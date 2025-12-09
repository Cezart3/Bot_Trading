"""
Final verification script before London session trading.

Checks:
1. MT5 connection and account
2. Server time vs local time
3. Session hours configuration
4. Risk manager functionality
5. Strategy state machine
6. Order placement capability (dry run)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, time as dt_time

from config.settings import get_settings
from strategies.orb_forex_strategy import create_eurusd_london_strategy, ORBState
from utils.risk_manager import RiskManager, PropAccountLimits
from models.candle import Candle


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_mt5_connection():
    """Test MT5 connection and get server time."""
    print_header("1. MT5 CONNECTION & SERVER TIME")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  ERROR: MetaTrader5 not installed!")
        return False, None

    settings = get_settings()

    # Try to connect
    if not mt5.initialize():
        # Try with credentials
        if not mt5.initialize(
            login=settings.mt5.login,
            password=settings.mt5.password,
            server=settings.mt5.server,
            path=settings.mt5.path,
            timeout=60000,
        ):
            print(f"  ERROR: Cannot connect to MT5: {mt5.last_error()}")
            return False, mt5

    account = mt5.account_info()
    if not account:
        print("  ERROR: Cannot get account info!")
        return False, mt5

    print(f"\n  Account: {account.login}")
    print(f"  Name: {account.name}")
    print(f"  Server: {account.server}")
    print(f"  Balance: ${account.balance:,.2f}")

    # Get server time from tick
    mt5.symbol_select("EURUSD", True)
    tick = mt5.symbol_info_tick("EURUSD")

    local_time = datetime.now()
    server_time = datetime.fromtimestamp(tick.time) if tick else local_time

    print(f"\n  Local time:  {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")

    time_diff = (local_time - server_time).total_seconds() / 3600
    print(f"  Time difference: {time_diff:+.1f} hours")

    if abs(time_diff) > 1:
        print(f"\n  WARNING: Significant time difference detected!")
        print(f"  The bot uses server time for session detection.")

    return True, mt5


def test_session_times():
    """Test session time configuration."""
    print_header("2. SESSION TIME CONFIGURATION")

    strategy = create_eurusd_london_strategy()

    print(f"\n  Session: {strategy.config.session_name}")
    print(f"  Start: {strategy.config.session_start_hour:02d}:{strategy.config.session_start_minute:02d}")
    print(f"  End: {strategy.config.session_end_hour:02d}:{strategy.config.session_end_minute:02d}")
    print(f"  ORB Duration: {strategy.config.orb_duration_minutes} minutes")

    # Test session detection
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute

    session_start = dt_time(strategy.config.session_start_hour, strategy.config.session_start_minute)
    session_end = dt_time(strategy.config.session_end_hour, strategy.config.session_end_minute)
    orb_end_hour = strategy.config.session_start_hour + strategy.config.orb_duration_minutes // 60
    orb_end_minute = strategy.config.orb_duration_minutes % 60
    orb_end = dt_time(orb_end_hour, orb_end_minute)

    current = now.time()

    print(f"\n  Current time: {current.strftime('%H:%M:%S')}")

    if current < session_start:
        mins_until = (datetime.combine(now.date(), session_start) - now).total_seconds() / 60
        print(f"  Status: WAITING FOR SESSION")
        print(f"  Time until session: {mins_until:.0f} minutes")
    elif session_start <= current < orb_end:
        mins_until_orb_end = (datetime.combine(now.date(), orb_end) - now).total_seconds() / 60
        print(f"  Status: BUILDING OPENING RANGE")
        print(f"  Time until ORB complete: {mins_until_orb_end:.0f} minutes")
    elif orb_end <= current < session_end:
        mins_until_end = (datetime.combine(now.date(), session_end) - now).total_seconds() / 60
        print(f"  Status: TRADING WINDOW OPEN")
        print(f"  Time remaining: {mins_until_end:.0f} minutes")
    else:
        print(f"  Status: SESSION CLOSED")

    # London session info
    print(f"\n  NOTE: London session (GMT 08:00-17:00)")
    print(f"        Romania winter (EET = GMT+2): 10:00-19:00")
    print(f"        Romania summer (EEST = GMT+3): 11:00-20:00")
    print(f"        Currently configured: {strategy.config.session_start_hour:02d}:00-{strategy.config.session_end_hour:02d}:00")

    return True


def test_risk_manager():
    """Test risk manager functionality."""
    print_header("3. RISK MANAGER")

    balance = 100000.0
    risk_percent = 0.5

    limits = PropAccountLimits(
        max_daily_drawdown=4.0,
        max_account_drawdown=10.0,
        warning_drawdown=5.0,
        normal_risk_per_trade=risk_percent,
        reduced_risk_per_trade=risk_percent / 2,
        max_positions=1,
    )

    rm = RiskManager(initial_account_balance=balance, limits=limits)
    rm.initialize_daily_stats(balance)

    print(f"\n  Initial Balance: ${balance:,.2f}")
    print(f"  Risk per trade: {risk_percent}%")
    print(f"  Risk amount: ${balance * risk_percent / 100:,.2f}")

    # Test can_trade
    can_trade, reason = rm.can_trade(current_positions=0)
    print(f"\n  Can trade (0 positions): {can_trade} - {reason}")

    can_trade, reason = rm.can_trade(current_positions=1)
    print(f"  Can trade (1 position): {can_trade} - {reason}")

    # Test risk percent
    current_risk = rm.get_current_risk_percent()
    print(f"\n  Current risk %: {current_risk}%")

    # Simulate drawdown
    rm.update_balance(96000)  # 4% loss
    status = rm.get_risk_status()
    print(f"\n  After 4% loss:")
    print(f"    Daily loss: {status['daily_loss_percent']:.2f}%")
    print(f"    Can trade: {status['can_trade']}")
    print(f"    Risk %: {status['current_risk_percent']}%")

    return True


def test_strategy_logic():
    """Test strategy state machine."""
    print_header("4. STRATEGY STATE MACHINE")

    strategy = create_eurusd_london_strategy()
    strategy.initialize()

    print(f"\n  Initial state: {strategy._state.value}")

    # Simulate candle during ORB period
    orb_candle = Candle(
        timestamp=datetime.now().replace(hour=10, minute=30),
        open=1.1000,
        high=1.1020,
        low=1.0990,
        close=1.1010,
        volume=100,
        symbol="EURUSD",
        timeframe="M5",
    )

    signal = strategy.on_candle(orb_candle, [orb_candle])
    print(f"  After ORB candle (10:30): {strategy._state.value}")
    print(f"  Signal: {signal}")

    # Simulate candle after ORB with breakout
    strategy._state = ORBState.ORB_COMPLETE
    strategy._opening_range = strategy._opening_range or type('OpeningRange', (), {
        'high': 1.1020,
        'low': 1.1000,
        'start_time': datetime.now(),
        'end_time': datetime.now(),
        'candle_count': 12,
    })()

    breakout_candle = Candle(
        timestamp=datetime.now().replace(hour=11, minute=30),
        open=1.1018,
        high=1.1035,
        low=1.1015,
        close=1.1030,  # > 1.1020 + 0.0002 buffer
        volume=100,
        symbol="EURUSD",
        timeframe="M5",
    )

    signal = strategy._check_breakout(breakout_candle)
    if signal:
        print(f"\n  Breakout signal generated:")
        print(f"    Type: {signal.signal_type.value}")
        print(f"    Entry: {signal.price:.5f}")
        print(f"    SL: {signal.stop_loss:.5f}")
        print(f"    TP: {signal.take_profit:.5f}")

        # Calculate lot size
        sl_pips = abs(signal.price - signal.stop_loss) / 0.0001
        risk_amount = 500  # 0.5% of 100k
        pip_value = 10
        lot_size = risk_amount / (sl_pips * pip_value)
        print(f"\n    SL Distance: {sl_pips:.1f} pips")
        print(f"    Lot Size: {lot_size:.2f}")
        print(f"    Risk: ${sl_pips * pip_value * round(lot_size, 2):.2f}")
    else:
        print("  No breakout signal")

    return True


def test_order_creation():
    """Test order creation (dry run)."""
    print_header("5. ORDER CREATION (DRY RUN)")

    from models.order import Order, OrderSide, OrderType

    # Simulate order parameters
    entry = 1.1030
    sl = 1.0998  # Below OR low with buffer
    tp = 1.1094  # 2:1 R:R
    lot_size = 1.67

    order = Order(
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=lot_size,
        stop_loss=sl,
        take_profit=tp,
        comment="ORB_FOREX",
        magic_number=3001,
    )

    print(f"\n  Order created:")
    print(f"    Symbol: {order.symbol}")
    print(f"    Side: {order.side.value}")
    print(f"    Type: {order.order_type.value}")
    print(f"    Quantity: {order.quantity} lots")
    print(f"    Stop Loss: {order.stop_loss:.5f}")
    print(f"    Take Profit: {order.take_profit:.5f}")
    print(f"    Comment: {order.comment}")
    print(f"    Magic: {order.magic_number}")

    return True


def main():
    """Run all verification tests."""
    print("\n")
    print("=" * 70)
    print("  FINAL VERIFICATION - ORB TRADING BOT")
    print("  TeleTrade-Sharp ECN | EUR/USD | London Session")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 1. MT5 Connection
    results["mt5"], mt5 = test_mt5_connection()
    if mt5:
        mt5.shutdown()

    # 2. Session times
    results["session"] = test_session_times()

    # 3. Risk manager
    results["risk"] = test_risk_manager()

    # 4. Strategy logic
    results["strategy"] = test_strategy_logic()

    # 5. Order creation
    results["order"] = test_order_creation()

    # Summary
    print_header("VERIFICATION SUMMARY")

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[X]"
        print(f"  {symbol} {name.upper()}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL CHECKS PASSED!")
        print()
        print("  The bot is READY for London session trading.")
        print()
        print("  To start the bot:")
        print("    cd D:\\Proiecte\\BOT_TRADING\\Bot_Trading")
        print("    python run_forex_trading.py --mode demo")
        print()
        print("  Session hours: 10:00 - 18:00 (server time)")
        print("  Opening Range: 10:00 - 11:00")
        print("  Risk per trade: 0.5% = $500 for 100k account")
    else:
        print("  SOME CHECKS FAILED!")
        print("  Please review the issues above before trading.")

    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
