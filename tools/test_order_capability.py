"""
Test order placement capability on MT5.
This tests if we CAN place orders, without actually placing them.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


def test_order_capability():
    """Test if we can place orders on MT5."""
    print("\n" + "=" * 70)
    print("  ORDER CAPABILITY TEST")
    print("=" * 70)

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  ERROR: MetaTrader5 not installed!")
        return False

    from config.settings import get_settings
    settings = get_settings()

    # Connect
    if not mt5.initialize():
        if not mt5.initialize(
            login=settings.mt5.login,
            password=settings.mt5.password,
            server=settings.mt5.server,
            path=settings.mt5.path,
            timeout=60000,
        ):
            print(f"  ERROR: Cannot connect: {mt5.last_error()}")
            return False

    account = mt5.account_info()
    print(f"\n  Account: {account.login}")
    print(f"  Balance: ${account.balance:,.2f}")
    print(f"  Leverage: 1:{account.leverage}")

    # Check if trading is allowed
    print(f"\n  Trade allowed: {account.trade_allowed}")
    print(f"  Trade expert: {account.trade_expert}")

    if not account.trade_allowed:
        print("\n  WARNING: Trading is NOT allowed on this account!")
        print("  Check MT5 settings: Tools -> Options -> Expert Advisors")
        mt5.shutdown()
        return False

    if not account.trade_expert:
        print("\n  WARNING: Expert Advisor trading is NOT allowed!")
        print("  Enable 'AutoTrading' button in MT5 toolbar")
        mt5.shutdown()
        return False

    # Select symbol and get info
    symbol = "EURUSD"
    if not mt5.symbol_select(symbol, True):
        print(f"\n  ERROR: Cannot select {symbol}")
        mt5.shutdown()
        return False

    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        print(f"\n  ERROR: Cannot get {symbol} info")
        mt5.shutdown()
        return False

    print(f"\n  Symbol: {symbol}")
    print(f"  Trade mode: {symbol_info.trade_mode}")  # 0=disabled, 4=full
    print(f"  Volume min: {symbol_info.volume_min}")
    print(f"  Volume max: {symbol_info.volume_max}")
    print(f"  Volume step: {symbol_info.volume_step}")
    print(f"  Filling mode: {symbol_info.filling_mode}")

    # Trade mode check
    trade_modes = {
        0: "DISABLED",
        1: "LONG_ONLY",
        2: "SHORT_ONLY",
        3: "CLOSE_ONLY",
        4: "FULL"
    }
    mode_name = trade_modes.get(symbol_info.trade_mode, "UNKNOWN")
    print(f"  Trade mode name: {mode_name}")

    if symbol_info.trade_mode != 4:  # SYMBOL_TRADE_MODE_FULL
        print(f"\n  WARNING: Symbol trade mode is {mode_name}, not FULL!")

    # Check order filling modes
    filling_modes = []
    if symbol_info.filling_mode & 1:
        filling_modes.append("FOK")
    if symbol_info.filling_mode & 2:
        filling_modes.append("IOC")
    if not filling_modes:
        filling_modes.append("RETURN")
    print(f"  Supported filling: {', '.join(filling_modes)}")

    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        print(f"\n  ERROR: Cannot get tick for {symbol}")
        mt5.shutdown()
        return False

    print(f"\n  Current Bid: {tick.bid:.5f}")
    print(f"  Current Ask: {tick.ask:.5f}")
    print(f"  Spread: {(tick.ask - tick.bid) / 0.0001:.1f} pips")

    # Check margin requirement for 1 lot
    margin = mt5.order_calc_margin(
        mt5.ORDER_TYPE_BUY,
        symbol,
        1.0,  # 1 lot
        tick.ask
    )
    if margin:
        print(f"\n  Margin for 1 lot: ${margin:,.2f}")
        print(f"  Free margin: ${account.margin_free:,.2f}")
        print(f"  Max lots possible: {account.margin_free / margin:.2f}")

    # Calculate profit for 1 lot, 10 pips
    profit = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY,
        symbol,
        1.0,  # 1 lot
        tick.ask,
        tick.ask + 0.0010  # 10 pips
    )
    if profit:
        print(f"  Profit for 1 lot, 10 pips: ${profit:,.2f}")

    # Check existing positions
    positions = mt5.positions_get(symbol=symbol)
    print(f"\n  Open positions on {symbol}: {len(positions) if positions else 0}")

    # Check pending orders
    orders = mt5.orders_get(symbol=symbol)
    print(f"  Pending orders on {symbol}: {len(orders) if orders else 0}")

    # Test order check (without placing)
    print("\n  Testing order validation (not placing)...")

    # Prepare a test order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,  # Minimum lot
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "sl": tick.ask - 0.0030,  # 30 pips SL
        "tp": tick.ask + 0.0060,  # 60 pips TP (2:1)
        "deviation": 20,
        "magic": 3001,
        "comment": "TEST_ORB_FOREX",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    # Check the order (without executing)
    result = mt5.order_check(request)

    if result is None:
        print(f"\n  ERROR: order_check failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print(f"\n  Order check result:")
    print(f"    Retcode: {result.retcode}")
    print(f"    Comment: {result.comment}")
    print(f"    Margin: ${result.margin:,.2f}")
    print(f"    Margin free: ${result.margin_free:,.2f}")

    # Retcode meanings
    retcodes = {
        0: "OK - Order can be placed",
        10004: "Requote",
        10006: "Request rejected",
        10007: "Request canceled by trader",
        10010: "Only position close allowed",
        10011: "No connection to trade server",
        10012: "Disabled for this account",
        10014: "Invalid volume",
        10015: "Invalid price",
        10016: "Invalid stops",
        10017: "Trade disabled",
        10018: "Market closed",
        10019: "Not enough money",
        10020: "Price changed",
        10021: "No quotes",
        10024: "Auto trading disabled",
        10025: "Protection level triggered",
        10026: "SL or TP near current price",
        10027: "Invalid SL or TP",
    }

    if result.retcode == 0:
        print(f"\n  ORDER CAN BE PLACED!")
        print(f"  The system is ready to trade.")
    else:
        desc = retcodes.get(result.retcode, "Unknown error")
        print(f"\n  ORDER CANNOT BE PLACED!")
        print(f"  Error: {result.retcode} - {desc}")

        if result.retcode == 10024:
            print("\n  FIX: Enable AutoTrading button in MT5 toolbar!")
        elif result.retcode == 10019:
            print("\n  FIX: Not enough margin. Reduce lot size.")
        elif result.retcode == 10016:
            print("\n  FIX: Invalid stops. Check SL/TP distances.")

    mt5.shutdown()

    return result.retcode == 0


def main():
    success = test_order_capability()

    print("\n" + "=" * 70)
    if success:
        print("  RESULT: TRADING IS ENABLED AND READY!")
    else:
        print("  RESULT: TRADING CAPABILITY ISSUE DETECTED!")
        print("  Please check the issues above.")
    print("=" * 70 + "\n")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
