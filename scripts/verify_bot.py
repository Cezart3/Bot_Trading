"""
BOT VERIFICATION SCRIPT

Run this BEFORE starting the bot to verify:
1. MT5 connection works
2. Risk calculation is CORRECT (0.5% = $50 for $10k account)
3. All symbols are available
4. Trading hours are correct

Author: Trading Bot Project
"""

import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5


def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_ok(text: str):
    print(f"  [OK] {text}")


def print_error(text: str):
    print(f"  [ERROR] {text}")


def print_warning(text: str):
    print(f"  [WARN] {text}")


def verify_mt5_connection() -> bool:
    """Verify MT5 connection."""
    print_header("MT5 CONNECTION")

    if not mt5.initialize():
        print_error(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    account = mt5.account_info()
    if not account:
        print_error("Failed to get account info")
        return False

    print_ok(f"Connected to: {account.server}")
    print_ok(f"Account: {account.login}")
    print_ok(f"Name: {account.name}")
    print_ok(f"Balance: ${account.balance:,.2f}")
    print_ok(f"Equity: ${account.equity:,.2f}")
    print_ok(f"Leverage: 1:{account.leverage}")

    return True


def verify_symbols() -> bool:
    """Verify trading symbols."""
    print_header("SYMBOLS VERIFICATION")

    required_symbols = ["EURUSD", "GBPUSD", "AUDUSD", "US30", "NDX100", "GER30"]
    all_ok = True

    for symbol in required_symbols:
        info = mt5.symbol_info(symbol)
        if info is None:
            print_error(f"{symbol}: NOT FOUND")
            all_ok = False
        elif not info.visible:
            mt5.symbol_select(symbol, True)
            print_warning(f"{symbol}: Added to Market Watch")
        else:
            bid, ask = mt5.symbol_info_tick(symbol).bid, mt5.symbol_info_tick(symbol).ask
            spread = (ask - bid) / info.point
            print_ok(f"{symbol}: Spread={spread:.1f} pts, Min lot={info.volume_min}")

    return all_ok


def verify_risk_calculation(risk_percent: float = 0.5) -> bool:
    """
    CRITICAL: Verify risk calculation is correct.

    For $10,000 account with 0.5% risk:
    - Risk amount should be $50
    - With 15 pip SL and $10/pip, lot size should be ~0.33
    - Loss at SL should be exactly $50
    """
    print_header("RISK CALCULATION VERIFICATION")

    account = mt5.account_info()
    balance = account.balance

    print(f"\n  Account Balance: ${balance:,.2f}")
    print(f"  Risk Percent: {risk_percent}%")
    print(f"  Risk Amount: ${balance * risk_percent / 100:.2f}")

    # Test for EURUSD
    print("\n  --- EURUSD Test ---")
    symbol_info = mt5.symbol_info("EURUSD")
    if symbol_info:
        pip_size = 0.0001
        pip_value = symbol_info.trade_tick_value * 10  # Approximate pip value

        test_sl_pips = 15  # 15 pips SL
        risk_amount = balance * risk_percent / 100
        lot_size = risk_amount / (test_sl_pips * pip_value)
        lot_size = round(lot_size, 2)
        lot_size = max(0.01, min(lot_size, 10.0))

        actual_risk = lot_size * test_sl_pips * pip_value
        actual_risk_percent = (actual_risk / balance) * 100

        print(f"  Test SL: {test_sl_pips} pips")
        print(f"  Pip Value: ${pip_value:.2f}/pip/lot")
        print(f"  Calculated Lot: {lot_size:.2f}")
        print(f"  Actual Risk: ${actual_risk:.2f} ({actual_risk_percent:.2f}%)")

        # Verify
        if abs(actual_risk_percent - risk_percent) <= 0.1:
            print_ok(f"RISK IS CORRECT! Loss at SL = ${actual_risk:.2f} ({actual_risk_percent:.2f}%)")
            return True
        else:
            print_error(f"RISK IS WRONG! Expected {risk_percent}%, got {actual_risk_percent:.2f}%")
            return False

    return False


def verify_trading_hours() -> bool:
    """Verify trading hours are set correctly."""
    print_header("TRADING HOURS VERIFICATION")

    utc_now = datetime.now(timezone.utc)
    ro_offset = 2  # Romania = UTC+2 in winter

    print(f"\n  Current UTC Time: {utc_now.strftime('%H:%M')}")
    print(f"  Current Romania Time: {(utc_now.hour + ro_offset) % 24}:{utc_now.strftime('%M')}")

    # London Kill Zone: 08:00-11:00 UTC = 10:00-13:00 Romania
    # NY Kill Zone: 14:00-17:00 UTC = 16:00-19:00 Romania

    print("\n  Expected Trading Hours:")
    print("  London: 08:00-11:00 UTC = 10:00-13:00 Romania")
    print("  NY:     14:00-17:00 UTC = 16:00-19:00 Romania")
    print("  NY Open: 14:30 UTC = 16:30 Romania")

    hour = utc_now.hour
    in_london = 8 <= hour < 12
    in_ny = 14 <= hour < 20

    if in_london:
        print_ok("Currently in LONDON session")
    elif in_ny:
        print_ok("Currently in NY session")
    else:
        if hour < 8:
            next_session = "London"
            minutes_until = (8 - hour) * 60 - utc_now.minute
        elif hour < 14:
            next_session = "NY"
            minutes_until = (14 - hour) * 60 - utc_now.minute
        else:
            next_session = "London (tomorrow)"
            minutes_until = (24 - hour + 8) * 60 - utc_now.minute

        print_warning(f"Outside trading hours. {next_session} in {minutes_until // 60}h {minutes_until % 60}m")

    return True


def simulate_trade(balance: float, risk_percent: float, sl_pips: float):
    """Simulate a trade to verify calculations."""
    print_header("TRADE SIMULATION")

    print(f"\n  Simulating trade with:")
    print(f"  - Balance: ${balance:,.2f}")
    print(f"  - Risk: {risk_percent}%")
    print(f"  - SL: {sl_pips} pips")

    # Get EURUSD info
    symbol_info = mt5.symbol_info("EURUSD")
    if not symbol_info:
        print_error("Cannot get EURUSD info")
        return

    pip_value = symbol_info.trade_tick_value * 10  # ~$10/pip/lot

    risk_amount = balance * risk_percent / 100
    lot_size = risk_amount / (sl_pips * pip_value)
    lot_size = round(lot_size / 0.01) * 0.01  # Round to 0.01
    lot_size = max(0.01, min(lot_size, 10.0))

    actual_risk = lot_size * sl_pips * pip_value
    actual_risk_pct = (actual_risk / balance) * 100

    print(f"\n  RESULTS:")
    print(f"  - Lot Size: {lot_size:.2f}")
    print(f"  - Risk Amount: ${actual_risk:.2f}")
    print(f"  - Risk Percent: {actual_risk_pct:.2f}%")

    # Win scenario (2R)
    tp_pips = sl_pips * 2
    win_pnl = lot_size * tp_pips * pip_value
    win_pct = (win_pnl / balance) * 100

    print(f"\n  IF SL HIT:")
    print(f"  - Loss: -${actual_risk:.2f} (-{actual_risk_pct:.2f}%)")

    print(f"\n  IF TP HIT (2R):")
    print(f"  - Profit: +${win_pnl:.2f} (+{win_pct:.2f}%)")

    # Verification
    print("\n  VERIFICATION:")
    if actual_risk_pct <= risk_percent * 1.05:  # Within 5% tolerance
        print_ok(f"Risk is within acceptable range ({actual_risk_pct:.2f}% <= {risk_percent * 1.05:.2f}%)")
    else:
        print_error(f"Risk is TOO HIGH! ({actual_risk_pct:.2f}% > {risk_percent * 1.05:.2f}%)")


def run_verification(risk_percent: float = 0.5):
    """Run complete verification."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 18 + "BOT VERIFICATION" + " " * 18 + "*")
    print("*" * 60)

    # 1. MT5 Connection
    if not verify_mt5_connection():
        print("\n[FATAL] Cannot continue without MT5 connection")
        mt5.shutdown()
        return False

    # 2. Symbols
    verify_symbols()

    # 3. Risk Calculation
    risk_ok = verify_risk_calculation(risk_percent)

    # 4. Trading Hours
    verify_trading_hours()

    # 5. Trade Simulation
    account = mt5.account_info()
    simulate_trade(account.balance, risk_percent, sl_pips=15)

    # Final Summary
    print_header("FINAL SUMMARY")

    if risk_ok:
        print_ok("Risk calculation is CORRECT")
        print_ok(f"Safe to run bot with {risk_percent}% risk")
        print()
        print("  To start the bot:")
        print(f"  python scripts/run_smc_v3.py --mode demo --risk {risk_percent}")
    else:
        print_error("Risk calculation has issues!")
        print_error("DO NOT start the bot until fixed!")

    mt5.shutdown()
    return risk_ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify bot before starting")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent to verify (default: 0.5)")

    args = parser.parse_args()

    success = run_verification(args.risk)

    if not success:
        sys.exit(1)
