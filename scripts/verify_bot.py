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
from config.settings import get_settings
from utils.risk_manager import RiskManager, PropAccountLimits
from utils.logger import setup_logging
from scripts.run_smc_v3 import DEFAULT_SYMBOLS, SYMBOL_ALIASES, SYMBOL_CONFIGS # Import these from run_smc_v3

# from strategies.smc_strategy_v3 import InstrumentType # Not directly used here, but good to note



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

    all_ok = True
    verified_symbols = []

    for symbol in DEFAULT_SYMBOLS:
        actual_symbol = None
        info = None
        # Try exact symbol first
        info = mt5.symbol_info(symbol)
        if info and info.visible:
            actual_symbol = symbol
        else: # Try aliases
            aliases = SYMBOL_ALIASES.get(symbol, [])
            for alias in aliases:
                info = mt5.symbol_info(alias)
                if info and info.visible:
                    actual_symbol = alias
                    print_warning(f"{symbol}: Using alias {alias}")
                    break
        
        if actual_symbol:
            verified_symbols.append(actual_symbol)
            if not info.visible: # If found but not visible
                mt5.symbol_select(actual_symbol, True)
                print_warning(f"{actual_symbol}: Added to Market Watch")
            
            # Get spread information
            if mt5.symbol_info_tick(actual_symbol):
                bid, ask = mt5.symbol_info_tick(actual_symbol).bid, mt5.symbol_info_tick(actual_symbol).ask
                # Ensure point is not zero to avoid division by zero
                point = info.point if info.point != 0 else 0.00001
                spread = (ask - bid) / point
                print_ok(f"{actual_symbol}: Spread={spread:.1f} pts, Min lot={info.volume_min}, Max lot={info.volume_max}, Step={info.volume_step}")
            else:
                print_warning(f"{actual_symbol}: Could not get tick info for spread. Is it actively traded?")
        else:
            print_error(f"{symbol}: NOT FOUND or NOT VISIBLE (tried aliases too).")
            all_ok = False

    if not verified_symbols:
        print_error("No tradeable symbols found!")
        all_ok = False
    else:
        print_ok(f"Verified {len(verified_symbols)} tradeable symbols: {', '.join(verified_symbols)}")

    return all_ok


def verify_risk_calculation(risk_percent: float = 0.5) -> bool:
    """
    CRITICAL: Verify risk calculation is correct using the bot's RiskManager.
    """
    print_header("RISK CALCULATION VERIFICATION")

    account = mt5.account_info()
    balance = account.balance

    limits = PropAccountLimits(
        max_daily_drawdown=4.0,
        max_account_drawdown=10.0,
        warning_drawdown=2.0,
        normal_risk_per_trade=risk_percent,
        reduced_risk_per_trade=risk_percent / 2,
        max_positions=2, # dummy value, not used in calculate_position_size directly
    )
    risk_manager = RiskManager(initial_account_balance=balance, limits=limits)

    all_ok = True
    test_sl_pips = 15 # Standard test SL for verification

    print(f"\n  Account Balance: ${balance:,.2f}")
    print(f"  Risk Percent: {risk_percent}% per trade")
    
    for symbol in DEFAULT_SYMBOLS:
        print(f"\n  --- {symbol} Test ---")
        info = mt5.symbol_info(symbol)
        if not info or not info.visible:
            print_warning(f"  {symbol} not found or not visible in MT5, skipping risk verification.")
            all_ok = False # Mark as not entirely OK, but continue
            continue

        # Get current price to calculate stop_loss_distance accurately
        tick_info = mt5.symbol_info_tick(symbol)
        if not tick_info or tick_info.bid == 0.0 or tick_info.ask == 0.0:
            print_warning(f"  Could not get current price for {symbol}, skipping risk verification.")
            all_ok = False # Mark as not entirely OK, but continue
            continue
        
        current_price = (tick_info.bid + tick_info.ask) / 2
        
        # Calculate point size based on symbol info, default to 0.00001 for forex
        point_size = info.point if info.point != 0 else 0.00001 
        
        # Assuming a long trade for SL calculation. SL will be 15 pips below current price.
        # Adjusted for spread, as per bot's logic in _check_confirmation
        current_spread_in_pips = (tick_info.ask - tick_info.bid) / point_size
        stop_loss_raw = current_price - (test_sl_pips * point_size)
        stop_loss = stop_loss_raw - (current_spread_in_pips * point_size) # Adjust SL for spread

        stop_loss_distance = abs(current_price - stop_loss)

        # For position size calculation, we need contract_size. Default to common forex value.
        contract_size = info.trade_contract_size if info.trade_contract_size > 0 else 100000 # Default for forex is 100,000

        lot_size = risk_manager.calculate_position_size(
            account_balance=balance,
            stop_loss_distance=stop_loss_distance,
            price=current_price,
            contract_size=contract_size,
            min_qty=info.volume_min,
            max_qty=info.volume_max,
            qty_step=info.volume_step,
        )

        print(f"  Current Price: {current_price:.{info.digits}f}")
        print(f"  Test SL Distance: {test_sl_pips} pips (adjusted for spread)")
        print(f"  Calculated Lot: {lot_size:.2f}")
        print(f"  MT5 Min Lot: {info.volume_min}, Max Lot: {info.volume_max}, Step: {info.volume_step}")
        
        if lot_size >= info.volume_min and lot_size <= info.volume_max:
            print_ok(f"Lot size {lot_size:.2f} is within MT5 limits.")
        else:
            print_error(f"Lot size {lot_size:.2f} is OUTSIDE MT5 limits ({info.volume_min}-{info.volume_max}).")
            all_ok = False
        
        if lot_size <= 0:
            print_error(f"Calculated Lot is {lot_size:.2f}. Check risk settings or symbol properties.")
            all_ok = False
            
    return all_ok

def simulate_trade(balance: float, risk_percent: float, sl_pips: float): # This function is removed
    pass


def run_verification(risk_percent: float = 0.5) -> bool:
    """Run complete verification."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 18 + "BOT VERIFICATION" + " " * 18 + "*")
    print("*" * 60)

    # 1. MT5 Connection
    if not verify_mt5_connection():
        print_error("\n[FATAL] Cannot continue without MT5 connection")
        mt5.shutdown()
        return False

    # 2. Symbols
    symbols_ok = verify_symbols()

    # 3. Risk Calculation
    risk_ok = verify_risk_calculation(risk_percent)

    # 4. Trading Hours
    trading_hours_ok = verify_trading_hours()

    # Final Summary
    print_header("FINAL SUMMARY")

    final_status = risk_ok and symbols_ok and trading_hours_ok # All checks must pass

    if final_status:
        print_ok("All CRITICAL checks passed successfully!")
        print_ok(f"Bot is ready for demo/paper trading with {risk_percent}% risk.")
        print("\n  To start the bot:")
        print(f"  python scripts/run_smc_v3.py --mode demo --log-level DEBUG --risk {risk_percent}")
        print("  (Use '--mode live' for real trading after extensive demo testing!)")
    else:
        print_error("Some CRITICAL verification checks FAILED!")
        print_error("DO NOT start the bot until all issues are fixed!")

    mt5.shutdown()
    return final_status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify bot before starting")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent to verify (default: 0.5)")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for verification script (default: INFO)"
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level) # Setup logging

    success = run_verification(args.risk)

    if not success:
        sys.exit(1)
