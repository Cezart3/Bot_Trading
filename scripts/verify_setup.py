"""
SMC V3 Setup Verification Script

Run this BEFORE starting the bot to verify everything is working correctly.

Checks:
1. MT5 connection and authentication
2. Account info and balance
3. Symbol availability and info
4. Data feed (candles) for all timeframes
5. Session times and kill zones
6. Strategy initialization
7. Risk manager configuration
8. Spread check for each symbol
9. Market status (open/closed)
10. Python dependencies

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --symbols EURUSD GBPUSD
    python scripts/verify_setup.py --verbose
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional
import importlib

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")


def print_check(name: str, passed: bool, details: str = ""):
    if passed:
        status = f"{Colors.GREEN}[OK]{Colors.RESET}"
    else:
        status = f"{Colors.RED}[FAIL]{Colors.RESET}"

    print(f"  {status} {name}")
    if details:
        print(f"       {Colors.YELLOW}{details}{Colors.RESET}")


def print_warning(text: str):
    print(f"  {Colors.YELLOW}[WARN] {text}{Colors.RESET}")


def print_info(text: str):
    print(f"  {Colors.BLUE}[INFO] {text}{Colors.RESET}")


class SetupVerifier:
    """Comprehensive setup verification for SMC V3 trading bot."""

    def __init__(self, symbols: List[str], verbose: bool = False):
        self.symbols = symbols
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.broker = None

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print_header("SMC V3 SETUP VERIFICATION")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Symbols: {', '.join(self.symbols)}")

        all_passed = True

        # 1. Python Dependencies
        print_header("1. PYTHON DEPENDENCIES")
        if not self._check_dependencies():
            all_passed = False

        # 2. Configuration Files
        print_header("2. CONFIGURATION FILES")
        if not self._check_config_files():
            all_passed = False

        # 3. MT5 Connection
        print_header("3. MT5 CONNECTION")
        if not self._check_mt5_connection():
            all_passed = False
            print_warning("Cannot continue without MT5 connection")
            return False

        # 4. Account Info
        print_header("4. ACCOUNT INFO")
        if not self._check_account_info():
            all_passed = False

        # 5. Symbols
        print_header("5. SYMBOLS CHECK")
        if not self._check_symbols():
            all_passed = False

        # 6. Data Feed
        print_header("6. DATA FEED")
        if not self._check_data_feed():
            all_passed = False

        # 7. Session Times
        print_header("7. SESSION TIMES")
        self._check_session_times()

        # 8. Strategy Initialization
        print_header("8. STRATEGY INITIALIZATION")
        if not self._check_strategy_init():
            all_passed = False

        # 9. Risk Manager
        print_header("9. RISK MANAGER")
        if not self._check_risk_manager():
            all_passed = False

        # 10. Market Status
        print_header("10. MARKET STATUS")
        self._check_market_status()

        # Summary
        self._print_summary(all_passed)

        # Cleanup
        if self.broker:
            self.broker.disconnect()

        return all_passed

    def _check_dependencies(self) -> bool:
        """Check required Python packages."""
        required = [
            ('MetaTrader5', 'MetaTrader5'),
            ('pytz', 'pytz'),
            ('numpy', 'numpy'),
        ]

        all_ok = True
        for name, import_name in required:
            try:
                importlib.import_module(import_name)
                print_check(f"{name}", True)
            except ImportError:
                print_check(f"{name}", False, f"pip install {name}")
                all_ok = False
                self.errors.append(f"Missing package: {name}")

        return all_ok

    def _check_config_files(self) -> bool:
        """Check configuration files exist."""
        files = [
            ('config/settings.py', 'Settings configuration'),
            ('strategies/smc_strategy_v3.py', 'SMC V3 Strategy'),
            ('brokers/mt5_broker.py', 'MT5 Broker'),
            ('utils/risk_manager.py', 'Risk Manager'),
        ]

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        all_ok = True

        for file, desc in files:
            full_path = os.path.join(base_path, file)
            exists = os.path.exists(full_path)
            print_check(f"{desc}", exists, file if not exists else "")
            if not exists:
                all_ok = False
                self.errors.append(f"Missing file: {file}")

        return all_ok

    def _check_mt5_connection(self) -> bool:
        """Check MT5 connection."""
        try:
            from config.settings import get_settings
            from brokers.mt5_broker import MT5Broker

            settings = get_settings()

            print_info(f"Server: {settings.mt5.server}")
            print_info(f"Login: {settings.mt5.login}")

            self.broker = MT5Broker(
                login=settings.mt5.login,
                password=settings.mt5.password,
                server=settings.mt5.server,
                path=settings.mt5.path,
                timeout=settings.mt5.timeout,
            )

            connected = self.broker.connect()
            print_check("MT5 Connection", connected)

            if not connected:
                self.errors.append("Failed to connect to MT5")
                # Try to get more details
                try:
                    import MetaTrader5 as mt5
                    error = mt5.last_error()
                    print_warning(f"MT5 Error: {error}")
                except:
                    pass
                return False

            return True

        except Exception as e:
            print_check("MT5 Connection", False, str(e))
            self.errors.append(f"MT5 connection error: {e}")
            return False

    def _check_account_info(self) -> bool:
        """Check account information."""
        try:
            account_info = self.broker.get_account_info()

            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            margin = account_info.get('margin', 0)
            free_margin = account_info.get('margin_free', 0)

            print_check(f"Balance: ${balance:,.2f}", balance > 0)
            print_check(f"Equity: ${equity:,.2f}", equity > 0)
            print_info(f"Margin Used: ${margin:,.2f}")
            print_info(f"Free Margin: ${free_margin:,.2f}")

            # Check account type
            account_type = account_info.get('trade_mode', 'unknown')
            if account_type == 0:
                print_info("Account Type: DEMO")
            elif account_type == 1:
                print_warning("Account Type: CONTEST")
            elif account_type == 2:
                print_warning("Account Type: REAL - BE CAREFUL!")

            if balance <= 0:
                self.errors.append("Account balance is 0 or negative")
                return False

            return True

        except Exception as e:
            print_check("Account Info", False, str(e))
            self.errors.append(f"Account info error: {e}")
            return False

    def _check_symbols(self) -> bool:
        """Check all symbols are available."""
        import MetaTrader5 as mt5

        all_ok = True

        for symbol in self.symbols:
            try:
                # Use MT5 directly for accurate trade_mode
                mt5_info = mt5.symbol_info(symbol)

                if mt5_info is None:
                    print_check(f"{symbol}", False, "Symbol not found")
                    self.errors.append(f"Symbol not found: {symbol}")
                    all_ok = False
                    continue

                # Check if symbol is tradeable
                # trade_mode: 0=disabled, 1=longonly, 2=shortonly, 3=closeonly, 4=full
                trade_mode = mt5_info.trade_mode
                if trade_mode == 0:
                    print_check(f"{symbol}", False, "Trading disabled")
                    self.errors.append(f"Trading disabled for: {symbol}")
                    all_ok = False
                    continue

                # Check if symbol is visible (enabled in Market Watch)
                if not mt5_info.visible:
                    # Try to enable it
                    mt5.symbol_select(symbol, True)
                    print_info(f"{symbol} was not visible, enabled it")

                trade_mode_names = {
                    0: "DISABLED", 1: "LONG_ONLY", 2: "SHORT_ONLY",
                    3: "CLOSE_ONLY", 4: "FULL"
                }
                trade_mode_str = trade_mode_names.get(trade_mode, str(trade_mode))

                # Get spread
                bid, ask = self.broker.get_current_price(symbol)
                if bid and ask:
                    spread = (ask - bid) / mt5_info.point
                    spread_ok = spread < 30  # Max 30 pips spread

                    details = f"Spread: {spread:.1f} pts, Mode: {trade_mode_str}"
                    if spread > 20:
                        details += " (HIGH!)"
                        self.warnings.append(f"High spread on {symbol}: {spread:.1f} pts")

                    print_check(f"{symbol}", spread_ok, details)
                else:
                    print_check(f"{symbol}", True, f"Mode: {trade_mode_str} (no price yet)")

                if self.verbose:
                    print_info(f"  Point: {mt5_info.point}")
                    print_info(f"  Min Lot: {mt5_info.volume_min}")
                    print_info(f"  Max Lot: {mt5_info.volume_max}")

            except Exception as e:
                print_check(f"{symbol}", False, str(e))
                self.errors.append(f"Symbol error {symbol}: {e}")
                all_ok = False

        return all_ok

    def _check_data_feed(self) -> bool:
        """Check data feed for all symbols and timeframes."""
        timeframes = ["M5", "M15", "H1", "H4", "D1"]
        all_ok = True

        for symbol in self.symbols:
            print_info(f"Checking {symbol}...")

            for tf in timeframes:
                try:
                    candles = self.broker.get_candles(symbol, tf, 10)

                    if candles and len(candles) > 0:
                        last_candle = candles[-1]
                        age_minutes = (datetime.now() - last_candle.timestamp.replace(tzinfo=None)).total_seconds() / 60

                        # Check data freshness
                        if tf == "M5" and age_minutes > 10:
                            print_check(f"  {tf}", True, f"Last: {age_minutes:.0f} min ago (check if market open)")
                        else:
                            print_check(f"  {tf}", True, f"{len(candles)} candles")
                    else:
                        print_check(f"  {tf}", False, "No data")
                        self.warnings.append(f"No {tf} data for {symbol}")

                except Exception as e:
                    print_check(f"  {tf}", False, str(e))
                    all_ok = False

        return all_ok

    def _check_session_times(self):
        """Check current session and time to next session."""
        utc_now = datetime.now(timezone.utc)
        hour = utc_now.hour
        minute = utc_now.minute

        print_info(f"Current UTC Time: {utc_now.strftime('%H:%M')}")

        # Session definitions
        sessions = {
            "London Kill Zone": (7, 0, 9, 0),
            "London Extended": (7, 0, 12, 0),
            "NY Kill Zone": (13, 0, 15, 0),
            "NY Extended": (13, 0, 20, 0),
        }

        current_session = None
        for name, (start_h, start_m, end_h, end_m) in sessions.items():
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            current_minutes = hour * 60 + minute

            if start_minutes <= current_minutes < end_minutes:
                current_session = name
                remaining = end_minutes - current_minutes
                print_check(f"IN {name}", True, f"{remaining} min remaining")

        if not current_session:
            # Calculate time to next session
            current_minutes = hour * 60 + minute

            if current_minutes < 7 * 60:
                next_session = "London"
                minutes_until = 7 * 60 - current_minutes
            elif current_minutes < 13 * 60:
                next_session = "NY"
                minutes_until = 13 * 60 - current_minutes
            else:
                next_session = "London (tomorrow)"
                minutes_until = (24 * 60 - current_minutes) + 7 * 60

            print_warning(f"Outside trading sessions")
            print_info(f"Next session: {next_session} in {minutes_until // 60}h {minutes_until % 60}m")

        # Check if weekend
        if utc_now.weekday() >= 5:
            print_warning("WEEKEND - Markets are closed!")
            days_until_monday = 7 - utc_now.weekday()
            print_info(f"Markets open in ~{days_until_monday} days")

    def _check_strategy_init(self) -> bool:
        """Check strategy can be initialized."""
        try:
            from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3

            for symbol in self.symbols[:2]:  # Test first 2 symbols
                config = SMCConfigV3(
                    risk_percent=0.5,
                    max_trades_per_day=2,
                )
                strategy = SMCStrategyV3(
                    symbol=symbol,
                    config=config
                )

                if strategy.initialize():
                    print_check(f"Strategy init ({symbol})", True)
                else:
                    print_check(f"Strategy init ({symbol})", False)
                    self.errors.append(f"Strategy init failed for {symbol}")
                    return False

            return True

        except Exception as e:
            print_check("Strategy Initialization", False, str(e))
            self.errors.append(f"Strategy init error: {e}")
            return False

    def _check_risk_manager(self) -> bool:
        """Check risk manager configuration."""
        try:
            from utils.risk_manager import RiskManager, PropAccountLimits

            limits = PropAccountLimits(
                max_daily_drawdown=4.0,
                max_account_drawdown=10.0,
                normal_risk_per_trade=0.5,
                reduced_risk_per_trade=0.25,
                max_positions=2,
            )

            rm = RiskManager(initial_account_balance=10000, limits=limits)
            rm.initialize_daily_stats(10000)

            # Check can trade
            can_trade, reason = rm.can_trade(0)
            print_check("Risk Manager Init", can_trade, reason if not can_trade else "")

            # Check risk calculation
            risk_pct = rm.get_current_risk_percent()
            print_info(f"Current Risk: {risk_pct}%")

            # Check position sizing
            pos_size = rm.calculate_position_size(
                account_balance=10000,
                stop_loss_distance=0.0020,  # 20 pips
                price=1.0850,
                contract_size=100000,
                min_qty=0.01,
                max_qty=100,
                qty_step=0.01
            )
            print_info(f"Sample Position Size: {pos_size} lots (for 20 pip SL)")

            return True

        except Exception as e:
            print_check("Risk Manager", False, str(e))
            self.errors.append(f"Risk manager error: {e}")
            return False

    def _check_market_status(self):
        """Check if markets are currently open."""
        utc_now = datetime.now(timezone.utc)

        # Check if weekend
        is_weekend = utc_now.weekday() >= 5

        # Check Forex market hours (Sunday 22:00 UTC to Friday 22:00 UTC)
        is_forex_open = not is_weekend
        if utc_now.weekday() == 4 and utc_now.hour >= 22:
            is_forex_open = False
        if utc_now.weekday() == 6 and utc_now.hour < 22:
            is_forex_open = False

        print_check("Forex Market", is_forex_open, "CLOSED" if not is_forex_open else "OPEN")

        # Check US market hours (13:30 - 20:00 UTC roughly)
        us_start = 13 * 60 + 30
        us_end = 20 * 60
        current = utc_now.hour * 60 + utc_now.minute
        is_us_open = not is_weekend and us_start <= current < us_end

        print_check("US Markets", is_us_open, "CLOSED" if not is_us_open else "OPEN")

        if is_weekend:
            print_warning("Markets are closed for the weekend!")

    def _print_summary(self, all_passed: bool):
        """Print verification summary."""
        print_header("VERIFICATION SUMMARY")

        if all_passed and not self.errors:
            print(f"\n  {Colors.GREEN}{Colors.BOLD}ALL CHECKS PASSED!{Colors.RESET}")
            print(f"  {Colors.GREEN}Bot is ready to run.{Colors.RESET}")
        else:
            print(f"\n  {Colors.RED}{Colors.BOLD}SOME CHECKS FAILED{Colors.RESET}")

        if self.errors:
            print(f"\n  {Colors.RED}Errors ({len(self.errors)}):{Colors.RESET}")
            for err in self.errors:
                print(f"    - {err}")

        if self.warnings:
            print(f"\n  {Colors.YELLOW}Warnings ({len(self.warnings)}):{Colors.RESET}")
            for warn in self.warnings:
                print(f"    - {warn}")

        # Print run command
        print(f"\n  {Colors.CYAN}To start the bot:{Colors.RESET}")
        print(f"  {Colors.BOLD}python scripts/run_smc_v3.py --mode demo --risk 0.5 --log-level DEBUG{Colors.RESET}")

        print()


def main():
    parser = argparse.ArgumentParser(description="SMC V3 Setup Verification")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD", "AUDUSD", "US500", "USTech100", "GER40"],
        help="Symbols to verify"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )

    args = parser.parse_args()

    verifier = SetupVerifier(symbols=args.symbols, verbose=args.verbose)
    success = verifier.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
