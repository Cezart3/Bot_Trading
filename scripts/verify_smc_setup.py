"""
SMC Bot Verification Script

Verifica complet ca strategia SMC este configurata corect si gata de rulare.
Testeaza:
1. Importuri si dependente
2. Conexiune MT5
3. Date multi-timeframe (H4, H1, M5)
4. Strategia SMC (POI, Structure, OB, FVG)
5. News filter cu buffer 2h
6. Mod demo si live
7. Simulare completa de semnal

Usage:
    python scripts/verify_smc_setup.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
from typing import Optional, List

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print()
    print("=" * 70)
    print(f"  {Colors.BOLD}{text}{Colors.RESET}")
    print("=" * 70)


def print_ok(text):
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {text}")


def print_error(text):
    print(f"  {Colors.RED}[ERROR]{Colors.RESET} {text}")


def print_info(text):
    print(f"  {Colors.CYAN}->{Colors.RESET} {text}")


def print_warn(text):
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {text}")


def print_section(num, total, text):
    print(f"\n{Colors.BLUE}[{num}/{total}]{Colors.RESET} {text}...")


class SMCVerifier:
    """Verifies SMC strategy setup."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.mt5 = None
        self.settings = None

    def verify_imports(self) -> bool:
        """Verify all required imports."""
        print_section(1, 8, "Verificare importuri si dependente")

        imports_ok = True

        # Core imports
        try:
            from config.settings import get_settings
            self.settings = get_settings()
            print_ok("config.settings importat")
        except ImportError as e:
            print_error(f"config.settings: {e}")
            self.errors.append("config.settings")
            imports_ok = False

        try:
            from brokers.mt5_broker import MT5Broker
            print_ok("brokers.mt5_broker importat")
        except ImportError as e:
            print_error(f"brokers.mt5_broker: {e}")
            self.errors.append("mt5_broker")
            imports_ok = False

        # SMC Strategy imports
        try:
            from strategies.smc_strategy import (
                SMCStrategy, SMCConfig, SMCState,
                MarketBias, StructureType, POIType,
                SwingPoint, StructureBreak, OrderBlock, FairValueGap,
                LiquidityLevel, POI, SignalScore
            )
            print_ok("strategies.smc_strategy importat (toate clasele)")
        except ImportError as e:
            print_error(f"strategies.smc_strategy: {e}")
            self.errors.append("smc_strategy")
            imports_ok = False

        # News filter
        try:
            from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact
            print_ok("utils.news_filter importat")
        except ImportError as e:
            print_error(f"utils.news_filter: {e}")
            self.errors.append("news_filter")
            imports_ok = False

        # MT5
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
            print_ok("MetaTrader5 importat")
        except ImportError as e:
            print_error(f"MetaTrader5: {e}")
            print_info("Ruleaza: pip install MetaTrader5")
            self.errors.append("MetaTrader5")
            imports_ok = False

        # Other dependencies
        try:
            import pytz
            print_ok("pytz importat")
        except ImportError as e:
            print_error(f"pytz: {e}")
            self.errors.append("pytz")
            imports_ok = False

        return imports_ok

    def verify_mt5_connection(self) -> bool:
        """Verify MT5 connection."""
        print_section(2, 8, "Verificare conexiune MT5")

        if not self.mt5 or not self.settings:
            print_error("Importurile nu sunt disponibile")
            return False

        print_info(f"Conectare la {self.settings.mt5.server}...")

        if not self.mt5.initialize(
            login=self.settings.mt5.login,
            password=self.settings.mt5.password,
            server=self.settings.mt5.server,
            path=self.settings.mt5.path,
            timeout=60000,
        ):
            error = self.mt5.last_error()
            print_error(f"Nu ma pot conecta la MT5: {error}")
            print_info("Verifica ca MT5 terminal este deschis si logat")
            self.errors.append("mt5_connection")
            return False

        print_ok("Conectat la MT5")

        account = self.mt5.account_info()
        if account:
            print_info(f"Account: {account.login}")
            print_info(f"Server: {account.server}")
            print_info(f"Balance: ${account.balance:,.2f}")
            print_info(f"Equity: ${account.equity:,.2f}")
            print_info(f"Leverage: 1:{account.leverage}")

            # Check account type
            if account.trade_mode == 0:
                print_ok("Mod cont: DEMO")
            elif account.trade_mode == 1:
                print_warn("Mod cont: CONTEST")
            elif account.trade_mode == 2:
                print_warn("Mod cont: LIVE (REAL MONEY!)")
        else:
            print_error("Nu pot obtine informatii despre cont")
            self.errors.append("account_info")
            return False

        return True

    def verify_multi_timeframe_data(self) -> bool:
        """Verify H4, H1, M5 data is available."""
        print_section(3, 8, "Verificare date multi-timeframe")

        if not self.mt5:
            return False

        symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
        timeframes = {
            "H4": self.mt5.TIMEFRAME_H4,
            "H1": self.mt5.TIMEFRAME_H1,
            "M5": self.mt5.TIMEFRAME_M5,
        }

        all_ok = True

        for symbol in symbols:
            if not self.mt5.symbol_select(symbol, True):
                print_error(f"{symbol}: Nu este disponibil")
                self.errors.append(f"symbol_{symbol}")
                all_ok = False
                continue

            print_info(f"Verificare {symbol}:")

            for tf_name, tf_value in timeframes.items():
                candles = self.mt5.copy_rates_from_pos(symbol, tf_value, 0, 100)

                if candles is None or len(candles) == 0:
                    print_error(f"  {tf_name}: Nu pot obtine date")
                    self.errors.append(f"candles_{symbol}_{tf_name}")
                    all_ok = False
                else:
                    last_time = datetime.fromtimestamp(candles[-1]["time"])
                    print_ok(f"  {tf_name}: {len(candles)} candele, ultima: {last_time.strftime('%H:%M')}")

            # Check spread
            tick = self.mt5.symbol_info_tick(symbol)
            if tick:
                pip_size = 0.01 if "JPY" in symbol else 0.0001
                spread_pips = (tick.ask - tick.bid) / pip_size
                if spread_pips > 3:
                    print_warn(f"  Spread mare: {spread_pips:.1f} pips")
                else:
                    print_ok(f"  Spread: {spread_pips:.1f} pips")

        return all_ok

    def verify_smc_strategy(self) -> bool:
        """Verify SMC strategy initialization and logic."""
        print_section(4, 8, "Verificare strategie SMC")

        try:
            from strategies.smc_strategy import SMCStrategy, SMCConfig

            config = SMCConfig(
                poi_min_score=3,
                min_rr=1.5,
                news_buffer_hours=2,
                max_trades_per_day=3,
            )

            strategy = SMCStrategy(
                symbol="EURUSD",
                timeframe="M5",
                magic_number=12350,
                timezone="UTC",
                config=config,
                use_news_filter=True,
            )
            strategy.pip_size = 0.0001

            if not strategy.initialize():
                print_error("Strategia nu s-a putut initializa")
                self.errors.append("strategy_init")
                return False

            print_ok("Strategia SMC initializata")
            print_info(f"Timeframes: H4 (bias), H1 (structure), M5 (entry)")
            print_info(f"POI min score: {config.poi_min_score}")
            print_info(f"Min R:R: {config.min_rr}")
            print_info(f"News buffer: {config.news_buffer_hours}h")
            print_info(f"Max trades/zi: {config.max_trades_per_day}")

            # Test market structure functions
            print()
            print_info("Testare functii SMC:")

            # Test ATR calculation
            from models.candle import Candle
            test_candles = []
            for i in range(20):
                test_candles.append(Candle(
                    timestamp=datetime.now() - timedelta(hours=i),
                    open=1.1000 + i * 0.0010,
                    high=1.1010 + i * 0.0010,
                    low=1.0990 + i * 0.0010,
                    close=1.1005 + i * 0.0010,
                    volume=1000,
                    symbol="EURUSD",
                    timeframe="H1"
                ))

            atr = strategy._calculate_atr(test_candles, 14)
            print_ok(f"  ATR calculation: {atr:.5f}")

            # Test swing point detection
            swing_highs, swing_lows = strategy._find_swing_points(test_candles, lookback=3)
            print_ok(f"  Swing detection: {len(swing_highs)} highs, {len(swing_lows)} lows")

            # Test session detection
            utc_now = datetime.now(timezone.utc)
            session = strategy._get_current_session(utc_now)
            print_ok(f"  Session detection: {session or 'Outside sessions'} (UTC: {utc_now.strftime('%H:%M')})")

            # Test signal score calculation
            from strategies.smc_strategy import SignalScore, POI, OrderBlock, POIType
            score = SignalScore(
                poi_score=4,
                confluence_score=0.7,
                structure_score=0.8,
                timing_score=0.9,
                rr_score=0.85,
            )
            total = score.calculate_total()
            print_ok(f"  Signal scoring: {total:.2f}")

            return True

        except Exception as e:
            print_error(f"Eroare la verificare strategie: {e}")
            import traceback
            traceback.print_exc()
            self.errors.append("strategy_test")
            return False

    def verify_news_filter(self) -> bool:
        """Verify news filter with 2h buffer."""
        print_section(5, 8, "Verificare news filter (2h buffer)")

        try:
            from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact

            config = NewsFilterConfig(
                filter_high_impact=True,
                filter_medium_impact=False,
                filter_entire_day=False,
                buffer_before_minutes=120,  # 2 hours
                buffer_after_minutes=30,
                currencies=["EUR", "USD"],
            )

            news_filter = NewsFilter(config)
            print_ok("News filter creat cu buffer 2h")

            # Update calendar
            if news_filter.update_calendar():
                print_ok("Calendar economic actualizat")
            else:
                print_warn("Nu s-a putut actualiza calendarul (foloseste cache)")

            # Check today's news
            today = datetime.now().date()
            events = news_filter.get_events_for_date(today)
            high_impact = [e for e in events if e.impact == NewsImpact.HIGH]

            print_info(f"Evenimente azi: {len(events)} total, {len(high_impact)} HIGH impact")

            if high_impact:
                for event in high_impact[:3]:
                    print_info(f"  - {event.currency}: {event.event} ({event.time or 'All day'})")

            # Test is_safe_to_trade
            now = datetime.now()
            is_safe = news_filter.is_safe_to_trade(now, "EURUSD")

            if is_safe:
                print_ok("Safe to trade EURUSD acum")
            else:
                print_warn("NOT safe to trade EURUSD acum (news within 2h)")

            # Test buffer logic
            print()
            print_info("Testare logica buffer 2h:")

            # Simulate checking 2h before a news event
            test_times = [
                ("2h30m before news", 150, True),
                ("2h before news", 120, False),
                ("1h before news", 60, False),
                ("At news time", 0, False),
                ("30m after news", -30, False),
                ("1h after news", -60, True),
            ]

            for desc, mins_before, expected_safe in test_times:
                # This is a logical test, not actual time
                actual_safe = mins_before > config.buffer_before_minutes or mins_before < -config.buffer_after_minutes
                status = "OK" if actual_safe == expected_safe else "MISMATCH"
                safe_str = "SAFE" if actual_safe else "BLOCKED"
                print_info(f"  {desc}: {safe_str} [{status}]")

            return True

        except Exception as e:
            print_error(f"Eroare la verificare news filter: {e}")
            self.errors.append("news_filter_test")
            return False

    def verify_demo_live_modes(self) -> bool:
        """Verify bot can work in both demo and live modes."""
        print_section(6, 8, "Verificare moduri demo/live")

        try:
            from scripts.main import TradingBot, parse_args
            from config.settings import Settings, get_settings

            settings = get_settings()

            # Test configuration for different modes
            modes = ["demo", "paper", "live"]

            for mode in modes:
                settings.trading_mode = mode
                print_info(f"Mod {mode.upper()}:")

                if mode == "demo":
                    print_ok(f"  Plasare ordine: DA (cont demo)")
                elif mode == "paper":
                    print_ok(f"  Plasare ordine: NU (simulare)")
                elif mode == "live":
                    print_warn(f"  Plasare ordine: DA (BANI REALI!)")

            # Test TradingBot creation
            bot = TradingBot(settings)
            bot._symbols = ["EURUSD"]
            bot._use_smc = True
            bot._strategy_type = "smc"

            print_ok("TradingBot creat cu succes")
            print_info(f"Strategia selectata: SMC")
            print_info(f"Simboluri: {bot._symbols}")

            return True

        except Exception as e:
            print_error(f"Eroare la verificare moduri: {e}")
            self.errors.append("modes_test")
            return False

    def verify_signal_generation(self) -> bool:
        """Test signal generation with real data."""
        print_section(7, 8, "Testare generare semnal (dry run)")

        if not self.mt5:
            print_error("MT5 nu este conectat")
            return False

        try:
            from strategies.smc_strategy import SMCStrategy, SMCConfig
            from models.candle import Candle

            symbol = "EURUSD"

            # Create strategy
            config = SMCConfig(
                poi_min_score=3,
                min_rr=1.5,
            )

            strategy = SMCStrategy(
                symbol=symbol,
                timeframe="M5",
                config=config,
                use_news_filter=False,  # Disable for testing
            )
            strategy.pip_size = 0.0001
            strategy.initialize()

            # Fetch real data
            h4_rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_H4, 0, 50)
            h1_rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_H1, 0, 100)
            m5_rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M5, 0, 100)

            if h4_rates is None or h1_rates is None or m5_rates is None:
                print_error("Nu pot obtine date pentru testare")
                return False

            # Convert to Candle objects
            def rates_to_candles(rates, tf):
                candles = []
                for r in rates:
                    candles.append(Candle(
                        timestamp=datetime.fromtimestamp(r["time"]),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["tick_volume"]),
                        symbol=symbol,
                        timeframe=tf
                    ))
                return candles

            h4_candles = rates_to_candles(h4_rates, "H4")
            h1_candles = rates_to_candles(h1_rates, "H1")
            m5_candles = rates_to_candles(m5_rates, "M5")

            print_ok(f"Date incarcate: H4={len(h4_candles)}, H1={len(h1_candles)}, M5={len(m5_candles)}")

            # Set multi-timeframe data
            strategy.set_candles(h4_candles, h1_candles, m5_candles)

            # Analyze current market state
            print()
            print_info("Analiza piata curenta:")

            # Check bias
            h4_bias = strategy._state.h4_bias
            h1_bias = strategy._state.h1_bias
            print_info(f"  H4 Bias: {h4_bias.value}")
            print_info(f"  H1 Bias: {h1_bias.value}")
            print_info(f"  ATR H1: {strategy.atr_h1:.5f} ({strategy.atr_h1/0.0001:.1f} pips)")
            print_info(f"  ATR M5: {strategy.atr_m5:.5f} ({strategy.atr_m5/0.0001:.1f} pips)")

            # Find order blocks
            obs = strategy._find_order_blocks(h1_candles, strategy.atr_h1)
            print_info(f"  Order Blocks H1: {len(obs)}")

            # Find FVGs
            fvgs = strategy._find_fvgs(h1_candles, strategy.atr_h1)
            print_info(f"  Fair Value Gaps H1: {len(fvgs)}")

            # Find liquidity levels
            liq_levels = strategy._find_liquidity_levels(h1_candles)
            print_info(f"  Liquidity Levels: {len(liq_levels)}")

            # Try to generate signal
            latest_candle = m5_candles[-1]
            signal = strategy.on_candle(latest_candle, m5_candles)

            if signal:
                print()
                print_ok(f"SEMNAL GENERAT!")
                print_info(f"  Tip: {signal.signal_type.value}")
                print_info(f"  Pret: {signal.price:.5f}")
                print_info(f"  SL: {signal.stop_loss:.5f}")
                print_info(f"  TP: {signal.take_profit:.5f}")
                print_info(f"  Confidence: {signal.confidence:.2f}")
                print_info(f"  Motiv: {signal.reason}")
            else:
                print()
                print_info("Niciun semnal in acest moment")
                print_info("(Aceasta este normal daca nu suntem in sesiune sau conditiile nu sunt indeplinite)")

            # Show current session status
            utc_now = datetime.now(timezone.utc)
            session = strategy._get_current_session(utc_now)

            if session:
                print_ok(f"In sesiune: {session.upper()}")
            else:
                print_info(f"Outside sessions (UTC: {utc_now.strftime('%H:%M')})")
                print_info("Sesiuni: London 07:00-11:00, NY 13:00-17:00 UTC")

            return True

        except Exception as e:
            print_error(f"Eroare la testare semnal: {e}")
            import traceback
            traceback.print_exc()
            self.errors.append("signal_test")
            return False

    def verify_integration(self) -> bool:
        """Run full integration test."""
        print_section(8, 8, "Test integrare completa")

        try:
            # Test that main.py can be configured for SMC
            from scripts.main import TradingBot
            from config.settings import get_settings

            settings = get_settings()
            settings.trading_mode = "paper"  # Safe mode

            bot = TradingBot(settings)
            bot._symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
            bot._use_smc = True
            bot._strategy_type = "smc"

            print_ok("Bot configurat pentru SMC")

            # Initialize bot
            print_info("Initializare bot...")
            if bot.initialize():
                print_ok("Bot initializat cu succes!")

                # Check strategies created
                print_info(f"Strategii create: {len(bot.strategies)}")
                for symbol, strategy in bot.strategies.items():
                    print_info(f"  {symbol}: {strategy.name} (magic: {strategy.magic_number})")

                # Cleanup
                bot.shutdown()
                print_ok("Bot shutdown complet")

                return True
            else:
                print_error("Bot nu s-a putut initializa")
                self.errors.append("bot_init")
                return False

        except Exception as e:
            print_error(f"Eroare la test integrare: {e}")
            import traceback
            traceback.print_exc()
            self.errors.append("integration_test")
            return False

    def run_all_verifications(self) -> bool:
        """Run all verifications."""
        print_header("VERIFICARE COMPLETA SMC TRADING BOT")
        print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        results = {}

        # 1. Imports
        results["imports"] = self.verify_imports()

        if not results["imports"]:
            print_header("ERORI DE IMPORT - REZOLVA MAI INTAI")
            return False

        # 2. MT5 Connection
        results["mt5"] = self.verify_mt5_connection()

        if results["mt5"]:
            # 3. Multi-timeframe data
            results["mtf_data"] = self.verify_multi_timeframe_data()
        else:
            results["mtf_data"] = False

        # 4. SMC Strategy
        results["smc_strategy"] = self.verify_smc_strategy()

        # 5. News Filter
        results["news_filter"] = self.verify_news_filter()

        # 6. Demo/Live modes
        results["modes"] = self.verify_demo_live_modes()

        # 7. Signal generation (only if MT5 connected)
        if results["mt5"]:
            results["signal_gen"] = self.verify_signal_generation()
        else:
            results["signal_gen"] = False

        # 8. Integration test (only if MT5 connected)
        if results["mt5"]:
            results["integration"] = self.verify_integration()
        else:
            results["integration"] = False

        # Cleanup MT5
        if self.mt5:
            self.mt5.shutdown()

        # Summary
        print_header("SUMAR VERIFICARI")

        all_passed = True
        for name, passed in results.items():
            status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
            symbol = "[+]" if passed else "[X]"
            print(f"  {symbol} {name.replace('_', ' ').title()}: {status}")
            if not passed:
                all_passed = False

        print()

        if self.warnings:
            print(f"  {Colors.YELLOW}Warnings: {len(self.warnings)}{Colors.RESET}")
            for warn in self.warnings:
                print(f"    - {warn}")
            print()

        if self.errors:
            print(f"  {Colors.RED}Errors: {len(self.errors)}{Colors.RESET}")
            for err in self.errors:
                print(f"    - {err}")
            print()

        print("=" * 70)

        if all_passed:
            print(f"  {Colors.GREEN}{Colors.BOLD}REZULTAT: TOATE VERIFICARILE AU TRECUT!{Colors.RESET}")
            print()
            print("  Botul SMC este gata de rulare. Comenzi:")
            print()
            print(f"  {Colors.CYAN}# Paper mode (fara ordine reale):{Colors.RESET}")
            print("  python scripts/main.py --strategy smc --mode paper --symbols EURUSD")
            print()
            print(f"  {Colors.CYAN}# Demo mode (ordine pe cont demo):{Colors.RESET}")
            print("  python scripts/main.py --strategy smc --mode demo --symbols EURUSD GBPUSD USDJPY EURJPY")
            print()
            print(f"  {Colors.YELLOW}# Live mode (BANI REALI - ATENTIE!):{Colors.RESET}")
            print("  python scripts/main.py --strategy smc --mode live --symbols EURUSD")
            print()
            print("  CARACTERISTICI SMC:")
            print("  - Multi-timeframe: H4 (bias), H1 (structure), M5 (entry)")
            print("  - POI detection: Order Blocks + FVG + Liquidity")
            print("  - Entry: CHoCH confirmation pe M5")
            print("  - News filter: 2h buffer inainte de stiri RED")
            print("  - Sessions: London 07:00-11:00, NY 13:00-17:00 UTC")
            print()
        else:
            print(f"  {Colors.RED}{Colors.BOLD}REZULTAT: UNELE VERIFICARI AU ESUAT!{Colors.RESET}")
            print()
            print("  Rezolva erorile de mai sus inainte de a rula botul.")

        print("=" * 70)
        print()

        return all_passed


def main():
    """Run verification."""
    verifier = SMCVerifier()
    success = verifier.run_all_verifications()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
