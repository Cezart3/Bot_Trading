"""
SMC V3 Trading Bot Runner

Runs SMC V3 strategy on multiple symbols with priority system:
- EURUSD (primary - London + NY)
- GBPUSD (secondary - London + NY)
- AUDUSD (secondary - NY overlap with Asian close)
- US30 (NY session only)
- USTech100/NASDAQ (NY session only)
- GER30/DAX (London + NY)

Usage:
    python scripts/run_smc_v3.py                    # Default: all symbols, demo mode
    python scripts/run_smc_v3.py --mode demo       # Demo account (default)
    python scripts/run_smc_v3.py --mode paper      # Paper trading (simulation only)
    python scripts/run_smc_v3.py --mode live       # LIVE - REAL MONEY!
    python scripts/run_smc_v3.py --symbols EURUSD  # Single symbol
    python scripts/run_smc_v3.py --risk 0.5        # 0.5% risk per trade

Author: Trading Bot Project
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import signal
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from config.settings import Settings, get_settings
from brokers.mt5_broker import MT5Broker
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType
from models.order import Order
from utils.logger import setup_logging, get_logger
from utils.risk_manager import RiskManager, PropAccountLimits
from utils.news_filter import NewsFilter

logger = get_logger(__name__)


# Symbol configurations with strict/relaxed settings based on backtest results
# GBPUSD: Best performer (PF 1.90, 50% WR) - highest priority
# AUDUSD: Decent (PF 1.10, 35% WR) - moderate settings
# EURUSD: Losing (-$177, 29% WR) - STRICT settings required
# US30: Losing (-$324, 17% WR) - VERY STRICT settings required

SYMBOL_CONFIGS = {
    # GBPUSD - BEST PERFORMER - Priority 1
    "GBPUSD": {
        "type": InstrumentType.FOREX,
        "priority": 3,  # Highest priority - best results
        "sessions": ["london", "ny"],
        "pip_size": 0.0001,
        "min_sl_pips": 12,
        "max_sl_pips": 30,
        # Relaxed settings - keep what works
        "poi_min_score": 1.8,
        "require_sweep": False,
        "adx_trending": 18.0,
        "min_rr": 1.8,
        "ob_min_impulse_atr": 0.7,
    },
    # AUDUSD - DECENT PERFORMER - Priority 2
    "AUDUSD": {
        "type": InstrumentType.FOREX,
        "priority": 2,
        "sessions": ["ny"],  # NY + Asian overlap
        "pip_size": 0.0001,
        "min_sl_pips": 10,
        "max_sl_pips": 25,
        # Moderate settings
        "poi_min_score": 2.2,
        "require_sweep": True,
        "adx_trending": 22.0,
        "min_rr": 1.8,
    },
    # EURUSD - LOSING - Priority 3 (STRICT SETTINGS)
    "EURUSD": {
        "type": InstrumentType.FOREX,
        "priority": 1,  # Lower priority due to losses
        "sessions": ["london", "ny"],
        "pip_size": 0.0001,
        "min_sl_pips": 10,
        "max_sl_pips": 25,
        # STRICT settings - require sweep for entry
        "poi_min_score": 2.5,  # Higher threshold
        "adx_trending": 22.0,
        "min_rr": 1.8,  # Better R:R
        "ob_min_impulse_atr": 0.7,
    },
    "EURGBP": {
        "type": InstrumentType.FOREX,
        "priority": 4,
        "sessions": ["london"],
        "pip_size": 0.0001,
        "min_sl_pips": 8,
        "max_sl_pips": 20,
        "poi_min_score": 2.0,
        "require_sweep": True,
        "adx_trending": 22.0,
        "min_rr": 2.0,
    },
    # US30 - LOSING BADLY - Priority 5 (VERY STRICT)
    "US30": {
        "type": InstrumentType.INDEX,
        "priority": 5,
        "sessions": ["ny"],
        "pip_size": 1.0,  # Points, not pips
        "min_sl_pips": 15,  # Increased from 5
        "max_sl_pips": 35,  # Increased from 20
        # VERY STRICT settings
        "poi_min_score": 2.5,  # Very high threshold
        "require_sweep": True,  # MUST have sweep
        "adx_trending": 25.0,
        "min_rr": 2.5,  # High R:R
    },
    # USTech100/USTEC - alternative names for NASDAQ
    "USTEC": {
        "type": InstrumentType.INDEX,
        "priority": 6,
        "sessions": ["ny"],
        "pip_size": 0.1,
        "min_sl_pips": 50,
        "max_sl_pips": 150,
        "poi_min_score": 2.5,
        "require_sweep": True,
        "adx_trending": 18.0, # RELAXED from 25.0
        "min_rr": 2.0,
    },
    "NAS100": {
        "type": InstrumentType.INDEX,
        "priority": 6,
        "sessions": ["ny"],
        "pip_size": 0.1,
        "min_sl_pips": 50,
        "max_sl_pips": 150,
        "poi_min_score": 2.5,
        "require_sweep": True,
        "adx_trending": 18.0, # RELAXED from 25.0
        "min_rr": 2.0,
    },
    # GER30/GER40/DE40 - multiple aliases
    "GER40": {
        "type": InstrumentType.INDEX,
        "priority": 7,
        "sessions": ["london", "ny"],
        "pip_size": 0.1,
        "min_sl_pips": 30,
        "max_sl_pips": 80,
        "poi_min_score": 2.5,
        "require_sweep": True,
        "adx_trending": 18.0, # RELAXED from 25.0
        "min_rr": 2.0,
    },
    "GER30": {
        "type": InstrumentType.INDEX,
        "priority": 7,
        "sessions": ["london", "ny"],
        "pip_size": 0.1,
        "min_sl_pips": 30,
        "max_sl_pips": 80,
        "poi_min_score": 2.5,
        "require_sweep": True,
        "adx_trending": 18.0, # RELAXED from 25.0
        "min_rr": 2.0,
    },
    "DE40": {
        "type": InstrumentType.INDEX,
        "priority": 7,
        "sessions": ["london", "ny"],
        "pip_size": 0.1,
        "min_sl_pips": 30,
        "max_sl_pips": 80,
        "poi_min_score": 2.5,
        "require_sweep": True,
        "adx_trending": 18.0, # RELAXED from 25.0
        "min_rr": 2.0,
    },
}

# Default symbols to trade (will be auto-detected in MT5)
DEFAULT_SYMBOLS = ["EURUSD", "AUDUSD", "GBPUSD", "US30", "USTEC", "GER40"]

# Alternative symbol names to try if default not found
SYMBOL_ALIASES = {
    "NAS100": ["USTEC", "USTech100", "NASDAQ", "NAS100.cash", "NAS100Cash", "US100", "NASDAQ100"],
    "USTEC": ["NAS100", "USTech100", "NASDAQ", "NAS100.cash", "US100", "NASDAQ100"],
    "GER30": ["GER40", "DE40", "DAX", "GER30.cash", "GER40.cash", "DE30", "DE40"],
    "GER40": ["GER30", "DE40", "DAX", "GER40.cash", "GER30.cash", "DE30"],
    "US30": ["DJI30", "DOW30", "US30.cash", "US30Cash", "DJ30"],
}


class SMCV3TradingBot:
    """
    SMC V3 Multi-Symbol Trading Bot.

    Features:
    - Multiple symbol monitoring with priority system
    - Percentage-based risk management
    - Session-aware trading
    - Intelligent signal selection
    """

    def __init__(
        self,
        symbols: List[str],
        risk_percent: float = 0.5,
        max_trades_per_day: int = 2,
        mode: str = "demo"
    ):
        self.symbols = symbols
        self.risk_percent = risk_percent
        self.max_trades_per_day = max_trades_per_day
        self.mode = mode

        self.broker: Optional[MT5Broker] = None
        self.strategies: Dict[str, SMCStrategyV3] = {}
        self.risk_manager: Optional[RiskManager] = None
        self.news_filter: Optional[NewsFilter] = None

        self._running = False
        self._last_candle_times: Dict[str, datetime] = {}
        self._trades_today = 0
        self._current_date = None
        self._last_heartbeat = None
        self._last_outside_session_log_time = None # New attribute
        self._scan_count = 0
        self._signals_found = 0

    def _create_strategy_for_symbol(self, symbol: str) -> SMCStrategyV3:
        """Create SMC V3 strategy for a symbol with optimized per-symbol settings."""
        config_data = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS.get("GBPUSD", {}))

        # Get per-symbol settings with defaults
        poi_min_score = config_data.get("poi_min_score", 2.0)
        require_sweep = config_data.get("require_sweep", False)
        adx_trending = config_data.get("adx_trending", 22.0)
        min_rr = config_data.get("min_rr", 1.5)
        ob_min_impulse_atr = config_data.get("ob_min_impulse_atr", 0.8)

        # Determine instrument type BEFORE creating config
        instrument_type = config_data.get("type", InstrumentType.FOREX)

        # Base config for all symbols
        base_config = {
            "instrument_type": instrument_type,
            "min_sl_pips": config_data.get("min_sl_pips", 10),
            "max_sl_pips": config_data.get("max_sl_pips", 30),

            # Risk settings - PERCENTAGE BASED
            "risk_percent": self.risk_percent,
            "max_trades_per_day": self.max_trades_per_day,

            # Use partial TPs for better risk management
            "use_partial_tp": True,
            "tp1_rr": 1.0,
            "tp1_percent": 50,
            "tp2_rr": 2.0,
            "tp2_percent": 30,
            "tp3_rr": 3.0,
            "tp3_percent": 20,

            # Time exit
            "use_time_exit": True,
            "time_exit_hours": 4,

            # Equity curve trading
            "use_equity_curve": True,
            "equity_losses_reduce": 2,
            "equity_reduced_risk": 0.25,  # Half of normal

            # Per-symbol optimized settings
            "poi_min_score": poi_min_score,
            "require_sweep_for_low_score": require_sweep,
            "adx_trending": adx_trending,
            "min_rr": min_rr,
            "ob_min_impulse_atr": ob_min_impulse_atr,
        }

        # Adjust config based on instrument type
        if instrument_type == InstrumentType.INDEX:
            # Config for indices
            base_config.update({
                "use_strict_kill_zones": False,
                "require_displacement": True,
                "displacement_min_atr": 1.2,
                "use_adx_filter": True,
                "adx_weak_trend": adx_trending - 5,
                "use_volatility_filter": False,
                "require_multi_candle": False,
                "ny_start_hour": 14, # 14:30 UTC
                "ny_end_hour": 20,
                "index_skip_first_minutes": 15,
                # For indices with require_sweep, always require it
                "sweep_score_threshold": poi_min_score if require_sweep else 999,
            })
        else:
            # Config for forex
            base_config.update({
                "use_strict_kill_zones": True,
                "require_displacement": True,
                "displacement_min_atr": 1.2,
                "use_adx_filter": True,
                "adx_weak_trend": adx_trending - 5,
                # For forex with require_sweep, set threshold appropriately
                "sweep_score_threshold": poi_min_score if require_sweep else 999,
            })

        logger.debug(f"[{symbol}] Strategy config: POI={poi_min_score}, Sweep={require_sweep}, ADX={adx_trending}, RR={min_rr}")


        config = SMCConfigV3(**base_config)

        strategy = SMCStrategyV3(
            symbol=symbol,
            timeframe="M1",
            magic_number=12370 + list(SYMBOL_CONFIGS.keys()).index(symbol) if symbol in SYMBOL_CONFIGS else 12370,
            config=config
        )

        return strategy

    def _get_current_session(self) -> Optional[str]:
        """Get current trading session."""
        utc_now = datetime.now(timezone.utc)
        hour = utc_now.hour
        minute = utc_now.minute

        # London Kill Zone: 08:00-11:00 UTC
        if 8 <= hour < 12:
            return "london"
        # NY Kill Zone: 14:30-17:00 UTC
        elif hour == 14 and minute >= 30 or 15 <= hour < 17:
             return "ny"
        return None

    def _find_symbol_in_mt5(self, symbol: str) -> Optional[str]:
        """Find a symbol in MT5, trying aliases if necessary."""
        # First try the exact symbol
        symbol_info = self.broker.get_symbol_info(symbol)
        if symbol_info:
            # Make sure it's selected in Market Watch
            self.broker.select_symbol(symbol)
            return symbol

        # Try aliases
        aliases = SYMBOL_ALIASES.get(symbol, [])
        for alias in aliases:
            symbol_info = self.broker.get_symbol_info(alias)
            if symbol_info:
                self.broker.select_symbol(alias)
                logger.info(f"Found alias {alias} for {symbol}")
                return alias

        return None

    def _symbol_can_trade_now(self, symbol: str) -> bool:
        """Check if symbol can trade in current session."""
        session = self._get_current_session()
        if not session:
            return False

        # Check both the actual symbol and any matching config
        config = SYMBOL_CONFIGS.get(symbol, {})
        if not config:
            # Try to find matching config from aliases
            for key, aliases in SYMBOL_ALIASES.items():
                if symbol in aliases:
                    config = SYMBOL_CONFIGS.get(key, {})
                    break

        allowed_sessions = config.get("sessions", ["london", "ny"])

        return session in allowed_sessions

    def initialize(self) -> bool:
        """Initialize the trading bot."""
        logger.info("=" * 60)
        logger.info("SMC V3 TRADING BOT")
        logger.info("=" * 60)
        
        # Initialize News Filter
        logger.info("Initializing News Filter...")
        self.news_filter = NewsFilter()
        self.news_filter.update_calendar()

        # Create MT5 broker
        settings = get_settings()

        self.broker = MT5Broker(
            login=settings.mt5.login,
            password=settings.mt5.password,
            server=settings.mt5.server,
            path=settings.mt5.path,
            timeout=settings.mt5.timeout,
        )

        if not self.broker.connect():
            logger.error("Failed to connect to MT5")
            return False

        # Get account info
        account_info = self.broker.get_account_info()
        balance = account_info.get("balance", 0)

        logger.info(f"Account: {account_info.get('login', 'N/A')}")
        logger.info(f"Balance: ${balance:,.2f}")
        logger.info(f"Equity: ${account_info.get('equity', 0):,.2f}")

        # Create risk manager
        limits = PropAccountLimits(
            max_daily_drawdown=4.0,
            max_account_drawdown=10.0,
            warning_drawdown=2.0,
            normal_risk_per_trade=self.risk_percent,
            reduced_risk_per_trade=self.risk_percent / 2,
            max_positions=self.max_trades_per_day,
        )
        self.risk_manager = RiskManager(initial_account_balance=balance, limits=limits)
        self.risk_manager.initialize_daily_stats(balance)

        # Create strategies for all symbols with alias support
        valid_symbols = []
        for symbol in self.symbols:
            # Try to find the symbol (or an alias) in MT5
            found_symbol = self._find_symbol_in_mt5(symbol)

            if not found_symbol:
                logger.warning(f"Symbol {symbol} and all aliases not found in MT5, skipping...")
                continue

            if found_symbol != symbol:
                logger.info(f"Using {found_symbol} instead of {symbol}")

            strategy = self._create_strategy_for_symbol(found_symbol)
            if strategy.initialize():
                self.strategies[found_symbol] = strategy
                valid_symbols.append(found_symbol)
                symbol_type = SYMBOL_CONFIGS.get(found_symbol, SYMBOL_CONFIGS.get(symbol, {})).get('type', InstrumentType.FOREX)
                logger.info(f"  [OK] {found_symbol} - {symbol_type.value if hasattr(symbol_type, 'value') else symbol_type}")
            else:
                logger.warning(f"  [X] Failed to initialize {found_symbol}")

        if not valid_symbols:
            logger.error("No valid symbols to trade!")
            return False

        self.symbols = valid_symbols

        logger.info("=" * 60)
        logger.info(f"Mode: {self.mode.upper()}")
        logger.info(f"Risk per trade: {self.risk_percent}%")
        logger.info(f"Max trades/day: {self.max_trades_per_day}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        return True

    def run(self) -> None:
        """Main trading loop."""
        self._running = True
        logger.info("Starting trading loop...")

        while self._running:
            try:
                self._trading_cycle()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(5)

        self.shutdown()

    def _trading_cycle(self) -> None:
        """Single trading cycle with detailed logging."""
        utc_now = datetime.now(timezone.utc)

        # Heartbeat logging every 5 minutes
        if self._last_heartbeat is None or (utc_now - self._last_heartbeat).total_seconds() >= 300:
            self._last_heartbeat = utc_now
            session = self._get_current_session()
            positions = []
            for symbol in self.symbols:
                positions.extend(self.broker.get_positions(symbol))

            balance = self.broker.get_account_balance()
            equity = self.broker.get_account_equity()

            logger.info(f"========== HEARTBEAT {utc_now.strftime('%H:%M:%S')} UTC ==========")
            logger.info(f"Session: {session or 'OUTSIDE'} | Scans: {self._scan_count} | Signals: {self._signals_found}")
            logger.info(f"Balance: ${balance:,.2f} | Equity: ${equity:,.2f} | Positions: {len(positions)}")
            logger.info(f"Trades today: {self._trades_today}/{self.max_trades_per_day}")
            logger.info("=" * 50)

        # Check session
        session = self._get_current_session()
        if not session:
                    # Calculate time until next session
                    london_start = utc_now.replace(hour=8, minute=0, second=0, microsecond=0)
                    ny_start = utc_now.replace(hour=14, minute=30, second=0, microsecond=0)
            
                    if utc_now < london_start:
                        next_session_time = london_start
                        next_session_name = "London"
                    elif utc_now < ny_start:
                        next_session_time = ny_start
                        next_session_name = "NY"
                    else:
                        # Next day's London session
                        next_session_time = london_start + timedelta(days=1)
                        next_session_name = "London (tomorrow)"
            
                    time_until = next_session_time - utc_now
                    minutes_until = int(time_until.total_seconds() / 60)
                    hours_until = minutes_until // 60
                    minutes_rem = minutes_until % 60
            
                    # Log outside trading hours message at the start of every 5-minute block
                    # e.g., at :00, :05, :10, :15 etc.
                    if utc_now.minute % 5 == 0 and \
                       (self._last_outside_session_log_time is None or (utc_now - self._last_outside_session_log_time).total_seconds() >= 290): # Allow for slight time drift
                        logger.info(f"Outside trading hours. {next_session_name} in {hours_until}h {minutes_rem}m. Checking again in 1 minute.")
                        self._last_outside_session_log_time = utc_now.replace(second=0, microsecond=0) # Reset to start of minute for precision

                    time.sleep(60) # Check every minute when outside sessions
                    return
        # Check for new day
        if self._current_date != utc_now.date():
            self._current_date = utc_now.date()
            self._trades_today = 0
            self._scan_count = 0
            self._signals_found = 0
            balance = self.broker.get_account_balance()
            self.risk_manager.initialize_daily_stats(balance)
            # Update news calendar daily
            if self.news_filter:
                logger.info("Updating news calendar for the new day...")
                self.news_filter.update_calendar()
            logger.info(f"============ NEW TRADING DAY: {self._current_date} ============")
            logger.info(f"Starting balance: ${balance:,.2f}")
            logger.info(f"Symbols active: {', '.join(self.symbols)}")
            logger.info("=" * 50)

        # Check max trades
        if self._trades_today >= self.max_trades_per_day:
            if self._scan_count % 300 == 0:  # Log every 5 min
                logger.info(f"Max trades reached ({self._trades_today}/{self.max_trades_per_day}) - waiting for next day")
            time.sleep(60)
            return

        # Check risk
        balance = self.broker.get_account_balance()
        self.risk_manager.update_balance(balance)

        all_positions = []
        for symbol in self.symbols:
            all_positions.extend(self.broker.get_positions(symbol))

        can_trade, reason = self.risk_manager.can_trade(len(all_positions))
        if not can_trade:
            logger.warning(f"Trading restricted: {reason}")
            time.sleep(60)
            return

        # Increment scan counter
        self._scan_count += 1

        # Collect signals from all tradeable symbols
        pending_signals = []
        symbols_scanned = []
        symbols_skipped = []

        for symbol in self.symbols:
            if not self._symbol_can_trade_now(symbol):
                symbols_skipped.append(symbol)
                continue

            symbols_scanned.append(symbol)
            signal = self._check_symbol(symbol)
            if signal:
                priority = SYMBOL_CONFIGS.get(symbol, {}).get("priority", 99)
                pending_signals.append((symbol, signal, priority))
                self._signals_found += 1

        # Log scan results every 30 seconds (approximately every 30 cycles at 1 sec sleep)
        if self._scan_count % 30 == 0:
            logger.debug(f"[SCAN #{self._scan_count}] Session: {session} | Scanned: {symbols_scanned} | Skipped: {symbols_skipped}")
            if not pending_signals:
                logger.debug(f"[SCAN #{self._scan_count}] No signals found this cycle")

        # Select best signal (lowest priority number = highest priority)
        if pending_signals and can_trade:
            # Sort by priority (lower is better), then by confidence (higher is better)
            pending_signals.sort(key=lambda x: (x[2], -x[1].confidence))

            best_symbol, best_signal, priority = pending_signals[0]

            logger.info(f">>> SIGNAL SELECTED: {best_symbol} (priority {priority}, confidence {best_signal.confidence:.2f})")
            logger.info(f">>> Type: {best_signal.signal_type.value} | Entry: {best_signal.price:.5f} | SL: {best_signal.stop_loss:.5f} | TP: {best_signal.take_profit:.5f}")

            # News Filter Check
            if self.news_filter and not self.news_filter.is_safe_to_trade(symbol=best_symbol):
                logger.warning(f"[{best_symbol}] Trade SKIPPED due to upcoming high-impact news.")
            else:
                self._execute_entry(best_symbol, best_signal)

        time.sleep(10)

    def _check_symbol(self, symbol: str) -> Optional[object]:
        """Check a symbol for signals with detailed logging."""
        strategy = self.strategies.get(symbol)
        if not strategy:
            return None

        # Get M1 candles as the primary timeframe for the new MTF logic
        m1_candles = self.broker.get_candles(symbol, "M1", 200) # Fetch more M1 candles for history
        if not m1_candles:
            logger.warning(f"[{symbol}] No M1 candle data available")
            return None
        
        latest_m1_candle = m1_candles[-1]

        # Check if new candle
        if self._last_candle_times.get(symbol) == latest_m1_candle.timestamp:
            return None  # Same candle, no need to recheck

        # New candle detected - log it
        is_first_check = self._last_candle_times.get(symbol) is None
        self._last_candle_times[symbol] = latest_m1_candle.timestamp

        if not is_first_check:
            logger.debug(f"[{symbol}] New M1 candle: {latest_m1_candle.timestamp.strftime('%H:%M')} | "
                        f"O:{latest_m1_candle.open:.5f} H:{latest_m1_candle.high:.5f} "
                        f"L:{latest_m1_candle.low:.5f} C:{latest_m1_candle.close:.5f}")

        # Get multi-timeframe data
        m5_candles = self.broker.get_candles(symbol, "M5", 100)
        h1_candles = self.broker.get_candles(symbol, "H1", 100)
        h4_candles = self.broker.get_candles(symbol, "H4", 50)
        daily_candles = self.broker.get_candles(symbol, "D1", 30)

        # Set candles on strategy
        strategy.set_candles(
            h4_candles=h4_candles or [],
            h1_candles=h1_candles or [],
            m5_candles=m5_candles or [],
            m1_candles=m1_candles, # Pass M1 candles
            daily_candles=daily_candles or []
        )

        # Update spread
        bid, ask = self.broker.get_current_price(symbol)
        spread_pips = 0
        if bid and ask:
            strategy.update_spread(ask - bid)
            pip_size = SYMBOL_CONFIGS.get(symbol, {}).get("pip_size", 0.0001)
            spread_pips = (ask - bid) / pip_size

        # Log strategy state for debugging
        status = strategy.get_status()
        if not is_first_check and self._scan_count % 60 == 0:  # Every minute
            logger.debug(f"[{symbol}] ADX: {status.get('adx_value', 0):.1f} | "
                        f"H1 Bias: {status.get('h1_bias', 'N/A')} | "
                        f"EMA Trend: {status.get('ema_trend', 'N/A')} | "
                        f"M5 POIs: {status.get('m5_pois_count', 0)} | "
                        f"Spread: {spread_pips:.1f} pips")

        # Check for existing positions
        positions = self.broker.get_positions(symbol)

        # Check exit signals
        for position in positions:
            exit_signal = strategy.should_exit(position, latest_m1_candle.close, m1_candles)
            if exit_signal:
                logger.info(f"[{symbol}] EXIT SIGNAL: {exit_signal.reason}")
                self._execute_exit(position)

        # Check entry signal (only if no position)
        if not positions:
            signal = strategy.on_candle(latest_m1_candle, m1_candles)
            if signal and signal.is_entry:
                logger.info(f"[{symbol}] ENTRY SIGNAL FOUND: {signal.signal_type.value}")
                logger.info(f"[{symbol}] Reason: {signal.reason}")
                logger.info(f"[{symbol}] Confidence: {signal.confidence:.2f}")
                return signal

        return None

    def _execute_entry(self, symbol: str, signal) -> None:
        """Execute entry order."""
        if self.mode == "paper":
            logger.info(f"[PAPER] [{symbol}] Would enter {signal.signal_type.value} at {signal.price}")
            logger.info(f"[PAPER] SL: {signal.stop_loss}, TP: {signal.take_profit}")
            self._trades_today += 1
            return

        strategy = self.strategies.get(symbol)
        symbol_info = self.broker.get_symbol_info(symbol)
        balance = self.broker.get_account_balance()

        # Calculate position size using PERCENTAGE RISK
        if signal.stop_loss:
            stop_loss_distance = abs(signal.price - signal.stop_loss)
            lot_size = self.risk_manager.calculate_position_size(
                account_balance=balance,
                stop_loss_distance=stop_loss_distance,
                price=signal.price,
                contract_size=symbol_info.get("contract_size", 100000),
                min_qty=symbol_info.get("volume_min", 0.01),
                max_qty=symbol_info.get("volume_max", 100.0),
                qty_step=symbol_info.get("volume_step", 0.01),
            )
        else:
            lot_size = 0.01  # Minimum

        # Create order
        if signal.is_long:
            order = Order.market_buy(
                symbol=symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"SMC_V3 Long - {signal.reason[:30]}",
            )
        else:
            order = Order.market_sell(
                symbol=symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"SMC_V3 Short - {signal.reason[:30]}",
            )

        if strategy:
            order.magic_number = strategy.magic_number
            order.strategy_name = strategy.name

        # Execute
        order_id = self.broker.place_order(order)

        if order_id:
            self._trades_today += 1
            logger.info(f"[{symbol}] Order placed: {order_id} | Lot: {lot_size} | Risk: {self.risk_percent}%")
        else:
            logger.error(f"[{symbol}] Failed to place order")

    def _execute_exit(self, position) -> None:
        """Execute exit."""
        if self.mode == "paper":
            logger.info(f"[PAPER] Would close position {position.broker_position_id}")
            return

        success = self.broker.close_position(position.broker_position_id)
        if success:
            logger.info(f"Position closed: {position.broker_position_id}")
        else:
            logger.error(f"Failed to close position: {position.broker_position_id}")

    def shutdown(self) -> None:
        """Shutdown the bot."""
        logger.info("Shutting down...")
        self._running = False

        if self.broker:
            self.broker.disconnect()

        logger.info("Shutdown complete")

    def stop(self) -> None:
        """Stop gracefully."""
        self._running = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SMC V3 Trading Bot")

    parser.add_argument(
        "--mode",
        choices=["demo", "paper", "live"],
        default="demo",
        help="Trading mode (default: demo)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to trade (default: {' '.join(DEFAULT_SYMBOLS)})"
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=0.5,
        help="Risk percent per trade (default: 0.5)"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=2,
        help="Max trades per day (default: 2)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    setup_logging(level=args.log_level)

    bot = SMCV3TradingBot(
        symbols=args.symbols,
        risk_percent=args.risk,
        max_trades_per_day=args.max_trades,
        mode=args.mode
    )

    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not bot.initialize():
        logger.error("Initialization failed")
        return 1

    try:
        bot.run()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
