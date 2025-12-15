"""
Trading Bot Main Entry Point

This is the main entry point for the Trading Bot.
Supports both MetaTrader 5 and NinjaTrader platforms.

Available Strategies:
- ORB (Open Range Breakout): Original strategy with larger SL
- LOCB (London Opening Candle Breakout): Optimized for shorter trades with ~6 pip SL

Usage:
    python main.py                              # Run with default (LOCB, 4 pairs, demo)
    python main.py --broker mt5                 # Run with MetaTrader 5
    python main.py --broker ninja               # Run with NinjaTrader
    python main.py --mode demo                  # Run on demo account (default)
    python main.py --mode paper                 # Run in paper mode - no real orders
    python main.py --mode live                  # Run on live account - REAL MONEY!
    python main.py --symbols EURUSD GBPUSD      # Trade specific symbols
    python main.py --strategy locb              # Use LOCB strategy (default)
    python main.py --strategy orb               # Use original ORB strategy

Default symbols: EURUSD, GBPUSD, USDJPY, EURJPY (4 pairs for more opportunities)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import signal
import time
from datetime import datetime
from typing import Optional

from config.settings import Settings, get_settings
from brokers.base_broker import BaseBroker
from brokers.mt5_broker import MT5Broker
from brokers.ninjatrader_broker import NinjaTraderBroker
from strategies.orb_strategy import ORBStrategy
from strategies.locb_strategy import LOCBStrategy
from models.order import Order
from utils.logger import setup_logging, get_logger
from utils.risk_manager import RiskManager
from utils.time_utils import (
    is_trading_hours,
    seconds_to_next_candle,
    time_to_session_end,
)

logger = get_logger(__name__)


class TradingBot:
    """
    Main Trading Bot class.

    Orchestrates the connection to broker, strategy execution,
    and trade management.

    Supports multiple symbols for increased trading opportunities.
    """

    def __init__(self, settings: Settings):
        """
        Initialize trading bot.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.broker: Optional[BaseBroker] = None
        self.strategies = {}  # Dict of symbol -> strategy
        self.risk_manager: Optional[RiskManager] = None
        self._running = False
        self._last_candle_times = {}  # Dict of symbol -> last candle time
        self._use_locb = False  # Flag to use LOCB strategy
        self._symbols = []  # List of symbols to trade

    def _create_broker(self) -> BaseBroker:
        """Create broker based on settings."""
        if self.settings.active_broker == "mt5":
            logger.info("Creating MetaTrader 5 broker connection...")
            return MT5Broker(
                login=self.settings.mt5.login,
                password=self.settings.mt5.password,
                server=self.settings.mt5.server,
                path=self.settings.mt5.path,
                timeout=self.settings.mt5.timeout,
            )
        elif self.settings.active_broker == "ninjatrader":
            logger.info("Creating NinjaTrader broker connection...")
            return NinjaTraderBroker(
                host=self.settings.ninjatrader.host,
                port=self.settings.ninjatrader.port,
                account=self.settings.ninjatrader.account,
            )
        else:
            raise ValueError(f"Unknown broker: {self.settings.active_broker}")

    def _create_strategy_for_symbol(self, symbol: str, magic_offset: int = 0):
        """Create strategy for a specific symbol."""
        # Symbol-specific pip sizes
        pip_sizes = {
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
            "USDJPY": 0.01,
            "EURJPY": 0.01,
            "AUDUSD": 0.0001,
            "USDCAD": 0.0001,
            "USDCHF": 0.0001,
        }
        pip_size = pip_sizes.get(symbol, 0.0001)

        if self._use_locb:
            strategy = LOCBStrategy(
                symbol=symbol,
                timeframe="M1",  # LOCB uses M1 for entries
                magic_number=12346 + magic_offset,
                # Server time settings (Teletrade MT5 uses UTC+4 equivalent)
                london_open_hour=12,  # 08:00 UTC = 12:00 server
                session_end_hour=15,  # 11:00 UTC = 15:00 server
                timezone="Etc/GMT-4",
                # Optimized parameters from backtest (best config)
                risk_reward_ratio=1.5,
                sl_buffer_pips=2.0,
                min_range_pips=2.0,
                max_range_pips=30.0,
                max_retest_candles=20,  # Optimized: was 30
                max_confirm_candles=10,  # Optimized: was 20
                retest_tolerance_pips=3.0,
                max_trades_per_day=1,
            )
            strategy.pip_size = pip_size
            return strategy
        else:
            strategy = ORBStrategy(
                symbol=symbol,
                timeframe=self.settings.default_timeframe,
                magic_number=12345 + magic_offset,
                session_start=self.settings.orb.session_start,
                session_end=self.settings.orb.session_end,
                timezone=self.settings.orb.timezone,
                range_minutes=self.settings.orb.range_minutes,
                breakout_buffer_pips=self.settings.orb.breakout_buffer_pips,
                use_atr_filter=self.settings.orb.use_atr_filter,
                min_range_pips=self.settings.orb.min_range_pips,
                max_range_pips=self.settings.orb.max_range_pips,
            )
            strategy.pip_size = pip_size
            return strategy

    def _create_strategies(self):
        """Create strategies for all symbols."""
        strategy_type = "LOCB" if self._use_locb else "ORB"
        logger.info(f"Creating {strategy_type} strategies for {len(self._symbols)} symbols...")

        for i, symbol in enumerate(self._symbols):
            self.strategies[symbol] = self._create_strategy_for_symbol(symbol, magic_offset=i)
            self._last_candle_times[symbol] = None
            logger.info(f"  - {symbol}: {strategy_type} strategy created")

    def _create_risk_manager(self) -> RiskManager:
        """Create risk manager with settings."""
        from utils.risk_manager import PropAccountLimits

        limits = PropAccountLimits(
            max_daily_drawdown=self.settings.risk.max_daily_loss,
            max_positions=self.settings.risk.max_positions,
            normal_risk_per_trade=self.settings.risk.max_risk_per_trade,
        )
        return RiskManager(limits=limits)

    def initialize(self) -> bool:
        """
        Initialize the trading bot.

        Returns:
            True if initialization successful, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("Initializing Trading Bot...")
        logger.info("=" * 60)

        # Create components
        try:
            self.broker = self._create_broker()
            self._create_strategies()  # Create strategies for all symbols
            self.risk_manager = self._create_risk_manager()
        except Exception as e:
            logger.error(f"Failed to create components: {e}")
            return False

        # Connect to broker
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            return False

        # Get account info
        account_info = self.broker.get_account_info()
        logger.info(f"Account: {account_info.get('login', 'N/A')}")
        logger.info(f"Balance: {account_info.get('balance', 0):.2f}")
        logger.info(f"Equity: {account_info.get('equity', 0):.2f}")

        # Initialize risk manager with current balance
        self.risk_manager.initialize_daily_stats(account_info.get("balance", 0))

        # Initialize all strategies and verify symbols
        for symbol, strategy in self.strategies.items():
            if not strategy.initialize():
                logger.error(f"Failed to initialize strategy for {symbol}")
                return False

            # Get symbol info and update pip size
            symbol_info = self.broker.get_symbol_info(symbol)
            if symbol_info:
                strategy.pip_size = symbol_info.get("pip_size", strategy.pip_size)
                logger.info(f"  {symbol}: pip_size={strategy.pip_size}, min_lot={symbol_info.get('volume_min', 0.01)}")

        strategy_name = "LOCB (London Opening Candle Breakout)" if self._use_locb else f"ORB ({self.settings.orb.range_minutes} min)"
        logger.info("=" * 60)
        logger.info("Trading Bot initialized successfully!")
        logger.info(f"Mode: {self.settings.trading_mode.upper()}")
        if self.settings.trading_mode == "demo":
            logger.info("  >> DEMO MODE: Real orders will be placed on demo account!")
        elif self.settings.trading_mode == "paper":
            logger.info("  >> PAPER MODE: No real orders, simulation only")
        elif self.settings.trading_mode == "live":
            logger.info("  >> LIVE MODE: REAL MONEY - BE CAREFUL!")
        logger.info(f"Broker: {self.settings.active_broker}")
        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Symbols: {', '.join(self._symbols)}")
        if self._use_locb:
            first_strategy = list(self.strategies.values())[0]
            logger.info(f"  - R:R Ratio: {first_strategy.risk_reward_ratio}:1")
            logger.info(f"  - Session: 08:00-11:00 UTC (12:00-15:00 MT5 server time)")
            logger.info(f"  - Avg SL: ~6 pips")
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
        """Single trading cycle - processes all symbols."""
        # Check if within trading hours using MT5 server time
        if self._use_locb:
            # LOCB uses MT5 server time directly (Teletrade server time)
            # London opens at 08:00 UTC = 12:00 server time
            session_start_hour = 12
            session_end_hour = 15

            # Get MT5 server time for accurate trading hours check
            server_time = self.broker.get_server_time()
            current_hour = server_time.hour

            if current_hour < session_start_hour or current_hour >= session_end_hour:
                # Calculate time until session starts
                if current_hour >= session_end_hour:
                    # Session ended, wait until tomorrow
                    hours_until = 24 - current_hour + session_start_hour
                else:
                    hours_until = session_start_hour - current_hour
                logger.debug(f"Outside trading hours. Server time: {server_time.strftime('%H:%M:%S')}. Session starts at {session_start_hour}:00")
                time.sleep(60)
                return
        else:
            session_start = self.settings.orb.session_start
            session_end = self.settings.orb.session_end
            tz = self.settings.orb.timezone

            if not is_trading_hours(session_start, session_end, tz):
                remaining = time_to_session_end(session_start, tz)
                logger.debug(f"Outside trading hours. Next session in: {remaining}")
                time.sleep(60)
                return

        # Update account balance
        balance = self.broker.get_account_balance()
        self.risk_manager.update_balance(balance)

        # Process each symbol
        for symbol in self._symbols:
            self._process_symbol(symbol)

        # Small delay before next cycle
        time.sleep(1)

    def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol for trading signals."""
        strategy = self.strategies.get(symbol)
        if not strategy:
            return

        # Check risk limits - count all open positions
        all_positions = []
        for sym in self._symbols:
            all_positions.extend(self.broker.get_positions(sym))

        can_trade, reason = self.risk_manager.can_trade(len(all_positions))
        if not can_trade:
            logger.debug(f"Trading restricted for {symbol}: {reason}")
            return

        # Get latest candles
        timeframe = "M1" if self._use_locb else self.settings.default_timeframe
        candles = self.broker.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            count=100,
        )

        if not candles:
            logger.debug(f"No candle data for {symbol}")
            return

        latest_candle = candles[-1]

        # Check if new candle for this symbol
        if self._last_candle_times.get(symbol) == latest_candle.timestamp:
            return

        self._last_candle_times[symbol] = latest_candle.timestamp
        logger.debug(f"[{symbol}] New candle: {latest_candle.timestamp} | C:{latest_candle.close:.5f}")

        # Get positions for this symbol
        positions = self.broker.get_positions(symbol)

        # Check for exit signals on existing positions
        for position in positions:
            exit_signal = strategy.should_exit(
                position, latest_candle.close, candles
            )
            if exit_signal:
                logger.info(f"[{symbol}] Exit signal: {exit_signal.reason}")
                self._execute_exit(position)

        # Check for entry signals (only if no positions for this symbol)
        symbol_positions = [p for p in positions if p.symbol == symbol]
        if not symbol_positions:
            signal = strategy.on_candle(latest_candle, candles)

            if signal and signal.is_entry:
                logger.info(f"[{symbol}] Entry signal: {signal.signal_type.value} - {signal.reason}")
                self._execute_entry(signal, symbol)

    def _execute_entry(self, signal, symbol: str = None) -> None:
        """
        Execute entry order based on signal.

        Args:
            signal: Strategy signal.
            symbol: Trading symbol (uses signal.symbol if not provided).
        """
        symbol = symbol or signal.symbol
        strategy = self.strategies.get(symbol)

        if self.settings.trading_mode == "paper":
            logger.info(f"[PAPER] [{symbol}] Would enter {signal.signal_type.value} at {signal.price}")
            logger.info(f"[PAPER] [{symbol}] SL: {signal.stop_loss}, TP: {signal.take_profit}")
            return

        # Calculate position size
        symbol_info = self.broker.get_symbol_info(symbol)
        balance = self.broker.get_account_balance()

        if signal.stop_loss and strategy:
            sl_pips = abs(signal.price - signal.stop_loss) / strategy.pip_size
            lot_size = self.risk_manager.calculate_position_size(
                account_balance=balance,
                stop_loss_pips=sl_pips,
                pip_value=symbol_info.get("pip_value", 10),
                min_lot=symbol_info.get("volume_min", 0.01),
                max_lot=symbol_info.get("volume_max", 100),
                lot_step=symbol_info.get("volume_step", 0.01),
            )
        else:
            lot_size = self.settings.default_lot_size

        # Create order
        strategy_prefix = "LOCB" if self._use_locb else "ORB"
        if signal.is_long:
            order = Order.market_buy(
                symbol=symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"{strategy_prefix} Long - {signal.reason}",
            )
        else:
            order = Order.market_sell(
                symbol=symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"{strategy_prefix} Short - {signal.reason}",
            )

        if strategy:
            order.magic_number = strategy.magic_number
            order.strategy_name = strategy.name

        # Execute order
        order_id = self.broker.place_order(order)

        if order_id:
            logger.info(f"[{symbol}] Order placed successfully: {order_id}")
        else:
            logger.error(f"[{symbol}] Failed to place order")

    def _execute_exit(self, position) -> None:
        """
        Execute exit for a position.

        Args:
            position: Position to close.
        """
        if self.settings.trading_mode == "paper":
            logger.info(f"[PAPER] Would close position {position.broker_position_id}")
            return

        success = self.broker.close_position(position.broker_position_id)

        if success:
            logger.info(f"Position closed: {position.broker_position_id}")
        else:
            logger.error(f"Failed to close position: {position.broker_position_id}")

    def shutdown(self) -> None:
        """Shutdown the trading bot."""
        logger.info("Shutting down Trading Bot...")
        self._running = False

        if self.broker:
            # Close all positions for all symbols if configured
            for symbol in self._symbols:
                positions = self.broker.get_positions(symbol)
                strategy = self.strategies.get(symbol)
                if strategy:
                    for position in positions:
                        if position.magic_number == strategy.magic_number:
                            logger.info(f"Closing {symbol} position on shutdown: {position.broker_position_id}")
                            self.broker.close_position(position.broker_position_id)

            self.broker.disconnect()

        logger.info("Trading Bot shutdown complete")

    def stop(self) -> None:
        """Stop the trading bot gracefully."""
        self._running = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ORB Trading Bot for MT5 and NinjaTrader"
    )
    parser.add_argument(
        "--broker",
        choices=["mt5", "ninja"],
        default="mt5",
        help="Broker to use (default: mt5)",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "demo", "paper", "backtest"],
        default="demo",
        help="Trading mode: demo (real orders on demo account), paper (simulated), live (real money). Default: demo",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD"],  # ONLY EURUSD is profitable with LOCB strategy
        help="Trading symbols (default: EURUSD). Use --symbols EURUSD GBPUSD etc.",
    )
    parser.add_argument(
        "--timeframe",
        default="M5",
        help="Timeframe (default: M5)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--strategy",
        choices=["orb", "locb"],
        default="locb",
        help="Strategy to use: orb (Open Range Breakout) or locb (London Opening Candle Breakout). Default: locb for demo mode",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Load settings
    settings = get_settings()

    # Override settings from command line
    if args.broker == "ninja":
        settings.active_broker = "ninjatrader"
    else:
        settings.active_broker = "mt5"

    settings.trading_mode = args.mode
    settings.default_symbol = args.symbols[0]  # First symbol as default
    settings.default_timeframe = args.timeframe

    # Create and run bot
    bot = TradingBot(settings)

    # Set symbols and strategy
    bot._symbols = args.symbols
    bot._use_locb = (args.strategy == "locb")

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize and run
    if not bot.initialize():
        logger.error("Bot initialization failed")
        return 1

    try:
        bot.run()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())