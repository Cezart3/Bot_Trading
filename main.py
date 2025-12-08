"""
Trading Bot Main Entry Point

This is the main entry point for the ORB (Open Range Breakout) Trading Bot.
Supports both MetaTrader 5 and NinjaTrader platforms.

Usage:
    python main.py                    # Run with default settings
    python main.py --broker mt5       # Run with MetaTrader 5
    python main.py --broker ninja     # Run with NinjaTrader
    python main.py --mode paper       # Run in paper trading mode
    python main.py --symbol EURUSD    # Trade specific symbol
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from typing import Optional

from config.settings import Settings, get_settings
from brokers.base_broker import BaseBroker
from brokers.mt5_broker import MT5Broker
from brokers.ninjatrader_broker import NinjaTraderBroker
from strategies.orb_strategy import ORBStrategy
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
    """

    def __init__(self, settings: Settings):
        """
        Initialize trading bot.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.broker: Optional[BaseBroker] = None
        self.strategy: Optional[ORBStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self._running = False
        self._last_candle_time: Optional[datetime] = None

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

    def _create_strategy(self) -> ORBStrategy:
        """Create ORB strategy with settings."""
        return ORBStrategy(
            symbol=self.settings.default_symbol,
            timeframe=self.settings.default_timeframe,
            session_start=self.settings.orb.session_start,
            session_end=self.settings.orb.session_end,
            timezone=self.settings.orb.timezone,
            range_minutes=self.settings.orb.range_minutes,
            breakout_buffer_pips=self.settings.orb.breakout_buffer_pips,
            use_atr_filter=self.settings.orb.use_atr_filter,
            min_range_pips=self.settings.orb.min_range_pips,
            max_range_pips=self.settings.orb.max_range_pips,
        )

    def _create_risk_manager(self) -> RiskManager:
        """Create risk manager with settings."""
        return RiskManager(
            max_risk_per_trade=self.settings.risk.max_risk_per_trade,
            max_daily_loss=self.settings.risk.max_daily_loss,
            max_positions=self.settings.risk.max_positions,
            stop_loss_atr_multiplier=self.settings.risk.stop_loss_atr_multiplier,
            take_profit_atr_multiplier=self.settings.risk.take_profit_atr_multiplier,
            use_trailing_stop=self.settings.risk.use_trailing_stop,
            trailing_stop_pips=self.settings.risk.trailing_stop_pips,
        )

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
            self.strategy = self._create_strategy()
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

        # Initialize strategy
        if not self.strategy.initialize():
            logger.error("Failed to initialize strategy")
            return False

        # Get symbol info
        symbol_info = self.broker.get_symbol_info(self.settings.default_symbol)
        if symbol_info:
            self.strategy.pip_size = symbol_info.get("pip_size", 0.0001)
            logger.info(f"Symbol: {self.settings.default_symbol}")
            logger.info(f"Pip size: {self.strategy.pip_size}")
            logger.info(f"Min lot: {symbol_info.get('volume_min', 0.01)}")

        logger.info("=" * 60)
        logger.info("Trading Bot initialized successfully!")
        logger.info(f"Mode: {self.settings.trading_mode}")
        logger.info(f"Broker: {self.settings.active_broker}")
        logger.info(f"Strategy: ORB ({self.settings.orb.range_minutes} min)")
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
        """Single trading cycle."""
        # Check if within trading hours
        if not is_trading_hours(
            self.settings.orb.session_start,
            self.settings.orb.session_end,
            self.settings.orb.timezone,
        ):
            remaining = time_to_session_end(
                self.settings.orb.session_start,
                self.settings.orb.timezone,
            )
            logger.debug(f"Outside trading hours. Next session in: {remaining}")
            time.sleep(60)
            return

        # Check risk limits
        positions = self.broker.get_positions(self.settings.default_symbol)
        can_trade, reason = self.risk_manager.can_trade(len(positions))

        if not can_trade:
            logger.warning(f"Trading restricted: {reason}")
            time.sleep(60)
            return

        # Get latest candles
        candles = self.broker.get_candles(
            symbol=self.settings.default_symbol,
            timeframe=self.settings.default_timeframe,
            count=100,
        )

        if not candles:
            logger.warning("No candle data received")
            time.sleep(10)
            return

        latest_candle = candles[-1]

        # Check if new candle
        if self._last_candle_time == latest_candle.timestamp:
            # Wait for next candle
            wait_time = seconds_to_next_candle(5)  # 5 minute timeframe
            time.sleep(min(wait_time, 10))
            return

        self._last_candle_time = latest_candle.timestamp
        logger.info(f"New candle: {latest_candle.timestamp} | O:{latest_candle.open:.5f} H:{latest_candle.high:.5f} L:{latest_candle.low:.5f} C:{latest_candle.close:.5f}")

        # Update account balance
        balance = self.broker.get_account_balance()
        self.risk_manager.update_balance(balance)

        # Check for exit signals on existing positions
        for position in positions:
            exit_signal = self.strategy.should_exit(
                position, latest_candle.close, candles
            )
            if exit_signal:
                logger.info(f"Exit signal: {exit_signal.reason}")
                self._execute_exit(position)

        # Check for entry signals (only if no positions)
        if not positions:
            signal = self.strategy.on_candle(latest_candle, candles)

            if signal and signal.is_entry:
                logger.info(f"Entry signal: {signal.signal_type.value} - {signal.reason}")
                self._execute_entry(signal)

        # Small delay before next cycle
        time.sleep(1)

    def _execute_entry(self, signal) -> None:
        """
        Execute entry order based on signal.

        Args:
            signal: Strategy signal.
        """
        if self.settings.trading_mode == "paper":
            logger.info(f"[PAPER] Would enter {signal.signal_type.value} at {signal.price}")
            logger.info(f"[PAPER] SL: {signal.stop_loss}, TP: {signal.take_profit}")
            return

        # Calculate position size
        symbol_info = self.broker.get_symbol_info(self.settings.default_symbol)
        balance = self.broker.get_account_balance()

        if signal.stop_loss:
            sl_pips = abs(signal.price - signal.stop_loss) / self.strategy.pip_size
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
        if signal.is_long:
            order = Order.market_buy(
                symbol=self.settings.default_symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB Long - {signal.reason}",
            )
        else:
            order = Order.market_sell(
                symbol=self.settings.default_symbol,
                quantity=lot_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB Short - {signal.reason}",
            )

        order.magic_number = self.strategy.magic_number
        order.strategy_name = self.strategy.name

        # Execute order
        order_id = self.broker.place_order(order)

        if order_id:
            logger.info(f"Order placed successfully: {order_id}")
        else:
            logger.error("Failed to place order")

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
            # Close all positions if configured
            positions = self.broker.get_positions(self.settings.default_symbol)
            for position in positions:
                if position.magic_number == self.strategy.magic_number:
                    logger.info(f"Closing position on shutdown: {position.broker_position_id}")
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
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Trading symbol (default: EURUSD)",
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
    settings.default_symbol = args.symbol
    settings.default_timeframe = args.timeframe

    # Create and run bot
    bot = TradingBot(settings)

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
