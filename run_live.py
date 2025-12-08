"""
Live/Demo Trading Script for ORB Strategy

This script runs the trading bot on live or demo accounts.
Supports multiple accounts and includes the news filter.

IMPORTANT: Always test on demo account first!

Usage:
    python run_live.py                           # Paper trading (no orders)
    python run_live.py --mode demo               # Demo account trading
    python run_live.py --mode live               # LIVE trading (CAREFUL!)
    python run_live.py --symbol NVDA             # Trade specific symbol
    python run_live.py --multi                   # Trade multiple accounts
"""

import argparse
import signal
import sys
import time
from datetime import datetime, date
from typing import Optional

from config.settings import Settings, get_settings
from brokers.mt5_broker import MT5Broker
from strategies.orb_vwap_strategy import ORBVWAPStrategy, ORBVWAPConfig
from models.order import Order
from utils.logger import setup_logging, get_logger
from utils.risk_manager import RiskManager, PropAccountLimits
from utils.news_filter import NewsFilter, NewsFilterConfig, create_news_filter
from utils.time_utils import is_trading_hours, seconds_to_next_candle
from multi_account.account_manager import MultiAccountManager, AccountConfig, AccountProvider

logger = get_logger(__name__)


class LiveTradingBot:
    """
    Live Trading Bot with News Filter.

    Features:
    - Connects to MT5 demo or live accounts
    - Applies news filter to avoid high-impact days
    - Risk management with prop trading limits
    - Real-time candle processing
    """

    def __init__(
        self,
        settings: Settings,
        symbol: str = "NVDA",
        use_news_filter: bool = True,
        mode: str = "paper",  # paper, demo, live
    ):
        """
        Initialize live trading bot.

        Args:
            settings: Application settings
            symbol: Trading symbol
            use_news_filter: Whether to use news filter
            mode: Trading mode (paper/demo/live)
        """
        self.settings = settings
        self.symbol = symbol
        self.use_news_filter = use_news_filter
        self.mode = mode

        self.broker: Optional[MT5Broker] = None
        self.strategy: Optional[ORBVWAPStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.news_filter: Optional[NewsFilter] = None

        self._running = False
        self._last_candle_time: Optional[datetime] = None
        self._trade_taken_today = False
        self._current_date: Optional[date] = None

    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info(f"  INITIALIZING LIVE TRADING BOT - {self.mode.upper()} MODE")
        logger.info("=" * 60)

        # 1. Initialize News Filter
        if self.use_news_filter:
            logger.info("Initializing news filter...")
            news_config = NewsFilterConfig(
                filter_high_impact=True,
                filter_medium_impact=False,
                currencies=["USD"],
            )
            self.news_filter = NewsFilter(news_config)
            self.news_filter.update_calendar()

            # Check if today is a news day
            if self.news_filter.is_high_impact_day():
                logger.warning("TODAY IS A HIGH IMPACT NEWS DAY - Trading disabled!")
                events = self.news_filter.get_events_for_date()
                for event in events:
                    logger.warning(f"  - {event.event} ({event.currency})")
                return False

            logger.info("News filter initialized - no high impact events today")

        # 2. Create Broker Connection
        logger.info("Creating broker connection...")
        self.broker = MT5Broker(
            login=self.settings.mt5.login,
            password=self.settings.mt5.password,
            server=self.settings.mt5.server,
            path=self.settings.mt5.path,
            timeout=self.settings.mt5.timeout,
        )

        if not self.broker.connect():
            logger.error("Failed to connect to MT5!")
            return False

        # 3. Get Account Info
        account_info = self.broker.get_account_info()
        logger.info(f"Account: {account_info.get('login', 'N/A')}")
        logger.info(f"Server: {account_info.get('server', 'N/A')}")
        logger.info(f"Balance: ${account_info.get('balance', 0):,.2f}")
        logger.info(f"Equity: ${account_info.get('equity', 0):,.2f}")

        # 4. Create Risk Manager
        logger.info("Creating risk manager...")
        limits = PropAccountLimits(
            max_daily_drawdown=self.settings.risk.max_daily_loss,
            max_account_drawdown=10.0,
            normal_risk_per_trade=self.settings.risk.max_risk_per_trade,
            reduced_risk_per_trade=0.5,
            max_positions=self.settings.risk.max_positions,
        )
        self.risk_manager = RiskManager(
            initial_account_balance=account_info.get('balance', 10000),
            limits=limits,
        )

        # 5. Create Strategy
        logger.info("Creating strategy...")
        strategy_config = ORBVWAPConfig(
            use_vwap_filter=True,
            use_ema_filter=True,
            use_atr_filter=True,
            use_time_filter=True,
            risk_reward_ratio=2.5,
        )
        self.strategy = ORBVWAPStrategy(
            config=strategy_config,
            symbol=self.symbol,
            timeframe="M5",
        )
        self.strategy.initialize()

        # 6. Get Symbol Info
        symbol_info = self.broker.get_symbol_info(self.symbol)
        if not symbol_info:
            logger.error(f"Symbol {self.symbol} not found!")
            return False

        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Point: {symbol_info.get('point', 0)}")
        logger.info(f"Spread: {symbol_info.get('spread', 0)} points")

        logger.info("=" * 60)
        logger.info("  BOT INITIALIZED SUCCESSFULLY")
        logger.info(f"  Mode: {self.mode.upper()}")
        logger.info(f"  Symbol: {self.symbol}")
        logger.info(f"  News Filter: {'Enabled' if self.use_news_filter else 'Disabled'}")
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
        # Check for new trading day
        today = date.today()
        if self._current_date != today:
            self._current_date = today
            self._trade_taken_today = False
            self.strategy.reset()
            logger.info(f"New trading day: {today}")

            # Re-check news filter for new day
            if self.use_news_filter and self.news_filter.is_high_impact_day():
                logger.warning("HIGH IMPACT NEWS DAY - Trading disabled!")
                self._running = False
                return

        # Check if within trading hours
        if not is_trading_hours(
            self.settings.orb.session_start,
            self.settings.orb.session_end,
            self.settings.orb.timezone,
        ):
            time.sleep(60)
            return

        # Check if trade already taken today
        if self._trade_taken_today:
            time.sleep(60)
            return

        # Check risk limits
        can_trade, reason = self.risk_manager.check_can_trade()
        if not can_trade:
            logger.warning(f"Trading restricted: {reason}")
            time.sleep(60)
            return

        # Get latest candles
        candles = self.broker.get_candles(
            symbol=self.symbol,
            timeframe="M5",
            count=100,
        )

        if not candles:
            logger.warning("No candle data received")
            time.sleep(10)
            return

        latest_candle = candles[-1]

        # Check if new candle
        if self._last_candle_time == latest_candle.timestamp:
            wait_time = seconds_to_next_candle(5)
            time.sleep(min(wait_time, 10))
            return

        self._last_candle_time = latest_candle.timestamp

        logger.info(
            f"New candle: {latest_candle.timestamp} | "
            f"O:{latest_candle.open:.2f} H:{latest_candle.high:.2f} "
            f"L:{latest_candle.low:.2f} C:{latest_candle.close:.2f}"
        )

        # Get existing positions
        positions = self.broker.get_positions(self.symbol)

        # Check for exit signals on existing positions
        for position in positions:
            if hasattr(position, 'magic_number') and position.magic_number == self.strategy.magic_number:
                # Position is managed by this strategy
                pass

        # Check for entry signals (only if no positions)
        if not positions and not self._trade_taken_today:
            signal = self.strategy.on_candle(latest_candle, candles)

            if signal and signal.is_entry:
                logger.info(f"ENTRY SIGNAL: {signal.signal_type.value}")
                logger.info(f"  Price: {signal.price:.2f}")
                logger.info(f"  SL: {signal.stop_loss:.2f}")
                logger.info(f"  TP: {signal.take_profit:.2f}")
                logger.info(f"  Reason: {signal.reason}")

                self._execute_entry(signal)
                self._trade_taken_today = True

        time.sleep(1)

    def _execute_entry(self, signal) -> None:
        """Execute entry order."""
        if self.mode == "paper":
            logger.info(f"[PAPER] Would enter {signal.signal_type.value} at {signal.price:.2f}")
            logger.info(f"[PAPER] SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
            return

        # Calculate position size
        account_info = self.broker.get_account_info()
        balance = account_info.get('balance', 10000)

        # Risk 1% per trade
        risk_amount = balance * (self.settings.risk.max_risk_per_trade / 100)

        if signal.stop_loss:
            sl_distance = abs(signal.price - signal.stop_loss)
            if sl_distance > 0:
                # For stocks: position size = risk / sl_distance
                position_size = risk_amount / sl_distance
                position_size = max(1, min(100, int(position_size)))  # 1-100 shares
            else:
                position_size = 1
        else:
            position_size = 1

        logger.info(f"Position size: {position_size} shares")

        # Create order
        if signal.is_long:
            order = Order.market_buy(
                symbol=self.symbol,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB Long - {self.mode}",
            )
        else:
            order = Order.market_sell(
                symbol=self.symbol,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB Short - {self.mode}",
            )

        order.magic_number = self.strategy.magic_number

        # Execute order
        if self.mode == "demo" or self.mode == "live":
            order_id = self.broker.place_order(order)

            if order_id:
                logger.info(f"ORDER PLACED: {order_id}")
                self.risk_manager.record_trade_entry(signal.price, signal.stop_loss)
            else:
                logger.error("FAILED TO PLACE ORDER!")

    def shutdown(self) -> None:
        """Shutdown the bot."""
        logger.info("Shutting down trading bot...")
        self._running = False

        if self.broker:
            self.broker.disconnect()

        logger.info("Bot shutdown complete")

    def stop(self) -> None:
        """Stop the bot gracefully."""
        self._running = False


def run_multi_account_trading(symbols: list[str] = None) -> None:
    """
    Run trading on multiple accounts.

    This loads account configurations from config/accounts.json
    """
    if symbols is None:
        symbols = ["NVDA", "AMD", "TSLA"]

    print("\n" + "=" * 70)
    print("       MULTI-ACCOUNT TRADING MODE")
    print("=" * 70)

    # Load account manager
    manager = MultiAccountManager()
    manager.load_config()

    if not manager.accounts:
        print("\nNo accounts configured!")
        print("Run: python multi_account/account_manager.py")
        print("This will create a sample config at config/accounts.json")
        return

    # Connect to all accounts
    manager.connect_all()
    manager.print_status()

    print("\nMulti-account trading would run here...")
    print("Each account trades independently with its own risk limits.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ORB Strategy Live Trading")
    parser.add_argument("--mode", choices=["paper", "demo", "live"], default="paper",
                        help="Trading mode")
    parser.add_argument("--symbol", type=str, default="NVDA",
                        help="Trading symbol")
    parser.add_argument("--no-news-filter", action="store_true",
                        help="Disable news filter")
    parser.add_argument("--multi", action="store_true",
                        help="Run multi-account mode")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level")

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Safety confirmation for live mode
    if args.mode == "live":
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  This will execute REAL trades with REAL money!")
        print("!" * 60)
        confirm = input("\nType 'YES' to confirm: ")
        if confirm != "YES":
            print("Live trading cancelled.")
            return

    if args.multi:
        run_multi_account_trading()
        return

    # Load settings
    settings = get_settings()

    # Create and run bot
    bot = LiveTradingBot(
        settings=settings,
        symbol=args.symbol.upper(),
        use_news_filter=not args.no_news_filter,
        mode=args.mode,
    )

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize and run
    if not bot.initialize():
        logger.error("Bot initialization failed!")
        return

    try:
        bot.run()
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
