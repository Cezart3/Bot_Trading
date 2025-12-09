#!/usr/bin/env python3
"""
ORB Trading Bot - Production Ready
===================================

IMPORTANT: This bot trades ONLY AMD based on backtesting analysis.
Demo and Live modes have IDENTICAL rules - no exceptions.

Based on 39-day backtesting results:
- AMD: +9.9%, Profit Factor 1.46, Max DD 7.4% -> APPROVED
- TSLA: +1.6%, Profit Factor 1.06 -> EXCLUDED (marginal)
- NVDA: -9.7%, Profit Factor 0.64 -> EXCLUDED (losing)

Usage:
    python run_trading.py                    # Paper mode (default)
    python run_trading.py --mode demo        # Demo with real orders
    python run_trading.py --mode live        # Live trading (requires validation)
    python run_trading.py --check-readiness  # Check if ready for live
    python run_trading.py --analysis         # Show stock analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import signal
import time
from datetime import datetime, date, timedelta
from typing import Optional

from config.settings import get_settings
from brokers.mt5_broker import MT5Broker
from strategies.orb_vwap_strategy import ORBVWAPStrategy, ORBVWAPConfig
from models.order import Order
from utils.logger import setup_logging, get_logger
from utils.news_filter import NewsFilter, NewsFilterConfig
from utils.prop_trading_rules import (
    PropTradingTracker,
    PropTradingConfig,
    get_analysis_report,
    ReadinessLevel,
)
from utils.time_utils import is_trading_hours, seconds_to_next_candle

logger = get_logger(__name__)

# ============================================================================
#                    CONFIGURATION - DO NOT CHANGE
# ============================================================================

# Based on backtesting: AMD is the ONLY profitable symbol
TRADING_SYMBOL = "AMD"

# Strict risk management (same for demo and live)
RISK_CONFIG = PropTradingConfig(
    allowed_symbol=TRADING_SYMBOL,
    max_risk_per_trade=0.5,  # 0.5% per trade
    max_daily_loss=2.0,  # 2% daily loss limit
    max_account_drawdown=5.0,  # 5% total drawdown limit
    max_positions=1,  # Only 1 position at a time
    max_trades_per_day=2,  # Max 2 trades per day
    max_consecutive_losses=2,  # Stop after 2 consecutive losses
    cooldown_after_losses=1,  # 1 day cooldown
    skip_news_days=True,  # Always skip news days
    stop_trading_after_time="15:30",  # No trades after 15:30
    # Readiness criteria
    min_demo_days=10,
    min_demo_trades=15,
    min_win_rate=35.0,
    min_profit_factor=1.2,
    max_allowed_drawdown=6.0,
    required_profitable_weeks=2,
)


# ============================================================================
#                    TRADING BOT CLASS
# ============================================================================

class ProductionTradingBot:
    """
    Production-ready trading bot with strict prop trading rules.

    CRITICAL: Demo and Live modes are IDENTICAL.
    If you can't follow rules on demo, you're not ready for live.
    """

    def __init__(self, mode: str = "paper"):
        """
        Initialize trading bot.

        Args:
            mode: Trading mode (paper/demo/live)
        """
        self.mode = mode
        self.settings = get_settings()

        # Components
        self.broker: Optional[MT5Broker] = None
        self.strategy: Optional[ORBVWAPStrategy] = None
        self.news_filter: Optional[NewsFilter] = None
        self.prop_tracker: Optional[PropTradingTracker] = None

        # State
        self._running = False
        self._last_candle_time: Optional[datetime] = None
        self._today: Optional[date] = None
        self._day_started = False

    def initialize(self) -> bool:
        """Initialize all components."""
        print("\n" + "=" * 70)
        print(f"  ORB TRADING BOT - {self.mode.upper()} MODE")
        print(f"  Symbol: {TRADING_SYMBOL} (ONLY - based on backtesting)")
        print("=" * 70)

        # 1. Initialize Prop Trading Tracker
        print("\n[1/5] Initializing prop trading rules...")
        self.prop_tracker = PropTradingTracker(RISK_CONFIG)

        # Check if live mode is allowed
        if self.mode == "live":
            status = self.prop_tracker.get_readiness_status()
            if not status["ready"]:
                print("\n" + "!" * 60)
                print("  LIVE TRADING NOT ALLOWED")
                print("  You have not met all readiness criteria on demo.")
                print("!" * 60)
                self.prop_tracker.print_readiness_report()
                return False

        # 2. Initialize News Filter
        print("[2/5] Initializing news filter...")
        news_config = NewsFilterConfig(
            filter_high_impact=True,
            filter_medium_impact=False,
            currencies=["USD"],
        )
        self.news_filter = NewsFilter(news_config)
        self.news_filter.update_calendar()

        # Check if today is a news day
        if self.news_filter.is_high_impact_day():
            print("\n" + "!" * 60)
            print("  HIGH IMPACT NEWS DAY - NO TRADING")
            print("!" * 60)
            events = self.news_filter.get_events_for_date()
            for event in events:
                if event.impact.value == "high":
                    print(f"  - {event.event} ({event.currency})")
            return False

        print("  No high impact news today - OK to trade")

        # 3. Connect to Broker
        print("[3/5] Connecting to broker...")
        self.broker = MT5Broker(
            login=self.settings.mt5.login,
            password=self.settings.mt5.password,
            server=self.settings.mt5.server,
            path=self.settings.mt5.path,
            timeout=self.settings.mt5.timeout,
        )

        if not self.broker.connect():
            print("  ERROR: Failed to connect to MT5!")
            return False

        account_info = self.broker.get_account_info()
        balance = account_info.get('balance', 0)
        print(f"  Account: {account_info.get('login', 'N/A')}")
        print(f"  Balance: ${balance:,.2f}")

        # 4. Initialize Strategy
        print("[4/5] Initializing strategy...")
        strategy_config = ORBVWAPConfig(
            use_vwap_filter=True,
            use_ema_filter=True,
            use_atr_filter=True,
            use_time_filter=True,
            risk_reward_ratio=2.5,
        )
        self.strategy = ORBVWAPStrategy(
            config=strategy_config,
            symbol=TRADING_SYMBOL,
            timeframe="M5",
        )
        self.strategy.initialize()

        # 5. Start the day in prop tracker
        print("[5/5] Starting trading day...")
        daily_stats = self.prop_tracker.start_day(balance)

        if not daily_stats.can_trade:
            print("\n" + "!" * 60)
            print("  TRADING BLOCKED")
            for violation in daily_stats.rules_violated:
                print(f"  - {violation}")
            print("!" * 60)
            return False

        print("\n" + "=" * 70)
        print("  BOT INITIALIZED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n  Mode:              {self.mode.upper()}")
        print(f"  Symbol:            {TRADING_SYMBOL}")
        print(f"  Risk per trade:    {RISK_CONFIG.max_risk_per_trade}%")
        print(f"  Max daily loss:    {RISK_CONFIG.max_daily_loss}%")
        print(f"  Max trades/day:    {RISK_CONFIG.max_trades_per_day}")
        print(f"  News filter:       ACTIVE")

        if self.mode == "paper":
            print("\n  NOTE: Paper mode - no real orders will be placed")
        elif self.mode == "demo":
            print("\n  NOTE: Demo mode - orders on demo account")
        else:
            print("\n  WARNING: LIVE MODE - REAL MONEY!")

        print("=" * 70 + "\n")

        self._today = date.today()
        self._day_started = True

        return True

    def run(self) -> None:
        """Main trading loop."""
        self._running = True
        print("Starting trading loop... Press Ctrl+C to stop.\n")

        while self._running:
            try:
                self._trading_cycle()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                time.sleep(5)

        self.shutdown()

    def _trading_cycle(self) -> None:
        """Single trading cycle."""
        now = datetime.now()
        today = now.date()

        # Check for new day
        if self._today != today:
            self._handle_new_day()
            return

        # Check trading hours
        if not is_trading_hours(
            self.settings.orb.session_start,
            self.settings.orb.session_end,
            self.settings.orb.timezone,
        ):
            time.sleep(60)
            return

        # Check if we can trade
        can_trade, reason = self.prop_tracker.check_can_open_trade(
            TRADING_SYMBOL,
            now,
        )

        if not can_trade:
            if "Max" in reason or "limit" in reason:
                logger.info(f"Trading restricted: {reason}")
            time.sleep(60)
            return

        # Get candles
        candles = self.broker.get_candles(
            symbol=TRADING_SYMBOL,
            timeframe="M5",
            count=100,
        )

        if not candles:
            time.sleep(10)
            return

        latest_candle = candles[-1]

        # Check for new candle
        if self._last_candle_time == latest_candle.timestamp:
            wait_time = seconds_to_next_candle(5)
            time.sleep(min(wait_time, 10))
            return

        self._last_candle_time = latest_candle.timestamp

        # Log candle
        print(
            f"[{latest_candle.timestamp.strftime('%H:%M')}] "
            f"O:{latest_candle.open:.2f} H:{latest_candle.high:.2f} "
            f"L:{latest_candle.low:.2f} C:{latest_candle.close:.2f}"
        )

        # Check for existing positions
        positions = self.broker.get_positions(TRADING_SYMBOL)

        if positions:
            # Already in a trade - wait
            return

        # Get signal
        signal = self.strategy.on_candle(latest_candle, candles)

        if signal and signal.is_entry:
            self._execute_signal(signal)

        time.sleep(1)

    def _execute_signal(self, signal) -> None:
        """Execute a trading signal."""
        print("\n" + "-" * 50)
        print(f"  SIGNAL: {signal.signal_type.value.upper()}")
        print(f"  Price:  ${signal.price:.2f}")
        print(f"  SL:     ${signal.stop_loss:.2f}")
        print(f"  TP:     ${signal.take_profit:.2f}")
        print(f"  Reason: {signal.reason}")
        print("-" * 50)

        if self.mode == "paper":
            print("  [PAPER] Signal logged - no order placed")
            # Still track it for statistics
            self.prop_tracker.record_trade(
                TRADING_SYMBOL,
                pnl=0,  # Unknown yet
                is_win=True,  # Placeholder
            )
            return

        # Calculate position size
        account_info = self.broker.get_account_info()
        balance = account_info.get('balance', 10000)

        risk_amount = balance * (RISK_CONFIG.max_risk_per_trade / 100)
        sl_distance = abs(signal.price - signal.stop_loss)

        if sl_distance > 0:
            position_size = risk_amount / sl_distance
            position_size = max(1, min(50, int(position_size)))  # 1-50 shares
        else:
            position_size = 1

        print(f"  Position size: {position_size} shares")
        print(f"  Risk amount: ${risk_amount:.2f}")

        # Create order
        if signal.is_long:
            order = Order.market_buy(
                symbol=TRADING_SYMBOL,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB {self.mode}",
            )
        else:
            order = Order.market_sell(
                symbol=TRADING_SYMBOL,
                quantity=position_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=f"ORB {self.mode}",
            )

        order.magic_number = self.strategy.magic_number

        # Place order
        order_id = self.broker.place_order(order)

        if order_id:
            print(f"  ORDER PLACED: {order_id}")
        else:
            print("  ORDER FAILED!")

    def _handle_new_day(self) -> None:
        """Handle transition to new day."""
        # End previous day
        if self._day_started:
            session = self.prop_tracker.end_day()
            print(f"\n[END OF DAY] PnL: ${session.pnl:.2f}")

        # Check news for new day
        if self.news_filter.is_high_impact_day():
            print("\n[NEW DAY] High impact news - no trading today")
            self._today = date.today()
            self._day_started = False
            time.sleep(3600)
            return

        # Start new day
        balance = self.broker.get_account_balance()
        daily_stats = self.prop_tracker.start_day(balance)

        if daily_stats.can_trade:
            print(f"\n[NEW DAY] Ready to trade. Balance: ${balance:,.2f}")
            self._day_started = True
        else:
            print(f"\n[NEW DAY] Trading blocked: {daily_stats.rules_violated}")
            self._day_started = False

        self._today = date.today()
        self.strategy.reset()
        self._last_candle_time = None

    def shutdown(self) -> None:
        """Shutdown the bot."""
        print("\nShutting down...")

        # End day in tracker
        if self._day_started and self.prop_tracker:
            try:
                session = self.prop_tracker.end_day()
                print(f"Day ended. PnL: ${session.pnl:.2f}")
            except Exception:
                pass

        # Disconnect broker
        if self.broker:
            self.broker.disconnect()

        print("Shutdown complete.\n")

    def stop(self) -> None:
        """Stop the bot."""
        self._running = False


# ============================================================================
#                    MAIN ENTRY POINT
# ============================================================================

def check_readiness():
    """Check and display readiness for live trading."""
    tracker = PropTradingTracker(RISK_CONFIG)
    tracker.print_readiness_report()


def show_analysis():
    """Show stock analysis report."""
    print(get_analysis_report())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ORB Trading Bot - AMD Only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading.py                    # Paper trading (safe)
  python run_trading.py --mode demo        # Demo account
  python run_trading.py --mode live        # Live trading (must pass readiness check)
  python run_trading.py --check-readiness  # Check if ready for live
  python run_trading.py --analysis         # Why only AMD?
        """
    )

    parser.add_argument(
        "--mode",
        choices=["paper", "demo", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Check readiness for live trading"
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Show stock analysis report"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level"
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Handle special commands
    if args.check_readiness:
        check_readiness()
        return

    if args.analysis:
        show_analysis()
        return

    # Live mode requires confirmation
    if args.mode == "live":
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  This will execute REAL trades with REAL money!")
        print("!" * 60)

        # Check readiness first
        tracker = PropTradingTracker(RISK_CONFIG)
        status = tracker.get_readiness_status()

        if not status["ready"]:
            print("\n  You have NOT passed the readiness check.")
            print("  Continue trading on demo until all criteria are met.")
            tracker.print_readiness_report()
            return

        confirm = input("\nType 'I UNDERSTAND' to confirm: ")
        if confirm != "I UNDERSTAND":
            print("Live trading cancelled.")
            return

    # Create and run bot
    bot = ProductionTradingBot(mode=args.mode)

    # Signal handlers
    def signal_handler(signum, frame):
        bot.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize and run
    if not bot.initialize():
        print("\nBot initialization failed. Check the errors above.")
        return

    try:
        bot.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
