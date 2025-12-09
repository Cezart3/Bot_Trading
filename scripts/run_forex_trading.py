"""
EUR/USD ORB Trading Bot - London Session.

Usage:
    python run_forex_trading.py --mode demo
    python run_forex_trading.py --mode live  (be careful!)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from datetime import datetime, time as dt_time
from typing import Optional
from brokers.base_broker import BaseBroker


from brokers.mt5_broker import MT5Broker
from config.settings import get_settings
from models.candle import Candle
from models.order import Order, OrderSide, OrderType
from strategies.orb_forex_strategy import ORBForexStrategy, ForexORBConfig, create_eurusd_london_strategy
from utils.logger import get_logger
from utils.risk_manager import RiskManager, PropAccountLimits

logger = get_logger(__name__)


def calculate_lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = 10.0,
) -> float:
    """Calculate lot size based on risk percentage."""
    risk_amount = account_balance * (risk_percent / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
    lot_size = round(lot_size, 2)
    lot_size = max(0.01, min(lot_size, 10.0))
    return lot_size


class ForexTradingBot:
    """EUR/USD ORB Trading Bot."""

    def __init__(self, mode: str = "demo", risk_percent: float = 0.5):
        """Initialize bot."""
        self.mode = mode
        self.risk_percent = risk_percent
        self.settings = get_settings()
        self.broker: Optional[MT5Broker] = None
        self.strategy: Optional[ORBForexStrategy] = None
        self.symbol = "EURUSD"
        self.timeframe = "M5"
        self.pip_size = 0.0001
        self.pip_value_per_lot = 10.0  # $10 per pip per lot for EUR/USD
        self._running = False

        # Risk manager - OPTIMAL settings based on backtest
        # Fixed 0.5% risk performed best (+4.26% with 4.52% max DD)
        limits = PropAccountLimits(
            max_daily_drawdown=4.0,        # 4% max daily loss - STOP trading
            max_account_drawdown=10.0,     # 10% max account drawdown - STOP trading
            warning_drawdown=3.0,          # 3% daily DD -> warning
            account_warning_drawdown=7.0,  # 7% account DD -> warning
            normal_risk_per_trade=0.5,     # Fixed 0.5% per trade (optimal)
            reduced_risk_per_trade=0.25,   # 0.25% if approaching limits
            max_positions=1,               # 1 position at a time
        )
        self.risk_manager = RiskManager(limits=limits)

    def initialize(self) -> bool:
        """Initialize all components."""
        print()
        print("=" * 70)
        print(f"  EUR/USD ORB TRADING BOT - {self.mode.upper()} MODE")
        print("=" * 70)
        print()

        # 1. Initialize broker
        print("[1/3] Connecting to broker...")
        self.broker = MT5Broker(
            login=self.settings.mt5.login,
            password=self.settings.mt5.password,
            server=self.settings.mt5.server,
            path=self.settings.mt5.path,
        )

        if not self.broker.connect():
            print("  ERROR: Failed to connect to MT5!")
            return False

        account_info = self.broker.get_account_info()
        if account_info:
            print(f"  Account: {account_info.get('login')}")
            print(f"  Balance: ${account_info.get('balance', 0):,.2f}")
            print(f"  Server: {account_info.get('server')}")

        # 2. Initialize strategy
        print()
        print("[2/3] Initializing strategy...")
        self.strategy = create_eurusd_london_strategy()
        self.strategy.initialize()

        print(f"  Symbol: {self.symbol}")
        print(f"  Session: London (10:00-18:00)")
        print(f"  Opening Range: 60 minutes")
        print(f"  Risk per trade: {self.risk_percent}%")
        print(f"  Risk/Reward: 2:1")

        # 3. Verify symbol
        print()
        print("[3/3] Verifying symbol...")
        if not self.broker.select_symbol(self.symbol):
            print(f"  ERROR: Symbol {self.symbol} not available!")
            return False

        symbol_info = self.broker.get_symbol_info(self.symbol)
        if symbol_info:
            print(f"  Spread: {symbol_info.get('spread', 'N/A')} points")
            print(f"  Digits: {symbol_info.get('digits', 'N/A')}")

        print()
        print("=" * 70)
        print("  BOT INITIALIZED SUCCESSFULLY")
        print("=" * 70)
        print()

        return True

    def run(self) -> None:
        """Run trading loop."""
        self._running = True
        print("Starting trading loop... Press Ctrl+C to stop.")
        print()

        last_candle_time = None
        last_status_print = None

        # Initialize risk manager with current balance
        account_info = self.broker.get_account_info()
        if account_info:
            balance = account_info.get("balance", 100000)
            self.risk_manager.initialize_daily_stats(balance)
            print(f"Risk Manager initialized with balance: ${balance:,.2f}")
            print()

        while self._running:
            try:
                # Update balance and check risk limits
                account_info = self.broker.get_account_info()
                if account_info:
                    balance = account_info.get("balance", 100000)
                    self.risk_manager.update_balance(balance)

                # Check if we can trade
                positions = self.broker.get_positions(self.symbol)
                can_trade, reason = self.risk_manager.can_trade(len(positions))

                # Get latest candle
                candles = self.broker.get_candles(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    count=100,
                )

                if not candles:
                    time.sleep(5)
                    continue

                # Get H1 candles for trend filter
                h1_candles = self.broker.get_candles(
                    symbol=self.symbol,
                    timeframe="H1",
                    count=100,
                )

                # Update strategy with H1 candles for trend detection
                if h1_candles:
                    self.strategy.update_h1_candles(h1_candles)

                latest_candle = candles[-1]

                # Check if new candle
                if last_candle_time != latest_candle.timestamp:
                    last_candle_time = latest_candle.timestamp

                    # Process candle
                    signal = self.strategy.on_candle(latest_candle, candles)

                    # Print status
                    status = self.strategy.get_status()
                    ts = latest_candle.timestamp.strftime("%H:%M")
                    state = status["state"]

                    or_info = ""
                    if status["opening_range"]:
                        or_info = f" | OR: H={status['opening_range']['high']:.5f} L={status['opening_range']['low']:.5f}"

                    risk_info = f" | Risk: {self.risk_manager.get_current_risk_percent()}%"
                    pos_info = f" | Pos: {len(positions)}"

                    # Show trend info
                    trend_info = ""
                    if self.strategy._current_trend:
                        trend_info = f" | Trend: {self.strategy._current_trend}"

                    print(
                        f"[{ts}] {latest_candle.close:.5f} | State: {state}{or_info}{trend_info}{risk_info}{pos_info}"
                    )

                    # Execute signal
                    if signal:
                        print()
                        print("=" * 50)
                        print(f"  SIGNAL: {signal.signal_type.value}")
                        print(f"  Entry: {signal.price:.5f}")
                        print(f"  SL: {signal.stop_loss:.5f}")
                        print(f"  TP: {signal.take_profit:.5f}")
                        print("=" * 50)
                        print()

                        # Check risk manager approval
                        if not can_trade:
                            print(f"  TRADE BLOCKED by Risk Manager: {reason}")
                            print("=" * 50)
                            continue

                        # Get risk percent from risk manager (may be reduced)
                        current_risk = self.risk_manager.get_current_risk_percent()

                        sl_pips = abs(signal.price - signal.stop_loss) / self.pip_size
                        lot_size = calculate_lot_size(
                            account_balance=balance,
                            risk_percent=current_risk,  # Use risk manager's risk %
                            stop_loss_pips=sl_pips,
                            pip_value_per_lot=self.pip_value_per_lot,
                        )
                        risk_dollars = sl_pips * self.pip_value_per_lot * lot_size

                        print(f"  Balance: ${balance:,.2f}")
                        print(f"  Risk: {current_risk}% = ${risk_dollars:.2f}")
                        print(f"  SL Distance: {sl_pips:.1f} pips")
                        print(f"  Lot Size: {lot_size}")
                        print("=" * 50)
                        print()

                        if self.mode == "demo":
                            # Create Order object
                            order_side = OrderSide.BUY if signal.signal_type.value == "long" else OrderSide.SELL
                            order = Order(
                                symbol=self.symbol,
                                side=order_side,
                                order_type=OrderType.MARKET,
                                quantity=lot_size,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                comment="ORB_FOREX",
                                magic_number=3001,
                            )

                            # Place order on demo account
                            order_result = self.broker.place_order(order)

                            if order_result:
                                print(f"  ORDER PLACED: {order_result}")
                            else:
                                print("  ORDER FAILED!")
                        else:
                            print("  LIVE MODE: Order would be placed here")

                # Sleep until next check
                time.sleep(10)

            except KeyboardInterrupt:
                print()
                print("Stopping bot...")
                self._running = False
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def shutdown(self) -> None:
        """Shutdown bot."""
        print()
        print("Shutting down...")
        if self.broker:
            self.broker.disconnect()
        print("Shutdown complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="EUR/USD ORB Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["demo", "live"],
        default="demo",
        help="Trading mode (default: demo)",
    )
    args = parser.parse_args()

    bot = ForexTradingBot(mode=args.mode)

    if not bot.initialize():
        print("Failed to initialize bot!")
        sys.exit(1)

    try:
        bot.run()
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
