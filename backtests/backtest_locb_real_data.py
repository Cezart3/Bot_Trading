"""
LOCB Strategy Backtest with REAL MT5 Data.

Uses actual historical data from MetaTrader 5.
Correct position sizing with EXACT risk percentage.

Author: Trading Bot Project

"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import MetaTrader5 as mt5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.candle import Candle
from strategies.locb_strategy import LOCBStrategy, TradingSession, M1CandleData
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    sl_pips: float
    lot_size: float
    risk_amount: float  # Exact $ risked
    pnl: float
    pnl_pips: float
    pnl_percent: float  # % of account
    result: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Results from backtesting."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_percent: float
    avg_win_percent: float
    avg_loss_percent: float
    max_drawdown_percent: float
    max_consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)


class LOCBRealDataBacktester:
    """Backtester using real MT5 data for LOCB strategy."""

    # Pip values per standard lot (100,000 units for forex)
    PIP_VALUES = {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "AUDUSD": 10.0,
        "EURGBP": 10.0,  # Approximately, depends on GBP/USD
        "US30": 1.0,     # $1 per point per contract
        "NDX100": 1.0,
        "GER30": 1.0,
        "USTEC": 1.0,
        "NAS100": 1.0,
    }

    PIP_SIZES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "AUDUSD": 0.0001,
        "EURGBP": 0.0001,
        "US30": 1.0,
        "NDX100": 0.1,
        "GER30": 0.1,
        "USTEC": 0.1,
        "NAS100": 0.1,
    }

    def __init__(
        self,
        symbol: str,
        initial_balance: float = 10000.0,
        risk_percent: float = 0.5,  # 0.5% risk per trade
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.balance = initial_balance
        self.trades: List[Trade] = []

        self.pip_size = self.PIP_SIZES.get(symbol, 0.0001)
        self.pip_value = self.PIP_VALUES.get(symbol, 10.0)

        # MT5 server timezone offset from UTC (e.g., "Etc/GMT-4" means UTC+4)
        # Assuming server is UTC+2 for our current setup
        self.mt5_timezone = "Etc/GMT-2" # This should match your MT5 server time, assuming UTC+2 for Romania
        
    def _create_strategy_for_symbol(self) -> LOCBStrategy:
        """Create LOCB strategy with backtest-appropriate configurations."""
        # These parameters are based on the default LOCBStrategy constructor
        # and tuned for backtesting (e.g., news filter disabled)
        strategy_params = {
            "symbol": self.symbol,
            "timeframe": "M1", # LOCB strategy uses M1 for entry confirmation
            "magic_number": 12346,
            "timezone": self.mt5_timezone, 
            "london_open_hour": 10, # If MT5 is UTC+2, London open (8 UTC) is 10 server time
            "london_open_minute": 0,
            "london_end_hour": 13,   # London close (11 UTC) is 13 server time
            "ny_open_hour": 16,     # NY open (14 UTC) is 16 server time
            "ny_open_minute": 0,
            "ny_end_hour": 19,      # NY close (17 UTC) is 19 server time
            "sl_buffer_pips": 1.0,
            "min_range_pips": 2.0,
            "max_range_pips": 30.0,
            "min_rr_ratio": 1.5,
            "max_rr_ratio": 4.0,
            "fallback_rr_ratio": 2.5,
            "max_trades_per_day": 2, # Max 1 per session
            "use_news_filter": False, # IMPORTANT: Disable for backtesting
            "close_before_session_end_minutes": 10,
        }
        
        # Adjust London/NY open/end hours based on a UTC+2 assumption for Romania (user's locale)
        # London Kill Zone: 08:00-11:00 UTC (10:00-13:00 Romania/Server time)
        # NY Kill Zone: 14:00-17:00 UTC (16:00-19:00 Romania/Server time)
        # These align with the `gemini.md` document for Kill Zones.
        strategy_params["london_open_hour"] = 10
        strategy_params["london_end_hour"] = 13
        strategy_params["ny_open_hour"] = 16
        strategy_params["ny_end_hour"] = 19


        strategy = LOCBStrategy(**strategy_params)
        return strategy

    def calculate_lot_size(self, sl_pips: float) -> float:
        """
        Calculate lot size for EXACT risk amount.

        Formula:
        Risk Amount = Balance * Risk% / 100
        Lot Size = Risk Amount / (sl_pips * Pip_Value_per_lot)

        Example with 0.5% risk, $10,000 balance, 15 pip SL:
        Risk Amount = $10,000 * 0.5% = $50
        Lot Size = $50 / (15 * $10) = 0.333 lots
        Loss at SL = 15 pips * 0.333 lots * $10 = $50 (exactly 0.5%)
        """
        if sl_pips <= 0:
            return 0.01  # Minimum

        risk_amount = self.balance * self.risk_percent / 100
        lot_size = risk_amount / (sl_pips * self.pip_value)

        # Apply broker limits
        lot_size = max(0.01, min(lot_size, 10.0))  # Min 0.01, max 10 lots
        lot_size = round(lot_size, 2)  # Round to 2 decimals

        return lot_size

    def calculate_pnl(self, pnl_pips: float, lot_size: float) -> float:
        """Calculate P&L from pips and lot size."""
        return pnl_pips * lot_size * self.pip_value

    def fetch_mt5_candles(
        self,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Candle]:
        """Fetch real candles from MT5."""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1) # LOCB uses M1 for data

        rates = mt5.copy_rates_range(self.symbol, mt5_tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"  [!] No data for {self.symbol} {timeframe}")
            return []

        candles = []
        for rate in rates:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(rate['time']),
                open=rate['open'],
                high=rate['high'],
                low=rate['low'],
                close=rate['close'],
                volume=rate['tick_volume']
            ))

        return candles

    def run_backtest(self, days: int = 60) -> Optional[BacktestResult]:
        """Run backtest on real MT5 data."""

        # Initialize MT5
        if not mt5.initialize():
            print("Failed to initialize MT5")
            return None

        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return None

        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)

        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"  Fetching real data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Fetch all timeframes - LOCB needs M1 primarily
        m1_candles = self.fetch_mt5_candles("M1", start_date, end_date)
        m5_candles = self.fetch_mt5_candles("M5", start_date, end_date)
        h1_candles = self.fetch_mt5_candles("H1", start_date, end_date)

        mt5.shutdown()

        if len(m1_candles) < 500: # Changed from M5 to M1 check
            print(f"  [!] Insufficient data: {len(m1_candles)} M1 candles")
            return None

        print(f"  M1: {len(m1_candles)}, M5: {len(m5_candles)}, H1: {len(h1_candles)}")


        # Create strategy
        try:
            strategy = self._create_strategy_for_symbol()
            strategy.initialize()
        except ValueError as e:
            print(f"  [!] Error creating strategy for {self.symbol}: {e}")
            return None

        # Backtest state
        self.balance = self.initial_balance
        self.trades = []
        position = None
        peak_balance = self.initial_balance
        max_drawdown = 0
        consecutive_losses = 0
        max_consecutive_losses = 0

        # Process candles
        # LOCB strategy expects M1 candles to be fed in the on_candle method.
        # It manages its own internal history and state.

        # We need to maintain a history of M1 candles to pass to strategy.on_candle
        # as the 'candles' argument.
        m1_history: List[Candle] = []

        for i, m1_candle in enumerate(m1_candles):
            m1_history.append(m1_candle)

            # Update spread (estimate)
            spread_estimate = self.pip_size * 1.0 # 1 pip spread
            strategy.update_spread(spread_estimate)

            # Check existing position
            if position:
                if position["direction"] == "long":
                    # Check SL
                    if m1_candle.low <= position["sl"]:
                        pnl_pips = (position["sl"] - position["entry"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="long",
                            entry_price=position["entry"],
                            exit_price=position["sl"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="loss",
                            exit_reason="sl_hit"
                        ))
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        continue

                    # Check TP
                    if m1_candle.high >= position["tp"]:
                        pnl_pips = (position["tp"] - position["entry"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="long",
                            entry_price=position["entry"],
                            exit_price=position["tp"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="win",
                            exit_reason="tp_hit"
                        ))
                        position = None
                        consecutive_losses = 0
                        continue

                else:  # Short
                    # Check SL
                    if m1_candle.high >= position["sl"]:
                        pnl_pips = (position["entry"] - position["sl"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="short",
                            entry_price=position["entry"],
                            exit_price=position["sl"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="loss",
                            exit_reason="sl_hit"
                        ))
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        continue

                    # Check TP
                    if m1_candle.low <= position["tp"]:
                        pnl_pips = (position["entry"] - position["tp"]) / self.pip_size
                        pnl = self.calculate_pnl(pnl_pips, position["lot_size"])
                        pnl_percent = (pnl / self.balance) * 100

                        self.balance += pnl
                        self.trades.append(Trade(
                            entry_time=position["entry_time"],
                            exit_time=m1_candle.timestamp,
                            direction="short",
                            entry_price=position["entry"],
                            exit_price=position["tp"],
                            sl=position["sl"],
                            tp=position["tp"],
                            sl_pips=position["sl_pips"],
                            lot_size=position["lot_size"],
                            risk_amount=position["risk_amount"],
                            pnl=pnl,
                            pnl_pips=pnl_pips,
                            pnl_percent=pnl_percent,
                            result="win",
                            exit_reason="tp_hit"
                        ))
                        position = None
                        consecutive_losses = 0
                        continue

            # Look for new signals
            if not position:
                signal = strategy.on_candle(m1_candle, m1_history)

                if signal and signal.stop_loss and signal.take_profit:
                    is_long = "LONG" in str(signal.signal_type)

                    # Calculate SL distance in pips
                    sl_pips = abs(signal.price - signal.stop_loss) / self.pip_size

                    # Calculate lot size for EXACT risk
                    risk_amount = self.balance * self.risk_percent / 100
                    lot_size = self.calculate_lot_size(sl_pips)

                    position = {
                        "direction": "long" if is_long else "short",
                        "entry": signal.price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "sl_pips": sl_pips,
                        "lot_size": lot_size,
                        "risk_amount": risk_amount,
                        "entry_time": m1_candle.timestamp,
                    }

            # Track drawdown
            if self.balance > peak_balance:
                peak_balance = self.balance
            drawdown = (peak_balance - self.balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate results
        if not self.trades:
            return BacktestResult(
                symbol=self.symbol,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0,
                profit_factor=0,
                total_pnl=0,
                total_pnl_percent=0,
                avg_win_percent=0,
                avg_loss_percent=0,
                max_drawdown_percent=max_drawdown,
                max_consecutive_losses=0,
                trades=[]
            )

        wins = [t for t in self.trades if t.result == "win"]
        losses = [t for t in self.trades if t.result == "loss"]

        total_win_pnl = sum(t.pnl for t in wins)
        total_loss_pnl = abs(sum(t.pnl for t in losses))

        return BacktestResult(
            symbol=self.symbol,
            total_trades=len(self.trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(self.trades) * 100,
            profit_factor=total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0,
            total_pnl=self.balance - self.initial_balance,
            total_pnl_percent=(self.balance - self.initial_balance) / self.initial_balance * 100,
            avg_win_percent=sum(t.pnl_percent for t in wins) / len(wins) if wins else 0,
            avg_loss_percent=sum(t.pnl_percent for t in losses) / len(losses) if losses else 0,
            max_drawdown_percent=max_drawdown,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades
        )


def run_locb_real_backtest(
    symbols: List[str] = None,
    days: int = 60,
    balance: float = 10000.0,
    risk: float = 0.5
):
    """Run backtest with real MT5 data."""

    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "AUDUSD", "US30", "NDX100", "USTEC", "NAS100", "GER30"]

    print("=" * 70)
    print("LOCB STRATEGY BACKTEST - REAL MT5 DATA")
    print("=" * 70)
    print(f"Period: Last {days} days")
    print(f"Initial Balance: ${balance:,.2f}")
    print(f"Risk per Trade: {risk}% = ${balance * risk / 100:.2f}")
    print(f"Symbols: {', '.join(symbols)}")
    print("=" * 70)
    print()

    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Testing: {symbol}")
        print(f"{'='*50}")

        backtester = LOCBRealDataBacktester(
            symbol=symbol,
            initial_balance=balance,
            risk_percent=risk
        )

        result = backtester.run_backtest(days=days)

        if result:
            all_results[symbol] = result

            print(f"\n--- {symbol} Results ---")
            print(f"Total Trades: {result.total_trades}")
            print(f"Wins: {result.wins} | Losses: {result.losses}")
            print(f"Win Rate: {result.win_rate:.1f}%")
            print(f"Profit Factor: {result.profit_factor:.2f}")
            print(f"Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_percent:+.2f}%)")
            print(f"Avg Win: {result.avg_win_percent:+.2f}% | Avg Loss: {result.avg_loss_percent:.2f}%")
            print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
            print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

            # Show individual trades
            if result.trades:
                print(f"\nLast 5 trades:")
                for t in result.trades[-5:]:
                    print(f"  {t.entry_time.strftime('%m/%d %H:%M')} {t.direction.upper():5} "
                          f"SL:{t.sl_pips:.1f}p Lot:{t.lot_size:.2f} "
                          f"Risk:${t.risk_amount:.0f} PnL:${t.pnl:+.2f} ({t.pnl_percent:+.2f}%) [{t.result}]")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY - ALL INSTRUMENTS (REAL DATA)")
    print("=" * 70)
    print(f"{ 'Symbol':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    print("-" * 70)

    total_pnl = 0
    total_trades = 0

    for symbol, result in all_results.items():
        print(f"{symbol:<10} {result.total_trades:<8} {result.win_rate:<8.1f} "
              f"{result.profit_factor:<8.2f} ${result.total_pnl:<11,.2f} {result.max_drawdown_percent:<8.2f}%")
        total_pnl += result.total_pnl
        total_trades += result.total_trades

    print("-" * 70)
    print(f"{ 'TOTAL':<10} ${total_pnl:<11,.2f}") # Corrected to show total pnl
    print("=" * 70)

    # Verification
    print("\n" + "=" * 70)
    print("RISK VERIFICATION")
    print("=" * 70)

    all_losses = []
    for result in all_results.values():
        for t in result.trades:
            if t.result == "loss":
                all_losses.append(t)

    if all_losses:
        avg_loss_pct = sum(t.pnl_percent for t in all_losses) / len(all_losses)
        max_loss_pct = min(t.pnl_percent for t in all_losses)
        print(f"Expected Loss per Trade: -{risk}%")
        print(f"Actual Avg Loss: {avg_loss_pct:.2f}%")
        print(f"Actual Max Loss: {max_loss_pct:.2f}%")

        if abs(avg_loss_pct) <= risk * 1.1:  # Within 10% tolerance
            print("[OK] RISK IS CORRECT!")
        else:
            print("[ERROR] WARNING: RISK CALCULATION MAY BE OFF!")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOCB Strategy Backtest with Real MT5 Data")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="Symbols to test")
    parser.add_argument("--days", type=int, default=60,
                       help="Days of history (default: 60)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent per trade (default: 0.5)")

    args = parser.parse_args()

    results = run_locb_real_backtest(
        symbols=args.symbols,
        days=args.days,
        balance=args.balance,
        risk=args.risk
    )
