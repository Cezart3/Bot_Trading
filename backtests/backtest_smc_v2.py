"""
SMC V2 Strategy Backtest - 6 Month Test.

This script tests the optimized SMC V2 strategy on EURUSD and optionally on indices.
Uses realistic market simulation with proper volatility patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pytz

from models.candle import Candle
from strategies.smc_strategy_v2 import SMCStrategyV2, SMCConfigV2, MarketBias
from utils.logger import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Trade record for backtest."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    entry_time: datetime
    exit_time: datetime
    pnl_pips: float
    pnl_dollars: float
    exit_reason: str
    rr_achieved: float
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Backtest results summary."""
    symbol: str
    period_days: int
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pips: float
    max_drawdown: float
    max_drawdown_percent: float
    profit_factor: float
    average_win: float
    average_loss: float
    average_rr: float
    best_trade: float
    worst_trade: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)


class RealisticMarketSimulator:
    """
    Generates realistic OHLC data with proper market characteristics:
    - Session-based volatility (London/NY higher, Asian lower)
    - Trend persistence
    - Mean reversion
    - Liquidity sweeps
    - Order block formation
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        base_price: float = 1.0850,
        pip_size: float = 0.0001,
        seed: Optional[int] = None
    ):
        self.symbol = symbol
        self.base_price = base_price
        self.pip_size = pip_size

        if seed is not None:
            random.seed(seed)

        # Volatility parameters (in pips)
        self.atr_london = 12  # Higher volatility
        self.atr_ny = 15
        self.atr_asian = 6
        self.atr_off_hours = 4

        # Trend parameters
        self.trend_strength = 0.0
        self.trend_duration = 0
        self.max_trend_duration = 200  # candles

    def _get_session_atr(self, hour_utc: int) -> float:
        """Get ATR based on session."""
        if 7 <= hour_utc < 12:
            return self.atr_london
        elif 12 <= hour_utc < 17:
            return self.atr_ny
        elif 0 <= hour_utc < 7:
            return self.atr_asian
        else:
            return self.atr_off_hours

    def _update_trend(self) -> None:
        """Update trend direction and strength."""
        self.trend_duration += 1

        # Trend reversal probability increases with duration
        reversal_prob = min(0.3, self.trend_duration / self.max_trend_duration)

        if random.random() < reversal_prob:
            # Reverse or go neutral
            if random.random() < 0.3:
                self.trend_strength = 0  # Neutral
            else:
                self.trend_strength = -self.trend_strength * random.uniform(0.5, 1.0)
            self.trend_duration = 0

        # Occasionally start new trend
        if self.trend_strength == 0 and random.random() < 0.05:
            self.trend_strength = random.choice([-1, 1]) * random.uniform(0.3, 0.8)
            self.trend_duration = 0

    def generate_candle(
        self,
        prev_close: float,
        timestamp: datetime,
        timeframe_minutes: int = 5
    ) -> Candle:
        """Generate a realistic candle."""
        hour_utc = timestamp.hour
        atr_pips = self._get_session_atr(hour_utc)
        atr = atr_pips * self.pip_size

        self._update_trend()

        # Base movement
        move_range = atr * random.uniform(0.3, 1.5)

        # Trend bias
        trend_bias = self.trend_strength * atr * 0.3

        # Random walk with trend
        open_price = prev_close
        direction = 1 if random.random() + self.trend_strength * 0.2 > 0.5 else -1

        # Generate OHLC
        body_size = move_range * random.uniform(0.2, 0.8)
        upper_wick = move_range * random.uniform(0.1, 0.5)
        lower_wick = move_range * random.uniform(0.1, 0.5)

        if direction > 0:  # Bullish
            close_price = open_price + body_size + trend_bias
            high = close_price + upper_wick
            low = open_price - lower_wick
        else:  # Bearish
            close_price = open_price - body_size + trend_bias
            low = close_price - lower_wick
            high = open_price + upper_wick

        # Occasionally create liquidity sweeps (wicks beyond recent levels)
        if random.random() < 0.05:  # 5% chance
            if direction > 0:
                low = open_price - lower_wick * 2
            else:
                high = open_price + upper_wick * 2

        return Candle(
            symbol=self.symbol,
            timestamp=timestamp,
            open=round(open_price, 5),
            high=round(max(high, open_price, close_price), 5),
            low=round(min(low, open_price, close_price), 5),
            close=round(close_price, 5),
            volume=random.randint(100, 10000)
        )

    def generate_candles(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe_minutes: int = 5
    ) -> List[Candle]:
        """Generate candles for a date range."""
        candles = []
        current_time = start_date
        current_price = self.base_price

        delta = timedelta(minutes=timeframe_minutes)

        while current_time <= end_date:
            # Skip weekends
            if current_time.weekday() < 5:
                candle = self.generate_candle(current_price, current_time, timeframe_minutes)
                candles.append(candle)
                current_price = candle.close

            current_time += delta

        return candles


def resample_candles(candles: List[Candle], target_minutes: int) -> List[Candle]:
    """Resample candles to a higher timeframe."""
    if not candles:
        return []

    resampled = []
    current_group = []

    for candle in candles:
        # Determine which group this candle belongs to
        group_start = candle.timestamp.replace(
            minute=(candle.timestamp.minute // target_minutes) * target_minutes,
            second=0,
            microsecond=0
        )

        if not current_group:
            current_group = [candle]
        elif candle.timestamp.replace(minute=(candle.timestamp.minute // target_minutes) * target_minutes) == \
             current_group[0].timestamp.replace(minute=(current_group[0].timestamp.minute // target_minutes) * target_minutes):
            current_group.append(candle)
        else:
            # Create resampled candle
            resampled.append(Candle(
                symbol=current_group[0].symbol,
                timestamp=current_group[0].timestamp.replace(
                    minute=(current_group[0].timestamp.minute // target_minutes) * target_minutes
                ),
                open=current_group[0].open,
                high=max(c.high for c in current_group),
                low=min(c.low for c in current_group),
                close=current_group[-1].close,
                volume=sum(c.volume for c in current_group)
            ))
            current_group = [candle]

    # Don't forget the last group
    if current_group:
        resampled.append(Candle(
            symbol=current_group[0].symbol,
            timestamp=current_group[0].timestamp.replace(
                minute=(current_group[0].timestamp.minute // target_minutes) * target_minutes
            ),
            open=current_group[0].open,
            high=max(c.high for c in current_group),
            low=min(c.low for c in current_group),
            close=current_group[-1].close,
            volume=sum(c.volume for c in current_group)
        ))

    return resampled


class SMCBacktestEngine:
    """Backtest engine optimized for SMC V2 strategy."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 1.0,
        commission_per_lot: float = 7.0,
        pip_value: float = 10.0,  # $10 per pip per standard lot for EURUSD
        lot_size: float = 0.1,  # Default lot size
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission_per_lot = commission_per_lot
        self.pip_value = pip_value
        self.lot_size = lot_size

        self.balance = initial_balance
        self.equity = initial_balance
        self.peak_equity = initial_balance
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def _calculate_lot_size(self, sl_pips: float, pip_size: float) -> float:
        """Calculate lot size based on risk."""
        risk_amount = self.balance * (self.risk_per_trade / 100)
        lot_size = risk_amount / (sl_pips * self.pip_value)
        return max(0.01, min(1.0, round(lot_size, 2)))

    def run(
        self,
        strategy: SMCStrategyV2,
        m5_candles: List[Candle],
        m15_candles: List[Candle],
        h1_candles: List[Candle],
        h4_candles: List[Candle],
    ) -> BacktestResult:
        """Run backtest on historical data."""
        logger.info(f"Starting backtest with {len(m5_candles)} M5 candles")
        logger.info(f"Period: {m5_candles[0].timestamp} to {m5_candles[-1].timestamp}")
        logger.info(f"Initial balance: ${self.initial_balance:,.2f}")

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.trades = []
        self.equity_curve = []

        strategy.initialize()
        pip_size = strategy.pip_size

        # Position tracking
        position = None
        position_entry_time = None

        # Create index maps for efficient lookups
        h1_by_time = {c.timestamp: c for c in h1_candles}
        h4_by_time = {c.timestamp: c for c in h4_candles}
        m15_by_time = {c.timestamp: c for c in m15_candles}

        last_date = None

        for i, candle in enumerate(m5_candles):
            ts = candle.timestamp

            # Skip if not enough history
            if i < 300:
                continue

            # New day reset
            if last_date != ts.date():
                last_date = ts.date()
                strategy.reset()

            # Get historical candles for each timeframe
            m5_history = m5_candles[max(0, i-100):i+1]

            # Find corresponding H1 candles
            h1_time = ts.replace(minute=0, second=0, microsecond=0)
            h1_idx = next((j for j, c in enumerate(h1_candles) if c.timestamp >= h1_time), len(h1_candles))
            h1_history = h1_candles[max(0, h1_idx-100):h1_idx+1] if h1_idx > 0 else []

            # Find corresponding H4 candles
            h4_hour = (ts.hour // 4) * 4
            h4_time = ts.replace(hour=h4_hour, minute=0, second=0, microsecond=0)
            h4_idx = next((j for j, c in enumerate(h4_candles) if c.timestamp >= h4_time), len(h4_candles))
            h4_history = h4_candles[max(0, h4_idx-50):h4_idx+1] if h4_idx > 0 else []

            # Find corresponding M15 candles
            m15_minute = (ts.minute // 15) * 15
            m15_time = ts.replace(minute=m15_minute, second=0, microsecond=0)
            m15_idx = next((j for j, c in enumerate(m15_candles) if c.timestamp >= m15_time), len(m15_candles))
            m15_history = m15_candles[max(0, m15_idx-100):m15_idx+1] if m15_idx > 0 else []

            if not h1_history or not h4_history:
                continue

            # Set multi-timeframe data
            strategy.set_candles(
                h4_candles=h4_history,
                h1_candles=h1_history,
                m5_candles=m5_history,
                m15_candles=m15_history
            )

            # Update equity curve
            self.equity_curve.append((ts, self.balance))

            # Check position status
            if position:
                # Check SL/TP
                if position["direction"] == "long":
                    if candle.low <= position["sl"]:
                        # Hit SL
                        exit_price = position["sl"]
                        pnl_pips = (exit_price - position["entry"]) / pip_size
                        pnl = pnl_pips * self.pip_value * position["lots"]
                        self._close_position(position, exit_price, ts, pnl_pips, pnl, "sl", pip_size)
                        position = None
                    elif candle.high >= position["tp"]:
                        # Hit TP
                        exit_price = position["tp"]
                        pnl_pips = (exit_price - position["entry"]) / pip_size
                        pnl = pnl_pips * self.pip_value * position["lots"]
                        self._close_position(position, exit_price, ts, pnl_pips, pnl, "tp", pip_size)
                        position = None
                else:  # Short
                    if candle.high >= position["sl"]:
                        exit_price = position["sl"]
                        pnl_pips = (position["entry"] - exit_price) / pip_size
                        pnl = pnl_pips * self.pip_value * position["lots"]
                        self._close_position(position, exit_price, ts, pnl_pips, pnl, "sl", pip_size)
                        position = None
                    elif candle.low <= position["tp"]:
                        exit_price = position["tp"]
                        pnl_pips = (position["entry"] - exit_price) / pip_size
                        pnl = pnl_pips * self.pip_value * position["lots"]
                        self._close_position(position, exit_price, ts, pnl_pips, pnl, "tp", pip_size)
                        position = None

            # Check for new signals only if no position
            if not position:
                signal = strategy.on_candle(candle, m5_history)

                if signal and signal.stop_loss and signal.take_profit:
                    sl_pips = abs(signal.price - signal.stop_loss) / pip_size
                    lots = self._calculate_lot_size(sl_pips, pip_size)

                    position = {
                        "direction": "long" if signal.signal_type.value == "long" else "short",
                        "entry": signal.price,
                        "sl": signal.stop_loss,
                        "tp": signal.take_profit,
                        "lots": lots,
                        "entry_time": ts,
                        "metadata": signal.metadata
                    }
                    position_entry_time = ts

        # Close any remaining position
        if position and m5_candles:
            last_candle = m5_candles[-1]
            exit_price = last_candle.close
            if position["direction"] == "long":
                pnl_pips = (exit_price - position["entry"]) / pip_size
            else:
                pnl_pips = (position["entry"] - exit_price) / pip_size
            pnl = pnl_pips * self.pip_value * position["lots"]
            self._close_position(position, exit_price, last_candle.timestamp, pnl_pips, pnl, "eod", pip_size)

        # Calculate results
        return self._calculate_results(strategy.symbol, len(m5_candles) // 288)  # Approximate days

    def _close_position(
        self,
        position: dict,
        exit_price: float,
        exit_time: datetime,
        pnl_pips: float,
        pnl: float,
        exit_reason: str,
        pip_size: float
    ) -> None:
        """Close a position and record the trade."""
        # Apply commission
        commission = self.commission_per_lot * position["lots"] * 2  # Entry + exit
        net_pnl = pnl - commission

        self.balance += net_pnl

        # Track peak for drawdown
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance

        # Calculate R:R achieved
        risk = abs(position["entry"] - position["sl"])
        if risk > 0:
            if position["direction"] == "long":
                rr_achieved = (exit_price - position["entry"]) / risk
            else:
                rr_achieved = (position["entry"] - exit_price) / risk
        else:
            rr_achieved = 0

        trade = BacktestTrade(
            symbol="EURUSD",
            direction=position["direction"],
            entry_price=position["entry"],
            exit_price=exit_price,
            sl=position["sl"],
            tp=position["tp"],
            entry_time=position["entry_time"],
            exit_time=exit_time,
            pnl_pips=pnl_pips,
            pnl_dollars=net_pnl,
            exit_reason=exit_reason,
            rr_achieved=rr_achieved,
            metadata=position.get("metadata", {})
        )

        self.trades.append(trade)

        logger.debug(
            f"Trade closed: {position['direction']} | "
            f"PnL: {pnl_pips:.1f} pips (${net_pnl:.2f}) | "
            f"Reason: {exit_reason} | Balance: ${self.balance:.2f}"
        )

    def _calculate_results(self, symbol: str, period_days: int) -> BacktestResult:
        """Calculate backtest performance metrics."""
        wins = [t for t in self.trades if t.pnl_dollars > 0]
        losses = [t for t in self.trades if t.pnl_dollars <= 0]

        total_pnl = sum(t.pnl_dollars for t in self.trades)
        total_pnl_pips = sum(t.pnl_pips for t in self.trades)

        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0

        avg_win = sum(t.pnl_dollars for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl_dollars for t in losses) / len(losses)) if losses else 0

        gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_rr = sum(t.rr_achieved for t in self.trades) / len(self.trades) if self.trades else 0

        # Calculate max drawdown
        max_dd = 0
        max_dd_pct = 0
        peak = self.initial_balance

        for ts, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        best_trade = max(t.pnl_dollars for t in self.trades) if self.trades else 0
        worst_trade = min(t.pnl_dollars for t in self.trades) if self.trades else 0

        return BacktestResult(
            symbol=symbol,
            period_days=period_days,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pips=total_pnl_pips,
            max_drawdown=max_dd,
            max_drawdown_percent=max_dd_pct,
            profit_factor=profit_factor,
            average_win=avg_win,
            average_loss=avg_loss,
            average_rr=avg_rr,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=self.trades,
            equity_curve=self.equity_curve
        )


def print_results(result: BacktestResult) -> None:
    """Print backtest results."""
    print("\n" + "=" * 70)
    print(f"  SMC V2 BACKTEST RESULTS - {result.symbol}")
    print("=" * 70)

    print(f"\n  Period: {result.period_days} days")
    print(f"  Initial Balance: ${result.initial_balance:,.2f}")
    print(f"  Final Balance:   ${result.final_balance:,.2f}")

    pnl_pct = (result.final_balance - result.initial_balance) / result.initial_balance * 100
    print(f"\n  Total P&L: ${result.total_pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"  Total Pips: {result.total_pnl_pips:+.1f}")

    print(f"\n  Total Trades: {result.total_trades}")
    print(f"  Winning Trades: {result.winning_trades}")
    print(f"  Losing Trades: {result.losing_trades}")
    print(f"  Win Rate: {result.win_rate:.1f}%")

    print(f"\n  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Average Win: ${result.average_win:.2f}")
    print(f"  Average Loss: ${result.average_loss:.2f}")
    print(f"  Average R:R: {result.average_rr:.2f}")

    print(f"\n  Best Trade: ${result.best_trade:.2f}")
    print(f"  Worst Trade: ${result.worst_trade:.2f}")

    print(f"\n  Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_percent:.2f}%)")

    # Monthly breakdown
    if result.trades:
        print("\n  Monthly Breakdown:")
        print("  " + "-" * 40)
        monthly_pnl = {}
        for trade in result.trades:
            month_key = trade.exit_time.strftime("%Y-%m")
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = {"pnl": 0, "trades": 0, "wins": 0}
            monthly_pnl[month_key]["pnl"] += trade.pnl_dollars
            monthly_pnl[month_key]["trades"] += 1
            if trade.pnl_dollars > 0:
                monthly_pnl[month_key]["wins"] += 1

        for month, data in sorted(monthly_pnl.items()):
            wr = (data["wins"] / data["trades"] * 100) if data["trades"] > 0 else 0
            print(f"  {month}: ${data['pnl']:+8.2f} | {data['trades']:2d} trades | {wr:.0f}% WR")

    print("\n" + "=" * 70)


def run_backtest(
    symbol: str = "EURUSD",
    days: int = 180,
    seed: int = 42
) -> BacktestResult:
    """Run complete backtest for a symbol."""
    logger.info(f"Running {days}-day backtest for {symbol}")

    # Generate data
    end_date = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    simulator = RealisticMarketSimulator(symbol=symbol, seed=seed)

    logger.info("Generating M5 candles...")
    m5_candles = simulator.generate_candles(start_date, end_date, timeframe_minutes=5)
    logger.info(f"Generated {len(m5_candles)} M5 candles")

    logger.info("Resampling to higher timeframes...")
    m15_candles = resample_candles(m5_candles, 15)
    h1_candles = resample_candles(m5_candles, 60)
    h4_candles = resample_candles(m5_candles, 240)

    logger.info(f"M15: {len(m15_candles)}, H1: {len(h1_candles)}, H4: {len(h4_candles)}")

    # Create strategy with optimized config - balanced approach
    config = SMCConfigV2(
        use_ema_filter=True,
        require_h4_agreement=False,  # Allow H1-only trades
        poi_min_score=1.8,  # Balanced score requirement
        poi_ob_score=1.2,   # Slightly higher OB score
        poi_sweep_score=1.0,  # Bonus for liquidity sweep
        poi_zone_score=0.5,  # Bonus for premium/discount
        poi_fresh_score=0.3,  # Bonus for fresh POI
        min_rr=1.8,  # Good min R:R
        target_rr=2.5,
        max_sl_pips=25.0,  # Reasonable SL
        min_sl_pips=8.0,
        max_trades_per_day=3,  # Allow up to 3 trades
        max_consecutive_losses=3,
        choch_lookback=12,  # Balanced lookback
    )

    strategy = SMCStrategyV2(
        symbol=symbol,
        timeframe="M5",
        config=config
    )

    # Run backtest
    engine = SMCBacktestEngine(
        initial_balance=10000.0,
        risk_per_trade=1.0,
        pip_value=10.0,
    )

    result = engine.run(
        strategy=strategy,
        m5_candles=m5_candles,
        m15_candles=m15_candles,
        h1_candles=h1_candles,
        h4_candles=h4_candles,
    )

    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("  SMC V2 STRATEGY BACKTEST - 6 MONTH TEST")
    print("=" * 70)

    # Run EURUSD backtest
    result = run_backtest(symbol="EURUSD", days=180, seed=42)
    print_results(result)

    # Summary
    print("\n\n" + "=" * 70)
    print("  PROFITABILITY SUMMARY")
    print("=" * 70)

    if result.total_pnl > 0:
        print(f"\n  [OK] Strategy is PROFITABLE")
        print(f"  [OK] Return: {((result.final_balance / result.initial_balance) - 1) * 100:.2f}%")
        print(f"  [OK] Profit Factor: {result.profit_factor:.2f}")
        print(f"  [OK] Win Rate: {result.win_rate:.1f}%")
    else:
        print(f"\n  [X] Strategy needs more optimization")
        print(f"  [X] Return: {((result.final_balance / result.initial_balance) - 1) * 100:.2f}%")

    # Key metrics for evaluation
    print(f"\n  Key Metrics for Prop Trading:")
    print(f"  - Max Drawdown: {result.max_drawdown_percent:.2f}% (Target: < 5%)")
    print(f"  - Profit Factor: {result.profit_factor:.2f} (Target: > 1.5)")
    print(f"  - Win Rate: {result.win_rate:.1f}% (Target: > 45%)")
    print(f"  - Avg R:R: {result.average_rr:.2f} (Target: > 1.5)")

    print("\n" + "=" * 70)

    return result


if __name__ == "__main__":
    main()
