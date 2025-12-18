"""
SMC (Smart Money Concepts) Strategy Backtest

Backtest the SMC strategy on historical data.
Tests multiple symbols: EURUSD, GBPUSD, USDJPY, EURJPY

Timeframes used:
- H4: Bias determination
- H1: Structure analysis, POI identification
- M5: Entry execution with CHoCH confirmation

Sessions (UTC):
- London: 07:00-11:00
- NY: 13:00-17:00

Usage:
    python backtests/backtest_smc.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not available. Using simulated data.")

from config.settings import get_settings
from models.candle import Candle
from strategies.smc_strategy import (
    SMCStrategy, SMCConfig, MarketBias,
    POI, OrderBlock, FairValueGap, LiquidityLevel
)


@dataclass
class SMCTrade:
    """Trade record for SMC backtest."""
    symbol: str
    day: str
    session: str  # "london" or "ny"
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    poi_score: int
    poi_type: str
    h4_bias: str
    h1_bias: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_pips: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    rr_actual: float = 0.0
    result: str = ""  # "WIN", "LOSS", "BE"


@dataclass
class BacktestStats:
    """Statistics from backtest."""
    symbol: str
    period_days: int
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    gross_profit_pips: float = 0.0
    gross_loss_pips: float = 0.0
    profit_factor: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    avg_rr: float = 0.0
    max_drawdown_pips: float = 0.0
    max_consecutive_losses: int = 0
    london_trades: int = 0
    ny_trades: int = 0
    london_win_rate: float = 0.0
    ny_win_rate: float = 0.0
    trades: List[SMCTrade] = field(default_factory=list)


def rates_to_candles(rates, symbol: str, timeframe: str) -> List[Candle]:
    """Convert MT5 rates to Candle objects."""
    candles = []
    for rate in rates:
        candles.append(Candle(
            timestamp=datetime.fromtimestamp(rate["time"], tz=timezone.utc),
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            volume=float(rate["tick_volume"]),
            symbol=symbol,
            timeframe=timeframe
        ))
    return candles


def get_pip_size(symbol: str) -> float:
    """Get pip size for symbol."""
    if "JPY" in symbol:
        return 0.01
    return 0.0001


def get_session(hour_utc: int) -> Optional[str]:
    """Determine trading session from UTC hour."""
    if 7 <= hour_utc < 11:
        return "london"
    elif 13 <= hour_utc < 17:
        return "ny"
    return None


def run_smc_backtest(
    symbol: str,
    h4_candles: List[Candle],
    h1_candles: List[Candle],
    m5_candles: List[Candle],
    config: Optional[SMCConfig] = None,
) -> BacktestStats:
    """
    Run SMC backtest on historical data.

    Args:
        symbol: Trading symbol
        h4_candles: H4 candle data
        h1_candles: H1 candle data
        m5_candles: M5 candle data (main timeframe)
        config: SMC configuration

    Returns:
        BacktestStats with all metrics
    """
    pip_size = get_pip_size(symbol)

    # Create strategy
    if config is None:
        config = SMCConfig(
            poi_min_score=3,
            min_rr=1.5,
            news_buffer_hours=2,
            max_trades_per_day=3,
        )

    strategy = SMCStrategy(
        symbol=symbol,
        timeframe="M5",
        config=config,
        use_news_filter=False,  # Disable for backtest
    )
    strategy.pip_size = pip_size
    strategy.initialize()

    trades: List[SMCTrade] = []
    current_trade: Optional[SMCTrade] = None

    # Build time-indexed data for lookups
    h4_by_time = {c.timestamp: c for c in h4_candles}
    h1_by_time = {c.timestamp: c for c in h1_candles}

    # Track daily trades
    last_date = None
    trades_today = 0

    # Process M5 candles
    for i in range(100, len(m5_candles)):  # Need history
        candle = m5_candles[i]
        ts = candle.timestamp

        # Check for new day
        candle_date = ts.date()
        if candle_date != last_date:
            last_date = candle_date
            trades_today = 0
            strategy._state.trades_today = 0

        # Skip weekends
        if ts.weekday() >= 5:
            continue

        # Check session
        hour_utc = ts.hour
        session = get_session(hour_utc)

        if session is None:
            continue

        # Get historical candles for each timeframe
        # H4: need ~50 candles for bias
        h4_history = [c for c in h4_candles if c.timestamp <= ts][-50:]
        # H1: need ~100 candles for structure/POI
        h1_history = [c for c in h1_candles if c.timestamp <= ts][-100:]
        # M5: need ~100 candles for entry
        m5_history = m5_candles[max(0, i-99):i+1]

        if len(h4_history) < 30 or len(h1_history) < 50:
            continue

        # Manage existing trade
        if current_trade:
            # Check SL/TP
            if current_trade.direction == "LONG":
                # Check SL hit
                if candle.low <= current_trade.stop_loss:
                    current_trade.exit_time = ts
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                    current_trade.result = "LOSS"
                    current_trade.exit_reason = "SL"
                    trades.append(current_trade)
                    current_trade = None
                    continue

                # Check TP hit
                if candle.high >= current_trade.take_profit:
                    current_trade.exit_time = ts
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                    current_trade.result = "WIN"
                    current_trade.exit_reason = "TP"
                    trades.append(current_trade)
                    current_trade = None
                    continue

            else:  # SHORT
                # Check SL hit
                if candle.high >= current_trade.stop_loss:
                    current_trade.exit_time = ts
                    current_trade.exit_price = current_trade.stop_loss
                    current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                    current_trade.result = "LOSS"
                    current_trade.exit_reason = "SL"
                    trades.append(current_trade)
                    current_trade = None
                    continue

                # Check TP hit
                if candle.low <= current_trade.take_profit:
                    current_trade.exit_time = ts
                    current_trade.exit_price = current_trade.take_profit
                    current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                    current_trade.result = "WIN"
                    current_trade.exit_reason = "TP"
                    trades.append(current_trade)
                    current_trade = None
                    continue

            # Check session end - close 10 min before
            if session == "london" and hour_utc >= 10 and ts.minute >= 50:
                current_trade.exit_time = ts
                current_trade.exit_price = candle.close
                if current_trade.direction == "LONG":
                    current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                else:
                    current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                current_trade.result = "WIN" if current_trade.pnl_pips > 0 else "LOSS"
                current_trade.exit_reason = "SESSION_END"
                trades.append(current_trade)
                current_trade = None
                continue

            if session == "ny" and hour_utc >= 16 and ts.minute >= 50:
                current_trade.exit_time = ts
                current_trade.exit_price = candle.close
                if current_trade.direction == "LONG":
                    current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                else:
                    current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                current_trade.result = "WIN" if current_trade.pnl_pips > 0 else "LOSS"
                current_trade.exit_reason = "SESSION_END"
                trades.append(current_trade)
                current_trade = None
                continue

        # Look for new entry if no position and not at max trades
        if current_trade is None and trades_today < config.max_trades_per_day:
            # Set multi-timeframe data
            strategy.set_candles(h4_history, h1_history, m5_history)

            # Get signal
            signal = strategy.on_candle(candle, m5_history)

            if signal and signal.is_entry:
                # Create trade
                direction = "LONG" if signal.is_long else "SHORT"
                sl_pips = abs(signal.price - signal.stop_loss) / pip_size
                tp_pips = abs(signal.take_profit - signal.price) / pip_size
                rr = tp_pips / sl_pips if sl_pips > 0 else 0

                current_trade = SMCTrade(
                    symbol=symbol,
                    day=str(candle_date),
                    session=session,
                    direction=direction,
                    entry_time=ts,
                    entry_price=signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    poi_score=signal.metadata.get("poi_score", 0),
                    poi_type=signal.metadata.get("poi_type", ""),
                    h4_bias=signal.metadata.get("h4_bias", ""),
                    h1_bias=signal.metadata.get("h1_bias", ""),
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    rr_actual=rr,
                )
                trades_today += 1

    # Close any remaining trade
    if current_trade and m5_candles:
        last_candle = m5_candles[-1]
        current_trade.exit_time = last_candle.timestamp
        current_trade.exit_price = last_candle.close
        if current_trade.direction == "LONG":
            current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
        else:
            current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
        current_trade.result = "WIN" if current_trade.pnl_pips > 0 else "LOSS"
        current_trade.exit_reason = "BACKTEST_END"
        trades.append(current_trade)

    # Calculate statistics
    stats = calculate_stats(symbol, trades, m5_candles)
    return stats


def calculate_stats(symbol: str, trades: List[SMCTrade], candles: List[Candle]) -> BacktestStats:
    """Calculate backtest statistics."""
    if candles:
        period_days = (candles[-1].timestamp - candles[0].timestamp).days
    else:
        period_days = 0

    stats = BacktestStats(
        symbol=symbol,
        period_days=period_days,
        trades=trades,
    )

    if not trades:
        return stats

    stats.total_trades = len(trades)

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]

    stats.wins = len(wins)
    stats.losses = len(losses)
    stats.win_rate = (len(wins) / len(trades)) * 100 if trades else 0

    # PnL
    stats.total_pips = sum(t.pnl_pips for t in trades)
    stats.gross_profit_pips = sum(t.pnl_pips for t in wins) if wins else 0
    stats.gross_loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0

    if stats.gross_loss_pips > 0:
        stats.profit_factor = stats.gross_profit_pips / stats.gross_loss_pips
    else:
        stats.profit_factor = stats.gross_profit_pips if stats.gross_profit_pips > 0 else 0

    if wins:
        stats.avg_win_pips = stats.gross_profit_pips / len(wins)
    if losses:
        stats.avg_loss_pips = stats.gross_loss_pips / len(losses)

    # Average R:R
    rr_values = [t.rr_actual for t in trades if t.rr_actual > 0]
    stats.avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    consecutive_losses = 0
    max_consec_losses = 0

    for t in trades:
        cumulative += t.pnl_pips
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

        if t.result == "LOSS":
            consecutive_losses += 1
            max_consec_losses = max(max_consec_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    stats.max_drawdown_pips = max_dd
    stats.max_consecutive_losses = max_consec_losses

    # Session breakdown
    london_trades = [t for t in trades if t.session == "london"]
    ny_trades = [t for t in trades if t.session == "ny"]

    stats.london_trades = len(london_trades)
    stats.ny_trades = len(ny_trades)

    london_wins = len([t for t in london_trades if t.result == "WIN"])
    ny_wins = len([t for t in ny_trades if t.result == "WIN"])

    stats.london_win_rate = (london_wins / len(london_trades)) * 100 if london_trades else 0
    stats.ny_win_rate = (ny_wins / len(ny_trades)) * 100 if ny_trades else 0

    return stats


def print_stats(stats: BacktestStats, show_trades: bool = False):
    """Print backtest statistics."""
    print()
    print("=" * 70)
    print(f"  {stats.symbol} - SMC BACKTEST RESULTS")
    print(f"  Period: {stats.period_days} days")
    print("=" * 70)

    print(f"\n  PERFORMANCE:")
    print(f"  {'-' * 40}")
    print(f"  Total Trades:     {stats.total_trades}")
    print(f"  Wins/Losses:      {stats.wins}/{stats.losses}")
    print(f"  Win Rate:         {stats.win_rate:.1f}%")
    print(f"  Total PnL:        {stats.total_pips:+.1f} pips")
    print(f"  Profit Factor:    {stats.profit_factor:.2f}")
    print(f"  Max Drawdown:     {stats.max_drawdown_pips:.1f} pips")
    print(f"  Max Consec Losses: {stats.max_consecutive_losses}")

    print(f"\n  AVERAGES:")
    print(f"  {'-' * 40}")
    print(f"  Avg Win:          {stats.avg_win_pips:.1f} pips")
    print(f"  Avg Loss:         {stats.avg_loss_pips:.1f} pips")
    print(f"  Avg R:R:          {stats.avg_rr:.2f}")

    print(f"\n  SESSION BREAKDOWN:")
    print(f"  {'-' * 40}")
    print(f"  London:           {stats.london_trades} trades ({stats.london_win_rate:.1f}% WR)")
    print(f"  New York:         {stats.ny_trades} trades ({stats.ny_win_rate:.1f}% WR)")

    if show_trades and stats.trades:
        print(f"\n  TRADE DETAILS:")
        print(f"  {'-' * 90}")
        print(f"  {'Date':<12} | {'Session':<7} | {'Dir':<5} | {'Entry':>8} | {'SL':>6} | {'TP':>6} | {'PnL':>7} | {'Result':<6}")
        print(f"  {'-' * 90}")

        for t in stats.trades[:50]:  # Limit to first 50
            print(
                f"  {t.day:<12} | {t.session:<7} | {t.direction:<5} | "
                f"{t.entry_price:>8.5f} | {t.sl_pips:>5.1f}p | {t.tp_pips:>5.1f}p | "
                f"{t.pnl_pips:>+6.1f}p | {t.result:<6}"
            )

        if len(stats.trades) > 50:
            print(f"  ... and {len(stats.trades) - 50} more trades")

    print()


def download_data(symbol: str, days: int = 365) -> Tuple[List[Candle], List[Candle], List[Candle]]:
    """Download historical data from MT5."""
    if not MT5_AVAILABLE:
        return [], [], []

    settings = get_settings()

    if not mt5.initialize(
        login=settings.mt5.login,
        password=settings.mt5.password,
        server=settings.mt5.server,
        path=settings.mt5.path,
        timeout=60000,
    ):
        print(f"MT5 init failed: {mt5.last_error()}")
        return [], [], []

    mt5.symbol_select(symbol, True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"  Downloading {symbol} data for {days} days...")

    # H4 data
    h4_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H4, start_date, end_date)
    h4_candles = rates_to_candles(h4_rates, symbol, "H4") if h4_rates is not None else []
    print(f"    H4: {len(h4_candles)} candles")

    # H1 data
    h1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
    h1_candles = rates_to_candles(h1_rates, symbol, "H1") if h1_rates is not None else []
    print(f"    H1: {len(h1_candles)} candles")

    # M5 data - may be limited to ~6 months
    m5_candles = []
    for try_days in [days, 180, 120, 90, 60]:
        try_start = end_date - timedelta(days=try_days)
        m5_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, try_start, end_date)
        if m5_rates is not None and len(m5_rates) > 1000:
            m5_candles = rates_to_candles(m5_rates, symbol, "M5")
            print(f"    M5: {len(m5_candles)} candles ({try_days} days)")
            break
        print(f"    M5: Trying {try_days} days... insufficient data")

    return h4_candles, h1_candles, m5_candles


def main():
    """Run SMC backtest."""
    print()
    print("=" * 70)
    print("       SMC (SMART MONEY CONCEPTS) STRATEGY BACKTEST")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Symbols to test
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]

    # Target period
    target_days = 365  # 1 year

    # SMC Configuration
    config = SMCConfig(
        # Structure
        poi_min_score=3,
        bos_min_move_atr=2.0,
        ob_min_impulse_atr=2.0,
        fvg_min_gap_atr=0.5,

        # Entry
        entry_mode="conservative",
        choch_lookback=8,

        # Risk
        min_rr=1.5,
        max_sl_atr=3.0,

        # Sessions
        london_start_hour=7,
        london_end_hour=11,
        ny_start_hour=13,
        ny_end_hour=17,

        # Limits
        max_trades_per_day=3,
    )

    all_stats: List[BacktestStats] = []

    if MT5_AVAILABLE:
        settings = get_settings()

        # Initialize MT5 once
        if not mt5.initialize(
            login=settings.mt5.login,
            password=settings.mt5.password,
            server=settings.mt5.server,
            path=settings.mt5.path,
            timeout=60000,
        ):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            print("Make sure MT5 terminal is open and logged in.")
            return

        print("MT5 connected successfully!")
        print()

        for symbol in symbols:
            print(f"\n{'='*70}")
            print(f"  Processing {symbol}...")
            print(f"{'='*70}")

            h4_candles, h1_candles, m5_candles = download_data(symbol, target_days)

            if not m5_candles or len(m5_candles) < 1000:
                print(f"  Insufficient data for {symbol}, skipping...")
                continue

            print(f"\n  Running backtest...")
            stats = run_smc_backtest(symbol, h4_candles, h1_candles, m5_candles, config)
            all_stats.append(stats)

            print_stats(stats, show_trades=True)

        mt5.shutdown()

    else:
        print("MT5 not available. Cannot run backtest without historical data.")
        print("Please ensure MetaTrader5 is installed and terminal is running.")
        return

    # Combined summary
    if all_stats:
        print("\n")
        print("=" * 70)
        print("       COMBINED SUMMARY - ALL SYMBOLS")
        print("=" * 70)

        total_trades = sum(s.total_trades for s in all_stats)
        total_wins = sum(s.wins for s in all_stats)
        total_pips = sum(s.total_pips for s in all_stats)
        gross_profit = sum(s.gross_profit_pips for s in all_stats)
        gross_loss = sum(s.gross_loss_pips for s in all_stats)

        combined_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        combined_pf = gross_profit / gross_loss if gross_loss > 0 else 0

        print(f"\n  {'Symbol':<10} | {'Trades':>7} | {'Win Rate':>8} | {'PnL':>10} | {'PF':>6} | {'MaxDD':>8}")
        print(f"  {'-' * 65}")

        for s in all_stats:
            print(
                f"  {s.symbol:<10} | {s.total_trades:>7} | {s.win_rate:>7.1f}% | "
                f"{s.total_pips:>+9.1f}p | {s.profit_factor:>5.2f} | {s.max_drawdown_pips:>7.1f}p"
            )

        print(f"  {'-' * 65}")
        print(
            f"  {'TOTAL':<10} | {total_trades:>7} | {combined_wr:>7.1f}% | "
            f"{total_pips:>+9.1f}p | {combined_pf:>5.2f} |"
        )

        print(f"\n  RECOMMENDATION:")
        print(f"  {'-' * 40}")

        # Find best performing symbol
        best = max(all_stats, key=lambda s: s.total_pips) if all_stats else None
        if best:
            print(f"  Best Symbol: {best.symbol} ({best.total_pips:+.1f} pips)")

        if combined_pf >= 1.5:
            print(f"  Strategy Status: PROFITABLE (PF {combined_pf:.2f})")
        elif combined_pf >= 1.0:
            print(f"  Strategy Status: BREAKEVEN (PF {combined_pf:.2f})")
        else:
            print(f"  Strategy Status: NEEDS OPTIMIZATION (PF {combined_pf:.2f})")

        print()
        print("=" * 70)
        print("  BACKTEST COMPLETE")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
