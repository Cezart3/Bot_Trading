"""
Analiza detaliata a strategiei LOCB pentru identificarea oportunitatilor de imbunatatire.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from config.settings import get_settings
from collections import defaultdict


@dataclass
class M1Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class TradeResult:
    day: str
    session: str
    direction: str
    confirmation_type: str
    entry_time: datetime
    entry_price: float
    sl_price: float
    tp_price: float
    exit_price: float
    sl_pips: float
    tp_pips: float
    pnl_pips: float
    result: str
    rr_ratio: float
    # Additional analysis fields
    breakout_candle_body_pct: float = 0.0
    retest_quality: str = ""
    time_to_entry_minutes: int = 0


def detect_choch(candles: List[M1Candle], direction: str, lookback: int = 10) -> bool:
    if len(candles) < lookback + 2:
        return False
    recent = candles[-(lookback + 1):-1]
    current = candles[-1]
    if direction == "bullish":
        swing_high = max(c.high for c in recent)
        return current.close > swing_high
    else:
        swing_low = min(c.low for c in recent)
        return current.close < swing_low


def detect_engulfing(candles: List[M1Candle], direction: str) -> bool:
    if len(candles) < 2:
        return False
    prev, curr = candles[-2], candles[-1]
    prev_body_high = max(prev.open, prev.close)
    prev_body_low = min(prev.open, prev.close)
    curr_body_high = max(curr.open, curr.close)
    curr_body_low = min(curr.open, curr.close)
    if direction == "bullish":
        return (prev.close < prev.open and curr.close > curr.open and
                curr_body_low <= prev_body_low and curr_body_high >= prev_body_high)
    else:
        return (prev.close > prev.open and curr.close < curr.open and
                curr_body_low <= prev_body_low and curr_body_high >= prev_body_high)


def detect_ifvg(candles: List[M1Candle], direction: str, lookback: int = 20) -> bool:
    if len(candles) < lookback:
        return False
    recent = candles[-lookback:]
    fvgs = []
    for i in range(2, len(recent) - 1):
        c0, c2 = recent[i-2], recent[i]
        if c2.low > c0.high:
            fvgs.append(("bullish", c0.high, c2.low))
        if c2.high < c0.low:
            fvgs.append(("bearish", c2.high, c0.low))
    current = candles[-1]
    for fvg_type, low, high in fvgs:
        if direction == "bullish" and fvg_type == "bearish":
            if current.close > high:
                return True
        elif direction == "bearish" and fvg_type == "bullish":
            if current.close < low:
                return True
    return False


def check_retest(candles: List[M1Candle], level: float, direction: str, tolerance_pips: float = 3.0) -> bool:
    if not candles:
        return False
    pip_size = 0.0001
    tolerance = tolerance_pips * pip_size
    current = candles[-1]
    if direction == "bullish":
        return current.low <= level + tolerance and current.close > level
    else:
        return current.high >= level - tolerance and current.close < level


def run_detailed_backtest(
    m1_by_time: dict,
    days: List[date],
    symbol: str,
    session_name: str,
    open_hour: int,
    open_minute: int,
    end_hour: int,
    pip_size: float = 0.0001,
    sl_buffer_pips: float = 1.0,
    rr_ratio: float = 2.5,
    max_retest_candles: int = 30,
    max_confirm_candles: int = 20,
) -> List[TradeResult]:
    """Run detailed backtest collecting all metrics."""

    trades = []

    for day in days:
        if day.weekday() >= 5:
            continue

        session_open = datetime(day.year, day.month, day.day, open_hour, open_minute)

        # Collect opening range (5 minutes)
        opening_candles = []
        for i in range(5):
            candle_time = session_open + timedelta(minutes=i)
            if candle_time in m1_by_time:
                rate = m1_by_time[candle_time]
                opening_candles.append(M1Candle(
                    time=candle_time, open=rate["open"],
                    high=rate["high"], low=rate["low"], close=rate["close"]
                ))

        if len(opening_candles) < 5:
            continue

        oc_high = max(c.high for c in opening_candles)
        oc_low = min(c.low for c in opening_candles)
        oc_range_pips = (oc_high - oc_low) / pip_size

        if not (2.0 <= oc_range_pips <= 30.0):
            continue

        # Check for breakout in next 5 minutes
        breakout_start = session_open + timedelta(minutes=5)
        direction = None
        retest_level = None
        breakout_candle = None

        for i in range(5):
            candle_time = breakout_start + timedelta(minutes=i)
            if candle_time in m1_by_time:
                rate = m1_by_time[candle_time]
                if rate["close"] > oc_high:
                    direction = "LONG"
                    retest_level = oc_high
                    breakout_candle = M1Candle(
                        time=candle_time, open=rate["open"],
                        high=rate["high"], low=rate["low"], close=rate["close"]
                    )
                    break
                elif rate["close"] < oc_low:
                    direction = "SHORT"
                    retest_level = oc_low
                    breakout_candle = M1Candle(
                        time=candle_time, open=rate["open"],
                        high=rate["high"], low=rate["low"], close=rate["close"]
                    )
                    break

        if direction is None or breakout_candle is None:
            continue

        # Calculate breakout candle body percentage
        bc_body = abs(breakout_candle.close - breakout_candle.open)
        bc_range = breakout_candle.high - breakout_candle.low
        breakout_body_pct = (bc_body / bc_range * 100) if bc_range > 0 else 0

        # Look for retest + confirmation
        m1_candles = []
        retest_found = False
        trade = None

        m1_start = breakout_start + timedelta(minutes=5)
        session_end = datetime(day.year, day.month, day.day, end_hour, 0)

        current_time = m1_start
        candles_since_breakout = 0
        candles_since_retest = 0
        retest_time = None

        while current_time < session_end:
            if current_time not in m1_by_time:
                current_time += timedelta(minutes=1)
                continue

            rate = m1_by_time[current_time]
            m1_candle = M1Candle(
                time=current_time, open=rate["open"],
                high=rate["high"], low=rate["low"], close=rate["close"]
            )
            m1_candles.append(m1_candle)
            candles_since_breakout += 1

            if not retest_found:
                if candles_since_breakout > max_retest_candles:
                    break
                if check_retest(m1_candles, retest_level, direction.lower()):
                    retest_found = True
                    retest_time = current_time
                    candles_since_retest = 0

            elif trade is None:
                candles_since_retest += 1
                if candles_since_retest > max_confirm_candles:
                    break

                # Check confirmations in order of strength
                confirm_type = None
                if detect_choch(m1_candles, direction.lower(), lookback=8):
                    confirm_type = "CHoCH"
                elif detect_ifvg(m1_candles, direction.lower(), lookback=15):
                    confirm_type = "iFVG"
                elif detect_engulfing(m1_candles, direction.lower()):
                    confirm_type = "Engulfing"

                if confirm_type:
                    entry_price = m1_candle.close

                    # Current SL logic (below/above confirmation candle)
                    if direction == "LONG":
                        sl = m1_candle.low - (sl_buffer_pips * pip_size)
                    else:
                        sl = m1_candle.high + (sl_buffer_pips * pip_size)

                    sl_pips = abs(entry_price - sl) / pip_size

                    if direction == "LONG":
                        tp = entry_price + (sl_pips * rr_ratio * pip_size)
                    else:
                        tp = entry_price - (sl_pips * rr_ratio * pip_size)

                    tp_pips = abs(entry_price - tp) / pip_size
                    time_to_entry = int((current_time - session_open).total_seconds() / 60)

                    trade = TradeResult(
                        day=str(day),
                        session=session_name,
                        direction=direction,
                        confirmation_type=confirm_type,
                        entry_time=current_time,
                        entry_price=entry_price,
                        sl_price=sl,
                        tp_price=tp,
                        exit_price=0,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        pnl_pips=0,
                        result="",
                        rr_ratio=rr_ratio,
                        breakout_candle_body_pct=breakout_body_pct,
                        time_to_entry_minutes=time_to_entry,
                    )

            elif trade and trade.result == "":
                if direction == "LONG":
                    if m1_candle.low <= trade.sl_price:
                        trade.exit_price = trade.sl_price
                        trade.pnl_pips = -trade.sl_pips
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    if m1_candle.high >= trade.tp_price:
                        trade.exit_price = trade.tp_price
                        trade.pnl_pips = trade.tp_pips
                        trade.result = "WIN"
                        trades.append(trade)
                        break
                else:
                    if m1_candle.high >= trade.sl_price:
                        trade.exit_price = trade.sl_price
                        trade.pnl_pips = -trade.sl_pips
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    if m1_candle.low <= trade.tp_price:
                        trade.exit_price = trade.tp_price
                        trade.pnl_pips = trade.tp_pips
                        trade.result = "WIN"
                        trades.append(trade)
                        break

            current_time += timedelta(minutes=1)

        # Session end close
        if trade and trade.result == "":
            last_times = [t for t in m1_by_time.keys() if t.date() == day and t < session_end]
            if last_times:
                last_rate = m1_by_time[max(last_times)]
                trade.exit_price = last_rate["close"]
                if direction == "LONG":
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                else:
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
                trades.append(trade)

    return trades


def analyze_results(trades: List[TradeResult], title: str):
    """Analyze trade results and print insights."""

    if not trades:
        print(f"\n{title}: No trades")
        return

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]

    total_pips = sum(t.pnl_pips for t in trades)
    win_rate = len(wins) / len(trades) * 100

    win_pips = sum(t.pnl_pips for t in wins) if wins else 0
    loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0.001
    pf = win_pips / loss_pips

    avg_win = win_pips / len(wins) if wins else 0
    avg_loss = loss_pips / len(losses) if losses else 0

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Trades: {len(trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total Pips: {total_pips:+.1f}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg Win: {avg_win:.1f} pips | Avg Loss: {avg_loss:.1f} pips")

    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": win_rate,
        "total_pips": total_pips,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    print("="*70)
    print("  ANALIZA DETALIATA STRATEGIE LOCB")
    print("="*70)

    # Initialize MT5
    s = get_settings()
    if not mt5.initialize(
        login=s.mt5.login, password=s.mt5.password,
        server=s.mt5.server, path=s.mt5.path, timeout=120000,
    ):
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    # Get 3 months of data for better analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    symbols_to_test = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY"]
    pip_sizes = {"EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "EURJPY": 0.01}

    all_results = {}

    for symbol in symbols_to_test:
        print(f"\n{'#'*70}")
        print(f"  SYMBOL: {symbol}")
        print(f"{'#'*70}")

        mt5.symbol_select(symbol, True)

        m1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)

        if m1_rates is None or len(m1_rates) == 0:
            print(f"  No data for {symbol}")
            continue

        print(f"  Data: {len(m1_rates)} M1 candles")

        m1_by_time = {datetime.fromtimestamp(r["time"]): r for r in m1_rates}
        days = sorted(set(datetime.fromtimestamp(r["time"]).date() for r in m1_rates))

        pip_size = pip_sizes.get(symbol, 0.0001)

        # Test London session
        london_trades = run_detailed_backtest(
            m1_by_time, days, symbol, "london",
            open_hour=12, open_minute=0, end_hour=15,
            pip_size=pip_size,
        )

        # Test NY session
        ny_trades = run_detailed_backtest(
            m1_by_time, days, symbol, "ny",
            open_hour=18, open_minute=30, end_hour=21,
            pip_size=pip_size,
        )

        all_trades = london_trades + ny_trades

        if not all_trades:
            print(f"  No trades for {symbol}")
            continue

        # Overall results
        result = analyze_results(all_trades, f"{symbol} - TOTAL")
        all_results[symbol] = result

        # By session
        analyze_results(london_trades, f"{symbol} - London Session")
        analyze_results(ny_trades, f"{symbol} - NY Session")

        # By confirmation type
        print(f"\n  --- Analiza pe tip de confirmare ---")
        for conf_type in ["CHoCH", "iFVG", "Engulfing"]:
            conf_trades = [t for t in all_trades if t.confirmation_type == conf_type]
            if conf_trades:
                wins = len([t for t in conf_trades if t.result == "WIN"])
                wr = wins / len(conf_trades) * 100
                pips = sum(t.pnl_pips for t in conf_trades)
                avg_sl = sum(t.sl_pips for t in conf_trades) / len(conf_trades)
                print(f"  {conf_type:10}: {len(conf_trades):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f} | Avg SL: {avg_sl:.1f}p")

        # By direction
        print(f"\n  --- Analiza pe directie ---")
        for direction in ["LONG", "SHORT"]:
            dir_trades = [t for t in all_trades if t.direction == direction]
            if dir_trades:
                wins = len([t for t in dir_trades if t.result == "WIN"])
                wr = wins / len(dir_trades) * 100
                pips = sum(t.pnl_pips for t in dir_trades)
                print(f"  {direction:5}: {len(dir_trades):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f}")

        # Breakout candle body analysis
        print(f"\n  --- Analiza calitate breakout candle ---")
        strong_breakout = [t for t in all_trades if t.breakout_candle_body_pct >= 60]
        weak_breakout = [t for t in all_trades if t.breakout_candle_body_pct < 60]

        if strong_breakout:
            wins = len([t for t in strong_breakout if t.result == "WIN"])
            wr = wins / len(strong_breakout) * 100
            pips = sum(t.pnl_pips for t in strong_breakout)
            print(f"  Strong breakout (body>=60%): {len(strong_breakout):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f}")

        if weak_breakout:
            wins = len([t for t in weak_breakout if t.result == "WIN"])
            wr = wins / len(weak_breakout) * 100
            pips = sum(t.pnl_pips for t in weak_breakout)
            print(f"  Weak breakout (body<60%):   {len(weak_breakout):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f}")

        # Time to entry analysis
        print(f"\n  --- Analiza timing intrare ---")
        early_entry = [t for t in all_trades if t.time_to_entry_minutes <= 30]
        mid_entry = [t for t in all_trades if 30 < t.time_to_entry_minutes <= 60]
        late_entry = [t for t in all_trades if t.time_to_entry_minutes > 60]

        for name, group in [("Early (<=30min)", early_entry), ("Mid (30-60min)", mid_entry), ("Late (>60min)", late_entry)]:
            if group:
                wins = len([t for t in group if t.result == "WIN"])
                wr = wins / len(group) * 100
                pips = sum(t.pnl_pips for t in group)
                print(f"  {name:18}: {len(group):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f}")

        # SL size analysis
        print(f"\n  --- Analiza marime SL ---")
        tight_sl = [t for t in all_trades if t.sl_pips <= 5]
        medium_sl = [t for t in all_trades if 5 < t.sl_pips <= 10]
        wide_sl = [t for t in all_trades if t.sl_pips > 10]

        for name, group in [("Tight SL (<=5p)", tight_sl), ("Medium SL (5-10p)", medium_sl), ("Wide SL (>10p)", wide_sl)]:
            if group:
                wins = len([t for t in group if t.result == "WIN"])
                wr = wins / len(group) * 100
                pips = sum(t.pnl_pips for t in group)
                print(f"  {name:18}: {len(group):3} trades | WR: {wr:5.1f}% | Pips: {pips:+7.1f}")

    mt5.shutdown()

    # Summary comparison
    print("\n" + "="*70)
    print("  COMPARATIE SIMBOLURI")
    print("="*70)

    if all_results:
        sorted_symbols = sorted(all_results.items(), key=lambda x: x[1]["total_pips"], reverse=True)

        print(f"\n  {'Symbol':<10} {'Trades':<8} {'WR%':<8} {'Pips':<10} {'PF':<8}")
        print(f"  {'-'*44}")
        for symbol, data in sorted_symbols:
            print(f"  {symbol:<10} {data['trades']:<8} {data['win_rate']:<8.1f} {data['total_pips']:<+10.1f} {data['pf']:<8.2f}")

    print("\n" + "="*70)
    print("  RECOMANDARI")
    print("="*70)


if __name__ == "__main__":
    main()
