"""
Backtest LOCB Strategy - OPTIMIZED parameters
Testing different configurations to find the best setup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from config.settings import get_settings


@dataclass
class M1Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class LOCBTrade:
    symbol: str
    day: str
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    confirmation_type: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    result: str = ""
    sl_pips: float = 0.0


def detect_choch(candles: List[M1Candle], direction: str, lookback: int = 8) -> bool:
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


def find_fvgs(candles: List[M1Candle]):
    fvgs = []
    if len(candles) < 3:
        return fvgs
    for i in range(2, len(candles)):
        c0 = candles[i - 2]
        c2 = candles[i]
        if c2.low > c0.high:
            fvgs.append({"type": "bullish", "high": c2.low, "low": c0.high})
        if c2.high < c0.low:
            fvgs.append({"type": "bearish", "high": c0.low, "low": c2.high})
    return fvgs


def detect_ifvg(candles: List[M1Candle], direction: str, lookback: int = 15) -> bool:
    if len(candles) < lookback:
        return False
    recent = candles[-lookback:]
    fvgs = find_fvgs(recent[:-1])
    if not fvgs:
        return False
    current = candles[-1]
    for fvg in fvgs:
        if direction == "bullish" and fvg["type"] == "bearish":
            if current.close > fvg["high"]:
                return True
        elif direction == "bearish" and fvg["type"] == "bullish":
            if current.close < fvg["low"]:
                return True
    return False


def detect_engulfing(candles: List[M1Candle], direction: str) -> bool:
    if len(candles) < 2:
        return False
    prev = candles[-2]
    curr = candles[-1]
    prev_body_high = max(prev.open, prev.close)
    prev_body_low = min(prev.open, prev.close)
    curr_body_high = max(curr.open, curr.close)
    curr_body_low = min(curr.open, curr.close)
    if direction == "bullish":
        prev_bearish = prev.close < prev.open
        curr_bullish = curr.close > curr.open
        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
        return prev_bearish and curr_bullish and engulfs
    else:
        prev_bullish = prev.close > prev.open
        curr_bearish = curr.close < curr.open
        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
        return prev_bullish and curr_bearish and engulfs


def check_retest(candles: List[M1Candle], level: float, direction: str, pip_size: float, tolerance_pips: float = 3.0) -> bool:
    if not candles:
        return False
    tolerance = tolerance_pips * pip_size
    current = candles[-1]
    if direction == "bullish":
        return current.low <= level + tolerance and current.close > level
    else:
        return current.high >= level - tolerance and current.close < level


def check_confirmation(candles: List[M1Candle], direction: str) -> Tuple[bool, str]:
    if detect_choch(candles, direction, lookback=8):
        return True, "CHoCH"
    if detect_ifvg(candles, direction, lookback=15):
        return True, "iFVG"
    if detect_engulfing(candles, direction):
        return True, "Engulfing"
    return False, ""


def run_backtest_for_symbol(
    symbol: str,
    m5_rates,
    m1_rates,
    pip_size: float,
    london_open_hour: int = 12,
    session_end_hour: int = 15,
    risk_reward: float = 1.5,
    sl_buffer_pips: float = 2.0,
    max_retest_candles: int = 30,
    max_confirm_candles: int = 15,  # Original was 15
    min_range_pips: float = 2.0,
    max_range_pips: float = 30.0,
) -> List[LOCBTrade]:
    """Run LOCB backtest for a single symbol."""

    trades = []

    m5_by_time = {}
    for rate in m5_rates:
        ts = datetime.fromtimestamp(rate["time"])
        m5_by_time[ts] = rate

    m1_by_time = {}
    for rate in m1_rates:
        ts = datetime.fromtimestamp(rate["time"])
        m1_by_time[ts] = rate

    days = set()
    for rate in m5_rates:
        ts = datetime.fromtimestamp(rate["time"])
        if ts.weekday() < 5:
            days.add(ts.date())

    for day in sorted(days):
        london_open = datetime(day.year, day.month, day.day, london_open_hour, 0)

        if london_open not in m5_by_time:
            continue

        opening_candle = m5_by_time[london_open]
        oc_high = opening_candle["high"]
        oc_low = opening_candle["low"]
        oc_range = oc_high - oc_low
        oc_range_pips = oc_range / pip_size

        # Filter by range size
        if oc_range_pips < min_range_pips or oc_range_pips > max_range_pips:
            continue

        second_candle_time = london_open + timedelta(minutes=5)
        if second_candle_time not in m5_by_time:
            continue

        second_candle = m5_by_time[second_candle_time]

        direction = None
        retest_level = None

        if second_candle["close"] > oc_high:
            direction = "LONG"
            retest_level = oc_high
        elif second_candle["close"] < oc_low:
            direction = "SHORT"
            retest_level = oc_low
        else:
            continue

        m1_candles = []
        retest_found = False
        entry_found = False
        trade = None

        m1_start = second_candle_time + timedelta(minutes=5)
        session_end = datetime(day.year, day.month, day.day, session_end_hour, 0)

        current_time = m1_start
        candles_since_breakout = 0
        candles_since_retest = 0

        while current_time < session_end:
            if current_time not in m1_by_time:
                current_time += timedelta(minutes=1)
                continue

            m1_rate = m1_by_time[current_time]
            m1_candle = M1Candle(
                time=current_time,
                open=m1_rate["open"],
                high=m1_rate["high"],
                low=m1_rate["low"],
                close=m1_rate["close"]
            )
            m1_candles.append(m1_candle)
            candles_since_breakout += 1

            if not retest_found:
                if candles_since_breakout > max_retest_candles:
                    break
                if check_retest(m1_candles, retest_level, direction.lower(), pip_size):
                    retest_found = True
                    candles_since_retest = 0

            elif not entry_found:
                candles_since_retest += 1
                if candles_since_retest > max_confirm_candles:
                    break

                confirmed, confirm_type = check_confirmation(m1_candles, direction.lower())

                if confirmed:
                    entry_price = m1_candle.close

                    if direction == "LONG":
                        sl = oc_low - (sl_buffer_pips * pip_size)
                        risk = entry_price - sl
                        tp = entry_price + (risk * risk_reward)
                    else:
                        sl = oc_high + (sl_buffer_pips * pip_size)
                        risk = sl - entry_price
                        tp = entry_price - (risk * risk_reward)

                    sl_pips = abs(entry_price - sl) / pip_size

                    trade = LOCBTrade(
                        symbol=symbol,
                        day=str(day),
                        direction=direction,
                        entry_time=current_time,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        confirmation_type=confirm_type,
                        sl_pips=sl_pips
                    )
                    entry_found = True

            elif trade and trade.result == "":
                if direction == "LONG":
                    if m1_candle.low <= trade.stop_loss:
                        trade.exit_time = current_time
                        trade.exit_price = trade.stop_loss
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    if m1_candle.high >= trade.take_profit:
                        trade.exit_time = current_time
                        trade.exit_price = trade.take_profit
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trade.result = "WIN"
                        trades.append(trade)
                        break
                else:
                    if m1_candle.high >= trade.stop_loss:
                        trade.exit_time = current_time
                        trade.exit_price = trade.stop_loss
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    if m1_candle.low <= trade.take_profit:
                        trade.exit_time = current_time
                        trade.exit_price = trade.take_profit
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trade.result = "WIN"
                        trades.append(trade)
                        break

            current_time += timedelta(minutes=1)

        if trade and trade.result == "":
            last_times = [t for t in m1_by_time.keys() if t.date() == day and t < session_end]
            if last_times:
                last_m1_time = max(last_times)
                last_m1 = m1_by_time[last_m1_time]
                trade.exit_time = last_m1_time
                trade.exit_price = last_m1["close"]
                if direction == "LONG":
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                else:
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
                trades.append(trade)

    return trades


def analyze_results(trades: List[LOCBTrade], symbol: str) -> Dict:
    if not trades:
        return {"symbol": symbol, "trades": 0, "profitable": False, "total_pips": 0, "win_rate": 0, "profit_factor": 0}

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    total_pips = sum(t.pnl_pips for t in trades)
    win_pips = sum(t.pnl_pips for t in wins) if wins else 0
    loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0.001
    pf = win_pips / loss_pips if loss_pips > 0 else 999
    wr = len(wins) / len(trades) * 100 if trades else 0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "total_pips": total_pips,
        "profit_factor": pf,
        "profitable": total_pips > 0 and pf > 1.0
    }


if __name__ == "__main__":
    s = get_settings()

    if not mt5.initialize(
        login=s.mt5.login,
        password=s.mt5.password,
        server=s.mt5.server,
        path=s.mt5.path,
        timeout=60000,
    ):
        print(f"MT5 init failed: {mt5.last_error()}")
        exit()

    symbols = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "EURJPY": 0.01,
    }

    end_date = datetime.now()

    m1_data = {}
    m5_data = {}

    for days_back in [60, 45, 30]:
        start_date = end_date - timedelta(days=days_back)
        print(f"Getting {days_back} days of data...")

        success = True
        for symbol in symbols.keys():
            mt5.symbol_select(symbol, True)
            m1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)
            m5_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)

            if m1_rates is None or len(m1_rates) == 0 or m5_rates is None or len(m5_rates) == 0:
                success = False
                break

            m1_data[symbol] = m1_rates
            m5_data[symbol] = m5_rates

        if success:
            print(f"Got data for all symbols")
            break
        m1_data = {}
        m5_data = {}

    if not m1_data:
        print("Could not get data!")
        mt5.shutdown()
        exit()

    # Test different parameter combinations
    print()
    print("=" * 90)
    print("  OPTIMIZATION: TESTING DIFFERENT PARAMETER COMBINATIONS")
    print("=" * 90)

    best_config = None
    best_pips = -9999

    configs = [
        {"rr": 1.5, "confirm": 15, "retest": 30, "end_hour": 15},
        {"rr": 1.5, "confirm": 15, "retest": 30, "end_hour": 14},
        {"rr": 2.0, "confirm": 15, "retest": 30, "end_hour": 15},
        {"rr": 1.5, "confirm": 10, "retest": 20, "end_hour": 15},
        {"rr": 1.5, "confirm": 20, "retest": 45, "end_hour": 15},
    ]

    for config in configs:
        all_trades = []
        for symbol, pip_size in symbols.items():
            trades = run_backtest_for_symbol(
                symbol=symbol,
                m5_rates=m5_data[symbol],
                m1_rates=m1_data[symbol],
                pip_size=pip_size,
                london_open_hour=12,
                session_end_hour=config["end_hour"],
                risk_reward=config["rr"],
                max_retest_candles=config["retest"],
                max_confirm_candles=config["confirm"],
            )
            all_trades.extend(trades)

        total_pips = sum(t.pnl_pips for t in all_trades)
        total_trades = len(all_trades)
        wins = len([t for t in all_trades if t.result == "WIN"])
        wr = (wins / total_trades * 100) if total_trades > 0 else 0

        print(f"  R:R {config['rr']}, Confirm {config['confirm']}m, Retest {config['retest']}m, End {config['end_hour']}:00")
        print(f"    -> {total_trades} trades, {wr:.1f}% WR, {total_pips:+.1f} pips")

        if total_pips > best_pips:
            best_pips = total_pips
            best_config = config

    print()
    print("=" * 90)
    print(f"  BEST CONFIG: R:R {best_config['rr']}, Confirm {best_config['confirm']}m, "
          f"Retest {best_config['retest']}m, End {best_config['end_hour']}:00")
    print(f"  BEST RESULT: {best_pips:+.1f} pips")
    print("=" * 90)

    # Detailed analysis with best config
    print()
    print("=" * 90)
    print("  DETAILED ANALYSIS WITH BEST CONFIG")
    print("=" * 90)

    all_trades = []
    all_results = []

    for symbol, pip_size in symbols.items():
        trades = run_backtest_for_symbol(
            symbol=symbol,
            m5_rates=m5_data[symbol],
            m1_rates=m1_data[symbol],
            pip_size=pip_size,
            london_open_hour=12,
            session_end_hour=best_config["end_hour"],
            risk_reward=best_config["rr"],
            max_retest_candles=best_config["retest"],
            max_confirm_candles=best_config["confirm"],
        )
        all_trades.extend(trades)
        result = analyze_results(trades, symbol)
        all_results.append(result)

        status = "[OK]    " if result["profitable"] else "[SLAB]  "
        print(f"  {status} {symbol}: {result['trades']:2} trades | WR: {result['win_rate']:5.1f}% | "
              f"Pips: {result['total_pips']:+7.1f} | PF: {result['profit_factor']:.2f}")

    # Only profitable pairs
    profitable_pairs = [r["symbol"] for r in all_results if r["profitable"]]
    unprofitable_pairs = [r["symbol"] for r in all_results if not r["profitable"]]

    print()
    print("=" * 90)
    print("  MONTHLY ESTIMATION")
    print("=" * 90)

    total_pips = sum(t.pnl_pips for t in all_trades)
    trading_days = len(set(t.day for t in all_trades))
    avg_per_day = total_pips / trading_days if trading_days > 0 else 0
    monthly_pips = avg_per_day * 22

    print(f"  ALL PAIRS:")
    print(f"    Total Pips: {total_pips:+.1f}")
    print(f"    Trading Days: {trading_days}")
    print(f"    Avg Pips/Day: {avg_per_day:+.2f}")
    print(f"    Monthly Est: {monthly_pips:+.1f} pips")

    # Estimate % return with 0.5% risk
    avg_sl = 8  # approximate
    monthly_pct = (monthly_pips / avg_sl) * 0.5 if avg_sl > 0 else 0
    print(f"    Monthly Return (0.5% risk): {monthly_pct:+.2f}%")

    if profitable_pairs:
        profitable_trades = [t for t in all_trades if t.symbol in profitable_pairs]
        profitable_pips = sum(t.pnl_pips for t in profitable_trades)
        profitable_days = len(set(t.day for t in profitable_trades))
        profitable_avg = profitable_pips / profitable_days if profitable_days > 0 else 0
        profitable_monthly = profitable_avg * 22
        profitable_monthly_pct = (profitable_monthly / avg_sl) * 0.5

        print()
        print(f"  ONLY PROFITABLE PAIRS ({', '.join(profitable_pairs)}):")
        print(f"    Total Pips: {profitable_pips:+.1f}")
        print(f"    Monthly Est: {profitable_monthly:+.1f} pips")
        print(f"    Monthly Return (0.5% risk): {profitable_monthly_pct:+.2f}%")

    print()
    print("=" * 90)
    print("  RECOMANDARI")
    print("=" * 90)
    if profitable_pairs:
        print(f"  PASTREAZA: {', '.join(profitable_pairs)}")
    if unprofitable_pairs:
        print(f"  ELIMINA: {', '.join(unprofitable_pairs)}")

    mt5.shutdown()
