"""
Backtest LOCB (London Opening Candle Breakout) Strategy on EUR/USD

Strategy Rules:
1. First M5 candle at London open (08:00 UTC) - mark HIGH/LOW
2. Second M5 candle closes above HIGH (bullish) or below LOW (bearish) -> direction confirmed
3. Switch to M1, wait for retest of HIGH (for buy) or LOW (for sell)
4. Wait for confirmation: CHoCH / Inversed FVG / Engulfing
5. Entry with SL below/above M5 opening candle extreme
6. TP: 2:1 R:R

Confirmations:
- CHoCH (Change of Character): Break of recent swing high/low in the direction of trade
- iFVG (Inversed Fair Value Gap): FVG that gets violated (price closes through it)
- Engulfing: Bullish/Bearish engulfing pattern
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Tuple
from config.settings import get_settings


@dataclass
class M1Candle:
    """M1 candle data."""
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class FVG:
    """Fair Value Gap."""
    type: str  # "bullish" or "bearish"
    high: float
    low: float
    candle_time: datetime
    invalidated: bool = False


@dataclass
class LOCBTrade:
    """LOCB trade record."""
    day: str
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    opening_candle_high: float
    opening_candle_low: float
    confirmation_type: str  # "CHoCH", "iFVG", "Engulfing"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    result: str = ""
    sl_pips: float = 0.0


def detect_choch(candles: List[M1Candle], direction: str, lookback: int = 10) -> bool:
    """
    Detect Change of Character (CHoCH).

    For bullish CHoCH: price breaks above recent swing high
    For bearish CHoCH: price breaks below recent swing low
    """
    if len(candles) < lookback + 2:
        return False

    recent = candles[-(lookback + 1):-1]  # Exclude current candle
    current = candles[-1]

    if direction == "bullish":
        # Find recent swing high (highest high in lookback period)
        swing_high = max(c.high for c in recent)
        # CHoCH if current candle closes above swing high
        return current.close > swing_high
    else:  # bearish
        # Find recent swing low
        swing_low = min(c.low for c in recent)
        # CHoCH if current candle closes below swing low
        return current.close < swing_low


def find_fvgs(candles: List[M1Candle]) -> List[FVG]:
    """
    Find Fair Value Gaps in candle data.

    Bullish FVG: Gap between candle[i-2].high and candle[i].low (gap up)
    Bearish FVG: Gap between candle[i-2].low and candle[i].high (gap down)
    """
    fvgs = []

    if len(candles) < 3:
        return fvgs

    for i in range(2, len(candles)):
        c0 = candles[i - 2]  # First candle
        c2 = candles[i]      # Third candle

        # Bullish FVG: gap between c0.high and c2.low
        if c2.low > c0.high:
            fvgs.append(FVG(
                type="bullish",
                high=c2.low,
                low=c0.high,
                candle_time=c2.time
            ))

        # Bearish FVG: gap between c0.low and c2.high
        if c2.high < c0.low:
            fvgs.append(FVG(
                type="bearish",
                high=c0.low,
                low=c2.high,
                candle_time=c2.time
            ))

    return fvgs


def detect_ifvg(candles: List[M1Candle], direction: str, lookback: int = 20) -> bool:
    """
    Detect Inversed Fair Value Gap (iFVG).

    iFVG = FVG that gets violated (price closes through it)
    For bullish setup: we need a bearish FVG that gets violated (bullish sign)
    For bearish setup: we need a bullish FVG that gets violated (bearish sign)
    """
    if len(candles) < lookback:
        return False

    recent = candles[-lookback:]
    fvgs = find_fvgs(recent[:-1])  # Find FVGs excluding current candle

    if not fvgs:
        return False

    current = candles[-1]

    for fvg in fvgs:
        if direction == "bullish" and fvg.type == "bearish":
            # Bullish iFVG: price closes above bearish FVG high
            if current.close > fvg.high:
                return True
        elif direction == "bearish" and fvg.type == "bullish":
            # Bearish iFVG: price closes below bullish FVG low
            if current.close < fvg.low:
                return True

    return False


def detect_engulfing(candles: List[M1Candle], direction: str) -> bool:
    """
    Detect Engulfing pattern.

    Bullish Engulfing: Current candle body completely engulfs previous candle body (bullish)
    Bearish Engulfing: Current candle body completely engulfs previous candle body (bearish)
    """
    if len(candles) < 2:
        return False

    prev = candles[-2]
    curr = candles[-1]

    prev_body_high = max(prev.open, prev.close)
    prev_body_low = min(prev.open, prev.close)
    curr_body_high = max(curr.open, curr.close)
    curr_body_low = min(curr.open, curr.close)

    if direction == "bullish":
        # Bullish engulfing: prev was bearish, curr is bullish and engulfs
        prev_bearish = prev.close < prev.open
        curr_bullish = curr.close > curr.open
        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
        return prev_bearish and curr_bullish and engulfs
    else:  # bearish
        # Bearish engulfing: prev was bullish, curr is bearish and engulfs
        prev_bullish = prev.close > prev.open
        curr_bearish = curr.close < curr.open
        engulfs = curr_body_low <= prev_body_low and curr_body_high >= prev_body_high
        return prev_bullish and curr_bearish and engulfs


def check_retest(candles: List[M1Candle], level: float, direction: str, tolerance_pips: float = 3.0) -> bool:
    """
    Check if price has retested a level.

    For bullish: price should come down and touch/approach the level from above
    For bearish: price should come up and touch/approach the level from below
    """
    if not candles:
        return False

    pip_size = 0.0001
    tolerance = tolerance_pips * pip_size
    current = candles[-1]

    if direction == "bullish":
        # Retest of HIGH for buy - price should touch or get close to the level
        return current.low <= level + tolerance and current.close > level
    else:  # bearish
        # Retest of LOW for sell - price should touch or get close to the level
        return current.high >= level - tolerance and current.close < level


def check_confirmation(candles: List[M1Candle], direction: str) -> Tuple[bool, str]:
    """
    Check for any confirmation pattern.
    Returns (confirmed, confirmation_type)
    """
    # Check CHoCH first (strongest signal)
    if detect_choch(candles, direction, lookback=8):
        return True, "CHoCH"

    # Check iFVG
    if detect_ifvg(candles, direction, lookback=15):
        return True, "iFVG"

    # Check Engulfing
    if detect_engulfing(candles, direction):
        return True, "Engulfing"

    return False, ""


def run_locb_backtest(
    m5_rates,
    m1_rates,
    london_open_hour: int = 10,  # MT5 server time (UTC+2), London 08:00 UTC = 10:00 server
    session_end_hour: int = 14,  # Before NY (12:00 UTC = 14:00 server)
    risk_reward: float = 2.0,
    sl_buffer_pips: float = 2.0,
    max_retest_candles: int = 30,  # Max M1 candles to wait for retest
    max_confirm_candles: int = 15,  # Max M1 candles after retest to wait for confirmation
) -> List[LOCBTrade]:
    """Run LOCB backtest."""

    trades = []
    pip_size = 0.0001

    # Convert rates to datetime-indexed dict
    m5_by_time = {}
    for rate in m5_rates:
        ts = datetime.fromtimestamp(rate["time"])
        m5_by_time[ts] = rate

    m1_by_time = {}
    for rate in m1_rates:
        ts = datetime.fromtimestamp(rate["time"])
        m1_by_time[ts] = rate

    # Get unique trading days
    days = set()
    for rate in m5_rates:
        ts = datetime.fromtimestamp(rate["time"])
        if ts.weekday() < 5:  # Skip weekends
            days.add(ts.date())

    debug_days = 999  # Disable debug output
    for day in sorted(days):
        # Find first M5 candle at London open
        london_open = datetime(day.year, day.month, day.day, london_open_hour, 0)

        if london_open not in m5_by_time:
            if debug_days < 5:
                # Debug: show what times we have for this day
                day_times = [t for t in m5_by_time.keys() if t.date() == day]
                if day_times:
                    print(f"DEBUG {day}: London open {london_open_hour}:00 not found. First candle: {min(day_times).hour}:{min(day_times).minute}")
                debug_days += 1
            continue

        opening_candle = m5_by_time[london_open]
        oc_high = opening_candle["high"]
        oc_low = opening_candle["low"]
        oc_range = oc_high - oc_low

        # Skip if range too small (reduced from 5 to 2 pips for single M5 candle)
        if oc_range < 2 * pip_size:
            if debug_days < 3:
                print(f"DEBUG {day}: Range too small: {oc_range/pip_size:.1f} pips")
            continue

        # Find second M5 candle
        second_candle_time = london_open + timedelta(minutes=5)
        if second_candle_time not in m5_by_time:
            if debug_days < 3:
                print(f"DEBUG {day}: Second candle not found at {second_candle_time}")
            continue

        second_candle = m5_by_time[second_candle_time]

        # Determine direction based on second candle close
        direction = None
        retest_level = None

        if second_candle["close"] > oc_high:
            direction = "LONG"
            retest_level = oc_high  # Wait for retest of HIGH
            if debug_days < 5:
                print(f"DEBUG {day}: LONG signal - close {second_candle['close']:.5f} > high {oc_high:.5f}")
                debug_days += 1
        elif second_candle["close"] < oc_low:
            direction = "SHORT"
            retest_level = oc_low  # Wait for retest of LOW
            if debug_days < 5:
                print(f"DEBUG {day}: SHORT signal - close {second_candle['close']:.5f} < low {oc_low:.5f}")
                debug_days += 1
        else:
            # No breakout, skip this day
            if debug_days < 3:
                print(f"DEBUG {day}: No breakout - close {second_candle['close']:.5f} between {oc_low:.5f} and {oc_high:.5f}")
            continue

        # Switch to M1 and look for retest + confirmation
        m1_candles = []
        retest_found = False
        entry_found = False
        trade = None

        # Start looking from after the second M5 candle
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

            # Phase 1: Look for retest
            if not retest_found:
                if candles_since_breakout > max_retest_candles:
                    # Timeout waiting for retest
                    if debug_days < 8:
                        print(f"DEBUG {day}: Retest timeout after {candles_since_breakout} candles")
                    break

                if check_retest(m1_candles, retest_level, direction.lower()):
                    retest_found = True
                    candles_since_retest = 0
                    if debug_days < 8:
                        print(f"DEBUG {day}: Retest found at {current_time}")

            # Phase 2: Look for confirmation after retest
            elif not entry_found:
                candles_since_retest += 1

                if candles_since_retest > max_confirm_candles:
                    # Timeout waiting for confirmation
                    if debug_days < 8:
                        print(f"DEBUG {day}: Confirmation timeout after {candles_since_retest} candles")
                    break

                confirmed, confirm_type = check_confirmation(m1_candles, direction.lower())

                if confirmed:
                    if debug_days < 8:
                        print(f"DEBUG {day}: Confirmation {confirm_type} at {current_time}")
                    # Entry!
                    entry_price = m1_candle.close

                    if direction == "LONG":
                        sl = oc_low - (sl_buffer_pips * pip_size)
                        risk = entry_price - sl
                        tp = entry_price + (risk * risk_reward)
                    else:  # SHORT
                        sl = oc_high + (sl_buffer_pips * pip_size)
                        risk = sl - entry_price
                        tp = entry_price - (risk * risk_reward)

                    sl_pips = abs(entry_price - sl) / pip_size

                    trade = LOCBTrade(
                        day=str(day),
                        direction=direction,
                        entry_time=current_time,
                        entry_price=entry_price,
                        stop_loss=sl,
                        take_profit=tp,
                        opening_candle_high=oc_high,
                        opening_candle_low=oc_low,
                        confirmation_type=confirm_type,
                        sl_pips=sl_pips
                    )
                    entry_found = True

            # Phase 3: Manage trade
            elif trade and trade.result == "":
                if direction == "LONG":
                    # Check SL
                    if m1_candle.low <= trade.stop_loss:
                        trade.exit_time = current_time
                        trade.exit_price = trade.stop_loss
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    # Check TP
                    if m1_candle.high >= trade.take_profit:
                        trade.exit_time = current_time
                        trade.exit_price = trade.take_profit
                        trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                        trade.result = "WIN"
                        trades.append(trade)
                        break
                else:  # SHORT
                    # Check SL
                    if m1_candle.high >= trade.stop_loss:
                        trade.exit_time = current_time
                        trade.exit_price = trade.stop_loss
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trade.result = "LOSS"
                        trades.append(trade)
                        break
                    # Check TP
                    if m1_candle.low <= trade.take_profit:
                        trade.exit_time = current_time
                        trade.exit_price = trade.take_profit
                        trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                        trade.result = "WIN"
                        trades.append(trade)
                        break

            current_time += timedelta(minutes=1)

        # Close at session end if still open
        if trade and trade.result == "":
            last_m1_time = max(t for t in m1_by_time.keys() if t.date() == day and t < session_end)
            if last_m1_time:
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


def print_results(trades: List[LOCBTrade], name: str, show_details: bool = False):
    """Print trade results."""
    if not trades:
        print(f"{name}: 0 tranzactii")
        return {}

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    total_pips = sum(t.pnl_pips for t in trades)
    win_pips = sum(t.pnl_pips for t in wins) if wins else 0
    loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0.001
    pf = win_pips / loss_pips if loss_pips > 0 else 999
    wr = len(wins) / len(trades) * 100 if trades else 0

    avg_sl = sum(t.sl_pips for t in trades) / len(trades) if trades else 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t.pnl_pips
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Confirmation breakdown
    confirm_types = {}
    for t in trades:
        ct = t.confirmation_type
        if ct not in confirm_types:
            confirm_types[ct] = {"total": 0, "wins": 0}
        confirm_types[ct]["total"] += 1
        if t.result == "WIN":
            confirm_types[ct]["wins"] += 1

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Total Trades: {len(trades)}")
    print(f"  Wins/Losses: {len(wins)}/{len(losses)}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total PnL: {total_pips:.1f} pips")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Max Drawdown: {max_dd:.1f} pips")
    print(f"  Avg SL Size: {avg_sl:.1f} pips")

    if show_details:
        print(f"\n  Confirmation Breakdown:")
        for ct, stats in confirm_types.items():
            ct_wr = (stats["wins"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"    {ct}: {stats['total']} trades, {ct_wr:.0f}% win rate")

        if wins:
            print(f"\n  Avg Win: {win_pips/len(wins):.1f} pips")
        if losses:
            print(f"  Avg Loss: {loss_pips/len(losses):.1f} pips")

    print()

    return {
        "name": name,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "total_pips": total_pips,
        "profit_factor": pf,
        "max_dd": max_dd,
        "avg_sl": avg_sl,
    }


def show_trade_details(trades: List[LOCBTrade]):
    """Show individual trade details."""
    print("\n" + "="*100)
    print("  DETALII TRANZACTII")
    print("="*100)

    for t in trades:
        oc_range = (t.opening_candle_high - t.opening_candle_low) / 0.0001
        print(
            f"  {t.day} | {t.direction:5} | Entry: {t.entry_price:.5f} | "
            f"SL: {t.stop_loss:.5f} ({t.sl_pips:.0f}p) | "
            f"Confirm: {t.confirmation_type:8} | "
            f"Result: {t.result:4} | PnL: {t.pnl_pips:+.1f}p"
        )


if __name__ == "__main__":
    # Initialize MT5
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

    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)

    # Get 3 months of data (M1 data limit)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    # Try getting data - M1 may need shorter period
    print(f"Downloading M5 data from {start_date.date()} to {end_date.date()}...")
    m5_rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        start_date,
        end_date,
    )

    if m5_rates is None:
        print(f"Nu am putut obtine date M5! Error: {mt5.last_error()}")
        mt5.shutdown()
        exit()

    print(f"Got {len(m5_rates)} M5 candles")

    # M1 data - try shorter periods if needed
    m1_rates = None
    for days_back in [90, 60, 45, 30]:
        start_date_m1 = end_date - timedelta(days=days_back)
        print(f"Trying M1 data for {days_back} days...")
        m1_rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M1,
            start_date_m1,
            end_date,
        )
        if m1_rates is not None and len(m1_rates) > 0:
            print(f"Got {len(m1_rates)} M1 candles for {days_back} days")
            # Also get matching M5 period
            m5_rates = mt5.copy_rates_range(
                symbol,
                mt5.TIMEFRAME_M5,
                start_date_m1,
                end_date,
            )
            break
        print(f"M1 failed for {days_back} days: {mt5.last_error()}")

    mt5.shutdown()

    if m1_rates is None or len(m1_rates) == 0:
        print("Nu am putut obtine date M1!")
        exit()

    print(f"M5 candles: {len(m5_rates)}")
    print(f"M1 candles: {len(m1_rates)}")

    print("\n" + "="*70)
    print("  BACKTEST EUR/USD LOCB (London Opening Candle Breakout)")
    print("  Perioada: 3 luni")
    print("="*70)

    # Run backtest with default parameters
    # MT5 server is UTC+2, so London 08:00 UTC = 10:00 server time
    print("\n>>> Test cu parametri default (R:R 2:1, London 08:00-12:00 UTC = 10:00-14:00 server)")
    trades = run_locb_backtest(
        m5_rates, m1_rates,
        london_open_hour=10,  # 08:00 UTC = 10:00 server
        session_end_hour=14,   # 12:00 UTC = 14:00 server
        risk_reward=2.0,
        sl_buffer_pips=2.0,
    )
    print_results(trades, "LOCB Default", show_details=True)
    show_trade_details(trades)

    # Test different R:R ratios
    print("\n" + "="*70)
    print("  TEST DIFERITE R:R RATIOS")
    print("="*70)

    best_result = None
    best_rr = 2.0

    for rr in [1.5, 2.0, 2.5, 3.0]:
        trades = run_locb_backtest(
            m5_rates, m1_rates,
            london_open_hour=10,
            session_end_hour=14,
            risk_reward=rr,
            sl_buffer_pips=2.0,
        )
        result = print_results(trades, f"R:R {rr}:1")
        if result and (best_result is None or result.get("total_pips", -999) > best_result.get("total_pips", -999)):
            best_result = result
            best_rr = rr

    # Test different session end times
    print("\n" + "="*70)
    print("  TEST DIFERITE ORE DE INCHIDERE SESIUNE (server time, -2h for UTC)")
    print("="*70)

    for end_hour in [13, 14, 15, 16]:  # 11-14 UTC = 13-16 server
        trades = run_locb_backtest(
            m5_rates, m1_rates,
            london_open_hour=10,
            session_end_hour=end_hour,
            risk_reward=best_rr,
            sl_buffer_pips=2.0,
        )
        print_results(trades, f"End Hour {end_hour}:00 server ({end_hour-2}:00 UTC)")

    # Test different confirmation wait times
    print("\n" + "="*70)
    print("  TEST DIFERITI PARAMETRI DE CONFIRMARE")
    print("="*70)

    for max_confirm in [10, 15, 20, 30]:
        for max_retest in [20, 30, 45]:
            trades = run_locb_backtest(
                m5_rates, m1_rates,
                london_open_hour=10,
                session_end_hour=15,
                risk_reward=3.0,
                sl_buffer_pips=2.0,
                max_retest_candles=max_retest,
                max_confirm_candles=max_confirm,
            )
            if trades:
                result = print_results(trades, f"Retest {max_retest}m, Confirm {max_confirm}m")

    # Final summary
    print("\n" + "="*70)
    print("  SUMAR FINAL")
    print("="*70)
    print(f"  Cel mai bun R:R: {best_rr}:1")
    if best_result:
        print(f"  Total Pips: {best_result['total_pips']:.1f}")
        print(f"  Win Rate: {best_result['win_rate']:.1f}%")
        print(f"  Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"  Avg SL: {best_result['avg_sl']:.1f} pips")
