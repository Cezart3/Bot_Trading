"""
Backtest LOCB Strategy - Parametrii EXACTI din main.py (demo/real mode)

Parametri identici cu ce rulează botul:
- London session: 12:00-15:00 server time (08:00-11:00 UTC)
- NY session: 18:30-21:00 server time (14:30-17:00 UTC)
- R:R dinamic: 1.5-4.0 (fallback 2.5)
- SL buffer: 1 pip
- Max 2 trades/day (1 per session)
- News filter: ON
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from typing import Optional, List, Tuple
from config.settings import get_settings
from utils.news_filter import NewsFilter, NewsFilterConfig, NewsImpact, EconomicEvent


@dataclass
class M1Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass
class LiquidityLevel:
    price: float
    type: str  # swing_high, swing_low
    strength: float


@dataclass
class LOCBTrade:
    day: str
    session: str  # "london" or "ny"
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    opening_candle_high: float
    opening_candle_low: float
    confirmation_type: str
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    result: str = ""
    sl_pips: float = 0.0
    lot_size: float = 0.0


def find_liquidity_levels(candles: List[M1Candle], lookback: int = 50) -> List[LiquidityLevel]:
    """Find swing highs/lows as liquidity targets."""
    levels = []
    if len(candles) < lookback:
        return levels

    recent = candles[-lookback:]

    # Find swing highs (local maxima)
    for i in range(2, len(recent) - 2):
        if (recent[i].high > recent[i-1].high and
            recent[i].high > recent[i-2].high and
            recent[i].high > recent[i+1].high and
            recent[i].high > recent[i+2].high):
            levels.append(LiquidityLevel(
                price=recent[i].high,
                type="swing_high",
                strength=0.8
            ))

    # Find swing lows (local minima)
    for i in range(2, len(recent) - 2):
        if (recent[i].low < recent[i-1].low and
            recent[i].low < recent[i-2].low and
            recent[i].low < recent[i+1].low and
            recent[i].low < recent[i+2].low):
            levels.append(LiquidityLevel(
                price=recent[i].low,
                type="swing_low",
                strength=0.8
            ))

    return levels


def calculate_dynamic_tp(
    entry_price: float,
    sl_price: float,
    direction: str,
    candles: List[M1Candle],
    min_rr: float = 1.5,
    max_rr: float = 4.0,
    fallback_rr: float = 2.5,
    pip_size: float = 0.0001,
) -> Tuple[float, float, str]:
    """
    Calculate dynamic TP based on liquidity levels.
    Returns (tp_price, actual_rr, tp_type)
    """
    risk = abs(entry_price - sl_price)

    # Find liquidity levels
    levels = find_liquidity_levels(candles)

    if direction == "LONG":
        # Look for swing highs above entry
        targets = [l for l in levels if l.type == "swing_high" and l.price > entry_price]
        targets.sort(key=lambda x: x.price)

        for target in targets:
            distance = target.price - entry_price
            rr = distance / risk if risk > 0 else 0

            if min_rr <= rr <= max_rr:
                return target.price, rr, "liquidity"

        # Fallback to fixed R:R
        tp = entry_price + (risk * fallback_rr)
        return tp, fallback_rr, "fallback"

    else:  # SHORT
        # Look for swing lows below entry
        targets = [l for l in levels if l.type == "swing_low" and l.price < entry_price]
        targets.sort(key=lambda x: x.price, reverse=True)

        for target in targets:
            distance = entry_price - target.price
            rr = distance / risk if risk > 0 else 0

            if min_rr <= rr <= max_rr:
                return target.price, rr, "liquidity"

        # Fallback to fixed R:R
        tp = entry_price - (risk * fallback_rr)
        return tp, fallback_rr, "fallback"


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


def detect_ifvg(candles: List[M1Candle], direction: str, lookback: int = 20) -> bool:
    if len(candles) < lookback:
        return False
    recent = candles[-lookback:]

    # Find FVGs
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


def check_confirmation(candles: List[M1Candle], direction: str) -> Tuple[bool, str]:
    if detect_choch(candles, direction, lookback=8):
        return True, "CHoCH"
    if detect_ifvg(candles, direction, lookback=15):
        return True, "iFVG"
    if detect_engulfing(candles, direction):
        return True, "Engulfing"
    return False, ""


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


def calculate_lot_size(balance: float, risk_pct: float, sl_pips: float, pip_value: float = 10.0) -> float:
    risk_amount = balance * (risk_pct / 100)
    if sl_pips <= 0:
        return 0.01
    lot_size = risk_amount / (sl_pips * pip_value)
    return max(0.01, min(round(lot_size, 2), 10.0))


def run_session_backtest(
    m5_by_time: dict,
    m1_by_time: dict,
    day: date,
    session_name: str,
    open_hour: int,
    open_minute: int,
    end_hour: int,
    current_balance: float,
    risk_percent: float,
    all_m1_candles: List[M1Candle],
    # Strategy params from main.py
    sl_buffer_pips: float = 1.0,
    min_range_pips: float = 2.0,
    max_range_pips: float = 30.0,
    min_rr_ratio: float = 1.5,
    max_rr_ratio: float = 4.0,
    fallback_rr_ratio: float = 2.5,
    max_retest_candles: int = 30,
    max_confirm_candles: int = 20,
    retest_tolerance_pips: float = 3.0,
) -> Optional[LOCBTrade]:
    """Run backtest for a single session."""

    pip_size = 0.0001
    pip_value = 10.0

    # Find opening candle (5-minute period from session open)
    session_open = datetime(day.year, day.month, day.day, open_hour, open_minute)

    # Collect M1 candles for the first 5 minutes (opening range)
    opening_candles = []
    for i in range(5):
        candle_time = session_open + timedelta(minutes=i)
        if candle_time in m1_by_time:
            rate = m1_by_time[candle_time]
            opening_candles.append(M1Candle(
                time=candle_time,
                open=rate["open"],
                high=rate["high"],
                low=rate["low"],
                close=rate["close"]
            ))

    if len(opening_candles) < 5:
        return None

    oc_high = max(c.high for c in opening_candles)
    oc_low = min(c.low for c in opening_candles)
    oc_range_pips = (oc_high - oc_low) / pip_size

    if not (min_range_pips <= oc_range_pips <= max_range_pips):
        return None

    # Breakout period (next 5 minutes)
    breakout_start = session_open + timedelta(minutes=5)
    direction = None
    retest_level = None

    for i in range(5):
        candle_time = breakout_start + timedelta(minutes=i)
        if candle_time in m1_by_time:
            rate = m1_by_time[candle_time]
            if rate["close"] > oc_high:
                direction = "LONG"
                retest_level = oc_high
                break
            elif rate["close"] < oc_low:
                direction = "SHORT"
                retest_level = oc_low
                break

    if direction is None:
        return None

    # Look for retest + confirmation
    m1_candles = []
    retest_found = False
    trade = None

    m1_start = breakout_start + timedelta(minutes=5)
    session_end = datetime(day.year, day.month, day.day, end_hour, 0)

    current_time = m1_start
    candles_since_breakout = 0
    candles_since_retest = 0

    while current_time < session_end:
        if current_time not in m1_by_time:
            current_time += timedelta(minutes=1)
            continue

        rate = m1_by_time[current_time]
        m1_candle = M1Candle(
            time=current_time,
            open=rate["open"],
            high=rate["high"],
            low=rate["low"],
            close=rate["close"]
        )
        m1_candles.append(m1_candle)
        candles_since_breakout += 1

        if not retest_found:
            if candles_since_breakout > max_retest_candles:
                break
            if check_retest(m1_candles, retest_level, direction.lower(), retest_tolerance_pips):
                retest_found = True
                candles_since_retest = 0

        elif trade is None:
            candles_since_retest += 1
            if candles_since_retest > max_confirm_candles:
                break

            confirmed, confirm_type = check_confirmation(m1_candles, direction.lower())

            if confirmed:
                entry_price = m1_candle.close

                if direction == "LONG":
                    sl = oc_low - (sl_buffer_pips * pip_size)
                else:
                    sl = oc_high + (sl_buffer_pips * pip_size)

                # Dynamic TP based on liquidity
                tp, actual_rr, tp_type = calculate_dynamic_tp(
                    entry_price, sl, direction,
                    all_m1_candles + m1_candles,
                    min_rr_ratio, max_rr_ratio, fallback_rr_ratio, pip_size
                )

                sl_pips = abs(entry_price - sl) / pip_size
                lot_size = calculate_lot_size(current_balance, risk_percent, sl_pips, pip_value)

                trade = LOCBTrade(
                    day=str(day),
                    session=session_name,
                    direction=direction,
                    entry_time=current_time,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    rr_ratio=actual_rr,
                    opening_candle_high=oc_high,
                    opening_candle_low=oc_low,
                    confirmation_type=confirm_type,
                    sl_pips=sl_pips,
                    lot_size=lot_size
                )

        elif trade and trade.result == "":
            if direction == "LONG":
                if m1_candle.low <= trade.stop_loss:
                    trade.exit_time = current_time
                    trade.exit_price = trade.stop_loss
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                    trade.pnl_dollars = trade.pnl_pips * pip_value * trade.lot_size
                    trade.result = "LOSS"
                    return trade
                if m1_candle.high >= trade.take_profit:
                    trade.exit_time = current_time
                    trade.exit_price = trade.take_profit
                    trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
                    trade.pnl_dollars = trade.pnl_pips * pip_value * trade.lot_size
                    trade.result = "WIN"
                    return trade
            else:
                if m1_candle.high >= trade.stop_loss:
                    trade.exit_time = current_time
                    trade.exit_price = trade.stop_loss
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                    trade.pnl_dollars = trade.pnl_pips * pip_value * trade.lot_size
                    trade.result = "LOSS"
                    return trade
                if m1_candle.low <= trade.take_profit:
                    trade.exit_time = current_time
                    trade.exit_price = trade.take_profit
                    trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
                    trade.pnl_dollars = trade.pnl_pips * pip_value * trade.lot_size
                    trade.result = "WIN"
                    return trade

        current_time += timedelta(minutes=1)

    # Close at session end
    if trade and trade.result == "":
        last_times = [t for t in m1_by_time.keys() if t.date() == day and t < session_end]
        if last_times:
            last_time = max(last_times)
            last_rate = m1_by_time[last_time]
            trade.exit_time = last_time
            trade.exit_price = last_rate["close"]
            if direction == "LONG":
                trade.pnl_pips = (trade.exit_price - trade.entry_price) / pip_size
            else:
                trade.pnl_pips = (trade.entry_price - trade.exit_price) / pip_size
            trade.pnl_dollars = trade.pnl_pips * pip_value * trade.lot_size
            trade.result = "WIN" if trade.pnl_pips > 0 else "LOSS"
            return trade

    return trade


def generate_news_events_for_period(news_filter: NewsFilter, start_date: date, end_date: date) -> None:
    """
    Generate high-impact news events for backtest period using the same logic as news_filter.
    This ensures backtest uses identical news filtering as live trading.
    """
    # Clear existing events
    news_filter.events = []

    current = start_date
    while current <= end_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        # NFP - First Friday of month
        if current.weekday() == 4 and current.day <= 7:
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="Non-Farm Payrolls"))
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="Unemployment Rate"))

        # Flash PMI - 3rd week (Mon/Tue) - HIGH impact
        if 15 <= current.day <= 23 and current.weekday() in [0, 1]:
            news_filter.events.append(EconomicEvent(
                date=current, time="04:30", currency="EUR",
                impact=NewsImpact.HIGH, event="German Flash Manufacturing PMI"))
            news_filter.events.append(EconomicEvent(
                date=current, time="04:30", currency="EUR",
                impact=NewsImpact.HIGH, event="German Flash Services PMI"))
            news_filter.events.append(EconomicEvent(
                date=current, time="09:45", currency="USD",
                impact=NewsImpact.HIGH, event="Flash Manufacturing PMI"))
            news_filter.events.append(EconomicEvent(
                date=current, time="09:45", currency="USD",
                impact=NewsImpact.HIGH, event="Flash Services PMI"))

        # CPI - Mid-month Wed/Thu
        if current.day in [10, 11, 12, 13, 14] and current.weekday() in [2, 3]:
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="CPI m/m"))
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="Core CPI m/m"))

        # Retail Sales - Around 15th-17th - HIGH impact
        if 14 <= current.day <= 17 and current.weekday() < 5:
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="Retail Sales m/m"))
            news_filter.events.append(EconomicEvent(
                date=current, time="08:30", currency="USD",
                impact=NewsImpact.HIGH, event="Core Retail Sales m/m"))

        # FOMC - Mid-month Wed in specific months
        if current.month in [1, 3, 5, 6, 7, 9, 11, 12]:
            if 14 <= current.day <= 21 and current.weekday() == 2:
                news_filter.events.append(EconomicEvent(
                    date=current, time="14:00", currency="USD",
                    impact=NewsImpact.HIGH, event="FOMC Statement"))
                news_filter.events.append(EconomicEvent(
                    date=current, time="14:30", currency="USD",
                    impact=NewsImpact.HIGH, event="FOMC Press Conference"))

        # ECB - Mid-month Thu in specific months
        if current.month in [1, 3, 4, 6, 7, 9, 10, 12]:
            if 10 <= current.day <= 17 and current.weekday() == 3:
                news_filter.events.append(EconomicEvent(
                    date=current, time="08:15", currency="EUR",
                    impact=NewsImpact.HIGH, event="ECB Interest Rate Decision"))

        # GDP - Last week of quarter end months
        if current.month in [1, 4, 7, 10] and 25 <= current.day <= 31:
            if current.weekday() in [2, 3, 4]:
                news_filter.events.append(EconomicEvent(
                    date=current, time="08:30", currency="USD",
                    impact=NewsImpact.HIGH, event="GDP q/q"))

        current += timedelta(days=1)

    # Rebuild high impact dates
    news_filter._build_high_impact_dates()


def main():
    print("="*70)
    print("  BACKTEST LOCB - PARAMETRI IDENTICI CU DEMO/REAL")
    print("="*70)
    print("  Cont: $100,000 | Risk: 1% per trade")
    print("  Sessions: London (12:00-15:00) + NY (18:30-21:00) server time")
    print("  R:R Dinamic: 1.5-4.0 (fallback 2.5)")
    print("  News Filter: ON")
    print("="*70)

    # Initialize MT5
    s = get_settings()
    if not mt5.initialize(
        login=s.mt5.login,
        password=s.mt5.password,
        server=s.mt5.server,
        path=s.mt5.path,
        timeout=120000,
    ):
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)

    # Get 2 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print(f"\nDownloading data: {start_date.date()} to {end_date.date()}...")

    m5_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
    m1_rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_date, end_date)

    mt5.shutdown()

    if m5_rates is None or m1_rates is None:
        print("Failed to get data!")
        return

    print(f"M5: {len(m5_rates)} candles | M1: {len(m1_rates)} candles")

    # Convert to dicts
    m5_by_time = {datetime.fromtimestamp(r["time"]): r for r in m5_rates}
    m1_by_time = {datetime.fromtimestamp(r["time"]): r for r in m1_rates}

    # Build M1 candles list for liquidity detection
    all_m1_candles = [
        M1Candle(time=datetime.fromtimestamp(r["time"]),
                 open=r["open"], high=r["high"], low=r["low"], close=r["close"])
        for r in m1_rates
    ]

    # Setup news filter (same logic as live trading)
    print("\nInitializing news filter...")
    config = NewsFilterConfig(
        filter_high_impact=True,
        filter_medium_impact=False,
        filter_entire_day=True,
        currencies=['EUR', 'USD'],
    )
    news_filter = NewsFilter(config)
    generate_news_events_for_period(news_filter, start_date.date(), end_date.date())

    high_days = sorted(set(e.date for e in news_filter.events if e.impact == NewsImpact.HIGH))
    print(f"High-impact news days: {len(high_days)}")

    # Strategy parameters from main.py
    PARAMS = {
        "sl_buffer_pips": 1.0,
        "min_range_pips": 2.0,
        "max_range_pips": 30.0,
        "min_rr_ratio": 1.5,
        "max_rr_ratio": 4.0,
        "fallback_rr_ratio": 2.5,
        "max_retest_candles": 30,
        "max_confirm_candles": 20,
        "retest_tolerance_pips": 3.0,
    }

    # Sessions from main.py
    SESSIONS = {
        "london": {"open_hour": 12, "open_minute": 0, "end_hour": 15},
        "ny": {"open_hour": 18, "open_minute": 30, "end_hour": 21},
    }

    account_balance = 100000.0
    risk_percent = 1.0
    current_balance = account_balance

    trades = []
    news_blocked_days = []

    # Get unique trading days
    days = sorted(set(datetime.fromtimestamp(r["time"]).date() for r in m5_rates if datetime.fromtimestamp(r["time"]).weekday() < 5))

    print(f"\nRunning backtest on {len(days)} trading days...")
    print("-"*70)

    for day in days:
        # Check news filter
        check_dt = datetime.combine(day, datetime.min.time().replace(hour=12))
        if not news_filter.is_safe_to_trade(check_dt, "EURUSD"):
            news_blocked_days.append(day)
            continue

        trades_today = 0

        # Run both sessions
        for session_name, session_params in SESSIONS.items():
            if trades_today >= 2:  # max_trades_per_day
                break

            trade = run_session_backtest(
                m5_by_time, m1_by_time, day, session_name,
                session_params["open_hour"],
                session_params["open_minute"],
                session_params["end_hour"],
                current_balance, risk_percent,
                all_m1_candles,
                **PARAMS
            )

            if trade:
                current_balance += trade.pnl_dollars
                trades.append(trade)
                trades_today += 1

    # Results
    print("\n" + "="*70)
    print("  REZULTATE BACKTEST")
    print("="*70)

    print(f"\n  Zile blocate de News Filter: {len(news_blocked_days)}")

    if not trades:
        print("  Nu au fost tranzacții!")
        return

    # Monthly breakdown
    months = {}
    for t in trades:
        dt = datetime.strptime(t.day, "%Y-%m-%d")
        key = (dt.year, dt.month)
        if key not in months:
            months[key] = []
        months[key].append(t)

    month_names = {10: "Octombrie", 11: "Noiembrie", 12: "Decembrie", 1: "Ianuarie"}

    for (year, month), month_trades in sorted(months.items()):
        wins = [t for t in month_trades if t.result == "WIN"]
        losses = [t for t in month_trades if t.result == "LOSS"]
        total_pips = sum(t.pnl_pips for t in month_trades)
        total_dollars = sum(t.pnl_dollars for t in month_trades)
        wr = len(wins) / len(month_trades) * 100 if month_trades else 0

        london_trades = [t for t in month_trades if t.session == "london"]
        ny_trades = [t for t in month_trades if t.session == "ny"]

        print(f"\n  {'='*60}")
        print(f"  {month_names.get(month, f'Luna {month}')} {year}")
        print(f"  {'='*60}")
        print(f"  Trades: {len(month_trades)} (London: {len(london_trades)}, NY: {len(ny_trades)})")
        print(f"  Win/Loss: {len(wins)}/{len(losses)} ({wr:.1f}% win rate)")
        print(f"  Total Pips: {total_pips:+.1f}")
        print(f"  Total P/L: ${total_dollars:+,.2f}")
        print(f"  Return: {(total_dollars/account_balance)*100:+.2f}%")

    # Overall
    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    total_pips = sum(t.pnl_pips for t in trades)
    total_dollars = sum(t.pnl_dollars for t in trades)
    wr = len(wins) / len(trades) * 100

    win_pips = sum(t.pnl_pips for t in wins) if wins else 0
    loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0.001
    pf = win_pips / loss_pips

    avg_rr = sum(t.rr_ratio for t in trades) / len(trades)

    london_trades = [t for t in trades if t.session == "london"]
    ny_trades = [t for t in trades if t.session == "ny"]

    print(f"\n  {'='*60}")
    print(f"  TOTAL (2 LUNI)")
    print(f"  {'='*60}")
    print(f"  Trades: {len(trades)}")
    print(f"    - London: {len(london_trades)} ({len([t for t in london_trades if t.result=='WIN'])} wins)")
    print(f"    - NY: {len(ny_trades)} ({len([t for t in ny_trades if t.result=='WIN'])} wins)")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Avg R:R folosit: {avg_rr:.2f}")
    print(f"  Total Pips: {total_pips:+.1f}")
    print(f"  Total P/L: ${total_dollars:+,.2f}")
    print(f"  Return: {(total_dollars/account_balance)*100:+.2f}%")
    print(f"  Sold final: ${current_balance:,.2f}")

    # Drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t.pnl_dollars
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    print(f"  Max Drawdown: ${max_dd:,.2f} ({(max_dd/account_balance)*100:.2f}%)")

    # Trade details
    print(f"\n  {'='*60}")
    print(f"  TRADE DETAILS")
    print(f"  {'='*60}")

    for t in trades:
        print(
            f"  {t.day} | {t.session:6} | {t.direction:5} | "
            f"R:R {t.rr_ratio:.1f} | {t.confirmation_type:8} | "
            f"{t.result:4} | {t.pnl_pips:+6.1f}p | ${t.pnl_dollars:+8,.2f}"
        )

    print("\n" + "="*70)
    print("  BACKTEST COMPLET")
    print("="*70)


if __name__ == "__main__":
    main()
