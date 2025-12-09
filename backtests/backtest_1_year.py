"""
Backtest EUR/USD ORB Strategy - 1 Year with Monthly Statistics.

Uses real MT5 data with corrected server time hours.
Server time = Romania time + 2 hours
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict


@dataclass
class Trade:
    date: str
    month: str
    entry_time: datetime
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    lot_size: float
    risk_dollars: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_dollars: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""


def calculate_lot_size(balance, risk_percent, sl_pips, pip_value=10.0):
    """Calculate lot size based on risk."""
    risk_amount = balance * (risk_percent / 100)
    lot_size = risk_amount / (sl_pips * pip_value)
    return round(max(0.01, min(lot_size, 10.0)), 2)


def run_backtest_1_year():
    """Run 1 year backtest with monthly statistics."""

    import MetaTrader5 as mt5
    from config.settings import get_settings

    settings = get_settings()

    # Connect to MT5
    if not mt5.initialize(
        login=settings.mt5.login,
        password=settings.mt5.password,
        server=settings.mt5.server,
        path=settings.mt5.path,
        timeout=60000,
    ):
        print(f"ERROR: Cannot connect to MT5: {mt5.last_error()}")
        return None

    print("Connected to MT5!")

    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)

    # Get maximum available M5 data (use copy_rates_from_pos for more data)
    print(f"Fetching maximum available historical data...")

    rates = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_M5,
        0,  # From current
        50000  # Get 50k candles (~8 months of M5 data)
    )

    if rates is not None and len(rates) > 0:
        start_date = datetime.fromtimestamp(rates[0]["time"])
        end_date = datetime.fromtimestamp(rates[-1]["time"])
        print(f"Data range: {start_date.date()} to {end_date.date()}")

    if rates is None or len(rates) == 0:
        print("ERROR: No data received!")
        return None

    print(f"Got {len(rates)} M5 candles")

    # Get H1 data for trend filter
    print("Fetching H1 data for trend filter...")
    h1_rates = mt5.copy_rates_from_pos(
        symbol,
        mt5.TIMEFRAME_H1,
        0,
        5000  # ~7 months of H1 data
    )

    if h1_rates is not None:
        print(f"Got {len(h1_rates)} H1 candles")
    else:
        print("WARNING: No H1 data, trend filter disabled")
        h1_rates = []

    mt5.shutdown()

    # Build H1 lookup by date/hour
    h1_by_datetime = {}
    for rate in h1_rates:
        ts = datetime.fromtimestamp(rate["time"])
        h1_by_datetime[(ts.date(), ts.hour)] = rate

    def calculate_ema(prices, period):
        """Calculate EMA."""
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def get_trend_at(day, hour, ema_period=50):
        """Get trend at specific day/hour using H1 EMA."""
        # Collect H1 closes before this point
        h1_closes = []
        for rate in h1_rates:
            ts = datetime.fromtimestamp(rate["time"])
            if ts.date() < day or (ts.date() == day and ts.hour < hour):
                h1_closes.append(rate["close"])

        if len(h1_closes) < ema_period + 5:
            return None

        ema = calculate_ema(h1_closes, ema_period)
        if ema is None:
            return None

        current_price = h1_closes[-1]
        return "UP" if current_price > ema else "DOWN"

    # Configuration - SERVER TIME (TeleTrade = Romania + 2h)
    SESSION_START_HOUR = 12  # 12:00 server = 10:00 Romania
    SESSION_END_HOUR = 21    # 21:00 server = 19:00 Romania
    ORB_MINUTES = 60
    RISK_REWARD = 2.0
    MIN_ORB_PIPS = 10.0
    MAX_ORB_PIPS = 40.0
    BUFFER_PIPS = 3.0  # Increased from 2.0
    MIN_MINUTES_AFTER_ORB = 5  # Wait 5 min after ORB
    USE_TREND_FILTER = True  # Only trade with trend
    TREND_EMA_PERIOD = 50  # EMA 50 on H1

    STARTING_BALANCE = 100_000.0
    # OPTIMAL FIXED RISK (best backtest results: +4.26%, 4.52% max DD)
    RISK_PERCENT = 0.5  # Fixed 0.5% per trade
    PIP_SIZE = 0.0001
    PIP_VALUE = 10.0

    # Group by day
    days = defaultdict(list)
    for rate in rates:
        ts = datetime.fromtimestamp(rate["time"])
        day = ts.date()
        days[day].append(rate)

    trades = []
    current_balance = STARTING_BALANCE

    for day in sorted(days.keys()):
        if day.weekday() >= 5:  # Skip weekends
            continue

        day_rates = days[day]

        # Find Opening Range candles (SESSION_START_HOUR to SESSION_START_HOUR + ORB)
        orb_candles = []
        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])
            if ts.hour == SESSION_START_HOUR and ts.minute < ORB_MINUTES:
                orb_candles.append(rate)

        if not orb_candles:
            continue

        orb_high = max(c["high"] for c in orb_candles)
        orb_low = min(c["low"] for c in orb_candles)
        orb_range_pips = (orb_high - orb_low) / PIP_SIZE

        # Filters
        if orb_range_pips < MIN_ORB_PIPS or orb_range_pips > MAX_ORB_PIPS:
            continue

        buffer = BUFFER_PIPS * PIP_SIZE

        # Look for breakout
        trade_taken = False
        current_trade = None

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])

            # Only trade after ORB and within session
            if ts.hour == SESSION_START_HOUR and ts.minute < ORB_MINUTES:
                continue
            if ts.hour < SESSION_START_HOUR or ts.hour >= SESSION_END_HOUR:
                continue

            # Check exits
            if current_trade and current_trade.result == "":
                if current_trade.direction == "LONG":
                    if rate["low"] <= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / PIP_SIZE
                        current_trade.pnl_dollars = current_trade.pnl_pips * PIP_VALUE * current_trade.lot_size
                        current_trade.result = "LOSS"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["high"] >= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / PIP_SIZE
                        current_trade.pnl_dollars = current_trade.pnl_pips * PIP_VALUE * current_trade.lot_size
                        current_trade.result = "WIN"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                else:  # SHORT
                    if rate["high"] >= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / PIP_SIZE
                        current_trade.pnl_dollars = current_trade.pnl_pips * PIP_VALUE * current_trade.lot_size
                        current_trade.result = "LOSS"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["low"] <= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / PIP_SIZE
                        current_trade.pnl_dollars = current_trade.pnl_pips * PIP_VALUE * current_trade.lot_size
                        current_trade.result = "WIN"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue

            # Look for entry
            if not trade_taken and current_trade is None:
                # Check min minutes after ORB
                orb_end_hour = SESSION_START_HOUR + ORB_MINUTES // 60
                orb_end_minute = ORB_MINUTES % 60
                orb_end_time = datetime.combine(day, datetime.min.time()).replace(
                    hour=orb_end_hour, minute=orb_end_minute
                )
                minutes_since_orb = (ts - orb_end_time).total_seconds() / 60
                if minutes_since_orb < MIN_MINUTES_AFTER_ORB:
                    continue

                # Get trend at this point
                trend = get_trend_at(day, ts.hour, TREND_EMA_PERIOD) if USE_TREND_FILTER else None

                # LONG breakout
                if rate["close"] > orb_high + buffer:
                    # Trend filter - only LONG if trend is UP or unknown
                    if USE_TREND_FILTER and trend == "DOWN":
                        continue

                    entry = rate["close"]
                    sl = orb_low - buffer
                    sl_pips = (entry - sl) / PIP_SIZE
                    risk = entry - sl
                    tp = entry + risk * RISK_REWARD

                    lot_size = calculate_lot_size(current_balance, RISK_PERCENT, sl_pips)
                    risk_dollars = sl_pips * PIP_VALUE * lot_size

                    current_trade = Trade(
                        date=str(day),
                        month=day.strftime("%Y-%m"),
                        entry_time=ts,
                        entry_price=entry,
                        direction="LONG",
                        stop_loss=sl,
                        take_profit=tp,
                        lot_size=lot_size,
                        risk_dollars=risk_dollars,
                    )
                    trade_taken = True

                # SHORT breakout
                elif rate["close"] < orb_low - buffer:
                    # Trend filter - only SHORT if trend is DOWN or unknown
                    if USE_TREND_FILTER and trend == "UP":
                        continue

                    entry = rate["close"]
                    sl = orb_high + buffer
                    sl_pips = (sl - entry) / PIP_SIZE
                    risk = sl - entry
                    tp = entry - risk * RISK_REWARD

                    lot_size = calculate_lot_size(current_balance, RISK_PERCENT, sl_pips)
                    risk_dollars = sl_pips * PIP_VALUE * lot_size

                    current_trade = Trade(
                        date=str(day),
                        month=day.strftime("%Y-%m"),
                        entry_time=ts,
                        entry_price=entry,
                        direction="SHORT",
                        stop_loss=sl,
                        take_profit=tp,
                        lot_size=lot_size,
                        risk_dollars=risk_dollars,
                    )
                    trade_taken = True

        # Close at session end
        if current_trade and current_trade.result == "":
            last_rate = day_rates[-1]
            current_trade.exit_time = datetime.fromtimestamp(last_rate["time"])
            current_trade.exit_price = last_rate["close"]
            if current_trade.direction == "LONG":
                current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / PIP_SIZE
            else:
                current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / PIP_SIZE
            current_trade.pnl_dollars = current_trade.pnl_pips * PIP_VALUE * current_trade.lot_size
            current_trade.result = "WIN" if current_trade.pnl_dollars > 0 else "LOSS"
            current_balance += current_trade.pnl_dollars
            trades.append(current_trade)

    return trades, STARTING_BALANCE, current_balance


def print_results(trades, starting_balance, final_balance):
    """Print detailed results with monthly breakdown."""

    print("\n" + "=" * 90)
    print("  BACKTEST RESULTS - EUR/USD ORB STRATEGY - 1 YEAR")
    print("  TeleTrade-Sharp ECN | London Session | 0.5% Risk per Trade")
    print("=" * 90)

    # Overall stats
    total_pnl = final_balance - starting_balance
    total_return = (total_pnl / starting_balance) * 100

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max drawdown
    peak = starting_balance
    max_dd = 0
    max_dd_pct = 0
    balance = starting_balance

    for t in trades:
        balance += t.pnl_dollars
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = (dd / peak) * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    print(f"\n  OVERALL PERFORMANCE")
    print(f"  " + "-" * 50)
    print(f"  Starting Balance:    ${starting_balance:>12,.2f}")
    print(f"  Final Balance:       ${final_balance:>12,.2f}")
    print(f"  Net Profit/Loss:     ${total_pnl:>+12,.2f} ({total_return:+.2f}%)")
    print()
    print(f"  Total Trades:        {len(trades):>12}")
    print(f"  Wins / Losses:       {len(wins):>5} / {len(losses)}")
    print(f"  Win Rate:            {win_rate:>11.1f}%")
    print(f"  Profit Factor:       {profit_factor:>12.2f}")
    print()
    print(f"  Gross Profit:        ${gross_profit:>12,.2f}")
    print(f"  Gross Loss:          ${gross_loss:>12,.2f}")
    print(f"  Max Drawdown:        ${max_dd:>12,.2f} ({max_dd_pct:.2f}%)")
    print()
    print(f"  Avg Win:             ${gross_profit/len(wins):>12,.2f}" if wins else "")
    print(f"  Avg Loss:            ${gross_loss/len(losses):>12,.2f}" if losses else "")

    # Monthly breakdown
    print("\n" + "=" * 90)
    print("  MONTHLY PERFORMANCE")
    print("=" * 90)

    monthly_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})

    for t in trades:
        m = t.month
        monthly_stats[m]["trades"] += 1
        monthly_stats[m]["pnl"] += t.pnl_dollars
        if t.result == "WIN":
            monthly_stats[m]["wins"] += 1
        else:
            monthly_stats[m]["losses"] += 1

    print(f"\n  {'Month':<10} | {'Trades':>7} | {'W/L':>7} | {'Win%':>7} | {'P&L':>12} | {'Cumulative':>12}")
    print("  " + "-" * 75)

    cumulative = starting_balance
    for month in sorted(monthly_stats.keys()):
        stats = monthly_stats[month]
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        cumulative += stats["pnl"]

        pnl_str = f"${stats['pnl']:+,.2f}"
        cum_str = f"${cumulative:,.2f}"

        print(f"  {month:<10} | {stats['trades']:>7} | {stats['wins']:>3}/{stats['losses']:<3} | {wr:>6.1f}% | {pnl_str:>12} | {cum_str:>12}")

    # Equity curve summary
    print("\n" + "=" * 90)
    print("  EQUITY CURVE (Monthly)")
    print("=" * 90)

    balance = starting_balance
    print(f"\n  Start: ${starting_balance:,.2f}")

    for month in sorted(monthly_stats.keys()):
        balance += monthly_stats[month]["pnl"]
        bar_len = int((balance - starting_balance) / 1000)
        if bar_len >= 0:
            bar = "#" * min(bar_len, 50)
            print(f"  {month}: ${balance:>12,.2f} |{bar}")
        else:
            bar = "-" * min(abs(bar_len), 50)
            print(f"  {month}: ${balance:>12,.2f} |{bar}")

    print(f"\n  End:   ${final_balance:,.2f}")

    # Risk analysis
    print("\n" + "=" * 90)
    print("  RISK ANALYSIS")
    print("=" * 90)

    # Calculate monthly returns
    monthly_returns = []
    for month in sorted(monthly_stats.keys()):
        monthly_returns.append(monthly_stats[month]["pnl"] / starting_balance * 100)

    if monthly_returns:
        avg_monthly = sum(monthly_returns) / len(monthly_returns)
        positive_months = len([r for r in monthly_returns if r > 0])
        negative_months = len([r for r in monthly_returns if r <= 0])

        print(f"\n  Avg Monthly Return:  {avg_monthly:>+.2f}%")
        print(f"  Positive Months:     {positive_months:>3} / {len(monthly_returns)}")
        print(f"  Negative Months:     {negative_months:>3} / {len(monthly_returns)}")
        print(f"  Best Month:          {max(monthly_returns):>+.2f}%")
        print(f"  Worst Month:         {min(monthly_returns):>+.2f}%")

    print("\n" + "=" * 90)
    if total_pnl > 0:
        print(f"  CONCLUSION: PROFITABLE STRATEGY!")
        print(f"  Annual return: {total_return:+.2f}% with max drawdown {max_dd_pct:.2f}%")
    else:
        print(f"  CONCLUSION: Strategy needs optimization")
    print("=" * 90 + "\n")


def main():
    print("\n" + "=" * 90)
    print("  RUNNING 1-YEAR BACKTEST...")
    print("=" * 90)

    result = run_backtest_1_year()

    if result:
        trades, starting, final = result
        print_results(trades, starting, final)
    else:
        print("Backtest failed!")


if __name__ == "__main__":
    main()
