"""
Backtest ORB Strategy on EUR/USD - Adjusted for Forex
Opening Range: 15-30 minutes (not 5 min like stocks)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaTrader5 as mt5
from datetime import datetime, timedelta, time
from config.settings import get_settings
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Trade:
    day: str
    entry_time: datetime
    entry_price: float
    direction: str
    stop_loss: float
    take_profit: float
    orb_high: float
    orb_low: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    result: str = ""


def run_forex_orb_backtest(
    rates,
    session_open_hour: int,
    session_close_hour: int,
    orb_minutes: int = 15,
    risk_reward: float = 2.0,
) -> List[Trade]:
    """Run ORB backtest with configurable opening range duration."""
    trades = []
    pip_size = 0.0001

    # Group candles by day
    days = {}
    for rate in rates:
        ts = datetime.fromtimestamp(rate["time"])
        day = ts.date()
        if day not in days:
            days[day] = []
        days[day].append(rate)

    for day in sorted(days.keys()):
        if day.weekday() >= 5:
            continue

        day_rates = days[day]

        # Find Opening Range (first N minutes after session open)
        orb_candles = []

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])
            if ts.hour == session_open_hour:
                minutes_after_open = ts.minute
                if minutes_after_open < orb_minutes:
                    orb_candles.append(rate)

        if not orb_candles:
            continue

        # Calculate ORB high/low from all candles in the range
        orb_high = max(c["high"] for c in orb_candles)
        orb_low = min(c["low"] for c in orb_candles)
        orb_range = orb_high - orb_low

        # Skip if range is 0
        if orb_range < 0.00001:
            continue

        # Look for breakout after ORB period
        trade_taken = False
        current_trade = None

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])

            # Skip during ORB period
            if ts.hour == session_open_hour and ts.minute < orb_minutes:
                continue

            # Skip outside session
            if ts.hour < session_open_hour or ts.hour >= session_close_hour:
                continue

            # Check exits for open trade
            if current_trade and current_trade.result == "":
                if current_trade.direction == "LONG":
                    if rate["low"] <= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (
                            current_trade.exit_price - current_trade.entry_price
                        ) / pip_size
                        current_trade.result = "LOSS"
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["high"] >= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (
                            current_trade.exit_price - current_trade.entry_price
                        ) / pip_size
                        current_trade.result = "WIN"
                        trades.append(current_trade)
                        current_trade = None
                        continue
                else:  # SHORT
                    if rate["high"] >= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (
                            current_trade.entry_price - current_trade.exit_price
                        ) / pip_size
                        current_trade.result = "LOSS"
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["low"] <= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (
                            current_trade.entry_price - current_trade.exit_price
                        ) / pip_size
                        current_trade.result = "WIN"
                        trades.append(current_trade)
                        current_trade = None
                        continue

            # Look for entry
            if not trade_taken and current_trade is None:
                # LONG breakout
                if rate["close"] > orb_high:
                    entry = rate["close"]
                    sl = orb_low
                    risk = entry - sl
                    tp = entry + risk * risk_reward

                    current_trade = Trade(
                        day=str(day),
                        entry_time=ts,
                        entry_price=entry,
                        direction="LONG",
                        stop_loss=sl,
                        take_profit=tp,
                        orb_high=orb_high,
                        orb_low=orb_low,
                    )
                    trade_taken = True

                # SHORT breakout
                elif rate["close"] < orb_low:
                    entry = rate["close"]
                    sl = orb_high
                    risk = sl - entry
                    tp = entry - risk * risk_reward

                    current_trade = Trade(
                        day=str(day),
                        entry_time=ts,
                        entry_price=entry,
                        direction="SHORT",
                        stop_loss=sl,
                        take_profit=tp,
                        orb_high=orb_high,
                        orb_low=orb_low,
                    )
                    trade_taken = True

        # Close at session end
        if current_trade and current_trade.result == "":
            last_rate = day_rates[-1]
            current_trade.exit_time = datetime.fromtimestamp(last_rate["time"])
            current_trade.exit_price = last_rate["close"]
            if current_trade.direction == "LONG":
                current_trade.pnl_pips = (
                    current_trade.exit_price - current_trade.entry_price
                ) / pip_size
            else:
                current_trade.pnl_pips = (
                    current_trade.entry_price - current_trade.exit_price
                ) / pip_size
            current_trade.result = "WIN" if current_trade.pnl_pips > 0 else "LOSS"
            trades.append(current_trade)

    return trades


def print_results(trades: List[Trade], name: str, show_details: bool = False):
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
    wr = len(wins) / len(trades) * 100

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

    print(f"{name}:")
    print(f"  Trades: {len(trades)} | W/L: {len(wins)}/{len(losses)} | WR: {wr:.0f}%")
    print(f"  Total: {total_pips:.1f} pips | PF: {pf:.2f} | MaxDD: {max_dd:.1f} pips")

    if show_details and trades:
        print(f"  Avg Win: {win_pips/len(wins):.1f} pips" if wins else "")
        print(f"  Avg Loss: {loss_pips/len(losses):.1f} pips" if losses else "")
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
    }


def show_trade_details(trades: List[Trade]):
    """Show individual trade details."""
    print("\nDetalii tranzactii:")
    print("-" * 90)
    for t in trades:
        range_pips = (t.orb_high - t.orb_low) / 0.0001
        print(
            f"  {t.day} | {t.direction:5} | Entry: {t.entry_price:.5f} | "
            f"SL: {t.stop_loss:.5f} | TP: {t.take_profit:.5f} | "
            f"Result: {t.result:4} | PnL: {t.pnl_pips:+.1f} pips | Range: {range_pips:.1f} pips"
        )


if __name__ == "__main__":
    # Get data
    s = get_settings()
    mt5.initialize(
        login=s.mt5.login,
        password=s.mt5.password,
        server=s.mt5.server,
        path=s.mt5.path,
        timeout=60000,
    )

    symbol = "EURUSD"
    mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        datetime.now() - timedelta(days=35),
        datetime.now(),
    )

    mt5.shutdown()

    if rates is None:
        print("Nu am putut obtine date!")
        exit()

    print("=" * 80)
    print("  BACKTEST EUR/USD ORB - PARAMETRI AJUSTATI PENTRU FOREX")
    print("=" * 80)
    print()

    # Test different sessions with 15-min ORB
    print("=" * 80)
    print("  TEST SESIUNI - ORB 15 minute, R:R 2:1")
    print("=" * 80)
    print()

    configs = [
        ("LONDRA (10:00-18:00)", 10, 18),
        ("NEW YORK (15:00-22:00)", 15, 22),
        ("ASIA (02:00-10:00)", 2, 10),
    ]

    best_session = None
    best_pf = 0

    for name, open_h, close_h in configs:
        trades = run_forex_orb_backtest(
            rates, open_h, close_h, orb_minutes=15, risk_reward=2.0
        )
        result = print_results(trades, name)
        if result and result.get("profit_factor", 0) > best_pf:
            best_pf = result["profit_factor"]
            best_session = (name, open_h, close_h)

    # Test different ORB durations for best session
    if best_session:
        print("=" * 80)
        print(f"  TEST DURATE ORB - {best_session[0]}")
        print("=" * 80)
        print()

        best_orb = 15
        best_orb_pf = 0

        for orb_min in [15, 30, 45, 60]:
            trades = run_forex_orb_backtest(
                rates, best_session[1], best_session[2], orb_minutes=orb_min, risk_reward=2.0
            )
            result = print_results(trades, f"ORB {orb_min} min")
            if result and result.get("profit_factor", 0) > best_orb_pf:
                best_orb_pf = result["profit_factor"]
                best_orb = orb_min

        # Test different R:R with best ORB
        print("=" * 80)
        print(f"  TEST R:R - {best_session[0]}, ORB {best_orb} min")
        print("=" * 80)
        print()

        best_rr = 2.0
        best_rr_result = None

        for rr in [1.5, 2.0, 2.5, 3.0]:
            trades = run_forex_orb_backtest(
                rates, best_session[1], best_session[2], orb_minutes=best_orb, risk_reward=rr
            )
            result = print_results(trades, f"R:R {rr}:1", show_details=True)
            if result and result.get("total_pips", -999) > (best_rr_result or {}).get("total_pips", -999):
                best_rr = rr
                best_rr_result = result

        # Show best configuration trades
        print("=" * 80)
        print("  CONFIGURATIE OPTIMA")
        print("=" * 80)
        print()
        print(f"  Sesiune: {best_session[0]}")
        print(f"  Opening Range: {best_orb} minute")
        print(f"  Risk/Reward: {best_rr}:1")
        print()

        # Run final backtest with best params and show details
        trades = run_forex_orb_backtest(
            rates, best_session[1], best_session[2], orb_minutes=best_orb, risk_reward=best_rr
        )
        print_results(trades, "REZULTAT FINAL", show_details=True)
        show_trade_details(trades)
