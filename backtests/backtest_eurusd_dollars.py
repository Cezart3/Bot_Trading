"""
Backtest EUR/USD ORB - Cu valoare cont in dolari si 0.5% risk per trade.

Opening Range: 60 minute (10:00-11:00 Londra)
Risk per trade: 0.5% din cont
Risk/Reward: 2:1
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
    lot_size: float
    risk_dollars: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_dollars: float = 0.0
    pnl_pips: float = 0.0
    result: str = ""


def calculate_lot_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = 10.0,  # $10 per pip for 1 lot EUR/USD
) -> float:
    """
    Calculate lot size based on risk percentage.

    Args:
        account_balance: Current account balance
        risk_percent: Risk percentage (e.g., 0.5 for 0.5%)
        stop_loss_pips: Distance to stop loss in pips
        pip_value_per_lot: Value of 1 pip per 1 standard lot ($10 for EUR/USD)

    Returns:
        Lot size rounded to 2 decimal places
    """
    risk_amount = account_balance * (risk_percent / 100)
    lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)

    # Round to 2 decimals and enforce min/max
    lot_size = round(lot_size, 2)
    lot_size = max(0.01, min(lot_size, 10.0))  # Min 0.01, Max 10 lots

    return lot_size


def run_backtest_with_money(
    rates,
    starting_balance: float = 10000.0,
    risk_percent: float = 0.5,
    session_open_hour: int = 10,
    session_close_hour: int = 18,
    orb_minutes: int = 60,
    risk_reward: float = 2.0,
    # Optimized filters
    min_orb_range_pips: float = 10.0,  # Avoid tight ranges (false breakouts)
    max_orb_range_pips: float = 40.0,  # Avoid extreme volatility
    breakout_buffer_pips: float = 2.0,  # Confirm breakout with buffer
) -> tuple[List[Trade], float]:
    """Run backtest with real money management and optimized filters."""
    trades = []
    pip_size = 0.0001
    pip_value_per_lot = 10.0  # $10 per pip per lot for EUR/USD

    current_balance = starting_balance
    filtered_days = 0

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

        # Find Opening Range
        orb_candles = []
        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])
            if ts.hour == session_open_hour and ts.minute < orb_minutes:
                orb_candles.append(rate)

        if not orb_candles:
            continue

        orb_high = max(c["high"] for c in orb_candles)
        orb_low = min(c["low"] for c in orb_candles)
        orb_range_pips = (orb_high - orb_low) / pip_size

        # Optimized filters - skip if range too small or too large
        if orb_range_pips < min_orb_range_pips:
            filtered_days += 1
            continue
        if orb_range_pips > max_orb_range_pips:
            filtered_days += 1
            continue

        # Calculate breakout buffer
        buffer = breakout_buffer_pips * pip_size

        # Look for breakout
        trade_taken = False
        current_trade = None

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate["time"])

            if ts.hour == session_open_hour and ts.minute < orb_minutes:
                continue
            if ts.hour < session_open_hour or ts.hour >= session_close_hour:
                continue

            # Check exits
            if current_trade and current_trade.result == "":
                if current_trade.direction == "LONG":
                    if rate["low"] <= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                        current_trade.pnl_dollars = current_trade.pnl_pips * pip_value_per_lot * current_trade.lot_size
                        current_trade.result = "LOSS"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["high"] >= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                        current_trade.pnl_dollars = current_trade.pnl_pips * pip_value_per_lot * current_trade.lot_size
                        current_trade.result = "WIN"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                else:  # SHORT
                    if rate["high"] >= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                        current_trade.pnl_dollars = current_trade.pnl_pips * pip_value_per_lot * current_trade.lot_size
                        current_trade.result = "LOSS"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate["low"] <= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                        current_trade.pnl_dollars = current_trade.pnl_pips * pip_value_per_lot * current_trade.lot_size
                        current_trade.result = "WIN"
                        current_balance += current_trade.pnl_dollars
                        trades.append(current_trade)
                        current_trade = None
                        continue

            # Look for entry with breakout buffer
            if not trade_taken and current_trade is None:
                # LONG breakout - close must be above OR high + buffer
                if rate["close"] > orb_high + buffer:
                    entry = rate["close"]
                    sl = orb_low - buffer  # SL below OR low with buffer
                    sl_pips = (entry - sl) / pip_size
                    risk = entry - sl
                    tp = entry + risk * risk_reward

                    # Calculate lot size based on current balance
                    lot_size = calculate_lot_size(current_balance, risk_percent, sl_pips)
                    risk_dollars = sl_pips * pip_value_per_lot * lot_size

                    current_trade = Trade(
                        day=str(day),
                        entry_time=ts,
                        entry_price=entry,
                        direction="LONG",
                        stop_loss=sl,
                        take_profit=tp,
                        lot_size=lot_size,
                        risk_dollars=risk_dollars,
                    )
                    trade_taken = True

                # SHORT breakout - close must be below OR low - buffer
                elif rate["close"] < orb_low - buffer:
                    entry = rate["close"]
                    sl = orb_high + buffer  # SL above OR high with buffer
                    sl_pips = (sl - entry) / pip_size
                    risk = sl - entry
                    tp = entry - risk * risk_reward

                    lot_size = calculate_lot_size(current_balance, risk_percent, sl_pips)
                    risk_dollars = sl_pips * pip_value_per_lot * lot_size

                    current_trade = Trade(
                        day=str(day),
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
                current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
            else:
                current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
            current_trade.pnl_dollars = current_trade.pnl_pips * pip_value_per_lot * current_trade.lot_size
            current_trade.result = "WIN" if current_trade.pnl_dollars > 0 else "LOSS"
            current_balance += current_trade.pnl_dollars
            trades.append(current_trade)

    return trades, current_balance


def main():
    # Connect to MT5
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

    # Get data
    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M5,
        datetime.now() - timedelta(days=35),
        datetime.now(),
    )

    mt5.shutdown()

    if rates is None:
        print("Nu am putut obtine date!")
        return

    # Configuration
    STARTING_BALANCE = 10000.0  # $10,000
    RISK_PERCENT = 0.5  # 0.5% risk per trade

    print("=" * 80)
    print("      BACKTEST EUR/USD ORB - LONDRA SESSION")
    print("=" * 80)
    print()
    print("  CONFIGURATIE:")
    print(f"    Cont initial:     ${STARTING_BALANCE:,.2f}")
    print(f"    Risk per trade:   {RISK_PERCENT}%")
    print(f"    Opening Range:    60 minute (10:00-11:00)")
    print(f"    Sesiune:          Londra (10:00-18:00)")
    print(f"    Risk/Reward:      2:1")
    print()

    # Run backtest
    trades, final_balance = run_backtest_with_money(
        rates,
        starting_balance=STARTING_BALANCE,
        risk_percent=RISK_PERCENT,
        session_open_hour=10,
        session_close_hour=18,
        orb_minutes=60,
        risk_reward=2.0,
    )

    # Calculate statistics
    total_pnl = final_balance - STARTING_BALANCE
    return_pct = (total_pnl / STARTING_BALANCE) * 100

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    # Max drawdown
    peak = STARTING_BALANCE
    max_dd = 0
    max_dd_pct = 0
    balance = STARTING_BALANCE

    for t in trades:
        balance += t.pnl_dollars
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = (dd / peak) * 100
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    # Print results
    print("=" * 80)
    print("      REZULTATE")
    print("=" * 80)
    print()
    print(f"  CONT INITIAL:       ${STARTING_BALANCE:>12,.2f}")
    print(f"  CONT FINAL:         ${final_balance:>12,.2f}")
    print(f"  PROFIT/PIERDERE:    ${total_pnl:>+12,.2f} ({return_pct:+.2f}%)")
    print()
    print(f"  Total tranzactii:   {len(trades)}")
    print(f"  Castiguri/Pierderi: {len(wins)}/{len(losses)}")
    print(f"  Win Rate:           {win_rate:.1f}%")
    print(f"  Profit Factor:      {profit_factor:.2f}")
    print()
    print(f"  Max Drawdown:       ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
    print(f"  Avg Win:            ${gross_profit/len(wins):,.2f}" if wins else "")
    print(f"  Avg Loss:           ${gross_loss/len(losses):,.2f}" if losses else "")
    print()

    # Equity curve points
    print("=" * 80)
    print("      EVOLUTIE CONT")
    print("=" * 80)
    print()
    print(f"  {'Data':<12} | {'Directie':<6} | {'Lot':<5} | {'Risk $':<8} | {'P&L $':<10} | {'Balanta':<12} | Rezultat")
    print("  " + "-" * 85)

    balance = STARTING_BALANCE
    for t in trades:
        balance += t.pnl_dollars
        print(
            f"  {t.day:<12} | {t.direction:<6} | {t.lot_size:<5.2f} | "
            f"${t.risk_dollars:<7.2f} | ${t.pnl_dollars:<+9.2f} | ${balance:<11,.2f} | {t.result}"
        )

    print()
    print("=" * 80)
    print(f"  CONCLUZIE: {'PROFITABIL' if total_pnl > 0 else 'NEPROFITABIL'}")
    print(f"  Cu 0.5% risk pe trade, ai fi facut ${total_pnl:+,.2f} in ultima luna")
    print("=" * 80)


if __name__ == "__main__":
    main()
