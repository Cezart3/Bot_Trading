"""
Backtest ORB Strategy on EUR/USD - Multiple Sessions
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
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pips: float = 0.0
    result: str = ""  # 'WIN', 'LOSS', 'OPEN'

@dataclass
class SessionConfig:
    name: str
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int
    orb_minutes: int = 5  # Opening Range duration

def run_backtest(rates, session: SessionConfig, risk_reward: float = 2.0, sl_buffer_pips: float = 3.0) -> List[Trade]:
    """Run ORB backtest on given session."""
    trades = []
    pip_size = 0.0001  # For EUR/USD

    # Group candles by day
    days = {}
    for rate in rates:
        ts = datetime.fromtimestamp(rate['time'])
        day = ts.date()
        if day not in days:
            days[day] = []
        days[day].append(rate)

    for day, day_rates in days.items():
        # Skip weekends
        if day.weekday() >= 5:
            continue

        # Find opening range candle
        orb_candle = None
        orb_time = time(session.open_hour, session.open_minute)

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate['time'])
            if ts.time() == orb_time:
                orb_candle = rate
                break

        if orb_candle is None:
            continue

        orb_high = orb_candle['high']
        orb_low = orb_candle['low']
        orb_range = orb_high - orb_low

        # Skip if range is too small (less than 5 pips) or too large (more than 50 pips)
        if orb_range < 5 * pip_size or orb_range > 50 * pip_size:
            continue

        # Look for breakout in subsequent candles
        session_end = time(session.close_hour, session.close_minute)
        trade_taken = False
        current_trade = None

        for rate in day_rates:
            ts = datetime.fromtimestamp(rate['time'])

            # Only trade during session hours
            if ts.time() <= orb_time or ts.time() > session_end:
                continue

            # If we have an open trade, check for exit
            if current_trade and current_trade.result == "":
                # Check stop loss
                if current_trade.direction == 'LONG':
                    if rate['low'] <= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                        current_trade.result = 'LOSS'
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate['high'] >= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
                        current_trade.result = 'WIN'
                        trades.append(current_trade)
                        current_trade = None
                        continue
                else:  # SHORT
                    if rate['high'] >= current_trade.stop_loss:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.stop_loss
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                        current_trade.result = 'LOSS'
                        trades.append(current_trade)
                        current_trade = None
                        continue
                    elif rate['low'] <= current_trade.take_profit:
                        current_trade.exit_time = ts
                        current_trade.exit_price = current_trade.take_profit
                        current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
                        current_trade.result = 'WIN'
                        trades.append(current_trade)
                        current_trade = None
                        continue

            # Look for new trade entry (only if no trade taken today)
            if not trade_taken and current_trade is None:
                # LONG breakout
                if rate['close'] > orb_high:
                    entry_price = rate['close']
                    sl = orb_low - sl_buffer_pips * pip_size
                    risk = entry_price - sl
                    tp = entry_price + risk * risk_reward

                    current_trade = Trade(
                        entry_time=ts,
                        entry_price=entry_price,
                        direction='LONG',
                        stop_loss=sl,
                        take_profit=tp
                    )
                    trade_taken = True

                # SHORT breakout
                elif rate['close'] < orb_low:
                    entry_price = rate['close']
                    sl = orb_high + sl_buffer_pips * pip_size
                    risk = sl - entry_price
                    tp = entry_price - risk * risk_reward

                    current_trade = Trade(
                        entry_time=ts,
                        entry_price=entry_price,
                        direction='SHORT',
                        stop_loss=sl,
                        take_profit=tp
                    )
                    trade_taken = True

        # Close any open trade at session end
        if current_trade and current_trade.result == "":
            last_rate = day_rates[-1]
            current_trade.exit_time = datetime.fromtimestamp(last_rate['time'])
            current_trade.exit_price = last_rate['close']
            if current_trade.direction == 'LONG':
                current_trade.pnl_pips = (current_trade.exit_price - current_trade.entry_price) / pip_size
            else:
                current_trade.pnl_pips = (current_trade.entry_price - current_trade.exit_price) / pip_size
            current_trade.result = 'WIN' if current_trade.pnl_pips > 0 else 'LOSS'
            trades.append(current_trade)

    return trades

def analyze_trades(trades: List[Trade], session_name: str) -> dict:
    """Analyze trade results."""
    if not trades:
        return {'session': session_name, 'trades': 0}

    wins = [t for t in trades if t.result == 'WIN']
    losses = [t for t in trades if t.result == 'LOSS']

    total_pips = sum(t.pnl_pips for t in trades)
    win_pips = sum(t.pnl_pips for t in wins) if wins else 0
    loss_pips = abs(sum(t.pnl_pips for t in losses)) if losses else 0

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    profit_factor = win_pips / loss_pips if loss_pips > 0 else float('inf')

    # Calculate max drawdown
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

    return {
        'session': session_name,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pips': total_pips,
        'profit_factor': profit_factor,
        'max_drawdown_pips': max_dd,
        'avg_win_pips': win_pips / len(wins) if wins else 0,
        'avg_loss_pips': loss_pips / len(losses) if losses else 0,
    }


if __name__ == "__main__":
    # Connect and get data
    s = get_settings()
    mt5.initialize(login=s.mt5.login, password=s.mt5.password, server=s.mt5.server, path=s.mt5.path, timeout=60000)

    symbol = 'EURUSD'
    mt5.symbol_select(symbol, True)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=35)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_time, end_time)

    mt5.shutdown()

    if rates is None:
        print("Nu am putut obtine date!")
        exit()

    print('=' * 80)
    print('         BACKTEST ORB EUR/USD - ULTIMA LUNA')
    print('         Risk/Reward: 2:1 | SL Buffer: 3 pips')
    print('=' * 80)
    print()

    # Define sessions (Romania time = Server time)
    sessions = [
        SessionConfig("LONDRA", 10, 0, 18, 0),       # 10:00 - 18:00 Romania
        SessionConfig("NEW YORK", 15, 0, 23, 0),     # 15:00 - 23:00 Romania
        SessionConfig("OVERLAP", 15, 0, 18, 0),      # 15:00 - 18:00 Romania
        SessionConfig("LONDRA EARLY", 10, 0, 14, 0), # 10:00 - 14:00 (before NY)
    ]

    results = []

    for session in sessions:
        trades = run_backtest(rates, session, risk_reward=2.0, sl_buffer_pips=3.0)
        stats = analyze_trades(trades, session.name)
        results.append(stats)

        print(f'SESIUNEA {session.name} ({session.open_hour}:00 - {session.close_hour}:00)')
        print('-' * 60)
        print(f'  Total tranzactii: {stats["trades"]}')
        if stats['trades'] > 0:
            print(f'  Castiguri/Pierderi: {stats["wins"]}/{stats["losses"]}')
            print(f'  Win Rate: {stats["win_rate"]:.1f}%')
            print(f'  Total Pips: {stats["total_pips"]:.1f}')
            print(f'  Profit Factor: {stats["profit_factor"]:.2f}')
            print(f'  Max Drawdown: {stats["max_drawdown_pips"]:.1f} pips')
            print(f'  Avg Win: {stats["avg_win_pips"]:.1f} pips')
            print(f'  Avg Loss: {stats["avg_loss_pips"]:.1f} pips')
        print()

    # Find best session
    print('=' * 80)
    print('                          CONCLUZIE')
    print('=' * 80)
    print()

    valid_results = [r for r in results if r['trades'] > 0 and r['profit_factor'] != float('inf')]
    if valid_results:
        # Sort by profit factor
        best_pf = max(valid_results, key=lambda x: x['profit_factor'])
        # Sort by total pips
        best_pips = max(valid_results, key=lambda x: x['total_pips'])

        print(f'Cel mai bun Profit Factor: {best_pf["session"]} (PF: {best_pf["profit_factor"]:.2f})')
        print(f'Cel mai profitabil (pips): {best_pips["session"]} ({best_pips["total_pips"]:.1f} pips)')
        print()

        # Recommendation
        if best_pf['session'] == best_pips['session']:
            print(f'>>> RECOMANDARE: {best_pf["session"]} <<<')
        else:
            # Choose based on consistency (profit factor) if they differ
            print(f'>>> RECOMANDARE: {best_pf["session"]} (mai consistent) <<<')
            print(f'    Alternativ: {best_pips["session"]} (mai profitabil dar riscant)')
