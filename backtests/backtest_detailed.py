"""
SMC V3 Detailed Backtest with Monthly Statistics

Provides comprehensive analysis:
- Monthly breakdown of trades
- Per-symbol statistics
- Drawdown analysis
- Trade distribution by session
- Detailed trade log

Usage:
    # Simulated data (fast, for testing)
    python backtests/backtest_detailed.py
    python backtests/backtest_detailed.py --symbols EURUSD GBPUSD
    python backtests/backtest_detailed.py --months 12
    python backtests/backtest_detailed.py --export results.csv

    # REAL MT5 DATA (requires MT5 connection)
    python backtests/backtest_detailed.py --real
    python backtests/backtest_detailed.py --real --symbols US500 USTech100 GER40
    python backtests/backtest_detailed.py --real --months 3 --export real_results.csv
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random
import math
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flag to track if MT5 is available
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    pass

from models.candle import Candle
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType


@dataclass
class Trade:
    """Detailed trade record."""
    id: int
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    pnl_pips: float
    pnl_percent: float
    result: str
    exit_reason: str
    session: str
    rr_achieved: float
    holding_time_hours: float


@dataclass
class MonthlyStats:
    """Monthly trading statistics."""
    month: str
    year: int
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    london_trades: int = 0
    ny_trades: int = 0


@dataclass
class SymbolStats:
    """Per-symbol statistics."""
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class InstrumentConfig:
    """Instrument configuration."""
    symbol: str
    instrument_type: InstrumentType
    pip_size: float
    pip_value: float
    typical_spread: float
    atr_london: float
    atr_ny: float
    atr_asian: float
    min_sl: float
    max_sl: float
    sessions: List[str]


class MT5DataLoader:
    """Load real market data from MT5."""

    def __init__(self):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    def __del__(self):
        if MT5_AVAILABLE:
            mt5.shutdown()

    def get_candles(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> List[Candle]:
        """Download candles from MT5."""
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Enable symbol if not visible
        if not mt5.symbol_select(symbol, True):
            print(f"  [!] Failed to select symbol {symbol}")
            return []

        # Download data
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            print(f"  [!] No data for {symbol} {timeframe}")
            return []

        candles = []
        for r in rates:
            candle = Candle(
                timestamp=datetime.fromtimestamp(r['time']),
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close']),
                volume=int(r['tick_volume'])
            )
            candles.append(candle)

        return candles

    def load_all_timeframes(self, symbol: str, start_date: datetime, end_date: datetime) -> Tuple[List[Candle], List[Candle], List[Candle], List[Candle], List[Candle]]:
        """Load M5, M15, H1, H4, D1 candles for symbol."""
        print(f"  Loading M5 data...")
        m5 = self.get_candles(symbol, "M5", start_date, end_date)
        print(f"    -> {len(m5)} candles")

        print(f"  Loading M15 data...")
        m15 = self.get_candles(symbol, "M15", start_date, end_date)
        print(f"    -> {len(m15)} candles")

        print(f"  Loading H1 data...")
        h1 = self.get_candles(symbol, "H1", start_date, end_date)
        print(f"    -> {len(h1)} candles")

        print(f"  Loading H4 data...")
        h4 = self.get_candles(symbol, "H4", start_date, end_date)
        print(f"    -> {len(h4)} candles")

        print(f"  Loading D1 data...")
        d1 = self.get_candles(symbol, "D1", start_date, end_date)
        print(f"    -> {len(d1)} candles")

        return m5, m15, h1, h4, d1


INSTRUMENTS = {
    "EURUSD": InstrumentConfig(
        symbol="EURUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=0.8,
        atr_london=0.0012,
        atr_ny=0.0015,
        atr_asian=0.0006,
        min_sl=10,
        max_sl=25,
        sessions=["london", "ny"]
    ),
    "GBPUSD": InstrumentConfig(
        symbol="GBPUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=1.2,
        atr_london=0.0015,
        atr_ny=0.0018,
        atr_asian=0.0008,
        min_sl=12,
        max_sl=30,
        sessions=["london", "ny"]
    ),
    "AUDUSD": InstrumentConfig(
        symbol="AUDUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=1.0,
        atr_london=0.0010,
        atr_ny=0.0012,
        atr_asian=0.0008,
        min_sl=10,
        max_sl=25,
        sessions=["ny"]
    ),
    "US500": InstrumentConfig(
        symbol="US500",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=0.4,
        atr_london=8.0,
        atr_ny=15.0,
        atr_asian=4.0,
        min_sl=5,
        max_sl=20,
        sessions=["ny"]
    ),
    "USTech100": InstrumentConfig(
        symbol="USTech100",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=1.0,
        atr_london=25.0,
        atr_ny=50.0,
        atr_asian=15.0,
        min_sl=15,
        max_sl=50,
        sessions=["ny"]
    ),
    "GER40": InstrumentConfig(
        symbol="GER40",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=1.2,
        atr_london=30.0,
        atr_ny=25.0,
        atr_asian=10.0,
        min_sl=10,
        max_sl=40,
        sessions=["london", "ny"]
    ),
}


class RealisticMarketSimulator:
    """Generate realistic market data."""

    def __init__(self, config: InstrumentConfig, seed: int = 42):
        self.config = config
        random.seed(seed)
        self.trend_direction = 0
        self.trend_strength = 0.5
        self.volatility_mult = 1.0

    def _get_session(self, hour: int) -> str:
        if 0 <= hour < 7:
            return "asian"
        elif 7 <= hour < 12:
            return "london"
        elif 12 <= hour < 17:
            return "ny"
        else:
            return "asian"

    def _get_session_atr(self, session: str) -> float:
        if session == "london":
            return self.config.atr_london
        elif session == "ny":
            return self.config.atr_ny
        return self.config.atr_asian

    def generate_candles(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe_minutes: int = 5
    ) -> List[Candle]:
        candles = []

        if self.config.symbol == "EURUSD":
            price = 1.0850
        elif self.config.symbol == "GBPUSD":
            price = 1.2650
        elif self.config.symbol == "AUDUSD":
            price = 0.6550
        elif self.config.symbol == "US500":
            price = 4500.0
        elif self.config.symbol == "USTech100":
            price = 15500.0
        elif self.config.symbol == "GER40":
            price = 16500.0
        else:
            price = 1.0

        current = start_date

        while current < end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                current = current.replace(hour=0, minute=0)
                continue

            session = self._get_session(current.hour)
            base_atr = self._get_session_atr(session)
            candle_atr = base_atr * math.sqrt(timeframe_minutes / 60) * self.volatility_mult

            if random.random() < 0.01:
                self.trend_direction = random.choice([-1, 0, 1])
                self.trend_strength = random.uniform(0.3, 0.8)

            if random.random() < 0.005:
                self.volatility_mult = random.uniform(0.7, 1.5)

            trend_bias = self.trend_direction * self.trend_strength * candle_atr * 0.3
            noise = random.gauss(0, candle_atr * 0.5)

            change = trend_bias + noise
            open_price = price
            close_price = price + change

            if close_price > open_price:
                high = close_price + abs(random.gauss(0, candle_atr * 0.3))
                low = open_price - abs(random.gauss(0, candle_atr * 0.2))
            else:
                high = open_price + abs(random.gauss(0, candle_atr * 0.2))
                low = close_price - abs(random.gauss(0, candle_atr * 0.3))

            decimals = 5 if self.config.pip_size < 0.01 else 2
            candle = Candle(
                timestamp=current,
                open=round(open_price, decimals),
                high=round(high, decimals),
                low=round(low, decimals),
                close=round(close_price, decimals),
                volume=random.randint(100, 10000)
            )
            candles.append(candle)

            price = close_price
            current += timedelta(minutes=timeframe_minutes)

        return candles

    def resample_candles(self, candles: List[Candle], target_minutes: int) -> List[Candle]:
        if not candles:
            return []

        resampled = []
        current_group = []

        for candle in candles:
            if not current_group:
                current_group = [candle]
            elif (candle.timestamp.hour * 60 + candle.timestamp.minute) // target_minutes == \
                 (current_group[0].timestamp.hour * 60 + current_group[0].timestamp.minute) // target_minutes and \
                 candle.timestamp.date() == current_group[0].timestamp.date():
                current_group.append(candle)
            else:
                if current_group:
                    resampled.append(Candle(
                        timestamp=current_group[0].timestamp,
                        open=current_group[0].open,
                        high=max(c.high for c in current_group),
                        low=min(c.low for c in current_group),
                        close=current_group[-1].close,
                        volume=sum(c.volume for c in current_group)
                    ))
                current_group = [candle]

        if current_group:
            resampled.append(Candle(
                timestamp=current_group[0].timestamp,
                open=current_group[0].open,
                high=max(c.high for c in current_group),
                low=min(c.low for c in current_group),
                close=current_group[-1].close,
                volume=sum(c.volume for c in current_group)
            ))

        return resampled


class DetailedBacktestEngine:
    """Backtest engine with detailed statistics."""

    def __init__(
        self,
        instrument_config: InstrumentConfig,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.5,
    ):
        self.config = instrument_config
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades: List[Trade] = []
        self.trade_counter = 0

        # Create strategy config
        if instrument_config.instrument_type == InstrumentType.INDEX:
            strategy_config = SMCConfigV3(
                instrument_type=instrument_config.instrument_type,
                min_sl_pips=instrument_config.min_sl,
                max_sl_pips=instrument_config.max_sl,
                use_strict_kill_zones=False,
                require_displacement=True,
                displacement_min_atr=1.0,
                poi_min_score=1.5,
                require_sweep_for_low_score=False,
                use_adx_filter=True,
                adx_trending=22.0,
                adx_weak_trend=15.0,
                use_volatility_filter=False,
                use_partial_tp=True,
                min_rr=1.5,
                require_multi_candle=False,
                ny_start_hour=13,
                ny_end_hour=20,
            )
        else:
            strategy_config = SMCConfigV3(
                instrument_type=instrument_config.instrument_type,
                min_sl_pips=instrument_config.min_sl,
                max_sl_pips=instrument_config.max_sl,
                use_strict_kill_zones=True,
                require_displacement=True,
                displacement_min_atr=1.2,
                poi_min_score=1.8,
                require_sweep_for_low_score=True,
                sweep_score_threshold=2.5,
                use_adx_filter=True,
                adx_trending=25.0,
                adx_weak_trend=18.0,
                use_partial_tp=True,
            )

        self.strategy = SMCStrategyV3(
            symbol=instrument_config.symbol,
            config=strategy_config
        )

    def run_backtest(
        self,
        m5_candles: List[Candle],
        m15_candles: List[Candle],
        h1_candles: List[Candle],
        h4_candles: List[Candle],
        daily_candles: List[Candle]
    ) -> List[Trade]:
        """Run backtest and return all trades."""
        self.strategy.initialize()
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades = []

        position = None
        position_entry_time = None
        position_direction = None
        position_entry_price = 0
        position_sl = 0
        position_tp = 0
        position_session = ""

        for i, candle in enumerate(m5_candles):
            if i < 100:
                continue

            current_m5 = m5_candles[:i+1]
            current_m15 = [c for c in m15_candles if c.timestamp <= candle.timestamp]
            current_h1 = [c for c in h1_candles if c.timestamp <= candle.timestamp]
            current_h4 = [c for c in h4_candles if c.timestamp <= candle.timestamp]
            current_daily = [c for c in daily_candles if c.timestamp.date() < candle.timestamp.date()]

            if len(current_h1) < 50 or len(current_h4) < 20:
                continue

            self.strategy.set_candles(
                h4_candles=current_h4[-100:],
                h1_candles=current_h1[-200:],
                m5_candles=current_m5[-100:],
                m15_candles=current_m15[-100:] if current_m15 else None,
                daily_candles=current_daily[-30:] if current_daily else None
            )

            self.strategy.update_spread(self.config.typical_spread * self.config.pip_size)

            # Check existing position
            if position:
                current_price = candle.close

                if position_direction == "long":
                    if candle.low <= position_sl:
                        self._close_trade(
                            candle.timestamp, position_sl, position_entry_time,
                            position_direction, position_entry_price,
                            position_sl, position_tp, "sl_hit", position_session
                        )
                        position = None
                        continue

                    if candle.high >= position_tp:
                        self._close_trade(
                            candle.timestamp, position_tp, position_entry_time,
                            position_direction, position_entry_price,
                            position_sl, position_tp, "tp_hit", position_session
                        )
                        position = None
                        continue
                else:
                    if candle.high >= position_sl:
                        self._close_trade(
                            candle.timestamp, position_sl, position_entry_time,
                            position_direction, position_entry_price,
                            position_sl, position_tp, "sl_hit", position_session
                        )
                        position = None
                        continue

                    if candle.low <= position_tp:
                        self._close_trade(
                            candle.timestamp, position_tp, position_entry_time,
                            position_direction, position_entry_price,
                            position_sl, position_tp, "tp_hit", position_session
                        )
                        position = None
                        continue

            # Look for new signals
            if not position:
                signal = self.strategy.on_candle(candle, current_m5[-50:])

                if signal and signal.stop_loss and signal.take_profit:
                    position = True
                    position_entry_time = candle.timestamp
                    position_direction = "long" if "LONG" in str(signal.signal_type) else "short"
                    position_entry_price = signal.price
                    position_sl = signal.stop_loss
                    position_tp = signal.take_profit
                    position_session = signal.metadata.get("session", "unknown")

            self.equity_curve.append(self.balance)

        return self.trades

    def _close_trade(
        self,
        exit_time: datetime,
        exit_price: float,
        entry_time: datetime,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        exit_reason: str,
        session: str
    ):
        """Record closed trade with full details."""
        self.trade_counter += 1

        if direction == "long":
            pnl_pips = (exit_price - entry_price) / self.config.pip_size
            risk_pips = (entry_price - sl) / self.config.pip_size
        else:
            pnl_pips = (entry_price - exit_price) / self.config.pip_size
            risk_pips = (sl - entry_price) / self.config.pip_size

        # Calculate P&L based on risk
        risk_amount = self.balance * (self.risk_per_trade / 100)
        if risk_pips != 0:
            pnl = (pnl_pips / abs(risk_pips)) * risk_amount
        else:
            pnl = 0

        pnl_percent = (pnl / self.balance) * 100
        rr_achieved = pnl_pips / abs(risk_pips) if risk_pips != 0 else 0

        self.balance += pnl
        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "be")

        holding_time = (exit_time - entry_time).total_seconds() / 3600

        self.trades.append(Trade(
            id=self.trade_counter,
            symbol=self.config.symbol,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            pnl=pnl,
            pnl_pips=pnl_pips,
            pnl_percent=pnl_percent,
            result=result,
            exit_reason=exit_reason,
            session=session,
            rr_achieved=rr_achieved,
            holding_time_hours=holding_time
        ))


def calculate_monthly_stats(trades: List[Trade], initial_balance: float) -> List[MonthlyStats]:
    """Calculate monthly statistics from trades."""
    monthly_data = defaultdict(list)

    for trade in trades:
        key = (trade.entry_time.year, trade.entry_time.month)
        monthly_data[key].append(trade)

    monthly_stats = []
    month_names = ["", "Ian", "Feb", "Mar", "Apr", "Mai", "Iun",
                   "Iul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for (year, month), month_trades in sorted(monthly_data.items()):
        wins = [t for t in month_trades if t.result == "win"]
        losses = [t for t in month_trades if t.result == "loss"]

        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))

        london_trades = len([t for t in month_trades if t.session == "london"])
        ny_trades = len([t for t in month_trades if t.session == "ny"])

        # Calculate max drawdown for month
        running_balance = initial_balance
        peak = running_balance
        max_dd = 0
        for t in sorted(month_trades, key=lambda x: x.entry_time):
            running_balance += t.pnl
            if running_balance > peak:
                peak = running_balance
            dd = (peak - running_balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        stats = MonthlyStats(
            month=month_names[month],
            year=year,
            trades=len(month_trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(month_trades) * 100 if month_trades else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0,
            total_pnl=sum(t.pnl for t in month_trades),
            total_pnl_percent=sum(t.pnl_percent for t in month_trades),
            max_drawdown=max_dd,
            avg_win=total_wins / len(wins) if wins else 0,
            avg_loss=total_losses / len(losses) if losses else 0,
            avg_rr=sum(t.rr_achieved for t in month_trades) / len(month_trades) if month_trades else 0,
            best_trade=max(t.pnl for t in month_trades) if month_trades else 0,
            worst_trade=min(t.pnl for t in month_trades) if month_trades else 0,
            london_trades=london_trades,
            ny_trades=ny_trades
        )
        monthly_stats.append(stats)

    return monthly_stats


def calculate_symbol_stats(all_trades: Dict[str, List[Trade]], initial_balance: float) -> List[SymbolStats]:
    """Calculate per-symbol statistics."""
    symbol_stats = []

    for symbol, trades in all_trades.items():
        if not trades:
            continue

        wins = [t for t in trades if t.result == "win"]
        losses = [t for t in trades if t.result == "loss"]
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))

        # Max drawdown
        running = initial_balance
        peak = running
        max_dd = 0
        for t in sorted(trades, key=lambda x: x.entry_time):
            running += t.pnl
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100
            if dd > max_dd:
                max_dd = dd

        stats = SymbolStats(
            symbol=symbol,
            trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(trades) * 100 if trades else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0,
            total_pnl=sum(t.pnl for t in trades),
            avg_pnl=sum(t.pnl for t in trades) / len(trades) if trades else 0,
            max_drawdown=max_dd
        )
        symbol_stats.append(stats)

    return sorted(symbol_stats, key=lambda x: x.total_pnl, reverse=True)


def print_detailed_report(
    all_trades: Dict[str, List[Trade]],
    monthly_stats: List[MonthlyStats],
    symbol_stats: List[SymbolStats],
    initial_balance: float,
    months: int
):
    """Print comprehensive backtest report."""

    all_trades_flat = []
    for trades in all_trades.values():
        all_trades_flat.extend(trades)
    all_trades_flat.sort(key=lambda x: x.entry_time)

    total_trades = len(all_trades_flat)
    total_wins = len([t for t in all_trades_flat if t.result == "win"])
    total_losses = len([t for t in all_trades_flat if t.result == "loss"])
    total_pnl = sum(t.pnl for t in all_trades_flat)
    total_win_amount = sum(t.pnl for t in all_trades_flat if t.pnl > 0)
    total_loss_amount = abs(sum(t.pnl for t in all_trades_flat if t.pnl < 0))

    print("\n" + "=" * 80)
    print("                    SMC V3 DETAILED BACKTEST REPORT")
    print("=" * 80)
    print(f"Period: {months} months | Initial Balance: ${initial_balance:,.2f}")
    print("=" * 80)

    # Overall Summary
    print("\n" + "-" * 40)
    print("           OVERALL SUMMARY")
    print("-" * 40)
    print(f"Total Trades:      {total_trades}")
    print(f"Wins:              {total_wins} ({total_wins/total_trades*100:.1f}%)" if total_trades else "Wins: 0")
    print(f"Losses:            {total_losses} ({total_losses/total_trades*100:.1f}%)" if total_trades else "Losses: 0")
    print(f"Profit Factor:     {total_win_amount/total_loss_amount:.2f}" if total_loss_amount > 0 else "Profit Factor: N/A")
    print(f"Total P&L:         ${total_pnl:+,.2f}")
    print(f"Return:            {total_pnl/initial_balance*100:+.2f}%")
    print(f"Avg Trade:         ${total_pnl/total_trades:+,.2f}" if total_trades else "Avg Trade: N/A")

    # Symbol Breakdown
    print("\n" + "-" * 40)
    print("           PER SYMBOL BREAKDOWN")
    print("-" * 40)
    print(f"{'Symbol':<10} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    print("-" * 60)
    for s in symbol_stats:
        pf_str = f"{s.profit_factor:.2f}" if s.profit_factor != float('inf') else "INF"
        print(f"{s.symbol:<10} {s.trades:<8} {s.win_rate:<8.1f} {pf_str:<8} ${s.total_pnl:<+11,.2f} {s.max_drawdown:.1f}%")

    # Monthly Breakdown
    print("\n" + "-" * 40)
    print("           MONTHLY BREAKDOWN")
    print("-" * 40)
    print(f"{'Month':<10} {'Trades':<8} {'W/L':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'DD%':<8}")
    print("-" * 70)
    for m in monthly_stats:
        pf_str = f"{m.profit_factor:.2f}" if m.profit_factor != float('inf') else "INF"
        print(f"{m.month} {m.year:<6} {m.trades:<8} {m.wins}/{m.losses:<5} {m.win_rate:<8.1f} {pf_str:<8} ${m.total_pnl:<+11,.2f} {m.max_drawdown:.1f}%")

    # Session Analysis
    print("\n" + "-" * 40)
    print("           SESSION ANALYSIS")
    print("-" * 40)
    london_total = sum(m.london_trades for m in monthly_stats)
    ny_total = sum(m.ny_trades for m in monthly_stats)
    print(f"London Session: {london_total} trades ({london_total/total_trades*100:.1f}%)" if total_trades else "London: 0")
    print(f"NY Session:     {ny_total} trades ({ny_total/total_trades*100:.1f}%)" if total_trades else "NY: 0")

    # Trade Distribution
    if all_trades_flat:
        print("\n" + "-" * 40)
        print("           TRADE DISTRIBUTION")
        print("-" * 40)
        avg_holding = sum(t.holding_time_hours for t in all_trades_flat) / len(all_trades_flat)
        avg_rr = sum(t.rr_achieved for t in all_trades_flat) / len(all_trades_flat)
        best = max(all_trades_flat, key=lambda x: x.pnl)
        worst = min(all_trades_flat, key=lambda x: x.pnl)

        print(f"Avg Holding Time: {avg_holding:.1f} hours")
        print(f"Avg R:R Achieved: {avg_rr:.2f}")
        print(f"Best Trade:       ${best.pnl:+,.2f} ({best.symbol}, {best.entry_time.strftime('%Y-%m-%d')})")
        print(f"Worst Trade:      ${worst.pnl:+,.2f} ({worst.symbol}, {worst.entry_time.strftime('%Y-%m-%d')})")

    # Recent Trades
    if all_trades_flat:
        print("\n" + "-" * 40)
        print("           LAST 10 TRADES")
        print("-" * 40)
        print(f"{'Date':<12} {'Symbol':<8} {'Dir':<6} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Result':<8}")
        print("-" * 70)
        for t in all_trades_flat[-10:]:
            print(f"{t.entry_time.strftime('%Y-%m-%d'):<12} {t.symbol:<8} {t.direction:<6} "
                  f"{t.entry_price:<10.5f} {t.exit_price:<10.5f} ${t.pnl:<+9.2f} {t.result.upper():<8}")

    print("\n" + "=" * 80)


def export_to_csv(all_trades: Dict[str, List[Trade]], filename: str):
    """Export all trades to CSV file."""
    all_trades_flat = []
    for trades in all_trades.values():
        all_trades_flat.extend(trades)
    all_trades_flat.sort(key=lambda x: x.entry_time)

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'ID', 'Symbol', 'Entry Time', 'Exit Time', 'Direction',
            'Entry Price', 'Exit Price', 'SL', 'TP', 'P&L', 'P&L Pips',
            'P&L %', 'Result', 'Exit Reason', 'Session', 'R:R', 'Holding Hours'
        ])
        for t in all_trades_flat:
            writer.writerow([
                t.id, t.symbol, t.entry_time, t.exit_time, t.direction,
                t.entry_price, t.exit_price, t.sl, t.tp, t.pnl, t.pnl_pips,
                t.pnl_percent, t.result, t.exit_reason, t.session,
                t.rr_achieved, t.holding_time_hours
            ])

    print(f"\nTrades exported to: {filename}")


def run_detailed_backtest(
    symbols: List[str] = None,
    months: int = 6,
    initial_balance: float = 10000.0,
    risk_percent: float = 0.5,
    export_file: str = None,
    use_real_data: bool = False
):
    """Run detailed backtest on multiple symbols."""

    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "AUDUSD", "US500", "USTech100", "GER40"]

    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=months * 30)

    data_source = "MT5 REAL DATA" if use_real_data else "SIMULATED DATA"

    print("=" * 80)
    print("SMC V3 DETAILED BACKTEST")
    print("=" * 80)
    print(f"Data Source: {data_source}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_percent}%")
    print(f"Symbols: {', '.join(symbols)}")
    print("=" * 80)

    # Initialize MT5 data loader if using real data
    mt5_loader = None
    if use_real_data:
        if not MT5_AVAILABLE:
            print("[!] ERROR: MetaTrader5 package not installed!")
            print("[!] Install with: pip install MetaTrader5")
            print("[!] Falling back to simulated data...")
            use_real_data = False
        else:
            try:
                mt5_loader = MT5DataLoader()
                print("[OK] Connected to MT5 for real data")
            except Exception as e:
                print(f"[!] Failed to connect to MT5: {e}")
                print("[!] Falling back to simulated data...")
                use_real_data = False

    all_trades: Dict[str, List[Trade]] = {}

    for symbol in symbols:
        if symbol not in INSTRUMENTS:
            print(f"[!] Unknown symbol: {symbol}, skipping...")
            continue

        config = INSTRUMENTS[symbol]
        print(f"\nProcessing: {symbol}...")

        if use_real_data and mt5_loader:
            # Use real MT5 data
            m5_candles, m15_candles, h1_candles, h4_candles, daily_candles = \
                mt5_loader.load_all_timeframes(symbol, start_date, end_date)

            if not m5_candles:
                print(f"  [!] No data for {symbol}, skipping...")
                continue
        else:
            # Use simulated data
            simulator = RealisticMarketSimulator(config, seed=hash(symbol) % 10000)
            m5_candles = simulator.generate_candles(start_date, end_date, 5)
            m15_candles = simulator.resample_candles(m5_candles, 15)
            h1_candles = simulator.resample_candles(m5_candles, 60)
            h4_candles = simulator.resample_candles(m5_candles, 240)
            daily_candles = simulator.resample_candles(m5_candles, 1440)

        engine = DetailedBacktestEngine(config, initial_balance, risk_percent)
        trades = engine.run_backtest(m5_candles, m15_candles, h1_candles, h4_candles, daily_candles)
        all_trades[symbol] = trades

        print(f"  {symbol}: {len(trades)} trades")

    # Calculate statistics
    all_trades_combined = []
    for trades in all_trades.values():
        all_trades_combined.extend(trades)

    monthly_stats = calculate_monthly_stats(all_trades_combined, initial_balance)
    symbol_stats = calculate_symbol_stats(all_trades, initial_balance)

    # Print report
    print_detailed_report(all_trades, monthly_stats, symbol_stats, initial_balance, months)

    # Export if requested
    if export_file:
        export_to_csv(all_trades, export_file)

    return all_trades, monthly_stats, symbol_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMC V3 Detailed Backtest")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="Symbols to test")
    parser.add_argument("--months", type=int, default=6,
                       help="Number of months (default: 6)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent per trade (default: 0.5)")
    parser.add_argument("--export", type=str, default=None,
                       help="Export trades to CSV file")
    parser.add_argument("--real", action="store_true",
                       help="Use real MT5 data instead of simulated (requires MT5 connection)")

    args = parser.parse_args()

    run_detailed_backtest(
        symbols=args.symbols,
        months=args.months,
        initial_balance=args.balance,
        risk_percent=args.risk,
        export_file=args.export,
        use_real_data=args.real
    )
