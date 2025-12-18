"""
Multi-Instrument Backtest for SMC V3 Strategy.

IMPORTANT: Uses the SAME configuration as run_smc_v3.py to ensure consistency!
No duplicated strategy logic - imports directly from the live runner.

Tests SMC V3 on multiple instruments with per-symbol optimized settings:
- GBPUSD (Priority 1 - BEST performer, relaxed settings)
- AUDUSD (Priority 2 - decent performer, moderate settings)
- EURUSD (Priority 3 - STRICT settings due to losses)
- US30 (Priority 5 - VERY STRICT settings)
- USTEC/NAS100 (Priority 6)
- GER40/GER30 (Priority 7)

Author: Trading Bot Project
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.candle import Candle
from models.position import Position, PositionStatus
from strategies.smc_strategy_v3 import SMCStrategyV3, SMCConfigV3, InstrumentType

# Import configurations from run_smc_v3.py to stay in sync!
from scripts.run_smc_v3 import SYMBOL_CONFIGS, SYMBOL_ALIASES


@dataclass
class InstrumentConfig:
    """Configuration for each instrument's market characteristics."""
    symbol: str
    instrument_type: InstrumentType
    pip_size: float
    pip_value: float  # Value per pip/point per lot
    typical_spread: float  # In pips/points
    atr_london: float  # Typical ATR in London session
    atr_ny: float  # Typical ATR in NY session
    atr_asian: float  # Typical ATR in Asian session
    sessions: List[str]  # Which sessions to trade


# Market characteristics for each instrument (separate from strategy config)
MARKET_DATA = {
    # === FOREX PAIRS ===
    "EURUSD": InstrumentConfig(
        symbol="EURUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=0.8,
        atr_london=0.0012,
        atr_ny=0.0015,
        atr_asian=0.0006,
        sessions=["london", "ny"]
    ),
    "GBPUSD": InstrumentConfig(
        symbol="GBPUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=1.0,
        atr_london=0.0015,
        atr_ny=0.0018,
        atr_asian=0.0008,
        sessions=["london", "ny"]
    ),
    "AUDUSD": InstrumentConfig(
        symbol="AUDUSD",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=0.9,
        atr_london=0.0010,
        atr_ny=0.0012,
        atr_asian=0.0008,
        sessions=["ny"]
    ),
    "EURGBP": InstrumentConfig(
        symbol="EURGBP",
        instrument_type=InstrumentType.FOREX,
        pip_size=0.0001,
        pip_value=10.0,
        typical_spread=1.2,
        atr_london=0.0010,
        atr_ny=0.0008,
        atr_asian=0.0005,
        sessions=["london"]
    ),
    # === INDICES ===
    "US30": InstrumentConfig(
        symbol="US30",
        instrument_type=InstrumentType.INDEX,
        pip_size=1.0,
        pip_value=1.0,
        typical_spread=2.0,
        atr_london=80.0,
        atr_ny=150.0,
        atr_asian=40.0,
        sessions=["ny"]
    ),
    "USTEC": InstrumentConfig(
        symbol="USTEC",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=1.0,
        atr_london=25.0,
        atr_ny=50.0,
        atr_asian=15.0,
        sessions=["ny"]
    ),
    "NAS100": InstrumentConfig(
        symbol="NAS100",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=1.0,
        atr_london=25.0,
        atr_ny=50.0,
        atr_asian=15.0,
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
        sessions=["london", "ny"]
    ),
    "GER30": InstrumentConfig(
        symbol="GER30",
        instrument_type=InstrumentType.INDEX,
        pip_size=0.1,
        pip_value=1.0,
        typical_spread=1.2,
        atr_london=30.0,
        atr_ny=25.0,
        atr_asian=10.0,
        sessions=["london", "ny"]
    ),
}


class RealisticMarketSimulator:
    """Generates realistic market data for backtesting."""

    def __init__(self, config: InstrumentConfig, seed: int = 42):
        self.config = config
        random.seed(seed)
        self.trend_direction = 0
        self.trend_strength = 0.5
        self.volatility_mult = 1.0

    def _get_session(self, hour: int) -> str:
        """Get current session based on hour (UTC)."""
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 12:
            return "london"
        elif 14 <= hour < 20:
            return "ny"
        else:
            return "asian"

    def _get_session_atr(self, session: str) -> float:
        """Get ATR for session."""
        if session == "london":
            return self.config.atr_london
        elif session == "ny":
            return self.config.atr_ny
        else:
            return self.config.atr_asian

    def generate_candles(
        self,
        start_date: datetime,
        end_date: datetime,
        timeframe_minutes: int = 5
    ) -> List[Candle]:
        """Generate realistic OHLC candles."""
        candles = []

        # Starting price based on instrument
        starting_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "AUDUSD": 0.6550,
            "EURGBP": 0.8550,
            "US30": 43500.0,
            "USTEC": 21500.0,
            "NAS100": 21500.0,
            "GER40": 20500.0,
            "GER30": 20500.0,
        }
        price = starting_prices.get(self.config.symbol, 1.0)

        current = start_date

        while current < end_date:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                current = current.replace(hour=0, minute=0)
                continue

            session = self._get_session(current.hour)
            base_atr = self._get_session_atr(session)

            # Scale ATR for timeframe
            candle_atr = base_atr * math.sqrt(timeframe_minutes / 60) * self.volatility_mult

            # Random trend changes
            if random.random() < 0.01:
                self.trend_direction = random.choice([-1, 0, 1])
                self.trend_strength = random.uniform(0.3, 0.8)

            # Volatility changes
            if random.random() < 0.005:
                self.volatility_mult = random.uniform(0.7, 1.5)

            # Generate candle
            trend_bias = self.trend_direction * self.trend_strength * candle_atr * 0.3
            noise = random.gauss(0, candle_atr * 0.5)

            change = trend_bias + noise
            open_price = price
            close_price = price + change

            # High and low with realistic wicks
            if close_price > open_price:
                high = close_price + abs(random.gauss(0, candle_atr * 0.3))
                low = open_price - abs(random.gauss(0, candle_atr * 0.2))
            else:
                high = open_price + abs(random.gauss(0, candle_atr * 0.2))
                low = close_price - abs(random.gauss(0, candle_atr * 0.3))

            # Create candle
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

    def resample_candles(
        self,
        candles: List[Candle],
        target_minutes: int
    ) -> List[Candle]:
        """Resample candles to higher timeframe."""
        if not candles:
            return []

        resampled = []
        current_group = []

        for candle in candles:
            minutes_in_day = candle.timestamp.hour * 60 + candle.timestamp.minute

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


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    pnl: float
    pnl_pips: float
    result: str
    exit_reason: str
    session: str


@dataclass
class BacktestResult:
    """Results from backtesting one instrument."""
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    total_pnl_pips: float
    avg_win: float
    avg_loss: float
    avg_rr: float
    max_drawdown: float
    max_consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)


def create_strategy_for_symbol(symbol: str, risk_percent: float = 0.5) -> SMCStrategyV3:
    """
    Create SMC V3 strategy for a symbol using SAME logic as run_smc_v3.py.
    This ensures backtest matches live trading exactly.
    """
    # Get config from run_smc_v3.py's SYMBOL_CONFIGS
    config_data = SYMBOL_CONFIGS.get(symbol, SYMBOL_CONFIGS.get("GBPUSD", {}))

    # Get per-symbol settings with defaults
    poi_min_score = config_data.get("poi_min_score", 2.0)
    require_sweep = config_data.get("require_sweep", False)
    adx_trending = config_data.get("adx_trending", 22.0)
    min_rr = config_data.get("min_rr", 1.5)
    min_sl_pips = config_data.get("min_sl_pips", 10)
    max_sl_pips = config_data.get("max_sl_pips", 30)
    instrument_type = config_data.get("type", InstrumentType.FOREX)

    # Base config
    base_config = {
        "instrument_type": instrument_type,
        "min_sl_pips": min_sl_pips,
        "max_sl_pips": max_sl_pips,
        "risk_percent": risk_percent,
        "max_trades_per_day": 4,

        # Partial TPs
        "use_partial_tp": True,
        "tp1_rr": 1.0,
        "tp1_percent": 50,
        "tp2_rr": 2.0,
        "tp2_percent": 30,
        "tp3_rr": 3.0,
        "tp3_percent": 20,

        # Time exit
        "use_time_exit": True,
        "time_exit_hours": 4,

        # Equity curve trading
        "use_equity_curve": True,
        "equity_losses_reduce": 2,
        "equity_reduced_risk": 0.25,

        # Per-symbol optimized settings
        "poi_min_score": poi_min_score,
        "require_sweep_for_low_score": require_sweep,
        "adx_trending": adx_trending,
        "min_rr": min_rr,
    }

    # Adjust based on instrument type
    if instrument_type == InstrumentType.INDEX:
        base_config.update({
            "use_strict_kill_zones": False,
            "require_displacement": True,
            "displacement_min_atr": 1.2,
            "use_adx_filter": True,
            "adx_weak_trend": adx_trending - 5,
            "use_volatility_filter": False,
            "require_multi_candle": False,
            "ny_start_hour": 13,
            "ny_end_hour": 20,
            "index_skip_first_minutes": 15,
            "sweep_score_threshold": poi_min_score if require_sweep else 999,
        })
    else:
        base_config.update({
            "use_strict_kill_zones": True,
            "require_displacement": True,
            "displacement_min_atr": 1.2,
            "use_adx_filter": True,
            "adx_weak_trend": adx_trending - 5,
            "sweep_score_threshold": poi_min_score if require_sweep else 999,
        })

    config = SMCConfigV3(**base_config)

    return SMCStrategyV3(
        symbol=symbol,
        timeframe="M5",
        magic_number=12370,
        config=config
    )


class SMCBacktestEngine:
    """
    Backtest engine that uses the ACTUAL SMCStrategyV3 strategy.
    No duplicated logic - same code path as live trading.
    """

    def __init__(
        self,
        symbol: str,
        market_config: InstrumentConfig,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.5,
    ):
        self.symbol = symbol
        self.market_config = market_config
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades: List[Trade] = []

        # Create strategy using SAME function as live trading
        self.strategy = create_strategy_for_symbol(symbol, risk_per_trade)

        # Get strategy config for logging
        strategy_config = SYMBOL_CONFIGS.get(symbol, {})
        print(f"  Strategy Config: POI={strategy_config.get('poi_min_score', 'default')}, "
              f"Sweep={strategy_config.get('require_sweep', False)}, "
              f"ADX={strategy_config.get('adx_trending', 'default')}, "
              f"RR={strategy_config.get('min_rr', 'default')}")

    def run_backtest(
        self,
        m5_candles: List[Candle],
        h1_candles: List[Candle],
        h4_candles: List[Candle],
        daily_candles: List[Candle]
    ) -> BacktestResult:
        """Run backtest on provided data using the actual strategy."""
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
        position_risk_amount = 0

        max_drawdown = 0
        peak_balance = self.initial_balance
        consecutive_losses = 0
        max_consecutive_losses = 0

        # Process each M5 candle
        for i, candle in enumerate(m5_candles):
            if i < 100:  # Need history
                continue

            # Get relevant candles up to current point
            current_m5 = m5_candles[:i+1]
            current_h1 = [c for c in h1_candles if c.timestamp <= candle.timestamp]
            current_h4 = [c for c in h4_candles if c.timestamp <= candle.timestamp]
            current_daily = [c for c in daily_candles if c.timestamp.date() < candle.timestamp.date()]

            if len(current_h1) < 50 or len(current_h4) < 20:
                continue

            # Set candles for strategy (same as live)
            self.strategy.set_candles(
                h4_candles=current_h4[-100:],
                h1_candles=current_h1[-200:],
                m5_candles=current_m5[-100:],
                daily_candles=current_daily[-30:] if current_daily else None
            )

            # Update spread (same as live)
            self.strategy.update_spread(self.market_config.typical_spread * self.market_config.pip_size)

            # Check existing position
            if position:
                current_price = candle.close

                # Check SL/TP hits
                if position_direction == "long":
                    if candle.low <= position_sl:
                        # Hit SL
                        pnl = -position_risk_amount  # Lost the risked amount
                        pnl_pips = (position_sl - position_entry_price) / self.market_config.pip_size
                        self._close_position(
                            candle.timestamp, position_sl, pnl, pnl_pips,
                            position_entry_time, position_direction, position_entry_price,
                            position_sl, position_tp, "sl_hit", position_session
                        )
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        # Notify strategy of loss
                        self.strategy.on_position_closed(None, pnl)
                        continue

                    if candle.high >= position_tp:
                        # Hit TP - calculate actual R:R achieved
                        sl_distance = abs(position_entry_price - position_sl)
                        tp_distance = abs(position_tp - position_entry_price)
                        rr_achieved = tp_distance / sl_distance if sl_distance > 0 else 2.0
                        pnl = position_risk_amount * rr_achieved
                        pnl_pips = (position_tp - position_entry_price) / self.market_config.pip_size
                        self._close_position(
                            candle.timestamp, position_tp, pnl, pnl_pips,
                            position_entry_time, position_direction, position_entry_price,
                            position_sl, position_tp, "tp_hit", position_session
                        )
                        position = None
                        consecutive_losses = 0
                        self.strategy.on_position_closed(None, pnl)
                        continue

                else:  # Short
                    if candle.high >= position_sl:
                        pnl = -position_risk_amount
                        pnl_pips = (position_entry_price - position_sl) / self.market_config.pip_size
                        self._close_position(
                            candle.timestamp, position_sl, pnl, pnl_pips,
                            position_entry_time, position_direction, position_entry_price,
                            position_sl, position_tp, "sl_hit", position_session
                        )
                        position = None
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        self.strategy.on_position_closed(None, pnl)
                        continue

                    if candle.low <= position_tp:
                        sl_distance = abs(position_sl - position_entry_price)
                        tp_distance = abs(position_entry_price - position_tp)
                        rr_achieved = tp_distance / sl_distance if sl_distance > 0 else 2.0
                        pnl = position_risk_amount * rr_achieved
                        pnl_pips = (position_entry_price - position_tp) / self.market_config.pip_size
                        self._close_position(
                            candle.timestamp, position_tp, pnl, pnl_pips,
                            position_entry_time, position_direction, position_entry_price,
                            position_sl, position_tp, "tp_hit", position_session
                        )
                        position = None
                        consecutive_losses = 0
                        self.strategy.on_position_closed(None, pnl)
                        continue

            # Look for new signals if no position (using actual strategy)
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
                    # Calculate risk amount
                    position_risk_amount = self.balance * (self.risk_per_trade / 100)

            # Update equity curve and drawdown
            self.equity_curve.append(self.balance)
            if self.balance > peak_balance:
                peak_balance = self.balance
            drawdown = (peak_balance - self.balance) / peak_balance * 100
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate results
        wins = sum(1 for t in self.trades if t.result == "win")
        losses = sum(1 for t in self.trades if t.result == "loss")

        total_wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        total_losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        return BacktestResult(
            symbol=self.symbol,
            total_trades=len(self.trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(self.trades) * 100 if self.trades else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else 0,
            total_pnl=self.balance - self.initial_balance,
            total_pnl_pips=sum(t.pnl_pips for t in self.trades),
            avg_win=total_wins / wins if wins > 0 else 0,
            avg_loss=total_losses / losses if losses > 0 else 0,
            avg_rr=(total_wins / wins) / (total_losses / losses) if wins > 0 and losses > 0 else 0,
            max_drawdown=max_drawdown,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades
        )

    def _close_position(
        self,
        exit_time: datetime,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        entry_time: datetime,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        exit_reason: str,
        session: str
    ):
        """Record closed trade."""
        self.balance += pnl

        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "be")

        self.trades.append(Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            pnl=pnl,
            pnl_pips=pnl_pips,
            result=result,
            exit_reason=exit_reason,
            session=session
        ))


def run_multi_instrument_backtest(
    symbols: List[str] = None,
    months: int = 6,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.5
) -> Dict[str, BacktestResult]:
    """Run backtest on multiple instruments using the actual strategy."""

    if symbols is None:
        # Use same default symbols as run_smc_v3.py (sorted by priority)
        symbols = ["GBPUSD", "AUDUSD", "EURUSD", "US30", "USTEC", "GER40"]

    results = {}

    # Date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=months * 30)

    print("=" * 70)
    print("SMC V3 MULTI-INSTRUMENT BACKTEST")
    print("Using SAME strategy config as run_smc_v3.py")
    print("=" * 70)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: {risk_per_trade}%")
    print(f"Instruments: {', '.join(symbols)}")
    print("=" * 70)

    # Show per-symbol configurations
    print("\nPer-Symbol Strategy Settings (from run_smc_v3.py):")
    print("-" * 70)
    for symbol in symbols:
        cfg = SYMBOL_CONFIGS.get(symbol, {})
        priority = cfg.get("priority", "?")
        poi = cfg.get("poi_min_score", "default")
        sweep = cfg.get("require_sweep", False)
        adx = cfg.get("adx_trending", "default")
        rr = cfg.get("min_rr", "default")
        print(f"  {symbol}: Priority={priority}, POI={poi}, Sweep={sweep}, ADX={adx}, RR={rr}")
    print("-" * 70)
    print()

    for symbol in symbols:
        if symbol not in MARKET_DATA:
            print(f"[!] Unknown symbol: {symbol}, skipping...")
            continue

        market_config = MARKET_DATA[symbol]
        strategy_config = SYMBOL_CONFIGS.get(symbol, {})

        print(f"\n{'='*50}")
        print(f"Testing: {symbol}")
        print(f"Priority: {strategy_config.get('priority', '?')}")
        print(f"Sessions: {', '.join(market_config.sessions)}")
        print(f"{'='*50}")

        # Generate data
        print("Generating market data...")
        simulator = RealisticMarketSimulator(market_config, seed=hash(symbol) % 10000)

        m5_candles = simulator.generate_candles(start_date, end_date, 5)
        h1_candles = simulator.resample_candles(m5_candles, 60)
        h4_candles = simulator.resample_candles(m5_candles, 240)
        daily_candles = simulator.resample_candles(m5_candles, 1440)

        print(f"  M5 candles: {len(m5_candles)}")
        print(f"  H1 candles: {len(h1_candles)}")
        print(f"  H4 candles: {len(h4_candles)}")
        print(f"  Daily candles: {len(daily_candles)}")

        # Run backtest with ACTUAL strategy
        print("Running backtest with actual strategy...")
        engine = SMCBacktestEngine(symbol, market_config, initial_balance, risk_per_trade)
        result = engine.run_backtest(m5_candles, h1_candles, h4_candles, daily_candles)
        results[symbol] = result

        # Print results
        print(f"\n--- {symbol} Results ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Wins: {result.wins} | Losses: {result.losses}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pips:.1f} pips)")
        print(f"Avg Win: ${result.avg_win:.2f} | Avg Loss: ${result.avg_loss:.2f}")
        print(f"Avg R:R: {result.avg_rr:.2f}")
        print(f"Max Drawdown: {result.max_drawdown:.2f}%")
        print(f"Max Consecutive Losses: {result.max_consecutive_losses}")

    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY - ALL INSTRUMENTS")
    print("=" * 70)
    print(f"{'Symbol':<10} {'Pri':<4} {'Trades':<8} {'Win%':<8} {'PF':<8} {'P&L':<12} {'MaxDD':<8}")
    print("-" * 70)

    total_pnl = 0
    total_trades = 0

    # Sort by priority
    sorted_results = sorted(results.items(),
                           key=lambda x: SYMBOL_CONFIGS.get(x[0], {}).get("priority", 99))

    for symbol, result in sorted_results:
        priority = SYMBOL_CONFIGS.get(symbol, {}).get("priority", "?")
        print(f"{symbol:<10} {priority:<4} {result.total_trades:<8} {result.win_rate:<8.1f} "
              f"{result.profit_factor:<8.2f} ${result.total_pnl:<11,.2f} {result.max_drawdown:<8.2f}%")
        total_pnl += result.total_pnl
        total_trades += result.total_trades

    print("-" * 70)
    print(f"{'TOTAL':<10} {'':<4} {total_trades:<8} {'-':<8} {'-':<8} ${total_pnl:<11,.2f}")
    print("=" * 70)

    # Ranking by Profit Factor
    print("\nRANKING BY PROFIT FACTOR:")
    ranked = sorted(results.items(), key=lambda x: x[1].profit_factor, reverse=True)
    for i, (symbol, result) in enumerate(ranked, 1):
        status = "[OK]" if result.profit_factor > 1.0 else "[NEEDS WORK]"
        cfg = SYMBOL_CONFIGS.get(symbol, {})
        sweep = "SWEEP" if cfg.get("require_sweep", False) else ""
        print(f"  {i}. {symbol}: PF={result.profit_factor:.2f}, Win={result.win_rate:.1f}%, "
              f"Trades={result.total_trades} {status} {sweep}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)

    profitable = [(s, r) for s, r in results.items() if r.profit_factor > 1.0 and r.total_trades >= 5]
    unprofitable = [(s, r) for s, r in results.items() if r.profit_factor <= 1.0 and r.total_trades >= 5]

    if profitable:
        print("Keep trading (PF > 1.0):")
        for s, r in profitable:
            print(f"  - {s}: ${r.total_pnl:,.2f}")

    if unprofitable:
        print("\nConsider adjusting or excluding:")
        for s, r in unprofitable:
            cfg = SYMBOL_CONFIGS.get(s, {})
            if cfg.get("require_sweep", False):
                print(f"  - {s}: Already strict (sweep required). May need exclusion.")
            else:
                print(f"  - {s}: Try enabling require_sweep=True")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SMC V3 Multi-Instrument Backtest")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="Symbols to test (default: GBPUSD AUDUSD EURUSD US30 USTEC GER40)")
    parser.add_argument("--months", type=int, default=6,
                       help="Number of months to backtest (default: 6)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Initial balance (default: 10000)")
    parser.add_argument("--risk", type=float, default=0.5,
                       help="Risk percent per trade (default: 0.5)")

    args = parser.parse_args()

    results = run_multi_instrument_backtest(
        symbols=args.symbols,
        months=args.months,
        initial_balance=args.balance,
        risk_per_trade=args.risk
    )
