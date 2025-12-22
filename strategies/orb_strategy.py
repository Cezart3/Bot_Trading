"""
Opening Range Breakout (ORB) Strategy - Multi-Session (London & NY).

Based on research best practices:
1. Define ASIAN SESSION RANGE (00:00-08:00 UTC)
2. Trade BREAKOUT at London Open (08:00 UTC)
3. Trade BREAKOUT at NY Open (14:30 UTC) - Reset if Session 1 completed
4. Entry: Candle CLOSE outside range with optional EMA and Momentum filters
5. SL: Opposite side of range (with min SL safety)
6. TP: 2:1 R:R
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

from models.candle import Candle
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)


class ORBPhase(Enum):
    """State machine phases."""
    BUILDING_RANGE = "building_range"
    WAITING_BREAKOUT = "waiting_breakout"
    TRADE_TAKEN = "trade_taken"


@dataclass
class ORBConfig:
    """Configuration for ORB Strategy (Asian Range Breakout)."""
    # Asian Range times (UTC)
    range_start_hour: int = 0
    range_end_hour: int = 8

    # Trading windows (UTC)
    trade_start_hour: int = 8
    trade_end_hour: int = 18 

    # New York specific kickoff (14:30 UTC = 16:30 Romania)
    ny_trade_start_hour: int = 14
    ny_trade_start_minute: int = 30

    # Range filters
    min_range_pips: float = 10.0
    max_range_pips: float = 150.0

    # Entry confirmation
    require_candle_close: bool = True
    breakout_buffer_pips: float = 0.0 # Extra pips beyond range to confirm breakout

    # Trend Filter (Optional)
    use_ema_filter: bool = False
    ema_period: int = 200

    # Momentum Filter (Optional)
    require_momentum: bool = False
    momentum_ratio: float = 0.6    # Body must be 60% of total candle range

    # Risk Management
    rr_ratio: float = 2.0
    min_sl_pips: float = 10.0
    max_sl_pips: float = 60.0
    spread_buffer_pips: float = 1.0

    # Protections
    use_breakeven: bool = True
    be_trigger_rr: float = 1.0     # Trigger BE at this R:R (e.g. 1.0 = 1:1)
    max_trades_per_day: int = 2

    # ADX Filter
    use_adx_filter: bool = True
    adx_period: int = 14
    min_adx: float = 20.0

class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy.
    Uses Asian session range and trades breakout at London/NY open.
    """

    def __init__(self, symbol: str, timeframe: str = "M15", config: ORBConfig = None, **kwargs):
        super().__init__(name="ORB-London-NY", symbol=symbol, timeframe=timeframe)

        self.pip_size = kwargs.get('pip_size', 0.0001)
        self.config = config or ORBConfig()

        # State
        self.current_date = None
        self.phase = ORBPhase.BUILDING_RANGE
        self.trades_today = 0
        self.last_signal_direction = None

        # Asian Range
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self.range_candles: List[Candle] = []

        # Indicators
        self.candle_history: List[Candle] = []
        self.ema_value: Optional[float] = None

    def initialize(self) -> bool:
        return True

    def on_tick(self, bid: float, ask: float, timestamp: datetime) -> Optional[StrategySignal]:
        return None

    def reset(self) -> None:
        """Reset for new day."""
        self.current_date = None
        self.phase = ORBPhase.BUILDING_RANGE
        self.trades_today = 0
        self.range_high = None
        self.range_low = None
        self.range_candles = []
        self.ema_value = None
        self.last_signal_direction = None

    def _calculate_ema(self, candles: List[Candle], period: int) -> Optional[float]:
        if len(candles) < period:
            return None
        closes = [c.close for c in candles[-period:]]
        return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

    def _is_momentum_candle(self, candle: Candle) -> bool:
        full_range = candle.high - candle.low
        if full_range == 0: return False
        body = abs(candle.close - candle.open)
        return (body / full_range) >= self.config.momentum_ratio

    def _calculate_adx(self, candles: List[Candle]) -> float:
        if len(candles) < self.config.adx_period + 10:
            return 0.0
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        tr_list = []
        for i in range(1, len(candles)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
        plus_dm = []
        minus_dm = []
        for i in range(1, len(candles)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
        period = self.config.adx_period
        def smooth_average(data, period):
            if len(data) < period: return [0] * len(data)
            result = [sum(data[:period]) / period]
            for i in range(period, len(data)):
                result.append((result[-1] * (period - 1) + data[i]) / period)
            return result
        atr = smooth_average(tr_list, period)
        plus_di_raw = smooth_average(plus_dm, period)
        minus_di_raw = smooth_average(minus_dm, period)
        if not atr or atr[-1] == 0: return 0.0
        plus_di = [(p / a * 100) if a > 0 else 0 for p, a in zip(plus_di_raw, atr)]
        minus_di = [(m / a * 100) if a > 0 else 0 for m, a in zip(minus_di_raw, atr)]
        dx = []
        for p, m in zip(plus_di, minus_di):
            dx.append(abs(p - m) / (p + m) * 100 if p + m > 0 else 0)
        adx_values = smooth_average(dx, period)
        return adx_values[-1] if adx_values else 0.0

    def _is_in_range_building_period(self, ts: datetime) -> bool:
        return self.config.range_start_hour <= ts.hour < self.config.range_end_hour

    def _is_in_trading_period(self, ts: datetime) -> bool:
        return self.config.trade_start_hour <= ts.hour < self.config.trade_end_hour

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        ts = candle.timestamp
        self.candle_history = candles[-300:] if len(candles) >= 300 else candles

        if self.current_date != ts.date():
            self.reset()
            self.current_date = ts.date()

        if self.trades_today >= self.config.max_trades_per_day:
            return None

        # Reset phase for NY session if first trade is done
        if self.phase == ORBPhase.TRADE_TAKEN:
            is_ny_time = ts.hour > self.config.ny_trade_start_hour or \
                        (ts.hour == self.config.ny_trade_start_hour and ts.minute >= self.config.ny_trade_start_minute)
            if is_ny_time:
                self.phase = ORBPhase.WAITING_BREAKOUT
                logger.info(f"[{self.symbol}] NY Session Kickoff at {ts.strftime('%H:%M')}. Resetting for possible second breakout.")

        if self.phase == ORBPhase.TRADE_TAKEN:
            return None

        if self.config.use_ema_filter:
            self.ema_value = self._calculate_ema(self.candle_history, self.config.ema_period)

        # PHASE 1: Asian Range
        if self._is_in_range_building_period(ts):
            self.phase = ORBPhase.BUILDING_RANGE
            self.range_candles.append(candle)
            if self.range_high is None:
                self.range_high, self.range_low = candle.high, candle.low
            else:
                self.range_high, self.range_low = max(self.range_high, candle.high), min(self.range_low, candle.low)
            return None

        # PHASE 2: Breakout
        if self._is_in_trading_period(ts):
            if self.phase == ORBPhase.BUILDING_RANGE:
                if self.range_high is None or self.range_low is None: 
                    return None
                
                range_pips = (self.range_high - self.range_low) / self.pip_size
                if not (self.config.min_range_pips <= range_pips <= self.config.max_range_pips):
                    self.phase = ORBPhase.TRADE_TAKEN
                    return None
                
                self.phase = ORBPhase.WAITING_BREAKOUT
                logger.debug(f"[{self.symbol}] Asian Range Confirmed: {self.range_low:.5f} - {self.range_high:.5f} ({range_pips:.1f}p)")

            if self.phase != ORBPhase.WAITING_BREAKOUT or self.range_high is None: 
                return None

            if self.config.use_adx_filter:
                if self._calculate_adx(self.candle_history) < self.config.min_adx: return None

            signal = None
            buffer = self.config.breakout_buffer_pips * self.pip_size

            # LONG
            if candle.close > self.range_high + buffer and self.last_signal_direction != "long":
                if self.config.use_ema_filter and self.ema_value and candle.close < self.ema_value: pass
                elif self.config.require_momentum and not self._is_momentum_candle(candle): pass
                else: signal = self._create_signal(candle, "long")

            # SHORT
            elif candle.close < self.range_low - buffer and self.last_signal_direction != "short":
                if self.config.use_ema_filter and self.ema_value and candle.close > self.ema_value: pass
                elif self.config.require_momentum and not self._is_momentum_candle(candle): pass
                else: signal = self._create_signal(candle, "short")

            if signal:
                self.phase = ORBPhase.TRADE_TAKEN
                self.trades_today += 1
                self.last_signal_direction = signal.signal_type.value
                return signal

        return None

    def _create_signal(self, candle: Candle, direction: str) -> Optional[StrategySignal]:
        is_long = direction == "long"
        side = SignalType.LONG if is_long else SignalType.SHORT
        entry = candle.close
        spread_buffer = self.config.spread_buffer_pips * self.pip_size
        sl = (self.range_low - spread_buffer) if is_long else (self.range_high + spread_buffer)
        sl_distance = abs(entry - sl)
        sl_pips = sl_distance / self.pip_size
        if sl_pips < self.config.min_sl_pips:
            sl_distance = self.config.min_sl_pips * self.pip_size
            sl = entry - sl_distance if is_long else entry + sl_distance
            sl_pips = self.config.min_sl_pips
        if sl_pips > self.config.max_sl_pips: return None
        tp_distance = sl_distance * self.config.rr_ratio
        tp = entry + tp_distance if is_long else entry - tp_distance
        return StrategySignal(signal_type=side, symbol=self.symbol, price=entry, stop_loss=sl, take_profit=tp, 
                              reason=f"ORB-Breakout-{side.value}", 
                              metadata={"range_high": self.range_high, "range_low": self.range_low, "sl_pips": sl_pips, "be_activated": False})

    def should_exit(self, position: Position, current_price: float, candles: Optional[list[Candle]] = None) -> Optional[StrategySignal]:
        if not self.config.use_breakeven or position.metadata.get("be_activated"): return None
        risk = abs(position.entry_price - position.stop_loss)
        trigger = risk * self.config.be_trigger_rr
        if (position.is_long and current_price >= position.entry_price + trigger) or \
           (not position.is_long and current_price <= position.entry_price - trigger):
            position.stop_loss = position.entry_price + (1 * self.pip_size if position.is_long else -1 * self.pip_size)
            position.metadata["be_activated"] = True
            logger.info(f"[{position.symbol}] BREAK-EVEN activated.")
        return None

    def get_config_dict(self) -> dict:
        return {"strategy": "ORB-London-NY", "min_adx": self.config.min_adx, "rr": self.config.rr_ratio}