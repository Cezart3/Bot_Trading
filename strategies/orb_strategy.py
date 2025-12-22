"""
Opening Range Breakout (ORB) Strategy - CORRECT Implementation.

Based on research best practices:
1. Define ASIAN SESSION RANGE (02:00-09:00 Romania time / 00:00-07:00 GMT)
2. Trade BREAKOUT at London Open (10:00 Romania / 08:00 GMT)
3. Entry: Candle CLOSE above HIGH (LONG) or below LOW (SHORT)
4. SL: Opposite side of range
5. TP: 2:1 R:R

Key Filters:
- Range height >= 10 pips (avoid noise)
- Range height <= 50 pips (avoid overextended)
- ADX > 20 (trending market)
- Trade WITH the breakout, NOT against it

Best pairs: GBPUSD, EURGBP, GBPJPY
Best session: London (10:00-13:00 Romania)
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
    """Configuration for ORB Strategy."""
    # Asian Range times (Romania timezone - matches MT5 data)
    range_start_hour: int = 2
    range_end_hour: int = 10

    # Trading window (London session)
    trade_start_hour: int = 10
    trade_end_hour: int = 13

    # Range filters
    min_range_pips: float = 10.0
    max_range_pips: float = 50.0

    # Entry confirmation
    require_candle_close: bool = True

    # Risk Management
    rr_ratio: float = 2.0
    min_sl_pips: float = 8.0
    max_sl_pips: float = 40.0
    spread_buffer_pips: float = 0.5

    # Trade limits
    max_trades_per_day: int = 2

    # ADX Filter
    use_adx_filter: bool = True
    adx_period: int = 14
    min_adx: float = 20.0


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy.
    Uses Asian session range and trades breakout at London open.
    """

    def __init__(self, symbol: str, timeframe: str = "M5", config: ORBConfig = None, **kwargs):
        super().__init__(name="ORB-London", symbol=symbol, timeframe=timeframe)

        self.pip_size = kwargs.get('pip_size', 0.0001)
        self.config = config or ORBConfig()

        # State
        self.current_date = None
        self.phase = ORBPhase.BUILDING_RANGE
        self.trades_today = 0

        # Asian Range
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self.range_candles: List[Candle] = []

        # For ADX calculation
        self.candle_history: List[Candle] = []

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

    def _calculate_adx(self, candles: List[Candle]) -> float:
        """Calculate ADX indicator."""
        if len(candles) < self.config.adx_period + 10:
            return 0.0

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]

        # True Range
        tr_list = []
        for i in range(1, len(candles)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        # +DM and -DM
        plus_dm = []
        minus_dm = []
        for i in range(1, len(candles)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        # Smoothed averages
        period = self.config.adx_period

        def smooth_average(data, period):
            if len(data) < period:
                return [0] * len(data)
            result = [sum(data[:period]) / period]
            for i in range(period, len(data)):
                result.append((result[-1] * (period - 1) + data[i]) / period)
            return result

        atr = smooth_average(tr_list, period)
        plus_di_raw = smooth_average(plus_dm, period)
        minus_di_raw = smooth_average(minus_dm, period)

        if not atr or atr[-1] == 0:
            return 0.0

        plus_di = [(p / a * 100) if a > 0 else 0 for p, a in zip(plus_di_raw, atr)]
        minus_di = [(m / a * 100) if a > 0 else 0 for m, a in zip(minus_di_raw, atr)]

        dx = []
        for p, m in zip(plus_di, minus_di):
            if p + m > 0:
                dx.append(abs(p - m) / (p + m) * 100)
            else:
                dx.append(0)

        adx_values = smooth_average(dx, period)
        return adx_values[-1] if adx_values else 0.0

    def _is_in_range_building_period(self, ts: datetime) -> bool:
        """Check if we're in Asian range building period."""
        hour = ts.hour
        return self.config.range_start_hour <= hour < self.config.range_end_hour

    def _is_in_trading_period(self, ts: datetime) -> bool:
        """Check if we're in London trading period."""
        hour = ts.hour
        return self.config.trade_start_hour <= hour < self.config.trade_end_hour

    def on_candle(self, candle: Candle, candles: list[Candle]) -> Optional[StrategySignal]:
        """Main candle processing."""
        ts = candle.timestamp

        self.candle_history = candles[-100:] if len(candles) >= 100 else candles

        # New day reset
        if self.current_date != ts.date():
            self.current_date = ts.date()
            self.phase = ORBPhase.BUILDING_RANGE
            self.trades_today = 0
            self.range_high = None
            self.range_low = None
            self.range_candles = []

        if self.trades_today >= self.config.max_trades_per_day:
            return None

        if self.phase == ORBPhase.TRADE_TAKEN:
            return None

        # PHASE 1: BUILD ASIAN RANGE
        if self._is_in_range_building_period(ts):
            self.phase = ORBPhase.BUILDING_RANGE
            self.range_candles.append(candle)

            if self.range_high is None:
                self.range_high = candle.high
                self.range_low = candle.low
            else:
                self.range_high = max(self.range_high, candle.high)
                self.range_low = min(self.range_low, candle.low)

            return None

        # PHASE 2: WAIT FOR BREAKOUT
        if self._is_in_trading_period(ts):
            if self.phase == ORBPhase.BUILDING_RANGE:
                self.phase = ORBPhase.WAITING_BREAKOUT

                if self.range_high is None or self.range_low is None:
                    self.phase = ORBPhase.TRADE_TAKEN
                    return None

                range_pips = (self.range_high - self.range_low) / self.pip_size

                if range_pips < self.config.min_range_pips:
                    self.phase = ORBPhase.TRADE_TAKEN
                    return None

                if range_pips > self.config.max_range_pips:
                    self.phase = ORBPhase.TRADE_TAKEN
                    return None

                logger.debug(f"[{self.symbol}] Asian Range: {self.range_low:.5f} - {self.range_high:.5f} ({range_pips:.1f}p)")

            if self.phase != ORBPhase.WAITING_BREAKOUT:
                return None

            # Check ADX filter
            if self.config.use_adx_filter:
                adx = self._calculate_adx(self.candle_history)
                if adx < self.config.min_adx:
                    return None

            # Check for breakout
            signal = None

            # LONG: Candle closes above range HIGH
            if candle.close > self.range_high:
                signal = self._create_signal(candle, "long")

            # SHORT: Candle closes below range LOW
            elif candle.close < self.range_low:
                signal = self._create_signal(candle, "short")

            if signal:
                self.phase = ORBPhase.TRADE_TAKEN
                self.trades_today += 1
                return signal

        return None

    def _create_signal(self, candle: Candle, direction: str) -> Optional[StrategySignal]:
        """Create trade signal with proper SL/TP."""
        is_long = direction == "long"
        side = SignalType.LONG if is_long else SignalType.SHORT

        entry = candle.close
        spread_buffer = self.config.spread_buffer_pips * self.pip_size

        if is_long:
            sl = self.range_low - spread_buffer
        else:
            sl = self.range_high + spread_buffer

        sl_distance = abs(entry - sl)
        sl_pips = sl_distance / self.pip_size

        if sl_pips < self.config.min_sl_pips:
            return None

        if sl_pips > self.config.max_sl_pips:
            return None

        tp_distance = sl_distance * self.config.rr_ratio
        tp = entry + tp_distance if is_long else entry - tp_distance

        range_pips = (self.range_high - self.range_low) / self.pip_size

        logger.info(
            f"[{self.symbol}] ORB SIGNAL: {side.value} @ {entry:.5f} | "
            f"Range: {range_pips:.1f}p | SL: {sl:.5f} ({sl_pips:.1f}p) | "
            f"TP: {tp:.5f} | R:R: {self.config.rr_ratio}"
        )

        return StrategySignal(
            signal_type=side,
            symbol=self.symbol,
            price=entry,
            stop_loss=sl,
            take_profit=tp,
            reason=f"ORB-Breakout-{side.value}",
            metadata={
                "range_high": self.range_high,
                "range_low": self.range_low,
                "range_pips": range_pips,
                "sl_pips": sl_pips,
            }
        )

    def should_exit(self, position: Position, current_price: float, candles: Optional[list[Candle]] = None) -> Optional[StrategySignal]:
        """No early exit - let SL/TP handle it."""
        return None

    def get_config_dict(self) -> dict:
        """Return config as dictionary."""
        return {
            "strategy": "ORB-London",
            "range_period": f"{self.config.range_start_hour:02d}:00-{self.config.range_end_hour:02d}:00",
            "trade_period": f"{self.config.trade_start_hour:02d}:00-{self.config.trade_end_hour:02d}:00",
            "min_range_pips": self.config.min_range_pips,
            "max_range_pips": self.config.max_range_pips,
            "rr_ratio": self.config.rr_ratio,
            "min_adx": self.config.min_adx,
        }
