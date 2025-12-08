"""
Technical indicators for trading strategies.
"""

from dataclasses import dataclass
from typing import Optional
from models.candle import Candle
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VWAPData:
    """VWAP calculation data."""

    vwap: float
    upper_band: float  # VWAP + 1 std dev
    lower_band: float  # VWAP - 1 std dev
    cumulative_volume: float
    cumulative_tp_volume: float


class VWAP:
    """
    Volume Weighted Average Price (VWAP) indicator.

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Resets at the start of each trading session.
    """

    def __init__(self):
        """Initialize VWAP calculator."""
        self._cumulative_volume: float = 0.0
        self._cumulative_tp_volume: float = 0.0
        self._cumulative_tp2_volume: float = 0.0  # For standard deviation
        self._current_date: Optional[str] = None
        self._vwap: float = 0.0
        self._std_dev: float = 0.0

    def reset(self) -> None:
        """Reset VWAP for new session."""
        self._cumulative_volume = 0.0
        self._cumulative_tp_volume = 0.0
        self._cumulative_tp2_volume = 0.0
        self._vwap = 0.0
        self._std_dev = 0.0

    def update(self, candle: Candle) -> VWAPData:
        """
        Update VWAP with new candle data.

        Args:
            candle: New candle data.

        Returns:
            Updated VWAP data with bands.
        """
        # Check if new session (reset at market open)
        candle_date = candle.timestamp.strftime("%Y-%m-%d")
        if self._current_date != candle_date:
            self.reset()
            self._current_date = candle_date
            logger.debug(f"VWAP reset for new session: {candle_date}")

        # Calculate typical price
        typical_price = (candle.high + candle.low + candle.close) / 3

        # Use volume if available, otherwise use 1
        volume = candle.volume if candle.volume and candle.volume > 0 else 1.0

        # Update cumulative values
        self._cumulative_volume += volume
        self._cumulative_tp_volume += typical_price * volume
        self._cumulative_tp2_volume += (typical_price ** 2) * volume

        # Calculate VWAP
        if self._cumulative_volume > 0:
            self._vwap = self._cumulative_tp_volume / self._cumulative_volume

            # Calculate standard deviation for bands
            variance = (self._cumulative_tp2_volume / self._cumulative_volume) - (self._vwap ** 2)
            self._std_dev = variance ** 0.5 if variance > 0 else 0.0

        return VWAPData(
            vwap=self._vwap,
            upper_band=self._vwap + self._std_dev,
            lower_band=self._vwap - self._std_dev,
            cumulative_volume=self._cumulative_volume,
            cumulative_tp_volume=self._cumulative_tp_volume,
        )

    def get_current(self) -> Optional[VWAPData]:
        """Get current VWAP data without updating."""
        if self._cumulative_volume == 0:
            return None

        return VWAPData(
            vwap=self._vwap,
            upper_band=self._vwap + self._std_dev,
            lower_band=self._vwap - self._std_dev,
            cumulative_volume=self._cumulative_volume,
            cumulative_tp_volume=self._cumulative_tp_volume,
        )

    @property
    def vwap(self) -> float:
        """Get current VWAP value."""
        return self._vwap

    @property
    def upper_band(self) -> float:
        """Get upper band (VWAP + 1 std dev)."""
        return self._vwap + self._std_dev

    @property
    def lower_band(self) -> float:
        """Get lower band (VWAP - 1 std dev)."""
        return self._vwap - self._std_dev


class EMA:
    """Exponential Moving Average indicator."""

    def __init__(self, period: int = 20):
        """
        Initialize EMA.

        Args:
            period: EMA period.
        """
        self.period = period
        self._multiplier = 2 / (period + 1)
        self._ema: Optional[float] = None
        self._count = 0
        self._sum = 0.0

    def reset(self) -> None:
        """Reset EMA calculation."""
        self._ema = None
        self._count = 0
        self._sum = 0.0

    def update(self, value: float) -> Optional[float]:
        """
        Update EMA with new value.

        Args:
            value: New price value.

        Returns:
            Current EMA value or None if not enough data.
        """
        self._count += 1

        if self._ema is None:
            # Build up initial SMA
            self._sum += value
            if self._count >= self.period:
                self._ema = self._sum / self.period
            return self._ema

        # Calculate EMA
        self._ema = (value - self._ema) * self._multiplier + self._ema
        return self._ema

    @property
    def value(self) -> Optional[float]:
        """Get current EMA value."""
        return self._ema


class ATR:
    """Average True Range indicator."""

    def __init__(self, period: int = 14):
        """
        Initialize ATR.

        Args:
            period: ATR period.
        """
        self.period = period
        self._atr: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._tr_values: list[float] = []

    def reset(self) -> None:
        """Reset ATR calculation."""
        self._atr = None
        self._prev_close = None
        self._tr_values = []

    def update(self, candle: Candle) -> Optional[float]:
        """
        Update ATR with new candle.

        Args:
            candle: New candle data.

        Returns:
            Current ATR value or None if not enough data.
        """
        # Calculate True Range
        if self._prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self._prev_close),
                abs(candle.low - self._prev_close)
            )

        self._prev_close = candle.close

        if self._atr is None:
            # Build up initial values
            self._tr_values.append(tr)
            if len(self._tr_values) >= self.period:
                self._atr = sum(self._tr_values) / self.period
            return self._atr

        # Calculate ATR using Wilder's smoothing
        self._atr = ((self._atr * (self.period - 1)) + tr) / self.period
        return self._atr

    @property
    def value(self) -> Optional[float]:
        """Get current ATR value."""
        return self._atr
