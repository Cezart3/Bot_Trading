"""Candle (OHLCV) data model."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Candle:
    """
    Represents a single candlestick (OHLCV bar).

    Attributes:
        timestamp: Candle open time
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        symbol: Trading symbol
        timeframe: Candle timeframe (e.g., 'M5', 'H1')
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = "M5"

    @property
    def body_size(self) -> float:
        """Calculate the absolute body size."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Calculate upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Calculate lower wick size."""
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        """Calculate total candle range (high - low)."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Check if candle is bearish (close < open)."""
        return self.close < self.open

    @property
    def mid_price(self) -> float:
        """Calculate mid price of the candle."""
        return (self.high + self.low) / 2

    def contains_price(self, price: float) -> bool:
        """Check if a price is within the candle's range."""
        return self.low <= price <= self.high

    def to_dict(self) -> dict:
        """Convert candle to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Candle":
        """Create Candle from dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            timestamp=timestamp,
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0)),
            symbol=data.get("symbol", ""),
            timeframe=data.get("timeframe", "M5"),
        )


@dataclass
class OpeningRange:
    """
    Represents the Opening Range for ORB strategy.

    Attributes:
        high: Highest price during opening range
        low: Lowest price during opening range
        start_time: Start of the opening range period
        end_time: End of the opening range period
        symbol: Trading symbol
        is_valid: Whether the range is valid for trading
    """

    high: float
    low: float
    start_time: datetime
    end_time: datetime
    symbol: str
    is_valid: bool = True
    candles: Optional[list[Candle]] = None

    @property
    def range_size(self) -> float:
        """Calculate the range size."""
        return self.high - self.low

    @property
    def mid_point(self) -> float:
        """Calculate the midpoint of the range."""
        return (self.high + self.low) / 2

    def is_breakout_long(self, price: float, buffer: float = 0.0) -> bool:
        """Check if price breaks above the range high."""
        return price > (self.high + buffer)

    def is_breakout_short(self, price: float, buffer: float = 0.0) -> bool:
        """Check if price breaks below the range low."""
        return price < (self.low - buffer)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "high": self.high,
            "low": self.low,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "symbol": self.symbol,
            "range_size": self.range_size,
            "is_valid": self.is_valid,
        }
