"""
Open Range Breakout (ORB) Strategy Implementation.

The ORB strategy captures price breakouts from the opening range.
The opening range is defined as the high and low of the first N minutes
of the trading session.

Entry Rules:
- Wait for the opening range to form (first N minutes)
- Go LONG when price breaks above the opening range high
- Go SHORT when price breaks below the opening range low

Exit Rules:
- Stop Loss: Below range low for longs, above range high for shorts
- Take Profit: Based on risk/reward ratio
- Time-based exit: Close all positions before session end
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional
import pytz

from models.candle import Candle, OpeningRange
from models.position import Position
from strategies.base_strategy import BaseStrategy, SignalType, StrategySignal
from utils.logger import get_logger
from utils.time_utils import (
    is_trading_hours,
    is_opening_range_complete,
    get_opening_range_window,
    time_to_session_end,
)

logger = get_logger(__name__)


@dataclass
class ORBState:
    """State tracking for ORB strategy."""

    opening_range: Optional[OpeningRange] = None
    range_calculated: bool = False
    breakout_triggered: bool = False
    breakout_direction: Optional[str] = None  # "long" or "short"
    entry_price: Optional[float] = None
    session_date: Optional[datetime] = None
    trades_today: int = 0
    max_trades_per_day: int = 1


class ORBStrategy(BaseStrategy):
    """
    Open Range Breakout Strategy.

    This strategy trades breakouts from the opening range of the trading session.
    Configurable parameters include:
    - Opening range duration (default: 5 minutes)
    - Session start/end times
    - Breakout buffer (pips above/below range)
    - Risk/reward ratio
    - ATR-based filtering
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "M5",
        magic_number: int = 12345,
        # Session settings
        session_start: str = "09:30",
        session_end: str = "16:00",
        timezone: str = "America/New_York",
        # ORB settings
        range_minutes: int = 5,
        breakout_buffer_pips: float = 2.0,
        # Risk settings
        risk_reward_ratio: float = 2.0,
        use_atr_filter: bool = True,
        min_range_pips: float = 5.0,
        max_range_pips: float = 50.0,
        # Trade management
        max_trades_per_day: int = 1,
        close_before_session_end_minutes: int = 15,
    ):
        """
        Initialize ORB strategy.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            magic_number: Unique identifier for orders.
            session_start: Trading session start time (HH:MM).
            session_end: Trading session end time (HH:MM).
            timezone: Trading timezone.
            range_minutes: Minutes to calculate opening range.
            breakout_buffer_pips: Buffer above/below range for entry.
            risk_reward_ratio: Take profit / stop loss ratio.
            use_atr_filter: Filter trades based on ATR.
            min_range_pips: Minimum range size in pips.
            max_range_pips: Maximum range size in pips.
            max_trades_per_day: Maximum trades per session.
            close_before_session_end_minutes: Close positions X minutes before session end.
        """
        super().__init__(
            name="ORB",
            symbol=symbol,
            timeframe=timeframe,
            magic_number=magic_number,
        )

        # Session settings
        self.session_start = session_start
        self.session_end = session_end
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)

        # ORB settings
        self.range_minutes = range_minutes
        self.breakout_buffer_pips = breakout_buffer_pips

        # Risk settings
        self.risk_reward_ratio = risk_reward_ratio
        self.use_atr_filter = use_atr_filter
        self.min_range_pips = min_range_pips
        self.max_range_pips = max_range_pips

        # Trade management
        self.max_trades_per_day = max_trades_per_day
        self.close_before_session_end_minutes = close_before_session_end_minutes

        # Pip size (will be set based on symbol)
        self.pip_size = 0.0001  # Default for forex

        # State
        self._state = ORBState(max_trades_per_day=max_trades_per_day)
        self._current_atr: Optional[float] = None

    def initialize(self) -> bool:
        """Initialize strategy."""
        logger.info(f"Initializing ORB strategy for {self.symbol}")
        logger.info(f"Session: {self.session_start} - {self.session_end} ({self.timezone})")
        logger.info(f"Opening range: {self.range_minutes} minutes")
        self._reset_daily_state()
        return True

    def _reset_daily_state(self, session_date=None) -> None:
        """Reset state for a new trading day."""
        self._state = ORBState(max_trades_per_day=self.max_trades_per_day)
        self._state.session_date = session_date or datetime.now(self.tz).date()
        logger.info("ORB state reset for new session")

    def _is_same_session_day(self, timestamp: datetime, reference_date: datetime) -> bool:
        """Check if timestamp is in the same trading session as reference date."""
        if timestamp.tzinfo is None:
            ts = self.tz.localize(timestamp)
        else:
            ts = timestamp.astimezone(self.tz)
        return ts.date() == reference_date.date()

    def _check_new_session(self, timestamp: datetime) -> None:
        """Check if it's a new trading session and reset if needed."""
        if timestamp.tzinfo is None:
            timestamp = self.tz.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(self.tz)

        current_date = timestamp.date()

        if self._state.session_date != current_date:
            self._reset_daily_state(current_date)

    def _calculate_opening_range(self, candles: list[Candle]) -> Optional[OpeningRange]:
        """
        Calculate the opening range from candles.

        Args:
            candles: List of candles.

        Returns:
            OpeningRange object or None if not enough data.
        """
        if not candles:
            return None

        # Use the date of the last candle (for backtesting compatibility)
        last_candle = candles[-1]
        candle_date = last_candle.timestamp
        if candle_date.tzinfo is None:
            candle_date = self.tz.localize(candle_date)
        else:
            candle_date = candle_date.astimezone(self.tz)

        # Get session start datetime for the candle's date
        start_parts = self.session_start.split(":")
        session_start_dt = candle_date.replace(
            hour=int(start_parts[0]),
            minute=int(start_parts[1]),
            second=0,
            microsecond=0,
        )
        range_end_dt = session_start_dt + timedelta(minutes=self.range_minutes)

        # Filter candles within opening range
        range_candles = []
        for candle in candles:
            candle_time = candle.timestamp
            if candle_time.tzinfo is None:
                candle_time = self.tz.localize(candle_time)
            else:
                candle_time = candle_time.astimezone(self.tz)

            if session_start_dt <= candle_time < range_end_dt:
                range_candles.append(candle)

        if not range_candles:
            # Fallback: use first candle of the session as opening range
            session_candles = [
                c for c in candles
                if self._is_same_session_day(c.timestamp, candle_date)
            ]
            if session_candles:
                # Use first candle as opening range
                first_candle = session_candles[0]
                range_candles = [first_candle]
                logger.debug(f"Using first candle as opening range: {first_candle.timestamp}")
            else:
                logger.debug("No candles found in opening range window")
                return None

        # Calculate range high and low
        range_high = max(c.high for c in range_candles)
        range_low = min(c.low for c in range_candles)

        opening_range = OpeningRange(
            high=range_high,
            low=range_low,
            start_time=session_start_dt,
            end_time=range_end_dt,
            symbol=self.symbol,
            is_valid=True,
            candles=range_candles,
        )

        # Validate range size
        range_size_pips = opening_range.range_size / self.pip_size

        if range_size_pips < self.min_range_pips:
            logger.warning(
                f"Opening range too small: {range_size_pips:.1f} pips < {self.min_range_pips} min"
            )
            opening_range.is_valid = False
        elif range_size_pips > self.max_range_pips:
            logger.warning(
                f"Opening range too large: {range_size_pips:.1f} pips > {self.max_range_pips} max"
            )
            opening_range.is_valid = False

        logger.info(
            f"Opening range calculated: High={range_high}, Low={range_low}, "
            f"Size={range_size_pips:.1f} pips, Valid={opening_range.is_valid}"
        )

        return opening_range

    def _calculate_atr(self, candles: list[Candle], period: int = 14) -> float:
        """
        Calculate Average True Range.

        Args:
            candles: List of candles.
            period: ATR period.

        Returns:
            ATR value.
        """
        if len(candles) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        # Simple moving average of TR
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
            return atr

        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def on_candle(
        self,
        candle: Candle,
        candles: list[Candle],
    ) -> Optional[StrategySignal]:
        """Process new candle and check for signals."""
        if not self._enabled:
            return None

        # Check for new session
        self._check_new_session(candle.timestamp)

        # Check if within trading hours
        if not is_trading_hours(
            self.session_start, self.session_end, self.timezone, candle.timestamp
        ):
            logger.debug("Outside trading hours")
            return None

        # Check if max trades reached
        if self._state.trades_today >= self._state.max_trades_per_day:
            logger.debug(f"Max trades per day reached ({self._state.max_trades_per_day})")
            return None

        # Check if breakout already triggered today
        if self._state.breakout_triggered:
            logger.debug("Breakout already triggered today")
            return None

        # Calculate ATR
        self._current_atr = self._calculate_atr(candles)

        # Check if opening range is complete
        if not is_opening_range_complete(
            self.session_start, self.range_minutes, self.timezone, candle.timestamp
        ):
            logger.debug("Opening range not yet complete")
            return None

        # Calculate opening range if not done
        if not self._state.range_calculated:
            self._state.opening_range = self._calculate_opening_range(candles)
            self._state.range_calculated = True

            if self._state.opening_range is None or not self._state.opening_range.is_valid:
                logger.warning("Invalid opening range, skipping session")
                return None

        # Check for breakout
        return self._check_breakout(candle)

    def _check_breakout(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Check if price has broken out of the opening range.

        Args:
            candle: Current candle.

        Returns:
            Signal if breakout detected, None otherwise.
        """
        if self._state.opening_range is None:
            return None

        or_range = self._state.opening_range
        buffer = self.breakout_buffer_pips * self.pip_size

        # Check for long breakout (close above range high)
        if candle.close > or_range.high + buffer:
            logger.info(f"LONG breakout detected! Price {candle.close} > {or_range.high + buffer}")

            stop_loss = or_range.low - buffer
            risk = candle.close - stop_loss
            take_profit = candle.close + (risk * self.risk_reward_ratio)

            self._state.breakout_triggered = True
            self._state.breakout_direction = "long"

            signal = StrategySignal(
                signal_type=SignalType.LONG,
                symbol=self.symbol,
                price=candle.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.8,
                reason=f"ORB Long Breakout - Range High: {or_range.high:.5f}",
                metadata={
                    "range_high": or_range.high,
                    "range_low": or_range.low,
                    "range_size": or_range.range_size,
                    "atr": self._current_atr,
                },
            )
            self._last_signal = signal
            return signal

        # Check for short breakout (close below range low)
        if candle.close < or_range.low - buffer:
            logger.info(f"SHORT breakout detected! Price {candle.close} < {or_range.low - buffer}")

            stop_loss = or_range.high + buffer
            risk = stop_loss - candle.close
            take_profit = candle.close - (risk * self.risk_reward_ratio)

            self._state.breakout_triggered = True
            self._state.breakout_direction = "short"

            signal = StrategySignal(
                signal_type=SignalType.SHORT,
                symbol=self.symbol,
                price=candle.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.8,
                reason=f"ORB Short Breakout - Range Low: {or_range.low:.5f}",
                metadata={
                    "range_high": or_range.high,
                    "range_low": or_range.low,
                    "range_size": or_range.range_size,
                    "atr": self._current_atr,
                },
            )
            self._last_signal = signal
            return signal

        return None

    def on_tick(
        self,
        bid: float,
        ask: float,
        timestamp: datetime,
    ) -> Optional[StrategySignal]:
        """Process tick update - can be used for real-time breakout detection."""
        # For now, we only use candle-based signals
        # This could be enhanced for immediate breakout detection
        return None

    def should_exit(
        self,
        position: Position,
        current_price: float,
        candles: Optional[list[Candle]] = None,
    ) -> Optional[StrategySignal]:
        """
        Check if position should be exited.

        Args:
            position: Current position.
            current_price: Current price.
            candles: Recent candles.

        Returns:
            Exit signal if should exit, None otherwise.
        """
        now = datetime.now(self.tz)

        # Check time-based exit
        time_remaining = time_to_session_end(self.session_end, self.timezone, now)
        if time_remaining.total_seconds() <= self.close_before_session_end_minutes * 60:
            logger.info(f"Session ending soon ({time_remaining}), closing position")
            exit_type = (
                SignalType.EXIT_LONG if position.is_long else SignalType.EXIT_SHORT
            )
            return StrategySignal(
                signal_type=exit_type,
                symbol=self.symbol,
                price=current_price,
                reason="Session end - time-based exit",
            )

        return None

    def on_position_opened(self, position: Position) -> None:
        """Called when position is opened."""
        self._state.trades_today += 1
        self._state.entry_price = position.entry_price
        logger.info(
            f"ORB position opened: {position.side} {position.quantity} @ {position.entry_price}"
        )

    def on_position_closed(self, position: Position, pnl: float) -> None:
        """Called when position is closed."""
        logger.info(
            f"ORB position closed: {position.side} @ {position.exit_price}, PnL: {pnl:.2f}"
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        candles: Optional[list[Candle]] = None,
    ) -> Optional[float]:
        """Calculate stop loss based on opening range."""
        if self._state.opening_range is None:
            return None

        buffer = self.breakout_buffer_pips * self.pip_size

        if is_long:
            return self._state.opening_range.low - buffer
        else:
            return self._state.opening_range.high + buffer

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
    ) -> Optional[float]:
        """Calculate take profit based on risk/reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio

        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward

    def get_status(self) -> dict:
        """Get strategy status."""
        base_status = super().get_status()
        base_status.update({
            "opening_range": (
                self._state.opening_range.to_dict()
                if self._state.opening_range
                else None
            ),
            "range_calculated": self._state.range_calculated,
            "breakout_triggered": self._state.breakout_triggered,
            "breakout_direction": self._state.breakout_direction,
            "trades_today": self._state.trades_today,
            "max_trades_per_day": self._state.max_trades_per_day,
            "current_atr": self._current_atr,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "timezone": self.timezone,
        })
        return base_status

    def reset(self) -> None:
        """Reset strategy for new session."""
        super().reset()
        self._reset_daily_state()
