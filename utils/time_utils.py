"""Time and trading session utilities."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional
import pytz


class TradingSession(Enum):
    """Trading session enumeration."""

    ASIAN = "asian"
    EUROPEAN = "european"
    US = "us"
    CUSTOM = "custom"


@dataclass
class SessionTime:
    """Trading session time configuration."""

    name: str
    start: time
    end: time
    timezone: str

    def is_active(self, dt: Optional[datetime] = None) -> bool:
        """Check if session is currently active."""
        if dt is None:
            dt = datetime.now(pytz.timezone(self.timezone))
        elif dt.tzinfo is None:
            dt = pytz.timezone(self.timezone).localize(dt)
        else:
            dt = dt.astimezone(pytz.timezone(self.timezone))

        current_time = dt.time()

        # Handle sessions that cross midnight
        if self.start > self.end:
            return current_time >= self.start or current_time <= self.end
        else:
            return self.start <= current_time <= self.end


# Predefined trading sessions
TRADING_SESSIONS = {
    TradingSession.ASIAN: SessionTime(
        name="Asian",
        start=time(0, 0),
        end=time(9, 0),
        timezone="Asia/Tokyo",
    ),
    TradingSession.EUROPEAN: SessionTime(
        name="European",
        start=time(8, 0),
        end=time(16, 0),
        timezone="Europe/London",
    ),
    TradingSession.US: SessionTime(
        name="US",
        start=time(9, 30),
        end=time(16, 0),
        timezone="America/New_York",
    ),
}


def get_session_times(
    session: TradingSession,
    session_start: Optional[str] = None,
    session_end: Optional[str] = None,
    timezone: Optional[str] = None,
) -> SessionTime:
    """
    Get session times for a trading session.

    Args:
        session: Trading session type.
        session_start: Custom session start (HH:MM format).
        session_end: Custom session end (HH:MM format).
        timezone: Custom timezone.

    Returns:
        SessionTime configuration.
    """
    if session == TradingSession.CUSTOM:
        if not session_start or not session_end or not timezone:
            raise ValueError("Custom session requires start, end, and timezone")
        start_parts = session_start.split(":")
        end_parts = session_end.split(":")
        return SessionTime(
            name="Custom",
            start=time(int(start_parts[0]), int(start_parts[1])),
            end=time(int(end_parts[0]), int(end_parts[1])),
            timezone=timezone,
        )
    return TRADING_SESSIONS[session]


def is_trading_hours(
    session_start: str,
    session_end: str,
    timezone: str = "America/New_York",
    dt: Optional[datetime] = None,
) -> bool:
    """
    Check if current time is within trading hours.

    Args:
        session_start: Session start time (HH:MM format).
        session_end: Session end time (HH:MM format).
        timezone: Timezone string.
        dt: Datetime to check (default: now).

    Returns:
        True if within trading hours, False otherwise.
    """
    tz = pytz.timezone(timezone)

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = tz.localize(dt)
    else:
        dt = dt.astimezone(tz)

    # Parse start and end times
    start_parts = session_start.split(":")
    end_parts = session_end.split(":")
    start = time(int(start_parts[0]), int(start_parts[1]))
    end = time(int(end_parts[0]), int(end_parts[1]))

    current_time = dt.time()

    # Handle sessions that cross midnight
    if start > end:
        return current_time >= start or current_time <= end
    else:
        return start <= current_time <= end


def time_to_session_end(
    session_end: str,
    timezone: str = "America/New_York",
    dt: Optional[datetime] = None,
) -> timedelta:
    """
    Calculate time remaining until session end.

    Args:
        session_end: Session end time (HH:MM format).
        timezone: Timezone string.
        dt: Current datetime (default: now).

    Returns:
        Time remaining as timedelta.
    """
    tz = pytz.timezone(timezone)

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = tz.localize(dt)
    else:
        dt = dt.astimezone(tz)

    end_parts = session_end.split(":")
    end_time = time(int(end_parts[0]), int(end_parts[1]))

    # Create end datetime
    end_dt = dt.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=0,
        microsecond=0,
    )

    # If end is tomorrow
    if end_dt <= dt:
        end_dt += timedelta(days=1)

    return end_dt - dt


def get_opening_range_window(
    session_start: str,
    range_minutes: int,
    timezone: str = "America/New_York",
    dt: Optional[datetime] = None,
) -> tuple[datetime, datetime]:
    """
    Get the opening range time window.

    Args:
        session_start: Session start time (HH:MM format).
        range_minutes: Duration of opening range in minutes.
        timezone: Timezone string.
        dt: Reference datetime (default: today).

    Returns:
        Tuple of (start_datetime, end_datetime) for opening range.
    """
    tz = pytz.timezone(timezone)

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = tz.localize(dt)
    else:
        dt = dt.astimezone(tz)

    start_parts = session_start.split(":")
    start_time = time(int(start_parts[0]), int(start_parts[1]))

    # Create start datetime for today
    range_start = dt.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=0,
        microsecond=0,
    )

    range_end = range_start + timedelta(minutes=range_minutes)

    return (range_start, range_end)


def is_opening_range_complete(
    session_start: str,
    range_minutes: int,
    timezone: str = "America/New_York",
    dt: Optional[datetime] = None,
) -> bool:
    """
    Check if opening range period has completed.

    Args:
        session_start: Session start time (HH:MM format).
        range_minutes: Duration of opening range in minutes.
        timezone: Timezone string.
        dt: Current datetime (default: now).

    Returns:
        True if opening range is complete, False otherwise.
    """
    tz = pytz.timezone(timezone)

    if dt is None:
        dt = datetime.now(tz)
    elif dt.tzinfo is None:
        dt = tz.localize(dt)
    else:
        dt = dt.astimezone(tz)

    _, range_end = get_opening_range_window(
        session_start, range_minutes, timezone, dt
    )

    return dt >= range_end


def get_next_candle_time(
    timeframe_minutes: int,
    dt: Optional[datetime] = None,
) -> datetime:
    """
    Get the time when the next candle will form.

    Args:
        timeframe_minutes: Candle timeframe in minutes.
        dt: Current datetime (default: now).

    Returns:
        Datetime of next candle close.
    """
    if dt is None:
        dt = datetime.now()

    # Calculate minutes since midnight
    minutes_since_midnight = dt.hour * 60 + dt.minute

    # Find current candle start
    current_candle_start = (minutes_since_midnight // timeframe_minutes) * timeframe_minutes

    # Next candle start
    next_candle_start = current_candle_start + timeframe_minutes

    # Calculate next candle datetime
    next_dt = dt.replace(
        hour=next_candle_start // 60,
        minute=next_candle_start % 60,
        second=0,
        microsecond=0,
    )

    # Handle day rollover
    if next_dt <= dt:
        next_dt += timedelta(days=1)

    return next_dt


def seconds_to_next_candle(
    timeframe_minutes: int,
    dt: Optional[datetime] = None,
) -> float:
    """
    Calculate seconds until next candle forms.

    Args:
        timeframe_minutes: Candle timeframe in minutes.
        dt: Current datetime (default: now).

    Returns:
        Seconds until next candle.
    """
    if dt is None:
        dt = datetime.now()

    next_candle = get_next_candle_time(timeframe_minutes, dt)
    return (next_candle - dt).total_seconds()
