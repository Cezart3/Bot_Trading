"""
Economic News Filter for Trading Bot.

Filters out high-impact news days to avoid volatile market conditions.
Data sources:
- Forex Factory (free)
- Investing.com economic calendar

Red = High Impact (avoid trading)
Orange = Medium Impact (optional filter)
Yellow = Low Impact (usually safe)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
import re

from utils.logger import get_logger

logger = get_logger(__name__)


class NewsImpact(Enum):
    """News impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EconomicEvent:
    """Represents an economic news event."""
    date: date
    time: Optional[str]  # HH:MM format or None if all-day
    currency: str
    impact: NewsImpact
    event: str
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None


@dataclass
class NewsFilterConfig:
    """Configuration for news filter."""

    # Impact levels to filter (avoid trading)
    filter_high_impact: bool = True
    filter_medium_impact: bool = False
    filter_low_impact: bool = False

    # Currencies to monitor (empty = all)
    currencies: list[str] = field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])

    # Time buffer around news (minutes)
    buffer_before_minutes: int = 90
    buffer_after_minutes: int = 15

    # Filter entire day or just around news time
    filter_entire_day: bool = False

    # Cache settings
    cache_days: int = 7


class NewsFilter:
    """
    Filters trading based on economic news calendar.

    Usage:
        news_filter = NewsFilter()
        news_filter.update_calendar()

        if news_filter.is_safe_to_trade(datetime.now()):
            # Execute trade
        else:
            # Skip - high impact news day
    """

    def __init__(
        self,
        config: Optional[NewsFilterConfig] = None,
        cache_path: str = "data/news_cache.json",
    ):
        """
        Initialize news filter.

        Args:
            config: Filter configuration.
            cache_path: Path to cache file.
        """
        self.config = config or NewsFilterConfig()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.events: list[EconomicEvent] = []
        self._high_impact_dates: set[date] = set()
        self._last_update: Optional[datetime] = None

        # Load cached data
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached news data."""
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path) as f:
                data = json.load(f)

            self._last_update = datetime.fromisoformat(data.get("last_update", "2000-01-01"))

            for event_data in data.get("events", []):
                event = EconomicEvent(
                    date=date.fromisoformat(event_data["date"]),
                    time=event_data.get("time"),
                    currency=event_data["currency"],
                    impact=NewsImpact(event_data["impact"]),
                    event=event_data["event"],
                )
                self.events.append(event)

            self._build_high_impact_dates()
            logger.info(f"Loaded {len(self.events)} events from cache")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Save news data to cache."""
        try:
            data = {
                "last_update": datetime.now().isoformat(),
                "events": [
                    {
                        "date": event.date.isoformat(),
                        "time": event.time,
                        "currency": event.currency,
                        "impact": event.impact.value,
                        "event": event.event,
                    }
                    for event in self.events
                ],
            }

            with open(self.cache_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.events)} events to cache")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _build_high_impact_dates(self) -> None:
        """Build set of high impact dates for quick lookup."""
        self._high_impact_dates.clear()

        for event in self.events:
            if event.impact == NewsImpact.HIGH and self.config.filter_high_impact:
                if not self.config.currencies or event.currency in self.config.currencies:
                    self._high_impact_dates.add(event.date)

            if event.impact == NewsImpact.MEDIUM and self.config.filter_medium_impact:
                if not self.config.currencies or event.currency in self.config.currencies:
                    self._high_impact_dates.add(event.date)

    def update_calendar(self, force: bool = False) -> bool:
        """
        Update economic calendar from online sources.

        Args:
            force: Force update even if cache is fresh.

        Returns:
            True if updated successfully.
        """
        # Check if update needed
        if not force and self._last_update:
            cache_age = datetime.now() - self._last_update
            if cache_age < timedelta(hours=6):
                logger.debug("Cache is fresh, skipping update")
                return True

        # Try multiple sources
        success = self._fetch_forex_factory() or self._fetch_hardcoded_calendar()

        if success:
            self._build_high_impact_dates()
            self._save_cache()
            self._last_update = datetime.now()

        return success

    def _fetch_forex_factory(self) -> bool:
        """
        Fetch calendar from Forex Factory.

        Note: This is a simplified version. Real implementation would
        need to handle their specific format and potential rate limits.
        """
        try:
            # Forex Factory calendar URL
            url = "https://www.forexfactory.com/calendar"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Forex Factory returned status {response.status_code}")
                return False

            # Parse HTML (simplified - real parsing would be more complex)
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for calendar rows
            calendar_rows = soup.find_all("tr", class_="calendar__row")

            if not calendar_rows:
                logger.warning("No calendar rows found on Forex Factory")
                return False

            new_events = []
            current_date = None

            for row in calendar_rows:
                # Extract date
                date_cell = row.find("td", class_="calendar__date")
                if date_cell and date_cell.text.strip():
                    date_text = date_cell.text.strip()
                    try:
                        # Parse date like "Mon Dec 9"
                        current_date = datetime.strptime(
                            f"{date_text} {datetime.now().year}",
                            "%a %b %d %Y"
                        ).date()
                    except ValueError:
                        pass

                if not current_date:
                    continue

                # Extract currency
                currency_cell = row.find("td", class_="calendar__currency")
                currency = currency_cell.text.strip() if currency_cell else ""

                # Extract impact
                impact_cell = row.find("td", class_="calendar__impact")
                impact = NewsImpact.LOW
                if impact_cell:
                    if impact_cell.find("span", class_="high"):
                        impact = NewsImpact.HIGH
                    elif impact_cell.find("span", class_="medium"):
                        impact = NewsImpact.MEDIUM

                # Extract event name
                event_cell = row.find("td", class_="calendar__event")
                event_name = event_cell.text.strip() if event_cell else ""

                if currency and event_name:
                    new_events.append(EconomicEvent(
                        date=current_date,
                        time=None,
                        currency=currency,
                        impact=impact,
                        event=event_name,
                    ))

            if new_events:
                self.events = new_events
                logger.info(f"Fetched {len(new_events)} events from Forex Factory")
                return True

            return False

        except requests.RequestException as e:
            logger.warning(f"Failed to fetch Forex Factory: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error parsing Forex Factory: {e}")
            return False

    def _fetch_hardcoded_calendar(self) -> bool:
        """
        Use hardcoded high-impact events as fallback.

        This includes recurring events that typically have high impact:
        - FOMC meetings
        - NFP (Non-Farm Payrolls)
        - CPI releases
        - Central bank decisions
        """
        logger.info("Using hardcoded economic calendar")

        # Generate events for the next 90 days
        today = date.today()
        new_events = []

        # High impact recurring events (approximate dates)
        # FOMC meetings are typically every 6 weeks
        # NFP is first Friday of each month
        # CPI is usually mid-month

        for days_ahead in range(90):
            check_date = today + timedelta(days=days_ahead)

            # Skip weekends
            if check_date.weekday() >= 5:
                continue

            # NFP - First Friday of month (but can be on other days too)
            if check_date.weekday() == 4 and check_date.day <= 7:
                new_events.append(EconomicEvent(
                    date=check_date,
                    time="08:30",
                    currency="USD",
                    impact=NewsImpact.HIGH,
                    event="Non-Farm Payrolls",
                ))
                # Add Unemployment Rate same day
                new_events.append(EconomicEvent(
                    date=check_date,
                    time="08:30",
                    currency="USD",
                    impact=NewsImpact.HIGH,
                    event="Unemployment Rate",
                ))

            # Flash PMI - Usually 3rd week of month (HIGH impact)
            if 15 <= check_date.day <= 23:
                # German Flash PMI (EUR) - usually Monday/Tuesday
                if check_date.weekday() in [0, 1]:
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="04:30",
                        currency="EUR",
                        impact=NewsImpact.HIGH,
                        event="German Flash Manufacturing PMI",
                    ))
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="04:30",
                        currency="EUR",
                        impact=NewsImpact.HIGH,
                        event="German Flash Services PMI",
                    ))
                # US Flash PMI - usually same day or day after
                if check_date.weekday() in [0, 1]:
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="09:45",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="Flash Manufacturing PMI",
                    ))
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="09:45",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="Flash Services PMI",
                    ))

            # CPI - Usually around 10th-15th of month
            if check_date.day in [10, 11, 12, 13, 14] and check_date.weekday() < 5:
                # Check if it's the second week Wednesday or Thursday
                if check_date.weekday() in [2, 3]:  # Wed or Thu
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="08:30",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="CPI m/m",
                    ))
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="08:30",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="Core CPI m/m",
                    ))

            # PPI - Usually day after CPI
            if check_date.day in [11, 12, 13, 14, 15] and check_date.weekday() < 5:
                if check_date.weekday() in [3, 4]:  # Thu or Fri
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="08:30",
                        currency="USD",
                        impact=NewsImpact.MEDIUM,
                        event="PPI m/m",
                    ))

            # FOMC - Typically 8 meetings per year
            # Jan, Mar, May, Jun, Jul, Sep, Nov, Dec
            fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
            if check_date.month in fomc_months:
                # FOMC usually meets mid-month for 2 days, statement on Wed
                if 14 <= check_date.day <= 21 and check_date.weekday() == 2:  # Wednesday
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="14:00",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="FOMC Statement",
                    ))
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="14:30",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="FOMC Press Conference",
                    ))

            # Retail Sales - Usually around 15th-17th (HIGH impact)
            if 14 <= check_date.day <= 17 and check_date.weekday() < 5:
                new_events.append(EconomicEvent(
                    date=check_date,
                    time="08:30",
                    currency="USD",
                    impact=NewsImpact.HIGH,
                    event="Retail Sales m/m",
                ))
                new_events.append(EconomicEvent(
                    date=check_date,
                    time="08:30",
                    currency="USD",
                    impact=NewsImpact.HIGH,
                    event="Core Retail Sales m/m",
                ))

            # GDP - Usually last week of month/quarter end months
            if check_date.month in [1, 4, 7, 10] and 25 <= check_date.day <= 31:
                if check_date.weekday() in [2, 3, 4]:  # Wed-Fri
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="08:30",
                        currency="USD",
                        impact=NewsImpact.HIGH,
                        event="GDP q/q",
                    ))

            # ISM Manufacturing - First business day of month
            if check_date.day <= 3 and check_date.weekday() < 5:
                new_events.append(EconomicEvent(
                    date=check_date,
                    time="10:00",
                    currency="USD",
                    impact=NewsImpact.MEDIUM,
                    event="ISM Manufacturing PMI",
                ))

            # ECB Decisions - Usually every 6 weeks on Thursday
            if check_date.month in [1, 3, 4, 6, 7, 9, 10, 12]:
                if 10 <= check_date.day <= 17 and check_date.weekday() == 3:  # Thursday
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="08:15",
                        currency="EUR",
                        impact=NewsImpact.HIGH,
                        event="ECB Interest Rate Decision",
                    ))

            # BOE Decisions - Usually every 6 weeks on Thursday
            if check_date.month in [2, 3, 5, 6, 8, 9, 11, 12]:
                if 1 <= check_date.day <= 10 and check_date.weekday() == 3:  # Thursday
                    new_events.append(EconomicEvent(
                        date=check_date,
                        time="07:00",
                        currency="GBP",
                        impact=NewsImpact.HIGH,
                        event="BOE Interest Rate Decision",
                    ))

        self.events = new_events
        logger.info(f"Generated {len(new_events)} hardcoded events")
        return True

    def add_custom_event(
        self,
        event_date: date,
        event_name: str,
        currency: str = "USD",
        impact: NewsImpact = NewsImpact.HIGH,
    ) -> None:
        """
        Add a custom event to the calendar.

        Args:
            event_date: Date of the event.
            event_name: Name of the event.
            currency: Affected currency.
            impact: Impact level.
        """
        event = EconomicEvent(
            date=event_date,
            time=None,
            currency=currency,
            impact=impact,
            event=event_name,
        )
        self.events.append(event)
        self._build_high_impact_dates()
        logger.info(f"Added custom event: {event_name} on {event_date}")

    def is_safe_to_trade(
        self,
        check_datetime: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> bool:
        """
        Check if it's safe to trade at the given time.

        Args:
            check_datetime: Datetime to check (default: now).
            symbol: Trading symbol (used to determine relevant currencies).

        Returns:
            True if safe to trade, False if high-impact news.
        """
        if check_datetime is None:
            check_datetime = datetime.now()

        check_date = check_datetime.date()

        # Determine relevant currencies based on symbol
        relevant_currencies = self.config.currencies
        if symbol:
            symbol_upper = symbol.upper().replace(".", "")
            # Check for forex pairs (6 characters like EURUSD, GBPUSD)
            if len(symbol_upper) >= 6:
                base = symbol_upper[:3]
                quote = symbol_upper[3:6]
                # Common currency codes
                valid_currencies = {"EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
                if base in valid_currencies and quote in valid_currencies:
                    relevant_currencies = [base, quote]
            # Fallback for US stocks/indices
            if not relevant_currencies or relevant_currencies == self.config.currencies:
                if any(x in symbol_upper for x in ["US", "SPY", "QQQ", "NVDA", "AMD", "TSLA", "ES", "NQ"]):
                    relevant_currencies = ["USD"]

        if self.config.filter_entire_day:
            # Check if date has any high impact events
            for event in self.events:
                if event.date != check_date:
                    continue

                if relevant_currencies and event.currency not in relevant_currencies:
                    continue

                if event.impact == NewsImpact.HIGH and self.config.filter_high_impact:
                    logger.info(f"Blocking trade: High impact news - {event.event} ({event.currency})")
                    return False

                if event.impact == NewsImpact.MEDIUM and self.config.filter_medium_impact:
                    logger.info(f"Blocking trade: Medium impact news - {event.event} ({event.currency})")
                    return False
        else:
            # Check specific time windows around news
            for event in self.events:
                if event.date != check_date:
                    continue

                if relevant_currencies and event.currency not in relevant_currencies:
                    continue

                if event.time:
                    # Parse event time
                    event_time = datetime.strptime(event.time, "%H:%M").time()
                    event_datetime = datetime.combine(check_date, event_time)

                    # Check if within buffer window
                    window_start = event_datetime - timedelta(minutes=self.config.buffer_before_minutes)
                    window_end = event_datetime + timedelta(minutes=self.config.buffer_after_minutes)

                    if window_start <= check_datetime <= window_end:
                        if event.impact == NewsImpact.HIGH and self.config.filter_high_impact:
                            logger.info(f"Blocking trade: Near high impact news - {event.event}")
                            return False

        return True

    def is_high_impact_day(self, check_date: Optional[date] = None) -> bool:
        """
        Quick check if a date has high impact news.

        Args:
            check_date: Date to check (default: today).

        Returns:
            True if high impact day.
        """
        if check_date is None:
            check_date = date.today()

        return check_date in self._high_impact_dates

    def get_events_for_date(self, check_date: Optional[date] = None) -> list[EconomicEvent]:
        """
        Get all events for a specific date.

        Args:
            check_date: Date to check (default: today).

        Returns:
            List of events on that date.
        """
        if check_date is None:
            check_date = date.today()

        return [e for e in self.events if e.date == check_date]

    def get_high_impact_dates(self, days_ahead: int = 30) -> list[date]:
        """
        Get list of high impact dates for the next N days.

        Args:
            days_ahead: Number of days to look ahead.

        Returns:
            List of dates with high impact news.
        """
        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        return sorted([
            d for d in self._high_impact_dates
            if today <= d <= end_date
        ])

    def print_calendar(self, days_ahead: int = 14) -> None:
        """Print upcoming high impact events."""
        print("\n" + "=" * 70)
        print("              ECONOMIC CALENDAR - HIGH IMPACT EVENTS")
        print("=" * 70)

        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        events = [
            e for e in self.events
            if today <= e.date <= end_date and e.impact == NewsImpact.HIGH
        ]
        events.sort(key=lambda x: x.date)

        if not events:
            print("\n  No high impact events in the next {days_ahead} days\n")
            return

        print(f"\n  {'Date':<12} | {'Time':<6} | {'Currency':<4} | {'Event':<35}")
        print("  " + "-" * 65)

        current_date = None
        for event in events:
            date_str = event.date.strftime("%Y-%m-%d") if event.date != current_date else ""
            current_date = event.date
            time_str = event.time or "All day"

            print(f"  {date_str:<12} | {time_str:<6} | {event.currency:<4} | {event.event[:35]:<35}")

        print("  " + "-" * 65)
        print(f"\n  Total high impact events: {len(events)}")
        print("=" * 70 + "\n")


# Convenience function
def create_news_filter(filter_high: bool = True, filter_medium: bool = False) -> NewsFilter:
    """
    Create a news filter with common settings.

    Args:
        filter_high: Filter high impact news days.
        filter_medium: Also filter medium impact news.

    Returns:
        Configured NewsFilter instance.
    """
    config = NewsFilterConfig(
        filter_high_impact=filter_high,
        filter_medium_impact=filter_medium,
        filter_entire_day=True,
        currencies=["USD"],  # Focus on USD for US stocks
    )

    news_filter = NewsFilter(config)
    news_filter.update_calendar()

    return news_filter


if __name__ == "__main__":
    # Test news filter
    news_filter = create_news_filter()
    news_filter.print_calendar(days_ahead=30)

    print(f"Is today safe to trade? {news_filter.is_safe_to_trade()}")
    print(f"Is today a high impact day? {news_filter.is_high_impact_day()}")

    high_dates = news_filter.get_high_impact_dates(30)
    print(f"\nHigh impact dates in next 30 days: {len(high_dates)}")
    for d in high_dates[:10]:
        print(f"  - {d}")
