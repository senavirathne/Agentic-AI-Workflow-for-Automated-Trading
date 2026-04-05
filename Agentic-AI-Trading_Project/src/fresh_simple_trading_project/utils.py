"""Small time and formatting utilities shared by the workflow modules."""

from __future__ import annotations

from datetime import date, datetime, time as datetime_time, timedelta, timezone
import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)

US_MARKET_TIMEZONE = ZoneInfo("America/New_York")
US_MARKET_CLOSE_TIME = datetime_time(hour=16, minute=0)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_HOLIDAY_CALENDAR = _NYSEHolidayCalendar()


def timestamp_slug() -> str:
    """Return a UTC timestamp formatted for filenames and record IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def next_top_of_hour(now: datetime | None = None) -> datetime:
    """Return the next whole-hour UTC timestamp after `now`."""
    current = now or datetime.now(timezone.utc)
    rounded = current.replace(minute=0, second=0, microsecond=0)
    return rounded + timedelta(hours=1)


def sleep_until(target: datetime, sleep_seconds: float | None = None) -> float:
    """Sleep until the target time or for an explicit number of seconds."""
    seconds = sleep_seconds
    if seconds is None:
        seconds = (target - datetime.now(timezone.utc)).total_seconds()
    seconds = max(float(seconds), 0.0)
    if seconds > 0:
        time.sleep(seconds)
    return seconds


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize timestamp-like inputs to UTC pandas timestamps."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def last_closed_us_market_day_cutoff(
    value: Any,
    *,
    available_timestamps: pd.Index | list[Any] | tuple[Any, ...] | None = None,
) -> pd.Timestamp:
    """Return the UTC cutoff for the most recent completed U.S. market day."""
    checkpoint_local = _as_utc_timestamp(value).tz_convert(US_MARKET_TIMEZONE)
    current_local_date = checkpoint_local.date()
    available_dates = _available_us_market_dates(available_timestamps)
    close_local = pd.Timestamp(
        datetime.combine(current_local_date, US_MARKET_CLOSE_TIME, tzinfo=US_MARKET_TIMEZONE)
    )

    if checkpoint_local >= close_local and current_local_date in available_dates:
        target_date = current_local_date
    else:
        target_date = _previous_available_market_date(available_dates, current_local_date)
        if target_date is None:
            target_date = _previous_us_market_day(current_local_date)

    return pd.Timestamp(
        datetime.combine(target_date, US_MARKET_CLOSE_TIME, tzinfo=US_MARKET_TIMEZONE)
    ).tz_convert("UTC")


def _available_us_market_dates(
    timestamps: pd.Index | list[Any] | tuple[Any, ...] | None,
) -> list[date]:
    if timestamps is None:
        return []
    normalized_dates = {
        local_date
        for timestamp in timestamps
        if _is_us_market_day(local_date := _as_utc_timestamp(timestamp).tz_convert(US_MARKET_TIMEZONE).date())
    }
    return sorted(normalized_dates)


def _previous_available_market_date(available_dates: list[date], current_date: date) -> date | None:
    for candidate in reversed(available_dates):
        if candidate < current_date:
            return candidate
    return None


def us_market_day_dates(
    timestamps: pd.Index | list[Any] | tuple[Any, ...],
    *,
    max_date: date | None = None,
) -> list[date]:
    """Return sorted U.S. market dates present in the supplied timestamps."""
    candidate_dates = _available_us_market_dates(timestamps)
    if max_date is None:
        return candidate_dates
    return [candidate for candidate in candidate_dates if candidate <= max_date]


def _previous_us_market_day(current_date: date) -> date:
    candidate = current_date - timedelta(days=1)
    while not _is_us_market_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def _is_us_market_day(value: date) -> bool:
    if value.weekday() >= 5:
        return False
    return value not in _nyse_holiday_dates(value, value)


def _nyse_holiday_dates(start_date: date, end_date: date) -> set[date]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    holidays = _NYSE_HOLIDAY_CALENDAR.holidays(start=start, end=end)
    return {pd.Timestamp(item).date() for item in holidays}


def _format_region(region: tuple[float, float]) -> str:
    """Format a numeric price region for human-readable output."""
    return f"{region[0]:.2f}-{region[1]:.2f}"


def _region_midpoint(region: tuple[float, float]) -> float:
    """Return the midpoint of a price region."""
    return (region[0] + region[1]) / 2


def _as_string(value: Any) -> str | None:
    """Normalize optional string-like values to stripped strings."""
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None
