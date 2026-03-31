from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def next_top_of_hour(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    rounded = current.replace(minute=0, second=0, microsecond=0)
    return rounded + timedelta(hours=1)


def sleep_until(target: datetime, sleep_seconds: float | None = None) -> float:
    seconds = sleep_seconds
    if seconds is None:
        seconds = (target - datetime.now(timezone.utc)).total_seconds()
    seconds = max(float(seconds), 0.0)
    if seconds > 0:
        time.sleep(seconds)
    return seconds
