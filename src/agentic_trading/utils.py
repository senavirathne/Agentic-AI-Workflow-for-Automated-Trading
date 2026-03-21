from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sleep_until(target_time: datetime, chunk_seconds: int = 30) -> None:
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)
    while True:
        remaining = (target_time - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, chunk_seconds))


def next_top_of_hour(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def timestamp_slug(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return ts.strftime("%Y%m%dT%H%M%SZ")


def normalize_bars_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])

    normalized = frame.copy()
    if isinstance(normalized.index, pd.MultiIndex) and "symbol" in normalized.index.names:
        normalized = normalized.xs(symbol, level="symbol", drop_level=True)

    if "symbol" in normalized.columns:
        normalized = normalized[normalized["symbol"] == symbol]

    if "timestamp" not in normalized.columns:
        normalized = normalized.reset_index()

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    normalized = normalized.set_index("timestamp")
    normalized = normalized[~normalized.index.duplicated(keep="last")]

    for column in normalized.columns:
        if column == "symbol":
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="ignore")

    return normalized


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)

