"""Serialization helpers shared by storage backends."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..models import (
    AlphaVantageIndicatorSnapshot,
    ForecastSnapshot,
    IndicatorHourChunk,
    NewsArticle,
    PerformanceSnapshot,
)


def _normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _timestamp_text(value: pd.Timestamp | str) -> str:
    return _normalize_timestamp(value).isoformat().replace("+00:00", "Z")


def _forecast_payload(snapshot: ForecastSnapshot) -> dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "generated_at": _timestamp_text(snapshot.generated_at),
        "valid_until": _timestamp_text(snapshot.valid_until),
        "reference_price": snapshot.reference_price,
        "reference_volume": snapshot.reference_volume,
        "trend_bias": snapshot.trend_bias,
        "continuation_price_target": snapshot.continuation_price_target,
        "continuation_volume_target": snapshot.continuation_volume_target,
        "reversal_price_target": snapshot.reversal_price_target,
        "reversal_volume_target": snapshot.reversal_volume_target,
        "continuation_signals": snapshot.continuation_signals,
        "reversal_signals": snapshot.reversal_signals,
        "summary": snapshot.summary,
        "confidence": snapshot.confidence,
    }


def _forecast_from_payload(payload: dict[str, Any]) -> ForecastSnapshot:
    return ForecastSnapshot(
        symbol=str(payload.get("symbol", "")).upper(),
        generated_at=_normalize_timestamp(payload["generated_at"]),
        valid_until=_normalize_timestamp(payload["valid_until"]),
        reference_price=float(payload.get("reference_price", 0.0)),
        reference_volume=float(payload.get("reference_volume", 0.0)),
        trend_bias=str(payload.get("trend_bias", "neutral")),
        continuation_price_target=_optional_float(payload.get("continuation_price_target")),
        continuation_volume_target=_optional_float(payload.get("continuation_volume_target")),
        reversal_price_target=_optional_float(payload.get("reversal_price_target")),
        reversal_volume_target=_optional_float(payload.get("reversal_volume_target")),
        continuation_signals=[str(item) for item in payload.get("continuation_signals", []) or []],
        reversal_signals=[str(item) for item in payload.get("reversal_signals", []) or []],
        summary=payload.get("summary"),
        confidence=float(payload.get("confidence", 0.0)),
    )


def _performance_payload(snapshot: PerformanceSnapshot) -> dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "as_of": _timestamp_text(snapshot.as_of),
        "position_qty": snapshot.position_qty,
        "trade_count": snapshot.trade_count,
        "market_price": snapshot.market_price,
        "avg_entry_price": snapshot.avg_entry_price,
        "realized_profit": snapshot.realized_profit,
        "unrealized_profit": snapshot.unrealized_profit,
        "current_profit": snapshot.current_profit,
    }


def _performance_from_payload(payload: dict[str, Any]) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        symbol=str(payload.get("symbol", "")).upper(),
        as_of=_normalize_timestamp(payload["as_of"]),
        position_qty=int(payload.get("position_qty", 0)),
        trade_count=int(payload.get("trade_count", 0)),
        market_price=float(payload.get("market_price", 0.0)),
        avg_entry_price=_optional_float(payload.get("avg_entry_price")),
        realized_profit=float(payload.get("realized_profit", 0.0)),
        unrealized_profit=float(payload.get("unrealized_profit", 0.0)),
        current_profit=float(payload.get("current_profit", 0.0)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _alpha_vantage_snapshot_from_payload(payload: dict[str, Any]) -> AlphaVantageIndicatorSnapshot:
    hourly_chunks = [
        _indicator_hour_chunk_from_payload(item)
        for item in payload.get("hourly_chunks", []) or []
        if isinstance(item, dict)
    ]
    latest_hour_chunk_payload = payload.get("latest_hour_chunk")
    latest_hour_chunk = None
    if isinstance(latest_hour_chunk_payload, dict):
        latest_hour_chunk = _indicator_hour_chunk_from_payload(latest_hour_chunk_payload)

    return AlphaVantageIndicatorSnapshot(
        symbol=str(payload.get("symbol", "")).upper(),
        interval=str(payload.get("interval", "5min")),
        trading_day=str(payload.get("trading_day", "")),
        latest_timestamp=str(payload.get("latest_timestamp", "")),
        indicator_columns=[str(item) for item in payload.get("indicator_columns", []) or []],
        rows=[dict(item) for item in payload.get("rows", []) or [] if isinstance(item, dict)],
        hourly_chunks=hourly_chunks,
        latest_hour_chunk=latest_hour_chunk,
    )


def _indicator_hour_chunk_from_payload(payload: dict[str, Any]) -> IndicatorHourChunk:
    return IndicatorHourChunk(
        slot_start=str(payload.get("slot_start", "")),
        slot_end=str(payload.get("slot_end", "")),
        rows=[dict(item) for item in payload.get("rows", []) or [] if isinstance(item, dict)],
    )


def _snapshot_hour_chunk(
    snapshot: AlphaVantageIndicatorSnapshot,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> IndicatorHourChunk | None:
    if not snapshot.rows:
        return None
    if as_of is None:
        return snapshot.latest_hour_chunk

    normalized_as_of = _normalize_timestamp(as_of)
    slot_start = normalized_as_of.floor("1h")
    rows = []
    for row in snapshot.rows:
        timestamp_text = row.get("time")
        if not timestamp_text:
            continue
        timestamp = _normalize_timestamp(str(timestamp_text))
        if slot_start <= timestamp <= normalized_as_of:
            rows.append(dict(row))
    if not rows:
        return None
    return IndicatorHourChunk(
        slot_start=slot_start.strftime("%Y-%m-%d %H:%M:%S"),
        slot_end=str(rows[-1]["time"]),
        rows=rows,
    )


def _news_article_key(article: NewsArticle) -> str:
    provider = article.provider.strip().lower()
    url = article.url.strip()
    if url:
        return f"{provider}|{url.lower()}" if provider else url.lower()
    return " | ".join(
        [
            provider,
            article.headline.strip().lower(),
            article.source.strip().lower(),
            _normalize_news_timestamp(article.published_at),
        ]
    )


def _news_article_sort_key(article: NewsArticle) -> str:
    return _normalize_news_timestamp(article.published_at)


def _normalize_news_timestamp(value: str | None, fallback: str | None = None) -> str:
    candidate = (value or "").strip()
    if candidate:
        try:
            timestamp = pd.Timestamp(candidate)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return timestamp.isoformat().replace("+00:00", "Z")
        except Exception:
            return candidate
    if fallback:
        return fallback
    return "1970-01-01T00:00:00Z"


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
