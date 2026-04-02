from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import requests

from .config import AlphaVantageConfig
from .models import AlphaVantageIndicatorSnapshot, IndicatorHourChunk


DEFAULT_INTERVAL = "5min"
MAX_TIMEOUT_SECONDS = 30


class AlphaVantageLimitError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    params: dict[str, Any]


def default_indicator_specs(interval: str = DEFAULT_INTERVAL) -> list[IndicatorSpec]:
    requested = [
        "KAMA",
        "MAMA",
        "SAR",
        "MACDEXT",
        "RSI",
        "ROC",
        "ADX",
        "AROON",
        "ATR",
        "BBANDS",
        "OBV",
        "MFI",
        "MFI",
        "OBV",
        "AD",
    ]
    unique_names = list(dict.fromkeys(requested))
    spec_map = {
        "KAMA": IndicatorSpec(
            name="KAMA",
            params={"function": "KAMA", "interval": interval, "series_type": "close", "time_period": 10},
        ),
        "MAMA": IndicatorSpec(
            name="MAMA",
            params={
                "function": "MAMA",
                "interval": interval,
                "series_type": "close",
                "fastlimit": 0.5,
                "slowlimit": 0.05,
            },
        ),
        "SAR": IndicatorSpec(
            name="SAR",
            params={"function": "SAR", "interval": interval, "acceleration": 0.02, "maximum": 0.2},
        ),
        "MACDEXT": IndicatorSpec(
            name="MACDEXT",
            params={
                "function": "MACDEXT",
                "interval": interval,
                "series_type": "close",
                "fastperiod": 12,
                "slowperiod": 26,
                "signalperiod": 9,
                "fastmatype": 0,
                "slowmatype": 0,
                "signalmatype": 0,
            },
        ),
        "RSI": IndicatorSpec(
            name="RSI",
            params={"function": "RSI", "interval": interval, "series_type": "close", "time_period": 14},
        ),
        "ROC": IndicatorSpec(
            name="ROC",
            params={"function": "ROC", "interval": interval, "series_type": "close", "time_period": 10},
        ),
        "ADX": IndicatorSpec(
            name="ADX",
            params={"function": "ADX", "interval": interval, "time_period": 14},
        ),
        "AROON": IndicatorSpec(
            name="AROON",
            params={"function": "AROON", "interval": interval, "time_period": 14},
        ),
        "ATR": IndicatorSpec(
            name="ATR",
            params={"function": "ATR", "interval": interval, "time_period": 14},
        ),
        "BBANDS": IndicatorSpec(
            name="BBANDS",
            params={
                "function": "BBANDS",
                "interval": interval,
                "series_type": "close",
                "time_period": 20,
                "nbdevup": 2,
                "nbdevdn": 2,
                "matype": 0,
            },
        ),
        "OBV": IndicatorSpec(name="OBV", params={"function": "OBV", "interval": interval}),
        "MFI": IndicatorSpec(
            name="MFI",
            params={"function": "MFI", "interval": interval, "time_period": 14},
        ),
        "AD": IndicatorSpec(name="AD", params={"function": "AD", "interval": interval}),
    }
    return [spec_map[name] for name in unique_names]


@dataclass
class AlphaVantageIndicatorService:
    config: AlphaVantageConfig
    http_get: Callable[..., Any] = requests.get
    sleep_fn: Callable[[float], None] = time.sleep
    indicator_specs: list[IndicatorSpec] | None = None

    def build_snapshot(self, symbol: str) -> AlphaVantageIndicatorSnapshot:
        self.config.require()
        interval = self.config.interval or DEFAULT_INTERVAL
        merged = self._build_aligned_frame(symbol.upper(), interval=interval)
        if merged.empty:
            raise ValueError(f"No Alpha Vantage indicator data returned for {symbol.upper()}.")

        merged = merged.sort_index()
        latest_timestamp = merged.index.max()
        latest_trading_day = latest_timestamp.date()
        latest_day_frame = merged.loc[merged.index.date == latest_trading_day].copy()
        records = _build_records(latest_day_frame)
        chunks = _build_hourly_chunks(latest_day_frame, records)

        return AlphaVantageIndicatorSnapshot(
            symbol=symbol.upper(),
            interval=interval,
            trading_day=latest_trading_day.isoformat(),
            latest_timestamp=_format_timestamp(latest_timestamp),
            indicator_columns=list(latest_day_frame.columns),
            rows=records,
            hourly_chunks=chunks,
            latest_hour_chunk=chunks[-1] if chunks else None,
        )

    def _build_aligned_frame(self, symbol: str, *, interval: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        specs = self.indicator_specs or default_indicator_specs(interval)

        for index, spec in enumerate(specs):
            raw = self._fetch_indicator(symbol, spec)
            frames.append(indicator_to_dataframe(spec.name, raw))
            if index < len(specs) - 1 and self.config.request_pause_seconds > 0:
                self.sleep_fn(self.config.request_pause_seconds)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()

    def _fetch_indicator(self, symbol: str, spec: IndicatorSpec) -> dict[str, Any]:
        params = {"symbol": symbol, "apikey": self.config.api_key, **spec.params}

        for attempt in range(1, self.config.max_retries + 1):
            response = self.http_get(self.config.base_url, params=params, timeout=MAX_TIMEOUT_SECONDS)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                raise_for_status()
            data = response.json()

            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage returned an error for {spec.name}: {data['Error Message']}")

            if "Note" in data or "Information" in data:
                message = data.get("Note") or data.get("Information") or "Rate limit reached."
                if attempt < self.config.max_retries:
                    self.sleep_fn(self.config.request_pause_seconds * attempt)
                    continue
                raise AlphaVantageLimitError(f"Rate limit hit for {spec.name}: {message}")

            return data

        raise AlphaVantageLimitError(f"Rate limit hit for {spec.name}.")


def indicator_to_dataframe(name: str, data: dict[str, Any]) -> pd.DataFrame:
    series = extract_technical_series(data)
    frame = pd.DataFrame.from_dict(series, orient="index")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rename_map = {column: normalize_indicator_column(name, column, len(frame.columns)) for column in frame.columns}
    return frame.rename(columns=rename_map)


def extract_technical_series(data: dict[str, Any]) -> dict[str, Any]:
    technical_key = next((key for key in data if "Technical Analysis" in key), None)
    if technical_key is None:
        raise ValueError(f"Could not find technical analysis data in Alpha Vantage response: {data}")
    return data[technical_key]


def normalize_indicator_column(indicator_name: str, column_name: str, column_count: int) -> str:
    canonical_indicator = indicator_name.upper()
    canonical_column = _canonical_name(column_name)
    column_aliases = {
        "MAMA": {
            "MAMA": "MAMA",
            "FAMA": "FAMA",
        },
        "MACDEXT": {
            "MACD": "MACDEXT",
            "MACD_SIGNAL": "MACDEXT_SIGNAL",
            "MACD_HIST": "MACDEXT_HIST",
        },
        "AROON": {
            "AROON_DOWN": "AROON_DOWN",
            "AROON_UP": "AROON_UP",
        },
        "BBANDS": {
            "REAL_UPPER_BAND": "BBANDS_UPPER",
            "REAL_MIDDLE_BAND": "BBANDS_MIDDLE",
            "REAL_LOWER_BAND": "BBANDS_LOWER",
        },
        "AD": {
            "CHAIKIN_A_D": "AD",
        },
    }
    mapped = column_aliases.get(canonical_indicator, {}).get(canonical_column)
    if mapped:
        return mapped
    if column_count == 1:
        return canonical_indicator
    return f"{canonical_indicator}_{canonical_column}"


def _canonical_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")


def _build_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    previous_row: pd.Series | None = None

    for timestamp, row in frame.iterrows():
        record: dict[str, object] = {"time": _format_timestamp(timestamp)}
        for column in frame.columns:
            value = row[column]
            record[column] = None if pd.isna(value) else round(float(value), 6)
        record["threshold_hits"] = _threshold_hits(row, previous_row)
        records.append(record)
        previous_row = row

    return records


def _build_hourly_chunks(frame: pd.DataFrame, records: list[dict[str, object]]) -> list[IndicatorHourChunk]:
    if frame.empty or not records:
        return []

    working = frame.copy()
    working["__record__"] = records
    chunks: list[IndicatorHourChunk] = []

    for slot_start, bucket in working.groupby(pd.Grouper(freq="1h")):
        if bucket.empty:
            continue
        chunks.append(
            IndicatorHourChunk(
                slot_start=_format_timestamp(slot_start),
                slot_end=_format_timestamp(bucket.index[-1]),
                rows=list(bucket["__record__"]),
            )
        )

    return chunks


def _threshold_hits(current: pd.Series, previous: pd.Series | None) -> list[str]:
    hits: list[str] = []

    rsi = _series_value(current, "RSI")
    if rsi is not None and rsi <= 30:
        hits.append("RSI_OVERSOLD")
    if rsi is not None and rsi >= 70:
        hits.append("RSI_OVERBOUGHT")

    mfi = _series_value(current, "MFI")
    if mfi is not None and mfi <= 20:
        hits.append("MFI_OVERSOLD")
    if mfi is not None and mfi >= 80:
        hits.append("MFI_OVERBOUGHT")

    adx = _series_value(current, "ADX")
    if adx is not None and adx >= 25:
        hits.append("ADX_STRONG_TREND")

    roc = _series_value(current, "ROC")
    if roc is not None and roc >= 2:
        hits.append("ROC_BULLISH")
    if roc is not None and roc <= -2:
        hits.append("ROC_BEARISH")

    aroon_up = _series_value(current, "AROON_UP")
    aroon_down = _series_value(current, "AROON_DOWN")
    if aroon_up is not None and aroon_down is not None:
        if aroon_up >= 70 and aroon_down <= 30:
            hits.append("AROON_BULLISH")
        if aroon_down >= 70 and aroon_up <= 30:
            hits.append("AROON_BEARISH")

    if previous is not None:
        current_macd = _series_value(current, "MACDEXT")
        current_signal = _series_value(current, "MACDEXT_SIGNAL")
        previous_macd = _series_value(previous, "MACDEXT")
        previous_signal = _series_value(previous, "MACDEXT_SIGNAL")
        if None not in (current_macd, current_signal, previous_macd, previous_signal):
            if previous_macd <= previous_signal and current_macd > current_signal:
                hits.append("MACDEXT_BULLISH_CROSS")
            if previous_macd >= previous_signal and current_macd < current_signal:
                hits.append("MACDEXT_BEARISH_CROSS")

        current_mama = _series_value(current, "MAMA")
        current_fama = _series_value(current, "FAMA")
        previous_mama = _series_value(previous, "MAMA")
        previous_fama = _series_value(previous, "FAMA")
        if None not in (current_mama, current_fama, previous_mama, previous_fama):
            if previous_mama <= previous_fama and current_mama > current_fama:
                hits.append("MAMA_BULLISH_CROSS")
            if previous_mama >= previous_fama and current_mama < current_fama:
                hits.append("MAMA_BEARISH_CROSS")

    return list(dict.fromkeys(hits))


def _series_value(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    return float(value)


def _format_timestamp(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
