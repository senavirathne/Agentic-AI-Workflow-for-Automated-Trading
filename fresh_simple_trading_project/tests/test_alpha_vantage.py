from __future__ import annotations

from typing import Any

from fresh_simple_trading_project.alpha_vantage import (
    AlphaVantageIndicatorService,
    default_indicator_specs,
)
from fresh_simple_trading_project.config import AlphaVantageConfig


def test_default_indicator_specs_deduplicate_requested_indicator_names() -> None:
    names = [spec.name for spec in default_indicator_specs()]

    assert names == [
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
        "AD",
    ]


def test_build_snapshot_aligns_latest_trading_day_and_hourly_chunks() -> None:
    service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=_fake_http_get,
        sleep_fn=lambda _: None,
    )

    snapshot = service.build_snapshot("AAPL")

    assert snapshot.symbol == "AAPL"
    assert snapshot.interval == "5min"
    assert snapshot.trading_day == "2025-01-03"
    assert snapshot.latest_timestamp == "2025-01-03 10:00:00"
    assert len(snapshot.rows) == 3
    assert snapshot.rows[0]["time"] == "2025-01-03 09:30:00"
    assert snapshot.rows[-1]["time"] == "2025-01-03 10:00:00"
    assert "MFI" in snapshot.indicator_columns
    assert "OBV" in snapshot.indicator_columns
    assert "MACDEXT_SIGNAL" in snapshot.indicator_columns
    assert snapshot.rows[-1]["threshold_hits"] == [
        "RSI_OVERBOUGHT",
        "MFI_OVERBOUGHT",
        "ADX_STRONG_TREND",
        "ROC_BULLISH",
        "AROON_BULLISH",
        "MACDEXT_BULLISH_CROSS",
        "MAMA_BULLISH_CROSS",
    ]
    assert len(snapshot.hourly_chunks) == 2
    assert snapshot.latest_hour_chunk is not None
    assert snapshot.latest_hour_chunk.slot_start == "2025-01-03 10:00:00"
    assert snapshot.latest_hour_chunk.slot_end == "2025-01-03 10:00:00"
    assert len(snapshot.latest_hour_chunk.rows) == 1


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def _fake_http_get(_url: str, *, params: dict[str, Any], timeout: int) -> _FakeResponse:
    del timeout
    function = str(params["function"])
    return _FakeResponse(_indicator_payload(function))


def _indicator_payload(function: str) -> dict[str, Any]:
    series_map = {
        "KAMA": _single_series("KAMA", [99.8, 100.1, 100.3, 100.9]),
        "MAMA": _multi_series(
            {
                "MAMA": [100.0, 100.1, 100.0, 100.8],
                "FAMA": [100.2, 100.2, 100.1, 100.4],
            }
        ),
        "SAR": _single_series("SAR", [99.4, 99.7, 99.9, 100.0]),
        "MACDEXT": _multi_series(
            {
                "MACD": [-0.2, -0.1, -0.05, 0.2],
                "MACD_Signal": [-0.1, 0.0, 0.02, 0.1],
                "MACD_Hist": [-0.1, -0.1, -0.07, 0.1],
            }
        ),
        "RSI": _single_series("RSI", [45.0, 55.0, 60.0, 75.0]),
        "ROC": _single_series("ROC", [-1.0, 0.5, 1.0, 2.5]),
        "ADX": _single_series("ADX", [18.0, 19.5, 21.0, 30.0]),
        "AROON": _multi_series(
            {
                "Aroon Down": [60.0, 45.0, 35.0, 20.0],
                "Aroon Up": [40.0, 55.0, 65.0, 80.0],
            }
        ),
        "ATR": _single_series("ATR", [1.1, 1.2, 1.3, 1.4]),
        "BBANDS": _multi_series(
            {
                "Real Upper Band": [102.0, 102.4, 102.8, 103.5],
                "Real Middle Band": [100.0, 100.5, 101.0, 101.5],
                "Real Lower Band": [98.0, 98.6, 99.2, 99.5],
            }
        ),
        "OBV": _single_series("OBV", [10_000.0, 12_000.0, 13_000.0, 15_000.0]),
        "MFI": _single_series("MFI", [35.0, 50.0, 65.0, 85.0]),
        "AD": _single_series("Chaikin A/D", [1_000.0, 1_200.0, 1_500.0, 1_800.0]),
    }
    return {
        "Meta Data": {"1: Symbol": "AAPL"},
        f"Technical Analysis: {function}": series_map[function],
    }


def _single_series(label: str, values: list[float]) -> dict[str, dict[str, str]]:
    return {timestamp: {label: str(value)} for timestamp, value in zip(_timestamps(), values, strict=True)}


def _multi_series(columns: dict[str, list[float]]) -> dict[str, dict[str, str]]:
    payload: dict[str, dict[str, str]] = {}
    for index, timestamp in enumerate(_timestamps()):
        payload[timestamp] = {name: str(values[index]) for name, values in columns.items()}
    return payload


def _timestamps() -> list[str]:
    return [
        "2025-01-02 15:55:00",
        "2025-01-03 09:30:00",
        "2025-01-03 09:35:00",
        "2025-01-03 10:00:00",
    ]
