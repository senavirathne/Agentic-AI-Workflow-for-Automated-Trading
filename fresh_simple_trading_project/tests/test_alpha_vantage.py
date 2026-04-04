from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from fresh_simple_trading_project.alpha_vantage import (
    AlphaVantageIndicatorService,
    LayeredCache,
    default_indicator_specs,
)
from fresh_simple_trading_project.config import AlphaVantageConfig
from fresh_simple_trading_project.storage import InMemoryResultStore


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


def test_alpha_vantage_fetches_once_per_day_and_reuses_stored_snapshot(tmp_path) -> None:
    counter = _CallCounter()
    service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        cache_dir=tmp_path / "alpha_vantage",
    )

    first = service.build_snapshot("AAPL", end_time="2025-01-03T09:35:00Z")

    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        cache_dir=tmp_path / "alpha_vantage",
    )
    second = second_service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    assert counter.calls == len(default_indicator_specs())
    assert first.latest_hour_chunk is not None
    assert first.latest_hour_chunk.slot_start == "2025-01-03 09:00:00"
    assert len(first.latest_hour_chunk.rows) == 2
    assert second.latest_hour_chunk is not None
    assert second.latest_hour_chunk.slot_start == "2025-01-03 10:00:00"
    assert len(second.latest_hour_chunk.rows) == 1


def test_alpha_vantage_feature_frame_reuses_stored_daily_data_without_refetch(tmp_path) -> None:
    counter = _CallCounter()
    cache_dir = tmp_path / "alpha_vantage"
    service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        cache_dir=cache_dir,
    )
    service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    counter.calls = 0
    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        cache_dir=cache_dir,
    )
    price_bars = _price_bars()

    feature_frame = second_service.build_feature_frame(
        "AAPL",
        price_bars,
        end_time="2025-01-03T10:00:00Z",
    )

    assert counter.calls == 0
    assert not feature_frame.empty
    assert feature_frame.index.max() == pd.Timestamp("2025-01-03T10:00:00Z")
    assert {"rsi", "macd", "macd_signal", "buy_trigger", "sell_trigger"}.issubset(feature_frame.columns)


def test_alpha_vantage_reuses_result_store_snapshot_without_api_key() -> None:
    counter = _CallCounter()
    store = InMemoryResultStore()
    first_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )

    first = first_service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    counter.calls = 0
    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key=None,
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    second = second_service.build_snapshot("AAPL", end_time="2025-01-03T09:35:00Z")

    assert first.latest_hour_chunk is not None
    assert counter.calls == 0
    assert second.latest_hour_chunk is not None
    assert second.latest_hour_chunk.slot_start == "2025-01-03 09:00:00"
    assert len(second.latest_hour_chunk.rows) == 2
    assert store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day="2025-01-03", interval="5min") is not None


def test_alpha_vantage_build_snapshot_reuses_current_day_store_without_refetch(monkeypatch) -> None:
    monkeypatch.setattr("fresh_simple_trading_project.alpha_vantage._requested_trading_day", lambda _: "2025-01-03")
    counter = _CallCounter()
    store = InMemoryResultStore()
    first_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    seeded = first_service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    counter.calls = 0
    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key=None,
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    snapshot = second_service.build_snapshot("AAPL")

    assert counter.calls == 0
    assert snapshot.trading_day == "2025-01-03"
    assert snapshot.latest_timestamp == seeded.latest_timestamp


def test_alpha_vantage_build_snapshot_refreshes_when_store_only_has_stale_day(monkeypatch) -> None:
    counter = _CallCounter()
    store = InMemoryResultStore()
    first_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    first_service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    monkeypatch.setattr("fresh_simple_trading_project.alpha_vantage._requested_trading_day", lambda _: "2025-01-04")
    counter.calls = 0
    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    snapshot = second_service.build_snapshot("AAPL")

    assert counter.calls == len(default_indicator_specs())
    assert snapshot.trading_day == "2025-01-03"


def test_alpha_vantage_refresh_merges_new_data_with_older_store_days() -> None:
    store = InMemoryResultStore()
    seed_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=_fake_http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    seed_service.build_snapshot("AAPL", end_time="2025-01-02T15:55:00Z")

    counter = _CallCounter(payload_factory=lambda function: _single_day_indicator_payload(function, "2025-01-03"))
    service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )

    frame = service.build_indicator_frame("AAPL", refresh=True)

    assert counter.calls == len(default_indicator_specs())
    assert frame.index.min() == pd.Timestamp("2025-01-02T15:55:00Z")
    assert frame.index.max() == pd.Timestamp("2025-01-03T10:00:00Z")
    assert store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day="2025-01-02", interval="5min") is not None
    assert store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day="2025-01-03", interval="5min") is not None


def test_alpha_vantage_ensure_data_for_window_fetches_and_persists_missing_days() -> None:
    counter = _CallCounter()
    store = InMemoryResultStore()
    service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )

    fetched = service.ensure_data_for_window(
        "AAPL",
        start_time="2025-01-03T09:30:00Z",
        end_time="2025-01-03T10:00:00Z",
    )

    assert fetched is True
    assert counter.calls == len(default_indicator_specs())
    assert store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day="2025-01-03", interval="5min") is not None


def test_alpha_vantage_ensure_data_for_window_reuses_store_without_refetch() -> None:
    counter = _CallCounter()
    store = InMemoryResultStore()
    first_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key="ABCDEFGHI1234",
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )
    first_service.build_snapshot("AAPL", end_time="2025-01-03T10:00:00Z")

    counter.calls = 0
    second_service = AlphaVantageIndicatorService(
        AlphaVantageConfig(
            api_key=None,
            interval="5min",
            request_pause_seconds=0.0,
            max_retries=1,
        ),
        http_get=counter.http_get,
        sleep_fn=lambda _: None,
        result_store=store,
    )

    fetched = second_service.ensure_data_for_window(
        "AAPL",
        start_time="2025-01-03T09:30:00Z",
        end_time="2025-01-03T10:00:00Z",
    )

    assert fetched is False
    assert counter.calls == 0


def test_layered_cache_prefers_memory_then_store_then_disk_and_backfills_memory() -> None:
    key = ("AAPL", "5min", "2025-01-03")
    frame = _price_bars().tail(2)
    calls: list[str] = []
    written_to_disk: dict[tuple[str, str, str], pd.DataFrame] = {}
    cache = LayeredCache(
        memory_cache={},
        store_getter=lambda requested_key: _record_frame_hit(calls, "store", requested_key, key, frame),
        disk_getter=lambda requested_key: _record_frame_hit(calls, "disk", requested_key, key, frame),
        disk_putter=lambda requested_key, value: written_to_disk.setdefault(requested_key, value.copy()),
    )

    first = cache.get(key)
    second = cache.get(key)

    assert first is not None
    assert first.equals(frame)
    assert second is not None
    assert second.equals(frame)
    assert calls == ["store"]
    assert key in written_to_disk
    assert written_to_disk[key].equals(frame)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _CallCounter:
    def __init__(self, payload_factory: Callable[[str], dict[str, Any]] | None = None) -> None:
        self.calls = 0
        self.payload_factory = payload_factory or _indicator_payload

    def http_get(self, _url: str, *, params: dict[str, Any], timeout: int) -> _FakeResponse:
        del timeout
        self.calls += 1
        function = str(params["function"])
        return _FakeResponse(self.payload_factory(function))


def _fake_http_get(_url: str, *, params: dict[str, Any], timeout: int) -> _FakeResponse:
    del timeout
    function = str(params["function"])
    return _FakeResponse(_indicator_payload(function))


def _record_frame_hit(
    calls: list[str],
    label: str,
    requested_key: tuple[str, str, str],
    expected_key: tuple[str, str, str],
    frame: pd.DataFrame,
) -> pd.DataFrame | None:
    if requested_key != expected_key:
        return None
    calls.append(label)
    return frame.copy()


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


def _single_day_indicator_payload(function: str, trading_day: str) -> dict[str, Any]:
    payload = _indicator_payload(function)
    technical_key = f"Technical Analysis: {function}"
    return {
        "Meta Data": dict(payload["Meta Data"]),
        technical_key: {
            timestamp: values
            for timestamp, values in payload[technical_key].items()
            if timestamp.startswith(trading_day)
        },
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


def _price_bars() -> pd.DataFrame:
    index = pd.to_datetime(_timestamps()[1:], utc=True)
    return pd.DataFrame(
        {
            "open": [100.0, 100.4, 101.1],
            "high": [100.8, 101.2, 102.0],
            "low": [99.8, 100.2, 100.9],
            "close": [100.5, 101.0, 101.8],
            "volume": [10_000, 11_000, 12_000],
        },
        index=index,
    )
