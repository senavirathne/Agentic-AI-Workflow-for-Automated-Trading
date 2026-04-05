"""Alpha Vantage indicator retrieval, alignment, caching, and snapshot helpers."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable

import pandas as pd
import requests

from .config import AlphaVantageConfig
from .models import AlphaVantageIndicatorSnapshot, IndicatorHourChunk
from .storage import ResultStore


DEFAULT_INTERVAL = "5min"
MAX_TIMEOUT_SECONDS = 30


class AlphaVantageLimitError(RuntimeError):
    """Raised when Alpha Vantage rate limits prevent a successful response."""

    pass


@dataclass(frozen=True)
class IndicatorSpec:
    """Describe a single Alpha Vantage indicator request."""

    name: str
    params: dict[str, Any]


@dataclass
class LayeredCache:
    """Resolve cached values in memory, then store, then disk."""

    memory_cache: dict[Hashable, pd.DataFrame]
    store_getter: Callable[[Hashable], pd.DataFrame | None] | None = None
    disk_getter: Callable[[Hashable], pd.DataFrame | None] | None = None
    store_putter: Callable[[Hashable, pd.DataFrame], None] | None = None
    disk_putter: Callable[[Hashable, pd.DataFrame], None] | None = None

    def get(self, key: Hashable) -> pd.DataFrame | None:
        """Load a dataframe from memory, then store, then disk."""

        memory_value = self.memory_cache.get(key)
        if memory_value is not None:
            return memory_value.copy()

        if self.store_getter is not None:
            store_value = self.store_getter(key)
            if store_value is not None:
                self.memory_cache[key] = store_value.copy()
                if self.disk_putter is not None:
                    self.disk_putter(key, store_value)
                return store_value.copy()

        if self.disk_getter is not None:
            disk_value = self.disk_getter(key)
            if disk_value is not None:
                self.memory_cache[key] = disk_value.copy()
                return disk_value.copy()

        return None

    def put(
        self,
        key: Hashable,
        value: pd.DataFrame,
        *,
        persist_store: bool = True,
        persist_disk: bool = True,
    ) -> pd.DataFrame:
        """Store a dataframe in memory and optionally persist it downstream."""

        normalized = value.sort_index().copy()
        self.memory_cache[key] = normalized.copy()
        if persist_store and self.store_putter is not None and not normalized.empty:
            self.store_putter(key, normalized)
        if persist_disk and self.disk_putter is not None:
            self.disk_putter(key, normalized)
        return normalized.copy()


def default_indicator_specs(interval: str = DEFAULT_INTERVAL) -> list[IndicatorSpec]:
    """Return the default set of 5-minute Alpha Vantage indicator requests."""
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
            params={
                "function": "KAMA",
                "interval": interval,
                "series_type": "close",
                "time_period": 10,
                "outputsize": "full",
            },
        ),
        "MAMA": IndicatorSpec(
            name="MAMA",
            params={
                "function": "MAMA",
                "interval": interval,
                "series_type": "close",
                "fastlimit": 0.5,
                "slowlimit": 0.05,
                "outputsize": "full",
            },
        ),
        "SAR": IndicatorSpec(
            name="SAR",
            params={
                "function": "SAR",
                "interval": interval,
                "acceleration": 0.02,
                "maximum": 0.2,
                "outputsize": "full",
            },
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
                "outputsize": "full",
            },
        ),
        "RSI": IndicatorSpec(
            name="RSI",
            params={
                "function": "RSI",
                "interval": interval,
                "series_type": "close",
                "time_period": 14,
                "outputsize": "full",
            },
        ),
        "ROC": IndicatorSpec(
            name="ROC",
            params={
                "function": "ROC",
                "interval": interval,
                "series_type": "close",
                "time_period": 10,
                "outputsize": "full",
            },
        ),
        "ADX": IndicatorSpec(
            name="ADX",
            params={"function": "ADX", "interval": interval, "time_period": 14, "outputsize": "full"},
        ),
        "AROON": IndicatorSpec(
            name="AROON",
            params={"function": "AROON", "interval": interval, "time_period": 14, "outputsize": "full"},
        ),
        "ATR": IndicatorSpec(
            name="ATR",
            params={"function": "ATR", "interval": interval, "time_period": 14, "outputsize": "full"},
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
                "outputsize": "full",
            },
        ),
        "OBV": IndicatorSpec(name="OBV", params={"function": "OBV", "interval": interval, "outputsize": "full"}),
        "MFI": IndicatorSpec(
            name="MFI",
            params={"function": "MFI", "interval": interval, "time_period": 14, "outputsize": "full"},
        ),
        "AD": IndicatorSpec(name="AD", params={"function": "AD", "interval": interval, "outputsize": "full"}),
    }
    return [spec_map[name] for name in unique_names]


@dataclass
class AlphaVantageIndicatorService:
    """Fetch, align, persist, and slice Alpha Vantage indicator tables."""

    config: AlphaVantageConfig
    http_get: Callable[..., Any] = requests.get
    sleep_fn: Callable[[float], None] = time.sleep
    indicator_specs: list[IndicatorSpec] | None = None
    cache_dir: Path | None = None
    result_store: ResultStore | None = None
    _aligned_frame_cache: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict, init=False, repr=False)
    _daily_frame_cache: dict[tuple[str, str, str], pd.DataFrame] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create the cache directory when disk caching is enabled."""

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_snapshot(
        self,
        symbol: str,
        *,
        end_time: pd.Timestamp | str | None = None,
    ) -> AlphaVantageIndicatorSnapshot:
        """Build a snapshot for the requested symbol and optional checkpoint.

        Args:
            symbol: Ticker symbol to load.
            end_time: Optional timestamp used to slice the latest available day.

        Returns:
            A normalized Alpha Vantage indicator snapshot.
        """

        interval = self.config.interval or DEFAULT_INTERVAL
        normalized_symbol = symbol.upper()
        if end_time is None:
            requested_day = _requested_trading_day(None)
            cached_snapshot = self._load_snapshot_for_trading_day(
                normalized_symbol,
                interval,
                requested_day,
                allow_aligned_cache=True,
            )
            if cached_snapshot is not None:
                return cached_snapshot
            merged = self.build_indicator_frame(normalized_symbol, interval=interval, refresh=True)
            current = merged.sort_index()
        else:
            requested_day = _requested_trading_day(end_time)
            day_frame = self._load_day_frame(
                normalized_symbol,
                interval,
                requested_day,
                allow_aligned_cache=True,
            )
            if day_frame is None or day_frame.empty:
                day_frame = self._indicator_frame_for_window(
                    normalized_symbol,
                    interval=interval,
                    start_time=pd.Timestamp(requested_day),
                    end_time=end_time,
                )
            current = _slice_to_timestamp(day_frame, end_time=end_time)
        if current.empty:
            raise ValueError(f"No Alpha Vantage indicator data returned for {normalized_symbol}.")
        if end_time is None:
            latest_trading_day = current.index.max().date()
            current = current.loc[current.index.date == latest_trading_day].copy()

        return _snapshot_from_frame(normalized_symbol, interval, current)

    def load_local_snapshot(
        self,
        symbol: str,
        *,
        trading_day: str | None = None,
        end_time: pd.Timestamp | str | None = None,
    ) -> AlphaVantageIndicatorSnapshot | None:
        """Load a locally available snapshot without triggering a network refresh.

        Args:
            symbol: Ticker symbol to load.
            trading_day: Optional specific trading day to load.
            end_time: Optional checkpoint to slice an already cached trading day.

        Returns:
            A snapshot from local cache/store when available, otherwise ``None``.
        """

        if trading_day is not None and end_time is not None:
            raise ValueError("Provide either trading_day or end_time when loading a local Alpha Vantage snapshot.")

        interval = self.config.interval or DEFAULT_INTERVAL
        normalized_symbol = symbol.upper()

        if end_time is not None:
            requested_day = _requested_trading_day(end_time)
            snapshot = self._load_snapshot_for_trading_day(
                normalized_symbol,
                interval,
                requested_day,
                allow_aligned_cache=True,
            )
            if snapshot is None:
                return None
            current = _slice_to_timestamp(_snapshot_to_frame(snapshot), end_time=end_time)
            if current.empty:
                return None
            return _snapshot_from_frame(normalized_symbol, interval, current)

        if trading_day is not None:
            return self._load_snapshot_for_trading_day(
                normalized_symbol,
                interval,
                trading_day,
                allow_aligned_cache=True,
            )

        return self._load_latest_snapshot(normalized_symbol, interval)

    def build_indicator_frame(
        self,
        symbol: str,
        *,
        interval: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Build the merged aligned indicator frame for one symbol.

        Args:
            symbol: Ticker symbol to load.
            interval: Optional interval override.
            refresh: Whether to force a fresh fetch before merging local history.

        Returns:
            The aligned indicator frame sorted by timestamp.
        """

        normalized_symbol = symbol.upper()
        resolved_interval = interval or self.config.interval or DEFAULT_INTERVAL
        cache_key = (normalized_symbol, resolved_interval)
        if not refresh and cache_key in self._aligned_frame_cache:
            return self._aligned_frame_cache[cache_key].copy()

        if not refresh:
            cached = self._load_cached_aligned_frame(normalized_symbol, resolved_interval)
            if cached is not None and not cached.empty:
                self._aligned_frame_cache[cache_key] = cached.copy()
                return cached

        fetched = self._build_aligned_frame(normalized_symbol, interval=resolved_interval)
        merged = self._merge_with_local_history(normalized_symbol, resolved_interval, fetched)
        self._persist_daily_frames(normalized_symbol, resolved_interval, fetched)
        self._aligned_frame_cache[cache_key] = merged.copy()
        return merged

    def build_feature_frame(
        self,
        symbol: str,
        price_bars: pd.DataFrame,
        *,
        end_time: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Join Alpha Vantage indicators onto 5-minute price bars.

        Args:
            symbol: Ticker symbol being analyzed.
            price_bars: Source 5-minute OHLCV bars.
            end_time: Optional checkpoint to slice both bars and indicators.

        Returns:
            A feature frame ready for technical analysis.
        """

        frame = price_bars.copy().sort_index()
        if end_time is not None:
            normalized_end = _normalize_timestamp(end_time)
            frame = frame.loc[frame.index <= normalized_end]
        if frame.empty:
            raise ValueError(f"No price bars available for {symbol.upper()} at the requested checkpoint.")

        indicator_frame = self._indicator_frame_for_window(
            symbol,
            interval=self.config.interval or DEFAULT_INTERVAL,
            start_time=frame.index.min(),
            end_time=frame.index.max(),
        )
        current_indicators = _slice_to_timestamp(indicator_frame, end_time=end_time)
        if current_indicators.empty:
            raise ValueError(f"No Alpha Vantage indicators available for {symbol.upper()} at the requested checkpoint.")

        frame = frame.join(current_indicators, how="left")

        required = [column for column in ["RSI", "MACDEXT", "MACDEXT_SIGNAL"] if column in frame.columns]
        if required:
            frame = frame.dropna(subset=required)
        if frame.empty:
            raise ValueError(f"Alpha Vantage indicators did not align with the provided 5-minute bars for {symbol.upper()}.")

        frame["return"] = frame["close"].pct_change()
        frame["ma_short"] = _coalesce_columns(frame, ["MAMA", "KAMA", "close"])
        frame["ma_long"] = _coalesce_columns(frame, ["FAMA", "BBANDS_MIDDLE", "KAMA", "close"])
        frame["rsi"] = _series_or_default(frame, "RSI", 50.0).fillna(50.0)
        frame["macd"] = _series_or_default(frame, "MACDEXT", 0.0).fillna(0.0)
        frame["macd_signal"] = _series_or_default(frame, "MACDEXT_SIGNAL", 0.0).fillna(0.0)

        atr_ratio = _series_or_default(frame, "ATR", 0.0) / frame["close"].replace(0, pd.NA)
        frame["rolling_volatility"] = pd.to_numeric(atr_ratio, errors="coerce").fillna(0.0)

        bullish_trend = frame["ma_short"] >= frame["ma_long"]
        bullish_momentum = frame["macd"] >= frame["macd_signal"]
        strong_trend = _series_or_default(frame, "ADX", 0.0) >= 20.0
        positive_roc = _series_or_default(frame, "ROC", 0.0) >= 0.0
        above_sar = frame["close"] >= _series_or_default(frame, "SAR", frame["close"])

        frame["buy_trigger"] = bullish_trend & bullish_momentum & strong_trend & positive_roc & above_sar & (
            frame["rsi"] >= 55.0
        )
        frame["sell_trigger"] = (
            (frame["rsi"] <= 45.0)
            | (frame["macd"] < frame["macd_signal"])
            | (_series_or_default(frame, "ROC", 0.0) < 0.0)
            | (frame["close"] < _series_or_default(frame, "SAR", frame["close"]))
            | (frame["ma_short"] < frame["ma_long"])
        )
        return frame

    def ensure_data_for_window(
        self,
        symbol: str,
        *,
        start_time: pd.Timestamp | str,
        end_time: pd.Timestamp | str,
        required_trading_days: list[str] | None = None,
    ) -> bool:
        """Ensure the requested trading-day windows exist in cache or storage.

        Args:
            symbol: Ticker symbol to prepare.
            start_time: Start of the required window.
            end_time: End of the required window.
            required_trading_days: Optional explicit trading-day list for the window.

        Returns:
            ``True`` when a fetch/persist cycle was needed, otherwise ``False``.
        """
        normalized_symbol = symbol.upper()
        interval = self.config.interval or DEFAULT_INTERVAL
        window_start = _normalize_timestamp(start_time)
        window_end = _normalize_timestamp(end_time)
        missing_days = self._missing_trading_days(
            normalized_symbol,
            interval=interval,
            start_time=window_start,
            end_time=window_end,
            trading_days=required_trading_days,
        )
        if not missing_days:
            print(
                f"[AlphaVantage] Local preload already satisfied for {normalized_symbol} {interval}: "
                f"{len(_trading_days_for_window(window_start, window_end))} trading day(s) available."
            )
            return False

        specs = self.indicator_specs or default_indicator_specs(interval)
        estimated_pause_seconds = max(len(specs) - 1, 0) * self.config.request_pause_seconds
        print(
            f"[AlphaVantage] Missing {len(missing_days)} trading day(s) for {normalized_symbol} {interval}. "
            f"Starting fetch/persist cycle with {len(specs)} indicator request(s). "
            f"Minimum configured pause budget: {estimated_pause_seconds:.1f}s."
        )
        print(f"[AlphaVantage] Missing trading days: {missing_days}")

        try:
            self.build_indicator_frame(normalized_symbol, interval=interval, refresh=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Alpha Vantage backtest data is missing from the DB/cache. "
                "Set ALPHA_VANTAGE_API_KEY so the workflow can retrieve and persist it before replay."
            ) from exc

        unresolved_days = self._missing_trading_days(
            normalized_symbol,
            interval=interval,
            start_time=window_start,
            end_time=window_end,
            trading_days=required_trading_days,
        )
        if unresolved_days:
            raise ValueError(
                f"Alpha Vantage indicators were still missing for {normalized_symbol} trading days: {unresolved_days}"
            )
        print(
            f"[AlphaVantage] Fetch/persist cycle completed for {normalized_symbol} {interval}. "
            f"Resolved {len(missing_days)} trading day(s) into local cache/storage."
        )
        return True

    def ensure_backtest_coverage(
        self,
        symbol: str,
        *,
        source_bars: pd.DataFrame,
        required_trading_days: list[str],
        result_store: ResultStore | None = None,
        database_path: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        """Ensure the local backtest store has day snapshots for the replay window.

        Args:
            symbol: Ticker symbol being prepared for replay.
            source_bars: Full source 5-minute bar history used by the backtest.
            required_trading_days: Trading days that must be available locally.
            result_store: Optional override for the store that should hold the
                preloaded snapshots.
            database_path: Optional database path used only for status messages.
            cache_path: Optional cache path used only for status messages.

        Raises:
            RuntimeError: If required local coverage is still unavailable after
                syncing from cache and attempting one fetch cycle.
            ValueError: If the supplied bar history is empty.
        """

        normalized_symbol = symbol.upper()
        if source_bars.empty:
            raise ValueError(f"No 5-minute bars available for {normalized_symbol}")

        store = result_store or self.result_store
        if store is None:
            raise RuntimeError("Alpha Vantage backtest coverage requires a configured result store.")

        interval = self.config.interval or DEFAULT_INTERVAL
        start_time = pd.Timestamp(source_bars.index.min())
        end_time = pd.Timestamp(source_bars.index.max())
        trading_day_count = len(required_trading_days)
        database_display = (
            str(Path(database_path).resolve()) if database_path is not None else "<not configured>"
        )
        resolved_cache_path = cache_path or self.cache_dir
        cache_display = (
            str(Path(resolved_cache_path).resolve()) if resolved_cache_path is not None else "<not configured>"
        )
        print(
            f"[Workflow] Alpha Vantage backtest preflight: ensuring local storage coverage for {normalized_symbol} "
            f"across {trading_day_count} trading day(s) | db={database_display} | cache={cache_display}"
        )
        self._sync_backtest_snapshots_from_local_cache(
            store,
            normalized_symbol,
            required_trading_days=required_trading_days,
            interval=interval,
        )
        missing_days = self._missing_backtest_snapshot_days(
            store,
            normalized_symbol,
            required_trading_days=required_trading_days,
            interval=interval,
        )
        if missing_days:
            self.ensure_data_for_window(
                normalized_symbol,
                start_time=start_time,
                end_time=end_time,
                required_trading_days=required_trading_days,
            )
            self._sync_backtest_snapshots_from_local_cache(
                store,
                normalized_symbol,
                required_trading_days=required_trading_days,
                interval=interval,
            )
            missing_days = self._missing_backtest_snapshot_days(
                store,
                normalized_symbol,
                required_trading_days=required_trading_days,
                interval=interval,
            )
        if missing_days:
            partial_coverage = self._resolve_backtest_partial_coverage(
                store,
                normalized_symbol,
                required_trading_days=required_trading_days,
                interval=interval,
            )
            if partial_coverage is None:
                raise RuntimeError(
                    "Alpha Vantage backtest data is still missing from the local store after preload. "
                    f"Missing {normalized_symbol} {interval} trading day snapshots: {missing_days}. "
                    f"DB path: {database_display}"
                )
            latest_snapshot, trailing_missing_days = partial_coverage
            latest_timestamp = _normalize_timestamp(latest_snapshot.latest_timestamp)
            print(
                "[Workflow] Alpha Vantage backtest preload accepted partial local coverage: "
                f"using the latest available snapshot through {latest_timestamp.isoformat()} "
                f"and skipping trailing unavailable trading day(s): {trailing_missing_days}."
            )
            return
        print(
            "[Workflow] Alpha Vantage backtest preload finished: required indicator snapshots and 1-hour chunks "
            f"are available in the local store at {database_display}."
        )

    def _indicator_frame_for_window(
        self,
        symbol: str,
        *,
        interval: str,
        start_time: pd.Timestamp | str,
        end_time: pd.Timestamp | str,
    ) -> pd.DataFrame:
        normalized_symbol = symbol.upper()
        window_start = _normalize_timestamp(start_time)
        window_end = _normalize_timestamp(end_time)
        trading_days = _trading_days_for_window(window_start, window_end)
        daily_frames: list[pd.DataFrame] = []
        missing_days: list[str] = []

        for trading_day in trading_days:
            day_frame = self._load_day_frame(
                normalized_symbol,
                interval,
                trading_day,
                allow_aligned_cache=True,
            )
            if day_frame is None or day_frame.empty:
                missing_days.append(trading_day)
                continue
            daily_frames.append(day_frame)

        if missing_days:
            merged = self.build_indicator_frame(normalized_symbol, interval=interval, refresh=True)
            for trading_day in missing_days:
                day_frame = _frame_for_trading_day(merged, trading_day)
                if day_frame.empty:
                    continue
                self._store_day_frame(normalized_symbol, interval, trading_day, day_frame)
                daily_frames.append(day_frame)

        if not daily_frames:
            return pd.DataFrame()
        combined = pd.concat(daily_frames, axis=0).sort_index()
        return combined.loc[(combined.index >= window_start) & (combined.index <= window_end)].copy()

    def _missing_backtest_snapshot_days(
        self,
        result_store: ResultStore,
        symbol: str,
        *,
        required_trading_days: list[str],
        interval: str,
    ) -> list[str]:
        return [
            trading_day
            for trading_day in required_trading_days
            if result_store.load_alpha_vantage_indicator_snapshot(
                symbol,
                trading_day=trading_day,
                interval=interval,
            )
            is None
        ]

    def _sync_backtest_snapshots_from_local_cache(
        self,
        result_store: ResultStore,
        symbol: str,
        *,
        required_trading_days: list[str],
        interval: str,
    ) -> list[str]:
        hydrated_days: list[str] = []
        for trading_day in required_trading_days:
            existing = result_store.load_alpha_vantage_indicator_snapshot(
                symbol,
                trading_day=trading_day,
                interval=interval,
            )
            if existing is not None:
                continue
            snapshot = self.load_local_snapshot(symbol, trading_day=trading_day)
            if snapshot is None:
                continue
            result_store.save_alpha_vantage_indicator_snapshot(snapshot)
            hydrated_days.append(trading_day)

        if hydrated_days:
            print(
                "[Workflow] Alpha Vantage backtest cache sync: hydrated "
                f"{len(hydrated_days)} trading day snapshot(s) from local cache/storage into the result store. "
                f"Trading days: {hydrated_days}"
            )
        return hydrated_days

    def _resolve_backtest_partial_coverage(
        self,
        result_store: ResultStore,
        symbol: str,
        *,
        required_trading_days: list[str],
        interval: str,
    ) -> tuple[AlphaVantageIndicatorSnapshot, list[str]] | None:
        first_missing_index: int | None = None
        latest_covered_snapshot: AlphaVantageIndicatorSnapshot | None = None

        for index, trading_day in enumerate(required_trading_days):
            snapshot = result_store.load_alpha_vantage_indicator_snapshot(
                symbol,
                trading_day=trading_day,
                interval=interval,
            )
            if snapshot is None:
                if first_missing_index is None:
                    first_missing_index = index
                continue
            if first_missing_index is not None:
                return None
            latest_covered_snapshot = snapshot

        if first_missing_index is None or first_missing_index == 0 or latest_covered_snapshot is None:
            return None

        return latest_covered_snapshot, required_trading_days[first_missing_index:]

    def _missing_trading_days(
        self,
        symbol: str,
        *,
        interval: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        trading_days: list[str] | None = None,
    ) -> list[str]:
        missing_days: list[str] = []
        for trading_day in trading_days or _trading_days_for_window(start_time, end_time):
            day_frame = self._load_day_frame(
                symbol,
                interval,
                trading_day,
                allow_aligned_cache=True,
            )
            if day_frame is None or day_frame.empty:
                missing_days.append(trading_day)
        return missing_days

    def _load_cached_aligned_frame(self, symbol: str, interval: str) -> pd.DataFrame | None:
        if self.cache_dir is None:
            return None
        interval_dir = self._interval_cache_dir(symbol, interval)
        if not interval_dir.exists():
            return None
        frames: list[pd.DataFrame] = []
        for daily_file in sorted(interval_dir.glob("*.csv")):
            frame = self._read_frame(daily_file)
            if frame.empty:
                continue
            trading_day = daily_file.stem
            self._cache_day_frame((symbol, interval, trading_day), frame)
            frames.append(frame)
        if not frames:
            return None
        return pd.concat(frames, axis=0).sort_index()

    def _load_aligned_frame_from_store(self, symbol: str, interval: str) -> pd.DataFrame | None:
        if self.result_store is None:
            return None
        snapshots = self.result_store.load_alpha_vantage_indicator_snapshots(symbol, interval=interval)
        if not snapshots:
            return None
        frames: list[pd.DataFrame] = []
        for snapshot in snapshots:
            frame = _snapshot_to_frame(snapshot)
            if frame.empty:
                continue
            self._cache_day_frame((symbol, interval, snapshot.trading_day), frame)
            frames.append(frame)
        if not frames:
            return None
        return pd.concat(frames, axis=0).sort_index()

    def _merge_with_local_history(self, symbol: str, interval: str, fresh: pd.DataFrame) -> pd.DataFrame:
        merged = fresh.sort_index().copy()
        if merged.empty:
            return merged

        existing_frames = [
            self._aligned_frame_cache.get((symbol, interval)),
            self._load_aligned_frame_from_store(symbol, interval),
            self._load_cached_aligned_frame(symbol, interval),
        ]
        for existing in existing_frames:
            if existing is None or existing.empty:
                continue
            merged = merged.combine_first(existing.sort_index())
        return merged.sort_index()

    def _load_day_frame(
        self,
        symbol: str,
        interval: str,
        trading_day: str,
        *,
        allow_aligned_cache: bool,
    ) -> pd.DataFrame | None:
        cache_key = (symbol, interval, trading_day)
        memory_value = self._daily_frame_cache.get(cache_key)
        if memory_value is not None:
            return memory_value.copy()

        store_value = self._load_day_frame_from_store(cache_key)
        if store_value is not None:
            cached = self._cache_day_frame(cache_key, store_value)
            self._persist_day_frame_to_disk(cache_key, cached)
            return cached

        disk_value = self._load_day_frame_from_disk(cache_key)
        if disk_value is not None:
            return self._cache_day_frame(cache_key, disk_value)

        if allow_aligned_cache:
            aligned = self._aligned_frame_cache.get((symbol, interval))
            if aligned is not None and not aligned.empty:
                frame = _frame_for_trading_day(aligned, trading_day)
                if not frame.empty:
                    self._store_day_frame(symbol, interval, trading_day, frame)
                    return self._daily_frame_cache[(symbol, interval, trading_day)].copy()
        return None

    def _persist_daily_frames(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        for trading_day, bucket in frame.groupby(frame.index.date):
            self._store_day_frame(symbol, interval, trading_day.isoformat(), bucket)

    def _store_day_frame(
        self,
        symbol: str,
        interval: str,
        trading_day: str,
        frame: pd.DataFrame,
        *,
        persist_store: bool = True,
    ) -> None:
        cache_key = (symbol, interval, trading_day)
        normalized = self._cache_day_frame(cache_key, frame)
        if persist_store and not normalized.empty:
            self._persist_day_frame_to_store(cache_key, normalized)
        self._persist_day_frame_to_disk(cache_key, normalized)

    def _daily_cache_path(self, symbol: str, interval: str, trading_day: str) -> Path:
        return self._interval_cache_dir(symbol, interval) / f"{trading_day}.csv"

    def _interval_cache_dir(self, symbol: str, interval: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("Alpha Vantage cache directory is not configured.")
        return self.cache_dir / symbol.upper() / interval

    def _read_frame(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame.sort_index()

    def _cache_day_frame(self, key: tuple[str, str, str], frame: pd.DataFrame) -> pd.DataFrame:
        """Store one trading-day frame in memory and return a defensive copy."""

        normalized = frame.sort_index().copy()
        self._daily_frame_cache[key] = normalized.copy()
        return normalized.copy()

    def _load_day_frame_from_store(self, key: Hashable) -> pd.DataFrame | None:
        if self.result_store is None:
            return None
        symbol, interval, trading_day = key
        snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(
            symbol,
            trading_day=trading_day,
            interval=interval,
        )
        if snapshot is None:
            return None
        return _snapshot_to_frame(snapshot)

    def _load_day_frame_from_disk(self, key: Hashable) -> pd.DataFrame | None:
        if self.cache_dir is None:
            return None
        symbol, interval, trading_day = key
        cache_file = self._daily_cache_path(symbol, interval, trading_day)
        if not cache_file.exists():
            return None
        return self._read_frame(cache_file)

    def _persist_day_frame_to_store(self, key: Hashable, frame: pd.DataFrame) -> None:
        if self.result_store is None or frame.empty:
            return
        symbol, interval, _ = key
        self.result_store.save_alpha_vantage_indicator_snapshot(_snapshot_from_frame(symbol, interval, frame))

    def _persist_day_frame_to_disk(self, key: Hashable, frame: pd.DataFrame) -> None:
        if self.cache_dir is None:
            return
        symbol, interval, trading_day = key
        cache_file = self._daily_cache_path(symbol, interval, trading_day)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_file, index_label="timestamp")

    def _build_aligned_frame(self, symbol: str, *, interval: str) -> pd.DataFrame:
        self.config.require()
        frames: list[pd.DataFrame] = []
        specs = self.indicator_specs or default_indicator_specs(interval)
        print(
            f"[AlphaVantage] Building aligned indicator frame for {symbol.upper()} {interval}. "
            f"Indicators to fetch: {len(specs)}."
        )

        for index, spec in enumerate(specs):
            print(
                f"[AlphaVantage] Fetching indicator {index + 1}/{len(specs)} for {symbol.upper()}: "
                f"{spec.name}."
            )
            raw = self._fetch_indicator(symbol, spec)
            frames.append(indicator_to_dataframe(spec.name, raw))
            if index < len(specs) - 1 and self.config.request_pause_seconds > 0:
                print(
                    f"[AlphaVantage] Waiting {self.config.request_pause_seconds:.1f}s before next indicator request."
                )
                self.sleep_fn(self.config.request_pause_seconds)

        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, axis=1).sort_index()
        print(
            f"[AlphaVantage] Completed aligned frame for {symbol.upper()} {interval}: "
            f"rows={len(merged)}, columns={len(merged.columns)}."
        )
        return merged

    def _fetch_indicator(self, symbol: str, spec: IndicatorSpec) -> dict[str, Any]:
        params = {"symbol": symbol, "apikey": self.config.api_key, **spec.params}

        for attempt in range(1, self.config.max_retries + 1):
            print(
                f"[AlphaVantage] Requesting {spec.name} for {symbol.upper()} "
                f"(attempt {attempt}/{self.config.max_retries})."
            )
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
                    retry_sleep = self.config.request_pause_seconds * attempt
                    print(
                        f"[AlphaVantage] Rate-limit/info response for {spec.name}: {message} "
                        f"Retrying after {retry_sleep:.1f}s."
                    )
                    self.sleep_fn(retry_sleep)
                    continue
                raise AlphaVantageLimitError(f"Rate limit hit for {spec.name}: {message}")

            print(f"[AlphaVantage] Received {spec.name} for {symbol.upper()}.")
            return data

        raise AlphaVantageLimitError(f"Rate limit hit for {spec.name}.")

    def _load_latest_snapshot(self, symbol: str, interval: str) -> AlphaVantageIndicatorSnapshot | None:
        if self.result_store is not None:
            snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(symbol, interval=interval)
            if snapshot is not None:
                return snapshot

        aligned = self._aligned_frame_cache.get((symbol, interval))
        if aligned is not None and not aligned.empty:
            latest_trading_day = aligned.index.max().date().isoformat()
            latest_frame = _frame_for_trading_day(aligned, latest_trading_day)
            if not latest_frame.empty:
                return _snapshot_from_frame(symbol, interval, latest_frame)

        cached = self._load_cached_aligned_frame(symbol, interval)
        if cached is None or cached.empty:
            return None
        latest_trading_day = cached.index.max().date().isoformat()
        latest_frame = _frame_for_trading_day(cached, latest_trading_day)
        if latest_frame.empty:
            return None
        return _snapshot_from_frame(symbol, interval, latest_frame)

    def _load_snapshot_for_trading_day(
        self,
        symbol: str,
        interval: str,
        trading_day: str,
        *,
        allow_aligned_cache: bool,
    ) -> AlphaVantageIndicatorSnapshot | None:
        frame = self._load_day_frame(
            symbol,
            interval,
            trading_day,
            allow_aligned_cache=allow_aligned_cache,
        )
        if frame is None or frame.empty:
            return None
        return _snapshot_from_frame(symbol, interval, frame)


def indicator_to_dataframe(name: str, data: dict[str, Any]) -> pd.DataFrame:
    """Convert one Alpha Vantage payload into a normalized dataframe."""

    series = extract_technical_series(data)
    frame = pd.DataFrame.from_dict(series, orient="index")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rename_map = {column: normalize_indicator_column(name, column, len(frame.columns)) for column in frame.columns}
    return frame.rename(columns=rename_map)


def extract_technical_series(data: dict[str, Any]) -> dict[str, Any]:
    """Extract the technical-analysis mapping from a raw API payload."""

    technical_key = next((key for key in data if "Technical Analysis" in key), None)
    if technical_key is None:
        raise ValueError(f"Could not find technical analysis data in Alpha Vantage response: {data}")
    return data[technical_key]


def normalize_indicator_column(indicator_name: str, column_name: str, column_count: int) -> str:
    """Normalize raw Alpha Vantage column names into stable internal names."""

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


def _snapshot_from_frame(
    symbol: str,
    interval: str,
    frame: pd.DataFrame,
) -> AlphaVantageIndicatorSnapshot:
    normalized = frame.sort_index().copy()
    latest_timestamp = normalized.index.max()
    trading_day = latest_timestamp.date().isoformat()
    records = _build_records(normalized)
    chunks = _build_hourly_chunks(normalized, records)
    return AlphaVantageIndicatorSnapshot(
        symbol=symbol.upper(),
        interval=interval,
        trading_day=trading_day,
        latest_timestamp=_format_timestamp(latest_timestamp),
        indicator_columns=list(normalized.columns),
        rows=records,
        hourly_chunks=chunks,
        latest_hour_chunk=chunks[-1] if chunks else None,
    )


def _snapshot_to_frame(snapshot: AlphaVantageIndicatorSnapshot) -> pd.DataFrame:
    if not snapshot.rows:
        return pd.DataFrame(columns=snapshot.indicator_columns)
    frame = pd.DataFrame(snapshot.rows)
    if frame.empty:
        return pd.DataFrame(columns=snapshot.indicator_columns)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    drop_columns = [column for column in ["threshold_hits"] if column in frame.columns]
    if drop_columns:
        frame = frame.drop(columns=drop_columns)
    frame = frame.set_index("time")
    ordered_columns = [column for column in snapshot.indicator_columns if column in frame.columns]
    if ordered_columns:
        frame = frame[ordered_columns]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index.name = None
    return frame.sort_index()


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


def _normalize_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _slice_to_timestamp(frame: pd.DataFrame, *, end_time: pd.Timestamp | str | None) -> pd.DataFrame:
    current = frame.sort_index()
    if end_time is None:
        return current
    normalized_end = _normalize_timestamp(end_time)
    return current.loc[current.index <= normalized_end].copy()


def _requested_trading_day(end_time: pd.Timestamp | str | None) -> str:
    if end_time is None:
        return pd.Timestamp.now(tz="UTC").date().isoformat()
    return _normalize_timestamp(end_time).date().isoformat()


def _trading_days_for_window(start_time: pd.Timestamp | str, end_time: pd.Timestamp | str) -> list[str]:
    start_day = _normalize_timestamp(start_time).normalize()
    end_day = _normalize_timestamp(end_time).normalize()
    return [timestamp.date().isoformat() for timestamp in pd.date_range(start_day, end_day, freq="1D", tz="UTC")]


def _frame_for_trading_day(frame: pd.DataFrame, trading_day: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    target_day = pd.Timestamp(trading_day).date()
    return frame.loc[frame.index.date == target_day].copy()


def _series_or_default(frame: pd.DataFrame, column: str, default: float | pd.Series) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    if isinstance(default, pd.Series):
        return pd.to_numeric(default, errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _coalesce_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    combined: pd.Series | None = None
    for column in columns:
        candidate = _series_or_default(frame, column, pd.Series(pd.NA, index=frame.index))
        combined = candidate if combined is None else combined.combine_first(candidate)
    if combined is None:
        return pd.Series(pd.NA, index=frame.index, dtype="float64")
    return pd.to_numeric(combined, errors="coerce")
