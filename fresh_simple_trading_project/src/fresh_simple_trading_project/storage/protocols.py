"""Storage protocols and references shared across persistence adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..models import (
    AlphaVantageIndicatorSnapshot,
    BacktestSummary,
    ForecastSnapshot,
    IndicatorHourChunk,
    NewsArticle,
    PerformanceSnapshot,
    WorkflowResult,
)


@dataclass(frozen=True)
class StorageRef:
    """Reference to a persisted raw artifact."""

    uri: str
    kind: str
    content_type: str | None = None


class RawStore(Protocol):
    """Protocol for raw artifact stores."""

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        """Persist raw bar data and return its storage reference."""
        ...

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        """Persist raw news data and return its storage reference."""
        ...


class ResultStore(Protocol):
    """Protocol for result stores that persist workflow state and summaries."""

    def save_workflow_run(
        self,
        result: WorkflowResult,
        *,
        raw_artifacts: dict[str, StorageRef] | None = None,
    ) -> None:
        """Persist one workflow result and optional raw-artifact references."""
        ...

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        """Persist one aggregated backtest summary."""
        ...

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        """Persist the last processed checkpoint for ``symbol``."""
        ...

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        """Load the last processed checkpoint for ``symbol``."""
        ...

    def save_alpha_vantage_indicator_snapshot(self, snapshot: AlphaVantageIndicatorSnapshot) -> None:
        """Persist an Alpha Vantage indicator snapshot."""
        ...

    def load_alpha_vantage_indicator_snapshot(
        self,
        symbol: str,
        *,
        trading_day: str | None = None,
        interval: str | None = None,
    ) -> AlphaVantageIndicatorSnapshot | None:
        """Load one Alpha Vantage indicator snapshot."""
        ...

    def load_alpha_vantage_indicator_snapshots(
        self,
        symbol: str,
        *,
        interval: str | None = None,
    ) -> list[AlphaVantageIndicatorSnapshot]:
        """Load all stored Alpha Vantage snapshots for ``symbol``."""
        ...

    def load_alpha_vantage_hour_chunk(
        self,
        symbol: str,
        *,
        as_of: pd.Timestamp | str | None = None,
        trading_day: str | None = None,
        interval: str | None = None,
    ) -> IndicatorHourChunk | None:
        """Load the relevant Alpha Vantage hour chunk for a checkpoint."""
        ...

    def save_forecast_snapshot(self, snapshot: ForecastSnapshot) -> None:
        """Persist a HOLD forecast snapshot."""
        ...

    def load_latest_forecast(
        self,
        symbol: str,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> ForecastSnapshot | None:
        """Load the latest valid forecast for ``symbol``."""
        ...

    def load_latest_performance(self, symbol: str) -> PerformanceSnapshot | None:
        """Load the latest persisted performance snapshot for ``symbol``."""
        ...

    def count_executed_trades(self, symbol: str) -> int:
        """Count executed trades recorded for ``symbol``."""
        ...

    def save_retrieved_news(self, symbol: str, query: str, articles: list[NewsArticle]) -> None:
        """Persist retrieved news articles for a query."""
        ...

    def save_news_query_fetch(
        self,
        symbol: str,
        query: str,
        *,
        provider: str,
        fetch_bucket: str,
    ) -> None:
        """Record that a provider/query fetch already happened for one bucket."""
        ...

    def has_news_query_fetch(
        self,
        symbol: str,
        query: str,
        *,
        provider: str,
        fetch_bucket: str,
    ) -> bool:
        """Report whether a provider/query fetch already happened for one bucket."""
        ...

    def load_retrieved_news(
        self,
        symbol: str,
        query: str,
        *,
        limit: int = 100,
        published_at_lte: pd.Timestamp | str | None = None,
    ) -> list[NewsArticle]:
        """Load cached news articles for a query."""
        ...
