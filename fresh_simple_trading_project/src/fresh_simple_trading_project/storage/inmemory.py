"""In-memory storage backends used by tests and lightweight workflows."""

from __future__ import annotations

import pandas as pd

from ..models import (
    AlphaVantageIndicatorSnapshot,
    BacktestSummary,
    ForecastSnapshot,
    NewsArticle,
    PerformanceSnapshot,
    WorkflowResult,
)
from .protocols import StorageRef
from .serialization import (
    _news_article_key,
    _news_article_sort_key,
    _normalize_news_timestamp,
    _normalize_timestamp,
    _snapshot_hour_chunk,
)


class InMemoryResultStore:
    """Store workflow results in memory for tests and local use."""

    def __init__(self) -> None:
        self.workflow_runs: list[WorkflowResult] = []
        self.backtest_runs: list[BacktestSummary] = []
        self.last_processed: dict[str, pd.Timestamp] = {}
        self.raw_artifacts_by_symbol: dict[str, dict[str, StorageRef]] = {}
        self.alpha_vantage_indicator_snapshots: list[AlphaVantageIndicatorSnapshot] = []
        self.forecast_snapshots: list[ForecastSnapshot] = []
        self.cached_news: dict[tuple[str, str, str], NewsArticle] = {}

    def save_workflow_run(
        self,
        result: WorkflowResult,
        *,
        raw_artifacts: dict[str, StorageRef] | None = None,
    ) -> None:
        self.workflow_runs.append(result)
        if raw_artifacts:
            self.raw_artifacts_by_symbol[result.symbol] = dict(raw_artifacts)

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        self.backtest_runs.append(summary)

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        self.last_processed[symbol] = pd.Timestamp(timestamp)

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        return self.last_processed.get(symbol)

    def save_alpha_vantage_indicator_snapshot(self, snapshot: AlphaVantageIndicatorSnapshot) -> None:
        normalized_symbol = snapshot.symbol.upper()
        normalized_interval = snapshot.interval
        normalized_trading_day = snapshot.trading_day
        self.alpha_vantage_indicator_snapshots = [
            existing
            for existing in self.alpha_vantage_indicator_snapshots
            if not (
                existing.symbol.upper() == normalized_symbol
                and existing.interval == normalized_interval
                and existing.trading_day == normalized_trading_day
            )
        ]
        self.alpha_vantage_indicator_snapshots.append(snapshot)

    def load_alpha_vantage_indicator_snapshot(
        self,
        symbol: str,
        *,
        trading_day: str | None = None,
        interval: str | None = None,
    ) -> AlphaVantageIndicatorSnapshot | None:
        normalized_symbol = symbol.upper()
        candidates = [
            snapshot
            for snapshot in self.alpha_vantage_indicator_snapshots
            if snapshot.symbol.upper() == normalized_symbol
            and (trading_day is None or snapshot.trading_day == trading_day)
            and (interval is None or snapshot.interval == interval)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda snapshot: (snapshot.trading_day, snapshot.latest_timestamp))

    def load_alpha_vantage_indicator_snapshots(
        self,
        symbol: str,
        *,
        interval: str | None = None,
    ) -> list[AlphaVantageIndicatorSnapshot]:
        normalized_symbol = symbol.upper()
        candidates = [
            snapshot
            for snapshot in self.alpha_vantage_indicator_snapshots
            if snapshot.symbol.upper() == normalized_symbol
            and (interval is None or snapshot.interval == interval)
        ]
        return sorted(candidates, key=lambda snapshot: (snapshot.trading_day, snapshot.latest_timestamp))

    def load_alpha_vantage_hour_chunk(
        self,
        symbol: str,
        *,
        as_of: pd.Timestamp | str | None = None,
        trading_day: str | None = None,
        interval: str | None = None,
    ):
        target_trading_day = trading_day
        if target_trading_day is None and as_of is not None:
            target_trading_day = _normalize_timestamp(as_of).date().isoformat()
        snapshot = self.load_alpha_vantage_indicator_snapshot(
            symbol,
            trading_day=target_trading_day,
            interval=interval,
        )
        if snapshot is None:
            return None
        return _snapshot_hour_chunk(snapshot, as_of=as_of)

    def save_forecast_snapshot(self, snapshot: ForecastSnapshot) -> None:
        self.forecast_snapshots.append(snapshot)

    def load_latest_forecast(
        self,
        symbol: str,
        *,
        as_of: pd.Timestamp | None = None,
    ) -> ForecastSnapshot | None:
        normalized_symbol = symbol.upper()
        as_of_timestamp = None if as_of is None else _normalize_timestamp(as_of)
        candidates = [
            snapshot
            for snapshot in self.forecast_snapshots
            if snapshot.symbol.upper() == normalized_symbol
            and (as_of_timestamp is None or _normalize_timestamp(snapshot.valid_until) >= as_of_timestamp)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda snapshot: _normalize_timestamp(snapshot.generated_at))

    def load_latest_performance(self, symbol: str) -> PerformanceSnapshot | None:
        normalized_symbol = symbol.upper()
        for result in reversed(self.workflow_runs):
            if result.symbol.upper() != normalized_symbol or result.performance is None:
                continue
            return result.performance
        return None

    def count_executed_trades(self, symbol: str) -> int:
        normalized_symbol = symbol.upper()
        return sum(
            1
            for result in self.workflow_runs
            if result.symbol.upper() == normalized_symbol and result.execution.executed
        )

    def save_retrieved_news(self, symbol: str, query: str, articles: list[NewsArticle]) -> None:
        normalized_symbol = symbol.upper()
        for article in articles:
            headline = article.headline.strip()
            if not headline:
                continue
            key = (normalized_symbol, query, _news_article_key(article))
            self.cached_news[key] = NewsArticle(
                headline=headline,
                summary=article.summary,
                source=article.source,
                url=article.url,
                published_at=_normalize_news_timestamp(article.published_at),
                provider=article.provider,
                primary_ticker=article.primary_ticker,
                primary_ticker_relevance=article.primary_ticker_relevance,
            )

    def load_retrieved_news(
        self,
        symbol: str,
        query: str,
        *,
        limit: int = 100,
        published_at_lte: pd.Timestamp | str | None = None,
    ) -> list[NewsArticle]:
        normalized_symbol = symbol.upper()
        articles = [
            article
            for (cached_symbol, cached_query, _), article in self.cached_news.items()
            if cached_symbol == normalized_symbol and cached_query == query
        ]
        if published_at_lte is not None:
            cutoff = _normalize_news_timestamp(str(published_at_lte))
            articles = [
                article for article in articles if _normalize_news_timestamp(article.published_at) <= cutoff
            ]
        return sorted(articles, key=_news_article_sort_key, reverse=True)[: max(1, limit)]
