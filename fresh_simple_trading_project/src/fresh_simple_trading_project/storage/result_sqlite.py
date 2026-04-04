"""SQLAlchemy-backed result storage, primarily used with SQLite."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from ..models import AlphaVantageIndicatorSnapshot, BacktestSummary, NewsArticle, PerformanceSnapshot, WorkflowResult
from ..utils import timestamp_slug
from .protocols import StorageRef
from .serialization import (
    _alpha_vantage_snapshot_from_payload,
    _forecast_from_payload,
    _forecast_payload,
    _news_article_key,
    _normalize_news_timestamp,
    _normalize_timestamp,
    _performance_from_payload,
    _performance_payload,
    _snapshot_hour_chunk,
    _timestamp_text,
    _utc_now_text,
)


class SQLiteResultStore:
    """Persist workflow state and snapshots through SQLAlchemy."""

    def __init__(
        self,
        database: Path | str,
        *,
        engine_factory: Callable[[str], Engine] | None = None,
    ) -> None:
        database_url = _coerce_database_url(database)
        if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
            database_path = Path(database_url.removeprefix("sqlite:///"))
            database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url
        self._metadata = sa.MetaData()
        self.workflow_runs_table = sa.Table(
            "workflow_runs",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("created_at", sa.String(32), nullable=False),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("action", sa.String(16), nullable=False),
            sa.Column("quantity", sa.Integer, nullable=False),
            sa.Column("executed", sa.Boolean, nullable=False, server_default=sa.false()),
            sa.Column("confidence", sa.Float, nullable=False),
            sa.Column("latest_price", sa.Float, nullable=False),
            sa.Column("sentiment_score", sa.Float, nullable=False),
            sa.Column("risk_score", sa.Float, nullable=False),
            sa.Column("metadata", sa.Text, nullable=False),
        )
        self.backtest_runs_table = sa.Table(
            "backtest_runs",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("created_at", sa.String(32), nullable=False),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("initial_cash", sa.Float, nullable=False),
            sa.Column("ending_cash", sa.Float, nullable=False),
            sa.Column("total_return_pct", sa.Float, nullable=False),
            sa.Column("benchmark_return_pct", sa.Float, nullable=False),
            sa.Column("max_drawdown_pct", sa.Float, nullable=False),
            sa.Column("sharpe_ratio", sa.Float, nullable=False),
            sa.Column("trade_count", sa.Integer, nullable=False),
            sa.Column("win_rate", sa.Float, nullable=False),
            sa.Column("signal_accuracy", sa.Float, nullable=False),
        )
        self.workflow_state_table = sa.Table(
            "workflow_state",
            self._metadata,
            sa.Column("symbol", sa.String(32), primary_key=True),
            sa.Column("last_processed_at", sa.String(64), nullable=False),
        )
        self.alpha_vantage_indicator_snapshots_table = sa.Table(
            "alpha_vantage_indicator_snapshots",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("created_at", sa.String(32), nullable=False),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("interval", sa.String(16), nullable=False),
            sa.Column("trading_day", sa.String(16), nullable=False),
            sa.Column("latest_timestamp", sa.String(32), nullable=False),
            sa.Column("latest_hour_slot_start", sa.String(32), nullable=True),
            sa.Column("latest_hour_slot_end", sa.String(32), nullable=True),
            sa.Column("payload", sa.Text, nullable=False),
        )
        self.forecast_snapshots_table = sa.Table(
            "forecast_snapshots",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("generated_at", sa.String(32), nullable=False),
            sa.Column("valid_until", sa.String(32), nullable=False),
            sa.Column("trend_bias", sa.String(32), nullable=False, server_default="neutral"),
            sa.Column("summary", sa.Text, nullable=True),
            sa.Column("confidence", sa.Float, nullable=False, server_default="0"),
            sa.Column("payload", sa.Text, nullable=False),
            sa.Index("ix_forecast_snapshots_symbol_generated_at", "symbol", "generated_at"),
            sa.Index("ix_forecast_snapshots_symbol_valid_until", "symbol", "valid_until"),
        )
        self.retrieved_news_table = sa.Table(
            "retrieved_news",
            self._metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("symbol", sa.String(32), nullable=False),
            # Keep the cached query indexable on SQL Server, which rejects VARCHAR(MAX)
            # columns in unique constraints and indexed key columns.
            sa.Column("query", sa.String(512), nullable=False),
            sa.Column("article_key", sa.String(1024), nullable=False),
            sa.Column("headline", sa.Text, nullable=False),
            sa.Column("summary", sa.Text, nullable=False),
            sa.Column("source", sa.String(255), nullable=False),
            sa.Column("url", sa.Text, nullable=False),
            sa.Column("published_at", sa.String(32), nullable=False),
            sa.Column("provider", sa.String(64), nullable=False, server_default=""),
            sa.Column("primary_ticker", sa.String(32), nullable=True),
            sa.Column("primary_ticker_relevance", sa.Float, nullable=True),
            sa.Column("fetched_at", sa.String(32), nullable=False),
            sa.UniqueConstraint("symbol", "query", "article_key", name="uq_retrieved_news_symbol_query_article"),
            sa.Index("ix_retrieved_news_symbol_query_published_at", "symbol", "query", "published_at"),
            sa.Index("ix_retrieved_news_published_at", "published_at"),
        )
        create_engine = engine_factory or _default_sqlalchemy_engine_factory
        self.engine = create_engine(database_url)
        self._metadata.create_all(self.engine)
        self._migrate_workflow_runs_table()
        self._migrate_retrieved_news_table()

    def save_workflow_run(
        self,
        result: WorkflowResult,
        *,
        raw_artifacts: dict[str, StorageRef] | None = None,
    ) -> None:
        metadata = {
            "analysis_interval": "5min",
            "loop_interval": "1h",
            "analysis_notes": result.analysis.notes,
            "technical_agent_summary": result.analysis.llm_summary,
            "news_agent_summary": result.retrieval.summary_note,
            "critical_news": result.retrieval.critical_news,
            "risk_warnings": result.risk.warnings,
            "risk_agent_summary": result.risk.summary_note,
            "headlines": result.retrieval.headline_summary,
            "decision_rationale": result.decision.rationale,
            "execution_status": result.execution.status,
            "protective_order_ids": result.execution.protective_order_ids,
        }
        if result.account is not None:
            metadata["account"] = {
                "cash": result.account.cash,
                "position_qty": result.account.position_qty,
                "market_open": result.account.market_open,
                "avg_entry_price": result.account.avg_entry_price,
                "realized_profit": result.account.realized_profit,
                "trade_count": result.account.trade_count,
            }
        if result.performance is not None:
            metadata["performance"] = _performance_payload(result.performance)
        if raw_artifacts:
            metadata["raw_artifacts"] = {
                name: {
                    "uri": artifact.uri,
                    "kind": artifact.kind,
                    "content_type": artifact.content_type,
                }
                for name, artifact in raw_artifacts.items()
            }
        if result.alpha_vantage_indicator_snapshot is not None:
            latest_hour_chunk = result.alpha_vantage_indicator_snapshot.latest_hour_chunk
            metadata["alpha_vantage_indicator_snapshot"] = {
                "trading_day": result.alpha_vantage_indicator_snapshot.trading_day,
                "latest_timestamp": result.alpha_vantage_indicator_snapshot.latest_timestamp,
                "indicator_columns": result.alpha_vantage_indicator_snapshot.indicator_columns,
                "row_count": len(result.alpha_vantage_indicator_snapshot.rows),
                "latest_hour_slot": None
                if latest_hour_chunk is None
                else {
                    "slot_start": latest_hour_chunk.slot_start,
                    "slot_end": latest_hour_chunk.slot_end,
                    "row_count": len(latest_hour_chunk.rows),
                },
            }
        if result.previous_forecast is not None:
            metadata["previous_forecast"] = _forecast_payload(result.previous_forecast)
        if result.hold_forecast is not None:
            metadata["hold_forecast"] = _forecast_payload(result.hold_forecast)
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(self.workflow_runs_table).values(
                    created_at=timestamp_slug(),
                    symbol=result.symbol,
                    action=result.decision.action.value,
                    quantity=result.decision.quantity,
                    executed=result.execution.executed,
                    confidence=result.decision.confidence,
                    latest_price=result.analysis.latest_price,
                    sentiment_score=result.retrieval.sentiment_score,
                    risk_score=result.risk.risk_score,
                    metadata=json.dumps(metadata),
                )
            )

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(self.backtest_runs_table).values(
                    created_at=timestamp_slug(),
                    symbol=summary.symbol,
                    initial_cash=summary.initial_cash,
                    ending_cash=summary.ending_cash,
                    total_return_pct=summary.total_return_pct,
                    benchmark_return_pct=summary.benchmark_return_pct,
                    max_drawdown_pct=summary.max_drawdown_pct,
                    sharpe_ratio=summary.sharpe_ratio,
                    trade_count=summary.trade_count,
                    win_rate=summary.win_rate,
                    signal_accuracy=summary.signal_accuracy,
                )
            )

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        normalized_timestamp = pd.Timestamp(timestamp).isoformat()
        with self.engine.begin() as connection:
            update_result = connection.execute(
                sa.update(self.workflow_state_table)
                .where(self.workflow_state_table.c.symbol == symbol)
                .values(last_processed_at=normalized_timestamp)
            )
            if update_result.rowcount == 0:
                connection.execute(
                    sa.insert(self.workflow_state_table).values(
                        symbol=symbol,
                        last_processed_at=normalized_timestamp,
                    )
                )

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        with self.engine.begin() as connection:
            row = connection.execute(
                sa.select(self.workflow_state_table.c.last_processed_at).where(
                    self.workflow_state_table.c.symbol == symbol
                )
            ).first()
        if row is None or not row[0]:
            return None
        return pd.Timestamp(row[0])

    def save_alpha_vantage_indicator_snapshot(self, snapshot: AlphaVantageIndicatorSnapshot) -> None:
        latest_hour_chunk = snapshot.latest_hour_chunk
        with self.engine.begin() as connection:
            existing = connection.execute(
                sa.select(self.alpha_vantage_indicator_snapshots_table.c.id)
                .where(self.alpha_vantage_indicator_snapshots_table.c.symbol == snapshot.symbol.upper())
                .where(self.alpha_vantage_indicator_snapshots_table.c.interval == snapshot.interval)
                .where(self.alpha_vantage_indicator_snapshots_table.c.trading_day == snapshot.trading_day)
                .order_by(self.alpha_vantage_indicator_snapshots_table.c.id.desc())
                .limit(1)
            ).first()
            values = {
                "created_at": timestamp_slug(),
                "symbol": snapshot.symbol.upper(),
                "interval": snapshot.interval,
                "trading_day": snapshot.trading_day,
                "latest_timestamp": snapshot.latest_timestamp,
                "latest_hour_slot_start": None if latest_hour_chunk is None else latest_hour_chunk.slot_start,
                "latest_hour_slot_end": None if latest_hour_chunk is None else latest_hour_chunk.slot_end,
                "payload": json.dumps(asdict(snapshot)),
            }
            if existing is None:
                connection.execute(sa.insert(self.alpha_vantage_indicator_snapshots_table).values(**values))
            else:
                connection.execute(
                    sa.update(self.alpha_vantage_indicator_snapshots_table)
                    .where(self.alpha_vantage_indicator_snapshots_table.c.id == existing.id)
                    .values(**values)
                )

    def load_alpha_vantage_indicator_snapshot(
        self,
        symbol: str,
        *,
        trading_day: str | None = None,
        interval: str | None = None,
    ) -> AlphaVantageIndicatorSnapshot | None:
        statement = (
            sa.select(self.alpha_vantage_indicator_snapshots_table.c.payload)
            .where(self.alpha_vantage_indicator_snapshots_table.c.symbol == symbol.upper())
            .order_by(
                self.alpha_vantage_indicator_snapshots_table.c.trading_day.desc(),
                self.alpha_vantage_indicator_snapshots_table.c.latest_timestamp.desc(),
                self.alpha_vantage_indicator_snapshots_table.c.id.desc(),
            )
            .limit(1)
        )
        if trading_day is not None:
            statement = statement.where(self.alpha_vantage_indicator_snapshots_table.c.trading_day == trading_day)
        if interval is not None:
            statement = statement.where(self.alpha_vantage_indicator_snapshots_table.c.interval == interval)
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
        if row is None or not row.payload:
            return None
        return _alpha_vantage_snapshot_from_payload(json.loads(row.payload))

    def load_alpha_vantage_indicator_snapshots(
        self,
        symbol: str,
        *,
        interval: str | None = None,
    ) -> list[AlphaVantageIndicatorSnapshot]:
        statement = (
            sa.select(
                self.alpha_vantage_indicator_snapshots_table.c.trading_day,
                self.alpha_vantage_indicator_snapshots_table.c.payload,
            )
            .where(self.alpha_vantage_indicator_snapshots_table.c.symbol == symbol.upper())
            .order_by(
                self.alpha_vantage_indicator_snapshots_table.c.trading_day.desc(),
                self.alpha_vantage_indicator_snapshots_table.c.latest_timestamp.desc(),
                self.alpha_vantage_indicator_snapshots_table.c.id.desc(),
            )
        )
        if interval is not None:
            statement = statement.where(self.alpha_vantage_indicator_snapshots_table.c.interval == interval)
        with self.engine.begin() as connection:
            rows = connection.execute(statement).all()

        snapshots_by_day: dict[str, AlphaVantageIndicatorSnapshot] = {}
        for row in rows:
            if row.trading_day in snapshots_by_day or not row.payload:
                continue
            snapshots_by_day[row.trading_day] = _alpha_vantage_snapshot_from_payload(json.loads(row.payload))
        return [snapshots_by_day[trading_day] for trading_day in sorted(snapshots_by_day)]

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

    def save_forecast_snapshot(self, snapshot) -> None:
        payload = _forecast_payload(snapshot)
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(self.forecast_snapshots_table).values(
                    symbol=snapshot.symbol.upper(),
                    generated_at=_timestamp_text(snapshot.generated_at),
                    valid_until=_timestamp_text(snapshot.valid_until),
                    trend_bias=snapshot.trend_bias,
                    summary=snapshot.summary,
                    confidence=snapshot.confidence,
                    payload=json.dumps(payload),
                )
            )

    def load_latest_forecast(
        self,
        symbol: str,
        *,
        as_of: pd.Timestamp | None = None,
    ):
        statement = (
            sa.select(self.forecast_snapshots_table.c.payload)
            .where(self.forecast_snapshots_table.c.symbol == symbol.upper())
            .order_by(self.forecast_snapshots_table.c.generated_at.desc())
        )
        if as_of is not None:
            statement = statement.where(self.forecast_snapshots_table.c.valid_until >= _timestamp_text(as_of))
        statement = statement.limit(1)
        with self.engine.begin() as connection:
            row = connection.execute(statement).first()
        if row is None or not row.payload:
            return None
        return _forecast_from_payload(json.loads(row.payload))

    def load_latest_performance(self, symbol: str) -> PerformanceSnapshot | None:
        statement = (
            sa.select(self.workflow_runs_table.c.metadata)
            .where(self.workflow_runs_table.c.symbol == symbol.upper())
            .order_by(self.workflow_runs_table.c.id.desc())
            .limit(25)
        )
        with self.engine.begin() as connection:
            rows = connection.execute(statement).all()
        for row in rows:
            if not row.metadata:
                continue
            try:
                metadata = json.loads(row.metadata)
            except json.JSONDecodeError:
                continue
            payload = metadata.get("performance")
            if isinstance(payload, dict):
                return _performance_from_payload(payload)
        return None

    def count_executed_trades(self, symbol: str) -> int:
        with self.engine.begin() as connection:
            count = connection.execute(
                sa.select(sa.func.count())
                .select_from(self.workflow_runs_table)
                .where(self.workflow_runs_table.c.symbol == symbol.upper())
                .where(self.workflow_runs_table.c.executed == sa.true())
            ).scalar_one()
        return int(count)

    def save_retrieved_news(self, symbol: str, query: str, articles: list[NewsArticle]) -> None:
        normalized_symbol = symbol.upper()
        fetched_at = _utc_now_text()
        rows = []
        for article in articles:
            headline = article.headline.strip()
            if not headline:
                continue
            rows.append(
                {
                    "symbol": normalized_symbol,
                    "query": query,
                    "article_key": _news_article_key(article),
                    "headline": headline,
                    "summary": article.summary,
                    "source": article.source,
                    "url": article.url,
                    "published_at": _normalize_news_timestamp(article.published_at, fallback=fetched_at),
                    "provider": article.provider,
                    "primary_ticker": article.primary_ticker,
                    "primary_ticker_relevance": article.primary_ticker_relevance,
                    "fetched_at": fetched_at,
                }
            )
        if not rows:
            return

        with self.engine.begin() as connection:
            for row in rows:
                try:
                    connection.execute(sa.insert(self.retrieved_news_table).values(**row))
                except sa.exc.IntegrityError:
                    continue

    def load_retrieved_news(
        self,
        symbol: str,
        query: str,
        *,
        limit: int = 100,
        published_at_lte: pd.Timestamp | str | None = None,
    ) -> list[NewsArticle]:
        statement = (
            sa.select(
                self.retrieved_news_table.c.headline,
                self.retrieved_news_table.c.summary,
                self.retrieved_news_table.c.source,
                self.retrieved_news_table.c.url,
                self.retrieved_news_table.c.published_at,
                self.retrieved_news_table.c.provider,
                self.retrieved_news_table.c.primary_ticker,
                self.retrieved_news_table.c.primary_ticker_relevance,
            )
            .where(self.retrieved_news_table.c.symbol == symbol.upper())
            .where(self.retrieved_news_table.c.query == query)
            .order_by(
                self.retrieved_news_table.c.published_at.desc(),
                self.retrieved_news_table.c.id.desc(),
            )
            .limit(max(1, limit))
        )
        if published_at_lte is not None:
            statement = statement.where(
                self.retrieved_news_table.c.published_at <= _normalize_news_timestamp(str(published_at_lte))
            )
        with self.engine.begin() as connection:
            rows = connection.execute(statement).all()
        return [
            NewsArticle(
                headline=row.headline,
                summary=row.summary,
                source=row.source,
                url=row.url,
                published_at=row.published_at,
                provider=row.provider,
                primary_ticker=row.primary_ticker,
                primary_ticker_relevance=row.primary_ticker_relevance,
            )
            for row in rows
        ]

    def _migrate_retrieved_news_table(self) -> None:
        inspector = sa.inspect(self.engine)
        if "retrieved_news" not in inspector.get_table_names():
            return
        existing_columns = {column["name"] for column in inspector.get_columns("retrieved_news")}
        statements = []
        if "provider" not in existing_columns:
            statements.append("ALTER TABLE retrieved_news ADD COLUMN provider VARCHAR(64) NOT NULL DEFAULT ''")
        if "primary_ticker" not in existing_columns:
            statements.append("ALTER TABLE retrieved_news ADD COLUMN primary_ticker VARCHAR(32)")
        if "primary_ticker_relevance" not in existing_columns:
            statements.append("ALTER TABLE retrieved_news ADD COLUMN primary_ticker_relevance FLOAT")
        if not statements:
            return
        with self.engine.begin() as connection:
            for statement in statements:
                connection.execute(sa.text(statement))

    def _migrate_workflow_runs_table(self) -> None:
        inspector = sa.inspect(self.engine)
        if "workflow_runs" not in inspector.get_table_names():
            return
        existing_columns = {column["name"] for column in inspector.get_columns("workflow_runs")}
        if "executed" in existing_columns:
            return
        with self.engine.begin() as connection:
            connection.execute(
                sa.text("ALTER TABLE workflow_runs ADD COLUMN executed BOOLEAN NOT NULL DEFAULT 0")
            )


def _coerce_database_url(database: Path | str) -> str:
    if isinstance(database, Path):
        return _sqlite_database_url(database)
    database_url = str(database).strip()
    if not database_url:
        raise RuntimeError("SQLiteResultStore requires a non-empty database path or database URL.")
    return database_url


def _sqlite_database_url(database_path: Path) -> str:
    return f"sqlite:///{database_path.resolve()}"


def _default_sqlalchemy_engine_factory(database_url: str) -> Engine:
    return sa.create_engine(database_url, future=True, pool_pre_ping=True)
