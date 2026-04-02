from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .models import AlphaVantageIndicatorSnapshot, BacktestSummary, NewsArticle, WorkflowResult
from .utils import timestamp_slug


@dataclass(frozen=True)
class StorageRef:
    uri: str
    kind: str
    content_type: str | None = None


class RawStore(Protocol):
    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        ...

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        ...


class ResultStore(Protocol):
    def save_workflow_run(
        self,
        result: WorkflowResult,
        *,
        raw_artifacts: dict[str, StorageRef] | None = None,
    ) -> None:
        ...

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        ...

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        ...

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        ...

    def save_alpha_vantage_indicator_snapshot(self, snapshot: AlphaVantageIndicatorSnapshot) -> None:
        ...


class LocalRawStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        target = self.root / "bars" / f"{symbol}_{timeframe}_{timestamp_slug()}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(target)
        return StorageRef(uri=target.resolve().as_uri(), kind="bars", content_type="text/csv")

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        target = self.root / "news" / f"{symbol}_{timestamp_slug()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [article.__dict__ for article in articles]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return StorageRef(uri=target.resolve().as_uri(), kind="news", content_type="application/json")


class AzureBlobRawStore:
    def __init__(
        self,
        *,
        container_name: str,
        account_url: str | None = None,
        connection_string: str | None = None,
        blob_prefix: str = "",
        blob_service_client_factory: Callable[["AzureBlobRawStore"], Any] | None = None,
    ) -> None:
        if not container_name.strip():
            raise RuntimeError("AzureBlobRawStore requires a non-empty container name.")
        if not (account_url or connection_string):
            raise RuntimeError(
                "AzureBlobRawStore requires AZURE_STORAGE_ACCOUNT_URL or AZURE_STORAGE_CONNECTION_STRING."
            )
        self.container_name = container_name.strip()
        self.account_url = account_url
        self.connection_string = connection_string
        self.blob_prefix = blob_prefix.strip("/")
        self._blob_service_client_factory = blob_service_client_factory
        self._blob_service_client: Any | None = None
        self._container_ready = False

    @property
    def blob_service_client(self) -> Any:
        if self._blob_service_client is None:
            factory = self._blob_service_client_factory or _default_blob_service_client_factory
            self._blob_service_client = factory(self)
        return self._blob_service_client

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> StorageRef:
        payload = bars.to_csv()
        blob_name = self._dated_blob_name(
            "bars",
            symbol=symbol,
            subfolder=timeframe,
            suffix=".csv",
        )
        return self._upload_text(blob_name, payload, kind="bars", content_type="text/csv")

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> StorageRef:
        payload = json.dumps([article.__dict__ for article in articles], indent=2)
        blob_name = self._dated_blob_name(
            "news",
            symbol=symbol,
            suffix=".json",
        )
        return self._upload_text(blob_name, payload, kind="news", content_type="application/json")

    def _dated_blob_name(
        self,
        category: str,
        *,
        symbol: str,
        suffix: str,
        subfolder: str | None = None,
    ) -> str:
        current_time = datetime.now(timezone.utc)
        parts = []
        if self.blob_prefix:
            parts.append(self.blob_prefix)
        parts.extend(
            [
                category,
                symbol.upper(),
            ]
        )
        if subfolder:
            parts.append(subfolder)
        parts.extend(
            [
                current_time.strftime("%Y"),
                current_time.strftime("%m"),
                current_time.strftime("%d"),
                f"{timestamp_slug()}{suffix}",
            ]
        )
        return "/".join(parts)

    def _upload_text(self, blob_name: str, payload: str, *, kind: str, content_type: str) -> StorageRef:
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        content_settings = _build_content_settings(content_type)
        blob_client.upload_blob(payload, overwrite=True, content_settings=content_settings)
        return StorageRef(
            uri=str(getattr(blob_client, "url", "")) or self._fallback_blob_uri(blob_name),
            kind=kind,
            content_type=content_type,
        )

    def _get_container_client(self) -> Any:
        container_client = self.blob_service_client.get_container_client(self.container_name)
        if not self._container_ready:
            try:
                container_client.create_container()
            except Exception as exc:  # pragma: no cover - exercised with Azure SDK.
                if exc.__class__.__name__ != "ResourceExistsError":
                    raise
            self._container_ready = True
        return container_client

    def _fallback_blob_uri(self, blob_name: str) -> str:
        if self.account_url:
            return f"{self.account_url.rstrip('/')}/{self.container_name}/{blob_name}"
        return f"az://{self.container_name}/{blob_name}"


class InMemoryResultStore:
    def __init__(self) -> None:
        self.workflow_runs: list[WorkflowResult] = []
        self.backtest_runs: list[BacktestSummary] = []
        self.last_processed: dict[str, pd.Timestamp] = {}
        self.raw_artifacts_by_symbol: dict[str, dict[str, StorageRef]] = {}
        self.alpha_vantage_indicator_snapshots: list[AlphaVantageIndicatorSnapshot] = []

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
        self.alpha_vantage_indicator_snapshots.append(snapshot)


class SQLAlchemyResultStore:
    def __init__(
        self,
        database_url: str,
        *,
        engine_factory: Callable[[str], Engine] | None = None,
    ) -> None:
        if not database_url.strip():
            raise RuntimeError("SQLAlchemyResultStore requires a non-empty database URL.")
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
        create_engine = engine_factory or _default_sqlalchemy_engine_factory
        self.engine = create_engine(database_url)
        self._metadata.create_all(self.engine)

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
        }
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
        with self.engine.begin() as connection:
            connection.execute(
                sa.insert(self.workflow_runs_table).values(
                    created_at=timestamp_slug(),
                    symbol=result.symbol,
                    action=result.decision.action.value,
                    quantity=result.decision.quantity,
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
            connection.execute(
                sa.insert(self.alpha_vantage_indicator_snapshots_table).values(
                    created_at=timestamp_slug(),
                    symbol=snapshot.symbol,
                    interval=snapshot.interval,
                    trading_day=snapshot.trading_day,
                    latest_timestamp=snapshot.latest_timestamp,
                    latest_hour_slot_start=None if latest_hour_chunk is None else latest_hour_chunk.slot_start,
                    latest_hour_slot_end=None if latest_hour_chunk is None else latest_hour_chunk.slot_end,
                    payload=json.dumps(asdict(snapshot)),
                )
            )


class SQLiteResultStore(SQLAlchemyResultStore):
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(_sqlite_database_url(self.database_path))


class AzureSQLResultStore(SQLAlchemyResultStore):
    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)


def _sqlite_database_url(database_path: Path) -> str:
    return f"sqlite:///{database_path.resolve()}"


def _default_sqlalchemy_engine_factory(database_url: str) -> Engine:
    return sa.create_engine(database_url, future=True, pool_pre_ping=True)


def _default_blob_service_client_factory(store: AzureBlobRawStore) -> Any:
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    if store.connection_string:
        return BlobServiceClient.from_connection_string(store.connection_string)
    return BlobServiceClient(account_url=store.account_url, credential=DefaultAzureCredential())


def _build_content_settings(content_type: str) -> Any:
    from azure.storage.blob import ContentSettings

    return ContentSettings(content_type=content_type)
