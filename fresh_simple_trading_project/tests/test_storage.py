from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from fresh_simple_trading_project.config import Settings
from fresh_simple_trading_project.models import (
    Action,
    AnalysisResult,
    AlphaVantageIndicatorSnapshot,
    BacktestSummary,
    Decision,
    EDAResult,
    ExecutionResult,
    IndicatorHourChunk,
    NewsArticle,
    RetrievalResult,
    RiskResult,
    WorkflowResult,
)
from fresh_simple_trading_project.storage import (
    AzureBlobRawStore,
    AzureSQLResultStore,
    LocalRawStore,
    SQLAlchemyResultStore,
    StorageRef,
)
from fresh_simple_trading_project.workflow import build_raw_store, build_result_store


def test_local_raw_store_returns_file_storage_refs(tmp_path: Path) -> None:
    store = LocalRawStore(tmp_path / "raw")

    bars_ref = store.save_bars("AAPL", "5min", _sample_bars())
    news_ref = store.save_news("AAPL", [NewsArticle(headline="Example headline")])

    assert bars_ref.uri.startswith("file://")
    assert news_ref.uri.startswith("file://")
    assert Path(bars_ref.uri.removeprefix("file://")).exists()
    assert Path(news_ref.uri.removeprefix("file://")).exists()


def test_sqlalchemy_result_store_persists_artifacts_and_state(tmp_path: Path) -> None:
    store = SQLAlchemyResultStore(f"sqlite:///{tmp_path / 'results.sqlite'}")
    result = _sample_workflow_result()

    store.save_workflow_run(
        result,
        raw_artifacts={
            "five_minute_bars": StorageRef("file:///tmp/aapl_5min.csv", "bars", "text/csv"),
            "news": StorageRef("file:///tmp/aapl_news.json", "news", "application/json"),
        },
    )
    store.save_last_processed("AAPL", result.analysis.timestamp)
    store.save_alpha_vantage_indicator_snapshot(_sample_alpha_vantage_snapshot())
    store.save_backtest_summary(
        BacktestSummary(
            symbol="AAPL",
            initial_cash=10_000.0,
            ending_cash=10_125.0,
            total_return_pct=1.25,
            benchmark_return_pct=0.9,
            max_drawdown_pct=0.4,
            sharpe_ratio=1.1,
            trade_count=2,
            win_rate=0.5,
            signal_accuracy=0.5,
            trades=[],
            equity_curve=pd.DataFrame({"equity": [10_000.0, 10_125.0]}),
        )
    )

    assert store.load_last_processed("AAPL") == pd.Timestamp("2025-01-01T12:00:00Z")
    with store.engine.begin() as connection:
        metadata_raw = connection.execute(sa.select(store.workflow_runs_table.c.metadata)).scalar_one()
        backtest_count = connection.execute(sa.select(sa.func.count()).select_from(store.backtest_runs_table)).scalar_one()
        snapshot_count = connection.execute(
            sa.select(sa.func.count()).select_from(store.alpha_vantage_indicator_snapshots_table)
        ).scalar_one()
    metadata = json.loads(metadata_raw)
    assert metadata["raw_artifacts"]["five_minute_bars"]["uri"] == "file:///tmp/aapl_5min.csv"
    assert snapshot_count == 1
    assert backtest_count == 1


def test_azure_blob_raw_store_uses_factory_and_returns_blob_urls() -> None:
    service_client = _FakeBlobServiceClient()
    store = AzureBlobRawStore(
        container_name="raw",
        account_url="https://example.blob.core.windows.net",
        blob_prefix="dev",
        blob_service_client_factory=lambda _: service_client,
    )

    bars_ref = store.save_bars("AAPL", "5min", _sample_bars())
    news_ref = store.save_news("AAPL", [NewsArticle(headline="Example headline")])

    assert bars_ref.uri.startswith("https://example.blob.core.windows.net/raw/dev/bars/AAPL/5min/")
    assert news_ref.uri.startswith("https://example.blob.core.windows.net/raw/dev/news/AAPL/")
    assert service_client.container_client.create_calls == 1
    assert len(service_client.container_client.uploads) == 2
    assert service_client.container_client.uploads[0]["overwrite"] is True
    assert "open,high,low,close,volume" in service_client.container_client.uploads[0]["payload"]


def test_store_builders_select_azure_providers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAW_STORE_PROVIDER", "azure_blob")
    monkeypatch.setenv("RESULT_STORE_PROVIDER", "azure_sql")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")
    monkeypatch.setenv("AZURE_BLOB_CONTAINER_RAW", "raw")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    settings = Settings.from_env(project_root=tmp_path)

    assert isinstance(build_raw_store(settings), AzureBlobRawStore)
    assert isinstance(build_result_store(settings), AzureSQLResultStore)


def _sample_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        },
        index=pd.date_range("2025-01-01T11:55:00Z", periods=2, freq="5min", tz="UTC"),
    )


def _sample_workflow_result() -> WorkflowResult:
    bars = _sample_bars()
    return WorkflowResult(
        symbol="AAPL",
        five_minute_bars=bars,
        hourly_bars=bars,
        eda=EDAResult(
            symbol="AAPL",
            missing_values=0,
            anomaly_count=0,
            candle_return_mean=0.01,
            candle_volatility=0.02,
            latest_close=101.5,
        ),
        analysis=AnalysisResult(
            symbol="AAPL",
            timestamp=pd.Timestamp("2025-01-01T12:00:00Z"),
            latest_price=101.5,
            trend="uptrend",
            bullish=True,
            entry_setup=True,
            exit_setup=False,
            confidence=0.8,
        ),
        retrieval=RetrievalResult(
            symbol="AAPL",
            articles=[NewsArticle(headline="Example headline")],
            headline_summary=["Example headline"],
            sentiment_score=0.4,
        ),
        risk=RiskResult(
            symbol="AAPL",
            risk_score=0.2,
            can_enter=True,
            recommended_qty=1,
        ),
        decision=Decision(
            symbol="AAPL",
            action=Action.BUY,
            quantity=1,
            confidence=0.8,
            rationale=["Positive setup"],
        ),
        execution=ExecutionResult(
            executed=False,
            order_id=None,
            status="dry_run",
        ),
        alpha_vantage_indicator_snapshot=_sample_alpha_vantage_snapshot(),
    )


def _sample_alpha_vantage_snapshot() -> AlphaVantageIndicatorSnapshot:
    rows = [
        {"time": "2025-01-01 11:00:00", "RSI": 52.0, "ADX": 23.5, "threshold_hits": ["ADX_STRONG_TREND"]},
        {"time": "2025-01-01 11:05:00", "RSI": 72.0, "ADX": 28.0, "threshold_hits": ["RSI_OVERBOUGHT"]},
    ]
    latest_chunk = IndicatorHourChunk(
        slot_start="2025-01-01 11:00:00",
        slot_end="2025-01-01 11:05:00",
        rows=rows,
    )
    return AlphaVantageIndicatorSnapshot(
        symbol="AAPL",
        interval="5min",
        trading_day="2025-01-01",
        latest_timestamp="2025-01-01 11:05:00",
        indicator_columns=["RSI", "ADX"],
        rows=rows,
        hourly_chunks=[latest_chunk],
        latest_hour_chunk=latest_chunk,
    )


class _FakeBlobServiceClient:
    def __init__(self) -> None:
        self.container_client = _FakeContainerClient()

    def get_container_client(self, name: str) -> "_FakeContainerClient":
        self.container_client.name = name
        return self.container_client


class _FakeContainerClient:
    def __init__(self) -> None:
        self.name: str | None = None
        self.create_calls = 0
        self.uploads: list[dict[str, object]] = []

    def create_container(self) -> None:
        self.create_calls += 1

    def get_blob_client(self, blob_name: str) -> "_FakeBlobClient":
        return _FakeBlobClient(self, blob_name)


class _FakeBlobClient:
    def __init__(self, container: _FakeContainerClient, blob_name: str) -> None:
        self.container = container
        self.blob_name = blob_name
        self.url = f"https://example.blob.core.windows.net/{container.name}/{blob_name}"

    def upload_blob(self, payload: str, *, overwrite: bool, content_settings) -> None:
        self.container.uploads.append(
            {
                "blob_name": self.blob_name,
                "payload": payload,
                "overwrite": overwrite,
                "content_settings": content_settings,
            }
        )
