from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fresh_simple_trading_project.config import RunMode, Settings
from fresh_simple_trading_project.data_collection import (
    DataCollectionModule,
    HistoricalReplayDataClient,
    SimulatedAccountClient,
)
from fresh_simple_trading_project.decision_engine import DecisionEngine
from fresh_simple_trading_project.eda import EDAModule
from fresh_simple_trading_project.execution import ExecutionModule, InMemoryBrokerClient
from fresh_simple_trading_project.features import FeatureEngineeringModule
from fresh_simple_trading_project.market_analysis import MarketAnalysisModule
from fresh_simple_trading_project.models import RetrievalResult
from fresh_simple_trading_project.risk_analysis import RiskAnalysisModule
from fresh_simple_trading_project.storage import InMemoryResultStore
from fresh_simple_trading_project.workflow import TradingWorkflow


class DummyRawStore:
    def save_bars(self, symbol, timeframe, bars):
        return Path("/tmp")

    def save_news(self, symbol, articles):
        return Path("/tmp")


class DummyInformationRetrieval:
    def retrieve(
        self,
        symbol: str,
        limit: int = 8,
        *,
        input_size_chars: int | None = None,
        published_at_lte=None,
    ) -> RetrievalResult:
        return RetrievalResult(symbol=symbol, articles=[], headline_summary=[], sentiment_score=0.0)


@pytest.mark.parametrize(
    ("run_mode", "expected_sleep_seconds"),
    [
        (RunMode.LIVE.value, 3600.0),
        (RunMode.BACKTEST.value, 1.0),
    ],
)
def test_settings_from_env_reads_run_mode(
    monkeypatch: pytest.MonkeyPatch,
    run_mode: str,
    expected_sleep_seconds: float,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RUN_MODE", run_mode)

    settings = Settings.from_env(project_root=project_root)

    assert settings.trading.mode == RunMode(run_mode)
    assert settings.trading.sr_lookback_days == 7
    assert settings.trading.indicator_lookback_hours == 24
    assert settings.trading.sleep_seconds == expected_sleep_seconds


def test_backtest_run_loop_uses_explicit_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("RUN_MODE", RunMode.BACKTEST.value)
    settings = Settings.from_env(project_root=project_root)
    settings = replace(settings, trading=replace(settings.trading, mode=RunMode.BACKTEST))

    replay_client = HistoricalReplayDataClient(
        five_min_history=_make_replay_bars(),
        hourly_history=pd.DataFrame(),
    )
    account_client = SimulatedAccountClient(cash=settings.trading.starting_cash)
    broker_client = InMemoryBrokerClient(account_client=account_client)
    workflow = TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=replay_client,
            account_client=account_client,
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(settings.trading),
        information_retrieval=DummyInformationRetrieval(),
        risk_analysis=RiskAnalysisModule(settings.trading),
        decision_engine=DecisionEngine(),
        execution_module=ExecutionModule(broker_client),
        raw_store=DummyRawStore(),
        result_store=InMemoryResultStore(),
        default_sleep_seconds=0.0,
    )

    results = workflow.run_loop(symbol="AAPL", max_iterations=3, sleep_seconds=0.0)

    assert len(results) == 3
    for result in results:
        assert (
            result.five_minute_bars.index.max() - result.five_minute_bars.index.min()
        ) == pd.Timedelta(hours=24)
        assert (result.hourly_bars.index.max() - result.hourly_bars.index.min()) == pd.Timedelta(days=7)


def _make_replay_bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3_000, freq="5min", tz="UTC")
    close = pd.Series(range(len(index)), dtype="float64").mul(0.02).add(100.0).to_numpy()
    return pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 50_000,
        },
        index=index,
    )
