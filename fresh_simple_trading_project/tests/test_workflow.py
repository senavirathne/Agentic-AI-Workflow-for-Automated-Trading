from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fresh_simple_trading_project.workflow as workflow_module
from fresh_simple_trading_project.config import Settings
from fresh_simple_trading_project.data_collection import DataCollectionModule, StaticAccountClient, StaticMarketDataClient
from fresh_simple_trading_project.decision_engine import DecisionEngine
from fresh_simple_trading_project.eda import EDAModule
from fresh_simple_trading_project.execution import ExecutionModule, InMemoryBrokerClient
from fresh_simple_trading_project.features import FeatureEngineeringModule
from fresh_simple_trading_project.information_retrieval import (
    AlpacaNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    StaticNewsSearchClient,
    WebSearchNewsClient,
)
from fresh_simple_trading_project.llm import LLMRequestError, TextGenerationClient
from fresh_simple_trading_project.market_analysis import MarketAnalysisModule
from fresh_simple_trading_project.models import NewsArticle
from fresh_simple_trading_project.risk_analysis import RiskAnalysisModule
from fresh_simple_trading_project.storage import InMemoryResultStore, LocalRawStore
from fresh_simple_trading_project.workflow import TradingWorkflow


def test_run_once_executes_fresh_hourly_architecture(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.symbol == "AAPL"
    assert len(result.five_minute_bars) > len(result.hourly_bars)
    assert result.analysis.entry_setup is True
    assert result.retrieval.sentiment_score > 0
    assert result.decision.action.value == "BUY"
    assert result.execution.status == "dry_run"


def test_analyze_range_then_resume_from_last_checkpoint(monkeypatch, tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    monkeypatch.setattr(workflow_module, "sleep_until", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow_module, "next_top_of_hour", lambda: None)

    history = workflow.analyze_range(
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T03:00:00Z",
        symbol="AAPL",
        execute_orders=False,
    )
    resumed = workflow.run_loop(symbol="AAPL", execute_orders=False, max_iterations=1)

    assert len(history) == 3
    assert len(resumed) == 1
    assert pd.Timestamp(history[-1].analysis.timestamp) == pd.Timestamp("2025-01-01T03:00:00Z")
    assert pd.Timestamp(resumed[0].analysis.timestamp) == pd.Timestamp("2025-01-01T04:00:00Z")
    assert workflow.result_store.load_last_processed("AAPL") == pd.Timestamp("2025-01-01T04:00:00Z")


def test_analyze_range_clamps_end_to_last_available_candlestick(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    history = workflow.analyze_range(
        start="2025-01-03T10:00:00Z",
        end="2025-01-10T00:00:00Z",
        symbol="AAPL",
        execute_orders=False,
    )

    assert len(history) == 1
    assert pd.Timestamp(history[0].analysis.timestamp) == pd.Timestamp("2025-01-03T11:00:00Z")


def test_workflow_uses_optional_llm_inside_existing_modules(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Bullish momentum is intact above support.",
            "News flow is mildly positive with no major red flags.",
            "Risk remains acceptable for a small position.",
            "OVERRIDE=KEEP\nNOTE=The module outputs support the current action.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert _expected_technical_prefix() in result.analysis.llm_summary
    assert "Bullish momentum is intact above support." in result.analysis.llm_summary
    assert result.retrieval.summary_note == "News flow is mildly positive with no major red flags."
    assert result.risk.summary_note == "Risk remains acceptable for a small position."
    assert any("LLM review" in item for item in result.decision.rationale)


def test_decision_agent_receives_other_agent_handoffs(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical agent sees improving momentum.",
            "News agent sees supportive catalysts.",
            "Risk agent keeps size small but acceptable.",
            "OVERRIDE=KEEP\nNOTE=All prior agent handoffs align with the proposed trade.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)

    workflow.run_once(symbol="AAPL", execute_orders=False)

    assert len(llm_client.calls) == 4
    decision_system_prompt, decision_prompt = llm_client.calls[-1]
    assert "decision coordinator agent" in decision_system_prompt.lower()
    assert f"Technical agent handoff: {_expected_technical_prefix()}" in decision_prompt
    assert "Technical agent sees improving momentum." in decision_prompt
    assert "News agent handoff: News agent sees supportive catalysts." in decision_prompt
    assert "Risk agent handoff: Risk agent keeps size small but acceptable." in decision_prompt


def test_agents_receive_plain_text_inputs_only(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical prompt confirms the uptrend is intact.",
            "News prompt shows positive earnings-driven headlines.",
            "Risk remains acceptable for a small position.",
            "ACTION=BUY\nQUANTITY=999\nNOTE=The prompt inputs support a guarded long entry.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert _expected_technical_prefix() in result.analysis.llm_summary
    assert "Technical prompt confirms the uptrend is intact." in result.analysis.llm_summary
    assert result.retrieval.summary_note == "News prompt shows positive earnings-driven headlines."
    assert result.decision.action.value == "BUY"
    assert result.decision.quantity == result.risk.recommended_qty
    technical_prompt = llm_client.calls[0][1]
    news_prompt = llm_client.calls[1][1]
    decision_prompt = llm_client.calls[3][1]
    assert "tool" not in technical_prompt.lower()
    assert "tool" not in news_prompt.lower()
    assert "tool" not in decision_prompt.lower()
    assert "RSI:" in technical_prompt
    assert "Recent articles:" in news_prompt
    assert "News headlines:" in decision_prompt


def test_workflow_wraps_llm_errors_with_stage_context(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path, llm_client=RaisingLLM())

    with pytest.raises(LLMRequestError, match="workflow stage 'market_analysis' for symbol 'AAPL'") as exc_info:
        workflow.run_once(symbol="AAPL", execute_orders=False)

    assert exc_info.value.operation == "generate"
    assert exc_info.value.model == "deepseek-reasoner"
    assert exc_info.value.base_url == "https://api.deepseek.com"


def test_technical_agent_handoff_always_includes_current_datetime_and_price(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical view stays constructive above support.",
            "News flow is mildly positive with no major red flags.",
            "Risk remains acceptable for a small position.",
            "OVERRIDE=KEEP\nNOTE=The module outputs support the current action.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert "2025-01-03T11:55:00Z" in result.analysis.llm_summary
    assert "$160.00" in result.analysis.llm_summary


def test_build_workflow_requires_deepseek_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        workflow_module.build_workflow(project_root=tmp_path)


def test_build_workflow_uses_alpaca_news_when_alpaca_credentials_are_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "synthetic")

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert workflow.settings.market_data.provider == "synthetic"
    assert isinstance(workflow.information_retrieval.news_client, CombinedNewsSearchClient)
    assert isinstance(workflow.information_retrieval.news_client.clients[0], AlpacaNewsSearchClient)
    assert isinstance(workflow.information_retrieval.news_client.clients[1], WebSearchNewsClient)


def test_format_reasoning_lines_show_agent_outputs_and_placeholders(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)
    lines = workflow_module.format_reasoning_lines(result)

    assert "Data Window:" in lines[0]
    assert "Technical Agent: <no output returned>" in lines[1]
    assert "News Agent: <no output returned>" in lines[2]
    assert "Risk Agent: <no output returned>" in lines[3]
    assert any(line.startswith("  Decision Reason") for line in lines)


def test_run_loop_passes_sleep_override_to_sleep_until(monkeypatch, tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    sleep_calls: list[float | None] = []

    def fake_sleep_until(_target, sleep_seconds=None):
        sleep_calls.append(sleep_seconds)
        return 0.0

    monkeypatch.setattr(workflow_module, "sleep_until", fake_sleep_until)

    results = workflow.run_loop(
        symbol="AAPL",
        execute_orders=False,
        max_iterations=2,
        sleep_seconds=0.25,
    )

    assert len(results) == 1
    assert sleep_calls == [0.25]


def _build_workflow(tmp_path: Path, llm_client: TextGenerationClient | None = None) -> TradingWorkflow:
    settings = Settings.from_env(project_root=tmp_path)
    return TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=StaticMarketDataClient(_make_uptrend_bars()),
            account_client=StaticAccountClient(cash=settings.trading.starting_cash),
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(settings.trading, llm_client=llm_client),
        information_retrieval=InformationRetrievalModule(
            StaticNewsSearchClient(
                [
                    NewsArticle(
                        headline="AAPL earnings beat expectations with strong growth",
                        summary="Analysts point to margin expansion and resilient demand.",
                    )
                ]
            ),
            llm_client=llm_client,
        ),
        risk_analysis=RiskAnalysisModule(settings.trading, llm_client=llm_client),
        decision_engine=DecisionEngine(llm_client=llm_client),
        execution_module=ExecutionModule(InMemoryBrokerClient()),
        raw_store=LocalRawStore(settings.paths.raw_dir),
        result_store=InMemoryResultStore(),
    )


def _make_uptrend_bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=720, freq="5min", tz="UTC")
    base = np.linspace(100.0, 160.0, num=len(index))
    return pd.DataFrame(
        {
            "open": base - 0.5,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.full(len(index), 50_000),
        },
        index=index,
    )


def _expected_technical_prefix() -> str:
    return "As of 2025-01-03T11:55:00Z, AAPL current price is $160.00."


class StubLLM(TextGenerationClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        if not self._responses:
            return None
        return self._responses.pop(0)


class RaisingLLM(TextGenerationClient):
    def generate(self, system_prompt: str, content: str) -> str | None:
        raise LLMRequestError(
            operation="generate",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com",
            detail="ModuleNotFoundError: No module named 'openai'",
        )
