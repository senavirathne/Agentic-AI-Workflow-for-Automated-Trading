from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import project.workflow as workflow_module
from project.config import RunMode, Settings
from project.agents import (
    DecisionCoordinatorAgent,
    HoldForecastAgent,
    NewsResearchAgent,
    RiskReviewAgent,
    TechnicalAnalysisAgent,
)
from project.data_collection import (
    DataCollectionModule,
    HistoricalReplayDataClient,
    SimulatedAccountClient,
    StaticMarketDataClient,
)
from project.decision_engine import DecisionEngine
from project.eda import EDAModule
from project.execution import ExecutionModule, InMemoryBrokerClient
from project.features import FeatureEngineeringModule
from project.information_retrieval import (
    AlphaVantageNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    StaticNewsSearchClient,
    WebSearchNewsClient,
)
from project.llm import FallbackLLMClient, LLMRequestError, OpenAILLMClient, TextGenerationClient
from project.market_analysis import MarketAnalysisModule
from project.models import AlphaVantageIndicatorSnapshot, IndicatorHourChunk, NewsArticle, RetrievalResult
from project.risk_analysis import RiskAnalysisModule
from project.storage import InMemoryResultStore, LocalRawStore
from project.utils import last_closed_us_market_day_cutoff, us_market_day_dates
from project.workflow import TradingWorkflow


def test_run_once_executes_fresh_hourly_architecture(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.symbol == "AAPL"
    assert len(result.five_minute_bars) > len(result.hourly_bars)
    assert result.analysis.entry_setup is True
    assert result.retrieval.headline_summary
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
    assert 'News context JSON: {"summary_note":"News agent sees supportive catalysts.","critical_news":[],"risk_flags":[],"catalysts":[]}' in decision_prompt
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
    assert "Rule-based sentiment score" not in news_prompt
    assert 'News context JSON: {"summary_note":"News prompt shows positive earnings-driven headlines.","critical_news":[],"risk_flags":[],"catalysts":[]}' in decision_prompt
    assert "Critical news (compressed):" not in decision_prompt
    assert "News headlines:" not in decision_prompt


def test_live_trading_uses_manual_indicators_even_when_alpha_vantage_is_configured(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical prompt stays on manually computed live indicators.",
            "News flow is positive.",
            "Risk remains acceptable.",
            "OVERRIDE=KEEP\nNOTE=The rule-based decision is acceptable.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)
    workflow.alpha_vantage_service = StubAlphaVantageService(_sample_alpha_vantage_snapshot())

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    technical_prompt = llm_client.calls[0][1]
    assert result.alpha_vantage_indicator_snapshot is None
    assert result.analysis.indicator_source == "manually computed indicators from live 5-minute bar data"
    assert "Alpha Vantage latest 1-hour indicator chunk" not in technical_prompt
    assert workflow.alpha_vantage_service.snapshot_calls == []
    assert workflow.alpha_vantage_service.feature_frame_calls == []


def test_backtest_uses_latest_alpha_vantage_hour_chunk_and_feature_frame(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical prompt confirms the Alpha Vantage hour chunk.",
            "News flow is positive.",
            "Risk remains acceptable.",
            "OVERRIDE=KEEP\nNOTE=The rule-based decision is acceptable.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
        market_data=replace(workflow.settings.market_data, provider="alpha_vantage"),
    )
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        feature_frame=_sample_alpha_vantage_feature_frame(_make_uptrend_bars()),
    )
    _seed_backtest_alpha_vantage_store(workflow)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    technical_prompt = llm_client.calls[0][1]
    assert result.alpha_vantage_indicator_snapshot is not None
    assert (
        result.analysis.indicator_source
        == "Alpha Vantage 5-minute indicators loaded from local storage and chunked into a 1-hour backtest step"
    )
    assert "Alpha Vantage latest 1-hour indicator chunk" in technical_prompt
    assert "\"time\":\"2025-01-03 11:00:00\"" in technical_prompt
    assert "Alpha Vantage threshold hits in latest chunk" in technical_prompt
    assert "RSI_OVERBOUGHT" in technical_prompt
    assert workflow.alpha_vantage_service.ensure_window_calls == []
    assert workflow.alpha_vantage_service.snapshot_calls == []
    assert workflow.alpha_vantage_service.feature_frame_calls == [("AAPL", pd.Timestamp("2025-01-03T11:55:00Z"))]
    assert result.alpha_vantage_indicator_snapshot.latest_hour_chunk is not None
    assert result.alpha_vantage_indicator_snapshot.latest_hour_chunk.slot_start == "2025-01-03 11:00:00"
    assert len(result.alpha_vantage_indicator_snapshot.hourly_chunks) == 1


def test_backtest_preloads_missing_alpha_vantage_snapshots_into_local_store(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        result_store=workflow.result_store,
    )

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.alpha_vantage_indicator_snapshot is not None
    assert len(workflow.alpha_vantage_service.ensure_window_calls) == 1
    unique_days = [
        candidate.isoformat()
        for candidate in us_market_day_dates(workflow.data_collection.fetch_five_minute_history("AAPL").index)
    ]
    assert all(
        workflow.result_store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day=trading_day, interval="5min")
        is not None
        for trading_day in unique_days
    )


def test_backtest_hydrates_cached_alpha_vantage_snapshots_into_local_store(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    trading_days = [
        candidate.isoformat()
        for candidate in us_market_day_dates(workflow.data_collection.fetch_five_minute_history("AAPL").index)
    ]
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        result_store=workflow.result_store,
        populate_store=False,
        local_snapshots={trading_day: _snapshot_for_trading_day(trading_day) for trading_day in trading_days},
    )

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.alpha_vantage_indicator_snapshot is not None
    assert workflow.alpha_vantage_service.ensure_window_calls == []
    assert all(
        workflow.result_store.load_alpha_vantage_indicator_snapshot("AAPL", trading_day=trading_day, interval="5min")
        is not None
        for trading_day in trading_days
    )


def test_backtest_accepts_latest_available_alpha_vantage_data_when_newest_days_are_missing(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    trading_days = [
        candidate.isoformat()
        for candidate in us_market_day_dates(workflow.data_collection.fetch_five_minute_history("AAPL").index)
    ]
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        result_store=workflow.result_store,
        populate_store=False,
        local_snapshots={
            trading_days[0]: _snapshot_for_trading_day(trading_days[0]),
            trading_days[1]: _snapshot_for_trading_day(trading_days[1]),
        },
    )

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.alpha_vantage_indicator_snapshot is not None
    assert result.alpha_vantage_indicator_snapshot.trading_day == trading_days[1]
    assert pd.Timestamp(result.analysis.timestamp) == pd.Timestamp(f"{trading_days[1]}T11:55:00Z")
    assert len(workflow.alpha_vantage_service.ensure_window_calls) == 1
    assert workflow.result_store.load_alpha_vantage_indicator_snapshot(
        "AAPL",
        trading_day=trading_days[1],
        interval="5min",
    ) is not None
    assert workflow.result_store.load_alpha_vantage_indicator_snapshot(
        "AAPL",
        trading_day=trading_days[-1],
        interval="5min",
    ) is None


def test_backtest_uses_explicit_price_lookup_method(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.data_collection.market_data_client = RecordingPriceMarketDataClient(
        _make_uptrend_bars(),
        price_to_return=161.25,
    )

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert result.analysis.latest_price == 161.25
    assert workflow.data_collection.market_data_client.price_calls == [pd.Timestamp("2025-01-03T11:55:00Z")]


def test_backtest_raises_when_alpha_vantage_preload_does_not_fill_local_store(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        populate_store=False,
    )

    with pytest.raises(RuntimeError, match="DB path"):
        workflow.run_once(symbol="AAPL", execute_orders=False)


def test_backtest_raises_when_alpha_vantage_gap_precedes_latest_available_day(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    trading_days = [
        candidate.isoformat()
        for candidate in us_market_day_dates(workflow.data_collection.fetch_five_minute_history("AAPL").index)
    ]
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        result_store=workflow.result_store,
        populate_store=False,
        local_snapshots={trading_days[-1]: _snapshot_for_trading_day(trading_days[-1])},
    )

    with pytest.raises(RuntimeError, match="DB path"):
        workflow.run_once(symbol="AAPL", execute_orders=False)


def test_backtest_news_cutoff_uses_last_closed_us_market_day(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.alpha_vantage_service = StubAlphaVantageService(
        _sample_alpha_vantage_snapshot(),
        result_store=workflow.result_store,
    )
    retrieval = RecordingInformationRetrievalModule()
    workflow.information_retrieval = retrieval

    workflow.run_once(symbol="AAPL", execute_orders=False)

    assert retrieval.calls
    assert retrieval.calls[-1]["published_at_lte"] == pd.Timestamp("2025-01-02T21:00:00Z")


def test_last_closed_us_market_day_cutoff_ignores_weekend_timestamps() -> None:
    timestamps = pd.date_range("2025-01-03T14:00:00Z", "2025-01-05T23:00:00Z", freq="5min", tz="UTC")

    cutoff = last_closed_us_market_day_cutoff(
        "2025-01-05T18:00:00Z",
        available_timestamps=timestamps,
    )

    assert cutoff == pd.Timestamp("2025-01-03T21:00:00Z")


def test_us_market_day_dates_exclude_weekends_and_market_holidays() -> None:
    timestamps = pd.date_range("2026-03-20T13:00:00Z", "2026-04-03T23:55:00Z", freq="5min", tz="UTC")

    trading_days = us_market_day_dates(timestamps, max_date=date(2026, 4, 2))

    assert date(2026, 3, 20) in trading_days
    assert date(2026, 4, 2) in trading_days
    assert date(2026, 3, 21) not in trading_days
    assert date(2026, 3, 22) not in trading_days
    assert date(2026, 3, 28) not in trading_days
    assert date(2026, 3, 29) not in trading_days
    assert date(2026, 4, 3) not in trading_days


def test_live_llms_receive_delayed_data_notice_and_current_price(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("LIVE_MARKET_DATA_DELAY_MINUTES", "15")
    llm_client = StubLLM(
        [
            "Technical prompt accounts for delayed bars and live price.",
            "News flow is positive.",
            "Risk prompt accounts for delayed bars and live price.",
            "OVERRIDE=KEEP\nNOTE=Decision prompt accounts for delayed bars and live price.",
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)
    workflow.alpaca_service = StubLivePriceService(161.25)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)

    technical_prompt = llm_client.calls[0][1]
    risk_prompt = llm_client.calls[2][1]
    decision_prompt = llm_client.calls[3][1]
    assert result.analysis.current_price == 161.25
    assert result.analysis.market_data_delay_minutes == 15
    assert "Live market data delay minutes: 15" in technical_prompt
    assert "Current live price snapshot: 161.25" in technical_prompt
    assert "Latest delayed 5-minute close: 160.00" in technical_prompt
    assert "Live market data delay minutes: 15" in risk_prompt
    assert "Current live price snapshot: 161.25" in risk_prompt
    assert "Live market data delay minutes: 15" in decision_prompt
    assert "Current live price snapshot: 161.25" in decision_prompt
    assert "latest delayed close is $160.00 and current live price is $161.25" in result.analysis.llm_summary


def test_live_workflow_tracks_trade_count_profit_and_adjusts_protective_orders(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        market_data=replace(workflow.settings.market_data, provider="alpaca"),
        trading=replace(workflow.settings.trading, live_market_data_delay_minutes=15),
    )
    simulated_account = SimulatedAccountClient(cash=workflow.settings.trading.starting_cash)
    broker = InMemoryBrokerClient(account_client=simulated_account)
    workflow.data_collection.account_client = simulated_account
    workflow.execution_module = ExecutionModule(broker)

    workflow.alpaca_service = StubLivePriceService(160.0)
    first = workflow.run_once(symbol="AAPL", execute_orders=True)

    workflow.alpaca_service = StubLivePriceService(164.0)
    second = workflow.run_once(symbol="AAPL", execute_orders=True)

    assert first.performance is not None
    assert first.performance.trade_count == 1
    assert first.execution.executed is True
    assert first.decision.action.value == "BUY"
    assert second.performance is not None
    assert second.performance.trade_count == 1
    assert second.performance.current_profit > 0
    assert second.execution.status == "protected"
    assert second.execution.protective_order_ids
    assert broker.protective_orders
    assert {order["type"] for order in broker.protective_orders} == {"stop", "limit"}


def test_workflow_reuses_previous_hold_forecast_on_next_hour(tmp_path: Path) -> None:
    llm_client = StubLLM(
        [
            "Technical prompt confirms range-bound conditions.",
            "News flow is neutral.",
            "Risk remains acceptable but no fresh entry is justified.",
            "ACTION=HOLD\nQUANTITY=0\nNOTE=Wait for confirmation.",
            (
                '{"trend_bias":"bullish","continuation_price_target":165.5,'
                '"continuation_volume_target":62000,"reversal_price_target":157.0,'
                '"reversal_volume_target":68000,'
                '"continuation_signals":["Breakout stays valid above 165.5 with expanding volume."],'
                '"reversal_signals":["Loss of 157.0 on heavy sell volume warns of reversal."],'
                '"summary":"If AAPL clears 165.50 on expanding volume, the uptrend is still intact.",'
                '"confidence":0.72}'
            ),
            "Technical prompt reviews the prior hold forecast.",
            "News flow is still neutral.",
            "Risk remains acceptable and references the prior hold forecast.",
            "ACTION=HOLD\nQUANTITY=0\nNOTE=Keep waiting for confirmation.",
            (
                '{"trend_bias":"bullish","continuation_price_target":166.0,'
                '"continuation_volume_target":63000,"reversal_price_target":156.5,'
                '"reversal_volume_target":69000,'
                '"continuation_signals":["Breakout above 166.0 keeps the move intact."],'
                '"reversal_signals":["Failure below 156.5 on heavy volume warns of reversal."],'
                '"summary":"The setup still needs a breakout confirmation.",'
                '"confidence":0.7}'
            ),
        ]
    )
    workflow = _build_workflow(tmp_path, llm_client=llm_client)

    first = workflow.run_once(symbol="AAPL", execute_orders=False)
    second = workflow.run_once(symbol="AAPL", execute_orders=False)

    assert first.hold_forecast is not None
    assert second.previous_forecast is not None
    assert second.previous_forecast.summary == "If AAPL clears 165.50 on expanding volume, the uptrend is still intact."
    second_technical_prompt = llm_client.calls[5][1]
    second_risk_prompt = llm_client.calls[7][1]
    second_decision_prompt = llm_client.calls[8][1]
    assert "Previous HOLD forecast:" in second_technical_prompt
    assert "165.5" in second_technical_prompt
    assert "Previous HOLD forecast:" in second_risk_prompt
    assert "Previous HOLD forecast:" in second_decision_prompt
    assert workflow.result_store.load_latest_forecast("AAPL") is not None


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


def test_build_workflow_requires_any_llm_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY or OPENAI_API_KEY"):
        workflow_module.build_workflow(project_root=tmp_path)


def test_build_workflow_uses_openai_when_deepseek_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert isinstance(workflow.market_analysis.technical_agent.llm_client, OpenAILLMClient)


def test_build_workflow_wraps_deepseek_with_openai_fallback_when_both_keys_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert isinstance(workflow.market_analysis.technical_agent.llm_client, FallbackLLMClient)


def test_build_workflow_uses_alpha_vantage_news_when_credentials_are_present(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "ABCDEFGHI1234")
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert workflow.settings.market_data.provider == "alpaca"
    assert isinstance(workflow.information_retrieval.news_client, CombinedNewsSearchClient)
    assert isinstance(workflow.information_retrieval.news_client.clients[0], AlphaVantageNewsSearchClient)
    assert isinstance(workflow.information_retrieval.news_client.clients[1], WebSearchNewsClient)
    assert workflow.information_retrieval.news_archive is workflow.result_store


def test_build_workflow_requires_alpaca_replay_bars_for_backtest_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("RUN_MODE", "backtest")
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Backtest OHLCV replay bars require Alpaca historical data"):
        workflow_module.build_workflow(project_root=tmp_path)


def test_build_workflow_requires_alpaca_market_data_for_live_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alpha_vantage")
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Live trading requires Alpaca market data"):
        workflow_module.build_workflow(project_root=tmp_path)


def test_format_reasoning_lines_show_agent_outputs_and_placeholders(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)
    lines = workflow_module.format_reasoning_lines(result)

    assert "Data Window:" in lines[0]
    assert "Alpha Vantage: <not configured>" in lines[1]
    assert "Technical Agent: <no output returned>" in lines[2]
    assert "News Agent: Latest available news:" in lines[3]
    assert "Risk Agent: <no output returned>" in lines[4]
    assert any(line.startswith("  Decision Reason") for line in lines)


def test_format_reasoning_lines_include_backtest_alpha_vantage_hour_chunks(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.alpha_vantage_service = StubAlphaVantageService(_sample_alpha_vantage_snapshot())
    _seed_backtest_alpha_vantage_store(workflow)

    result = workflow.run_once(symbol="AAPL", execute_orders=False)
    lines = workflow_module.format_reasoning_lines(result)

    assert any("Alpha Vantage 1h Chunk 1:" in line for line in lines)


def test_run_loop_passes_sleep_override_to_sleep_until(monkeypatch, tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    sleep_calls: list[float | None] = []

    class StopLoop(RuntimeError):
        pass

    def fake_sleep_until(_target, sleep_seconds=None):
        sleep_calls.append(sleep_seconds)
        raise StopLoop

    monkeypatch.setattr(workflow_module, "sleep_until", fake_sleep_until)

    with pytest.raises(StopLoop):
        workflow.run_loop(
            symbol="AAPL",
            execute_orders=False,
            max_iterations=2,
            sleep_seconds=0.25,
        )

    assert sleep_calls == [0.25]


def test_live_run_loop_waits_for_delayed_hourly_checkpoint(monkeypatch, tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(
            workflow.settings.trading,
            mode=RunMode.LIVE,
            live_market_data_delay_minutes=15,
        ),
        market_data=replace(workflow.settings.market_data, provider="alpaca"),
    )
    workflow.result_store.save_last_processed("AAPL", pd.Timestamp("2025-01-03T11:00:00Z"))
    sleep_targets: list[tuple[object, float | None]] = []

    class StopLoop(RuntimeError):
        pass

    def fake_sleep_until(target, sleep_seconds=None):
        sleep_targets.append((target, sleep_seconds))
        raise StopLoop

    monkeypatch.setattr(workflow_module, "sleep_until", fake_sleep_until)

    with pytest.raises(StopLoop):
        workflow.run_loop(symbol="AAPL", execute_orders=False, max_iterations=1, sleep_seconds=0.25)

    assert sleep_targets == [(pd.Timestamp("2025-01-03T12:15:00Z").to_pydatetime(), 0.25)]


def test_live_run_loop_stops_immediately_when_market_is_closed(tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.LIVE),
    )
    workflow.alpaca_service = StubClockService([False])

    results = workflow.run_loop(symbol="AAPL", execute_orders=False, max_iterations=3, sleep_seconds=0.0)

    assert results == []


def test_live_run_loop_stops_after_current_iteration_when_market_closes(monkeypatch, tmp_path: Path) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.LIVE),
    )
    workflow.alpaca_service = StubClockService([True, False])
    sleep_targets: list[tuple[object, float | None]] = []

    monkeypatch.setattr(
        workflow_module,
        "sleep_until",
        lambda target, sleep_seconds=None: sleep_targets.append((target, sleep_seconds)),
    )

    results = workflow.run_loop(symbol="AAPL", execute_orders=False, max_iterations=3, sleep_seconds=0.25)

    assert len(results) == 1
    assert sleep_targets == []


def test_backtest_run_loop_sleeps_briefly_without_using_live_hour_target(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workflow = _build_workflow(tmp_path)
    workflow.settings = replace(
        workflow.settings,
        trading=replace(workflow.settings.trading, mode=RunMode.BACKTEST),
    )
    workflow.data_collection = DataCollectionModule(
        market_data_client=HistoricalReplayDataClient(
            five_min_history=_make_backtest_bars(),
            hourly_history=pd.DataFrame(),
        ),
        account_client=SimulatedAccountClient(cash=workflow.settings.trading.starting_cash),
    )
    sleep_targets: list[tuple[object, float | None]] = []
    simulated_now = datetime(2025, 1, 3, 12, 34, tzinfo=timezone.utc)

    class StopLoop(RuntimeError):
        pass

    class FakeDatetime:
        @staticmethod
        def now(tz=None):
            return simulated_now if tz is not None else simulated_now.replace(tzinfo=None)

    def fake_sleep_until(target, sleep_seconds=None):
        sleep_targets.append((target, sleep_seconds))
        raise StopLoop

    monkeypatch.setattr(workflow_module, "datetime", FakeDatetime)
    monkeypatch.setattr(
        workflow_module,
        "next_top_of_hour",
        lambda: (_ for _ in ()).throw(AssertionError("backtest should not wait for the next real hour")),
    )
    monkeypatch.setattr(workflow_module, "sleep_until", fake_sleep_until)

    with pytest.raises(StopLoop):
        workflow.run_loop(symbol="AAPL", execute_orders=False, max_iterations=2, sleep_seconds=0.25)

    assert sleep_targets == [(simulated_now, 0.25)]


def _build_workflow(tmp_path: Path, llm_client: TextGenerationClient | None = None) -> TradingWorkflow:
    settings = Settings.from_env(project_root=tmp_path)
    technical_agent = None if llm_client is None else TechnicalAnalysisAgent(llm_client=llm_client)
    news_agent = None if llm_client is None else NewsResearchAgent(llm_client=llm_client)
    risk_agent = None if llm_client is None else RiskReviewAgent(llm_client=llm_client)
    decision_agent = None if llm_client is None else DecisionCoordinatorAgent(llm_client=llm_client)
    return TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=StaticMarketDataClient(_make_uptrend_bars()),
            account_client=SimulatedAccountClient(cash=settings.trading.starting_cash),
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(settings.trading, technical_agent=technical_agent),
        information_retrieval=InformationRetrievalModule(
            StaticNewsSearchClient(
                [
                    NewsArticle(
                        headline="AAPL earnings beat expectations with strong growth",
                        summary="Analysts point to margin expansion and resilient demand.",
                    )
                ]
            ),
            news_agent=news_agent,
        ),
        risk_analysis=RiskAnalysisModule(settings.trading, risk_agent=risk_agent),
        decision_engine=DecisionEngine(decision_agent=decision_agent),
        execution_module=ExecutionModule(InMemoryBrokerClient()),
        raw_store=LocalRawStore(settings.paths.raw_dir),
        result_store=InMemoryResultStore(),
        hold_forecast_agent=HoldForecastAgent(llm_client=llm_client),
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


def _make_backtest_bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3_000, freq="5min", tz="UTC")
    base = np.linspace(100.0, 180.0, num=len(index))
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


class StubAlphaVantageService:
    def __init__(
        self,
        snapshot: AlphaVantageIndicatorSnapshot,
        feature_frame: pd.DataFrame | None = None,
        *,
        result_store=None,
        populate_store: bool = True,
        local_snapshots: dict[str, AlphaVantageIndicatorSnapshot] | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.feature_frame = feature_frame
        self.result_store = result_store
        self.populate_store = populate_store
        self.local_snapshots = {} if local_snapshots is None else dict(local_snapshots)
        self.ensure_window_calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        self.snapshot_calls: list[tuple[str, pd.Timestamp | str | None]] = []
        self.feature_frame_calls: list[tuple[str, pd.Timestamp | str | None]] = []

    def ensure_data_for_window(
        self,
        symbol: str,
        *,
        start_time: pd.Timestamp | str,
        end_time: pd.Timestamp | str,
        required_trading_days: list[str] | None = None,
    ) -> bool:
        self.ensure_window_calls.append((symbol, pd.Timestamp(start_time), pd.Timestamp(end_time)))
        if self.result_store is not None and self.populate_store:
            trading_days = required_trading_days
            if trading_days is None:
                trading_days = [
                    candidate.isoformat()
                    for candidate in us_market_day_dates(
                        pd.date_range(
                            pd.Timestamp(start_time).normalize(),
                            pd.Timestamp(end_time).normalize(),
                            freq="1D",
                            tz="UTC",
                        )
                    )
                ]
            for trading_day in trading_days:
                self.result_store.save_alpha_vantage_indicator_snapshot(
                    _snapshot_for_trading_day(trading_day, symbol=symbol)
                )
        return True

    def load_local_snapshot(
        self,
        symbol: str,
        *,
        trading_day: str | None = None,
        end_time: pd.Timestamp | str | None = None,
    ) -> AlphaVantageIndicatorSnapshot | None:
        normalized_symbol = symbol.upper()
        if trading_day is None and end_time is not None:
            trading_day = pd.Timestamp(end_time).date().isoformat()

        if trading_day is not None:
            snapshot = None
            if self.result_store is not None:
                snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(
                    normalized_symbol,
                    trading_day=trading_day,
                    interval="5min",
                )
            if snapshot is None:
                snapshot = self.local_snapshots.get(trading_day)
            return snapshot

        if self.result_store is not None:
            snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(normalized_symbol, interval="5min")
            if snapshot is not None:
                return snapshot
        if not self.local_snapshots:
            return None
        return max(self.local_snapshots.values(), key=lambda candidate: (candidate.trading_day, candidate.latest_timestamp))

    def build_snapshot(
        self,
        symbol: str,
        *,
        end_time: pd.Timestamp | str | None = None,
    ) -> AlphaVantageIndicatorSnapshot:
        self.snapshot_calls.append((symbol, end_time))
        return self.snapshot

    def build_feature_frame(
        self,
        symbol: str,
        price_bars: pd.DataFrame,
        *,
        end_time: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        self.feature_frame_calls.append((symbol, end_time))
        frame = self.feature_frame.copy() if self.feature_frame is not None else _sample_alpha_vantage_feature_frame(price_bars)
        if end_time is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end_time)]
        return frame


class StubLivePriceService:
    def __init__(self, current_price: float) -> None:
        self.current_price = current_price

    def get_current_price(self, symbol: str) -> float:
        return self.current_price


class StubClockService:
    def __init__(self, open_states: list[bool]) -> None:
        self.open_states = list(open_states)

    def get_clock(self):
        is_open = self.open_states.pop(0) if self.open_states else False
        return type("Clock", (), {"is_open": is_open})()


class RecordingPriceMarketDataClient(StaticMarketDataClient):
    def __init__(self, frame: pd.DataFrame, *, price_to_return: float) -> None:
        super().__init__(frame)
        self.price_to_return = price_to_return
        self.price_calls: list[pd.Timestamp] = []

    def get_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        self.price_calls.append(pd.Timestamp(timestamp))
        return float(self.price_to_return)


class RecordingInformationRetrievalModule:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(
        self,
        symbol: str,
        limit: int = 10,
        *,
        input_size_chars: int | None = None,
        published_at_lte=None,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "symbol": symbol,
                "limit": limit,
                "input_size_chars": input_size_chars,
                "published_at_lte": published_at_lte,
            }
        )
        return RetrievalResult(
            symbol=symbol,
            articles=[],
            headline_summary=[],
            sentiment_score=0.0,
            critical_news=[],
            risk_flags=[],
            catalysts=[],
            summary_note=None,
        )


def _sample_alpha_vantage_snapshot() -> AlphaVantageIndicatorSnapshot:
    rows = [
        {
            "time": "2025-01-03 11:00:00",
            "RSI": 62.0,
            "ADX": 22.0,
            "threshold_hits": ["ADX_STRONG_TREND"],
        },
        {
            "time": "2025-01-03 11:05:00",
            "RSI": 72.0,
            "ADX": 28.0,
            "threshold_hits": ["RSI_OVERBOUGHT", "ADX_STRONG_TREND"],
        },
        {
            "time": "2025-01-03 11:55:00",
            "RSI": 68.0,
            "ADX": 25.0,
            "threshold_hits": ["ADX_STRONG_TREND"],
        },
    ]
    latest_chunk = IndicatorHourChunk(
        slot_start="2025-01-03 11:00:00",
        slot_end="2025-01-03 11:55:00",
        rows=rows,
    )
    return AlphaVantageIndicatorSnapshot(
        symbol="AAPL",
        interval="5min",
        trading_day="2025-01-03",
        latest_timestamp="2025-01-03 11:55:00",
        indicator_columns=["RSI", "ADX"],
        rows=rows,
        hourly_chunks=[latest_chunk],
        latest_hour_chunk=latest_chunk,
    )


def _sample_alpha_vantage_feature_frame(price_bars: pd.DataFrame) -> pd.DataFrame:
    frame = price_bars.copy()
    frame["return"] = frame["close"].pct_change().fillna(0.0)
    frame["ma_short"] = frame["close"] - 0.25
    frame["ma_long"] = frame["close"] - 1.0
    frame["rsi"] = 62.0
    frame["macd"] = 0.8
    frame["macd_signal"] = 0.5
    frame["rolling_volatility"] = 0.01
    frame["buy_trigger"] = True
    frame["sell_trigger"] = False
    return frame


def _seed_backtest_alpha_vantage_store(workflow: TradingWorkflow, symbol: str = "AAPL") -> None:
    trading_days = [
        candidate.isoformat()
        for candidate in us_market_day_dates(workflow.data_collection.fetch_five_minute_history(symbol).index)
    ]
    for trading_day in trading_days:
        workflow.result_store.save_alpha_vantage_indicator_snapshot(_snapshot_for_trading_day(trading_day, symbol=symbol))


def _snapshot_for_trading_day(trading_day: str, *, symbol: str = "AAPL") -> AlphaVantageIndicatorSnapshot:
    rows = [
        {
            "time": f"{trading_day} 11:00:00",
            "RSI": 62.0,
            "ADX": 22.0,
            "threshold_hits": ["ADX_STRONG_TREND"],
        },
        {
            "time": f"{trading_day} 11:05:00",
            "RSI": 72.0,
            "ADX": 28.0,
            "threshold_hits": ["RSI_OVERBOUGHT", "ADX_STRONG_TREND"],
        },
        {
            "time": f"{trading_day} 11:55:00",
            "RSI": 68.0,
            "ADX": 25.0,
            "threshold_hits": ["ADX_STRONG_TREND"],
        },
    ]
    latest_chunk = IndicatorHourChunk(
        slot_start=f"{trading_day} 11:00:00",
        slot_end=f"{trading_day} 11:55:00",
        rows=rows,
    )
    return AlphaVantageIndicatorSnapshot(
        symbol=symbol,
        interval="5min",
        trading_day=trading_day,
        latest_timestamp=f"{trading_day} 11:55:00",
        indicator_columns=["RSI", "ADX"],
        rows=rows,
        hourly_chunks=[latest_chunk],
        latest_hour_chunk=latest_chunk,
    )
