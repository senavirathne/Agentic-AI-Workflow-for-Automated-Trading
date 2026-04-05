from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project.agents import DecisionCoordinatorAgent
from project.config import Settings
from project.decision_engine import DecisionEngine
from project.features import FeatureEngineeringModule
from project.llm import TextGenerationClient
from project.market_analysis import MarketAnalysisModule
from project.models import AccountState, AnalysisResult, RetrievalResult, RiskResult


def test_market_analysis_identifies_support_and_resistance_from_history(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    features = FeatureEngineeringModule(settings.trading).build(_make_range_bound_bars())
    analysis = MarketAnalysisModule(settings.trading).analyze("AAPL", features)

    assert analysis.support_levels
    assert analysis.resistance_levels
    assert analysis.support_regions
    assert analysis.resistance_regions
    assert analysis.nearest_support is not None
    assert analysis.nearest_resistance is not None
    assert analysis.nearest_support_region is not None
    assert analysis.nearest_resistance_region is not None
    assert analysis.nearest_support_region_strength >= 1
    assert analysis.nearest_resistance_region_strength >= 1
    assert analysis.nearest_support < analysis.latest_price < analysis.nearest_resistance
    assert analysis.nearest_support_region[0] <= analysis.nearest_support <= analysis.nearest_support_region[1]
    assert analysis.nearest_resistance_region[0] <= analysis.nearest_resistance <= analysis.nearest_resistance_region[1]


def test_market_analysis_combines_hourly_major_levels_with_five_minute_local_regions(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    features = FeatureEngineeringModule(settings.trading).build(_make_range_bound_bars())
    analysis = MarketAnalysisModule(settings.trading).analyze("AAPL", features, _make_hourly_major_bars())

    assert analysis.support_levels
    assert analysis.resistance_levels
    assert analysis.support_regions
    assert analysis.resistance_regions
    assert analysis.nearest_support is not None
    assert analysis.nearest_resistance is not None
    assert analysis.nearest_support_region is not None
    assert analysis.nearest_resistance_region is not None
    assert analysis.nearest_support < analysis.nearest_support_region[0]
    assert analysis.nearest_resistance > analysis.nearest_resistance_region[1]
    assert any("hourly pivot candles" in note for note in analysis.notes)
    assert any("5-minute pivot candles" in note for note in analysis.notes)


def test_decision_engine_blocks_buy_when_price_is_near_resistance() -> None:
    engine = DecisionEngine(resistance_buffer_pct=1.0)
    analysis = AnalysisResult(
        symbol="AAPL",
        timestamp=pd.Timestamp("2025-01-01T12:00:00Z"),
        latest_price=109.8,
        trend="bullish",
        bullish=True,
        entry_setup=True,
        exit_setup=False,
        confidence=0.8,
        support_levels=[100.0, 102.0],
        resistance_levels=[110.0, 112.0],
        support_regions=[(99.4, 100.6), (101.4, 102.6)],
        resistance_regions=[(109.4, 110.6), (111.4, 112.6)],
        support_region_strengths=[2, 2],
        resistance_region_strengths=[3, 2],
        nearest_support=102.0,
        nearest_resistance=110.0,
        nearest_support_region=(101.4, 102.6),
        nearest_resistance_region=(109.4, 110.6),
        nearest_support_region_strength=2,
        nearest_resistance_region_strength=3,
        distance_to_support_pct=7.65,
        distance_to_resistance_pct=0.18,
    )
    retrieval = RetrievalResult(symbol="AAPL", articles=[], headline_summary=[], sentiment_score=0.3)
    risk = RiskResult(symbol="AAPL", risk_score=0.2, can_enter=True, recommended_qty=5)

    decision = engine.decide(AccountState(cash=10_000.0), analysis, retrieval, risk)

    assert decision.action.value == "HOLD"
    assert any("resistance" in item.lower() for item in decision.rationale)


def test_decision_engine_sells_when_price_breaks_below_support_region() -> None:
    engine = DecisionEngine()
    analysis = AnalysisResult(
        symbol="AAPL",
        timestamp=pd.Timestamp("2025-01-01T12:00:00Z"),
        latest_price=99.0,
        trend="bearish",
        bullish=False,
        entry_setup=False,
        exit_setup=False,
        confidence=0.5,
        support_levels=[100.0],
        resistance_levels=[110.0],
        support_regions=[(99.5, 100.5)],
        resistance_regions=[(109.5, 110.5)],
        support_region_strengths=[2],
        resistance_region_strengths=[1],
        nearest_support=100.0,
        nearest_resistance=110.0,
        nearest_support_region=(99.5, 100.5),
        nearest_resistance_region=(109.5, 110.5),
        nearest_support_region_strength=2,
        nearest_resistance_region_strength=1,
        distance_to_support_pct=-1.0,
        distance_to_resistance_pct=10.0,
    )
    retrieval = RetrievalResult(symbol="AAPL", articles=[], headline_summary=[], sentiment_score=0.1)
    risk = RiskResult(symbol="AAPL", risk_score=0.3, can_enter=False, recommended_qty=0)

    decision = engine.decide(AccountState(cash=10_000.0, position_qty=4), analysis, retrieval, risk)

    assert decision.action.value == "SELL"
    assert any("support region" in item.lower() for item in decision.rationale)


def test_decision_engine_can_use_llm_review_without_changing_simple_module_flow() -> None:
    engine = DecisionEngine(
        decision_agent=DecisionCoordinatorAgent(
            llm_client=StubLLM(["OVERRIDE=HOLD\nNOTE=Wait for a cleaner entry above resistance."])
        )
    )
    analysis = AnalysisResult(
        symbol="AAPL",
        timestamp=pd.Timestamp("2025-01-01T12:00:00Z"),
        latest_price=105.0,
        trend="bullish",
        bullish=True,
        entry_setup=True,
        exit_setup=False,
        confidence=0.8,
        support_levels=[101.0],
        resistance_levels=[110.0],
        support_regions=[(100.4, 101.6)],
        resistance_regions=[(109.4, 110.6)],
        support_region_strengths=[2],
        resistance_region_strengths=[2],
        nearest_support=101.0,
        nearest_resistance=110.0,
        nearest_support_region=(100.4, 101.6),
        nearest_resistance_region=(109.4, 110.6),
        nearest_support_region_strength=2,
        nearest_resistance_region_strength=2,
        distance_to_support_pct=3.96,
        distance_to_resistance_pct=4.76,
    )
    retrieval = RetrievalResult(
        symbol="AAPL",
        articles=[],
        headline_summary=[],
        sentiment_score=0.4,
        summary_note="News flow is constructive."
    )
    risk = RiskResult(
        symbol="AAPL",
        risk_score=0.2,
        can_enter=True,
        recommended_qty=5,
        summary_note="Risk is acceptable for a starter position.",
    )

    decision = engine.decide(AccountState(cash=10_000.0), analysis, retrieval, risk)

    assert decision.action.value == "HOLD"
    assert any("LLM review" in item for item in decision.rationale)
    assert any("vetoed" in item.lower() for item in decision.rationale)


def _make_range_bound_bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=360, freq="5min", tz="UTC")
    base = 105 + 4 * np.sin(np.linspace(0, 12 * np.pi, len(index)))
    return pd.DataFrame(
        {
            "open": base - 0.2,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base,
            "volume": np.full(len(index), 20_000),
        },
        index=index,
    )


def _make_hourly_major_bars() -> pd.DataFrame:
    index = pd.date_range("2024-12-28", periods=120, freq="1h", tz="UTC")
    base = 105 + 10 * np.sin(np.linspace(0, 8 * np.pi, len(index)))
    return pd.DataFrame(
        {
            "open": base - 0.4,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base,
            "volume": np.full(len(index), 8_000),
        },
        index=index,
    )


class StubLLM(TextGenerationClient):
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def generate(self, system_prompt: str, content: str) -> str | None:
        if not self._responses:
            return None
        return self._responses.pop(0)
