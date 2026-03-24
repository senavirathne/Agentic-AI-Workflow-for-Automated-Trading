from __future__ import annotations

import pandas as pd

from agentic_trading.agents import DecisionAgent, InformationRetrievalAgent, TradingAgentGraph
from agentic_trading.models import Action, AnalysisResult, Decision, MarketContext, RetrievalResult, RiskPlan


class StubAnalysisAgent:
    def __init__(self, result: AnalysisResult, call_log: list[str]) -> None:
        self.result = result
        self.call_log = call_log
        self.memory: dict | None = None

    def analyze(self, context: MarketContext, memory: dict | None = None) -> AnalysisResult:
        self.call_log.append("analysis")
        self.memory = memory
        return self.result


class StubRetrievalAgent:
    def __init__(self, result: RetrievalResult, call_log: list[str]) -> None:
        self.result = result
        self.call_log = call_log
        self.memory: dict | None = None

    def retrieve(self, symbol: str, articles: list[dict], memory: dict | None = None) -> RetrievalResult:
        self.call_log.append("retrieval")
        self.memory = memory
        return self.result


class StubRiskAgent:
    def __init__(self, result: RiskPlan, call_log: list[str]) -> None:
        self.result = result
        self.call_log = call_log
        self.memory: dict | None = None

    def evaluate(self, context: MarketContext, memory: dict | None = None) -> RiskPlan:
        self.call_log.append("risk")
        self.memory = memory
        return self.result


class StubDecisionAgent:
    def __init__(self, result: Decision, call_log: list[str]) -> None:
        self.result = result
        self.call_log = call_log
        self.memory: dict | None = None

    def decide(
        self,
        context: MarketContext,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskPlan,
        memory: dict | None = None,
    ) -> Decision:
        self.call_log.append("decision")
        self.memory = memory
        return self.result


class StubExecutionAgent:
    def __init__(self, order_id: str, call_log: list[str]) -> None:
        self.order_id = order_id
        self.call_log = call_log

    def execute(self, decision: Decision, execute_orders: bool) -> str | None:
        self.call_log.append("execution")
        return self.order_id if execute_orders else None


class StubLLMClient:
    def __init__(self, response: str | None) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        return self.response


def _make_context() -> MarketContext:
    return MarketContext(
        symbol="TQQQ",
        short_bars=pd.DataFrame(),
        medium_bars=pd.DataFrame(),
        long_bars=pd.DataFrame(),
        latest_price=100.0,
        market_open=True,
        buying_power=10_000.0,
        position_open=False,
        current_qty=0,
        news=[{"headline": "Example headline"}],
    )


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        symbol="TQQQ",
        latest_timestamp=pd.Timestamp("2025-01-01T00:00:00Z"),
        rsi_now=52.0,
        macd_now=0.2,
        signal_now=0.1,
        ma_fast=100.0,
        ma_mid=99.0,
        ma_slow=98.0,
        in_uptrend=True,
        entry_setup=True,
        exit_setup=False,
        signal_frame=pd.DataFrame(),
        notes=["Entry setup is active."],
    )


def _make_retrieval() -> RetrievalResult:
    return RetrievalResult(
        symbol="TQQQ",
        articles=[{"headline": "Example headline"}],
        headline_summary=["Example headline"],
        positive_hits=1,
        negative_hits=0,
        risk_flags=[],
    )


def _make_risk() -> RiskPlan:
    return RiskPlan(
        symbol="TQQQ",
        max_notional=500.0,
        recommended_qty=5,
        can_enter=True,
        notes=[],
    )


def test_trading_agent_graph_executes_order_when_decision_requires_it() -> None:
    call_log: list[str] = []
    decision = Decision(
        symbol="TQQQ",
        action=Action.BUY,
        quantity=5,
        confidence=0.8,
        rationale=["Entry setup and risk gate passed."],
    )
    graph = TradingAgentGraph(
        analysis_agent=StubAnalysisAgent(_make_analysis(), call_log),
        retrieval_agent=StubRetrievalAgent(_make_retrieval(), call_log),
        risk_agent=StubRiskAgent(_make_risk(), call_log),
        decision_agent=StubDecisionAgent(decision, call_log),
        execution_agent=StubExecutionAgent("order-123", call_log),
    )

    result = graph.run(_make_context(), execute_orders=True)

    assert result.decision.action == Action.BUY
    assert result.order_id == "order-123"
    assert call_log == ["analysis", "retrieval", "risk", "decision", "execution"]


def test_trading_agent_graph_skips_execution_when_order_submission_is_disabled() -> None:
    call_log: list[str] = []
    decision = Decision(
        symbol="TQQQ",
        action=Action.BUY,
        quantity=5,
        confidence=0.8,
        rationale=["Entry setup and risk gate passed."],
    )
    graph = TradingAgentGraph(
        analysis_agent=StubAnalysisAgent(_make_analysis(), call_log),
        retrieval_agent=StubRetrievalAgent(_make_retrieval(), call_log),
        risk_agent=StubRiskAgent(_make_risk(), call_log),
        decision_agent=StubDecisionAgent(decision, call_log),
        execution_agent=StubExecutionAgent("order-123", call_log),
    )

    result = graph.run(_make_context(), execute_orders=False)

    assert result.decision.action == Action.BUY
    assert result.order_id is None
    assert call_log == ["analysis", "retrieval", "risk", "decision"]


def test_trading_agent_graph_passes_loaded_memories_into_agents() -> None:
    call_log: list[str] = []
    analysis_agent = StubAnalysisAgent(_make_analysis(), call_log)
    retrieval_agent = StubRetrievalAgent(_make_retrieval(), call_log)
    risk_agent = StubRiskAgent(_make_risk(), call_log)
    decision_agent = StubDecisionAgent(
        Decision(
            symbol="TQQQ",
            action=Action.HOLD,
            quantity=0,
            confidence=0.5,
            rationale=["No setup."],
        ),
        call_log,
    )
    graph = TradingAgentGraph(
        analysis_agent=analysis_agent,
        retrieval_agent=retrieval_agent,
        risk_agent=risk_agent,
        decision_agent=decision_agent,
        execution_agent=StubExecutionAgent("order-123", call_log),
    )

    graph.run(
        _make_context(),
        analysis_memory={"rsi": 40.0},
        retrieval_memory={"positive_hits": 2},
        risk_memory={"buying_power": 9_000.0},
        decision_memory={"consecutive_holds": 3},
    )

    assert analysis_agent.memory == {"rsi": 40.0}
    assert retrieval_agent.memory == {"positive_hits": 2}
    assert risk_agent.memory == {"buying_power": 9_000.0}
    assert decision_agent.memory == {"consecutive_holds": 3}


def test_information_retrieval_agent_collects_structured_news_inputs() -> None:
    agent = InformationRetrievalAgent(max_news_items=5)

    result = agent.retrieve(
        "TQQQ",
        [
            {"headline": "TQQQ sees strong inflows", "summary": "Demand stays elevated with profit growth."},
            {"headline": "Analysts cite stable leverage profile", "summary": "Risk remains contained."},
            {"headline": "Regulators open lawsuit review", "summary": "Probe expands across leveraged ETFs."},
        ],
    )

    assert result.headline_summary == [
        "TQQQ sees strong inflows",
        "Analysts cite stable leverage profile",
        "Regulators open lawsuit review",
    ]
    assert result.positive_hits > 0
    assert result.negative_hits > 0
    assert "At least one headline contains a high-risk keyword." in result.risk_flags


def test_decision_agent_can_veto_rule_based_buy_via_llm_review() -> None:
    llm_client = StubLLMClient(
        '{"action": "HOLD", "quantity": 0, "confidence": 0.83, '
        '"rationale": ["Recent headlines add event risk to a fresh entry."]}'
    )
    agent = DecisionAgent(llm_client=llm_client)

    result = agent.decide(
        _make_context(),
        _make_analysis(),
        _make_retrieval(),
        _make_risk(),
    )

    assert result.action == Action.HOLD
    assert result.quantity == 0
    assert result.metadata["decision_source"] == "llm_entry_veto"
    assert any("downgraded the entry to hold" in item for item in result.rationale)


def test_decision_agent_does_not_allow_llm_to_invent_new_buy_signal() -> None:
    llm_client = StubLLMClient(
        '{"action": "BUY", "quantity": 5, "confidence": 0.91, '
        '"rationale": ["Momentum could improve from here."]}'
    )
    agent = DecisionAgent(llm_client=llm_client)
    analysis = _make_analysis()
    analysis.entry_setup = False

    result = agent.decide(
        _make_context(),
        analysis,
        _make_retrieval(),
        _make_risk(),
    )

    assert result.action == Action.HOLD
    assert result.quantity == 0
    assert result.metadata["decision_source"] == "rules_with_llm_review"


def test_decision_agent_includes_memory_context_in_llm_prompt() -> None:
    llm_client = StubLLMClient(
        '{"action": "HOLD", "quantity": 0, "confidence": 0.50, "rationale": ["Stay patient."]}'
    )
    agent = DecisionAgent(llm_client=llm_client)

    agent.decide(
        _make_context(),
        _make_analysis(),
        _make_retrieval(),
        _make_risk(),
        memory={
            "consecutive_holds": 2,
            "avg_confidence_24h": 0.62,
            "win_rate_recent": 0.67,
            "last_buy_timestamp": "2025-01-01T12:00:00+00:00",
            "last_sell_timestamp": "2025-01-01T14:00:00+00:00",
            "decisions_24h": {"BUY": 1, "SELL": 1, "HOLD": 6},
        },
    )

    assert llm_client.calls
    _, prompt = llm_client.calls[0]
    assert "Previous average confidence (24h): 0.62" in prompt
    assert "Previous recent win rate: 0.67" in prompt
    assert "Previous decisions_24h: {'BUY': 1, 'SELL': 1, 'HOLD': 6}" in prompt
