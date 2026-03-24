from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from agentic_trading.config import CloudConfig, Credentials, LLMConfig, Paths, Settings, TradingConfig
from agentic_trading.models import Action, AnalysisResult, Decision, MarketContext, RetrievalResult, RiskPlan, WorkflowResult
from agentic_trading.storage import AgentMemoryStore
from agentic_trading.workflow import TradingWorkflow


def test_agent_memory_store_round_trip(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path / "memory.sqlite")

    assert store.load_latest_memory("market_analysis", "TQQQ") is None
    assert store.load_memory_window("market_analysis", "TQQQ") == []

    store.save_memory(
        "market_analysis",
        "TQQQ",
        "2025-01-01T12:00:00+00:00",
        {"rsi": 41.2, "in_uptrend": True},
    )
    store.save_memory(
        "market_analysis",
        "TQQQ",
        "2025-01-01T13:00:00+00:00",
        {"rsi": 43.5, "in_uptrend": True},
    )

    latest = store.load_latest_memory("market_analysis", "TQQQ")
    window = store.load_memory_window("market_analysis", "TQQQ", hours=24)

    assert latest is not None
    assert latest["rsi"] == 43.5
    assert [entry["rsi"] for entry in window] == [41.2, 43.5]


def test_workflow_loads_and_saves_agent_memories(tmp_path: Path) -> None:
    workflow = TradingWorkflow(_make_settings(tmp_path))
    context = _make_context()
    result = WorkflowResult(
        context=context,
        analysis=_make_analysis(),
        retrieval=_make_retrieval(),
        risk=_make_risk(),
        decision=_make_decision(),
        order_id=None,
    )

    memory_store = StubMemoryStore()
    graph = StubAgentGraph(result)
    workflow.service = StubService(context.short_bars, context.news, context.latest_price)
    workflow.raw_lake = StubRawLake()
    workflow.structured_store = StubStructuredStore()
    workflow.memory_store = memory_store
    workflow.analysis_agent = StubAnalysisMemoryAgent()
    workflow.retrieval_agent = StubRetrievalMemoryAgent()
    workflow.risk_agent = StubRiskMemoryAgent()
    workflow.decision_agent = StubDecisionMemoryAgent()
    workflow.agent_graph = graph

    run_result = workflow.run_once(symbol="TQQQ", execute_orders=False)

    assert run_result is result
    assert memory_store.loaded == [
        ("market_analysis", "TQQQ"),
        ("information_retrieval", "TQQQ"),
        ("risk_management", "TQQQ"),
        ("decision", "TQQQ"),
    ]
    assert graph.calls[0]["analysis_memory"] == {"rsi": 40.0}
    assert graph.calls[0]["retrieval_memory"] == {"positive_hits": 2}
    assert graph.calls[0]["risk_memory"] == {"buying_power": 9_500.0}
    assert graph.calls[0]["decision_memory"] == {"consecutive_holds": 2}
    assert [item[0] for item in memory_store.saved] == [
        "market_analysis",
        "information_retrieval",
        "risk_management",
        "decision",
    ]


def _make_settings(tmp_path: Path) -> Settings:
    paths = Paths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        database_path=tmp_path / "workflow.sqlite",
    )
    paths.create_directories()
    return Settings(
        credentials=Credentials(api_key="key", api_secret="secret"),
        trading=TradingConfig(),
        cloud=CloudConfig(),
        llm=LLMConfig(),
        paths=paths,
    )


def _make_context() -> MarketContext:
    index = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1_000] * 5,
        },
        index=index,
    )
    return MarketContext(
        symbol="TQQQ",
        short_bars=bars,
        medium_bars=bars,
        long_bars=bars,
        latest_price=104.0,
        market_open=True,
        buying_power=10_000.0,
        position_open=False,
        current_qty=0,
        news=[{"headline": "Example headline"}],
    )


def _make_analysis() -> AnalysisResult:
    return AnalysisResult(
        symbol="TQQQ",
        latest_timestamp=pd.Timestamp("2025-01-01T16:00:00+00:00"),
        rsi_now=51.0,
        macd_now=0.2,
        signal_now=0.15,
        ma_fast=103.0,
        ma_mid=102.0,
        ma_slow=101.0,
        in_uptrend=True,
        entry_setup=False,
        exit_setup=False,
        signal_frame=pd.DataFrame(
            {
                "medium_trend_bullish": [True],
                "in_uptrend": [True],
            },
            index=[pd.Timestamp("2025-01-01T16:00:00+00:00")],
        ),
        notes=["Trend filter is bullish on the daily timeframe."],
    )


def _make_retrieval() -> RetrievalResult:
    return RetrievalResult(
        symbol="TQQQ",
        articles=[{"headline": "Example headline"}],
        headline_summary=["Example headline"],
        positive_hits=2,
        negative_hits=0,
        risk_flags=[],
    )


def _make_risk() -> RiskPlan:
    return RiskPlan(
        symbol="TQQQ",
        max_notional=200.0,
        recommended_qty=1,
        can_enter=True,
        notes=[],
    )


def _make_decision() -> Decision:
    return Decision(
        symbol="TQQQ",
        action=Action.HOLD,
        quantity=0,
        confidence=0.55,
        rationale=["No actionable setup, holding."],
        metadata={"decision_source": "rules"},
    )


class StubService:
    def __init__(self, bars: pd.DataFrame, news: list[dict], latest_price: float) -> None:
        self.bars = bars
        self.news = news
        self.latest_price = latest_price

    def fetch_stock_bars(self, symbols: list[str], timeframe, days: int) -> dict[str, pd.DataFrame]:
        return {symbols[0]: self.bars}

    def fetch_recent_news(self, symbol: str, days: int) -> list[dict]:
        return self.news

    def fetch_latest_trade(self, symbol: str) -> float:
        return self.latest_price

    def get_open_position(self, symbol: str) -> tuple[bool, int, float | None]:
        return False, 0, None

    def get_buying_power(self) -> float:
        return 10_000.0

    def get_clock(self) -> SimpleNamespace:
        return SimpleNamespace(is_open=True, next_open=None)


class StubRawLake:
    def save_bars(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        return None

    def save_news(self, symbol: str, news: list[dict]) -> None:
        return None


class StubStructuredStore:
    def __init__(self) -> None:
        self.saved_results: list[WorkflowResult] = []

    def save_workflow_run(self, result: WorkflowResult) -> None:
        self.saved_results.append(result)


class StubMemoryStore:
    def __init__(self) -> None:
        self.loaded: list[tuple[str, str]] = []
        self.saved: list[tuple[str, str, str, dict]] = []
        self.latest = {
            "market_analysis": {"rsi": 40.0},
            "information_retrieval": {"positive_hits": 2},
            "risk_management": {"buying_power": 9_500.0},
            "decision": {"consecutive_holds": 2},
        }

    def load_latest_memory(self, agent_name: str, symbol: str) -> dict | None:
        self.loaded.append((agent_name, symbol))
        return self.latest.get(agent_name)

    def load_memory_window(self, agent_name: str, symbol: str, hours: int = 24) -> list[dict]:
        return []

    def save_memory(self, agent_name: str, symbol: str, cycle_timestamp: str, memory: dict) -> None:
        self.saved.append((agent_name, symbol, cycle_timestamp, memory))


class StubAgentGraph:
    def __init__(self, result: WorkflowResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    def run(
        self,
        context: MarketContext,
        execute_orders: bool = False,
        analysis_memory: dict | None = None,
        retrieval_memory: dict | None = None,
        risk_memory: dict | None = None,
        decision_memory: dict | None = None,
    ) -> WorkflowResult:
        self.calls.append(
            {
                "context": context,
                "execute_orders": execute_orders,
                "analysis_memory": analysis_memory,
                "retrieval_memory": retrieval_memory,
                "risk_memory": risk_memory,
                "decision_memory": decision_memory,
            }
        )
        return self.result


@dataclass
class StubAnalysisMemoryAgent:
    def build_memory(self, analysis: AnalysisResult, previous_memory: dict | None, cycle_timestamp: str) -> dict:
        return {"agent": "analysis", "previous": previous_memory, "cycle_timestamp": cycle_timestamp}


@dataclass
class StubRetrievalMemoryAgent:
    def build_memory(
        self,
        retrieval: RetrievalResult,
        previous_memory: dict | None,
        recent_memories: list[dict] | None,
        cycle_timestamp: str,
    ) -> dict:
        return {
            "agent": "retrieval",
            "previous": previous_memory,
            "recent_memories": recent_memories or [],
            "cycle_timestamp": cycle_timestamp,
        }


@dataclass
class StubRiskMemoryAgent:
    def build_memory(
        self,
        context: MarketContext,
        risk: RiskPlan,
        previous_memory: dict | None,
        recent_memories: list[dict] | None,
        cycle_timestamp: str,
    ) -> dict:
        return {
            "agent": "risk",
            "previous": previous_memory,
            "recent_memories": recent_memories or [],
            "cycle_timestamp": cycle_timestamp,
        }


@dataclass
class StubDecisionMemoryAgent:
    def build_memory(
        self,
        context: MarketContext,
        decision: Decision,
        previous_memory: dict | None,
        recent_memories: list[dict] | None,
        cycle_timestamp: str,
    ) -> dict:
        return {
            "agent": "decision",
            "previous": previous_memory,
            "recent_memories": recent_memories or [],
            "cycle_timestamp": cycle_timestamp,
        }
