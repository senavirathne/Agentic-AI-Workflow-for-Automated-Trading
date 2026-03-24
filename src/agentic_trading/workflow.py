from __future__ import annotations

import logging
from dataclasses import dataclass, field

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from .agents import (
    DecisionAgent,
    DeepSeekLLMClient,
    ExecutionAgent,
    InformationRetrievalAgent,
    MarketAnalysisAgent,
    RiskManagementAgent,
    TradingAgentGraph,
)
from .alpaca_clients import AlpacaService
from .config import Settings
from .models import MarketContext, WorkflowResult
from .storage import AgentMemoryStore, CloudObjectStorePlaceholder, LocalDataLake, SQLiteStructuredStore
from .utils import next_top_of_hour, sleep_until


@dataclass
class TradingWorkflow:
    settings: Settings
    service: AlpacaService = field(init=False)
    raw_lake: LocalDataLake | CloudObjectStorePlaceholder = field(init=False)
    structured_store: SQLiteStructuredStore = field(init=False)
    memory_store: AgentMemoryStore = field(init=False)
    analysis_agent: MarketAnalysisAgent = field(init=False)
    retrieval_agent: InformationRetrievalAgent = field(init=False)
    risk_agent: RiskManagementAgent = field(init=False)
    decision_agent: DecisionAgent = field(init=False)
    execution_agent: ExecutionAgent = field(init=False)
    agent_graph: TradingAgentGraph = field(init=False)
    llm_client: DeepSeekLLMClient | None = field(init=False)

    def __post_init__(self) -> None:
        self.service = AlpacaService(self.settings)
        self.raw_lake = (
            CloudObjectStorePlaceholder(self.settings.cloud.provider, self.settings.cloud.raw_store_uri)
            if self.settings.cloud.uses_placeholder_cloud
            else LocalDataLake(self.settings.paths.raw_dir)
        )
        self.structured_store = SQLiteStructuredStore(self.settings.paths.database_path)
        self.memory_store = AgentMemoryStore(self.settings.paths.database_path)
        self.llm_client = DeepSeekLLMClient(self.settings.llm) if self.settings.llm.enabled else None
        self.analysis_agent = MarketAnalysisAgent(self.settings.trading)
        self.retrieval_agent = InformationRetrievalAgent()
        self.risk_agent = RiskManagementAgent(self.settings.trading)
        self.decision_agent = DecisionAgent(llm_client=self.llm_client)
        self.execution_agent = ExecutionAgent(self.service)
        self.agent_graph = TradingAgentGraph(
            analysis_agent=self.analysis_agent,
            retrieval_agent=self.retrieval_agent,
            risk_agent=self.risk_agent,
            decision_agent=self.decision_agent,
            execution_agent=self.execution_agent,
        )

    def run_once(self, symbol: str | None = None, execute_orders: bool = False) -> WorkflowResult:
        symbol = (symbol or self.settings.trading.primary_symbol).upper()
        analysis_memory = self.memory_store.load_latest_memory("market_analysis", symbol)
        retrieval_memory = self.memory_store.load_latest_memory("information_retrieval", symbol)
        risk_memory = self.memory_store.load_latest_memory("risk_management", symbol)
        decision_memory = self.memory_store.load_latest_memory("decision", symbol)
        trading = self.settings.trading
        short_timeframe = TimeFrame(
            amount=trading.short_timeframe_multiplier,
            unit=trading.short_timeframe,
        )
        medium_timeframe = TimeFrame(
            amount=trading.medium_timeframe_multiplier,
            unit=trading.medium_timeframe,
        )
        long_timeframe = TimeFrame(amount=1, unit=trading.long_timeframe)

        short_bars = self.service.fetch_stock_bars(
            [symbol],
            short_timeframe,
            trading.short_lookback_days,
        )[symbol]
        medium_bars = self.service.fetch_stock_bars(
            [symbol],
            medium_timeframe,
            trading.medium_lookback_days,
        )[symbol]
        long_bars = self.service.fetch_stock_bars(
            [symbol],
            long_timeframe,
            trading.long_lookback_days,
        )[symbol]
        news = self.service.fetch_recent_news(
            symbol=symbol,
            days=trading.news_lookback_days,
        )
        latest_price = self.service.fetch_latest_trade(symbol)
        position_open, current_qty, avg_entry_price = self._get_position_state(symbol)
        buying_power = self.service.get_buying_power()
        clock = self.service.get_clock()

        self.raw_lake.save_bars(
            symbol,
            _timeframe_storage_key(trading.short_timeframe, trading.short_timeframe_multiplier),
            short_bars,
        )
        self.raw_lake.save_bars(
            symbol,
            _timeframe_storage_key(trading.medium_timeframe, trading.medium_timeframe_multiplier),
            medium_bars,
        )
        self.raw_lake.save_bars(
            symbol,
            _timeframe_storage_key(trading.long_timeframe),
            long_bars,
        )
        self.raw_lake.save_news(symbol, news)

        context = MarketContext(
            symbol=symbol,
            short_bars=short_bars,
            medium_bars=medium_bars,
            long_bars=long_bars,
            latest_price=latest_price,
            market_open=clock.is_open,
            buying_power=buying_power,
            position_open=position_open,
            current_qty=current_qty,
            avg_entry_price=avg_entry_price,
            news=news,
        )
        result = self.agent_graph.run(
            context,
            execute_orders=execute_orders,
            analysis_memory=analysis_memory,
            retrieval_memory=retrieval_memory,
            risk_memory=risk_memory,
            decision_memory=decision_memory,
        )
        cycle_timestamp = result.analysis.latest_timestamp.isoformat()
        self.memory_store.save_memory(
            "market_analysis",
            symbol,
            cycle_timestamp,
            self.analysis_agent.build_memory(result.analysis, analysis_memory, cycle_timestamp),
        )
        self.memory_store.save_memory(
            "information_retrieval",
            symbol,
            cycle_timestamp,
            self.retrieval_agent.build_memory(
                result.retrieval,
                retrieval_memory,
                self.memory_store.load_memory_window("information_retrieval", symbol, hours=24),
                cycle_timestamp,
            ),
        )
        self.memory_store.save_memory(
            "risk_management",
            symbol,
            cycle_timestamp,
            self.risk_agent.build_memory(
                result.context,
                result.risk,
                risk_memory,
                self.memory_store.load_memory_window("risk_management", symbol, hours=24),
                cycle_timestamp,
            ),
        )
        self.memory_store.save_memory(
            "decision",
            symbol,
            cycle_timestamp,
            self.decision_agent.build_memory(
                result.context,
                result.decision,
                decision_memory,
                self.memory_store.load_memory_window("decision", symbol, hours=24),
                cycle_timestamp,
            ),
        )
        self.structured_store.save_workflow_run(result)
        return result

    def run_loop(
        self,
        symbol: str | None = None,
        execute_orders: bool = False,
        max_iterations: int | None = None,
    ) -> list[WorkflowResult]:
        market_open = self.service.get_clock().is_open
        results: list[WorkflowResult] = []

        while max_iterations is None or len(results) < max_iterations:
            clock = self.service.get_clock()
            if market_open and not clock.is_open:
                logging.info("Market closed. Sleeping until %s", clock.next_open)
                market_open = False
                sleep_until(clock.next_open)
                continue
            if not market_open and clock.is_open:
                logging.info("Market opened. Resuming workflow.")
                market_open = True
            if not clock.is_open:
                logging.info("Market is closed. Sleeping until %s", clock.next_open)
                sleep_until(clock.next_open)
                continue

            results.append(self.run_once(symbol=symbol, execute_orders=execute_orders))
            sleep_until(next_top_of_hour())

        return results

    def _get_position_state(self, symbol: str) -> tuple[bool, int, float | None]:
        position_open, current_qty, avg_entry_price = self.service.get_open_position(symbol)
        return position_open, current_qty, avg_entry_price


def _timeframe_storage_key(timeframe_unit: TimeFrameUnit, multiplier: int = 1) -> str:
    return f"{multiplier}{timeframe_unit.name.lower()}"
