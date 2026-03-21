from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .agents import DecisionAgent, ExecutionAgent, InformationRetrievalAgent, MarketAnalysisAgent, RiskManagementAgent
from .alpaca_clients import AlpacaService
from .config import Settings
from .models import MarketContext, WorkflowResult
from .storage import CloudObjectStorePlaceholder, LocalDataLake, SQLiteStructuredStore
from .utils import next_top_of_hour, sleep_until


@dataclass
class TradingWorkflow:
    settings: Settings
    service: AlpacaService = field(init=False)
    raw_lake: LocalDataLake | CloudObjectStorePlaceholder = field(init=False)
    structured_store: SQLiteStructuredStore = field(init=False)
    analysis_agent: MarketAnalysisAgent = field(init=False)
    retrieval_agent: InformationRetrievalAgent = field(init=False)
    risk_agent: RiskManagementAgent = field(init=False)
    decision_agent: DecisionAgent = field(init=False)
    execution_agent: ExecutionAgent = field(init=False)

    def __post_init__(self) -> None:
        self.service = AlpacaService(self.settings)
        self.raw_lake = (
            CloudObjectStorePlaceholder(self.settings.cloud.provider, self.settings.cloud.raw_store_uri)
            if self.settings.cloud.uses_placeholder_cloud
            else LocalDataLake(self.settings.paths.raw_dir)
        )
        self.structured_store = SQLiteStructuredStore(self.settings.paths.database_path)
        self.analysis_agent = MarketAnalysisAgent(self.settings.trading)
        self.retrieval_agent = InformationRetrievalAgent()
        self.risk_agent = RiskManagementAgent(self.settings.trading)
        self.decision_agent = DecisionAgent()
        self.execution_agent = ExecutionAgent(self.service)

    def run_once(self, symbol: str | None = None, execute_orders: bool = False) -> WorkflowResult:
        symbol = (symbol or self.settings.trading.primary_symbol).upper()
        main_bars = self.service.fetch_stock_bars(
            [symbol],
            self.settings.trading.main_timeframe,
            self.settings.trading.main_lookback_days,
        )[symbol]
        trend_bars = self.service.fetch_stock_bars(
            [symbol],
            self.settings.trading.trend_timeframe,
            self.settings.trading.trend_lookback_days,
        )[symbol]
        news = self.service.fetch_recent_news(
            symbol=symbol,
            days=self.settings.trading.news_lookback_days,
        )
        latest_price = self.service.fetch_latest_trade(symbol)
        position_open, current_qty, avg_entry_price = self._get_position_state(symbol)
        buying_power = self.service.get_buying_power()

        self.raw_lake.save_bars(symbol, self.settings.trading.main_timeframe.value, main_bars)
        self.raw_lake.save_bars(symbol, self.settings.trading.trend_timeframe.value, trend_bars)
        self.raw_lake.save_news(symbol, news)

        context = MarketContext(
            symbol=symbol,
            main_bars=main_bars,
            trend_bars=trend_bars,
            latest_price=latest_price,
            market_open=self.service.get_clock().is_open,
            buying_power=buying_power,
            position_open=position_open,
            current_qty=current_qty,
            avg_entry_price=avg_entry_price,
            news=news,
        )
        analysis = self.analysis_agent.analyze(context)
        retrieval = self.retrieval_agent.retrieve(symbol, news)
        risk = self.risk_agent.evaluate(context)
        decision = self.decision_agent.decide(context, analysis, retrieval, risk)
        order_id = self.execution_agent.execute(decision, execute_orders=execute_orders)

        result = WorkflowResult(
            context=context,
            analysis=analysis,
            retrieval=retrieval,
            risk=risk,
            decision=decision,
            order_id=order_id,
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
