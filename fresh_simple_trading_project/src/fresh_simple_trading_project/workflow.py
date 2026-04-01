from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import pandas as pd

from .agents import DecisionCoordinatorAgent, NewsResearchAgent, RiskReviewAgent, TechnicalAnalysisAgent
from .alpaca_integration import (
    AlpacaAccountClient,
    AlpacaBrokerClient,
    AlpacaMarketDataClient,
    AlpacaService,
)
from .config import RunMode, Settings
from .data_collection import (
    DataCollectionModule,
    HistoricalReplayDataClient,
    SimulatedAccountClient,
    SyntheticMarketDataClient,
)
from .decision_engine import DecisionEngine
from .eda import EDAModule
from .execution import ExecutionModule, InMemoryBrokerClient
from .features import FeatureEngineeringModule
from .information_retrieval import (
    AlpacaNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    WebSearchNewsClient,
)
from .llm import DeepSeekLLMClient, LLMRequestError
from .market_analysis import MarketAnalysisModule
from .models import CollectedMarketData, WorkflowResult
from .risk_analysis import RiskAnalysisModule
from .storage import (
    AzureBlobRawStore,
    AzureSQLResultStore,
    LocalRawStore,
    RawStore,
    ResultStore,
    SQLiteResultStore,
)


@dataclass
class TradingWorkflow:
    settings: Settings
    data_collection: DataCollectionModule
    eda_module: EDAModule
    feature_engineering: FeatureEngineeringModule
    market_analysis: MarketAnalysisModule
    information_retrieval: InformationRetrievalModule
    risk_analysis: RiskAnalysisModule
    decision_engine: DecisionEngine
    execution_module: ExecutionModule
    raw_store: RawStore
    result_store: ResultStore
    default_sleep_seconds: float | None = None

    def run_once(self, symbol: str | None = None, execute_orders: bool = False) -> WorkflowResult:
        target_symbol = (symbol or self.settings.trading.symbol).upper()
        collected = self.data_collection.collect(target_symbol, self.settings.trading)
        return self._run_collected(
            target_symbol,
            collected,
            self._should_execute_orders(execute_orders),
        )

    def _run_collected(
        self,
        symbol: str,
        collected: CollectedMarketData,
        execute_orders: bool,
    ) -> WorkflowResult:
        raw_artifacts = {
            "five_minute_bars": self.raw_store.save_bars(symbol, "5min", collected.five_minute_bars),
            "hourly_bars": self.raw_store.save_bars(symbol, "1h", collected.hourly_bars),
        }
        eda = self.eda_module.summarize(
            symbol,
            collected.five_minute_bars.tail(self.settings.trading.eda_window_hours * 12),
        )
        feature_frame = self.feature_engineering.build(collected.five_minute_bars)
        analysis = self._run_llm_stage(
            "market_analysis",
            symbol,
            lambda: self.market_analysis.analyze(symbol, feature_frame, collected.hourly_bars),
        )
        retrieval = self._run_llm_stage(
            "information_retrieval",
            symbol,
            lambda: self.information_retrieval.retrieve(symbol, limit=self.settings.trading.news_limit),
        )
        raw_artifacts["news"] = self.raw_store.save_news(symbol, retrieval.articles)
        risk = self._run_llm_stage(
            "risk_analysis",
            symbol,
            lambda: self.risk_analysis.assess(symbol, collected.account, analysis, eda, retrieval),
        )
        decision = self._run_llm_stage(
            "decision_engine",
            symbol,
            lambda: self.decision_engine.decide(collected.account, analysis, retrieval, risk),
        )
        _set_broker_market_price(
            self.execution_module.broker_client,
            symbol=symbol,
            price=analysis.latest_price,
            timestamp=analysis.timestamp,
        )
        execution = self.execution_module.execute(decision, execute_orders=execute_orders)
        result = WorkflowResult(
            symbol=symbol,
            five_minute_bars=collected.five_minute_bars,
            hourly_bars=collected.hourly_bars,
            eda=eda,
            analysis=analysis,
            retrieval=retrieval,
            risk=risk,
            decision=decision,
            execution=execution,
        )
        self.result_store.save_workflow_run(result, raw_artifacts=raw_artifacts)
        self.result_store.save_last_processed(symbol, result.analysis.timestamp)
        return result

    def _run_llm_stage(self, stage: str, symbol: str, operation):
        try:
            return operation()
        except LLMRequestError as exc:
            raise LLMRequestError(
                operation=exc.operation,
                model=exc.model,
                base_url=exc.base_url,
                detail=f"workflow stage '{stage}' for symbol '{symbol}': {exc.detail}",
            ) from exc

    def run_loop(
        self,
        symbol: str | None = None,
        execute_orders: bool = False,
        max_iterations: int | None = None,
        sleep_seconds: float | None = None,
    ) -> list[WorkflowResult]:
        target_symbol = (symbol or self.settings.trading.symbol).upper()
        mode = self.settings.trading.mode
        effective_sleep_seconds = self.settings.trading.sleep_seconds
        if self.default_sleep_seconds is not None:
            effective_sleep_seconds = self.default_sleep_seconds
        if sleep_seconds is not None:
            effective_sleep_seconds = max(float(sleep_seconds), 0.0)
        should_execute_orders = self._should_execute_orders(execute_orders)

        print(
            f"\n[Workflow] Starting {mode.value} loop for {target_symbol}. "
            f"Max iterations: {max_iterations}, sleep_seconds: {effective_sleep_seconds}"
        )
        results: list[WorkflowResult] = []
        checkpoints = self._compute_backtest_checkpoints(target_symbol) if mode == RunMode.BACKTEST else None
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            if mode == RunMode.BACKTEST:
                if not checkpoints:
                    print("[Workflow] Historical replay is exhausted.")
                    break
                checkpoint = checkpoints.pop(0)
                self._advance_replay_client(checkpoint)
                print(f"\n[Workflow] Replaying checkpoint {checkpoint.isoformat()} for {target_symbol}...")
            else:
                print(f"\n[Workflow] Processing live iteration {iteration + 1} for {target_symbol}...")

            collected = self.data_collection.collect(target_symbol, self.settings.trading)
            _print_collected_windows(collected)
            result = self._run_collected(target_symbol, collected, should_execute_orders)
            print(
                f"[Workflow] Decision: {result.decision.action.value} "
                f"(Qty: {result.decision.quantity}) | Executed: {result.execution.status}"
            )
            _print_reasoning(result)
            results.append(result)
            iteration += 1

            if not (max_iterations is None or iteration < max_iterations):
                break
            if mode == RunMode.BACKTEST and checkpoints is not None and not checkpoints:
                break
            if effective_sleep_seconds > 0:
                print(f"[Workflow] Sleeping for {effective_sleep_seconds:.1f}s...")
                time.sleep(effective_sleep_seconds)

        return results

    def _should_execute_orders(self, execute_orders: bool) -> bool:
        return execute_orders or self.settings.trading.mode == RunMode.BACKTEST

    def _compute_backtest_checkpoints(self, symbol: str) -> list[pd.Timestamp]:
        replay_client = self._get_replay_market_data_client()
        if replay_client is None:
            raise RuntimeError("Backtest mode requires HistoricalReplayDataClient.")
        replay_client.reset()
        source_five_minute_bars = self.data_collection.fetch_five_minute_history(symbol)
        source_hourly_bars = self.data_collection.fetch_hourly_history(symbol)
        if source_five_minute_bars.empty or source_hourly_bars.empty:
            return []

        earliest_indicator_checkpoint = _as_utc_timestamp(source_five_minute_bars.index[0]) + pd.Timedelta(
            hours=self.settings.trading.indicator_lookback_hours
        )
        earliest_sr_checkpoint = _as_utc_timestamp(source_hourly_bars.index[0]) + pd.Timedelta(
            days=self.settings.trading.sr_lookback_days
        )
        earliest_checkpoint = max(earliest_indicator_checkpoint, earliest_sr_checkpoint)
        checkpoints = _hourly_checkpoints(source_five_minute_bars)
        return [checkpoint for checkpoint in checkpoints if checkpoint >= earliest_checkpoint]

    def _advance_replay_client(self, checkpoint: pd.Timestamp) -> None:
        replay_client = self._get_replay_market_data_client()
        if replay_client is None:
            raise RuntimeError("Backtest mode requires HistoricalReplayDataClient.")
        replay_client.advance_to(checkpoint)

    def _get_replay_market_data_client(self) -> HistoricalReplayDataClient | None:
        client = self.data_collection.market_data_client
        if isinstance(client, HistoricalReplayDataClient):
            return client
        return None


def build_workflow(
    project_root: Path | None = None,
    mode: RunMode | str | None = None,
) -> TradingWorkflow:
    settings = Settings.from_env(project_root=project_root)
    if mode is not None:
        settings = replace(
            settings,
            trading=replace(settings.trading, mode=_normalize_run_mode(mode)),
        )
    settings.llm.require()
    llm_client = DeepSeekLLMClient(settings.llm)
    technical_agent = TechnicalAnalysisAgent(llm_client=llm_client)
    news_agent = NewsResearchAgent(llm_client=llm_client)
    risk_agent = RiskReviewAgent(llm_client=llm_client)
    decision_agent = DecisionCoordinatorAgent(llm_client=llm_client)
    alpaca_service = AlpacaService(settings.alpaca) if settings.alpaca.enabled else None

    if settings.trading.mode == RunMode.LIVE:
        market_data_client, account_client, broker_client = _build_live_clients(settings, alpaca_service)
    else:
        market_data_client, account_client, broker_client = _build_backtest_clients(settings, alpaca_service)

    live_news_clients = [WebSearchNewsClient(max_age_days=settings.news.max_age_days)]
    if alpaca_service is not None:
        live_news_clients.insert(
            0,
            AlpacaNewsSearchClient(alpaca_service, max_age_days=settings.news.max_age_days),
        )
    news_client = CombinedNewsSearchClient(live_news_clients)

    return TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=market_data_client,
            account_client=account_client,
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(
            settings.trading,
            llm_client=llm_client,
            technical_agent=technical_agent,
        ),
        information_retrieval=InformationRetrievalModule(
            news_client,
            llm_client=llm_client,
            news_agent=news_agent,
            max_article_age_days=settings.news.max_age_days,
        ),
        risk_analysis=RiskAnalysisModule(
            settings.trading,
            llm_client=llm_client,
            risk_agent=risk_agent,
        ),
        decision_engine=DecisionEngine(
            llm_client=llm_client,
            decision_agent=decision_agent,
        ),
        execution_module=ExecutionModule(broker_client),
        raw_store=build_raw_store(settings),
        result_store=build_result_store(settings),
        default_sleep_seconds=settings.trading.sleep_seconds,
    )


def build_raw_store(settings: Settings) -> RawStore:
    provider = settings.raw_store.provider
    if provider == "local":
        return LocalRawStore(settings.paths.raw_dir)
    if provider == "azure_blob":
        return AzureBlobRawStore(
            container_name=settings.azure.blob_container_raw,
            account_url=settings.azure.storage_account_url,
            connection_string=settings.azure.storage_connection_string,
            blob_prefix=settings.azure.blob_prefix,
        )
    raise RuntimeError(f"Unsupported raw store provider: {provider}")


def build_result_store(settings: Settings) -> ResultStore:
    provider = settings.result_store.provider
    if provider == "sqlite":
        return SQLiteResultStore(settings.paths.database_path)
    if provider == "azure_sql":
        if not settings.result_store.database_url:
            raise RuntimeError("RESULT_STORE_PROVIDER=azure_sql requires DATABASE_URL.")
        return AzureSQLResultStore(settings.result_store.database_url)
    raise RuntimeError(f"Unsupported result store provider: {provider}")


def _build_live_clients(
    settings: Settings,
    alpaca_service: AlpacaService | None,
):
    if settings.market_data.provider == "alpaca":
        if alpaca_service is None:
            raise RuntimeError(
                "Alpaca market data was requested, but ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY are missing."
            )
        return (
            AlpacaMarketDataClient(
                alpaca_service,
                five_minute_lookback_days=_lookback_days_for_indicator_window(
                    settings.trading.indicator_lookback_hours
                ),
                hourly_lookback_days=max(settings.trading.sr_lookback_days, 7),
            ),
            AlpacaAccountClient(alpaca_service),
            AlpacaBrokerClient(alpaca_service),
        )

    simulated_account = SimulatedAccountClient(cash=settings.trading.starting_cash)
    return (
        SyntheticMarketDataClient(),
        simulated_account,
        InMemoryBrokerClient(account_client=simulated_account),
    )


def _build_backtest_clients(
    settings: Settings,
    alpaca_service: AlpacaService | None,
):
    history_lookback_days = _backtest_history_lookback_days(settings)
    symbol = settings.trading.symbol
    if settings.market_data.provider == "alpaca":
        if alpaca_service is None:
            raise RuntimeError(
                "Alpaca market data was requested, but ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET_KEY are missing."
            )
        five_min_history = alpaca_service.fetch_five_minute_bars(symbol, history_lookback_days)
        hourly_history = alpaca_service.fetch_hourly_bars(symbol, history_lookback_days)
    else:
        synthetic_client = SyntheticMarketDataClient()
        five_min_history = synthetic_client.fetch_five_minute_bars(symbol)
        hourly_history = synthetic_client.fetch_hourly_bars(symbol)

    simulated_account = SimulatedAccountClient(cash=settings.trading.starting_cash)
    return (
        HistoricalReplayDataClient(
            five_min_history=five_min_history,
            hourly_history=hourly_history,
        ),
        simulated_account,
        InMemoryBrokerClient(account_client=simulated_account),
    )


def _normalize_run_mode(mode: RunMode | str) -> RunMode:
    if isinstance(mode, RunMode):
        return mode
    return RunMode(mode.strip().lower())


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _backtest_history_lookback_days(settings: Settings) -> int:
    indicator_days = _lookback_days_for_indicator_window(settings.trading.indicator_lookback_hours)
    return max(
        settings.trading.backtest_history_days,
        settings.trading.sr_lookback_days + 7,
        indicator_days + 1,
    )


def _hourly_checkpoints(source_bars: pd.DataFrame) -> list[pd.Timestamp]:
    if source_bars.empty:
        return []
    checkpoints: list[pd.Timestamp] = []
    for _, bucket in source_bars.groupby(pd.Grouper(freq="1h")):
        if bucket.empty:
            continue
        checkpoints.append(_as_utc_timestamp(bucket.index[-1]))
    return checkpoints


def _lookback_days_for_indicator_window(lookback_hours: int) -> int:
    return max(2, (lookback_hours // 24) + 1)


def _set_broker_market_price(broker_client, *, symbol: str, price: float, timestamp: pd.Timestamp) -> None:
    setter = getattr(broker_client, "set_market_price", None)
    if callable(setter):
        setter(symbol, price, timestamp)


def _print_collected_windows(collected: CollectedMarketData) -> None:
    print(f"[Workflow] Indicator window (5min): {_frame_window(collected.five_minute_bars)}")
    print(f"[Workflow] Support/Resistance window (1h): {_frame_window(collected.hourly_bars)}")


def _print_reasoning(result: WorkflowResult) -> None:
    for line in format_reasoning_lines(result):
        print(line)


def format_reasoning_lines(result: WorkflowResult, prefix: str = "  ") -> list[str]:
    lines = [
        f"{prefix}Data Window: {_data_window(result)}",
        f"{prefix}Technical Agent: {_display_reason(result.analysis.llm_summary)}",
        f"{prefix}News Agent: {_display_reason(result.retrieval.summary_note)}",
        f"{prefix}Risk Agent: {_display_reason(result.risk.summary_note)}",
    ]
    for idx, headline in enumerate(result.retrieval.headline_summary, 1):
        lines.append(f"{prefix}Headline {idx}: {headline}")
    for idx, warning in enumerate(result.risk.warnings, 1):
        lines.append(f"{prefix}Risk Warning {idx}: {warning}")
    if result.decision.rationale:
        for idx, rationale in enumerate(result.decision.rationale, 1):
            lines.append(f"{prefix}Decision Reason {idx}: {rationale}")
    else:
        lines.append(f"{prefix}Decision Reason: <none>")
    return lines


def _display_reason(value: str | None) -> str:
    if value is None:
        return "<no output returned>"
    candidate = value.strip()
    return candidate or "<no output returned>"


def _data_window(result: WorkflowResult) -> str:
    return (
        f"5min {_frame_window(result.five_minute_bars)} | "
        f"1h {_frame_window(result.hourly_bars)}"
    )


def _frame_window(frame: pd.DataFrame) -> str:
    try:
        start = frame.index.min()
        end = frame.index.max()
        if start is None or end is None:
            return "<unknown>"
        return f"{pd.Timestamp(start).isoformat()} -> {pd.Timestamp(end).isoformat()}"
    except Exception:
        return "<unknown>"
