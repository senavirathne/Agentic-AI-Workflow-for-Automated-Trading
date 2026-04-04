"""Workflow assembly and orchestration for live trading and backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from .alpha_vantage import AlphaVantageIndicatorService
from .agents import DecisionCoordinatorAgent, HoldForecastAgent, NewsResearchAgent, RiskReviewAgent, TechnicalAnalysisAgent
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
)
from .decision_engine import DecisionEngine
from .eda import EDAModule
from .execution import ExecutionModule, InMemoryBrokerClient
from .features import FeatureEngineeringModule
from .information_retrieval import (
    AlphaVantageNewsSearchClient,
    CombinedNewsSearchClient,
    InformationRetrievalModule,
    WebSearchNewsClient,
)
from .llm import DeepSeekLLMClient, FallbackLLMClient, LLMRequestError, OpenAILLMClient, TextGenerationClient
from .market_analysis import MarketAnalysisModule
from .models import AlphaVantageIndicatorSnapshot, CollectedMarketData, ForecastSnapshot, PerformanceSnapshot, WorkflowResult
from .reporting import (
    _artifact_location,
    _display_reason,
    _print_collected_windows,
    _print_indicator_context,
    _print_labeled_items,
    _print_reasoning,
    format_reasoning_lines,
)
from .risk_analysis import RiskAnalysisModule
from .storage import (
    AzureBlobRawStore,
    AzureSQLResultStore,
    LocalRawStore,
    RawStore,
    ResultStore,
    SQLiteResultStore,
)
from .utils import _as_utc_timestamp, last_closed_us_market_day_cutoff, next_top_of_hour, sleep_until
from .utils import us_market_day_dates


@dataclass
class TradingWorkflow:
    """Run the end-to-end trading workflow for a symbol or replay window."""

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
    alpha_vantage_service: AlphaVantageIndicatorService | None = None
    alpaca_service: AlpacaService | None = None
    hold_forecast_agent: HoldForecastAgent | None = None
    default_sleep_seconds: float | None = None
    _alpha_vantage_preloaded_symbols: set[str] = field(default_factory=set, init=False, repr=False)

    def run_once(self, symbol: str | None = None, execute_orders: bool = False) -> WorkflowResult:
        target_symbol = (symbol or self.settings.trading.symbol).upper()
        print(f"[Workflow] stating {self.settings.trading.mode.value} workflow for {target_symbol}...")
        self._ensure_backtest_alpha_vantage_data(target_symbol)
        if self.settings.trading.mode == RunMode.BACKTEST:
            collected = self.data_collection.collect_until(
                target_symbol,
                self.settings.trading,
                end_time=self._latest_backtest_available_timestamp(target_symbol),
            )
        else:
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
        previous_forecast = self.result_store.load_latest_forecast(
            symbol,
            as_of=collected.five_minute_bars.index[-1],
        )
        raw_artifacts = {
            "five_minute_bars": self.raw_store.save_bars(symbol, "5min", collected.five_minute_bars),
            "hourly_bars": self.raw_store.save_bars(symbol, "1h", collected.hourly_bars),
        }
        print(
            f"[Workflow] Market data collection finished: "
            f"5min rows={len(collected.five_minute_bars)} | 1h rows={len(collected.hourly_bars)}"
        )
        print(
            f"[Workflow] Raw artifacts saved: "
            f"5min={_artifact_location(raw_artifacts['five_minute_bars'])} | "
            f"1h={_artifact_location(raw_artifacts['hourly_bars'])}"
        )
        eda = self.eda_module.summarize(
            symbol,
            collected.five_minute_bars.tail(self.settings.trading.eda_window_hours * 12),
        )
        print(
            f"[Workflow] EDA finished: latest_close={eda.latest_close:.2f} | "
            f"volatility={eda.candle_volatility:.4f} | anomalies={eda.anomaly_count} | missing={eda.missing_values}"
        )
        feature_frame, alpha_vantage_snapshot, indicator_source = self._build_indicator_context(symbol, collected)
        _print_indicator_context(alpha_vantage_snapshot, indicator_source=indicator_source, feature_frame=feature_frame)
        price_at_timestamp = self._resolve_price_at_analysis_time(symbol, collected)
        market_data_delay_minutes = self._market_data_delay_minutes()
        current_price = self._fetch_live_current_price(symbol, market_data_delay_minutes)
        analysis = self._run_llm_stage(
            "market_analysis",
            symbol,
            lambda: self.market_analysis.analyze(
                symbol,
                feature_frame,
                collected.hourly_bars,
                alpha_vantage_snapshot=alpha_vantage_snapshot,
                price_at_timestamp=price_at_timestamp,
                current_price=current_price,
                market_data_delay_minutes=market_data_delay_minutes,
                indicator_source=indicator_source,
                previous_forecast=previous_forecast,
            ),
        )
        print(
            f"[Workflow] Market analysis finished: trend={analysis.trend} | bullish={analysis.bullish} | "
            f"entry_setup={analysis.entry_setup} | exit_setup={analysis.exit_setup} | "
            f"latest_price={analysis.latest_price:.2f} | confidence={analysis.confidence:.2f}"
        )
        print(f"[Workflow] Technical agent output: {_display_reason(analysis.llm_summary)}")
        news_published_at_lte = self._news_published_at_lte(symbol, collected.five_minute_bars.index[-1])
        retrieval = self._run_llm_stage(
            "information_retrieval",
            symbol,
            lambda: self.information_retrieval.retrieve(
                symbol,
                limit=self.settings.trading.news_limit,
                published_at_lte=news_published_at_lte,
            ),
        )
        raw_artifacts["news"] = self.raw_store.save_news(symbol, retrieval.articles)
        print(
            f"[Workflow] Information retrieval finished: articles={len(retrieval.articles)} | "
            f"critical_news={len(retrieval.critical_news)} | catalysts={len(retrieval.catalysts)}"
        )
        print(f"[Workflow] News artifact saved: {_artifact_location(raw_artifacts['news'])}")
        print(f"[Workflow] News agent output: {_display_reason(retrieval.summary_note)}")
        _print_labeled_items("Critical news", retrieval.critical_news)
        _print_labeled_items("Headline", retrieval.headline_summary)
        risk = self._run_llm_stage(
            "risk_analysis",
            symbol,
            lambda: self.risk_analysis.assess(
                symbol,
                collected.account,
                analysis,
                eda,
                retrieval,
                previous_forecast=previous_forecast,
            ),
        )
        print(
            f"[Workflow] Risk analysis finished: risk_score={risk.risk_score:.2f} | "
            f"can_enter={risk.can_enter} | recommended_qty={risk.recommended_qty} | "
            f"stop_loss={risk.stop_loss_price} | take_profit={risk.take_profit_price}"
        )
        print(f"[Workflow] Risk agent output: {_display_reason(risk.summary_note)}")
        _print_labeled_items("Risk warning", risk.warnings)
        decision = self._run_llm_stage(
            "decision_engine",
            symbol,
            lambda: self.decision_engine.decide(
                collected.account,
                analysis,
                retrieval,
                risk,
                previous_forecast=previous_forecast,
            ),
        )
        print(
            f"[Workflow] Decision engine finished: action={decision.action.value} | "
            f"quantity={decision.quantity} | confidence={decision.confidence:.2f}"
        )
        _print_labeled_items("Decision reason", decision.rationale)
        _set_broker_market_price(
            self.execution_module.broker_client,
            symbol=symbol,
            price=float(analysis.current_price or analysis.latest_price),
            timestamp=analysis.timestamp,
        )
        execution = self.execution_module.execute(
            decision,
            execute_orders=execute_orders,
            account=collected.account,
            risk=risk,
        )
        print(
            f"[Workflow] Execution finished: status={execution.status} | executed={execution.executed} | "
            f"order_id={execution.order_id} | protective_orders={execution.protective_order_ids or '<none>'}"
        )
        account_snapshot = self._refresh_account_state(symbol, fallback=collected.account)
        performance = self._build_performance_snapshot(
            symbol=symbol,
            account=account_snapshot,
            analysis=analysis,
            execution=execution,
        )
        print(
            f"[Workflow] Trading count={performance.trade_count} | position_qty={performance.position_qty} | "
            f"realized_profit={performance.realized_profit:.2f} | "
            f"unrealized_profit={performance.unrealized_profit:.2f} | "
            f"current_profit={performance.current_profit:.2f}"
        )
        hold_forecast = self._maybe_build_hold_forecast(
            decision=decision,
            analysis=analysis,
            retrieval=retrieval,
            risk=risk,
            previous_forecast=previous_forecast,
        )
        if hold_forecast is not None:
            print(f"[Workflow] Hold forecast generated: {_display_reason(hold_forecast.summary)}")
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
            alpha_vantage_indicator_snapshot=alpha_vantage_snapshot,
            account=account_snapshot,
            performance=performance,
            previous_forecast=previous_forecast,
            hold_forecast=hold_forecast,
        )
        self.result_store.save_workflow_run(result, raw_artifacts=raw_artifacts)
        if alpha_vantage_snapshot is not None and self.settings.trading.mode != RunMode.BACKTEST:
            self.result_store.save_alpha_vantage_indicator_snapshot(alpha_vantage_snapshot)
        if hold_forecast is not None:
            self.result_store.save_forecast_snapshot(hold_forecast)
        self.result_store.save_last_processed(symbol, result.analysis.timestamp)
        print(f"[Workflow] Persistence finished: last_processed={pd.Timestamp(result.analysis.timestamp).isoformat()}")
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

    def _build_indicator_context(
        self,
        symbol: str,
        collected: CollectedMarketData,
    ) -> tuple[pd.DataFrame, AlphaVantageIndicatorSnapshot | None, str]:
        if self._uses_backtest_alpha_vantage_indicators():
            end_time = collected.five_minute_bars.index[-1]
            snapshot = self._build_alpha_vantage_indicator_snapshot(symbol, end_time=end_time)
            return (
                self.alpha_vantage_service.build_feature_frame(
                    symbol,
                    collected.five_minute_bars,
                    end_time=end_time,
                ),
                snapshot,
                "Alpha Vantage 5-minute indicators loaded from local storage and chunked into a 1-hour backtest step",
            )
        return (
            self.feature_engineering.build(collected.five_minute_bars),
            None,
            self._manual_indicator_source(),
        )

    def _resolve_price_at_analysis_time(
        self,
        symbol: str,
        collected: CollectedMarketData,
    ) -> float | None:
        if self.settings.trading.mode != RunMode.BACKTEST:
            return None
        checkpoint = collected.five_minute_bars.index[-1]
        price = self.data_collection.fetch_price_at_or_before(symbol, checkpoint)
        print(f"[Workflow] Backtest price lookup finished: timestamp={checkpoint.isoformat()} | price={price:.2f}")
        return price

    def _build_alpha_vantage_indicator_snapshot(
        self,
        symbol: str,
        *,
        end_time: pd.Timestamp | str | None = None,
    ):
        if not self._uses_backtest_alpha_vantage_indicators():
            return None
        if end_time is not None:
            stored_snapshot = self._load_backtest_indicator_snapshot_from_store(symbol, end_time=end_time)
            if stored_snapshot is not None:
                return stored_snapshot
        return self.alpha_vantage_service.build_snapshot(symbol, end_time=end_time)

    def _load_backtest_indicator_snapshot_from_store(
        self,
        symbol: str,
        *,
        end_time: pd.Timestamp | str,
    ) -> AlphaVantageIndicatorSnapshot | None:
        interval = self.settings.alpha_vantage.interval or "5min"
        normalized_end = _as_utc_timestamp(end_time)
        trading_day = normalized_end.date().isoformat()
        snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(
            symbol,
            trading_day=trading_day,
            interval=interval,
        )
        if snapshot is None:
            return None

        rows = [
            dict(row)
            for row in snapshot.rows
            if row.get("time") and _as_utc_timestamp(str(row["time"])) <= normalized_end
        ]
        if not rows:
            return None

        latest_hour_chunk = self.result_store.load_alpha_vantage_hour_chunk(
            symbol,
            as_of=normalized_end,
            interval=interval,
        )
        return replace(
            snapshot,
            latest_timestamp=str(rows[-1]["time"]),
            rows=rows,
            hourly_chunks=[] if latest_hour_chunk is None else [latest_hour_chunk],
            latest_hour_chunk=latest_hour_chunk,
        )

    def _uses_backtest_alpha_vantage_indicators(self) -> bool:
        return self.settings.trading.mode == RunMode.BACKTEST and self.alpha_vantage_service is not None

    def _manual_indicator_source(self) -> str:
        if self.settings.trading.mode == RunMode.LIVE and self.settings.market_data.provider == "alpaca":
            return "manually computed indicators from Alpaca 5-minute bars"
        if self.settings.trading.mode == RunMode.LIVE:
            return "manually computed indicators from live 5-minute bar data"
        return "manually computed indicators from 5-minute bar data"

    def _market_data_delay_minutes(self) -> int:
        if self.settings.trading.mode != RunMode.LIVE:
            return 0
        if self.settings.market_data.provider != "alpaca":
            return 0
        return max(0, self.settings.trading.live_market_data_delay_minutes)

    def _fetch_live_current_price(self, symbol: str, market_data_delay_minutes: int) -> float | None:
        if market_data_delay_minutes <= 0 or self.alpaca_service is None:
            return None
        getter = getattr(self.alpaca_service, "get_current_price", None)
        if not callable(getter):
            return None
        try:
            return float(getter(symbol))
        except Exception as exc:
            print(f"[Workflow] Live current price fetch failed for {symbol}: {exc}")
            return None

    def _refresh_account_state(self, symbol: str, fallback):
        getter = getattr(self.data_collection.account_client, "get_account_state", None)
        if not callable(getter):
            return fallback
        try:
            return getter(symbol)
        except Exception:
            return fallback

    def _build_performance_snapshot(
        self,
        *,
        symbol: str,
        account,
        analysis,
        execution,
    ) -> PerformanceSnapshot:
        market_price = float(analysis.current_price or analysis.latest_price)
        unrealized_profit = 0.0
        if account.position_qty > 0 and account.avg_entry_price is not None:
            unrealized_profit = (market_price - account.avg_entry_price) * account.position_qty
        previous_performance = self.result_store.load_latest_performance(symbol)
        realized_profit = float(account.realized_profit)
        if (
            previous_performance is not None
            and int(account.trade_count) == 0
            and float(account.realized_profit) == 0.0
        ):
            realized_profit = float(previous_performance.realized_profit)
        persisted_trade_count = self.result_store.count_executed_trades(symbol)
        trade_count = max(
            int(account.trade_count),
            persisted_trade_count + (1 if execution.executed else 0),
            0 if previous_performance is None else int(previous_performance.trade_count),
        )
        return PerformanceSnapshot(
            symbol=symbol,
            as_of=pd.Timestamp(analysis.timestamp),
            position_qty=int(account.position_qty),
            trade_count=trade_count,
            market_price=market_price,
            avg_entry_price=account.avg_entry_price,
            realized_profit=round(realized_profit, 2),
            unrealized_profit=round(unrealized_profit, 2),
            current_profit=round(realized_profit + unrealized_profit, 2),
        )

    def _maybe_build_hold_forecast(
        self,
        *,
        decision,
        analysis,
        retrieval,
        risk,
        previous_forecast: ForecastSnapshot | None,
    ) -> ForecastSnapshot | None:
        if decision.action.value != "HOLD" or self.hold_forecast_agent is None:
            return None
        return self._run_llm_stage(
            "hold_forecast",
            analysis.symbol,
            lambda: self.hold_forecast_agent.forecast(
                analysis=analysis,
                retrieval=retrieval,
                risk=risk,
                previous_forecast=previous_forecast,
                valid_for_minutes=self.settings.trading.loop_interval_minutes,
            ),
        )

    def analyze_range(
        self,
        *,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        symbol: str | None = None,
        execute_orders: bool = False,
    ) -> list[WorkflowResult]:
        target_symbol = (symbol or self.settings.trading.symbol).upper()
        self._ensure_backtest_alpha_vantage_data(target_symbol)
        should_execute_orders = self._should_execute_orders(execute_orders)
        checkpoints = self._compute_range_checkpoints(target_symbol, start=start, end=end)
        results: list[WorkflowResult] = []
        for checkpoint in checkpoints:
            collected = self.data_collection.collect_until(
                target_symbol,
                self.settings.trading,
                end_time=checkpoint,
            )
            _print_collected_windows(collected)
            results.append(self._run_collected(target_symbol, collected, should_execute_orders))
        return results

    def run_loop(
        self,
        symbol: str | None = None,
        execute_orders: bool = False,
        max_iterations: int | None = None,
        sleep_seconds: float | None = None,
    ) -> list[WorkflowResult]:
        target_symbol = (symbol or self.settings.trading.symbol).upper()
        self._ensure_backtest_alpha_vantage_data(target_symbol)
        mode = self.settings.trading.mode
        effective_sleep_seconds = self.settings.trading.sleep_seconds
        if self.default_sleep_seconds is not None:
            effective_sleep_seconds = self.default_sleep_seconds
        if sleep_seconds is not None:
            effective_sleep_seconds = max(float(sleep_seconds), 0.0)
        should_execute_orders = self._should_execute_orders(execute_orders)

        print(
            f"\n[Workflow] stating {mode.value} loop for {target_symbol}. "
            f"Max iterations: {max_iterations}, sleep_seconds: {effective_sleep_seconds}"
        )
        results: list[WorkflowResult] = []
        checkpoints = self._compute_backtest_checkpoints(target_symbol) if mode == RunMode.BACKTEST else None
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            checkpoint: pd.Timestamp | None = None
            if mode == RunMode.BACKTEST:
                if not checkpoints:
                    print("[Workflow] Historical replay is exhausted.")
                    break
                checkpoint = checkpoints.pop(0)
                self._advance_replay_client(checkpoint)
                print(f"\n[Workflow] Replaying checkpoint {checkpoint.isoformat()} for {target_symbol}...")
            else:
                checkpoint = self._next_live_checkpoint(target_symbol)
                if checkpoint is None:
                    if effective_sleep_seconds > 0:
                        wait_target = self._next_live_wait_target(target_symbol)
                        if wait_target is not None:
                            print(
                                "[Workflow] No new completed delayed hourly checkpoint is available yet. "
                                f"Waiting until {wait_target.isoformat()} ({effective_sleep_seconds:.1f}s)..."
                            )
                            sleep_until(wait_target, effective_sleep_seconds)
                            continue
                    print("[Workflow] No new completed hourly checkpoint is available yet.")
                    break
                print(
                    f"\n[Workflow] Processing live iteration {iteration + 1} for {target_symbol} "
                    f"at {checkpoint.isoformat()}..."
                )

            if checkpoint is not None and mode == RunMode.LIVE:
                collected = self.data_collection.collect_until(
                    target_symbol,
                    self.settings.trading,
                    end_time=checkpoint,
                )
            else:
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
                if mode == RunMode.LIVE:
                    wait_target = self._next_live_wait_target(target_symbol)
                    if wait_target is None:
                        wait_target = next_top_of_hour()
                    print(
                        "[Workflow] Waiting for the next delayed hourly checkpoint "
                        f"at {wait_target.isoformat()} ({effective_sleep_seconds:.1f}s)..."
                    )
                    sleep_until(wait_target, effective_sleep_seconds)
                else:
                    print(f"[Workflow] Sleeping for {effective_sleep_seconds:.1f}s...")
                    sleep_until(next_top_of_hour(), effective_sleep_seconds)

        return results

    def _should_execute_orders(self, execute_orders: bool) -> bool:
        return execute_orders or self.settings.trading.mode == RunMode.BACKTEST

    def _ensure_backtest_alpha_vantage_data(self, symbol: str) -> None:
        if not self._uses_backtest_alpha_vantage_indicators():
            return

        normalized_symbol = symbol.upper()
        if normalized_symbol in self._alpha_vantage_preloaded_symbols:
            return

        source_bars = self.data_collection.fetch_five_minute_history(normalized_symbol)
        if source_bars.empty:
            raise ValueError(f"No 5-minute bars available for {normalized_symbol}")

        start_time = pd.Timestamp(source_bars.index.min())
        end_time = pd.Timestamp(source_bars.index.max())
        required_trading_days = self._required_backtest_alpha_vantage_trading_days(source_bars.index)
        trading_day_count = len(required_trading_days)
        interval = self.settings.alpha_vantage.interval or "5min"
        database_path = self.settings.paths.database_path.resolve()
        cache_path = (self.settings.paths.data_dir / "alpha_vantage").resolve()
        print(
            f"[Workflow] Alpha Vantage backtest preflight: ensuring local storage coverage for {normalized_symbol} "
            f"across {trading_day_count} trading day(s) | db={database_path} | cache={cache_path}"
        )
        missing_days = [
            trading_day
            for trading_day in required_trading_days
            if self.result_store.load_alpha_vantage_indicator_snapshot(
                normalized_symbol,
                trading_day=trading_day,
                interval=interval,
            )
            is None
        ]
        if missing_days:
            self.alpha_vantage_service.ensure_data_for_window(
                normalized_symbol,
                start_time=start_time,
                end_time=end_time,
                required_trading_days=required_trading_days,
            )
            missing_days = [
                trading_day
                for trading_day in required_trading_days
                if self.result_store.load_alpha_vantage_indicator_snapshot(
                    normalized_symbol,
                    trading_day=trading_day,
                    interval=interval,
                )
                is None
            ]
        if missing_days:
            raise RuntimeError(
                "Alpha Vantage backtest data is still missing from the local store after preload. "
                f"Missing {normalized_symbol} {interval} trading day snapshots: {missing_days}. "
                f"DB path: {database_path}"
            )
        print(
            "[Workflow] Alpha Vantage backtest preload finished: required indicator snapshots and 1-hour chunks "
            f"are available in the local store at {database_path}."
        )
        self._alpha_vantage_preloaded_symbols.add(normalized_symbol)

    def _required_backtest_alpha_vantage_trading_days(self, timestamps: pd.Index) -> list[str]:
        if timestamps.empty:
            return []
        cutoff = last_closed_us_market_day_cutoff(
            pd.Timestamp.now(tz="UTC"),
            available_timestamps=timestamps,
        )
        cutoff_date = cutoff.tz_convert("America/New_York").date()
        return [candidate.isoformat() for candidate in us_market_day_dates(timestamps, max_date=cutoff_date)]

    def _latest_backtest_available_timestamp(self, symbol: str) -> pd.Timestamp | None:
        source_bars = self.data_collection.fetch_five_minute_history(symbol)
        if source_bars.empty:
            return None
        latest_source_timestamp = _as_utc_timestamp(source_bars.index.max())
        if not self._uses_backtest_alpha_vantage_indicators():
            return latest_source_timestamp
        interval = self.settings.alpha_vantage.interval or "5min"
        latest_snapshot = self.result_store.load_alpha_vantage_indicator_snapshot(symbol, interval=interval)
        if latest_snapshot is None:
            return latest_source_timestamp
        latest_indicator_timestamp = _as_utc_timestamp(latest_snapshot.latest_timestamp)
        return min(latest_source_timestamp, latest_indicator_timestamp)

    def _news_published_at_lte(
        self,
        symbol: str,
        checkpoint: pd.Timestamp | str,
    ) -> pd.Timestamp | None:
        if self.settings.trading.mode != RunMode.BACKTEST:
            return None
        source_bars = self.data_collection.fetch_five_minute_history(symbol)
        if source_bars.empty:
            return None
        cutoff = last_closed_us_market_day_cutoff(
            checkpoint,
            available_timestamps=source_bars.index,
        )
        print(f"[Workflow] Backtest news cutoff: published_at<={cutoff.isoformat()}")
        return cutoff

    def _compute_range_checkpoints(
        self,
        symbol: str,
        *,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> list[pd.Timestamp]:
        source_bars = self.data_collection.fetch_five_minute_history(symbol)
        if source_bars.empty:
            return []
        start_timestamp = _as_utc_timestamp(start).floor("1h")
        end_timestamp = _as_utc_timestamp(end).floor("1h")
        latest_available = _as_utc_timestamp(source_bars.index.max()).floor("1h")
        earliest_available = _as_utc_timestamp(source_bars.index.min()).ceil("1h")
        effective_end = min(end_timestamp, latest_available)
        checkpoint = max(start_timestamp + pd.Timedelta(hours=1), earliest_available)
        checkpoints: list[pd.Timestamp] = []
        while checkpoint <= effective_end:
            checkpoints.append(checkpoint)
            checkpoint += pd.Timedelta(hours=1)
        return checkpoints

    def _next_live_checkpoint(self, symbol: str) -> pd.Timestamp | None:
        source_bars = self.data_collection.fetch_five_minute_history(symbol)
        if source_bars.empty:
            return None
        latest_available = _as_utc_timestamp(source_bars.index.max()).floor("1h")
        earliest_available = _as_utc_timestamp(source_bars.index.min()).ceil("1h")
        last_processed = self.result_store.load_last_processed(symbol)
        if last_processed is None:
            return latest_available if latest_available >= earliest_available else None
        next_checkpoint = max(_as_utc_timestamp(last_processed) + pd.Timedelta(hours=1), earliest_available)
        if next_checkpoint > latest_available:
            return None
        return next_checkpoint

    def _next_live_wait_target(self, symbol: str) -> datetime | None:
        source_bars = self.data_collection.fetch_five_minute_history(symbol)
        if source_bars.empty:
            return None
        earliest_available = _as_utc_timestamp(source_bars.index.min()).ceil("1h")
        last_processed = self.result_store.load_last_processed(symbol)
        if last_processed is None:
            latest_available = _as_utc_timestamp(source_bars.index.max()).floor("1h")
            next_checkpoint = max(latest_available + pd.Timedelta(hours=1), earliest_available)
        else:
            next_checkpoint = max(_as_utc_timestamp(last_processed) + pd.Timedelta(hours=1), earliest_available)
        wait_target = next_checkpoint + pd.Timedelta(minutes=self._market_data_delay_minutes())
        return wait_target.to_pydatetime()

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
        latest_checkpoint_cap = self._latest_backtest_available_timestamp(symbol)
        checkpoints = _hourly_checkpoints(source_five_minute_bars)
        return [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint >= earliest_checkpoint
            and (latest_checkpoint_cap is None or checkpoint <= latest_checkpoint_cap)
        ]

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
    """Construct a fully wired workflow from project settings."""
    settings = Settings.from_env(project_root=project_root)
    if mode is not None:
        settings = replace(
            settings,
            trading=replace(settings.trading, mode=_normalize_run_mode(mode)),
        )
    settings.require_llm()
    llm_client = _build_llm_client(settings)
    technical_agent = TechnicalAnalysisAgent(llm_client=llm_client)
    news_agent = NewsResearchAgent(llm_client=llm_client)
    risk_agent = RiskReviewAgent(llm_client=llm_client)
    decision_agent = DecisionCoordinatorAgent(llm_client=llm_client)
    hold_forecast_agent = HoldForecastAgent(llm_client=llm_client)
    raw_store = build_raw_store(settings)
    result_store = build_result_store(settings)
    alpha_vantage_service = None
    if settings.trading.mode == RunMode.BACKTEST or settings.alpha_vantage.enabled:
        alpha_vantage_service = AlphaVantageIndicatorService(
            settings.alpha_vantage,
            cache_dir=settings.paths.data_dir / "alpha_vantage",
            result_store=result_store,
        )
    alpaca_service = AlpacaService(settings.alpaca) if settings.alpaca.enabled else None

    live_news_clients = [WebSearchNewsClient(max_age_days=settings.news.max_age_days)]
    if settings.alpha_vantage.enabled and settings.alpha_vantage.api_key is not None:
        live_news_clients.insert(
            0,
            AlphaVantageNewsSearchClient(
                api_key=settings.alpha_vantage.api_key,
                base_url=settings.alpha_vantage.base_url,
                max_age_days=settings.news.max_age_days,
            ),
        )
    news_client = CombinedNewsSearchClient(live_news_clients)

    if settings.trading.mode == RunMode.LIVE:
        if alpaca_service is None:
            raise RuntimeError(
                "Live trading requires Alpaca market data. "
                "Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY."
            )
        if settings.market_data.provider != "alpaca":
            settings = replace(
                settings,
                market_data=replace(settings.market_data, provider="alpaca"),
            )
        return _build_live_workflow(
            settings=settings,
            technical_agent=technical_agent,
            news_agent=news_agent,
            risk_agent=risk_agent,
            decision_agent=decision_agent,
            hold_forecast_agent=hold_forecast_agent,
            raw_store=raw_store,
            result_store=result_store,
            alpha_vantage_service=alpha_vantage_service,
            alpaca_service=alpaca_service,
            news_client=news_client,
        )
    return _build_backtest_workflow(
        settings=settings,
        technical_agent=technical_agent,
        news_agent=news_agent,
        risk_agent=risk_agent,
        decision_agent=decision_agent,
        hold_forecast_agent=hold_forecast_agent,
        raw_store=raw_store,
        result_store=result_store,
        alpha_vantage_service=alpha_vantage_service,
        alpaca_service=alpaca_service,
        news_client=news_client,
    )


def _build_llm_client(settings: Settings) -> TextGenerationClient:
    """Build the configured primary/secondary LLM stack."""
    primary = DeepSeekLLMClient(settings.llm) if settings.llm.enabled else None
    secondary_config = settings.secondary_llm
    secondary = (
        OpenAILLMClient(secondary_config)
        if secondary_config is not None and secondary_config.enabled
        else None
    )
    if primary is not None and secondary is not None:
        return FallbackLLMClient(primary=primary, secondary=secondary)
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    settings.require_llm()
    raise RuntimeError("At least one LLM client must be configured.")


def _build_live_workflow(
    *,
    settings: Settings,
    technical_agent: TechnicalAnalysisAgent,
    news_agent: NewsResearchAgent,
    risk_agent: RiskReviewAgent,
    decision_agent: DecisionCoordinatorAgent,
    hold_forecast_agent: HoldForecastAgent,
    raw_store: RawStore,
    result_store: ResultStore,
    alpha_vantage_service: AlphaVantageIndicatorService | None,
    alpaca_service: AlpacaService,
    news_client: CombinedNewsSearchClient,
) -> TradingWorkflow:
    market_data_client, account_client, broker_client = _build_live_clients(settings, alpaca_service)
    return TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=market_data_client,
            account_client=account_client,
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(settings.trading, technical_agent=technical_agent),
        information_retrieval=InformationRetrievalModule(
            news_client,
            news_agent=news_agent,
            news_archive=result_store,
            max_article_age_days=settings.news.max_age_days,
        ),
        risk_analysis=RiskAnalysisModule(settings.trading, risk_agent=risk_agent),
        decision_engine=DecisionEngine(decision_agent=decision_agent),
        execution_module=ExecutionModule(broker_client),
        raw_store=raw_store,
        result_store=result_store,
        alpha_vantage_service=alpha_vantage_service,
        alpaca_service=alpaca_service,
        hold_forecast_agent=hold_forecast_agent,
        default_sleep_seconds=settings.trading.sleep_seconds,
    )


def _build_backtest_workflow(
    *,
    settings: Settings,
    technical_agent: TechnicalAnalysisAgent,
    news_agent: NewsResearchAgent,
    risk_agent: RiskReviewAgent,
    decision_agent: DecisionCoordinatorAgent,
    hold_forecast_agent: HoldForecastAgent,
    raw_store: RawStore,
    result_store: ResultStore,
    alpha_vantage_service: AlphaVantageIndicatorService | None,
    alpaca_service: AlpacaService | None,
    news_client: CombinedNewsSearchClient,
) -> TradingWorkflow:
    market_data_client, account_client, broker_client = _build_backtest_clients(settings, alpaca_service)
    return TradingWorkflow(
        settings=settings,
        data_collection=DataCollectionModule(
            market_data_client=market_data_client,
            account_client=account_client,
        ),
        eda_module=EDAModule(),
        feature_engineering=FeatureEngineeringModule(settings.trading),
        market_analysis=MarketAnalysisModule(settings.trading, technical_agent=technical_agent),
        information_retrieval=InformationRetrievalModule(
            news_client,
            news_agent=news_agent,
            news_archive=result_store,
            max_article_age_days=settings.news.max_age_days,
        ),
        risk_analysis=RiskAnalysisModule(settings.trading, risk_agent=risk_agent),
        decision_engine=DecisionEngine(decision_agent=decision_agent),
        execution_module=ExecutionModule(broker_client),
        raw_store=raw_store,
        result_store=result_store,
        alpha_vantage_service=alpha_vantage_service,
        alpaca_service=alpaca_service,
        hold_forecast_agent=hold_forecast_agent,
        default_sleep_seconds=settings.trading.sleep_seconds,
    )


def build_raw_store(settings: Settings) -> RawStore:
    """Construct the configured raw artifact store."""
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
    """Construct the configured result store."""
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
    raise RuntimeError(
        "Live trading requires Alpaca market data. "
        "Set LIVE_MARKET_DATA_PROVIDER=alpaca and configure Alpaca credentials."
    )


def _build_backtest_clients(
    settings: Settings,
    alpaca_service: AlpacaService | None,
):
    if alpaca_service is None:
        raise RuntimeError(
            "Backtest OHLCV replay bars require Alpaca historical data. "
            "Configure ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY, or pass explicit bars to BacktestingEngine.run()."
        )
    history_lookback_days = _backtest_history_lookback_days(settings)
    symbol = settings.trading.symbol
    five_min_history = alpaca_service.fetch_five_minute_bars(symbol, history_lookback_days)
    hourly_history = alpaca_service.fetch_hourly_bars(symbol, history_lookback_days)

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
