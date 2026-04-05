"""Backtesting helpers for replaying the unified trading workflow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import (
    AlpacaConfig,
    AlphaVantageConfig,
    AzureConfig,
    LLMConfig,
    MarketDataConfig,
    NewsConfig,
    Paths,
    RawStoreConfig,
    ResultStoreConfig,
    RunMode,
    Settings,
    TradingConfig,
)
from .data_collection import DataCollectionModule, HistoricalReplayDataClient, SimulatedAccountClient
from .decision_engine import DecisionEngine
from .eda import EDAModule
from .execution import ExecutionModule, InMemoryBrokerClient
from .features import FeatureEngineeringModule
from .market_analysis import MarketAnalysisModule
from .models import Action, BacktestSummary, BacktestTrade, RetrievalResult, WorkflowResult
from .risk_analysis import RiskAnalysisModule
from .storage import InMemoryResultStore
from .workflow import TradingWorkflow


class BacktestingEngine:
    """Thin wrapper over the unified workflow loop."""

    workflow: TradingWorkflow | None

    def __init__(
        self,
        workflow: TradingWorkflow | None = None,
        *,
        config: TradingConfig | None = None,
        feature_engineering: FeatureEngineeringModule | None = None,
        eda_module: EDAModule | None = None,
        market_analysis: MarketAnalysisModule | None = None,
        risk_analysis: RiskAnalysisModule | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        self.workflow = workflow
        self._legacy_config = config
        self._legacy_feature_engineering = feature_engineering
        self._legacy_eda_module = eda_module
        self._legacy_market_analysis = market_analysis
        self._legacy_risk_analysis = risk_analysis
        self._legacy_decision_engine = decision_engine

    def run(
        self,
        symbol: str | None = None,
        five_minute_bars: pd.DataFrame | None = None,
        hourly_bars: pd.DataFrame | None = None,
        *,
        max_iterations: int | None = None,
        sleep_seconds: float | None = None,
    ) -> BacktestSummary:
        """Run the configured workflow or legacy modules as a backtest."""
        if five_minute_bars is not None:
            return self._run_legacy(
                symbol=symbol,
                five_minute_bars=five_minute_bars,
                hourly_bars=hourly_bars,
                max_iterations=max_iterations,
                sleep_seconds=sleep_seconds,
            )
        if self.workflow is None:
            raise RuntimeError("BacktestingEngine requires either a workflow or explicit historical bars.")
        if self.workflow.settings.trading.mode != RunMode.BACKTEST:
            raise RuntimeError("BacktestingEngine requires a workflow configured with RunMode.BACKTEST.")
        results = self.workflow.run_loop(
            symbol=symbol,
            execute_orders=True,
            max_iterations=max_iterations,
            sleep_seconds=sleep_seconds,
        )
        return summarize_backtest_results(
            results,
            starting_cash=self.workflow.settings.trading.starting_cash,
            symbol=(symbol or self.workflow.settings.trading.symbol).upper(),
        )

    def _run_legacy(
        self,
        *,
        symbol: str | None,
        five_minute_bars: pd.DataFrame,
        hourly_bars: pd.DataFrame | None,
        max_iterations: int | None,
        sleep_seconds: float | None,
    ) -> BacktestSummary:
        config = self._legacy_config
        if (
            config is None
            or self._legacy_feature_engineering is None
            or self._legacy_eda_module is None
            or self._legacy_market_analysis is None
            or self._legacy_risk_analysis is None
            or self._legacy_decision_engine is None
        ):
            raise RuntimeError("Legacy BacktestingEngine usage requires config and workflow modules.")

        target_symbol = (symbol or config.symbol).upper()
        isolated_root = Path("/tmp/fresh_simple_trading_project_backtesting")
        isolated_root.mkdir(parents=True, exist_ok=True)
        paths = Paths(
            project_root=isolated_root,
            data_dir=isolated_root / "data",
            raw_dir=isolated_root / "data" / "raw",
            reports_dir=isolated_root / "reports",
            database_path=isolated_root / "data" / "workflow.sqlite",
        )
        paths.create_directories()
        settings = Settings(
            trading=replace(config, mode=RunMode.BACKTEST),
            market_data=MarketDataConfig(provider="alpha_vantage"),
            raw_store=RawStoreConfig(provider="local"),
            result_store=ResultStoreConfig(provider="sqlite", database_url=f"sqlite:///{paths.database_path.resolve()}"),
            alpha_vantage=AlphaVantageConfig(),
            azure=AzureConfig(),
            alpaca=AlpacaConfig(),
            news=NewsConfig(),
            llm=LLMConfig(),
            secondary_llm=None,
            paths=paths,
        )
        simulated_account = SimulatedAccountClient(cash=config.starting_cash)
        replay_client = HistoricalReplayDataClient(
            five_min_history=five_minute_bars,
            hourly_history=hourly_bars if hourly_bars is not None else pd.DataFrame(),
        )
        workflow = TradingWorkflow(
            settings=settings,
            data_collection=DataCollectionModule(
                market_data_client=replay_client,
                account_client=simulated_account,
            ),
            eda_module=self._legacy_eda_module,
            feature_engineering=self._legacy_feature_engineering,
            market_analysis=self._legacy_market_analysis,
            information_retrieval=_BacktestRetrievalModule(),
            risk_analysis=self._legacy_risk_analysis,
            decision_engine=self._legacy_decision_engine,
            execution_module=ExecutionModule(InMemoryBrokerClient(account_client=simulated_account)),
            raw_store=_NullRawStore(),
            result_store=InMemoryResultStore(),
            default_sleep_seconds=0.0,
        )
        results = workflow.run_loop(
            symbol=target_symbol,
            execute_orders=True,
            max_iterations=max_iterations,
            sleep_seconds=sleep_seconds,
        )
        return summarize_backtest_results(
            results,
            starting_cash=config.starting_cash,
            symbol=target_symbol,
        )


class _NullRawStore:
    """Minimal raw-store stub used by the legacy backtesting path."""

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> Path:
        """Return a placeholder bars path for legacy backtests."""
        return Path("/tmp")

    def save_news(self, symbol: str, articles) -> Path:
        """Return a placeholder news path for legacy backtests."""
        return Path("/tmp")


class _BacktestRetrievalModule:
    """Static retrieval stub used by the legacy backtesting path."""

    def retrieve(
        self,
        symbol: str,
        limit: int = 8,
        *,
        input_size_chars: int | None = None,
        published_at_lte=None,
    ) -> RetrievalResult:
        """Return an empty retrieval payload for legacy backtests."""
        return RetrievalResult(
            symbol=symbol,
            articles=[],
            headline_summary=[],
            sentiment_score=0.0,
        )


def summarize_backtest_results(
    results: list[WorkflowResult],
    *,
    starting_cash: float,
    symbol: str | None = None,
) -> BacktestSummary:
    """Aggregate workflow iteration results into a backtest summary."""
    target_symbol = (symbol or (results[0].symbol if results else "")).upper()
    if not results:
        return BacktestSummary(
            symbol=target_symbol,
            initial_cash=starting_cash,
            ending_cash=starting_cash,
            total_return_pct=0.0,
            benchmark_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            trade_count=0,
            win_rate=0.0,
            signal_accuracy=0.0,
            trades=[],
            equity_curve=pd.DataFrame(columns=["equity"]),
            net_profit=0.0,
        )

    cash = float(starting_cash)
    position_qty = 0
    entry_price: float | None = None
    entry_time: pd.Timestamp | None = None
    trades: list[BacktestTrade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []
    first_price = float(results[0].analysis.latest_price)

    for result in results:
        price = float(result.analysis.latest_price)
        timestamp = pd.Timestamp(result.analysis.timestamp)
        if result.execution.executed and result.decision.quantity > 0:
            quantity = int(result.decision.quantity)
            if result.decision.action == Action.BUY and position_qty == 0:
                cash -= quantity * price
                position_qty = quantity
                entry_price = price
                entry_time = timestamp
            elif (
                result.decision.action == Action.SELL
                and position_qty > 0
                and entry_price is not None
                and entry_time is not None
            ):
                sold_qty = min(quantity, position_qty)
                cash += sold_qty * price
                pnl = (price - entry_price) * sold_qty
                trades.append(
                    BacktestTrade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        qty=sold_qty,
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=round(pnl, 2),
                        pnl_pct=round(((price / entry_price) - 1) * 100, 2),
                        reason="signal_exit",
                    )
                )
                position_qty -= sold_qty
                if position_qty == 0:
                    entry_price = None
                    entry_time = None

        equity = cash + (position_qty * price)
        equity_points.append((timestamp, round(equity, 2)))

    final_price = float(results[-1].analysis.latest_price)
    final_time = pd.Timestamp(results[-1].analysis.timestamp)
    if position_qty > 0 and entry_price is not None and entry_time is not None:
        cash += position_qty * final_price
        pnl = (final_price - entry_price) * position_qty
        trades.append(
            BacktestTrade(
                entry_time=entry_time,
                exit_time=final_time,
                qty=position_qty,
                entry_price=entry_price,
                exit_price=final_price,
                pnl=round(pnl, 2),
                pnl_pct=round(((final_price / entry_price) - 1) * 100, 2),
                reason="forced_close",
            )
        )
        equity_points.append((final_time, round(cash, 2)))

    equity_curve = pd.DataFrame(equity_points, columns=["timestamp", "equity"]).drop_duplicates("timestamp")
    if equity_curve.empty:
        equity_curve = pd.DataFrame({"timestamp": [final_time], "equity": [cash]})
    equity_curve = equity_curve.set_index("timestamp")
    ending_cash = float(equity_curve["equity"].iloc[-1])
    benchmark_return_pct = round(((final_price / first_price) - 1) * 100, 2) if first_price > 0 else 0.0
    total_return_pct = round(((ending_cash / starting_cash) - 1) * 100, 2) if starting_cash > 0 else 0.0
    max_drawdown_pct = _max_drawdown_pct(equity_curve["equity"])
    sharpe_ratio = _sharpe_ratio(equity_curve["equity"])
    trade_count = len(trades)
    win_rate = round(sum(trade.pnl > 0 for trade in trades) / trade_count, 2) if trade_count else 0.0
    net_profit = round(ending_cash - starting_cash, 2)

    return BacktestSummary(
        symbol=target_symbol,
        initial_cash=starting_cash,
        ending_cash=round(ending_cash, 2),
        total_return_pct=total_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        trade_count=trade_count,
        win_rate=win_rate,
        signal_accuracy=win_rate,
        trades=trades,
        equity_curve=equity_curve,
        net_profit=net_profit,
    )


def _max_drawdown_pct(equity: pd.Series) -> float:
    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks.replace(0, pd.NA)
    minimum = drawdown.min()
    if pd.isna(minimum):
        return 0.0
    return round(abs(float(minimum)) * 100, 2)


def _sharpe_ratio(equity: pd.Series) -> float:
    returns = equity.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    bars_per_day = 24
    return round(float((returns.mean() / returns.std()) * (bars_per_day ** 0.5)), 2)
