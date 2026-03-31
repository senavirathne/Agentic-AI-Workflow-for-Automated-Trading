from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import RunMode
from .models import Action, BacktestSummary, BacktestTrade, WorkflowResult
from .workflow import TradingWorkflow


@dataclass
class BacktestingEngine:
    """Thin wrapper over the unified workflow loop."""

    workflow: TradingWorkflow

    def run(
        self,
        symbol: str | None = None,
        *,
        max_iterations: int | None = None,
        sleep_seconds: float | None = None,
    ) -> BacktestSummary:
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


def summarize_backtest_results(
    results: list[WorkflowResult],
    *,
    starting_cash: float,
    symbol: str | None = None,
) -> BacktestSummary:
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
