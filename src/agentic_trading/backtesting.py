from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TradingConfig
from .indicators import generate_signal_frame
from .models import BacktestSummary, BacktestTrade
from .utils import dump_json


@dataclass
class StrategyBacktester:
    config: TradingConfig

    def prepare_signal_frame(self, main_bars: pd.DataFrame, trend_bars: pd.DataFrame) -> pd.DataFrame:
        return generate_signal_frame(main_bars, trend_bars, self.config)

    def backtest(self, symbol: str, main_bars: pd.DataFrame, trend_bars: pd.DataFrame) -> BacktestSummary:
        signal_frame = self.prepare_signal_frame(main_bars, trend_bars)
        cash = self.config.initial_capital
        qty = 0
        entry_price: float | None = None
        entry_time: pd.Timestamp | None = None
        equity_points: list[dict[str, float | pd.Timestamp]] = []
        trades: list[BacktestTrade] = []

        for timestamp, row in signal_frame.iterrows():
            price = float(row["close"])
            if qty == 0 and bool(row["buy_signal"]):
                notional = cash * self.config.buy_power_limit
                candidate_qty = int(notional / price)
                if candidate_qty > 0:
                    qty = candidate_qty
                    cash -= qty * price
                    entry_price = price
                    entry_time = timestamp
            elif qty > 0 and bool(row["sell_signal"]):
                assert entry_price is not None and entry_time is not None
                cash += qty * price
                pnl = (price - entry_price) * qty
                trades.append(
                    BacktestTrade(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=timestamp,
                        qty=qty,
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=pnl,
                        pnl_pct=((price / entry_price) - 1) * 100,
                        reason="sell_signal",
                    )
                )
                qty = 0
                entry_price = None
                entry_time = None

            equity_points.append({"timestamp": timestamp, "equity": cash + (qty * price)})

        if qty > 0 and entry_price is not None and entry_time is not None:
            final_price = float(signal_frame.iloc[-1]["close"])
            final_time = signal_frame.index[-1]
            cash += qty * final_price
            trades.append(
                BacktestTrade(
                    symbol=symbol,
                    entry_time=entry_time,
                    exit_time=final_time,
                    qty=qty,
                    entry_price=entry_price,
                    exit_price=final_price,
                    pnl=(final_price - entry_price) * qty,
                    pnl_pct=((final_price / entry_price) - 1) * 100,
                    reason="end_of_test",
                )
            )
            qty = 0

        equity_curve = pd.DataFrame(equity_points).set_index("timestamp")
        ending_capital = float(cash)
        benchmark_return_pct = (
            (float(signal_frame["close"].iloc[-1]) / float(signal_frame["close"].iloc[0]) - 1) * 100
        )
        win_rate = (
            sum(trade.pnl > 0 for trade in trades) / len(trades)
            if trades
            else 0.0
        )

        return BacktestSummary(
            symbol=symbol,
            initial_capital=self.config.initial_capital,
            ending_capital=ending_capital,
            total_return_pct=((ending_capital / self.config.initial_capital) - 1) * 100,
            benchmark_return_pct=benchmark_return_pct,
            max_drawdown_pct=_max_drawdown_pct(equity_curve["equity"]),
            sharpe_ratio=_sharpe_ratio(equity_curve["equity"].pct_change().dropna()),
            signal_accuracy=_signal_accuracy(signal_frame, horizon=self.config.signal_horizon_bars),
            trade_count=len(trades),
            win_rate=win_rate,
            trades=trades,
            equity_curve=equity_curve,
        )

    def write_outputs(self, summary: BacktestSummary, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "symbol": summary.symbol,
            "initial_capital": summary.initial_capital,
            "ending_capital": summary.ending_capital,
            "total_return_pct": summary.total_return_pct,
            "benchmark_return_pct": summary.benchmark_return_pct,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "sharpe_ratio": summary.sharpe_ratio,
            "signal_accuracy": summary.signal_accuracy,
            "trade_count": summary.trade_count,
            "win_rate": summary.win_rate,
        }
        dump_json(output_dir / "backtest_summary.json", summary_payload)
        summary.equity_curve.to_csv(output_dir / "equity_curve.csv")
        pd.DataFrame([trade.__dict__ for trade in summary.trades]).to_csv(
            output_dir / "trades.csv", index=False
        )

        figure, axis = plt.subplots(figsize=(12, 5))
        summary.equity_curve["equity"].plot(ax=axis, color="#005f73", linewidth=2)
        axis.set_title(f"{summary.symbol} strategy equity curve")
        axis.set_ylabel("Portfolio value")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / "equity_curve.png", dpi=180)
        plt.close(figure)


def _max_drawdown_pct(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1
    return float(drawdown.min() * 100) if not drawdown.empty else 0.0


def _sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def _signal_accuracy(signal_frame: pd.DataFrame, horizon: int) -> float:
    future_return = signal_frame["close"].shift(-horizon) / signal_frame["close"] - 1
    buy_checks = future_return[signal_frame["buy_signal"]] > 0
    sell_checks = future_return[signal_frame["sell_signal"]] < 0
    total = len(buy_checks) + len(sell_checks)
    if total == 0:
        return 0.0
    correct = int(buy_checks.sum()) + int(sell_checks.sum())
    return correct / total

