from __future__ import annotations

from dataclasses import replace

import pandas as pd

from agentic_trading.backtesting import StrategyBacktester
from agentic_trading.config import TradingConfig


def test_backtester_generates_closed_trade(monkeypatch) -> None:
    config = replace(TradingConfig(), initial_capital=1_000.0, buy_power_limit=0.5, signal_horizon_bars=1)
    backtester = StrategyBacktester(config)

    signal_frame = pd.DataFrame(
        {
            "close": [100.0, 102.0, 104.0, 103.0],
            "buy_signal": [False, True, False, False],
            "sell_signal": [False, False, False, True],
        },
        index=pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
    )

    monkeypatch.setattr(
        backtester,
        "prepare_signal_frame",
        lambda short, medium, long: signal_frame,
    )
    summary = backtester.backtest("TQQQ", signal_frame, signal_frame, signal_frame)

    assert summary.trade_count == 1
    assert len(summary.trades) == 1
    assert summary.ending_capital == 1004.0
    assert summary.total_return_pct > 0
