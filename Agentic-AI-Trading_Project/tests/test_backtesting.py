from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from project.backtesting import BacktestingEngine
from project.config import Settings
from project.decision_engine import DecisionEngine
from project.eda import EDAModule
from project.features import FeatureEngineeringModule
from project.market_analysis import MarketAnalysisModule
from project.risk_analysis import RiskAnalysisModule


def test_backtesting_engine_generates_summary_and_trades(tmp_path: Path) -> None:
    settings = Settings.from_env(project_root=tmp_path)
    engine = BacktestingEngine(
        config=settings.trading,
        feature_engineering=FeatureEngineeringModule(settings.trading),
        eda_module=EDAModule(),
        market_analysis=MarketAnalysisModule(settings.trading),
        risk_analysis=RiskAnalysisModule(settings.trading),
        decision_engine=DecisionEngine(),
    )
    five_minute_bars = _make_backtest_bars()
    hourly_bars = five_minute_bars.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    summary = engine.run("AAPL", five_minute_bars, hourly_bars, sleep_seconds=0)

    assert summary.symbol == "AAPL"
    assert not summary.equity_curve.empty
    assert summary.ending_cash > 0


def _make_backtest_bars() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=(7 * 24 * 12) + (6 * 12), freq="5min", tz="UTC")
    uptrend = np.linspace(100.0, 145.0, num=7 * 24 * 12)
    pullback = np.linspace(145.0, 130.0, num=len(index) - len(uptrend))
    close = np.concatenate([uptrend, pullback])
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "volume": np.full(len(index), 40_000),
        },
        index=index,
    )
