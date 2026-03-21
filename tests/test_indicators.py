from __future__ import annotations

from dataclasses import replace

import pandas as pd

from agentic_trading.config import TradingConfig
from agentic_trading.indicators import WindowSignalTracker, compute_rsi, generate_signal_frame


def test_compute_rsi_stays_within_bounds() -> None:
    prices = pd.Series([100, 101, 102, 101, 103, 104, 102, 101, 100, 102, 103, 104, 105, 106, 108, 110])
    rsi = compute_rsi(prices, period=5).dropna()
    assert not rsi.empty
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_window_signal_tracker_can_trigger_buy_and_sell() -> None:
    tracker = WindowSignalTracker(window_size=2)
    buy_signal, sell_signal = tracker.update(1, rsi_prev=25, rsi_now=31, macd_prev=-0.5, macd_now=-0.4, sig_prev=-0.3, sig_now=-0.35)
    assert buy_signal is False
    assert sell_signal is False

    buy_signal, sell_signal = tracker.update(2, rsi_prev=35, rsi_now=40, macd_prev=-0.5, macd_now=0.1, sig_prev=-0.1, sig_now=0.0)
    assert buy_signal is True
    assert sell_signal is False

    buy_signal, sell_signal = tracker.update(3, rsi_prev=75, rsi_now=64, macd_prev=0.5, macd_now=0.4, sig_prev=0.3, sig_now=0.35)
    assert buy_signal is False
    assert sell_signal is False

    buy_signal, sell_signal = tracker.update(4, rsi_prev=68, rsi_now=60, macd_prev=0.4, macd_now=-0.2, sig_prev=0.2, sig_now=0.15)
    assert buy_signal is False
    assert sell_signal is True


def test_generate_signal_frame_returns_strategy_columns() -> None:
    config = replace(
        TradingConfig(),
        rsi_period=3,
        ma_fast=3,
        ma_mid=4,
        ma_slow=5,
        window_size=2,
    )
    main_index = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    trend_index = pd.date_range("2024-12-01", periods=40, freq="D", tz="UTC")
    main_bars = pd.DataFrame(
        {
            "open": range(30),
            "high": [value + 1 for value in range(30)],
            "low": [max(value - 1, 0) for value in range(30)],
            "close": [100 + (value * 0.8) for value in range(30)],
            "volume": [1_000] * 30,
        },
        index=main_index,
    )
    trend_bars = pd.DataFrame(
        {
            "open": range(40),
            "high": [value + 1 for value in range(40)],
            "low": [max(value - 1, 0) for value in range(40)],
            "close": [90 + (value * 1.2) for value in range(40)],
            "volume": [5_000] * 40,
        },
        index=trend_index,
    )
    signal_frame = generate_signal_frame(main_bars, trend_bars, config)
    for column in ["rsi", "macd", "macd_signal", "trend_ma_fast", "buy_signal", "sell_signal", "signal"]:
        assert column in signal_frame.columns

