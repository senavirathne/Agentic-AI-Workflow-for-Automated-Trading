from __future__ import annotations

from dataclasses import replace

import pandas as pd

from agentic_trading.config import TradingConfig
from agentic_trading.indicators import WindowSignalTracker, compute_rsi, generate_signal_frame


def _build_bar_frame(
    start: str,
    periods: int,
    freq: str,
    closes: list[float],
    volume: int = 1_000,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [max(value - 1, 0) for value in closes],
            "close": closes,
            "volume": [volume] * periods,
        },
        index=pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
    )


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
    short_bars = _build_bar_frame(
        "2025-01-01",
        periods=30,
        freq="15min",
        closes=[100 + (value * 0.8) for value in range(30)],
    )
    medium_bars = _build_bar_frame(
        "2024-12-01",
        periods=40,
        freq="2h",
        closes=[95 + (value * 1.0) for value in range(40)],
        volume=2_500,
    )
    long_bars = _build_bar_frame(
        "2024-01-01",
        periods=260,
        freq="D",
        closes=[90 + (value * 0.5) for value in range(260)],
        volume=5_000,
    )
    signal_frame = generate_signal_frame(short_bars, medium_bars, long_bars, config)
    for column in [
        "rsi",
        "macd",
        "macd_signal",
        "medium_ma_fast",
        "medium_trend_bullish",
        "trend_ma_fast",
        "buy_signal",
        "sell_signal",
        "signal",
    ]:
        assert column in signal_frame.columns


def test_generate_signal_frame_requires_medium_and_long_trend_confirmation(monkeypatch) -> None:
    config = replace(
        TradingConfig(),
        rsi_period=2,
        ma_fast=2,
        ma_mid=3,
        ma_slow=4,
        window_size=1,
    )
    short_bars = _build_bar_frame(
        "2025-01-01",
        periods=5,
        freq="15min",
        closes=[100.0, 99.0, 100.0, 99.5, 100.5],
    )
    medium_bullish = _build_bar_frame(
        "2024-12-01",
        periods=6,
        freq="2h",
        closes=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    )
    medium_bearish = _build_bar_frame(
        "2024-12-01",
        periods=6,
        freq="2h",
        closes=[15.0, 14.0, 13.0, 12.0, 11.0, 10.0],
    )
    long_bullish = _build_bar_frame(
        "2024-01-01",
        periods=6,
        freq="D",
        closes=[20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        volume=5_000,
    )
    long_bearish = _build_bar_frame(
        "2024-01-01",
        periods=6,
        freq="D",
        closes=[25.0, 24.0, 23.0, 22.0, 21.0, 20.0],
        volume=5_000,
    )

    monkeypatch.setattr(
        WindowSignalTracker,
        "update",
        lambda self, bar_index, **kwargs: (bar_index == 4, False),
    )

    fully_aligned = generate_signal_frame(short_bars, medium_bullish, long_bullish, config)
    blocked_by_medium = generate_signal_frame(short_bars, medium_bearish, long_bullish, config)
    blocked_by_long = generate_signal_frame(short_bars, medium_bullish, long_bearish, config)

    assert bool(fully_aligned.iloc[-1]["buy_signal"]) is True
    assert bool(fully_aligned.iloc[-1]["medium_trend_bullish"]) is True
    assert bool(fully_aligned.iloc[-1]["in_uptrend"]) is True
    assert bool(blocked_by_medium.iloc[-1]["buy_signal"]) is False
    assert bool(blocked_by_medium.iloc[-1]["medium_trend_bullish"]) is False
    assert bool(blocked_by_long.iloc[-1]["buy_signal"]) is False
    assert bool(blocked_by_long.iloc[-1]["in_uptrend"]) is False
