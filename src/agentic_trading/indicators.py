from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TradingConfig


def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0.0)
    losses = (-deltas).where(deltas < 0, 0.0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(prices: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def add_features(frame: pd.DataFrame, config: TradingConfig) -> pd.DataFrame:
    features = frame.copy()
    features["returns"] = features["close"].pct_change()
    features["rsi"] = compute_rsi(features["close"], config.rsi_period)
    features["ma_fast"] = features["close"].rolling(config.ma_fast).mean()
    features["ma_mid"] = features["close"].rolling(config.ma_mid).mean()
    features["ma_slow"] = features["close"].rolling(config.ma_slow).mean()
    features["volatility_20"] = features["returns"].rolling(20).std() * np.sqrt(252)
    macd_line, signal_line = compute_macd(
        features["close"], config.macd_fast, config.macd_slow, config.macd_signal
    )
    features["macd"] = macd_line
    features["macd_signal"] = signal_line
    features["macd_hist"] = macd_line - signal_line
    return features


def build_trend_context(
    main_index: pd.Index, trend_frame: pd.DataFrame, config: TradingConfig
) -> pd.DataFrame:
    trend_features = add_features(trend_frame, config)
    trend_view = trend_features[["ma_fast", "ma_mid", "ma_slow"]].rename(
        columns={
            "ma_fast": "trend_ma_fast",
            "ma_mid": "trend_ma_mid",
            "ma_slow": "trend_ma_slow",
        }
    )
    return trend_view.reindex(main_index, method="ffill")


@dataclass
class WindowSignalTracker:
    window_size: int
    rsi_bounce_bar: int | None = None
    macd_cross_bar: int | None = None
    rsi_retreat_bar: int | None = None
    macd_death_cross_bar: int | None = None
    macd_centerline_bar: int | None = None

    def update(
        self,
        bar_index: int,
        rsi_prev: float,
        rsi_now: float,
        macd_prev: float,
        macd_now: float,
        sig_prev: float,
        sig_now: float,
    ) -> tuple[bool, bool]:
        if rsi_prev < 30 and rsi_now > 30:
            self.rsi_bounce_bar = bar_index
        if macd_prev < sig_prev and macd_now > sig_now:
            self.macd_cross_bar = bar_index

        buy_trigger = (
            self.rsi_bounce_bar is not None
            and self.macd_cross_bar is not None
            and abs(self.rsi_bounce_bar - self.macd_cross_bar) <= self.window_size
        )
        if buy_trigger:
            self.rsi_bounce_bar = None
            self.macd_cross_bar = None

        if rsi_prev > 70 and rsi_now < 65:
            self.rsi_retreat_bar = bar_index
        if macd_prev > sig_prev and macd_now < sig_now:
            self.macd_death_cross_bar = bar_index
        elif macd_prev > 0 and macd_now < 0:
            self.macd_centerline_bar = bar_index

        sell_trigger = (
            self.rsi_retreat_bar is not None
            and (
                (
                    self.macd_death_cross_bar is not None
                    and abs(self.rsi_retreat_bar - self.macd_death_cross_bar) <= self.window_size
                )
                or (
                    self.macd_centerline_bar is not None
                    and abs(self.rsi_retreat_bar - self.macd_centerline_bar) <= self.window_size
                )
            )
        )
        if sell_trigger:
            self.rsi_retreat_bar = None
            self.macd_death_cross_bar = None
            self.macd_centerline_bar = None

        return buy_trigger, sell_trigger


def generate_signal_frame(
    main_frame: pd.DataFrame, trend_frame: pd.DataFrame, config: TradingConfig
) -> pd.DataFrame:
    features = add_features(main_frame, config)
    trend_context = build_trend_context(features.index, trend_frame, config)
    features = features.join(trend_context)
    features["in_uptrend"] = (
        (features["trend_ma_fast"] > features["trend_ma_mid"])
        & (features["trend_ma_mid"] > features["trend_ma_slow"])
    )
    features["buy_signal"] = False
    features["sell_signal"] = False
    features["signal"] = "HOLD"

    tracker = WindowSignalTracker(window_size=config.window_size)
    for index in range(1, len(features)):
        previous = features.iloc[index - 1]
        current = features.iloc[index]
        required = [
            previous["rsi"],
            current["rsi"],
            previous["macd"],
            current["macd"],
            previous["macd_signal"],
            current["macd_signal"],
        ]
        if any(pd.isna(required)):
            continue

        buy_signal, sell_signal = tracker.update(
            bar_index=index,
            rsi_prev=float(previous["rsi"]),
            rsi_now=float(current["rsi"]),
            macd_prev=float(previous["macd"]),
            macd_now=float(current["macd"]),
            sig_prev=float(previous["macd_signal"]),
            sig_now=float(current["macd_signal"]),
        )

        if buy_signal and bool(current["in_uptrend"]):
            features.iloc[index, features.columns.get_loc("buy_signal")] = True
        if sell_signal:
            features.iloc[index, features.columns.get_loc("sell_signal")] = True

    features.loc[features["buy_signal"], "signal"] = "BUY"
    features.loc[features["sell_signal"], "signal"] = "SELL"
    return features

