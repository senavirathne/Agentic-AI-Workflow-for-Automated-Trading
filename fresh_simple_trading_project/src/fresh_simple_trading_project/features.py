"""Feature engineering for 5-minute trading bars."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import TradingConfig


@dataclass
class FeatureEngineeringModule:
    """Compute technical features and rule-based buy/sell triggers."""

    config: TradingConfig

    def build(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Create indicator columns used by analysis and decision modules."""
        frame = bars.copy().sort_index()
        frame["return"] = frame["close"].pct_change()
        frame["ma_short"] = frame["close"].rolling(self.config.short_ma, min_periods=1).mean()
        frame["ma_long"] = frame["close"].rolling(self.config.long_ma, min_periods=1).mean()
        frame["rsi"] = _compute_rsi(frame["close"], self.config.rsi_period)
        macd, macd_signal = _compute_macd(
            frame["close"],
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
        )
        frame["macd"] = macd
        frame["macd_signal"] = macd_signal
        frame["rolling_volatility"] = frame["return"].rolling(20, min_periods=5).std().fillna(0.0)
        frame["buy_trigger"] = (
            (frame["ma_short"] > frame["ma_long"])
            & (frame["macd"] >= frame["macd_signal"])
            & (frame["rsi"] >= self.config.buy_rsi_threshold)
        )
        frame["sell_trigger"] = (
            (frame["ma_short"] < frame["ma_long"])
            | (frame["macd"] < frame["macd_signal"])
            | (frame["rsi"] <= self.config.sell_rsi_threshold)
        )
        return frame


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Compute a simple rolling RSI series."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    relative_strength = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(loss != 0, 100.0)
    rsi = rsi.where(~((gain == 0) & (loss == 0)), 50.0)
    return rsi.fillna(50.0)


def _compute_macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series]:
    """Compute MACD and MACD signal series from closing prices."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal
