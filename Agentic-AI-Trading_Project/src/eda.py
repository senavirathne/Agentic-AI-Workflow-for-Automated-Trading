"""Exploratory data analysis helpers for recent bar windows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import EDAResult


@dataclass
class EDAModule:
    """Compute lightweight descriptive statistics for recent market data."""

    def summarize(self, symbol: str, bars: pd.DataFrame) -> EDAResult:
        """Summarize missing data, volatility, returns, and anomalies."""
        cleaned = bars.copy().sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
        missing_values = int(cleaned.isna().sum().sum())
        cleaned = cleaned.ffill().bfill()
        returns = cleaned["close"].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.0
        mean_return = float(returns.mean()) if not returns.empty else 0.0
        if returns.empty or returns.std() == 0:
            anomaly_count = 0
        else:
            zscore = (returns - returns.mean()) / returns.std()
            anomaly_count = int((zscore.abs() > 3).sum())
        return EDAResult(
            symbol=symbol,
            missing_values=missing_values,
            anomaly_count=anomaly_count,
            candle_return_mean=round(mean_return, 6),
            candle_volatility=round(volatility, 6),
            latest_close=float(cleaned["close"].iloc[-1]),
        )
