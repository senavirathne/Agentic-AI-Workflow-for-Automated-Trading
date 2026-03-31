from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import EDAResult


@dataclass
class EDAModule:
    def summarize(self, symbol: str, bars: pd.DataFrame) -> EDAResult:
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

    def cluster_assets(self, bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for symbol, bars in bars_by_symbol.items():
            cleaned = bars.copy().sort_index()
            returns = cleaned["close"].pct_change().dropna()
            if cleaned.empty or returns.empty:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "mean_return": float(returns.mean()),
                    "volatility": float(returns.std()),
                    "latest_close": float(cleaned["close"].iloc[-1]),
                }
            )
        features = pd.DataFrame(rows)
        if features.empty:
            return features
        if len(features) == 1:
            features["cluster"] = 0
            return features

        values = features[["mean_return", "volatility", "latest_close"]].to_numpy(dtype=float)
        scaled = _standardize(values)
        features["cluster"] = _fit_simple_kmeans(scaled, cluster_count=min(3, len(features)))
        return features


def _standardize(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    return (values - means) / stds


def _fit_simple_kmeans(values: np.ndarray, cluster_count: int, max_iterations: int = 25) -> np.ndarray:
    centroids = values[:cluster_count].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(max_iterations):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        updated_labels = distances.argmin(axis=1)
        if np.array_equal(labels, updated_labels):
            break
        labels = updated_labels
        for cluster in range(cluster_count):
            members = values[labels == cluster]
            if len(members) > 0:
                centroids[cluster] = members.mean(axis=0)
    return labels
