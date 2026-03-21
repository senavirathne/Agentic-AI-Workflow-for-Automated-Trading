from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import TradingConfig
from .indicators import add_features
from .utils import dump_json


def clean_price_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    cleaned = frame.copy().sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    numeric_columns = [column for column in cleaned.columns if column != "symbol"]
    missing_before = float(cleaned[numeric_columns].isna().sum().sum())
    cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()
    returns = cleaned["close"].pct_change()
    zscore = (returns - returns.mean()) / returns.std() if returns.std() else pd.Series(index=returns.index, dtype=float)
    cleaned["return_zscore"] = zscore
    cleaned["return_anomaly"] = cleaned["return_zscore"].abs() > 3
    anomaly_count = float(cleaned["return_anomaly"].sum())
    return cleaned, {
        "missing_values_filled": missing_before,
        "anomaly_count": anomaly_count,
    }


def generate_eda_artifacts(
    symbol_frames: dict[str, pd.DataFrame], output_dir: Path, config: TradingConfig
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_frames: dict[str, pd.DataFrame] = {}
    data_quality: dict[str, dict[str, float]] = {}

    for symbol, frame in symbol_frames.items():
        cleaned, quality = clean_price_frame(frame)
        cleaned_frames[symbol] = add_features(cleaned, config)
        data_quality[symbol] = quality

    close_panel = pd.concat(
        {symbol: frame["close"] for symbol, frame in cleaned_frames.items()},
        axis=1,
    )
    close_panel.columns = list(cleaned_frames.keys())
    returns_panel = close_panel.pct_change().dropna(how="all")

    statistics = _descriptive_statistics(cleaned_frames, data_quality)
    correlations = returns_panel.corr()
    clusters = _cluster_assets(returns_panel, close_panel)

    statistics.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    correlations.to_csv(output_dir / "correlations.csv")
    clusters.to_csv(output_dir / "clusters.csv", index=False)
    dump_json(output_dir / "data_quality.json", data_quality)

    _plot_prices(close_panel, output_dir / "price_history.png")
    _plot_volatility(returns_panel, output_dir / "rolling_volatility.png")
    _plot_correlation(correlations, output_dir / "correlation_heatmap.png")
    _plot_clusters(clusters, output_dir / "asset_clusters.png")
    _write_summary(statistics, correlations, clusters, output_dir / "eda_summary.md")

    return {
        "stats": output_dir / "descriptive_statistics.csv",
        "correlations": output_dir / "correlations.csv",
        "clusters": output_dir / "clusters.csv",
        "summary": output_dir / "eda_summary.md",
    }


def _descriptive_statistics(
    cleaned_frames: dict[str, pd.DataFrame], data_quality: dict[str, dict[str, float]]
) -> pd.DataFrame:
    rows = []
    for symbol, frame in cleaned_frames.items():
        close = frame["close"]
        returns = close.pct_change().dropna()
        rows.append(
            {
                "symbol": symbol,
                "observations": len(frame),
                "start": frame.index.min(),
                "end": frame.index.max(),
                "close_mean": close.mean(),
                "close_std": close.std(),
                "close_min": close.min(),
                "close_max": close.max(),
                "annual_return_pct": ((close.iloc[-1] / close.iloc[0]) - 1) * 100,
                "annualized_volatility_pct": returns.std() * np.sqrt(252) * 100 if not returns.empty else 0.0,
                "latest_rsi": frame["rsi"].iloc[-1],
                "latest_macd": frame["macd"].iloc[-1],
                "missing_values_filled": data_quality[symbol]["missing_values_filled"],
                "anomaly_count": data_quality[symbol]["anomaly_count"],
            }
        )
    return pd.DataFrame(rows)


def _cluster_assets(returns_panel: pd.DataFrame, close_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in close_panel.columns:
        returns = returns_panel[symbol].dropna()
        price = close_panel[symbol].dropna()
        if returns.empty or price.empty:
            continue
        rows.append(
            {
                "symbol": symbol,
                "annual_return_pct": ((price.iloc[-1] / price.iloc[0]) - 1) * 100,
                "annualized_volatility_pct": returns.std() * np.sqrt(252) * 100,
                "sharpe_proxy": (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else 0.0,
            }
        )

    features = pd.DataFrame(rows)
    if len(features) < 2:
        features["cluster"] = 0
        return features

    cluster_count = min(3, len(features))
    scaled = StandardScaler().fit_transform(
        features[["annual_return_pct", "annualized_volatility_pct", "sharpe_proxy"]]
    )
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
    features["cluster"] = model.fit_predict(scaled)
    return features


def _plot_prices(close_panel: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    close_panel.plot(ax=axis, linewidth=1.5)
    axis.set_title("Asset price history")
    axis.set_ylabel("Close price")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_volatility(returns_panel: pd.DataFrame, path: Path) -> None:
    rolling_vol = returns_panel.rolling(20).std() * np.sqrt(252)
    figure, axis = plt.subplots(figsize=(12, 6))
    rolling_vol.plot(ax=axis, linewidth=1.5)
    axis.set_title("20-day rolling annualized volatility")
    axis.set_ylabel("Volatility")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_correlation(correlation: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.index)))
    axis.set_yticklabels(correlation.index)
    axis.set_title("Return correlation heatmap")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_clusters(clusters: pd.DataFrame, path: Path) -> None:
    if clusters.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    scatter = axis.scatter(
        clusters["annualized_volatility_pct"],
        clusters["annual_return_pct"],
        c=clusters["cluster"],
        cmap="viridis",
        s=120,
    )
    for _, row in clusters.iterrows():
        axis.annotate(row["symbol"], (row["annualized_volatility_pct"], row["annual_return_pct"]))
    axis.set_xlabel("Annualized volatility %")
    axis.set_ylabel("Annual return %")
    axis.set_title("K-Means asset clusters")
    figure.colorbar(scatter, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_summary(
    statistics: pd.DataFrame, correlations: pd.DataFrame, clusters: pd.DataFrame, path: Path
) -> None:
    highest_vol = statistics.sort_values("annualized_volatility_pct", ascending=False).iloc[0]
    strongest_pair = _strongest_correlation_pair(correlations)
    highest_return = statistics.sort_values("annual_return_pct", ascending=False).iloc[0]

    lines = [
        "# EDA Summary",
        "",
        "## Data Cleaning",
        f"- Missing values filled across the universe: {int(statistics['missing_values_filled'].sum())}",
        f"- Return anomalies flagged across the universe: {int(statistics['anomaly_count'].sum())}",
        "",
        "## Patterns",
        f"- Highest trailing return: {highest_return['symbol']} at {highest_return['annual_return_pct']:.2f}%.",
        f"- Highest annualized volatility: {highest_vol['symbol']} at {highest_vol['annualized_volatility_pct']:.2f}%.",
        f"- Strongest correlation pair: {strongest_pair}.",
        "",
        "## Risk Indicators",
        "- Volatility and correlation outputs are saved alongside this summary.",
        "- Latest RSI and MACD values are included in `descriptive_statistics.csv` for watchlist screening.",
        "",
        "## Clustering",
        f"- Assets were grouped into {clusters['cluster'].nunique() if not clusters.empty else 0} cluster(s) using return, volatility, and Sharpe proxy.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _strongest_correlation_pair(correlations: pd.DataFrame) -> str:
    best_pair = "N/A"
    best_value = -1.0
    for left, right in combinations(correlations.columns, 2):
        value = abs(float(correlations.loc[left, right]))
        if value > best_value:
            best_value = value
            best_pair = f"{left}/{right} ({correlations.loc[left, right]:.2f})"
    return best_pair

