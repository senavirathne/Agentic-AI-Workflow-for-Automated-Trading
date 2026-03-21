# Results And Metrics

## EDA Outputs

The `eda` command writes:

- `descriptive_statistics.csv`
- `correlations.csv`
- `clusters.csv`
- `data_quality.json`
- `price_history.png`
- `rolling_volatility.png`
- `correlation_heatmap.png`
- `asset_clusters.png`
- `eda_summary.md`

These outputs cover:

- data cleaning and missing-value handling
- anomaly flagging
- descriptive statistics
- volatility analysis
- correlation analysis
- K-Means clustering
- feature engineering with RSI, MACD, and moving averages

## Backtest Outputs

The `backtest` command writes:

- `backtest_summary.json`
- `equity_curve.csv`
- `trades.csv`
- `equity_curve.png`

The summary reports:

- total return
- benchmark return
- max drawdown
- Sharpe ratio
- trade count
- win rate
- signal accuracy

## Interpretation Notes

- `signal_accuracy` is a practical proxy for model accuracy in this rule-based system.
- `win_rate` measures closed-trade hit rate.
- `benchmark_return_pct` compares the workflow against passive buy-and-hold on the same symbol.
- `max_drawdown_pct` is the main downside-risk indicator for the backtest.

