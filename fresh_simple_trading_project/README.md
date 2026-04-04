# Fresh Simple Trading Project

This is a standalone project inside the repository. It does not reuse the existing trading package structure.

The project implements a simple Agentic AI trading workflow that analyzes 5-minute candlestick data while the main loop runs once per hour.

Live trading and backtesting now use the same hourly workflow shape:

- Last 24 hours of 5-minute candles are used to compute and tune indicators.
- Last 7 days of 1-hour candles are used to calculate support and resistance.
- Live mode uses the live/recent market-data provider and defaults to a `3600` second sleep.
- Backtest mode uses the historical market-data provider and defaults to a `1` second replay sleep.
- Backtest checkpoints load Alpha Vantage indicator snapshots and 1-hour chunks from the local SQLite result store.

The workflow always builds its LLM path. `build_workflow()` and the CLI fail fast unless at least one LLM key is configured. DeepSeek remains the primary client, and if an OpenAI key is present the workflow automatically retries on OpenAI when the DeepSeek request fails due to exhausted credits or quota. The LLM path is split into explicit text-only agents:

- `TechnicalAnalysisAgent` receives the computed market snapshot as plain text before writing its handoff.
- `NewsResearchAgent` receives recent articles, query context, sentiment, catalysts, and risk flags as plain text.
- `RiskReviewAgent` turns the quantitative risk state into a trade-size/risk handoff.
- `DecisionCoordinatorAgent` reviews the market, news, and risk context as plain text before producing the final guarded trade decision.

- Data Collection
- EDA
- Feature Engineering
- Market Analysis
- Information Retrieval
- Risk Analysis
- Decision Engine
- Execution
- Backtesting

## Fresh Project Layout

```text
fresh_simple_trading_project/
  docs/
    architecture.md
  src/fresh_simple_trading_project/
    agents.py
    backtesting.py
    cli.py
    config.py
    data_collection.py
    decision_engine.py
    eda.py
    execution.py
    features.py
    information_retrieval.py
    market_analysis.py
    models.py
    risk_analysis.py
    storage.py
    utils.py
    workflow.py
  tests/
```

## Quick Start

```bash
cd fresh_simple_trading_project
python -m pytest
PYTHONPATH=src python -m fresh_simple_trading_project.cli trade-once --symbol AAPL
PYTHONPATH=src python -m fresh_simple_trading_project.cli alpha-vantage-indicators --symbol AAPL
PYTHONPATH=src python -m fresh_simple_trading_project.cli trade-loop --symbol AAPL --sleep-seconds 3600
PYTHONPATH=src python -m fresh_simple_trading_project.cli backtest --symbol AAPL --sleep-seconds 1
PYTHONPATH=src python -m fresh_simple_trading_project.cli trade-loop --symbol AAPL --sleep-seconds 1
```

`trade-once` now prints the LLM agent handoffs and final decision rationale by default. Use `trade-once --json` if you want the structured JSON payload instead.

`alpha-vantage-indicators` fetches the requested Alpha Vantage technical indicators on `5min` candles, aligns them on common timestamps, filters to the latest trading day, stores the JSON snapshot in the configured SQLite/Azure SQL result store, and prints the JSON table plus hourly chunks.

`trade-loop` accepts `--sleep-seconds` so you can compress the wall-clock wait between hourly workflow iterations during demos or simulations while keeping the trading checkpoint cadence at one hour. Live mode defaults to `3600`.

`backtest` now replays the same hourly workflow cadence using historical market data, with a default replay sleep of `1` second between hourly checkpoints.

The workflow now merges two live news sources for the news agent: Alpha Vantage news plus live web-search news. If `ALPHA_VANTAGE_API_KEY` is present, Alpha Vantage `NEWS_SENTIMENT` is included automatically with a dynamic `time_from` window that expands backward based on the news-agent input budget; the web-search feed is always enabled. Set `LIVE_MARKET_DATA_PROVIDER=alpaca` for live 5-minute Alpaca market data and live paper-account state. Backtests replay Alpaca historical OHLCV bars and layer Alpha Vantage-backed indicator snapshots from the local SQLite DB on top.

Before using the CLI or `build_workflow()`, export at least one LLM key:

```bash
export DEEPSEEK_API_KEY=your_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-reasoner
export OPENAI_API_KEY=your_openai_key
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-5.4-mini
export ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

To use the Alpaca setup pattern from the notebook, export:

```bash
export ALPACA_PAPER_API_KEY=your_key
export ALPACA_PAPER_SECRET_KEY=your_secret
export PAPER=true
export TRADE_API_URL=https://paper-api.alpaca.markets
```

When those credentials are present, the fresh project switches its market data, account state, and live order execution to Alpaca while keeping the same simple workflow modules.

Keep the DeepSeek and OpenAI keys in environment variables or `.env`. Do not hardcode API keys in notebooks or source files.

EDA, feature engineering, and the risk/decision guardrails remain deterministic even though the workflow now always includes the LLM agents.
