# Fresh Simple Trading Project

This is a standalone project inside the repository. It does not reuse the existing trading package structure.

The project implements a simple Agentic AI trading workflow that analyzes 5-minute candlestick data while the main loop runs once per hour.

Live trading and backtesting now use the same hourly workflow shape:

- Last 24 hours of 5-minute candles are used to compute and tune indicators.
- Last 7 days of 1-hour candles are used to calculate support and resistance.
- Live mode uses the live/recent market-data provider and defaults to a `3600` second sleep.
- Backtest mode uses the historical market-data provider and defaults to a `1` second replay sleep.

The workflow always builds its LLM path. `build_workflow()` and the CLI fail fast if `DEEPSEEK_API_KEY` is missing, and the LLM path is split into explicit text-only agents:

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
PYTHONPATH=src python -m fresh_simple_trading_project.cli trade-loop --symbol AAPL --sleep-seconds 3600
PYTHONPATH=src python -m fresh_simple_trading_project.cli backtest --symbol AAPL --sleep-seconds 1
PYTHONPATH=src python -m fresh_simple_trading_project.cli trade-loop --symbol AAPL --sleep-seconds 1
```

`trade-once` now prints the LLM agent handoffs and final decision rationale by default. Use `trade-once --json` if you want the structured JSON payload instead.

`trade-loop` accepts `--sleep-seconds` so you can compress the wall-clock wait between hourly workflow iterations during demos or simulations while keeping the trading checkpoint cadence at one hour. Live mode defaults to `3600`.

`backtest` now replays the same hourly workflow cadence using historical market data, with a default replay sleep of `1` second between hourly checkpoints.

The workflow now merges two live news sources for the news agent: Alpaca news plus live web-search news. If `ALPACA_PAPER_API_KEY` and `ALPACA_PAPER_SECRET_KEY` are present, Alpaca news is included automatically; the web-search feed is always enabled. Set `MARKET_DATA_PROVIDER=alpaca` if you also want live 5-minute Alpaca market data and live paper-account state.

Before using the CLI or `build_workflow()`, export:

```bash
export DEEPSEEK_API_KEY=your_key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-reasoner
```

To use the Alpaca setup pattern from the notebook, export:

```bash
export ALPACA_PAPER_API_KEY=your_key
export ALPACA_PAPER_SECRET_KEY=your_secret
export PAPER=true
export TRADE_API_URL=https://paper-api.alpaca.markets
```

When those credentials are present, the fresh project switches its market data, account state, and live order execution to Alpaca while keeping the same simple workflow modules.

Keep the DeepSeek key in environment variables or `.env`. Do not hardcode API keys in notebooks or source files.

EDA, feature engineering, and the risk/decision guardrails remain deterministic even though the workflow now always includes the LLM agents.
