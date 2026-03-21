# Agentic AI Workflow for Automated Trading

This repository implements a stock-focused trading workflow using the Alpaca reference bot in:

- `examples/stocks/build_trading_bot_with_ChatGPT/trading_bot_chatgpt.ipynb`
- `examples/stocks/build_trading_bot_with_ChatGPT/strategy.py`

The current implementation intentionally stays close to those resources. It turns the single-script Alpaca RSI/MACD strategy into a modular workflow with:

- a market analysis module
- an information retrieval module using Alpaca News
- a decision engine
- a risk management module
- a paper-trade execution module
- EDA and backtesting utilities

Where the supplied resources do not fully cover the assignment, the code leaves explicit extension seams open instead of guessing at a larger solution. That is most visible in the cloud storage layer and future AI/LLM decision hooks.

## Project Structure

```text
src/agentic_trading/
  alpaca_clients.py    # Alpaca data, news, and paper-trading integration
  agents.py            # Market analysis, retrieval, decision, risk, execution agents
  analysis.py          # EDA, volatility/correlation analysis, K-Means clustering
  backtesting.py       # Strategy backtester and performance metrics
  cli.py               # CLI entry point
  indicators.py        # RSI, MACD, moving averages, signal generation
  storage.py           # Local raw-data storage + SQL store + cloud placeholders
  workflow.py          # Hourly orchestration loop based on the Alpaca example
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Populate `.env` with your Alpaca paper credentials before using the data collection, backtesting, or trading commands.

## Commands

Collect raw daily bars for the default universe:

```bash
python -m agentic_trading.cli collect --include-news
```

Run EDA:

```bash
python -m agentic_trading.cli eda
```

Backtest the Alpaca RSI/MACD workflow on the primary symbol:

```bash
python -m agentic_trading.cli backtest --symbol TQQQ
```

Run one dry-run workflow cycle:

```bash
python -m agentic_trading.cli trade-once --symbol TQQQ
```

Run the hourly paper-trading loop:

```bash
python -m agentic_trading.cli trade-loop --symbol TQQQ --execute
```

## Outputs

- Raw bars and news: `data/raw/`
- Structured run history: `data/trading_workflow.db`
- EDA artifacts: `reports/output/eda/`
- Backtest artifacts: `reports/output/backtests/<symbol>/`
- Logs: `logs/workflow.log`

## Documentation

- [Architecture](docs/architecture.md)
- [Cloud Setup And Boundaries](docs/cloud_setup.md)
- [Results And Metrics](docs/results.md)

## Scope Boundaries

- Current asset coverage is US stocks because the provided Alpaca example is a stocks workflow.
- The decision engine still uses the RSI/MACD + moving average logic from Alpaca as its trade rule.
- The information retrieval module is implemented with Alpaca News, but it only enriches rationale in this version.
- AWS S3 / Azure Blob and AWS RDS / Azure SQL are left as adapter boundaries, not fully provisioned infrastructure.
- This code is for paper trading and academic analysis, not production deployment.
