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

## Azure Function App VM Trigger

This project already includes a standalone Azure Functions entrypoint in `function_app.py`. The Function App does not run the trading workflow itself. Instead, it:

1. Accepts an HTTP or timer trigger.
2. Starts the target Azure VM if it is stopped.
3. Uses Azure VM Run Command to execute the trading CLI inside the VM.

### What must exist on the VM

- The project must already be copied to the VM at the same path you configure in `VM_PROJECT_DIR`.
- The VM must have a working virtual environment that can be activated from `VM_VENV_ACTIVATE`.
- The VM copy of the project must have its own `.env` with trading credentials such as `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ALPHA_VANTAGE_API_KEY`, and Alpaca keys when used.

Example VM command executed by the Function App:

```bash
cd /opt/fresh_simple_trading_project
source /opt/fresh_simple_trading_project/.venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m fresh_simple_trading_project.cli run --mode live --symbol AAPL
```

If `mode=backtest`, the Function App adds `--max-iterations <loops>`. If `live_after_backtest=true`, it runs a live pass after the backtest. If `VM_AUTO_SHUTDOWN=true`, the VM is shut down after the command finishes.

The workflow is launched in the background on the VM so the Function App can serve later log-tail requests while the workflow is still running.

### Function App settings

Set these application settings in the Azure Function App:

```text
FUNCTIONS_WORKER_RUNTIME=python
AZURE_SUBSCRIPTION_ID=<subscription-id>
AZURE_VM_RESOURCE_GROUP=<resource-group>
AZURE_VM_NAME=<vm-name>
VM_PROJECT_DIR=/opt/fresh_simple_trading_project
VM_VENV_ACTIVATE=/opt/fresh_simple_trading_project/.venv/bin/activate
VM_DEFAULT_SYMBOL=AAPL
VM_DEFAULT_LOOPS=1
VM_DEFAULT_MODE=live
VM_TIMER_ENABLED=false
VM_AUTO_SHUTDOWN=true
VM_LOG_DIR=/opt/fresh_simple_trading_project/logs
```

Optional settings:

- `AZURE_CLIENT_ID` if the Function App uses a user-assigned managed identity.
- `FUNCTION_APP_STORAGE_ROOT` if you want the dispatch-state file written to a specific writable directory instead of the default temp location inside Azure Functions.
- `VM_LOG_DIR` if you want VM workflow logs written somewhere other than `<VM_PROJECT_DIR>/logs`.

For local testing, copy `local.settings.json.example` to `local.settings.json` and fill in the real values.

### Azure permissions

Enable a managed identity on the Function App and grant it RBAC on the target VM or its resource group so it can:

- start the VM
- read instance view
- invoke Run Command

In practice, assign a role that includes those `Microsoft.Compute/virtualMachines/*` actions. `Virtual Machine Contributor` at VM or resource-group scope is the usual starting point.

### Build and deploy the standalone Function App

The Function App no longer depends on `src/fresh_simple_trading_project`. Build the upload bundle with:

```bash
./scripts/build_function_app_package.sh
```

That command creates:

- `dist/function_app_vm_dispatch/`
- `dist/function_app_vm_dispatch.zip`

Upload only that generated bundle to Azure Function App. It contains:

- `function_app.py`
- `host.json`
- minimal `requirements.txt`
- `.funcignore`
- `local.settings.json.example`

You can publish with Azure Functions Core Tools from the generated folder:

```bash
cd dist/function_app_vm_dispatch
func azure functionapp publish <YOUR_FUNCTION_APP_NAME>
```

Or deploy the generated zip package:

```bash
az functionapp deployment source config-zip \
  --resource-group <RESOURCE_GROUP> \
  --name <YOUR_FUNCTION_APP_NAME> \
  --src dist/function_app_vm_dispatch.zip
```

### Trigger the VM run

The Function App exposes these `AuthLevel.FUNCTION` routes:

- `GET|POST /api/trading/vm/start`
- `GET /api/trading/vm/status`
- `GET /api/trading/vm/log`
- Compatibility aliases: `/api/trading/session/start`, `/api/trading/session/status`, `/api/trading/session/log`

Behavior mapping for the VM trigger:

- Backtest only: call `GET|POST /api/trading/vm/start` with `mode=backtest` and `loops=<n>`. The VM runs the backtest loop for exactly `<n>` iterations, then stops at the end when `VM_AUTO_SHUTDOWN=true`.
- Live only: call `GET|POST /api/trading/vm/start` with `mode=live`. The VM runs the live workflow, waits roughly one hour between hourly checkpoints, and exits when the live market is closed. When `VM_AUTO_SHUTDOWN=true`, the VM then powers off.
- Backtest then live: call `GET|POST /api/trading/vm/start` with `mode=backtest`, `loops=<n>`, and `live_after_backtest=true`. The VM completes the backtest iterations first, then starts the live workflow, and keeps going until the market closes. When `VM_AUTO_SHUTDOWN=true`, shutdown happens only after the live run finishes.
- Each accepted dispatch response includes `log_file_path`, `log_tail_command`, `log_url`, and `log_download_url`.

Example start request:

```bash
curl -X POST \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/start?code=<FUNCTION_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "loops": 1,
    "mode": "live",
    "force": false,
    "live_after_backtest": false
  }'
```

Equivalent GET request:

```bash
curl \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/session/start?code=<FUNCTION_KEY>&symbol=AAPL&mode=backtest&loops=6"
```

Example status request:

```bash
curl \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/status?code=<FUNCTION_KEY>"
```

Download the current full log snapshot:

```bash
curl -L \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&log_file_path=<URL_ENCODED_LOG_FILE_PATH>&download=true" \
  -o workflow.log
```

Example log-tail request from your local terminal:

```bash
curl \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>"
```

Fetch more lines from a specific log file:

```bash
curl \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&lines=120&log_file_path=/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log"
```

If the VM has already shut down and you still want to read the log through the Function App, start it on demand:

```bash
curl \
  "https://<YOUR_FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&start_if_needed=true"
```

### Timer trigger

`trading_timer_trigger()` is scheduled hourly. To let the timer dispatch a VM run:

- set `VM_TIMER_ENABLED=true`
- keep `VM_DEFAULT_MODE=live`

The timer skips dispatch when it detects an active VM run.
