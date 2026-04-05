# Azure Function VM Dispatcher

This bundle is a standalone Azure Function App that starts a VM and runs the trading workflow already installed on that VM.

## Files in this bundle

- `function_app.py`
- `host.json`
- `requirements.txt`
- `.funcignore`
- `local.settings.json.example`

## Required Function App settings

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

- `AZURE_CLIENT_ID` for a user-assigned managed identity
- `FUNCTION_APP_STORAGE_ROOT` to override where the dispatch state file is written

## VM prerequisites

The target VM must already contain the trading project and its virtual environment.

Expected VM command shape:

```bash
cd /opt/fresh_simple_trading_project
source /opt/fresh_simple_trading_project/.venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m fresh_simple_trading_project.cli run --mode live --symbol AAPL
```

Each Azure-triggered run writes stdout/stderr to a VM log file under `VM_LOG_DIR`.

Example:

```bash
tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log
```

## Endpoints

- `GET|POST /api/trading/vm/start`
- `GET /api/trading/vm/status`
- `GET /api/trading/vm/log`
- Compatibility aliases: `/api/trading/session/start`, `/api/trading/session/status`, `/api/trading/session/log`

Example:

```bash
curl -X POST \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/start?code=<FUNCTION_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","mode":"backtest","loops":3,"live_after_backtest":true}'
```

Equivalent GET request:

```bash
curl \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/session/start?code=<FUNCTION_KEY>&symbol=AAPL&mode=backtest&loops=3"
```

Accepted start responses include `log_url` and `log_download_url`. Those generated URLs already include
`start_if_needed=true` and `wait_for_running_seconds=90` so they can tolerate the VM still transitioning
through Azure's `starting` state when opened immediately after dispatch.

Fetch the latest saved log tail from your local terminal:

```bash
curl \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>"
```

Fetch a specific number of log lines:

```bash
curl \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&lines=120"
```

Download the full log file snapshot:

```bash
curl -L \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&log_file_path=<URL_ENCODED_LOG_FILE_PATH>&download=true" \
  -o workflow.log
```

If you want the endpoint to wait for a booting VM to reach `running`, you can also pass:

```bash
curl \
  "https://<FUNCTION_APP_NAME>.azurewebsites.net/api/trading/vm/log?code=<FUNCTION_KEY>&log_file_path=<URL_ENCODED_LOG_FILE_PATH>&start_if_needed=true&wait_for_running_seconds=90"
```
