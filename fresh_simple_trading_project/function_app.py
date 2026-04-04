import azure.functions as func
import logging
import os
import sys
from pathlib import Path

# Ensure the 'src' directory is in the Python path so we can import our project
project_root = Path(__file__).parent.resolve()
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from fresh_simple_trading_project.workflow import build_workflow

app = func.FunctionApp()

# The schedule "0 0 * * * *" means: 
# Run at the start (0 seconds, 0 minutes) of every hour.
@app.timer_trigger(schedule="0 0 * * * *", arg_name="myTimer", run_on_startup=False, use_monitor=True)
def trading_timer_trigger(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Azure Function started the Trading Workflow...')
    
    # 2. Build and run the workflow
    try:
        workflow = build_workflow(project_root=project_root)
        symbol = os.environ.get("TRADING_SYMBOL", "AAPL")
        
        logging.info(f"Running trade-once for {symbol}")
        # Note: We set execute_orders=True because this is our automated production run.
        result = workflow.run_once(symbol=symbol, execute_orders=True)
        
        logging.info(f"Workflow completed: Decision={result.decision.action.value}, Status={result.execution.status}")
        
    except Exception as e:
        logging.error(f"Workflow failed with error: {str(e)}")
        raise  # Re-raise so Azure Functions marks the execution as failed

    logging.info('Azure Function execution finished.')
