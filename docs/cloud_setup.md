# Cloud Setup And Boundaries

## Implemented Now

- `deploy/aws/ec2_bootstrap.sh`
  - Bootstraps an EC2 host with Python, a virtual environment, and project dependencies.
- `deploy/aws/cron.example`
  - Mirrors the EC2 scheduling direction from the Alpaca example README.
- Local raw-data persistence
  - The current implementation stores raw bars and news under `data/raw/`.
- Local structured persistence
  - The current implementation stores workflow and backtest results in `data/trading_workflow.db`.

## Extension Paths Left Open

These are required by the assignment, but the provided Alpaca resources do not include enough material to implement them responsibly yet:

- AWS S3 / Azure Blob Storage
  - `CloudObjectStorePlaceholder` is the extension seam for object storage.
  - Replace `LocalDataLake` in `workflow.py` once a bucket/container convention is defined.
- AWS RDS / Azure SQL Database
  - `SQLiteStructuredStore` defines the current schema and persistence behavior.
  - Promote the same schema to PostgreSQL/MySQL/Azure SQL once infrastructure details are chosen.
- AWS Lambda / Azure Functions
  - Not implemented.
  - The current hourly entry point is better suited to EC2/VM scheduling at this stage.

## Recommended Next Step

If the next phase is cloud deployment rather than strategy expansion, keep the workflow contract unchanged and swap these pieces only:

1. Replace `LocalDataLake` with an S3/Blob-backed implementation.
2. Replace `SQLiteStructuredStore` with an RDS/Azure SQL implementation.
3. Keep `TradingWorkflow.run_once()` as the stable execution unit for cron, Lambda, or Functions.

