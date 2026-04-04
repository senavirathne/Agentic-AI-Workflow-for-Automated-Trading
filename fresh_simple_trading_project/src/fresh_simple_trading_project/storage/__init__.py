"""Persistence adapters for raw artifacts, workflow runs, and snapshots."""

from __future__ import annotations

from .inmemory import InMemoryResultStore
from .protocols import RawStore, ResultStore, StorageRef
from .raw import AzureBlobRawStore, LocalRawStore
from .result_azure import AzureSQLResultStore
from .result_sqlite import SQLiteResultStore
from .serialization import (
    _alpha_vantage_snapshot_from_payload,
    _forecast_payload,
    _news_article_key,
    _performance_payload,
)

SQLAlchemyResultStore = SQLiteResultStore

__all__ = [
    "AzureBlobRawStore",
    "AzureSQLResultStore",
    "InMemoryResultStore",
    "LocalRawStore",
    "RawStore",
    "ResultStore",
    "SQLiteResultStore",
    "SQLAlchemyResultStore",
    "StorageRef",
    "_alpha_vantage_snapshot_from_payload",
    "_forecast_payload",
    "_news_article_key",
    "_performance_payload",
]
