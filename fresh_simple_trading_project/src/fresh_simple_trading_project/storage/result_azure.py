"""Azure SQL result-store adapter."""

from __future__ import annotations

from .result_sqlite import SQLiteResultStore


class AzureSQLResultStore(SQLiteResultStore):
    """Azure SQL-backed result store implementation."""

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)
