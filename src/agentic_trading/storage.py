from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd

from .models import BacktestSummary, WorkflowResult
from .utils import timestamp_slug


class RawDataLake(Protocol):
    def save_bars(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> Path:
        ...

    def save_news(self, symbol: str, articles: list[dict]) -> Path:
        ...


class LocalDataLake:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bars(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> Path:
        target = self.root / "bars" / f"{symbol}_{timeframe}_{timestamp_slug()}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target)
        return target

    def save_news(self, symbol: str, articles: list[dict]) -> Path:
        target = self.root / "news" / f"{symbol}_{timestamp_slug()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(articles, handle, indent=2, default=str)
        return target


class CloudObjectStorePlaceholder:
    def __init__(self, provider: str, uri: str | None) -> None:
        self.provider = provider
        self.uri = uri

    def save_bars(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> Path:
        raise NotImplementedError(
            f"{self.provider} object storage wiring is intentionally left open. "
            "This project currently persists raw data locally and documents the "
            "S3/Azure Blob extension path."
        )

    def save_news(self, symbol: str, articles: list[dict]) -> Path:
        raise NotImplementedError(
            f"{self.provider} object storage wiring is intentionally left open. "
            "This project currently persists raw data locally and documents the "
            "S3/Azure Blob extension path."
        )


@dataclass
class AgentMemoryStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cycle_timestamp TEXT NOT NULL,
                    memory_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save_memory(self, agent_name: str, symbol: str, cycle_timestamp: str, memory: dict) -> None:
        payload = dict(memory)
        payload.setdefault("cycle_timestamp", cycle_timestamp)
        payload.setdefault("symbol", symbol)
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_memories (
                    agent_name, symbol, cycle_timestamp, memory_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    agent_name,
                    symbol,
                    cycle_timestamp,
                    json.dumps(payload, default=str),
                    created_at,
                ),
            )

    def load_latest_memory(self, agent_name: str, symbol: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT memory_json
                FROM agent_memories
                WHERE agent_name = ? AND symbol = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_name, symbol),
            ).fetchone()
        if row is None:
            return None
        return _decode_memory_json(row[0])

    def load_memory_window(self, agent_name: str, symbol: str, hours: int = 24) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_json
                FROM agent_memories
                WHERE agent_name = ? AND symbol = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (agent_name, symbol, cutoff),
            ).fetchall()
        return [memory for row in rows if (memory := _decode_memory_json(row[0])) is not None]


class SQLiteStructuredStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    order_id TEXT,
                    latest_price REAL NOT NULL,
                    in_uptrend INTEGER NOT NULL,
                    rationale TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    ending_capital REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    benchmark_return_pct REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    signal_accuracy REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    reason TEXT NOT NULL,
                    FOREIGN KEY(backtest_run_id) REFERENCES backtest_runs(id)
                );
                """
            )

    def save_workflow_run(self, result: WorkflowResult) -> None:
        payload = {
            "news_risk_flags": result.retrieval.risk_flags,
            "decision_metadata": result.decision.metadata,
            "analysis_notes": result.analysis.notes,
            "risk_notes": result.risk.notes,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    created_at, symbol, action, quantity, confidence, order_id,
                    latest_price, in_uptrend, rationale, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_slug(),
                    result.context.symbol,
                    result.decision.action.value,
                    result.decision.quantity,
                    result.decision.confidence,
                    result.order_id,
                    result.context.latest_price,
                    int(result.analysis.in_uptrend),
                    json.dumps(result.decision.rationale),
                    json.dumps(payload),
                ),
            )

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO backtest_runs (
                    created_at, symbol, initial_capital, ending_capital,
                    total_return_pct, benchmark_return_pct, max_drawdown_pct,
                    sharpe_ratio, signal_accuracy, trade_count, win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_slug(),
                    summary.symbol,
                    summary.initial_capital,
                    summary.ending_capital,
                    summary.total_return_pct,
                    summary.benchmark_return_pct,
                    summary.max_drawdown_pct,
                    summary.sharpe_ratio,
                    summary.signal_accuracy,
                    summary.trade_count,
                    summary.win_rate,
                ),
            )
            run_id = cursor.lastrowid
            for trade in summary.trades:
                connection.execute(
                    """
                    INSERT INTO backtest_trades (
                        backtest_run_id, symbol, entry_time, exit_time, qty,
                        entry_price, exit_price, pnl, pnl_pct, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        trade.symbol,
                        trade.entry_time.isoformat(),
                        trade.exit_time.isoformat(),
                        trade.qty,
                        trade.entry_price,
                        trade.exit_price,
                        trade.pnl,
                        trade.pnl_pct,
                        trade.reason,
                    ),
                )


def _decode_memory_json(payload: str) -> dict | None:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
