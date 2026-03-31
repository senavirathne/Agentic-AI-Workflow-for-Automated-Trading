from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

import pandas as pd

from .models import BacktestSummary, NewsArticle, WorkflowResult
from .utils import timestamp_slug


class RawStore(Protocol):
    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> Path:
        ...

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> Path:
        ...


class LocalRawStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bars(self, symbol: str, timeframe: str, bars: pd.DataFrame) -> Path:
        target = self.root / "bars" / f"{symbol}_{timeframe}_{timestamp_slug()}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        bars.to_csv(target)
        return target

    def save_news(self, symbol: str, articles: list[NewsArticle]) -> Path:
        target = self.root / "news" / f"{symbol}_{timestamp_slug()}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [article.__dict__ for article in articles]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target


class InMemoryResultStore:
    def __init__(self) -> None:
        self.workflow_runs: list[WorkflowResult] = []
        self.backtest_runs: list[BacktestSummary] = []
        self.last_processed: dict[str, pd.Timestamp] = {}

    def save_workflow_run(self, result: WorkflowResult) -> None:
        self.workflow_runs.append(result)

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        self.backtest_runs.append(summary)

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        self.last_processed[symbol] = pd.Timestamp(timestamp)

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        return self.last_processed.get(symbol)


class SQLiteResultStore:
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
                    latest_price REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    metadata TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    initial_cash REAL NOT NULL,
                    ending_cash REAL NOT NULL,
                    total_return_pct REAL NOT NULL,
                    benchmark_return_pct REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    signal_accuracy REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_state (
                    symbol TEXT PRIMARY KEY,
                    last_processed_at TEXT NOT NULL
                );
                """
            )

    def save_workflow_run(self, result: WorkflowResult) -> None:
        metadata = {
            "analysis_interval": "5min",
            "loop_interval": "1h",
            "analysis_notes": result.analysis.notes,
            "technical_agent_summary": result.analysis.llm_summary,
            "news_agent_summary": result.retrieval.summary_note,
            "risk_warnings": result.risk.warnings,
            "risk_agent_summary": result.risk.summary_note,
            "headlines": result.retrieval.headline_summary,
            "decision_rationale": result.decision.rationale,
            "execution_status": result.execution.status,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    created_at, symbol, action, quantity, confidence, latest_price,
                    sentiment_score, risk_score, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_slug(),
                    result.symbol,
                    result.decision.action.value,
                    result.decision.quantity,
                    result.decision.confidence,
                    result.analysis.latest_price,
                    result.retrieval.sentiment_score,
                    result.risk.risk_score,
                    json.dumps(metadata),
                ),
            )

    def save_backtest_summary(self, summary: BacktestSummary) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO backtest_runs (
                    created_at, symbol, initial_cash, ending_cash, total_return_pct,
                    benchmark_return_pct, max_drawdown_pct, sharpe_ratio,
                    trade_count, win_rate, signal_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp_slug(),
                    summary.symbol,
                    summary.initial_cash,
                    summary.ending_cash,
                    summary.total_return_pct,
                    summary.benchmark_return_pct,
                    summary.max_drawdown_pct,
                    summary.sharpe_ratio,
                    summary.trade_count,
                    summary.win_rate,
                    summary.signal_accuracy,
                ),
            )

    def save_last_processed(self, symbol: str, timestamp: pd.Timestamp) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO workflow_state (symbol, last_processed_at)
                VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET last_processed_at = excluded.last_processed_at
                """,
                (symbol, pd.Timestamp(timestamp).isoformat()),
            )

    def load_last_processed(self, symbol: str) -> pd.Timestamp | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT last_processed_at
                FROM workflow_state
                WHERE symbol = ?
                """,
                (symbol,),
            ).fetchone()
        if row is None or not row[0]:
            return None
        return pd.Timestamp(row[0])
