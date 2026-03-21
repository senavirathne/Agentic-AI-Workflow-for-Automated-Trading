from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketContext:
    symbol: str
    main_bars: pd.DataFrame
    trend_bars: pd.DataFrame
    latest_price: float
    market_open: bool
    buying_power: float
    position_open: bool
    current_qty: int
    avg_entry_price: float | None = None
    news: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    symbol: str
    latest_timestamp: pd.Timestamp
    rsi_now: float
    macd_now: float
    signal_now: float
    ma_fast: float | None
    ma_mid: float | None
    ma_slow: float | None
    in_uptrend: bool
    entry_setup: bool
    exit_setup: bool
    signal_frame: pd.DataFrame
    notes: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    symbol: str
    articles: list[dict[str, Any]]
    headline_summary: list[str]
    positive_hits: int
    negative_hits: int
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class RiskPlan:
    symbol: str
    max_notional: float
    recommended_qty: int
    can_enter: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Decision:
    symbol: str
    action: Action
    quantity: int
    confidence: float
    rationale: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    context: MarketContext
    analysis: AnalysisResult
    retrieval: RetrievalResult
    risk: RiskPlan
    decision: Decision
    order_id: str | None = None


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    qty: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    reason: str


@dataclass
class BacktestSummary:
    symbol: str
    initial_capital: float
    ending_capital: float
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    signal_accuracy: float
    trade_count: int
    win_rate: float
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame

