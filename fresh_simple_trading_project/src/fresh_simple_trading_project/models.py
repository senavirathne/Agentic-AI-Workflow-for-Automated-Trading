from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class NewsArticle:
    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    published_at: str | None = None


@dataclass
class AccountState:
    cash: float
    position_qty: int = 0
    market_open: bool = True


@dataclass
class CollectedMarketData:
    """Workflow input snapshot for one symbol.

    `five_minute_bars` is the indicator-tuning window used by feature engineering.
    `hourly_bars` is the support/resistance window used by market analysis.
    """

    symbol: str
    # Last indicator window, typically 24 hours of 5-minute candles.
    five_minute_bars: pd.DataFrame
    # Last support/resistance window, typically 7 days of 1-hour candles.
    hourly_bars: pd.DataFrame
    account: AccountState


@dataclass
class EDAResult:
    symbol: str
    missing_values: int
    anomaly_count: int
    candle_return_mean: float
    candle_volatility: float
    latest_close: float


@dataclass
class AnalysisResult:
    symbol: str
    timestamp: pd.Timestamp
    latest_price: float
    trend: str
    bullish: bool
    entry_setup: bool
    exit_setup: bool
    confidence: float
    notes: list[str] = field(default_factory=list)
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    support_regions: list[tuple[float, float]] = field(default_factory=list)
    resistance_regions: list[tuple[float, float]] = field(default_factory=list)
    support_region_strengths: list[int] = field(default_factory=list)
    resistance_region_strengths: list[int] = field(default_factory=list)
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    nearest_support_region: tuple[float, float] | None = None
    nearest_resistance_region: tuple[float, float] | None = None
    nearest_support_region_strength: int = 0
    nearest_resistance_region_strength: int = 0
    distance_to_support_pct: float | None = None
    distance_to_resistance_pct: float | None = None
    llm_summary: str | None = None


@dataclass
class RetrievalResult:
    symbol: str
    articles: list[NewsArticle]
    headline_summary: list[str]
    sentiment_score: float
    risk_flags: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    summary_note: str | None = None


@dataclass
class RiskResult:
    symbol: str
    risk_score: float
    can_enter: bool
    recommended_qty: int
    warnings: list[str] = field(default_factory=list)
    summary_note: str | None = None


@dataclass
class Decision:
    symbol: str
    action: Action
    quantity: int
    confidence: float
    rationale: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    executed: bool
    order_id: str | None
    status: str


@dataclass
class WorkflowResult:
    symbol: str
    five_minute_bars: pd.DataFrame
    hourly_bars: pd.DataFrame
    eda: EDAResult
    analysis: AnalysisResult
    retrieval: RetrievalResult
    risk: RiskResult
    decision: Decision
    execution: ExecutionResult


@dataclass
class BacktestTrade:
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
    initial_cash: float
    ending_cash: float
    total_return_pct: float
    benchmark_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trade_count: int
    win_rate: float
    signal_accuracy: float
    trades: list[BacktestTrade]
    equity_curve: pd.DataFrame
