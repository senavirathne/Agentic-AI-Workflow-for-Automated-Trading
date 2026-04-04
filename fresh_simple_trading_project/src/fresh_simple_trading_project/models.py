"""Datamodels shared across the trading workflow package."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum

import pandas as pd


class Action(str, Enum):
    """Trading actions that can be emitted by the workflow."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class NewsArticle:
    """Normalized news article metadata used by retrieval and reasoning stages."""

    headline: str
    summary: str = ""
    source: str = ""
    url: str = ""
    published_at: str | None = None
    provider: str = ""
    primary_ticker: str | None = None
    primary_ticker_relevance: float | None = None


@dataclass
class AccountState:
    """Portfolio and market-open state visible to the workflow."""

    cash: float
    position_qty: int = 0
    market_open: bool = True
    avg_entry_price: float | None = None
    realized_profit: float = 0.0
    trade_count: int = 0


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
class IndicatorHourChunk:
    """One hourly bucket of aligned 5-minute indicator rows."""

    slot_start: str
    slot_end: str
    rows: list[dict[str, object]]


@dataclass
class AlphaVantageIndicatorSnapshot:
    """Daily Alpha Vantage indicator table plus hourly chunk views."""

    symbol: str
    interval: str
    trading_day: str
    latest_timestamp: str
    indicator_columns: list[str]
    rows: list[dict[str, object]]
    hourly_chunks: list[IndicatorHourChunk] = field(default_factory=list)
    latest_hour_chunk: IndicatorHourChunk | None = None


@dataclass
class EDAResult:
    """Exploratory statistics calculated for the recent 5-minute window."""

    symbol: str
    missing_values: int
    anomaly_count: int
    candle_return_mean: float
    candle_volatility: float
    latest_close: float


@dataclass
class PriceLevelContext:
    """Support, resistance, and local-region context derived from price history."""

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


@dataclass
class AnalysisResult:
    """Technical market-analysis output used by downstream stages."""

    symbol: str
    timestamp: pd.Timestamp
    latest_price: float
    trend: str
    bullish: bool
    entry_setup: bool
    exit_setup: bool
    confidence: float
    price_levels: PriceLevelContext = field(default_factory=PriceLevelContext)
    notes: list[str] = field(default_factory=list)
    llm_summary: str | None = None
    current_price: float | None = None
    market_data_delay_minutes: int = 0
    indicator_source: str = "manually computed indicators from 5-minute bar data"
    latest_volume: float = 0.0
    previous_forecast_summary: str | None = None
    support_levels: InitVar[list[float] | None] = None
    resistance_levels: InitVar[list[float] | None] = None
    support_regions: InitVar[list[tuple[float, float]] | None] = None
    resistance_regions: InitVar[list[tuple[float, float]] | None] = None
    support_region_strengths: InitVar[list[int] | None] = None
    resistance_region_strengths: InitVar[list[int] | None] = None
    nearest_support: InitVar[float | None] = None
    nearest_resistance: InitVar[float | None] = None
    nearest_support_region: InitVar[tuple[float, float] | None] = None
    nearest_resistance_region: InitVar[tuple[float, float] | None] = None
    nearest_support_region_strength: InitVar[int | None] = None
    nearest_resistance_region_strength: InitVar[int | None] = None
    distance_to_support_pct: InitVar[float | None] = None
    distance_to_resistance_pct: InitVar[float | None] = None

    def __post_init__(
        self,
        support_levels: list[float] | None,
        resistance_levels: list[float] | None,
        support_regions: list[tuple[float, float]] | None,
        resistance_regions: list[tuple[float, float]] | None,
        support_region_strengths: list[int] | None,
        resistance_region_strengths: list[int] | None,
        nearest_support: float | None,
        nearest_resistance: float | None,
        nearest_support_region: tuple[float, float] | None,
        nearest_resistance_region: tuple[float, float] | None,
        nearest_support_region_strength: int | None,
        nearest_resistance_region_strength: int | None,
        distance_to_support_pct: float | None,
        distance_to_resistance_pct: float | None,
    ) -> None:
        if not any(
            value is not None
            for value in (
                support_levels,
                resistance_levels,
                support_regions,
                resistance_regions,
                support_region_strengths,
                resistance_region_strengths,
                nearest_support,
                nearest_resistance,
                nearest_support_region,
                nearest_resistance_region,
                nearest_support_region_strength,
                nearest_resistance_region_strength,
                distance_to_support_pct,
                distance_to_resistance_pct,
            )
        ):
            return

        base = self.price_levels
        self.price_levels = PriceLevelContext(
            support_levels=list(base.support_levels if support_levels is None else support_levels),
            resistance_levels=list(base.resistance_levels if resistance_levels is None else resistance_levels),
            support_regions=list(base.support_regions if support_regions is None else support_regions),
            resistance_regions=list(base.resistance_regions if resistance_regions is None else resistance_regions),
            support_region_strengths=list(
                base.support_region_strengths if support_region_strengths is None else support_region_strengths
            ),
            resistance_region_strengths=list(
                base.resistance_region_strengths
                if resistance_region_strengths is None
                else resistance_region_strengths
            ),
            nearest_support=base.nearest_support if nearest_support is None else nearest_support,
            nearest_resistance=base.nearest_resistance if nearest_resistance is None else nearest_resistance,
            nearest_support_region=(
                base.nearest_support_region if nearest_support_region is None else nearest_support_region
            ),
            nearest_resistance_region=(
                base.nearest_resistance_region if nearest_resistance_region is None else nearest_resistance_region
            ),
            nearest_support_region_strength=(
                base.nearest_support_region_strength
                if nearest_support_region_strength is None
                else nearest_support_region_strength
            ),
            nearest_resistance_region_strength=(
                base.nearest_resistance_region_strength
                if nearest_resistance_region_strength is None
                else nearest_resistance_region_strength
            ),
            distance_to_support_pct=(
                base.distance_to_support_pct if distance_to_support_pct is None else distance_to_support_pct
            ),
            distance_to_resistance_pct=(
                base.distance_to_resistance_pct
                if distance_to_resistance_pct is None
                else distance_to_resistance_pct
            ),
        )


def _price_level_property(name: str) -> property:
    def getter(self: AnalysisResult):
        return getattr(self.price_levels, name)

    def setter(self: AnalysisResult, value) -> None:
        setattr(self.price_levels, name, value)

    return property(getter, setter)


for _legacy_price_level_name in (
    "support_levels",
    "resistance_levels",
    "support_regions",
    "resistance_regions",
    "support_region_strengths",
    "resistance_region_strengths",
    "nearest_support",
    "nearest_resistance",
    "nearest_support_region",
    "nearest_resistance_region",
    "nearest_support_region_strength",
    "nearest_resistance_region_strength",
    "distance_to_support_pct",
    "distance_to_resistance_pct",
):
    setattr(AnalysisResult, _legacy_price_level_name, _price_level_property(_legacy_price_level_name))


@dataclass
class RetrievalResult:
    """News retrieval output consumed by risk and decision modules."""

    symbol: str
    articles: list[NewsArticle]
    headline_summary: list[str]
    sentiment_score: float = 0.0
    critical_news: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    summary_note: str | None = None


@dataclass
class RiskResult:
    """Risk scoring, sizing, and protective-order guidance."""

    symbol: str
    risk_score: float
    can_enter: bool
    recommended_qty: int
    position_in_profit: bool = False
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    warnings: list[str] = field(default_factory=list)
    summary_note: str | None = None


@dataclass
class Decision:
    """Final workflow decision before execution."""

    symbol: str
    action: Action
    quantity: int
    confidence: float
    rationale: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Execution outcome returned by the execution module."""

    executed: bool
    order_id: str | None
    status: str
    protective_order_ids: list[str] = field(default_factory=list)


@dataclass
class ForecastSnapshot:
    """Forward-looking HOLD forecast generated by the hold agent."""

    symbol: str
    generated_at: pd.Timestamp
    valid_until: pd.Timestamp
    reference_price: float
    reference_volume: float
    trend_bias: str = "neutral"
    continuation_price_target: float | None = None
    continuation_volume_target: float | None = None
    reversal_price_target: float | None = None
    reversal_volume_target: float | None = None
    continuation_signals: list[str] = field(default_factory=list)
    reversal_signals: list[str] = field(default_factory=list)
    summary: str | None = None
    confidence: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Current trading performance metrics for the active symbol."""

    symbol: str
    as_of: pd.Timestamp
    position_qty: int
    trade_count: int
    market_price: float
    avg_entry_price: float | None = None
    realized_profit: float = 0.0
    unrealized_profit: float = 0.0
    current_profit: float = 0.0


@dataclass
class WorkflowResult:
    """Complete output of a single workflow iteration."""

    symbol: str
    five_minute_bars: pd.DataFrame
    hourly_bars: pd.DataFrame
    eda: EDAResult
    analysis: AnalysisResult
    retrieval: RetrievalResult
    risk: RiskResult
    decision: Decision
    execution: ExecutionResult
    alpha_vantage_indicator_snapshot: AlphaVantageIndicatorSnapshot | None = None
    account: AccountState | None = None
    performance: PerformanceSnapshot | None = None
    previous_forecast: ForecastSnapshot | None = None
    hold_forecast: ForecastSnapshot | None = None


@dataclass
class BacktestTrade:
    """A completed simulated trade captured during backtesting."""

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
    """Aggregate performance summary for a backtest run."""

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
    net_profit: float = 0.0
