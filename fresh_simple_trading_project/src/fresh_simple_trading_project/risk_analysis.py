"""Risk scoring and trade-sizing logic for the trading workflow."""

from __future__ import annotations

from dataclasses import dataclass

from .agents import RiskReviewAgent
from .config import TradingConfig
from .models import AccountState, AnalysisResult, EDAResult, ForecastSnapshot, RetrievalResult, RiskResult


@dataclass
class RiskAnalysisModule:
    """Convert market, news, and account context into a risk assessment."""

    config: TradingConfig
    risk_agent: RiskReviewAgent | None = None

    def assess(
        self,
        symbol: str,
        account: AccountState,
        analysis: AnalysisResult,
        eda: EDAResult,
        retrieval: RetrievalResult,
        previous_forecast: ForecastSnapshot | None = None,
    ) -> RiskResult:
        risk_score = 0.2
        warnings: list[str] = []
        if eda.candle_volatility > self.config.risk_volatility_cutoff:
            risk_score += 0.3
            warnings.append("5-minute candle volatility is above the configured cutoff.")
        if eda.anomaly_count > 0:
            risk_score += 0.15
            warnings.append("Recent data contains anomaly signals.")
        if retrieval.risk_flags:
            risk_score += 0.15
            warnings.extend(retrieval.risk_flags)
        if not account.market_open:
            warnings.append("Market is closed.")

        risk_score = max(0.0, min(1.0, round(risk_score, 2)))
        notional_budget = account.cash * self.config.max_position_pct * max(0.25, 1.0 - risk_score)
        recommended_qty = int(notional_budget / analysis.latest_price) if analysis.latest_price > 0 else 0
        can_enter = account.market_open and recommended_qty > 0 and risk_score < 0.85
        market_price = float(analysis.current_price or analysis.latest_price)
        position_in_profit = bool(
            account.position_qty > 0
            and account.avg_entry_price is not None
            and market_price > account.avg_entry_price
        )
        stop_loss_price = _recommended_stop_loss(analysis, eda, market_price)
        take_profit_price = _recommended_take_profit(analysis, eda, market_price) if position_in_profit else None
        if stop_loss_price is not None:
            warnings.append(f"Protective stop loss level={stop_loss_price:.2f}.")
        if take_profit_price is not None:
            warnings.append(f"Active take profit level={take_profit_price:.2f}.")
        summary_note = self._summarize_with_llm(
            symbol=symbol,
            risk_score=risk_score,
            recommended_qty=recommended_qty,
            can_enter=can_enter,
            warnings=warnings,
            analysis=analysis,
            eda=eda,
            retrieval=retrieval,
            previous_forecast=previous_forecast,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_in_profit=position_in_profit,
        )
        return RiskResult(
            symbol=symbol,
            risk_score=risk_score,
            can_enter=can_enter,
            recommended_qty=recommended_qty,
            position_in_profit=position_in_profit,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            warnings=warnings,
            summary_note=summary_note,
        )

    def _summarize_with_llm(
        self,
        *,
        symbol: str,
        risk_score: float,
        recommended_qty: int,
        can_enter: bool,
        warnings: list[str],
        analysis: AnalysisResult,
        eda: EDAResult,
        retrieval: RetrievalResult,
        previous_forecast: ForecastSnapshot | None,
        stop_loss_price: float | None,
        take_profit_price: float | None,
        position_in_profit: bool,
    ) -> str | None:
        if self.risk_agent is None:
            return None

        return self.risk_agent.summarize(
            symbol=symbol,
            risk_score=risk_score,
            recommended_qty=recommended_qty,
            can_enter=can_enter,
            warnings=warnings,
            analysis=analysis,
            retrieval=retrieval,
            eda=eda,
            previous_forecast=previous_forecast,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            position_in_profit=position_in_profit,
        )


def _recommended_stop_loss(
    analysis: AnalysisResult,
    eda: EDAResult,
    market_price: float,
) -> float | None:
    price_levels = analysis.price_levels
    candidates: list[float] = []
    if price_levels.nearest_support_region is not None:
        candidates.append(price_levels.nearest_support_region[0] * 0.9975)
    if price_levels.nearest_support is not None:
        candidates.append(price_levels.nearest_support * 0.9975)
    volatility_floor = market_price * (1 - max(0.01, eda.candle_volatility * 1.5))
    candidates.append(volatility_floor)
    eligible = [round(candidate, 2) for candidate in candidates if candidate < market_price]
    if not eligible:
        return None
    return max(eligible)


def _recommended_take_profit(
    analysis: AnalysisResult,
    eda: EDAResult,
    market_price: float,
) -> float | None:
    price_levels = analysis.price_levels
    candidates: list[float] = []
    if price_levels.nearest_resistance_region is not None:
        candidates.append(price_levels.nearest_resistance_region[1])
    if price_levels.nearest_resistance is not None:
        candidates.append(price_levels.nearest_resistance)
    candidates.append(market_price * (1 + max(0.01, eda.candle_volatility * 2)))
    eligible = [round(candidate, 2) for candidate in candidates if candidate > market_price]
    if not eligible:
        return None
    return min(eligible)
