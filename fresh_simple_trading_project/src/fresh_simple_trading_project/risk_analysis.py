from __future__ import annotations

from dataclasses import dataclass

from .agents import RiskReviewAgent
from .config import TradingConfig
from .llm import TextGenerationClient
from .models import AccountState, AnalysisResult, EDAResult, RetrievalResult, RiskResult


@dataclass
class RiskAnalysisModule:
    config: TradingConfig
    llm_client: TextGenerationClient | None = None
    risk_agent: RiskReviewAgent | None = None

    def __post_init__(self) -> None:
        if self.risk_agent is None and self.llm_client is not None:
            self.risk_agent = RiskReviewAgent(llm_client=self.llm_client)

    def assess(
        self,
        symbol: str,
        account: AccountState,
        analysis: AnalysisResult,
        eda: EDAResult,
        retrieval: RetrievalResult,
    ) -> RiskResult:
        risk_score = 0.2
        warnings: list[str] = []
        if eda.candle_volatility > self.config.risk_volatility_cutoff:
            risk_score += 0.3
            warnings.append("5-minute candle volatility is above the configured cutoff.")
        if eda.anomaly_count > 0:
            risk_score += 0.15
            warnings.append("Recent data contains anomaly signals.")
        if retrieval.sentiment_score < 0:
            risk_score += 0.15
            warnings.append("Recent news sentiment is negative.")
        if retrieval.risk_flags:
            risk_score += 0.1
            warnings.extend(retrieval.risk_flags)
        if not account.market_open:
            warnings.append("Market is closed.")

        risk_score = max(0.0, min(1.0, round(risk_score, 2)))
        notional_budget = account.cash * self.config.max_position_pct * max(0.25, 1.0 - risk_score)
        recommended_qty = int(notional_budget / analysis.latest_price) if analysis.latest_price > 0 else 0
        can_enter = account.market_open and recommended_qty > 0 and risk_score < 0.85
        summary_note = self._summarize_with_llm(
            symbol=symbol,
            risk_score=risk_score,
            recommended_qty=recommended_qty,
            can_enter=can_enter,
            warnings=warnings,
            analysis=analysis,
            eda=eda,
            retrieval=retrieval,
        )
        return RiskResult(
            symbol=symbol,
            risk_score=risk_score,
            can_enter=can_enter,
            recommended_qty=recommended_qty,
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
        )
