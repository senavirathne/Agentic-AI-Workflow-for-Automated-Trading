from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from alpaca.trading.enums import OrderSide

from .config import TradingConfig
from .indicators import generate_signal_frame
from .models import Action, AnalysisResult, Decision, MarketContext, RetrievalResult, RiskPlan


NEGATIVE_KEYWORDS = {
    "downgrade",
    "lawsuit",
    "probe",
    "fraud",
    "miss",
    "recall",
    "slump",
    "layoff",
    "bankruptcy",
    "offering",
}
POSITIVE_KEYWORDS = {
    "upgrade",
    "beat",
    "surge",
    "growth",
    "profit",
    "record",
    "approval",
    "partnership",
    "buyback",
}


@dataclass
class MarketAnalysisAgent:
    config: TradingConfig

    def analyze(self, context: MarketContext) -> AnalysisResult:
        signal_frame = generate_signal_frame(context.main_bars, context.trend_bars, self.config)
        latest = signal_frame.iloc[-1]
        notes = []
        if bool(latest["buy_signal"]):
            notes.append("Entry signal is active from the Alpaca RSI/MACD window logic.")
        if bool(latest["sell_signal"]):
            notes.append("Exit signal is active from the Alpaca RSI/MACD window logic.")
        if bool(latest["in_uptrend"]):
            notes.append("Trend filter is bullish on the higher timeframe.")

        return AnalysisResult(
            symbol=context.symbol,
            latest_timestamp=signal_frame.index[-1],
            rsi_now=float(latest["rsi"]),
            macd_now=float(latest["macd"]),
            signal_now=float(latest["macd_signal"]),
            ma_fast=float(latest["trend_ma_fast"]) if latest["trend_ma_fast"] == latest["trend_ma_fast"] else None,
            ma_mid=float(latest["trend_ma_mid"]) if latest["trend_ma_mid"] == latest["trend_ma_mid"] else None,
            ma_slow=float(latest["trend_ma_slow"]) if latest["trend_ma_slow"] == latest["trend_ma_slow"] else None,
            in_uptrend=bool(latest["in_uptrend"]),
            entry_setup=bool(latest["buy_signal"]),
            exit_setup=bool(latest["sell_signal"]),
            signal_frame=signal_frame,
            notes=notes,
        )


@dataclass
class InformationRetrievalAgent:
    def retrieve(self, symbol: str, articles: list[dict]) -> RetrievalResult:
        positive_hits = 0
        negative_hits = 0
        risk_flags: list[str] = []
        summary: list[str] = []

        for article in articles[:5]:
            headline = article.get("headline", "")
            body = f"{headline} {article.get('summary', '')} {article.get('content', '')}".lower()
            positive_hits += _count_hits(body, POSITIVE_KEYWORDS)
            negative_hits += _count_hits(body, NEGATIVE_KEYWORDS)
            summary.append(headline)

        if negative_hits > positive_hits:
            risk_flags.append("Recent Alpaca news flow is net negative by simple keyword scan.")
        if any(_contains_any(item.lower(), {"downgrade", "lawsuit", "fraud", "bankruptcy"}) for item in summary):
            risk_flags.append("At least one headline contains a high-risk keyword.")

        return RetrievalResult(
            symbol=symbol,
            articles=articles,
            headline_summary=summary,
            positive_hits=positive_hits,
            negative_hits=negative_hits,
            risk_flags=risk_flags,
        )


@dataclass
class RiskManagementAgent:
    config: TradingConfig

    def evaluate(self, context: MarketContext) -> RiskPlan:
        notes = []
        max_notional = context.buying_power * self.config.buy_power_limit
        recommended_qty = int(max_notional / context.latest_price) if context.latest_price > 0 else 0

        if not context.market_open:
            notes.append("Market is closed, so no new entry can be opened.")
        if recommended_qty <= 0:
            notes.append("Buying power limit is too small for a new position.")

        return RiskPlan(
            symbol=context.symbol,
            max_notional=max_notional,
            recommended_qty=recommended_qty,
            can_enter=context.market_open and recommended_qty > 0,
            notes=notes,
        )


@dataclass
class DecisionAgent:
    def decide(
        self,
        context: MarketContext,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskPlan,
    ) -> Decision:
        rationale = [
            f"RSI={analysis.rsi_now:.2f}, MACD={analysis.macd_now:.4f}, signal={analysis.signal_now:.4f}.",
            f"Trend filter bullish={analysis.in_uptrend}.",
            f"Recent news articles retrieved={len(retrieval.articles)}.",
        ]
        if retrieval.risk_flags:
            rationale.append("News module raised risk flags, but order rules remain technical-only in this version.")
        if not context.market_open:
            rationale.append("Market is closed, so the workflow will hold.")
            return Decision(
                symbol=context.symbol,
                action=Action.HOLD,
                quantity=0,
                confidence=0.9,
                rationale=rationale,
                metadata={"news_risk_flags": retrieval.risk_flags},
            )

        if context.position_open:
            if analysis.exit_setup:
                rationale.append("Open position plus exit setup -> sell.")
                return Decision(
                    symbol=context.symbol,
                    action=Action.SELL,
                    quantity=context.current_qty,
                    confidence=0.8,
                    rationale=rationale,
                    metadata={"news_risk_flags": retrieval.risk_flags},
                )
            rationale.append("Open position without exit setup -> hold.")
            return Decision(
                symbol=context.symbol,
                action=Action.HOLD,
                quantity=0,
                confidence=0.55,
                rationale=rationale,
                metadata={"news_risk_flags": retrieval.risk_flags},
            )

        if analysis.entry_setup and risk.can_enter:
            rationale.append("Entry setup and risk gate passed -> buy.")
            return Decision(
                symbol=context.symbol,
                action=Action.BUY,
                quantity=risk.recommended_qty,
                confidence=0.75,
                rationale=rationale,
                metadata={"news_risk_flags": retrieval.risk_flags, "max_notional": risk.max_notional},
            )

        rationale.append("No actionable setup -> hold.")
        return Decision(
            symbol=context.symbol,
            action=Action.HOLD,
            quantity=0,
            confidence=0.5,
            rationale=rationale,
            metadata={"news_risk_flags": retrieval.risk_flags},
        )


@dataclass
class ExecutionAgent:
    service: Any

    def execute(self, decision: Decision, execute_orders: bool) -> str | None:
        if not execute_orders or decision.action == Action.HOLD or decision.quantity <= 0:
            return None

        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
        order = self.service.submit_market_order(
            symbol=decision.symbol,
            qty=decision.quantity,
            side=side,
        )
        return str(order.id)


def _count_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(keyword in text for keyword in keywords)


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)
