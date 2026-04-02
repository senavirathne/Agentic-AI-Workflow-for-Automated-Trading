from __future__ import annotations

from dataclasses import dataclass

from .agents import DecisionCoordinatorAgent, DecisionReview
from .llm import TextGenerationClient
from .models import AccountState, Action, AnalysisResult, Decision, RetrievalResult, RiskResult


@dataclass
class DecisionEngine:
    resistance_buffer_pct: float = 1.0
    support_break_pct: float = 0.5
    llm_client: TextGenerationClient | None = None
    decision_agent: DecisionCoordinatorAgent | None = None

    def __post_init__(self) -> None:
        if self.decision_agent is None and self.llm_client is not None:
            self.decision_agent = DecisionCoordinatorAgent(llm_client=self.llm_client)

    def decide(
        self,
        account: AccountState,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskResult,
    ) -> Decision:
        rule_based_decision = self._rule_based_decision(account, analysis, retrieval, risk)
        return self._apply_llm_review(rule_based_decision, account, analysis, retrieval, risk)

    def _rule_based_decision(
        self,
        account: AccountState,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskResult,
    ) -> Decision:
        rationale = [
            f"Trend={analysis.trend}",
            f"Critical news items={len(retrieval.critical_news)}",
            f"Risk score={risk.risk_score:.2f}",
        ]
        if analysis.llm_summary:
            rationale.append(f"Market analysis module: {analysis.llm_summary}")
        if retrieval.summary_note:
            rationale.append(f"Information retrieval module: {retrieval.summary_note}")
        if retrieval.critical_news:
            rationale.append(f"Top critical news: {retrieval.critical_news[0]}")
        if risk.summary_note:
            rationale.append(f"Risk analysis module: {risk.summary_note}")
        confidence = max(0.0, min(1.0, round((analysis.confidence + (1.0 - risk.risk_score)) / 2, 2)))

        if not account.market_open:
            rationale.append("Market closed, so the engine holds.")
            return Decision(
                symbol=analysis.symbol,
                action=Action.HOLD,
                quantity=0,
                confidence=confidence,
                rationale=rationale,
            )

        in_support_region = _price_in_region(analysis.latest_price, analysis.nearest_support_region)
        in_resistance_region = _price_in_region(analysis.latest_price, analysis.nearest_resistance_region)
        broke_support_region = _below_region(analysis.latest_price, analysis.nearest_support_region)
        confirmed_support_region = analysis.nearest_support_region_strength >= 1
        confirmed_resistance_region = analysis.nearest_resistance_region_strength >= 2
        resistance_region_ceiling = (
            analysis.nearest_resistance_region is None
            or analysis.latest_price <= _region_midpoint(analysis.nearest_resistance_region)
        )
        resistance_cap_active = (
            analysis.nearest_resistance is None or analysis.latest_price <= analysis.nearest_resistance
        )
        broke_support = (
            analysis.distance_to_support_pct is not None
            and analysis.distance_to_support_pct <= -self.support_break_pct
        )
        near_resistance = (
            in_resistance_region and confirmed_resistance_region and resistance_cap_active and resistance_region_ceiling
        ) or (
            analysis.distance_to_resistance_pct is not None
            and 0.0 <= analysis.distance_to_resistance_pct <= self.resistance_buffer_pct
            and confirmed_resistance_region
            and resistance_cap_active
        )

        if account.position_qty > 0 and (analysis.exit_setup or broke_support or (broke_support_region and confirmed_support_region)):
            if broke_support_region and confirmed_support_region and analysis.nearest_support_region is not None:
                rationale.append(
                    "Price broke below the nearest support region "
                    f"{_format_region(analysis.nearest_support_region)}."
                )
            elif broke_support:
                rationale.append("Price broke below the nearest support line.")
            else:
                rationale.append("Open position and exit setup detected.")
            return Decision(
                symbol=analysis.symbol,
                action=Action.SELL,
                quantity=account.position_qty,
                confidence=confidence,
                rationale=rationale,
            )

        if (
            account.position_qty == 0
            and analysis.entry_setup
            and risk.can_enter
            and not near_resistance
        ):
            if in_support_region and confirmed_support_region and analysis.nearest_support_region is not None:
                rationale.append(
                    "Price is trading inside the local support region "
                    f"{_format_region(analysis.nearest_support_region)}."
                )
            elif analysis.nearest_support is not None:
                rationale.append(f"Major support={analysis.nearest_support:.2f}.")
            rationale.append("Entry setup, news review completed, and risk approval.")
            return Decision(
                symbol=analysis.symbol,
                action=Action.BUY,
                quantity=risk.recommended_qty,
                confidence=confidence,
                rationale=rationale,
            )

        if near_resistance:
            if in_resistance_region and analysis.nearest_resistance_region is not None:
                rationale.append(
                    "Price is trading inside the local resistance region "
                    f"{_format_region(analysis.nearest_resistance_region)}, so the engine waits."
                )
            elif analysis.nearest_resistance is not None:
                rationale.append(
                    f"Price is too close to major resistance at {analysis.nearest_resistance:.2f}, so the engine waits."
                )
        rationale.append("Conditions for a new trade are not strong enough.")
        return Decision(
            symbol=analysis.symbol,
            action=Action.HOLD,
            quantity=0,
            confidence=confidence,
            rationale=rationale,
        )

    def _apply_llm_review(
        self,
        rule_based_decision: Decision,
        account: AccountState,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskResult,
    ) -> Decision:
        if self.decision_agent is None:
            return rule_based_decision

        review = self.decision_agent.review(
            rule_based_action=rule_based_decision.action.value,
            rule_based_quantity=rule_based_decision.quantity,
            account=account,
            analysis=analysis,
            retrieval=retrieval,
            risk=risk,
        )
        if review.action is None and review.override is None and review.note is None:
            return rule_based_decision

        rationale = list(rule_based_decision.rationale)
        if review.note:
            rationale.append(f"LLM review: {review.note}")

        decision = self._decision_from_review(
            review=review,
            rule_based_decision=rule_based_decision,
            account=account,
            risk=risk,
            rationale=rationale,
        )
        if decision is not None:
            return decision
        return Decision(
            symbol=rule_based_decision.symbol,
            action=rule_based_decision.action,
            quantity=rule_based_decision.quantity,
            confidence=rule_based_decision.confidence,
            rationale=rationale,
        )

    def _decision_from_review(
        self,
        *,
        review: DecisionReview,
        rule_based_decision: Decision,
        account: AccountState,
        risk: RiskResult,
        rationale: list[str],
    ) -> Decision | None:
        if review.action is None:
            if review.override == "HOLD" and rule_based_decision.action == Action.BUY:
                rationale.append("LLM review vetoed the new entry.")
                return Decision(
                    symbol=rule_based_decision.symbol,
                    action=Action.HOLD,
                    quantity=0,
                    confidence=rule_based_decision.confidence,
                    rationale=rationale,
                )
            return Decision(
                symbol=rule_based_decision.symbol,
                action=rule_based_decision.action,
                quantity=rule_based_decision.quantity,
                confidence=rule_based_decision.confidence,
                rationale=rationale,
            )

        action = Action(review.action)
        if action == Action.HOLD:
            rationale.append("Decision agent selected HOLD after reviewing the provided context.")
            return Decision(
                symbol=rule_based_decision.symbol,
                action=Action.HOLD,
                quantity=0,
                confidence=rule_based_decision.confidence,
                rationale=rationale,
            )

        if action == Action.BUY:
            if not account.market_open or account.position_qty > 0 or not risk.can_enter or risk.recommended_qty <= 0:
                rationale.append("Decision agent BUY proposal was rejected by engine guardrails.")
                return None
            requested_qty = review.quantity or risk.recommended_qty
            quantity = max(1, min(requested_qty, risk.recommended_qty))
            rationale.append("Decision agent selected BUY after reviewing the provided context.")
            return Decision(
                symbol=rule_based_decision.symbol,
                action=Action.BUY,
                quantity=quantity,
                confidence=rule_based_decision.confidence,
                rationale=rationale,
            )

        if action == Action.SELL:
            if account.position_qty <= 0:
                rationale.append("Decision agent SELL proposal was rejected because there is no open position.")
                return None
            requested_qty = review.quantity or account.position_qty
            quantity = max(1, min(requested_qty, account.position_qty))
            rationale.append("Decision agent selected SELL after reviewing the provided context.")
            return Decision(
                symbol=rule_based_decision.symbol,
                action=Action.SELL,
                quantity=quantity,
                confidence=rule_based_decision.confidence,
                rationale=rationale,
            )

        return None


def _price_in_region(price: float, region: tuple[float, float] | None) -> bool:
    if region is None:
        return False
    return region[0] <= price <= region[1]


def _below_region(price: float, region: tuple[float, float] | None) -> bool:
    if region is None:
        return False
    return price < region[0]


def _format_region(region: tuple[float, float]) -> str:
    return f"{region[0]:.2f}-{region[1]:.2f}"


def _region_midpoint(region: tuple[float, float]) -> float:
    return (region[0] + region[1]) / 2
