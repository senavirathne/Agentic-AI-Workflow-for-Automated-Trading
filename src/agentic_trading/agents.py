from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, TypedDict

from .config import LLMConfig, TradingConfig
from .indicators import generate_signal_frame
from .models import Action, AnalysisResult, Decision, MarketContext, RetrievalResult, RiskPlan, WorkflowResult

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised when langgraph is not installed.
    # LangGraph is optional at runtime so the package still works before extras are installed.
    END = "__end__"
    StateGraph = None


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
TRADING_SYSTEM_PROMPT = "You are an expert systematic trading analyst."
DECISION_REVIEW_PROMPT = (
    "Review the proposed trading decision using only the supplied context. Return JSON only with keys "
    "'action', 'quantity', 'confidence', and 'rationale'. "
    "Only choose from BUY, SELL, HOLD."
)


class TextGenerationClient(Protocol):
    def generate(self, system_prompt: str, content: str) -> str | None:
        ...


@dataclass
class DeepSeekLLMClient:
    config: LLMConfig
    _client: Any | None = field(init=False, default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled and bool(self.config.api_key)

    def generate(self, system_prompt: str, content: str) -> str | None:
        if not self.enabled:
            return None

        try:
            client = self._get_client()
            response_stream = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                stream=True,
            )
            parts: list[str] = []
            for chunk in response_stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
            result = "".join(parts).strip()
            return result or None
        except Exception as exc:  # pragma: no cover - network and SDK failures are environment dependent.
            logging.warning("LLM request failed: %s", exc)
            return None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client


@dataclass
class MarketAnalysisAgent:
    config: TradingConfig

    def analyze(self, context: MarketContext) -> AnalysisResult:
        signal_frame = generate_signal_frame(
            context.short_bars,
            context.medium_bars,
            context.long_bars,
            self.config,
        )
        latest = signal_frame.iloc[-1]
        notes = []
        if bool(latest["buy_signal"]):
            notes.append("Entry signal is active from the Alpaca RSI/MACD window logic.")
        if bool(latest["sell_signal"]):
            notes.append("Exit signal is active from the Alpaca RSI/MACD window logic.")
        if bool(latest["medium_trend_bullish"]):
            notes.append("Intermediate 2-hour trend confirmation is bullish.")
        if bool(latest["in_uptrend"]):
            notes.append("Trend filter is bullish on the daily timeframe.")

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
    max_news_items: int = 5

    def retrieve(self, symbol: str, articles: list[dict]) -> RetrievalResult:
        positive_hits = 0
        negative_hits = 0
        risk_flags: list[str] = []
        summary: list[str] = []
        selected_articles = articles[: self.max_news_items]

        for article in selected_articles:
            headline = article.get("headline", "")
            body = f"{headline} {article.get('summary', '')} {article.get('content', '')}".lower()
            positive_hits += _count_hits(body, POSITIVE_KEYWORDS)
            negative_hits += _count_hits(body, NEGATIVE_KEYWORDS)
            if headline:
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
    llm_client: TextGenerationClient | None = None

    def decide(
        self,
        context: MarketContext,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskPlan,
    ) -> Decision:
        rule_based_decision = self._rule_based_decision(context, analysis, retrieval, risk)
        reviewed_decision = self._review_with_llm(context, analysis, retrieval, risk, rule_based_decision)
        return reviewed_decision or rule_based_decision

    def _rule_based_decision(
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
                metadata={"news_risk_flags": retrieval.risk_flags, "decision_source": "rules"},
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
                    metadata={"news_risk_flags": retrieval.risk_flags, "decision_source": "rules"},
                )
            rationale.append("Open position without exit setup -> hold.")
            return Decision(
                symbol=context.symbol,
                action=Action.HOLD,
                quantity=0,
                confidence=0.55,
                rationale=rationale,
                metadata={"news_risk_flags": retrieval.risk_flags, "decision_source": "rules"},
            )

        if analysis.entry_setup and risk.can_enter:
            rationale.append("Entry setup and risk gate passed -> buy.")
            return Decision(
                symbol=context.symbol,
                action=Action.BUY,
                quantity=risk.recommended_qty,
                confidence=0.75,
                rationale=rationale,
                metadata={
                    "news_risk_flags": retrieval.risk_flags,
                    "max_notional": risk.max_notional,
                    "decision_source": "rules",
                },
            )

        rationale.append("No actionable setup -> hold.")
        return Decision(
            symbol=context.symbol,
            action=Action.HOLD,
            quantity=0,
            confidence=0.5,
            rationale=rationale,
            metadata={"news_risk_flags": retrieval.risk_flags, "decision_source": "rules"},
        )

    def _review_with_llm(
        self,
        context: MarketContext,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskPlan,
        rule_based_decision: Decision,
    ) -> Decision | None:
        if self.llm_client is None:
            return None

        raw = self.llm_client.generate(
            TRADING_SYSTEM_PROMPT,
            "\n".join(
                [
                    DECISION_REVIEW_PROMPT,
                    f"Symbol: {context.symbol}",
                    f"Market open: {context.market_open}",
                    f"Position open: {context.position_open}",
                    f"Current quantity: {context.current_qty}",
                    f"Latest price: {context.latest_price:.4f}",
                    f"Analysis: RSI={analysis.rsi_now:.2f}, MACD={analysis.macd_now:.4f}, "
                    f"signal={analysis.signal_now:.4f}, in_uptrend={analysis.in_uptrend}, "
                    f"entry_setup={analysis.entry_setup}, exit_setup={analysis.exit_setup}",
                    f"News summary: {retrieval.headline_summary}",
                    f"News risk flags: {retrieval.risk_flags}",
                    f"Risk plan: can_enter={risk.can_enter}, recommended_qty={risk.recommended_qty}, "
                    f"max_notional={risk.max_notional:.2f}",
                    f"Rule-based action: {rule_based_decision.action.value}",
                    f"Rule-based quantity: {rule_based_decision.quantity}",
                    f"Rule-based confidence: {rule_based_decision.confidence:.2f}",
                    "Return JSON only.",
                ]
            ),
        )
        payload = _parse_json_object(raw)
        if payload is None:
            return None

        reviewed_action = _parse_action(payload.get("action"))
        reviewed_quantity = _coerce_int(payload.get("quantity"), rule_based_decision.quantity)
        reviewed_confidence = _coerce_confidence(payload.get("confidence"), rule_based_decision.confidence)
        reviewed_rationale = _as_string_list(payload.get("rationale")) or ["LLM review completed."]

        metadata = dict(rule_based_decision.metadata)
        metadata["llm_review_action"] = reviewed_action.value if reviewed_action is not None else None
        metadata["llm_review_used"] = True

        if reviewed_action is None:
            return Decision(
                symbol=rule_based_decision.symbol,
                action=rule_based_decision.action,
                quantity=rule_based_decision.quantity,
                confidence=rule_based_decision.confidence,
                rationale=rule_based_decision.rationale + reviewed_rationale,
                metadata={**metadata, "decision_source": "rules_with_llm_rationale"},
            )

        # Guardrail: the LLM can veto a rule-based BUY, but it cannot invent a new trade.
        if rule_based_decision.action == Action.BUY:
            if reviewed_action == Action.HOLD:
                return Decision(
                    symbol=rule_based_decision.symbol,
                    action=Action.HOLD,
                    quantity=0,
                    confidence=reviewed_confidence,
                    rationale=rule_based_decision.rationale + ["LLM review downgraded the entry to hold."] + reviewed_rationale,
                    metadata={**metadata, "decision_source": "llm_entry_veto"},
                )
            quantity = min(max(reviewed_quantity, 0), risk.recommended_qty)
            if quantity <= 0:
                return Decision(
                    symbol=rule_based_decision.symbol,
                    action=Action.HOLD,
                    quantity=0,
                    confidence=reviewed_confidence,
                    rationale=rule_based_decision.rationale + ["LLM review returned a non-positive quantity, so the entry was held."] + reviewed_rationale,
                    metadata={**metadata, "decision_source": "llm_entry_veto"},
                )
            return Decision(
                symbol=rule_based_decision.symbol,
                action=Action.BUY,
                quantity=quantity,
                confidence=(rule_based_decision.confidence + reviewed_confidence) / 2,
                rationale=rule_based_decision.rationale + reviewed_rationale,
                metadata={**metadata, "decision_source": "rules_with_llm_review"},
            )

        return Decision(
            symbol=rule_based_decision.symbol,
            action=rule_based_decision.action,
            quantity=rule_based_decision.quantity,
            confidence=(rule_based_decision.confidence + reviewed_confidence) / 2,
            rationale=rule_based_decision.rationale + reviewed_rationale,
            metadata={**metadata, "decision_source": "rules_with_llm_review"},
        )


@dataclass
class ExecutionAgent:
    service: Any

    def execute(self, decision: Decision, execute_orders: bool) -> str | None:
        if not execute_orders or decision.action == Action.HOLD or decision.quantity <= 0:
            return None

        from alpaca.trading.enums import OrderSide

        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
        order = self.service.submit_market_order(
            symbol=decision.symbol,
            qty=decision.quantity,
            side=side,
        )
        return str(order.id)


class TradingAgentState(TypedDict, total=False):
    context: MarketContext
    execute_orders: bool
    analysis: AnalysisResult
    retrieval: RetrievalResult
    risk: RiskPlan
    decision: Decision
    order_id: str | None


@dataclass
class TradingAgentGraph:
    analysis_agent: MarketAnalysisAgent
    retrieval_agent: InformationRetrievalAgent
    risk_agent: RiskManagementAgent
    decision_agent: DecisionAgent
    execution_agent: ExecutionAgent
    _app: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if StateGraph is not None:
            self._app = self._build_graph()

    def run(self, context: MarketContext, execute_orders: bool = False) -> WorkflowResult:
        initial_state: TradingAgentState = {"context": context, "execute_orders": execute_orders}
        state = self._app.invoke(initial_state) if self._app is not None else self._run_sequential(initial_state)
        return WorkflowResult(
            context=context,
            analysis=state["analysis"],
            retrieval=state["retrieval"],
            risk=state["risk"],
            decision=state["decision"],
            order_id=state.get("order_id"),
        )

    def _build_graph(self) -> Any:
        workflow = StateGraph(TradingAgentState)
        workflow.add_node("analyze_market", self._analyze_market)
        workflow.add_node("retrieve_information", self._retrieve_information)
        workflow.add_node("evaluate_risk", self._evaluate_risk)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("execute_order", self._execute_order)
        workflow.set_entry_point("analyze_market")
        workflow.add_edge("analyze_market", "retrieve_information")
        workflow.add_edge("retrieve_information", "evaluate_risk")
        workflow.add_edge("evaluate_risk", "make_decision")
        workflow.add_conditional_edges(
            "make_decision",
            self._route_after_decision,
            {"execute": "execute_order", "end": END},
        )
        workflow.add_edge("execute_order", END)
        return workflow.compile()

    def _run_sequential(self, initial_state: TradingAgentState) -> TradingAgentState:
        state: TradingAgentState = dict(initial_state)
        state.update(self._analyze_market(state))
        state.update(self._retrieve_information(state))
        state.update(self._evaluate_risk(state))
        state.update(self._make_decision(state))
        if self._route_after_decision(state) == "execute":
            state.update(self._execute_order(state))
        else:
            state["order_id"] = None
        return state

    def _analyze_market(self, state: TradingAgentState) -> dict[str, AnalysisResult]:
        return {"analysis": self.analysis_agent.analyze(state["context"])}

    def _retrieve_information(self, state: TradingAgentState) -> dict[str, RetrievalResult]:
        context = state["context"]
        return {"retrieval": self.retrieval_agent.retrieve(context.symbol, context.news)}

    def _evaluate_risk(self, state: TradingAgentState) -> dict[str, RiskPlan]:
        return {"risk": self.risk_agent.evaluate(state["context"])}

    def _make_decision(self, state: TradingAgentState) -> dict[str, Decision]:
        return {
            "decision": self.decision_agent.decide(
                state["context"],
                state["analysis"],
                state["retrieval"],
                state["risk"],
            )
        }

    def _execute_order(self, state: TradingAgentState) -> dict[str, str | None]:
        return {
            "order_id": self.execution_agent.execute(
                state["decision"],
                execute_orders=bool(state.get("execute_orders", False)),
            )
        }

    def _route_after_decision(self, state: TradingAgentState) -> str:
        decision = state["decision"]
        if bool(state.get("execute_orders", False)) and decision.action != Action.HOLD and decision.quantity > 0:
            return "execute"
        return "end"


def _count_hits(text: str, keywords: Iterable[str]) -> int:
    return sum(keyword in text for keyword in keywords)


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _parse_json_object(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None

    candidate = raw.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        payload = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _parse_action(value: Any) -> Action | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    try:
        return Action(candidate)
    except ValueError:
        return None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))
