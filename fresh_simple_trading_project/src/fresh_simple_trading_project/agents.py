from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import TextGenerationClient, clean_llm_text
from .models import AccountState, AnalysisResult, EDAResult, NewsArticle, RetrievalResult, RiskResult


@dataclass(frozen=True)
class DecisionReview:
    action: str | None = None
    quantity: int | None = None
    note: str | None = None
    override: str | None = None


@dataclass(frozen=True)
class NewsResearchSummary:
    summary_note: str | None = None
    critical_news: list[str] | None = None
    risk_flags: list[str] | None = None
    catalysts: list[str] | None = None


@dataclass
class TechnicalAnalysisAgent:
    llm_client: TextGenerationClient | None = None

    SYSTEM_PROMPT = (
        "You are the technical analysis agent in a multi-agent trading workflow. "
        "Review the provided market snapshot and write one concise handoff."
    )

    def summarize(
        self,
        *,
        symbol: str,
        as_of: str,
        trend: str,
        latest_price: float,
        latest_rsi: float,
        latest_macd: float,
        latest_macd_signal: float,
        short_ma: float,
        long_ma: float,
        rolling_volatility: float,
        nearest_support: float | None,
        nearest_resistance: float | None,
        local_support_region: tuple[float, float] | None,
        local_resistance_region: tuple[float, float] | None,
        alpha_vantage_latest_hour_chunk: list[dict[str, object]] | None = None,
        alpha_vantage_threshold_hits: list[str] | None = None,
    ) -> str | None:
        if self.llm_client is None:
            return None

        alpha_vantage_lines: list[str] = []
        if alpha_vantage_latest_hour_chunk:
            alpha_vantage_lines.extend(
                [
                    "Alpha Vantage latest 1-hour indicator chunk (5-minute rows, JSON):",
                    json.dumps(alpha_vantage_latest_hour_chunk, separators=(",", ":")),
                ]
            )
        if alpha_vantage_threshold_hits:
            alpha_vantage_lines.append(f"Alpha Vantage threshold hits in latest chunk: {alpha_vantage_threshold_hits}")

        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    f"Symbol: {symbol}",
                    f"Current datetime: {as_of}",
                    f"Current price: {latest_price:.2f}",
                    "Timeframe analyzed: 5-minute candles within the latest hourly cycle.",
                    f"Trend: {trend}",
                    f"Latest close: {latest_price:.2f}",
                    f"RSI: {latest_rsi:.2f}",
                    f"MACD: {latest_macd:.4f}",
                    f"MACD signal: {latest_macd_signal:.4f}",
                    f"Short MA: {short_ma:.2f}",
                    f"Long MA: {long_ma:.2f}",
                    f"Rolling volatility: {rolling_volatility:.4f}",
                    f"Major support (hourly): {nearest_support}",
                    f"Major resistance (hourly): {nearest_resistance}",
                    f"Local support region (5-minute): {local_support_region}",
                    f"Local resistance region (5-minute): {local_resistance_region}",
                    *alpha_vantage_lines,
                    "The handoff must explicitly include the current datetime and current price.",
                    "Return one short technical handoff for the decision-making agent.",
                ]
            ),
        )
        return _ensure_technical_handoff_fields(
            symbol=symbol,
            as_of=as_of,
            latest_price=latest_price,
            summary=clean_llm_text(raw),
        )


@dataclass
class NewsResearchAgent:
    llm_client: TextGenerationClient | None = None

    SYSTEM_PROMPT = (
        "You are the news research agent in a multi-agent trading workflow. "
        "Review the provided news context before preparing one concise handoff."
    )

    def summarize(
        self,
        *,
        symbol: str,
        articles: list[NewsArticle],
        search_queries: list[str],
        limit: int | None = None,
    ) -> NewsResearchSummary | None:
        if self.llm_client is None or not articles:
            return None

        article_limit = min(limit or len(articles), len(articles))
        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    f"Symbol: {symbol}",
                    "Search focus: symbol-specific news, American economy developments, and overall U.S. stock market trend.",
                    "Queries used:",
                    *[f"- {query}" for query in search_queries],
                    "Recent articles:",
                    *[
                        f"- {article.headline} | {article.summary} | source={article.source} | published_at={article.published_at}"
                        for article in articles[:article_limit]
                    ],
                    "Review all provided articles.",
                    "Identify only the critical news items that materially affect trading decisions.",
                    "Return JSON with keys: summary_note, critical_news, risk_flags, catalysts.",
                    "summary_note must be a precise short summary for the decision-making agent.",
                    "critical_news must be an ordered array of the most material items, highest impact first.",
                    "risk_flags must list concrete downside risks from the provided news only.",
                    "catalysts must list concrete catalysts from the provided news only.",
                    "Do not produce or rely on a sentiment score.",
                ]
            ),
        )
        return _parse_news_research_summary(clean_llm_text(raw))


@dataclass
class RiskReviewAgent:
    llm_client: TextGenerationClient | None = None

    SYSTEM_PROMPT = (
        "You are the risk manager agent in a multi-agent trading workflow. "
        "Summarize the main trade risk and size constraint in one concise sentence."
    )

    def summarize(
        self,
        *,
        symbol: str,
        risk_score: float,
        recommended_qty: int,
        can_enter: bool,
        warnings: list[str],
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        eda: EDAResult,
    ) -> str | None:
        if self.llm_client is None:
            return None

        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    f"Symbol: {symbol}",
                    f"Risk score: {risk_score:.2f}",
                    f"Can enter: {can_enter}",
                    f"Recommended quantity: {recommended_qty}",
                    f"Latest price: {analysis.latest_price:.2f}",
                    f"Trend: {analysis.trend}",
                    f"EDA volatility: {eda.candle_volatility:.4f}",
                    f"EDA anomaly count: {eda.anomaly_count}",
                    f"Critical news: {retrieval.critical_news}",
                    f"News risks: {retrieval.risk_flags}",
                    f"News catalysts: {retrieval.catalysts}",
                    f"News agent handoff: {retrieval.summary_note}",
                    f"Warnings: {warnings}",
                    "Return one short risk handoff for the decision-making agent.",
                ]
            ),
        )
        return clean_llm_text(raw)


@dataclass
class DecisionCoordinatorAgent:
    llm_client: TextGenerationClient | None = None

    SYSTEM_PROMPT = (
        "You are the decision coordinator agent in a multi-agent trading workflow. "
        "Review the provided market, news, and risk context, then produce the final trading decision."
    )

    def review(
        self,
        *,
        rule_based_action: str,
        rule_based_quantity: int,
        account: AccountState,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskResult,
    ) -> DecisionReview:
        if self.llm_client is None:
            return DecisionReview()

        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    "Review the proposed decision from the existing simple workflow modules.",
                    "Return exactly three lines:",
                    "ACTION=BUY or SELL or HOLD",
                    "QUANTITY=<integer>",
                    "NOTE=<one short sentence>",
                    f"Symbol: {analysis.symbol}",
                    f"Market open: {account.market_open}",
                    f"Current position quantity: {account.position_qty}",
                    f"Cash available: {account.cash:.2f}",
                    f"Rule-based action: {rule_based_action}",
                    f"Rule-based quantity: {rule_based_quantity}",
                    f"Latest price: {analysis.latest_price:.2f}",
                    f"Trend: {analysis.trend}",
                    f"Bullish: {analysis.bullish}",
                    f"Entry setup: {analysis.entry_setup}",
                    f"Exit setup: {analysis.exit_setup}",
                    f"Confidence: {analysis.confidence:.2f}",
                    f"Major support levels (hourly): {analysis.support_levels}",
                    f"Major resistance levels (hourly): {analysis.resistance_levels}",
                    f"Local support regions (5-minute): {analysis.support_regions}",
                    f"Local resistance regions (5-minute): {analysis.resistance_regions}",
                    f"Nearest support: {analysis.nearest_support}",
                    f"Nearest resistance: {analysis.nearest_resistance}",
                    f"Distance to support pct: {analysis.distance_to_support_pct}",
                    f"Distance to resistance pct: {analysis.distance_to_resistance_pct}",
                    f"Technical agent handoff: {analysis.llm_summary}",
                    f"Critical news: {retrieval.critical_news}",
                    f"News risks: {retrieval.risk_flags}",
                    f"News catalysts: {retrieval.catalysts}",
                    f"News headlines: {retrieval.headline_summary}",
                    f"News agent handoff: {retrieval.summary_note}",
                    f"Risk score: {risk.risk_score:.2f}",
                    f"Can enter: {risk.can_enter}",
                    f"Recommended quantity: {risk.recommended_qty}",
                    f"Risk warnings: {risk.warnings}",
                    f"Risk agent handoff: {risk.summary_note}",
                    "BUY is only valid for a new entry. SELL is only valid for reducing or closing an existing position.",
                ]
            ),
        )
        return _parse_review(clean_llm_text(raw))


def _parse_review(raw: str | None) -> DecisionReview:
    if not raw:
        return DecisionReview()

    action: str | None = None
    quantity: int | None = None
    note: str | None = None
    override: str | None = None

    if raw.lstrip().startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            action = _normalize_action(payload.get("action"))
            quantity = _parse_int(payload.get("quantity"))
            note = _as_string(payload.get("note"))
            override = _normalize_override(payload.get("override"))
            return DecisionReview(action=action, quantity=quantity, note=note, override=override)

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        normalized = stripped.replace(":", "=", 1)
        if "=" not in normalized:
            continue
        key, value = normalized.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if key == "ACTION":
            action = _normalize_action(value)
        elif key == "QUANTITY":
            quantity = _parse_int(value)
        elif key == "NOTE":
            note = value
        elif key == "OVERRIDE":
            override = _normalize_override(value)

    if note is None:
        note = raw.strip()
    return DecisionReview(action=action, quantity=quantity, note=note, override=override)


def _parse_news_research_summary(raw: str | None) -> NewsResearchSummary:
    cleaned = clean_llm_text(raw)
    if not cleaned:
        return NewsResearchSummary(summary_note=None, critical_news=[], risk_flags=[], catalysts=[])

    if cleaned.lstrip().startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return NewsResearchSummary(
                summary_note=_as_string(payload.get("summary_note") or payload.get("summary") or payload.get("note")),
                critical_news=_as_string_list(payload.get("critical_news")),
                risk_flags=_as_string_list(payload.get("risk_flags")),
                catalysts=_as_string_list(payload.get("catalysts")),
            )

    return NewsResearchSummary(
        summary_note=cleaned,
        critical_news=[],
        risk_flags=[],
        catalysts=[],
    )


def _ensure_technical_handoff_fields(
    *,
    symbol: str,
    as_of: str,
    latest_price: float,
    summary: str | None,
) -> str:
    prefix = f"As of {as_of}, {symbol} current price is ${latest_price:.2f}."
    if not summary:
        return prefix
    if summary.startswith(prefix):
        return summary
    return f"{prefix} {summary}"


def _normalize_action(value: Any) -> str | None:
    candidate = _as_string(value)
    if candidate is None:
        return None
    normalized = candidate.upper()
    return normalized if normalized in {"BUY", "SELL", "HOLD"} else None


def _normalize_override(value: Any) -> str | None:
    candidate = _as_string(value)
    if candidate is None:
        return None
    normalized = candidate.upper()
    return normalized if normalized in {"KEEP", "HOLD"} else None


def _parse_int(value: Any) -> int | None:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [_as_string(item) for item in value]
        return [item for item in items if item]
    candidate = _as_string(value)
    return [candidate] if candidate else []
