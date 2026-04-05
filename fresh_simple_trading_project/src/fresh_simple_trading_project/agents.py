"""Text-only agent wrappers and parsing helpers for the workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .llm import TextGenerationClient, clean_llm_text
from .models import AccountState, AnalysisResult, EDAResult, ForecastSnapshot, NewsArticle, RetrievalResult, RiskResult
from .utils import _as_string, _as_utc_timestamp


@dataclass(frozen=True)
class DecisionReview:
    """Normalized decision-review payload returned by the coordinator agent."""

    action: str | None = None
    quantity: int | None = None
    note: str | None = None
    override: str | None = None


@dataclass(frozen=True)
class NewsResearchSummary:
    """Structured summary returned by the news research agent."""

    summary_note: str | None = None
    critical_news: list[str] | None = None
    risk_flags: list[str] | None = None
    catalysts: list[str] | None = None


@dataclass
class TechnicalAnalysisAgent:
    """Produce the technical-analysis handoff consumed by later agents."""

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
        current_price: float | None = None,
        market_data_delay_minutes: int = 0,
        indicator_source: str = "manually computed indicators from 5-minute bar data",
        previous_forecast: ForecastSnapshot | None = None,
    ) -> str | None:
        """Generate the technical handoff consumed by later workflow stages."""

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
        live_context_lines = _live_market_context_lines(
            latest_price=latest_price,
            current_price=current_price,
            market_data_delay_minutes=market_data_delay_minutes,
        )
        forecast_context_lines = _forecast_context_lines(previous_forecast)

        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    f"Symbol: {symbol}",
                    f"Current datetime: {as_of}",
                    f"Current price: {latest_price:.2f}",
                    f"Indicator source: {indicator_source}",
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
                    *live_context_lines,
                    *forecast_context_lines,
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
            current_price=current_price,
            market_data_delay_minutes=market_data_delay_minutes,
            summary=clean_llm_text(raw),
        )


@dataclass
class NewsResearchAgent:
    """Summarize recent news into compact trading-relevant context."""

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
        """Summarize recent articles into structured trading context."""

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
                    "Focus only on three scopes: the given symbol, the American economy, and the overall U.S. stock market.",
                    "Return JSON with keys: summary_note, critical_news, risk_flags, catalysts.",
                    "summary_note must be a compressed precise summary for the decision-making agent.",
                    "critical_news must contain at most 3 short items, ordered by impact, covering only the symbol, the economy, and the market when material.",
                    "risk_flags must list concrete downside risks from the provided news only.",
                    "catalysts must list concrete catalysts from the provided news only.",
                    "Do not produce or rely on a sentiment score.",
                ]
            ),
        )
        return _parse_news_research_summary(clean_llm_text(raw))


@dataclass
class RiskReviewAgent:
    """Produce a concise risk-management handoff for the decision stage."""

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
        previous_forecast: ForecastSnapshot | None = None,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        position_in_profit: bool = False,
    ) -> str | None:
        """Generate the concise risk handoff used by the decision stage."""

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
                    *_analysis_live_context_lines(analysis),
                    f"Trend: {analysis.trend}",
                    f"EDA volatility: {eda.candle_volatility:.4f}",
                    f"EDA anomaly count: {eda.anomaly_count}",
                    f"Position in profit: {position_in_profit}",
                    f"Recommended stop loss: {stop_loss_price}",
                    f"Recommended take profit: {take_profit_price}",
                    f"Critical news: {retrieval.critical_news}",
                    f"News risks: {retrieval.risk_flags}",
                    f"News catalysts: {retrieval.catalysts}",
                    f"News agent handoff: {retrieval.summary_note}",
                    *_forecast_context_lines(previous_forecast),
                    f"Warnings: {warnings}",
                    "Return one short risk handoff for the decision-making agent.",
                ]
            ),
        )
        return clean_llm_text(raw)


@dataclass
class DecisionCoordinatorAgent:
    """Review rule-based output and produce the final trade instruction."""

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
        previous_forecast: ForecastSnapshot | None = None,
    ) -> DecisionReview:
        """Review the rule-based trade and return the coordinator payload."""

        if self.llm_client is None:
            return DecisionReview()
        price_levels = analysis.price_levels

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
                    f"Major support levels (hourly): {price_levels.support_levels}",
                    f"Major resistance levels (hourly): {price_levels.resistance_levels}",
                    f"Local support regions (5-minute): {price_levels.support_regions}",
                    f"Local resistance regions (5-minute): {price_levels.resistance_regions}",
                    f"Nearest support: {price_levels.nearest_support}",
                    f"Nearest resistance: {price_levels.nearest_resistance}",
                    f"Distance to support pct: {price_levels.distance_to_support_pct}",
                    f"Distance to resistance pct: {price_levels.distance_to_resistance_pct}",
                    *_analysis_live_context_lines(analysis),
                    f"Technical agent handoff: {analysis.llm_summary}",
                    f"News context JSON: {_decision_news_context_json(retrieval)}",
                    f"Risk score: {risk.risk_score:.2f}",
                    f"Can enter: {risk.can_enter}",
                    f"Recommended quantity: {risk.recommended_qty}",
                    f"Position in profit: {risk.position_in_profit}",
                    f"Stop loss price: {risk.stop_loss_price}",
                    f"Take profit price: {risk.take_profit_price}",
                    f"Risk warnings: {risk.warnings}",
                    f"Risk agent handoff: {risk.summary_note}",
                    *_forecast_context_lines(previous_forecast),
                    "BUY is only valid for a new entry. SELL is only valid for reducing or closing an existing position.",
                ]
            ),
        )
        return _parse_review(clean_llm_text(raw))


@dataclass
class HoldForecastAgent:
    """Generate continuation and reversal expectations for HOLD decisions."""

    llm_client: TextGenerationClient | None = None

    SYSTEM_PROMPT = (
        "You are the hold-forecast agent in a multi-agent trading workflow. "
        "When the trading decision is HOLD, produce one structured one-hour forecast."
    )

    def forecast(
        self,
        *,
        analysis: AnalysisResult,
        retrieval: RetrievalResult,
        risk: RiskResult,
        previous_forecast: ForecastSnapshot | None = None,
        valid_for_minutes: int = 60,
    ) -> ForecastSnapshot:
        """Generate the next HOLD forecast snapshot for the active symbol."""

        generated_at = _as_utc_timestamp(analysis.timestamp)
        valid_until = generated_at + pd.Timedelta(minutes=max(1, valid_for_minutes))
        latest_price = float(analysis.current_price or analysis.latest_price)
        latest_volume = float(analysis.latest_volume)
        price_levels = analysis.price_levels

        if self.llm_client is None:
            return _heuristic_hold_forecast(
                analysis=analysis,
                generated_at=generated_at,
                valid_until=valid_until,
            )

        raw = self.llm_client.generate(
            self.SYSTEM_PROMPT,
            "\n".join(
                [
                    f"Symbol: {analysis.symbol}",
                    f"As of: {generated_at.isoformat().replace('+00:00', 'Z')}",
                    f"Valid until: {valid_until.isoformat().replace('+00:00', 'Z')}",
                    f"Latest price: {latest_price:.2f}",
                    f"Latest volume: {latest_volume:.0f}",
                    f"Trend: {analysis.trend}",
                    f"Confidence: {analysis.confidence:.2f}",
                    f"Nearest support: {price_levels.nearest_support}",
                    f"Nearest resistance: {price_levels.nearest_resistance}",
                    f"Nearest support region: {price_levels.nearest_support_region}",
                    f"Nearest resistance region: {price_levels.nearest_resistance_region}",
                    f"Risk warnings: {risk.warnings}",
                    f"Critical news: {retrieval.critical_news}",
                    f"Risk catalysts: {retrieval.catalysts}",
                    *_forecast_context_lines(previous_forecast),
                    "Return JSON with keys:",
                    "trend_bias, continuation_price_target, continuation_volume_target, reversal_price_target, reversal_volume_target, continuation_signals, reversal_signals, summary, confidence",
                    "The continuation target must explain the price and volume level that keeps the trend intact.",
                    "The reversal target must explain the price and volume level that signals likely near-term reversal.",
                    "continuation_signals and reversal_signals must be short string lists.",
                ]
            ),
        )
        return _parse_hold_forecast(
            clean_llm_text(raw),
            analysis=analysis,
            generated_at=generated_at,
            valid_until=valid_until,
        )


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
    current_price: float | None,
    market_data_delay_minutes: int,
    summary: str | None,
) -> str:
    if market_data_delay_minutes > 0 and current_price is not None:
        prefix = (
            f"As of {as_of}, {symbol} latest delayed close is ${latest_price:.2f} "
            f"and current live price is ${current_price:.2f}."
        )
    elif market_data_delay_minutes > 0:
        prefix = (
            f"As of {as_of}, {symbol} latest delayed close is ${latest_price:.2f}."
        )
    else:
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


def _parse_float(value: Any) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [_as_string(item) for item in value]
        return [item for item in items if item]
    candidate = _as_string(value)
    return [candidate] if candidate else []


def _forecast_context_lines(previous_forecast: ForecastSnapshot | None) -> list[str]:
    if previous_forecast is None:
        return ["Previous HOLD forecast: <none>"]
    return [
        "Previous HOLD forecast:",
        f"- Generated at: {previous_forecast.generated_at.isoformat().replace('+00:00', 'Z')}",
        f"- Valid until: {previous_forecast.valid_until.isoformat().replace('+00:00', 'Z')}",
        f"- Trend bias: {previous_forecast.trend_bias}",
        f"- Continuation target: price={previous_forecast.continuation_price_target}, volume={previous_forecast.continuation_volume_target}",
        f"- Reversal target: price={previous_forecast.reversal_price_target}, volume={previous_forecast.reversal_volume_target}",
        f"- Continuation signals: {previous_forecast.continuation_signals}",
        f"- Reversal signals: {previous_forecast.reversal_signals}",
        f"- Summary: {previous_forecast.summary}",
    ]


def _parse_hold_forecast(
    raw: str | None,
    *,
    analysis: AnalysisResult,
    generated_at: pd.Timestamp,
    valid_until: pd.Timestamp,
) -> ForecastSnapshot:
    heuristic = _heuristic_hold_forecast(
        analysis=analysis,
        generated_at=generated_at,
        valid_until=valid_until,
    )
    cleaned = clean_llm_text(raw)
    if cleaned and cleaned.lstrip().startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return ForecastSnapshot(
                symbol=analysis.symbol,
                generated_at=generated_at,
                valid_until=valid_until,
                reference_price=float(analysis.current_price or analysis.latest_price),
                reference_volume=float(analysis.latest_volume),
                trend_bias=_as_string(payload.get("trend_bias")) or heuristic.trend_bias,
                continuation_price_target=_coalesce_float(
                    _parse_float(payload.get("continuation_price_target")),
                    heuristic.continuation_price_target,
                ),
                continuation_volume_target=_coalesce_float(
                    _parse_float(payload.get("continuation_volume_target")),
                    heuristic.continuation_volume_target,
                ),
                reversal_price_target=_coalesce_float(
                    _parse_float(payload.get("reversal_price_target")),
                    heuristic.reversal_price_target,
                ),
                reversal_volume_target=_coalesce_float(
                    _parse_float(payload.get("reversal_volume_target")),
                    heuristic.reversal_volume_target,
                ),
                continuation_signals=_as_string_list(payload.get("continuation_signals"))
                or heuristic.continuation_signals,
                reversal_signals=_as_string_list(payload.get("reversal_signals")) or heuristic.reversal_signals,
                summary=_as_string(payload.get("summary")) or heuristic.summary,
                confidence=max(
                    0.0,
                    min(1.0, round(_coalesce_float(_parse_float(payload.get("confidence")), analysis.confidence), 2)),
                ),
            )
    return heuristic


def _heuristic_hold_forecast(
    *,
    analysis: AnalysisResult,
    generated_at: pd.Timestamp,
    valid_until: pd.Timestamp,
) -> ForecastSnapshot:
    reference_price = float(analysis.current_price or analysis.latest_price)
    reference_volume = float(analysis.latest_volume)
    price_levels = analysis.price_levels
    continuation_price_target = price_levels.nearest_resistance or round(reference_price * 1.01, 2)
    continuation_volume_target = round(reference_volume * 1.1, 2) if reference_volume > 0 else None
    reversal_price_target = price_levels.nearest_support or round(reference_price * 0.99, 2)
    reversal_volume_target = round(reference_volume * 1.15, 2) if reference_volume > 0 else None
    return ForecastSnapshot(
        symbol=analysis.symbol,
        generated_at=generated_at,
        valid_until=valid_until,
        reference_price=reference_price,
        reference_volume=reference_volume,
        trend_bias=analysis.trend,
        continuation_price_target=continuation_price_target,
        continuation_volume_target=continuation_volume_target,
        reversal_price_target=reversal_price_target,
        reversal_volume_target=reversal_volume_target,
        continuation_signals=[
            "Trend stays intact if price holds above support and expands on volume.",
            "Momentum continuation is stronger if buyers clear resistance with volume confirmation.",
        ],
        reversal_signals=[
            "A drop back through support with heavier volume warns of reversal.",
            "Momentum loss near resistance followed by weak closes signals fading trend strength.",
        ],
        summary=(
            f"Monitor {continuation_price_target:.2f} with volume expansion for continuation and "
            f"{reversal_price_target:.2f} with heavier sell volume for reversal risk."
        ),
        confidence=max(0.0, min(1.0, round(analysis.confidence, 2))),
    )


def _coalesce_float(primary: float | None, fallback: float | None) -> float | None:
    if primary is not None:
        return primary
    return fallback


def _decision_news_context_json(retrieval: RetrievalResult) -> str:
    return json.dumps(
        {
            "summary_note": retrieval.summary_note,
            "critical_news": retrieval.critical_news,
            "risk_flags": retrieval.risk_flags,
            "catalysts": retrieval.catalysts,
        },
        separators=(",", ":"),
    )


def _live_market_context_lines(
    *,
    latest_price: float,
    current_price: float | None,
    market_data_delay_minutes: int,
) -> list[str]:
    if market_data_delay_minutes <= 0:
        return []
    lines = [
        f"Live market data delay minutes: {market_data_delay_minutes}",
        f"Latest delayed 5-minute close: {latest_price:.2f}",
    ]
    if current_price is not None:
        lines.append(f"Current live price snapshot: {current_price:.2f}")
    lines.append("Use the live price snapshot for current context; indicators still come from delayed bars.")
    return lines


def _analysis_live_context_lines(analysis: AnalysisResult) -> list[str]:
    return [
        f"Indicator source: {analysis.indicator_source}",
        *_live_market_context_lines(
            latest_price=analysis.latest_price,
            current_price=analysis.current_price,
            market_data_delay_minutes=analysis.market_data_delay_minutes,
        ),
    ]
