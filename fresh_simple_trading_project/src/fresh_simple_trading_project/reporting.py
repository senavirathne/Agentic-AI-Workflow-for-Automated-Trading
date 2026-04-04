"""Workflow presentation helpers used by the CLI and run loop logging."""

from __future__ import annotations

import pandas as pd

from .models import AlphaVantageIndicatorSnapshot, CollectedMarketData, WorkflowResult


def _print_collected_windows(collected: CollectedMarketData) -> None:
    print(f"[Workflow] Indicator window (5min): {_frame_window(collected.five_minute_bars)}")
    print(f"[Workflow] Support/Resistance window (1h): {_frame_window(collected.hourly_bars)}")


def _print_reasoning(result: WorkflowResult) -> None:
    for line in format_reasoning_lines(result):
        print(line)


def format_reasoning_lines(result: WorkflowResult, prefix: str = "  ") -> list[str]:
    lines = [
        f"{prefix}Data Window: {_data_window(result)}",
        f"{prefix}Alpha Vantage: {_alpha_vantage_window(result)}",
        f"{prefix}Technical Agent: {_display_reason(result.analysis.llm_summary)}",
        f"{prefix}News Agent: {_display_reason(result.retrieval.summary_note)}",
        f"{prefix}Risk Agent: {_display_reason(result.risk.summary_note)}",
    ]
    if result.performance is not None:
        lines.append(
            f"{prefix}Performance: trades={result.performance.trade_count}, "
            f"realized={result.performance.realized_profit:.2f}, "
            f"unrealized={result.performance.unrealized_profit:.2f}, "
            f"current_profit={result.performance.current_profit:.2f}, "
            f"position_qty={result.performance.position_qty}"
        )
    if result.previous_forecast is not None:
        lines.append(f"{prefix}Prior Forecast: {_display_reason(result.previous_forecast.summary)}")
    if result.hold_forecast is not None:
        lines.append(f"{prefix}New HOLD Forecast: {_display_reason(result.hold_forecast.summary)}")
    if result.execution.protective_order_ids:
        lines.append(f"{prefix}Protective Orders: {result.execution.protective_order_ids}")
    lines.extend(_alpha_vantage_hour_chunk_lines(result, prefix=prefix))
    if result.retrieval.critical_news:
        for idx, item in enumerate(result.retrieval.critical_news, 1):
            lines.append(f"{prefix}Critical News {idx}: {item}")
    else:
        for idx, headline in enumerate(result.retrieval.headline_summary, 1):
            lines.append(f"{prefix}Headline {idx}: {headline}")
    for idx, warning in enumerate(result.risk.warnings, 1):
        lines.append(f"{prefix}Risk Warning {idx}: {warning}")
    if result.decision.rationale:
        for idx, rationale in enumerate(result.decision.rationale, 1):
            lines.append(f"{prefix}Decision Reason {idx}: {rationale}")
    else:
        lines.append(f"{prefix}Decision Reason: <none>")
    return lines


def _display_reason(value: str | None) -> str:
    if value is None:
        return "<no output returned>"
    candidate = value.strip()
    return candidate or "<no output returned>"


def _artifact_location(artifact) -> str:
    uri = getattr(artifact, "uri", None)
    if uri:
        return str(uri)
    return str(artifact)


def _print_indicator_context(
    snapshot: AlphaVantageIndicatorSnapshot | None,
    *,
    indicator_source: str,
    feature_frame: pd.DataFrame,
) -> None:
    if snapshot is None:
        print(
            f"[Workflow] Feature engineering finished: source={indicator_source} | "
            f"rows={len(feature_frame)}"
        )
        return
    latest_chunk = (
        "<none>"
        if snapshot.latest_hour_chunk is None
        else f"{snapshot.latest_hour_chunk.slot_start} -> {snapshot.latest_hour_chunk.slot_end}"
    )
    print(
        f"[Workflow] Alpha Vantage data retrieval finished: "
        f"trading_day={snapshot.trading_day} | rows={len(snapshot.rows)} | "
        f"hourly_chunks={len(snapshot.hourly_chunks)} | latest_chunk={latest_chunk}"
    )


def _print_labeled_items(label: str, items: list[str]) -> None:
    if not items:
        return
    for index, item in enumerate(items, 1):
        print(f"[Workflow] {label} {index}: {item}")


def _data_window(result: WorkflowResult) -> str:
    return f"5min {_frame_window(result.five_minute_bars)} | 1h {_frame_window(result.hourly_bars)}"


def _alpha_vantage_window(result: WorkflowResult) -> str:
    snapshot = result.alpha_vantage_indicator_snapshot
    if snapshot is None:
        return "<not configured>"
    if snapshot.latest_hour_chunk is None:
        return f"{snapshot.trading_day} | no hourly chunk"
    return (
        f"{snapshot.trading_day} | "
        f"{snapshot.latest_hour_chunk.slot_start} -> {snapshot.latest_hour_chunk.slot_end}"
    )


def _alpha_vantage_hour_chunk_lines(result: WorkflowResult, *, prefix: str) -> list[str]:
    snapshot = result.alpha_vantage_indicator_snapshot
    if snapshot is None or not snapshot.hourly_chunks:
        return []
    return [
        (
            f"{prefix}Alpha Vantage 1h Chunk {idx}: "
            f"{chunk.slot_start} -> {chunk.slot_end} | rows={len(chunk.rows)}"
        )
        for idx, chunk in enumerate(snapshot.hourly_chunks, 1)
    ]


def _frame_window(frame: pd.DataFrame) -> str:
    try:
        start = frame.index.min()
        end = frame.index.max()
        if start is None or end is None:
            return "<unknown>"
        return f"{pd.Timestamp(start).isoformat()} -> {pd.Timestamp(end).isoformat()}"
    except Exception:
        return "<unknown>"
