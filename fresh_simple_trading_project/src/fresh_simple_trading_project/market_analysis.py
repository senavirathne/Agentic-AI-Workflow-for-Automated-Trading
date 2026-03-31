from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .agents import TechnicalAnalysisAgent
from .config import TradingConfig
from .llm import TextGenerationClient
from .models import AnalysisResult


@dataclass
class MarketAnalysisModule:
    config: TradingConfig
    llm_client: TextGenerationClient | None = None
    technical_agent: TechnicalAnalysisAgent | None = None

    def __post_init__(self) -> None:
        if self.technical_agent is None and self.llm_client is not None:
            self.technical_agent = TechnicalAnalysisAgent(llm_client=self.llm_client)

    def analyze(
        self,
        symbol: str,
        feature_frame: pd.DataFrame,
        hourly_bars: pd.DataFrame | None = None,
    ) -> AnalysisResult:
        latest = feature_frame.iloc[-1]
        analysis_timestamp = _as_utc_timestamp(feature_frame.index[-1])
        major_frame = hourly_bars if hourly_bars is not None and not hourly_bars.empty else feature_frame
        support_levels, _, resistance_levels, _ = _identify_price_levels(major_frame)
        local_support_levels, local_support_strengths, local_resistance_levels, local_resistance_strengths = _identify_price_levels(feature_frame)
        latest_price = float(latest["close"])
        region_half_width_pct = _region_half_width_pct(feature_frame)
        support_regions = _build_regions(local_support_levels, region_half_width_pct)
        resistance_regions = _build_regions(local_resistance_levels, region_half_width_pct)
        nearest_support = _nearest_support(support_levels, latest_price)
        nearest_resistance = _nearest_resistance(resistance_levels, latest_price)
        nearest_support_region = _nearest_support_region(support_regions, latest_price)
        nearest_resistance_region = _nearest_resistance_region(resistance_regions, latest_price)
        nearest_support_region_strength = _nearest_region_strength(
            nearest_support_region,
            support_regions,
            local_support_strengths,
        )
        nearest_resistance_region_strength = _nearest_region_strength(
            nearest_resistance_region,
            resistance_regions,
            local_resistance_strengths,
        )
        distance_to_support_pct = _distance_pct(nearest_support, latest_price)
        distance_to_resistance_pct = _distance_pct(nearest_resistance, latest_price)
        bullish = bool(latest["ma_short"] >= latest["ma_long"] and latest["macd"] >= latest["macd_signal"])
        entry_setup = bool(latest["buy_trigger"])
        exit_setup = bool(latest["sell_trigger"])
        trend = "bullish" if bullish else "bearish"
        confidence = 0.5
        if bullish:
            confidence += 0.2
        if float(latest["rsi"]) >= self.config.buy_rsi_threshold:
            confidence += 0.1
        if float(latest["rolling_volatility"]) <= self.config.risk_volatility_cutoff:
            confidence += 0.1
        confidence = max(0.0, min(1.0, round(confidence, 2)))
        notes = [
            "Signal computed from 5-minute candlestick data.",
            "Major support and resistance lines are derived from hourly pivot candles.",
            "Local support and resistance regions are derived from 5-minute pivot candles and recent candle ranges.",
            f"Latest close={latest_price:.2f}",
            f"RSI={float(latest['rsi']):.2f}",
            f"MACD={float(latest['macd']):.4f} vs signal={float(latest['macd_signal']):.4f}",
            f"Trend={trend}",
        ]
        if nearest_support is not None:
            notes.append(f"Major support={nearest_support:.2f} ({distance_to_support_pct:.2f}% away).")
        if nearest_support_region is not None:
            notes.append(
                "Local support region="
                + _format_region(nearest_support_region)
                + f" (historical touches={nearest_support_region_strength})."
            )
        if nearest_resistance is not None:
            notes.append(f"Major resistance={nearest_resistance:.2f} ({distance_to_resistance_pct:.2f}% away).")
        if nearest_resistance_region is not None:
            notes.append(
                "Local resistance region="
                + _format_region(nearest_resistance_region)
                + f" (historical touches={nearest_resistance_region_strength})."
            )
        llm_summary = self._summarize_with_llm(
            symbol=symbol,
            analysis_timestamp=analysis_timestamp,
            latest=latest,
            trend=trend,
            latest_price=latest_price,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            local_support_region=nearest_support_region,
            local_resistance_region=nearest_resistance_region,
        )
        if llm_summary:
            notes.append(f"LLM technical summary: {llm_summary}")
        return AnalysisResult(
            symbol=symbol,
            timestamp=analysis_timestamp,
            latest_price=latest_price,
            trend=trend,
            bullish=bullish,
            entry_setup=entry_setup,
            exit_setup=exit_setup,
            confidence=confidence,
            notes=notes,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            support_regions=support_regions,
            resistance_regions=resistance_regions,
            support_region_strengths=local_support_strengths,
            resistance_region_strengths=local_resistance_strengths,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            nearest_support_region=nearest_support_region,
            nearest_resistance_region=nearest_resistance_region,
            nearest_support_region_strength=nearest_support_region_strength,
            nearest_resistance_region_strength=nearest_resistance_region_strength,
            distance_to_support_pct=distance_to_support_pct,
            distance_to_resistance_pct=distance_to_resistance_pct,
            llm_summary=llm_summary,
        )

    def _summarize_with_llm(
        self,
        *,
        symbol: str,
        analysis_timestamp: pd.Timestamp,
        latest: pd.Series,
        trend: str,
        latest_price: float,
        nearest_support: float | None,
        nearest_resistance: float | None,
        local_support_region: tuple[float, float] | None,
        local_resistance_region: tuple[float, float] | None,
    ) -> str | None:
        if self.technical_agent is None:
            return None

        return self.technical_agent.summarize(
            symbol=symbol,
            as_of=analysis_timestamp.isoformat().replace("+00:00", "Z"),
            trend=trend,
            latest_price=latest_price,
            latest_rsi=float(latest["rsi"]),
            latest_macd=float(latest["macd"]),
            latest_macd_signal=float(latest["macd_signal"]),
            short_ma=float(latest["ma_short"]),
            long_ma=float(latest["ma_long"]),
            rolling_volatility=float(latest["rolling_volatility"]),
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            local_support_region=local_support_region,
            local_resistance_region=local_resistance_region,
        )


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _identify_price_levels(
    frame: pd.DataFrame,
    pivot_window: int = 3,
    max_levels: int = 3,
) -> tuple[list[float], list[int], list[float], list[int]]:
    recent = frame.copy()
    if len(recent) <= pivot_window * 2:
        return [], [], [], []
    history = recent.iloc[:-1] if len(recent) > 1 else recent

    support_candidates: list[float] = []
    resistance_candidates: list[float] = []
    lows = recent["low"].reset_index(drop=True)
    highs = recent["high"].reset_index(drop=True)

    for index in range(pivot_window, len(recent) - pivot_window):
        low_window = lows.iloc[index - pivot_window : index + pivot_window + 1]
        high_window = highs.iloc[index - pivot_window : index + pivot_window + 1]
        current_low = float(lows.iloc[index])
        current_high = float(highs.iloc[index])
        if current_low == float(low_window.min()):
            support_candidates.append(current_low)
        if current_high == float(high_window.max()):
            resistance_candidates.append(current_high)

    if not support_candidates and not history.empty:
        support_candidates.append(float(history["low"].min()))
    if not resistance_candidates and not history.empty:
        resistance_candidates.append(float(history["high"].max()))

    supports, support_strengths = _compress_levels(sorted(support_candidates), max_levels=max_levels)
    resistances, resistance_strengths = _compress_levels(
        sorted(resistance_candidates, reverse=True),
        max_levels=max_levels,
    )
    return supports, support_strengths, resistances, resistance_strengths


def _region_half_width_pct(frame: pd.DataFrame) -> float:
    candle_range_pct = ((frame["high"] - frame["low"]) / frame["close"].replace(0, pd.NA)).tail(20).dropna()
    if candle_range_pct.empty:
        return 0.005
    half_width_pct = float(candle_range_pct.mean()) / 2
    return round(min(max(half_width_pct, 0.003), 0.015), 4)


def _build_regions(levels: list[float], half_width_pct: float) -> list[tuple[float, float]]:
    regions: list[tuple[float, float]] = []
    for level in levels:
        lower = round(level * (1 - half_width_pct), 2)
        upper = round(level * (1 + half_width_pct), 2)
        regions.append((lower, upper))
    return regions


def _compress_levels(levels: list[float], max_levels: int, tolerance_pct: float = 0.003) -> tuple[list[float], list[int]]:
    compressed: list[float] = []
    strengths: list[int] = []
    for level in levels:
        if not compressed:
            compressed.append(level)
            strengths.append(1)
            continue
        match_index = next(
            (
                index
                for index, existing in enumerate(compressed)
                if abs(level - existing) / existing <= tolerance_pct
            ),
            None,
        )
        if match_index is None:
            compressed.append(level)
            strengths.append(1)
            continue
        total = strengths[match_index] + 1
        compressed[match_index] = ((compressed[match_index] * strengths[match_index]) + level) / total
        strengths[match_index] = total

    rounded_levels = [round(level, 2) for level in compressed[:max_levels]]
    return rounded_levels, strengths[:max_levels]


def _nearest_support(levels: list[float], latest_price: float) -> float | None:
    eligible = [level for level in levels if level <= latest_price]
    if not eligible:
        return None
    return max(eligible)


def _nearest_resistance(levels: list[float], latest_price: float) -> float | None:
    eligible = [level for level in levels if level >= latest_price]
    if not eligible:
        return None
    return min(eligible)


def _nearest_support_region(regions: list[tuple[float, float]], latest_price: float) -> tuple[float, float] | None:
    eligible = [region for region in regions if region[0] <= latest_price]
    if not eligible:
        return None
    return max(eligible, key=lambda region: _region_midpoint(region))


def _nearest_resistance_region(regions: list[tuple[float, float]], latest_price: float) -> tuple[float, float] | None:
    eligible = [region for region in regions if region[1] >= latest_price]
    if not eligible:
        return None
    return min(eligible, key=lambda region: _region_midpoint(region))


def _nearest_region_strength(
    target_region: tuple[float, float] | None,
    regions: list[tuple[float, float]],
    strengths: list[int],
) -> int:
    if target_region is None:
        return 0
    for index, region in enumerate(regions):
        if region == target_region:
            return strengths[index]
    return 0


def _distance_pct(reference: float | None, latest_price: float) -> float | None:
    if reference is None or reference == 0:
        return None
    return round(((latest_price - reference) / reference) * 100, 2)


def _region_midpoint(region: tuple[float, float]) -> float:
    return (region[0] + region[1]) / 2


def _format_region(region: tuple[float, float]) -> str:
    return f"{region[0]:.2f}-{region[1]:.2f}"
