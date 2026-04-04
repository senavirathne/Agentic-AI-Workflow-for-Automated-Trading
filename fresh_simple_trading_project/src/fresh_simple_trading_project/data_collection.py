"""Market data and account-state collection helpers for workflow inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .config import TradingConfig
from .models import AccountState, CollectedMarketData


class MarketDataClient(Protocol):
    """Protocol for market-data providers that return bar history."""

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        ...

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        ...

    def get_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        ...


class AccountClient(Protocol):
    """Protocol for account-state providers used by the workflow."""

    def get_account_state(self, symbol: str) -> AccountState:
        ...


@dataclass
class SimulatedAccountClient:
    """In-memory account model that updates itself on simulated fills."""

    cash: float
    position_qty: int = 0
    market_open: bool = True
    avg_entry_price: float | None = None
    realized_profit: float = 0.0
    trade_count: int = 0

    def get_account_state(self, symbol: str) -> AccountState:
        return AccountState(
            cash=float(self.cash),
            position_qty=int(self.position_qty),
            market_open=self.market_open,
            avg_entry_price=self.avg_entry_price,
            realized_profit=float(self.realized_profit),
            trade_count=int(self.trade_count),
        )

    def apply_fill(self, symbol: str, qty: int, side: str, price: float) -> AccountState:
        normalized_side = side.strip().upper()
        quantity = max(int(qty), 0)
        if quantity <= 0:
            return self.get_account_state(symbol)
        if normalized_side == "BUY":
            cost = float(quantity) * float(price)
            if cost > self.cash:
                raise ValueError(
                    f"Simulated account rejected BUY {quantity} {symbol} at {price:.2f}: insufficient cash."
                )
            if self.position_qty > 0 and self.avg_entry_price is not None:
                existing_cost = self.position_qty * self.avg_entry_price
                self.avg_entry_price = (existing_cost + cost) / (self.position_qty + quantity)
            else:
                self.avg_entry_price = float(price)
            self.cash -= cost
            self.position_qty += quantity
            self.trade_count += 1
            return self.get_account_state(symbol)
        if normalized_side == "SELL":
            if quantity > self.position_qty:
                raise ValueError(
                    f"Simulated account rejected SELL {quantity} {symbol}: position size is {self.position_qty}."
                )
            if self.avg_entry_price is not None:
                self.realized_profit += (float(price) - self.avg_entry_price) * quantity
            self.cash += float(quantity) * float(price)
            self.position_qty -= quantity
            self.trade_count += 1
            if self.position_qty == 0:
                self.avg_entry_price = None
            return self.get_account_state(symbol)
        raise ValueError(f"Unsupported simulated order side: {side}")


@dataclass
class StaticMarketDataClient:
    """Simple market-data client backed by a fixed dataframe."""

    frame: pd.DataFrame

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        return _normalize_ohlcv_bars(self.frame.copy())

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        return resample_to_hourly_bars(self.fetch_five_minute_bars(symbol))

    def get_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        return _close_price_at_or_before(self.fetch_five_minute_bars(symbol), timestamp)


@dataclass
class HistoricalReplayDataClient:
    """Replays historical bars through a movable virtual clock."""

    five_min_history: pd.DataFrame
    hourly_history: pd.DataFrame
    current_time: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.five_min_history = _normalize_ohlcv_bars(self.five_min_history)
        if self.hourly_history.empty:
            self.hourly_history = resample_to_hourly_bars(self.five_min_history)
        else:
            self.hourly_history = _normalize_ohlcv_bars(self.hourly_history)
        if self.current_time is not None:
            self.current_time = _normalize_end_timestamp(self.current_time)

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        return self._slice_until(self.five_min_history)

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        return self._slice_until(self.hourly_history)

    def get_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        return _close_price_at_or_before(self._slice_until(self.five_min_history), timestamp)

    def advance_to(self, timestamp: pd.Timestamp) -> None:
        self.current_time = _normalize_end_timestamp(timestamp)

    def reset(self) -> None:
        self.current_time = None

    def _slice_until(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.current_time is None:
            return frame.copy()
        return frame.loc[frame.index <= self.current_time].copy()


@dataclass
class DataCollectionModule:
    """Assemble market and account data into workflow-ready snapshots."""

    market_data_client: MarketDataClient
    account_client: AccountClient

    def fetch_history(self, symbol: str) -> pd.DataFrame:
        return self.fetch_indicator_bars(symbol)

    def fetch_five_minute_history(self, symbol: str) -> pd.DataFrame:
        return self.market_data_client.fetch_five_minute_bars(symbol)

    def fetch_hourly_history(self, symbol: str) -> pd.DataFrame:
        return self.market_data_client.fetch_hourly_bars(symbol)

    def fetch_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        return float(self.market_data_client.get_price_at_or_before(symbol, timestamp))

    def fetch_sr_bars(
        self,
        symbol: str,
        lookback_days: int = 7,
        *,
        end_time: pd.Timestamp | str | None = None,
        source_hourly_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        hourly = source_hourly_bars if source_hourly_bars is not None else self.fetch_hourly_history(symbol)
        return _slice_window(hourly, end_time=end_time, window=pd.Timedelta(days=lookback_days))

    def fetch_indicator_bars(
        self,
        symbol: str,
        lookback_hours: int = 24,
        *,
        end_time: pd.Timestamp | str | None = None,
        source_five_minute_bars: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        five_minute = (
            source_five_minute_bars
            if source_five_minute_bars is not None
            else self.fetch_five_minute_history(symbol)
        )
        return _slice_window(five_minute, end_time=end_time, window=pd.Timedelta(hours=lookback_hours))

    def collect(
        self,
        symbol: str,
        config: TradingConfig,
    ) -> CollectedMarketData:
        return self.collect_until(
            symbol,
            config,
        )

    def collect_until(
        self,
        symbol: str,
        config: TradingConfig,
        end_time: pd.Timestamp | str | None = None,
        source_five_minute_bars: pd.DataFrame | None = None,
        source_hourly_bars: pd.DataFrame | None = None,
    ) -> CollectedMarketData:
        five_minute_bars = self.fetch_indicator_bars(
            symbol,
            lookback_hours=config.indicator_lookback_hours,
            end_time=end_time,
            source_five_minute_bars=source_five_minute_bars,
        )
        if five_minute_bars.empty:
            raise ValueError(f"No 5-minute bars available for {symbol}")
        hourly_bars = self.fetch_sr_bars(
            symbol,
            lookback_days=config.sr_lookback_days,
            end_time=end_time,
            source_hourly_bars=source_hourly_bars,
        )
        if hourly_bars.empty:
            raise ValueError(f"No 1-hour bars available for {symbol}")
        return CollectedMarketData(
            symbol=symbol,
            five_minute_bars=five_minute_bars,
            hourly_bars=hourly_bars,
            account=self.account_client.get_account_state(symbol),
        )


def resample_to_hourly_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample a 5-minute OHLCV frame into hourly bars."""
    aggregated = frame.resample("1h").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def _normalize_end_timestamp(end_time: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if end_time is None:
        return None
    end_timestamp = pd.Timestamp(end_time)
    if end_timestamp.tzinfo is None:
        return end_timestamp.tz_localize("UTC")
    return end_timestamp.tz_convert("UTC")


def _slice_window(
    frame: pd.DataFrame,
    *,
    end_time: pd.Timestamp | str | None,
    window: pd.Timedelta,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    end_timestamp = _normalize_end_timestamp(end_time)
    if end_timestamp is None:
        end_timestamp = pd.Timestamp(frame.index[-1]).tz_convert("UTC")
    cutoff = end_timestamp - window
    return frame.loc[(frame.index >= cutoff) & (frame.index <= end_timestamp)].copy()


def _close_price_at_or_before(
    frame: pd.DataFrame,
    timestamp: pd.Timestamp | str,
) -> float:
    normalized = _normalize_ohlcv_bars(frame)
    target = _normalize_end_timestamp(timestamp)
    if target is None:
        raise ValueError("A timestamp is required to fetch a historical price.")
    eligible = normalized.loc[normalized.index <= target]
    if eligible.empty:
        raise ValueError(f"No OHLCV bar is available at or before {target.isoformat()}.")
    return float(eligible.iloc[-1]["close"])


def _normalize_ohlcv_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["open", "high", "low", "close", "volume"]
    normalized = frame.copy().sort_index()
    if normalized.index.tz is None:
        normalized.index = pd.to_datetime(normalized.index, utc=True)
    else:
        normalized.index = pd.to_datetime(normalized.index, utc=True)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"OHLCV bars are missing columns: {missing}")
    normalized = normalized[required_columns].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized["volume"] = normalized["volume"].fillna(0.0)
    return normalized
