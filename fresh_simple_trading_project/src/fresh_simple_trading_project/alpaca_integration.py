from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any

import pandas as pd

from .config import AlpacaConfig
from .data_collection import _normalize_ohlcv_bars
from .models import AccountState, NewsArticle


@dataclass
class AlpacaService:
    config: AlpacaConfig

    @cached_property
    def stock_data_client(self) -> Any:
        self.config.require()
        from alpaca.data.historical.stock import StockHistoricalDataClient

        return StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
        )

    @cached_property
    def trading_client(self) -> Any:
        self.config.require()
        from alpaca.trading.client import TradingClient

        return TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
            paper=self.config.paper_trading,
            url_override=self.config.trade_api_url,
        )

    @cached_property
    def news_client(self) -> Any:
        self.config.require()
        from alpaca.data.historical import NewsClient

        return NewsClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
        )

    def fetch_five_minute_bars(self, symbol: str, lookback_days: int=None) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = datetime.now(timezone.utc)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(amount=5, unit=TimeFrameUnit.Minute),
            start=end - timedelta(days=lookback_days) if lookback_days else None,         
        )
        frame = self.stock_data_client.get_stock_bars(request).df
        return _normalize_alpaca_bars(frame, symbol)

    def fetch_hourly_bars(self, symbol: str, lookback_days: int=None) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        end = datetime.now(timezone.utc)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Hour),
            start=end - timedelta(days=lookback_days) if lookback_days else None,           
        )
        frame = self.stock_data_client.get_stock_bars(request).df
        return _normalize_alpaca_bars(frame, symbol)

    def fetch_latest_trade(self, symbol: str) -> float:
        from alpaca.data.historical.stock import StockLatestTradeRequest

        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = self.stock_data_client.get_stock_latest_trade(request)
        return float(response[symbol].price)

    def fetch_news(
        self,
        *,
        symbols: str | list[str] | None = None,
        days: int = 7,
        limit: int = 10,
    ) -> list[NewsArticle]:
        from alpaca.data.requests import NewsRequest

        normalized_symbols: str | None
        if isinstance(symbols, list):
            normalized_symbols = ",".join(symbol.strip().upper() for symbol in symbols if symbol.strip()) or None
        elif isinstance(symbols, str):
            normalized_symbols = symbols.strip().upper() or None
        else:
            normalized_symbols = None

        request = NewsRequest(
            symbols=normalized_symbols,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            limit=limit,
            include_content=True,
            exclude_contentless=False,
            sort="desc",
        )
        news_set = self.news_client.get_news(request)
        articles = getattr(news_set, "data", {}).get("news", [])
        normalized: list[NewsArticle] = []
        for article in articles:
            normalized.append(
                NewsArticle(
                    headline=str(getattr(article, "headline", "") or "").strip(),
                    summary=str(getattr(article, "summary", "") or "").strip(),
                    source=str(getattr(article, "source", "") or "").strip(),
                    url=str(getattr(article, "url", "") or "").strip(),
                    published_at=_to_isoformat(getattr(article, "updated_at", None)),
                )
            )
        return [article for article in normalized if article.headline]

    def fetch_recent_news(self, symbol: str, days: int = 7, limit: int = 10) -> list[NewsArticle]:
        return self.fetch_news(symbols=symbol, days=days, limit=limit)

    def get_clock(self) -> Any:
        return self.trading_client.get_clock()

    def get_buying_power(self) -> float:
        return float(self.trading_client.get_account().buying_power)

    def get_open_position(self, symbol: str) -> tuple[bool, int, float | None]:
        try:
            position = self.trading_client.get_open_position(symbol)
        except Exception:
            return False, 0, None
        avg_entry_price = getattr(position, "avg_entry_price", None)
        return True, int(float(position.qty)), (float(avg_entry_price) if avg_entry_price is not None else None)

    def get_account_state(self, symbol: str) -> AccountState:
        position_open, position_qty, _ = self.get_open_position(symbol)
        clock = self.get_clock()
        return AccountState(
            cash=self.get_buying_power(),
            position_qty=position_qty,
            market_open=bool(clock.is_open),
        )

    def submit_market_order(self, symbol: str, qty: int, side: str) -> str:
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side_name = side.upper()
        if side_name not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported order side: {side}")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side_name == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        order = self.trading_client.submit_order(request)
        return str(getattr(order, "id", ""))


@dataclass
class AlpacaMarketDataClient:
    service: AlpacaService
    five_minute_lookback_days: int = 2
    hourly_lookback_days: int = 7

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_five_minute_bars(symbol, self.five_minute_lookback_days)

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_hourly_bars(symbol, self.hourly_lookback_days)


@dataclass
class AlpacaHistoricalMarketDataClient:
    service: AlpacaService
    history_lookback_days: int = 30

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_five_minute_bars(symbol, self.history_lookback_days)

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_hourly_bars(symbol, self.history_lookback_days)


@dataclass
class AlpacaAccountClient:
    service: AlpacaService

    def get_account_state(self, symbol: str) -> AccountState:
        return self.service.get_account_state(symbol)


@dataclass
class AlpacaBrokerClient:
    service: AlpacaService

    def place_order(self, symbol: str, qty: int, side: str) -> str:
        return self.service.submit_market_order(symbol, qty, side)


def _normalize_alpaca_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    normalized = frame.copy()
    if isinstance(normalized.index, pd.MultiIndex) and "symbol" in normalized.index.names:
        normalized = normalized.xs(symbol, level="symbol", drop_level=True)

    if "symbol" in normalized.columns:
        normalized = normalized[normalized["symbol"] == symbol]

    if "timestamp" not in normalized.columns:
        normalized = normalized.reset_index()
    if "timestamp" not in normalized.columns:
        fallback_column = next(
            (column for column in normalized.columns if str(column).lower() in {"index", "level_0"}),
            normalized.columns[0],
        )
        normalized = normalized.rename(columns={fallback_column: "timestamp"})

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).set_index("timestamp")
    return _normalize_ohlcv_bars(normalized)


def _to_isoformat(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)
