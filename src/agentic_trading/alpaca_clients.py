from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any

from alpaca.data.historical import NewsClient
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from .config import Settings
from .utils import normalize_bars_frame


class AlpacaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def stock_data_client(self) -> StockHistoricalDataClient:
        self.settings.credentials.require()
        return StockHistoricalDataClient(
            api_key=self.settings.credentials.api_key,
            secret_key=self.settings.credentials.api_secret,
        )

    @cached_property
    def trading_client(self) -> TradingClient:
        self.settings.credentials.require()
        return TradingClient(
            api_key=self.settings.credentials.api_key,
            secret_key=self.settings.credentials.api_secret,
            paper=self.settings.credentials.paper_trading,
            url_override=self.settings.credentials.trade_api_url,
        )

    @cached_property
    def news_client(self) -> NewsClient:
        self.settings.credentials.require()
        return NewsClient(
            api_key=self.settings.credentials.api_key,
            secret_key=self.settings.credentials.api_secret,
        )

    def fetch_stock_bars(
        self,
        symbols: list[str],
        timeframe: TimeFrame | TimeFrameUnit,
        days: int,
        timeframe_multiplier: int = 1,
    ) -> dict[str, Any]:
        today = datetime.now(timezone.utc)
        if timeframe_multiplier <= 0:
            raise ValueError("timeframe_multiplier must be a positive integer")

        resolved_timeframe = (
            TimeFrame(amount=timeframe_multiplier, unit=timeframe)
            if isinstance(timeframe, TimeFrameUnit)
            else timeframe
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=resolved_timeframe,
            start=today - timedelta(days=days),
        )
        frame = self.stock_data_client.get_stock_bars(request).df
        return {symbol: normalize_bars_frame(frame, symbol) for symbol in symbols}

    def fetch_latest_trade(self, symbol: str) -> float:
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = self.stock_data_client.get_stock_latest_trade(request)
        return float(response[symbol].price)

    def fetch_recent_news(self, symbol: str, days: int, limit: int = 10) -> list[dict[str, Any]]:
        request = NewsRequest(
            symbols=symbol,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            end=datetime.now(timezone.utc),
            limit=limit,
            include_content=True,
            exclude_contentless=False,
            sort="desc",
        )
        news_set = self.news_client.get_news(request)
        articles = getattr(news_set, "data", {}).get("news", [])
        normalized: list[dict[str, Any]] = []
        for article in articles:
            normalized.append(
                {
                    "id": article.id,
                    "headline": article.headline,
                    "summary": article.summary,
                    "content": article.content,
                    "updated_at": article.updated_at,
                    "created_at": article.created_at,
                    "source": article.source,
                    "author": article.author,
                    "symbols": article.symbols,
                    "url": article.url,
                }
            )
        return normalized

    def get_clock(self) -> Any:
        return self.trading_client.get_clock()

    def get_buying_power(self) -> float:
        return float(self.trading_client.get_account().buying_power)

    def get_open_position(self, symbol: str) -> tuple[bool, int, float | None]:
        try:
            position = self.trading_client.get_open_position(symbol)
        except Exception:
            return False, 0, None
        return True, int(float(position.qty)), float(position.avg_entry_price)

    def submit_market_order(self, symbol: str, qty: int, side: OrderSide) -> Any:
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
        )
        return self.trading_client.submit_order(request)
