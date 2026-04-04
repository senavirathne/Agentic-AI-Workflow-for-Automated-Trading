"""Adapters for fetching market data and placing orders through Alpaca."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any

import pandas as pd

from .config import AlpacaConfig
from .data_collection import _close_price_at_or_before, _normalize_ohlcv_bars
from .models import AccountState


@dataclass
class AlpacaService:
    """Wrap Alpaca SDK clients behind a workflow-friendly interface."""

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

    def fetch_price_at_or_before(
        self,
        symbol: str,
        timestamp: pd.Timestamp | str,
        *,
        lookback_days: int | None = None,
    ) -> float:
        frame = self.fetch_five_minute_bars(symbol, lookback_days)
        return _close_price_at_or_before(frame, timestamp)

    def fetch_latest_trade(self, symbol: str) -> float:
        from alpaca.data.historical.stock import StockLatestTradeRequest

        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = self.stock_data_client.get_stock_latest_trade(request)
        return float(response[symbol].price)

    def get_current_price(self, symbol: str) -> float:
        import yfinance as yf

        normalized_symbol = symbol.strip().upper()
        tickers = yf.Tickers(normalized_symbol)
        ticker = tickers.tickers.get(normalized_symbol)
        if ticker is None:
            raise RuntimeError(f"Could not find current price for {normalized_symbol} via yfinance.")

        info = getattr(ticker, "info", {}) or {}
        price = info.get("currentPrice")
        if price is None:
            price = info.get("regularMarketPrice")
        if price is None:
            raise RuntimeError(f"yfinance did not return a current price for {normalized_symbol}.")
        return float(price)

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
        position_open, position_qty, avg_entry_price = self.get_open_position(symbol)
        clock = self.get_clock()
        account = self.trading_client.get_account()
        trade_count, realized_profit = self._trade_metrics(symbol)
        buying_power = _optional_float(getattr(account, "buying_power", None))
        cash = _optional_float(getattr(account, "cash", None))
        return AccountState(
            cash=float(buying_power if buying_power is not None else (cash or 0.0)),
            position_qty=position_qty,
            market_open=bool(clock.is_open),
            avg_entry_price=avg_entry_price if position_open else None,
            realized_profit=realized_profit,
            trade_count=trade_count,
        )

    def _trade_metrics(self, symbol: str) -> tuple[int, float]:
        orders = self._filled_orders(symbol)
        if not orders:
            return 0, 0.0

        realized_profit = 0.0
        open_qty = 0.0
        avg_entry_price: float | None = None
        trade_count = 0

        for order in orders:
            filled_qty = _optional_float(getattr(order, "filled_qty", None))
            filled_price = _optional_float(getattr(order, "filled_avg_price", None))
            side = str(getattr(order, "side", "") or "").strip().upper()
            if filled_qty is None or filled_qty <= 0 or filled_price is None or side not in {"BUY", "SELL"}:
                continue

            trade_count += 1
            if side == "BUY":
                existing_cost = 0.0 if avg_entry_price is None else open_qty * avg_entry_price
                open_qty += filled_qty
                avg_entry_price = (existing_cost + (filled_qty * filled_price)) / open_qty if open_qty > 0 else None
                continue

            if open_qty > 0 and avg_entry_price is not None:
                sold_qty = min(filled_qty, open_qty)
                realized_profit += (filled_price - avg_entry_price) * sold_qty
                open_qty -= sold_qty
                if open_qty == 0:
                    avg_entry_price = None

        return trade_count, round(realized_profit, 2)

    def _filled_orders(self, symbol: str) -> list[Any]:
        from alpaca.trading.enums import OrderStatus, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[symbol.upper()],
            limit=500,
            nested=False,
        )
        try:
            orders = self.trading_client.get_orders(filter=request)
        except Exception:
            return []

        filled_orders = [
            order
            for order in orders or []
            if str(getattr(order, "status", "") or "").strip().lower() == OrderStatus.FILLED.value
            and _optional_float(getattr(order, "filled_qty", None)) not in {None, 0.0}
        ]
        return sorted(
            filled_orders,
            key=lambda order: _order_timestamp(order),
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

    def sync_risk_orders(
        self,
        symbol: str,
        qty: int,
        *,
        stop_loss_price: float | None,
        take_profit_price: float | None,
    ) -> list[str]:
        self.clear_risk_orders(symbol)
        if qty <= 0:
            return []
        if stop_loss_price is None and take_profit_price is None:
            return []

        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, StopOrderRequest, TakeProfitRequest

        order_ids: list[str] = []
        if stop_loss_price is not None and take_profit_price is not None:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=round(float(take_profit_price), 2)),
                stop_loss=StopLossRequest(stop_price=round(float(stop_loss_price), 2)),
                client_order_id=_managed_risk_order_id(symbol, "oco"),
            )
            order = self.trading_client.submit_order(request)
            order_ids.append(str(getattr(order, "id", "")))
            return order_ids

        if stop_loss_price is not None:
            stop_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(float(stop_loss_price), 2),
                client_order_id=_managed_risk_order_id(symbol, "stop"),
            )
            stop_order = self.trading_client.submit_order(stop_request)
            order_ids.append(str(getattr(stop_order, "id", "")))
        if take_profit_price is not None:
            limit_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=round(float(take_profit_price), 2),
                client_order_id=_managed_risk_order_id(symbol, "tp"),
            )
            limit_order = self.trading_client.submit_order(limit_request)
            order_ids.append(str(getattr(limit_order, "id", "")))
        return order_ids

    def clear_risk_orders(self, symbol: str) -> None:
        from alpaca.trading.enums import OrderSide, QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            side=OrderSide.SELL,
            symbols=[symbol.upper()],
        )
        try:
            open_orders = self.trading_client.get_orders(filter=request)
        except Exception:
            return
        prefix = _managed_risk_order_prefix(symbol)
        for order in open_orders or []:
            client_order_id = str(getattr(order, "client_order_id", "") or "")
            order_id = getattr(order, "id", None)
            if not client_order_id.startswith(prefix) or order_id is None:
                continue
            try:
                self.trading_client.cancel_order_by_id(order_id)
            except Exception:
                continue


@dataclass
class AlpacaMarketDataClient:
    """Market-data adapter that fetches recent Alpaca bar history."""

    service: AlpacaService
    five_minute_lookback_days: int = 2
    hourly_lookback_days: int = 7

    def fetch_five_minute_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_five_minute_bars(symbol, self.five_minute_lookback_days)

    def fetch_hourly_bars(self, symbol: str) -> pd.DataFrame:
        return self.service.fetch_hourly_bars(symbol, self.hourly_lookback_days)

    def get_price_at_or_before(self, symbol: str, timestamp: pd.Timestamp | str) -> float:
        return self.service.fetch_price_at_or_before(
            symbol,
            timestamp,
            lookback_days=self.five_minute_lookback_days,
        )


@dataclass
class AlpacaAccountClient:
    """Expose Alpaca account state through the workflow account interface."""

    service: AlpacaService

    def get_account_state(self, symbol: str) -> AccountState:
        return self.service.get_account_state(symbol)


@dataclass
class AlpacaBrokerClient:
    """Expose Alpaca order placement through the workflow broker interface."""

    service: AlpacaService

    def place_order(self, symbol: str, qty: int, side: str) -> str:
        return self.service.submit_market_order(symbol, qty, side)

    def sync_risk_orders(
        self,
        symbol: str,
        qty: int,
        *,
        stop_loss_price: float | None,
        take_profit_price: float | None,
    ) -> list[str]:
        return self.service.sync_risk_orders(
            symbol,
            qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

    def clear_risk_orders(self, symbol: str) -> None:
        self.service.clear_risk_orders(symbol)


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


def _managed_risk_order_prefix(symbol: str) -> str:
    return f"risk-{symbol.strip().lower()}-"


def _managed_risk_order_id(symbol: str, suffix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{_managed_risk_order_prefix(symbol)}{suffix}-{timestamp}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_timestamp(order: Any) -> pd.Timestamp:
    for field_name in ("filled_at", "updated_at", "submitted_at", "created_at"):
        value = getattr(order, field_name, None)
        if value is None:
            continue
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
    return pd.Timestamp(datetime.min, tz="UTC")
