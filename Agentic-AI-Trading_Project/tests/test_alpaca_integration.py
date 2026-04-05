from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit

import project.workflow as workflow_module
from project.alpaca_integration import (
    AlpacaAccountClient,
    AlpacaBrokerClient,
    AlpacaMarketDataClient,
    AlpacaService,
)
from project.data_collection import HistoricalReplayDataClient
from project.config import AlpacaConfig
from project.execution import ExecutionModule
from project.storage import InMemoryResultStore


def test_alpaca_service_fetches_and_normalizes_five_minute_bars() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    stock_client = FakeStockHistoricalDataClient(_make_alpaca_bar_frame())
    service.__dict__["stock_data_client"] = stock_client

    frame = service.fetch_five_minute_bars("AAPL", lookback_days=15)

    assert stock_client.last_request is not None
    assert stock_client.last_request.timeframe.amount == 5
    assert stock_client.last_request.timeframe.unit == TimeFrameUnit.Minute
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert str(frame.index.tz) == "UTC"
    assert len(frame) == 4


def test_alpaca_historical_market_data_client_fetches_hourly_bars() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    stock_client = FakeStockHistoricalDataClient(_make_alpaca_bar_frame_with_freq("1h"))
    service.__dict__["stock_data_client"] = stock_client

    frame = AlpacaMarketDataClient(service, hourly_lookback_days=30).fetch_hourly_bars("AAPL")

    assert stock_client.last_request is not None
    assert stock_client.last_request.timeframe.amount == 1
    assert stock_client.last_request.timeframe.unit == TimeFrameUnit.Hour
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_alpaca_historical_market_data_client_fetches_price_at_or_before() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    stock_client = FakeStockHistoricalDataClient(_make_alpaca_bar_frame())
    service.__dict__["stock_data_client"] = stock_client

    price = AlpacaMarketDataClient(service, five_minute_lookback_days=30).get_price_at_or_before(
        "AAPL",
        "2025-01-01T00:07:00Z",
    )

    assert stock_client.last_request is not None
    assert stock_client.last_request.timeframe.amount == 5
    assert stock_client.last_request.timeframe.unit == TimeFrameUnit.Minute
    assert price == 101.5


def test_alpaca_account_client_reads_buying_power_position_and_clock() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    service.__dict__["trading_client"] = FakeTradingClient(position_qty="3")

    account = AlpacaAccountClient(service).get_account_state("AAPL")

    assert account.cash == 25_000.0
    assert account.position_qty == 3
    assert account.market_open is True


def test_alpaca_account_client_returns_zero_when_no_position_exists() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    service.__dict__["trading_client"] = FakeTradingClient(position_qty=None)

    account = AlpacaAccountClient(service).get_account_state("AAPL")

    assert account.position_qty == 0


def test_alpaca_broker_client_submits_market_order() -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)
    trading_client = FakeTradingClient(position_qty=None)
    service.__dict__["trading_client"] = trading_client

    order_id = AlpacaBrokerClient(service).place_order("AAPL", 2, "BUY")

    assert order_id == "alpaca-order-1"
    assert trading_client.last_submitted_order is not None
    assert trading_client.last_submitted_order.symbol == "AAPL"
    assert trading_client.last_submitted_order.qty == 2
    assert trading_client.last_submitted_order.side.value == "buy"


def test_alpaca_service_get_current_price_uses_yfinance(monkeypatch) -> None:
    config = AlpacaConfig(api_key="key", api_secret="secret")
    service = AlpacaService(config)

    fake_ticker = SimpleNamespace(info={"currentPrice": 173.45})
    fake_yfinance = SimpleNamespace(Tickers=lambda symbols: SimpleNamespace(tickers={"AAPL": fake_ticker}))
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    price = service.get_current_price("AAPL")

    assert price == 173.45


def test_build_workflow_uses_alpaca_clients_when_notebook_credentials_are_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    class DummyAlpacaService:
        def __init__(self, config: AlpacaConfig) -> None:
            self.config = config

        def fetch_five_minute_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="5min", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def fetch_hourly_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="1h", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def get_account_state(self, symbol: str):
            return SimpleNamespace(cash=10_000.0, position_qty=0, market_open=True)

        def submit_market_order(self, symbol: str, qty: int, side: str) -> str:
            return "dummy-order"

    monkeypatch.setattr(workflow_module, "AlpacaService", DummyAlpacaService)
    monkeypatch.setattr(workflow_module, "SQLiteResultStore", lambda _: InMemoryResultStore())

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert isinstance(workflow.data_collection.market_data_client, AlpacaMarketDataClient)
    assert isinstance(workflow.data_collection.account_client, AlpacaAccountClient)
    assert isinstance(workflow.execution_module, ExecutionModule)
    assert isinstance(workflow.execution_module.broker_client, AlpacaBrokerClient)


def test_build_workflow_forces_alpaca_provider_for_live_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("RUN_MODE", "live")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alpha_vantage")

    class DummyAlpacaService:
        def __init__(self, config: AlpacaConfig) -> None:
            self.config = config

        def fetch_five_minute_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="5min", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def fetch_hourly_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="1h", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def get_account_state(self, symbol: str):
            return SimpleNamespace(cash=10_000.0, position_qty=0, market_open=True)

        def submit_market_order(self, symbol: str, qty: int, side: str) -> str:
            return "dummy-order"

    monkeypatch.setattr(workflow_module, "AlpacaService", DummyAlpacaService)
    monkeypatch.setattr(workflow_module, "SQLiteResultStore", lambda _: InMemoryResultStore())

    workflow = workflow_module.build_workflow(project_root=tmp_path)

    assert workflow.settings.market_data.provider == "alpaca"
    assert isinstance(workflow.data_collection.market_data_client, AlpacaMarketDataClient)


def test_build_workflow_uses_historical_alpaca_client_in_backtest_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    class DummyAlpacaService:
        def __init__(self, config: AlpacaConfig) -> None:
            self.config = config

        def fetch_five_minute_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="5min", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def fetch_hourly_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
            index = pd.date_range("2025-01-01", periods=24, freq="1h", tz="UTC")
            return pd.DataFrame(
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                },
                index=index,
            )

        def get_account_state(self, symbol: str):
            return SimpleNamespace(cash=10_000.0, position_qty=0, market_open=True)

        def submit_market_order(self, symbol: str, qty: int, side: str) -> str:
            return "dummy-order"

    monkeypatch.setattr(workflow_module, "AlpacaService", DummyAlpacaService)
    monkeypatch.setattr(workflow_module, "SQLiteResultStore", lambda _: InMemoryResultStore())

    workflow = workflow_module.build_workflow(project_root=tmp_path, mode="backtest")

    assert isinstance(workflow.data_collection.market_data_client, HistoricalReplayDataClient)


class FakeStockHistoricalDataClient:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.last_request = None

    def get_stock_bars(self, request):
        self.last_request = request
        return SimpleNamespace(df=self.frame)


class FakeTradingClient:
    def __init__(self, position_qty: str | None) -> None:
        self.position_qty = position_qty
        self.last_submitted_order = None

    def get_account(self):
        return SimpleNamespace(buying_power="25000")

    def get_clock(self):
        return SimpleNamespace(is_open=True)

    def get_open_position(self, symbol: str):
        if self.position_qty is None:
            raise RuntimeError("no position")
        return SimpleNamespace(qty=self.position_qty)

    def submit_order(self, request):
        self.last_submitted_order = request
        return SimpleNamespace(id="alpaca-order-1")


def _make_alpaca_bar_frame() -> pd.DataFrame:
    return _make_alpaca_bar_frame_with_freq("5min")


def _make_alpaca_bar_frame_with_freq(freq: str) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [["AAPL"], pd.date_range("2025-01-01", periods=4, freq=freq, tz="UTC")],
        names=["symbol", "timestamp"],
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1000, 1001, 1002, 1003],
        },
        index=index,
    )
