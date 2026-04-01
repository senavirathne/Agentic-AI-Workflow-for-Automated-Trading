from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency during bootstrap.
    def load_dotenv(dotenv_path: str | Path | None = None) -> bool:
        return False


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class RunMode(str, Enum):
    LIVE = "live"
    BACKTEST = "backtest"


@dataclass(frozen=True)
class TradingConfig:
    symbol: str = "AAPL"
    mode: RunMode = RunMode.LIVE
    lookback_hours: int = 240
    eda_window_hours: int = 24
    sr_lookback_days: int = 7
    sr_timeframe: str = "1h"
    indicator_lookback_hours: int = 24
    indicator_timeframe: str = "5min"
    backtest_history_days: int = 30
    live_sleep_seconds: float = 3600.0
    backtest_sleep_seconds: float = 1.0
    source_interval_minutes: int = 5
    loop_interval_minutes: int = 60
    short_ma: int = 10
    long_ma: int = 30
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    buy_rsi_threshold: float = 55.0
    sell_rsi_threshold: float = 45.0
    max_position_pct: float = 0.10
    starting_cash: float = 10_000.0
    news_limit: int = 8
    risk_volatility_cutoff: float = 0.03

    @property
    def support_resistance_lookback_days(self) -> int:
        return self.sr_lookback_days

    @property
    def sleep_seconds(self) -> float:
        if self.mode == RunMode.BACKTEST:
            return self.backtest_sleep_seconds
        return self.live_sleep_seconds


@dataclass(frozen=True)
class MarketDataConfig:
    provider: str = "synthetic"


@dataclass(frozen=True)
class RawStoreConfig:
    provider: str = "local"


@dataclass(frozen=True)
class ResultStoreConfig:
    provider: str = "sqlite"
    database_url: str | None = None


@dataclass(frozen=True)
class AzureConfig:
    storage_account_url: str | None = None
    storage_connection_string: str | None = None
    blob_container_raw: str = "raw"
    blob_prefix: str = ""
    key_vault_url: str | None = None


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str | None = None
    api_secret: str | None = None
    paper_trading: bool = True
    trade_api_url: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def require(self) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Missing Alpaca API credentials. Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY."
            )


@dataclass(frozen=True)
class NewsConfig:
    max_age_days: int = 7


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-reasoner"
    timeout_seconds: float = 45.0
    show_progress: bool = True
    heartbeat_seconds: float = 5.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def require(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "Missing DeepSeek API credentials. Set DEEPSEEK_API_KEY in fresh_simple_trading_project/.env."
            )


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    reports_dir: Path
    database_path: Path

    def create_directories(self) -> None:
        for directory in [self.data_dir, self.raw_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    trading: TradingConfig
    market_data: MarketDataConfig
    raw_store: RawStoreConfig
    result_store: ResultStoreConfig
    azure: AzureConfig
    alpaca: AlpacaConfig
    news: NewsConfig
    llm: LLMConfig
    paths: Paths

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = Path(project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env")
        paths = Paths(
            project_root=root,
            data_dir=root / "data",
            raw_dir=root / "data" / "raw",
            reports_dir=root / "reports",
            database_path=root / "data" / "workflow.sqlite",
        )
        paths.create_directories()
        alpaca = AlpacaConfig(
            api_key=os.getenv("ALPACA_PAPER_API_KEY"),
            api_secret=os.getenv("ALPACA_PAPER_SECRET_KEY"),
            paper_trading=_parse_bool(os.getenv("PAPER"), True),
            trade_api_url=os.getenv("TRADE_API_URL") or os.getenv("ALPACA_TRADE_API_URL"),
        )
        market_provider = _resolve_market_provider(alpaca_enabled=alpaca.enabled)
        run_mode = _parse_run_mode(os.getenv("RUN_MODE"))
        legacy_lookback_hours = max(1, int(os.getenv("LOOKBACK_HOURS", "24")))
        legacy_sr_lookback_days = max(
            1,
            int(os.getenv("SUPPORT_RESISTANCE_LOOKBACK_DAYS", "7")),
        )
        return cls(
            trading=TradingConfig(
                symbol=os.getenv("TRADING_SYMBOL", "AAPL").upper(),
                mode=run_mode,
                lookback_hours=max(1, int(os.getenv("LOOKBACK_HOURS", "240"))),
                eda_window_hours=max(1, int(os.getenv("EDA_WINDOW_HOURS", "24"))),
                sr_lookback_days=max(
                    1,
                    int(os.getenv("SR_LOOKBACK_DAYS", str(legacy_sr_lookback_days))),
                ),
                sr_timeframe=os.getenv("SR_TIMEFRAME", "1h"),
                indicator_lookback_hours=max(
                    1,
                    int(os.getenv("INDICATOR_LOOKBACK_HOURS", str(legacy_lookback_hours))),
                ),
                indicator_timeframe=os.getenv("INDICATOR_TIMEFRAME", "5min"),
                backtest_history_days=max(8, int(os.getenv("BACKTEST_HISTORY_DAYS", "30"))),
                live_sleep_seconds=max(0.0, float(os.getenv("LIVE_SLEEP_SECONDS", "3600.0"))),
                backtest_sleep_seconds=max(0.0, float(os.getenv("BACKTEST_SLEEP_SECONDS", "1.0"))),
            ),
            market_data=MarketDataConfig(provider=market_provider),
            raw_store=RawStoreConfig(provider=_resolve_raw_store_provider()),
            result_store=ResultStoreConfig(
                provider=_resolve_result_store_provider(),
                database_url=os.getenv("DATABASE_URL") or _sqlite_database_url(paths.database_path),
            ),
            azure=AzureConfig(
                storage_account_url=os.getenv("AZURE_STORAGE_ACCOUNT_URL"),
                storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
                blob_container_raw=os.getenv("AZURE_BLOB_CONTAINER_RAW", "raw"),
                blob_prefix=os.getenv("AZURE_BLOB_PREFIX", "").strip("/"),
                key_vault_url=os.getenv("AZURE_KEY_VAULT_URL"),
            ),
            alpaca=alpaca,
            news=NewsConfig(
                max_age_days=max(1, int(os.getenv("NEWS_MAX_AGE_DAYS", "7"))),
            ),
            llm=LLMConfig(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
                timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0")),
                show_progress=_parse_bool(os.getenv("LLM_SHOW_PROGRESS"), True),
                heartbeat_seconds=float(os.getenv("LLM_HEARTBEAT_SECONDS", "5.0")),
            ),
            paths=paths,
        )


def _parse_run_mode(raw: str | None) -> RunMode:
    candidate = (raw or RunMode.LIVE.value).strip().lower()
    try:
        return RunMode(candidate)
    except ValueError as exc:
        raise RuntimeError(
            f"Unsupported RUN_MODE={candidate!r}. Use '{RunMode.LIVE.value}' or '{RunMode.BACKTEST.value}'."
        ) from exc


def _resolve_market_provider(*, alpaca_enabled: bool) -> str:
    explicit = os.getenv("MARKET_DATA_PROVIDER")
    if explicit:
        provider = explicit.strip().lower()
        if provider in {"alpaca", "synthetic"}:
            return provider
        raise RuntimeError(
            f"Unsupported MARKET_DATA_PROVIDER={provider!r}. Use 'alpaca' or 'synthetic'."
        )
    if alpaca_enabled:
        return "alpaca"
    return "synthetic"


def _resolve_raw_store_provider() -> str:
    provider = os.getenv("RAW_STORE_PROVIDER", "local").strip().lower()
    if provider in {"local", "azure_blob"}:
        return provider
    raise RuntimeError(
        f"Unsupported RAW_STORE_PROVIDER={provider!r}. Use 'local' or 'azure_blob'."
    )


def _resolve_result_store_provider() -> str:
    provider = os.getenv("RESULT_STORE_PROVIDER", "sqlite").strip().lower()
    if provider in {"sqlite", "azure_sql"}:
        return provider
    raise RuntimeError(
        f"Unsupported RESULT_STORE_PROVIDER={provider!r}. Use 'sqlite' or 'azure_sql'."
    )


def _sqlite_database_url(database_path: Path) -> str:
    return f"sqlite:///{database_path.resolve()}"
