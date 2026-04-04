"""Configuration models and environment loading helpers for the trading workflow."""

from __future__ import annotations

import os
import secrets
import string
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - optional dependency during bootstrap.
    def dotenv_values(dotenv_path: str | Path | None = None) -> dict[str, str]:
        return {}


def _parse_bool(raw: str | None, default: bool) -> bool:
    """Convert a string-like environment value into a boolean."""
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class RunMode(str, Enum):
    """Supported operating modes for the workflow."""

    LIVE = "live"
    BACKTEST = "backtest"


@dataclass(frozen=True)
class TradingConfig:
    """Core trading, indicator, and loop cadence settings."""

    symbol: str = "AAPL"
    mode: RunMode = RunMode.LIVE
    live_market_data_delay_minutes: int = 15
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
        """Return the configured support and resistance lookback window."""
        return self.sr_lookback_days

    @property
    def sleep_seconds(self) -> float:
        """Return the effective loop sleep for the active run mode."""
        if self.mode == RunMode.BACKTEST:
            return self.backtest_sleep_seconds
        return self.live_sleep_seconds


@dataclass(frozen=True)
class MarketDataConfig:
    """Select the market-data provider used by the workflow."""

    provider: str = "alpha_vantage"


@dataclass(frozen=True)
class RawStoreConfig:
    """Select where raw bars and news artifacts are written."""

    provider: str = "local"


@dataclass(frozen=True)
class ResultStoreConfig:
    """Select where workflow summaries and snapshots are persisted."""

    provider: str = "sqlite"
    database_url: str | None = None


def generate_random_alpha_vantage_api_key(length: int = 13) -> str:
    """Generate a random uppercase alphanumeric placeholder key."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass(frozen=True)
class AlphaVantageConfig:
    """Settings for Alpha Vantage technical-indicator and news calls."""

    api_key: str | None = None
    base_url: str = "https://www.alphavantage.co/query"
    interval: str = "5min"
    request_pause_seconds: float = 15.0
    max_retries: int = 3

    @property
    def generated_api_key(self) -> str:
        """Return a random placeholder key for demos or validation messages."""
        return generate_random_alpha_vantage_api_key()

    @property
    def enabled(self) -> bool:
        """Report whether a real Alpha Vantage API key is configured."""
        return bool(self.api_key)

    def require(self) -> None:
        """Raise when an Alpha Vantage-backed operation has no real credentials."""
        if self.enabled:
            return
        raise RuntimeError(
            "Missing Alpha Vantage credentials. Set ALPHA_VANTAGE_API_KEY. "
            "A 13-character placeholder was generated, but Alpha Vantage requests require a real issued key."
        )


@dataclass(frozen=True)
class AzureConfig:
    """Azure storage and Key Vault configuration."""

    storage_account_url: str | None = None
    storage_connection_string: str | None = None
    blob_container_raw: str = "raw"
    blob_prefix: str = ""
    key_vault_url: str | None = None


@dataclass(frozen=True)
class AlpacaConfig:
    """Credentials and connection settings for Alpaca integration."""

    api_key: str | None = None
    api_secret: str | None = None
    paper_trading: bool = True
    trade_api_url: str | None = None

    @property
    def enabled(self) -> bool:
        """Report whether Alpaca credentials are configured."""
        return bool(self.api_key and self.api_secret)

    def require(self) -> None:
        """Raise when an Alpaca-backed operation has no credentials."""
        if not self.enabled:
            raise RuntimeError(
                "Missing Alpaca API credentials. Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY."
            )


@dataclass(frozen=True)
class NewsConfig:
    """Controls how much historical news can be considered."""

    max_age_days: int = 7


@dataclass(frozen=True)
class LLMConfig:
    """Settings for the text-generation client used by the agents."""

    provider: str = "deepseek"
    provider_label: str = "DeepSeek"
    api_key: str | None = None
    base_url: str | None = "https://api.deepseek.com"
    model: str = "deepseek-reasoner"
    timeout_seconds: float = 45.0
    show_progress: bool = True
    heartbeat_seconds: float = 5.0

    @property
    def enabled(self) -> bool:
        """Report whether LLM credentials are configured."""
        return bool(self.api_key)

    def require(self) -> None:
        """Raise when an LLM-backed operation has no API key."""
        if not self.api_key:
            if self.provider.strip().lower().replace("-", "_") == "openai":
                env_var = "OPENAI_API_KEY"
            else:
                env_var = "DEEPSEEK_API_KEY"
            raise RuntimeError(
                f"Missing {self.provider_label} API credentials. "
                f"Set {env_var} in fresh_simple_trading_project/.env."
            )


@dataclass(frozen=True)
class Paths:
    """Filesystem paths used by the standalone trading project."""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    reports_dir: Path
    database_path: Path

    def create_directories(self) -> None:
        """Ensure the configured data and report directories exist."""
        for directory in [self.data_dir, self.raw_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Complete application settings assembled from environment variables."""

    trading: TradingConfig
    market_data: MarketDataConfig
    raw_store: RawStoreConfig
    result_store: ResultStoreConfig
    alpha_vantage: AlphaVantageConfig
    azure: AzureConfig
    alpaca: AlpacaConfig
    news: NewsConfig
    llm: LLMConfig
    secondary_llm: LLMConfig | None
    paths: Paths

    def require_llm(self) -> None:
        """Raise when neither the primary nor secondary LLM is configured."""
        if self.llm.enabled or (self.secondary_llm is not None and self.secondary_llm.enabled):
            return
        raise RuntimeError(
            "Missing LLM API credentials. "
            "Set DEEPSEEK_API_KEY or OPENAI_API_KEY in fresh_simple_trading_project/.env."
        )

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Build settings from the working directory and optional `.env` file."""
        root = Path(project_root or Path.cwd()).resolve()
        env = _merged_env(root / ".env")
        storage_root = _resolve_storage_root(root, env)
        paths = Paths(
            project_root=root,
            data_dir=storage_root / "data",
            raw_dir=storage_root / "data" / "raw",
            reports_dir=storage_root / "reports",
            database_path=storage_root / "data" / "workflow.sqlite",
        )
        paths.create_directories()
        alpaca = AlpacaConfig(
            api_key=env.get("ALPACA_PAPER_API_KEY"),
            api_secret=env.get("ALPACA_PAPER_SECRET_KEY"),
            paper_trading=_parse_bool(env.get("PAPER"), True),
            trade_api_url=env.get("TRADE_API_URL") or env.get("ALPACA_TRADE_API_URL"),
        )
        run_mode = _parse_run_mode(env.get("RUN_MODE"))
        explicit_market_provider, market_provider_env = _market_provider_override(env, run_mode)
        market_provider = _resolve_market_provider(
            explicit=explicit_market_provider,
            alpaca_enabled=alpaca.enabled,
            run_mode=run_mode,
            env_var_name=market_provider_env,
        )
        legacy_lookback_hours = max(1, int(env.get("LOOKBACK_HOURS", "24")))
        legacy_sr_lookback_days = max(
            1,
            int(env.get("SUPPORT_RESISTANCE_LOOKBACK_DAYS", "7")),
        )
        return cls(
            trading=TradingConfig(
                symbol=env.get("TRADING_SYMBOL", "AAPL").upper(),
                mode=run_mode,
                live_market_data_delay_minutes=max(0, int(env.get("LIVE_MARKET_DATA_DELAY_MINUTES", "15"))),
                lookback_hours=max(1, int(env.get("LOOKBACK_HOURS", "240"))),
                eda_window_hours=max(1, int(env.get("EDA_WINDOW_HOURS", "24"))),
                sr_lookback_days=max(
                    1,
                    int(env.get("SR_LOOKBACK_DAYS", str(legacy_sr_lookback_days))),
                ),
                sr_timeframe=env.get("SR_TIMEFRAME", "1h"),
                indicator_lookback_hours=max(
                    1,
                    int(env.get("INDICATOR_LOOKBACK_HOURS", str(legacy_lookback_hours))),
                ),
                indicator_timeframe=env.get("INDICATOR_TIMEFRAME", "5min"),
                backtest_history_days=max(8, int(env.get("BACKTEST_HISTORY_DAYS", "30"))),
                live_sleep_seconds=max(0.0, float(env.get("LIVE_SLEEP_SECONDS", "3600.0"))),
                backtest_sleep_seconds=max(0.0, float(env.get("BACKTEST_SLEEP_SECONDS", "1.0"))),
            ),
            market_data=MarketDataConfig(provider=market_provider),
            raw_store=RawStoreConfig(provider=_resolve_raw_store_provider(env.get("RAW_STORE_PROVIDER"))),
            result_store=ResultStoreConfig(
                provider=_resolve_result_store_provider(env.get("RESULT_STORE_PROVIDER")),
                database_url=env.get("DATABASE_URL") or _sqlite_database_url(paths.database_path),
            ),
            alpha_vantage=AlphaVantageConfig(
                api_key=env.get("ALPHA_VANTAGE_API_KEY"),
                base_url=env.get("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"),
                interval=env.get("ALPHA_VANTAGE_INTERVAL", "5min"),
                request_pause_seconds=max(0.0, float(env.get("ALPHA_VANTAGE_CALL_PAUSE_SECONDS", "15.0"))),
                max_retries=max(1, int(env.get("ALPHA_VANTAGE_MAX_RETRIES", "3"))),
            ),
            azure=AzureConfig(
                storage_account_url=env.get("AZURE_STORAGE_ACCOUNT_URL"),
                storage_connection_string=env.get("AZURE_STORAGE_CONNECTION_STRING"),
                blob_container_raw=env.get("AZURE_BLOB_CONTAINER_RAW", "raw"),
                blob_prefix=env.get("AZURE_BLOB_PREFIX", "").strip("/"),
                key_vault_url=env.get("AZURE_KEY_VAULT_URL"),
            ),
            alpaca=alpaca,
            news=NewsConfig(
                max_age_days=max(1, int(env.get("NEWS_MAX_AGE_DAYS", "7"))),
            ),
            llm=LLMConfig(
                provider="deepseek",
                provider_label="DeepSeek",
                api_key=env.get("DEEPSEEK_API_KEY"),
                base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                model=env.get("DEEPSEEK_MODEL", "deepseek-reasoner"),
                timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "45.0")),
                show_progress=_parse_bool(env.get("LLM_SHOW_PROGRESS"), True),
                heartbeat_seconds=float(env.get("LLM_HEARTBEAT_SECONDS", "5.0")),
            ),
            secondary_llm=LLMConfig(
                provider="openai",
                provider_label="OpenAI",
                api_key=env.get("OPENAI_API_KEY"),
                base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=env.get("OPENAI_MODEL", "gpt-5.4-mini"),
                timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "45.0")),
                show_progress=_parse_bool(env.get("LLM_SHOW_PROGRESS"), True),
                heartbeat_seconds=float(env.get("LLM_HEARTBEAT_SECONDS", "5.0")),
            ),
            paths=paths,
        )


def _parse_run_mode(raw: str | None) -> RunMode:
    """Parse the configured run mode with a live default."""
    candidate = (raw or RunMode.LIVE.value).strip().lower()
    try:
        return RunMode(candidate)
    except ValueError as exc:
        raise RuntimeError(
            f"Unsupported RUN_MODE={candidate!r}. Use '{RunMode.LIVE.value}' or '{RunMode.BACKTEST.value}'."
        ) from exc


def _market_provider_override(env: dict[str, str], run_mode: RunMode) -> tuple[str | None, str]:
    """Resolve the environment variable used for the active mode's market-data provider."""
    if run_mode == RunMode.BACKTEST:
        return None, "MARKET_DATA_PROVIDER"
    explicit = env.get("LIVE_MARKET_DATA_PROVIDER")
    if explicit is not None:
        return explicit, "LIVE_MARKET_DATA_PROVIDER"
    return env.get("MARKET_DATA_PROVIDER"), "MARKET_DATA_PROVIDER"


def _resolve_market_provider(
    *,
    explicit: str | None,
    alpaca_enabled: bool,
    run_mode: RunMode,
    env_var_name: str,
) -> str:
    """Resolve the market-data provider from config and available credentials."""
    if explicit:
        provider = explicit.strip().lower().replace("-", "_")
        if provider == "alphavantage":
            provider = "alpha_vantage"
        if provider in {"alpaca", "alpha_vantage"}:
            return provider
        raise RuntimeError(
            f"Unsupported {env_var_name}={provider!r}. "
            "Use 'alpha_vantage' or 'alpaca'."
        )
    if run_mode == RunMode.BACKTEST:
        return "alpha_vantage"
    if alpaca_enabled:
        return "alpaca"
    return "alpha_vantage"


def _resolve_raw_store_provider(provider: str | None = None) -> str:
    """Resolve the raw artifact store provider name."""
    provider = (provider or "local").strip().lower()
    if provider in {"local", "azure_blob"}:
        return provider
    raise RuntimeError(
        f"Unsupported RAW_STORE_PROVIDER={provider!r}. Use 'local' or 'azure_blob'."
    )


def _resolve_result_store_provider(provider: str | None = None) -> str:
    """Resolve the result store provider name."""
    provider = (provider or "sqlite").strip().lower()
    if provider in {"sqlite", "azure_sql"}:
        return provider
    raise RuntimeError(
        f"Unsupported RESULT_STORE_PROVIDER={provider!r}. Use 'sqlite' or 'azure_sql'."
    )


def _merged_env(dotenv_path: Path) -> dict[str, str]:
    """Merge process environment variables with optional dotenv values."""
    merged = {
        key: value
        for key, value in dotenv_values(dotenv_path).items()
        if value is not None
    }
    merged.update(os.environ)
    return merged


def _resolve_storage_root(project_root: Path, env: dict[str, str]) -> Path:
    """Resolve a writable root for runtime-generated files and caches."""
    override = env.get("FUNCTION_APP_STORAGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if _is_azure_functions_environment(env):
        return Path(tempfile.gettempdir()).resolve() / "fresh_simple_trading_project"
    return project_root


def _is_azure_functions_environment(env: dict[str, str]) -> bool:
    """Detect whether the process is running inside Azure Functions."""
    return any(
        env.get(key)
        for key in (
            "FUNCTIONS_WORKER_RUNTIME",
            "WEBSITE_INSTANCE_ID",
            "WEBSITE_SITE_NAME",
        )
    )


def _sqlite_database_url(database_path: Path) -> str:
    """Build a SQLite SQLAlchemy URL for the configured database file."""
    return f"sqlite:///{database_path.resolve()}"
