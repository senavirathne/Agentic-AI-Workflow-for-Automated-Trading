from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from alpaca.data.timeframe import TimeFrameUnit


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return default
    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return values or default


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Credentials:
    api_key: str | None
    api_secret: str | None
    paper_trading: bool = True
    trade_api_url: str | None = None

    def require(self) -> None:
        if not self.api_key or not self.api_secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Populate ALPACA_PAPER_API_KEY and "
                "ALPACA_PAPER_SECRET_KEY in the environment or .env file."
            )


@dataclass(frozen=True)
class TradingConfig:
    primary_symbol: str = "TQQQ"
    universe_symbols: list[str] = field(
        default_factory=lambda: ["TQQQ", "QQQ", "SPY", "AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
    )
    market_timezone: str = "America/New_York"
    short_timeframe: TimeFrameUnit = TimeFrameUnit.Minute
    short_timeframe_multiplier: int = 15
    short_lookback_days: int = 30
    medium_timeframe: TimeFrameUnit = TimeFrameUnit.Hour
    medium_timeframe_multiplier: int = 2
    medium_lookback_days: int = 90
    long_timeframe: TimeFrameUnit = TimeFrameUnit.Day
    long_lookback_days: int = 260
    news_lookback_days: int = 7
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_fast: int = 50
    ma_mid: int = 100
    ma_slow: int = 200
    buy_power_limit: float = 0.02
    max_risk_pct: float = 0.03
    window_size: int = 5
    initial_capital: float = 100_000.0
    signal_horizon_bars: int = 5


@dataclass(frozen=True)
class CloudConfig:
    provider: str = "local"
    raw_store_uri: str | None = None
    structured_store_uri: str | None = None
    compute_target: str = "local"

    @property
    def uses_placeholder_cloud(self) -> bool:
        return self.provider.lower() in {"aws", "azure"}


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-reasoner"
    timeout_seconds: float = 45.0


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    logs_dir: Path
    database_path: Path

    def create_directories(self) -> None:
        for directory in [
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.reports_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    credentials: Credentials
    trading: TradingConfig
    cloud: CloudConfig
    llm: LLMConfig
    paths: Paths

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        load_dotenv()
        root = Path(project_root or Path.cwd()).resolve()
        paths = Paths(
            project_root=root,
            data_dir=root / "data",
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            reports_dir=root / "reports" / "output",
            logs_dir=root / "logs",
            database_path=root / "data" / "trading_workflow.db",
        )
        paths.create_directories()

        credentials = Credentials(
            api_key=os.getenv("ALPACA_PAPER_API_KEY"),
            api_secret=os.getenv("ALPACA_PAPER_SECRET_KEY"),
            paper_trading=_parse_bool(os.getenv("ALPACA_PAPER_TRADE"), True),
            trade_api_url=os.getenv("ALPACA_TRADE_API_URL") or None,
        )
        trading = TradingConfig(
            primary_symbol=os.getenv("PRIMARY_SYMBOL", "TQQQ").upper(),
            universe_symbols=_split_csv(os.getenv("UNIVERSE_SYMBOLS"), TradingConfig().universe_symbols),
        )
        cloud = CloudConfig(
            provider=os.getenv("CLOUD_PROVIDER", "local").lower(),
            raw_store_uri=os.getenv("RAW_STORE_URI") or None,
            structured_store_uri=os.getenv("STRUCTURED_STORE_URI") or None,
            compute_target=os.getenv("COMPUTE_TARGET", "local").lower(),
        )
        llm = LLMConfig(
            enabled=_parse_bool(os.getenv("LLM_ENABLED"), False),
            api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
            timeout_seconds=_parse_float(os.getenv("LLM_TIMEOUT_SECONDS"), 45.0),
        )
        return cls(credentials=credentials, trading=trading, cloud=cloud, llm=llm, paths=paths)
