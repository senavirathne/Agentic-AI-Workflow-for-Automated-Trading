from __future__ import annotations

from pathlib import Path

import fresh_simple_trading_project.config as config_module
import pytest
from fresh_simple_trading_project.config import Settings


ENV_KEYS = [
    "FUNCTIONS_WORKER_RUNTIME",
    "FUNCTION_APP_STORAGE_ROOT",
    "TRADING_SYMBOL",
    "LOOKBACK_HOURS",
    "EDA_WINDOW_HOURS",
    "SUPPORT_RESISTANCE_LOOKBACK_DAYS",
    "BACKTEST_HISTORY_DAYS",
    "LIVE_SLEEP_SECONDS",
    "LIVE_MARKET_DATA_DELAY_MINUTES",
    "BACKTEST_SLEEP_SECONDS",
    "RUN_MODE",
    "LIVE_MARKET_DATA_PROVIDER",
    "MARKET_DATA_PROVIDER",
    "ALPACA_PAPER_API_KEY",
    "ALPACA_PAPER_SECRET_KEY",
    "PAPER",
    "TRADE_API_URL",
    "ALPACA_TRADE_API_URL",
    "NEWS_MAX_AGE_DAYS",
    "RAW_STORE_PROVIDER",
    "RESULT_STORE_PROVIDER",
    "DATABASE_URL",
    "ALPHA_VANTAGE_API_KEY",
    "ALPHA_VANTAGE_BASE_URL",
    "ALPHA_VANTAGE_INTERVAL",
    "ALPHA_VANTAGE_CALL_PAUSE_SECONDS",
    "ALPHA_VANTAGE_MAX_RETRIES",
    "AZURE_STORAGE_ACCOUNT_URL",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_BLOB_CONTAINER_RAW",
    "AZURE_BLOB_PREFIX",
    "AZURE_KEY_VAULT_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
]


def test_settings_default_to_alpha_vantage_provider(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpha_vantage"
    assert settings.raw_store.provider == "local"
    assert settings.result_store.provider == "sqlite"
    assert settings.alpha_vantage.enabled is False
    assert len(settings.alpha_vantage.generated_api_key) == 13
    assert settings.alpaca.enabled is False
    assert settings.alpaca.paper_trading is True
    assert settings.news.max_age_days == 7
    assert settings.trading.lookback_hours == 240
    assert settings.trading.eda_window_hours == 24
    assert settings.trading.support_resistance_lookback_days == 7
    assert settings.trading.live_sleep_seconds == 3600.0
    assert settings.trading.live_market_data_delay_minutes == 15
    assert settings.trading.backtest_sleep_seconds == 1.0
    assert settings.azure.blob_container_raw == "raw"
    assert settings.result_store.database_url is not None


def test_settings_use_temp_storage_root_inside_azure_functions(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    azure_temp_root = tmp_path / "azure-temp"
    monkeypatch.setattr(config_module.tempfile, "gettempdir", lambda: str(azure_temp_root))
    monkeypatch.setenv("FUNCTIONS_WORKER_RUNTIME", "python")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.paths.project_root == tmp_path
    assert settings.paths.data_dir == azure_temp_root / "fresh_simple_trading_project" / "data"
    assert settings.paths.raw_dir == settings.paths.data_dir / "raw"
    assert settings.paths.reports_dir == azure_temp_root / "fresh_simple_trading_project" / "reports"
    assert settings.paths.data_dir.exists()
    assert settings.paths.reports_dir.exists()


def test_settings_honor_explicit_function_storage_root_override(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    explicit_storage_root = tmp_path / "function-storage"
    monkeypatch.setenv("FUNCTIONS_WORKER_RUNTIME", "python")
    monkeypatch.setenv("FUNCTION_APP_STORAGE_ROOT", str(explicit_storage_root))

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.paths.data_dir == explicit_storage_root / "data"
    assert settings.paths.database_path == explicit_storage_root / "data" / "workflow.sqlite"


def test_alpha_vantage_generated_key_refreshes_on_each_access(monkeypatch) -> None:
    generated_keys = iter(["ABCDEFGHI1234", "ZYXWVUT987654"])
    monkeypatch.setattr(
        config_module,
        "generate_random_alpha_vantage_api_key",
        lambda length=13: next(generated_keys),
    )

    alpha_vantage = config_module.AlphaVantageConfig()

    assert alpha_vantage.generated_api_key == "ABCDEFGHI1234"
    assert alpha_vantage.generated_api_key == "ZYXWVUT987654"


def test_settings_use_notebook_alpaca_variables_when_present(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("PAPER", "false")
    monkeypatch.setenv("TRADE_API_URL", "https://paper-api.example.test")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpaca"
    assert settings.alpaca.enabled is True
    assert settings.alpaca.api_key == "alpaca-key"
    assert settings.alpaca.api_secret == "alpaca-secret"
    assert settings.alpaca.paper_trading is False
    assert settings.alpaca.trade_api_url == "https://paper-api.example.test"


def test_settings_allow_alpha_vantage_live_market_data_with_alpaca_credentials(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alpha_vantage")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpha_vantage"
    assert settings.alpaca.enabled is True


def test_settings_default_to_alpha_vantage_in_backtest_even_with_alpaca_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RUN_MODE", "backtest")
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alpaca")
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpha_vantage"
    assert settings.alpaca.enabled is True


def test_settings_parse_news_max_age_days(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("NEWS_MAX_AGE_DAYS", "5")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.news.max_age_days == 5


def test_settings_parse_live_market_data_delay_minutes(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LIVE_MARKET_DATA_DELAY_MINUTES", "20")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.trading.live_market_data_delay_minutes == 20


def test_settings_parse_azure_storage_configuration(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RAW_STORE_PROVIDER", "azure_blob")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")
    monkeypatch.setenv("AZURE_BLOB_CONTAINER_RAW", "trading-raw")
    monkeypatch.setenv("AZURE_BLOB_PREFIX", "dev")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://vault-example.vault.azure.net")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.raw_store.provider == "azure_blob"
    assert settings.azure.storage_account_url == "https://example.blob.core.windows.net"
    assert settings.azure.blob_container_raw == "trading-raw"
    assert settings.azure.blob_prefix == "dev"
    assert settings.azure.key_vault_url == "https://vault-example.vault.azure.net"


def test_settings_parse_azure_sql_configuration(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("RESULT_STORE_PROVIDER", "azure_sql")
    monkeypatch.setenv("DATABASE_URL", "mssql+pyodbc://server/database")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.result_store.provider == "azure_sql"
    assert settings.result_store.database_url == "mssql+pyodbc://server/database"


def test_settings_parse_alpha_vantage_configuration(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "ABCDEFGHI1234")
    monkeypatch.setenv("ALPHA_VANTAGE_CALL_PAUSE_SECONDS", "0.5")
    monkeypatch.setenv("ALPHA_VANTAGE_MAX_RETRIES", "4")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.alpha_vantage.enabled is True
    assert settings.alpha_vantage.api_key == "ABCDEFGHI1234"
    assert settings.alpha_vantage.request_pause_seconds == 0.5
    assert settings.alpha_vantage.max_retries == 4


def test_settings_parse_secondary_openai_llm_configuration(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.secondary_llm is not None
    assert settings.secondary_llm.enabled is True
    assert settings.secondary_llm.provider == "openai"
    assert settings.secondary_llm.api_key == "openai-key"
    assert settings.secondary_llm.model == "gpt-5.4-mini"
    assert settings.secondary_llm.base_url == "https://api.openai.com/v1"


def test_settings_require_at_least_one_llm_api_key(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    settings = Settings.from_env(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY or OPENAI_API_KEY"):
        settings.require_llm()


def test_settings_accept_alpha_vantage_provider_alias(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LIVE_MARKET_DATA_PROVIDER", "alphavantage")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpha_vantage"


def test_settings_accept_legacy_market_data_provider_for_live_mode(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alpaca")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "alpaca"


def _clear_env(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
