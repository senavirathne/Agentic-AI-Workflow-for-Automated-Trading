from __future__ import annotations

from pathlib import Path

from fresh_simple_trading_project.config import Settings


ENV_KEYS = [
    "TRADING_SYMBOL",
    "LOOKBACK_HOURS",
    "EDA_WINDOW_HOURS",
    "SUPPORT_RESISTANCE_LOOKBACK_DAYS",
    "BACKTEST_HISTORY_DAYS",
    "LIVE_SLEEP_SECONDS",
    "BACKTEST_SLEEP_SECONDS",
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
]


def test_settings_default_to_synthetic_provider(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "synthetic"
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
    assert settings.trading.backtest_sleep_seconds == 1.0
    assert settings.azure.blob_container_raw == "raw"
    assert settings.result_store.database_url is not None


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


def test_settings_allow_synthetic_market_data_with_alpaca_credentials(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "synthetic")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "synthetic"
    assert settings.alpaca.enabled is True


def test_settings_parse_news_max_age_days(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("NEWS_MAX_AGE_DAYS", "5")

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.news.max_age_days == 5


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


def test_settings_reject_removed_market_data_provider(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "alphavantage")

    try:
        Settings.from_env(project_root=tmp_path)
    except RuntimeError as exc:
        assert "MARKET_DATA_PROVIDER" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected removed alphavantage provider to be rejected.")


def _clear_env(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
