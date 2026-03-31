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
]


def test_settings_default_to_synthetic_provider(tmp_path: Path, monkeypatch) -> None:
    _clear_env(monkeypatch)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.market_data.provider == "synthetic"
    assert settings.alpaca.enabled is False
    assert settings.alpaca.paper_trading is True
    assert settings.news.max_age_days == 7
    assert settings.trading.lookback_hours == 24
    assert settings.trading.eda_window_hours == 24
    assert settings.trading.support_resistance_lookback_days == 7
    assert settings.trading.live_sleep_seconds == 3600.0
    assert settings.trading.backtest_sleep_seconds == 1.0


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
