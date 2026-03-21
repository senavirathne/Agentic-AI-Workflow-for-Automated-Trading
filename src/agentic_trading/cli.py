from __future__ import annotations

import argparse
import logging
from pathlib import Path

from alpaca.data.timeframe import TimeFrameUnit

from .alpaca_clients import AlpacaService
from .analysis import generate_eda_artifacts
from .backtesting import StrategyBacktester
from .config import Settings
from .workflow import TradingWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic Alpaca trading workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect raw market data and news.")
    collect.add_argument("--symbols", nargs="+", help="Symbols to collect.")
    collect.add_argument("--days", type=int, default=365, help="Number of days of daily bars.")
    collect.add_argument("--include-news", action="store_true", help="Collect recent Alpaca news.")

    eda = subparsers.add_parser("eda", help="Run exploratory data analysis over a symbol universe.")
    eda.add_argument("--symbols", nargs="+", help="Symbols to analyze.")
    eda.add_argument("--days", type=int, default=365, help="Lookback window for daily bars.")

    backtest = subparsers.add_parser("backtest", help="Backtest the Alpaca RSI/MACD workflow.")
    backtest.add_argument("--symbol", help="Symbol to backtest.")

    trade_once = subparsers.add_parser("trade-once", help="Run one workflow cycle.")
    trade_once.add_argument("--symbol", help="Symbol to trade.")
    trade_once.add_argument("--execute", action="store_true", help="Submit paper-trade orders.")

    trade_loop = subparsers.add_parser("trade-loop", help="Run the hourly workflow loop.")
    trade_loop.add_argument("--symbol", help="Symbol to trade.")
    trade_loop.add_argument("--execute", action="store_true", help="Submit paper-trade orders.")
    trade_loop.add_argument("--iterations", type=int, help="Optional maximum loop iterations.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env(Path.cwd())
    _configure_logging(settings.paths.logs_dir / "workflow.log")

    if args.command == "collect":
        _collect(args, settings)
    elif args.command == "eda":
        _run_eda(args, settings)
    elif args.command == "backtest":
        _run_backtest(args, settings)
    elif args.command == "trade-once":
        _trade_once(args, settings)
    elif args.command == "trade-loop":
        _trade_loop(args, settings)


def _collect(args: argparse.Namespace, settings: Settings) -> None:
    service = AlpacaService(settings)
    workflow = TradingWorkflow(settings)
    symbols = [symbol.upper() for symbol in (args.symbols or settings.trading.universe_symbols)]
    bars_by_symbol = service.fetch_stock_bars(symbols, TimeFrameUnit.Day, args.days)
    for symbol, frame in bars_by_symbol.items():
        workflow.raw_lake.save_bars(symbol, "day", frame)
        if args.include_news:
            news = service.fetch_recent_news(symbol, days=settings.trading.news_lookback_days)
            workflow.raw_lake.save_news(symbol, news)
    print(f"Collected daily bars for {', '.join(symbols)}")


def _run_eda(args: argparse.Namespace, settings: Settings) -> None:
    service = AlpacaService(settings)
    symbols = [symbol.upper() for symbol in (args.symbols or settings.trading.universe_symbols)]
    bars_by_symbol = service.fetch_stock_bars(symbols, TimeFrameUnit.Day, args.days)
    output_dir = settings.paths.reports_dir / "eda"
    artifacts = generate_eda_artifacts(bars_by_symbol, output_dir, settings.trading)
    print(f"EDA outputs written to {artifacts['summary']}")


def _run_backtest(args: argparse.Namespace, settings: Settings) -> None:
    symbol = (args.symbol or settings.trading.primary_symbol).upper()
    service = AlpacaService(settings)
    main_bars = service.fetch_stock_bars(
        [symbol], settings.trading.main_timeframe, settings.trading.main_lookback_days
    )[symbol]
    trend_bars = service.fetch_stock_bars(
        [symbol], settings.trading.trend_timeframe, settings.trading.trend_lookback_days
    )[symbol]
    backtester = StrategyBacktester(settings.trading)
    summary = backtester.backtest(symbol, main_bars, trend_bars)
    output_dir = settings.paths.reports_dir / "backtests" / symbol.lower()
    backtester.write_outputs(summary, output_dir)
    workflow = TradingWorkflow(settings)
    workflow.structured_store.save_backtest_summary(summary)
    print(f"Backtest summary written to {output_dir / 'backtest_summary.json'}")


def _trade_once(args: argparse.Namespace, settings: Settings) -> None:
    workflow = TradingWorkflow(settings)
    result = workflow.run_once(symbol=args.symbol, execute_orders=args.execute)
    print(f"{result.context.symbol}: {result.decision.action.value} qty={result.decision.quantity}")


def _trade_loop(args: argparse.Namespace, settings: Settings) -> None:
    workflow = TradingWorkflow(settings)
    results = workflow.run_loop(
        symbol=args.symbol,
        execute_orders=args.execute,
        max_iterations=args.iterations,
    )
    print(f"Completed {len(results)} workflow iterations.")


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
