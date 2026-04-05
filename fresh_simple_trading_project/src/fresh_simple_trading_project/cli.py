"""Command-line entry points for running the trading workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .alpha_vantage import AlphaVantageIndicatorService
from .backtesting import summarize_backtest_results
from .config import RunMode, Settings
from .models import AlphaVantageIndicatorSnapshot, WorkflowResult
from .workflow import build_result_store, build_workflow, format_reasoning_lines


def main() -> None:
    """Parse CLI arguments and execute the requested workflow command."""
    parser = argparse.ArgumentParser(description="Unified live trading and backtesting workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_cmd = subparsers.add_parser("run")
    run_cmd.add_argument("--mode", choices=[mode.value for mode in RunMode], default=None)
    run_cmd.add_argument("--symbol", default=None)
    run_cmd.add_argument("--execute", action="store_true")
    run_cmd.add_argument("--max-iterations", type=int, default=None)
    run_cmd.add_argument("--sleep-seconds", type=float, default=None)
    run_cmd.add_argument("--json", action="store_true")

    trade_once = subparsers.add_parser("trade-once")
    trade_once.add_argument("--mode", choices=[mode.value for mode in RunMode], default=None)
    trade_once.add_argument("--symbol", default=None)
    trade_once.add_argument("--execute", action="store_true")
    trade_once.add_argument("--json", action="store_true")

    alpha_vantage_cmd = subparsers.add_parser("alpha-vantage-indicators")
    alpha_vantage_cmd.add_argument("--symbol", default=None)
    alpha_vantage_cmd.add_argument("--json", action="store_true")

    args = parser.parse_args()
    project_root = _resolve_project_root()

    if args.command == "alpha-vantage-indicators":
        _print_alpha_vantage_snapshot(symbol=args.symbol, project_root=project_root)
        return

    workflow = build_workflow(project_root=project_root, mode=getattr(args, "mode", None))

    if args.command == "run":
        results = workflow.run_loop(
            symbol=args.symbol,
            execute_orders=args.execute,
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep_seconds,
        )
        _print_run_summary(workflow, results, json_output=args.json, symbol=args.symbol)
        return

    result = workflow.run_once(symbol=args.symbol, execute_orders=args.execute)
    if args.json:
        print(json.dumps(_trade_once_payload(workflow.settings.trading.mode, result), indent=2))
        return
    print(_format_trade_once_output(workflow.settings.trading.mode, result))


def _resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the standalone project root even when invoked from `src/`."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "fresh_simple_trading_project").is_dir():
            return candidate

    fallback = Path(__file__).resolve().parents[2]
    if (fallback / "src" / "fresh_simple_trading_project").is_dir():
        return fallback
    return current


def _print_run_summary(
    workflow,
    results: list[WorkflowResult],
    *,
    json_output: bool,
    symbol: str | None,
) -> None:
    """Render a run-loop summary in text or JSON form."""
    target_symbol = (symbol or workflow.settings.trading.symbol).upper()
    mode = workflow.settings.trading.mode
    if mode == RunMode.BACKTEST:
        summary = summarize_backtest_results(
            results,
            starting_cash=workflow.settings.trading.starting_cash,
            symbol=target_symbol,
        )
        workflow.result_store.save_backtest_summary(summary)
        if json_output:
            print(
                json.dumps(
                    {
                        "mode": mode.value,
                        "symbol": target_symbol,
                        "iterations": len(results),
                        "ending_cash": summary.ending_cash,
                        "net_profit": summary.net_profit,
                        "total_return_pct": summary.total_return_pct,
                        "benchmark_return_pct": summary.benchmark_return_pct,
                        "trade_count": summary.trade_count,
                        "win_rate": summary.win_rate,
                    },
                    indent=2,
                )
            )
            return
        print(
            "\n".join(
                [
                    f"Mode: {mode.value}",
                    f"Symbol: {target_symbol}",
                    f"Iterations: {len(results)}",
                    f"Ending Cash: {summary.ending_cash:.2f}",
                    f"Net Profit: {summary.net_profit:.2f}",
                    f"Total Return %: {summary.total_return_pct:.2f}",
                    f"Benchmark Return %: {summary.benchmark_return_pct:.2f}",
                    f"Trade Count: {summary.trade_count}",
                    f"Win Rate: {summary.win_rate:.2f}",
                ]
            )
        )
        return

    if json_output:
        payload: dict[str, object] = {
            "mode": mode.value,
            "symbol": target_symbol,
            "iterations": len(results),
        }
        if results:
            payload["last_result"] = _trade_once_payload(mode, results[-1])
        print(json.dumps(payload, indent=2))
        return

    lines = [
        f"Mode: {mode.value}",
        f"Symbol: {target_symbol}",
        f"Iterations: {len(results)}",
    ]
    if results:
        lines.extend(
            [
                f"Last Decision: {results[-1].decision.action.value}",
                f"Last Quantity: {results[-1].decision.quantity}",
                f"Last Execution: {results[-1].execution.status}",
            ]
        )
    print("\n".join(lines))


def _trade_once_payload(mode: RunMode, result: WorkflowResult) -> dict[str, object]:
    """Build the structured payload for a single workflow iteration."""
    return {
        "mode": mode.value,
        "symbol": result.symbol,
        "analysis_interval": "5min",
        "loop_interval": "1h",
        "technical_agent_summary": result.analysis.llm_summary,
        "news_agent_summary": result.retrieval.summary_note,
        "risk_agent_summary": result.risk.summary_note,
        "action": result.decision.action.value,
        "quantity": result.decision.quantity,
        "confidence": result.decision.confidence,
        "execution_status": result.execution.status,
        "protective_order_ids": result.execution.protective_order_ids,
        "performance": _performance_payload(result.performance),
        "previous_forecast": _forecast_payload(result.previous_forecast),
        "hold_forecast": _forecast_payload(result.hold_forecast),
        "alpha_vantage_indicator_snapshot": _alpha_vantage_payload(result.alpha_vantage_indicator_snapshot),
        "reasoning": _reasoning_payload(result),
    }


def _reasoning_payload(result: WorkflowResult) -> dict[str, object]:
    """Serialize the agent handoffs and supporting reasoning fields."""
    return {
        "technical_agent": result.analysis.llm_summary,
        "news_agent": result.retrieval.summary_note,
        "critical_news": result.retrieval.critical_news,
        "risk_agent": result.risk.summary_note,
        "risk_warnings": result.risk.warnings,
        "decision_rationale": result.decision.rationale,
        "headline_summary": result.retrieval.headline_summary,
        "performance": _performance_payload(result.performance),
        "previous_forecast": _forecast_payload(result.previous_forecast),
        "hold_forecast": _forecast_payload(result.hold_forecast),
        "alpha_vantage_indicator_snapshot": _alpha_vantage_payload(result.alpha_vantage_indicator_snapshot),
    }


def _format_trade_once_output(mode: RunMode, result: WorkflowResult) -> str:
    """Format a readable text summary for a single iteration."""
    lines = [
        f"Mode: {mode.value}",
        f"Symbol: {result.symbol}",
        f"Decision: {result.decision.action.value}",
        f"Quantity: {result.decision.quantity}",
        f"Confidence: {result.decision.confidence:.2f}",
        f"Execution: {result.execution.status}",
        f"Protective Orders: {result.execution.protective_order_ids or '<none>'}",
        "",
        "Reasoning:",
    ]
    lines.extend(format_reasoning_lines(result, prefix="  "))
    return "\n".join(lines)


def _alpha_vantage_payload(snapshot: AlphaVantageIndicatorSnapshot | None) -> dict[str, object] | None:
    """Serialize an Alpha Vantage snapshot dataclass for CLI output."""
    if snapshot is None:
        return None
    return asdict(snapshot)


def _forecast_payload(snapshot) -> dict[str, object] | None:
    """Serialize a forecast snapshot for CLI output."""
    if snapshot is None:
        return None
    payload = asdict(snapshot)
    for key in ("generated_at", "valid_until"):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    return payload


def _performance_payload(snapshot) -> dict[str, object] | None:
    """Serialize a performance snapshot for CLI output."""
    if snapshot is None:
        return None
    payload = asdict(snapshot)
    if payload.get("as_of") is not None:
        payload["as_of"] = str(payload["as_of"])
    return payload


def _print_alpha_vantage_snapshot(*, symbol: str | None, project_root: Path | None = None) -> None:
    """Fetch, persist, and print the latest Alpha Vantage indicator snapshot."""
    settings = Settings.from_env(project_root=project_root or _resolve_project_root())
    target_symbol = (symbol or settings.trading.symbol).upper()
    result_store = build_result_store(settings)
    service = AlphaVantageIndicatorService(
        settings.alpha_vantage,
        cache_dir=settings.paths.data_dir / "alpha_vantage",
        result_store=result_store,
    )
    snapshot = service.build_snapshot(target_symbol)
    result_store.save_alpha_vantage_indicator_snapshot(snapshot)
    print(json.dumps(asdict(snapshot), indent=2))


if __name__ == "__main__":
    main()
