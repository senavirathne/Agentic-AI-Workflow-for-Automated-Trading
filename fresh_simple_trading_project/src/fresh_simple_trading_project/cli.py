from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backtesting import summarize_backtest_results
from .config import RunMode
from .models import WorkflowResult
from .workflow import build_workflow, format_reasoning_lines


def main() -> None:
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

    args = parser.parse_args()
    _override_runtime_env(mode=args.mode, symbol=args.symbol)
    workflow = build_workflow(project_root=Path.cwd())

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


def _override_runtime_env(*, mode: str | None, symbol: str | None) -> None:
    if mode:
        os.environ["RUN_MODE"] = mode
    if symbol:
        os.environ["TRADING_SYMBOL"] = symbol.upper()


def _print_run_summary(
    workflow,
    results: list[WorkflowResult],
    *,
    json_output: bool,
    symbol: str | None,
) -> None:
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
        "reasoning": _reasoning_payload(result),
    }


def _reasoning_payload(result: WorkflowResult) -> dict[str, object]:
    return {
        "technical_agent": result.analysis.llm_summary,
        "news_agent": result.retrieval.summary_note,
        "risk_agent": result.risk.summary_note,
        "risk_warnings": result.risk.warnings,
        "decision_rationale": result.decision.rationale,
        "headline_summary": result.retrieval.headline_summary,
    }


def _format_trade_once_output(mode: RunMode, result: WorkflowResult) -> str:
    lines = [
        f"Mode: {mode.value}",
        f"Symbol: {result.symbol}",
        f"Decision: {result.decision.action.value}",
        f"Quantity: {result.decision.quantity}",
        f"Confidence: {result.decision.confidence:.2f}",
        f"Execution: {result.execution.status}",
        "",
        "Reasoning:",
    ]
    lines.extend(format_reasoning_lines(result, prefix="  "))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
