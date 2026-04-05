import json
import logging
import os
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import azure.functions as func

project_root = Path(__file__).parent.resolve()

app = func.FunctionApp()

LIVE_MODE = "live"
BACKTEST_MODE = "backtest"
VALID_MODES = {LIVE_MODE, BACKTEST_MODE}

DEFAULT_SYMBOL = "AAPL"
DEFAULT_LOOPS = 1
DEFAULT_MODE = LIVE_MODE
DEFAULT_TIMER_ENABLED = False
DEFAULT_AUTO_SHUTDOWN = True
DEFAULT_LOG_DIR_NAME = "logs"
DEFAULT_LOG_TAIL_LINES = 80
MAX_LOG_TAIL_LINES = 200
LOG_OUTPUT_START_MARKER = "__CODEX_VM_LOG_START__"
LOG_OUTPUT_END_MARKER = "__CODEX_VM_LOG_END__"
LOG_OUTPUT_ERROR_MARKER = "__CODEX_VM_LOG_ERROR__"
RUN_COMMAND_LAUNCH_SUCCESS_MARKER = "__CODEX_VM_RUN_LAUNCH_OK__"
RUN_COMMAND_LAUNCH_ERROR_MARKER = "__CODEX_VM_RUN_LAUNCH_ERROR__"
RUN_COMMAND_LOG_FILE_WAIT_SECONDS = 15
DISPATCH_FILE_NAME = "vm_dispatch_state.json"
ACTIVE_VM_POWER_STATES = {"starting", "running", "stopping", "deallocating"}


class DispatchConflictError(RuntimeError):
    """Raised when a VM dispatch already appears active and force was not requested."""


class VmLogAccessError(RuntimeError):
    """Raised when the VM log cannot be fetched or parsed."""


class VmDispatchLaunchError(RuntimeError):
    """Raised when the VM dispatch command does not confirm the workflow runner started."""


@dataclass(frozen=True)
class VmRunnerConfig:
    subscription_id: str
    resource_group: str
    vm_name: str
    project_dir: str
    venv_activate: str
    default_symbol: str
    default_loops: int
    default_mode: str
    timer_enabled: bool
    auto_shutdown: bool
    log_dir: str
    managed_identity_client_id: str | None = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_azure_functions_environment() -> bool:
    return any(
        os.environ.get(key)
        for key in (
            "FUNCTIONS_WORKER_RUNTIME",
            "WEBSITE_INSTANCE_ID",
            "WEBSITE_SITE_NAME",
        )
    )


def _dispatch_storage_root() -> Path:
    override = os.environ.get("FUNCTION_APP_STORAGE_ROOT")
    if override and override.strip():
        return Path(override).expanduser().resolve()
    if _is_azure_functions_environment():
        return Path(tempfile.gettempdir()).resolve() / "fresh_simple_trading_project"
    return project_root


def _dispatch_state_path() -> Path:
    return _dispatch_storage_root() / "data" / DISPATCH_FILE_NAME


def _empty_dispatch_state() -> dict[str, Any]:
    return {
        "accepted": False,
        "active": False,
        "symbol": None,
        "loops": None,
        "mode": None,
        "live_after_backtest": False,
        "trigger_source": None,
        "submitted_at": None,
        "vm_name": None,
        "resource_group": None,
        "start_requested": False,
        "power_state": None,
        "log_file_path": None,
        "log_tail_command": None,
    }


def _load_dispatch_state() -> dict[str, Any]:
    state_path = _dispatch_state_path()
    if not state_path.exists():
        return _empty_dispatch_state()

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logging.warning("VM dispatch state file could not be read. Resetting dispatch state.")
        return _empty_dispatch_state()

    if not isinstance(payload, dict):
        return _empty_dispatch_state()

    state = _empty_dispatch_state()
    state.update(payload)
    return state


def _save_dispatch_state(state: dict[str, Any]) -> None:
    state_path = _dispatch_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_positive_int(value: Any, *, default: int) -> int:
    if value in {None, ""}:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("loops must be a positive integer")
    return parsed


def _parse_mode(value: Any, *, default: str) -> str:
    candidate = str(value or default).strip().lower()
    if candidate not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    return candidate


def _parse_log_tail_lines(value: Any, *, default: int = DEFAULT_LOG_TAIL_LINES) -> int:
    parsed = _parse_positive_int(value, default=default)
    return min(parsed, MAX_LOG_TAIL_LINES)


def _request_payload(req: func.HttpRequest) -> dict[str, Any]:
    try:
        payload = req.get_json()
    except ValueError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _request_value(req: func.HttpRequest, payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    return req.params.get(key)


def _sibling_route_url(req: func.HttpRequest, sibling_name: str, **extra_params: Any) -> str:
    split = urlsplit(req.url)
    base_query = dict(parse_qsl(split.query, keep_blank_values=True))
    if "code" in req.params and "code" not in base_query:
        base_query["code"] = req.params["code"]
    base_query.update({key: str(value) for key, value in extra_params.items() if value not in {None, ""}})
    route_prefix = split.path.rsplit("/", 1)[0]
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            f"{route_prefix}/{sibling_name}",
            urlencode(base_query),
            "",
        )
    )


def _dispatch_with_log_urls(req: func.HttpRequest, dispatch: dict[str, Any]) -> dict[str, Any]:
    enriched_dispatch = dict(dispatch)
    log_file_path = enriched_dispatch.get("log_file_path")
    if not log_file_path:
        return enriched_dispatch

    enriched_dispatch["log_url"] = _sibling_route_url(req, "log", log_file_path=log_file_path)
    enriched_dispatch["log_download_url"] = _sibling_route_url(
        req,
        "log",
        log_file_path=log_file_path,
        download="true",
    )
    return enriched_dispatch


def _response(*, message: str, dispatch: dict[str, Any], status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(
            {
                "message": message,
                "dispatch": dispatch,
            },
            indent=2,
        ),
        status_code=status_code,
        mimetype="application/json",
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required Function App setting: {name}")
    return value.strip()


def _load_vm_runner_config() -> VmRunnerConfig:
    project_dir = _require_env("VM_PROJECT_DIR")
    return VmRunnerConfig(
        subscription_id=_require_env("AZURE_SUBSCRIPTION_ID"),
        resource_group=_require_env("AZURE_VM_RESOURCE_GROUP"),
        vm_name=_require_env("AZURE_VM_NAME"),
        project_dir=project_dir,
        venv_activate=_require_env("VM_VENV_ACTIVATE"),
        default_symbol=os.environ.get("VM_DEFAULT_SYMBOL", os.environ.get("TRADING_SYMBOL", DEFAULT_SYMBOL)).upper(),
        default_loops=_parse_positive_int(os.environ.get("VM_DEFAULT_LOOPS"), default=DEFAULT_LOOPS),
        default_mode=_parse_mode(os.environ.get("VM_DEFAULT_MODE"), default=os.environ.get("RUN_MODE", DEFAULT_MODE)),
        timer_enabled=_parse_bool(os.environ.get("VM_TIMER_ENABLED"), DEFAULT_TIMER_ENABLED),
        auto_shutdown=_parse_bool(os.environ.get("VM_AUTO_SHUTDOWN"), DEFAULT_AUTO_SHUTDOWN),
        log_dir=os.environ.get("VM_LOG_DIR", str(Path(project_dir) / DEFAULT_LOG_DIR_NAME)),
        managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID"),
    )


def _dispatch_log_file_path(
    *,
    config: VmRunnerConfig,
    symbol: str,
    mode: str,
    trigger_source: str,
    submitted_at: str,
) -> str:
    safe_symbol = "".join(character.lower() if character.isalnum() else "_" for character in symbol).strip("_") or "symbol"
    safe_mode = "".join(character.lower() if character.isalnum() else "_" for character in mode).strip("_") or "mode"
    safe_trigger = "".join(
        character.lower() if character.isalnum() else "_" for character in trigger_source
    ).strip("_") or "trigger"
    timestamp = datetime.fromisoformat(submitted_at).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"workflow_{safe_mode}_{safe_symbol}_{safe_trigger}_{timestamp}.log"
    return str(Path(config.log_dir) / filename)


def _build_compute_client(config: VmRunnerConfig):
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.compute import ComputeManagementClient

    credential = DefaultAzureCredential(
        managed_identity_client_id=config.managed_identity_client_id,
        exclude_interactive_browser_credential=True,
    )
    return ComputeManagementClient(credential, config.subscription_id)


def _extract_power_state(instance_view: Any) -> str | None:
    for status in getattr(instance_view, "statuses", []) or []:
        code = str(getattr(status, "code", "") or "")
        if code.startswith("PowerState/"):
            return code.split("/", 1)[1].lower()
    return None


def _get_vm_power_state(compute_client: Any, config: VmRunnerConfig) -> str | None:
    instance_view = compute_client.virtual_machines.instance_view(config.resource_group, config.vm_name)
    return _extract_power_state(instance_view)


def _dispatch_is_active(state: dict[str, Any], power_state: str | None) -> bool:
    if power_state in ACTIVE_VM_POWER_STATES:
        return True
    if power_state in {"stopped", "stopped(deallocated)", "deallocated"}:
        return False
    return bool(state.get("active"))


def _build_run_command_script(
    *,
    config: VmRunnerConfig,
    symbol: str,
    loops: int,
    mode: str,
    live_after_backtest: bool,
    submitted_at: str,
    log_file_path: str,
) -> list[str]:
    runner_script = "\n".join(
        _build_workflow_runner_script(
            config=config,
            symbol=symbol,
            loops=loops,
            mode=mode,
            live_after_backtest=live_after_backtest,
            submitted_at=submitted_at,
            log_file_path=log_file_path,
        )
    )
    quoted_runner_script = shlex.quote(runner_script)
    quoted_log_file = shlex.quote(log_file_path)
    return [
        "set -euo pipefail",
        f"nohup bash -lc {quoted_runner_script} >/dev/null 2>&1 </dev/null &",
        "runner_pid=$!",
        f"for _ in $(seq 1 {RUN_COMMAND_LOG_FILE_WAIT_SECONDS}); do",
        f"  if [ -f {quoted_log_file} ]; then",
        f'    echo "{RUN_COMMAND_LAUNCH_SUCCESS_MARKER} pid=${{runner_pid}} log_file_path={log_file_path}"',
        "    exit 0",
        "  fi",
        "  sleep 1",
        "done",
        f'echo "{RUN_COMMAND_LAUNCH_ERROR_MARKER} Log file was not created: {log_file_path}"',
        "exit 1",
    ]


def _build_workflow_runner_script(
    *,
    config: VmRunnerConfig,
    symbol: str,
    loops: int,
    mode: str,
    live_after_backtest: bool,
    submitted_at: str,
    log_file_path: str,
) -> list[str]:
    project_dir = shlex.quote(config.project_dir)
    venv_activate = shlex.quote(config.venv_activate)
    quoted_symbol = shlex.quote(symbol)
    quoted_mode = shlex.quote(mode)
    quoted_log_dir = shlex.quote(str(Path(log_file_path).parent))
    quoted_log_file = shlex.quote(log_file_path)
    primary_run_command = (
        "python -m fresh_simple_trading_project.cli run "
        f"--mode {quoted_mode} --symbol {quoted_symbol}"
    )
    if mode == BACKTEST_MODE:
        primary_run_command = f"{primary_run_command} --max-iterations {loops}"
    lines = [
        "set -euo pipefail",
        f"mkdir -p {quoted_log_dir}",
        f"exec >> {quoted_log_file} 2>&1",
        'trap \'exit_code=$?; echo "[Dispatcher] Workflow run finished with exit_code=${exit_code}"\' EXIT',
        f'echo "[Dispatcher] Submitted at {submitted_at}"',
        f'echo "[Dispatcher] Logging to {log_file_path}"',
        (
            f'echo "[Dispatcher] Starting workflow dispatch: symbol={symbol} '
            f'mode={mode} loops={loops} live_after_backtest={str(live_after_backtest).lower()}"'
        ),
        f"cd {project_dir}",
        f"source {venv_activate}",
        'export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"',
        primary_run_command,
    ]
    if live_after_backtest and mode == BACKTEST_MODE:
        lines.append('echo "[Dispatcher] Backtest finished. Starting live follow-up run."')
        lines.append(
            "python -m fresh_simple_trading_project.cli run "
            f"--mode {LIVE_MODE} --symbol {quoted_symbol}"
        )
    if config.auto_shutdown:
        lines.append('echo "[Dispatcher] Auto shutdown requested."')
        lines.append("sudo shutdown -h now")
    return lines


def _build_run_command_input(script_lines: list[str]):
    from azure.mgmt.compute.models import RunCommandInput

    return RunCommandInput(command_id="RunShellScript", script=script_lines)


def _build_log_tail_script(*, log_file_path: str, lines: int) -> list[str]:
    return _build_log_read_script(log_file_path=log_file_path, lines=lines)


def _build_log_read_script(*, log_file_path: str, lines: int | None = None) -> list[str]:
    quoted_log_file = shlex.quote(log_file_path)
    read_command = f"cat {quoted_log_file}" if lines is None else f"tail -n {lines} {quoted_log_file}"
    return [
        "set -euo pipefail",
        f'if [ ! -f {quoted_log_file} ]; then echo "{LOG_OUTPUT_ERROR_MARKER} Log file not found: {log_file_path}"; exit 3; fi',
        f'echo "{LOG_OUTPUT_START_MARKER}"',
        read_command,
        f'echo "{LOG_OUTPUT_END_MARKER}"',
    ]


def _extract_run_command_output(run_command_result: Any) -> str:
    if run_command_result is None:
        return ""

    messages: list[str] = []
    for attribute in ("output", "error", "execution_message"):
        value = getattr(run_command_result, attribute, None)
        if value:
            messages.append(str(value))

    for status in getattr(run_command_result, "value", []) or []:
        message = getattr(status, "message", None)
        if message:
            messages.append(str(message))

    if not messages:
        return ""
    return "\n".join(messages)


def _normalize_run_command_output(raw_output: str) -> str:
    normalized = raw_output.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        try:
            decoded = json.loads(normalized)
            if isinstance(decoded, str):
                return decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
        except json.JSONDecodeError:
            pass
    return normalized


def _strip_run_command_wrappers(raw_output: str) -> str:
    ignored_prefixes = {
        "Enable succeeded:",
        "Enable failed:",
        "[stdout]",
        "[stderr]",
    }
    filtered_lines = [line for line in raw_output.splitlines() if line.strip() not in ignored_prefixes]
    return "\n".join(filtered_lines).strip()


def _extract_marked_output(raw_output: str) -> str | None:
    normalized = _normalize_run_command_output(raw_output)
    start_index = normalized.find(LOG_OUTPUT_START_MARKER)
    end_index = normalized.find(LOG_OUTPUT_END_MARKER)

    if start_index != -1:
        start_index += len(LOG_OUTPUT_START_MARKER)
        if end_index == -1 or end_index < start_index:
            return normalized[start_index:].strip("\n")
        return normalized[start_index:end_index].strip("\n")

    if end_index != -1:
        return normalized[:end_index].strip("\n")

    fallback = _strip_run_command_wrappers(normalized)
    return fallback or None


def _run_vm_shell_script_sync(compute_client: Any, config: VmRunnerConfig, script_lines: list[str]) -> Any:
    return compute_client.virtual_machines.begin_run_command(
        config.resource_group,
        config.vm_name,
        _build_run_command_input(script_lines),
    ).result()


def _confirm_vm_runner_launch(
    *,
    compute_client: Any,
    config: VmRunnerConfig,
    symbol: str,
    loops: int,
    mode: str,
    live_after_backtest: bool,
    submitted_at: str,
    log_file_path: str,
) -> None:
    result = _run_vm_shell_script_sync(
        compute_client,
        config,
        _build_run_command_script(
            config=config,
            symbol=symbol,
            loops=loops,
            mode=mode,
            live_after_backtest=live_after_backtest,
            submitted_at=submitted_at,
            log_file_path=log_file_path,
        ),
    )
    raw_output = _extract_run_command_output(result)
    normalized = _normalize_run_command_output(raw_output)
    if RUN_COMMAND_LAUNCH_SUCCESS_MARKER in normalized:
        return

    if RUN_COMMAND_LAUNCH_ERROR_MARKER in normalized:
        error_message = normalized.split(RUN_COMMAND_LAUNCH_ERROR_MARKER, 1)[1].strip()
    else:
        error_message = _strip_run_command_wrappers(normalized)
    raise VmDispatchLaunchError(
        error_message or "VM run command did not confirm that the workflow runner started."
    )


def _fetch_vm_log_tail(
    *,
    compute_client: Any,
    config: VmRunnerConfig,
    log_file_path: str,
    lines: int,
) -> str:
    return _fetch_vm_log(
        compute_client=compute_client,
        config=config,
        log_file_path=log_file_path,
        lines=lines,
    )


def _fetch_vm_log(
    *,
    compute_client: Any,
    config: VmRunnerConfig,
    log_file_path: str,
    lines: int | None = None,
) -> str:
    result = _run_vm_shell_script_sync(
        compute_client,
        config,
        _build_log_read_script(log_file_path=log_file_path, lines=lines),
    )
    raw_output = _extract_run_command_output(result)
    if LOG_OUTPUT_ERROR_MARKER in raw_output:
        error_message = raw_output.split(LOG_OUTPUT_ERROR_MARKER, 1)[1].strip()
        raise VmLogAccessError(error_message or f"Log file not found: {log_file_path}")
    content = _extract_marked_output(raw_output)
    if content is None:
        raise VmLogAccessError("VM log tail output could not be parsed from Run Command response.")
    return content


def _record_dispatch(
    *,
    config: VmRunnerConfig,
    symbol: str,
    loops: int,
    mode: str,
    live_after_backtest: bool,
    trigger_source: str,
    start_requested: bool,
    power_state: str | None,
    submitted_at: str,
    log_file_path: str,
) -> dict[str, Any]:
    state = {
        "accepted": True,
        "active": True,
        "symbol": symbol,
        "loops": loops,
        "mode": mode,
        "live_after_backtest": live_after_backtest,
        "trigger_source": trigger_source,
        "submitted_at": submitted_at,
        "vm_name": config.vm_name,
        "resource_group": config.resource_group,
        "start_requested": start_requested,
        "power_state": power_state,
        "log_file_path": log_file_path,
        "log_tail_command": f"tail -f {shlex.quote(log_file_path)}",
    }
    _save_dispatch_state(state)
    return state


def launch_vm_run(
    *,
    symbol: str,
    loops: int,
    mode: str,
    live_after_backtest: bool,
    trigger_source: str,
    force: bool = False,
) -> dict[str, Any]:
    config = _load_vm_runner_config()
    compute_client = _build_compute_client(config)
    previous_state = _load_dispatch_state()
    power_state = _get_vm_power_state(compute_client, config)

    if _dispatch_is_active(previous_state, power_state) and not force:
        raise DispatchConflictError(
            f"VM dispatch already appears active for {config.vm_name} (power_state={power_state or 'unknown'})."
        )

    start_requested = False
    if power_state != "running":
        logging.info(
            "Starting VM %s in resource group %s before dispatch. Current power state: %s",
            config.vm_name,
            config.resource_group,
            power_state or "unknown",
        )
        compute_client.virtual_machines.begin_start(config.resource_group, config.vm_name).result()
        start_requested = True
        power_state = "running"

    submitted_at = _utcnow_iso()
    log_file_path = _dispatch_log_file_path(
        config=config,
        symbol=symbol,
        mode=mode,
        trigger_source=trigger_source,
        submitted_at=submitted_at,
    )

    logging.info(
        "Dispatching VM run command for %s with symbol=%s loops=%s mode=%s via %s trigger. log_file_path=%s",
        config.vm_name,
        symbol,
        loops,
        mode,
        trigger_source,
        log_file_path,
    )
    _confirm_vm_runner_launch(
        compute_client=compute_client,
        config=config,
        symbol=symbol,
        loops=loops,
        mode=mode,
        live_after_backtest=live_after_backtest,
        submitted_at=submitted_at,
        log_file_path=log_file_path,
    )
    return _record_dispatch(
        config=config,
        symbol=symbol,
        loops=loops,
        mode=mode,
        live_after_backtest=live_after_backtest,
        trigger_source=trigger_source,
        start_requested=start_requested,
        power_state=power_state,
        submitted_at=submitted_at,
        log_file_path=log_file_path,
    )


def _default_dispatch_values(config: VmRunnerConfig) -> tuple[str, int, str]:
    return (
        config.default_symbol.upper(),
        config.default_loops,
        config.default_mode,
    )


def _build_status_payload() -> dict[str, Any]:
    state = _load_dispatch_state()
    try:
        config = _load_vm_runner_config()
        compute_client = _build_compute_client(config)
        power_state = _get_vm_power_state(compute_client, config)
        state["vm_name"] = config.vm_name
        state["resource_group"] = config.resource_group
        state["power_state"] = power_state
        state["active"] = _dispatch_is_active(state, power_state)
    except Exception as exc:
        state["vm_status_error"] = str(exc)
    return state


def _log_response(
    *,
    message: str,
    log_file_path: str,
    power_state: str | None,
    lines: int,
    content: str | None,
    status_code: int = 200,
) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(
            {
                "message": message,
                "log_file_path": log_file_path,
                "power_state": power_state,
                "lines": lines,
                "content": content,
            },
            indent=2,
        ),
        status_code=status_code,
        mimetype="application/json",
    )


def _log_download_response(*, log_file_path: str, content: str) -> func.HttpResponse:
    return func.HttpResponse(
        content,
        status_code=200,
        mimetype="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(log_file_path).name}"',
        },
    )


# The schedule "0 0 * * * *" means:
# Run at the start (0 seconds, 0 minutes) of every hour.
@app.timer_trigger(schedule="0 0 * * * *", arg_name="myTimer", run_on_startup=False, use_monitor=True)
def trading_timer_trigger(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.info("The timer is past due!")

    config = _load_vm_runner_config()
    if not config.timer_enabled:
        logging.info("VM timer dispatch is disabled. Skipping hourly timer execution.")
        return

    symbol, loops, mode = _default_dispatch_values(config)
    if mode != LIVE_MODE:
        logging.info(
            "VM timer dispatch only supports live mode. Current VM_DEFAULT_MODE=%s, skipping hourly timer execution.",
            mode,
        )
        return
    try:
        dispatch = launch_vm_run(
            symbol=symbol,
            loops=loops,
            mode=mode,
            live_after_backtest=False,
            trigger_source="timer",
            force=False,
        )
    except DispatchConflictError as exc:
        logging.info("Skipping timer dispatch because a previous VM run still appears active: %s", str(exc))
        return

    logging.info(
        "Timer dispatch accepted for VM %s. symbol=%s loops=%s mode=%s",
        dispatch["vm_name"],
        dispatch["symbol"],
        dispatch["loops"],
        dispatch["mode"],
    )


@app.route(route="trading/vm/start", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def start_trading_vm(req: func.HttpRequest) -> func.HttpResponse:
    """Start a VM-backed trading run and return dispatch details."""
    payload = _request_payload(req)
    config = _load_vm_runner_config()
    default_symbol, default_loops, default_mode = _default_dispatch_values(config)

    try:
        loops = _parse_positive_int(_request_value(req, payload, "loops"), default=default_loops)
        mode = _parse_mode(_request_value(req, payload, "mode"), default=default_mode)
    except ValueError as exc:
        return func.HttpResponse(str(exc), status_code=400)

    symbol = str(_request_value(req, payload, "symbol") or default_symbol).upper()
    force = _parse_bool(_request_value(req, payload, "force"), False)
    live_after_backtest = _parse_bool(_request_value(req, payload, "live_after_backtest"), False)
    if live_after_backtest and mode != BACKTEST_MODE:
        return func.HttpResponse("live_after_backtest can only be used when mode=backtest", status_code=400)

    try:
        dispatch = launch_vm_run(
            symbol=symbol,
            loops=loops,
            mode=mode,
            live_after_backtest=live_after_backtest,
            trigger_source="http",
            force=force,
        )
    except DispatchConflictError as exc:
        return _response(
            message=f"A VM trading dispatch is already active. Use force=true to replace it. {exc}",
            dispatch=_build_status_payload(),
            status_code=409,
        )
    except VmDispatchLaunchError as exc:
        return _response(
            message=f"Trading VM dispatch failed before the workflow runner started. {exc}",
            dispatch=_build_status_payload(),
            status_code=500,
        )

    return _response(
        message="Trading VM dispatch accepted.",
        dispatch=_dispatch_with_log_urls(req, dispatch),
        status_code=202,
    )


@app.route(route="trading/vm/status", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def trading_vm_status(req: func.HttpRequest) -> func.HttpResponse:
    """Return the latest VM trading dispatch status."""
    del req
    return _response(
        message="Current VM trading dispatch status.",
        dispatch=_build_status_payload(),
    )


@app.route(route="trading/vm/log", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def trading_vm_log(req: func.HttpRequest) -> func.HttpResponse:
    """Return the current VM log tail or the full log as a download."""
    state = _load_dispatch_state()
    log_file_path = str(req.params.get("log_file_path") or state.get("log_file_path") or "").strip()
    if not log_file_path:
        return func.HttpResponse(
            "No VM log file is available yet. Trigger a run first or provide log_file_path.",
            status_code=404,
        )

    lines = _parse_log_tail_lines(req.params.get("lines"), default=DEFAULT_LOG_TAIL_LINES)
    start_if_needed = _parse_bool(req.params.get("start_if_needed"), False)
    download = _parse_bool(req.params.get("download"), False)

    try:
        config = _load_vm_runner_config()
        compute_client = _build_compute_client(config)
        power_state = _get_vm_power_state(compute_client, config)

        if power_state != "running":
            if not start_if_needed:
                return _log_response(
                    message="VM is not running. Set start_if_needed=true to start the VM before reading logs.",
                    log_file_path=log_file_path,
                    power_state=power_state,
                    lines=lines,
                    content=None,
                    status_code=409,
                )
            compute_client.virtual_machines.begin_start(config.resource_group, config.vm_name).result()
            power_state = "running"

        content = _fetch_vm_log(
            compute_client=compute_client,
            config=config,
            log_file_path=log_file_path,
            lines=None if download else lines,
        )
    except VmLogAccessError as exc:
        return _log_response(
            message=str(exc),
            log_file_path=log_file_path,
            power_state="running",
            lines=lines,
            content=None,
            status_code=404,
        )
    except Exception as exc:
        return _log_response(
            message=f"Failed to fetch VM log tail. {exc}",
            log_file_path=log_file_path,
            power_state=None,
            lines=lines,
            content=None,
            status_code=500,
        )

    if download:
        return _log_download_response(log_file_path=log_file_path, content=content)

    return _log_response(
        message="Current VM workflow log tail.",
        log_file_path=log_file_path,
        power_state=power_state,
        lines=lines,
        content=content,
    )


@app.route(route="trading/session/start", methods=["GET", "POST"], auth_level=func.AuthLevel.FUNCTION)
def start_trading_session(req: func.HttpRequest) -> func.HttpResponse:
    """Compatibility route for older trading session start callers."""
    return start_trading_vm(req)


@app.route(route="trading/session/status", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def trading_session_status(req: func.HttpRequest) -> func.HttpResponse:
    """Compatibility route for older trading session status callers."""
    return trading_vm_status(req)


@app.route(route="trading/session/log", methods=["GET"], auth_level=func.AuthLevel.FUNCTION)
def trading_session_log(req: func.HttpRequest) -> func.HttpResponse:
    """Compatibility route for older trading session log callers."""
    return trading_vm_log(req)
