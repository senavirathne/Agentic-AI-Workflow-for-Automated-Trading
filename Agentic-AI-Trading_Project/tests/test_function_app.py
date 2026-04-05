from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import azure.functions as func
except ModuleNotFoundError:
    azure_module = types.ModuleType("azure")
    functions_module = types.ModuleType("azure.functions")

    class HttpRequest:
        def __init__(self, *, method, url, headers=None, params=None, route_params=None, body=b""):
            self.method = method
            self.url = url
            self.headers = headers or {}
            self.params = params or {}
            self.route_params = route_params or {}
            self._body = body

        def get_json(self):
            if not self._body:
                raise ValueError("no body")
            return json.loads(self._body.decode("utf-8"))

    class HttpResponse:
        def __init__(self, body="", *, status_code=200, mimetype=None, headers=None):
            self._body = body.encode("utf-8") if isinstance(body, str) else body
            self.status_code = status_code
            self.mimetype = mimetype
            self.headers = headers or {}

        def get_body(self):
            return self._body

    class FunctionApp:
        def timer_trigger(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def route(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    class AuthLevel:
        FUNCTION = "function"

    class TimerRequest:
        pass

    functions_module.HttpRequest = HttpRequest
    functions_module.HttpResponse = HttpResponse
    functions_module.FunctionApp = FunctionApp
    functions_module.AuthLevel = AuthLevel
    functions_module.TimerRequest = TimerRequest
    azure_module.functions = functions_module
    sys.modules["azure"] = azure_module
    sys.modules["azure.functions"] = functions_module
    import azure.functions as func

import function_app as function_app_module


class FakeStatus:
    def __init__(self, code: str) -> None:
        self.code = code


class FakeInstanceView:
    def __init__(self, power_state: str) -> None:
        self.statuses = [FakeStatus(f"PowerState/{power_state}")]


class FakePoller:
    def __init__(self, result_value=None) -> None:
        self._result_value = result_value

    def result(self):
        return self._result_value


class FakeRunCommandStatus:
    def __init__(self, message: str) -> None:
        self.message = message


class FakeRunCommandResult:
    def __init__(self, *messages: str) -> None:
        self.value = [FakeRunCommandStatus(message) for message in messages]


class FakeVirtualMachinesClient:
    def __init__(self, power_state: str, run_command_result=None) -> None:
        self.power_state = power_state
        self.run_command_result = run_command_result or FakeRunCommandResult(
            function_app_module.RUN_COMMAND_LAUNCH_SUCCESS_MARKER
        )
        self.start_calls: list[tuple[str, str]] = []
        self.run_command_calls: list[tuple[str, str, object]] = []

    def instance_view(self, resource_group: str, vm_name: str):
        return FakeInstanceView(self.power_state)

    def begin_start(self, resource_group: str, vm_name: str):
        self.start_calls.append((resource_group, vm_name))
        self.power_state = "running"
        return FakePoller()

    def begin_run_command(self, resource_group: str, vm_name: str, payload: object):
        self.run_command_calls.append((resource_group, vm_name, payload))
        return FakePoller(self.run_command_result)


class FakeComputeClient:
    def __init__(self, power_state: str, run_command_result=None) -> None:
        self.virtual_machines = FakeVirtualMachinesClient(power_state, run_command_result=run_command_result)


class DummyTimer:
    def __init__(self, past_due: bool = False) -> None:
        self.past_due = past_due


@pytest.fixture(autouse=True)
def vm_function_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "FUNCTIONS_WORKER_RUNTIME",
        "FUNCTION_APP_STORAGE_ROOT",
        "WEBSITE_INSTANCE_ID",
        "WEBSITE_SITE_NAME",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_VM_RESOURCE_GROUP",
        "AZURE_VM_NAME",
        "VM_PROJECT_DIR",
        "VM_VENV_ACTIVATE",
        "VM_DEFAULT_SYMBOL",
        "VM_DEFAULT_LOOPS",
        "VM_DEFAULT_MODE",
        "VM_TIMER_ENABLED",
        "VM_AUTO_SHUTDOWN",
        "VM_AUTO_SHUTDOWN_DELAY_SECONDS",
        "VM_LOG_DIR",
        "VM_LOG_BLOB_CONTAINER",
        "VM_LOG_BLOB_PREFIX",
        "VM_LOG_BLOB_ACCOUNT_URL",
        "VM_LOG_BLOB_CONNECTION_STRING",
        "VM_LOG_BLOB_SHARE_TTL_HOURS",
        "TRADING_SYMBOL",
        "RUN_MODE",
        "AZURE_STORAGE_ACCOUNT_URL",
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_BLOB_CONTAINER_LOGS",
        "AzureWebJobsStorage",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("FUNCTIONS_WORKER_RUNTIME", "python")
    monkeypatch.setenv("FUNCTION_APP_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-id")
    monkeypatch.setenv("AZURE_VM_RESOURCE_GROUP", "rg-test")
    monkeypatch.setenv("AZURE_VM_NAME", "vm-test")
    monkeypatch.setenv("VM_PROJECT_DIR", "/opt/fresh_simple_trading_project")
    monkeypatch.setenv("VM_VENV_ACTIVATE", "/opt/fresh_simple_trading_project/.venv/bin/activate")
    monkeypatch.setenv("VM_DEFAULT_SYMBOL", "MSFT")
    monkeypatch.setenv("VM_DEFAULT_LOOPS", "2")
    monkeypatch.setenv("VM_DEFAULT_MODE", "live")
    monkeypatch.setenv("VM_TIMER_ENABLED", "false")
    monkeypatch.setenv("VM_AUTO_SHUTDOWN", "true")
    monkeypatch.setenv("VM_AUTO_SHUTDOWN_DELAY_SECONDS", "900")
    monkeypatch.setenv("VM_LOG_DIR", "/opt/fresh_simple_trading_project/logs")
    monkeypatch.setattr(
        function_app_module,
        "_build_run_command_input",
        lambda script_lines: {"command_id": "RunShellScript", "script": script_lines},
    )


def test_dispatch_state_path_uses_temp_storage_without_project_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FUNCTION_APP_STORAGE_ROOT", raising=False)

    state_path = function_app_module._dispatch_state_path()

    assert state_path == Path(tempfile.gettempdir()).resolve() / "fresh_simple_trading_project" / "data" / "vm_dispatch_state.json"


def test_start_vm_uses_defaults_when_payload_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("deallocated")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    monkeypatch.setattr(function_app_module, "_utcnow_iso", lambda: "2026-04-05T09:32:43+00:00")

    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b"{}",
    )

    response = function_app_module.start_trading_vm(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 202
    assert payload["dispatch"]["symbol"] == "MSFT"
    assert payload["dispatch"]["loops"] == 2
    assert payload["dispatch"]["mode"] == "live"
    assert payload["dispatch"]["live_after_backtest"] is False
    assert payload["dispatch"]["start_requested"] is True
    assert payload["dispatch"]["log_file_path"] == "/opt/fresh_simple_trading_project/logs/workflow_live_msft_http_20260405T093243Z.log"
    assert payload["dispatch"]["log_tail_command"] == (
        "tail -f /opt/fresh_simple_trading_project/logs/workflow_live_msft_http_20260405T093243Z.log"
    )
    assert fake_compute.virtual_machines.start_calls == [("rg-test", "vm-test")]
    assert len(fake_compute.virtual_machines.run_command_calls) == 1


def test_start_vm_rejects_invalid_loops() -> None:
    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b'{"loops": 0}',
    )

    response = function_app_module.start_trading_vm(req)

    assert response.status_code == 400
    assert response.get_body().decode("utf-8") == "loops must be a positive integer"


def test_start_vm_accepts_get_query_params_and_returns_log_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("deallocated")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    monkeypatch.setattr(function_app_module, "_utcnow_iso", lambda: "2026-04-05T09:32:43+00:00")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")
    monkeypatch.setenv("VM_LOG_BLOB_CONTAINER", "workflow-logs")
    monkeypatch.setattr(
        function_app_module,
        "_shareable_blob_url",
        lambda **kwargs: "https://example.blob.core.windows.net/workflow-logs/share.log?sas-token",
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/session/start?code=test-key",
        headers={},
        params={"symbol": "AAPL", "loops": "3", "mode": "backtest", "code": "test-key"},
        route_params={},
        body=b"",
    )

    response = function_app_module.start_trading_session(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 202
    assert payload["dispatch"]["symbol"] == "AAPL"
    assert payload["dispatch"]["loops"] == 3
    assert payload["dispatch"]["mode"] == "backtest"
    assert payload["dispatch"]["log_url"] == (
        "http://localhost/api/trading/session/log"
        "?code=test-key"
        "&log_file_path=%2Fopt%2Ffresh_simple_trading_project%2Flogs%2Fworkflow_backtest_aapl_http_20260405T093243Z.log"
        "&start_if_needed=true"
        "&wait_for_running_seconds=90"
    )
    assert payload["dispatch"]["log_download_url"] == (
        "http://localhost/api/trading/session/log"
        "?code=test-key"
        "&log_file_path=%2Fopt%2Ffresh_simple_trading_project%2Flogs%2Fworkflow_backtest_aapl_http_20260405T093243Z.log"
        "&start_if_needed=true"
        "&wait_for_running_seconds=90"
        "&download=true"
    )
    assert payload["dispatch"]["blob_log_url"] == (
        "https://example.blob.core.windows.net/workflow-logs/logs/2026/04/05/"
        "workflow_backtest_aapl_http_20260405T093243Z.log"
    )
    assert payload["dispatch"]["blob_log_share_url"] == (
        "https://example.blob.core.windows.net/workflow-logs/share.log?sas-token"
    )


def test_start_vm_force_parses_from_json_and_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("running")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": True,
            "symbol": "MSFT",
            "loops": 2,
            "mode": "live",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T00:00:00+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "running",
        }
    )

    req_without_force = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b'{"symbol": "AAPL"}',
    )
    assert function_app_module.start_trading_vm(req_without_force).status_code == 409

    req_force_json = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b'{"symbol": "AAPL", "force": true}',
    )
    assert function_app_module.start_trading_vm(req_force_json).status_code == 202

    req_force_query = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={"force": "true"},
        route_params={},
        body=b'{"symbol": "AAPL"}',
    )
    assert function_app_module.start_trading_vm(req_force_query).status_code == 202


def test_launch_vm_run_starts_stopped_vm_and_runs_command(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("deallocated")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    monkeypatch.setattr(function_app_module, "_utcnow_iso", lambda: "2026-04-05T09:32:43+00:00")

    dispatch = function_app_module.launch_vm_run(
        symbol="AAPL",
        loops=3,
        mode="live",
        live_after_backtest=False,
        trigger_source="http",
    )

    assert dispatch["accepted"] is True
    assert dispatch["start_requested"] is True
    assert dispatch["log_file_path"] == "/opt/fresh_simple_trading_project/logs/workflow_live_aapl_http_20260405T093243Z.log"
    assert fake_compute.virtual_machines.start_calls == [("rg-test", "vm-test")]
    assert len(fake_compute.virtual_machines.run_command_calls) == 1


def test_launch_vm_run_skips_start_when_vm_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("running")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    monkeypatch.setattr(function_app_module, "_utcnow_iso", lambda: "2026-04-05T09:32:43+00:00")

    dispatch = function_app_module.launch_vm_run(
        symbol="AAPL",
        loops=2,
        mode="backtest",
        live_after_backtest=False,
        trigger_source="http",
        force=True,
    )

    assert dispatch["accepted"] is True
    assert dispatch["start_requested"] is False
    assert fake_compute.virtual_machines.start_calls == []
    assert len(fake_compute.virtual_machines.run_command_calls) == 1


def test_launch_vm_run_raises_when_run_command_does_not_confirm_runner_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_compute = FakeComputeClient(
        "running",
        run_command_result=FakeRunCommandResult("Enable succeeded:", "[stdout]"),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    monkeypatch.setattr(function_app_module, "_utcnow_iso", lambda: "2026-04-05T09:32:43+00:00")

    with pytest.raises(
        function_app_module.VmDispatchLaunchError,
        match="did not confirm that the workflow runner started",
    ):
        function_app_module.launch_vm_run(
            symbol="AAPL",
            loops=2,
            mode="backtest",
            live_after_backtest=False,
            trigger_source="http",
            force=True,
        )

    assert function_app_module._load_dispatch_state()["accepted"] is False


def test_timer_trigger_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VM_TIMER_ENABLED", "false")
    called = {"launch": False}

    def fail_launch(**kwargs):
        called["launch"] = True
        raise AssertionError("launch_vm_run should not be called when the timer is disabled")

    monkeypatch.setattr(function_app_module, "launch_vm_run", fail_launch)

    function_app_module.trading_timer_trigger(DummyTimer())

    assert called["launch"] is False


def test_timer_trigger_skips_when_default_mode_is_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VM_TIMER_ENABLED", "true")
    monkeypatch.setenv("VM_DEFAULT_MODE", "backtest")
    called = {"launch": False}

    def fail_launch(**kwargs):
        called["launch"] = True
        raise AssertionError("launch_vm_run should not be called for hourly backtest dispatches")

    monkeypatch.setattr(function_app_module, "launch_vm_run", fail_launch)

    function_app_module.trading_timer_trigger(DummyTimer())

    assert called["launch"] is False


def test_build_run_command_script_includes_shutdown_only_when_enabled() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    with_shutdown_live = function_app_module._build_run_command_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="live",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
    )
    with_shutdown_backtest = function_app_module._build_run_command_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="backtest",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
    )
    without_shutdown = function_app_module._build_run_command_script(
        config=function_app_module.VmRunnerConfig(**{**config.__dict__, "auto_shutdown": False}),
        symbol="AAPL",
        loops=4,
        mode="live",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
    )

    assert with_shutdown_live[1].startswith("nohup bash -lc ")
    assert "workflow_live_aapl_http_20260405T093243Z.log" in with_shutdown_live[1]
    assert "--mode live --symbol AAPL" in with_shutdown_live[1]
    assert "--max-iterations 4" not in with_shutdown_live[1]
    assert "--mode backtest --symbol AAPL --max-iterations 4" in with_shutdown_backtest[1]
    assert "sudo shutdown -h now" in with_shutdown_live[1]
    assert "sudo shutdown -h now" not in without_shutdown[1]


def test_build_workflow_runner_script_includes_log_redirection() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    lines = function_app_module._build_workflow_runner_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="live",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
    )

    assert lines[0] == "set -euo pipefail"
    assert lines[1] == "mkdir -p /srv/trader/logs"
    assert lines[2] == "exec >> /srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log 2>&1"
    assert any("--mode live --symbol AAPL" in line for line in lines)
    assert lines[-2] == "sleep 900"
    assert lines[-1] == "sudo shutdown -h now"


def test_build_workflow_runner_script_uploads_blob_and_delays_shutdown() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
        auto_shutdown_delay_seconds=900,
        log_blob_container="workflow-logs",
        log_blob_account_url="https://example.blob.core.windows.net",
    )

    lines = function_app_module._build_workflow_runner_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="live",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
        blob_log_blob_name="logs/2026/04/05/workflow_live_aapl_http_20260405T093243Z.log",
    )

    assert any("python -m fresh_simple_trading_project.log_blob_uploader" in line for line in lines)
    assert any("--container workflow-logs" in line for line in lines)
    assert any(
        "--blob-name logs/2026/04/05/workflow_live_aapl_http_20260405T093243Z.log" in line for line in lines
    )
    assert lines[-2] == "sleep 900"
    assert lines[-1] == "sudo shutdown -h now"


def test_build_run_command_script_waits_for_log_file_before_reporting_success() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    lines = function_app_module._build_run_command_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="live",
        live_after_backtest=False,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
    )

    assert lines[0] == "set -eu"
    assert lines[2] == "runner_pid=$!"
    assert lines[3] == f"for _ in $(seq 1 {function_app_module.RUN_COMMAND_LOG_FILE_WAIT_SECONDS}); do"
    assert lines[4] == "  if [ -f /srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log ]; then"
    assert function_app_module.RUN_COMMAND_LAUNCH_SUCCESS_MARKER in lines[5]
    assert function_app_module.RUN_COMMAND_LAUNCH_ERROR_MARKER in lines[10]


def test_build_log_read_script_uses_posix_safe_shell_options() -> None:
    lines = function_app_module._build_log_read_script(
        log_file_path="/srv/trader/logs/workflow_live_aapl_http_20260405T093243Z.log",
        lines=50,
    )

    assert lines[0] == "set -eu"


def test_start_vm_rejects_live_after_backtest_for_live_mode() -> None:
    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b'{"mode": "live", "live_after_backtest": true}',
    )

    response = function_app_module.start_trading_vm(req)

    assert response.status_code == 400
    assert response.get_body().decode("utf-8") == "live_after_backtest can only be used when mode=backtest"


def test_start_vm_returns_500_when_vm_runner_launch_is_not_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient(
        "deallocated",
        run_command_result=FakeRunCommandResult(
            f"{function_app_module.RUN_COMMAND_LAUNCH_ERROR_MARKER} Log file was not created: /tmp/missing.log"
        ),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)

    req = func.HttpRequest(
        method="POST",
        url="http://localhost/api/trading/vm/start",
        headers={},
        params={},
        route_params={},
        body=b'{"symbol": "AAPL"}',
    )

    response = function_app_module.start_trading_vm(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 500
    assert payload["message"] == (
        "Trading VM dispatch failed before the workflow runner started. "
        "Log file was not created: /tmp/missing.log"
    )


def test_build_run_command_script_appends_live_run_after_backtest() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    lines = function_app_module._build_run_command_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="backtest",
        live_after_backtest=True,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
    )

    assert "--mode backtest --symbol AAPL --max-iterations 4" in lines[1]
    assert "python -m fresh_simple_trading_project.cli run --mode live --symbol AAPL" in lines[1]
    assert "sudo shutdown -h now" in lines[1]


def test_build_run_command_script_orders_backtest_then_live_then_shutdown() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    lines = function_app_module._build_run_command_script(
        config=config,
        symbol="AAPL",
        loops=4,
        mode="backtest",
        live_after_backtest=True,
        submitted_at="2026-04-05T09:32:43+00:00",
        log_file_path="/srv/trader/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
    )

    wrapper_line = lines[1]
    backtest_index = wrapper_line.index("--mode backtest --symbol AAPL --max-iterations 4")
    live_index = wrapper_line.index("python -m fresh_simple_trading_project.cli run --mode live --symbol AAPL")
    shutdown_index = wrapper_line.index("sudo shutdown -h now")

    assert backtest_index < live_index < shutdown_index


def test_dispatch_log_file_path_uses_mode_symbol_trigger_and_timestamp() -> None:
    config = function_app_module.VmRunnerConfig(
        subscription_id="sub-id",
        resource_group="rg-test",
        vm_name="vm-test",
        project_dir="/srv/trader",
        venv_activate="/srv/trader/.venv/bin/activate",
        default_symbol="AAPL",
        default_loops=1,
        default_mode="live",
        timer_enabled=False,
        auto_shutdown=True,
        log_dir="/srv/trader/logs",
    )

    path = function_app_module._dispatch_log_file_path(
        config=config,
        symbol="BRK.B",
        mode="backtest",
        trigger_source="http",
        submitted_at="2026-04-05T09:32:43+00:00",
    )

    assert path == "/srv/trader/logs/workflow_backtest_brk_b_http_20260405T093243Z.log"


def test_extract_marked_output_returns_log_content() -> None:
    content = function_app_module._extract_marked_output(
        "\n".join(
            [
                "Enable succeeded:",
                function_app_module.LOG_OUTPUT_START_MARKER,
                "line one",
                "line two",
                function_app_module.LOG_OUTPUT_END_MARKER,
            ]
        )
    )

    assert content == "line one\nline two"


def test_extract_marked_output_falls_back_to_wrapper_stripped_output() -> None:
    content = function_app_module._extract_marked_output(
        "\n".join(
            [
                "Enable succeeded:",
                "[stdout]",
                "decision line",
                "risk line",
            ]
        )
    )

    assert content == "decision line\nrisk line"


def test_trading_vm_log_returns_latest_log_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient(
        "running",
        run_command_result=FakeRunCommandResult(
            "\n".join(
                [
                    "Enable succeeded:",
                    function_app_module.LOG_OUTPUT_START_MARKER,
                    "decision line",
                    "risk line",
                    function_app_module.LOG_OUTPUT_END_MARKER,
                ]
            )
        ),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": True,
            "symbol": "AAPL",
            "loops": 1,
            "mode": "backtest",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T09:32:43+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "running",
            "log_file_path": "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
            "log_tail_command": "tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
        }
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/vm/log",
        headers={},
        params={"lines": "20"},
        route_params={},
        body=b"",
    )

    response = function_app_module.trading_vm_log(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 200
    assert payload["log_file_path"] == "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log"
    assert payload["lines"] == 20
    assert payload["content"] == "decision line\nrisk line"
    assert len(fake_compute.virtual_machines.run_command_calls) == 1


def test_trading_vm_log_returns_409_when_vm_is_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient("deallocated")
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": False,
            "symbol": "AAPL",
            "loops": 1,
            "mode": "backtest",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T09:32:43+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "deallocated",
            "log_file_path": "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
            "log_tail_command": "tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
        }
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/vm/log",
        headers={},
        params={},
        route_params={},
        body=b"",
    )

    response = function_app_module.trading_vm_log(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 409
    assert payload["power_state"] == "deallocated"
    assert payload["content"] is None
    assert fake_compute.virtual_machines.start_calls == []


def test_trading_vm_log_can_start_vm_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient(
        "deallocated",
        run_command_result=FakeRunCommandResult(
            "\n".join(
                [
                    function_app_module.LOG_OUTPUT_START_MARKER,
                    "decision line",
                    function_app_module.LOG_OUTPUT_END_MARKER,
                ]
            )
        ),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": False,
            "symbol": "AAPL",
            "loops": 1,
            "mode": "backtest",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T09:32:43+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "deallocated",
            "log_file_path": "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
            "log_tail_command": "tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
        }
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/vm/log",
        headers={},
        params={"start_if_needed": "true"},
        route_params={},
        body=b"",
    )

    response = function_app_module.trading_vm_log(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 200
    assert payload["power_state"] == "running"
    assert payload["content"] == "decision line"
    assert fake_compute.virtual_machines.start_calls == [("rg-test", "vm-test")]


def test_trading_vm_log_waits_for_vm_to_finish_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient(
        "starting",
        run_command_result=FakeRunCommandResult(
            "\n".join(
                [
                    function_app_module.LOG_OUTPUT_START_MARKER,
                    "decision line",
                    function_app_module.LOG_OUTPUT_END_MARKER,
                ]
            )
        ),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    states = iter(["starting", "running"])
    monkeypatch.setattr(function_app_module, "_get_vm_power_state", lambda compute_client, config: next(states))
    monkeypatch.setattr(function_app_module.time, "sleep", lambda _: None)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": True,
            "symbol": "AAPL",
            "loops": 1,
            "mode": "backtest",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T09:32:43+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "starting",
            "log_file_path": "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
            "log_tail_command": "tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
        }
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/vm/log",
        headers={},
        params={"wait_for_running_seconds": "15"},
        route_params={},
        body=b"",
    )

    response = function_app_module.trading_vm_log(req)
    payload = json.loads(response.get_body().decode("utf-8"))

    assert response.status_code == 200
    assert payload["power_state"] == "running"
    assert payload["content"] == "decision line"
    assert fake_compute.virtual_machines.start_calls == []


def test_trading_vm_log_can_download_full_log(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_compute = FakeComputeClient(
        "running",
        run_command_result=FakeRunCommandResult(
            "\n".join(
                [
                    function_app_module.LOG_OUTPUT_START_MARKER,
                    "decision line",
                    "risk line",
                    function_app_module.LOG_OUTPUT_END_MARKER,
                ]
            )
        ),
    )
    monkeypatch.setattr(function_app_module, "_build_compute_client", lambda config: fake_compute)
    function_app_module._save_dispatch_state(
        {
            "accepted": True,
            "active": True,
            "symbol": "AAPL",
            "loops": 1,
            "mode": "backtest",
            "live_after_backtest": False,
            "trigger_source": "http",
            "submitted_at": "2026-04-05T09:32:43+00:00",
            "vm_name": "vm-test",
            "resource_group": "rg-test",
            "start_requested": False,
            "power_state": "running",
            "log_file_path": "/opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
            "log_tail_command": "tail -f /opt/fresh_simple_trading_project/logs/workflow_backtest_aapl_http_20260405T093243Z.log",
        }
    )

    req = func.HttpRequest(
        method="GET",
        url="http://localhost/api/trading/vm/log",
        headers={},
        params={"download": "true"},
        route_params={},
        body=b"",
    )

    response = function_app_module.trading_vm_log(req)

    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert response.headers["Content-Disposition"] == (
        'attachment; filename="workflow_backtest_aapl_http_20260405T093243Z.log"'
    )
    assert response.get_body().decode("utf-8") == "decision line\nrisk line"
