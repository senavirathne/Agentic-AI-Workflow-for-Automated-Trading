from __future__ import annotations

from pathlib import Path

import pytest

import fresh_simple_trading_project.cli as cli_module


def test_resolve_project_root_finds_parent_project_root(tmp_path: Path) -> None:
    project_root = _make_project_layout(tmp_path)
    src_dir = project_root / "src"
    package_dir = src_dir / "fresh_simple_trading_project"

    assert cli_module._resolve_project_root(project_root) == project_root
    assert cli_module._resolve_project_root(src_dir) == project_root
    assert cli_module._resolve_project_root(package_dir) == project_root


def test_main_uses_resolved_project_root_when_invoked_from_src(monkeypatch, tmp_path: Path) -> None:
    project_root = _make_project_layout(tmp_path)
    src_dir = project_root / "src"
    recorded: dict[str, Path] = {}

    class StopExecution(RuntimeError):
        pass

    def fake_build_workflow(*, project_root: Path | None = None, mode=None):
        del mode
        recorded["project_root"] = project_root
        raise StopExecution

    monkeypatch.chdir(src_dir)
    monkeypatch.setattr(cli_module, "build_workflow", fake_build_workflow)
    monkeypatch.setattr(
        cli_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: cli_module.argparse.Namespace(
            command="trade-once",
            mode="backtest",
            symbol="AAPL",
            execute=False,
            json=False,
        ),
    )

    with pytest.raises(StopExecution):
        cli_module.main()

    assert recorded["project_root"] == project_root


def _make_project_layout(tmp_path: Path) -> Path:
    project_root = tmp_path / "fresh_simple_trading_project"
    (project_root / "src" / "fresh_simple_trading_project").mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'fresh-simple-trading-project'\n", encoding="utf-8")
    return project_root
