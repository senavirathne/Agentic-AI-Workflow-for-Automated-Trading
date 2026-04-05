from __future__ import annotations

from datetime import datetime, timedelta, timezone

from project import utils


def test_sleep_until_uses_explicit_sleep_seconds(monkeypatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(utils.time, "sleep", calls.append)

    slept = utils.sleep_until(datetime.now(timezone.utc) + timedelta(hours=1), sleep_seconds=0.5)

    assert slept == 0.5
    assert calls == [0.5]


def test_sleep_until_clamps_negative_override(monkeypatch) -> None:
    calls: list[float] = []
    monkeypatch.setattr(utils.time, "sleep", calls.append)

    slept = utils.sleep_until(datetime.now(timezone.utc), sleep_seconds=-2)

    assert slept == 0.0
    assert calls == []
