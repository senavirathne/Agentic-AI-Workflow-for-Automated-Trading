from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fresh_simple_trading_project.config import LLMConfig
from fresh_simple_trading_project.llm import DeepSeekLLMClient, LLMRequestError


def test_generate_raises_on_transport_failure() -> None:
    client = DeepSeekLLMClient(LLMConfig(api_key="test-key"))
    client._client = _fake_client(lambda **_: (_ for _ in ()).throw(RuntimeError("network down")))

    with pytest.raises(LLMRequestError, match="LLM generate failed"):
        client.generate("system", "content")


def test_generate_raises_on_empty_streamed_response() -> None:
    client = DeepSeekLLMClient(LLMConfig(api_key="test-key"))
    client._client = _fake_client(
        lambda **_: [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])]
    )

    with pytest.raises(LLMRequestError, match="empty streamed response"):
        client.generate("system", "content")


def test_run_with_heartbeat_emits_progress_messages(monkeypatch) -> None:
    client = DeepSeekLLMClient(LLMConfig(api_key="test-key", heartbeat_seconds=0.01))
    messages: list[str] = []
    monkeypatch.setattr(client, "_emit_progress", messages.append)

    result = client._run_with_heartbeat("generate", "waiting for streamed response", lambda: _sleep_then_return("ok"))

    assert result == "ok"
    assert any("still thinking" in message for message in messages)
    assert any("completed after" in message for message in messages)


def _fake_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def _sleep_then_return(value: str) -> str:
    time.sleep(0.12)
    return value
