from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from fresh_simple_trading_project.config import LLMConfig
from fresh_simple_trading_project.llm import DeepSeekLLMClient, FallbackLLMClient, LLMRequestError, is_quota_exhaustion_error


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


def test_fallback_llm_client_switches_to_secondary_on_quota_error() -> None:
    primary = RaisingStubLLM(
        LLMRequestError(
            operation="generate",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com",
            detail="HTTP 402 insufficient balance",
        )
    )
    secondary = RecordingStubLLM("secondary response")
    client = FallbackLLMClient(primary=primary, secondary=secondary)

    result = client.generate("system", "content")

    assert result == "secondary response"
    assert secondary.calls == [("system", "content")]


def test_fallback_llm_client_preserves_non_quota_errors() -> None:
    primary_error = LLMRequestError(
        operation="generate",
        model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        detail="RuntimeError: network down",
    )
    client = FallbackLLMClient(primary=RaisingStubLLM(primary_error), secondary=RecordingStubLLM("unused"))

    with pytest.raises(LLMRequestError, match="network down"):
        client.generate("system", "content")


def test_is_quota_exhaustion_error_matches_known_balance_failures() -> None:
    assert is_quota_exhaustion_error(
        LLMRequestError(
            operation="generate",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com",
            detail="Error 402: insufficient balance",
        )
    )
    assert not is_quota_exhaustion_error(
        LLMRequestError(
            operation="generate",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com",
            detail="RuntimeError: socket timeout",
        )
    )


def _fake_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def _sleep_then_return(value: str) -> str:
    time.sleep(0.12)
    return value


class RecordingStubLLM:
    def __init__(self, response: str | None) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, content: str) -> str | None:
        self.calls.append((system_prompt, content))
        return self.response


class RaisingStubLLM:
    def __init__(self, error: LLMRequestError) -> None:
        self.error = error

    def generate(self, system_prompt: str, content: str) -> str | None:
        raise self.error
