"""LLM client abstractions and helpers for agent prompts."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import LLMConfig


class TextGenerationClient(Protocol):
    """Minimal interface for text-generation backends used by agents."""

    def generate(self, system_prompt: str, content: str) -> str | None:
        """Generate text from the supplied system and user prompts."""
        ...


class LLMRequestError(RuntimeError):
    """Structured exception raised when an LLM call fails."""

    def __init__(self, *, operation: str, model: str, base_url: str, detail: str) -> None:
        super().__init__(f"LLM {operation} failed for model '{model}' via {base_url}: {detail}")
        self.operation = operation
        self.model = model
        self.base_url = base_url
        self.detail = detail


@dataclass
class OpenAICompatibleLLMClient:
    """Streaming text-generation client backed by an OpenAI-compatible API."""

    config: LLMConfig
    _client: Any | None = field(init=False, default=None, repr=False)

    @property
    def enabled(self) -> bool:
        """Report whether the client has the credentials needed to run."""
        return self.config.enabled

    def generate(self, system_prompt: str, content: str) -> str | None:
        """Generate a completion for the supplied system and user prompts."""
        if not self.enabled:
            return None

        try:
            response = self._run_with_heartbeat(
                "generate",
                "waiting for streamed response",
                lambda: self._generate_streamed(system_prompt, content),
            )
            if not response:
                raise self._request_error("generate", "the API returned an empty streamed response")
            return response
        except LLMRequestError:
            raise
        except Exception as exc:
            raise self._request_error("generate", f"{type(exc).__name__}: {exc}") from exc

    def _get_client(self) -> Any:
        """Lazily initialize the underlying OpenAI-compatible client."""
        if self._client is None:
            from openai import OpenAI

            client_kwargs = dict(
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _generate_streamed(self, system_prompt: str, content: str) -> str:
        """Collect a streamed chat completion into a single string."""
        client = self._get_client()
        response_stream = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            stream=True,
        )
        parts: list[str] = []
        for chunk in response_stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def _request_error(self, operation: str, detail: str) -> LLMRequestError:
        """Create a normalized request error for this configured client."""
        return LLMRequestError(
            operation=operation,
            model=self.config.model,
            base_url=self.config.base_url or "default OpenAI base URL",
            detail=detail,
        )

    def _run_with_heartbeat(self, operation: str, waiting_for: str, fn: Callable[[], Any]) -> Any:
        """Run a blocking operation while emitting periodic progress messages."""
        heartbeat_seconds = max(0.01, float(self.config.heartbeat_seconds))
        if not self.config.show_progress:
            return fn()

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        started_at = time.monotonic()
        heartbeat_count = 0

        def worker() -> None:
            try:
                result_queue.put(("result", fn()))
            except BaseException as exc:  # pragma: no cover - surfaced in caller.
                result_queue.put(("error", exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                status, payload = result_queue.get(timeout=heartbeat_seconds)
            except queue.Empty:
                heartbeat_count += 1
                elapsed = time.monotonic() - started_at
                self._emit_progress(
                    f"LLM {operation} for model '{self.config.model}' is still thinking "
                    f"after {elapsed:.1f}s ({waiting_for})."
                )
                continue

            if heartbeat_count:
                elapsed = time.monotonic() - started_at
                self._emit_progress(
                    f"LLM {operation} for model '{self.config.model}' completed after {elapsed:.1f}s."
                )

            if status == "error":
                raise payload
            return payload

    def _emit_progress(self, message: str) -> None:
        """Emit a progress message to stdout."""
        print(message, flush=True)


class DeepSeekLLMClient(OpenAICompatibleLLMClient):
    """Compatibility subclass for callers that still import ``DeepSeekLLMClient``."""


class OpenAILLMClient(OpenAICompatibleLLMClient):
    """Compatibility subclass for callers that still import ``OpenAILLMClient``."""


@dataclass
class FallbackLLMClient:
    """Try a primary client first and switch to a secondary client on quota exhaustion."""

    primary: TextGenerationClient
    secondary: TextGenerationClient
    fallback_predicate: Callable[[LLMRequestError], bool] = field(default=lambda error: is_quota_exhaustion_error(error))

    def generate(self, system_prompt: str, content: str) -> str | None:
        """Generate text from the primary client and retry with the secondary when quota is exhausted."""
        try:
            return self.primary.generate(system_prompt, content)
        except LLMRequestError as exc:
            if not self.fallback_predicate(exc):
                raise
            self._emit_progress(
                f"Primary LLM '{exc.model}' via {exc.base_url} appears out of credits or quota; "
                "retrying with the secondary LLM."
            )
            try:
                return self.secondary.generate(system_prompt, content)
            except LLMRequestError as fallback_exc:
                raise LLMRequestError(
                    operation=fallback_exc.operation,
                    model=fallback_exc.model,
                    base_url=fallback_exc.base_url,
                    detail=(
                        f"primary fallback trigger from model '{exc.model}' via {exc.base_url}: {exc.detail}; "
                        f"secondary request failed: {fallback_exc.detail}"
                    ),
                ) from fallback_exc

    def _emit_progress(self, message: str) -> None:
        """Emit fallback progress to stdout."""
        print(message, flush=True)


def is_quota_exhaustion_error(error: LLMRequestError) -> bool:
    """Report whether an LLM failure looks like exhausted balance or quota."""
    detail = error.detail.lower()
    return any(
        signal in detail
        for signal in (
            "insufficient balance",
            "insufficient_balance",
            "insufficient quota",
            "insufficient_quota",
            "quota exceeded",
            "exceeded your current quota",
            "out of credits",
            "credit balance",
            "billing hard limit",
            "payment required",
            "not enough balance",
        )
    )


def clean_llm_text(value: str | None) -> str | None:
    """Normalize raw LLM text and strip surrounding fenced-code wrappers."""
    if value is None:
        return None
    candidate = value.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    return candidate or None
