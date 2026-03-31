from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .config import LLMConfig


class TextGenerationClient(Protocol):
    def generate(self, system_prompt: str, content: str) -> str | None:
        ...


class LLMRequestError(RuntimeError):
    def __init__(self, *, operation: str, model: str, base_url: str, detail: str) -> None:
        super().__init__(f"LLM {operation} failed for model '{model}' via {base_url}: {detail}")
        self.operation = operation
        self.model = model
        self.base_url = base_url
        self.detail = detail


@dataclass
class DeepSeekLLMClient:
    config: LLMConfig
    _client: Any | None = field(init=False, default=None, repr=False)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def generate(self, system_prompt: str, content: str) -> str | None:
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
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _generate_streamed(self, system_prompt: str, content: str) -> str:
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
        return LLMRequestError(
            operation=operation,
            model=self.config.model,
            base_url=self.config.base_url,
            detail=detail,
        )

    def _run_with_heartbeat(self, operation: str, waiting_for: str, fn: Callable[[], Any]) -> Any:
        heartbeat_seconds = max(0.1, float(self.config.heartbeat_seconds))
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
        print(message, flush=True)


def clean_llm_text(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    return candidate or None
