"""Provider-agnostic LLM access.

One interface, two backends. Everything above this file asks for "JSON that
matches this shape" and does not care whether Claude or GPT produced it, which
means switching providers is a dropdown, not a refactor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    """Rate limits, overloads, timeouts — worth retrying."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


# USD per million tokens. Update as list prices change; used only for the
# on-screen estimate, never for billing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o": (2.5, 10.0),
}


def estimate_cost(model: str, usage: Usage) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (usage.input_tokens / 1e6) * inp + (usage.output_tokens / 1e6) * out


@dataclass
class LLMClient:
    """Thin wrapper over the Anthropic and OpenAI SDKs."""

    provider: str
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 8000
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise LLMError(
                f"No API key for {self.provider}. Add it in the sidebar, or set "
                f"{'ANTHROPIC_API_KEY' if self.provider == 'anthropic' else 'OPENAI_API_KEY'} "
                "in your Streamlit secrets."
            )
        self._client = self._build_client()

    def _build_client(self) -> Any:
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install anthropic") from exc
            return Anthropic(api_key=self.api_key)
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install openai") from exc
            return OpenAI(api_key=self.api_key)
        raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------ #

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Send one prompt, return raw text."""
        tokens = max_tokens or self.max_tokens
        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self.usage.add(
                    Usage(resp.usage.input_tokens, resp.usage.output_tokens, 1)
                )
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                )

            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            if resp.usage:
                self.usage.add(
                    Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens, 1)
                )
            return resp.choices[0].message.content or ""

        except Exception as exc:
            if _is_transient(exc):
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

    def complete_json(
        self, system: str, user: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Send one prompt and parse the reply as a JSON object.

        Models occasionally wrap JSON in prose or a code fence even when told
        not to, so the response is salvaged rather than thrown away.
        """
        system = system.rstrip() + "\n\nRespond with a single valid JSON object and nothing else."
        raw = self.complete(system, user, max_tokens=max_tokens)
        return parse_json_object(raw)


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "rate limit",
        "ratelimit",
        "overloaded",
        "429",
        "500",
        "502",
        "503",
        "529",
        "timeout",
        "timed out",
        "connection",
        "temporarily",
    )
    return any(m in text for m in markers)


def parse_json_object(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    text = (raw or "").strip()
    if not text:
        raise LLMError("The model returned an empty response.")

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Could not find JSON in the model response: {text[:300]}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMError("Expected a JSON object at the top level.")
    return parsed
