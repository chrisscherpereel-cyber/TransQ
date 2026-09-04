"""Provider-agnostic LLM access.

One interface, three code paths, five vendors. Everything above this file asks
for "JSON that matches this shape" and does not care who produced it, so
switching providers is a dropdown rather than a refactor.

* Gemini    — native ``google-genai`` SDK (``response_mime_type`` gives real
              JSON guarantees; the OpenAI-compat endpoint is still beta)
* Claude    — native ``anthropic`` SDK
* OpenAI / xAI / OpenRouter — the ``openai`` SDK with a different ``base_url``
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Provider, get_provider


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


# USD per million tokens (input, output). Update as list prices change; used
# only for the on-screen estimate, never for billing.
#
# The OpenRouter entries are keyed by slug and are approximate: OpenRouter routes
# to whichever upstream host is cheapest or fastest at the moment, so the real
# rate moves. Treat those figures as an order of magnitude and check the
# OpenRouter dashboard for the actual spend. Any slug not listed here (including
# anything you type into the custom-model box) shows tokens but no dollar figure,
# which is the honest answer rather than a fabricated one.
PRICING: dict[str, tuple[float, float]] = {
    # OpenRouter (approximate — see note above)
    "deepseek/deepseek-v4-pro": (0.87, 1.74),
    "deepseek/deepseek-v4-flash": (0.07, 0.17),
    "deepseek/deepseek-v3.2": (0.21, 0.31),
    "deepseek/deepseek-chat-v3.1": (0.25, 0.95),
    "deepseek/deepseek-r1": (0.70, 2.50),
    "google/gemini-3.8-flash": (0.75, 3.75),
    "anthropic/claude-sonnet-4.5": (3.0, 15.0),
    "openai/gpt-4.1": (2.0, 8.0),
    "x-ai/grok-4.6": (2.0, 6.0),
    # Google
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    # Anthropic
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # OpenAI
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4o": (2.5, 10.0),
    # xAI
    "grok-4.6": (2.0, 6.0),
    "grok-4": (3.0, 15.0),
}


def register_pricing(rates: dict[str, tuple[float, float]]) -> None:
    """Merge live prices (currently OpenRouter's catalog) into the table.

    Live figures beat the static ones above, which are only a starting point for
    providers with no price API.
    """
    PRICING.update(rates)


def estimate_cost(model: str, usage: Usage) -> float:
    """Best-effort USD estimate. Returns 0.0 for models with no listed price."""
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (usage.input_tokens / 1e6) * inp + (usage.output_tokens / 1e6) * out


def has_pricing(model: str) -> bool:
    return model in PRICING


@dataclass
class LLMClient:
    """Thin wrapper over whichever vendor SDK the selected provider needs."""

    provider: str
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 8000
    usage: Usage = field(default_factory=Usage)
    app_name: str = "Lecture Quiz Builder"
    app_url: str = "https://github.com/"

    def __post_init__(self) -> None:
        self.spec: Provider = get_provider(self.provider)
        if not self.api_key:
            raise LLMError(
                f"No API key for {self.spec.label}. Add it in the sidebar, or set "
                f"{self.spec.env_var} in your Streamlit secrets. "
                f"Get one at {self.spec.console_url}"
            )
        self._client = self._build_client()

    # ------------------------------------------------------------------ #
    # Client construction
    # ------------------------------------------------------------------ #

    def _build_client(self) -> Any:
        if self.spec.sdk == "gemini":
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install google-genai") from exc
            return genai.Client(api_key=self.api_key)

        if self.spec.sdk == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install anthropic") from exc
            return Anthropic(api_key=self.api_key)

        if self.spec.sdk == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install openai") from exc
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.spec.base_url:
                kwargs["base_url"] = self.spec.base_url
            if self.spec.key == "openrouter":
                # Optional attribution headers OpenRouter uses for its leaderboard.
                kwargs["default_headers"] = {
                    "HTTP-Referer": self.app_url,
                    "X-Title": self.app_name,
                }
            return OpenAI(**kwargs)

        raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------ #
    # Completion
    # ------------------------------------------------------------------ #

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def complete(
        self, system: str, user: str, max_tokens: int | None = None, json_mode: bool = False
    ) -> str:
        """Send one prompt, return raw text."""
        tokens = max_tokens or self.max_tokens
        try:
            if self.spec.sdk == "gemini":
                return self._complete_gemini(system, user, tokens, json_mode)
            if self.spec.sdk == "anthropic":
                return self._complete_anthropic(system, user, tokens)
            return self._complete_openai(system, user, tokens, json_mode)
        except (LLMError, TransientLLMError):
            raise
        except Exception as exc:
            if _is_transient(exc):
                raise TransientLLMError(str(exc)) from exc
            raise LLMError(f"{self.spec.label} request failed: {exc}") from exc

    def _complete_gemini(self, system: str, user: str, tokens: int, json_mode: bool) -> str:
        from google.genai import types

        config: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.temperature,
            # Gemini 3.x counts reasoning tokens against this budget, so give it
            # real headroom or long question batches get truncated mid-JSON.
            "max_output_tokens": max(tokens, 4096),
        }
        if json_mode:
            config["response_mime_type"] = "application/json"

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(**config),
        )

        meta = getattr(resp, "usage_metadata", None)
        if meta:
            self.usage.add(
                Usage(
                    int(getattr(meta, "prompt_token_count", 0) or 0),
                    int(getattr(meta, "candidates_token_count", 0) or 0),
                    1,
                )
            )
        else:
            self.usage.add(Usage(0, 0, 1))

        text = getattr(resp, "text", None)
        if not text:
            reason = ""
            for candidate in getattr(resp, "candidates", None) or []:
                reason = str(getattr(candidate, "finish_reason", "") or "")
                break
            raise LLMError(
                "Gemini returned no text"
                + (f" (finish reason: {reason})." if reason else ".")
                + " If this repeats, try a smaller chunk length or a different model."
            )
        return text

    def _complete_anthropic(self, system: str, user: str, tokens: int) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.usage.add(Usage(resp.usage.input_tokens, resp.usage.output_tokens, 1))
        return "".join(block.text for block in resp.content if block.type == "text")

    def _complete_openai(self, system: str, user: str, tokens: int, json_mode: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode and self.spec.supports_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        if usage:
            self.usage.add(
                Usage(
                    int(getattr(usage, "prompt_tokens", 0) or 0),
                    int(getattr(usage, "completion_tokens", 0) or 0),
                    1,
                )
            )
        else:
            self.usage.add(Usage(0, 0, 1))

        choices = getattr(resp, "choices", None) or []
        if not choices:
            raise LLMError(f"{self.spec.label} returned no choices.")
        return choices[0].message.content or ""

    def complete_json(
        self, system: str, user: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Send one prompt and parse the reply as a JSON object.

        Models occasionally wrap JSON in prose or a code fence even when told
        not to — and on OpenRouter, whether native JSON mode is honored depends
        on the underlying model — so the response is salvaged rather than
        thrown away.
        """
        system = system.rstrip() + "\n\nRespond with a single valid JSON object and nothing else."
        raw = self.complete(system, user, max_tokens=max_tokens, json_mode=True)
        return parse_json_object(raw)


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    markers = (
        "rate limit",
        "ratelimit",
        "resource_exhausted",
        "resource exhausted",
        "overloaded",
        "unavailable",
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
