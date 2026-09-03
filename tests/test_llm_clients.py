"""Client-construction tests for every provider.

The vendor SDKs are replaced with recording stubs, so these run offline and with
no API keys. What they pin down is the wiring that is easy to get quietly wrong:
which base_url each provider dials, which SDK path it takes, whether native JSON
mode is requested, and whether token usage is accounted correctly.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.config import PROVIDERS
from src.llm import LLMClient, LLMError


# --------------------------------------------------------------------------- #
# Stub SDKs
# --------------------------------------------------------------------------- #


class Recorder:
    """Collects the kwargs each fake SDK was constructed and called with."""

    def __init__(self) -> None:
        self.client_kwargs: dict = {}
        self.call_kwargs: dict = {}
        self.config: dict = {}


@pytest.fixture
def rec() -> Recorder:
    return Recorder()


@pytest.fixture
def fake_openai(monkeypatch, rec):
    class Message:
        content = '{"ok": true}'

    class Choice:
        message = Message()

    class Usage:
        prompt_tokens = 100
        completion_tokens = 42

    class Completions:
        @staticmethod
        def create(**kwargs):
            rec.call_kwargs = kwargs
            return types.SimpleNamespace(choices=[Choice()], usage=Usage())

    class Chat:
        completions = Completions()

    class OpenAI:
        def __init__(self, **kwargs):
            rec.client_kwargs = kwargs
            self.chat = Chat()

    module = types.ModuleType("openai")
    module.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return rec


@pytest.fixture
def fake_gemini(monkeypatch, rec):
    class Models:
        @staticmethod
        def generate_content(**kwargs):
            rec.call_kwargs = kwargs
            return types.SimpleNamespace(
                text='{"ok": true}',
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=200, candidates_token_count=50
                ),
                candidates=[],
            )

    class Client:
        def __init__(self, **kwargs):
            rec.client_kwargs = kwargs
            self.models = Models()

    genai = types.ModuleType("google.genai")
    genai.Client = Client

    genai_types = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            # Recorded separately: generate_content() rebinds call_kwargs after
            # this runs, so stashing it there would be overwritten.
            rec.config = dict(kwargs)

    genai_types.GenerateContentConfig = GenerateContentConfig
    genai.types = genai_types

    google = types.ModuleType("google")
    google.genai = genai

    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return rec


@pytest.fixture
def fake_anthropic(monkeypatch, rec):
    class Block:
        type = "text"
        text = '{"ok": true}'

    class Messages:
        @staticmethod
        def create(**kwargs):
            rec.call_kwargs = kwargs
            return types.SimpleNamespace(
                content=[Block()],
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
            )

    class Anthropic:
        def __init__(self, **kwargs):
            rec.client_kwargs = kwargs
            self.messages = Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return rec


# --------------------------------------------------------------------------- #
# Missing key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_missing_key_names_the_right_env_var(provider):
    with pytest.raises(LLMError) as exc:
        LLMClient(provider=provider, model="m", api_key="")
    assert PROVIDERS[provider].env_var in str(exc.value)
    assert PROVIDERS[provider].console_url in str(exc.value)


# --------------------------------------------------------------------------- #
# Gemini (default)
# --------------------------------------------------------------------------- #


def test_gemini_uses_native_sdk_and_requests_json(fake_gemini):
    client = LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k")
    assert fake_gemini.client_kwargs == {"api_key": "k"}

    assert client.complete_json("sys", "user") == {"ok": True}
    config = fake_gemini.config
    assert config["response_mime_type"] == "application/json"
    assert config["system_instruction"].startswith("sys")
    assert config["max_output_tokens"] >= 4096  # headroom for thinking tokens
    assert fake_gemini.call_kwargs["model"] == "gemini-3.8-flash"


def test_gemini_skips_json_mime_for_plain_completions(fake_gemini):
    client = LLMClient(provider="gemini", model="gemini-2.5-flash", api_key="k")
    client.complete("sys", "user")
    assert "response_mime_type" not in fake_gemini.config


def test_gemini_records_usage(fake_gemini):
    client = LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k")
    client.complete_json("sys", "user")
    assert (client.usage.input_tokens, client.usage.output_tokens) == (200, 50)
    assert client.usage.calls == 1


def test_gemini_empty_response_is_a_clear_error(fake_gemini, monkeypatch):
    import google.genai as genai  # the stub

    def blocked(**kwargs):
        return types.SimpleNamespace(
            text=None,
            usage_metadata=None,
            candidates=[types.SimpleNamespace(finish_reason="MAX_TOKENS")],
        )

    client = LLMClient(provider="gemini", model="gemini-3.8-flash", api_key="k")
    monkeypatch.setattr(client._client.models, "generate_content", blocked)
    with pytest.raises(LLMError) as exc:
        client.complete_json("sys", "user")
    assert "MAX_TOKENS" in str(exc.value)


# --------------------------------------------------------------------------- #
# OpenAI-compatible trio
# --------------------------------------------------------------------------- #


def test_openai_uses_sdk_default_base_url(fake_openai):
    LLMClient(provider="openai", model="gpt-4.1", api_key="k")
    assert "base_url" not in fake_openai.client_kwargs
    assert fake_openai.client_kwargs["api_key"] == "k"


def test_xai_points_at_api_x_ai(fake_openai):
    client = LLMClient(provider="xai", model="grok-4.6", api_key="k")
    assert fake_openai.client_kwargs["base_url"] == "https://api.x.ai/v1"
    assert client.complete_json("sys", "user") == {"ok": True}
    assert fake_openai.call_kwargs["response_format"] == {"type": "json_object"}
    assert fake_openai.call_kwargs["model"] == "grok-4.6"


def test_openrouter_sets_base_url_and_attribution_headers(fake_openai):
    LLMClient(provider="openrouter", model="x-ai/grok-4.6", api_key="k")
    assert fake_openai.client_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    headers = fake_openai.client_kwargs["default_headers"]
    assert "HTTP-Referer" in headers and "X-Title" in headers


def test_openrouter_does_not_send_response_format(fake_openai):
    """Not every proxied model honors it; JSON is salvaged from the text instead."""
    client = LLMClient(provider="openrouter", model="deepseek/deepseek-chat", api_key="k")
    assert client.complete_json("sys", "user") == {"ok": True}
    assert "response_format" not in fake_openai.call_kwargs


def test_openai_path_records_usage(fake_openai):
    client = LLMClient(provider="xai", model="grok-4.6", api_key="k")
    client.complete_json("sys", "user")
    assert (client.usage.input_tokens, client.usage.output_tokens) == (100, 42)


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


def test_anthropic_passes_system_separately(fake_anthropic):
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-5", api_key="k")
    assert client.complete_json("sys", "user") == {"ok": True}
    assert fake_anthropic.call_kwargs["system"].startswith("sys")
    assert fake_anthropic.call_kwargs["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------- #
# Shared behavior
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "provider,model",
    [("gemini", "gemini-3.8-flash"), ("xai", "grok-4.6"), ("anthropic", "claude-sonnet-4-5")],
)
def test_json_instruction_is_appended_to_every_system_prompt(
    provider, model, fake_gemini, fake_openai, fake_anthropic
):
    client = LLMClient(provider=provider, model=model, api_key="k")
    client.complete_json("Be concise.", "user")
    rec = {"gemini": fake_gemini, "xai": fake_openai, "anthropic": fake_anthropic}[provider]
    if provider == "gemini":
        system = rec.config["system_instruction"]
    elif provider == "anthropic":
        system = rec.call_kwargs["system"]
    else:
        system = rec.call_kwargs["messages"][0]["content"]
    assert "single valid JSON object" in system
