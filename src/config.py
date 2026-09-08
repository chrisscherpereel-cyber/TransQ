"""Runtime settings, the provider registry, and secret resolution.

Secrets are looked up in this order: Streamlit secrets -> environment ->
whatever the user typed into the sidebar. That lets the same code run on
Streamlit Community Cloud, on a laptop with a .env, or as a shared campus
deployment where each instructor supplies their own key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # Streamlit is present at runtime but not in unit tests.
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# Whisper model sizes that are realistic on Streamlit Community Cloud
# (1 CPU core, ~1 GB RAM). "small" is the practical ceiling there.
WHISPER_MODELS: dict[str, str] = {
    "tiny": "Fastest, roughly 5–8% WER on clean lecture audio. Good for a smoke test.",
    "base": "Fast. Acceptable for clear, close-mic recordings.",
    "small": "Best quality that still fits Streamlit Cloud's memory budget. Recommended.",
    "medium": "Noticeably better on jargon and accents. Needs a local machine or a paid host.",
    "large-v3": "Best accuracy. GPU strongly recommended.",
}


# --------------------------------------------------------------------------- #
# LLM providers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provider:
    """Everything the app needs to know to talk to one LLM vendor.

    ``sdk`` picks the code path in ``src.llm``:

    * ``"gemini"``     — the native ``google-genai`` SDK
    * ``"anthropic"``  — the native ``anthropic`` SDK
    * ``"openai"``     — the ``openai`` SDK, pointed at ``base_url``

    Three vendors share the ``openai`` path because OpenAI, xAI, and OpenRouter
    all expose the same ``/chat/completions`` contract. Gemini also offers an
    OpenAI-compatible endpoint, but it is still marked beta and its handling of
    ``response_format`` is inconsistent — since this app depends on reliable JSON,
    Gemini uses its native SDK, where ``response_mime_type`` is a first-class
    guarantee.
    """

    key: str
    label: str
    sdk: str
    env_var: str
    # For most providers this is the model dropdown. For OpenRouter it is only a
    # seed — the first entry is the default selection, and the real list is
    # fetched live in src/openrouter_catalog.py because that roster changes
    # weekly and a hardcoded copy would be wrong within a month.
    models: tuple[str, ...]
    base_url: str | None = None
    supports_json_mode: bool = True
    allow_custom_model: bool = False
    console_url: str = ""
    note: str = ""


# OpenRouter's Free Models Router. It reads each request, filters to free models
# that can serve it, and picks one at random — so it costs nothing and never
# goes stale as free models come and go.
#
# The trade-offs are real and worth stating, because they are not obvious from
# the price: lower rate limits, higher latency at peak, availability that varies,
# and a *different model per call*, so two runs over the same lecture can differ
# in quality in a way a pinned model's do not. Good for drafting and for anyone
# without a paid key; pin a model when a particular set matters.
FREE_ROUTER = "openrouter/free"


PROVIDERS: dict[str, Provider] = {
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter",
        sdk="openai",
        env_var="OPENROUTER_API_KEY",
        models=(
            FREE_ROUTER,
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v3.2",
            "deepseek/deepseek-chat-v3.1",
            "deepseek/deepseek-r1",
            "google/gemini-3.8-flash",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-4.1",
            "x-ai/grok-4.6",
        ),
        base_url="https://openrouter.ai/api/v1",
        # OpenRouter proxies hundreds of models and not all of them honor
        # response_format, so JSON is requested in the prompt and salvaged from
        # the reply rather than enforced by the API.
        supports_json_mode=False,
        allow_custom_model=True,
        console_url="https://openrouter.ai/keys",
        note="Default. The full model list is fetched live from OpenRouter.",
    ),
    "gemini": Provider(
        key="gemini",
        label="Google Gemini",
        sdk="gemini",
        env_var="GEMINI_API_KEY",
        models=(
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ),
        console_url="https://aistudio.google.com/apikey",
        note="Fast, inexpensive, and has a free tier that covers light use.",
    ),
    "anthropic": Provider(
        key="anthropic",
        label="Anthropic Claude",
        sdk="anthropic",
        env_var="ANTHROPIC_API_KEY",
        models=(
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
        ),
        console_url="https://console.anthropic.com/settings/keys",
        note="Strongest at following the item-writing rules in the prompts.",
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        sdk="openai",
        env_var="OPENAI_API_KEY",
        models=("gpt-4.1", "gpt-4.1-mini", "gpt-4o"),
        base_url=None,  # SDK default
        console_url="https://platform.openai.com/api-keys",
    ),
    "xai": Provider(
        key="xai",
        label="xAI Grok",
        sdk="openai",
        env_var="XAI_API_KEY",
        models=("grok-4.6", "grok-4"),
        base_url="https://api.x.ai/v1",
        console_url="https://console.x.ai",
        note="OpenAI-compatible endpoint at api.x.ai.",
    ),
}

DEFAULT_PROVIDER = "openrouter"

# Back-compat for anything that imported the old flat mapping.
LLM_MODELS: dict[str, list[str]] = {k: list(p.models) for k, p in PROVIDERS.items()}

AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac", "aac"]


def get_provider(key: str) -> Provider:
    return PROVIDERS.get(key, PROVIDERS[DEFAULT_PROVIDER])


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets, then the environment."""
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
    return os.environ.get(name, default)


def available_providers() -> list[str]:
    """Provider keys that already have a key configured, default first."""
    return [k for k, p in PROVIDERS.items() if get_secret(p.env_var)]


@dataclass
class AppSettings:
    # Transcription
    whisper_model: str = "small"
    compute_type: str = "int8"
    language: str | None = None  # None = autodetect
    vad_filter: bool = True
    beam_size: int = 1

    # Generation
    provider: str = DEFAULT_PROVIDER
    llm_model: str = PROVIDERS[DEFAULT_PROVIDER].models[0]
    api_key: str = ""
    temperature: float = 0.3

    # Quiz shape
    num_questions: int = 10
    options_per_question: int = 4
    bloom_targets: list[str] = field(
        default_factory=lambda: ["Remember", "Understand", "Apply", "Analyze"]
    )
    difficulty_mix: str = "Balanced"
    course_context: str = ""

    # Chunking
    chunk_seconds: int = 600
    chunk_overlap_seconds: int = 30

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        return get_secret(get_provider(self.provider).env_var)
