"""Runtime settings and secret resolution.

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

LLM_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-opus-4-1",
        "claude-haiku-4-5",
    ],
    "openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
    ],
}

AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "mp4", "mpeg", "mpga", "webm", "ogg", "flac", "aac"]


def get_secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets, then the environment."""
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:
            pass
    return os.environ.get(name, default)


@dataclass
class AppSettings:
    # Transcription
    whisper_model: str = "small"
    compute_type: str = "int8"
    language: str | None = None  # None = autodetect
    vad_filter: bool = True
    beam_size: int = 1

    # Generation
    provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-5"
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
        env_name = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        return get_secret(env_name)
