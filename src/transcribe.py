"""Local speech-to-text with faster-whisper.

faster-whisper runs the Whisper weights through CTranslate2, which is roughly
4x quicker than openai-whisper on CPU and uses about half the memory — the
difference between "runs on Streamlit Community Cloud" and "doesn't".

Segment-level timestamps are kept because every generated question carries a
pointer back to the moment in the lecture it came from.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .schema import Segment, Transcript

ProgressFn = Callable[[float, str], None]


class TranscriptionError(RuntimeError):
    pass


def load_model(model_size: str = "small", compute_type: str = "int8") -> Any:
    """Load (and cache on disk) a faster-whisper model.

    The caller is expected to wrap this in ``st.cache_resource`` so the weights
    are downloaded and held in memory once per app process, not once per rerun.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc

    download_root = os.environ.get("WHISPER_CACHE_DIR", ".cache/whisper")
    os.makedirs(download_root, exist_ok=True)

    return WhisperModel(
        model_size,
        device="auto",
        compute_type=compute_type,
        download_root=download_root,
    )


def transcribe_file(
    model: Any,
    audio_path: str,
    language: str | None = None,
    vad_filter: bool = True,
    beam_size: int = 1,
    progress: ProgressFn | None = None,
) -> Transcript:
    """Transcribe an audio file into a :class:`Transcript`.

    ``progress`` receives (fraction_complete, human_readable_status) as segments
    stream in, so the UI can show real movement instead of a fake spinner.
    """
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    try:
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
            condition_on_previous_text=False,  # avoids runaway repetition loops
            word_timestamps=False,
        )
    except Exception as exc:
        raise TranscriptionError(f"Whisper could not decode this file: {exc}") from exc

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []

    for i, seg in enumerate(segments_iter):
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            Segment(index=i, start=float(seg.start), end=float(seg.end), text=text)
        )
        if progress and duration > 0:
            frac = min(0.99, float(seg.end) / duration)
            progress(frac, f"Transcribed {int(frac * 100)}% ({len(segments)} segments)")

    if not segments:
        raise TranscriptionError(
            "No speech was detected. Check that the file contains audible speech, "
            "and try turning off the voice-activity filter."
        )

    if progress:
        progress(1.0, f"Transcription complete — {len(segments)} segments")

    return Transcript(
        segments=segments,
        language=str(getattr(info, "language", language or "en")),
        duration=duration or segments[-1].end,
        model_name=getattr(model, "model_size_or_path", "faster-whisper"),
    )


def probe_duration(audio_path: str) -> float:
    """Best-effort media duration in seconds, used for time/cost estimates."""
    try:
        import av

        with av.open(audio_path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        pass
    return 0.0


def estimate_transcription_minutes(audio_seconds: float, model_size: str) -> float:
    """Rough wall-clock estimate on a single CPU core.

    Multipliers are measured real-time factors for int8 CTranslate2 on a modest
    cloud vCPU. They are deliberately pessimistic — better to beat the estimate.
    """
    factors = {
        "tiny": 0.10,
        "base": 0.16,
        "small": 0.35,
        "medium": 0.90,
        "large-v3": 1.60,
    }
    return (audio_seconds / 60.0) * factors.get(model_size, 0.4)
