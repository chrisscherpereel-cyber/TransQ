"""Local speech-to-text with faster-whisper.

faster-whisper runs the Whisper weights through CTranslate2, which is roughly
4x quicker than openai-whisper on CPU and uses about half the memory — the
difference between "runs on Streamlit Community Cloud" and "doesn't".

Segment-level timestamps are kept because every generated question carries a
pointer back to the moment in the lecture it came from.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

from .schema import Segment, Transcript, TranscriptPart

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

    total = duration or segments[-1].end
    name = os.path.basename(audio_path)
    return Transcript(
        segments=segments,
        language=str(getattr(info, "language", language or "en")),
        duration=total,
        model_name=getattr(model, "model_size_or_path", "faster-whisper"),
        parts=[
            TranscriptPart(
                index=0, filename=name, offset=0.0, duration=total, segments=len(segments)
            )
        ],
    )


def transcribe_parts(
    model: Any,
    audio_paths: list[str],
    display_names: list[str] | None = None,
    language: str | None = None,
    vad_filter: bool = True,
    beam_size: int = 1,
    progress: ProgressFn | None = None,
) -> Transcript:
    """Transcribe several files in order and stitch them into one transcript.

    This exists because the practical ceiling on a single upload is not the
    lecture — it is the host. Splitting a 75-minute recording into three
    25-minute files and handing them over in order produces exactly the same
    combined transcript as one long upload, because each part's timestamps are
    shifted by the running total before its segments are appended. A question
    generated from part three still points at 0:52:14 of the lecture, not
    0:02:14 of the third file.

    Files are transcribed **sequentially**, not in parallel: the Whisper model is
    a single shared object and running it concurrently on one CPU core would be
    slower, not faster.

    Splits are assumed to be contiguous and non-overlapping. If parts overlap,
    the overlapping speech appears twice — the near-duplicate check will flag
    resulting questions, but trimming the overlap beforehand is better.
    """
    if not audio_paths:
        raise TranscriptionError("No audio files were provided.")

    names = display_names or [os.path.basename(p) for p in audio_paths]
    if len(names) != len(audio_paths):
        raise TranscriptionError("display_names must line up with audio_paths.")

    # Probe every part first so the progress bar reflects the whole recording
    # rather than restarting at zero for each file.
    part_durations = [probe_duration(p) for p in audio_paths]
    known_total = sum(d for d in part_durations if d > 0)

    all_segments: list[Segment] = []
    parts: list[TranscriptPart] = []
    offset = 0.0
    language_seen = language or "en"
    failures: list[str] = []

    for i, (path, name) in enumerate(zip(audio_paths, names)):
        part_label = f"part {i + 1} of {len(audio_paths)} ({name})"

        def part_progress(frac: float, message: str, _i=i, _offset=offset) -> None:
            if progress is None:
                return
            if known_total > 0:
                elapsed = _offset + frac * max(part_durations[_i], 0.0)
                overall = min(0.99, elapsed / known_total)
            else:
                overall = min(0.99, (_i + frac) / len(audio_paths))
            progress(overall, f"Transcribing {part_label} — {message}")

        try:
            part = transcribe_file(
                model,
                path,
                language=language,
                vad_filter=vad_filter,
                beam_size=beam_size,
                progress=part_progress,
            )
        except TranscriptionError as exc:
            # One unreadable or silent part should not throw away the rest.
            failures.append(f"{name}: {exc}")
            continue

        for seg in part.segments:
            all_segments.append(
                Segment(
                    index=len(all_segments),
                    start=seg.start + offset,
                    end=seg.end + offset,
                    text=seg.text,
                    part=i,
                )
            )

        # Prefer the container's reported duration over the last segment's end,
        # so trailing silence in a part does not shift everything after it.
        measured = max(part_durations[i], part.duration, part.segments[-1].end)
        parts.append(
            TranscriptPart(
                index=i,
                filename=name,
                offset=offset,
                duration=measured,
                segments=len(part.segments),
            )
        )
        offset += measured
        language_seen = part.language or language_seen

    if not all_segments:
        raise TranscriptionError(
            "No speech was detected in any part. "
            + (" ".join(failures) if failures else "")
        )

    if progress:
        note = f" ({len(failures)} part(s) skipped)" if failures else ""
        progress(1.0, f"Combined {len(parts)} part(s) — {len(all_segments)} segments{note}")

    return Transcript(
        segments=all_segments,
        language=language_seen,
        duration=offset,
        model_name=getattr(model, "model_size_or_path", "faster-whisper"),
        parts=parts,
        skipped_parts=failures,
    )


def natural_sort_key(name: str) -> tuple:
    """Sort key that orders ``part2`` before ``part10``.

    Split recordings are almost always named with a trailing number, and plain
    alphabetical sorting puts part 10 immediately after part 1 — which would
    silently scramble the middle of a lecture.
    """
    return tuple(
        int(chunk) if chunk.isdigit() else chunk.lower()
        for chunk in re.split(r"(\d+)", name)
        if chunk != ""
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
