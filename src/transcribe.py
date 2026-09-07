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


class TranscriptBuilder:
    """Stitches parts onto one continuous timeline, and can be resumed.

    Pulled out of ``transcribe_parts`` so that a half-finished recording is a
    real, saveable object rather than a local variable. That is the whole point:
    the accumulated state used to live only in a Python list, so a process death
    on the last part discarded every part before it.

    Each part's timestamps are shifted by the running total before its segments
    are appended, so a question generated from part three points at 0:52:14 of
    the lecture rather than 0:02:14 of the third file.
    """

    def __init__(self, language: str = "en", model_name: str = ""):
        self.segments: list[Segment] = []
        self.parts: list[TranscriptPart] = []
        self.failures: list[str] = []
        self.offset = 0.0
        self.language = language or "en"
        self.model_name = model_name

    @classmethod
    def resuming(cls, transcript: Transcript) -> "TranscriptBuilder":
        """Continue from a checkpoint, appending after everything already done."""
        builder = cls(transcript.language, transcript.model_name)
        builder.segments = list(transcript.segments)
        builder.parts = list(transcript.parts)
        builder.failures = list(transcript.skipped_parts)
        # Resume where the saved timeline ends, not at zero — otherwise the new
        # part would overwrite the old one's timestamps.
        builder.offset = transcript.duration or (
            transcript.segments[-1].end if transcript.segments else 0.0
        )
        return builder

    @property
    def part_index(self) -> int:
        return len(self.parts)

    def add(self, part: Transcript, filename: str, probed_duration: float = 0.0) -> None:
        index = self.part_index
        for seg in part.segments:
            self.segments.append(
                Segment(
                    index=len(self.segments),
                    start=seg.start + self.offset,
                    end=seg.end + self.offset,
                    text=seg.text,
                    part=index,
                )
            )

        # Prefer the container's reported duration over the last segment's end,
        # so trailing silence in a part does not shift everything after it.
        measured = max(probed_duration, part.duration, part.segments[-1].end)
        self.parts.append(
            TranscriptPart(
                index=index,
                filename=filename,
                offset=self.offset,
                duration=measured,
                segments=len(part.segments),
            )
        )
        self.offset += measured
        self.language = part.language or self.language

    def skip(self, filename: str, reason: str) -> None:
        self.failures.append(f"{filename}: {reason}")

    def build(self, pending: list[str] | None = None) -> Transcript:
        """Snapshot the work so far. ``pending`` names the parts still to come."""
        return Transcript(
            segments=list(self.segments),
            language=self.language,
            duration=self.offset,
            model_name=self.model_name,
            parts=list(self.parts),
            skipped_parts=list(self.failures),
            pending_parts=list(pending or []),
        )


def transcribe_parts(
    model: Any,
    audio_paths: list[str],
    display_names: list[str] | None = None,
    language: str | None = None,
    vad_filter: bool = True,
    beam_size: int = 1,
    progress: ProgressFn | None = None,
    on_part_complete: Callable[[Transcript], None] | None = None,
    resume_from: Transcript | None = None,
) -> Transcript:
    """Transcribe several files in order and stitch them into one transcript.

    This exists because the practical ceiling on a single upload is not the
    lecture — it is the host. Splitting a 75-minute recording into three
    25-minute files and handing them over in order produces exactly the same
    combined transcript as one long upload.

    Files are transcribed **sequentially**, not in parallel: the Whisper model is
    a single shared object and running it concurrently on one CPU core would be
    slower, not faster.

    ``on_part_complete`` is called with a valid, saveable :class:`Transcript`
    after each part finishes, carrying the remaining filenames in
    ``pending_parts``. **Callers should persist it.** A container that runs out
    of memory is SIGKILLed: no exception is raised, no ``except`` runs, no
    ``finally`` runs. The only state that survives is state already written down,
    so a checkpoint after each part is the difference between losing one part and
    losing the lecture. This matters most for split recordings, because splitting
    is what people do when a single run is already too big for the host.

    ``resume_from`` continues a checkpoint: the new parts are appended after
    everything it already contains. Splits are assumed contiguous and
    non-overlapping; overlapping parts duplicate the overlapping speech.
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

    builder = (
        TranscriptBuilder.resuming(resume_from)
        if resume_from is not None
        else TranscriptBuilder(
            language or "en", getattr(model, "model_size_or_path", "faster-whisper")
        )
    )
    already_done = len(builder.parts)
    total_parts = already_done + len(audio_paths)

    for i, (path, name) in enumerate(zip(audio_paths, names)):
        part_label = f"part {already_done + i + 1} of {total_parts} ({name})"
        base_offset = sum(d for d in part_durations[:i] if d > 0)

        def part_progress(frac: float, message: str, _i=i, _base=base_offset) -> None:
            if progress is None:
                return
            if known_total > 0:
                elapsed = _base + frac * max(part_durations[_i], 0.0)
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
            builder.skip(name, str(exc))
            _checkpoint(builder, names[i + 1 :], on_part_complete)
            continue

        builder.add(part, name, part_durations[i])
        _checkpoint(builder, names[i + 1 :], on_part_complete)

    if not builder.segments:
        raise TranscriptionError(
            "No speech was detected in any part. "
            + (" ".join(builder.failures) if builder.failures else "")
        )

    if progress:
        note = f" ({len(builder.failures)} part(s) skipped)" if builder.failures else ""
        progress(
            1.0,
            f"Combined {len(builder.parts)} part(s) — "
            f"{len(builder.segments)} segments{note}",
        )

    return builder.build()


def _checkpoint(
    builder: TranscriptBuilder,
    pending: list[str],
    on_part_complete: Callable[[Transcript], None] | None,
) -> None:
    """Hand the caller a saveable snapshot, without letting a save failure kill
    the run. Storage being briefly unavailable is a reason to keep transcribing,
    not to discard the parts already done."""
    if on_part_complete is None or not builder.segments:
        return
    try:
        on_part_complete(builder.build(pending))
    except Exception:  # noqa: BLE001 - a checkpoint is best-effort by design
        pass


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
