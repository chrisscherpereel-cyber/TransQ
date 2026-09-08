"""Import an existing transcript, skipping Whisper entirely.

Plenty of lectures are already transcribed — Panopto, Zoom, Echo360, Teams and
YouTube all emit captions, and a department may have human-corrected transcripts
that are better than anything Whisper would produce locally. Re-transcribing
those wastes twenty minutes to arrive at a worse result.

Four shapes are recognized:

* **SRT** and **WebVTT** — real caption files with real timings. These are the
  best input: every question keeps a true timestamp back into the recording.
* **Timestamped text** — lines like ``[12:34] ...`` or ``00:12:34 ...``, which is
  what most "download transcript" buttons produce (and what this app exports).
* **Plain text or Markdown** — no timings at all. Still perfectly usable; the
  text is split into evenly-spaced pseudo-segments so chunking and the
  summarizer behave normally, and the app is explicit that the resulting
  timestamps are estimates rather than measurements.
"""

from __future__ import annotations

import re

from .schema import Segment, Transcript, TranscriptPart

# 00:01:02,500 / 00:01:02.500 / 01:02.500 / 1:02
_TS = r"(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"
SRT_ARROW = re.compile(rf"({_TS})\s*-->\s*({_TS})")
LEADING_TS = re.compile(rf"^\s*[\[(<]?\s*({_TS})\s*[\])>]?\s*[-–—:]?\s*")
SPEAKER = re.compile(r"^\s*(?:[A-Z][A-Za-z.'\- ]{1,40}|SPEAKER_\d+)\s*:\s+")

# Assumed speaking rate for text with no timings. 150 wpm is a normal lecturing
# pace; it only affects the scale of estimated timestamps, never the content.
WORDS_PER_MINUTE = 150.0
SECONDS_PER_WORD = 60.0 / WORDS_PER_MINUTE


class TranscriptImportError(ValueError):
    pass


def parse_clock(value: str) -> float:
    """"1:02:03,500" -> seconds."""
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise TranscriptImportError(f"Unreadable timestamp: {value!r}") from exc
    total = 0.0
    for number in numbers:
        total = total * 60 + number
    return total


def detect_format(text: str, filename: str = "") -> str:
    """Return one of: vtt, srt, timestamped, plain."""
    head = text.lstrip()[:400]
    lower = filename.lower()

    if head.upper().startswith("WEBVTT") or lower.endswith(".vtt"):
        return "vtt"
    if SRT_ARROW.search(head):
        return "vtt" if lower.endswith(".vtt") else "srt"

    lines = [ln for ln in text.splitlines() if ln.strip()][:25]
    if lines and sum(1 for ln in lines if LEADING_TS.match(ln)) >= max(2, len(lines) // 3):
        return "timestamped"
    return "plain"


# --------------------------------------------------------------------------- #
# Caption formats
# --------------------------------------------------------------------------- #


def _parse_cues(text: str) -> list[tuple[float, float, str]]:
    """Pull (start, end, text) out of SRT or WebVTT. Shared: they differ only
    in the header, the cue numbering, and ``.`` versus ``,`` in the timestamps —
    none of which changes how the cues are read."""
    cues: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))

    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        arrow_at = next((i for i, ln in enumerate(lines) if SRT_ARROW.search(ln)), None)
        if arrow_at is None:
            continue
        match = SRT_ARROW.search(lines[arrow_at])
        start, end = parse_clock(match.group(1)), parse_clock(match.group(2))

        body_lines = lines[arrow_at + 1 :]
        body = " ".join(ln.strip() for ln in body_lines).strip()
        # WebVTT inline tags (<v Speaker>, <c.colour>) carry no meaning here.
        body = re.sub(r"<[^>]+>", "", body).strip()
        body = SPEAKER.sub("", body)
        if body:
            cues.append((start, end, body))

    return cues


def from_captions(text: str) -> list[Segment]:
    cues = _parse_cues(text)
    if not cues:
        raise TranscriptImportError(
            "No caption cues were found. Check that the file really is SRT or WebVTT."
        )
    return [
        Segment(index=i, start=start, end=end, text=body)
        for i, (start, end, body) in enumerate(cues)
    ]


# --------------------------------------------------------------------------- #
# Timestamped plain text
# --------------------------------------------------------------------------- #


def from_timestamped_text(text: str) -> list[Segment]:
    """Lines that begin with a clock reading, as most transcript exports do."""
    segments: list[Segment] = []
    pending: list[str] = []
    start: float | None = None

    def flush(end: float | None) -> None:
        nonlocal pending, start
        body = SPEAKER.sub("", " ".join(pending).strip()).strip()
        if body and start is not None:
            segments.append(
                Segment(
                    index=len(segments),
                    start=start,
                    end=end if end is not None else start + _estimate_seconds(body),
                    text=body,
                )
            )
        pending = []

    for line in text.splitlines():
        if not line.strip():
            continue
        match = LEADING_TS.match(line)
        if match:
            new_start = parse_clock(match.group(1))
            flush(new_start)
            start = new_start
            remainder = line[match.end() :].strip()
            if remainder:
                pending.append(remainder)
        elif start is not None:
            pending.append(line.strip())

    flush(None)
    if not segments:
        raise TranscriptImportError("No timestamped lines were found.")
    return segments


# --------------------------------------------------------------------------- #
# Untimed text
# --------------------------------------------------------------------------- #


def _estimate_seconds(text: str) -> float:
    return max(1.0, len(text.split()) * SECONDS_PER_WORD)


def from_plain_text(text: str, words_per_segment: int = 90) -> list[Segment]:
    """Split untimed prose into evenly-spaced pseudo-segments.

    The timings are estimates from a normal speaking rate, not measurements.
    Everything downstream — chunking, the summarizer's windows, the timestamp on
    each question — needs *a* timeline to work with, and an estimated one keeps
    those features functioning. The UI labels it as estimated so nobody treats a
    question's timestamp as a place to scrub to in the recording.
    """
    cleaned = _strip_markdown(text)
    words = cleaned.split()
    if not words:
        raise TranscriptImportError("The transcript is empty.")

    segments: list[Segment] = []
    clock = 0.0
    for i in range(0, len(words), words_per_segment):
        body = " ".join(words[i : i + words_per_segment])
        span = _estimate_seconds(body)
        segments.append(
            Segment(index=len(segments), start=clock, end=clock + span, text=body)
        )
        clock += span
    return segments


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^```.*?^```", " ", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", r"\1\2", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def import_transcript(
    text: str, filename: str = "transcript.txt", language: str = "en"
) -> tuple[Transcript, str]:
    """Parse any supported transcript. Returns ``(transcript, description)``."""
    if not text or not text.strip():
        raise TranscriptImportError("That file is empty.")

    kind = detect_format(text, filename)
    if kind in ("srt", "vtt"):
        segments = from_captions(text)
        described = f"{kind.upper()} captions with real timings"
        timings_are_real = True
    elif kind == "timestamped":
        segments = from_timestamped_text(text)
        described = "timestamped text"
        timings_are_real = True
    else:
        segments = from_plain_text(text)
        described = "plain text — timestamps are estimated from a 150 wpm speaking rate"
        timings_are_real = False

    if not segments:
        raise TranscriptImportError("Nothing usable was found in that file.")

    duration = segments[-1].end
    transcript = Transcript(
        segments=segments,
        language=language,
        duration=duration,
        model_name="imported" if timings_are_real else "imported (estimated timings)",
        parts=[
            TranscriptPart(
                index=0,
                filename=filename,
                offset=0.0,
                duration=duration,
                segments=len(segments),
            )
        ],
    )
    return transcript, described
