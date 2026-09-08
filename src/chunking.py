"""Split a transcript into overlapping time windows.

A 75-minute lecture is roughly 11,000 words. That fits in a modern context
window, but quality degrades: the model summarizes the first ten minutes well
and skims the rest. Chunking by time gives every part of the lecture equal
attention and lets questions be distributed across the whole session instead of
clustering at the start.

The overlap keeps an idea that straddles a boundary from being cut in half.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import Chunk, Transcript


def chunk_transcript(
    transcript: Transcript,
    chunk_seconds: int = 600,
    overlap_seconds: int = 30,
    min_chars: int = 200,
) -> list[Chunk]:
    """Break the transcript into windows of ``chunk_seconds`` with overlap."""
    if not transcript.segments:
        return []

    total = transcript.duration or transcript.segments[-1].end
    if total <= chunk_seconds:
        return [
            Chunk(index=0, start=0.0, end=total, text=transcript.text)
        ]

    step = max(60, chunk_seconds - overlap_seconds)
    chunks: list[Chunk] = []
    start = 0.0
    index = 0

    while start < total:
        end = min(start + chunk_seconds, total)
        text = " ".join(
            s.text.strip()
            for s in transcript.segments
            if s.end > start and s.start < end
        ).strip()
        if len(text) >= min_chars:
            chunks.append(Chunk(index=index, start=start, end=end, text=text))
            index += 1
        if end >= total:
            break
        start += step

    return chunks


def allocate_questions(num_questions: int, num_chunks: int) -> list[int]:
    """Spread N questions across M chunks as evenly as possible.

    The fallback used when there is no summary to judge importance from.
    Remainders go to the earliest chunks, which usually carry the framing.
    """
    if num_chunks <= 0:
        return []
    base, extra = divmod(max(0, num_questions), num_chunks)
    return [base + (1 if i < extra else 0) for i in range(num_chunks)]


# Words that say nothing about what a lecture was about. Deliberately short: an
# aggressive stop list starts deleting the domain terms that carry the signal.
STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from by
    with without into over under again further once here there when where why how all any
    both each few more most other some such no nor not only own same so too very can will
    just should now is are was were be been being have has had do does did as it its
    they them their we our you your i he she his her explain describe discuss understand
    students lecture today going want like know think about""".split()
)

# A window carrying none of the lecture's key material still gets this share of
# the busiest window's weight. Not zero: the summary is itself a model's
# opinion, and it should not be able to silence ten minutes outright.
FLOOR_WEIGHT = 0.15

# No single window may hold more than this share of the quiz, however dense it
# looks. Twelve questions about one ten-minute stretch is not a lecture quiz.
MAX_SHARE = 0.45


def _keywords(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z][a-z'-]{2,}", text.lower())
        if word not in STOPWORDS
    }


def _timestamp_seconds(value: str) -> float | None:
    """Parse "M:SS" or "H:MM:SS" out of a summary outline entry."""
    parts = value.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.strip().isdigit() for p in parts):
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def chunk_importance(chunks: list[Chunk], summary: Any) -> list[float]:
    """Score each window by how much of the lecture's key material it carries.

    The summary already names what mattered — learning objectives, key points,
    and a timestamped outline. This asks, window by window, how much of that
    lands here; the answer decides how many questions the window is worth.

    Two signals, deliberately different in kind. Outline timestamps say *where*
    the important material is and are exact when present. Objectives and key
    points are prose, matched by shared vocabulary — cruder, but it still works
    when the model produced no usable timestamps, which happens.
    """
    if not chunks:
        return []

    objectives = list(getattr(summary, "learning_objectives", None) or [])
    key_points = list(getattr(summary, "key_points", None) or [])
    outline = list(getattr(summary, "outline", None) or [])

    scores = [0.0] * len(chunks)

    for entry in outline:
        if not isinstance(entry, dict):
            continue
        when = _timestamp_seconds(str(entry.get("timestamp", "")))
        if when is None:
            continue
        for i, chunk in enumerate(chunks):
            if chunk.start <= when < chunk.end:
                scores[i] += 1.0
                break

    # Objectives count double: what the instructor means to assess outranks
    # what merely came up.
    for weight, lines in ((2.0, objectives), (1.0, key_points)):
        for line in lines:
            wanted = _keywords(str(line))
            if not wanted:
                continue
            overlaps = [len(wanted & _keywords(c.text)) / len(wanted) for c in chunks]
            best = max(overlaps)
            if best <= 0:
                continue
            # Credit every window covering this point nearly as well as the best
            # one — an idea developed across a boundary belongs to both.
            for i, share in enumerate(overlaps):
                if share >= best * 0.75:
                    scores[i] += weight * share

    return scores


def allocate_by_importance(
    num_questions: int, chunks: list[Chunk], summary: Any = None
) -> list[int]:
    """Distribute questions by what matters, not by the clock.

    The old behaviour was one question per ten-minute window, which treats a
    lecture as though every minute is equally worth examining. It is not: the
    first five minutes are admin, some stretch in the middle is the actual
    argument, and there is usually a tangent and a Q&A. Even allocation spends
    the same number of questions on each, and the quiz reads like a stopwatch.

    So allocation follows the summary's own account of what the lecture was
    about. Windows carrying objectives and key points get more; windows carrying
    none get a floor rather than a zero. One window is capped at
    :data:`MAX_SHARE` of the quiz so a dense passage cannot take the whole thing.

    Falls back to even allocation with no summary — the transcript-import path,
    or a run where summarization failed.
    """
    if not chunks or num_questions <= 0:
        return [0] * len(chunks)
    if len(chunks) == 1:
        return [num_questions]

    scores = chunk_importance(chunks, summary) if summary is not None else []
    if not scores or max(scores) <= 0:
        return allocate_questions(num_questions, len(chunks))

    top = max(scores)
    weights = [max(score, top * FLOOR_WEIGHT) for score in scores]
    cap = max(1, int(num_questions * MAX_SHARE))

    # Largest-remainder apportionment: proportional, integral, and it sums to
    # exactly the target — which matters, because the count is a promise.
    total = sum(weights)
    exact = [num_questions * w / total for w in weights]
    counts = [min(cap, int(value)) for value in exact]

    remaining = num_questions - sum(counts)
    order = sorted(
        range(len(chunks)),
        key=lambda i: (exact[i] - int(exact[i]), exact[i]),
        reverse=True,
    )
    while remaining > 0:
        placed = False
        for i in order:
            if remaining == 0:
                break
            if counts[i] < cap:
                counts[i] += 1
                remaining -= 1
                placed = True
        if not placed:
            cap += 1  # everything is at the cap; raise it rather than under-deliver
    return counts
