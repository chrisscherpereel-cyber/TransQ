"""Split a transcript into overlapping time windows.

A 75-minute lecture is roughly 11,000 words. That fits in a modern context
window, but quality degrades: the model summarizes the first ten minutes well
and skims the rest. Chunking by time gives every part of the lecture equal
attention and lets questions be distributed across the whole session instead of
clustering at the start.

The overlap keeps an idea that straddles a boundary from being cut in half.
"""

from __future__ import annotations

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

    Remainders go to the earliest chunks, which usually carry the framing
    material an instructor most wants covered.
    """
    if num_chunks <= 0:
        return []
    base, extra = divmod(max(0, num_questions), num_chunks)
    return [base + (1 if i < extra else 0) for i in range(num_chunks)]
