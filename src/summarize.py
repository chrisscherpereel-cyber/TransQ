"""Map-reduce summarization over transcript chunks."""

from __future__ import annotations

from collections.abc import Callable

from . import prompts
from .chunking import chunk_transcript
from .llm import LLMClient
from .schema import Chunk, Summary, Transcript

ProgressFn = Callable[[float, str], None]


def _course_context_block(course_context: str) -> str:
    if not course_context.strip():
        return ""
    return (
        "Course context provided by the instructor (use it to judge what matters, "
        f"not as a source of content):\n{course_context.strip()}\n"
    )


def summarize_chunk(client: LLMClient, chunk: Chunk, course_context: str = "") -> dict:
    data = client.complete_json(
        prompts.SUMMARY_SYSTEM,
        prompts.SUMMARY_CHUNK_USER.format(
            label=chunk.label,
            course_context=_course_context_block(course_context),
            text=chunk.text[:24000],
        ),
        max_tokens=2000,
    )
    data["_label"] = chunk.label
    data["_start"] = chunk.start
    return data


def summarize_transcript(
    client: LLMClient,
    transcript: Transcript,
    course_context: str = "",
    chunk_seconds: int = 600,
    overlap_seconds: int = 30,
    progress: ProgressFn | None = None,
) -> tuple[Summary, list[Chunk], list[dict]]:
    """Summarize a full transcript.

    Returns the synthesized summary plus the chunks and per-chunk summaries, so
    question generation can reuse them instead of paying for them twice.
    """
    chunks = chunk_transcript(transcript, chunk_seconds, overlap_seconds)
    if not chunks:
        return Summary(), [], []

    sections: list[dict] = []
    for i, chunk in enumerate(chunks):
        if progress:
            progress(i / max(1, len(chunks)) * 0.8, f"Summarizing {chunk.label}")
        try:
            sections.append(summarize_chunk(client, chunk, course_context))
        except Exception as exc:  # one bad chunk should not sink the run
            sections.append(
                {"_label": chunk.label, "_start": chunk.start,
                 "heading": f"Segment {i + 1}", "key_points": [],
                 "key_terms": [], "_error": str(exc)}
            )

    if progress:
        progress(0.85, "Synthesizing the overall summary")

    summary = _reduce(client, sections, course_context)

    if progress:
        progress(1.0, "Summary complete")
    return summary, chunks, sections


def _reduce(client: LLMClient, sections: list[dict], course_context: str) -> Summary:
    rendered: list[str] = []
    for s in sections:
        lines = [f"### [{s.get('_label', '')}] {s.get('heading', '')}"]
        lines += [f"- {p}" for p in s.get("key_points", [])]
        for term in s.get("key_terms", []) or []:
            lines.append(f"- TERM {term.get('term', '')}: {term.get('definition', '')}")
        if s.get("notable_quote"):
            lines.append(f'- QUOTE "{s["notable_quote"]}"')
        rendered.append("\n".join(lines))

    data = client.complete_json(
        prompts.SUMMARY_SYSTEM,
        prompts.SUMMARY_REDUCE_USER.format(
            course_context=_course_context_block(course_context),
            sections="\n\n".join(rendered)[:40000],
        ),
        max_tokens=4000,
    )

    return Summary(
        title=str(data.get("title", "") or ""),
        abstract=str(data.get("abstract", "") or ""),
        key_points=[str(p) for p in data.get("key_points", []) or []],
        learning_objectives=[str(o) for o in data.get("learning_objectives", []) or []],
        key_terms=[
            {"term": str(t.get("term", "")), "definition": str(t.get("definition", ""))}
            for t in data.get("key_terms", []) or []
            if isinstance(t, dict)
        ],
        outline=[
            {
                "timestamp": str(o.get("timestamp", "")),
                "heading": str(o.get("heading", "")),
                "detail": str(o.get("detail", "")),
            }
            for o in data.get("outline", []) or []
            if isinstance(o, dict)
        ],
    )
