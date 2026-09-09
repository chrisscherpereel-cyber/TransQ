"""Map-reduce summarization over transcript chunks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import prompts
from .chunking import chunk_transcript
from .diagnostics import (
    FAILED,
    OK,
    PHASE_SUMMARY,
    SKIPPED,
    RunReport,
    classify,
    short_reason,
)
from .llm import (
    LLMClient,
    LLMError,
    TruncatedResponseError,
    salvage_object_fields,
)
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
    try:
        data = client.complete_json(
            prompts.SUMMARY_SYSTEM,
            prompts.SUMMARY_CHUNK_USER.format(
                label=chunk.label,
                course_context=_course_context_block(course_context),
                text=chunk.text[:24000],
            ),
            max_tokens=2000,
        )
    except TruncatedResponseError as exc:
        # A window whose reply was cut off has usually produced its heading and
        # several key points already. Keeping those costs nothing and means one
        # long-winded window does not leave a hole in the summary.
        data = salvage_object_fields(exc.raw)
        if not data.get("key_points"):
            raise
    data["_label"] = chunk.label
    data["_start"] = chunk.start
    return data


def _material_block(material: Any) -> str:
    """The deck's own structure, for the synthesis step.

    Only titles, not full slide text. The reduce call already carries every
    section summary; adding a whole deck on top is the fastest way to truncate
    it. Titles are enough to fix terminology and to reveal material the
    transcript skated over — a slide the instructor put up and barely narrated
    still belongs in an account of what the lecture covered.
    """
    if material is None or not getattr(material, "sections", None):
        return ""
    return (
        f"The instructor's {material.section_noun}s for this lecture, in order. "
        "Use their wording for technical terms, and treat a topic that appears "
        "here but is thin in the transcript as covered rather than absent:\n"
        f"{material.outline()}\n\n"
    )


def _exam_block(exam_topics: list[str] | None) -> str:
    """What the instructor says will be tested."""
    topics = [str(t).strip() for t in (exam_topics or []) if str(t).strip()]
    if not topics:
        return ""
    listed = "\n".join(f"- {t}" for t in topics[:20])
    return (
        "The instructor will examine these topics. Make sure the learning "
        "objectives and key points cover them explicitly, using the "
        "instructor's own wording:\n"
        f"{listed}\n\n"
    )


def summarize_transcript(
    client: LLMClient,
    transcript: Transcript,
    course_context: str = "",
    chunk_seconds: int = 600,
    overlap_seconds: int = 30,
    material: Any = None,
    exam_topics: list[str] | None = None,
    cached_sections: list[dict] | None = None,
    on_section: Callable[[dict], None] | None = None,
    progress: ProgressFn | None = None,
    report: RunReport | None = None,
) -> tuple[Summary, list[Chunk], list[dict]]:
    """Summarize a full transcript.

    Returns the synthesized summary plus the chunks and per-chunk summaries, so
    question generation can reuse them instead of paying for them twice.

    One bad chunk still does not sink the run — but it is now *recorded*. If
    every chunk fails, that is not a quiet empty summary, it is an error: the
    old behaviour was to carry on and generate questions from nothing, which is
    what made an expired key look like a lecture with nothing in it.
    """
    report = report or RunReport()
    chunks = chunk_transcript(transcript, chunk_seconds, overlap_seconds)
    if not chunks:
        return Summary(), [], []

    # Windows summarized by an earlier attempt. Re-running after a failure must
    # not re-pay for work that already succeeded — that was the real cost of a
    # cut-off synthesis: eight good section summaries thrown away because the
    # ninth call, the one that combines them, ran out of room.
    done = {
        str(s.get("_label")): s
        for s in (cached_sections or [])
        if isinstance(s, dict) and s.get("_label") and not s.get("_error")
    }

    sections: list[dict] = []
    for i, chunk in enumerate(chunks):
        if chunk.label in done:
            sections.append(done[chunk.label])
            report.record(
                PHASE_SUMMARY, chunk.label, SKIPPED,
                detail="already summarized in an earlier attempt",
                produced=len(done[chunk.label].get("key_points") or []),
            )
            continue
        if progress:
            progress(i / max(1, len(chunks)) * 0.8, f"Summarizing {chunk.label}")
        try:
            section = summarize_chunk(client, chunk, course_context)
            sections.append(section)
            _checkpoint(on_section, section)
            report.record(
                PHASE_SUMMARY, chunk.label, OK,
                produced=len(section.get("key_points") or []),
            )
        except Exception as exc:  # one bad chunk should not sink the run
            sections.append(
                {"_label": chunk.label, "_start": chunk.start,
                 "heading": f"Segment {i + 1}", "key_points": [],
                 "key_terms": [], "_error": str(exc)}
            )
            report.record(
                PHASE_SUMMARY, chunk.label, FAILED,
                detail=short_reason(exc), cause=classify(exc),
            )

    if report.phase_failed_entirely(PHASE_SUMMARY):
        raise LLMError(
            f"Every one of the {len(chunks)} summary requests failed. "
            f"Last reason: {report.failures[-1].detail}"
        )

    if progress:
        progress(0.85, "Synthesizing the overall summary")

    try:
        summary = _reduce(client, sections, course_context, material, exam_topics)
        report.record(PHASE_SUMMARY, "overall synthesis", OK)
    except Exception as exc:
        # The synthesis is one call over material that is already paid for.
        # Failing here used to discard every section summary with it. Assemble
        # one locally instead: the sections already carry headings, key points
        # and terms, so a usable summary needs no further model call. It is
        # plainer than a synthesized one, and it is not nothing.
        report.record(
            PHASE_SUMMARY, "overall synthesis", FAILED,
            detail=short_reason(exc), cause=classify(exc),
        )
        summary = summary_from_sections(sections)
        if not summary.key_points:
            raise
        report.record(
            PHASE_SUMMARY, "assembled locally", OK,
            detail="synthesis failed; built from the section summaries instead",
            produced=len(summary.key_points),
        )

    if progress:
        progress(1.0, "Summary complete")
    return summary, chunks, sections


def _checkpoint(on_section: Callable[[dict], None] | None, section: dict) -> None:
    """Hand a finished window to the caller the moment it exists.

    Returning the sections at the end is no use to a run that never reaches the
    end. Every window is a paid-for model call, so the caller gets each one as it
    lands and can resume from it — the same reason transcription checkpoints each
    part. A caller whose save fails must not take the summary down with it: the
    window succeeded either way.
    """
    if on_section is None:
        return
    try:
        on_section(section)
    except Exception:  # noqa: BLE001 - a failed checkpoint is not a failed window
        pass


def summary_from_sections(sections: list[dict]) -> Summary:
    """Build a Summary from section summaries, with no model call.

    The fallback when synthesis fails. Deliberately mechanical — it collates
    rather than rewrites, so it cannot invent anything the sections did not
    already say, and it costs nothing to run.
    """
    key_points: list[str] = []
    key_terms: list[dict[str, str]] = []
    outline: list[dict[str, str]] = []
    seen_terms: set[str] = set()

    for section in sections:
        if not isinstance(section, dict) or section.get("_error"):
            continue
        heading = str(section.get("heading") or "").strip()
        label = str(section.get("_label") or "")
        if heading:
            outline.append(
                {
                    "timestamp": label.split("–")[0].strip(),
                    "heading": heading,
                    "detail": "",
                }
            )
        for point in section.get("key_points") or []:
            text = str(point).strip()
            if text and text not in key_points:
                key_points.append(text)
        for term in section.get("key_terms") or []:
            if not isinstance(term, dict):
                continue
            name = str(term.get("term") or "").strip()
            if name and name.lower() not in seen_terms:
                seen_terms.add(name.lower())
                key_terms.append(
                    {"term": name, "definition": str(term.get("definition") or "")}
                )

    return Summary(
        title=(outline[0]["heading"] if outline else "Lecture summary"),
        abstract=(
            "Assembled from the per-section summaries because the synthesis step "
            "did not complete. The points below are what each part of the lecture "
            "produced, in order, without an overall narrative."
        )
        if key_points
        else "",
        key_points=key_points[:20],
        learning_objectives=[],
        key_terms=key_terms[:30],
        outline=outline,
    )


def _reduce(
    client: LLMClient,
    sections: list[dict],
    course_context: str,
    material: Any = None,
    exam_topics: list[str] | None = None,
) -> Summary:
    rendered: list[str] = []
    for s in sections:
        lines = [f"### [{s.get('_label', '')}] {s.get('heading', '')}"]
        lines += [f"- {p}" for p in s.get("key_points", [])]
        for term in s.get("key_terms", []) or []:
            lines.append(f"- TERM {term.get('term', '')}: {term.get('definition', '')}")
        if s.get("notable_quote"):
            lines.append(f'- QUOTE "{s["notable_quote"]}"')
        rendered.append("\n".join(lines))

    try:
        data = client.complete_json(
            prompts.SUMMARY_SYSTEM,
            prompts.SUMMARY_REDUCE_USER.format(
                course_context=_course_context_block(course_context),
                material_clause=_material_block(material),
                exam_clause=_exam_block(exam_topics),
                sections="\n\n".join(rendered)[:40000],
            ),
            max_tokens=4000,
        )
    except TruncatedResponseError as exc:
        # The synthesis writes title and abstract first and the outline last, so
        # a cut-off reply is usually a summary missing its tail rather than
        # nothing at all. Take what closed; the caller's local assembly is the
        # backstop if even that is empty.
        data = salvage_object_fields(exc.raw)
        if not (data.get("key_points") or data.get("abstract")):
            raise

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
