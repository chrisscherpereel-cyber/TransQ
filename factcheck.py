"""Flagging lecture claims worth a second look before students are examined.

**What this is not.** It is not a fact-checker, and the app must never present it
as one. A language model assessing claims from its own training has a cutoff
date, uneven coverage, and a real rate of confident error — including confidently
contradicting a correct lecture. Treated as a verdict it would be worse than
useless, because it would spend the instructor's trust on noise.

What it is: a reading pass that surfaces things an instructor might want to look
at again, with its reasoning attached and its confidence stated, so a thirty-
second glance can dismiss most of it. The instructor remains the authority
throughout. Every verdict is a question, not a finding.

**Two signals of very different strength**, and the UI should say which is which:

*Against the slides* — both texts are supplied, so "the transcript says 40% and
the slide says 14%" is a real, checkable observation. This is the reliable half,
and it catches the thing instructors actually want caught: misspeaking relative
to their own prepared material.

*Against the model's knowledge* — advisory only. Useful for "this was superseded
around 2019" and "that's one school of thought, the other is…", unreliable for
anything niche, recent, or local to a course.

**The transcription caveat does most of the work.** Whisper mishears numbers,
names, technical terms and negations. Most claims that look wrong in a lecture
transcript are transcription errors, so the prompt asks for that verdict
explicitly and separately — otherwise this feature would mostly report Whisper's
mistakes as the instructor's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from . import prompts
from .diagnostics import FAILED, OK, RunReport, classify, short_reason
from .llm import LLMClient, TruncatedResponseError, salvage_array_objects
from .schema import Chunk, format_timestamp

ProgressFn = Callable[[float, str], None]

PHASE_REVIEW_CLAIMS = "Consistency review"

# Ordered by how much they should interrupt an instructor's day.
VERDICTS: dict[str, dict[str, str]] = {
    "questionable": {
        "label": "Appears to conflict with established understanding",
        "icon": "🔴",
        "rank": "0",
    },
    "outdated": {
        "label": "May have been superseded",
        "icon": "🟠",
        "rank": "1",
    },
    "contested": {
        "label": "Legitimate disagreement in the field",
        "icon": "🟡",
        "rank": "2",
    },
    "transcription": {
        "label": "Probably a transcription error, not the instructor",
        "icon": "🎙️",
        "rank": "3",
    },
    "unclear": {
        "label": "Too garbled to assess",
        "icon": "❓",
        "rank": "4",
    },
    "consistent": {
        "label": "Matches mainstream understanding",
        "icon": "🟢",
        "rank": "5",
    },
}

NEEDS_ATTENTION = ("questionable", "outdated", "contested")


@dataclass
class Finding:
    claim: str = ""
    quote: str = ""
    timestamp: str = ""
    verdict: str = "unclear"
    confidence: str = "low"
    explanation: str = ""
    mainstream_view: str = ""
    slide_conflict: str = ""

    @property
    def icon(self) -> str:
        return VERDICTS.get(self.verdict, VERDICTS["unclear"])["icon"]

    @property
    def label(self) -> str:
        return VERDICTS.get(self.verdict, VERDICTS["unclear"])["label"]

    @property
    def rank(self) -> int:
        # Low confidence sorts below high confidence within the same verdict:
        # a hedged guess should not head the list.
        order = int(VERDICTS.get(self.verdict, VERDICTS["unclear"])["rank"])
        penalty = {"high": 0, "medium": 1, "low": 2}.get(self.confidence, 2)
        return order * 10 + penalty

    @property
    def needs_attention(self) -> bool:
        return self.verdict in NEEDS_ATTENTION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Finding":
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: str(v or "") for k, v in data.items() if k in known}
        if clean.get("verdict") not in VERDICTS:
            clean["verdict"] = "unclear"
        if clean.get("confidence") not in ("high", "medium", "low"):
            clean["confidence"] = "low"
        return cls(**clean)


@dataclass
class ReviewResult:
    findings: list[Finding] = field(default_factory=list)
    checked_windows: int = 0
    model_used: str = ""
    reviewed_at: str = ""

    @property
    def attention(self) -> list[Finding]:
        return [f for f in self.findings if f.needs_attention]

    @property
    def transcription_issues(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "transcription"]

    @property
    def slide_conflicts(self) -> list[Finding]:
        return [f for f in self.findings if f.slide_conflict.strip()]

    def headline(self) -> str:
        if not self.findings:
            return f"Nothing flagged across {self.checked_windows} window(s)."
        parts = []
        if self.attention:
            parts.append(f"{len(self.attention)} worth a look")
        if self.slide_conflicts:
            parts.append(f"{len(self.slide_conflicts)} differ from your slides")
        if self.transcription_issues:
            parts.append(f"{len(self.transcription_issues)} likely mis-transcribed")
        return " · ".join(parts) or f"{len(self.findings)} note(s)."

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "checked_windows": self.checked_windows,
            "model_used": self.model_used,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewResult":
        return cls(
            findings=[
                Finding.from_dict(f)
                for f in (data.get("findings") or [])
                if isinstance(f, dict)
            ],
            checked_windows=int(data.get("checked_windows") or 0),
            model_used=str(data.get("model_used") or ""),
            reviewed_at=str(data.get("reviewed_at") or ""),
        )


def review_transcript(
    client: LLMClient,
    chunks: list[Chunk],
    material: Any = None,
    exam_topics: list[str] | None = None,
    progress: ProgressFn | None = None,
    report: RunReport | None = None,
) -> ReviewResult:
    """Read the lecture window by window and flag what is worth re-checking.

    Windows are reviewed independently, like summarization, so one failed call
    costs one window rather than the whole review — and so a long lecture never
    needs a single enormous request.
    """
    report = report or RunReport()
    result = ReviewResult(model_used=getattr(client, "model", ""))
    if not chunks:
        return result

    topics = [str(t).strip() for t in (exam_topics or []) if str(t).strip()]
    topic_clause = ""
    if topics:
        listed = "\n".join(f"- {t}" for t in topics[:20])
        topic_clause = (
            "These topics will be examined, so claims bearing on them matter more "
            "than the rest. Prioritise them:\n"
            f"{listed}\n\n"
        )

    for i, chunk in enumerate(chunks):
        if progress:
            progress(i / max(1, len(chunks)), f"Reviewing {chunk.label}")

        material_clause = ""
        if material is not None:
            from .materials import material_context, relevant_sections

            matched = relevant_sections(chunk.text, material)
            if matched:
                material_clause = (
                    material_context(matched, material.section_noun)
                    + "Where the transcript and these differ on a fact, a figure or "
                    "a definition, say so in \"slide_conflict\". This is the most "
                    "useful thing you can find: both texts are in front of you, so "
                    "it is a real observation rather than a judgement call.\n\n"
                )

        try:
            data = _ask(client, chunk, material_clause, topic_clause)
        except TruncatedResponseError as exc:
            rescued = salvage_array_objects(exc.raw, "findings")
            if not rescued:
                report.record(
                    PHASE_REVIEW_CLAIMS, chunk.label, FAILED,
                    detail=short_reason(exc), cause=classify(exc),
                )
                continue
            data = {"findings": rescued}
        except Exception as exc:  # noqa: BLE001 - one window is not the lecture
            report.record(
                PHASE_REVIEW_CLAIMS, chunk.label, FAILED,
                detail=short_reason(exc), cause=classify(exc),
            )
            continue

        found = [
            Finding.from_dict(raw)
            for raw in (data.get("findings") or [])
            if isinstance(raw, dict)
        ]
        found = [f for f in found if f.claim.strip()]
        for finding in found:
            if not finding.timestamp:
                finding.timestamp = format_timestamp(chunk.start)

        result.findings.extend(found)
        result.checked_windows += 1
        report.record(
            PHASE_REVIEW_CLAIMS, chunk.label, OK, produced=len(found)
        )

    result.findings.sort(key=lambda f: (f.rank, f.timestamp))
    if progress:
        progress(1.0, result.headline())
    return result


def _ask(
    client: LLMClient, chunk: Chunk, material_clause: str, topic_clause: str
) -> dict:
    from .mcq import _with_inline_timestamps

    return client.complete_json(
        prompts.REVIEW_SYSTEM,
        prompts.REVIEW_USER.format(
            label=chunk.label,
            material_clause=material_clause,
            topic_clause=topic_clause,
            text=_with_inline_timestamps(chunk)[:24000],
        ),
        max_tokens=4000,
    )
