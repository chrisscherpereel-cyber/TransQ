"""Typed data structures shared across transcription, generation, and export.

Everything downstream (validators, editors, exporters) speaks these objects, so
a new export format never has to know how a question was produced.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

BloomLevel = Literal[
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
]
Difficulty = Literal["Easy", "Medium", "Hard"]

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


class Segment(BaseModel):
    """One Whisper segment, kept so every claim can be traced back to audio.

    ``start``/``end`` are always on the *lecture* timeline. When a recording was
    uploaded in parts, each part's times are offset by the running total before
    its segments land here, so a timestamp on a question means the same thing
    whether the lecture arrived as one file or five.
    """

    index: int
    start: float
    end: float
    text: str
    part: int = 0

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)


class TranscriptPart(BaseModel):
    """Bookkeeping for one uploaded file within a multi-part recording."""

    index: int
    filename: str
    offset: float  # where this part begins on the combined timeline
    duration: float
    segments: int = 0

    # Recorded when this part finished, for diagnosing a run that later died
    # with no error. Memory rising steadily across parts is a different problem
    # from memory sitting flat, and after a crash these are the only evidence.
    memory_gb: float = 0.0
    elapsed_seconds: float = 0.0
    finished_at: str = ""

    @property
    def label(self) -> str:
        return (
            f"{self.index + 1}. {self.filename} "
            f"({format_timestamp(self.offset)}–{format_timestamp(self.offset + self.duration)})"
        )


class Transcript(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    language: str = "en"
    duration: float = 0.0
    model_name: str = ""
    parts: list[TranscriptPart] = Field(default_factory=list)
    skipped_parts: list[str] = Field(default_factory=list)

    # Filenames of parts that have not been transcribed yet, in order. A
    # transcript checkpointed halfway through a split recording carries the rest
    # of the queue here, which is what makes resuming possible: the saved
    # document knows what is missing, so a crashed run costs one part rather
    # than the whole lecture. Empty means finished.
    pending_parts: list[str] = Field(default_factory=list)

    @property
    def is_multipart(self) -> bool:
        return len(self.parts) > 1

    @property
    def is_complete(self) -> bool:
        return not self.pending_parts

    @property
    def progress_label(self) -> str:
        """e.g. "2 of 3 parts transcribed" — for the library list and resume UI."""
        done = len(self.parts)
        total = done + len(self.pending_parts)
        return f"{done} of {total} parts transcribed"

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def text_with_timestamps(self) -> str:
        return "\n".join(f"[{s.timestamp}] {s.text.strip()}" for s in self.segments)

    def slice_by_time(self, start: float, end: float) -> str:
        return " ".join(
            s.text.strip() for s in self.segments if s.start >= start and s.start < end
        )


class Chunk(BaseModel):
    """A window of transcript small enough to fit an LLM context comfortably."""

    index: int
    start: float
    end: float
    text: str

    @property
    def label(self) -> str:
        return f"{format_timestamp(self.start)}–{format_timestamp(self.end)}"


class Summary(BaseModel):
    title: str = ""
    abstract: str = ""
    key_points: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    key_terms: list[dict[str, str]] = Field(default_factory=list)
    outline: list[dict[str, str]] = Field(default_factory=list)

    def as_markdown(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"# {self.title}\n")
        if self.abstract:
            parts.append(f"{self.abstract}\n")
        if self.learning_objectives:
            parts.append("## Learning objectives\n")
            parts += [f"- {o}" for o in self.learning_objectives]
            parts.append("")
        if self.key_points:
            parts.append("## Key points\n")
            parts += [f"- {p}" for p in self.key_points]
            parts.append("")
        if self.outline:
            parts.append("## Outline\n")
            for item in self.outline:
                ts = item.get("timestamp", "")
                head = item.get("heading", "")
                body = item.get("detail", "")
                parts.append(f"- **{ts} {head}** — {body}" if ts else f"- **{head}** — {body}")
            parts.append("")
        if self.key_terms:
            parts.append("## Key terms\n")
            for term in self.key_terms:
                parts.append(f"- **{term.get('term', '')}** — {term.get('definition', '')}")
            parts.append("")
        return "\n".join(parts).strip()


class MCQ(BaseModel):
    """One multiple-choice item, with provenance and pedagogical metadata."""

    id: str = Field(default_factory=lambda: f"q{uuid.uuid4().hex[:10]}")
    stem: str
    options: list[str]
    correct_index: int
    rationale: str = ""
    distractor_rationales: list[str] = Field(default_factory=list)
    bloom: BloomLevel = "Understand"
    difficulty: Difficulty = "Medium"
    topic: str = ""
    source_timestamp: str = ""
    source_quote: str = ""
    points: float = 1.0
    flags: list[str] = Field(default_factory=list)
    include: bool = True

    @field_validator("options")
    @classmethod
    def _at_least_three(cls, v: list[str]) -> list[str]:
        if len(v) < 3:
            raise ValueError("a multiple-choice item needs at least three options")
        if len(v) > len(OPTION_LETTERS):
            raise ValueError(f"at most {len(OPTION_LETTERS)} options are supported")
        return [o.strip() for o in v]

    @field_validator("correct_index")
    @classmethod
    def _in_range(cls, v: int) -> int:
        if v < 0:
            raise ValueError("correct_index must be non-negative")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index points past the end of options")

    @property
    def answer_letter(self) -> str:
        return OPTION_LETTERS[self.correct_index]

    @property
    def correct_option(self) -> str:
        return self.options[self.correct_index]

    def lettered_options(self) -> list[tuple[str, str]]:
        return list(zip(OPTION_LETTERS, self.options))


class QuizMeta(BaseModel):
    title: str = "Lecture Quiz"
    course: str = ""
    description: str = ""
    source_filename: str = ""
    generated_on: str = ""
    model_used: str = ""


class Quiz(BaseModel):
    meta: QuizMeta = Field(default_factory=QuizMeta)
    questions: list[MCQ] = Field(default_factory=list)

    @property
    def included(self) -> list[MCQ]:
        return [q for q in self.questions if q.include]

    @property
    def total_points(self) -> float:
        return sum(q.points for q in self.included)


def format_timestamp(seconds: float) -> str:
    """Seconds -> H:MM:SS (or M:SS under an hour)."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_timestamp(value: str) -> float:
    """H:MM:SS or M:SS -> seconds. Returns 0.0 on anything unparseable."""
    if not value:
        return 0.0
    match = re.findall(r"\d+", value)
    if not match:
        return 0.0
    nums = [int(n) for n in match][-3:]
    total = 0.0
    for n in nums:
        total = total * 60 + n
    return total
