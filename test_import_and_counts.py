"""Tests for transcript import and for the question-count guarantee.

The count tests are the important half: "I asked for 12 and got 8" was the
reported bug, and the fix is only real if it holds when the model under-delivers,
when items fail to parse, and when the reviewer drops things — which is exactly
what the stub LLMs here simulate.
"""

from __future__ import annotations

import json

import pytest

from src.chunking import chunk_transcript
from src.mcq import _allocate_topup, generate_question_set
from src.schema import Segment, Transcript
from src.transcript_import import (
    TranscriptImportError,
    detect_format,
    import_transcript,
    parse_clock,
)


# --------------------------------------------------------------------------- #
# Transcript import
# --------------------------------------------------------------------------- #

SRT = """1
00:00:00,000 --> 00:00:04,500
Welcome to operations management.

2
00:00:04,500 --> 00:00:09,000
Today we cover aggregate planning.

3
00:01:30,000 --> 00:01:35,250
A chase strategy varies capacity with demand.
"""

VTT = """WEBVTT

00:00:00.000 --> 00:00:04.500
<v Instructor>Welcome to operations management.

00:00:04.500 --> 00:00:09.000
Today we cover aggregate planning.
"""

TIMESTAMPED = """[0:00] Welcome to operations management.
[0:30] Today we cover aggregate planning.
and this line continues the same thought.
[2:15] A chase strategy varies capacity with demand.
"""

PLAIN = " ".join(["Operations management concerns the design of processes."] * 60)

MARKDOWN = """# Lecture 4

## Aggregate planning

- A **chase strategy** varies capacity.
- A [level strategy](http://example.com) holds output constant.

Both trade off inventory against workforce stability, and the choice depends on
the cost structure of the operation being planned.
"""


def test_clock_parsing():
    assert parse_clock("00:00:04,500") == pytest.approx(4.5)
    assert parse_clock("01:02:03") == pytest.approx(3723)
    assert parse_clock("2:15") == pytest.approx(135)
    with pytest.raises(TranscriptImportError):
        parse_clock("not-a-time")


@pytest.mark.parametrize(
    "text,filename,expected",
    [
        (SRT, "captions.srt", "srt"),
        (VTT, "captions.vtt", "vtt"),
        (TIMESTAMPED, "notes.txt", "timestamped"),
        (PLAIN, "notes.txt", "plain"),
        (MARKDOWN, "notes.md", "plain"),
    ],
)
def test_format_detection(text, filename, expected):
    assert detect_format(text, filename) == expected


def test_srt_keeps_real_timings():
    transcript, described = import_transcript(SRT, "lecture.srt")
    assert "SRT" in described
    assert len(transcript.segments) == 3
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[2].start == pytest.approx(90.0)
    assert transcript.segments[2].timestamp == "1:30"
    assert transcript.duration == pytest.approx(95.25)


def test_vtt_strips_speaker_tags():
    transcript, described = import_transcript(VTT, "lecture.vtt")
    assert "VTT" in described
    assert "<v" not in transcript.text
    assert transcript.text.startswith("Welcome to operations management.")


def test_timestamped_text_joins_continuation_lines():
    transcript, described = import_transcript(TIMESTAMPED, "lecture.txt")
    assert described == "timestamped text"
    assert len(transcript.segments) == 3
    assert transcript.segments[1].start == pytest.approx(30.0)
    assert "continues the same thought" in transcript.segments[1].text
    assert transcript.segments[2].start == pytest.approx(135.0)


def test_plain_text_gets_estimated_timings():
    transcript, described = import_transcript(PLAIN, "lecture.txt")
    assert "estimated" in described
    assert len(transcript.segments) > 1
    assert transcript.segments[0].start == 0.0
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert transcript.duration > 0


def test_markdown_formatting_is_stripped():
    transcript, _ = import_transcript(MARKDOWN, "lecture.md")
    text = transcript.text
    assert "**" not in text and "##" not in text
    assert "http://example.com" not in text
    assert "level strategy" in text  # link text survives


def test_round_trip_through_our_own_srt_export():
    """A transcript exported by this app must import back cleanly."""
    from src.exporters.transcript_formats import export_srt

    original = Transcript(
        segments=[
            Segment(index=i, start=i * 30.0, end=(i + 1) * 30.0, text=f"Segment {i} text.")
            for i in range(5)
        ],
        duration=150.0,
    )
    reimported, _ = import_transcript(export_srt(original).decode("utf-8"), "x.srt")
    assert len(reimported.segments) == len(original.segments)
    assert reimported.segments[3].start == pytest.approx(90.0)
    assert reimported.text == original.text


def test_empty_and_unusable_input_is_rejected():
    for bad in ("", "   \n\n  "):
        with pytest.raises(TranscriptImportError):
            import_transcript(bad, "x.txt")


def test_imported_transcript_chunks_like_any_other():
    transcript, _ = import_transcript(SRT * 40, "long.srt")
    chunks = chunk_transcript(transcript, chunk_seconds=30, overlap_seconds=0)
    assert chunks and all(c.text for c in chunks)


# --------------------------------------------------------------------------- #
# Question count
# --------------------------------------------------------------------------- #


_SUBJECTS = [
    "capacity utilization", "takt time", "bottleneck identification",
    "safety stock", "reorder points", "queueing discipline",
    "workforce leveling", "overtime premiums", "backorder penalties",
    "forecast error", "inventory turnover", "throughput accounting",
    "setup reduction", "line balancing", "demand smoothing",
    "supplier lead times", "quality inspection", "process yield",
    "cycle counting", "master scheduling", "rough-cut planning",
    "kanban sizing", "batch economics", "labour productivity",
    "seasonal indexing", "chase scheduling", "level scheduling",
    "hiring costs", "subcontracting decisions", "distribution routing",
]


def _distinct_stem(counter: int) -> str:
    """Stems with genuinely different content words.

    Numbering alone would not do: the duplicate detector strips digits and short
    words, so "question 1" and "question 2" are identical to it — as they should
    be, since a real model does not distinguish questions by serial number.
    """
    subject = _SUBJECTS[(counter - 1) % len(_SUBJECTS)]
    round_marker = "revised " if counter > len(_SUBJECTS) else ""
    return f"Which statement about {round_marker}{subject} follows from the lecture?"


class StubLLM:
    """A model that returns ``yield_ratio`` of what it is asked for.

    Real models under-deliver, which is the root of the reported bug; this makes
    that behavior reproducible.
    """

    def __init__(self, yield_ratio: float = 1.0, drop_in_review: int = 0, malformed: int = 0):
        self.yield_ratio = yield_ratio
        self.drop_in_review = drop_in_review
        self.malformed = malformed
        self.counter = 0
        self.generate_calls = 0

    def complete_json(self, system, user, max_tokens=None):
        if "Review the following" in user:
            ids = [q["id"] for q in json.loads(user.split("DRAFT QUESTIONS:")[1].strip())]
            reviews = []
            for i, qid in enumerate(ids):
                verdict = "drop" if i < self.drop_in_review else "keep"
                reviews.append({"id": qid, "verdict": verdict, "issues": ["weak"]})
            self.drop_in_review = 0  # only the first review drops
            return {"reviews": reviews}

        if "multiple-choice questions" not in user:
            return {"heading": "S", "key_points": ["p"], "key_terms": [], "notable_quote": "q"}

        self.generate_calls += 1
        asked = int(user.split("exactly ")[1].split(" ")[0])
        produce = max(0, int(round(asked * self.yield_ratio)))
        out = []
        for _ in range(produce):
            self.counter += 1
            if self.malformed > 0:
                self.malformed -= 1
                out.append({"stem": "broken", "options": ["only-one"]})  # unusable
                continue
            out.append(
                {
                    "stem": _distinct_stem(self.counter),
                    "options": ["Alpha", "Beta", "Gamma", "Delta"],
                    "correct_index": 0,
                    "rationale": "Because alpha.",
                    "distractor_rationales": ["", "b", "c", "d"],
                    "bloom": "Apply",
                    "difficulty": "Medium",
                    "topic": f"Topic {self.counter}",
                    "source_timestamp": "5:00",
                    "source_quote": "alpha is correct",
                }
            )
        return {"questions": out}


@pytest.fixture
def chunks():
    segments = [
        Segment(index=i, start=i * 60.0, end=(i + 1) * 60.0, text=f"Content {i}. " * 25)
        for i in range(30)
    ]
    return chunk_transcript(Transcript(segments=segments, duration=1800.0), chunk_seconds=600)


def included(questions):
    return [q for q in questions if q.include]


def test_exact_count_when_the_model_cooperates(chunks):
    questions, notes = generate_question_set(
        StubLLM(1.0), chunks, target=12, do_review=False
    )
    assert len(included(questions)) == 12


def test_shortfall_is_topped_up(chunks):
    """The reported bug: ask for 12, a model that returns two-thirds gives 8."""
    llm = StubLLM(yield_ratio=0.67)
    questions, notes = generate_question_set(llm, chunks, target=12, do_review=False)

    assert len(included(questions)) == 12, "top-up rounds must close the gap"
    assert llm.generate_calls > len(chunks), "a second round should have run"


def test_reviewer_drops_are_replaced(chunks):
    llm = StubLLM(yield_ratio=1.0, drop_in_review=4)
    questions, notes = generate_question_set(llm, chunks, target=10, do_review=True)

    assert len(included(questions)) == 10
    assert any(not q.include for q in questions), "dropped drafts stay visible"
    assert any("rejected" in n for n in notes)


def test_malformed_items_do_not_reduce_the_count(chunks):
    questions, _ = generate_question_set(
        StubLLM(1.0, malformed=3), chunks, target=10, do_review=False
    )
    assert len(included(questions)) == 10


def test_surplus_is_trimmed_to_the_request_but_kept_in_the_bank(chunks):
    questions, notes = generate_question_set(
        StubLLM(yield_ratio=2.0), chunks, target=6, do_review=False
    )
    assert len(included(questions)) == 6
    assert len(questions) > 6, "extras stay available"
    assert any("extra" in n.lower() for n in notes)
    assert all(
        "Extra — beyond the requested count" in q.flags
        for q in questions
        if not q.include
    )


def test_an_impossible_request_reports_instead_of_pretending(chunks):
    llm = StubLLM(yield_ratio=0.0)
    questions, notes = generate_question_set(llm, chunks, target=10, do_review=False)
    assert included(questions) == []
    assert any("Stopped topping up" in n or "survived" in n for n in notes)


def test_top_up_rounds_are_bounded(chunks):
    llm = StubLLM(yield_ratio=0.2)
    generate_question_set(llm, chunks, target=20, do_review=False, max_rounds=2)
    assert llm.generate_calls <= len(chunks) * 2 + 1, "must not retry forever"


def test_duplicate_stems_across_rounds_are_dropped(chunks):
    class Repeater(StubLLM):
        def complete_json(self, system, user, max_tokens=None):
            payload = super().complete_json(system, user, max_tokens)
            for q in payload.get("questions", []):
                q["stem"] = "The one and only question about planning strategy?"
            return payload

    questions, _ = generate_question_set(Repeater(1.0), chunks, target=8, do_review=False)
    assert len(questions) == 1, "near-identical stems collapse to one"


def test_answer_positions_are_still_balanced(chunks):
    questions, _ = generate_question_set(StubLLM(1.0), chunks, target=12, do_review=False)
    letters = {q.answer_letter for q in included(questions)}
    assert len(letters) > 1, "the stub always answers A; balancing must spread them"


def test_zero_target_and_no_chunks_are_handled(chunks):
    assert generate_question_set(StubLLM(), chunks, target=0) == ([], [])
    assert generate_question_set(StubLLM(), [], target=5) == ([], [])


def test_topup_allocation_rotates_across_chunks():
    chunk_list = [object()] * 4
    first = _allocate_topup(2, chunk_list, offset=0)
    later = _allocate_topup(2, chunk_list, offset=2)
    assert sum(first) == sum(later) == 2
    assert first != later, "rotating avoids hammering the opening minutes"
