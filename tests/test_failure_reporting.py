"""Tests for the failure paths that used to be silent.

The reported bug: summarize-and-generate would "finish" without completing, and
the app said nothing useful. The cause was three layers of `except: continue`
writing their only evidence into a progress bar that was cleared moments later.

These tests assert the inverse property throughout — **a failed call must leave
evidence that outlives the progress bar** — for each realistic cause: a truncated
reply, a rate limit, an exhausted key, a rejected key, and a model that returns
prose instead of JSON.
"""

from __future__ import annotations

import json

import pytest

from src.chunking import chunk_transcript
from src.diagnostics import (
    CAUSE_AUTH,
    CAUSE_BAD_JSON,
    CAUSE_CREDIT,
    CAUSE_RATE_LIMIT,
    CAUSE_TRUNCATED,
    FAILED,
    OK,
    PHASE_GENERATE,
    PHASE_REVIEW,
    PHASE_SUMMARY,
    RunReport,
    classify,
    short_reason,
)
from src.llm import LLMError, TruncatedResponseError, _looks_truncated, parse_json_object
from src.mcq import critique_and_revise, generate_question_set, generate_questions
from src.schema import Segment, Transcript
from src.summarize import summarize_transcript


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def transcript() -> Transcript:
    return Transcript(
        segments=[
            Segment(index=i, start=i * 60.0, end=(i + 1) * 60.0,
                    text=f"Operations content {i}. " * 25)
            for i in range(30)
        ],
        duration=1800.0,
    )


@pytest.fixture
def chunks(transcript):
    return chunk_transcript(transcript, chunk_seconds=600)


SUBJECTS = [
    "capacity utilization", "takt time", "safety stock", "reorder points",
    "line balancing", "forecast error", "setup reduction", "batch economics",
    "queue discipline", "hiring cost", "overtime premium", "yield loss",
]


def question_payload(n: int, offset: int = 0) -> dict:
    return {
        "questions": [
            {
                "stem": f"Which claim about {SUBJECTS[(offset + i) % len(SUBJECTS)]} holds?",
                "options": ["Alpha", "Beta", "Gamma", "Delta"],
                "correct_index": 0,
                "rationale": "Because alpha.",
                "distractor_rationales": ["", "b", "c", "d"],
                "bloom": "Apply",
                "difficulty": "Medium",
                "topic": SUBJECTS[(offset + i) % len(SUBJECTS)],
                "source_timestamp": "5:00",
                "source_quote": "alpha is correct",
            }
            for i in range(n)
        ]
    }


class FailingLLM:
    """Raises ``error`` on the calls named in ``fail_on``."""

    def __init__(self, error: Exception, fail_on: str = "all"):
        self.error = error
        self.fail_on = fail_on
        self.counter = 0

    def complete_json(self, system, user, max_tokens=None):
        kind = (
            "generate" if "multiple-choice questions" in user
            else "review" if "Review the following" in user
            else "reduce" if "section summaries" in user.lower()
            else "summary"
        )
        if self.fail_on in ("all", kind):
            raise self.error
        if kind == "generate":
            n = int(user.split("exactly ")[1].split(" ")[0])
            self.counter += n
            return question_payload(n, self.counter)
        if kind == "reduce":
            return {"title": "T", "abstract": "A", "learning_objectives": [],
                    "key_points": [], "key_terms": [], "outline": []}
        if kind == "review":
            ids = [q["id"] for q in json.loads(user.split("DRAFT QUESTIONS:")[1].strip())]
            return {"reviews": [{"id": i, "verdict": "keep", "issues": []} for i in ids]}
        return {"heading": "S", "key_points": ["p"], "key_terms": [], "notable_quote": "q"}


# --------------------------------------------------------------------------- #
# Truncation detection
# --------------------------------------------------------------------------- #


def test_unbalanced_json_is_truncation_not_nonsense():
    assert not _looks_truncated('{"a": 1}')
    assert _looks_truncated('{"questions": [{"stem": "why')
    assert not _looks_truncated('{"a": "a { brace in a string"}')
    assert _looks_truncated('{"a": "unterminated string')


def test_a_cut_off_reply_raises_the_specific_error():
    with pytest.raises(TruncatedResponseError) as exc:
        parse_json_object('{"questions": [{"stem": "Which claim abo')
    assert "cut off" in str(exc.value)
    assert "fewer questions" in str(exc.value)


def test_prose_is_not_mistaken_for_truncation():
    with pytest.raises(LLMError) as exc:
        parse_json_object("I'm sorry, I can't help with that.")
    assert not isinstance(exc.value, TruncatedResponseError)


def test_truncation_is_not_retried_as_transient():
    """Retrying an over-long request unchanged just bills for the same failure."""
    from src.llm import TransientLLMError

    assert not issubclass(TruncatedResponseError, TransientLLMError)


# --------------------------------------------------------------------------- #
# Cause classification and advice
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 429 - rate limit exceeded", CAUSE_RATE_LIMIT),
        ("402 Payment Required: insufficient credits", CAUSE_CREDIT),
        ("401 Unauthorized: invalid api key", CAUSE_AUTH),
        ("The reply was cut off before the JSON closed", CAUSE_TRUNCATED),
        ("Model returned malformed JSON: Expecting value", CAUSE_BAD_JSON),
    ],
)
def test_provider_errors_are_classified(message, expected):
    assert classify(RuntimeError(message)) == expected


def test_every_classified_cause_has_advice():
    report = RunReport()
    for cause in (CAUSE_TRUNCATED, CAUSE_RATE_LIMIT, CAUSE_CREDIT, CAUSE_AUTH, CAUSE_BAD_JSON):
        report.steps.clear()
        report.record(PHASE_GENERATE, "x", FAILED, cause=cause)
        assert report.advice(), f"no advice for {cause}"


def test_reasons_are_readable_not_stack_traces():
    reason = short_reason(RuntimeError("x" * 500))
    assert len(reason) <= 201 and reason.endswith("…")


def test_report_headline_counts_failures():
    report = RunReport()
    report.record(PHASE_GENERATE, "0:00–10:00", OK, produced=3)
    report.record(PHASE_GENERATE, "10:00–20:00", FAILED, detail="429")
    assert report.headline() == "1 of 2 model calls failed."
    assert not report.phase_failed_entirely(PHASE_GENERATE)


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #


def test_a_single_bad_chunk_is_recorded_but_not_fatal(transcript):
    class OneBad(FailingLLM):
        """Fails the second chunk only. Counted rather than matched on a label,
        because chunk windows overlap and their labels are not round numbers."""

        seen = 0

        def complete_json(self, system, user, max_tokens=None):
            if "Below is one segment" in user:
                OneBad.seen += 1
                if OneBad.seen == 2:
                    raise RuntimeError("429 rate limit exceeded")
            return FailingLLM.complete_json(self, system, user, max_tokens)

    OneBad.seen = 0

    report = RunReport()
    summary, chunks, _ = summarize_transcript(
        OneBad(RuntimeError("unused"), fail_on="none"), transcript, report=report
    )
    assert summary is not None
    assert len(report.failures) == 1
    assert report.failures[0].cause == CAUSE_RATE_LIMIT


def test_when_every_summary_call_fails_the_run_stops(transcript):
    """It used to carry on and generate questions from an empty summary."""
    report = RunReport()
    with pytest.raises(LLMError) as exc:
        summarize_transcript(
            FailingLLM(RuntimeError("401 invalid api key")), transcript, report=report
        )
    assert "Every one of the" in str(exc.value)
    assert "invalid api key" in str(exc.value)
    assert report.phase_failed_entirely(PHASE_SUMMARY)
    assert report.dominant_cause() == CAUSE_AUTH


def test_a_failed_synthesis_is_recorded_and_then_worked_around(transcript):
    """Synthesis is one call over material that has already been paid for.

    It used to take every section summary down with it, which is the expensive
    way to fail: the sections were bought and then thrown away. Now the failure
    is still recorded — the instructor has to be told the good summary was not
    written — but the sections are collated locally rather than discarded.
    """
    report = RunReport()
    summary, _, sections = summarize_transcript(
        FailingLLM(RuntimeError("500 server error"), fail_on="reduce"),
        transcript, report=report,
    )

    synthesis = [s for s in report.phase_steps(PHASE_SUMMARY) if "synthesis" in s.label]
    assert synthesis and synthesis[0].is_failure, "the failure must not be hidden"

    assert summary.key_points, "the paid-for sections must survive the failed call"
    assert sections, "and be handed back, so a re-run does not buy them again"
    assert any(
        "assembled locally" in s.label for s in report.phase_steps(PHASE_SUMMARY)
    ), "this summary is plainer than a synthesized one; the report should say so"


def test_a_failed_synthesis_with_nothing_to_salvage_still_raises(transcript):
    """The fallback is only a fallback when there is something to fall back on.

    Sections that came back empty give an empty local summary, and returning
    that silently is the original bug — a run that "finished" with nothing in it.
    """

    class EmptySections(FailingLLM):
        def complete_json(self, system, user, max_tokens=None):
            if "Below is one segment" in user:
                return {"heading": "", "key_points": [], "key_terms": []}
            return FailingLLM.complete_json(self, system, user, max_tokens)

    report = RunReport()
    with pytest.raises(Exception):
        summarize_transcript(
            EmptySections(RuntimeError("500 server error"), fail_on="reduce"),
            transcript, report=report,
        )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_a_failed_chunk_leaves_evidence(chunks):
    """The core bug: this used to vanish into a cleared progress bar."""
    report = RunReport()
    generate_questions(
        FailingLLM(RuntimeError("429 rate limit exceeded")),
        chunks, [2] * len(chunks), report=report,
    )
    assert len(report.failures) == len(chunks)
    assert all(f.detail for f in report.failures), "every failure must carry a reason"
    assert report.dominant_cause() == CAUSE_RATE_LIMIT
    assert report.advice()


def test_total_generation_failure_raises_instead_of_returning_nothing(chunks):
    report = RunReport()
    with pytest.raises(LLMError) as exc:
        generate_question_set(
            FailingLLM(RuntimeError("402 insufficient credits")),
            chunks, target=10, do_review=False, report=report,
        )
    assert "Every question request failed" in str(exc.value)
    assert report.dominant_cause() == CAUSE_CREDIT


def test_a_thin_lecture_is_reported_differently_from_a_broken_api(chunks):
    """Two very different problems that used to share one message."""
    class NothingUsable(FailingLLM):
        def complete_json(self, system, user, max_tokens=None):
            if "multiple-choice questions" in user:
                return {"questions": []}
            return FailingLLM.complete_json(self, system, user, max_tokens)

    report = RunReport()
    questions, notes = generate_question_set(
        NothingUsable(RuntimeError("unused"), fail_on="none"),
        chunks, target=10, do_review=False, report=report,
    )
    assert questions == []
    assert not report.failures, "an empty answer is not a failed call"
    assert any("nothing new for this material" in n for n in notes)


def test_empty_chunks_are_recorded_as_empty_not_ok(chunks):
    class NothingUsable(FailingLLM):
        def complete_json(self, system, user, max_tokens=None):
            if "multiple-choice questions" in user:
                return {"questions": []}
            return FailingLLM.complete_json(self, system, user, max_tokens)

    report = RunReport()
    generate_questions(
        NothingUsable(RuntimeError("unused"), fail_on="none"),
        chunks, [2] * len(chunks), report=report,
    )
    statuses = {s.status for s in report.phase_steps(PHASE_GENERATE)}
    assert statuses == {"empty"}
    assert not report.failures


# --------------------------------------------------------------------------- #
# Truncation recovery
# --------------------------------------------------------------------------- #


class TruncatesLargeAsks:
    """Refuses batches above ``ceiling`` the way a real output limit does."""

    def __init__(self, ceiling: int = 2):
        self.ceiling = ceiling
        self.asks: list[int] = []
        self.counter = 0

    def complete_json(self, system, user, max_tokens=None):
        if "multiple-choice questions" not in user:
            return {"heading": "S", "key_points": ["p"], "key_terms": [], "notable_quote": "q"}
        n = int(user.split("exactly ")[1].split(" ")[0])
        self.asks.append(n)
        if n > self.ceiling:
            raise TruncatedResponseError("stopped at its output limit")
        self.counter += n
        return question_payload(n, self.counter)


def test_a_truncated_batch_is_retried_smaller_instead_of_lost(chunks):
    """Half a chunk's questions beats none, and the top-up rounds cover the rest."""
    llm = TruncatesLargeAsks(ceiling=2)
    report = RunReport()
    questions = generate_questions(llm, chunks, [4] * len(chunks), report=report)

    assert questions, "the retry should salvage the chunk"
    assert 4 in llm.asks and 2 in llm.asks, "it should ask again, smaller"
    retries = [s for s in report.steps if s.status == "skipped"]
    assert retries and "cut off" in retries[0].detail
    assert not report.failures, "a successful retry is not a failure"


def test_truncation_that_persists_is_reported(chunks):
    llm = TruncatesLargeAsks(ceiling=0)  # even one question is too much
    report = RunReport()
    questions = generate_questions(llm, chunks, [2] * len(chunks), report=report)

    assert questions == []
    assert report.failures
    assert report.dominant_cause() == CAUSE_TRUNCATED
    assert "fewer questions" in report.advice()


# --------------------------------------------------------------------------- #
# Review
# --------------------------------------------------------------------------- #


def test_a_failed_review_keeps_the_drafts_and_says_so(chunks):
    good = FailingLLM(RuntimeError("unused"), fail_on="none")
    drafts = generate_questions(good, chunks, [2] * len(chunks))
    assert drafts

    report = RunReport()
    kept, reviews = critique_and_revise(
        FailingLLM(RuntimeError("429 rate limit exceeded")), drafts, report=report
    )
    assert len(kept) == len(drafts), "a failed review must not lose the drafts"
    assert reviews == []
    failures = [s for s in report.phase_steps(PHASE_REVIEW) if s.is_failure]
    assert failures and failures[0].cause == CAUSE_RATE_LIMIT


def test_a_successful_review_is_recorded_too(chunks):
    good = FailingLLM(RuntimeError("unused"), fail_on="none")
    drafts = generate_questions(good, chunks, [2] * len(chunks))
    report = RunReport()
    critique_and_revise(good, drafts, report=report)
    assert any(s.status == OK for s in report.phase_steps(PHASE_REVIEW))


# --------------------------------------------------------------------------- #
# The report as the UI sees it
# --------------------------------------------------------------------------- #


def test_report_rows_are_renderable(chunks):
    report = RunReport()
    generate_questions(
        FailingLLM(RuntimeError("429 rate limit")), chunks, [1] * len(chunks), report=report
    )
    rows = report.as_rows()
    assert rows and set(rows[0]) == {"Step", "Part", "Result", "Produced", "Detail"}
    assert all(row["Result"] == "✗" for row in rows)


def test_a_clean_run_reports_success(chunks):
    report = RunReport()
    generate_question_set(
        FailingLLM(RuntimeError("unused"), fail_on="none"),
        chunks, target=6, do_review=False, report=report,
    )
    assert not report.failures
    assert "succeeded" in report.headline()
    assert report.advice() == ""
