"""Question wording, and the consistency review.

**Framing.** "According to the lecture, which of the following..." is a worse
exam item than the same question asked plainly: it cues the student that recall
is wanted, it cannot be reused on a midterm spanning six weeks, and it reads as
an artefact of how the item was made. The default is now standalone. Provenance
does not disappear — the timestamp and supporting quote are still recorded on
every item; they simply stop appearing in the stem.

**Review.** A model assessing claims against its own training is wrong often
enough that presenting this as fact-checking would be actively harmful: the first
confident wrong flag is the one that makes an instructor stop reading. So the
tests here are as much about what the feature *claims* as what it computes — the
verdict vocabulary keeps transcription errors separate from factual doubt, low
confidence sorts below high, and a slide conflict (both texts supplied, a real
observation) outranks a judgement drawn from training.
"""

from __future__ import annotations

import json

import pytest

from src import prompts
from src.config import DEFAULT_FRAMING, QUESTION_FRAMING
from src.factcheck import (
    NEEDS_ATTENTION,
    VERDICTS,
    Finding,
    ReviewResult,
    review_transcript,
)
from src.mcq import generate_questions
from src.schema import Chunk

TRANSCRIPT = (
    "Chase strategy tracks demand through hiring and firing each period. "
    "Level strategy holds capacity steady and absorbs variation with inventory. "
)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [Chunk(index=0, start=0.0, end=600.0, text=TRANSCRIPT * 20)]


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


class PromptSpy:
    """Captures the prompts rather than answering usefully."""

    def __init__(self) -> None:
        self.systems: list[str] = []
        self.users: list[str] = []

    def complete_json(self, system, user, max_tokens=None):
        self.systems.append(system)
        self.users.append(user)
        if "Review the following" in user:
            ids = [q["id"] for q in json.loads(user.split("DRAFT QUESTIONS:")[1].strip())]
            return {"reviews": [{"id": i, "verdict": "keep", "issues": []} for i in ids]}
        return {"questions": []}


def test_standalone_is_the_default():
    from src.config import AppSettings

    assert DEFAULT_FRAMING == "standalone"
    assert AppSettings().framing == "standalone"


def test_the_default_forbids_referring_to_the_lecture(chunks):
    spy = PromptSpy()
    generate_questions(spy, chunks, [3])

    system = spy.systems[0]
    assert "according to the lecture" in system.lower()
    assert "Never" in system, "the rule has to be a prohibition, not a preference"


def test_the_lecture_framing_is_still_available(chunks):
    spy = PromptSpy()
    generate_questions(spy, chunks, [3], framing="lecture")

    system = spy.systems[0]
    assert "may refer to the lecture" in system
    assert "Never refer to the source" not in system


def test_the_scenario_framing_asks_for_a_situation(chunks):
    spy = PromptSpy()
    generate_questions(spy, chunks, [3], framing="scenario")
    assert "concrete situation" in spy.systems[0]


def test_an_unknown_framing_falls_back_rather_than_crashing(chunks):
    spy = PromptSpy()
    generate_questions(spy, chunks, [3], framing="nonsense")
    assert QUESTION_FRAMING[DEFAULT_FRAMING]["rule"] in spy.systems[0]


def test_provenance_is_still_required_under_every_framing(chunks):
    """Dropping the lecture from the stem must not drop the supporting quote —
    that is what makes these drafts reviewable at speed."""
    for framing in QUESTION_FRAMING:
        spy = PromptSpy()
        generate_questions(spy, chunks, [2], framing=framing)
        assert "verbatim" in spy.systems[0]
        assert "timestamp" in spy.systems[0]


def test_every_framing_renders_a_complete_prompt():
    for key, spec in QUESTION_FRAMING.items():
        rendered = prompts.MCQ_SYSTEM.format(n_options=4, framing_rule=spec["rule"])
        assert "{" not in rendered.replace("{{", "").replace("}}", "")
        assert spec["rule"] in rendered
        assert spec["label"] and spec["help"]


def test_the_reviewer_is_told_the_same_framing_rule(chunks):
    """Otherwise the review pass "repairs" a standalone stem back into one that
    cites the lecture, silently undoing the setting."""
    rendered = prompts.CRITIQUE_USER.format(
        framing_rule=QUESTION_FRAMING["standalone"]["rule"], questions_json="[]"
    )
    assert "Never refer to the source" in rendered


# --------------------------------------------------------------------------- #
# The review's honesty about itself
# --------------------------------------------------------------------------- #


def test_transcription_errors_are_their_own_verdict():
    """Most claims that look wrong in an automatic transcript are Whisper
    mishearing a number or a term. Without this, the feature would mostly report
    the transcriber's mistakes as the instructor's."""
    assert "transcription" in VERDICTS
    assert "transcription" not in NEEDS_ATTENTION
    assert "TRANSCRIPT" in prompts.REVIEW_SYSTEM


def test_the_prompt_refuses_the_authority_it_would_otherwise_assume():
    system = prompts.REVIEW_SYSTEM
    assert "NOT the authority" in system
    assert "cutoff" in system
    assert "simplification" in system.lower(), "teaching approximates on purpose"


def test_contested_is_kept_distinct_from_questionable():
    """A claim reflecting one legitimate school of thought is not an error."""
    assert "contested" in VERDICTS and "questionable" in VERDICTS
    assert "contested" in NEEDS_ATTENTION


def test_low_confidence_sorts_below_high_within_a_verdict():
    high = Finding(claim="a", verdict="questionable", confidence="high")
    low = Finding(claim="b", verdict="questionable", confidence="low")
    assert high.rank < low.rank


def test_serious_verdicts_outrank_mild_ones():
    assert Finding(claim="a", verdict="questionable").rank < Finding(
        claim="b", verdict="consistent"
    ).rank


# --------------------------------------------------------------------------- #
# Running a review
# --------------------------------------------------------------------------- #


class ReviewingLLM:
    def __init__(self, findings=None, fail: bool = False):
        self.findings = findings or []
        self.fail = fail
        self.calls = 0
        self.prompts: list[str] = []

    def complete_json(self, system, user, max_tokens=None):
        self.calls += 1
        self.prompts.append(user)
        if self.fail:
            raise RuntimeError("provider is down")
        return {"findings": self.findings}


def test_findings_come_back_sorted_by_seriousness(chunks):
    client = ReviewingLLM(
        [
            {"claim": "fine", "verdict": "consistent", "confidence": "high"},
            {"claim": "wrong", "verdict": "questionable", "confidence": "high"},
            {"claim": "debated", "verdict": "contested", "confidence": "high"},
        ]
    )
    result = review_transcript(client, chunks)
    assert [f.verdict for f in result.findings] == [
        "questionable",
        "contested",
        "consistent",
    ]


def test_a_clean_lecture_returns_nothing_and_says_so(chunks):
    result = review_transcript(ReviewingLLM([]), chunks)
    assert result.findings == []
    assert "Nothing flagged" in result.headline()


def test_slide_conflicts_are_separated_from_knowledge_claims(chunks):
    client = ReviewingLLM(
        [
            {"claim": "a", "verdict": "questionable", "slide_conflict": "slide 7 says 14%"},
            {"claim": "b", "verdict": "contested", "slide_conflict": ""},
        ]
    )
    result = review_transcript(client, chunks)
    assert len(result.slide_conflicts) == 1
    assert result.slide_conflicts[0].slide_conflict == "slide 7 says 14%"


def test_one_failed_window_does_not_sink_the_review():
    from src.diagnostics import RunReport

    many = [
        Chunk(index=i, start=i * 600.0, end=(i + 1) * 600.0, text=TRANSCRIPT * 20)
        for i in range(3)
    ]
    report = RunReport()
    result = review_transcript(ReviewingLLM(fail=True), many, report=report)

    assert result.findings == []
    assert result.checked_windows == 0
    assert len(report.failures) == 3, "each failure must leave a trace"


def test_a_finding_without_a_timestamp_gets_its_windows(chunks):
    client = ReviewingLLM([{"claim": "a", "verdict": "contested"}])
    result = review_transcript(client, chunks)
    assert result.findings[0].timestamp == "0:00"


def test_garbage_verdicts_degrade_to_unclear_rather_than_crashing(chunks):
    client = ReviewingLLM(
        [{"claim": "a", "verdict": "definitely-wrong", "confidence": "absolute"}]
    )
    result = review_transcript(client, chunks)
    assert result.findings[0].verdict == "unclear"
    assert result.findings[0].confidence == "low"


def test_claims_with_no_text_are_dropped(chunks):
    client = ReviewingLLM([{"claim": "", "verdict": "questionable"}, {"claim": "real"}])
    result = review_transcript(client, chunks)
    assert len(result.findings) == 1


def test_exam_topics_are_named_in_the_review_prompt(chunks):
    client = ReviewingLLM([])
    review_transcript(client, chunks, exam_topics=["Chase versus level"])
    assert "Chase versus level" in client.prompts[0]


def test_a_review_round_trips_through_storage(chunks):
    client = ReviewingLLM(
        [{"claim": "a", "verdict": "outdated", "confidence": "medium",
          "explanation": "superseded around 2019"}]
    )
    result = review_transcript(client, chunks)
    restored = ReviewResult.from_dict(result.to_dict())

    assert [f.claim for f in restored.findings] == [f.claim for f in result.findings]
    assert restored.findings[0].verdict == "outdated"
    assert restored.checked_windows == result.checked_windows


def test_a_malformed_stored_review_degrades_to_empty():
    assert ReviewResult.from_dict({}).findings == []
    assert ReviewResult.from_dict({"findings": ["nope"]}).findings == []
