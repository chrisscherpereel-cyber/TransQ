"""Generating more than one alternative set, and getting different questions.

Two reports, one root cause. "It does not appear that more than 2 question sets
can be created", and "a different model can be used to create a new question
set". There was never a cap on the number of sets — but ``run_alternative_set``
passed no ``avoid_stems``, so every alternative pass was written in ignorance of
the sets before it and reproduced them. A third set that repeats the first two
is indistinguishable from a third set that failed to appear, and a set from a
different model that repeats the first is indistinguishable from a model switch
that did nothing.

So the property under test is not "does a third set exist" but "is a third set
*different*" — and, because that fix could easily under-deliver instead, that
the count still comes out exact.
"""

from __future__ import annotations

import pytest

from src.mcq import _drop_duplicates, generate_question_set
from src.schema import MCQ, Chunk, Summary

# --------------------------------------------------------------------------- #
# A stub that behaves like a model: it writes about whatever it is not told to
# avoid, and repeats itself when it is told nothing.
# --------------------------------------------------------------------------- #

# Enough genuinely distinct material that running out is a deliberate test
# below, not an accident that makes every other assertion ambiguous.
#
# Distinct in *wording*, not by a trailing index: near-duplicate detection
# strips digits and short tokens, so "topic 1" and "topic 2" are correctly the
# same question to it. Varying only an index would have this file testing the
# duplicate checker instead of alternative sets.
_SUBJECTS = [
    "chase strategy hiring", "level strategy inventory", "demand forecasting",
    "capacity cushion", "planning horizon", "make to order",
    "overtime subcontracting", "bottleneck utilisation", "seasonal smoothing",
    "stockout penalties", "workforce flexibility", "backlog carrying",
]
_ASPECTS = [
    "cost behaviour", "risk exposure", "measurement difficulty",
    "customer impact", "supplier dependency",
]
TOPICS = [f"{subject} and {aspect}" for aspect in _ASPECTS for subject in _SUBJECTS]


class RepeatingLLM:
    """Writes the first topics it has not been told to avoid.

    This models the behaviour that made the bug invisible: a real model, asked
    twice about the same chunk with no avoid list, returns the same salient
    points both times. That is not a failure — it is what "most important"
    means, and it is exactly why an alternative set has to be told what already
    exists.

    One instance represents one generation run, so ``served`` is per set. Within
    a run the app already threads earlier chunks' stems into the prompt; this
    stub honours that without depending on the exact truncation of the avoid
    clause, which would make the test about prompt formatting rather than about
    whether alternative sets differ.
    """

    def __init__(self) -> None:
        self.generate_calls = 0
        self.avoid_seen: list[str] = []
        self.served: list[str] = []

    def complete_json(self, system, user, max_tokens=None):
        if "Review the following" in user:
            import json

            ids = [q["id"] for q in json.loads(user.split("DRAFT QUESTIONS:")[1].strip())]
            return {"reviews": [{"id": i, "verdict": "keep", "issues": []} for i in ids]}

        if "multiple-choice questions" not in user:
            return {"heading": "S", "key_points": ["p"], "key_terms": [], "notable_quote": "q"}

        self.generate_calls += 1
        asked = int(user.split("exactly ")[1].split(" ")[0])

        banned = [t for t in TOPICS if t in user]
        self.avoid_seen.extend(banned)
        available = [
            t for t in TOPICS if t not in banned and t not in self.served
        ]

        out = []
        for topic in available[:asked]:
            self.served.append(topic)
            out.append(
                {
                    "stem": f"Which claim about {topic} follows from the lecture?",
                    "options": ["Alpha", "Beta", "Gamma", "Delta"],
                    "correct_index": 0,
                    "rationale": "Because alpha.",
                    "distractor_rationales": ["", "b", "c", "d"],
                    "bloom": "Understand",
                    "difficulty": "Medium",
                    "topic": topic,
                    "source_timestamp": "5:00",
                    "source_quote": f"the lecture discussed {topic}",
                }
            )
        return {"questions": out}


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(index=i, start=i * 600.0, end=(i + 1) * 600.0,
              text=f"Operations content about planning window {i}. " * 40)
        for i in range(3)
    ]


@pytest.fixture
def summary() -> Summary:
    return Summary(
        title="Aggregate Planning",
        learning_objectives=["Explain chase and level strategies"],
        key_points=["Chase tracks demand", "Level holds capacity steady"],
    )


def make_set(chunks, summary, avoid, target=4, client=None):
    """One generation run. A fresh stub each time, because a set is a run."""
    questions, _ = generate_question_set(
        client or RepeatingLLM(), chunks, target=target, summary=summary,
        avoid_stems=avoid, do_review=False,
    )
    return questions


# --------------------------------------------------------------------------- #
# There is no ceiling — and never was
# --------------------------------------------------------------------------- #


def test_a_third_and_fourth_set_are_produced(chunks, summary):
    stems: list[str] = []
    sets = []
    for _ in range(4):
        produced = make_set(chunks, summary, stems)
        sets.append(produced)
        stems.extend(q.stem for q in produced)

    assert all(s for s in sets), "every set must contain questions"
    assert len(sets) == 4


def test_each_new_set_is_actually_different(chunks, summary):
    """The real complaint. A third set identical to the first two reads as a
    third set that could not be created."""
    stems: list[str] = []
    for _ in range(3):
        produced = make_set(chunks, summary, stems)
        new = {q.stem for q in produced}
        assert not (new & set(stems)), "an alternative set repeated an earlier one"
        stems.extend(new)

    assert len(set(stems)) == len(stems), "no duplicates across the whole lecture"


def test_the_old_behaviour_repeats_itself(chunks, summary):
    """A witness for the bug: with no avoid list — what the app used to pass —
    the second set is the first set again."""
    first = make_set(chunks, summary, avoid=[])
    second = make_set(chunks, summary, avoid=[])

    assert {q.stem for q in first} == {q.stem for q in second}


def test_the_model_is_told_what_to_avoid(chunks, summary):
    first = make_set(chunks, summary, avoid=[])
    watcher = RepeatingLLM()
    make_set(chunks, summary, avoid=[q.stem for q in first], client=watcher)

    assert watcher.avoid_seen, "the avoid list never reached the prompt"


# --------------------------------------------------------------------------- #
# The fix must not under-deliver
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", [3, 5, 8])
def test_an_alternative_set_still_hits_the_requested_count(chunks, summary, target):
    """Dropping cross-set duplicates creates a shortfall; the top-up rounds have
    to make it good, or "different" would be bought with "fewer"."""
    first = make_set(chunks, summary, avoid=[], target=target)
    second = make_set(
        chunks, summary, avoid=[q.stem for q in first], target=target
    )

    assert len([q for q in first if q.include]) == target
    assert len([q for q in second if q.include]) == target


def test_a_lecture_that_runs_out_of_material_stops_rather_than_repeating(
    chunks, summary
):
    """The material is finite. Once it is used up the app must hand back what it
    has — never loop, never invent, and never quietly repeat an earlier set to
    make the number."""
    stems: list[str] = []
    for _ in range(20):
        produced, notes = generate_question_set(
            RepeatingLLM(), chunks, target=6, summary=summary,
            avoid_stems=stems, do_review=False,
        )
        stems.extend(q.stem for q in produced)

    assert len(set(stems)) == len(stems), "exhaustion must never cause a repeat"
    assert len(stems) <= len(TOPICS), "nothing beyond the available material"


# --------------------------------------------------------------------------- #
# Cross-set duplicate detection
# --------------------------------------------------------------------------- #


def item(stem: str) -> MCQ:
    return MCQ(
        stem=stem, options=["A", "B", "C", "D"], correct_index=0,
        rationale="r", source_timestamp="1:00", source_quote="q",
    )


def test_dedup_can_measure_against_earlier_sets():
    earlier = ["Which claim about chase strategy hiring costs follows from the lecture?"]
    fresh = [
        item("Which claim about chase strategy hiring costs follows from the lecture?"),
        item("Which claim about seasonal demand smoothing follows from the lecture?"),
    ]
    kept = _drop_duplicates(fresh, earlier)
    assert len(kept) == 1
    assert "seasonal" in kept[0].stem


def test_dedup_without_earlier_sets_is_unchanged():
    fresh = [item("Alpha question here about planning"), item("Beta question here about costs")]
    assert len(_drop_duplicates(fresh)) == 2
    assert len(_drop_duplicates(fresh, [])) == 2
