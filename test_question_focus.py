"""Questions should follow the lecture's argument, not its clock.

The complaint: "avoid making one question for each ten minutes of the lecture,
but provide the best questions." That was literally what the code did —
``allocate_questions`` divided the target evenly across ten-minute windows, so a
quiz spent as many questions on the opening admin and the closing Q&A as on the
central argument.

Allocation now follows the summary's own account of what the lecture was for.
The tests below pin the properties that makes safe: the count is still exact,
no window can swallow the quiz, thin windows are quieted but never silenced,
and with no summary the old even split still applies.
"""

from __future__ import annotations

import pytest

from src.chunking import (
    FLOOR_WEIGHT,
    MAX_SHARE,
    allocate_by_importance,
    allocate_questions,
    chunk_importance,
)
from src.mcq import _allocate_topup, _build_focus_clause
from src.schema import Chunk, Summary

ADMIN = "welcome syllabus office hours attendance policy exam dates housekeeping"
CORE_A = (
    "aggregate planning chase strategy tracks demand hiring firing workforce "
    "level strategy holds capacity steady inventory absorbs variation"
)
CORE_B = (
    "comparing chase and level numerically hiring cost inventory carrying cost "
    "total cost tradeoff which strategy wins under volatile demand"
)
TANGENT = "a story from my consulting days and a long question from the back row"


def windows(*texts: str) -> list[Chunk]:
    return [
        Chunk(index=i, start=i * 600.0, end=(i + 1) * 600.0, text=t)
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def lecture() -> list[Chunk]:
    return windows(ADMIN, CORE_A, CORE_B, TANGENT)


@pytest.fixture
def summary() -> Summary:
    return Summary(
        title="Aggregate Planning",
        learning_objectives=[
            "Explain the chase and level aggregate planning strategies",
            "Compare their cost behaviour under volatile demand",
        ],
        key_points=[
            "Chase tracks demand through hiring and firing",
            "Level holds capacity steady and absorbs variation with inventory",
        ],
        outline=[
            {"timestamp": "12:00", "title": "Chase versus level"},
            {"timestamp": "22:00", "title": "Cost comparison"},
        ],
    )


# --------------------------------------------------------------------------- #
# The complaint itself
# --------------------------------------------------------------------------- #


def test_questions_go_where_the_content_is_not_where_the_minutes_are(lecture, summary):
    counts = allocate_by_importance(10, lecture, summary)
    admin, core_a, core_b, tangent = counts

    assert core_a > admin and core_a > tangent
    assert core_b > admin and core_b > tangent
    assert core_a + core_b >= 7, "the argument should carry most of the quiz"


def test_the_old_behaviour_was_a_flat_split_and_is_what_this_replaces(lecture):
    """Kept as a witness: this is what the quiz used to look like."""
    assert allocate_questions(10, len(lecture)) == [3, 3, 2, 2]

    counts = allocate_by_importance(10, lecture, Summary(
        learning_objectives=["Explain chase and level strategies"],
    ))
    assert counts != [3, 3, 2, 2]


def test_a_thin_window_is_quieted_but_never_silenced(lecture, summary):
    """The summary is a model's opinion. It may down-weight ten minutes; it must
    not be able to delete them, because it is sometimes wrong."""
    counts = allocate_by_importance(20, lecture, summary)
    assert all(c >= 1 for c in counts)
    assert FLOOR_WEIGHT > 0


# --------------------------------------------------------------------------- #
# Properties that must survive the change
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("target", [1, 3, 5, 7, 10, 13, 20, 40])
def test_the_count_is_still_exact(lecture, summary, target):
    """The count is a promise made elsewhere in the app; weighting must not
    quietly turn 10 into 9."""
    assert sum(allocate_by_importance(target, lecture, summary)) == target


def test_no_single_window_can_swallow_the_quiz(summary):
    """One dense stretch must not become the whole exam."""
    lopsided = windows(ADMIN, CORE_A + " " + CORE_B, ADMIN, ADMIN)
    counts = allocate_by_importance(20, lopsided, summary)
    assert max(counts) <= int(20 * MAX_SHARE) + 1
    assert sum(counts) == 20


def test_no_summary_falls_back_to_the_even_split(lecture):
    """The transcript-import path, and any run where summarization failed."""
    assert allocate_by_importance(10, lecture, None) == allocate_questions(10, 4)


def test_an_empty_summary_falls_back_rather_than_producing_nonsense(lecture):
    assert sum(allocate_by_importance(9, lecture, Summary())) == 9


def test_edge_cases_do_not_raise(summary):
    assert allocate_by_importance(0, windows(ADMIN), summary) == [0]
    assert allocate_by_importance(5, [], summary) == []
    assert allocate_by_importance(5, windows(CORE_A), summary) == [5]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_outline_timestamps_place_importance_exactly(lecture):
    scores = chunk_importance(
        lecture, Summary(outline=[{"timestamp": "22:30", "title": "Cost comparison"}])
    )
    assert scores.index(max(scores)) == 2, "22:30 falls in the third window"


def test_scoring_survives_an_outline_with_unusable_timestamps(lecture):
    """Models produce "early on", "n/a", or nothing at all. Prose matching is
    what keeps the feature working when they do."""
    summary = Summary(
        outline=[{"timestamp": "early on", "title": "x"}, {"title": "no timestamp"}],
        key_points=["Chase tracks demand through hiring and firing"],
    )
    scores = chunk_importance(lecture, summary)
    assert max(scores) > 0
    assert scores.index(max(scores)) in (1, 2)


def test_objectives_outrank_incidental_key_points(lecture):
    """What the instructor means to assess beats what merely came up."""
    as_objective = chunk_importance(
        lecture, Summary(learning_objectives=["Explain chase and level strategies"])
    )
    as_key_point = chunk_importance(
        lecture, Summary(key_points=["Explain chase and level strategies"])
    )
    assert max(as_objective) > max(as_key_point)


def test_hour_long_timestamps_parse(lecture):
    long_lecture = windows(ADMIN, CORE_A, CORE_B, TANGENT, ADMIN, CORE_A, CORE_B)
    scores = chunk_importance(
        long_lecture, Summary(outline=[{"timestamp": "1:02:00", "title": "late point"}])
    )
    assert scores.index(max(scores)) == 6


# --------------------------------------------------------------------------- #
# Top-up rounds
# --------------------------------------------------------------------------- #


def test_topups_are_taken_from_substantial_windows(lecture, summary):
    """Where the padding used to come from: a shortfall was spread round-robin
    over every window, so the extra questions came from the admin and the Q&A."""
    weights = chunk_importance(lecture, summary)
    counts = _allocate_topup(4, lecture, offset=0, weights=weights)

    assert sum(counts) == 4
    assert counts[1] + counts[2] == 4, "top-ups belong in the substantive windows"


def test_topups_without_weights_still_cover_every_window(lecture):
    counts = _allocate_topup(4, lecture, offset=0, weights=None)
    assert sum(counts) == 4
    assert all(c == 1 for c in counts)


def test_topup_rotation_avoids_hammering_the_same_window(lecture, summary):
    weights = chunk_importance(lecture, summary)
    first = _allocate_topup(1, lecture, offset=0, weights=weights)
    second = _allocate_topup(1, lecture, offset=1, weights=weights)
    assert first != second


# --------------------------------------------------------------------------- #
# What each call is told to aim at
# --------------------------------------------------------------------------- #


def test_the_focus_clause_names_what_the_lecture_was_for():
    clause = _build_focus_clause(
        ["Explain chase versus level", "Compare cost behaviour"]
    )
    assert "Explain chase versus level" in clause
    assert "Compare cost behaviour" in clause
    assert "fills a count" in clause, "the anti-padding instruction must survive"


def test_no_focus_points_produces_no_clause_rather_than_an_empty_heading():
    assert _build_focus_clause([]) == ""
    assert _build_focus_clause(["", "   "]) == ""


def test_the_focus_clause_stays_bounded():
    """A 40-point summary must not crowd out the transcript in the prompt."""
    clause = _build_focus_clause([f"Objective number {i}" for i in range(40)])
    assert clause.count("\n- ") <= 12
