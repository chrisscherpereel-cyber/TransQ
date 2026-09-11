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


def test_a_thin_window_is_quieted_and_an_empty_one_is_dropped(lecture, summary):
    """This used to assert that no window could ever be silenced, on the grounds
    that the summary is a model's opinion and opinions are sometimes wrong.

    That was too cautious, and it caused the thing it meant to prevent: with a
    floor under every window, ten questions over seven windows put roughly one in
    each — the stopwatch allocation that importance weighting exists to replace,
    and the instructor's complaint verbatim.

    The distinction that holds is between *thin* and *empty*. A window that
    scored a little is a judgement call and keeps its floor. A window that scored
    zero matched nothing at all — no objective, no key point, no term, no exam
    topic — and against a summary drawn from these same windows, that is a signal
    rather than a near miss. The case where the scoring itself has failed is
    covered by the next test, not by refusing to ever act on a score.
    """
    counts = allocate_by_importance(20, lecture, summary)
    scores = chunk_importance(lecture, summary)

    for count, score in zip(counts, scores):
        if score > 0:
            assert count >= 1, "a window that scored anything keeps its floor"
        else:
            assert count == 0, "a window that scored nothing is not worth examining"
    assert sum(counts) == 20
    assert FLOOR_WEIGHT > 0


def test_the_scores_are_distrusted_when_they_stop_discriminating():
    """A thin or oddly worded summary can leave nearly every window with no
    keyword overlap — which looks identical to a lecture with one good passage,
    and is not. A single survivor is a failure of the scoring, not a verdict, so
    the floor returns for everyone rather than the quiz collapsing onto one
    window."""
    from src.chunking import MIN_SCORING_WINDOWS

    windows = [
        Chunk(index=0, start=0.0, end=600.0, text="chase strategy hiring firing " * 40),
        Chunk(index=1, start=600.0, end=1200.0, text="unrelated words entirely " * 40),
        Chunk(index=2, start=1200.0, end=1800.0, text="different words again " * 40),
    ]
    summary = Summary(learning_objectives=["Chase strategy hiring firing"])
    scoring = sum(1 for s in chunk_importance(windows, summary) if s > 0)
    assert scoring < MIN_SCORING_WINDOWS, "the fixture must be degenerate"

    counts = allocate_by_importance(9, windows, summary)
    assert all(c >= 1 for c in counts), "one survivor means the scores are noise"
    assert sum(counts) == 9


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


# --------------------------------------------------------------------------- #
# An empty window earns nothing
# --------------------------------------------------------------------------- #
#
# Reported after importance weighting shipped: "it still appears that questions
# are written for each section of the transcript." It was — the floor weight,
# meant to keep a *quiet* window in the running, was also lifting windows that
# scored literally zero. Five minutes of "the midterm is next week, office hours
# moved to Thursday" matched nothing in the summary because there is nothing in
# it to examine, and it still collected a question.


def empty_and_dense_windows():
    """A lecture with real content in the middle and none at either end."""
    from src.schema import Chunk, Summary

    admin = "Reminder the midterm is next week. Office hours moved to Thursday. "
    content = (
        "Chase strategy tracks demand through hiring and firing each period. "
        "Level strategy holds capacity steady and absorbs variation with inventory. "
    )
    goodbye = "Questions? Right. Good question. Anything else? See you Thursday. "

    chunks = [
        Chunk(index=0, start=0.0, end=600.0, text=admin * 30),
        Chunk(index=1, start=600.0, end=1200.0, text=content * 30),
        Chunk(index=2, start=1200.0, end=1800.0, text=content * 30),
        Chunk(index=3, start=1800.0, end=2400.0, text=goodbye * 30),
    ]
    summary = Summary(
        title="Aggregate planning",
        abstract="Chase versus level strategies.",
        learning_objectives=["Distinguish chase from level aggregate planning"],
        key_points=[
            "Chase tracks demand through hiring and firing",
            "Level holds capacity steady and absorbs variation with inventory",
        ],
        key_terms=[],
        outline=[],
    )
    return chunks, summary


def test_a_window_with_nothing_to_examine_gets_nothing():
    from src.chunking import allocate_by_importance, chunk_importance

    chunks, summary = empty_and_dense_windows()
    scores = chunk_importance(chunks, summary, [])
    assert scores[0] == 0 and scores[3] == 0, "the fixture must have empty windows"

    counts = allocate_by_importance(10, chunks, summary, [])
    assert counts[0] == 0 and counts[3] == 0
    assert sum(counts) == 10, "the count is still a promise"


def test_the_leftovers_do_not_leak_back_into_empty_windows():
    """Largest-remainder distributes what rounding left over. Handing that to an
    excluded window would reproduce the same one-per-window result by another
    route."""
    from src.chunking import allocate_by_importance

    chunks, summary = empty_and_dense_windows()
    for target in range(3, 31):
        counts = allocate_by_importance(target, chunks, summary, [])
        assert sum(counts) == target
        assert counts[0] == 0 and counts[3] == 0, f"leaked at {target}"


def test_a_quiet_window_is_still_kept_in_the_running():
    """The floor exists for a reason. Only a *zero* is excluded — a window that
    scored a little still competes, or the quiz would cover only the peaks."""
    from src.chunking import allocate_by_importance, chunk_importance
    from src.schema import Chunk, Summary

    strong = "Chase strategy tracks demand through hiring and firing. "
    faint = "Inventory sits between the two, roughly speaking, as a buffer. "
    chunks = [
        Chunk(index=0, start=0.0, end=600.0, text=strong * 30),
        Chunk(index=1, start=600.0, end=1200.0, text=strong * 30),
        Chunk(index=2, start=1200.0, end=1800.0, text=faint * 30),
    ]
    summary = Summary(
        title="t", abstract="a",
        learning_objectives=["Chase strategy tracks demand through hiring"],
        key_points=["Inventory is a buffer"], key_terms=[], outline=[],
    )
    scores = chunk_importance(chunks, summary, [])
    assert scores[2] > 0, "the fixture must have a quiet-but-not-empty window"
    assert allocate_by_importance(12, chunks, summary, [])[2] > 0


def test_an_even_lecture_is_unaffected():
    """Where every window scores, this change does nothing at all."""
    from src.chunking import allocate_by_importance
    from src.schema import Chunk, Summary

    text = "Chase strategy tracks demand through hiring and firing each period. "
    chunks = [
        Chunk(index=i, start=i * 600.0, end=(i + 1) * 600.0, text=text * 30)
        for i in range(4)
    ]
    summary = Summary(
        title="t", abstract="a",
        learning_objectives=["Chase strategy tracks demand"],
        key_points=["Hiring and firing each period"], key_terms=[], outline=[],
    )
    counts = allocate_by_importance(8, chunks, summary, [])
    assert all(c > 0 for c in counts) and sum(counts) == 8


def test_with_no_summary_the_clock_is_still_the_only_information():
    """Excluding windows requires knowing which ones do not matter. Without a
    summary there is no such knowledge, so even allocation remains correct."""
    from src.chunking import allocate_by_importance
    from src.schema import Chunk

    chunks = [
        Chunk(index=i, start=i * 600.0, end=(i + 1) * 600.0, text="words " * 200)
        for i in range(4)
    ]
    counts = allocate_by_importance(8, chunks, None, [])
    assert all(c > 0 for c in counts) and sum(counts) == 8


def test_the_progress_line_names_the_count_not_just_the_window():
    """"Writing questions for 9:30–19:30", repeated, reads as one question per
    ten minutes even when it is not — the skipped windows are invisible."""
    from src.mcq import generate_questions
    from src.schema import Chunk

    class Silent:
        def complete_json(self, system, user, max_tokens=None):
            return {"questions": []}

    seen: list[str] = []
    chunks = [Chunk(index=0, start=0.0, end=600.0, text="content " * 300)]
    generate_questions(
        Silent(), chunks, [3], progress=lambda f, m: seen.append(m)
    )
    assert any("3 questions" in m for m in seen)
