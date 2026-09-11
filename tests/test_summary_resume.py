"""Summarizing must be resumable, because it is the expensive half of a run.

The report: "if the app fails during summarization there is no option to
continue — how do I get it to finish rather than start over." There wasn't one.
A thirty-minute lecture is five or six windows, each its own model call, and the
whole set was thrown away if any one of them ended the run: the per-window
summaries were only ever returned on success, so a failure at window five
discarded the four that had already been bought.

Three properties fix that, and each is tested here:

**Nothing finished is lost.** Windows are handed back as they complete, not at
the end, so a run that never reaches the end still leaves its work behind.

**A re-run resumes.** Given those windows again, the second attempt skips them —
no second charge for the same text — and does only what remains.

**The cache belongs to one lecture.** Window labels are timestamps, so summaries
from one lecture would match another of the same length. Reuse is keyed to the
transcript and the window geometry that produced them, because a silently wrong
summary is worse than a slow one.

And, underneath all three, the salvage: a window whose reply was cut off has
usually written its heading and several key points already. Those are kept.
"""

from __future__ import annotations

import pytest

from src.diagnostics import PHASE_SUMMARY, RunReport
from src.llm import TruncatedResponseError, salvage_object_fields
from src.schema import Segment, Transcript
from src.summarize import summarize_chunk, summarize_transcript, summary_from_sections
from src.chunking import chunk_transcript

SECTION = {"heading": "Capacity", "key_points": ["chase tracks demand"], "key_terms": []}
REDUCED = {
    "title": "Aggregate planning",
    "abstract": "A",
    "key_points": ["p"],
    "learning_objectives": [],
    "key_terms": [],
    "outline": [],
}


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


class CountingLLM:
    """Counts window calls, and can be told to die on the nth one."""

    def __init__(self, die_on: int | None = None, error: Exception | None = None):
        self.die_on = die_on
        self.error = error or RuntimeError("429 rate limit exceeded")
        self.windows = 0
        self.reduces = 0

    def complete_json(self, system, user, max_tokens=None):
        if "Below is one segment" in user:
            self.windows += 1
            if self.die_on is not None and self.windows >= self.die_on:
                raise self.error
            return dict(SECTION)
        self.reduces += 1
        return dict(REDUCED)


# --------------------------------------------------------------------------- #
# Nothing finished is lost
# --------------------------------------------------------------------------- #


def test_finished_windows_are_handed_back_as_they_land(transcript):
    """Not at the end. The run that most needs this is the one with no end."""
    kept: list[dict] = []
    client = CountingLLM(die_on=3, error=RuntimeError("401 invalid api key"))

    summarize_transcript(client, transcript, on_section=kept.append, report=RunReport())

    assert len(kept) == 2, "both windows that finished before the failure survive"
    assert all(k["_label"] for k in kept), "each must know which window it is"


def test_a_run_that_dies_completely_still_leaves_its_finished_windows(transcript):
    from src.llm import LLMError

    kept: list[dict] = []

    class DiesAfterFirst(CountingLLM):
        def complete_json(self, system, user, max_tokens=None):
            if "Below is one segment" in user and self.windows >= 1:
                self.windows += 1
                raise RuntimeError("401 invalid api key")
            return CountingLLM.complete_json(self, system, user, max_tokens)

    try:
        summarize_transcript(
            DiesAfterFirst(), transcript, on_section=kept.append, report=RunReport()
        )
    except LLMError:
        pass

    assert kept, "the window that succeeded was paid for; it must outlive the raise"


def test_a_failing_checkpoint_does_not_fail_the_window(transcript):
    """Saving is best-effort. The model call succeeded either way, and taking the
    summary down because a cache write failed would be the tail wagging the dog."""

    def explode(section: dict) -> None:
        raise RuntimeError("session state is gone")

    summary, _, sections = summarize_transcript(
        CountingLLM(), transcript, on_section=explode, report=RunReport()
    )
    assert summary.title and sections


# --------------------------------------------------------------------------- #
# A re-run resumes
# --------------------------------------------------------------------------- #


def test_cached_windows_are_not_bought_twice(transcript):
    first = CountingLLM()
    _, _, sections = summarize_transcript(first, transcript, report=RunReport())
    assert first.windows > 1, "the fixture must be long enough to matter"

    second = CountingLLM()
    report = RunReport()
    summary, _, again = summarize_transcript(
        second, transcript, cached_sections=sections, report=report
    )

    assert second.windows == 0, "every window was already summarized"
    assert second.reduces == 1, "only the synthesis is left to do"
    assert summary.title
    assert len(again) == len(sections)


def test_a_resumed_run_does_only_the_windows_that_are_missing(transcript):
    chunks = chunk_transcript(transcript, 600, 30)
    partial = [
        {"_label": chunks[0].label, "_start": chunks[0].start, **SECTION}
    ]

    client = CountingLLM()
    summarize_transcript(client, transcript, cached_sections=partial, report=RunReport())

    assert client.windows == len(chunks) - 1


def test_a_reused_window_is_reported_as_skipped_not_invented(transcript):
    """The run report is the only honest account of what a run cost. A window
    that was reused must not appear as one that was requested."""
    _, _, sections = summarize_transcript(CountingLLM(), transcript, report=RunReport())

    report = RunReport()
    summarize_transcript(
        CountingLLM(), transcript, cached_sections=sections, report=report
    )

    skipped = [s for s in report.phase_steps(PHASE_SUMMARY) if s.status == "skipped"]
    assert len(skipped) == len(sections)
    assert all("earlier attempt" in s.detail for s in skipped)


def test_a_failed_window_is_never_cached_as_done(transcript):
    """Otherwise the resume would inherit the hole and call it finished."""
    client = CountingLLM(die_on=3, error=RuntimeError("429 rate limit"))
    _, _, sections = summarize_transcript(client, transcript, report=RunReport())

    retry = CountingLLM()
    summarize_transcript(retry, transcript, cached_sections=sections, report=RunReport())

    assert retry.windows == len([s for s in sections if s.get("_error")])
    assert retry.windows > 0


def test_garbage_in_the_cache_is_ignored_rather_than_trusted(transcript):
    client = CountingLLM()
    summarize_transcript(
        client, transcript,
        cached_sections=["not a dict", {}, {"_label": ""}],  # type: ignore[list-item]
        report=RunReport(),
    )
    assert client.windows == len(chunk_transcript(transcript, 600, 30))


# --------------------------------------------------------------------------- #
# Salvage
# --------------------------------------------------------------------------- #


def test_a_cut_off_window_keeps_the_points_it_finished():
    cut = (
        '{"heading": "Aggregate planning", '
        '"key_points": ["chase tracks demand", "level holds capacity steady"], '
        '"key_terms": [{"term": "takt ti'
    )

    class Truncating:
        def complete_json(self, system, user, max_tokens=None):
            raise TruncatedResponseError("cut off", raw=cut)

    chunk = chunk_transcript(
        Transcript(
            segments=[Segment(index=0, start=0.0, end=600.0, text="content " * 200)],
            duration=600.0,
        ),
        600, 30,
    )[0]

    section = summarize_chunk(Truncating(), chunk)
    assert section["heading"] == "Aggregate planning"
    assert len(section["key_points"]) == 2
    assert "key_terms" not in section or section["key_terms"] == []


def test_a_cut_off_window_with_nothing_finished_still_fails():
    class Truncating:
        def complete_json(self, system, user, max_tokens=None):
            raise TruncatedResponseError("cut off", raw='{"heading": "Aggregate pla')

    chunk = chunk_transcript(
        Transcript(
            segments=[Segment(index=0, start=0.0, end=600.0, text="content " * 200)],
            duration=600.0,
        ),
        600, 30,
    )[0]

    with pytest.raises(TruncatedResponseError):
        summarize_chunk(Truncating(), chunk)


def test_object_salvage_keeps_only_fields_that_closed():
    data = salvage_object_fields('{"a": 1, "b": [1, 2], "c": {"d": "unterminated')
    assert data == {"a": 1, "b": [1, 2]}


def test_object_salvage_handles_a_complete_object_and_a_fence():
    assert salvage_object_fields('```json\n{"a": 1}\n```')["a"] == 1
    assert salvage_object_fields('{"a": 1}') == {"a": 1}


def test_object_salvage_is_not_fooled_by_punctuation_inside_strings():
    raw = '{"a": "one, two, three", "b": "unclosed, with, commas'
    assert salvage_object_fields(raw) == {"a": "one, two, three"}


def test_object_salvage_returns_nothing_rather_than_guessing():
    assert salvage_object_fields("") == {}
    assert salvage_object_fields("I'm sorry, I can't help with that.") == {}
    assert salvage_object_fields('{"a": "unterminated') == {}


# --------------------------------------------------------------------------- #
# The local assembly, and what it admits about itself
# --------------------------------------------------------------------------- #


def test_the_local_summary_collates_and_never_invents():
    sections = [
        {"_label": "0:00–10:00", "heading": "Chase", "key_points": ["a", "b"],
         "key_terms": [{"term": "Takt", "definition": "d"}]},
        {"_label": "10:00–20:00", "heading": "Level", "key_points": ["b", "c"],
         "key_terms": [{"term": "takt", "definition": "d"}]},
        {"_label": "20:00–30:00", "heading": "", "key_points": [], "_error": "429"},
    ]
    summary = summary_from_sections(sections)

    assert summary.key_points == ["a", "b", "c"], "deduplicated, in order"
    assert len(summary.key_terms) == 1, "the same term twice is one term"
    assert [o["heading"] for o in summary.outline] == ["Chase", "Level"]
    assert "did not complete" in summary.abstract, "it must not pass as synthesized"


def test_the_local_summary_of_nothing_is_empty_not_misleading():
    assert summary_from_sections([]).key_points == []
    assert summary_from_sections([{"_error": "x"}]).abstract == ""
