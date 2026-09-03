"""Tests for split-recording transcription and question regeneration.

The Whisper model and the LLM are both stubbed, so these run offline. What they
pin down is the arithmetic and the prompt plumbing — the two places where a
quiet bug would produce plausible-looking but wrong output: timestamps that
drift after the first part, and an "alternative" set that is really the same
questions again.
"""

from __future__ import annotations

import json
import types

import pytest

from src.chunking import chunk_transcript
from src.mcq import (
    _build_avoid_clause,
    find_chunk_for_timestamp,
    generate_questions,
    generate_replacement,
)
from src.schema import MCQ, Segment, Transcript, format_timestamp
from src.transcribe import natural_sort_key, transcribe_parts


# --------------------------------------------------------------------------- #
# Stub Whisper
# --------------------------------------------------------------------------- #


class FakeInfo:
    def __init__(self, duration: float, language: str = "en"):
        self.duration = duration
        self.language = language


class FakeWhisper:
    """Returns one 30-second segment per half minute of the named part."""

    model_size_or_path = "fake"

    def __init__(self, durations: dict[str, float], silent: set[str] | None = None):
        self.durations = durations
        self.silent = silent or set()
        self.calls: list[str] = []

    def transcribe(self, path, **kwargs):
        import os

        name = os.path.basename(path)
        self.calls.append(name)
        duration = self.durations[name]
        if name in self.silent:
            return iter([]), FakeInfo(duration)

        segments = [
            types.SimpleNamespace(start=float(t), end=float(t + 30), text=f"{name} at {t}s.")
            for t in range(0, int(duration), 30)
        ]
        return iter(segments), FakeInfo(duration)


@pytest.fixture
def parts(tmp_path):
    """Three 10-minute parts of one 30-minute lecture."""
    names = ["lecture_part1.mp3", "lecture_part2.mp3", "lecture_part3.mp3"]
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"fake audio")
        paths.append(str(p))
    return names, paths


@pytest.fixture(autouse=True)
def no_probe(monkeypatch):
    """probe_duration reads real media; stub it to the fixture durations."""
    import src.transcribe as tr

    monkeypatch.setattr(tr, "probe_duration", lambda path: 0.0)


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_natural_sort_puts_part2_before_part10():
    names = ["lec_part10.mp3", "lec_part2.mp3", "lec_part1.mp3"]
    assert sorted(names, key=natural_sort_key) == [
        "lec_part1.mp3",
        "lec_part2.mp3",
        "lec_part10.mp3",
    ]
    # Plain alphabetical sorting is what this guards against.
    assert sorted(names) != sorted(names, key=natural_sort_key)


def test_natural_sort_is_case_insensitive_and_stable():
    assert natural_sort_key("Part1.mp3") < natural_sort_key("part2.mp3")


# --------------------------------------------------------------------------- #
# Stitching
# --------------------------------------------------------------------------- #


def test_parts_are_transcribed_in_the_order_given(parts):
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names})
    transcribe_parts(model, paths, display_names=names)
    assert model.calls == names


def test_timestamps_are_offset_onto_one_lecture_timeline(parts):
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names})
    t = transcribe_parts(model, paths, display_names=names)

    assert t.is_multipart
    assert len(t.parts) == 3
    assert [p.offset for p in t.parts] == [0.0, 600.0, 1200.0]
    assert t.duration == pytest.approx(1800.0)

    # Part 2's first segment must land at 10:00, not back at 0:00.
    part2_first = next(s for s in t.segments if s.part == 1)
    assert part2_first.start == pytest.approx(600.0)
    assert part2_first.timestamp == "10:00"

    part3_first = next(s for s in t.segments if s.part == 2)
    assert part3_first.timestamp == "20:00"


def test_segments_are_monotonic_and_reindexed(parts):
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names})
    t = transcribe_parts(model, paths, display_names=names)

    assert [s.index for s in t.segments] == list(range(len(t.segments)))
    starts = [s.start for s in t.segments]
    assert starts == sorted(starts), "combined timeline must never go backwards"


def test_uneven_part_lengths_still_line_up(parts):
    names, paths = parts
    model = FakeWhisper({names[0]: 300.0, names[1]: 900.0, names[2]: 600.0})
    t = transcribe_parts(model, paths, display_names=names)
    assert [p.offset for p in t.parts] == [0.0, 300.0, 1200.0]
    assert t.duration == pytest.approx(1800.0)


def test_a_silent_part_is_skipped_without_losing_the_others(parts):
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names}, silent={names[1]})
    t = transcribe_parts(model, paths, display_names=names)

    assert len(t.parts) == 2
    assert len(t.skipped_parts) == 1
    assert names[1] in t.skipped_parts[0]
    assert {s.part for s in t.segments} == {0, 2}


def test_all_parts_silent_raises(parts):
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names}, silent=set(names))
    with pytest.raises(Exception) as exc:
        transcribe_parts(model, paths, display_names=names)
    assert "No speech" in str(exc.value)


def test_no_files_raises():
    with pytest.raises(Exception):
        transcribe_parts(FakeWhisper({}), [])


def test_single_file_still_reports_one_part(parts):
    names, paths = parts
    model = FakeWhisper({names[0]: 600.0})
    t = transcribe_parts(model, paths[:1], display_names=names[:1])
    assert not t.is_multipart
    assert t.parts[0].filename == names[0]


def test_stitched_transcript_chunks_across_part_boundaries(parts):
    """The whole point: summarization sees one lecture, not three files."""
    names, paths = parts
    model = FakeWhisper({n: 600.0 for n in names})
    t = transcribe_parts(model, paths, display_names=names)

    chunks = chunk_transcript(t, chunk_seconds=900, overlap_seconds=0)
    assert len(chunks) == 2
    # The first 15-minute window must contain text from both part 1 and part 2.
    assert "lecture_part1" in chunks[0].text and "lecture_part2" in chunks[0].text


# --------------------------------------------------------------------------- #
# Avoid-lists and regeneration
# --------------------------------------------------------------------------- #


class ScriptedLLM:
    """Records prompts and returns a fresh numbered question each time."""

    def __init__(self):
        self.prompts: list[str] = []
        self.counter = 0

    def complete_json(self, system, user, max_tokens=None):
        self.prompts.append(user)
        n = int(user.split("exactly ")[1].split(" ")[0]) if "exactly " in user else 1
        out = []
        for _ in range(n):
            self.counter += 1
            out.append(
                {
                    "stem": f"Generated question {self.counter} about planning?",
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
        Segment(index=i, start=i * 60.0, end=(i + 1) * 60.0, text=f"Content {i}. " * 20)
        for i in range(30)
    ]
    return chunk_transcript(Transcript(segments=segments, duration=1800.0), chunk_seconds=600)


def test_avoid_clause_lists_prior_questions():
    clause = _build_avoid_clause(["Capacity"], ["What is takt time?"])
    assert "Capacity" in clause
    assert "What is takt time?" in clause
    assert _build_avoid_clause([], []) == ""


def test_generation_passes_prior_stems_into_the_prompt(chunks):
    llm = ScriptedLLM()
    generate_questions(llm, chunks, [1] * len(chunks), avoid_stems=["An earlier question?"])
    assert all("An earlier question?" in p for p in llm.prompts)


def test_questions_from_earlier_chunks_are_avoided_in_later_ones(chunks):
    llm = ScriptedLLM()
    generate_questions(llm, chunks, [1] * len(chunks))
    # The second chunk's prompt must mention what the first chunk produced.
    assert "Generated question 1" in llm.prompts[1]


def test_alternative_set_is_steered_away_from_the_first(chunks):
    llm = ScriptedLLM()
    first = generate_questions(llm, chunks, [2] * len(chunks))
    llm.prompts.clear()

    second = generate_questions(
        llm, chunks, [2] * len(chunks), avoid_stems=[q.stem for q in first]
    )
    assert {q.stem for q in first}.isdisjoint({q.stem for q in second})
    for stem in [q.stem for q in first][-25:]:
        assert stem in llm.prompts[0]


# --------------------------------------------------------------------------- #
# Single-question replacement
# --------------------------------------------------------------------------- #


def make_q(**kw) -> MCQ:
    base = dict(
        stem="An original question about aggregate planning?",
        options=["A", "B", "C", "D"],
        correct_index=2,
        rationale="r",
        bloom="Analyze",
        difficulty="Hard",
        source_timestamp="12:30",
        source_quote="q",
        points=2.5,
    )
    base.update(kw)
    return MCQ(**base)


def test_find_chunk_for_timestamp(chunks):
    assert find_chunk_for_timestamp(chunks, "0:30").index == 0
    found = find_chunk_for_timestamp(chunks, "12:30")
    assert found.start <= 750 < found.end
    # Out-of-range or unparseable timestamps still resolve to something.
    assert find_chunk_for_timestamp(chunks, "99:99:99") is not None
    assert find_chunk_for_timestamp(chunks, "") is not None
    assert find_chunk_for_timestamp([], "1:00") is None


def test_replacement_comes_from_the_same_section_by_default(chunks):
    llm = ScriptedLLM()
    old = make_q()
    new = generate_replacement(llm, chunks, old, existing_stems=[old.stem])

    assert new is not None and new.stem != old.stem
    assert len(llm.prompts) == 1
    expected = find_chunk_for_timestamp(chunks, old.source_timestamp)
    assert expected.label in llm.prompts[0], "should quote the same stretch of lecture"
    assert old.stem in llm.prompts[0], "should be told what it is replacing"


def test_replacement_preserves_point_value(chunks):
    new = generate_replacement(ScriptedLLM(), chunks, make_q(points=2.5), existing_stems=[])
    assert new.points == 2.5


def test_replacement_is_validated(chunks):
    new = generate_replacement(ScriptedLLM(), chunks, make_q(), existing_stems=[])
    assert isinstance(new.flags, list)  # checks ran, clean item has none
    assert new.flags == []


def test_replacement_without_chunks_returns_none():
    assert generate_replacement(ScriptedLLM(), [], make_q(), existing_stems=[]) is None


def test_timestamp_formatting_across_a_long_multipart_lecture():
    assert format_timestamp(3134) == "52:14"
    assert format_timestamp(3600 + 314) == "1:05:14"
