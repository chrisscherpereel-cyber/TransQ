"""Exporting this app's real prompts for an external benchmark.

Ori Eval — and any harness like it — answers "which model is best at *this*"
only if it is given *this*. A generic benchmark tells you which model is better
at generic tasks; it cannot tell you which one writes sound multiple-choice
items from a lecture transcript, in JSON, with a verbatim supporting quote,
under fifteen item-writing rules. So the export has to carry the app's actual
prompts, including the clauses a particular lecture produces.

The failure to guard against is an export that *looks* right and quietly tests
something else: prompts missing the exam-topic steer, missing the slides, or
carrying a run-specific avoid list that makes two models incomparable.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

from src.library import SavedLecture, LibraryEntry
from src.materials import extract_material
from src.schema import Segment, Summary, Transcript

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "export_eval.py",
)


@pytest.fixture(scope="module")
def exporter():
    sys.path.insert(0, os.path.dirname(SCRIPT))
    spec = importlib.util.spec_from_file_location("export_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lecture() -> SavedLecture:
    segments = [
        Segment(
            index=i,
            start=i * 30.0,
            end=(i + 1) * 30.0,
            text=(
                "Chase strategy tracks demand through hiring and firing workers "
                "each period, trading stability for lower inventory. "
            ),
        )
        for i in range(80)
    ]
    deck = extract_material(
        "week4.md",
        b"# Chase strategy\nChase tracks demand by hiring and firing workers.",
    )
    return SavedLecture(
        entry=LibraryEntry(id="abc123", title="MGT 301 — Week 4"),
        transcript=Transcript(segments=segments, duration=2400.0),
        summary=Summary(
            title="Aggregate Planning",
            learning_objectives=["Explain chase strategy"],
            key_points=["Chase tracks demand"],
        ),
        material=deck,
        exam_topics=["Chase versus level"],
    )


def test_the_export_carries_the_apps_real_prompts(exporter, lecture):
    records = exporter.build_records(lecture, target=10)

    assert records
    for record in records:
        assert record["system"], "the item-writing rules must travel with it"
        assert "multiple-choice questions" in record["prompt"]
        assert record["expected_questions"] >= 1


def test_the_exam_topics_reach_the_exported_prompt(exporter, lecture):
    """Benchmarking a prompt without the steer measures the wrong thing."""
    records = exporter.build_records(lecture, target=10)
    assert all("EXAM" in r["prompt"] for r in records)
    assert any("Chase versus level" in r["prompt"] for r in records)


def test_the_slides_reach_the_exported_prompt(exporter, lecture):
    records = exporter.build_records(lecture, target=10)
    assert any("Chase strategy" in r["prompt"] for r in records)


def test_the_avoid_list_is_deliberately_left_out(exporter, lecture):
    """It changes between runs, so including it would make two models'
    results incomparable — the one thing a benchmark cannot tolerate."""
    records = exporter.build_records(lecture, target=10)
    assert all("already exist" not in r["prompt"] for r in records)


def test_the_counts_match_the_apps_own_allocation(exporter, lecture):
    """The export must ask for what the app asks for, or under-delivery gets
    measured against the wrong denominator."""
    records = exporter.build_records(lecture, target=10)
    assert sum(r["expected_questions"] for r in records) == 10


def test_a_grader_gets_the_source_text_to_check_quotes_against(exporter, lecture):
    """Provenance is the property that most separates usable models here, and
    it cannot be graded on plausibility."""
    records = exporter.build_records(lecture, target=6)
    assert all(r["transcript_excerpt"] for r in records)
    assert all(r["window"] for r in records)


def test_records_are_identifiable_per_window(exporter, lecture):
    records = exporter.build_records(lecture, target=8)
    ids = [r["id"] for r in records]
    assert len(set(ids)) == len(ids)
    assert all(r["id"].startswith("abc123-") for r in records)


def test_a_lecture_without_slides_or_topics_still_exports(exporter, lecture):
    """Both are optional in the app, so both must be optional here."""
    bare = SavedLecture(entry=lecture.entry, transcript=lecture.transcript)
    records = exporter.build_records(bare, target=6)

    assert records
    assert all("EXAM" not in r["prompt"] for r in records)


def test_an_empty_transcript_exports_nothing_rather_than_raising(exporter):
    empty = SavedLecture(entry=LibraryEntry(id="x", title="t"), transcript=Transcript())
    assert exporter.build_records(empty, target=5) == []


def test_the_rubric_names_the_disqualifying_failures(exporter):
    """A rubric that rewards fluency would pick the wrong model for this task."""
    rubric = exporter.RUBRIC
    assert "verbatim" in rubric
    assert "correct_index" in rubric
    assert "JSON" in rubric
    assert "Do not reward fluency" in rubric


def test_the_readme_warns_that_the_export_contains_the_transcript(exporter):
    assert "transcript" in exporter.READ_ME
    assert "version control" in exporter.READ_ME
