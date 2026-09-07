"""Tests for the saved-lecture library.

The reported bug: transcripts did not persist. They were never written anywhere —
only `st.session_state` held them, so a refresh, an idle sign-out or a Streamlit
restart discarded twenty minutes of transcription.

The property under test throughout: **what you get back must be what you put in,
after the process that wrote it is gone.** So the round-trip tests rebuild the
store from disk rather than reusing the object that saved.
"""

from __future__ import annotations

import pytest

from src.library import (
    LibraryEntry,
    LibraryError,
    TranscriptLibrary,
    _default_title,
)
from src.schema import (
    MCQ,
    Quiz,
    QuizMeta,
    Segment,
    Summary,
    Transcript,
    TranscriptPart,
)
from src.storage import GuardedStore, MemoryStore, build_store

SECRET = "a-long-random-app-secret-for-tests"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_transcript(filename: str = "mgt301_lecture4.mp3", parts: int = 1) -> Transcript:
    segments = [
        Segment(index=i, start=i * 30.0, end=(i + 1) * 30.0,
                text=f"Operations content {i}. " * 8, part=i % parts)
        for i in range(60)
    ]
    return Transcript(
        segments=segments,
        duration=1800.0,
        language="en",
        parts=[
            TranscriptPart(
                index=p, filename=f"{filename}" if parts == 1 else f"part{p + 1}.mp3",
                offset=p * (1800.0 / parts), duration=1800.0 / parts, segments=60 // parts,
            )
            for p in range(parts)
        ],
    )


def make_quiz(n: int = 6, title: str = "Set 1") -> Quiz:
    return Quiz(
        meta=QuizMeta(title=title),
        questions=[
            MCQ(
                stem=f"Which claim about subject {i} follows from the lecture?",
                options=["Alpha", "Beta", "Gamma", "Delta"],
                correct_index=i % 4,
                rationale="Because alpha.",
                source_timestamp="5:00",
                source_quote="alpha is correct",
            )
            for i in range(n)
        ],
    )


@pytest.fixture
def library() -> TranscriptLibrary:
    return TranscriptLibrary(GuardedStore(MemoryStore()), "chris")


# --------------------------------------------------------------------------- #
# Saving and listing
# --------------------------------------------------------------------------- #


def test_a_library_needs_an_account():
    with pytest.raises(LibraryError):
        TranscriptLibrary(MemoryStore(), "")


def test_saving_returns_an_entry_describing_the_lecture(library):
    entry = library.save(make_transcript(), origin="audio")
    assert entry.id
    assert entry.duration == 1800.0
    assert entry.word_count > 0
    assert entry.length_label == "30:00"
    assert entry.origin == "audio"
    assert not entry.has_summary and entry.question_sets == 0


def test_an_empty_transcript_is_refused(library):
    with pytest.raises(LibraryError):
        library.save(Transcript())


def test_the_default_title_comes_from_the_filename():
    assert _default_title(make_transcript("mgt301_lecture4.mp3")) == "mgt301 lecture4"
    assert _default_title(make_transcript("MGT-301-Week-2.m4a")) == "MGT 301 Week 2"
    bare = Transcript(segments=[Segment(index=0, start=0, end=60, text="x")], duration=600.0)
    assert "10:00" in _default_title(bare)


def test_entries_are_listed_newest_first(library):
    first = library.save(make_transcript(), title="Week 1")
    second = library.save(make_transcript(), title="Week 2")
    second.updated_at = "2099-01-01T00:00:00+00:00"
    library._upsert(second)

    listed = library.entries()
    assert [e.title for e in listed] == ["Week 2", "Week 1"]
    assert library.count() == 2
    assert first.id != second.id


def test_an_empty_library_lists_nothing(library):
    assert library.entries() == []
    assert library.count() == 0


# --------------------------------------------------------------------------- #
# Round-trip — the actual bug
# --------------------------------------------------------------------------- #


def test_a_transcript_survives_a_restart(tmp_path):
    """A fresh store object, as after a Streamlit restart."""
    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    original = make_transcript()
    entry = TranscriptLibrary(store, "chris").save(original, origin="audio")

    reopened, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    saved = TranscriptLibrary(reopened, "chris").load(entry.id)

    assert saved.transcript.text == original.text
    assert saved.transcript.word_count == original.word_count
    assert len(saved.transcript.segments) == len(original.segments)
    assert saved.transcript.duration == original.duration


def test_timestamps_and_parts_survive_the_round_trip(library):
    original = make_transcript(parts=3)
    entry = library.save(original)
    saved = library.load(entry.id)

    assert saved.transcript.is_multipart
    assert len(saved.transcript.parts) == 3
    assert [p.offset for p in saved.transcript.parts] == [p.offset for p in original.parts]
    assert saved.transcript.segments[17].timestamp == original.segments[17].timestamp
    assert [s.part for s in saved.transcript.segments] == [s.part for s in original.segments]


def test_the_summary_and_every_question_set_come_back(library):
    summary = Summary(title="Aggregate Planning", abstract="A" * 120,
                      key_points=["Chase tracks demand"],
                      learning_objectives=["Explain chase vs level"])
    quizzes = [make_quiz(6, "Set 1"), make_quiz(4, "Set 2")]
    entry = library.save(make_transcript(), summary=summary, quizzes=quizzes)

    saved = library.load(entry.id)
    assert saved.summary.title == "Aggregate Planning"
    assert saved.summary.learning_objectives == summary.learning_objectives
    assert len(saved.quizzes) == 2
    assert [len(q.included) for q in saved.quizzes] == [6, 4]


def test_answer_keys_and_provenance_survive(library):
    quiz = make_quiz(8)
    entry = library.save(make_transcript(), quizzes=[quiz])
    restored = library.load(entry.id).quizzes[0]

    assert [q.answer_letter for q in restored.questions] == [
        q.answer_letter for q in quiz.questions
    ]
    assert [q.correct_option for q in restored.questions] == [
        q.correct_option for q in quiz.questions
    ]
    assert all(q.source_timestamp == "5:00" for q in restored.questions)
    assert all(q.source_quote for q in restored.questions)


def test_the_index_reflects_what_was_generated(library):
    entry = library.save(
        make_transcript(), summary=Summary(title="T"), quizzes=[make_quiz(6), make_quiz(4)]
    )
    listed = library.entries()[0]
    assert listed.has_summary
    assert listed.question_sets == 2
    assert listed.questions == 10
    assert "2 question sets" in listed.summary_label


def test_transcript_only_entries_say_so(library):
    library.save(make_transcript())
    assert library.entries()[0].summary_label == "transcript only"


# --------------------------------------------------------------------------- #
# Updating in place
# --------------------------------------------------------------------------- #


def test_generating_updates_the_same_entry_rather_than_duplicating(library):
    """Auto-save runs twice per lecture: after transcription, after generation."""
    transcript = make_transcript()
    first = library.save(transcript, origin="audio")
    second = library.save(
        transcript, summary=Summary(title="T"), quizzes=[make_quiz()],
        entry_id=first.id,
    )

    assert second.id == first.id
    assert library.count() == 1
    assert second.created_at == first.created_at, "creation time must not move"
    assert library.entries()[0].has_summary


def test_a_stale_entry_id_makes_a_new_entry_not_an_orphan(library):
    """The lecture was deleted in another tab; saving must not write a record
    that nothing in the index points at."""
    entry = library.save(make_transcript())
    library.delete(entry.id)

    fresh = library.save(make_transcript(), entry_id=entry.id)
    assert fresh.id != entry.id
    assert library.get_entry(fresh.id) is not None


def test_updating_keeps_the_original_title(library):
    first = library.save(make_transcript(), title="Week 4 — capacity")
    second = library.save(make_transcript(), entry_id=first.id, quizzes=[make_quiz()])
    assert second.title == "Week 4 — capacity"


# --------------------------------------------------------------------------- #
# Rename and delete
# --------------------------------------------------------------------------- #


def test_rename(library):
    entry = library.save(make_transcript())
    library.rename(entry.id, "MGT 301 — Lecture 4")

    assert library.entries()[0].title == "MGT 301 — Lecture 4"
    assert library.load(entry.id).entry.title == "MGT 301 — Lecture 4"


def test_rename_rejects_an_empty_title_and_a_missing_entry(library):
    entry = library.save(make_transcript())
    with pytest.raises(LibraryError):
        library.rename(entry.id, "   ")
    with pytest.raises(LibraryError):
        library.rename("nonexistent", "Anything")


def test_delete_removes_it_from_the_index_and_the_content(library):
    entry = library.save(make_transcript())
    library.delete(entry.id)

    assert library.entries() == []
    assert library.get_entry(entry.id) is None
    with pytest.raises(LibraryError):
        library.load(entry.id)


def test_deleting_leaves_no_readable_transcript_behind(library):
    entry = library.save(make_transcript())
    library.delete(entry.id)
    record = library.store.read(f"library/chris/t/{entry.id}") or {}
    assert record.get("deleted") is True
    assert "transcript" not in record


def test_loading_something_that_never_existed(library):
    with pytest.raises(LibraryError):
        library.load("nope")


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


def test_libraries_are_per_account(library):
    library.save(make_transcript(), title="Mine")
    other = TranscriptLibrary(library.store, "jsmith")
    assert other.entries() == []

    other.save(make_transcript(), title="Theirs")
    assert [e.title for e in library.entries()] == ["Mine"]
    assert [e.title for e in other.entries()] == ["Theirs"]


def test_one_save_does_not_rewrite_another_lecture(library):
    first = library.save(make_transcript(), title="Week 1")
    before = library.store.read(f"library/chris/t/{first.id}")["_v"]

    library.save(make_transcript(), title="Week 2")
    after = library.store.read(f"library/chris/t/{first.id}")["_v"]
    assert after == before, "per-lecture documents must not touch each other"


def test_transcripts_are_encrypted_at_rest(tmp_path):
    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    entry = TranscriptLibrary(store, "chris").save(make_transcript())

    blob = (tmp_path / "library" / "chris" / "t" / f"{entry.id}.enc").read_bytes()
    assert b"Operations content" not in blob
    assert b"mgt301" not in blob.lower()


# --------------------------------------------------------------------------- #
# Entry presentation
# --------------------------------------------------------------------------- #


def test_entry_labels_are_readable():
    entry = LibraryEntry(
        id="x", title="T", duration=3725.0,
        updated_at="2026-09-07T14:32:11+00:00",
        has_summary=True, question_sets=1, questions=12,
    )
    assert entry.length_label == "1:02:05"
    assert entry.saved_label == "2026-09-07 14:32"
    assert entry.summary_label == "summary · 1 question set (12 questions)"


def test_entry_tolerates_fields_from_a_future_version():
    entry = LibraryEntry.from_dict({"id": "x", "title": "T", "unexpected_field": 1})
    assert entry.id == "x" and entry.title == "T"
