"""Surviving a process that dies partway through a split recording.

The reported failure: a three-part lecture crashed during part 3, and parts 1
and 2 — twenty minutes of successful transcription — were gone. They had been
accumulating in a list local to ``transcribe_parts``, and the only write to
storage happened after all three parts returned.

What makes this class of bug nasty is that the usual defence does not apply. A
container that exceeds its memory limit is SIGKILLed: no exception is raised, no
``except`` clause runs, no ``finally`` runs, no atexit handler runs. **The only
state that survives is state already written down.** So the tests here do not
simulate an exception — an exception would have been survivable all along. They
simulate the process simply ceasing, by asserting on what the checkpoint
callback had already handed over at the moment of death.

Three properties:

1. Every completed part is durable *before* the next one starts.
2. A checkpoint is a valid, loadable transcript — not a fragment needing repair.
3. Resuming appends to the saved timeline, so timestamps stay correct across the
   join and no part is transcribed twice.
"""

from __future__ import annotations

import os
import types

import pytest

from src.library import TranscriptLibrary
from src.schema import Segment, Transcript
from src.storage import GuardedStore, MemoryStore
from src.transcribe import TranscriptBuilder, TranscriptionError, transcribe_parts

PART_SECONDS = 600.0


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


class FakeInfo:
    def __init__(self, duration: float, language: str = "en"):
        self.duration = duration
        self.language = language


class DyingWhisper:
    """Transcribes normally until the named part, then the process 'dies'.

    ``SystemExit`` stands in for SIGKILL. It does not inherit from ``Exception``,
    so the ``except TranscriptionError`` handlers in the transcription path
    cannot catch it — which is exactly the property being modelled. Whatever the
    checkpoint wrote before this point is all that survives.
    """

    model_size_or_path = "fake"

    def __init__(self, dies_on: str | None = None, silent: set[str] | None = None):
        self.dies_on = dies_on
        self.silent = silent or set()
        self.calls: list[str] = []

    def transcribe(self, path, **kwargs):
        name = os.path.basename(path)
        self.calls.append(name)
        if name == self.dies_on:
            raise SystemExit("container killed (out of memory)")
        if name in self.silent:
            return iter([]), FakeInfo(PART_SECONDS)
        segments = [
            types.SimpleNamespace(
                start=float(t), end=float(t + 30), text=f"{name} at {t}s."
            )
            for t in range(0, int(PART_SECONDS), 30)
        ]
        return iter(segments), FakeInfo(PART_SECONDS)


@pytest.fixture
def parts(tmp_path):
    names = ["lecture_part1.mp3", "lecture_part2.mp3", "lecture_part3.mp3"]
    paths = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"fake audio")
        paths.append(str(p))
    return names, paths


@pytest.fixture(autouse=True)
def no_probe(monkeypatch):
    import src.transcribe as tr

    monkeypatch.setattr(tr, "probe_duration", lambda path: PART_SECONDS)


@pytest.fixture
def library() -> TranscriptLibrary:
    return TranscriptLibrary(GuardedStore(MemoryStore()), "chris")


# --------------------------------------------------------------------------- #
# 1. Checkpoints happen, and happen in time
# --------------------------------------------------------------------------- #


def test_each_part_is_handed_over_as_it_finishes(parts):
    names, paths = parts
    seen: list[Transcript] = []
    transcribe_parts(
        DyingWhisper(), paths, display_names=names, on_part_complete=seen.append
    )

    assert len(seen) == 3, "one checkpoint per part, not one at the end"
    assert [len(t.parts) for t in seen] == [1, 2, 3]
    assert [len(t.pending_parts) for t in seen] == [2, 1, 0]


def test_a_checkpoint_lands_before_the_next_part_is_read(parts):
    """Ordering is the whole guarantee: written *before*, not after."""
    names, paths = parts
    model = DyingWhisper()
    events: list[str] = []

    original = model.transcribe

    def watched(path, **kwargs):
        events.append(f"read:{os.path.basename(path)}")
        return original(path, **kwargs)

    model.transcribe = watched
    transcribe_parts(
        model,
        paths,
        display_names=names,
        on_part_complete=lambda t: events.append(f"saved:{len(t.parts)}"),
    )

    assert events == [
        "read:lecture_part1.mp3", "saved:1",
        "read:lecture_part2.mp3", "saved:2",
        "read:lecture_part3.mp3", "saved:3",
    ]


def test_the_reported_bug_two_parts_survive_a_kill_on_the_third(parts, library):
    """The regression, end to end, with a real library behind the callback."""
    names, paths = parts
    model = DyingWhisper(dies_on=names[2])

    entry_id: list[str] = []

    def save(partial: Transcript) -> None:
        entry = library.save(
            partial, title="MGT 301 Week 4",
            entry_id=entry_id[0] if entry_id else None,
        )
        entry_id[:] = [entry.id]

    with pytest.raises(SystemExit):
        transcribe_parts(
            model, paths, display_names=names, on_part_complete=save
        )

    # The process is gone. What is left in storage is all there is.
    saved = library.load(entry_id[0])
    assert len(saved.transcript.parts) == 2
    assert saved.transcript.word_count > 0
    assert saved.transcript.duration == pytest.approx(2 * PART_SECONDS)
    assert saved.transcript.pending_parts == [names[2]]
    assert not saved.transcript.is_complete


def test_checkpoints_update_one_entry_rather_than_piling_up(parts, library):
    names, paths = parts
    entry_id: list[str] = []

    def save(partial: Transcript) -> None:
        entry = library.save(
            partial, entry_id=entry_id[0] if entry_id else None
        )
        entry_id[:] = [entry.id]

    transcribe_parts(DyingWhisper(), paths, display_names=names, on_part_complete=save)
    assert library.count() == 1, "three checkpoints, one lecture"


def test_a_failing_save_does_not_stop_the_transcription(parts):
    """Storage being briefly unavailable is a reason to keep going, not to stop
    — otherwise the checkpoint would cause the loss it exists to prevent."""
    names, paths = parts

    def explode(_):
        raise RuntimeError("Dropbox is unreachable")

    result = transcribe_parts(
        DyingWhisper(), paths, display_names=names, on_part_complete=explode
    )
    assert len(result.parts) == 3


def test_a_skipped_part_still_checkpoints_the_rest(parts):
    """A silent part 2 must not leave part 1 unsaved while part 3 runs."""
    names, paths = parts
    seen: list[Transcript] = []
    transcribe_parts(
        DyingWhisper(silent={names[1]}),
        paths, display_names=names, on_part_complete=seen.append,
    )
    assert [len(t.parts) for t in seen] == [1, 1, 2]
    assert seen[-1].skipped_parts


def test_nothing_is_checkpointed_before_there_is_anything_to_save(parts):
    """A first part that yields no speech should not write an empty transcript."""
    names, paths = parts
    seen: list[Transcript] = []
    transcribe_parts(
        DyingWhisper(silent={names[0]}),
        paths, display_names=names, on_part_complete=seen.append,
    )
    assert all(t.segments for t in seen)


# --------------------------------------------------------------------------- #
# 2. A checkpoint is a real transcript
# --------------------------------------------------------------------------- #


def test_a_checkpoint_round_trips_through_storage(parts, library):
    names, paths = parts
    seen: list[Transcript] = []
    transcribe_parts(
        DyingWhisper(), paths, display_names=names, on_part_complete=seen.append
    )

    midway = seen[0]
    entry = library.save(midway, title="Half done")
    restored = library.load(entry.id).transcript

    assert restored.text == midway.text
    assert restored.pending_parts == midway.pending_parts
    assert restored.duration == midway.duration


def test_the_library_entry_reports_being_unfinished(parts, library):
    names, paths = parts
    seen: list[Transcript] = []
    transcribe_parts(
        DyingWhisper(), paths, display_names=names, on_part_complete=seen.append
    )

    library.save(seen[1], title="Two of three")
    listed = library.entries()[0]
    assert not listed.is_complete
    assert listed.pending_parts == [names[2]]
    assert listed.progress_label == "2 of 3 parts transcribed"
    assert "incomplete" in listed.summary_label


def test_a_finished_transcript_has_nothing_pending(parts):
    names, paths = parts
    final = transcribe_parts(DyingWhisper(), paths, display_names=names)
    assert final.is_complete and final.pending_parts == []


# --------------------------------------------------------------------------- #
# 3. Resuming
# --------------------------------------------------------------------------- #


def test_resuming_transcribes_only_what_is_missing(parts):
    names, paths = parts
    checkpoints: list[Transcript] = []
    model = DyingWhisper(dies_on=names[2])
    with pytest.raises(SystemExit):
        transcribe_parts(
            model, paths, display_names=names, on_part_complete=checkpoints.append
        )

    partial = checkpoints[-1]
    resumed_model = DyingWhisper()
    final = transcribe_parts(
        resumed_model, paths[2:], display_names=names[2:], resume_from=partial
    )

    assert resumed_model.calls == [names[2]], "part 3 only — not the whole lecture again"
    assert len(final.parts) == 3
    assert final.is_complete


def test_the_timeline_continues_across_the_join(parts):
    """The point of resuming: a question from part 3 still cites 0:22:00 of the
    lecture, not 0:02:00 of the third file."""
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2]), paths, display_names=names,
            on_part_complete=checkpoints.append,
        )

    resumed = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:],
        resume_from=checkpoints[-1],
    )

    assert [p.offset for p in resumed.parts] == [0.0, 600.0, 1200.0]
    assert resumed.duration == pytest.approx(3 * PART_SECONDS)

    third = next(s for s in resumed.segments if s.part == 2)
    assert third.start >= 1200.0

    starts = [s.start for s in resumed.segments]
    assert starts == sorted(starts), "the joined timeline must never go backwards"
    assert [s.index for s in resumed.segments] == list(
        range(len(resumed.segments))
    ), "segment indices must stay contiguous across the join"


def test_resuming_keeps_the_text_that_was_already_transcribed(parts):
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2]), paths, display_names=names,
            on_part_complete=checkpoints.append,
        )
    before = checkpoints[-1].text

    resumed = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:],
        resume_from=checkpoints[-1],
    )
    assert resumed.text.startswith(before)
    assert names[2] in resumed.text


def test_resuming_carries_forward_a_part_that_was_skipped(parts):
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2], silent={names[1]}),
            paths, display_names=names, on_part_complete=checkpoints.append,
        )

    resumed = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:],
        resume_from=checkpoints[-1],
    )
    assert resumed.skipped_parts, "a skipped part must not be forgotten on resume"


def test_resuming_a_checkpoint_that_was_reloaded_from_storage(parts, library):
    """The real path: the app restarted, so the transcript comes off disk."""
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2]), paths, display_names=names,
            on_part_complete=checkpoints.append,
        )
    entry = library.save(checkpoints[-1], title="Interrupted")

    reloaded = library.load(entry.id).transcript
    final = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:], resume_from=reloaded
    )

    assert final.is_complete
    assert len(final.parts) == 3
    assert final.duration == pytest.approx(3 * PART_SECONDS)


# --------------------------------------------------------------------------- #
# The builder itself
# --------------------------------------------------------------------------- #


def test_an_empty_builder_builds_an_empty_transcript():
    assert TranscriptBuilder().build().segments == []


def test_resuming_an_empty_transcript_starts_at_zero():
    builder = TranscriptBuilder.resuming(Transcript())
    assert builder.offset == 0.0 and builder.part_index == 0


def test_a_builder_resumed_from_a_transcript_without_duration_uses_the_last_segment():
    """Defensive: a transcript saved by an older version has no duration set,
    and resuming at offset zero would overwrite it."""
    partial = Transcript(
        segments=[Segment(index=0, start=0.0, end=90.0, text="x")], duration=0.0
    )
    assert TranscriptBuilder.resuming(partial).offset == 90.0


def test_all_parts_silent_still_raises(parts):
    names, paths = parts
    with pytest.raises(TranscriptionError):
        transcribe_parts(
            DyingWhisper(silent=set(names)), paths, display_names=names
        )


# --------------------------------------------------------------------------- #
# Telemetry — the evidence a silent death leaves behind
# --------------------------------------------------------------------------- #
#
# Once memory was ruled out as the cause of one deployment's failures, the
# question became "then what?", and there was nothing recorded to answer it.
# These figures are captured per part so that a run which dies with no error
# still leaves numbers that discriminate between the remaining explanations.


def test_each_part_records_memory_and_timing(parts):
    names, paths = parts
    result = transcribe_parts(DyingWhisper(), paths, display_names=names)

    for part in result.parts:
        assert part.finished_at, "a part with no timestamp cannot be placed in a run"
        assert part.elapsed_seconds >= 0.0


def test_telemetry_survives_the_checkpoint_and_a_reload(parts, library):
    """Useless unless it outlives the process it describes."""
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2]), paths, display_names=names,
            on_part_complete=checkpoints.append,
        )

    entry = library.save(checkpoints[-1], title="Interrupted")
    reloaded = library.load(entry.id).transcript

    assert len(reloaded.parts) == 2
    for part in reloaded.parts:
        assert part.finished_at
        assert part.elapsed_seconds >= 0.0


def test_memory_is_recorded_when_the_platform_reports_it(parts, monkeypatch):
    import src.transcribe as tr

    monkeypatch.setattr(tr, "process_memory_gb", lambda: 1.25)
    result = transcribe_parts(DyingWhisper(), parts[1], display_names=parts[0])
    assert [p.memory_gb for p in result.parts] == [1.25, 1.25, 1.25]


def test_an_unreadable_memory_figure_does_not_break_the_run(parts, monkeypatch):
    """macOS and Windows have no /proc; transcription must not care."""
    import src.transcribe as tr

    monkeypatch.setattr(tr, "process_memory_gb", lambda: None)
    result = transcribe_parts(DyingWhisper(), parts[1], display_names=parts[0])
    assert len(result.parts) == 3
    assert all(p.memory_gb == 0.0 for p in result.parts)


def test_resuming_keeps_the_telemetry_of_the_earlier_parts(parts):
    """Otherwise finishing a lecture would erase the record of why it broke."""
    names, paths = parts
    checkpoints: list[Transcript] = []
    with pytest.raises(SystemExit):
        transcribe_parts(
            DyingWhisper(dies_on=names[2]), paths, display_names=names,
            on_part_complete=checkpoints.append,
        )
    before = [p.finished_at for p in checkpoints[-1].parts]

    resumed = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:],
        resume_from=checkpoints[-1],
    )
    assert [p.finished_at for p in resumed.parts[:2]] == before


# --------------------------------------------------------------------------- #
# One part per run
# --------------------------------------------------------------------------- #
#
# The final diagnosis on the reported deployment: memory flat, file fine, model
# fitting comfortably — but a three-part lecture died every time on part 3, and
# part 3 alone succeeded. What failed was the length of a single run. So the app
# now hands over one part per script execution, and `remaining_after` is what
# keeps a lecture whole while that happens.


def one_part_at_a_time(model, paths, names, library, title="Split run"):
    """Drive the whole lecture the way the app does: one call per part."""
    entry_id = None
    for index in range(len(paths)):
        resume = library.load(entry_id).transcript if entry_id else None
        transcript = transcribe_parts(
            model,
            [paths[index]],
            display_names=[names[index]],
            resume_from=resume,
            remaining_after=names[index + 1 :],
        )
        entry_id = library.save(transcript, title=title, entry_id=entry_id).id
    return library.load(entry_id).transcript


def test_a_single_part_call_does_not_claim_the_lecture_is_finished(parts):
    """Without remaining_after, transcribing part 1 of 3 alone would write
    pending_parts=[] and the other two would be silently forgotten."""
    names, paths = parts
    first = transcribe_parts(
        DyingWhisper(), paths[:1], display_names=names[:1],
        remaining_after=names[1:],
    )
    assert first.pending_parts == names[1:]
    assert not first.is_complete


def test_the_last_part_completes_the_lecture(parts):
    names, paths = parts
    last = transcribe_parts(
        DyingWhisper(), paths[2:], display_names=names[2:], remaining_after=[]
    )
    assert last.is_complete


def test_one_part_per_run_produces_the_same_lecture_as_one_long_run(parts, library):
    """The property that makes this safe to adopt: splitting the *runs* must not
    change the transcript."""
    names, paths = parts
    stepwise = one_part_at_a_time(DyingWhisper(), paths, names, library)
    in_one_go = transcribe_parts(DyingWhisper(), paths, display_names=names)

    assert stepwise.text == in_one_go.text
    assert stepwise.duration == pytest.approx(in_one_go.duration)
    assert [p.offset for p in stepwise.parts] == [p.offset for p in in_one_go.parts]
    assert [s.start for s in stepwise.segments] == [
        s.start for s in in_one_go.segments
    ]
    assert [s.part for s in stepwise.segments] == [s.part for s in in_one_go.segments]
    assert stepwise.is_complete


def test_each_run_transcribes_exactly_one_file(parts, library):
    """The whole point — no run may be longer than a single part."""
    names, paths = parts
    model = DyingWhisper()
    one_part_at_a_time(model, paths, names, library)
    assert model.calls == names, "each file read once, in order"


def test_the_lecture_is_recoverable_between_every_run(parts, library):
    """Stopping between runs — a closed laptop — must lose nothing."""
    names, paths = parts
    entry_id = None
    for index in range(2):  # the user walks away after two parts
        resume = library.load(entry_id).transcript if entry_id else None
        transcript = transcribe_parts(
            DyingWhisper(), [paths[index]], display_names=[names[index]],
            resume_from=resume, remaining_after=names[index + 1 :],
        )
        entry_id = library.save(transcript, entry_id=entry_id).id

    entry = library.entries()[0]
    assert not entry.is_complete
    assert entry.pending_parts == [names[2]]
    assert library.load(entry_id).transcript.word_count > 0


def test_a_single_file_lecture_needs_no_special_casing(parts, library):
    names, paths = parts
    only = transcribe_parts(
        DyingWhisper(), paths[:1], display_names=names[:1], remaining_after=[]
    )
    assert only.is_complete and not only.is_multipart
