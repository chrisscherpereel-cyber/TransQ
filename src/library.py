"""A saved lecture library, per account.

Transcripts used to live only in ``st.session_state`` — which means a browser
refresh, an idle sign-out, or a Streamlit restart threw away twenty minutes of
transcription. Nothing was ever written to the store. This module fixes that.

What gets saved is the whole working state, not just the text: the transcript,
the summary, and every question set generated from it. Reloading a lecture
therefore puts you back exactly where you left off, rather than handing back the
transcript and making you pay for the summary and questions again.

**On encryption.** Entries are stored through the normal :class:`~src.storage.Store`
path, so they are encrypted at rest under the app key like every other document.
They are deliberately *not* sealed to the user's password the way personal API
keys are. A key is trivially re-entered if it becomes unreadable; a semester of
transcripts is not, and an administrator password reset would destroy them. If a
deployment's threat model needs transcripts unreadable without the user present —
recordings of class discussion, say — that is a one-line change to encrypt the
``payload`` field with the session data key, at the cost of losing them on reset.

Layout, one document per lecture so a save never rewrites the whole library:

    library/<username>/index          metadata for the list view
    library/<username>/t/<id>         the full record
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .schema import Quiz, Summary, Transcript, format_timestamp
from .storage import Store, StorageError

INDEX_SUFFIX = "index"
RECORD_PREFIX = "t"

# A soft ceiling. Nothing is deleted automatically — silently discarding
# somebody's saved work is worse than telling them the shelf is full.
SOFT_LIMIT = 200


class LibraryError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class LibraryEntry:
    """What the list view needs, without loading the transcript itself."""

    id: str
    title: str
    source_filename: str = ""
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    duration: float = 0.0
    word_count: int = 0
    parts: int = 1
    language: str = "en"
    origin: str = ""          # "audio" | the transcript-import description
    has_summary: bool = False
    question_sets: int = 0
    questions: int = 0
    material_filename: str = ""
    exam_topics: int = 0
    # Filenames still to transcribe. Non-empty means this lecture was
    # checkpointed partway through a split recording, so the list view can offer
    # to finish it — and, just as importantly, will not present a truncated
    # lecture as though it were whole.
    pending_parts: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.pending_parts

    @property
    def length_label(self) -> str:
        return format_timestamp(self.duration) if self.duration else "—"

    @property
    def saved_label(self) -> str:
        return self.updated_at.replace("T", " ")[:16]

    @property
    def progress_label(self) -> str:
        done = self.parts
        return f"{done} of {done + len(self.pending_parts)} parts transcribed"

    @property
    def summary_label(self) -> str:
        if self.pending_parts:
            noun = "part" if len(self.pending_parts) == 1 else "parts"
            return (
                f"incomplete — {self.progress_label}, "
                f"{len(self.pending_parts)} {noun} still to do"
            )
        if not self.has_summary:
            return "transcript only"
        if self.question_sets:
            noun = "set" if self.question_sets == 1 else "sets"
            return f"summary · {self.question_sets} question {noun} ({self.questions} questions)"
        return "summary"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LibraryEntry":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SavedLecture:
    """A whole working state, restored."""

    entry: LibraryEntry
    transcript: Transcript
    summary: Summary | None = None
    quizzes: list[Quiz] = field(default_factory=list)
    # The slides the lecture was built from, and what the instructor said would
    # be examined. Saved with the lecture because re-uploading a deck and
    # retyping a topic list to regenerate one question set is exactly the
    # friction that stops people regenerating.
    material: Any = None
    exam_topics: list[str] = field(default_factory=list)
    review: Any = None


class TranscriptLibrary:
    """Saved lectures for one account."""

    def __init__(self, store: Store, username: str):
        if not username:
            raise LibraryError("A library belongs to an account.")
        self.store = store
        self.username = username

    # -- paths -- #

    @property
    def _root(self) -> str:
        return f"library/{self.username}"

    def _index_path(self) -> str:
        return f"{self._root}/{INDEX_SUFFIX}"

    def _record_path(self, entry_id: str) -> str:
        return f"{self._root}/{RECORD_PREFIX}/{entry_id}"

    # -- index -- #

    def entries(self) -> list[LibraryEntry]:
        """Newest first — the one you just made is the one you want."""
        document = self.store.read(self._index_path()) or {}
        rows = document.get("entries")
        if not isinstance(rows, list):
            return []
        parsed = [LibraryEntry.from_dict(r) for r in rows if isinstance(r, dict)]
        return sorted(parsed, key=lambda e: e.updated_at, reverse=True)

    def _write_index(self, entries: list[LibraryEntry]) -> None:
        self.store.write(
            self._index_path(),
            {"version": 1, "entries": [asdict(e) for e in entries]},
        )

    def _upsert(self, entry: LibraryEntry) -> None:
        entries = [e for e in self.entries() if e.id != entry.id]
        entries.append(entry)
        self._write_index(entries)

    def get_entry(self, entry_id: str) -> LibraryEntry | None:
        return next((e for e in self.entries() if e.id == entry_id), None)

    def count(self) -> int:
        return len(self.entries())

    # -- saving -- #

    def save(
        self,
        transcript: Transcript,
        title: str = "",
        summary: Summary | None = None,
        quizzes: list[Quiz] | None = None,
        origin: str = "",
        entry_id: str | None = None,
        material: Any = None,
        exam_topics: list[str] | None = None,
        review: Any = None,
    ) -> LibraryEntry:
        """Create or replace a saved lecture. Returns its index entry."""
        if transcript is None or not transcript.segments:
            raise LibraryError("There is no transcript to save.")

        existing = self.get_entry(entry_id) if entry_id else None
        if entry_id and existing is None:
            # The id refers to something deleted elsewhere; make a new one rather
            # than silently writing an orphan the index will never show.
            entry_id = None

        quizzes = quizzes or []
        entry = LibraryEntry(
            id=entry_id or uuid.uuid4().hex[:12],
            title=(title or (existing.title if existing else "") or _default_title(transcript)).strip(),
            source_filename=transcript.parts[0].filename if transcript.parts else "",
            created_at=existing.created_at if existing else now(),
            updated_at=now(),
            duration=transcript.duration,
            word_count=transcript.word_count,
            parts=len(transcript.parts) or 1,
            language=transcript.language,
            origin=origin or (existing.origin if existing else ""),
            has_summary=summary is not None,
            question_sets=len(quizzes),
            questions=sum(len(q.included) for q in quizzes),
            pending_parts=list(transcript.pending_parts),
            material_filename=getattr(material, "filename", "") or "",
            exam_topics=len(exam_topics or []),
        )

        # Record first, index second: an index row pointing at nothing is worse
        # than a record nothing points at yet.
        self.store.write(
            self._record_path(entry.id),
            {
                "version": 1,
                "entry": asdict(entry),
                "transcript": transcript.model_dump(),
                "summary": summary.model_dump() if summary is not None else None,
                "quizzes": [q.model_dump() for q in quizzes],
                "material": material.to_dict() if material is not None else None,
                "exam_topics": list(exam_topics or []),
                "review": review.to_dict() if review is not None else None,
            },
        )
        self._upsert(entry)
        return entry

    def rename(self, entry_id: str, title: str) -> LibraryEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise LibraryError("That lecture is no longer in your library.")
        title = title.strip()
        if not title:
            raise LibraryError("A title is required.")
        entry.title = title
        entry.updated_at = now()
        self._upsert(entry)

        document = self.store.read(self._record_path(entry_id))
        if document:
            document["entry"] = asdict(entry)
            self.store.write(self._record_path(entry_id), document)
        return entry

    # -- loading -- #

    def load(self, entry_id: str) -> SavedLecture:
        document = self.store.read(self._record_path(entry_id))
        if not document:
            raise LibraryError(
                "That lecture could not be read — it may have been deleted."
            )
        try:
            transcript = Transcript.model_validate(document["transcript"])
            summary = (
                Summary.model_validate(document["summary"])
                if document.get("summary")
                else None
            )
            quizzes = [Quiz.model_validate(q) for q in document.get("quizzes") or []]
        except Exception as exc:
            raise LibraryError(f"That saved lecture could not be read back: {exc}") from exc

        material = None
        if document.get("material"):
            try:
                from .materials import Material

                material = Material.from_dict(document["material"])
            except Exception:  # noqa: BLE001 - a bad deck must not block a reopen
                material = None

        review = None
        if document.get("review"):
            try:
                from .factcheck import ReviewResult

                review = ReviewResult.from_dict(document["review"])
            except Exception:  # noqa: BLE001 - never block a reopen
                review = None

        entry = LibraryEntry.from_dict(document.get("entry") or {"id": entry_id, "title": ""})
        return SavedLecture(
            entry=entry,
            transcript=transcript,
            summary=summary,
            quizzes=quizzes,
            material=material,
            exam_topics=list(document.get("exam_topics") or []),
            review=review,
        )

    def delete(self, entry_id: str) -> None:
        self._write_index([e for e in self.entries() if e.id != entry_id])
        # Tombstone rather than a missing file, so a backend without delete
        # (Dropbox app folders, notably) does not leave readable content behind.
        try:
            self.store.write(self._record_path(entry_id), {"version": 1, "deleted": True})
        except StorageError:
            pass


def _default_title(transcript: Transcript) -> str:
    """A name to show before anyone has typed one."""
    if transcript.parts and transcript.parts[0].filename:
        stem = transcript.parts[0].filename.rsplit(".", 1)[0]
        stem = re.sub(r"[_-]+", " ", stem).strip()
        if stem:
            return stem[:80]
    return f"Lecture ({format_timestamp(transcript.duration)})"
