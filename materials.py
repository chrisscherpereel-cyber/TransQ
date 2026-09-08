"""Reading the slides and handouts a lecture was actually built from.

A transcript is what was *said*. It is a poor record of what was *shown*: the
term spelled out on a slide arrives in the audio as "this thing here", the
four-item framework is spoken as "these", and the numbers in a table are never
read aloud at all. Whisper transcribes a gesture at a diagram as silence.

So the deck is not decoration — it is the half of the lecture that survives
badly. Given it, the summary can name what was on screen, and questions can use
the exact terminology a student saw rather than the pronoun the instructor used.

Design decisions worth stating:

**Sections, not one blob.** A deck arrives as titled slides, a PDF as pages, a
document as headed sections. Keeping that structure is what lets a ten-minute
transcript window be matched to the two or three slides that belong to it,
rather than every question being handed the whole deck.

**Extraction never raises.** A file that cannot be read is reported, not fatal.
Losing an upload should cost you the upload, not the lecture — and a corrupt
deck arriving as an exception at the top of the pipeline would take the
transcript with it.

**Speaker notes count.** They are often where the argument actually lives, and
they are invisible in the room, so they are exactly the material a transcript is
least likely to duplicate.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# A section shorter than this carries a title and nothing else — a section
# divider, a "Questions?" slide. Keeping them dilutes every relevance match.
MIN_SECTION_CHARS = 25

# Guards against a 300-slide deck crowding the transcript out of the prompt.
MAX_SECTIONS = 400
MAX_SECTION_CHARS = 4000

SUPPORTED = {
    "pptx": "PowerPoint",
    "pdf": "PDF",
    "docx": "Word",
    "txt": "text",
    "md": "Markdown",
}


class MaterialError(RuntimeError):
    """The file could not be read at all."""


@dataclass
class MaterialSection:
    """One slide, page, or headed section."""

    index: int
    title: str = ""
    text: str = ""
    notes: str = ""

    @property
    def label(self) -> str:
        name = self.title.strip() or "(untitled)"
        return f"{self.index + 1}. {name}"

    @property
    def full_text(self) -> str:
        parts = [self.title.strip(), self.text.strip()]
        if self.notes.strip():
            parts.append(f"Speaker notes: {self.notes.strip()}")
        return "\n".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialSection":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Material:
    """One uploaded supporting file, parsed into sections."""

    filename: str = ""
    kind: str = ""
    sections: list[MaterialSection] = field(default_factory=list)

    @property
    def section_noun(self) -> str:
        return {"PowerPoint": "slide", "PDF": "page"}.get(self.kind, "section")

    @property
    def text(self) -> str:
        return "\n\n".join(s.full_text for s in self.sections)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def outline(self, limit: int = 60) -> str:
        """Titles only — what the deck covered, cheap enough to always include."""
        return "\n".join(f"- {s.label}" for s in self.sections[:limit])

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "kind": self.kind,
            "sections": [asdict(s) for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Material":
        return cls(
            filename=str(data.get("filename") or ""),
            kind=str(data.get("kind") or ""),
            sections=[
                MaterialSection.from_dict(s)
                for s in (data.get("sections") or [])
                if isinstance(s, dict)
            ],
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_material(filename: str, blob: bytes) -> Material:
    """Parse an uploaded file into titled sections.

    Raises :class:`MaterialError` only when nothing at all could be read —
    a caller that wants to keep going on a bad file catches this one type.
    """
    extension = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    if extension not in SUPPORTED:
        raise MaterialError(
            f"{filename}: {extension or 'this file type'} is not supported. "
            "Use PowerPoint (.pptx), PDF, Word (.docx), or plain text."
        )

    readers = {
        "pptx": _read_pptx,
        "pdf": _read_pdf,
        "docx": _read_docx,
        "txt": _read_text,
        "md": _read_text,
    }
    try:
        sections = readers[extension](blob)
    except MaterialError:
        raise
    except Exception as exc:  # noqa: BLE001 - the file, not the app, is at fault
        raise MaterialError(f"{filename} could not be read: {exc}") from exc

    sections = _tidy(sections)
    if not sections:
        raise MaterialError(
            f"{filename} contained no readable text. A deck of images, or a "
            "scanned PDF, needs OCR before it can be used here."
        )
    return Material(filename=filename, kind=SUPPORTED[extension], sections=sections)


def _tidy(sections: list[MaterialSection]) -> list[MaterialSection]:
    """Drop the empties, cap the extremes, renumber what survives."""
    kept: list[MaterialSection] = []
    for section in sections:
        if len(section.full_text) < MIN_SECTION_CHARS:
            continue
        section.text = section.text[:MAX_SECTION_CHARS]
        section.notes = section.notes[:MAX_SECTION_CHARS]
        section.index = len(kept)
        kept.append(section)
        if len(kept) >= MAX_SECTIONS:
            break
    return kept


def _read_pptx(blob: bytes) -> list[MaterialSection]:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover
        raise MaterialError("pip install python-pptx to read PowerPoint files") from exc

    deck = Presentation(io.BytesIO(blob))
    sections: list[MaterialSection] = []

    for i, slide in enumerate(deck.slides):
        title, body = "", []
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                # Tables carry the numbers a lecture talks around without
                # reading out, so they are worth more than the text beside them.
                if getattr(shape, "has_table", False):
                    body.append(_table_text(shape.table))
                continue
            text = shape.text_frame.text.strip()
            if not text:
                continue
            if not title and _is_title(shape, slide):
                title = text.splitlines()[0][:200]
            else:
                body.append(text)

        notes = ""
        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()

        sections.append(
            MaterialSection(index=i, title=title, text="\n".join(body), notes=notes)
        )
    return sections


def _is_title(shape: Any, slide: Any) -> bool:
    try:
        return shape == slide.shapes.title
    except Exception:  # pragma: no cover - layouts without a title placeholder
        return False


def _table_text(table: Any) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _read_pdf(blob: bytes) -> list[MaterialSection]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise MaterialError("pip install pypdf to read PDF files") from exc

    reader = PdfReader(io.BytesIO(blob))
    sections: list[MaterialSection] = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - one bad page is not a bad document
            continue
        # A slide-style PDF puts the slide title on the first line; a prose PDF
        # gives us a first line that is merely the first line. Either is a more
        # useful label than "Page 7".
        first = text.splitlines()[0].strip() if text else ""
        sections.append(
            MaterialSection(
                index=i,
                title=first[:120] if 0 < len(first) <= 120 else f"Page {i + 1}",
                text=text,
            )
        )
    return sections


def _read_docx(blob: bytes) -> list[MaterialSection]:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise MaterialError("pip install python-docx to read Word files") from exc

    document = docx.Document(io.BytesIO(blob))
    sections: list[MaterialSection] = []
    title, body = "", []

    def flush() -> None:
        if title or body:
            sections.append(
                MaterialSection(index=len(sections), title=title, text="\n".join(body))
            )

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if (paragraph.style.name or "").startswith("Heading"):
            flush()
            title, body = text[:200], []
        else:
            body.append(text)
    flush()

    if not sections:  # a document with no headings at all
        whole = "\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
        sections = [MaterialSection(index=0, title="Document", text=whole)]
    return sections


def _read_text(blob: bytes) -> list[MaterialSection]:
    text = blob.decode("utf-8", errors="replace")
    # Split on Markdown headings when present; otherwise on blank-line blocks.
    if re.search(r"^#{1,6}\s+\S", text, re.MULTILINE):
        parts = re.split(r"^(#{1,6}\s+.*)$", text, flags=re.MULTILINE)
        sections: list[MaterialSection] = []
        heading = ""
        for part in parts:
            if re.match(r"^#{1,6}\s+", part or ""):
                heading = part.lstrip("#").strip()[:200]
            elif (part or "").strip():
                sections.append(
                    MaterialSection(
                        index=len(sections), title=heading, text=part.strip()
                    )
                )
        if sections:
            return sections

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [
        MaterialSection(index=i, title=b.splitlines()[0][:120], text=b)
        for i, b in enumerate(blocks)
    ]


# --------------------------------------------------------------------------- #
# Matching material to a moment in the lecture
# --------------------------------------------------------------------------- #


def relevant_sections(
    text: str, material: Material | None, limit: int = 3, floor: float = 0.04
) -> list[MaterialSection]:
    """The few sections that belong to this stretch of transcript.

    Handing every chunk the whole deck would work and be wrong: it triples the
    prompt, pushes the transcript toward the context limit — the truncation this
    app already fights — and invites questions about slides the instructor never
    reached in that ten minutes.

    Matching is on shared vocabulary, which is crude but well suited here,
    because a slide and the speech about it share exactly the thing that
    matters: the domain terms. ``floor`` keeps an unrelated slide out rather
    than always returning the least-bad three.
    """
    if material is None or not material.sections:
        return []

    from .chunking import _keywords  # local: chunking imports nothing from here

    wanted = _keywords(text)
    if not wanted:
        return []

    scored: list[tuple[float, MaterialSection]] = []
    for section in material.sections:
        tokens = _keywords(section.full_text)
        if not tokens:
            continue
        # Asymmetric on purpose: what fraction of the *slide* appears in the
        # speech. A slide covered thoroughly in this window scores high even if
        # the window also covers much else, which plain Jaccard would punish.
        score = len(wanted & tokens) / len(tokens)
        if score >= floor:
            scored.append((score, section))

    scored.sort(key=lambda pair: (-pair[0], pair[1].index))
    return [section for _, section in scored[:limit]]


def material_context(sections: list[MaterialSection], noun: str = "slide") -> str:
    """The prompt fragment for a set of matched sections."""
    if not sections:
        return ""
    body = "\n\n".join(f"[{noun} {s.label}]\n{s.full_text}" for s in sections)
    return (
        f"The instructor's own {noun}s for this part of the lecture are below. "
        "Use their exact terminology and any figures they give — the transcript "
        "often refers to what was on screen as \"this\" or \"here\", and the "
        f"{noun}s are where the term actually appears:\n\n"
        f"{body}\n\n"
    )
