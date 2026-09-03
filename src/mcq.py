"""Multiple-choice question generation, validation, and repair.

Generation is the easy half. The valuable half is what happens after: LLMs
reliably produce items with defects that a human item-writer would catch —
the correct answer is the longest option, two options say the same thing,
"all of the above" sneaks in, every answer lands on C. This module measures
those defects, fixes what can be fixed mechanically, and flags the rest for
the instructor rather than pretending the output is ready to deploy.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Callable

from . import prompts
from .llm import LLMClient
from .schema import MCQ, Chunk, format_timestamp, parse_timestamp

ProgressFn = Callable[[float, str], None]

BANNED_OPTION_PATTERNS = [
    r"^all of the above",
    r"^none of the above",
    r"^both [a-f] and [a-f]",
    r"^a and b\b",
    r"^any of the above",
]

NEGATIVE_STEM_PATTERNS = [r"\bNOT\b", r"\bEXCEPT\b", r"\bnever\b", r"\bincorrect\b"]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _build_avoid_clause(topics: list[str], stems: list[str]) -> str:
    """Tell the model what has already been asked, so it writes something new."""
    parts: list[str] = []
    if topics:
        parts.append(
            "Do not duplicate these topics already covered: " + "; ".join(topics[-12:])
        )
    if stems:
        listed = "\n".join(f"- {s}" for s in stems[-25:])
        parts.append(
            "These questions already exist. Write questions that test DIFFERENT "
            "content or a different aspect of the same content — not a reworded "
            "version of any of them:\n" + listed
        )
    return "\n\n".join(parts)


def generate_questions(
    client: LLMClient,
    chunks: list[Chunk],
    allocation: list[int],
    n_options: int = 4,
    bloom_targets: list[str] | None = None,
    difficulty_mix: str = "Balanced",
    course_context: str = "",
    avoid_stems: list[str] | None = None,
    progress: ProgressFn | None = None,
) -> list[MCQ]:
    """Generate questions chunk by chunk, avoiding repeats across chunks.

    ``avoid_stems`` carries questions that already exist — from earlier chunks in
    this run, and from a previous run when the instructor asks for an alternative
    set. Without it, a second pass over the same lecture reliably reproduces the
    first pass, because the salient points of a chunk are the salient points
    whichever time you ask.
    """
    bloom_targets = bloom_targets or ["Remember", "Understand", "Apply", "Analyze"]
    options_placeholder = ", ".join(f'"option {chr(65 + i)}"' for i in range(n_options))
    context_block = (
        f"Course context from the instructor:\n{course_context.strip()}\n"
        if course_context.strip()
        else ""
    )

    questions: list[MCQ] = []
    seen_topics: list[str] = []
    prior_stems = list(avoid_stems or [])

    for chunk, want in zip(chunks, allocation):
        if want <= 0:
            continue
        if progress:
            done = len(questions)
            progress(
                min(0.95, done / max(1, sum(allocation))),
                f"Writing questions for {chunk.label}",
            )

        avoid = _build_avoid_clause(seen_topics, prior_stems)

        try:
            data = client.complete_json(
                prompts.MCQ_SYSTEM.format(n_options=n_options),
                prompts.MCQ_USER.format(
                    n=want,
                    label=chunk.label,
                    course_context=context_block,
                    bloom_targets=", ".join(bloom_targets),
                    difficulty_mix=difficulty_mix,
                    avoid_clause=avoid,
                    options_placeholder=options_placeholder,
                    text=_with_inline_timestamps(chunk)[:24000],
                ),
                max_tokens=max(2000, want * 700),
            )
        except Exception as exc:
            if progress:
                progress(0.95, f"Skipped {chunk.label}: {exc}")
            continue

        for raw in data.get("questions", []) or []:
            item = _coerce_mcq(raw, fallback_timestamp=format_timestamp(chunk.start))
            if item is None:
                continue
            questions.append(item)
            prior_stems.append(item.stem)
            if item.topic:
                seen_topics.append(item.topic)

    if progress:
        progress(1.0, f"Drafted {len(questions)} questions")
    return questions


def find_chunk_for_timestamp(chunks: list[Chunk], timestamp: str) -> Chunk | None:
    """Locate the transcript window a question came from."""
    if not chunks:
        return None
    seconds = parse_timestamp(timestamp)
    for chunk in chunks:
        if chunk.start <= seconds < chunk.end:
            return chunk
    # A malformed or out-of-range timestamp still deserves an answer.
    return min(chunks, key=lambda c: abs(c.start - seconds))


def generate_replacement(
    client: LLMClient,
    chunks: list[Chunk],
    question: MCQ,
    existing_stems: list[str],
    n_options: int = 4,
    bloom_targets: list[str] | None = None,
    difficulty_mix: str = "Balanced",
    course_context: str = "",
    same_section: bool = True,
) -> MCQ | None:
    """Write one new question to stand in for ``question``.

    By default the replacement is drawn from the same part of the lecture, so
    swapping out a bad item does not quietly leave a hole in the coverage. Set
    ``same_section=False`` to draw from anywhere in the recording instead.
    """
    if not chunks:
        return None

    if same_section:
        chunk = find_chunk_for_timestamp(chunks, question.source_timestamp)
        pool = [chunk] if chunk else chunks[:1]
    else:
        pool = chunks

    avoid = [s for s in existing_stems if s]
    for chunk in pool:
        drafted = generate_questions(
            client,
            [chunk],
            [1],
            n_options=n_options,
            bloom_targets=bloom_targets or [question.bloom],
            difficulty_mix=difficulty_mix,
            course_context=course_context,
            avoid_stems=avoid,
        )
        if drafted:
            new = drafted[0]
            new.points = question.points
            new.flags = validate_question(new)
            return new
    return None


def _with_inline_timestamps(chunk: Chunk) -> str:
    """Prefix the chunk with its window so the model cites plausible times."""
    return f"[segment covers {chunk.label}]\n{chunk.text}"


def _coerce_mcq(raw: dict, fallback_timestamp: str = "") -> MCQ | None:
    """Build an MCQ from loose model output, or return None if unusable."""
    try:
        options = [str(o).strip() for o in raw.get("options", []) if str(o).strip()]
        stem = str(raw.get("stem", "")).strip()
        if not stem or len(options) < 3:
            return None

        idx = raw.get("correct_index", raw.get("answer_index", 0))
        if isinstance(idx, str):
            letter = idx.strip().upper()
            idx = ord(letter[0]) - 65 if letter[:1].isalpha() else int(re.sub(r"\D", "", letter) or 0)
        idx = int(idx)
        if not 0 <= idx < len(options):
            return None

        rationales = [str(r) for r in raw.get("distractor_rationales", []) or []]
        rationales = (rationales + [""] * len(options))[: len(options)]

        bloom = str(raw.get("bloom", "Understand")).strip().title()
        if bloom not in {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}:
            bloom = "Understand"
        difficulty = str(raw.get("difficulty", "Medium")).strip().title()
        if difficulty not in {"Easy", "Medium", "Hard"}:
            difficulty = "Medium"

        return MCQ(
            stem=stem,
            options=options,
            correct_index=idx,
            rationale=str(raw.get("rationale", "")).strip(),
            distractor_rationales=rationales,
            bloom=bloom,  # type: ignore[arg-type]
            difficulty=difficulty,  # type: ignore[arg-type]
            topic=str(raw.get("topic", "")).strip(),
            source_timestamp=str(raw.get("source_timestamp", "")).strip() or fallback_timestamp,
            source_quote=str(raw.get("source_quote", "")).strip(),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_question(q: MCQ) -> list[str]:
    """Return a list of human-readable defects. Empty means clean."""
    flags: list[str] = []
    opts = q.options

    for pattern in BANNED_OPTION_PATTERNS:
        if any(re.match(pattern, o.strip(), re.IGNORECASE) for o in opts):
            flags.append("Uses an 'all/none of the above' style option")
            break

    normalized = [re.sub(r"[^a-z0-9]+", " ", o.lower()).strip() for o in opts]
    if len(set(normalized)) < len(normalized):
        flags.append("Two or more options are duplicates")

    lengths = [len(o) for o in opts]
    others = [n for i, n in enumerate(lengths) if i != q.correct_index]
    if others and lengths[q.correct_index] > 1.6 * (sum(others) / len(others)):
        flags.append("Correct answer is conspicuously longer than the distractors")

    for pattern in NEGATIVE_STEM_PATTERNS:
        if re.search(pattern, q.stem):
            flags.append("Negatively phrased stem (NOT/EXCEPT)")
            break

    if len(q.stem.split()) < 5:
        flags.append("Stem may be too short to pose a complete problem")
    if len(q.stem) > 400:
        flags.append("Stem is unusually long")

    if not q.source_quote:
        flags.append("No supporting quote from the transcript")
    if not q.rationale:
        flags.append("No rationale for the correct answer")

    if any(re.search(r"\b(always|never)\b", o, re.IGNORECASE) for o in opts):
        flags.append("An option uses an absolute qualifier (always/never)")

    stem_words = set(re.findall(r"[a-z]{5,}", q.stem.lower()))
    correct_words = set(re.findall(r"[a-z]{5,}", q.correct_option.lower()))
    distractor_words: set[str] = set()
    for i, o in enumerate(opts):
        if i != q.correct_index:
            distractor_words |= set(re.findall(r"[a-z]{5,}", o.lower()))
    give_away = (stem_words & correct_words) - distractor_words
    if give_away:
        flags.append(
            "Stem repeats wording found only in the correct answer: "
            + ", ".join(sorted(give_away)[:3])
        )

    return flags


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "best", "but", "by", "does", "do",
    "following", "for", "from", "has", "have", "how", "in", "is", "it", "its",
    "most", "of", "on", "or", "that", "the", "their", "these", "this", "to",
    "was", "were", "what", "when", "which", "who", "why", "will", "with",
}

DUPLICATE_THRESHOLD = 0.7


def _content_tokens(text: str) -> set[str]:
    words = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).split()
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_all(questions: list[MCQ]) -> list[MCQ]:
    """Attach flags in place, including cross-item near-duplicate detection.

    Duplicates are compared on content-word overlap rather than exact text,
    because the same question asked twice in slightly different words is the
    failure mode that actually shows up when several chunks cover related
    material.
    """
    for q in questions:
        q.flags = validate_question(q)

    tokens = [_content_tokens(q.stem) for q in questions]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if _jaccard(tokens[i], tokens[j]) >= DUPLICATE_THRESHOLD:
                note = f"Near-duplicate of question {j + 1}"
                if note not in questions[i].flags:
                    questions[i].flags.append(note)
                note_j = f"Near-duplicate of question {i + 1}"
                if note_j not in questions[j].flags:
                    questions[j].flags.append(note_j)
    return questions


def balance_answer_positions(questions: list[MCQ], seed: int | None = 17) -> list[MCQ]:
    """Shuffle options so correct answers are spread evenly across positions.

    LLMs have a strong positional bias — left alone, a generated set often puts
    60%+ of correct answers in the same slot, which students notice within one
    semester of using the tool.
    """
    if not questions:
        return questions
    rng = random.Random(seed)
    n_slots = min(len(q.options) for q in questions)

    targets: list[int] = []
    while len(targets) < len(questions):
        block = list(range(n_slots))
        rng.shuffle(block)
        targets.extend(block)
    targets = targets[: len(questions)]

    for q, target in zip(questions, targets):
        target = min(target, len(q.options) - 1)
        if target == q.correct_index:
            continue
        opts = list(q.options)
        rats = list(q.distractor_rationales) or [""] * len(opts)
        rats = (rats + [""] * len(opts))[: len(opts)]
        opts[q.correct_index], opts[target] = opts[target], opts[q.correct_index]
        rats[q.correct_index], rats[target] = rats[target], rats[q.correct_index]
        q.options = opts
        q.distractor_rationales = rats
        q.correct_index = target
    return questions


def answer_distribution(questions: list[MCQ]) -> dict[str, int]:
    return dict(Counter(q.answer_letter for q in questions))


def coverage_report(questions: list[MCQ]) -> dict[str, dict[str, int]]:
    return {
        "bloom": dict(Counter(q.bloom for q in questions)),
        "difficulty": dict(Counter(q.difficulty for q in questions)),
        "answer_position": answer_distribution(questions),
    }


# --------------------------------------------------------------------------- #
# Optional second-pass critique
# --------------------------------------------------------------------------- #

def critique_and_revise(
    client: LLMClient, questions: list[MCQ], progress: ProgressFn | None = None
) -> tuple[list[MCQ], list[dict]]:
    """Have the model review its own items and apply the revisions it proposes.

    A separate pass with a reviewer persona catches a meaningful share of the
    defects a single generate call leaves behind — it is the cheapest quality
    improvement available here. Items marked "drop" are kept but unchecked, so
    the instructor sees what was rejected and why.
    """
    if not questions:
        return questions, []

    payload = [
        {
            "id": q.id,
            "stem": q.stem,
            "options": q.options,
            "correct_index": q.correct_index,
            "source_quote": q.source_quote,
        }
        for q in questions
    ]

    if progress:
        progress(0.2, "Reviewing draft questions")

    try:
        data = client.complete_json(
            prompts.CRITIQUE_SYSTEM,
            prompts.CRITIQUE_USER.format(
                questions_json=json.dumps(payload, indent=1)[:40000]
            ),
            max_tokens=max(3000, len(questions) * 400),
        )
    except Exception:
        return questions, []

    by_id = {q.id: q for q in questions}
    reviews = data.get("reviews", []) or []

    for review in reviews:
        q = by_id.get(str(review.get("id", "")))
        if q is None:
            continue
        verdict = str(review.get("verdict", "keep")).lower()
        issues = [str(i) for i in review.get("issues", []) or []]

        if verdict == "drop":
            q.include = False
            q.flags = list(dict.fromkeys(q.flags + ["Reviewer recommended dropping"] + issues))
        elif verdict == "revise":
            revised = review.get("revised") or {}
            new_opts = [str(o).strip() for o in revised.get("options", []) if str(o).strip()]
            new_stem = str(revised.get("stem", "")).strip()
            new_idx = revised.get("correct_index", q.correct_index)
            try:
                new_idx = int(new_idx)
            except (TypeError, ValueError):
                new_idx = q.correct_index
            if new_stem and len(new_opts) >= 3 and 0 <= new_idx < len(new_opts):
                q.stem = new_stem
                q.options = new_opts
                q.correct_index = new_idx
                q.distractor_rationales = ([""] * len(new_opts))
                if revised.get("rationale"):
                    q.rationale = str(revised["rationale"])
                q.flags = list(dict.fromkeys(q.flags + [f"Revised by reviewer: {i}" for i in issues]))

    if progress:
        progress(1.0, "Review complete")

    return list(by_id.values()), reviews
