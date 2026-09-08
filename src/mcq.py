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
from typing import Any

from . import prompts
from .chunking import allocate_by_importance, chunk_importance
from .diagnostics import (
    EMPTY,
    FAILED,
    OK,
    PHASE_GENERATE,
    PHASE_REPLACE,
    PHASE_REVIEW,
    SKIPPED,
    RunReport,
    classify,
    short_reason,
)
from .llm import LLMClient, LLMError, TruncatedResponseError
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


def _build_focus_clause(focus_points: list[str]) -> str:
    """Name what this lecture was for, so "important" is not left to the model.

    Without it, each chunk is judged on its own and the model has no way to know
    that the segment it is reading is a digression. The summary already worked
    this out for the whole lecture; passing it down is what makes a per-chunk
    call aim at the lecture's actual argument.
    """
    points = [str(p).strip() for p in focus_points if str(p).strip()]
    if not points:
        return ""
    listed = "\n".join(f"- {p}" for p in points[:12])
    return (
        "This lecture is meant to teach the following. Prioritise items that "
        "assess these; write nothing that merely fills a count:\n"
        f"{listed}\n\n"
    )


def generate_questions(
    client: LLMClient,
    chunks: list[Chunk],
    allocation: list[int],
    n_options: int = 4,
    bloom_targets: list[str] | None = None,
    difficulty_mix: str = "Balanced",
    course_context: str = "",
    avoid_stems: list[str] | None = None,
    focus_points: list[str] | None = None,
    progress: ProgressFn | None = None,
    report: RunReport | None = None,
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

    report = report or RunReport()
    focus_clause = _build_focus_clause(list(focus_points or []))
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

        def _ask(n: int, budget: int) -> dict:
            return client.complete_json(
                prompts.MCQ_SYSTEM.format(n_options=n_options),
                prompts.MCQ_USER.format(
                    n=n,
                    label=chunk.label,
                    course_context=context_block,
                    bloom_targets=", ".join(bloom_targets),
                    difficulty_mix=difficulty_mix,
                    focus_clause=focus_clause,
                    avoid_clause=avoid,
                    options_placeholder=options_placeholder,
                    text=_with_inline_timestamps(chunk)[:24000],
                ),
                max_tokens=budget,
            )

        try:
            try:
                data = _ask(want, max(2000, want * 700))
            except TruncatedResponseError as exc:
                # The model ran out of room mid-answer. Asking for half as many
                # with a bigger budget usually succeeds, and half a chunk's
                # questions beats none — the top-up rounds cover the rest.
                reduced = max(1, want // 2)
                report.record(
                    PHASE_GENERATE, chunk.label, SKIPPED,
                    detail=f"reply was cut off at {want} questions; retrying with {reduced}",
                    cause=classify(exc),
                )
                if progress:
                    progress(0.95, f"{chunk.label}: reply cut off, retrying smaller")
                data = _ask(reduced, max(3000, reduced * 1200))
                want = reduced
        except Exception as exc:
            # Was: a progress message that the next repaint erased. A failure
            # here is the single most likely reason a run "finishes" with
            # nothing, so it has to leave a durable trace.
            report.record(
                PHASE_GENERATE, chunk.label, FAILED,
                detail=short_reason(exc), cause=classify(exc),
            )
            if progress:
                progress(0.95, f"{chunk.label} failed: {short_reason(exc)}")
            continue

        produced_here = 0
        for raw in data.get("questions", []) or []:
            item = _coerce_mcq(raw, fallback_timestamp=format_timestamp(chunk.start))
            if item is None:
                continue
            questions.append(item)
            prior_stems.append(item.stem)
            produced_here += 1
            if item.topic:
                seen_topics.append(item.topic)

        report.record(
            PHASE_GENERATE, chunk.label,
            OK if produced_here else EMPTY,
            detail="" if produced_here else f"asked for {want}, got nothing usable",
            produced=produced_here,
        )

    if progress:
        progress(1.0, f"Drafted {len(questions)} questions")
    return questions


def _allocate_topup(
    shortfall: int, chunks: list[Chunk], offset: int, weights: list[float] | None = None
) -> list[int]:
    """Spread a shortfall round-robin over the chunks worth asking about.

    Rotating the starting point matters: always restarting at chunk 0 would make
    every top-up round hammer the opening minutes of the lecture, which is
    exactly the part already best covered.

    When importance weights are known, the rotation runs over the substantial
    windows only. Otherwise a lecture whose good material is in the middle gets
    topped up from its admin and its Q&A, which is where the padding used to
    come from.
    """
    if not chunks:
        return []
    eligible = list(range(len(chunks)))
    if weights and max(weights) > 0:
        threshold = max(weights) * 0.25
        preferred = [i for i, w in enumerate(weights) if w >= threshold]
        if preferred:
            eligible = preferred

    counts = [0] * len(chunks)
    for i in range(shortfall):
        counts[eligible[(offset + i) % len(eligible)]] += 1
    return counts


def generate_question_set(
    client: LLMClient,
    chunks: list[Chunk],
    target: int,
    n_options: int = 4,
    bloom_targets: list[str] | None = None,
    difficulty_mix: str = "Balanced",
    course_context: str = "",
    summary: Any = None,
    avoid_stems: list[str] | None = None,
    do_review: bool = True,
    max_rounds: int = 3,
    progress: ProgressFn | None = None,
    report: RunReport | None = None,
) -> tuple[list[MCQ], list[str]]:
    """Produce ``target`` usable questions — not "roughly target".

    A single generation pass reliably under-delivers, for three compounding
    reasons, none of which used to be corrected:

    1. Asked for 3 questions from a chunk, a model often returns 2.
    2. Malformed items (bad ``correct_index``, two options, truncated JSON) are
       dropped during parsing.
    3. The review pass then *drops* weak items, shrinking the set again.

    So the count is now enforced rather than hoped for: generate, count what
    survived, and run further rounds for the shortfall — including after the
    review pass, so an item the reviewer rejected is replaced instead of simply
    lost. Extra items beyond the target are kept in the bank but unchecked, and
    if the lecture genuinely cannot support the request, that is reported as a
    note instead of quietly handing back fewer.

    Returns ``(questions, notes)``.
    """
    notes: list[str] = []
    report = report or RunReport()
    if not chunks or target <= 0:
        return [], notes

    bloom_targets = bloom_targets or ["Remember", "Understand", "Apply", "Analyze"]

    # What the lecture was for, in the summary's own words. Drives both where
    # questions are placed and what each call is told to aim at, so the quiz
    # follows the argument rather than the clock.
    focus_points = []
    if summary is not None:
        focus_points = [
            *(getattr(summary, "learning_objectives", None) or []),
            *(getattr(summary, "key_points", None) or []),
        ]
    weights = chunk_importance(chunks, summary) if summary is not None else []

    common = dict(
        n_options=n_options,
        bloom_targets=bloom_targets,
        difficulty_mix=difficulty_mix,
        course_context=course_context,
        focus_points=focus_points,
    )

    # Questions that already exist in *other* sets for this lecture. Without
    # these, an "alternative" set is written in ignorance of the first one and
    # reproduces it — the salient points of a chunk are the salient points
    # whichever time you ask. That made a second set look redundant and a third
    # look impossible.
    existing = list(avoid_stems or [])

    questions: list[MCQ] = []
    rotation = 0

    def _emit(fraction: float, message: str) -> None:
        if progress:
            progress(fraction, message)

    # --- Round 1: the straightforward pass ------------------------------- #
    questions = generate_questions(
        client,
        chunks,
        allocate_by_importance(target, chunks, summary),
        avoid_stems=existing,
        progress=lambda f, m: _emit(f * 0.5, m),
        report=report,
        **common,
    )
    questions = _drop_duplicates(questions, existing)

    # Distinguish "the model could not be reached" from "this lecture is thin".
    # The old code reported both as "returned nothing new for this material".
    if report.phase_failed_entirely(PHASE_GENERATE):
        raise LLMError(
            f"Every question request failed. Last reason: "
            f"{report.failures[-1].detail}"
        )

    # --- Top-up rounds --------------------------------------------------- #
    for round_no in range(2, max_rounds + 1):
        shortfall = target - len(questions)
        if shortfall <= 0:
            break
        rotation += max(1, len(questions))
        _emit(
            0.5 + (round_no - 2) * 0.1,
            f"Only {len(questions)} of {target} so far — writing {shortfall} more",
        )
        extra = generate_questions(
            client,
            chunks,
            _allocate_topup(shortfall, chunks, rotation, weights),
            avoid_stems=existing + [q.stem for q in questions],
            report=report,
            **common,
        )
        before = len(questions)
        questions = _drop_duplicates(questions + extra, existing)
        if len(questions) == before:
            # Another round will not help if this one added nothing usable —
            # but say *which* kind of nothing, since a failing API and a thin
            # lecture need completely different responses from the reader.
            recent = [s for s in report.phase_steps(PHASE_GENERATE) if s.is_failure]
            if recent:
                notes.append(
                    f"Stopped topping up after round {round_no}: "
                    f"{len(recent)} request(s) failed — {recent[-1].detail}"
                )
            else:
                notes.append(
                    f"Stopped topping up after round {round_no}: the model returned "
                    "nothing new for this material."
                )
            break

    # --- Review, then replace whatever it dropped ------------------------ #
    if do_review and questions:
        _emit(0.8, "Reviewing drafted questions")
        questions, _ = critique_and_revise(client, questions, report=report)

        dropped = [q for q in questions if not q.include]
        if dropped and len(_included(questions)) < target:
            need = target - len(_included(questions))
            _emit(0.9, f"Reviewer dropped {len(dropped)} — writing {need} replacement(s)")
            rotation += len(questions)
            replacements = generate_questions(
                client,
                chunks,
                _allocate_topup(need, chunks, rotation, weights),
                avoid_stems=existing + [q.stem for q in questions],
                report=report,
                **common,
            )
            questions = _drop_duplicates(questions + replacements, existing)

    # --- Settle on exactly `target` -------------------------------------- #
    questions = balance_answer_positions(questions)
    questions = validate_all(questions)
    questions = _select_best(questions, target, notes)

    _emit(1.0, f"{len(_included(questions))} questions ready")
    return questions, notes


def _included(questions: list[MCQ]) -> list[MCQ]:
    return [q for q in questions if q.include]


def _drop_duplicates(
    questions: list[MCQ], against: list[str] | None = None
) -> list[MCQ]:
    """Remove near-identical stems, keeping the first.

    ``against`` seeds the comparison with stems from question sets generated
    earlier for this lecture, so an alternative set is measured against what
    already exists rather than only against itself. A dropped near-duplicate
    then shows up as a shortfall, which the top-up rounds make good — the set
    comes back to full size with genuinely different questions instead of
    quietly repeating the first set.
    """
    kept: list[MCQ] = []
    seen: list[set[str]] = [_content_tokens(s) for s in (against or []) if s]
    for q in questions:
        tokens = _content_tokens(q.stem)
        if any(_jaccard(tokens, other) >= DROP_DUPLICATE_THRESHOLD for other in seen):
            continue
        kept.append(q)
        seen.append(tokens)
    return kept


def _select_best(questions: list[MCQ], target: int, notes: list[str]) -> list[MCQ]:
    """Check exactly ``target`` items, preferring the cleanest ones.

    Surplus items are kept in the bank rather than thrown away — an instructor
    who wanted ten and got thirteen usually wants to look at the other three.
    """
    reviewer_dropped = [q for q in questions if not q.include]
    candidates = [q for q in questions if q.include]

    # Fewest defects first; ties keep the original order, which follows the
    # lecture, so the checked set stays spread across the recording.
    ranked = sorted(
        range(len(candidates)), key=lambda i: (len(candidates[i].flags), i)
    )
    chosen = {id(candidates[i]) for i in ranked[:target]}

    for q in candidates:
        if id(q) not in chosen:
            q.include = False
            q.flags = list(dict.fromkeys(q.flags + ["Extra — beyond the requested count"]))

    kept = len(chosen)
    if kept < target:
        notes.append(
            f"Asked for {target} questions; {kept} survived generation and review. "
            "A longer recording, a lower question count, or a stronger model will "
            "close the gap."
        )
    elif len(candidates) > target:
        notes.append(
            f"{len(candidates) - target} extra question(s) are in the bank, "
            "unchecked — tick any of them to include it."
        )
    if reviewer_dropped:
        notes.append(
            f"The reviewer rejected {len(reviewer_dropped)} draft(s); they are kept "
            "unchecked so you can see what was cut and why."
        )
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
    report: RunReport | None = None,
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

    report = report or RunReport()
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
            report=report,
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

# Flagging is advisory, so it can afford to be sensitive. Dropping destroys work,
# so it needs a higher bar: two questions about the same concept legitimately
# share a lot of vocabulary, and only near-identical wording should be discarded.
DUPLICATE_THRESHOLD = 0.7
DROP_DUPLICATE_THRESHOLD = 0.85


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
    client: LLMClient,
    questions: list[MCQ],
    progress: ProgressFn | None = None,
    report: RunReport | None = None,
) -> tuple[list[MCQ], list[dict]]:
    """Have the model review its own items and apply the revisions it proposes.

    A separate pass with a reviewer persona catches a meaningful share of the
    defects a single generate call leaves behind — it is the cheapest quality
    improvement available here. Items marked "drop" are kept but unchecked, so
    the instructor sees what was rejected and why.
    """
    report = report or RunReport()
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
    except Exception as exc:
        # The review is optional, so a failure here must not lose the drafts —
        # but silently skipping it made "why were none dropped?" unanswerable.
        report.record(
            PHASE_REVIEW, f"{len(questions)} questions", FAILED,
            detail=short_reason(exc), cause=classify(exc),
        )
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

    report.record(
        PHASE_REVIEW, f"{len(questions)} questions", OK, produced=len(reviews)
    )
    if progress:
        progress(1.0, "Review complete")

    return list(by_id.values()), reviews
