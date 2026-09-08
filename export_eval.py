#!/usr/bin/env python3
"""Export this app's real prompts so a benchmark can score the actual task.

Written for OpenRouter's Ori Eval, though nothing here is specific to it: the
output is a plain JSONL of prompts plus a rubric, which any harness can read.

**Why this rather than a plug-in.** Ori Eval is a CLI tool. It installs with a
shell script, holds an interactive `ori login`, and prints human-readable
reports. None of that survives inside a Streamlit app: Community Cloud wipes the
container on restart, a multi-user deployment cannot share one person's login,
and parsing an undocumented report format would break on its next release. So
the integration is a file, not a call — the app exports what it actually asks
models to do, you run the benchmark on your own machine, and you bring back a
model slug to pin in the sidebar.

**Why the app has to be the one to export it.** A generic benchmark tells you
which model is better at generic tasks. It cannot tell you which model writes
good multiple-choice items *from a lecture transcript, with a verbatim
supporting quote, in JSON, under this app's fifteen item-writing rules*. That is
the only question that matters here, and answering it needs these prompts —
including the material and exam-topic clauses your own lecture produces.

    python3 scripts/export_eval.py --list
    python3 scripts/export_eval.py --lecture "MGT 301 — Week 4" --out eval/

Then, in that directory:

    ori login
    ori eval

The prompts contain your lecture transcript. Treat the export directory the way
you would treat the recording.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prompts  # noqa: E402
from src.chunking import allocate_by_importance, chunk_transcript  # noqa: E402
from src.library import TranscriptLibrary  # noqa: E402
from src.materials import material_context, relevant_sections  # noqa: E402
from src.mcq import (  # noqa: E402
    _build_exam_clause,
    _build_focus_clause,
    _with_inline_timestamps,
)
from src.config import DEFAULT_FRAMING, QUESTION_FRAMING  # noqa: E402
from src.storage import StorageError, build_store  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotate_key import load_secrets  # noqa: E402

RUBRIC = """# What a good answer looks like

Score each reply against the task this app actually performs: writing
multiple-choice items from a lecture transcript. These criteria are ordered by
how often they are what separates a usable model from an unusable one.

## Disqualifying

- **Valid JSON, complete.** A single object with a `questions` array, closed. A
  reply cut off mid-array is the most common real failure and costs the whole
  input again on retry.
- **`correct_index` points at the correct option.** Small models routinely
  misalign this. An item with the wrong key is worse than no item, because it
  looks fine until a student challenges it.
- **`source_quote` appears verbatim in the transcript.** Paraphrase here breaks
  provenance, which is the thing that makes these drafts reviewable at speed.
  Check the quote against the supplied transcript text, not for plausibility.

## Quality

- **Exactly the number of questions requested.** Under-delivery forces extra
  rounds, so a cheap model that returns 6 of 10 can cost more than a dearer one
  that returns 10.
- **Distractors are specific misconceptions**, not filler. "Beta", a restatement
  of the stem, or an obviously absurd option all fail this.
- **The stem is answerable without reading the options**, positively phrased, no
  "all of the above" or "none of the above", no giveaway from option length.
- **Tests understanding, not recall of phrasing** — not a date mentioned in
  passing, not the instructor's exact words.
- **Follows the requested Bloom levels and difficulty mix.**
- **Uses the supplied slide terminology** where slides were provided, rather
  than the pronoun the transcript contains.

## What not to reward

Do not reward fluency, length, or a confident tone. A model that writes
beautiful items with a fabricated quote is worse for this task than one that
writes plain items with real ones.
"""

READ_ME = """# Eval set for Lecture Quiz Builder

`prompts.jsonl` holds the exact system and user messages this app sends when
writing questions, generated from one real lecture — including the material and
exam-topic clauses that lecture produced. `rubric.md` says what a good answer
looks like for this task specifically.

    ori login
    ori eval

**These files contain your lecture transcript.** Keep the directory out of
version control and delete it when you are done.

Cost: an eval sends real requests to real models. Start with two or three
candidates and a handful of prompts before running a wide comparison.

When you have a winner, set it in the app's sidebar. The model you last
generated with becomes your default on the next sign-in.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--user", default="", help="account whose library to read")
    parser.add_argument("--lecture", default="", help="lecture name, or part of it")
    parser.add_argument("--out", default="eval", help="directory to write (default: eval)")
    parser.add_argument("--questions", type=int, default=10, help="questions per run")
    parser.add_argument("--list", action="store_true", help="list lectures and exit")
    args = parser.parse_args()

    secrets = load_secrets()
    if not secrets.get("APP_SECRET"):
        print("APP_SECRET is not set — the library cannot be decrypted.")
        return 1

    try:
        store, _ = build_store(secrets)
    except StorageError as exc:
        print(f"Storage could not start: {exc}")
        return 1

    username = args.user or _sole_user(store)
    if not username:
        print("Pass --user; the account name is the one you sign in with.")
        return 1

    library = TranscriptLibrary(store, username)
    entries = [e for e in library.entries() if e.is_complete]
    if not entries:
        print(f"No finished lectures in {username}'s library.")
        return 1

    if args.list or not args.lecture:
        print(f"Lectures for {username}:\n")
        for entry in entries:
            print(f"  {entry.title}  ({entry.length_label}, {entry.summary_label})")
        print("\nRe-run with --lecture \"<name>\".")
        return 0

    wanted = args.lecture.lower()
    match = next((e for e in entries if wanted in e.title.lower()), None)
    if match is None:
        print(f'No lecture matching "{args.lecture}". Use --list to see them.')
        return 1

    saved = library.load(match.id)
    records = build_records(saved, args.questions)
    if not records:
        print("That lecture produced no prompts — is the transcript empty?")
        return 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "prompts.jsonl"), "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "rubric.md"), "w", encoding="utf-8") as handle:
        handle.write(RUBRIC)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(READ_ME)

    print(
        f"Wrote {len(records)} prompt(s) from “{match.title}” to {args.out}/\n"
        f"  prompts.jsonl   the real generation prompts\n"
        f"  rubric.md       what a good answer looks like here\n"
        f"  README.md       how to run it, and the privacy note\n\n"
        f"These contain your transcript. Do not commit them.\n\n"
        f"Next:  cd {args.out} && ori login && ori eval"
    )
    return 0


def build_records(saved, target: int) -> list[dict]:
    """One record per transcript window, carrying the app's real prompts."""
    transcript = saved.transcript
    summary = saved.summary
    material = saved.material
    topics = list(saved.exam_topics or [])

    chunks = chunk_transcript(transcript, 600, 30)
    if not chunks:
        return []
    allocation = allocate_by_importance(target, chunks, summary, topics)

    focus_points = []
    if summary is not None:
        focus_points = [
            *(getattr(summary, "learning_objectives", None) or []),
            *(getattr(summary, "key_points", None) or []),
        ]
    focus_clause = _build_exam_clause(topics) + _build_focus_clause(focus_points)
    options_placeholder = ", ".join(f'"option {chr(65 + i)}"' for i in range(4))

    records: list[dict] = []
    for chunk, want in zip(chunks, allocation):
        if want <= 0:
            continue
        material_clause = ""
        if material is not None:
            matched = relevant_sections(chunk.text, material)
            material_clause = material_context(matched, material.section_noun)

        records.append(
            {
                "id": f"{saved.entry.id}-{chunk.index}",
                "system": prompts.MCQ_SYSTEM.format(
                    n_options=4,
                    framing_rule=QUESTION_FRAMING[DEFAULT_FRAMING]["rule"],
                ),
                "prompt": prompts.MCQ_USER.format(
                    n=want,
                    label=chunk.label,
                    course_context="",
                    bloom_targets="Remember, Understand, Apply, Analyze",
                    difficulty_mix="Balanced",
                    focus_clause=focus_clause,
                    material_clause=material_clause,
                    # Left empty on purpose: the avoid list changes between runs
                    # and would make two models' results incomparable.
                    avoid_clause="",
                    options_placeholder=options_placeholder,
                    text=_with_inline_timestamps(chunk)[:24000],
                ),
                # Carried so a grader can check quotes against the source rather
                # than judging them for plausibility.
                "expected_questions": want,
                "window": chunk.label,
                "transcript_excerpt": chunk.text[:24000],
            }
        )
    return records


def _sole_user(store) -> str:
    """Use the only account when there is only one — the common case."""
    try:
        index = store.read("users/index") or {}
        names = [str(n) for n in (index.get("usernames") or []) if n]
    except StorageError:
        return ""
    return names[0] if len(names) == 1 else ""


if __name__ == "__main__":
    raise SystemExit(main())
