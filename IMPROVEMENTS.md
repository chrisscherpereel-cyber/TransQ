# Improvement roadmap

Ordered by value per unit of effort. The first four are worth doing before you
put this in front of students; the rest are genuine features, not polish.

---

## Tier 1 — do these next

### 1. Verify every source quote actually appears in the transcript

Right now each question *claims* a supporting quote. Nothing checks it. A fuzzy
substring match against the transcript turns that claim into a guarantee and
catches the single most damaging failure mode — a confidently wrong question
about something the lecture never said.

```python
from difflib import SequenceMatcher

def quote_is_grounded(quote: str, transcript_text: str, threshold: float = 0.75) -> bool:
    """Sliding-window fuzzy match; ASR text won't match verbatim."""
    q = " ".join(quote.lower().split())
    if not q:
        return False
    hay = " ".join(transcript_text.lower().split())
    if q in hay:
        return True
    words, qlen = hay.split(), len(q.split())
    return any(
        SequenceMatcher(None, q, " ".join(words[i : i + qlen])).ratio() >= threshold
        for i in range(0, max(1, len(words) - qlen), max(1, qlen // 2))
    )
```

Wire it into `validate_question` as a hard flag, and default those items to
`include = False`. Roughly 30 lines, and it converts the app from "trust the
model" to "trust but verify."

### 2. Feed Whisper your course vocabulary

faster-whisper accepts an `initial_prompt`. Passing a short glossary
("kanban, heijunka, takt time, EOQ, MRP, bullwhip effect") measurably reduces
mistranscribed jargon — and jargon is exactly what the questions are about.

```python
model.transcribe(path, initial_prompt="Terms used: " + ", ".join(glossary))
```

Add a glossary text box next to the course-context field. One sidebar input, one
kwarg, disproportionate payoff.

### 3. Map questions to your stated learning objectives

Let the instructor paste course learning objectives, ask the model to tag each
question with the objective it assesses, and render a coverage matrix showing
which objectives have no items. This is the difference between "a quiz" and
"assessment evidence," and it produces exactly the artifact AACSB
assurance-of-learning review asks for. It also surfaces the more useful finding:
objectives the *lecture itself* never supported.

### 4. Cache transcripts by audio hash

Transcription is the expensive step, and Streamlit reruns the whole script on
every widget interaction. Hash the uploaded bytes, store the transcript JSON
under `.cache/transcripts/<sha256>.json`, and check it first. Re-generating
questions with different settings then costs seconds instead of twenty minutes —
which is what makes the tool usable iteratively rather than once.

Worth doing **per part**, not per upload: re-uploading a four-part lecture with
one part re-cut should re-transcribe one part, not four. The `TranscriptPart`
records already carry the filename and offset needed to stitch cached parts back
together.

---

## Tier 2 — meaningful features

### 5. More question types

Multiple choice alone is a narrow instrument. In rough order of value for an
operations course:

- **Numeric / calculation items** with tolerance bands and worked solutions
  (EOQ, takt time, capacity utilization). These are what actually discriminate
  in a quantitative course, and QTI supports them.
- **Multi-select** (`rcardinality="Multiple"` in QTI) for "select all that apply."
- **Short answer** with a model answer and a scoring rubric, for the Analyze and
  Evaluate levels where MCQ genuinely struggles.
- **True/False with required justification** — cheap to write, hard to guess.

The schema already carries `points` and per-option rationales; adding a `kind`
discriminator to `MCQ` and branching in the exporters is the main work.

### 6. Parallel forms at the item level

Alternative *sets* already exist (the ✨ button on the Questions tab), which
covers makeup exams and a separate practice bank. The finer-grained version is
still open: 3–5 variants of the **same item** — same concept, different numbers
and surface features — grouped so an LMS can draw one at random per student.
That is what meaningfully reduces exposure when answers circulate, and in Canvas
it maps onto question groups rather than a second quiz.

### 7. Close the loop with item analysis

Export a quiz, run it, then import Canvas's item-analysis CSV back in. Store the
p-value and point-biserial per item, retire items that discriminate poorly, and
feed the survivors back as few-shot examples. Within two semesters the generator
is calibrated to your students rather than to a generic prior. This is the single
feature that would make the tool compound in value.

### 8. Second input: slides or readings

Let the instructor upload the lecture's PPTX or the assigned chapter PDF
alongside the audio. Use it to correct ASR errors on technical terms, to weight
what matters, and to catch material the slides covered but the recording garbled.
The `pptx` and `pdf` extraction path is straightforward.

### 9. Speaker diarization

`pyannote.audio` or WhisperX separates instructor speech from student speech.
Two payoffs: questions stop being written from a student's half-formed answer,
and you can strip student utterances entirely before anything is sent to an API —
which is the clean answer to the FERPA question in the README rather than a
policy caveat.

### 10. Persistent question bank

SQLite (or Postgres) storing every generated item across lectures, with
similarity-based dedupe against prior weeks and tagging by module. Turns a
per-lecture utility into a course-level asset, and makes cumulative final-exam
assembly a query instead of a project.

---

## Tier 3 — infrastructure and scale

### 11. Move long jobs off the Streamlit request

Streamlit's execution model — rerun the whole script on every interaction — is a
poor fit for a 20-minute transcription. If more than a handful of people use
this, split the pipeline into a worker (Celery + Redis, or a simple SQLite job
table) and let the UI poll. This is the change that makes the app survive being
shared with a department.

### 12. Batch mode

Point it at a folder or a Panopto/Kaltura/Zoom export and process a whole
semester overnight. Most of the marginal value of this tool shows up when it runs
over fourteen lectures, not one.

### 13. Real authentication

The shared password is fine for one instructor. For a department, use OIDC
against campus SSO and give each user their own key and usage quota. Add a
per-session token ceiling regardless — an accidental 40-question run on a
three-hour recording should not be able to surprise anyone.

### 14. Docker + GPU

A `Dockerfile` with CUDA base image makes `large-v3` practical (roughly 30×
faster than CPU `small`, and materially more accurate on accented speech and
technical vocabulary). Worth it if there's any lab machine or campus VM available.

### 15. An evaluation harness

Build a fixed set of 3–5 lectures with human-graded reference questions, then
score model/prompt changes against it: defect rate per item, grounding rate,
Bloom-level accuracy, answer-position entropy. Without this, "Claude vs. GPT" and
"is the new prompt better" are opinions. With it, they are a table. Given your
research interests, this is also the part that could become a paper — an
instrument for measuring AI-generated assessment quality is more publishable than
the generator itself.

---

## Deliberately not doing

- **Auto-publishing to the LMS.** The export-and-import step is friction, and the
  friction is the review gate. Keep it.
- **Hiding the flags.** Surfacing "this item may have two defensible answers" is
  the app's most useful output. Do not let a future cleanup pass bury it.
- **Auto-grading student responses.** Different product, much higher stakes, and
  the failure modes land on students rather than on you.

---

## Known limitations today

| Limitation | Impact |
|---|---|
| Source quotes are unverified | A question can cite something never said (Tier 1 #1 fixes this) |
| No transcript caching | Changing a generation setting re-transcribes from scratch (#4) |
| Chunk overlap can duplicate content | Near-duplicate detection flags it, but does not merge |
| Bloom self-labeling is unreliable | The model's "Analyze" is often Understand; treat labels as hints |
| Overlapping upload parts are transcribed twice | Split on clean boundaries; the app assumes parts are contiguous and does not detect or trim overlap |
| A skipped (silent) part shortens the timeline | Timestamps after the gap are off by that part's length; the app warns which part was dropped |
| OpenRouter costs are approximate | It routes to whichever upstream host is cheapest at the moment; custom slugs show tokens only |
| Gemini's free tier is rate-limited | A long lecture can trip requests-per-minute limits; the client retries with backoff, but a paid key is smoother |
| Streamlit Cloud caps at `small` | Accented or noisy audio transcribes poorly there |
| Alternative sets live only in the session | Closing the tab loses every set but the one you exported (Tier 2 #10 fixes this) |
| Single-user session state | Two people using one deployment share nothing but also collide on nothing; there is no saved work |
| QTI 1.2 tested against Canvas semantics only | Other LMSs accept the package but may map feedback fields differently |
