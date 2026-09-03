# 🎓 Lecture Quiz Builder

Upload a lecture recording. Get back a transcript, a structured summary, and a
reviewable bank of multiple-choice questions you can import straight into Canvas
— or hand out as a printed worksheet with an answer key.

Transcription runs **locally** with faster-whisper, so audio never leaves the
server. Only the transcript text is sent to an LLM, and only for summarizing and
question writing.

```
audio ──► faster-whisper ──► transcript ──► chunks ──► summary
                                              │
                                              └──► MCQ draft ──► review pass
                                                                    │
                            QTI · XLSX · CSV · DOCX · PDF · MD ◄────┘
```

---

## What it does

| Stage | Detail |
|---|---|
| **Transcribe** | faster-whisper (CTranslate2), segment timestamps, voice-activity filtering, live progress |
| **Summarize** | Map-reduce over 10-minute windows so a 75-minute lecture gets even attention: title, abstract, learning objectives, key points, timestamped outline, key terms |
| **Generate** | Questions written per chunk with Bloom-level and difficulty targets, each carrying the timestamp and a verbatim quote that supports the answer |
| **Review** | An automatic second pass critiques the drafts and repairs or drops weak items |
| **Validate** | Mechanical checks for "all of the above", duplicate options, giveaway answer length, negative stems, near-duplicate questions, missing provenance |
| **Balance** | Correct answers are redistributed across A/B/C/D — LLMs have a strong positional bias students notice fast |
| **Edit** | Every stem, option, and answer key is editable in the browser before export |
| **Export** | QTI 1.2 (Canvas) · QTI 2.1 · XLSX · CSV · DOCX · PDF · Markdown · SRT/VTT captions |

Every generated question is a **draft for your review**, not a finished exam item.
The app is built to make review fast, not to make it unnecessary.

---

## Quick start (local)

```bash
git clone https://github.com/<you>/lecture-quiz-builder.git
cd lecture-quiz-builder

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # add ANTHROPIC_API_KEY or OPENAI_API_KEY
streamlit run app.py
```

`ffmpeg` is needed for some container formats. macOS: `brew install ffmpeg`.
Ubuntu: `sudo apt install ffmpeg`. Windows: `winget install ffmpeg`.

The first run downloads the Whisper weights (~150 MB for `small`) into
`.cache/whisper/`. Subsequent runs are instant.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo (Community Cloud requires public,
   or a private repo on a paid plan).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the
   repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets** and paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."
   APP_PASSWORD = "pick-something"     # optional shared-password gate
   ```

4. Deploy.

`requirements.txt` and `packages.txt` (which installs `ffmpeg`) are picked up
automatically.

### Community Cloud limits — read this before you rely on it

The free tier gives you **1 CPU core and about 1 GB of RAM**. That has real
consequences:

- **Use the `small` Whisper model or below.** `medium` and `large-v3` will
  exhaust memory and the app will restart mid-transcription.
- **Transcription is roughly 0.35× real time on `small`** — a 50-minute lecture
  takes about 18 minutes. The app shows an estimate before it starts.
- **Apps sleep after inactivity** and cold-start by re-downloading the model.
  Expect a slow first request after idle time.
- **Uploads are capped at 400 MB** by `.streamlit/config.toml`. A 90-minute MP3
  at 128 kbps is about 85 MB, so this is usually fine; MP4 video is not.
  Extract the audio track first: `ffmpeg -i lecture.mp4 -vn -b:a 96k lecture.mp3`.

If any of that bites, run it on a campus machine or in Docker instead — the
codebase is identical, and a GPU makes `large-v3` practical.

---

## Configuration

Everything is in the sidebar; nothing needs a code change to try.

**Transcription** — model size, language, voice-activity filter, beam size, compute type.
**Generation** — provider (Claude / OpenAI), model, question count, options per question, Bloom levels, difficulty mix, temperature.
**Context** — a free-text course description that steers what the model treats as important. This is worth filling in. "MGT 301 Operations Management, junior level; emphasize the trade-offs between chase and level strategies" produces materially better questions than the default.

Prompts live in `src/prompts.py` and are meant to be edited for your discipline.

---

## Importing into Canvas

1. Export **QTI 1.2 · Canvas (.zip)**.
2. Canvas → your course → **Settings → Import Course Content**.
3. Content type: **QTI .zip file**. Upload, import.
4. The quiz appears under **Quizzes** as an unpublished assignment quiz, with
   the correct answers keyed and your rationales attached as answer feedback.

Use the QTI 2.1 export only if your LMS specifically demands 2.1 — Canvas
handles the 1.2 package more reliably.

---

## Project layout

```
lecture-quiz-builder/
├── app.py                        Streamlit UI and pipeline orchestration
├── requirements.txt
├── packages.txt                  apt packages for Streamlit Cloud (ffmpeg)
├── conftest.py
├── .streamlit/
│   ├── config.toml               upload cap, theme
│   └── secrets.toml.example
├── src/
│   ├── schema.py                 Transcript, Chunk, Summary, MCQ, Quiz
│   ├── config.py                 settings, model catalogs, secret resolution
│   ├── transcribe.py             faster-whisper wrapper
│   ├── chunking.py               time-window splitting, question allocation
│   ├── llm.py                    Anthropic/OpenAI abstraction, retries, cost
│   ├── prompts.py                every prompt, in one editable place
│   ├── summarize.py              map-reduce summarization
│   ├── mcq.py                    generation, validation, balancing, critique
│   └── exporters/
│       ├── qti.py                QTI 1.2 (Canvas) and QTI 2.1
│       ├── tabular.py            CSV, XLSX
│       ├── documents.py          DOCX, PDF, Markdown
│       └── transcript_formats.py TXT, SRT, WebVTT
└── tests/test_pipeline.py        31 tests: schema, validation, all exporters
```

```bash
pytest -q          # run the suite
```

---

## Privacy and student data

Audio is transcribed locally and written only to a temp file that is deleted
immediately after. Transcript text **is** sent to whichever LLM provider you
select.

That matters for FERPA. A recording of you lecturing is your own content. A
recording of a class **discussion** contains identifiable student speech, and
sending it to a third-party API is a disclosure decision your institution — not
this README — should make. Before using this on discussion recordings, check
with your campus's privacy office about whether the provider is covered by an
existing agreement.

Consider also: disclose recording and AI processing in your syllabus, and review
every generated item before it reaches a student.

---

## Costs

Transcription is free (local compute). Generation runs roughly:

| Lecture length | Approx. transcript tokens | Cost with Claude Sonnet, review pass on |
|---|---|---|
| 30 min | ~6k | ~$0.05 |
| 60 min | ~12k | ~$0.10 |
| 90 min | ~18k | ~$0.15 |

The sidebar shows the actual estimated cost after each run. Turning off the
review pass roughly halves it and noticeably lowers question quality.

---

## License

MIT. See [LICENSE](LICENSE).
