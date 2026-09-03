# 🎓 Lecture Quiz Builder

Upload a lecture recording. Get back a transcript, a structured summary, and a
reviewable bank of multiple-choice questions you can import straight into Canvas
— or hand out as a printed worksheet with an answer key.

Transcription runs **locally** with faster-whisper, so audio never leaves the
server. Only the transcript text is sent to an LLM, and only for summarizing and
question writing — DeepSeek via OpenRouter by default, or Gemini, Claude,
OpenAI, Grok, or any other model OpenRouter carries.

Long recordings can be uploaded **in parts**: split a 90-minute lecture into
three files and they are transcribed in order, then stitched into one continuous
transcript before anything else happens.

```
part 1 ─┐
part 2 ─┼─► faster-whisper ─► stitched transcript ─► chunks ─► summary
part 3 ─┘    (sequential)       (one timeline)         │
                                                       └─► MCQ draft ─► review
                                                                          │
                                    QTI · XLSX · CSV · DOCX · PDF · MD ◄──┘
                                                     ▲
                             alternative sets · per-question replacement
```

---

## What it does

| Stage | Detail |
|---|---|
| **Transcribe** | faster-whisper (CTranslate2), segment timestamps, voice-activity filtering, live progress. Accepts a split recording as several files and stitches them onto one timeline |
| **Summarize** | Map-reduce over 10-minute windows so a 75-minute lecture gets even attention: title, abstract, learning objectives, key points, timestamped outline, key terms |
| **Generate** | Questions written per chunk with Bloom-level and difficulty targets, each carrying the timestamp and a verbatim quote that supports the answer. Provider is a dropdown: OpenRouter, Gemini, Claude, OpenAI, Grok |
| **Review** | An automatic second pass critiques the drafts and repairs or drops weak items |
| **Validate** | Mechanical checks for "all of the above", duplicate options, giveaway answer length, negative stems, near-duplicate questions, missing provenance |
| **Balance** | Correct answers are redistributed across A/B/C/D — LLMs have a strong positional bias students notice fast |
| **Edit** | Every stem, option, and answer key is editable in the browser before export |
| **Regenerate** | Ask for a whole alternative set (kept side by side with the first), or replace any single question with a newly written one |
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

cp .env.example .env        # add OPENROUTER_API_KEY (or any other provider's key)
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
3. Open **Advanced settings → Secrets** and paste keys for whichever providers
   you plan to use — one is enough:

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."  # default provider
   GEMINI_API_KEY = "AIza..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."
   XAI_API_KEY = "xai-..."
   APP_PASSWORD = "pick-something"      # optional shared-password gate
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
  If a file is still too large, or a single transcription run is taking longer
  than you want to sit through, split it and upload the parts — see
  [Splitting a long recording](#splitting-a-long-recording).

If any of that bites, run it on a campus machine or in Docker instead — the
codebase is identical, and a GPU makes `large-v3` practical.

---

## Configuration

Everything is in the sidebar; nothing needs a code change to try.

**Transcription** — model size, language, voice-activity filter, beam size, compute type.
**Generation** — provider, model, question count, options per question, Bloom levels, difficulty mix, temperature.
**Context** — a free-text course description that steers what the model treats as important. This is worth filling in. "MGT 301 Operations Management, junior level; emphasize the trade-offs between chase and level strategies" produces materially better questions than the default.

Prompts live in `src/prompts.py` and are meant to be edited for your discipline.

### LLM providers

| Provider | Key | Where to get one | Notes |
|---|---|---|---|
| **OpenRouter** *(default)* | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | One key, any model — paste any slug from [openrouter.ai/models](https://openrouter.ai/models) |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier covers light use |
| Anthropic Claude | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Best at following the item-writing rules |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | |
| xAI Grok | `XAI_API_KEY` | [console.x.ai](https://console.x.ai) | OpenAI-compatible endpoint |

The default model is **`deepseek/deepseek-v4-pro`** — strong enough to follow the
item-writing rules in the prompts, and roughly a tenth the price of the frontier
alternatives. `deepseek/deepseek-v4-flash` is cheaper again if you're generating
across a whole semester; `deepseek/deepseek-r1` is available but its reasoning
output makes it slower here for no gain on this task.

Adding a provider is a single entry in `PROVIDERS` in `src/config.py`. Anything
that speaks the OpenAI `/chat/completions` protocol needs only a `base_url`;
Gemini and Claude have native paths because their SDKs give stronger JSON
guarantees. OpenRouter deliberately skips native JSON mode — not every proxied
model honors `response_format`, and a rejected parameter fails the whole call, so
JSON is requested in the prompt and salvaged from the reply instead.

### Splitting a long recording

Upload the parts together. They are ordered by filename with numbers read as
numbers — so `part2` comes before `part10`, which plain alphabetical sorting gets
wrong — and you can override that with **Use the order I uploaded them in**. The
detected order is shown before you commit.

Each part is transcribed in turn and its timestamps are shifted by the running
total, so the combined transcript is one continuous lecture. A question drawn
from part three is tagged `52:14` of the lecture, not `2:14` of the third file,
and the summarizer's 10-minute windows straddle part boundaries exactly as they
would in a single file.

Two things to know:

- **Split on a clean boundary, without overlap.** If parts overlap, the
  overlapping speech is transcribed twice. The near-duplicate check will flag
  questions that result, but trimming the overlap first is better.
- **A part with no detectable speech is skipped, not fatal.** The rest still
  processes and the app tells you which part was dropped — but the timeline after
  that gap will be short by the missing part's length.

To split a file with ffmpeg, in 25-minute pieces:

```bash
ffmpeg -i lecture.mp3 -f segment -segment_time 1500 -c copy lecture_part%d.mp3
```

### Alternative sets and replacing single questions

On the **Questions** tab:

- **✨ Generate an alternative set** writes a whole new set over the same
  lecture. Every question you already have is passed in as an explicit
  avoid-list, which matters more than it sounds: without it, a second pass
  reproduces the first almost verbatim, because the salient points of a lecture
  are the salient points however many times you ask. Previous sets are kept —
  switch between them with the radio buttons, and export whichever you want.
  Useful for a makeup exam, a practice bank that isn't the graded bank, or simply
  a second opinion on a chunk you thought was under-covered.
- **🔄 Replace this question** swaps one item for a newly written one. By default
  the replacement is drawn from the same stretch of the lecture, so removing a
  bad item doesn't quietly leave a hole in your coverage; toggle that off to draw
  from anywhere. The point value is carried over, and the new item is validated
  immediately.

Both actions know about every question in every set, so replacements and
alternative sets don't collide with each other.

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
│   ├── transcribe.py             faster-whisper wrapper, multi-part stitching
│   ├── chunking.py               time-window splitting, question allocation
│   ├── llm.py                    five-provider abstraction, retries, cost
│   ├── prompts.py                every prompt, in one editable place
│   ├── summarize.py              map-reduce summarization
│   ├── mcq.py                    generation, validation, balancing, critique,
│   │                             avoid-lists, single-question replacement
│   └── exporters/
│       ├── qti.py                QTI 1.2 (Canvas) and QTI 2.1
│       ├── tabular.py            CSV, XLSX
│       ├── documents.py          DOCX, PDF, Markdown
│       └── transcript_formats.py TXT, SRT, WebVTT
└── tests/
    ├── test_pipeline.py            schema, validation, balancing, all exporters
    ├── test_llm_clients.py         provider wiring, against stubbed SDKs
    └── test_multipart_and_regen.py part stitching, avoid-lists, replacement
```

```bash
pytest -q          # 81 tests, no API keys or network needed
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

Transcription is free (local compute). Generation, for a 60-minute lecture
(~12k transcript tokens) with the review pass on:

| Model | Approx. cost per lecture |
|---|---|
| `deepseek/deepseek-v4-flash` | ~$0.005 |
| `deepseek/deepseek-v3.2` | ~$0.01 |
| **`deepseek/deepseek-v4-pro`** *(default)* | ~$0.03 |
| `gemini-3.8-flash` | ~$0.04 |
| `grok-4.6` | ~$0.07 |
| `gpt-4.1` | ~$0.09 |
| `claude-sonnet-4-5` | ~$0.10 |

The sidebar shows actual token counts and an estimated cost after each run.
OpenRouter figures are approximate — it routes to whichever upstream host is
cheapest or fastest at the moment, so the real rate moves; check the OpenRouter
dashboard for actual spend. Any model slug typed into the custom box shows tokens
but no dollar figure, which is the honest answer rather than a fabricated one.

An alternative set costs about the same as the original run. A single replacement
costs one small call.

Turning off the review pass roughly halves the cost and noticeably lowers
question quality. At three cents a lecture that is not a trade worth making.

---

## License

MIT. See [LICENSE](LICENSE).
