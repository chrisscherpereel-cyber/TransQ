# 🎓 Lecture Quiz Builder

Upload a lecture recording. Get back a transcript, a structured summary, and a
reviewable bank of multiple-choice questions you can import straight into Canvas
— or hand out as a printed worksheet with an answer key.

Transcription runs **locally** with faster-whisper, so audio never leaves the
server. Only the transcript text is sent to an LLM, and only for summarizing and
question writing — free of charge by default, via OpenRouter's Free Models
Router, or Gemini, Claude, DeepSeek, OpenAI, Grok, or any other model OpenRouter
carries.

Long recordings can be uploaded **in parts**: split a 90-minute lecture into
three files and they are transcribed in order, then stitched into one continuous
transcript before anything else happens. Already have captions from Panopto,
Zoom or YouTube? **Import them** and skip transcription entirely.

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
| **Supporting material** | Optionally upload the lecture's own slides or handout (PPTX, PDF, DOCX, text). Speaker notes are read too. A transcript records what was *said*; the deck restores what was *shown* — the term the audio renders as "this thing here" |
| **Exam topics** | Optionally list the topics you will actually test. They outrank everything the app infers about importance, steering both where questions are placed and what each request is told to aim at |
| **Summarize** | Map-reduce over 10-minute windows so a 75-minute lecture gets even attention: title, abstract, learning objectives, key points, timestamped outline, key terms — informed by the deck's structure and your exam topics when supplied. **Resumable**: every window that finishes is kept, so a run that fails partway carries on from the next one instead of starting over and charging twice |
| **Generate** | Questions are placed by **importance, not by the clock**: the summary's objectives and key points decide which parts of the lecture are worth examining, so admin and tangents are quieted and the argument carries the quiz. Each item carries its timestamp and a verbatim supporting quote. Provider is a dropdown: OpenRouter, Gemini, Claude, OpenAI, Grok |
| **Wording** | Choose how stems are framed: **standalone** (default — asks the question directly, no "according to the lecture", reusable across a whole unit), reference the lecture, or an applied scenario. Provenance is unaffected: the timestamp and supporting quote are still recorded on every item |
| **Review** | An automatic second pass critiques the drafts and repairs or drops weak items |
| **Consistency check** | A separate 🔍 Review tab reads the lecture and flags claims worth checking again — against your uploaded slides (reliable: both texts are supplied) and against the model's general knowledge (advisory only). Transcription errors are reported separately, because most claims that look wrong in an automatic transcript are Whisper mishearing a term |
| **Validate** | Mechanical checks for "all of the above", duplicate options, giveaway answer length, negative stems, near-duplicate questions, missing provenance |
| **Balance** | Correct answers are redistributed across A/B/C/D — LLMs have a strong positional bias students notice fast |
| **Edit** | Every stem, option, and answer key is editable in the browser before export |
| **Regenerate** | Ask for as many alternative sets as you like — each is steered away from every question already written for that lecture, and labelled with the model that produced it, so switching models is a real comparison. Or replace any single question |
| **Export** | QTI 1.2 (Canvas) · QTI 2.1 · XLSX · CSV · DOCX · PDF · Markdown · SRT/VTT captions |
| **Accounts** | Sign-in with lockout and idle timeout, admin-created users, per-account settings, and personal API keys encrypted so only that user can read them |
| **Issued keys** | Mint a capped, revocable OpenRouter key per person — no collecting personal credentials |
| **Save** | Every lecture is stored to your account automatically — transcript, summary and question sets — and reopens where you left off |
| **Track** | Tokens and estimated cost, live during a run and cumulative per account |

Every generated question is a **draft for your review**, not a finished exam item.
The app is built to make review fast, not to make it unnecessary.

---

## Quick start (local)

```bash
git clone https://github.com/<you>/lecture-quiz-builder.git
cd lecture-quiz-builder

python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # set APP_SECRET (required); an API key is optional
streamlit run app.py
```

You do **not** need to install ffmpeg — audio decoding goes through `av` (PyAV),
whose wheels ship FFmpeg compiled in. Install it only if you want the
command-line tools for splitting recordings yourself: `brew install ffmpeg`
(macOS), `sudo apt install ffmpeg` (Ubuntu), `winget install ffmpeg` (Windows).

The first run downloads the Whisper weights (~150 MB for `small`) into
`.cache/whisper/`. Subsequent runs are instant.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repo (Community Cloud requires public,
   or a private repo on a paid plan).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the
   repo, branch `main`, main file `app.py`.
3. Open **Advanced settings → Secrets**:

   ```toml
   # Required — everything persisted is encrypted with a key derived from this.
   APP_SECRET = "a-long-random-string"

   # Strongly recommended on Streamlit Cloud: the container disk is wiped on
   # every restart, so without Dropbox your accounts will not survive one.
   DROPBOX_APP_KEY = "..."
   DROPBOX_APP_SECRET = "..."
   DROPBOX_REFRESH_TOKEN = "..."

   # Optional shared fallback keys — each user can save their own instead.
   OPENROUTER_API_KEY = "sk-or-v1-..."  # default provider
   ```

4. Deploy, then open the app: it will ask you to create the administrator
   account. See [Accounts and persistence](#accounts-and-persistence).

`requirements.txt` is picked up automatically. There is deliberately **no
`packages.txt`**: its presence makes Community Cloud run `apt-get` on every
build, and an expired Debian repository on the platform's base image then fails
the deploy before Python is reached — a breakage no app author can fix. Nothing
here needs a system package. See [docs/DEPLOY_NOTES.md](docs/DEPLOY_NOTES.md).

### Community Cloud limits — read this before you rely on it

The free tier gives you **1 CPU core and about 1 GB of RAM**. That has real
consequences:

- **Use the `small` Whisper model or below.** `medium` and `large-v3` exhaust
  memory and the container is killed mid-transcription. The app now measures the
  host's actual memory limit (cgroup-aware, so a container on a large machine is
  not fooled by the host's total) and refuses a model that cannot fit, naming one
  that can. `small` on 1 GB is flagged as tight rather than refused: it usually
  works, and blocking it would remove a setup people rely on.
- **Transcription is roughly 0.35× real time on `small`** — a 50-minute lecture
  takes about 18 minutes. The app shows an estimate before it starts.
- **A split recording is transcribed one part per run, and checkpointed.** Each
  part gets its own short script run, is saved the moment it finishes, and the
  next part is scheduled after it. This is what keeps a long lecture from being
  cut off: the platform is never asked for a half-hour unbroken run, only for a
  series of short ones. Reopening the app offers to finish anything left, so an
  interruption costs one part rather than the lecture — which matters because a
  killed container raises no exception, and anything not already written down is
  gone. Continuing does depend on the browser tab staying open; close it and the
  lecture pauses safely rather than failing.
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
| **OpenRouter** *(default)* | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) | One key, every model — the full catalog is fetched live and listed A–Z, free models marked. Starts on `openrouter/free` |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier covers light use |
| Anthropic Claude | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) | Best at following the item-writing rules |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | |
| xAI Grok | `XAI_API_KEY` | [console.x.ai](https://console.x.ai) | OpenAI-compatible endpoint |

#### Starting free, and staying on what you choose

The default model is **`openrouter/free`**, OpenRouter's Free Models Router: it
reads each request, filters to free models that can serve it, and picks one. So
the app works on a fresh OpenRouter key with no billing set up, and the list
never goes stale as free models come and go.

The trade-offs are real and not visible in the price. Free models have lower
rate limits and higher latency at busy times, availability varies, and — the one
that matters most here — **a different model may answer each call**, so two runs
over the same lecture can differ in quality in a way a pinned model's do not.
Fine for drafting; pin a specific model when a particular set matters.

Whatever you last generated with becomes your starting point next time you sign
in, per account. That is recorded on *use*, not on selection, so browsing the
model list does not change what you come back to.

#### Benchmarking models on *your* lectures

A generic leaderboard cannot tell you which model writes sound multiple-choice
items from a lecture transcript, in JSON, with a verbatim supporting quote,
under this app's item-writing rules. Only your own prompts can.

`scripts/export_eval.py` writes them out — the real system and user messages,
generated from one saved lecture, including the slide and exam-topic clauses
that lecture produces:

```bash
python3 scripts/export_eval.py --list
python3 scripts/export_eval.py --lecture "MGT 301 — Week 4" --out eval/
```

It also writes a rubric describing what a good answer looks like *here*: valid
complete JSON, `correct_index` aligned with the options, and a `source_quote`
that appears verbatim in the transcript — the three things that separate a
usable model from an unusable one, none of which a general benchmark measures.

Run [OpenRouter's Ori Eval](https://openrouter.ai/docs/guides/ori/eval) over
that directory (`ori login && ori eval`), then pin the winner in the sidebar.
Ori is a CLI with an interactive login, so it stays outside the deployed app —
see the script's docstring for why embedding it would not survive a restart.

**The export contains your transcript.** Keep it out of version control.

#### The OpenRouter model list

The full catalog is **fetched live** from `https://openrouter.ai/api/v1/models`
(public, no key needed), cached for an hour, and listed **alphabetically** with
🆓 marking every model OpenRouter serves at $0. A hardcoded list would be wrong
within a month — OpenRouter's roster turns over weekly, and a stale list both
offers retired models and hides new ones.

Sidebar controls: **Free models only**, a vendor filter, type-to-search in the
dropdown, **↻ Refresh list**, and a free-text box for any slug that isn't listed
yet. Live prices from the catalog feed the cost estimator, so the figure in the
sidebar reflects what the model actually costs today rather than a number written
into this repo months ago.

Two kinds of entry are filtered out, because the catalog carries more than this
app can use:

- **Image and video models** (Recraft, Wan, Hailuo) — they cannot return a quiz.
- **`:batch` variants** — asynchronous batch endpoints that accept the request
  but don't answer it interactively.

If the fetch fails, a bundled snapshot is used and the sidebar says so in as many
words. It is never presented as current.

The default model is **`deepseek/deepseek-v4-pro`** — strong enough to follow the
item-writing rules in the prompts, and roughly a tenth the price of the frontier
alternatives. `deepseek/deepseek-v4-flash` is cheaper again if you're generating
across a whole semester; `deepseek/deepseek-r1` is available but its reasoning
output makes it slower here for no gain on this task.

A note on the free models: they're genuinely free and fine for trying the app
out, but they're rate-limited and generally smaller, and it shows in the
questions — weaker adherence to the item-writing rules, more items caught by the
validators. At around three cents a lecture for `deepseek-v4-pro`, free is rarely
the economical choice once your time reviewing the output is counted.

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

Three things to know:

- **Every finished part is saved before the next one starts.** If the run is
  interrupted — a container killed for memory, a deploy, a restart — the parts
  already done are in your library. Reopen the app and it offers to finish the
  lecture: upload only the missing parts and they are appended to the saved
  timeline, so timestamps stay correct across the join and nothing is transcribed
  twice. You can also choose **Keep what I have** and work from a partial lecture.

  This is deliberate rather than incidental. A container that exceeds its memory
  limit is killed outright: no exception is raised, so no error handler runs and
  nothing gets a chance to save on the way down. Only what was already written to
  storage survives, which is why the checkpoint has to happen after each part
  rather than at the end.
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

### Importing a transcript instead of audio

Pick **📄 An existing transcript** and upload one of:

| Format | Timestamps |
|---|---|
| `.srt`, `.vtt` | Real — the best input; every question keeps a true position in the recording |
| Timestamped text (`[12:34] …`) | Real |
| `.txt`, `.md` | **Estimated** from a 150 wpm speaking rate |

Untimed text still works: everything downstream needs *a* timeline, so one is
estimated and the app labels it as such. Treat a question's timestamp as
approximate rather than a place to scrub to.

This path is often better than transcribing. A human-corrected department
transcript beats what `small` Whisper produces on a CPU, and it arrives in
seconds rather than twenty minutes.

### Getting the number of questions you asked for

Ask for 12 and you get 12. That takes more than one prompt, because three things
independently eat into the count: a model asked for three questions from a
section often returns two; malformed items are dropped during parsing; and the
review pass then *rejects* weak drafts.

So the count is enforced rather than hoped for. After the first pass the app
counts what survived and runs further rounds for the shortfall — including after
the review pass, so a rejected item is replaced instead of simply lost. Surplus
items stay in the bank unchecked (tick any to include it), rejected drafts stay
visible with the reviewer's reason, and if the lecture genuinely cannot support
the request, the app says so instead of quietly handing back fewer.

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

## Your saved lectures

Transcription is the slow step, so nothing relies on you remembering to save. A
lecture is written to your account the moment it is transcribed or imported, and
the same entry is updated when you generate a summary and questions.

### Name it first

**Name this lecture** sits above the file picker, before you upload anything.
Whatever you type there names the library entry, the quiz title that lands in
Canvas, and the filename of every export — so `MGT 301 — Week 4, Aggregate
Planning` follows the work instead of `20260901-111056-sp001_combined`.

The field is deliberately *before* the upload rather than after. Renaming later
has always been possible, but a name you have to remember to fix is a name that
stays wrong, and a semester of those reads as a list of recorder timestamps.
Leave it blank and the filename is still used, as before.

The name is set once and then holds: a summary and question sets generated hours
later attach to that same entry, and the per-part checkpoints during a split
recording never overwrite it. Renaming in the Library moves the whole lecture,
not just the listing — and if that lecture is the one you have open, the header
and your next export follow it.

### Finding it again

The **📚 Library** tab lists what you have: length, word count, whether it has a
summary, and how many question sets. Past five lectures a search box appears,
matching on the name and on the source filename for when you only remember which
recording it came from. **Open this lecture** restores the whole working state —
transcript, summary, and every set — so you can export or regenerate without
paying for any of it again. Rename and delete are there too.

Once a lecture is open, its name appears under the app title, because every tab
below — transcript, summary, questions, exports — belongs to that one lecture.

Two things worth knowing:

- **On Streamlit Community Cloud, configure Dropbox.** Local files are wiped on
  every restart, so without it the library empties itself periodically. The
  Library tab warns when storage is not durable.
- **Lectures are encrypted at rest under the app key**, like everything else in
  the store — *not* sealed to your password the way personal API keys are. That
  is deliberate: a key is trivially re-entered, a semester of transcripts is not,
  and an admin password reset would otherwise destroy them. If your deployment
  needs transcripts unreadable without the user present — recordings of class
  discussion, say — `src/library.py` documents the one-line change, and the
  trade it makes.

## When a run doesn't finish

Every model call is recorded, and the **Run report** under the result shows each
one — which part of the lecture it covered, whether it succeeded, and what went
wrong if it didn't. It stays on screen after the progress bar disappears, and
persists on the Questions tab, so a failed run can be diagnosed rather than
guessed at.

| What you see | What it means |
|---|---|
| **The run stopped early** | Every call in a stage failed — the error names the reason from the provider |
| **Finished with gaps** | Some chunks failed; the questions you have came from the ones that worked |
| ✗ in the run report | That call never reached the model, or came back unusable |
| `skipped` in the run report | The reply was cut off, so it was retried with a smaller batch |

Common causes, each with specific advice shown in-app:

- **Reply cut off** — the model hit its output limit. The app now retries that
  chunk automatically with half the batch size; if it still fails, ask for fewer
  questions or shorten the chunk length under **Context & advanced**. Reasoning
  models are the usual culprits, because their thinking counts against the
  output budget.
- **Out of credit / cap reached** — top up at the provider, or raise the cap in
  **Admin → Issued API keys**.
- **Rate limited** — free OpenRouter models have tight limits; wait or switch.
- **Key rejected** — check it's an inference key for the selected provider, not a
  management key.

## Accounts and persistence

### First run

Set `APP_SECRET` (see below), start the app, and it asks you to create the
administrator account. From then on, **only an administrator creates accounts** —
there is no self-registration, which is the right default for a public URL.

New users get a temporary password and are made to choose their own at first
sign-in. Admins can change roles, disable accounts, reset passwords, and delete
users from the **🛠️ Admin** tab. The last remaining administrator cannot be
deleted, demoted, or disabled — locking yourself out of your own deployment is an
easy mistake and an annoying one to undo.

### What persists per account

- The provider and model you last used, plus question count, options, Bloom
  levels, difficulty and course context — press **💾 Save these settings to my
  account** and the next sign-in starts where you left off.
- **An API key**, encrypted — either one you saved yourself, or one an
  administrator issued to you (below). Each account holds its own, so a
  colleague's usage bills their account and nobody can spend your budget.
  A key you save yourself is encrypted with a key derived from **your password**,
  so nobody else — including an administrator — can read it back. The
  consequence: if an administrator resets your password, your saved keys are
  gone and you re-enter them. That is the guarantee working, not a bug.
- Your usage history.

### Security features

- **Sign-in throttling** — 5 failed attempts locks an account for 15 minutes.
- **Idle timeout** — sessions end after 8 hours, discarding the key that reads
  your saved credentials.
- **Security log** — sign-ins, lockouts, account changes and key events, visible
  to admins and exportable. Credentials themselves are never logged.
- **My security panel** — change your password (saved keys survive), or forget
  every personal key at once.
- **Key rotation** — `python3 scripts/rotate_key.py --new-secret …` re-encrypts
  the whole store, so `APP_SECRET` can actually be changed after an exposure.

[SECURITY.md](SECURITY.md) explains the design, what it deliberately costs, and
what remains true anyway.

### Issued keys — the safer way to give colleagues access

Instead of asking people for their personal API key, issue each of them a
**capped, revocable OpenRouter key** from the Admin tab. This is the recommended
setup for any deployment other people sign in to.

1. Create a **management key** at
   [openrouter.ai/settings/management-keys](https://openrouter.ai/settings/management-keys).
   It is a different kind of key from an inference key — an ordinary `sk-or-v1-…`
   used for chat will not work here, and the app says so if you paste one.
2. **Admin → Issued API keys** → paste it. The app verifies it against OpenRouter
   before saving, then stores it encrypted.
3. Set a default cap (e.g. `$5.00`) and a reset period (monthly, weekly, daily).
4. Open any account and press **🔑 Issue an OpenRouter key**.

That user now has a working key on their account without pasting anything. From
the same panel you can watch spend against the cap, raise or lower it, pause the
key, or revoke it outright — and deleting an account revokes its key first, so a
departed colleague never leaves a live credential behind.

Why this is better than storing personal keys:

| | Personal key saved in the app | Issued key |
|---|---|---|
| A leak costs | Their entire OpenRouter balance | At most that key's cap |
| Revoking it | Their problem, on their account | One click, here |
| Who is the custodian | You, of their credential | Nobody — the key is yours to begin with |
| Per-user spend | This app's estimate | OpenRouter's own accounting |

At roughly three cents a lecture, a `$5.00` monthly cap is about 150 lectures —
generous for a colleague and survivable as a mistake.

The management key is the one credential that still matters: it can mint and
revoke keys on your account. It is stored encrypted under `APP_SECRET`, and
[SECURITY.md](SECURITY.md) covers what that does and does not protect against.

### Setting `APP_SECRET`

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"

# No Python handy? openssl ships with macOS and most Linux distros:
openssl rand -base64 48
```

On macOS use `python3` — plain `python` is not a command there.

Put it in `.streamlit/secrets.toml` (or `.env` locally). Everything stored is
encrypted with a key derived from it: accounts, settings, API keys and the usage
ledger are opaque blobs on disk or in Dropbox.

**Back `APP_SECRET` up where you back up passwords.** To change it, use
`scripts/rotate_key.py` *while you still have the current one* — that re-encrypts
everything under the new secret. Losing it outright is still unrecoverable.

Without `APP_SECRET` the app still runs, but nothing is saved and it says so.

### Storage backends

| | When to use it | Survives a Streamlit Cloud restart? |
|---|---|---|
| **Encrypted local files** (default) | A machine you control, or local development | ❌ — the container disk is wiped |
| **Dropbox app folder** | A shared deployment on Streamlit Cloud | ✅ |

Dropbox wins automatically when all three of its credentials are present.

**The quickest route:** create the app and tick its four permissions in the
Dropbox App Console (steps 1–2 below), then run
`python3 scripts/setup_dropbox.py` — it does the authorization and token
exchange and prints a finished secrets block to paste. The full walkthrough,
with the ordering gotcha that catches most people, is
**[docs/DROPBOX.md](docs/DROPBOX.md)**. By hand:

1. [dropbox.com/developers/apps](https://www.dropbox.com/developers/apps) →
   **Create app** → Scoped access → **App folder** → name it.
2. Under **Permissions**, tick `files.metadata.read`, `files.metadata.write`,
   `files.content.read` and `files.content.write`, then press **Submit**.
   Do this *before* step 3 — a token keeps whatever scopes it was issued with,
   so ticking permissions afterwards does not upgrade it.
3. Visit, with your app key substituted in:
   `https://www.dropbox.com/oauth2/authorize?client_id=APP_KEY&response_type=code&token_access_type=offline`
   Approve, and copy the authorization code. (`token_access_type=offline` is what
   makes Dropbox return a long-lived refresh token instead of one that dies in
   four hours.)
4. Exchange it for a refresh token — the code is single-use and expires in
   minutes, so do this immediately:

   ```bash
   curl -u APP_KEY:APP_SECRET https://api.dropboxapi.com/oauth2/token \
     -d code=THE_CODE -d grant_type=authorization_code
   ```

5. Put `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET` and the `refresh_token` from that
   response into your secrets, alongside `APP_SECRET`.
6. **Verify it, rather than assuming:** `python3 scripts/check_dropbox.py`. Bad
   credentials do not crash the app — it falls back to local files, which look
   identical until a restart wipes them. The checker does a full encrypted
   round trip and names the step that failed. The sidebar also reports the
   live backend.

**An honest note on Dropbox.** It is file storage, not a database — no
transactions, no row locking. Two people saving at the same instant means one
write lands on top of the other. The backend uses revision-checked writes so a
collision is detected and retried rather than silently swallowed, which is enough
for a handful of instructors sharing a deployment. For a whole department, use a
real database: `Store` in `src/storage.py` is a deliberately small three-method
interface, so a Postgres or Supabase backend is one class, not a rewrite.

### Before you share the URL

Storing other people's API keys on a free public host is a real decision, not a
formality. Encrypted at rest is not the same as safe: anyone who can read your
Streamlit secrets can decrypt everything. If that matters, run it on a campus
machine, or have each user paste their key per session rather than saving it.

**[SECURITY.md](SECURITY.md)** works through this properly — what the current
design protects against, what it does not, and the changes worth making before
colleagues sign in. The short version: issue capped, revocable per-user keys
through OpenRouter's provisioning API rather than protecting unlimited ones, and
key the encryption on each user's password so the server cannot decrypt without
them.

## Usage and cost tracking

The sidebar shows tokens and estimated cost climbing **during** a run — the LLM
client reports after every call, so you see the meter move rather than learning
the cost afterwards.

The **📊 Usage** tab keeps the history: totals, estimated cost per day, per model,
and per pipeline step, with every run in a table you can export as CSV.
Administrators can switch the scope to **Everyone** and see spend per user.

Costs are estimates from published rates. Your provider's dashboard is the
authority on what you were actually billed — this is here so nothing is a
surprise before you get there.

## Project layout

```
lecture-quiz-builder/
├── app.py                        Streamlit UI and pipeline orchestration
├── requirements.txt
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
│   ├── openrouter_catalog.py     live model list: fetch, filter, sort, free flags
│   ├── storage.py                encrypted Store: local files and Dropbox
│   ├── hostinfo.py               memory ceiling detection and the model guard
│   ├── materials.py              slides/handouts: extract, match to a window
│   ├── factcheck.py              consistency review: flag, never adjudicate
│   ├── accounts.py               users, roles, scrypt passwords, saved keys
│   ├── usage.py                  live meter and the persistent usage ledger
│   ├── provisioning.py           issue/cap/revoke per-user OpenRouter keys
│   ├── appconfig.py              encrypted deployment settings
│   ├── keymgmt.py                key derivation, purpose subkeys, rotation
│   ├── audit.py                  security log (never records credentials)
│   ├── diagnostics.py            per-call run report and failure advice
│   ├── library.py                saved lectures per account
│   ├── transcript_import.py      SRT / VTT / timestamped / plain-text import
│   ├── prompts.py                every prompt, in one editable place
│   ├── summarize.py              map-reduce summarization
│   ├── mcq.py                    generation, validation, balancing, critique,
│   │                             avoid-lists, single-question replacement
│   └── exporters/
│       ├── qti.py                QTI 1.2 (Canvas) and QTI 2.1
│       ├── tabular.py            CSV, XLSX
│       ├── documents.py          DOCX, PDF, Markdown
│       └── transcript_formats.py TXT, SRT, WebVTT
├── docs/
│   ├── DROPBOX.md                step-by-step encrypted-storage setup
│   └── DEPLOY_NOTES.md           why there is no packages.txt
├── scripts/
│   ├── rotate_key.py             re-encrypt everything under a new APP_SECRET
│   ├── setup_dropbox.py          interactive OAuth: prints a secrets block
│   ├── export_eval.py            this app's real prompts, for Ori Eval et al.
│   ├── check_dropbox.py          preflight: full encrypted round trip to Dropbox
│   └── check_build.py            preflight: app.py and src/ are the same vintage
└── tests/
    ├── test_pipeline.py             schema, validation, balancing, all exporters
    ├── test_llm_clients.py          provider wiring, against stubbed SDKs
    ├── test_multipart_and_regen.py  part stitching, avoid-lists, replacement
    ├── test_openrouter_catalog.py   catalog parsing, filters, free detection
    ├── test_import_and_counts.py    transcript import, question-count guarantee
    ├── test_accounts_storage_usage.py  crypto, accounts, roles, usage ledger
    ├── test_provisioning.py         issued keys: mint, cap, inspect, revoke
    ├── test_security_hardening.py   crypto, envelope encryption, lockout, audit
    ├── test_failure_reporting.py    truncation, rate limits, silent-failure guards
    ├── test_library.py              save/load round-trips, per-account isolation
    ├── test_question_focus.py       importance-weighted allocation, focus clause
    ├── test_alternative_sets.py     third and fourth sets, cross-set difference
    ├── test_truncation_salvage.py   keeping questions written before a cut-off
    ├── test_materials.py            real PPTX/PDF/DOCX, matching, exam topics
    ├── test_eval_export.py          the benchmark export carries the real task
    ├── test_framing_and_review.py   stem wording, and the review's honesty
    ├── test_dropbox_backend.py      health check, error advice, encrypted round trip
    ├── test_crash_recovery.py       per-part checkpoints, resume, timeline joins
    ├── test_summary_resume.py       resuming a summary, salvage, local fallback
    ├── test_deploy_integrity.py     app.py and src/ must be the same vintage
    └── test_hostinfo.py             memory detection and the model-fit verdicts
```

`scripts/setup_dropbox.py` walks through Dropbox authorization;
`scripts/check_dropbox.py` verifies it end to end before you rely on it;
`scripts/rotate_key.py` re-encrypts the store under a new `APP_SECRET`;
`scripts/check_build.py` catches a half-copied deploy before you push it.

```bash
pytest -q          # 524 tests, no API keys or network needed
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

Free models on OpenRouter cost nothing at all — see the note above on what you
give up.

The sidebar shows actual token counts and an estimated cost after each run, using
live prices from OpenRouter's catalog. They are still approximate: OpenRouter
routes to whichever upstream host is cheapest or fastest at the moment, so the
real rate moves. Check the OpenRouter dashboard for actual spend. A slug typed
into the free-text box that isn't in the catalog shows tokens but no dollar
figure, which is the honest answer rather than a fabricated one.

An alternative set costs about the same as the original run. A single replacement
costs one small call.

Turning off the review pass roughly halves the cost and noticeably lowers
question quality. At three cents a lecture that is not a trade worth making.

---

## License

MIT. See [LICENSE](LICENSE).
