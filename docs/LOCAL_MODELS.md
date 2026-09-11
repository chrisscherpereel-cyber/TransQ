# Running the questions on your own computer

The app can send the lecture to a model running on your own machine instead of a
hosted API. Nothing about the lecture leaves the computer, there is no
per-lecture cost, and it works with the internet off.

---

## Read this first: it only works when the app runs on your machine too

**A model on your laptop is not reachable from Streamlit Community Cloud.**
When the app runs on Streamlit's servers, `localhost` means *Streamlit's*
container — not your computer. There is no address you can type that bridges
that, and no setting that fixes it. It is what "local" means.

So using a local model is really two things:

1. Install a model runner and download a model. *(15 minutes, once.)*
2. Run the app on that same computer, instead of visiting the Streamlit URL.
   *(10 minutes, once. It keeps working after that.)*

Your hosted deployment keeps working exactly as it does now. This is a second
way to run the app, not a replacement — and the two share nothing, including
your saved lectures, unless you have Dropbox storage configured.

---

## Step 1 — Install a model runner

Two good options. **Ollama** is the recommendation: it installs as a background
service, so the app can find it without you opening anything, and it is the same
two commands on both platforms. **LM Studio** is the one to choose if you would
rather click than type.

### Ollama — macOS

1. Download from [ollama.com/download](https://ollama.com/download) and drag
   Ollama to Applications. Open it once; it puts an icon in the menu bar and
   starts automatically from then on.
2. Open **Terminal** (⌘-Space, type "Terminal") and download a model:

   ```bash
   ollama pull qwen2.5:14b
   ```

   The first pull is several gigabytes and takes a few minutes. Later pulls of
   other models reuse nothing, so each one costs its own download.

3. Check it worked:

   ```bash
   ollama list
   ```

If you prefer Homebrew: `brew install ollama` then `brew services start ollama`.

### Ollama — Windows

1. Download **OllamaSetup.exe** from
   [ollama.com/download](https://ollama.com/download) and run it. It installs a
   background service and starts on login.
2. Open **PowerShell** (Start menu, type "PowerShell") and run the same two
   commands as above:

   ```powershell
   ollama pull qwen2.5:14b
   ollama list
   ```

### LM Studio — macOS and Windows

1. Download from [lmstudio.ai](https://lmstudio.ai) and install it normally.
2. Use the **search tab** (magnifying glass) to find a model and click
   **Download**. LM Studio marks which ones fit in your machine's memory, which
   is genuinely useful the first time.
3. Go to the **Developer** / **Local Server** tab and click **Start Server**. It
   listens on port **1234**.
4. Leave LM Studio open. Unlike Ollama, it stops serving when you quit it.

---

## Step 2 — Pick a model that fits

Model names change every few months, so treat the names below as starting points
and browse [ollama.com/library](https://ollama.com/library) for what is current.
The **size class** is the part that matters and does not go stale.

On an Apple Silicon Mac, the GPU can use most of your unified memory, so the
figure to check is your total RAM (Apple menu → About This Mac). On Windows with
an NVIDIA card, the limit is the card's own VRAM, not system RAM — a 32 GB PC
with an 8 GB card is an 8 GB machine for this purpose.

| Your memory | Size class | Starting points | What to expect |
|---|---|---|---|
| 8 GB | ~3B | `llama3.2:3b`, `phi4-mini` | Struggles with the JSON structure this app needs. A fallback, not a choice. |
| 16 GB | ~8B | `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b` | The usable floor. Drafts real questions; review them closely. |
| 24–32 GB | ~14B | `qwen2.5:14b`, `phi4` | Noticeably better at following the item-writing rules. **The sweet spot.** |
| 36–48 GB | ~30B | `qwen2.5:32b` | Comparable to a mid-tier hosted model for this task. |
| 64 GB+ | ~70B | `llama3.3:70b` | The best that runs locally, and slow enough that you will feel it. |

The app reads your machine's memory and tells you which class it can hold, right
under the server address in the sidebar — so you do not have to work this out
yourself.

**Reasoning models are a poor fit here.** Models that "think" before answering
spend their output budget on reasoning, which is exactly what causes the
truncation this app has to salvage. A plain instruction-tuned model of the same
size does better on this task and is several times faster.

---

## Step 3 — Run the app on that computer

### The short way

1. On GitHub, click the green **Code** button → **Download ZIP**. Unzip it —
   on a Mac it lands in Downloads as a folder called `TransQ-main`.
2. Open the `scripts` folder inside it and double-click:
   - **macOS** — `mac_setup.command`
   - **Windows** — `windows_setup.bat`

That script creates a private Python environment inside the folder, installs
everything, generates your `APP_SECRET`, tells you whether it found Ollama, and
starts the app. First run takes a few minutes; after that it starts in seconds.
Run it again any time you want to start the app.

> **If macOS refuses to open it** — "permission denied", or the file opens in a
> text editor — that is because downloading a ZIP strips the permission that
> makes a file runnable. Open **Terminal** and paste this instead, which does not
> need it:
>
> ```bash
> cd ~/Downloads/TransQ-main && bash scripts/mac_setup.command
> ```
>
> Adjust the folder name if yours differs. If macOS says the file is from an
> unidentified developer, right-click it → **Open** → **Open**, which is the
> standard way to approve a file you downloaded yourself.

### The manual way

If you would rather see each step, or the script failed and you want to know
where:

```bash
cd TransQ

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Set `APP_SECRET` so your work is saved. On macOS:

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

On Windows (PowerShell):

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Paste the printed string into `.env` as `APP_SECRET=...`. Then:

```bash
streamlit run app.py
```

Either way, your browser opens at `http://localhost:8501` and the app asks you
to create an admin account the first time.

> **Windows note:** if the script says Python is not installed, get it from
> [python.org/downloads](https://www.python.org/downloads/) and **tick "Add
> python.exe to PATH"** on the installer's first screen. That checkbox is the
> difference between the script working and not.

> **This is a separate library from your hosted app.** Lectures saved here live
> in this folder, encrypted under this `APP_SECRET`. If you want one library
> across both, configure Dropbox storage the same way in each — see
> `DROPBOX.md`. And back up `APP_SECRET` where you keep passwords: lose it and
> the saved lectures cannot be read, by anyone, ever.

---

## Step 4 — Point the app at the model

In the sidebar, open **Question generation**:

1. Set **LLM provider** to **On this computer**.
2. The **Server address** should already be right —
   `http://localhost:11434/v1` for Ollama, `http://localhost:1234/v1` for LM
   Studio. A bare `localhost:1234` works too; the app tidies it up.
3. The **Model** dropdown fills with whatever you have downloaded. If it is
   empty, press **Check again**.

There is no API key field, because a local server does not use one.

---

## What to actually expect

**Speed.** On an Apple Silicon Mac with a ~14B model, budget roughly 5–15
minutes to summarize and generate questions for a 60-minute lecture, against
under a minute for a hosted model. On Windows without a dedicated GPU, several
times that. The app's per-window progress makes this bearable to watch, and the
resume behaviour means a failure partway does not cost you the finished windows.

**Quality.** A ~14B local model writes usable draft questions but is meaningfully
weaker than a good hosted model at the fiddly parts: plausible distractors,
staying inside a Bloom level, and not giving the answer away through option
length. The app's validation and review passes catch some of that. Read the
drafts more carefully than you would from a frontier model.

**Comparing them.** This is worth doing directly rather than guessing. Generate
one set with your local model, then use **Regenerate** to make a second set with
a hosted model on the same lecture. Each set is labelled with the model that
produced it, and the second is steered away from the first's questions, so you
are comparing two genuine attempts rather than the same questions twice.

**Transcription was always local.** Whisper has never sent your audio anywhere —
that part of the privacy story was already true. What changes here is the
*transcript text*, which previously went to whichever API you chose.

---

## Why this matters for class recordings

The FERPA question in the README turns on one thing: transcript text leaving
your machine. With a local model, it does not — the audio is transcribed by
Whisper locally and the text goes to a model on the same computer.

That makes recordings of class *discussion*, where students are audible and
identifiable, a different proposition than they are with a hosted API. It does
not make them unproblematic — the recording still exists, and the disclosure
decision still belongs to your campus privacy office — but it removes the
third-party disclosure from the chain.

---

## Troubleshooting

**"Nothing is listening on port 11434."** Ollama is not running. On macOS, open
the Ollama app. On Windows, check for its icon in the system tray, or restart
the machine — it starts on login. `ollama list` in a terminal is the fastest
test: if that works, the server is up.

**LM Studio worked earlier and not now.** LM Studio only serves while it is open
and the server is started. Reopen it, go to the Developer tab, and press
**Start Server** again. Ollama does not have this problem, which is the main
reason it is the recommendation.

**The model list is empty but the server is connected.** The server is running
and no models are downloaded. Run `ollama pull qwen2.5:14b`, or use LM Studio's
search tab.

**Everything is extremely slow, or the machine freezes.** The model is bigger
than your memory and the machine is swapping to disk. Drop one size class —
a smaller model that fits is dramatically faster than a larger one that does
not. `ollama rm <model>` removes the one you no longer want.

**Replies keep getting cut off.** Shorten the chunk length in **Context &
advanced**, ask for fewer questions per run, or move to a larger model. Whatever
finished before the cut is kept either way, and pressing the button again
resumes rather than starting over.

**It worked locally but not on the Streamlit URL.** Expected, and not fixable —
see the top of this page. The hosted app cannot reach your computer.
