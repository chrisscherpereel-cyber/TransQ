# Setting up encrypted Dropbox storage

**Time:** about 15 minutes, once. **Cost:** nothing — a free Dropbox account is enough.

## Why this step exists

Streamlit Community Cloud gives your app a container that is wiped whenever the
app restarts — which happens on every code push, after periods of inactivity, and
whenever Streamlit feels like it. Without Dropbox, that wipe takes your accounts,
your users' saved API keys, your usage ledger and every saved lecture with it.

Dropbox is where those files go so they survive. What lands there is **not** your
transcripts in readable form: every document is encrypted on your machine, with a
key derived from `APP_SECRET`, before it is uploaded. Dropbox stores opaque
blobs. Anyone who gets into the Dropbox account — including Dropbox — sees
`4f2a…` and file sizes, not lecture content.

That also means the two secrets do different jobs and are not interchangeable:

| | What it protects | If you lose it |
|---|---|---|
| `APP_SECRET` | The encryption key. Makes stored data readable. | Everything stored is unrecoverable. **Back it up.** |
| Dropbox credentials | Where the encrypted files live. | Generate new ones; nothing is lost. |

---

## The short way

Steps 3–5 below — copying the app key, hand-building an authorize URL, and
running a `curl` command against a code that expires in minutes — are where this
usually goes wrong. A script does them for you:

```bash
python3 scripts/setup_dropbox.py
```

It walks you through the App Console prerequisites, opens the authorize page,
takes the code, exchanges it, checks that all four permissions actually came
back, and prints a finished secrets block to paste. Do **Step 1** and **Step 2**
first — the script cannot tick permissions for you, and a token generated before
you press Submit is the single most common failure.

The rest of this document explains each step by hand, for when you want to know
what the script is doing or something has gone wrong.

---

## Step 1 — Create the Dropbox app

1. Sign in at [dropbox.com/developers/apps](https://www.dropbox.com/developers/apps)
   and click **Create app**.
2. Choose **Scoped access**.
3. Choose **App folder** — *not* Full Dropbox.

   This matters. "App folder" confines the app to a single folder Dropbox creates
   for it (`Apps/your-app-name/`). A leaked token then exposes that folder and
   nothing else in your Dropbox. There is no reason to grant more.
4. Name it something you will recognise later, e.g. `mgt301-quiz-builder`. The
   name must be unique across all of Dropbox, so expect to try twice.

You land on the app's settings page. Leave it open.

## Step 2 — Grant the four permissions

Click the **Permissions** tab and tick exactly these:

- `files.metadata.read`
- `files.metadata.write`
- `files.content.read`
- `files.content.write`

Then press **Submit** at the bottom of the page. Nothing takes effect until you do.

> **The single most common way this goes wrong.** Scopes are baked into a token
> at the moment it is issued. If you generate a token first and tick permissions
> afterwards, that token keeps the old, narrower scopes forever, and the app will
> fail with a `missing_scope` error that looks like a credentials problem.
> **Permissions first, Submit, then token.** If you have already made a token,
> just make a new one after submitting.

You do **not** need `account_info.read`. The app never asks who you are.

## Step 3 — Copy the app key and secret

Back on the **Settings** tab, find **App key** and **App secret** (click *Show*).
Keep them somewhere you can paste from in a minute. These are not the final
credentials — they are used to mint the token in the next step.

## Step 4 — Authorize the app and get a code

Paste this into your browser with your own app key substituted for `APP_KEY`:

```
https://www.dropbox.com/oauth2/authorize?client_id=APP_KEY&response_type=code&token_access_type=offline
```

`token_access_type=offline` is the part that matters: it is what makes Dropbox
return a **refresh token**. Without it you get an access token that expires in
four hours and your app breaks over lunch.

Click **Allow**. Dropbox shows you an authorization code — a long string on the
page. Copy it. It is single-use and expires within minutes, so do the next step
straight away.

## Step 5 — Exchange the code for a refresh token

In Terminal (macOS: use `python3`, never `python`):

```bash
curl -u APP_KEY:APP_SECRET https://api.dropboxapi.com/oauth2/token \
  -d code=THE_CODE_YOU_JUST_COPIED \
  -d grant_type=authorization_code
```

The response is one line of JSON:

```json
{"access_token":"sl.u.AF...","token_type":"bearer","expires_in":14400,
 "refresh_token":"3TgH...","scope":"files.content.read files.content.write ...","uid":"...","account_id":"..."}
```

Two things to check before moving on:

- **`refresh_token` is present.** If it is missing, you left
  `token_access_type=offline` out of the URL in step 4. Redo step 4.
- **`scope` lists all four permissions.** If it does not, you generated the code
  before pressing Submit in step 2. Redo steps 4 and 5.

Copy the `refresh_token` value. Ignore `access_token` — the app never uses it.

## Step 6 — Generate your APP_SECRET

This is the encryption key, and it has nothing to do with the "App secret" from
Dropbox. Generate a fresh one:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

or, if Python is being difficult:

```bash
openssl rand -base64 48
```

**Put it in your password manager now.** Everything the app stores is encrypted
with a key derived from it. If you lose it, the data is gone — that is the design,
not a bug. (You can *change* it later with `scripts/rotate_key.py`, but only while
you still have the current one.)

## Step 7 — Put the four values into your secrets

**On Streamlit Community Cloud:** open your app → **⋮** → **Settings** →
**Secrets**, and paste:

```toml
APP_SECRET = "the-long-random-string-from-step-6"

DROPBOX_APP_KEY = "your-app-key"
DROPBOX_APP_SECRET = "your-app-secret"
DROPBOX_REFRESH_TOKEN = "the-refresh-token-from-step-5"

# Optional — defaults to /lecture-quiz-builder inside the app folder.
# DROPBOX_FOLDER = "/lecture-quiz-builder"
```

Save. Streamlit restarts the app automatically.

**Running locally:** the same keys go in `.streamlit/secrets.toml`, or as
`KEY=value` lines in a `.env` file. Both are already in `.gitignore`. Never commit
either one — a refresh token in a public repo is a live credential.

## Step 8 — Verify it actually works

Do not skip this. If any credential is wrong, the app does not crash: it warns
once and quietly falls back to local files, which look identical until the next
restart eats them.

```bash
python3 scripts/check_dropbox.py
```

It connects, writes a real encrypted document, reads it back, confirms the bytes
sitting on Dropbox are not readable, and deletes the probe. Every step is named,
so a failure tells you which one and what to change.

You can also confirm from inside the app: the sidebar reports the active storage
backend. It should say **Dropbox**, not "local files".

---

## First run

The first time the app starts with storage working:

1. Sign in with the bootstrap administrator account (see the README).
2. **Change that password immediately.**
3. Create accounts for your colleagues under **Admin → Accounts**. There is no
   self-registration by design — the only way in is an account you made.
4. Each user enters their own LLM API key under the key panel; it is sealed with
   a key derived from *their* password, so no one else on the deployment — you
   included — can read it.

Look in Dropbox afterwards and you will see `Apps/your-app-name/lecture-quiz-builder/`
containing `.enc` files and a small plaintext `keyring.json`. The keyring holds a
salt, not a key; it is meant to be readable.

---

## When something is wrong

| Symptom | Cause | Fix |
|---|---|---|
| `missing_scope` / "missing a permission" | Token was issued before you pressed Submit | Redo steps 4–5. Ticking scopes does not upgrade an existing token. |
| `invalid_grant` | The authorization code was reused or expired | Codes are single-use and short-lived. Redo step 4, then 5 immediately. |
| `invalid_client` | App key/secret wrong, or from a different app | Re-copy from the Settings tab. Watch for a trailing space. |
| `expired_access_token` | You saved the access token, not the refresh token | Use the `refresh_token` field from step 5. |
| Sidebar says "local files" with credentials set | One of the three values is empty or misspelled | All three are required; any missing one means local files. Run the checker. |
| "Encryption is not configured" | `APP_SECRET` is unset | Step 6, then step 7. |
| Everything unreadable after a redeploy | `APP_SECRET` changed | Restore the old value from your password manager. There is no other route. |

**Rotating a leaked Dropbox token:** in the App Console, generate a new refresh
token (step 4–5 again) and update the secret. Your data is untouched — it is
encrypted under `APP_SECRET`, which has not changed.

**Rotating `APP_SECRET`:** use `scripts/rotate_key.py` while you still hold the
current one. It re-encrypts every document under the new key. Do not just change
the value in secrets — that makes everything unreadable.

---

## What Dropbox is not

Dropbox is file storage, not a database. There are no transactions and no row
locking. Two people saving at the same instant means one write can land on top of
the other. The backend mitigates this with revision-checked writes (a collision is
detected and retried rather than silently swallowed) and one document per user and
per lecture, so concurrent saves usually do not touch the same file at all.

That is enough for a handful of instructors sharing a deployment. It is not enough
for a whole department. At that scale the right answer is a real database, and
`Store` in `src/storage.py` is a deliberately small three-method interface — a
Postgres or Supabase backend is one class, not a rewrite.
