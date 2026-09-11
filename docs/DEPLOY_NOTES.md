# Why there is no `packages.txt`

Streamlit Community Cloud runs `apt-get` on every build if — and only if — the
repository contains a `packages.txt`. This one used to contain a single line,
`ffmpeg`, and that line was enough to make deployment fail entirely:

```
E: Release file for http://deb.debian.org/debian-security/dists/bullseye-security/InRelease
   is expired (invalid since 1h 52min 44s).
❗️ installer returned a non-zero exit code
❗️ Error during processing dependencies!
```

Nothing in this app was wrong. Debian's `bullseye-security` metadata expired on
the platform's base image, `apt-get update` returned non-zero, and the build
stopped before Python was ever reached. It is not a failure anyone deploying an
app can fix, and it recurs whenever that metadata lapses again.

**The system `ffmpeg` package was never needed.** Nothing in this codebase shells
out to `ffmpeg` or `ffprobe`. Audio decoding goes through `av` (PyAV), a direct
dependency in `requirements.txt`, and its manylinux wheels ship FFmpeg compiled
in — `libavcodec`, `libavformat`, `libavutil`, `libswresample` and the rest are
inside the wheel. faster-whisper decodes through the same library.

So `packages.txt` was removed. With no apt manifest, Community Cloud skips the
apt step, an expired Debian repository cannot break the build, and deploys are
faster besides.

## If you run this outside Streamlit Cloud

You still do not need system ffmpeg — `pip install -r requirements.txt` brings a
working decoder. Install it anyway if you want the command-line tools for
splitting recordings by hand:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install ffmpeg        # Windows
```

That is a convenience for you, not a requirement of the app.

## If a build fails on apt again

Do not add `packages.txt` back to install something unless you are certain it is
needed. A pure-Python dependency with binary wheels is always the safer choice on
a platform whose base image you do not control: it cannot be broken by someone
else's repository metadata.

---

# Half-copied deploys, and how to recognise one

This has now cost two debugging sessions, and it looks like an application bug
both times. It is not one.

Deploying this app is a file copy — a zip extracted over a checkout, or a push
to the branch Streamlit Cloud serves. If `app.py` arrives and something under
`src/` does not, **nothing complains**. Python imports the older module happily.
The app starts, signs you in, transcribes, and then dies partway through a run.

## The give-away

```
File "/mount/src/…/app.py", line 1593, in run_generation
    summary, chunks, sections = summarize_transcript(
        client,
    ...<9 lines>...
        report=report,
    )
```

Read what is *missing*: there is **no frame inside `summarize_transcript`**. The
traceback stops at the call. That only happens when the exception comes from
binding the arguments — Python never entered the function. Which means the
function on disk has a different signature from the one `app.py` is calling, and
that means the two files are different vintages.

A traceback that ends *inside* a `src/` module is an ordinary bug. A traceback
that ends *at a call into* a `src/` module is almost always this.

## Preventing it

`app.py` declares what it needs from `src/` in `REQUIRED_API` and checks it once
at startup, so a mismatched tree is refused with a message naming the files
rather than failing later with a traceback that names nothing.

Check before you push:

```bash
python3 scripts/check_build.py
```

Exit code 0 means the tree is consistent; 1 means it is not, and the output says
which file. It imports no Streamlit, needs no keys and touches no network, so it
works as a CI step or a pre-push hook.

## Two things that cause it

**Extracting an archive does not delete files.** A file removed in a new version
survives in your checkout. `packages.txt` above is exactly this: it kept breaking
builds after it had been removed upstream, because unzipping cannot delete. Use
`git rm <file>` for removals.

**`git status` before you push.** If you deploy from GitHub, Streamlit Cloud
serves whatever the branch contains — not what is in your working directory. New
or changed files under `src/` that were never staged will not be there, and the
app you are debugging is not the app you have.
