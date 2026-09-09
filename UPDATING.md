# Updating a deployed copy

Every deployment failure this project has had was a file that did not arrive.
Not a bug, not a platform problem — a copy that missed something. The app now
refuses to start when that happens and names the files, but it is easier to
avoid than to diagnose.

**The rule: replace the whole folder, never individual files.** `app.py` and
`src/` change together. Updating one without the other produces an app that
starts, signs you in, transcribes, and then fails partway through a run.

---

## Route A — GitHub in the browser

No git installed, nothing to configure. Use this one if you are not sure.

1. Extract the zip. You will get a folder containing `app.py`, `src/`,
   `scripts/`, `docs/`, `requirements.txt` and `MANIFEST.sha256`.
2. In your repository on github.com: **Add file → Upload files**.
3. Drag in the **folders**, not the files. Select `app.py`, `src`, `scripts`,
   `docs`, `requirements.txt` and `MANIFEST.sha256` together and drop them on
   the page in one go. GitHub keeps the folder structure and replaces anything
   with a matching path.
4. Write a commit message and **Commit changes**.
5. Streamlit Cloud redeploys within a minute or so.

**The trap:** dragging `app.py` on its own. It is the file whose name you know,
it is at the top of the folder, and it is the one thing that must never travel
alone. If you take nothing else from this page: `src/` goes with it, every time.

**The other trap:** uploading cannot *delete*. If a file was removed in the new
version, it survives in your repository until you delete it by hand — open it on
GitHub and use the ⋯ menu → **Delete file**. This is how `packages.txt` kept
breaking builds after it was removed (see `DEPLOY_NOTES.md`).

---

## Route B — git on your own machine

Faster once set up, and it shows you exactly what is about to change.

```bash
cd /path/to/your/repo

# Replace the two folders wholesale rather than merging into them, so a file
# deleted upstream actually disappears here too.
rm -rf src scripts docs
cp -R /path/to/extracted/{app.py,src,scripts,docs,requirements.txt,MANIFEST.sha256} .

python3 scripts/check_build.py     # must print "Safe to deploy"

git add -A
git status                         # read this. It is the last honest look.
git commit -m "Update to latest build"
git push
```

`git status` is the step people skip and the one that would have caught both
incidents. Anything listed under *Changes not staged for commit* or *Untracked
files* is **not** going to the server — Streamlit Cloud serves the branch, not
your working directory.

---

## Checking before you push

```bash
python3 scripts/check_build.py
```

Exit code 0 means the tree is consistent. 1 means it is not, and the output
names the files. It imports no Streamlit, needs no keys and touches no network,
so it also works as a CI step or a pre-push hook:

```bash
printf '#!/bin/sh\npython3 scripts/check_build.py\n' > .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Two checks run, and they catch different things:

| Check | Catches | On failure |
|---|---|---|
| **Signatures** — does `src/` provide the functions and arguments `app.py` calls? | A module older than the app. This is what crashes mid-run | The app refuses to start |
| **Fingerprints** — does each file hash to what `MANIFEST.sha256` recorded? | Everything else: a changed prompt, a corrected threshold, a file that never arrived | Reported, never blocked |

Drift is reported rather than blocked because during your own development it is
simply the work you are doing. A check that cries wolf while you edit is one you
learn to click past, which costs more than it ever saved.

**If you edit the code yourself**, re-record the fingerprints so the app stops
mentioning it:

```bash
python3 scripts/make_manifest.py
```

---

## Reading a failure that got through

If a traceback ends **at a call into** a `src/` module — with no frame *inside*
the function being called — that is not an application bug. Python failed while
binding the arguments, meaning the function on disk has a different signature
from the one calling it. The two files are different vintages.

A traceback that ends *inside* a `src/` module is an ordinary bug. A traceback
that stops at the doorway is a half-copied deploy.
