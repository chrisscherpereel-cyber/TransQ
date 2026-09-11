#!/usr/bin/env python3
"""Record a fingerprint of every Python file the app runs.

The signature check in ``app.py`` catches a stale module only when its *shape*
changed — a new keyword argument, a new function. A file whose body changed but
whose signatures did not is invisible to it, and that is most changes: a fixed
prompt, a corrected threshold, a bug fix inside an existing function.

So the packaged build also carries ``MANIFEST.sha256``, listing a hash of
``app.py`` and every file under ``src/``. The app compares the files it is
actually running against that list at startup, which turns "I think I copied
everything" into something checkable — including on Streamlit Cloud, where you
cannot run a script to find out.

Regenerate after making your own changes, or the app will keep reporting drift
that you introduced deliberately:

    python3 scripts/make_manifest.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"


def tracked_files() -> list[Path]:
    """Everything whose contents change what the app does."""
    files = [ROOT / "app.py"]
    files += sorted(p for p in (ROOT / "src").rglob("*.py") if "__pycache__" not in p.parts)
    return [p for p in files if p.is_file()]


def digest(path: Path) -> str:
    # Normalise line endings so a Windows checkout does not read as drift.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def build() -> str:
    lines = [
        "# Fingerprints of the files this build runs. Regenerate with",
        "# python3 scripts/make_manifest.py after changing any of them.",
    ]
    lines += [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in tracked_files()
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = build()
    MANIFEST.write_text(text)
    print(f"Wrote {MANIFEST.relative_to(ROOT)} — {len(text.splitlines()) - 2} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
