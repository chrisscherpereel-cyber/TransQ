"""Is the code actually running the code that was shipped?

Deploying this app is a file copy, and a copy that misses a file is silent:
Python imports the older module without complaint. The failure lands later, at
the moment of use, as a traceback that names a symptom and not a cause. It has
cost this project three debugging sessions — a `packages.txt` that survived an
unzip (archives cannot delete files), and twice a `src/` left behind by an
update that took `app.py`.

Two checks, because they catch different things and neither is enough alone.

**Signatures** — does the module provide the functions and keyword arguments
that `app.py` calls? Catches exactly the failures that would otherwise be a
`TypeError` mid-run, and needs no bookkeeping to stay true. Blind, though, to a
file whose body changed while its shape did not, which is most changes.

**Fingerprints** — does each file hash to what the packaged build recorded?
Catches everything, including a corrected prompt or a changed threshold, and
works on a host where you cannot run a script to find out. Costs one artefact,
``MANIFEST.sha256``, that has to be regenerated when you edit the code
deliberately — so drift here is reported as information, never as a refusal.

The rule that follows: a signature mismatch stops the app, because it *will*
crash. A fingerprint mismatch only tells you, because it might be you.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.sha256"


def digest(path: Path) -> str:
    """Hash of a file's contents, line endings normalised.

    A checkout on Windows rewrites newlines, which would otherwise read as every
    file having drifted — a check that cries wolf is one people learn to skip.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_manifest() -> dict[str, str]:
    """Recorded fingerprints, keyed by repo-relative path. Empty when absent."""
    if not MANIFEST.is_file():
        return {}
    recorded: dict[str, str] = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1].strip()] = parts[0]
    return recorded


@dataclass
class BuildStatus:
    """What differs between the files on disk and the build they came from."""

    changed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    checked: int = 0
    has_manifest: bool = True

    @property
    def is_clean(self) -> bool:
        return not (self.changed or self.missing)

    def summary(self) -> str:
        if not self.has_manifest:
            return "No MANIFEST.sha256 in this build — nothing to compare against."
        if self.is_clean:
            return f"All {self.checked} files match the packaged build."
        bits = []
        if self.changed:
            noun = "file differs" if len(self.changed) == 1 else "files differ"
            bits.append(f"{len(self.changed)} {noun} from the packaged build")
        if self.missing:
            bits.append(f"{len(self.missing)} missing")
        return " · ".join(bits)


def compare() -> BuildStatus:
    """Compare the files on disk against the fingerprints the build recorded."""
    recorded = read_manifest()
    if not recorded:
        return BuildStatus(has_manifest=False)

    status = BuildStatus(checked=len(recorded))
    for relative, expected in sorted(recorded.items()):
        path = ROOT / relative
        if not path.is_file():
            status.missing.append(relative)
        elif digest(path) != expected:
            status.changed.append(relative)

    on_disk = {
        p.relative_to(ROOT).as_posix()
        for p in [ROOT / "app.py", *(ROOT / "src").rglob("*.py")]
        if p.is_file() and "__pycache__" not in p.parts
    }
    status.extra = sorted(on_disk - set(recorded))
    return status
