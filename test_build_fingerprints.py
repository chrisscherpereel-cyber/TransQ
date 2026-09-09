"""Catching a stale file whose *shape* did not change.

The signature guard in `app.py` catches a module that lost a keyword argument,
because that is what crashes. It is blind to the more common case: a file whose
body changed while its functions kept their arguments — a corrected prompt, a
changed threshold, a fixed off-by-one. Such a file runs happily and does the
wrong thing, and on a hosted deployment there is no shell to go and check.

So the packaged build carries `MANIFEST.sha256` and the app compares itself to
it at startup. The design constraint these tests hold in place is the one that
decides whether anybody keeps the feature switched on:

**A signature mismatch stops the app; a fingerprint mismatch only reports.**

The first will certainly crash. The second might just be you, editing your own
code — and a check that cries wolf during ordinary work is one people learn to
click past, which costs more than it ever saved.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from src import buildinfo

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# The manifest itself
# --------------------------------------------------------------------------- #


def test_the_shipped_manifest_matches_the_shipped_code():
    """If this fails, the build was packaged without regenerating fingerprints,
    and every user would see drift that is not theirs."""
    status = buildinfo.compare()
    assert status.has_manifest, "the build must ship a MANIFEST.sha256"
    assert status.changed == [], f"stale fingerprints for {status.changed}"
    assert status.missing == [], f"manifest names files that are gone: {status.missing}"


def test_every_python_file_the_app_runs_is_fingerprinted():
    """A file nobody records is a file nobody can check."""
    status = buildinfo.compare()
    assert status.extra == [], f"not covered by the manifest: {status.extra}"


def test_the_manifest_covers_app_and_src_and_nothing_else():
    recorded = buildinfo.read_manifest()
    assert "app.py" in recorded
    assert any(p.startswith("src/") for p in recorded)
    assert all(
        p == "app.py" or p.startswith("src/") for p in recorded
    ), "tests and scripts change constantly; fingerprinting them would only nag"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature repo: two files and a manifest that describes them."""
    (tmp_path / "src").mkdir()
    (tmp_path / "app.py").write_text("print('app')\n")
    (tmp_path / "src" / "thing.py").write_text("VALUE = 1\n")

    lines = ["# fingerprints"]
    for rel in ("app.py", "src/thing.py"):
        data = (tmp_path / rel).read_bytes()
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    return tmp_path


def compare_in(tree: Path) -> buildinfo.BuildStatus:
    original_root, original_manifest = buildinfo.ROOT, buildinfo.MANIFEST
    buildinfo.ROOT = tree
    buildinfo.MANIFEST = tree / "MANIFEST.sha256"
    try:
        return buildinfo.compare()
    finally:
        buildinfo.ROOT, buildinfo.MANIFEST = original_root, original_manifest


def test_a_body_only_change_is_detected(tree):
    """The case the signature check cannot see at all."""
    (tree / "src" / "thing.py").write_text("VALUE = 2\n")
    status = compare_in(tree)
    assert status.changed == ["src/thing.py"]
    assert not status.is_clean


def test_a_missing_file_is_reported_as_missing_not_as_changed(tree):
    (tree / "src" / "thing.py").unlink()
    status = compare_in(tree)
    assert status.missing == ["src/thing.py"] and status.changed == []


def test_an_untracked_new_file_is_noticed(tree):
    (tree / "src" / "extra.py").write_text("NEW = 1\n")
    status = compare_in(tree)
    assert status.extra == ["src/extra.py"]
    assert status.is_clean, "a new file breaks nothing; it is not a failure"


def test_windows_line_endings_are_not_drift(tree):
    """A checkout that rewrites newlines would otherwise report every file as
    changed — and a check that is wrong the first time is never trusted again."""
    (tree / "src" / "thing.py").write_bytes(b"VALUE = 1\r\n")
    assert compare_in(tree).is_clean


def test_a_build_with_no_manifest_says_so_rather_than_failing(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "app.py").write_text("x = 1\n")
    status = compare_in(tmp_path)
    assert not status.has_manifest
    assert "No MANIFEST" in status.summary()


def test_comments_and_blank_lines_in_a_manifest_are_ignored(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "app.py").write_text("x = 1\n")
    digest = hashlib.sha256(b"x = 1\n").hexdigest()
    (tmp_path / "MANIFEST.sha256").write_text(
        f"# a comment\n\n{digest}  app.py\n\n# trailing note\n"
    )
    status = compare_in(tmp_path)
    assert status.checked == 1 and status.is_clean


# --------------------------------------------------------------------------- #
# What each check is allowed to do about it
# --------------------------------------------------------------------------- #


def test_drift_reports_but_never_blocks():
    """The rule that keeps the feature switched on. A fingerprint mismatch during
    your own development is normal; refusing to start would be intolerable."""
    source = (ROOT / "app.py").read_text()
    start = source.index("def version_check")
    body = source[start : source.index("\ndef ", start + 10)]

    assert "return False" in body, "a signature mismatch must stop the app"
    after_drift = body[body.index("drift = build_drift()") :]
    assert "return False" not in after_drift, "drift alone must never stop the app"
    assert "return True" in after_drift


def test_the_scripts_exist_and_are_runnable():
    for name in ("check_build.py", "make_manifest.py"):
        assert (ROOT / "scripts" / name).is_file()


def test_check_build_passes_on_this_tree():
    """The end-to-end contract: a correctly assembled tree exits zero."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_build.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Safe to deploy" in result.stdout
