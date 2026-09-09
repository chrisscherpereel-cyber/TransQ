#!/usr/bin/env python3
"""Verify that app.py and the modules in src/ are the same vintage.

Deploying this app is a file copy. A copy that misses one file is silent:
Python imports the older module without complaint, and the mismatch surfaces
much later as a bare ``TypeError`` inside a Streamlit traceback, at the moment
of use, with nothing pointing at the cause. The give-away in that traceback is
that it *stops at the call* — there is no frame inside the function being
called, because Python never got past binding the arguments.

Run this after copying in a new version and before redeploying:

    python3 scripts/check_build.py

It exits 0 when the tree is consistent and 1 when it is not, so it also works as
a CI step or a pre-push hook. It imports nothing from Streamlit and needs no
keys, no network and no configuration.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def declared_requirements() -> list[tuple[str, str, tuple[str, ...]]]:
    """Read REQUIRED_API out of app.py as text, so Streamlit is never imported."""
    tree = ast.parse((ROOT / "app.py").read_text())
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        if any(isinstance(t, ast.Name) and t.id == "REQUIRED_API" for t in targets):
            if node.value is not None:
                return ast.literal_eval(node.value)
    raise SystemExit(
        "app.py does not declare REQUIRED_API — it is older than this script.\n"
        "Replace app.py as well as src/."
    )


def problems() -> list[str]:
    found: list[str] = []
    for module_name, attr, kwargs in declared_requirements():
        path = module_name.replace(".", "/") + ".py"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # One absent file makes every module importing it unimportable. Name
            # the file that is missing, once — six lines with one cause reads as
            # six problems and sends people looking in the wrong places.
            missing_module = (exc.name or "").replace(".", "/")
            found.append(
                f"{missing_module}.py is missing entirely"
                if missing_module
                else f"{path} will not import: {exc}"
            )
            continue
        except Exception as exc:  # noqa: BLE001 - report it, do not crash on it
            found.append(f"{path} will not import: {exc}")
            continue

        target = getattr(module, attr, None)
        if target is None:
            found.append(f"{path} has no {attr}() — the file predates app.py")
            continue

        try:
            accepted = set(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            continue

        missing = [k for k in kwargs if k not in accepted]
        if missing:
            found.append(
                f"{path}: {attr}() does not accept "
                + ", ".join(missing)
                + " — the file predates app.py"
            )
    return list(dict.fromkeys(found))


ADVICE = (
    "\nFix: replace the whole src/ folder, not just app.py."
    "\n\nExtracting an archive over a checkout never removes or replaces what it"
    "\ndoes not contain, so old files survive an update. And if you deploy from"
    "\nGitHub, Streamlit Cloud serves THE BRANCH, not your working directory:"
    "\nrun `git status`, and anything under 'Changes not staged' or 'Untracked"
    "\nfiles' is not on the server."
)


def main() -> int:
    issues = problems()

    drift: list[str] = []
    try:
        from src.buildinfo import compare

        status = compare()
        # Anything the signature check already named is not news a second time.
        named = " ".join(issues)
        drift = [
            f"{p} differs from the packaged build"
            for p in status.changed
            if p not in named
        ]
        drift += [f"{p} is missing entirely" for p in status.missing if p not in named]
    except Exception:  # noqa: BLE001 - the fingerprint check is a bonus, not a gate
        status = None

    if not issues and not drift:
        if status is not None and status.has_manifest:
            print(f"✅ app.py and src/ agree, and all {status.checked} files match")
            print("   the packaged build. Safe to deploy.")
        else:
            print("✅ app.py and src/ agree. Safe to deploy.")
            print("   (No MANIFEST.sha256 — run scripts/make_manifest.py to add one.)")
        return 0

    if issues:
        print("❌ Some files in src/ are older than app.py.\n")
        for issue in issues:
            print(f"  · {issue}")
        print(
            "\nThe app would start normally and then fail partway through a run,"
            "\nwith a traceback that stops at the call and names no cause."
        )

    if drift:
        if issues:
            print("\nAlso out of date, for the same reason:\n")
        else:
            print("⚠️  Some files differ from the build they were packaged in.\n")
        for item in drift:
            print(f"  · {item}")
        if not issues:
            print(
                "\nNothing will crash — these are body changes, not signature"
                "\nchanges. Expected if you have been editing the code, in which"
                "\ncase run `python3 scripts/make_manifest.py` to re-record them."
                "\nIf you have not been editing, part of an update did not arrive."
            )

    if issues or status is None or not status.has_manifest:
        print(ADVICE)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
