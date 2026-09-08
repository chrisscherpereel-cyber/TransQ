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
    return found


def main() -> int:
    issues = problems()
    if not issues:
        print("✅ app.py and src/ agree. Safe to deploy.")
        return 0

    print("❌ Some files in src/ are older than app.py.\n")
    for issue in issues:
        print(f"  · {issue}")
    print(
        "\nThe app would start normally and then fail partway through a run,"
        "\nwith a traceback that stops at the call and names no cause."
        "\n\nFix: replace the whole src/ folder, not just app.py."
        "\nIf you deploy from GitHub, run `git status` — the src/ changes were"
        "\nprobably never committed, and Streamlit Cloud serves the branch."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
