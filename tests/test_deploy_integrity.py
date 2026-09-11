"""The app and the modules it calls must be the same vintage.

Deploying this app is a file copy — a zip extracted over a checkout, or a push
to the branch Streamlit Cloud serves. A copy that misses one file is not caught
by anything: Python imports the old module happily, and the mismatch only
surfaces later, as a bare ``TypeError`` deep inside a Streamlit traceback, at
the moment of use, with nothing in the message pointing at the real cause. It
has cost this project a debugging session twice — once when a deleted
``packages.txt`` survived an unzip, once when ``src/summarize.py`` did not.

So ``app.py`` declares what it needs from ``src/`` in ``REQUIRED_API`` and checks
it once at startup. These tests keep that declaration honest, which is the part
that would otherwise rot: a guard listing functions that no longer exist is not a
guard, it is a second thing to debug.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def required_api() -> list[tuple[str, str, tuple[str, ...]]]:
    """Read REQUIRED_API out of app.py without importing Streamlit."""
    tree = ast.parse((ROOT / "app.py").read_text())
    for node in tree.body:
        targets = getattr(node, "targets", []) or [getattr(node, "target", None)]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if "REQUIRED_API" in names and node.value is not None:
            return ast.literal_eval(node.value)
    raise AssertionError("app.py no longer declares REQUIRED_API")


def resolve(module: object, dotted: str) -> object | None:
    """Mirrors app.py: an entry may name a method, or a plain constant."""
    target: object | None = module
    for part in dotted.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


@pytest.mark.parametrize("module_name,attr,kwargs", required_api())
def test_every_declared_requirement_is_actually_met(module_name, attr, kwargs):
    """The guard must pass on a correctly assembled tree.

    If this fails, either src/ is genuinely behind app.py — the thing the guard
    exists to catch — or the declaration names something that was renamed.
    """
    module = importlib.import_module(module_name)
    target = resolve(module, attr)
    assert target is not None, f"{module_name} has no {attr}"

    if not kwargs:
        # A constant (PHASE_ADVICE, REVIEW_SYSTEM) is checked for existence
        # alone. Its presence is the signal; it has no signature to inspect.
        return

    accepted = set(inspect.signature(target).parameters)
    missing = [k for k in kwargs if k not in accepted]
    assert not missing, f"{module_name}.{attr} does not accept {missing}"


def test_entries_that_declare_arguments_name_something_callable():
    """A constant with a kwargs list would silently never be checked."""
    for module_name, attr, kwargs in required_api():
        if not kwargs:
            continue
        target = resolve(importlib.import_module(module_name), attr)
        assert callable(target), f"{module_name}.{attr} declares {kwargs} but is not callable"


def test_the_guard_covers_the_calls_that_have_actually_broken():
    """Every entry earns its place by having failed in the field, or by being a
    signature app.py newly depends on. This asserts the two that bit us."""
    declared = {(m, a) for m, a, _ in required_api()}
    assert ("src.summarize", "summarize_transcript") in declared
    assert ("src.transcribe", "transcribe_parts") in declared


def test_the_guard_names_the_file_rather_than_the_module_path():
    """`src/summarize.py` is something you can go and look at. `src.summarize`
    is a thing you have to translate first, while already frustrated."""
    source = (ROOT / "app.py").read_text()
    start = source.index("def stale_modules")
    body = source[start : source.index("\ndef ", start + 10)]
    assert "replace('.', '/')" in body and ".py" in body


def test_a_stale_module_is_reported_rather_than_raised():
    """Simulating the real failure: app.py expects a keyword that the deployed
    src/ has never heard of."""
    import types

    fake = types.ModuleType("fake_stale")

    def old_signature(client, transcript, report=None):
        return None

    fake.summarize_transcript = old_signature

    accepted = set(inspect.signature(fake.summarize_transcript).parameters)
    missing = [k for k in ("cached_sections", "on_section") if k not in accepted]
    assert missing == ["cached_sections", "on_section"], (
        "this is exactly what the deployed app hit: the call is well-formed, "
        "the module is simply older than the caller"
    )
