"""Talking to a model running on the same machine as the app.

Ollama and LM Studio both expose OpenAI's ``/chat/completions`` contract on
localhost, so the existing ``openai`` code path in :mod:`src.llm` already speaks
their language. What they do *not* share with a hosted vendor is everything
around the request: there is no API key, the port varies, the model list is
whatever happens to be downloaded, and the server is frequently not running at
all. This module covers that gap.

**Why discovery instead of a hardcoded list.** Local model names change monthly
and every machine has a different set downloaded. A curated dropdown would be
wrong on most machines within a month and would offer models the user does not
have. So the list comes from the server itself — what is offered is what is
actually installed and ready to run.

**The constraint the UI must state plainly.** A model on your laptop is not
reachable from Streamlit Community Cloud. ``localhost`` there means Streamlit's
own container, not your computer. This is not a setting that can be fixed; it is
what "local" means. Using a local model requires running the app on the same
machine as the model, so the app has to say so rather than let someone spend an
afternoon on a connection that cannot exist.

No new dependencies: ``urllib`` from the standard library is enough for a GET
against localhost, and adding ``requests`` for one call would be a poor trade.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Where the two common runners listen by default. Ollama first: it is the one
# that runs as a background service, so it is the one most likely to answer
# without the user having opened anything.
KNOWN_SERVERS: tuple[tuple[str, str], ...] = (
    ("Ollama", "http://localhost:11434/v1"),
    ("LM Studio", "http://localhost:1234/v1"),
)

DEFAULT_BASE_URL = KNOWN_SERVERS[0][1]

# Long enough to survive a busy machine, short enough that a closed port does
# not stall the sidebar. A refused connection returns immediately regardless.
PROBE_TIMEOUT = 2.5


@dataclass
class LocalServer:
    """The result of asking a local address whether anything is home."""

    base_url: str
    reachable: bool = False
    models: list[str] = field(default_factory=list)
    label: str = ""
    detail: str = ""

    @property
    def is_usable(self) -> bool:
        return self.reachable and bool(self.models)

    def status_line(self) -> str:
        if not self.reachable:
            return self.detail or "No local model server responded."
        if not self.models:
            return (
                f"{self.label or 'A server'} is running, but no models are "
                "downloaded yet."
            )
        noun = "model" if len(self.models) == 1 else "models"
        return f"{self.label or 'Connected'} · {len(self.models)} {noun} ready"


def normalise(base_url: str) -> str:
    """Accept what people actually paste.

    A bare host, a URL with a trailing slash, or one that already ends in
    ``/v1`` all mean the same thing, and asking someone to get it exactly right
    is a support ticket waiting to happen.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_BASE_URL
    if "://" not in url:
        url = "http://" + url
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _name_for(base_url: str) -> str:
    for label, known in KNOWN_SERVERS:
        if normalise(known) == normalise(base_url):
            return label
    return "Local server"


def probe(base_url: str, timeout: float = PROBE_TIMEOUT) -> LocalServer:
    """Ask one address what models it has.

    Never raises. A server that is not running is the normal case here, not an
    exceptional one — the whole point of this call is to find out.
    """
    url = normalise(base_url)
    server = LocalServer(base_url=url, label=_name_for(url))

    try:
        request = urllib.request.Request(
            f"{url}/models", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        server.detail = (
            f"Something is listening at {url}, but it answered {exc.code} rather "
            "than a model list. Check the address and the port."
        )
        return server
    except urllib.error.URLError as exc:
        server.detail = _explain_unreachable(url, exc)
        return server
    except Exception as exc:  # noqa: BLE001 - a probe must never break the page
        server.detail = f"Could not read a model list from {url}: {exc}"
        return server

    server.reachable = True
    server.models = _extract_models(payload)
    return server


def _explain_unreachable(url: str, exc: Exception) -> str:
    """Say which of the two likely causes it is, rather than printing errno."""
    reason = str(getattr(exc, "reason", exc)).lower()
    port = url.split(":")[-1].split("/")[0]
    if "refused" in reason:
        return (
            f"Nothing is listening on port {port}. Start Ollama or LM Studio, "
            "then check again."
        )
    if "timed out" in reason or "timeout" in reason:
        return (
            f"{url} accepted the connection but did not answer in time. The "
            "server may still be loading a model."
        )
    return f"Could not reach {url}: {getattr(exc, 'reason', exc)}"


def _extract_models(payload: Any) -> list[str]:
    """Read the OpenAI ``/models`` shape, tolerating the variations in the wild.

    Ollama and LM Studio both return ``{"data": [{"id": ...}]}``. Ollama's own
    native endpoint uses ``{"models": [{"name": ...}]}``, and a user who pastes
    that address instead should still get a working list rather than a blank
    dropdown and no explanation.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("models")
    if not isinstance(rows, list):
        return []

    names: list[str] = []
    for row in rows:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = str(row.get("id") or row.get("name") or "")
        else:
            continue
        name = name.strip()
        if name and name not in names:
            names.append(name)
    return sorted(names, key=str.lower)


def discover(timeout: float = PROBE_TIMEOUT) -> LocalServer:
    """Try the well-known ports and return the first server with models.

    Preferring a server that *has* models over one that merely answers matters:
    someone with LM Studio loaded and Ollama freshly installed should get the
    one they can actually use, not the one that happens to sort first.
    """
    answered: LocalServer | None = None
    for _, url in KNOWN_SERVERS:
        server = probe(url, timeout=timeout)
        if server.is_usable:
            return server
        if server.reachable and answered is None:
            answered = server
    if answered is not None:
        return answered
    return LocalServer(
        base_url=DEFAULT_BASE_URL,
        detail=(
            "No local model server answered on the usual ports (11434 for "
            "Ollama, 1234 for LM Studio). Install one and start it, then check "
            "again — see docs/LOCAL_MODELS.md."
        ),
    )


# --------------------------------------------------------------------------- #
# What will actually run on this machine
# --------------------------------------------------------------------------- #

# Memory each size class wants, at the 4-bit quantisation these runners download
# by default, once the context window this app uses is accounted for.
#
# Deliberately conservative, and deliberately size classes rather than model
# names. Conservative because this app is not the only thing running: when the
# model is local, Whisper is loaded on the same machine, and a model that
# technically fits but forces swapping is slower than the next size down while
# looking, to the user, like the app has hung. Size classes because local model
# names turn over monthly — a list frozen here would name models the user has
# not downloaded and miss the ones they have.
SIZE_CLASSES: tuple[tuple[str, float, str], ...] = (
    ("~3B", 3.0, "Fast, and weak at the JSON structure this app needs. A fallback, not a choice."),
    ("~8B", 6.0, "The usable floor for question writing. Good enough to draft, still worth reviewing closely."),
    ("~14B", 14.0, "Noticeably better at following the item-writing rules. A sensible target."),
    ("~30B", 22.0, "Comparable to a mid-tier hosted model for this task. Slower per lecture."),
    ("~70B", 42.0, "Best quality that runs locally at all, and slow enough that you will feel it."),
)

# Left for the operating system, the browser, and Whisper — which is loaded on
# this same machine whenever the model is.
RESERVED_GB = 3.0


def largest_class_for(memory_gb: float) -> str:
    """The biggest size class this machine can comfortably hold, or "" if none."""
    budget = max(0.0, memory_gb - RESERVED_GB)
    best = ""
    for name, needs, _ in SIZE_CLASSES:
        if needs <= budget:
            best = name
    return best


def guidance(memory_gb: float) -> str:
    """One sentence pointing at a size class, given the memory actually present."""
    if memory_gb <= 0:
        return "Could not read this machine's memory, so pick a size by trial."
    best = largest_class_for(memory_gb)
    if not best:
        return (
            f"{memory_gb:.0f} GB is below what any usable model needs alongside "
            "the app. A hosted model is the better option on this machine."
        )
    detail = next(d for name, _, d in SIZE_CLASSES if name == best)
    return f"{memory_gb:.0f} GB of memory — up to a {best} model. {detail}"
