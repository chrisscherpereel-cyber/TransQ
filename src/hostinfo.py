"""How much machine is actually underneath us.

Written after a three-part lecture died partway through part three on Streamlit
Community Cloud. The container has about 1 GB of RAM; the failure mode when a
Whisper model does not fit is not an exception but a SIGKILL, so nothing is
raised, nothing is caught, and the user sees the progress bar stop at 80% with
no explanation.

A guess made *before* loading the weights is worth far more than a diagnosis
made after the process is dead. So: work out the memory ceiling, compare it to
what the chosen model needs, and refuse up front with a message naming the
largest model that will actually run.

Two things this deliberately does not do. It does not import psutil — a
dependency for one number that ``/sys`` already has. And it does not trust
``os.sysconf`` alone: inside a container that reports the *host's* physical
memory, which on a shared cloud host can be a hundred times the real limit.
cgroup limits come first for exactly that reason.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

GB = 1024 ** 3

# Peak resident memory per model, in GB: int8 CTranslate2 weights plus the
# decoder's working set and audio buffers. Roughly weights × 1.4.
#
# These are calibrated so that `small` on a 1 GB host reads as *tight* rather
# than impossible, because that combination demonstrably does work much of the
# time. Getting this wrong in either direction is costly: too generous and the
# guard fails to prevent the crash it exists for; too strict and it blocks a
# configuration people run successfully every day.
MODEL_MEMORY_GB: dict[str, float] = {
    "tiny": 0.25,
    "base": 0.35,
    "small": 0.70,
    "medium": 1.90,
    "large-v3": 3.60,
}

# Non-int8 compute types keep wider weights in memory.
COMPUTE_MULTIPLIER: dict[str, float] = {
    "int8": 1.0,
    "int8_float16": 1.15,
    "float16": 1.7,
    "float32": 2.8,
}

# Streamlit, Python, the model catalog and the encrypted store all want memory
# before Whisper asks for any. Reserve it rather than pretending it is free.
OVERHEAD_GB = 0.35

CGROUP_V2 = "/sys/fs/cgroup/memory.max"
CGROUP_V1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

# A cgroup with no limit reports a sentinel — either the literal string "max" or
# an absurd number. Anything above this is "unlimited", not a real ceiling.
UNLIMITED_ABOVE_GB = 1024.0


def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def cgroup_limit_gb() -> float | None:
    """The container's memory ceiling, if it has one."""
    for path in (CGROUP_V2, CGROUP_V1):
        value = _read_int(path)
        if value and value > 0:
            gb = value / GB
            if gb < UNLIMITED_ABOVE_GB:
                return gb
    return None


def physical_memory_gb() -> float | None:
    """What the kernel reports, which inside a container is the host's total."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return (pages * page_size) / GB


def available_memory_gb() -> float | None:
    """The real ceiling: the tighter of the cgroup limit and physical memory.

    ``None`` means we could not tell, which is treated as permission rather than
    prohibition — refusing to run because a number was unreadable would be worse
    than the problem being solved.
    """
    candidates = [v for v in (cgroup_limit_gb(), physical_memory_gb()) if v]
    return min(candidates) if candidates else None


def model_memory_gb(model_size: str, compute_type: str = "int8") -> float:
    base = MODEL_MEMORY_GB.get(model_size, MODEL_MEMORY_GB["small"])
    return base * COMPUTE_MULTIPLIER.get(compute_type, 1.0)


def largest_model_that_fits(
    budget_gb: float, compute_type: str = "int8"
) -> str | None:
    """The biggest model this host can hold, or ``None`` if even ``tiny`` cannot."""
    usable = budget_gb - OVERHEAD_GB
    fitting = [
        name
        for name in MODEL_MEMORY_GB
        if model_memory_gb(name, compute_type) <= usable
    ]
    return fitting[-1] if fitting else None


@dataclass
class MemoryVerdict:
    """Whether a model will run here, and what to say about it.

    Three outcomes rather than two, because the interesting case is in the
    middle. ``small`` on a 1 GB container fits with almost nothing to spare: it
    usually works, sometimes dies on a long file, and blocking it outright would
    take away a configuration people rely on. Refusing is for models that cannot
    fit at all; a tight fit gets a warning and a suggestion, not a veto.
    """

    level: str  # "ok" | "tight" | "refused"
    message: str = ""
    budget_gb: float | None = None
    needed_gb: float = 0.0
    suggestion: str = ""

    @property
    def allowed(self) -> bool:
        return self.level != "refused"


def check_model_fits(model_size: str, compute_type: str = "int8") -> MemoryVerdict:
    """Decide before the weights load, because afterwards there is no decision.

    An unreadable memory limit is treated as permission, not prohibition: this
    guards against a known failure, and it must never itself be the reason a
    working deployment stops working.
    """
    budget = available_memory_gb()
    if budget is None:
        return MemoryVerdict("ok")

    needed = model_memory_gb(model_size, compute_type)
    best = largest_model_that_fits(budget, compute_type)
    smaller = best if best and best != model_size else ""

    if needed >= budget:
        detail = (
            f"The **{model_size}** model needs about {needed:.1f} GB and this "
            f"server has {budget:.1f} GB in total, so it cannot load."
        )
        advice = (
            f" Use **{smaller}** here"
            if smaller
            else " No Whisper model fits on this server"
        )
        return MemoryVerdict(
            "refused",
            f"{detail}{advice}, run the app on a machine with more memory, or "
            f"import an existing transcript instead of transcribing. "
            f"Refusing now rather than letting it be killed partway through: the "
            f"process is terminated outright, so there is no error to show you.",
            budget, needed, smaller,
        )

    if needed > budget - OVERHEAD_GB:
        return MemoryVerdict(
            "tight",
            f"**{model_size}** needs about {needed:.1f} GB of this server's "
            f"{budget:.1f} GB, leaving very little for the app itself. It will "
            f"probably work, but a long recording may be killed partway through "
            f"with no error message."
            + (f" **{smaller}** would be a safer choice here." if smaller else ""),
            budget, needed, smaller,
        )

    return MemoryVerdict("ok", "", budget, needed, "")


# Community Cloud checks the repo out here. It is the most reliable signal
# available, and it is what the platform's own docs describe.
STREAMLIT_CLOUD_MARKER = "/mount/src"

STREAMLIT_CLOUD_ENV_VARS = ("STREAMLIT_SHARING_MODE", "STREAMLIT_RUNTIME_ENV")


def is_ephemeral_host() -> bool:
    """Is this a host whose disk disappears on restart?

    Used to decide how loudly to complain about local-file storage. On a laptop,
    local files are exactly right and a standing warning is noise that teaches
    people to ignore warnings. On Community Cloud the same configuration quietly
    destroys a semester of work.

    Deliberately conservative: only positive evidence of an ephemeral host counts.
    Being wrong in the quiet direction costs a warning; being wrong in the loud
    direction costs the credibility of every warning the app shows.
    """
    if os.path.isdir(STREAMLIT_CLOUD_MARKER):
        return True
    return any(os.environ.get(name) for name in STREAMLIT_CLOUD_ENV_VARS)


def describe_host() -> str:
    """A one-line summary for the sidebar."""
    budget = available_memory_gb()
    if budget is None:
        return "Memory limit unknown"
    best = largest_model_that_fits(budget)
    return f"{budget:.1f} GB RAM · largest usable Whisper model: {best or 'none'}"
