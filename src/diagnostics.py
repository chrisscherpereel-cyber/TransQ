"""A record of what actually happened during a run.

The bug this exists to fix: the pipeline used to catch every per-chunk failure,
write a note into a progress bar that was cleared moments later, and carry on.
A run where the API rejected the key, ran out of credit, or truncated every
reply looked identical to a run where the lecture simply had little to say —
"no questions could be generated", and nothing about why.

So every call now records its outcome here, successes included, and the UI shows
the result. The guiding rule: **a step that fails must leave evidence that
outlives the progress bar.**
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Phases, so the report can be read in pipeline order.
PHASE_SUMMARY = "Summarize"
PHASE_GENERATE = "Generate"
PHASE_REVIEW = "Review"
PHASE_REPLACE = "Replace"
# Named here rather than imported, to keep diagnostics free of a factcheck import.
PHASE_REVIEW_CLAIMS_NAME = "Consistency review"

OK = "ok"
FAILED = "failed"
EMPTY = "empty"      # the call worked but produced nothing usable
SKIPPED = "skipped"


@dataclass
class StepResult:
    phase: str
    label: str            # which chunk, or which pass
    status: str
    detail: str = ""      # human-readable cause
    cause: str = ""       # short machine-ish classification
    produced: int = 0     # questions / sections returned

    @property
    def is_failure(self) -> bool:
        return self.status == FAILED


@dataclass
class RunReport:
    """Everything that happened, in order."""

    steps: list[StepResult] = field(default_factory=list)

    def record(
        self,
        phase: str,
        label: str,
        status: str,
        detail: str = "",
        cause: str = "",
        produced: int = 0,
    ) -> StepResult:
        step = StepResult(phase, label, status, detail, cause, produced)
        self.steps.append(step)
        return step

    # -- queries -- #

    @property
    def failures(self) -> list[StepResult]:
        return [s for s in self.steps if s.is_failure]

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.steps if s.status == OK)

    def phase_steps(self, phase: str) -> list[StepResult]:
        return [s for s in self.steps if s.phase == phase]

    def phase_failed_entirely(self, phase: str) -> bool:
        """The phase produced nothing at all — not just a bad patch of audio.

        Judged on output, not on status. A chunk whose reply was cut off logs a
        retry *and* may still yield questions salvaged from the truncated text;
        counting that as a total failure would abandon work the model actually
        did. Equally, a chunk that logged a retry and then failed produced
        nothing and must still read as failed — which "no step was skipped"
        got wrong in the other direction.
        """
        steps = self.phase_steps(phase)
        if not steps or any(s.produced for s in steps):
            return False
        return any(s.is_failure for s in steps)

    def causes(self) -> Counter:
        return Counter(s.cause for s in self.failures if s.cause)

    def dominant_cause(self) -> str:
        counts = self.causes()
        return counts.most_common(1)[0][0] if counts else ""

    # -- presentation -- #

    def headline(self) -> str:
        failures = len(self.failures)
        if not self.steps:
            return "Nothing ran."
        if not failures:
            return f"All {len(self.steps)} model calls succeeded."
        return f"{failures} of {len(self.steps)} model calls failed."

    def advice(self) -> str:
        """What to actually do about the most common failure in this run.

        Phase-aware, because the same cause has different fixes in different
        places. A truncated reply during *summarization* has nothing to do with
        the question count, and telling someone to "ask for fewer questions"
        while summarizing sends them to a setting that cannot help — which is
        worse than saying nothing, because they will try it.
        """
        cause = self.dominant_cause()
        phase = self._dominant_failing_phase()
        override = PHASE_ADVICE.get((phase, cause))
        return override or ADVICE.get(cause, "")

    def _dominant_failing_phase(self) -> str:
        phases = Counter(s.phase for s in self.failures if s.phase)
        return phases.most_common(1)[0][0] if phases else ""

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "Step": s.phase,
                "Part": s.label,
                "Result": {OK: "✓", FAILED: "✗", EMPTY: "—", SKIPPED: "skipped"}.get(
                    s.status, s.status
                ),
                "Produced": s.produced,
                "Detail": s.detail,
            }
            for s in self.steps
        ]


# --------------------------------------------------------------------------- #
# Cause classification
# --------------------------------------------------------------------------- #

CAUSE_TRUNCATED = "truncated"
CAUSE_RATE_LIMIT = "rate_limit"
CAUSE_CREDIT = "credit"
CAUSE_AUTH = "auth"
CAUSE_BAD_JSON = "bad_json"
CAUSE_MODEL = "model_unavailable"
CAUSE_NETWORK = "network"
CAUSE_TIMEOUT = "timeout"
CAUSE_OTHER = "other"

ADVICE = {
    CAUSE_TRUNCATED: (
        "The model's reply was cut off before it finished. Whatever it completed "
        "before the cut has been kept — check the steps above for how many — so "
        "this is a shortfall rather than a loss. To stop it recurring: ask for "
        "fewer questions per run, shorten the chunk length in **Context & "
        "advanced**, or pick a model with a larger output limit. Reasoning models "
        "are especially prone to it because their thinking counts against the "
        "output budget, so switching to a non-reasoning model often fixes it "
        "outright."
    ),
    CAUSE_RATE_LIMIT: (
        "The provider is rate-limiting this key. Free models on OpenRouter have "
        "tight limits — wait a minute and retry, or switch to a paid model."
    ),
    CAUSE_CREDIT: (
        "The key is out of credit, or has hit its cap. Top up at your provider, "
        "or raise this account's cap under **Admin → Issued API keys**."
    ),
    CAUSE_AUTH: (
        "The provider rejected the API key. Check it under the sidebar, and make "
        "sure it is an inference key for this provider — not a management key."
    ),
    CAUSE_BAD_JSON: (
        "The model returned something that was not valid JSON. Some models ignore "
        "the format instruction; try a different one — the DeepSeek and Gemini "
        "defaults are reliable here."
    ),
    CAUSE_MODEL: (
        "That model is not available on this key. Pick another from the model "
        "list; the list is fetched live, but a model can be retired between "
        "refreshes."
    ),
    CAUSE_NETWORK: "The provider could not be reached. Check connectivity and retry.",
    CAUSE_TIMEOUT: (
        "The request timed out. Long chunks on a slow model are the usual cause — "
        "shorten the chunk length in **Context & advanced**."
    ),
}


# Advice that depends on *where* the failure happened, not only on what it was.
PHASE_ADVICE: dict[tuple[str, str], str] = {
    (PHASE_SUMMARY, CAUSE_TRUNCATED): (
        "The model's reply was cut off while summarizing. Every window that "
        "finished has been kept, so pressing **Summarize & generate** again "
        "resumes from where it stopped rather than starting over — it will not "
        "re-pay for the windows already done. The question count is irrelevant "
        "here; if it keeps happening, shorten the chunk length in **Context & "
        "advanced** or switch to a non-reasoning model, whose thinking does not "
        "count against the output budget."
    ),
    (PHASE_REVIEW_CLAIMS_NAME, CAUSE_TRUNCATED): (
        "The consistency review was cut off. Windows that finished are kept. "
        "A shorter chunk length or a model with a larger output limit will get "
        "through the rest."
    ),
}


def classify(exc: BaseException | str) -> str:
    """Reduce a provider error to a cause the UI can give advice about."""
    text = str(exc).lower()
    name = type(exc).__name__.lower() if isinstance(exc, BaseException) else ""

    if "truncat" in text or "cut off" in text or "max_tokens" in text or "length" == text:
        return CAUSE_TRUNCATED
    if "truncated" in name:
        return CAUSE_TRUNCATED
    if any(w in text for w in ("insufficient", "credit", "quota", "billing", "402", "payment")):
        return CAUSE_CREDIT
    if any(w in text for w in ("rate limit", "ratelimit", "429", "too many requests")):
        return CAUSE_RATE_LIMIT
    if any(w in text for w in ("unauthor", "forbidden", "invalid api key", "401", "403",
                               "no auth", "api key")):
        return CAUSE_AUTH
    if any(w in text for w in ("not a valid model", "no endpoints", "model not found", "404")):
        return CAUSE_MODEL
    if any(w in text for w in ("json", "expecting value", "unterminated")):
        return CAUSE_BAD_JSON
    if any(w in text for w in ("timed out", "timeout")):
        return CAUSE_TIMEOUT
    if any(w in text for w in ("connection", "unreachable", "network", "dns", "ssl")):
        return CAUSE_NETWORK
    return CAUSE_OTHER


def short_reason(exc: BaseException) -> str:
    """One line an instructor can read, not a stack trace."""
    text = " ".join(str(exc).split())
    return text[:200] + ("…" if len(text) > 200 else "")
