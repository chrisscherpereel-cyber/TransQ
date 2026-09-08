"""The live OpenRouter model catalog.

OpenRouter carries several hundred models and the roster changes every week —
new releases land, old ones are retired, prices move, and free variants come and
go. A list hardcoded in this file would be wrong within a month and would go on
being confidently wrong, offering models that no longer exist and hiding ones
that do.

So the catalog is fetched at runtime from ``GET /api/v1/models`` (public, no key
required), cached for an hour, and sorted alphabetically. A small bundled
snapshot is used only when the network is unavailable, and the UI says plainly
when that has happened.

Two filters are applied, because "every model OpenRouter lists" includes things
this app cannot use:

* **Text output required.** Image and video generators (Recraft, Wan, Hailuo)
  appear in the same catalog. They cannot return a quiz.
* **No ``:batch`` variants.** Those are asynchronous batch endpoints; they take
  the same request but do not answer it interactively.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

MODELS_URL = "https://openrouter.ai/api/v1/models"
REQUEST_TIMEOUT = 20


@dataclass(frozen=True)
class ORModel:
    """One OpenRouter model, with prices normalized to USD per million tokens."""

    id: str
    name: str
    context_length: int
    prompt_per_m: float
    completion_per_m: float
    is_free: bool
    price_known: bool = True

    @property
    def vendor(self) -> str:
        return self.id.lstrip("~").split("/")[0]

    @property
    def has_known_price(self) -> bool:
        return self.price_known and (
            self.is_free or self.prompt_per_m > 0 or self.completion_per_m > 0
        )

    @property
    def price_label(self) -> str:
        if self.is_free:
            return "free"
        if not self.has_known_price:
            # Only reachable from the bundled snapshot, where some prices were
            # not captured. Saying "$0.00" there would read as free and be wrong.
            return "price not known"
        return f"${self.prompt_per_m:.2f}/${self.completion_per_m:.2f} per M"

    @property
    def option_label(self) -> str:
        """What the sidebar shows. Free models are marked so they stand out."""
        marker = "🆓 " if self.is_free else ""
        return f"{marker}{self.id} — {self.price_label}"


class CatalogError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _to_per_million(raw: object) -> float | None:
    """OpenRouter quotes prices per token as decimal strings ("0.0000001").

    Returns None when the field is missing or unparseable. That is deliberately
    distinct from 0.0: an explicit "0" means the model is free, whereas a price
    we could not read means we do not know — and calling the second one free
    would understate somebody's bill.
    """
    try:
        return float(raw) * 1_000_000  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def is_usable(entry: dict) -> bool:
    """Can this catalog entry actually answer a chat request with text?"""
    model_id = str(entry.get("id", ""))
    if not model_id or model_id.endswith(":batch"):
        return False
    architecture = entry.get("architecture") or {}
    outputs = architecture.get("output_modalities")
    if outputs is None:
        # Older entries omit the field; fall back to the modality string.
        modality = str(architecture.get("modality", ""))
        return "text" in modality.split("->")[-1] if "->" in modality else True
    return "text" in outputs


def parse_models(payload: dict) -> list[ORModel]:
    """Turn an ``/api/v1/models`` response into a sorted list of usable models.

    Sorting is case-insensitive on the slug, so the list reads the way a person
    scans it — all of ``anthropic/`` together, then ``deepseek/``, and so on.
    """
    entries = payload.get("data")
    if not isinstance(entries, list):
        raise CatalogError("Unexpected response from OpenRouter: no 'data' array.")

    models: list[ORModel] = []
    for entry in entries:
        if not isinstance(entry, dict) or not is_usable(entry):
            continue
        pricing = entry.get("pricing") or {}
        prompt_per_m = _to_per_million(pricing.get("prompt"))
        completion_per_m = _to_per_million(pricing.get("completion"))
        priced = prompt_per_m is not None and completion_per_m is not None
        model_id = str(entry["id"])
        models.append(
            ORModel(
                id=model_id,
                name=str(entry.get("name") or model_id),
                context_length=int(entry.get("context_length") or 0),
                prompt_per_m=prompt_per_m or 0.0,
                completion_per_m=completion_per_m or 0.0,
                # ":free" is the usual marker, but the price is the ground truth.
                is_free=priced and prompt_per_m == 0.0 and completion_per_m == 0.0,
                price_known=priced,
            )
        )

    models.sort(key=lambda m: m.id.lstrip("~").lower())
    return models


# The Free Models Router. It is a router rather than a model, so it does not
# reliably appear in /api/v1/models — and it is what a new account starts on,
# which would leave the picker opening on an option it does not contain.
#
# Guaranteed in ``load_models`` rather than in ``parse_models`` on purpose:
# parsing reports exactly what OpenRouter returned, and an empty response has to
# stay recognisable as a failure. Deciding what the picker should offer is a
# separate job from reading the API.
FREE_ROUTER_ID = "openrouter/free"


def _with_free_router(models: list[ORModel]) -> list[ORModel]:
    if any(m.id == FREE_ROUTER_ID for m in models):
        return models
    router = ORModel(
        id=FREE_ROUTER_ID,
        name="Free Models Router — picks a capable free model per request",
        context_length=0,
        prompt_per_m=0.0,
        completion_per_m=0.0,
        is_free=True,
        price_known=True,
    )
    return sorted([*models, router], key=lambda m: m.id.lstrip("~").lower())


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def fetch_models(timeout: int = REQUEST_TIMEOUT) -> list[ORModel]:
    """Download and parse the live catalog. Raises :class:`CatalogError`."""
    request = urllib.request.Request(
        MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "lecture-quiz-builder"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise CatalogError(f"OpenRouter returned HTTP {exc.code} for the model list.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise CatalogError(f"Could not reach OpenRouter: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"OpenRouter sent a malformed model list: {exc}") from exc

    models = parse_models(payload)
    if not models:
        raise CatalogError("OpenRouter's model list came back empty.")
    return models


def load_models(timeout: int = REQUEST_TIMEOUT) -> tuple[list[ORModel], str | None]:
    """Live catalog if possible, bundled snapshot if not.

    Returns ``(models, warning)``. ``warning`` is None on a live fetch and
    explains the fallback otherwise, so the UI can say which one is on screen
    rather than silently showing a stale list as if it were current.
    """
    try:
        return _with_free_router(fetch_models(timeout=timeout)), None
    except CatalogError as exc:
        return list(FALLBACK_MODELS), (
            f"{exc} Showing a bundled snapshot instead — it may be out of date, "
            "and you can still type any model slug by hand."
        )


def pricing_map(models: list[ORModel]) -> dict[str, tuple[float, float]]:
    """Slug -> (input, output) per million tokens, for the cost estimator.

    Models with no known price are left out, so the sidebar shows "—" for them
    rather than a confident $0.00.
    """
    return {
        m.id: (m.prompt_per_m, m.completion_per_m)
        for m in models
        if m.has_known_price
    }


def vendors(models: list[ORModel]) -> list[str]:
    return sorted({m.vendor for m in models}, key=str.lower)


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #

# A snapshot taken while building this app. It exists so the app still works
# without a network, NOT as the source of truth — the live fetch above is.
# Prices are USD per million tokens.
_FALLBACK_ROWS: tuple[tuple[str, float, float], ...] = (
    ("anthropic/claude-fable-5.1", 10.0, 50.0),
    ("anthropic/claude-opus-5", 15.0, 75.0),
    ("anthropic/claude-sonnet-4.5", 3.0, 15.0),
    ("bytedance-seed/seed-2-1-turbo", 0.0, 0.0),
    ("bytedance-seed/seed-2.0-code", 0.0, 0.0),
    ("deepseek/deepseek-chat-v3.1", 0.25, 0.95),
    ("deepseek/deepseek-r1", 0.70, 2.50),
    ("deepseek/deepseek-v3.2", 0.21, 0.31),
    ("deepseek/deepseek-v4-flash", 0.07, 0.17),
    ("deepseek/deepseek-v4-flash-0731", 0.05, 0.16),
    ("deepseek/deepseek-v4-pro", 0.87, 1.74),
    ("deepseek/deepseek-v4-pro-0813", 0.66, 1.98),
    ("dots-studio/dots-3-note-preview:free", 0.0, 0.0),
    ("google/gemini-3.5-flash-lite", 0.30, 2.50),
    ("google/gemini-3.6-flash", 0.0, 0.0),
    ("google/gemini-3.7-flash", 0.0, 0.0),
    ("google/gemini-3.8-flash", 0.75, 3.75),
    ("ibm-granite/granite-4.2-8b", 0.10, 0.15),
    ("inception/mercury-2.5-preview", 0.04, 0.15),
    ("inclusionai/ling-3.0-flash", 0.0, 0.0),
    ("inclusionai/ling-3.0-flash-fin:free", 0.0, 0.0),
    ("liquid/lfm-2.5-2.6b:free", 0.0, 0.0),
    ("meta/muse-glimmer-30b", 0.0, 0.0),
    ("meta/muse-spark-1.2", 0.0, 0.0),
    ("meta/muse-spark-1.3", 1.25, 4.25),
    ("meta/muse-spark-1.3-contributor", 0.10, 0.20),
    ("nvidia/nemotron-3.5-lightning", 0.0, 0.0),
    ("nvidia/nemotron-3.5-lightning:free", 0.0, 0.0),
    ("openai/gpt-4.1", 2.0, 8.0),
    ("poolside/laguna-s-2.1", 0.0, 0.0),
    ("poolside/laguna-s-2.1:free", 0.0, 0.0),
    ("qwen/qwen3.7-flash", 0.0, 0.0),
    ("qwen/qwen3.8-27b", 0.0, 0.0),
    ("qwen/qwen3.8-2.4t-a95b", 0.0, 0.0),
    ("qwen/qwen3.8-flash", 0.15, 0.47),
    ("qwen/qwen3.8-max", 0.0, 0.0),
    ("sakana/sakana-namazu", 0.0, 0.0),
    ("tencent/hy-mt2-30b-a3b", 0.0, 0.0),
    ("tencent/hy4-preview", 0.834, 2.501),
    ("thinkingmachines/inkling-small", 0.0, 0.0),
    ("thinkingmachines/inkling-small:free", 0.0, 0.0),
    ("upstage/solar-pro4", 0.0, 0.0),
    ("x-ai/grok-4.6", 2.0, 6.0),
    ("z-ai/glm-5.3", 0.0, 0.0),
    ("z-ai/glm-5.3-flash", 0.075, 0.25),
    ("z-ai/glm-flash-latest", 0.075, 0.25),
)

# Entries whose price we did not capture are marked free only when the slug says
# so — guessing "$0" from a missing figure would understate real spend.
FALLBACK_MODELS: tuple[ORModel, ...] = tuple(
    sorted(
        (
            ORModel(
                id=slug,
                name=slug,
                context_length=0,
                prompt_per_m=prompt,
                completion_per_m=completion,
                is_free=slug.endswith(":free"),
                price_known=(prompt > 0 or completion > 0 or slug.endswith(":free")),
            )
            for slug, prompt, completion in _FALLBACK_ROWS
        ),
        key=lambda m: m.id.lstrip("~").lower(),
    )
)

FALLBACK_MODELS = tuple(_with_free_router(list(FALLBACK_MODELS)))
