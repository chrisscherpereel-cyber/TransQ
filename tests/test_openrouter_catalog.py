"""Tests for the OpenRouter catalog.

The network is never touched: parsing runs against a fixture shaped like a real
``/api/v1/models`` response, and the fetch path is exercised by monkeypatching
``urlopen``. What matters here is that the app never silently offers a model it
cannot use, never calls a paid model free, and never presents a stale bundled
list as if it were live.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.llm import PRICING, estimate_cost, has_pricing, register_pricing
from src.openrouter_catalog import (
    FALLBACK_MODELS,
    CatalogError,
    ORModel,
    fetch_models,
    is_usable,
    load_models,
    parse_models,
    pricing_map,
    vendors,
)
from src.llm import Usage


def entry(
    model_id: str,
    prompt: str = "0.000001",
    completion: str = "0.000002",
    outputs: list[str] | None = None,
    **extra,
) -> dict:
    data = {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "context_length": 128000,
        "architecture": {
            "modality": "text->text",
            "input_modalities": ["text"],
            "output_modalities": ["text"] if outputs is None else outputs,
            "tokenizer": "Other",
        },
        "pricing": {"prompt": prompt, "completion": completion},
    }
    data.update(extra)
    return data


@pytest.fixture
def payload() -> dict:
    return {
        "data": [
            entry("zebra-labs/zeta-9"),
            entry("anthropic/claude-sonnet-4.5", "0.000003", "0.000015"),
            entry("deepseek/deepseek-v4-pro", "0.00000087", "0.00000174"),
            entry("deepseek/deepseek-v4-pro:batch", "0.0000006", "0.0000019"),
            entry("liquid/lfm-2.5-2.6b:free", "0", "0"),
            entry("nvidia/nemotron-3.5-lightning:free", "0", "0"),
            entry("recraft/recraft-v4-styles-pro", "0.0001", "0", outputs=["image"]),
            entry("alibaba/wan-3.0-prime", "0.00007", "0", outputs=["video"]),
            entry("Qwen/QWEN3.8-Max", "0.0000012", "0.0000060"),
            entry("~z-ai/glm-latest", "0.000000075", "0.00000025"),
        ]
    }


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


def test_image_and_video_models_are_excluded(payload):
    ids = {m.id for m in parse_models(payload)}
    assert "recraft/recraft-v4-styles-pro" not in ids
    assert "alibaba/wan-3.0-prime" not in ids


def test_batch_variants_are_excluded(payload):
    ids = {m.id for m in parse_models(payload)}
    assert "deepseek/deepseek-v4-pro" in ids
    assert "deepseek/deepseek-v4-pro:batch" not in ids


def test_is_usable_handles_a_missing_architecture_block():
    assert is_usable({"id": "vendor/model"})  # older entries: assume text
    assert not is_usable({"id": ""})
    assert is_usable({"id": "v/m", "architecture": {"modality": "text+image->text"}})
    assert not is_usable({"id": "v/m", "architecture": {"modality": "text->image"}})


def test_multimodal_input_is_kept_when_output_is_text():
    models = parse_models(
        {"data": [entry("v/m", outputs=["text"], architecture=None)]}
    )
    assert len(models) == 1


# --------------------------------------------------------------------------- #
# Sorting
# --------------------------------------------------------------------------- #


def test_models_are_sorted_alphabetically_case_insensitively(payload):
    ids = [m.id for m in parse_models(payload)]
    assert ids == sorted(ids, key=lambda s: s.lstrip("~").lower())
    # Uppercase and tilde-prefixed slugs sort with their peers, not at the ends.
    assert ids.index("anthropic/claude-sonnet-4.5") < ids.index("Qwen/QWEN3.8-Max")
    assert ids.index("Qwen/QWEN3.8-Max") < ids.index("zebra-labs/zeta-9")
    assert ids.index("~z-ai/glm-latest") < ids.index("zebra-labs/zeta-9")


# --------------------------------------------------------------------------- #
# Pricing and free detection
# --------------------------------------------------------------------------- #


def test_per_token_strings_convert_to_per_million(payload):
    sonnet = next(m for m in parse_models(payload) if "sonnet" in m.id)
    assert sonnet.prompt_per_m == pytest.approx(3.0)
    assert sonnet.completion_per_m == pytest.approx(15.0)
    assert sonnet.price_label == "$3.00/$15.00 per M"


def test_zero_priced_models_are_marked_free(payload):
    free = [m for m in parse_models(payload) if m.is_free]
    assert {m.id for m in free} == {
        "liquid/lfm-2.5-2.6b:free",
        "nvidia/nemotron-3.5-lightning:free",
    }
    assert all(m.price_label == "free" for m in free)
    assert all(m.option_label.startswith("🆓 ") for m in free)


def test_paid_models_are_never_marked_free(payload):
    for model in parse_models(payload):
        if model.is_free:
            continue
        assert not model.option_label.startswith("🆓")


def test_free_detection_uses_price_not_just_the_suffix():
    """A ":free" suffix that is actually billed must not be labelled free."""
    models = parse_models({"data": [entry("v/mislabelled:free", "0.000002", "0.000004")]})
    assert models[0].is_free is False
    assert "$2.00" in models[0].price_label


def test_malformed_prices_do_not_crash():
    models = parse_models({"data": [entry("v/m", prompt=None, completion="abc")]})
    assert models[0].prompt_per_m == 0.0
    assert models[0].has_known_price is False
    assert models[0].price_label == "price not known"


def test_pricing_map_omits_unknown_prices(payload):
    rates = pricing_map(parse_models(payload))
    assert rates["deepseek/deepseek-v4-pro"] == pytest.approx((0.87, 1.74))
    assert rates["liquid/lfm-2.5-2.6b:free"] == (0.0, 0.0)
    unknown = ORModel("v/m", "m", 0, 0.0, 0.0, is_free=False)
    assert pricing_map([unknown]) == {}


# --------------------------------------------------------------------------- #
# Vendors
# --------------------------------------------------------------------------- #


def test_vendors_are_deduplicated_and_sorted(payload):
    names = vendors(parse_models(payload))
    assert names == sorted(names, key=str.lower)
    assert "deepseek" in names and "z-ai" in names  # tilde stripped
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# Fetch and fallback
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_parses_a_live_response(monkeypatch, payload):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat.urllib.request,
        "urlopen",
        lambda req, timeout=None: FakeResponse(json.dumps(payload).encode()),
    )
    models = fetch_models()
    assert len(models) == 7  # 10 entries minus one :batch, one image, one video
    assert models[0].id == "anthropic/claude-sonnet-4.5"


def test_network_failure_falls_back_with_a_warning(monkeypatch):
    import src.openrouter_catalog as cat

    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(cat.urllib.request, "urlopen", boom)
    models, warning = load_models()

    assert models == list(FALLBACK_MODELS)
    assert warning and "snapshot" in warning
    assert "out of date" in warning, "the UI must not present stale data as live"


def test_successful_load_reports_no_warning(monkeypatch, payload):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat.urllib.request,
        "urlopen",
        lambda req, timeout=None: FakeResponse(json.dumps(payload).encode()),
    )
    models, warning = load_models()
    assert warning is None and models


def test_malformed_payloads_raise(monkeypatch):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat.urllib.request, "urlopen", lambda req, timeout=None: FakeResponse(b"not json")
    )
    with pytest.raises(CatalogError):
        fetch_models()

    with pytest.raises(CatalogError):
        parse_models({"models": []})


def test_empty_catalog_is_treated_as_a_failure(monkeypatch):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat.urllib.request,
        "urlopen",
        lambda req, timeout=None: FakeResponse(json.dumps({"data": []}).encode()),
    )
    with pytest.raises(CatalogError):
        fetch_models()


# --------------------------------------------------------------------------- #
# Bundled snapshot
# --------------------------------------------------------------------------- #


def test_fallback_is_sorted_and_has_free_models():
    ids = [m.id for m in FALLBACK_MODELS]
    assert ids == sorted(ids, key=lambda s: s.lstrip("~").lower())
    assert any(m.is_free for m in FALLBACK_MODELS)
    assert all("/" in m.id for m in FALLBACK_MODELS)


def test_fallback_contains_the_default_model():
    from src.config import AppSettings

    assert AppSettings().llm_model in {m.id for m in FALLBACK_MODELS}


def test_fallback_never_labels_a_paid_model_free():
    """In the snapshot, "free" is trusted only from the `:free` suffix, because
    real prices were not captured — guessing $0 from a missing figure would
    understate spend. `openrouter/free` is the one documented exception: it is a
    router over free models only, free by definition rather than by suffix."""
    for model in FALLBACK_MODELS:
        if model.is_free:
            assert model.id.endswith(":free") or model.id == FREE_ROUTER_ID


# --------------------------------------------------------------------------- #
# Integration with the cost estimator
# --------------------------------------------------------------------------- #


def test_live_prices_override_the_static_table(payload):
    original = PRICING.get("deepseek/deepseek-v4-pro")
    try:
        register_pricing({"deepseek/deepseek-v4-pro": (9.99, 19.99)})
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        assert estimate_cost("deepseek/deepseek-v4-pro", usage) == pytest.approx(9.99)
    finally:
        if original:
            PRICING["deepseek/deepseek-v4-pro"] = original


def test_registering_the_catalog_makes_new_slugs_priceable(payload):
    slug = "zebra-labs/zeta-9"
    PRICING.pop(slug, None)
    assert not has_pricing(slug)
    register_pricing(pricing_map(parse_models(payload)))
    assert has_pricing(slug)
    PRICING.pop(slug, None)


# --------------------------------------------------------------------------- #
# The free router is the default, so it must always be offerable
# --------------------------------------------------------------------------- #
#
# `openrouter/free` is a router rather than a model, so it does not reliably
# appear in /api/v1/models. Since it is what a new account starts on, a picker
# that could not list it would open on an option it does not contain.


from src.config import DEFAULT_PROVIDER, FREE_ROUTER, PROVIDERS  # noqa: E402
from src.openrouter_catalog import FREE_ROUTER_ID, parse_models  # noqa: E402


def test_the_shipped_default_is_the_free_router():
    assert PROVIDERS[DEFAULT_PROVIDER].models[0] == FREE_ROUTER
    assert FREE_ROUTER == FREE_ROUTER_ID


def test_parse_models_reports_the_api_verbatim_without_the_router():
    """Parsing must not invent entries: an empty response has to stay
    recognisable as a failure rather than arriving as a one-model catalog."""
    models = parse_models(
        {
            "data": [
                {
                    "id": "deepseek/deepseek-r1",
                    "name": "R1",
                    "context_length": 64000,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                }
            ]
        }
    )
    assert [m.id for m in models] == ["deepseek/deepseek-r1"]


def test_the_free_router_is_added_when_the_api_omits_it(monkeypatch):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat, "fetch_models", lambda timeout=None: [
            m for m in FALLBACK_MODELS if m.id != FREE_ROUTER_ID
        ]
    )
    models, warning = cat.load_models()
    assert warning is None
    router = next(m for m in models if m.id == FREE_ROUTER_ID)
    assert router.is_free
    assert router.price_label == "free"
    assert router.price_known, "an unpriced router would show as 'price unknown'"


def test_the_free_router_is_not_duplicated_when_the_api_lists_it(monkeypatch):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(cat, "fetch_models", lambda timeout=None: list(FALLBACK_MODELS))
    models, _ = cat.load_models()
    assert [m.id for m in models].count(FREE_ROUTER_ID) == 1


def test_adding_the_router_keeps_the_list_alphabetical(monkeypatch):
    import src.openrouter_catalog as cat

    monkeypatch.setattr(
        cat, "fetch_models",
        lambda timeout=None: parse_models({
            "data": [
                {"id": "zzz/last", "name": "Z", "context_length": 1,
                 "pricing": {"prompt": "0.001", "completion": "0.001"}},
                {"id": "aaa/first", "name": "A", "context_length": 1,
                 "pricing": {"prompt": "0.001", "completion": "0.001"}},
            ]
        }),
    )
    models, _ = cat.load_models()
    slugs = [m.id for m in models]
    assert slugs == sorted(slugs, key=lambda s: s.lstrip("~").lower())
    assert FREE_ROUTER_ID in slugs
