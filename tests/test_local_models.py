"""Running the questions through a model on your own machine.

Two things make this different from every other provider, and both are places
where the obvious implementation would be wrong.

**There is no key.** A local server does not authenticate, so demanding a key
would be a box that does nothing, and the client must not refuse to start
without one. It also bills nothing — which has to be an explicit zero, because
"no published price" reads as a gap in the app's knowledge rather than the
actual answer.

**The model list cannot be written down.** Local model names turn over monthly
and every machine has a different set downloaded, so a curated list would offer
models the user does not have and hide the ones they do. It comes from the
server, or it is wrong.

And the constraint that governs the whole feature: **a model on your laptop is
unreachable from Streamlit Cloud**, where `localhost` is Streamlit's own
container. That cannot be fixed by configuration, so the app has to say it.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src import localmodels
from src.config import PROVIDERS, AppSettings, available_providers, get_provider
from src.llm import LLMClient, LLMError


# --------------------------------------------------------------------------- #
# Addresses people actually type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "typed",
    [
        "http://localhost:11434/v1",
        "http://localhost:11434/v1/",
        "http://localhost:11434",
        "localhost:11434",
        "  localhost:11434/  ",
    ],
)
def test_every_way_of_writing_the_same_address_works(typed):
    """Asking someone to get a URL exactly right is a support ticket waiting."""
    assert localmodels.normalise(typed) == "http://localhost:11434/v1"


def test_an_empty_address_falls_back_rather_than_producing_nonsense():
    assert localmodels.normalise("") == localmodels.DEFAULT_BASE_URL
    assert localmodels.normalise("   ") == localmodels.DEFAULT_BASE_URL


# --------------------------------------------------------------------------- #
# Reading the model list
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def serve(monkeypatch, payload):
    monkeypatch.setattr(
        localmodels.urllib.request, "urlopen",
        lambda *a, **k: FakeResponse(payload),
    )


def fail(monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(localmodels.urllib.request, "urlopen", boom)


def test_the_openai_shape_is_read(monkeypatch):
    serve(monkeypatch, {"data": [{"id": "qwen3:8b"}, {"id": "llama3.1:8b"}]})
    server = localmodels.probe("http://localhost:11434")
    assert server.models == ["llama3.1:8b", "qwen3:8b"], "sorted, case-insensitively"
    assert server.is_usable


def test_ollamas_own_shape_is_read_too(monkeypatch):
    """Someone who pastes Ollama's native address should get a working list, not
    a blank dropdown with no explanation of why."""
    serve(monkeypatch, {"models": [{"name": "mistral:7b"}]})
    assert localmodels.probe("http://localhost:11434").models == ["mistral:7b"]


def test_duplicate_and_empty_names_are_dropped(monkeypatch):
    serve(monkeypatch, {"data": [{"id": "a"}, {"id": "a"}, {"id": ""}, {}, "b"]})
    assert localmodels.probe("http://localhost:11434").models == ["a", "b"]


def test_a_running_server_with_nothing_downloaded_says_which_it_is(monkeypatch):
    """"Install a model" and "start the server" are different problems, and
    conflating them sends people to the wrong fix."""
    serve(monkeypatch, {"data": []})
    server = localmodels.probe("http://localhost:11434")
    assert server.reachable and not server.is_usable
    assert "no models are downloaded" in server.status_line()


def test_garbage_from_the_server_does_not_raise(monkeypatch):
    serve(monkeypatch, ["not", "a", "dict"])
    assert localmodels.probe("http://localhost:11434").models == []


# --------------------------------------------------------------------------- #
# When nothing is there — the normal case, not an exceptional one
# --------------------------------------------------------------------------- #


def test_a_closed_port_names_the_port_and_the_fix(monkeypatch):
    fail(monkeypatch, urllib.error.URLError("Connection refused"))
    server = localmodels.probe("http://localhost:11434")
    assert not server.reachable
    assert "11434" in server.detail
    assert "Ollama" in server.detail, "say what to start, not just that it failed"


def test_a_probe_never_raises_whatever_happens(monkeypatch):
    for exc in (
        urllib.error.URLError("timed out"),
        urllib.error.HTTPError("u", 404, "nope", {}, None),
        ValueError("nonsense"),
        OSError("network is down"),
    ):
        fail(monkeypatch, exc)
        server = localmodels.probe("http://localhost:11434")
        assert not server.is_usable
        assert server.detail, "a failure the user cannot read is a failure twice"


def test_discovery_prefers_a_server_that_has_models(monkeypatch):
    """Ollama freshly installed and LM Studio actually loaded: offer the one
    that can be used, not the one that sorts first."""
    def by_port(request, *a, **k):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        if "11434" in url:
            return FakeResponse({"data": []})
        return FakeResponse({"data": [{"id": "loaded-model"}]})

    monkeypatch.setattr(localmodels.urllib.request, "urlopen", by_port)
    found = localmodels.discover()
    assert found.models == ["loaded-model"]
    assert "1234" in found.base_url


def test_discovery_with_nothing_running_points_at_the_documentation(monkeypatch):
    fail(monkeypatch, urllib.error.URLError("Connection refused"))
    found = localmodels.discover()
    assert not found.reachable
    assert "11434" in found.detail and "1234" in found.detail
    assert "LOCAL_MODELS" in found.detail


# --------------------------------------------------------------------------- #
# What will actually run on this machine
# --------------------------------------------------------------------------- #


def test_bigger_machines_are_offered_bigger_models():
    assert localmodels.largest_class_for(8) == "~3B"
    assert localmodels.largest_class_for(16) == "~8B"
    assert localmodels.largest_class_for(24) == "~14B"
    assert localmodels.largest_class_for(64) == "~70B"


def test_a_machine_that_cannot_run_anything_is_told_so_plainly():
    """Better than recommending a model that will swap. A model which technically
    loads but forces swapping is slower than the next size down, and reads to the
    user as the app having hung."""
    assert localmodels.largest_class_for(4) == ""
    assert "hosted model is the better option" in localmodels.guidance(4)


def test_guidance_survives_not_knowing_the_memory():
    assert localmodels.guidance(0)


def test_the_size_classes_are_ordered_and_described():
    sizes = [need for _, need, _ in localmodels.SIZE_CLASSES]
    assert sizes == sorted(sizes)
    assert all(desc for _, _, desc in localmodels.SIZE_CLASSES)


# --------------------------------------------------------------------------- #
# The provider, and the client it builds
# --------------------------------------------------------------------------- #


def test_the_local_provider_asks_for_no_key():
    spec = get_provider("local")
    assert spec.requires_key is False
    assert spec.is_local is True
    assert spec.sdk == "openai", "reuses the OpenAI-compatible path, not a new one"


def test_it_is_offered_even_with_no_keys_configured(monkeypatch):
    """The one provider that is always available should never be hidden behind
    a credential check it does not use."""
    monkeypatch.setattr("src.config.get_secret", lambda *a, **k: "")
    assert "local" in available_providers()


def test_a_client_starts_with_no_key(monkeypatch):
    """The check that protects every other provider must not fire here."""
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("m", (), {"OpenAI": FakeOpenAI}))

    client = LLMClient(provider="local", model="qwen3:8b", api_key="")
    assert client is not None
    assert built["base_url"] == "http://localhost:11434/v1"
    assert built["api_key"], "the SDK still needs a non-empty string"


def test_the_address_can_be_overridden_at_runtime(monkeypatch):
    """LM Studio is on a different port, and people move things."""
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("m", (), {"OpenAI": FakeOpenAI}))

    LLMClient(
        provider="local", model="m", api_key="",
        base_url_override="http://localhost:1234/v1",
    )
    assert built["base_url"] == "http://localhost:1234/v1"


def test_local_calls_get_a_long_timeout_and_no_retries(monkeypatch):
    """A local model takes minutes where a hosted one takes seconds, and a
    local failure is a machine out of memory — not a busy server that will
    recover, so retrying only makes the user wait through it twice."""
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("m", (), {"OpenAI": FakeOpenAI}))

    LLMClient(provider="local", model="m", api_key="")
    assert built["timeout"] >= 300
    assert built["max_retries"] == 0


def test_a_hosted_provider_still_demands_its_key():
    with pytest.raises(LLMError) as exc:
        LLMClient(provider="openai", model="gpt-4.1", api_key="")
    assert "No API key" in str(exc.value)


def test_local_usage_is_priced_at_exactly_zero(monkeypatch):
    """Not "no published price" — that reads as a gap in the app's knowledge
    rather than the actual answer, which is that it is free."""
    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(__import__("sys").modules, "openai",
                        type("m", (), {"OpenAI": FakeOpenAI}))

    seen: list = []
    client = LLMClient(
        provider="local", model="some-model-nobody-has-priced", api_key="",
        on_usage=lambda i, o, rate: seen.append(rate),
    )
    client._account(100, 50)
    assert seen == [(0.0, 0.0)]


def test_settings_carry_the_address():
    assert AppSettings().local_base_url.startswith("http://localhost")


def test_the_provider_note_points_at_the_instructions():
    """Someone reading the dropdown should learn the constraint before they hit
    it, not after."""
    note = PROVIDERS["local"].note
    assert "LOCAL_MODELS.md" in note
    assert "same machine" in note


# --------------------------------------------------------------------------- #
# A hosted container is not a broken setup
# --------------------------------------------------------------------------- #
#
# Reported from the live deployment: "Could not reach http://localhost:11434/v1:
# [Errno 99] Cannot assign requested address", under a caption telling the user
# to install Ollama, under a memory reading of 3 GB.
#
# Every part of that was wrong as advice. Streamlit's container forbids loopback
# connections outright, so the errno is not a symptom of anything the user can
# fix; installing Ollama would not have helped; and the 3 GB was *Streamlit's*
# memory being reported as though it were their Mac's, which would steer someone
# with 32 GB toward a 3B model.


def test_a_sandbox_that_forbids_loopback_is_named_as_such(monkeypatch):
    """The obvious reading of errno 99 is "my Ollama is broken". The actual
    answer is that this machine was never going to reach one."""
    fail(monkeypatch, urllib.error.URLError("[Errno 99] Cannot assign requested address"))
    detail = localmodels.probe("http://localhost:11434").detail

    assert "hosted container" in detail
    assert "own computer" in detail
    assert "Errno" not in detail, "an errno is not an explanation"


def test_the_hosted_branch_never_reports_the_servers_memory():
    """A memory figure read on Streamlit's container describes Streamlit, not the
    user's machine, and would send someone with 32 GB to a 3B model. The hosted
    path must not call the guidance at all."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text()
    start = source.index("def local_model_picker")
    body = source[start : source.index("\ndef ", start + 10)]

    hosted = body[body.index("if is_ephemeral_host():") : body.index("Server address")]
    assert "guidance(" not in hosted
    assert "available_memory_gb" not in hosted
    assert "return " in hosted, "the hosted branch must stop, not fall through"
    ast.parse(body.strip())


def test_the_hosted_branch_offers_the_route_that_actually_works():
    """Telling someone what cannot work is half an answer. The other half is the
    setup script, and the free model that works right now."""
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent.joinpath("app.py").read_text()
    start = source.index("def local_model_picker")
    body = source[start : source.index("\ndef ", start + 10)]
    hosted = body[body.index("if is_ephemeral_host():") : body.index("Server address")]

    assert "mac_setup.command" in hosted
    assert "windows_setup.bat" in hosted
    assert "openrouter/free" in hosted, "say what works while they set this up"


def test_the_setup_scripts_exist_and_are_not_empty():
    from pathlib import Path

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    for name in ("mac_setup.command", "windows_setup.bat"):
        path = scripts / name
        assert path.is_file(), f"{name} is missing"
        assert len(path.read_text()) > 500


def test_the_setup_scripts_never_overwrite_an_existing_secret():
    """Regenerating APP_SECRET makes every saved lecture unreadable. A setup
    script that runs twice must be safe the second time."""
    from pathlib import Path

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    mac = (scripts / "mac_setup.command").read_text()
    win = (scripts / "windows_setup.bat").read_text()

    assert 'if [ ! -f ".env" ]' in mac
    assert 'if not exist ".env"' in win
    for text in (mac, win):
        assert "back up" in text.lower()


def test_the_recommended_pull_command_matches_the_documentation():
    """A model name in the UI that the docs never mention sends people to a
    dead end."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    app = root.joinpath("app.py").read_text()
    docs = root.joinpath("docs", "LOCAL_MODELS.md").read_text()

    start = app.index("def local_model_picker")
    body = app[start : app.index("\ndef ", start + 10)]
    assert "ollama pull qwen2.5:14b" in body
    assert "qwen2.5:14b" in docs
