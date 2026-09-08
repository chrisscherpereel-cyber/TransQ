"""Tests for issued (provisioned) OpenRouter keys.

No network: ``urlopen`` is replaced with a small fake that records requests and
returns realistic payloads. What matters here is that the app never loses track
of a key it created — an orphaned key that still bills, or a stored key with no
hash to revoke it by, are the two failures this feature exists to prevent.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.accounts import AuthError, UserDirectory
from src.appconfig import AppConfig
from src.provisioning import (
    DEFAULT_LIMIT,
    LIMIT_PRESETS,
    KeyInfo,
    ProvisioningClient,
    ProvisioningError,
    key_name_for,
)
from src.storage import Cipher, MemoryStore

SECRET = "a-long-random-app-secret-for-tests"


# --------------------------------------------------------------------------- #
# Fake transport
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeOpenRouter:
    """Just enough of /api/v1/keys to exercise the client."""

    def __init__(self, wrap_single_in_list: bool = False):
        self.keys: dict[str, dict] = {}
        self.requests: list[tuple[str, str, dict | None]] = []
        self.counter = 0
        self.wrap_single_in_list = wrap_single_in_list

    def __call__(self, request, timeout=None):
        method = request.get_method()
        url = request.full_url
        body = json.loads(request.data.decode()) if request.data else None
        path = url.split("/api/v1/keys", 1)[1].strip("/")
        self.requests.append((method, path, body))

        if method == "POST":
            self.counter += 1
            key_hash = f"hash-{self.counter}"
            record = {
                "hash": key_hash,
                "name": body.get("name", ""),
                "label": f"sk-or-v1-abc...{self.counter:03d}",
                "limit": body.get("limit"),
                "limit_remaining": body.get("limit"),
                "limit_reset": body.get("limit_reset"),
                "usage": 0,
                "usage_monthly": 0,
                "disabled": False,
                "created_at": "2026-09-04T12:00:00+00:00",
            }
            self.keys[key_hash] = record
            return FakeResponse(
                json.dumps({"data": record, "key": f"sk-or-v1-secret{self.counter}"}).encode()
            )

        if method == "GET" and path:
            record = self._require(path)
            payload = [record] if self.wrap_single_in_list else record
            return FakeResponse(json.dumps({"data": payload}).encode())

        if method == "GET":
            return FakeResponse(json.dumps({"data": list(self.keys.values())}).encode())

        if method == "PATCH":
            record = self._require(path)
            if "limit" in body:
                record["limit"] = body["limit"]
                record["limit_remaining"] = body["limit"]
            if "disabled" in body:
                record["disabled"] = body["disabled"]
            return FakeResponse(json.dumps({"data": record}).encode())

        if method == "DELETE":
            self._require(path)
            del self.keys[path]
            return FakeResponse(b"")

        raise AssertionError(f"unexpected {method} {path}")

    def _require(self, key_hash: str) -> dict:
        if key_hash not in self.keys:
            raise urllib.error.HTTPError(key_hash, 404, "Not Found", {}, None)
        return self.keys[key_hash]


@pytest.fixture
def api(monkeypatch) -> FakeOpenRouter:
    import src.provisioning as prov

    fake = FakeOpenRouter()
    monkeypatch.setattr(prov.urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def client(api) -> ProvisioningClient:
    return ProvisioningClient("mgmt-key")


# --------------------------------------------------------------------------- #
# Client basics
# --------------------------------------------------------------------------- #


def test_a_management_key_is_required():
    with pytest.raises(ProvisioningError) as exc:
        ProvisioningClient("")
    assert "management key" in str(exc.value).lower()


def test_create_sends_the_documented_fields(client, api):
    client.create_key("lecture-quiz-builder/jsmith", limit=5.0, limit_reset="monthly")
    method, path, body = api.requests[-1]
    assert (method, path) == ("POST", "")
    assert body == {
        "name": "lecture-quiz-builder/jsmith",
        "limit": 5.0,
        "limit_reset": "monthly",
    }


def test_create_returns_the_one_time_secret_and_its_hash(client):
    minted = client.create_key("x", limit=5.0)
    assert minted.secret.startswith("sk-or-v1-")
    assert minted.info.hash == "hash-1"
    assert minted.info.limit == 5.0


def test_an_uncapped_key_omits_the_limit_fields(client, api):
    client.create_key("x", limit=None)
    assert api.requests[-1][2] == {"name": "x"}


def test_bearer_token_is_sent(monkeypatch):
    import src.provisioning as prov

    seen = {}

    def capture(request, timeout=None):
        seen.update(request.headers)
        return FakeResponse(json.dumps({"data": []}).encode())

    monkeypatch.setattr(prov.urllib.request, "urlopen", capture)
    ProvisioningClient("mgmt-abc").verify()
    assert seen.get("Authorization") == "Bearer mgmt-abc"


def test_update_and_revoke(client, api):
    minted = client.create_key("x", limit=5.0)
    updated = client.update_key(minted.info.hash, limit=25.0)
    assert updated.limit == 25.0

    paused = client.update_key(minted.info.hash, disabled=True)
    assert paused.disabled is True

    client.delete_key(minted.info.hash)
    assert api.keys == {}
    with pytest.raises(ProvisioningError):
        client.get_key(minted.info.hash)


def test_update_with_nothing_to_change_is_a_read(client, api):
    minted = client.create_key("x", limit=5.0)
    api.requests.clear()
    client.update_key(minted.info.hash)
    assert api.requests[-1][0] == "GET"


def test_list_keys(client):
    client.create_key("a", limit=1.0)
    client.create_key("b", limit=2.0)
    assert {k.name for k in client.list_keys()} == {"a", "b"}


# --------------------------------------------------------------------------- #
# Response-shape tolerance and errors
# --------------------------------------------------------------------------- #


def test_single_records_wrapped_in_a_list_are_accepted(monkeypatch):
    """The docs show GET /keys/{hash} returning data as a one-element array."""
    import src.provisioning as prov

    fake = FakeOpenRouter(wrap_single_in_list=True)
    monkeypatch.setattr(prov.urllib.request, "urlopen", fake)
    client = ProvisioningClient("mgmt")
    minted = client.create_key("x", limit=5.0)
    assert client.get_key(minted.info.hash).hash == minted.info.hash


def test_a_create_reply_without_a_key_is_an_error(monkeypatch):
    import src.provisioning as prov

    monkeypatch.setattr(
        prov.urllib.request,
        "urlopen",
        lambda r, timeout=None: FakeResponse(json.dumps({"data": {"hash": "h"}}).encode()),
    )
    with pytest.raises(ProvisioningError) as exc:
        ProvisioningClient("mgmt").create_key("x")
    assert "orphan" in str(exc.value)


@pytest.mark.parametrize(
    "code,expected",
    [
        (401, "management"),
        (403, "management"),
        (404, "no longer exists"),
        (429, "rate-limiting"),
        (500, "HTTP 500"),
    ],
)
def test_http_errors_explain_themselves(monkeypatch, code, expected):
    import src.provisioning as prov

    def fail(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, code, "err", {}, None)

    monkeypatch.setattr(prov.urllib.request, "urlopen", fail)
    with pytest.raises(ProvisioningError) as exc:
        ProvisioningClient("mgmt").get_key("h")
    assert expected in str(exc.value)


def test_network_failure_is_reported(monkeypatch):
    import src.provisioning as prov

    def fail(request, timeout=None):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(prov.urllib.request, "urlopen", fail)
    with pytest.raises(ProvisioningError) as exc:
        ProvisioningClient("mgmt").verify()
    assert "reach OpenRouter" in str(exc.value)


def test_malformed_prices_and_missing_fields_do_not_crash():
    from src.provisioning import _coerce_key_info

    info = _coerce_key_info({"hash": "h", "limit": "not-a-number"})
    assert info.limit is None and info.usage == 0.0
    assert _coerce_key_info({}).hash == ""


# --------------------------------------------------------------------------- #
# KeyInfo arithmetic
# --------------------------------------------------------------------------- #


def test_spend_prefers_limit_remaining():
    """With a resetting cap, cumulative usage overstates the current period."""
    info = KeyInfo(hash="h", limit=10.0, limit_remaining=7.5, usage=93.0, usage_monthly=2.5)
    assert info.spent == pytest.approx(2.5)
    assert info.fraction_used == pytest.approx(0.25)
    assert info.status_label == "$2.50 of $10.00"


def test_spend_falls_back_to_usage_when_uncapped():
    info = KeyInfo(hash="h", limit=None, usage_monthly=4.0)
    assert not info.is_capped
    assert info.spent == pytest.approx(4.0)
    assert info.fraction_used == 0.0
    assert info.status_label == "no cap set"


def test_fraction_is_clamped_and_disabled_wins():
    over = KeyInfo(hash="h", limit=5.0, limit_remaining=-1.0)
    assert over.fraction_used == 1.0
    assert KeyInfo(hash="h", limit=5.0, disabled=True).status_label == "disabled"


def test_key_names_identify_the_person_and_the_app():
    assert key_name_for("jsmith") == "lecture-quiz-builder/jsmith"


def test_limit_presets_are_sane():
    assert DEFAULT_LIMIT in LIMIT_PRESETS
    assert list(LIMIT_PRESETS) == sorted(LIMIT_PRESETS)
    assert min(LIMIT_PRESETS) > 0


# --------------------------------------------------------------------------- #
# App config
# --------------------------------------------------------------------------- #


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(MemoryStore(), Cipher(SECRET))


def test_config_plain_values(config):
    assert config.get("default_key_limit", 5.0) == 5.0
    config.set("default_key_limit", 25.0)
    assert config.get("default_key_limit") == 25.0


def test_management_key_is_stored_encrypted(config):
    config.set_secret("openrouter_management_key", "sk-or-v1-MANAGEMENT")
    raw = str(config.store.read("config"))
    assert "MANAGEMENT" not in raw
    assert config.get_secret("openrouter_management_key") == "sk-or-v1-MANAGEMENT"
    assert config.has_secret("openrouter_management_key")


def test_management_key_can_be_removed(config):
    config.set_secret("openrouter_management_key", "sk-or-v1-x")
    config.set_secret("openrouter_management_key", "")
    assert not config.has_secret("openrouter_management_key")
    assert config.get_secret("openrouter_management_key") == ""


def test_secrets_need_encryption():
    plain = AppConfig(MemoryStore(), cipher=None)
    with pytest.raises(RuntimeError):
        plain.set_secret("k", "v")
    assert plain.get_secret("k") == ""


# --------------------------------------------------------------------------- #
# Account records
# --------------------------------------------------------------------------- #


@pytest.fixture
def directory() -> UserDirectory:
    directory = UserDirectory(MemoryStore(), Cipher(SECRET))
    directory.bootstrap_admin("admin", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    return directory


def test_issued_key_and_its_handle_are_saved_together(directory):
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-theirs",
        {"hash": "hash-1", "name": "lecture-quiz-builder/jsmith", "limit": 5.0,
         "limit_reset": "monthly"},
    )
    user = directory.get("jsmith")
    assert directory.get_issued_key(user, "openrouter") == "sk-or-v1-theirs"
    record = directory.provisioned_record(user, "openrouter")
    assert record["hash"] == "hash-1" and record["limit"] == 5.0
    # Without the hash the key could never be revoked.
    assert record["hash"], "a stored key must always carry its revocation handle"


def test_issued_keys_are_encrypted_at_rest(directory):
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-VERYSECRET", {"hash": "hash-1"}
    )
    assert "VERYSECRET" not in str(directory.store.read("users/u/jsmith"))


def test_revoking_clears_both_halves(directory):
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-theirs", {"hash": "hash-1", "limit": 5.0}
    )
    record = directory.clear_provisioned_key("jsmith", "openrouter")
    assert record["hash"] == "hash-1"

    user = directory.get("jsmith")
    assert directory.get_issued_key(user, "openrouter") == ""
    assert directory.provisioned_record(user, "openrouter") == {}


def test_issued_keys_do_not_leak_between_accounts(directory):
    directory.record_provisioned_key("admin", "openrouter", "sk-admin", {"hash": "h1"})
    directory.record_provisioned_key("jsmith", "openrouter", "sk-jsmith", {"hash": "h2"})
    assert directory.get_issued_key(directory.get("admin"), "openrouter") == "sk-admin"
    assert directory.provisioned_record(directory.get("jsmith"), "openrouter")["hash"] == "h2"


def test_recording_against_a_missing_account_fails(directory):
    with pytest.raises(AuthError):
        directory.record_provisioned_key("ghost", "openrouter", "sk", {"hash": "h"})


def test_provisioned_records_survive_a_reload(directory):
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-theirs", {"hash": "hash-1", "limit": 5.0}
    )
    reloaded = UserDirectory(directory.store, directory.cipher)
    session = reloaded.authenticate("jsmith", "temp-password-1")
    assert reloaded.provisioned_record(session.user, "openrouter")["hash"] == "hash-1"


# --------------------------------------------------------------------------- #
# The whole loop
# --------------------------------------------------------------------------- #


def test_issue_use_inspect_revoke(directory, api):
    """Mint for a user, read the cap back, then revoke it everywhere."""
    client = ProvisioningClient("mgmt")

    minted = client.create_key(key_name_for("jsmith"), limit=5.0, limit_reset="monthly")
    directory.record_provisioned_key(
        "jsmith", "openrouter", minted.secret,
        {"hash": minted.info.hash, "name": minted.info.name, "limit": 5.0,
         "limit_reset": "monthly"},
    )

    user = directory.get("jsmith")
    assert directory.get_issued_key(user, "openrouter") == minted.secret

    # Spend half of it upstream.
    api.keys[minted.info.hash]["limit_remaining"] = 2.5
    info = client.get_key(minted.info.hash)
    assert info.status_label == "$2.50 of $5.00"
    assert info.fraction_used == pytest.approx(0.5)

    record = directory.clear_provisioned_key("jsmith", "openrouter")
    client.delete_key(record["hash"])

    assert api.keys == {}
    assert directory.get_issued_key(directory.get("jsmith"), "openrouter") == ""
