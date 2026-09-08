"""Tests for persistence, accounts, and the usage ledger.

Nothing here touches the network or Dropbox. The point is to pin down the
properties that would be quietly dangerous to get wrong: that plaintext never
reaches storage, that a wrong secret fails loudly instead of returning garbage,
that the last administrator cannot be removed, and that costs are attributed to
the account that incurred them.
"""

from __future__ import annotations

import pytest

from src.accounts import (
    ROLES,
    AuthError,
    User,
    UserDirectory,
    hash_password,
    suggest_password,
    verify_password,
)
from src.keymgmt import KEYRING_PATH, load_or_create_keyring
from src.storage import (
    Cipher,
    GuardedStore,
    LocalStore,
    MemoryStore,
    StorageError,
    build_store,
    derive_key,
)
from src.usage import LiveMeter, UsageLog, UsageRecord


SECRET = "a-long-random-app-secret-for-tests"


@pytest.fixture
def cipher() -> Cipher:
    return Cipher(SECRET)


@pytest.fixture
def directory(cipher) -> UserDirectory:
    return UserDirectory(MemoryStore(), cipher)


# --------------------------------------------------------------------------- #
# Encryption
# --------------------------------------------------------------------------- #


def test_key_derivation_is_deterministic_and_secret_specific():
    assert derive_key(SECRET) == derive_key(SECRET)
    assert derive_key(SECRET) != derive_key(SECRET + "x")
    with pytest.raises(StorageError):
        derive_key("")


def test_round_trip(cipher):
    payload = {"users": {"c": {"role": "admin"}}, "n": 3}
    assert cipher.decrypt(cipher.encrypt(payload)) == payload


def test_ciphertext_does_not_leak_plaintext(cipher):
    blob = cipher.encrypt({"api_key": "sk-or-v1-SUPERSECRET", "user": "chris"})
    assert b"SUPERSECRET" not in blob
    assert b"chris" not in blob
    assert b"api_key" not in blob


def test_encryption_is_randomized(cipher):
    """Identical payloads must not produce identical blobs."""
    payload = {"same": "value"}
    assert cipher.encrypt(payload) != cipher.encrypt(payload)


def test_a_different_secret_fails_loudly(cipher):
    blob = cipher.encrypt({"a": 1})
    with pytest.raises(StorageError) as exc:
        Cipher("some-other-secret").decrypt(blob)
    assert "APP_SECRET" in str(exc.value)


def test_text_encryption_round_trip_and_bad_token(cipher):
    token = cipher.encrypt_text("sk-or-v1-abc")
    assert token != "sk-or-v1-abc"
    assert cipher.decrypt_text(token) == "sk-or-v1-abc"
    assert cipher.decrypt_text("garbage") == ""


# --------------------------------------------------------------------------- #
# Local store
# --------------------------------------------------------------------------- #


def test_local_store_persists_encrypted(tmp_path, cipher):
    store = LocalStore(cipher, root=str(tmp_path))
    assert store.read("users") is None

    store.write("users", {"users": {"c": {"password_hash": "deadbeef"}}})
    assert store.read("users")["users"]["c"]["password_hash"] == "deadbeef"

    on_disk = (tmp_path / "users.enc").read_bytes()
    assert b"deadbeef" not in on_disk, "the file must not contain plaintext"
    assert b"password_hash" not in on_disk


def test_local_store_survives_a_new_instance(tmp_path):
    """The keyring is shared so both instances derive the same key."""
    first = LocalStore(Cipher(SECRET), root=str(tmp_path))
    first.write("k", {"v": 1})
    header = first.read_plain(KEYRING_PATH)
    if header is None:  # a fresh Cipher creates its own keyring; persist it
        first.write_plain(KEYRING_PATH, first.cipher.keyring.to_dict())
        header = first.read_plain(KEYRING_PATH)

    keyring, _ = load_or_create_keyring(SECRET, header)
    second = LocalStore(Cipher(SECRET, keyring), root=str(tmp_path))
    assert second.read("k") == {"v": 1}


def test_local_store_leaves_no_temp_files(tmp_path, cipher):
    store = LocalStore(cipher, root=str(tmp_path))
    store.write("users", {"v": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_path_traversal_is_neutralized(tmp_path, cipher):
    store = LocalStore(cipher, root=str(tmp_path))
    store.write("../../escape", {"v": 1})
    assert not (tmp_path.parent.parent / "escape.enc").exists()


def test_build_store_without_a_secret_warns_and_saves_nothing():
    store, warning = build_store({})
    assert isinstance(store, MemoryStore)
    assert warning and "not being saved" in warning


def test_build_store_with_a_secret_uses_local_files(tmp_path):
    store, warning = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    assert warning is None
    assert isinstance(store, GuardedStore)
    assert isinstance(store.inner, LocalStore)


def test_build_store_writes_a_keyring_header(tmp_path):
    """The salt is not secret, but it must be recorded or nothing reopens."""
    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    header = store.read_plain(KEYRING_PATH)
    assert header["primary"]["kdf"] == "scrypt"
    assert len(bytes.fromhex(header["primary"]["salt"])) == 16
    # The old PBKDF2 scheme is kept as a reader so existing stores still open.
    assert any(spec["kdf"] == "pbkdf2" for spec in header["legacy"])


def test_two_deployments_derive_different_keys(tmp_path):
    """A random per-deployment salt: same secret, different key material."""
    a, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path / "a")})
    b, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path / "b")})
    assert (
        a.read_plain(KEYRING_PATH)["primary"]["salt"]
        != b.read_plain(KEYRING_PATH)["primary"]["salt"]
    )


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #


def test_password_hashing_round_trip():
    digest, salt = hash_password("correct horse battery")
    assert verify_password("correct horse battery", digest, salt)
    assert not verify_password("wrong", digest, salt)


def test_same_password_gets_different_salts():
    a_hash, a_salt = hash_password("same-password-here")
    b_hash, b_salt = hash_password("same-password-here")
    assert a_salt != b_salt and a_hash != b_hash


def test_short_passwords_are_refused():
    with pytest.raises(AuthError):
        hash_password("short")


def test_verify_is_safe_with_empty_inputs():
    assert not verify_password("", "abc", "def")
    assert not verify_password("password123", "", "")


def test_suggested_passwords_are_long_and_unambiguous():
    for _ in range(20):
        candidate = suggest_password()
        assert len(candidate) == 14
        assert not set(candidate) & set("l1IO0")
    assert suggest_password() != suggest_password()


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


def test_bootstrap_then_login(directory):
    assert directory.is_empty()
    session = directory.bootstrap_admin("Admin", "first-password", "C S")
    assert session.user.username == "admin" and session.user.is_admin
    assert not session.user.must_change_password
    assert session.dek is not None, "sign-in must yield the account data key"
    assert not directory.is_empty()

    again = directory.authenticate("ADMIN", "first-password")
    assert again.user.username == "admin"
    assert again.user.last_login


def test_bootstrap_refuses_once_accounts_exist(directory):
    directory.bootstrap_admin("admin", "first-password")
    with pytest.raises(AuthError):
        directory.bootstrap_admin("intruder", "another-password")


def test_passwords_are_never_stored_in_the_clear(directory):
    directory.bootstrap_admin("admin", "plaintext-password")
    assert "plaintext-password" not in str(directory.store.read("users/u/admin"))


def test_wrong_password_and_unknown_user_give_the_same_message(directory):
    directory.bootstrap_admin("admin", "first-password")
    with pytest.raises(AuthError) as wrong:
        directory.authenticate("admin", "nope")
    with pytest.raises(AuthError) as unknown:
        directory.authenticate("ghost", "nope")
    assert str(wrong.value) == str(unknown.value), "must not reveal which usernames exist"


def test_admin_creates_users_who_must_change_password(directory):
    directory.bootstrap_admin("admin", "first-password")
    created = directory.create_user("jsmith", "temp-password-1", "instructor", "J Smith")
    assert created.must_change_password
    assert created.role == "instructor"

    directory.set_password("jsmith", "their-own-password")
    assert not directory.get("jsmith").must_change_password


def test_duplicate_and_invalid_usernames_are_refused(directory):
    directory.bootstrap_admin("admin", "first-password")
    with pytest.raises(AuthError):
        directory.create_user("admin", "another-password")
    with pytest.raises(AuthError):
        directory.create_user("bad user!", "another-password")
    with pytest.raises(AuthError):
        directory.create_user("", "another-password")
    with pytest.raises(AuthError):
        directory.create_user("ok", "another-password", role="superuser")


def test_disabled_accounts_cannot_sign_in(directory):
    directory.bootstrap_admin("admin", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    directory.set_active("jsmith", False)
    with pytest.raises(AuthError) as exc:
        directory.authenticate("jsmith", "temp-password-1")
    assert "disabled" in str(exc.value)


def test_the_last_administrator_is_protected(directory):
    directory.bootstrap_admin("admin", "first-password")
    for action in (
        lambda: directory.delete_user("admin"),
        lambda: directory.set_role("admin", "instructor"),
        lambda: directory.set_active("admin", False),
    ):
        with pytest.raises(AuthError) as exc:
            action()
        assert "only administrator" in str(exc.value)


def test_protection_lifts_once_a_second_admin_exists(directory):
    directory.bootstrap_admin("admin", "first-password")
    directory.create_user("deputy", "temp-password-1", role="admin")
    directory.set_role("admin", "instructor")
    assert directory.get("admin").role == "instructor"
    assert directory.admin_count() == 1


def test_usernames_are_case_insensitive(directory):
    directory.bootstrap_admin("admin", "first-password")
    directory.create_user("JSmith", "temp-password-1")
    assert directory.get("jsmith") is not None
    assert directory.get("JSMITH") is not None


# --------------------------------------------------------------------------- #
# Per-account settings and keys
# --------------------------------------------------------------------------- #


def test_settings_persist_per_account(directory):
    directory.bootstrap_admin("admin", "first-password")
    directory.create_user("jsmith", "temp-password-1")

    directory.save_settings("admin", {"llm_model": "deepseek/deepseek-v4-pro"})
    directory.save_settings("jsmith", {"llm_model": "google/gemini-3.8-flash"})

    assert directory.get("admin").settings["llm_model"] == "deepseek/deepseek-v4-pro"
    assert directory.get("jsmith").settings["llm_model"] == "google/gemini-3.8-flash"


def test_settings_merge_rather_than_replace(directory):
    directory.bootstrap_admin("admin", "first-password")
    directory.save_settings("admin", {"llm_model": "a", "num_questions": 12})
    directory.save_settings("admin", {"llm_model": "b"})
    saved = directory.get("admin").settings
    assert saved == {"llm_model": "b", "num_questions": 12}


def test_api_keys_are_stored_encrypted_and_scoped_to_the_account(directory):
    admin = directory.bootstrap_admin("admin", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    jsmith = directory.authenticate("jsmith", "temp-password-1")

    directory.save_api_key("admin", "openrouter", "sk-or-v1-ADMINKEY", admin.dek)
    directory.save_api_key("jsmith", "openrouter", "sk-or-v1-OTHERKEY", jsmith.dek)

    raw = str(directory.store.read("users/u/admin")) + str(
        directory.store.read("users/u/jsmith")
    )
    assert "ADMINKEY" not in raw and "OTHERKEY" not in raw

    assert (
        directory.get_api_key(directory.get("admin"), "openrouter", admin.dek)
        == "sk-or-v1-ADMINKEY"
    )
    assert (
        directory.get_api_key(directory.get("jsmith"), "openrouter", jsmith.dek)
        == "sk-or-v1-OTHERKEY"
    )
    # One user's data key must not open another user's saved key.
    assert directory.get_api_key(directory.get("jsmith"), "openrouter", admin.dek) == ""


def test_api_keys_are_per_provider_and_removable(directory):
    session = directory.bootstrap_admin("admin", "first-password")
    directory.save_api_key("admin", "openrouter", "sk-or-1", session.dek)
    directory.save_api_key("admin", "gemini", "AIza-2", session.dek)

    admin = directory.get("admin")
    assert directory.get_api_key(admin, "openrouter", session.dek) == "sk-or-1"
    assert directory.get_api_key(admin, "gemini", session.dek) == "AIza-2"
    assert directory.get_api_key(admin, "anthropic", session.dek) == ""

    directory.save_api_key("admin", "gemini", "", session.dek)
    assert not directory.has_api_key(directory.get("admin"), "gemini")


def test_personal_keys_need_the_session_data_key(directory):
    """Without the user's own key material there is nothing to encrypt under."""
    directory.bootstrap_admin("admin", "first-password")
    with pytest.raises(AuthError):
        directory.save_api_key("admin", "openrouter", "sk-or-1", None)


def test_issued_keys_still_need_the_app_cipher():
    plain = UserDirectory(MemoryStore(), cipher=None)
    plain.bootstrap_admin("admin", "first-password")
    with pytest.raises(AuthError):
        plain.record_provisioned_key("admin", "openrouter", "sk-or-1", {"hash": "h"})


def test_user_serialization_round_trip():
    user = User(username="c", role="admin", settings={"a": 1}, api_keys={"x": "y"})
    restored = User.from_dict(user.to_dict())
    assert restored == user
    # Unknown fields from a future version must not break loading.
    assert User.from_dict({"username": "c", "unexpected": 1}).username == "c"


# --------------------------------------------------------------------------- #
# Usage ledger
# --------------------------------------------------------------------------- #


def record(username="c", model="m", cost=1.0, operation="generate", **kw) -> UsageRecord:
    base = dict(
        timestamp=kw.pop("timestamp", "2026-09-04T10:00:00+00:00"),
        username=username,
        provider="openrouter",
        model=model,
        operation=operation,
        input_tokens=kw.pop("input_tokens", 1000),
        output_tokens=kw.pop("output_tokens", 500),
        cost=cost,
    )
    base.update(kw)
    return UsageRecord(**base)


@pytest.fixture
def log() -> UsageLog:
    return UsageLog(MemoryStore())


def test_usage_totals_and_attribution(log):
    log.append(record("chris", cost=0.10))
    log.append(record("chris", cost=0.20))
    log.append(record("jsmith", cost=1.00))

    assert log.totals().cost == pytest.approx(1.30)
    assert log.totals("chris").cost == pytest.approx(0.30)
    assert log.totals("chris").runs == 2
    assert log.totals("jsmith").cost == pytest.approx(1.00)
    assert log.totals("nobody").runs == 0


def test_usage_breakdowns(log):
    log.append(record("chris", model="a", cost=1.0, operation="generate"))
    log.append(record("chris", model="b", cost=2.0, operation="replace"))
    log.append(record("jsmith", model="a", cost=4.0, operation="generate"))

    by_model = log.by_model()
    assert list(by_model) == ["a", "b"]  # sorted by cost, descending
    assert by_model["a"].cost == pytest.approx(5.0)

    assert log.by_user()["jsmith"].cost == pytest.approx(4.0)
    assert log.by_operation("chris")["replace"].cost == pytest.approx(2.0)


def test_by_day_buckets_and_windows(log):
    import datetime as dt

    today = dt.date.today().isoformat()
    log.append(record(timestamp=f"{today}T09:00:00+00:00", cost=1.0))
    log.append(record(timestamp=f"{today}T15:00:00+00:00", cost=2.0))
    log.append(record(timestamp="2020-01-01T00:00:00+00:00", cost=99.0))

    daily = log.by_day(days=30)
    assert daily[today] == pytest.approx(3.0)
    assert "2020-01-01" not in daily, "old records fall outside the window"


def test_unpriced_runs_are_counted_but_not_costed(log):
    log.append(record(cost=0.0, cost_known=False))
    totals = log.totals()
    assert totals.unpriced_runs == 1
    assert totals.cost_label == "—"

    log.append(record(cost=0.5))
    assert log.totals().cost_label == "$0.5000 +"


def test_ledger_is_bounded(log, monkeypatch):
    import src.usage as usage_module

    monkeypatch.setattr(usage_module, "MAX_RECORDS", 5)
    for _ in range(9):
        log.append(record())
    assert len(log.records()) == 5


# --------------------------------------------------------------------------- #
# Live meter
# --------------------------------------------------------------------------- #


def test_live_meter_accumulates_per_call():
    meter = LiveMeter(provider="openrouter", model="deepseek/deepseek-v4-pro")
    meter.record(1_000_000, 0, (0.87, 1.74))
    assert meter.cost == pytest.approx(0.87)
    meter.record(0, 1_000_000, (0.87, 1.74))
    assert meter.cost == pytest.approx(0.87 + 1.74)
    assert meter.calls == 2
    assert meter.total_tokens == 2_000_000
    assert len(meter.history) == 2, "history drives the live readout"


def test_live_meter_marks_unknown_pricing():
    meter = LiveMeter(model="mystery/model")
    meter.record(1000, 500, None)
    assert meter.cost_known is False
    assert meter.cost_label == "—"


def test_meter_reset_clears_everything():
    meter = LiveMeter()
    meter.record(10, 10, None)
    meter.reset()
    assert (meter.total_tokens, meter.calls, meter.cost, meter.cost_known) == (0, 0, 0.0, True)


def test_meter_converts_to_a_ledger_record():
    meter = LiveMeter(provider="openrouter", model="m")
    meter.record(100, 50, (1.0, 2.0))
    entry = meter.to_record("chris", "generate", "lecture4.mp3")
    assert entry.username == "chris" and entry.operation == "generate"
    assert entry.source == "lecture4.mp3"
    assert entry.total_tokens == 150 and entry.calls == 1


def test_client_reports_every_call_to_the_meter(monkeypatch):
    """The callback is what makes the sidebar move mid-run."""
    import sys
    import types

    from src.llm import LLMClient

    class Completions:
        @staticmethod
        def create(**kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))],
                usage=types.SimpleNamespace(prompt_tokens=700, completion_tokens=300),
            )

    module = types.ModuleType("openai")
    module.OpenAI = lambda **kw: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions())
    )
    monkeypatch.setitem(sys.modules, "openai", module)

    meter = LiveMeter()
    client = LLMClient(provider="openrouter", model="deepseek/deepseek-v4-pro", api_key="k")
    client.on_usage = lambda i, o, rate: meter.record(i, o, rate)

    client.complete_json("sys", "user")
    client.complete_json("sys", "user")

    assert meter.calls == 2
    assert meter.total_tokens == 2000
    assert meter.cost > 0, "a priced model should accumulate cost live"
    assert meter.total_tokens == client.usage.input_tokens + client.usage.output_tokens


# --------------------------------------------------------------------------- #
# The model you last used is the one you come back to
# --------------------------------------------------------------------------- #
#
# The sidebar has always *read* a saved provider and model, but nothing wrote
# them unless you found the "Save these settings" button — so every sign-in put
# you back on the shipped default. The write now happens on use.


def test_the_provider_and_model_last_used_survive_a_restart(tmp_path):
    from src.config import DEFAULT_PROVIDER, PROVIDERS

    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    directory = UserDirectory(store, Cipher(SECRET))
    directory.create_user("chris", "a-good-long-password", role="admin")

    # Nothing saved yet: a new account starts on the shipped default.
    user = directory.get("chris")
    assert user.settings.get("provider", DEFAULT_PROVIDER) == DEFAULT_PROVIDER
    assert (
        user.settings.get("llm_model", PROVIDERS[DEFAULT_PROVIDER].models[0])
        == "openrouter/free"
    )

    directory.save_settings(
        "chris", {"provider": "openrouter", "llm_model": "deepseek/deepseek-r1"}
    )

    # A fresh process, as after a sign-out or a restart.
    reopened, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    later = UserDirectory(reopened, Cipher(SECRET)).get("chris")
    assert later.settings["llm_model"] == "deepseek/deepseek-r1"
    assert later.settings["provider"] == "openrouter"


def test_saving_the_model_leaves_other_settings_alone(tmp_path):
    """Recording a model on use must not quietly reset question count or Bloom
    levels — it runs after every generation."""
    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    directory = UserDirectory(store, Cipher(SECRET))
    directory.create_user("chris", "a-good-long-password")

    directory.save_settings("chris", {"num_questions": 25, "whisper_model": "base"})
    directory.save_settings("chris", {"llm_model": "x-ai/grok-4.6"})

    settings = directory.get("chris").settings
    assert settings["llm_model"] == "x-ai/grok-4.6"
    assert settings["num_questions"] == 25
    assert settings["whisper_model"] == "base"


def test_each_account_remembers_its_own_model(tmp_path):
    store, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    directory = UserDirectory(store, Cipher(SECRET))
    directory.create_user("chris", "a-good-long-password")
    directory.create_user("jsmith", "another-good-password")

    directory.save_settings("chris", {"llm_model": "deepseek/deepseek-r1"})
    directory.save_settings("jsmith", {"llm_model": "openrouter/free"})

    assert directory.get("chris").settings["llm_model"] == "deepseek/deepseek-r1"
    assert directory.get("jsmith").settings["llm_model"] == "openrouter/free"
