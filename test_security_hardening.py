"""Tests for the security hardening: key management, envelope encryption,
rollback detection, lockout, and the audit trail.

Each test states the property it is defending, because that is the part worth
keeping when someone later refactors this. The valuable assertions here are the
negative ones — what must *not* be readable, and by whom.
"""

from __future__ import annotations

import base64
import datetime as dt

import pytest
from cryptography.fernet import Fernet

from src import audit as audit_module
from src.accounts import (
    LOCKOUT_MINUTES,
    MAX_FAILED_ATTEMPTS,
    AuthError,
    UserDirectory,
    mask_key,
)
from src.appconfig import AppConfig
from src.audit import AuditLog
from src.keymgmt import (
    PURPOSE_APIKEYS,
    PURPOSE_CONFIG,
    PURPOSE_DOCUMENTS,
    KeySpec,
    Keyring,
    KeyError_,
    decrypt_with_dek,
    derive_kek,
    encrypt_with_dek,
    legacy_spec,
    load_or_create_keyring,
    new_dek,
    new_spec,
    unwrap_dek,
    wrap_dek,
)
from src.storage import Cipher, GuardedStore, LocalStore, MemoryStore, RollbackError

SECRET = "a-long-random-app-secret-for-tests"
OTHER = "an-entirely-different-secret-value"


@pytest.fixture
def keyring() -> Keyring:
    ring, _ = load_or_create_keyring(SECRET, None)
    return ring


@pytest.fixture
def directory() -> UserDirectory:
    store = GuardedStore(MemoryStore())
    return UserDirectory(store, Cipher(SECRET), AuditLog(store))


# --------------------------------------------------------------------------- #
# 3. Key derivation
# --------------------------------------------------------------------------- #


def test_the_default_kdf_is_memory_hard(keyring):
    assert keyring.primary.kdf == "scrypt"
    assert keyring.primary.n >= 2**15


def test_the_salt_is_random_per_deployment():
    a, _ = load_or_create_keyring(SECRET, None)
    b, _ = load_or_create_keyring(SECRET, None)
    assert a.primary.salt != b.primary.salt
    assert len(a.primary.salt) == 16


def test_purposes_get_different_keys(keyring):
    """The old scheme used one key for everything, so the inner layer of
    encryption around API keys protected against nothing."""
    token = keyring.encrypt(PURPOSE_APIKEYS, b"sk-or-v1-secret")

    assert keyring.decrypt(PURPOSE_APIKEYS, token) == b"sk-or-v1-secret"
    for other in (PURPOSE_DOCUMENTS, PURPOSE_CONFIG):
        with pytest.raises(KeyError_):
            keyring.decrypt(other, token)


def test_every_ciphertext_records_its_key_version(keyring):
    import json

    envelope = json.loads(keyring.encrypt(PURPOSE_DOCUMENTS, b"x").decode())
    assert envelope["kv"] == keyring.primary.key_version
    assert envelope["env"] == 2


def test_a_wrong_secret_fails_loudly_not_silently(keyring):
    blob = keyring.encrypt(PURPOSE_DOCUMENTS, b"payload")
    stranger = Keyring(secret=OTHER, primary=keyring.primary, legacy=[])
    with pytest.raises(KeyError_) as exc:
        stranger.decrypt(PURPOSE_DOCUMENTS, blob)
    assert "APP_SECRET" in str(exc.value)


def test_key_specs_round_trip_through_the_header(keyring):
    restored = KeySpec.from_dict(keyring.primary.to_dict())
    assert restored == keyring.primary


# --------------------------------------------------------------------------- #
# 3b. Backward compatibility — the upgrade must not lose data
# --------------------------------------------------------------------------- #


def test_data_written_by_the_old_scheme_still_opens():
    """PBKDF2, fixed salt, no HKDF, bare Fernet token with no envelope."""
    old = Keyring(secret=SECRET, primary=legacy_spec(), legacy=[])
    blob = old._fernet(legacy_spec(), PURPOSE_DOCUMENTS).encrypt(b"legacy payload")

    modern, created = load_or_create_keyring(SECRET, None)
    assert created
    assert modern.decrypt(PURPOSE_DOCUMENTS, blob) == b"legacy payload"


def test_upgrading_a_store_rewrites_under_the_new_key(tmp_path):
    from src.storage import build_store

    old_cipher = Cipher(SECRET, Keyring(secret=SECRET, primary=legacy_spec(), legacy=[]))
    legacy_store = LocalStore(old_cipher, root=str(tmp_path))
    legacy_store.write("users", {"users": {"chris": {"role": "admin"}}})

    upgraded, _ = build_store({"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path)})
    assert upgraded.read("users")["users"]["chris"]["role"] == "admin"

    upgraded.write("users", {"users": {"chris": {"role": "instructor"}}})
    raw = (tmp_path / "users.enc").read_bytes()
    assert raw.lstrip()[:1] == b"{", "rewritten data should carry the new envelope"


def test_rotation_keeps_old_keys_readable_then_drops_them(keyring):
    old_blob = keyring.encrypt(PURPOSE_DOCUMENTS, b"before rotation")
    rotated = keyring.rotated(OTHER)

    # Mid-rotation: both keys work.
    assert rotated.decrypt(PURPOSE_DOCUMENTS, old_blob) == b"before rotation"
    new_blob = rotated.encrypt(PURPOSE_DOCUMENTS, b"after rotation")
    assert rotated.primary.key_version == keyring.primary.key_version + 1

    # After rotation completes, only the new secret opens the new data.
    final = Keyring(secret=OTHER, primary=rotated.primary, legacy=[])
    assert final.decrypt(PURPOSE_DOCUMENTS, new_blob) == b"after rotation"
    with pytest.raises(KeyError_):
        final.decrypt(PURPOSE_DOCUMENTS, old_blob)


# --------------------------------------------------------------------------- #
# 2. Envelope encryption
# --------------------------------------------------------------------------- #


def test_dek_wrap_and_unwrap():
    dek, salt = new_dek(), b"0123456789abcdef"
    wrapped = wrap_dek(dek, "correct-password", salt)
    assert wrapped != dek.hex()
    assert unwrap_dek(wrapped, "correct-password", salt) == dek
    assert unwrap_dek(wrapped, "wrong-password", salt) is None
    assert unwrap_dek("", "correct-password", salt) is None


def test_kek_depends_on_both_password_and_salt():
    assert derive_kek("p", b"a" * 16) != derive_kek("p", b"b" * 16)
    assert derive_kek("p", b"a" * 16) != derive_kek("q", b"a" * 16)


def test_dek_encryption_round_trip():
    dek = new_dek()
    token = encrypt_with_dek(dek, "sk-or-v1-personal")
    assert "personal" not in token
    assert decrypt_with_dek(dek, token) == "sk-or-v1-personal"
    assert decrypt_with_dek(new_dek(), token) == "", "another key must not open it"


def test_app_secret_cannot_read_a_personal_key(directory):
    """The property the whole design exists for."""
    session = directory.bootstrap_admin("chris", "first-password")
    directory.save_api_key("chris", "openrouter", "sk-or-v1-PERSONAL", session.dek)

    user = directory.get("chris")
    token = user.api_keys["openrouter"]

    # The app key — the thing an APP_SECRET leak would hand an attacker.
    assert directory.cipher.decrypt_text(token) == ""
    assert directory.get_api_key(user, "openrouter", None) == ""
    # Only the session that signed in can read it.
    assert directory.get_api_key(user, "openrouter", session.dek) == "sk-or-v1-PERSONAL"


def test_one_users_data_key_does_not_open_anothers(directory):
    admin = directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    jsmith = directory.authenticate("jsmith", "temp-password-1")

    directory.save_api_key("jsmith", "openrouter", "sk-or-v1-THEIRS", jsmith.dek)
    user = directory.get("jsmith")
    assert directory.get_api_key(user, "openrouter", admin.dek) == ""
    assert directory.get_api_key(user, "openrouter", jsmith.dek) == "sk-or-v1-THEIRS"


def test_changing_your_own_password_keeps_your_keys(directory):
    session = directory.bootstrap_admin("chris", "first-password")
    directory.save_api_key("chris", "openrouter", "sk-or-v1-PERSONAL", session.dek)

    directory.set_password("chris", "a-new-password", current_dek=session.dek, actor="chris")
    after = directory.authenticate("chris", "a-new-password")
    assert directory.get_api_key(after.user, "openrouter", after.dek) == "sk-or-v1-PERSONAL"


def test_an_admin_reset_destroys_the_users_personal_keys(directory):
    """The deliberate cost of the guarantee, asserted so it is never quietly
    'fixed' by adding a recovery path."""
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    jsmith = directory.authenticate("jsmith", "temp-password-1")
    directory.save_api_key("jsmith", "openrouter", "sk-or-v1-THEIRS", jsmith.dek)

    directory.set_password("jsmith", "reset-by-admin", clear_flag=False, actor="chris")

    after = directory.authenticate("jsmith", "reset-by-admin")
    assert after.user.api_keys == {}
    assert directory.get_api_key(after.user, "openrouter", after.dek) == ""


def test_an_admin_reset_leaves_issued_keys_working(directory):
    """Issued keys are the app's own, capped and revocable — they must survive."""
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-ISSUED", {"hash": "h1", "limit": 5.0}
    )
    directory.set_password("jsmith", "reset-by-admin", clear_flag=False, actor="chris")

    user = directory.get("jsmith")
    assert directory.get_issued_key(user, "openrouter") == "sk-or-v1-ISSUED"


def test_personal_keys_win_over_issued_ones(directory):
    session = directory.bootstrap_admin("chris", "first-password")
    directory.record_provisioned_key(
        "chris", "openrouter", "sk-or-v1-ISSUED", {"hash": "h1"}
    )
    directory.save_api_key("chris", "openrouter", "sk-or-v1-MINE", session.dek)

    user = directory.get("chris")
    assert directory.resolve_key(user, "openrouter", session.dek) == ("sk-or-v1-MINE", "personal")
    assert directory.resolve_key(user, "openrouter", None) == ("sk-or-v1-ISSUED", "issued")
    assert directory.resolve_key(user, "gemini", session.dek) == ("", "")


def test_accounts_from_before_envelope_encryption_get_a_data_key(directory):
    directory.bootstrap_admin("chris", "first-password")
    user = directory.get("chris")
    user.wrapped_dek = ""
    user.dek_salt = ""
    directory._save_user(user)

    session = directory.authenticate("chris", "first-password")
    assert session.dek is not None
    directory.save_api_key("chris", "openrouter", "sk-later", session.dek)
    assert directory.get_api_key(directory.get("chris"), "openrouter", session.dek) == "sk-later"


def test_self_service_revocation_clears_every_personal_key(directory):
    session = directory.bootstrap_admin("chris", "first-password")
    directory.save_api_key("chris", "openrouter", "sk-1", session.dek)
    directory.save_api_key("chris", "gemini", "sk-2", session.dek)

    assert directory.clear_all_api_keys("chris", actor="chris") == 2
    assert directory.get("chris").api_keys == {}


def test_mask_key_shows_existence_not_content():
    assert mask_key("sk-or-v1-abcdefgh") == "…efgh"
    assert mask_key("abc") == "…"
    assert mask_key("") == ""


# --------------------------------------------------------------------------- #
# 4. Per-user documents and rollback
# --------------------------------------------------------------------------- #


def test_accounts_live_in_separate_documents(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")

    paths = directory.store.list_paths("users")
    assert "users/u/chris" in paths and "users/u/jsmith" in paths
    # Saving one account must not rewrite the other's document.
    before = directory.store.read("users/u/chris")["_v"]
    directory.save_settings("jsmith", {"llm_model": "x"})
    assert directory.store.read("users/u/chris")["_v"] == before


def test_a_legacy_single_document_store_is_migrated():
    store = GuardedStore(MemoryStore())
    cipher = Cipher(SECRET)
    store.write(
        "users",
        {
            "users": {
                "chris": {
                    "username": "chris", "role": "admin",
                    "api_keys": {"openrouter": cipher.encrypt_text("sk-old")},
                }
            }
        },
    )
    directory = UserDirectory(store, cipher)

    assert [u.username for u in directory.all_users()] == ["chris"]
    user = directory.get("chris")
    # Keys encrypted under the app key move to issued_keys, where app-key
    # material now belongs — nothing is lost, nothing is mislabelled.
    assert directory.get_issued_key(user, "openrouter") == "sk-old"
    assert user.api_keys == {}


def test_rollback_is_detected():
    inner = MemoryStore()
    store = GuardedStore(inner)
    store.write("users/u/chris", {"active": True})
    snapshot = inner.read("users/u/chris")

    store.write("users/u/chris", {"active": False})
    inner.write("users/u/chris", snapshot)  # someone restores the older file

    with pytest.raises(RollbackError) as exc:
        store.read("users/u/chris")
    assert "older copy" in str(exc.value)


def test_versions_increase_and_normal_reads_pass():
    store = GuardedStore(MemoryStore())
    store.write("k", {"a": 1})
    store.write("k", {"a": 2})
    assert store.read("k")["_v"] == 2
    assert store.read("k")["a"] == 2


def test_the_watermark_document_is_not_itself_guarded():
    store = GuardedStore(MemoryStore())
    store.write("k", {"a": 1})
    assert store.read("_watermarks") is not None
    assert "_watermarks" not in store.list_paths()


# --------------------------------------------------------------------------- #
# 5. Lockout, audit
# --------------------------------------------------------------------------- #


def test_repeated_failures_lock_the_account(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")

    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthError):
            directory.authenticate("jsmith", "wrong")

    assert directory.get("jsmith").is_locked
    with pytest.raises(AuthError) as exc:
        directory.authenticate("jsmith", "temp-password-1")
    assert "Try again in" in str(exc.value), "the correct password is refused while locked"


def test_a_successful_sign_in_clears_the_failure_count(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")

    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(AuthError):
            directory.authenticate("jsmith", "wrong")
    assert directory.get("jsmith").failed_attempts == MAX_FAILED_ATTEMPTS - 1

    directory.authenticate("jsmith", "temp-password-1")
    assert directory.get("jsmith").failed_attempts == 0
    assert not directory.get("jsmith").is_locked


def test_an_admin_can_unlock(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthError):
            directory.authenticate("jsmith", "wrong")

    directory.unlock("jsmith", actor="chris")
    assert directory.authenticate("jsmith", "temp-password-1").user.username == "jsmith"


def test_a_lock_expires(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    user = directory.get("jsmith")
    user.locked_until = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1)
    ).isoformat(timespec="seconds")
    directory._save_user(user)

    assert not directory.get("jsmith").is_locked
    assert directory.authenticate("jsmith", "temp-password-1").user.username == "jsmith"


def test_lock_duration_is_reported(directory):
    directory.bootstrap_admin("chris", "first-password")
    user = directory.get("chris")
    user.locked_until = (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=LOCKOUT_MINUTES)
    ).isoformat(timespec="seconds")
    assert 1 <= user.lock_minutes_left <= LOCKOUT_MINUTES + 1


def test_unknown_usernames_are_not_enumerable(directory):
    directory.bootstrap_admin("chris", "first-password")
    with pytest.raises(AuthError) as wrong:
        directory.authenticate("chris", "nope")
    with pytest.raises(AuthError) as ghost:
        directory.authenticate("nobody", "nope")
    assert str(wrong.value) == str(ghost.value)


def test_audit_records_events_without_recording_secrets(directory):
    session = directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1", actor="chris")
    directory.save_api_key("chris", "openrouter", "sk-or-v1-SUPERSECRET", session.dek)
    directory.record_provisioned_key(
        "jsmith", "openrouter", "sk-or-v1-ISSUEDSECRET", {"hash": "h1", "limit": 5.0},
        actor="chris",
    )

    events = directory.audit.events()
    names = {e.event for e in events}
    assert {"user.create", "key.saved", "key.issued", "login.success"} <= names

    dump = str([e.__dict__ for e in events])
    assert "SUPERSECRET" not in dump and "ISSUEDSECRET" not in dump
    assert "first-password" not in dump and "temp-password-1" not in dump


def test_audit_records_failures_and_lockouts(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1")
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthError):
            directory.authenticate("jsmith", "wrong")

    names = [e.event for e in directory.audit.events()]
    assert "login.failure" in names and "login.locked" in names


def test_audit_can_be_filtered_by_person(directory):
    directory.bootstrap_admin("chris", "first-password")
    directory.create_user("jsmith", "temp-password-1", actor="chris")
    assert all(
        "jsmith" in (e.actor, e.subject)
        for e in directory.audit.events(subject="jsmith")
    )


def test_audit_is_bounded(monkeypatch):
    store = GuardedStore(MemoryStore())
    log = AuditLog(store)
    monkeypatch.setattr(audit_module, "MAX_EVENTS", 5)
    for i in range(9):
        log.record("login.success", f"user{i}")
    assert len(log.events(limit=100)) == 5


def test_audit_failure_never_blocks_the_operation():
    class Broken(MemoryStore):
        def write(self, path, payload):
            raise RuntimeError("storage is down")

    log = AuditLog(Broken())
    log.record("login.success", "chris")  # must not raise


# --------------------------------------------------------------------------- #
# Config credentials
# --------------------------------------------------------------------------- #


def test_the_management_key_uses_its_own_subkey():
    store = GuardedStore(MemoryStore())
    cipher = Cipher(SECRET)
    config = AppConfig(store, cipher)
    config.set_secret("openrouter_management_key", "sk-or-v1-MGMT")

    token = store.read("config")["values"]["secret:openrouter_management_key"]
    assert config.get_secret("openrouter_management_key") == "sk-or-v1-MGMT"
    # The API-key subkey must not open a config credential.
    assert cipher.decrypt_text(token) == ""


def test_a_fernet_key_is_never_reused_across_purposes(keyring):
    keys = {
        purpose: keyring._fernet(keyring.primary, purpose)._signing_key
        for purpose in (PURPOSE_DOCUMENTS, PURPOSE_APIKEYS, PURPOSE_CONFIG)
    }
    assert len(set(keys.values())) == 3


def test_legacy_spec_does_not_use_hkdf():
    """Legacy blobs must be read exactly the way they were written."""
    spec = legacy_spec()
    assert spec.use_hkdf is False and spec.kdf == "pbkdf2"
    assert new_spec().use_hkdf is True
