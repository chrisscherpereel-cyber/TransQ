"""The Dropbox backend, against a stubbed SDK — no network, no account.

Two failures motivate this file, both of the same shape: **a misconfiguration
that leaves the app apparently working.** Dropbox never crashes the app; when it
does not work the app falls back to local files, which behave identically until a
Streamlit Cloud restart wipes them. So the health check and the error messages
are load-bearing, and they need tests.

1. ``available()`` once called ``users_get_current_account()``, which needs the
   ``account_info.read`` scope the app never otherwise uses. A correctly-scoped
   deployment failed its own health check and was silently downgraded.
2. When it does fail, the message has to name the fix. "Dropbox connection
   failed" sends an administrator to re-check credentials that are fine.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.keymgmt import load_or_create_keyring
from src.storage import Cipher, DropboxStore, StorageError, _explain_dropbox

SECRET = "a-long-random-app-secret-for-tests"


# --------------------------------------------------------------------------- #
# A stub Dropbox SDK
# --------------------------------------------------------------------------- #


class StubApiError(Exception):
    pass


class StubMetadata:
    def __init__(self, rev: str = "rev1"):
        self.rev = rev


class StubListing:
    def __init__(self) -> None:
        self.entries: list = []
        self.has_more = False
        self.cursor = ""


class StubClient:
    """Records what the app asked Dropbox to do, and can be told to refuse."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.files: dict[str, bytes] = {}
        self.list_error: Exception | None = None

    def files_list_folder(self, path, **kwargs):
        self.calls.append("files_list_folder")
        if self.list_error is not None:
            raise self.list_error
        return StubListing()

    def files_download(self, remote):
        self.calls.append("files_download")
        if remote not in self.files:
            raise StubApiError("path/not_found/")
        response = types.SimpleNamespace(content=self.files[remote])
        return StubMetadata(), response

    def files_upload(self, blob, remote, mode=None, mute=False):
        self.calls.append("files_upload")
        self.files[remote] = blob
        return StubMetadata()

    def users_get_current_account(self):  # pragma: no cover - must never run
        self.calls.append("users_get_current_account")
        raise AssertionError(
            "available() must not need the account_info.read scope"
        )


def install_stub_sdk(monkeypatch) -> type[StubClient]:
    module = types.ModuleType("dropbox")
    module.Dropbox = StubClient
    module.exceptions = types.SimpleNamespace(ApiError=StubApiError)

    class WriteMode:
        @staticmethod
        def update(rev):
            return ("update", rev)

        overwrite = ("overwrite", None)

    module.files = types.SimpleNamespace(WriteMode=WriteMode)
    monkeypatch.setitem(sys.modules, "dropbox", module)
    return StubClient


@pytest.fixture
def store(monkeypatch) -> DropboxStore:
    install_stub_sdk(monkeypatch)
    keyring, _ = load_or_create_keyring(SECRET, None)
    return DropboxStore(Cipher(SECRET, keyring), "key", "secret", "refresh")


# --------------------------------------------------------------------------- #
# The health check
# --------------------------------------------------------------------------- #


def test_the_health_check_uses_a_scope_the_app_actually_needs(store):
    """The regression: probing with an endpoint outside the app's own scopes."""
    assert store.available() is True
    assert store._client.calls == ["files_list_folder"]
    assert "users_get_current_account" not in store._client.calls


def test_an_empty_app_folder_is_healthy(store):
    """A brand-new app folder does not exist until something is written to it."""
    store._client.list_error = StubApiError("path/not_found/...")
    assert store.available() is True
    assert store.last_error == ""


def test_a_real_failure_is_reported_not_swallowed(store):
    store._client.list_error = StubApiError("missing_scope/files.metadata.read")
    assert store.available() is False
    assert store.last_error


def test_a_recovered_connection_clears_the_stale_error(store):
    store._client.list_error = StubApiError("invalid_grant")
    store.available()
    assert store.last_error

    store._client.list_error = None
    assert store.available() is True
    assert store.last_error == "", "a stale error would misdiagnose the next failure"


def test_credentials_are_passed_to_the_sdk_as_a_refresh_token(store):
    """Not an access token — those expire in four hours and the app dies quietly."""
    assert store._client.kwargs["oauth2_refresh_token"] == "refresh"
    assert store._client.kwargs["app_key"] == "key"


# --------------------------------------------------------------------------- #
# Error messages have to name the fix
# --------------------------------------------------------------------------- #


def test_a_missing_scope_explains_the_ordering_trap():
    """Ticking permissions does not upgrade a token that already exists."""
    message = _explain_dropbox(Exception("missing_scope/files.content.write"))
    assert "files.content.write" in message
    assert "NEW refresh token" in message
    assert "Submit" in message


def test_a_rejected_token_points_at_the_credentials():
    message = _explain_dropbox(Exception("invalid_grant"))
    assert "refresh token" in message.lower()
    assert "spaces" in message


def test_an_expired_token_says_to_regenerate():
    assert "new refresh token" in _explain_dropbox(
        Exception("expired_access_token")
    ).lower()


def test_an_unrecognised_failure_still_says_something_and_stays_short():
    message = _explain_dropbox(Exception("x" * 500))
    assert message and len(message) <= 200


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


def test_documents_round_trip_and_are_encrypted_on_dropbox(store):
    store.write("library/chris/t/abc", {"transcript": "Aggregate planning matters"})

    blob = next(iter(store._client.files.values()))
    assert b"Aggregate planning" not in blob, "plaintext must never reach Dropbox"
    assert store.read("library/chris/t/abc")["transcript"] == (
        "Aggregate planning matters"
    )


def test_a_missing_document_reads_as_none_rather_than_raising(store):
    assert store.read("users/u/nobody") is None


def test_the_keyring_header_is_stored_as_readable_json(store):
    """A salt is not a secret, and it must be readable before any key exists."""
    store.write_plain("keyring", {"kdf": "scrypt", "salt": "abc"})
    blob = store._client.files["/lecture-quiz-builder/keyring.json"]
    assert b"scrypt" in blob
    assert store.read_plain("keyring")["salt"] == "abc"


def test_paths_land_inside_the_configured_folder(monkeypatch):
    install_stub_sdk(monkeypatch)
    keyring, _ = load_or_create_keyring(SECRET, None)
    store = DropboxStore(
        Cipher(SECRET, keyring), "k", "s", "r", folder="mgt301"
    )
    assert store._path("users/index") == "/mgt301/users/index.enc"


def test_an_unreachable_dropbox_raises_storage_error_not_a_dropbox_exception(store):
    """Callers catch StorageError; a raw SDK exception would escape the app."""
    def boom(*args, **kwargs):
        raise ConnectionError("network is down")

    store._client.files_download = boom
    with pytest.raises(StorageError):
        store.read("users/index")


# --------------------------------------------------------------------------- #
# Choosing a backend, and saying why
# --------------------------------------------------------------------------- #
#
# The reported symptom: a sidebar reading "local encrypted files" on a
# deployment the user believed was configured for Dropbox. The old selection
# logic required all three credentials and, if any were absent, fell through to
# local files returning **no warning at all** — so a single misspelled key name
# produced an app that looked healthy and threw everything away on restart.


from src.storage import build_store, dropbox_credentials  # noqa: E402

FULL = {
    "DROPBOX_APP_KEY": "k",
    "DROPBOX_APP_SECRET": "s",
    "DROPBOX_REFRESH_TOKEN": "r",
}


def base_secrets(tmp_path, **extra):
    return {"APP_SECRET": SECRET, "DATA_DIR": str(tmp_path), **extra}


def test_credentials_are_counted_as_present_or_missing():
    present, missing = dropbox_credentials(FULL)
    assert present == list(FULL) and missing == []

    present, missing = dropbox_credentials({"DROPBOX_APP_KEY": "k"})
    assert present == ["DROPBOX_APP_KEY"]
    assert "DROPBOX_REFRESH_TOKEN" in missing


def test_a_blank_placeholder_counts_as_missing():
    """`DROPBOX_APP_KEY = ""` left in a secrets file is not configuration."""
    _, missing = dropbox_credentials({**FULL, "DROPBOX_APP_SECRET": "   "})
    assert missing == ["DROPBOX_APP_SECRET"]


def test_half_configured_dropbox_warns_and_names_what_is_missing(tmp_path):
    """The silent case that motivated this: some credentials, no complaint."""
    store, warning = build_store(
        base_secrets(tmp_path, DROPBOX_APP_KEY="k", DROPBOX_APP_SECRET="s")
    )
    assert warning is not None
    assert "DROPBOX_REFRESH_TOKEN" in warning
    assert "local" in store.name


def test_half_configured_warns_on_a_laptop_too(tmp_path, monkeypatch):
    """Unlike plain local storage, partial credentials are a mistake anywhere."""
    import src.storage as storage

    monkeypatch.setattr(storage, "is_ephemeral_host", lambda: False)
    _, warning = build_store(base_secrets(tmp_path, DROPBOX_APP_KEY="k"))
    assert warning is not None


def test_no_credentials_is_quiet_on_a_laptop(tmp_path, monkeypatch):
    """Local files are the right answer on a machine you control; a standing
    warning there only teaches people to ignore warnings."""
    import src.storage as storage

    monkeypatch.setattr(storage, "is_ephemeral_host", lambda: False)
    store, warning = build_store(base_secrets(tmp_path))
    assert warning is None
    assert store.fallback_reason, "the reason is still available on request"


def test_no_credentials_warns_loudly_on_an_ephemeral_host(tmp_path, monkeypatch):
    import src.storage as storage

    monkeypatch.setattr(storage, "is_ephemeral_host", lambda: True)
    _, warning = build_store(base_secrets(tmp_path))
    assert warning is not None
    assert "restart" in warning


def test_a_failed_connection_reports_the_reason_and_how_to_test_it(
    tmp_path, monkeypatch
):
    install_stub_sdk(monkeypatch)
    monkeypatch.setattr(
        StubClient,
        "files_list_folder",
        lambda self, p, **kw: (_ for _ in ()).throw(
            StubApiError("missing_scope/files.content.write")
        ),
    )
    store, warning = build_store(base_secrets(tmp_path, **FULL))

    assert "local" in store.name
    assert "check_dropbox" in warning
    assert "Submit" in warning, "the actual fix, not just 'connection failed'"


def test_a_working_dropbox_produces_no_warning_and_no_reason(tmp_path, monkeypatch):
    install_stub_sdk(monkeypatch)
    store, warning = build_store(base_secrets(tmp_path, **FULL))

    assert store.name == "Dropbox"
    assert warning is None
    assert not store.fallback_reason


def test_the_fallback_reason_survives_the_guarded_wrapper(tmp_path):
    """build_store returns a GuardedStore; the sidebar reads the reason off it."""
    store, _ = build_store(base_secrets(tmp_path, DROPBOX_APP_KEY="k"))
    assert type(store).__name__ == "GuardedStore"
    assert "DROPBOX_APP_SECRET" in store.fallback_reason
