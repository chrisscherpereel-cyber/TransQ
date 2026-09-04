"""Encrypted persistence, with a local backend and a Dropbox backend.

Everything written here — accounts, saved settings, API keys, the usage ledger —
is serialized to JSON, encrypted, and stored as an opaque blob. Key derivation,
purpose separation and rotation live in :mod:`src.keymgmt`; this module is about
*where* blobs go and *what guarantees* surround them.

Two guarantees are worth naming:

**Atomic writes.** Files are written to a temporary path and renamed, so a crash
mid-write cannot leave a truncated file where the account database used to be.

**Rollback detection.** Every document carries a monotonically increasing version,
and the highest version seen for each path is recorded separately. Restoring an
old copy — an accidental Dropbox "restore previous version", or a deliberate one
to re-enable a disabled account — is refused rather than silently accepted. An
attacker who can roll back *both* the document and the watermark defeats this;
it is a tripwire, not a vault, and it catches the realistic case.

**A word on Dropbox.** It is file storage, not a database. There are no
transactions and no row locking. The Dropbox backend uses revision-checked writes
so a collision is *detected* and retried rather than silently swallowed, and
accounts are stored one document per user so two people saving at once no longer
touch the same file at all. That is enough for a handful of instructors sharing a
deployment. It is not enough for fifty concurrent users — at that point the right
answer is a real database, and ``Store`` is deliberately a small interface so
swapping one in means writing one class, not rewriting the app.
"""

from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Any

from .keymgmt import (
    KEYRING_PATH,
    PURPOSE_APIKEYS,
    PURPOSE_CONFIG,
    PURPOSE_DOCUMENTS,
    Keyring,
    KeyError_,
    load_or_create_keyring,
)

WATERMARK_PATH = "_watermarks"
VERSION_FIELD = "_v"


class StorageError(RuntimeError):
    pass


class RollbackError(StorageError):
    """A document came back older than a version we have already seen."""


# --------------------------------------------------------------------------- #
# Encryption facade
# --------------------------------------------------------------------------- #


class Cipher:
    """Encrypt and decrypt payloads, with purpose-separated subkeys.

    Kept as a thin facade because the rest of the app talks to *this*, not to the
    keyring: adding purposes or rotating keys does not ripple outward.
    """

    def __init__(self, secret: str, keyring: Keyring | None = None):
        if keyring is not None:
            self.keyring = keyring
        else:
            self.keyring, _ = load_or_create_keyring(secret, None)

    # -- documents -- #

    def encrypt(self, payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        return self.keyring.encrypt(PURPOSE_DOCUMENTS, raw)

    def decrypt(self, blob: bytes) -> dict[str, Any]:
        try:
            return json.loads(self.keyring.decrypt(PURPOSE_DOCUMENTS, blob).decode("utf-8"))
        except KeyError_ as exc:
            raise StorageError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise StorageError(f"Stored data is corrupt: {exc}") from exc

    # -- individual secrets -- #

    def encrypt_text(self, text: str) -> str:
        """For app-owned credentials held inside an already-encrypted file.

        Uses a different subkey from the surrounding document, so this is now a
        real second layer rather than the decorative one it used to be.
        """
        return self.keyring.encrypt_text(PURPOSE_APIKEYS, text)

    def decrypt_text(self, token: str) -> str:
        return self.keyring.decrypt_text(PURPOSE_APIKEYS, token)

    def encrypt_config(self, text: str) -> str:
        return self.keyring.encrypt_text(PURPOSE_CONFIG, text)

    def decrypt_config(self, token: str) -> str:
        return self.keyring.decrypt_text(PURPOSE_CONFIG, token)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class Store(ABC):
    """Read and write encrypted documents, plus a plaintext side channel."""

    name = "store"

    @abstractmethod
    def read(self, path: str) -> dict[str, Any] | None:
        """Return the decoded document, or None if it does not exist."""

    @abstractmethod
    def write(self, path: str, payload: dict[str, Any]) -> None:
        """Create or replace the document."""

    @abstractmethod
    def read_plain(self, path: str) -> dict[str, Any] | None:
        """Read an unencrypted document — the keyring header lives here."""

    @abstractmethod
    def write_plain(self, path: str, payload: dict[str, Any]) -> None:
        """Write an unencrypted document. Never used for anything secret."""

    @abstractmethod
    def available(self) -> bool:
        """Is the backend reachable and configured?"""

    def list_paths(self, prefix: str = "") -> list[str]:
        """Every document path under a prefix. Used by rotation and migration."""
        return []

    def describe(self) -> str:
        return self.name


class LocalStore(Store):
    """Encrypted files on disk.

    Fine on a machine you control. On Streamlit Community Cloud the container
    filesystem is wiped whenever the app restarts or sleeps, so accounts written
    here will not survive — use Dropbox there.
    """

    name = "local encrypted files"

    def __init__(self, cipher: Cipher, root: str = ".data"):
        self.cipher = cipher
        self.root = root
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

    def _path(self, path: str, suffix: str = ".enc") -> str:
        safe = "/".join(
            part for part in path.strip("/").split("/") if part not in ("", ".", "..")
        )
        full = os.path.join(self.root, safe + suffix)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        return full

    def _write_bytes(self, full: str, blob: bytes) -> None:
        with self._lock:
            temporary = full + ".tmp"
            with open(temporary, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, full)

    def read(self, path: str) -> dict[str, Any] | None:
        full = self._path(path)
        if not os.path.exists(full):
            return None
        with open(full, "rb") as handle:
            return self.cipher.decrypt(handle.read())

    def write(self, path: str, payload: dict[str, Any]) -> None:
        self._write_bytes(self._path(path), self.cipher.encrypt(payload))

    def read_plain(self, path: str) -> dict[str, Any] | None:
        full = self._path(path, ".json")
        if not os.path.exists(full):
            return None
        try:
            with open(full, "rb") as handle:
                return json.loads(handle.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StorageError(f"{full} is not readable: {exc}") from exc

    def write_plain(self, path: str, payload: dict[str, Any]) -> None:
        blob = json.dumps(payload, indent=2).encode("utf-8")
        self._write_bytes(self._path(path, ".json"), blob)

    def list_paths(self, prefix: str = "") -> list[str]:
        found: list[str] = []
        base = os.path.join(self.root, prefix.strip("/")) if prefix else self.root
        for directory, _, files in os.walk(base):
            for name in files:
                if not name.endswith(".enc"):
                    continue
                full = os.path.join(directory, name)
                found.append(os.path.relpath(full, self.root)[: -len(".enc")])
        return sorted(found)

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return f"local encrypted files ({os.path.abspath(self.root)})"


class DropboxStore(Store):
    """Encrypted blobs in a Dropbox app folder.

    Uses a refresh token rather than a short-lived access token, so the app keeps
    working past the four-hour expiry without anyone re-authorizing it.
    """

    name = "Dropbox"

    def __init__(
        self,
        cipher: Cipher,
        app_key: str,
        app_secret: str,
        refresh_token: str,
        folder: str = "/lecture-quiz-builder",
    ):
        self.cipher = cipher
        self.folder = "/" + folder.strip("/")
        self._lock = threading.Lock()
        self._revisions: dict[str, str] = {}
        try:
            import dropbox
        except ImportError as exc:  # pragma: no cover
            raise StorageError("pip install dropbox") from exc

        self._dropbox = dropbox
        self._client = dropbox.Dropbox(
            app_key=app_key,
            app_secret=app_secret,
            oauth2_refresh_token=refresh_token,
            timeout=30,
        )

    def _path(self, path: str, suffix: str = ".enc") -> str:
        return f"{self.folder}/{path.strip('/')}{suffix}"

    def _download(self, remote: str) -> bytes | None:
        try:
            metadata, response = self._client.files_download(remote)
        except self._dropbox.exceptions.ApiError as exc:
            if _is_not_found(exc):
                return None
            raise StorageError(f"Dropbox read failed: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"Dropbox is unreachable: {exc}") from exc
        self._revisions[remote] = metadata.rev
        return response.content

    def _upload(self, remote: str, blob: bytes) -> None:
        known_rev = self._revisions.get(remote)
        with self._lock:
            try:
                mode = (
                    self._dropbox.files.WriteMode.update(known_rev)
                    if known_rev
                    else self._dropbox.files.WriteMode.overwrite
                )
                metadata = self._client.files_upload(blob, remote, mode=mode, mute=True)
            except self._dropbox.exceptions.ApiError as exc:
                if known_rev and _is_conflict(exc):
                    # Somebody else saved first. Take their revision and retry.
                    self._revisions.pop(remote, None)
                    metadata = self._client.files_upload(
                        blob, remote,
                        mode=self._dropbox.files.WriteMode.overwrite, mute=True,
                    )
                else:
                    raise StorageError(f"Dropbox write failed: {exc}") from exc
            except Exception as exc:
                raise StorageError(f"Dropbox is unreachable: {exc}") from exc
            self._revisions[remote] = metadata.rev

    def read(self, path: str) -> dict[str, Any] | None:
        content = self._download(self._path(path))
        return self.cipher.decrypt(content) if content is not None else None

    def write(self, path: str, payload: dict[str, Any]) -> None:
        self._upload(self._path(path), self.cipher.encrypt(payload))

    def read_plain(self, path: str) -> dict[str, Any] | None:
        content = self._download(self._path(path, ".json"))
        if content is None:
            return None
        try:
            return json.loads(content.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StorageError(f"{path} is not readable: {exc}") from exc

    def write_plain(self, path: str, payload: dict[str, Any]) -> None:
        self._upload(self._path(path, ".json"), json.dumps(payload, indent=2).encode("utf-8"))

    def list_paths(self, prefix: str = "") -> list[str]:
        folder = f"{self.folder}/{prefix.strip('/')}" if prefix else self.folder
        try:
            result = self._client.files_list_folder(folder, recursive=True)
        except Exception:
            return []
        found: list[str] = []
        while True:
            for entry in result.entries:
                name = getattr(entry, "path_lower", "")
                if name.endswith(".enc"):
                    found.append(name[len(self.folder) + 1 : -len(".enc")])
            if not result.has_more:
                break
            result = self._client.files_list_folder_continue(result.cursor)
        return sorted(found)

    def available(self) -> bool:
        try:
            self._client.users_get_current_account()
            return True
        except Exception:
            return False

    def describe(self) -> str:
        return f"Dropbox app folder ({self.folder})"


def _is_not_found(exc: Exception) -> bool:
    return "not_found" in str(exc).lower()


def _is_conflict(exc: Exception) -> bool:
    return "conflict" in str(exc).lower()


class MemoryStore(Store):
    """In-process only. Used by tests, and as a last resort so the app still
    runs (without remembering anything) when no secret is configured."""

    name = "in-memory (nothing is saved)"

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._plain: dict[str, dict[str, Any]] = {}

    def read(self, path: str) -> dict[str, Any] | None:
        value = self._data.get(path)
        return json.loads(json.dumps(value)) if value is not None else None

    def write(self, path: str, payload: dict[str, Any]) -> None:
        self._data[path] = json.loads(json.dumps(payload))

    def read_plain(self, path: str) -> dict[str, Any] | None:
        value = self._plain.get(path)
        return json.loads(json.dumps(value)) if value is not None else None

    def write_plain(self, path: str, payload: dict[str, Any]) -> None:
        self._plain[path] = json.loads(json.dumps(payload))

    def list_paths(self, prefix: str = "") -> list[str]:
        return sorted(p for p in self._data if p.startswith(prefix))

    def available(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Rollback protection
# --------------------------------------------------------------------------- #


class GuardedStore(Store):
    """Wraps a backend and refuses documents older than one already seen.

    Fernet tells you a blob is authentic. It cannot tell you it is *current* —
    an old ciphertext is perfectly valid forever. That gap is what a rollback
    exploits: restore last week's account file and a disabled account works
    again, or a changed password reverts. A monotonic version per document,
    recorded outside it, closes the accidental and partial cases.
    """

    def __init__(self, inner: Store):
        self.inner = inner
        self.name = inner.name
        self._lock = threading.Lock()

    # -- watermarks -- #

    def _watermarks(self) -> dict[str, int]:
        document = self.inner.read(WATERMARK_PATH) or {}
        marks = document.get("marks")
        return {str(k): int(v) for k, v in marks.items()} if isinstance(marks, dict) else {}

    def _remember(self, path: str, version: int) -> None:
        marks = self._watermarks()
        if version > marks.get(path, 0):
            marks[path] = version
            self.inner.write(WATERMARK_PATH, {"version": 1, "marks": marks})

    # -- Store -- #

    def read(self, path: str) -> dict[str, Any] | None:
        document = self.inner.read(path)
        if document is None or path == WATERMARK_PATH:
            return document

        seen = self._watermarks().get(path, 0)
        version = int(document.get(VERSION_FIELD, 0) or 0)
        if seen and version < seen:
            raise RollbackError(
                f"'{path}' came back at version {version} but version {seen} was "
                "already seen. An older copy has been restored over the current "
                "one. Recover the newer file, or delete the watermark record if "
                "the rollback was intentional."
            )
        return document

    def write(self, path: str, payload: dict[str, Any]) -> None:
        if path == WATERMARK_PATH:
            self.inner.write(path, payload)
            return
        with self._lock:
            current = self._watermarks().get(path, 0)
            existing = self.inner.read(path) or {}
            version = max(current, int(existing.get(VERSION_FIELD, 0) or 0)) + 1
            self.inner.write(path, {**payload, VERSION_FIELD: version})
            self._remember(path, version)

    def read_plain(self, path: str) -> dict[str, Any] | None:
        return self.inner.read_plain(path)

    def write_plain(self, path: str, payload: dict[str, Any]) -> None:
        self.inner.write_plain(path, payload)

    def list_paths(self, prefix: str = "") -> list[str]:
        return [p for p in self.inner.list_paths(prefix) if p != WATERMARK_PATH]

    def available(self) -> bool:
        return self.inner.available()

    def describe(self) -> str:
        return self.inner.describe()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_store(secrets: dict[str, str]) -> tuple[Store, str | None]:
    """Pick a backend from configuration. Returns ``(store, warning)``.

    Dropbox wins when its credentials are present, because that is the only
    option that survives a Streamlit Cloud restart. Otherwise encrypted local
    files. With no ``APP_SECRET`` at all, an in-memory store keeps the app usable
    for a single session and the UI says clearly that nothing is being saved.
    """
    secret = secrets.get("APP_SECRET", "")
    if not secret:
        return MemoryStore(), (
            "No APP_SECRET is set, so accounts and settings are not being saved — "
            "they will vanish when this app restarts. See the README to set one."
        )

    # The keyring header is plaintext (a salt is not a secret) and has to be read
    # before anything else, so it comes from a bootstrap backend with no cipher.
    bootstrap = _bootstrap_backend(secrets)
    try:
        header = bootstrap.read_plain(KEYRING_PATH) if bootstrap else None
    except StorageError:
        header = None

    try:
        keyring, created = load_or_create_keyring(secret, header)
    except KeyError_ as exc:
        return MemoryStore(), f"Encryption could not start ({exc}). Nothing is being saved."

    cipher = Cipher(secret, keyring)
    backend, warning = _make_backend(secrets, cipher)

    if created:
        try:
            backend.write_plain(KEYRING_PATH, keyring.to_dict())
        except StorageError:
            pass  # a missing header only means the next start recreates it

    return GuardedStore(backend), warning


def _bootstrap_backend(secrets: dict[str, str]) -> Store | None:
    """A backend used only to read the plaintext keyring header."""
    try:
        placeholder = Cipher.__new__(Cipher)  # never used for crypto here
        backend, _ = _make_backend(secrets, placeholder)  # type: ignore[arg-type]
        return backend
    except Exception:
        return None


def _make_backend(secrets: dict[str, str], cipher: Cipher) -> tuple[Store, str | None]:
    app_key = secrets.get("DROPBOX_APP_KEY", "")
    app_secret = secrets.get("DROPBOX_APP_SECRET", "")
    refresh_token = secrets.get("DROPBOX_REFRESH_TOKEN", "")
    data_dir = secrets.get("DATA_DIR") or ".data"

    if app_key and app_secret and refresh_token:
        try:
            store = DropboxStore(
                cipher, app_key, app_secret, refresh_token,
                secrets.get("DROPBOX_FOLDER") or "/lecture-quiz-builder",
            )
            if store.available():
                return store, None
            return LocalStore(cipher, data_dir), (
                "Dropbox credentials are set but Dropbox rejected them. Falling back "
                "to local files, which do NOT survive a restart on Streamlit Cloud."
            )
        except StorageError as exc:
            return LocalStore(cipher, data_dir), (
                f"Dropbox could not be initialized ({exc}). Falling back to local files."
            )

    return LocalStore(cipher, data_dir), None


# Backwards-compatible name: earlier versions imported this from here.
def derive_key(secret: str) -> bytes:  # pragma: no cover - compatibility shim
    from .keymgmt import legacy_spec

    if not secret:
        raise StorageError("APP_SECRET is not set; encrypted storage cannot start.")
    import base64

    return base64.urlsafe_b64encode(legacy_spec().derive(secret))
