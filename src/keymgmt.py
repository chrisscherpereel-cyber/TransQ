"""Key derivation, purpose separation, and rotation.

This replaces the original scheme — PBKDF2 with a hardcoded salt, one derived key
used for everything, no version tag — with four properties it lacked:

**A random salt per deployment.** The old fixed salt made the derived key a pure
function of ``APP_SECRET``, identical in every copy of this app in the world. The
salt is not secret, so it lives in a plaintext ``keyring`` document beside the
data; what it buys is that precomputation against one deployment buys nothing
against another.

**A memory-hard KDF.** scrypt instead of PBKDF2. PBKDF2 parallelizes well on a
GPU, which is the wrong property for something guarding an account store.

**Distinct subkeys per purpose.** Documents, API keys and config are encrypted
under separate HKDF-derived subkeys. Previously the "second layer" of encryption
around API keys used the very same key as the file around them, so it protected
against nothing. Now compromising one context does not hand over the others.

**A version on every ciphertext.** Each blob records which key version wrote it,
and the keyring keeps old specs as read-only fallbacks. That is what makes
rotation possible at all: ``scripts/rotate_key.py`` can re-encrypt everything
under a new secret while the app keeps reading what has not moved yet. Without
it, changing ``APP_SECRET`` orphans the entire store — which in practice means it
never gets changed, including after an exposure.

Upgrading is automatic and lossless. A store written by the old scheme has no
keyring; on first use one is created with modern parameters, and the legacy
PBKDF2 spec is retained as a fallback reader so existing data still opens. Every
subsequent write uses the new key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

KEYRING_PATH = "keyring"
ENVELOPE_VERSION = 2

# Purposes get their own subkeys. Adding one is safe; renaming one is not — it
# would orphan everything written under the old name.
PURPOSE_DOCUMENTS = "documents"
PURPOSE_APIKEYS = "apikeys"
PURPOSE_CONFIG = "config"
PURPOSES = (PURPOSE_DOCUMENTS, PURPOSE_APIKEYS, PURPOSE_CONFIG)

# scrypt at n=2^15 costs ~32 MB and a fraction of a second. It runs once per
# process, not per request, so this is comfortably affordable.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1

LEGACY_SALT = b"lecture-quiz-builder/v1"
LEGACY_ITERATIONS = 200_000


class KeyError_(RuntimeError):
    """Raised when key material is missing, wrong, or unusable."""


@dataclass(frozen=True)
class KeySpec:
    """One way of turning ``APP_SECRET`` into key material."""

    key_version: int
    kdf: str                      # "scrypt" | "pbkdf2"
    salt: bytes
    iterations: int = LEGACY_ITERATIONS
    n: int = SCRYPT_N
    r: int = SCRYPT_R
    p: int = SCRYPT_P
    # The original scheme used the derived key directly, with no purpose
    # separation. Legacy blobs must be read exactly the way they were written.
    use_hkdf: bool = True

    def derive(self, secret: str) -> bytes:
        if not secret:
            raise KeyError_("APP_SECRET is not set; encrypted storage cannot start.")
        material = secret.encode("utf-8")
        if self.kdf == "scrypt":
            return hashlib.scrypt(
                material, salt=self.salt, n=self.n, r=self.r, p=self.p,
                dklen=32, maxmem=256 * 1024 * 1024,
            )
        if self.kdf == "pbkdf2":
            return hashlib.pbkdf2_hmac("sha256", material, self.salt, self.iterations)
        raise KeyError_(f"Unknown key derivation function: {self.kdf}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_version": self.key_version,
            "kdf": self.kdf,
            "salt": self.salt.hex(),
            "iterations": self.iterations,
            "n": self.n,
            "r": self.r,
            "p": self.p,
            "use_hkdf": self.use_hkdf,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeySpec":
        return cls(
            key_version=int(data.get("key_version", 1)),
            kdf=str(data.get("kdf", "scrypt")),
            salt=bytes.fromhex(str(data.get("salt", ""))),
            iterations=int(data.get("iterations", LEGACY_ITERATIONS)),
            n=int(data.get("n", SCRYPT_N)),
            r=int(data.get("r", SCRYPT_R)),
            p=int(data.get("p", SCRYPT_P)),
            use_hkdf=bool(data.get("use_hkdf", True)),
        )


def legacy_spec() -> KeySpec:
    """How the first version of this app derived its key."""
    return KeySpec(
        key_version=0, kdf="pbkdf2", salt=LEGACY_SALT,
        iterations=LEGACY_ITERATIONS, use_hkdf=False,
    )


def new_spec(key_version: int = 1) -> KeySpec:
    return KeySpec(key_version=key_version, kdf="scrypt", salt=os.urandom(16))


def _fernet_key(master: bytes, purpose: str, use_hkdf: bool) -> bytes:
    if not use_hkdf:
        return base64.urlsafe_b64encode(master)
    derived = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None,
        info=f"lecture-quiz-builder/{purpose}".encode("utf-8"),
    ).derive(master)
    return base64.urlsafe_b64encode(derived)


@dataclass
class Keyring:
    """The primary key for writing, plus older keys kept for reading."""

    secret: str
    primary: KeySpec
    legacy: list[KeySpec] = field(default_factory=list)
    _cache: dict[tuple[int, str], Fernet] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #

    def _fernet(self, spec: KeySpec, purpose: str) -> Fernet:
        cache_key = (spec.key_version, purpose)
        if cache_key not in self._cache:
            master = spec.derive(self.secret)
            self._cache[cache_key] = Fernet(_fernet_key(master, purpose, spec.use_hkdf))
        return self._cache[cache_key]

    def _readers(self, purpose: str) -> list[tuple[KeySpec, Fernet]]:
        specs = [self.primary, *self.legacy]
        return [(spec, self._fernet(spec, purpose)) for spec in specs]

    # ------------------------------------------------------------------ #

    def encrypt(self, purpose: str, plaintext: bytes) -> bytes:
        token = self._fernet(self.primary, purpose).encrypt(plaintext)
        envelope = {
            "env": ENVELOPE_VERSION,
            "kv": self.primary.key_version,
            "ct": token.decode("ascii"),
        }
        return json.dumps(envelope, separators=(",", ":")).encode("utf-8")

    def decrypt(self, purpose: str, blob: bytes) -> bytes:
        """Open a blob written by this or any earlier key version."""
        token, declared_version = _unwrap_envelope(blob)

        candidates = self._readers(purpose)
        if declared_version is not None:
            # Try the key it says wrote it first; still fall back, because a
            # half-finished rotation can leave either version on disk.
            candidates.sort(key=lambda pair: pair[0].key_version != declared_version)

        for _, fernet in candidates:
            try:
                return fernet.decrypt(token)
            except InvalidToken:
                continue

        raise KeyError_(
            "Stored data could not be decrypted with any known key. Either "
            "APP_SECRET has changed since it was written, or the keyring file is "
            "missing. Restore the original secret, or delete the store to start "
            "over (all accounts will be lost)."
        )

    def encrypt_text(self, purpose: str, text: str) -> str:
        return self.encrypt(purpose, text.encode("utf-8")).decode("utf-8")

    def decrypt_text(self, purpose: str, token: str) -> str:
        try:
            return self.decrypt(purpose, token.encode("utf-8")).decode("utf-8")
        except (KeyError_, ValueError, UnicodeDecodeError):
            return ""

    # ------------------------------------------------------------------ #

    def rotated(self, new_secret: str) -> "Keyring":
        """A keyring whose primary is a fresh key over ``new_secret``.

        The old specs are kept as readers so a rotation can proceed document by
        document rather than needing to be atomic across the whole store.
        """
        return MultiSecretKeyring(
            secret=new_secret,
            primary=new_spec(self.primary.key_version + 1),
            legacy=[self.primary, *self.legacy],
            legacy_secret=self.secret,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "primary": self.primary.to_dict(),
            "legacy": [spec.to_dict() for spec in self.legacy],
        }


def _unwrap_envelope(blob: bytes) -> tuple[bytes, int | None]:
    """Return ``(fernet_token, key_version)``.

    Version 1 of this app wrote bare Fernet tokens with no envelope. Those still
    have to open, so anything that is not our JSON envelope is treated as one.
    """
    stripped = blob.lstrip()
    if stripped[:1] == b"{":
        try:
            document = json.loads(stripped.decode("utf-8"))
            if isinstance(document, dict) and "ct" in document:
                return str(document["ct"]).encode("ascii"), document.get("kv")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return blob, None


class MultiSecretKeyring(Keyring):
    """A keyring whose legacy specs derive from a different secret.

    Only used during rotation, where the new key comes from the new secret and
    everything already on disk still needs the old one.
    """

    def __init__(self, secret: str, primary: KeySpec, legacy: list[KeySpec], legacy_secret: str):
        super().__init__(secret=secret, primary=primary, legacy=legacy)
        self.legacy_secret = legacy_secret

    def _fernet(self, spec: KeySpec, purpose: str) -> Fernet:  # type: ignore[override]
        cache_key = (spec.key_version, purpose)
        if cache_key not in self._cache:
            secret = self.secret if spec is self.primary else self.legacy_secret
            master = spec.derive(secret)
            self._cache[cache_key] = Fernet(_fernet_key(master, purpose, spec.use_hkdf))
        return self._cache[cache_key]


def load_or_create_keyring(secret: str, header: dict[str, Any] | None) -> tuple[Keyring, bool]:
    """Build the keyring from a stored header, creating one if absent.

    Returns ``(keyring, created)``. When ``created`` is True the caller should
    persist ``keyring.to_dict()`` — and note that the legacy PBKDF2 spec is
    included as a reader, which is what lets a store written by the old scheme
    keep working without a migration step.
    """
    if not secret:
        raise KeyError_("APP_SECRET is not set; encrypted storage cannot start.")

    if header:
        primary = KeySpec.from_dict(header.get("primary", {}))
        legacy = [KeySpec.from_dict(item) for item in header.get("legacy", []) or []]
        return Keyring(secret=secret, primary=primary, legacy=legacy), False

    return Keyring(secret=secret, primary=new_spec(1), legacy=[legacy_spec()]), True


# --------------------------------------------------------------------------- #
# Per-user envelope encryption
# --------------------------------------------------------------------------- #

WRAP_N = 2**14  # a login should not cost the user a visible pause


def derive_kek(password: str, salt: bytes) -> bytes:
    """Key-encrypting key from a password. Never stored, anywhere."""
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=WRAP_N, r=8, p=1,
        dklen=32, maxmem=64 * 1024 * 1024,
    )


def new_dek() -> bytes:
    return os.urandom(32)


def wrap_dek(dek: bytes, password: str, salt: bytes) -> str:
    kek = derive_kek(password, salt)
    return Fernet(base64.urlsafe_b64encode(kek)).encrypt(dek).decode("ascii")


def unwrap_dek(wrapped: str, password: str, salt: bytes) -> bytes | None:
    """The user's data key, or None if the password is wrong."""
    if not wrapped:
        return None
    kek = derive_kek(password, salt)
    try:
        return Fernet(base64.urlsafe_b64encode(kek)).decrypt(wrapped.encode("ascii"))
    except (InvalidToken, ValueError):
        return None


def dek_fernet(dek: bytes) -> Fernet:
    return Fernet(base64.urlsafe_b64encode(dek))


def encrypt_with_dek(dek: bytes, text: str) -> str:
    return dek_fernet(dek).encrypt(text.encode("utf-8")).decode("ascii")


def decrypt_with_dek(dek: bytes, token: str) -> str:
    try:
        return dek_fernet(dek).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeDecodeError):
        return ""
