"""Accounts, roles, saved settings, and two different kinds of API key.

Design decisions worth knowing:

**Passwords are never stored.** Only a scrypt hash with a per-user random salt.

**A personal API key is encrypted so that this app cannot read it.** Each account
holds a random data key (DEK); the DEK is stored only in a form wrapped under a
key derived from the user's password, which is never stored anywhere. Signing in
unwraps it into memory for that session. The consequence is the point: leaking
``APP_SECRET`` no longer exposes anyone's personal key, and an administrator
reading the store cannot decrypt one either. The cost is equally real — an
administrator password reset **destroys** that user's saved personal keys,
because if it did not, the administrator could recover them, which is the thing
being prevented.

**An issued key is different, and encrypted differently.** Keys this deployment
mints through a provider's provisioning API belong to the deployment, not to the
person. An administrator has to be able to create one *for* a user who is not
present, so those are encrypted under the app's own key. That is a weaker
guarantee, and it is acceptable only because an issued key is capped and
revocable — the reason to prefer issuing keys over collecting personal ones.

**One document per account.** Every save used to rewrite one file containing
everybody, which is how concurrent saves collided and how a partial write could
damage unrelated accounts. Accounts now live at ``users/u/<username>`` with a
separate index, and a legacy single-document store is migrated on first read.

**No self-registration**, **the last administrator cannot be removed**, and
**repeated failed sign-ins lock an account** for a cooling-off period.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import secrets as pysecrets
from dataclasses import dataclass, field
from typing import Any

from .keymgmt import (
    decrypt_with_dek,
    encrypt_with_dek,
    new_dek,
    unwrap_dek,
    wrap_dek,
)
from .storage import Cipher, Store, StorageError

LEGACY_USERS_PATH = "users"
INDEX_PATH = "users/index"
USER_PREFIX = "users/u"

SCRYPT_N = 2**14  # ~16 MB per hash: slow for an attacker, unnoticeable here
SCRYPT_R = 8
SCRYPT_P = 1
KEY_LENGTH = 32

ROLES = ("admin", "instructor")

# Sign-in throttling. scrypt already makes each guess expensive; a lockout stops
# an attacker from simply queueing thousands of them against a public URL.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


class AuthError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse(stamp: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Password hashing
# --------------------------------------------------------------------------- #


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return ``(hash_hex, salt_hex)``."""
    if len(password) < 8:
        raise AuthError("Passwords must be at least 8 characters.")
    salt_bytes = bytes.fromhex(salt) if salt else os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt_bytes,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        dklen=KEY_LENGTH, maxmem=64 * 1024 * 1024,
    )
    return digest.hex(), salt_bytes.hex()


def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
    if not password or not hash_hex or not salt_hex:
        return False
    try:
        candidate, _ = hash_password(password, salt_hex)
    except (AuthError, ValueError):
        return False
    # Constant-time: a timing difference here leaks how much of a hash matched.
    return hmac.compare_digest(candidate, hash_hex)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class User:
    username: str
    display_name: str = ""
    role: str = "instructor"
    password_hash: str = ""
    password_salt: str = ""
    active: bool = True
    must_change_password: bool = False
    created_at: str = field(default_factory=now)
    last_login: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    # Personal keys, encrypted under the user's DEK. Unreadable without them.
    api_keys: dict[str, str] = field(default_factory=dict)
    # Keys this app minted, encrypted under the app key so an admin can issue
    # them. Capped and revocable, which is what makes that acceptable.
    issued_keys: dict[str, str] = field(default_factory=dict)
    provisioned_keys: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Envelope encryption material.
    dek_salt: str = ""
    wrapped_dek: str = ""

    # Sign-in throttling.
    failed_attempts: int = 0
    locked_until: str = ""

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def label(self) -> str:
        return self.display_name or self.username

    @property
    def is_locked(self) -> bool:
        deadline = _parse(self.locked_until)
        return bool(deadline and deadline > _utcnow())

    @property
    def lock_minutes_left(self) -> int:
        deadline = _parse(self.locked_until)
        if not deadline:
            return 0
        return max(0, int((deadline - _utcnow()).total_seconds() // 60) + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "password_hash": self.password_hash,
            "password_salt": self.password_salt,
            "active": self.active,
            "must_change_password": self.must_change_password,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "settings": self.settings,
            "api_keys": self.api_keys,
            "issued_keys": self.issued_keys,
            "provisioned_keys": self.provisioned_keys,
            "dek_salt": self.dek_salt,
            "wrapped_dek": self.wrapped_dek,
            "failed_attempts": self.failed_attempts,
            "locked_until": self.locked_until,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Session:
    """What a successful sign-in yields: the account, and its unwrapped data key."""

    user: User
    dek: bytes | None = None


# --------------------------------------------------------------------------- #
# Directory
# --------------------------------------------------------------------------- #


class UserDirectory:
    """All account operations. One encrypted document per account."""

    def __init__(self, store: Store, cipher: Cipher | None = None, audit=None):
        self.store = store
        self.cipher = cipher
        self.audit = audit

    # -- audit helper -- #

    def _log(self, event: str, actor: str, subject: str = "", detail: str = "") -> None:
        if self.audit is not None:
            self.audit.record(event, actor, subject, detail)

    # -- persistence -- #

    def _path(self, username: str) -> str:
        return f"{USER_PREFIX}/{username}"

    def _index(self) -> list[str]:
        document = self.store.read(INDEX_PATH)
        if document is None:
            return self._migrate_legacy()
        names = document.get("usernames")
        return sorted(str(n) for n in names) if isinstance(names, list) else []

    def _write_index(self, names: list[str]) -> None:
        self.store.write(INDEX_PATH, {"version": 1, "usernames": sorted(set(names))})

    def _migrate_legacy(self) -> list[str]:
        """Split a pre-existing single ``users`` document into per-user files."""
        legacy = self.store.read(LEGACY_USERS_PATH)
        if not legacy:
            self._write_index([])
            return []

        records = legacy.get("users") or {}
        names: list[str] = []
        for name, record in records.items():
            if not isinstance(record, dict):
                continue
            user = User.from_dict(record)
            # Pre-envelope keys were encrypted under the app key. Move them to
            # issued_keys, which is where app-key-encrypted material now lives,
            # so nothing is lost and nothing is mislabelled as user-private.
            if user.api_keys and not user.issued_keys:
                user.issued_keys = dict(user.api_keys)
                user.api_keys = {}
            self._save_user(user)
            names.append(user.username)

        self._write_index(names)
        return sorted(names)

    def _load_user(self, username: str) -> User | None:
        document = self.store.read(self._path(username))
        return User.from_dict(document["user"]) if document and "user" in document else None

    def _save_user(self, user: User) -> None:
        self.store.write(self._path(user.username), {"version": 2, "user": user.to_dict()})

    # -- queries -- #

    def all_users(self) -> list[User]:
        users = [self._load_user(name) for name in self._index()]
        return sorted(
            (u for u in users if u is not None), key=lambda u: u.username.lower()
        )

    def get(self, username: str) -> User | None:
        name = _normalize(username)
        if not name or name not in self._index():
            return None
        return self._load_user(name)

    def is_empty(self) -> bool:
        return not self._index()

    def admin_count(self) -> int:
        return sum(1 for u in self.all_users() if u.is_admin and u.active)

    # -- mutations -- #

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "instructor",
        display_name: str = "",
        must_change_password: bool = True,
        actor: str = "system",
    ) -> User:
        username = _normalize(username)
        if not username:
            raise AuthError("A username is required.")
        if not username.replace(".", "").replace("_", "").replace("-", "").isalnum():
            raise AuthError("Usernames may contain letters, digits, dot, dash, underscore.")
        if role not in ROLES:
            raise AuthError(f"Unknown role: {role}")

        names = self._index()
        if username in names:
            raise AuthError(f"'{username}' already exists.")

        password_hash, salt = hash_password(password)
        dek_salt = os.urandom(16)
        user = User(
            username=username,
            display_name=display_name.strip(),
            role=role,
            password_hash=password_hash,
            password_salt=salt,
            must_change_password=must_change_password,
            dek_salt=dek_salt.hex(),
            wrapped_dek=wrap_dek(new_dek(), password, dek_salt),
        )
        self._save_user(user)
        self._write_index([*names, username])
        self._log("user.create", actor, username, f"role={role}")
        return user

    def authenticate(self, username: str, password: str) -> Session:
        """Verify credentials and unwrap the account's data key.

        Both failure modes return the same message: naming which usernames exist
        is free reconnaissance for anyone probing a public URL.
        """
        name = _normalize(username)
        user = self._load_user(name) if name in self._index() else None

        if user is None:
            self._log("login.failure", name or "?", detail="unknown account")
            raise AuthError("Incorrect username or password.")

        if user.is_locked:
            self._log("login.locked", user.username, user.username)
            raise AuthError(
                f"Too many failed attempts. Try again in {user.lock_minutes_left} minute(s), "
                "or ask an administrator to reset the password."
            )

        if not verify_password(password, user.password_hash, user.password_salt):
            user.failed_attempts += 1
            if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
                user.locked_until = (
                    _utcnow() + dt.timedelta(minutes=LOCKOUT_MINUTES)
                ).isoformat(timespec="seconds")
                user.failed_attempts = 0
                self._save_user(user)
                self._log("login.locked", user.username, user.username, "too many failures")
                raise AuthError(
                    f"Too many failed attempts. This account is locked for "
                    f"{LOCKOUT_MINUTES} minutes."
                )
            self._save_user(user)
            self._log("login.failure", user.username, user.username)
            raise AuthError("Incorrect username or password.")

        if not user.active:
            self._log("login.failure", user.username, user.username, "disabled")
            raise AuthError("That account has been disabled. Ask an administrator.")

        dek = self._unwrap(user, password)
        user.failed_attempts = 0
        user.locked_until = ""
        user.last_login = now()
        self._save_user(user)
        self._log("login.success", user.username, user.username)
        return Session(user=user, dek=dek)

    def _unwrap(self, user: User, password: str) -> bytes | None:
        """Recover the account's data key, creating one for older accounts."""
        if user.wrapped_dek and user.dek_salt:
            return unwrap_dek(user.wrapped_dek, password, bytes.fromhex(user.dek_salt))
        # An account created before envelope encryption existed: give it a data
        # key now, so the next personal key it saves is protected properly.
        salt = os.urandom(16)
        dek = new_dek()
        user.dek_salt = salt.hex()
        user.wrapped_dek = wrap_dek(dek, password, salt)
        return dek

    def set_password(
        self, username: str, password: str, clear_flag: bool = True,
        current_dek: bytes | None = None, actor: str = "",
    ) -> None:
        """Change a password, keeping personal keys when the data key is known.

        A user changing their own password passes the data key from their
        session, so it is simply re-wrapped and their saved keys survive. An
        administrator resetting somebody else's password has no data key — a new
        one is generated and the old personal keys become permanently unreadable,
        which is exactly the guarantee envelope encryption exists to provide.
        """
        user = self._require(username)
        user.password_hash, user.password_salt = hash_password(password)

        salt = os.urandom(16)
        user.dek_salt = salt.hex()
        if current_dek is not None:
            user.wrapped_dek = wrap_dek(current_dek, password, salt)
        else:
            user.wrapped_dek = wrap_dek(new_dek(), password, salt)
            if user.api_keys:
                user.api_keys = {}
        if clear_flag:
            user.must_change_password = False
        user.failed_attempts = 0
        user.locked_until = ""
        self._save_user(user)
        self._log(
            "password.change" if current_dek is not None else "password.reset",
            actor or user.username, user.username,
        )

    def set_role(self, username: str, role: str, actor: str = "") -> None:
        if role not in ROLES:
            raise AuthError(f"Unknown role: {role}")
        user = self._require(username)
        if user.is_admin and role != "admin" and self.admin_count() <= 1:
            raise AuthError("This is the only administrator — promote someone else first.")
        user.role = role
        self._save_user(user)
        self._log("user.role", actor, user.username, f"role={role}")

    def set_active(self, username: str, active: bool, actor: str = "") -> None:
        user = self._require(username)
        if user.is_admin and not active and self.admin_count() <= 1:
            raise AuthError("This is the only administrator — you would lock yourself out.")
        user.active = active
        if active:
            user.failed_attempts = 0
            user.locked_until = ""
        self._save_user(user)
        self._log("user.active", actor, user.username, f"active={active}")

    def unlock(self, username: str, actor: str = "") -> None:
        user = self._require(username)
        user.failed_attempts = 0
        user.locked_until = ""
        self._save_user(user)
        self._log("user.active", actor, user.username, "unlocked")

    def delete_user(self, username: str, actor: str = "") -> None:
        user = self._require(username)
        if user.is_admin and self.admin_count() <= 1:
            raise AuthError("This is the only administrator — it cannot be deleted.")
        self._write_index([n for n in self._index() if n != user.username])
        self.store.write(self._path(user.username), {"version": 2, "deleted": True})
        self._log("user.delete", actor, user.username)

    def _require(self, username: str) -> User:
        user = self._load_user(_normalize(username))
        if user is None:
            raise AuthError("No such account.")
        return user

    # -- per-user settings -- #

    def save_settings(self, username: str, settings: dict[str, Any]) -> None:
        user = self._load_user(_normalize(username))
        if user is None:
            return
        user.settings.update(settings)
        self._save_user(user)

    # -- personal keys (user-private) -- #

    def save_api_key(
        self, username: str, provider: str, api_key: str, dek: bytes | None, actor: str = ""
    ) -> None:
        """Store a key only this user can read back."""
        if dek is None:
            raise AuthError(
                "Your session does not hold the key needed to encrypt this. "
                "Sign out and back in, then try again."
            )
        user = self._require(username)
        if api_key:
            user.api_keys[provider] = encrypt_with_dek(dek, api_key)
            event = "key.saved"
        else:
            user.api_keys.pop(provider, None)
            event = "key.removed"
        self._save_user(user)
        self._log(event, actor or user.username, user.username, f"provider={provider}")

    def get_api_key(self, user: User, provider: str, dek: bytes | None) -> str:
        """The user's own key, if their session can decrypt it."""
        token = user.api_keys.get(provider, "")
        if not token or dek is None:
            return ""
        return decrypt_with_dek(dek, token)

    def has_api_key(self, user: User, provider: str) -> bool:
        return bool(user.api_keys.get(provider))

    def clear_all_api_keys(self, username: str, actor: str = "") -> int:
        """Self-service revocation: forget every personal key at once."""
        user = self._require(username)
        removed = len(user.api_keys)
        user.api_keys = {}
        self._save_user(user)
        if removed:
            self._log("key.removed", actor or user.username, user.username, "all personal keys")
        return removed

    # -- issued keys (app-owned, capped, revocable) -- #

    def record_provisioned_key(
        self, username: str, provider: str, secret: str, record: dict[str, Any],
        actor: str = "",
    ) -> None:
        """Save a freshly minted key and the handle needed to manage it.

        Both halves are written in one operation: a key stored without its hash
        could never be revoked, and a hash without its key would leave the
        account unable to run anything.
        """
        if self.cipher is None:
            raise AuthError("Encryption is not configured, so keys cannot be saved.")
        user = self._require(username)
        user.issued_keys[provider] = self.cipher.encrypt_text(secret)
        user.provisioned_keys[provider] = dict(record)
        self._save_user(user)
        self._log(
            "key.issued", actor, user.username,
            f"provider={provider} limit={record.get('limit')}",
        )

    def clear_provisioned_key(
        self, username: str, provider: str, actor: str = ""
    ) -> dict[str, Any]:
        """Forget a minted key locally. Returns the record, for revocation."""
        user = self._require(username)
        record = user.provisioned_keys.pop(provider, {})
        user.issued_keys.pop(provider, None)
        self._save_user(user)
        if record:
            self._log("key.revoked", actor, user.username, f"provider={provider}")
        return record

    def get_issued_key(self, user: User, provider: str) -> str:
        if self.cipher is None:
            return ""
        token = user.issued_keys.get(provider, "")
        return self.cipher.decrypt_text(token) if token else ""

    def provisioned_record(self, user: User, provider: str) -> dict[str, Any]:
        return user.provisioned_keys.get(provider, {})

    # -- resolution -- #

    def resolve_key(self, user: User, provider: str, dek: bytes | None) -> tuple[str, str]:
        """The key this session should use, and where it came from.

        A personal key wins over an issued one: somebody who deliberately saved
        their own key meant to use it.
        """
        personal = self.get_api_key(user, provider, dek)
        if personal:
            return personal, "personal"
        issued = self.get_issued_key(user, provider)
        if issued:
            return issued, "issued"
        return "", ""

    # -- bootstrap -- #

    def bootstrap_admin(self, username: str, password: str, display_name: str = "") -> Session:
        """Create the very first administrator. Refuses once anyone exists."""
        if not self.is_empty():
            raise AuthError("Accounts already exist; the first-run setup is closed.")
        self.create_user(
            username, password, role="admin",
            display_name=display_name, must_change_password=False, actor="first-run",
        )
        return self.authenticate(username, password)


def _normalize(username: str) -> str:
    return (username or "").strip().lower()


def suggest_password(length: int = 14) -> str:
    """A password worth handing to a colleague — no ambiguous characters."""
    alphabet = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(pysecrets.choice(alphabet) for _ in range(length))


def mask_key(key: str) -> str:
    """Show that a key exists without showing the key."""
    if not key:
        return ""
    return f"…{key[-4:]}" if len(key) > 4 else "…"


__all__ = [
    "AuthError",
    "LOCKOUT_MINUTES",
    "MAX_FAILED_ATTEMPTS",
    "ROLES",
    "Session",
    "StorageError",
    "User",
    "UserDirectory",
    "hash_password",
    "mask_key",
    "suggest_password",
    "verify_password",
]
