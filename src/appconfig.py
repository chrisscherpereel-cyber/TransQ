"""Deployment-wide settings, kept in the same encrypted store as accounts.

Only one thing lives here so far — the OpenRouter management key — but it needed
a home that is neither a per-user record nor a Streamlit secret. Streamlit
secrets are edited by redeploying; this is set from the Admin tab by whoever runs
the deployment, and it is a credential, so it is encrypted individually on top of
the already-encrypted document.
"""

from __future__ import annotations

from typing import Any

from .storage import Cipher, Store

CONFIG_PATH = "config"

# Keys under this prefix are encrypted individually before being written.
SECRET_PREFIX = "secret:"

# Config credentials use their own HKDF subkey, distinct from the one guarding
# users' API keys — compromising one context should not hand over the other.


class AppConfig:
    """Small key/value store for deployment settings."""

    def __init__(self, store: Store, cipher: Cipher | None = None):
        self.store = store
        self.cipher = cipher

    def _load(self) -> dict[str, Any]:
        document = self.store.read(CONFIG_PATH) or {}
        values = document.get("values")
        return values if isinstance(values, dict) else {}

    def _save(self, values: dict[str, Any]) -> None:
        self.store.write(CONFIG_PATH, {"version": 1, "values": values})

    # -- plain values -- #

    def get(self, name: str, default: Any = None) -> Any:
        return self._load().get(name, default)

    def set(self, name: str, value: Any) -> None:
        values = self._load()
        values[name] = value
        self._save(values)

    # -- credentials -- #

    def get_secret(self, name: str) -> str:
        token = self._load().get(SECRET_PREFIX + name, "")
        if not token or self.cipher is None:
            return ""
        return self.cipher.decrypt_config(token)

    def set_secret(self, name: str, value: str) -> None:
        if self.cipher is None:
            raise RuntimeError("Encryption is not configured, so secrets cannot be saved.")
        values = self._load()
        if value:
            values[SECRET_PREFIX + name] = self.cipher.encrypt_config(value)
        else:
            values.pop(SECRET_PREFIX + name, None)
        self._save(values)

    def has_secret(self, name: str) -> bool:
        return bool(self._load().get(SECRET_PREFIX + name))
