"""Mint capped, revocable OpenRouter keys — one per person.

The point of this module is to change what a leaked key costs.

Storing someone's personal, unlimited API key means a leak exposes their whole
budget and revoking it is their problem, not yours. OpenRouter's provisioning API
inverts that: you hold **one** management key, and each account gets its own
sub-key with a hard credit ceiling that resets monthly. A leaked sub-key costs at
most that ceiling, revocation is one call, and per-user spend comes from the
provider rather than from this app's estimates.

It also means nobody has to hand you a personal credential. You stop being the
custodian of other people's keys, which is the part of the old arrangement that
does not scale past trusting each other.

API surface used (all under ``https://openrouter.ai/api/v1/keys``, authenticated
with the management key as a bearer token):

===========================  ======================================
``POST   /keys``             create a key; the reply carries the
                             only copy of the key string
``GET    /keys``             list keys
``GET    /keys/{hash}``      one key's limit and usage
``PATCH  /keys/{hash}``      change the cap, or disable it
``DELETE /keys/{hash}``      revoke permanently
===========================  ======================================

The key string itself is returned **once**, at creation. After that OpenRouter
only ever shows a masked ``label``, so this app saves the real value against the
account (encrypted) at the moment it is minted, and identifies it forever after
by its ``hash``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

KEYS_URL = "https://openrouter.ai/api/v1/keys"
REQUEST_TIMEOUT = 25

# Credit ceilings offered in the UI. Deliberately small: the whole point is that
# an accident or a leak is survivable, and a lecture costs a few cents.
LIMIT_PRESETS = (2.0, 5.0, 10.0, 25.0, 50.0)
DEFAULT_LIMIT = 5.0
RESET_PERIODS = ("monthly", "weekly", "daily")


class ProvisioningError(RuntimeError):
    pass


@dataclass(frozen=True)
class KeyInfo:
    """One provisioned key as OpenRouter reports it."""

    hash: str
    name: str = ""
    label: str = ""           # masked form, e.g. "sk-or-v1-abc...123"
    limit: float | None = None
    limit_remaining: float | None = None
    limit_reset: str | None = None
    usage: float = 0.0
    usage_monthly: float = 0.0
    disabled: bool = False
    created_at: str = ""

    @property
    def is_capped(self) -> bool:
        return self.limit is not None and self.limit > 0

    @property
    def fraction_used(self) -> float:
        """0.0–1.0 against the cap; 0.0 when uncapped."""
        if not self.is_capped:
            return 0.0
        return min(1.0, max(0.0, self.spent / float(self.limit)))

    @property
    def spent(self) -> float:
        """What this key has used against its cap.

        ``limit_remaining`` is authoritative when present — with a resetting
        limit, cumulative ``usage`` keeps climbing past what the current period
        actually shows as spent.
        """
        if self.is_capped and self.limit_remaining is not None:
            return max(0.0, float(self.limit) - float(self.limit_remaining))
        return float(self.usage_monthly or self.usage or 0.0)

    @property
    def status_label(self) -> str:
        if self.disabled:
            return "disabled"
        if not self.is_capped:
            return "no cap set"
        return f"${self.spent:,.2f} of ${float(self.limit):,.2f}"


@dataclass(frozen=True)
class MintedKey:
    """A freshly created key. ``secret`` is available only here, only once."""

    secret: str
    info: KeyInfo


def _coerce_key_info(payload: dict[str, Any]) -> KeyInfo:
    def number(field: str) -> float | None:
        value = payload.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return KeyInfo(
        hash=str(payload.get("hash") or payload.get("key_hash") or ""),
        name=str(payload.get("name") or ""),
        label=str(payload.get("label") or ""),
        limit=number("limit"),
        limit_remaining=number("limit_remaining"),
        limit_reset=payload.get("limit_reset"),
        usage=number("usage") or 0.0,
        usage_monthly=number("usage_monthly") or 0.0,
        disabled=bool(payload.get("disabled", False)),
        created_at=str(payload.get("created_at") or ""),
    )


def _unwrap(document: Any) -> dict[str, Any]:
    """Pull the key object out of a response.

    OpenRouter returns the record under ``data``, sometimes as an object and
    sometimes as a single-element list depending on the endpoint. Accepting both
    (and a bare object) costs three lines and removes a whole class of breakage
    when the API tightens up.
    """
    if isinstance(document, dict):
        data = document.get("data", document)
    else:
        data = document
    if isinstance(data, list):
        if not data:
            raise ProvisioningError("OpenRouter returned an empty key record.")
        data = data[0]
    if not isinstance(data, dict):
        raise ProvisioningError("Unexpected key record from OpenRouter.")
    return data


def _find_secret(document: Any) -> str:
    """Locate the one-time key string in a create response."""
    if isinstance(document, dict):
        for candidate in (document.get("key"), (document.get("data") or {}).get("key")
                          if isinstance(document.get("data"), dict) else None):
            if isinstance(candidate, str) and candidate.startswith("sk-"):
                return candidate
    return ""


class ProvisioningClient:
    """Thin wrapper over the OpenRouter key-management endpoints."""

    def __init__(self, management_key: str, timeout: int = REQUEST_TIMEOUT):
        if not management_key:
            raise ProvisioningError(
                "No OpenRouter management key is configured. An administrator "
                "sets one under Admin → Issued API keys."
            )
        self.management_key = management_key
        self.timeout = timeout

    # ------------------------------------------------------------------ #

    def _request(self, method: str, path: str = "", body: dict | None = None) -> Any:
        url = KEYS_URL + (f"/{path.lstrip('/')}" if path else "")
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.management_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "lecture-quiz-builder",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise ProvisioningError(_explain_http(exc)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProvisioningError(f"Could not reach OpenRouter: {exc}") from exc

        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProvisioningError(f"OpenRouter sent a malformed reply: {exc}") from exc

    # ------------------------------------------------------------------ #

    def verify(self) -> bool:
        """Is this management key usable? Called before saving it."""
        self._request("GET")
        return True

    def create_key(
        self, name: str, limit: float | None = DEFAULT_LIMIT, limit_reset: str | None = "monthly"
    ) -> MintedKey:
        body: dict[str, Any] = {"name": name}
        if limit is not None and limit > 0:
            body["limit"] = float(limit)
            if limit_reset in RESET_PERIODS:
                body["limit_reset"] = limit_reset

        document = self._request("POST", body=body)
        secret = _find_secret(document)
        if not secret:
            raise ProvisioningError(
                "OpenRouter created the key but did not return its value. "
                "Check openrouter.ai/settings/keys and delete the orphan."
            )
        return MintedKey(secret=secret, info=_coerce_key_info(_unwrap(document)))

    def get_key(self, key_hash: str) -> KeyInfo:
        return _coerce_key_info(_unwrap(self._request("GET", key_hash)))

    def update_key(
        self, key_hash: str, limit: float | None = None, disabled: bool | None = None
    ) -> KeyInfo:
        body: dict[str, Any] = {}
        if limit is not None:
            body["limit"] = float(limit)
        if disabled is not None:
            body["disabled"] = bool(disabled)
        if not body:
            return self.get_key(key_hash)
        return _coerce_key_info(_unwrap(self._request("PATCH", key_hash, body)))

    def delete_key(self, key_hash: str) -> None:
        self._request("DELETE", key_hash)

    def list_keys(self) -> list[KeyInfo]:
        document = self._request("GET")
        data = document.get("data", []) if isinstance(document, dict) else document
        if not isinstance(data, list):
            return []
        return [_coerce_key_info(item) for item in data if isinstance(item, dict)]


def _explain_http(exc: urllib.error.HTTPError) -> str:
    """Turn a status code into something an administrator can act on."""
    detail = ""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        detail = str(payload.get("error", {}).get("message") or payload.get("message") or "")
    except Exception:
        pass

    if exc.code in (401, 403):
        return (
            "OpenRouter rejected the management key. Provisioning needs a "
            "*management* key from openrouter.ai/settings/management-keys — an "
            "ordinary inference key will not work here."
            + (f" ({detail})" if detail else "")
        )
    if exc.code == 404:
        return "That key no longer exists at OpenRouter — it may already have been deleted."
    if exc.code == 429:
        return "OpenRouter is rate-limiting key management right now. Try again shortly."
    return f"OpenRouter returned HTTP {exc.code}" + (f": {detail}" if detail else ".")


def key_name_for(username: str, app_name: str = "lecture-quiz-builder") -> str:
    """A name that identifies the person and the app in OpenRouter's own list."""
    return f"{app_name}/{username}"
