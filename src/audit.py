"""An audit trail for the events that matter after something goes wrong.

Deliberately narrow. This records *that* a credential was saved, issued, or
revoked, and *who* did it — never the credential, never a fragment of one, never
enough to reconstruct one. An audit log that leaks the thing it audits is worse
than no log, because it concentrates every secret in one file and calls it
diligence.

Sign-in failures are recorded too, because "when did this account start failing
to log in" is the question you cannot answer retrospectively without them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from .storage import Store, StorageError

AUDIT_PATH = "audit"
MAX_EVENTS = 2000

# Event names, so a typo cannot quietly create a category nobody reads.
LOGIN_OK = "login.success"
LOGIN_FAIL = "login.failure"
LOGIN_LOCKED = "login.locked"
USER_CREATE = "user.create"
USER_DELETE = "user.delete"
USER_ROLE = "user.role"
USER_ACTIVE = "user.active"
PASSWORD_CHANGE = "password.change"
PASSWORD_RESET = "password.reset"
KEY_SAVED = "key.saved"
KEY_REMOVED = "key.removed"
KEY_ISSUED = "key.issued"
KEY_REVOKED = "key.revoked"
KEY_CAP = "key.cap_changed"
MGMT_KEY_SET = "management_key.set"
MGMT_KEY_CLEARED = "management_key.cleared"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class AuditEvent:
    timestamp: str
    event: str
    actor: str            # who did it
    subject: str = ""     # who it was done to
    detail: str = ""      # never a secret

    @property
    def day(self) -> str:
        return self.timestamp[:10]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class AuditLog:
    def __init__(self, store: Store):
        self.store = store

    def _rows(self) -> list[dict[str, Any]]:
        document = self.store.read(AUDIT_PATH) or {}
        rows = document.get("events")
        return rows if isinstance(rows, list) else []

    def record(self, event: str, actor: str, subject: str = "", detail: str = "") -> None:
        """Append one event. Never raises — an audit failure must not block work."""
        try:
            rows = self._rows()
            rows.append(
                asdict(AuditEvent(now(), event, actor, subject, detail))
            )
            if len(rows) > MAX_EVENTS:
                rows = rows[-MAX_EVENTS:]
            self.store.write(AUDIT_PATH, {"version": 1, "events": rows})
        except (StorageError, Exception):
            pass

    def events(self, subject: str | None = None, limit: int = 200) -> list[AuditEvent]:
        rows = [AuditEvent.from_dict(r) for r in self._rows()]
        if subject:
            rows = [e for e in rows if subject in (e.actor, e.subject)]
        return list(reversed(rows))[:limit]
