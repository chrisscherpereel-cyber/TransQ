"""Token and cost accounting.

Two audiences, one ledger. While a job runs, the sidebar shows tokens and dollars
climbing call by call, so nobody discovers the cost only after the fact. Across
sessions, the same records answer the questions that matter later: what has this
account spent this month, which model is actually cheapest for this work, and —
for an administrator — who is spending what.

Costs are estimates from published rates, not billed amounts. The provider's own
dashboard is the authority; this is here so nothing is a surprise before you get
there.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

from .storage import Store

USAGE_PATH = "usage"

# Keep the ledger bounded: an encrypted blob is rewritten whole on every append,
# so unbounded growth would eventually make each save slow.
MAX_RECORDS = 5000


@dataclass
class UsageRecord:
    """One billable interaction with a model."""

    timestamp: str
    username: str
    provider: str
    model: str
    operation: str  # summarize | generate | review | replace | alternative
    input_tokens: int
    output_tokens: int
    cost: float
    cost_known: bool = True
    calls: int = 1
    source: str = ""  # the lecture this was for

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def day(self) -> str:
        return self.timestamp[:10]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Totals:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    calls: int = 0
    runs: int = 0
    unpriced_runs: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, record: UsageRecord) -> None:
        self.input_tokens += record.input_tokens
        self.output_tokens += record.output_tokens
        self.cost += record.cost
        self.calls += record.calls
        self.runs += 1
        if not record.cost_known:
            self.unpriced_runs += 1

    @property
    def cost_label(self) -> str:
        if self.unpriced_runs and self.cost == 0:
            return "—"
        suffix = " +" if self.unpriced_runs else ""
        return f"${self.cost:,.4f}{suffix}"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class UsageLog:
    """Append-only ledger, persisted through whichever Store is configured."""

    def __init__(self, store: Store):
        self.store = store

    def _load_raw(self) -> list[dict[str, Any]]:
        document = self.store.read(USAGE_PATH) or {}
        records = document.get("records")
        return records if isinstance(records, list) else []

    def records(self, username: str | None = None) -> list[UsageRecord]:
        parsed = [UsageRecord.from_dict(r) for r in self._load_raw()]
        if username is not None:
            parsed = [r for r in parsed if r.username == username]
        return parsed

    def append(self, record: UsageRecord) -> None:
        rows = self._load_raw()
        rows.append(asdict(record))
        if len(rows) > MAX_RECORDS:
            rows = rows[-MAX_RECORDS:]
        self.store.write(USAGE_PATH, {"version": 1, "records": rows})

    # -- aggregation -- #

    def totals(self, username: str | None = None, since: str | None = None) -> Totals:
        total = Totals()
        for record in self.records(username):
            if since and record.day < since:
                continue
            total.add(record)
        return total

    def by_day(self, username: str | None = None, days: int = 30) -> dict[str, float]:
        cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
        buckets: dict[str, float] = {}
        for record in self.records(username):
            if record.day >= cutoff:
                buckets[record.day] = buckets.get(record.day, 0.0) + record.cost
        return dict(sorted(buckets.items()))

    def by_model(self, username: str | None = None) -> dict[str, Totals]:
        buckets: dict[str, Totals] = {}
        for record in self.records(username):
            buckets.setdefault(record.model, Totals()).add(record)
        return dict(sorted(buckets.items(), key=lambda kv: -kv[1].cost))

    def by_user(self) -> dict[str, Totals]:
        buckets: dict[str, Totals] = {}
        for record in self.records():
            buckets.setdefault(record.username, Totals()).add(record)
        return dict(sorted(buckets.items(), key=lambda kv: -kv[1].cost))

    def by_operation(self, username: str | None = None) -> dict[str, Totals]:
        buckets: dict[str, Totals] = {}
        for record in self.records(username):
            buckets.setdefault(record.operation, Totals()).add(record)
        return buckets


@dataclass
class LiveMeter:
    """Running totals for the job in progress.

    The LLM client calls :meth:`record` after every request, which is what makes
    the on-screen figures move during a run rather than jumping at the end.
    """

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost: float = 0.0
    cost_known: bool = True
    history: list[tuple[int, float]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, input_tokens: int, output_tokens: int, rate: tuple[float, float] | None) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        if rate is None:
            self.cost_known = False
        else:
            self.cost += (input_tokens / 1e6) * rate[0] + (output_tokens / 1e6) * rate[1]
        self.history.append((self.total_tokens, self.cost))

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.calls = 0
        self.cost = 0.0
        self.cost_known = True
        self.history.clear()

    @property
    def cost_label(self) -> str:
        return f"${self.cost:,.4f}" if self.cost_known else "—"

    def to_record(self, username: str, operation: str, source: str = "") -> UsageRecord:
        return UsageRecord(
            timestamp=now(),
            username=username,
            provider=self.provider,
            model=self.model,
            operation=operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=self.cost,
            cost_known=self.cost_known,
            calls=self.calls,
            source=source,
        )
