"""Public records and watch specifications. Source text is data, never instructions."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc(value: str | datetime) -> str:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        raise ValueError("timestamps require an explicit timezone")
    return stamp.astimezone(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    """An observation, not necessarily a trade. Dates remain distinct.

    ``published_at`` is the source's actual timestamp (unknown stays None);
    ``observed_at`` is when the collector saw this revision. Historical availability
    must not be inferred from a transaction date or a filing date with no time of day.
    ``data`` retains source units, parse status, and source text where needed.
    """

    id: str
    source: str
    kind: str
    title: str
    url: str
    observed_at: str
    published_at: str | None = None
    actor: str = ""
    symbol: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not self.source or not self.kind:
            raise ValueError("event id, source and kind are required")
        object.__setattr__(self, "observed_at", utc(self.observed_at))
        if self.published_at is not None:
            object.__setattr__(self, "published_at", utc(self.published_at))
        digest(self.to_dict())                  # reject NaN and unserializable values

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        value = self.to_dict()
        value.pop("observed_at")
        return digest(value)


@dataclass(frozen=True)
class Watch:
    """A persisted collector configuration; credentials belong in environment variables."""

    id: str
    source: str
    target: str
    interval_seconds: float = 300
    options: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if not self.id or not self.source or not self.target:
            raise ValueError("watch id, source and target are required")
        if self.source in ('rss', 'page'):
            from .http import public_url
            public_url(self.target)
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 1:
            raise ValueError("interval_seconds must be finite and at least 1")
        if not isinstance(self.options, dict) or not isinstance(self.enabled, bool):
            raise TypeError("options must be an object and enabled must be a boolean")
        forbidden = ("token", "secret", "password", "api_key", "apikey", "authorization")
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if any(x in str(key).lower() for x in forbidden):
                        raise ValueError("credentials must be supplied through environment variables")
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(self.options)
        digest(asdict(self))


@dataclass
class Batch:
    events: list[Event] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    retry_after: float | None = None
