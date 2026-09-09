"""Provenance: what `result_manifest.DataSource` needs, plus what it should have asked for.

A DataSource records a name, a retrieval time and an adjustment string. That is enough to
print a result card and not enough to reproduce anything: it does not say which version of
the client ran, what was actually requested, or whether the bytes are still the bytes. This
module adds those three and then refuses to carry the one thing that must never be written
down - a credential.

Two rules hold for every Provenance ever constructed:

  * `content_sha256` hashes the DATA, not the path and not the retrieval time. Same arrays,
    same hash, whatever order the columns arrived in; one changed cell, different hash. That
    is what makes `Cache.verify()` and `Cache.refetch()` mean something.
  * `request` is scrubbed in `__post_init__`, before anything can read it. A key-shaped name,
    a value equal to a live credential in the environment, or a 32-hex FRED-key-shaped string
    is replaced by "<redacted>". The scrub happens at construction, so a careless adapter
    cannot leak a key into a repr, a manifest row, a JSON sidecar or the hash.

Derived from this repository's own trap tables (market-data-sourcing,
research-integrity-guards), not from any other library's schema.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fin_skills.core.result_manifest import DataSource

#: version of THIS layer, recorded in every Provenance. Bumped when normalisation changes.
ADAPTER_VERSION = "0.1.0"

REDACTED = "<redacted>"

# a mapping key whose NAME says it holds a credential
_KEY_NAME = re.compile(
    r"(?i)(api[_-]?key|apikey|access[_-]?key|secret|password|passwd|token|bearer|"
    r"credential|authorization|auth[_-]?header|cookie|session[_-]?id|private[_-]?key)")

# a value shaped like a FRED key: 32 lowercase hex characters, nothing else
_FRED_KEY_SHAPE = re.compile(r"^[0-9a-f]{32}$")


def _live_credentials() -> frozenset[str]:
    """Every value currently sitting in a credential-shaped environment variable.

    Redacting by NAME misses a key passed under an innocent name; redacting by VALUE
    catches it. This reads the environment and never stores what it read.
    """
    return frozenset(v for k, v in os.environ.items()
                     if v and len(v) >= 8 and _KEY_NAME.search(k))


def scrub(obj: Any) -> Any:
    """Return `obj` with anything credential-shaped replaced by "<redacted>".

    Recurses through mappings and sequences. Three tests, any of which redacts:
    the mapping key names a credential, the string value equals a live credential in the
    environment, or the string value has the shape of a FRED key (32 lowercase hex).
    """
    live = _live_credentials()
    return _scrub(obj, live)


def _scrub(obj: Any, live: frozenset[str]) -> Any:
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            out[k] = REDACTED if isinstance(k, str) and _KEY_NAME.search(k) \
                else _scrub(v, live)
        return out
    if isinstance(obj, (list, tuple)):
        cls = type(obj) if not isinstance(obj, tuple) else tuple
        return cls(_scrub(v, live) for v in obj)
    if isinstance(obj, str):
        if obj in live or _FRED_KEY_SHAPE.match(obj):
            return REDACTED
        return obj
    return obj


# ------------------------------------------------------------------------- content hash
def _index_bytes(idx: pd.Index) -> bytes:
    if isinstance(idx, pd.MultiIndex):
        return b"\0".join(_index_bytes(idx.get_level_values(i)) for i in range(idx.nlevels))
    if isinstance(idx, pd.DatetimeIndex):
        # tz is metadata, not content: the same instants hash the same in any zone
        return idx.tz_convert("UTC").asi8.tobytes() if idx.tz is not None else idx.asi8.tobytes()
    return _values_bytes(np.asarray(idx))


def _values_bytes(values: np.ndarray) -> bytes:
    kind = values.dtype.kind
    if kind in "iub":
        return np.asarray(values, dtype="int64").tobytes()
    if kind == "f":
        return np.asarray(values, dtype="float64").tobytes()
    if kind == "M":
        return values.astype("datetime64[ns]").view("int64").tobytes()
    if kind == "m":
        return values.astype("timedelta64[ns]").view("int64").tobytes()
    return "\0".join("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                     for v in values.tolist()).encode("utf-8", "replace")


def content_hash(obj: Any) -> str:
    """sha256 over the canonicalised content of a frame, a series or a JSON-able object.

    Canonical means: rows in index order after `sort_index()`, columns in sorted NAME
    order, dtypes included, timezone normalised to UTC. Column order therefore does not
    change the hash and a single changed cell always does. Nothing about WHERE the object
    came from or WHEN it was fetched enters the digest.
    """
    h = hashlib.sha256()
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(name=obj.name if obj.name is not None else "value")
    if isinstance(obj, pd.DataFrame):
        d = obj.sort_index()
        h.update(b"frame\0")
        h.update(_index_bytes(d.index))
        for col in sorted(d.columns, key=repr):
            s = d[col]
            if isinstance(s, pd.DataFrame):        # duplicate column labels
                s = s.iloc[:, 0]
            h.update(repr(col).encode("utf-8", "replace") + b"\0")
            h.update(str(s.dtype).encode("ascii", "replace") + b"\0")
            h.update(_values_bytes(s.to_numpy()))
        return h.hexdigest()
    h.update(b"json\0")
    h.update(canonical_json(obj).encode("utf-8"))
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace jitter, str() for the rest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def utc_now() -> str:
    """ISO-8601 UTC, second resolution - the stamp every Provenance carries."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- Provenance
@dataclass(frozen=True)
class Provenance:
    """Where one artefact came from, in enough detail to fetch it again and check it.

    `library_version` is read at CALL time rather than taken from a lockfile, because
    akshare deletes its own PyPI history (219 releases listed, oldest 1.16.72 of
    2025-04-05, against 1,437 in the upstream changelog - verified 2026-09-09), so a pin
    is not evidence of what ran.
    """

    source: str                       # "yfinance" | "akshare" | "ccxt:binance" | "fred" | "edgar"
    adapter_version: str
    library_version: str
    retrieved_at: str                 # ISO-8601 UTC
    request: dict                     # the NORMALIZED request; scrubbed at construction
    content_sha256: str               # over the canonicalised arrays
    vendor_vintage: str = ""          # the vendor's own stamp (FRED realtime, EDGAR accn)
    terms_url: str = ""
    redistributable: bool = False
    attribution: str = ""             # the exact string the vendor's terms require, if any

    def __post_init__(self) -> None:
        object.__setattr__(self, "request", scrub(dict(self.request)))

    # ------------------------------------------------------------------ conversions
    def as_data_source(self, adjustment: str = "unknown") -> DataSource:
        """The row `result_manifest.ResultCard` wants. `adjustment` is the convention the
        schema object declares - "anchored_start", "PIT, filed<=asof", "vintage as-of"."""
        return DataSource(name=self.source, retrieved_at=self.retrieved_at,
                          adjustment=str(adjustment))

    def to_dict(self) -> dict:
        """A JSON-able mapping. Already scrubbed - `request` cannot hold a credential."""
        return {"source": self.source, "adapter_version": self.adapter_version,
                "library_version": self.library_version, "retrieved_at": self.retrieved_at,
                "request": self.request, "content_sha256": self.content_sha256,
                "vendor_vintage": self.vendor_vintage, "terms_url": self.terms_url,
                "redistributable": bool(self.redistributable),
                "attribution": self.attribution}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Provenance":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def with_content(self, obj: Any) -> "Provenance":
        """The same provenance re-hashed against `obj`."""
        return replace(self, content_sha256=content_hash(obj))

    def __repr__(self) -> str:
        req = canonical_json(self.request)
        if len(req) > 120:
            req = req[:117] + "..."
        return (f"Provenance(source={self.source!r}, library_version="
                f"{self.library_version!r}, retrieved_at={self.retrieved_at!r}, "
                f"request={req}, content_sha256={self.content_sha256[:12]!r}...)")


def make(source: str, *, library_version: str, request: Mapping[str, Any], content: Any,
         vendor_vintage: str = "", terms_url: str = "", redistributable: bool = False,
         attribution: str = "", retrieved_at: str | None = None) -> Provenance:
    """Build a Provenance, hashing `content` and stamping the clock. The one constructor
    adapters use, so no adapter has to remember to hash or to scrub."""
    return Provenance(source=source, adapter_version=ADAPTER_VERSION,
                      library_version=str(library_version),
                      retrieved_at=retrieved_at or utc_now(), request=dict(request),
                      content_sha256=content_hash(content), vendor_vintage=vendor_vintage,
                      terms_url=terms_url, redistributable=bool(redistributable),
                      attribution=attribution)


__all__ = ["ADAPTER_VERSION", "REDACTED", "Provenance", "canonical_json", "content_hash",
           "make", "scrub", "utc_now"]
