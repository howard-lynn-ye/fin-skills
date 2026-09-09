"""The JSON boundary: pandas objects in, JSON-friendly payloads out, and back.

A guard wants a `pd.Series` with a sorted DatetimeIndex. An agent has JSON. This module
owns the one conversion between them, in both directions, so the JSON Schema that
`fin_skills.tools.schema` publishes and the decoder that runs at call time can never
describe different things - the schema fragments below ARE the shapes `decode()` accepts.

    series   {"index": ["2024-01-02", ...], "values": [0.001, ...],
              "name": "returns", "dtype": "float64", "freq": "B"}
    frame    {"index": [...], "columns": ["open", "close"],
              "records": [{"open": 1.0, "close": 2.0}, ...]}

`index`, `name`, `dtype` and `freq` are optional; a bare list of numbers is accepted
wherever a Series is expected. Dates cross as ISO strings. `freq` is carried because a
`pd.bdate_range` index has one and `pandas.testing.assert_series_equal` compares it: drop
it and the round trip is no longer exact.

JSON has no NaN or Infinity. `encode()` writes a missing value as `null` and a non-finite
one as the string "Infinity" / "-Infinity" (a breakeven cost of infinity is a real, useful
answer - it must not become null); `decode()` reads all three back.

Every failure raises TypeError naming the field, because that is the convention every
guard already follows (`Guard.run` raises TypeError on bad inputs, never on a failed check).

Size caps are enforced on the way in, not left to the machine's memory:

    MAX_ARGUMENT_BYTES   the whole `arguments` object, as JSON
    MAX_POINTS           cells in one series or frame (rows x columns)

Both are module constants so the documentation can quote the code instead of a memory.
"""
from __future__ import annotations

import json
import math
import numbers
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- size caps
MAX_ARGUMENT_BYTES = 8_000_000
"""Largest `arguments` object one tool call may carry, measured as compact JSON (8 MB)."""

MAX_POINTS = 1_000_000
"""Largest single series or frame, in cells (rows x columns). 1e6 float64 cells is ~8 MB."""

MAX_EVIDENCE_ROWS = 50
"""Rows of any one evidence table that survive into the result."""

MAX_EVIDENCE_ITEMS = 500
"""Elements of any one evidence series or list that survive into the result."""

MAX_EVIDENCE_CHARS = 4_000
"""Characters of any one evidence string that survive into the result."""

_POS_INF = "Infinity"
_NEG_INF = "-Infinity"

# ------------------------------------------------------------------- the schema fragments
# These are the payload shapes, as JSON Schema. schema.py imports them rather than writing
# its own copy: one definition, so the published schema and the decoder cannot drift.
# These travel inside every tool schema that uses them, so the prose is kept short: the
# long form lives in this module's docstring and in the fin-skills-as-tools skill.
_INDEX_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Row labels; ISO-8601 strings for a time index.",
    "items": {"type": ["string", "number"]},
}

SERIES_SCHEMA: dict[str, Any] = {
    "description": "A pandas Series: {'index': [...], 'values': [...]}, index optional. "
                   "A bare list of numbers also works. null = NaN; use the strings "
                   "'Infinity' / '-Infinity'.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "index": _INDEX_SCHEMA,
                "values": {"type": "array",
                           "items": {"type": ["number", "string", "boolean", "null"]}},
                "name": {"type": ["string", "null"]},
                "dtype": {"type": "string", "description": "numpy dtype, e.g. 'float64'."},
                "freq": {"type": ["string", "null"],
                         "description": "pandas offset alias, e.g. 'B'."},
            },
            "required": ["values"],
            "additionalProperties": False,
        },
        {"type": "array", "items": {"type": ["number", "null"]}},
    ],
}

FRAME_SCHEMA: dict[str, Any] = {
    "description": "A pandas DataFrame: {'index': [...], 'records': [{col: value}, ...]}, "
                   "index optional. A bare list of records also works. null = NaN.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "index": _INDEX_SCHEMA,
                "records": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "array", "items": {"type": "string"},
                            "description": "Column order; defaults to the first record's keys."},
                "dtypes": {"type": "object", "additionalProperties": {"type": "string"}},
                "freq": {"type": ["string", "null"]},
                "index_name": {"type": ["string", "null"]},
            },
            "required": ["records"],
            "additionalProperties": False,
        },
        {"type": "array", "items": {"type": "object"}},
    ],
}

_SERIES_KEYS = frozenset({"index", "values", "name", "dtype", "freq"})
_FRAME_KEYS = frozenset({"index", "records", "columns", "dtypes", "freq", "index_name"})


# ------------------------------------------------------------------------------ scalars
def _json_number(x: Any) -> Any:
    """A float as JSON: finite -> float, NaN -> None, +/-inf -> the string form."""
    v = float(x)
    if math.isnan(v):
        return None
    if math.isinf(v):
        return _POS_INF if v > 0 else _NEG_INF
    return v


def _from_json_number(x: Any) -> Any:
    if x is None:
        return float("nan")
    if x == _POS_INF:
        return float("inf")
    if x == _NEG_INF:
        return float("-inf")
    return x


def _index_to_json(idx: pd.Index) -> list[Any]:
    if isinstance(idx, pd.DatetimeIndex):
        return [None if pd.isna(t) else pd.Timestamp(t).isoformat() for t in idx]
    return [_json_number(v) if isinstance(v, (float, np.floating)) else
            (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else
             (None if v is None else str(v)))
            for v in idx]


def _index_from_json(values: Sequence[Any] | None, field: str, n: int,
                     freq: str | None = None, name: Any = None) -> pd.Index | None:
    if values is None:
        return None
    if len(values) != n:
        raise TypeError(f"{field}: index has {len(values)} label(s) but there are {n} value(s)")
    if values and all(isinstance(v, str) for v in values):
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(list(values)), name=name)
        except (TypeError, ValueError):
            return pd.Index(list(values), name=name)
        if freq:
            try:
                idx.freq = pd.tseries.frequencies.to_offset(freq)
            except (TypeError, ValueError):  # a freq the data does not actually follow
                pass
        return idx
    return pd.Index(list(values), name=name)


# ------------------------------------------------------------------------------- encode
def series_to_payload(s: pd.Series, limit: int | None = None) -> dict[str, Any]:
    """A Series as the JSON payload shape. `limit` truncates and flags the result."""
    truncated = limit is not None and len(s) > limit
    body = s.iloc[:limit] if truncated else s
    values = [_json_number(v) if isinstance(v, (float, np.floating)) else
              (bool(v) if isinstance(v, (bool, np.bool_)) else
               (int(v) if isinstance(v, (int, np.integer)) else
                (None if v is None or (isinstance(v, float) and math.isnan(v)) else
                 (v if isinstance(v, str) else str(v)))))
              for v in body.to_numpy(dtype=object)]
    out: dict[str, Any] = {"index": _index_to_json(body.index), "values": values,
                           "name": None if s.name is None else str(s.name),
                           "dtype": str(s.dtype)}
    freq = getattr(body.index, "freqstr", None)
    if freq:
        out["freq"] = freq
    if truncated:
        out["truncated"] = {"kept": len(body), "of": len(s)}
    return out


def frame_to_payload(df: pd.DataFrame, limit: int | None = None) -> dict[str, Any]:
    """A DataFrame as the JSON payload shape (records plus an index)."""
    truncated = limit is not None and len(df) > limit
    body = df.iloc[:limit] if truncated else df
    cols = [str(c) for c in body.columns]
    records = []
    for row in body.itertuples(index=False, name=None):
        records.append({c: (_json_number(v) if isinstance(v, (float, np.floating)) else
                            (bool(v) if isinstance(v, (bool, np.bool_)) else
                             (int(v) if isinstance(v, (int, np.integer)) else
                              (None if v is None else
                               (v if isinstance(v, str) else
                                (pd.Timestamp(v).isoformat()
                                 if isinstance(v, (pd.Timestamp, np.datetime64)) else str(v)))))))
                        for c, v in zip(cols, row)})
    out: dict[str, Any] = {"index": _index_to_json(body.index), "columns": cols,
                           "records": records,
                           "dtypes": {c: str(t) for c, t in zip(cols, body.dtypes)}}
    if body.index.name is not None:
        out["index_name"] = str(body.index.name)
    freq = getattr(body.index, "freqstr", None)
    if freq:
        out["freq"] = freq
    if truncated:
        out["truncated"] = {"kept": len(body), "of": len(df)}
    return out


def encode(obj: Any, field: str = "value", *, rows: int = MAX_EVIDENCE_ROWS,
           items: int = MAX_EVIDENCE_ITEMS, chars: int = MAX_EVIDENCE_CHARS,
           _depth: int = 0) -> Any:
    """Anything a guard put in `evidence` -> JSON-safe scalars and small tables.

    Series and frames become the payload shape, truncated to `items` / `rows`; numpy
    scalars become Python ones; non-finite floats become null / 'Infinity'; anything with
    no JSON form at all becomes its repr, truncated. Never raises.
    """
    if _depth > 6:
        return _clip(repr(obj), chars)
    if obj is None or isinstance(obj, (bool, str)):
        return _clip(obj, chars) if isinstance(obj, str) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _json_number(obj)
    if isinstance(obj, complex):
        return {"real": _json_number(obj.real), "imag": _json_number(obj.imag)}
    if isinstance(obj, pd.Series):
        return series_to_payload(obj, limit=items)
    if isinstance(obj, pd.DataFrame):
        return frame_to_payload(obj, limit=rows)
    if isinstance(obj, pd.Index):
        return {"index": _index_to_json(obj[:items]), "n": int(len(obj))}
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.ndarray):
        if obj.ndim <= 1:
            return [encode(v, field, rows=rows, items=items, chars=chars, _depth=_depth + 1)
                    for v in obj[:items].tolist()]
        return frame_to_payload(pd.DataFrame(obj), limit=rows)
    if isinstance(obj, Mapping):
        return {str(k): encode(v, field, rows=rows, items=items, chars=chars, _depth=_depth + 1)
                for k, v in list(obj.items())[:items]}
    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = list(obj)
        out = [encode(v, field, rows=rows, items=items, chars=chars, _depth=_depth + 1)
               for v in seq[:items]]
        if len(seq) > items:
            out.append(f"... {len(seq) - items} more")
        return out
    for attr in ("_asdict", "to_dict"):                      # namedtuple / small containers
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return encode(fn(), field, rows=rows, items=items, chars=chars,
                              _depth=_depth + 1)
            except Exception:                                 # noqa: BLE001 - evidence only
                break
    if hasattr(obj, "__dataclass_fields__"):
        return {f: encode(getattr(obj, f, None), field, rows=rows, items=items, chars=chars,
                          _depth=_depth + 1) for f in obj.__dataclass_fields__}
    return _clip(repr(obj), chars)


def _clip(text: Any, chars: int) -> str:
    s = str(text)
    return s if len(s) <= chars else s[:chars] + f"... [{len(s) - chars} more chars]"


# ------------------------------------------------------------------------------- decode
def looks_like_series(value: Any) -> bool:
    return (isinstance(value, Mapping) and "values" in value
            and isinstance(value.get("values"), list)
            and set(value) <= _SERIES_KEYS | {"truncated"})


def looks_like_frame(value: Any) -> bool:
    return (isinstance(value, Mapping) and "records" in value
            and isinstance(value.get("records"), list)
            and set(value) <= _FRAME_KEYS | {"truncated"})


_NEAR = ("values", "records", "index", "columns")


def _reject_near_miss(value: Any, field: str) -> None:
    """A mapping that ALMOST spells a payload is a typo, not a mapping the guard wanted.

    `{"vals": [...]}` would otherwise be handed to the guard as a dict and come back as
    'returns must be a Series', which does not say what is wrong. Only fires on a key that
    is close to a payload key, so a real mapping input (greeks, a bar, a universe) passes.
    """
    import difflib

    if not isinstance(value, Mapping) or not value:
        return
    keys = [k for k in value if isinstance(k, str)]
    if len(keys) != len(value):
        return
    for k in keys:
        if k in _NEAR:
            continue
        close = difflib.get_close_matches(k, _NEAR, n=1, cutoff=0.75)
        if close:
            raise TypeError(
                f"{field}: key {k!r} looks like a mistyped {close[0]!r}. A series is "
                f"{{'index': [...], 'values': [...]}} and a frame is {{'index': [...], "
                f"'records': [{{col: value}}, ...]}}; got keys {sorted(keys)}")


def _check_points(n: int, field: str, what: str) -> None:
    if n > MAX_POINTS:
        raise TypeError(f"{field}: {what} has {n} cell(s); the limit is "
                        f"{MAX_POINTS} (fin_skills.tools.payloads.MAX_POINTS). Aggregate or "
                        f"slice the window before sending it")


def to_series(value: Any, field: str) -> pd.Series:
    """A payload (or a bare list) -> pd.Series. TypeError names `field` on anything else."""
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, list):
        value = {"values": value}
    if not isinstance(value, Mapping) or not isinstance(value.get("values"), list):
        raise TypeError(f"{field} must be a series payload {{'index': [...], 'values': [...]}} "
                        f"or a list of numbers, got {type(value).__name__}")
    bad = sorted(set(value) - _SERIES_KEYS - {"truncated"})
    if bad:
        raise TypeError(f"{field}: unknown key(s) {bad} in the series payload; "
                        f"allowed: {sorted(_SERIES_KEYS)}")
    raw = [_from_json_number(v) for v in value["values"]]
    _check_points(len(raw), field, "series")
    name = value.get("name")
    idx = _index_from_json(value.get("index"), field, len(raw), value.get("freq"))
    try:
        s = pd.Series(raw, index=idx, name=name)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field}: cannot build a Series - {exc}") from exc
    dtype = value.get("dtype")
    if dtype:
        try:
            s = s.astype(dtype)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field}: dtype {dtype!r} does not fit the values - {exc}") from exc
    return s


def to_frame(value: Any, field: str) -> pd.DataFrame:
    """A payload (or a bare list of records) -> pd.DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, list):
        value = {"records": value}
    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list):
        raise TypeError(f"{field} must be a frame payload {{'index': [...], 'records': "
                        f"[{{col: value}}, ...]}} or a list of records, got "
                        f"{type(value).__name__}")
    bad = sorted(set(value) - _FRAME_KEYS - {"truncated"})
    if bad:
        raise TypeError(f"{field}: unknown key(s) {bad} in the frame payload; "
                        f"allowed: {sorted(_FRAME_KEYS)}")
    records = value["records"]
    for i, r in enumerate(records):
        if not isinstance(r, Mapping):
            raise TypeError(f"{field}: record {i} is a {type(r).__name__}, expected an object "
                            f"of {{column: value}}")
    cols = list(value.get("columns") or (list(records[0]) if records else []))
    _check_points(len(records) * max(len(cols), 1), field, "frame")
    rows = [{c: _from_json_number(r.get(c)) for c in cols} for r in records]
    idx = _index_from_json(value.get("index"), field, len(rows), value.get("freq"),
                           name=value.get("index_name"))
    try:
        df = pd.DataFrame(rows, columns=cols, index=idx)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field}: cannot build a DataFrame - {exc}") from exc
    for col, dtype in (value.get("dtypes") or {}).items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{field}: column {col!r} does not fit dtype {dtype!r} "
                                f"- {exc}") from exc
    return df


def decode(value: Any, field: str, kind: str = "any") -> Any:
    """JSON -> the Python value a guard wants.

    `kind` is the declared kind from the schema ('series', 'frame', or anything else). A
    payload-shaped object is converted whatever the declared kind, so a guard whose input
    is annotated `Any` (purge_effect's `x`, regime_lookahead's `p_calm`) still receives a
    real Series or frame. Lists and mappings are walked, so a sequence of frames
    (brinson_attribution's `panels`) converts element by element.
    """
    if kind == "series":
        return to_series(value, field)
    if kind == "frame":
        return to_frame(value, field)
    if looks_like_series(value):
        return to_series(value, field)
    if looks_like_frame(value):
        return to_frame(value, field)
    _reject_near_miss(value, field)
    if isinstance(value, list):
        return [decode(v, f"{field}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        return {k: decode(v, f"{field}[{k!r}]") for k, v in value.items()}
    if isinstance(value, str) and value in (_POS_INF, _NEG_INF):
        return _from_json_number(value)
    return value


def check_size(arguments: Mapping[str, Any]) -> int:
    """Reject an oversized `arguments` object before anything is decoded. Returns its size."""
    try:
        size = len(json.dumps(arguments, default=str, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"arguments are not JSON: {exc}") from exc
    if size > MAX_ARGUMENT_BYTES:
        raise TypeError(f"arguments are {size} bytes as JSON; the limit is "
                        f"{MAX_ARGUMENT_BYTES} (fin_skills.tools.payloads."
                        f"MAX_ARGUMENT_BYTES). Send a shorter window")
    return size


def is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


__all__ = ["FRAME_SCHEMA", "MAX_ARGUMENT_BYTES", "MAX_EVIDENCE_CHARS", "MAX_EVIDENCE_ITEMS",
           "MAX_EVIDENCE_ROWS", "MAX_POINTS", "SERIES_SCHEMA", "check_size", "decode",
           "encode", "frame_to_payload", "is_number", "looks_like_frame", "looks_like_series",
           "series_to_payload", "to_frame", "to_series"]
