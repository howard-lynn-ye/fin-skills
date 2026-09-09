"""A content-addressed store that treats a changed refetch as EVIDENCE, not as an update.

Three properties, each of which exists because of a specific failure this repo has already
recorded:

  * **It sits above the client, not inside the session.** yfinance now RAISES on a
    `requests_cache` session (`YFDataException: Caching sessions ... are not supported`),
    so the old HTTP-cache recipe is dead for the layer's first adapter. Caching the
    normalised RESULT is the only place left that works for every vendor.

  * **`put()` refuses on a `no-persist` provider.** Tiingo's Starter plan forbids
    retaining the data in "any persistent or durable storage" (ToS 1.6a). One global cache
    applied to every provider would put free-tier users in breach, so the policy is per
    adapter and the refusal is an exception, not a warning.

  * **`refetch()` never overwrites.** A forward-adjusted series is anchored at the present,
    so every new dividend rewrites the whole history and the same query run a month later
    returns different numbers. The cache keeps both vintages and returns a `Divergence`,
    whose (close, other) pair is exactly the input shape of the `reconcile_sources` guard.

The on-disk format is CSV plus a JSON sidecar, on purpose. Parquet does not store
timezones by design and CSV degrades `America/New_York` to a fixed `UTC-05:00`, destroying
DST - so the zone, the dtypes and every declaration come back from the SIDECAR and the
layer never trusts the data file for them.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from fin_skills.api.base import Finding
from fin_skills.data import declare
from fin_skills.data.provenance import (Provenance, canonical_json, content_hash, scrub,
                                        utc_now)
from fin_skills.data.schema import Adjustment, Bars, Fundamentals, Macro

META = "meta.json"


class CachePolicyError(PermissionError):
    """Raised when a provider's terms forbid the write that was attempted."""


# -------------------------------------------------------------------------- Divergence
@dataclass(frozen=True)
class Divergence:
    """What changed between two vintages of the same request, kept as evidence."""

    key: str
    old: Provenance
    new: Provenance
    n_changed: int
    first_changed: pd.Timestamp | None
    max_abs_bps: float
    old_values: pd.Series
    new_values: pd.Series
    actions: pd.DataFrame | None = None
    ticker: str = ""

    @property
    def changed(self) -> bool:
        return self.n_changed > 0

    def to_bundle(self):
        """-> `reconcile_sources` (and `adjustment_check`, when actions came along).

        `close` is the vintage that was on disk and `other` is what the vendor returns
        now. A guard that says DATA ERROR here is saying the two disagree at a date with
        no corporate action near it, which is the shape of a silent vendor correction.
        """
        from fin_skills.api.bundle import Bundle                     # noqa: PLC0415
        slots: dict[str, Any] = {"close": self.old_values, "other": self.new_values}
        if self.actions is not None and len(self.actions):
            slots["actions"] = self.actions
        return Bundle(**slots)

    def report(self) -> str:
        if not self.changed:
            return (f"{self.key[:12]}  UNCHANGED  {self.old.retrieved_at} -> "
                    f"{self.new.retrieved_at}")
        first = self.first_changed.date() if self.first_changed is not None else "?"
        return (f"{self.key[:12]}  {self.n_changed} cell(s) changed, first {first}, "
                f"max {self.max_abs_bps:.1f} bps   {self.old.retrieved_at} -> "
                f"{self.new.retrieved_at}\n"
                f"  both vintages retained; nothing was overwritten")

    def __str__(self) -> str:
        return self.report()


# ------------------------------------------------------------------------- (de)serialise
def _col_key(col: Any) -> str:
    return "\x1f".join(str(c) for c in col) if isinstance(col, tuple) else str(col)


def _dump_frame(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Write one frame and return the sidecar entry that lets it be read back exactly."""
    df.to_csv(path, encoding="utf-8")
    idx = df.index
    return {
        "columns_nlevels": int(getattr(df.columns, "nlevels", 1)),
        "column_names": [None if n is None else str(n)
                         for n in list(getattr(df.columns, "names", [None]))],
        "index_name": None if idx.name is None else str(idx.name),
        "index_is_datetime": isinstance(idx, pd.DatetimeIndex),
        "index_tz": str(idx.tz) if getattr(idx, "tz", None) is not None else None,
        "dtypes": {_col_key(c): str(df[c].dtype) for c in df.columns},
        "shape": [int(df.shape[0]), int(df.shape[1])],
    }


def _load_frame(path: Path, meta: dict[str, Any]) -> pd.DataFrame:
    nlev = int(meta.get("columns_nlevels", 1))
    header = list(range(nlev)) if nlev > 1 else 0
    want = meta.get("dtypes", {})

    # A zero-padded identifier is a number to a CSV reader: a CIK of "0000320193" comes
    # back as 320193 and stops matching anything. The sidecar says which columns were
    # text, so they are read as text - the file is never trusted to say so itself.
    probe = pd.read_csv(path, header=header, index_col=0, nrows=0, encoding="utf-8")
    as_text = {lbl: "object" for lbl in probe.columns
               if want.get(_col_key(lbl)) == "object"}

    # float_precision="round_trip" is not decoration either: the default C parser is a
    # fast approximate one and loses about 1e-14 relative, which is enough to change a
    # content hash and turn every verify() into a false alarm.
    df = pd.read_csv(path, header=header, index_col=0, encoding="utf-8",
                     float_precision="round_trip", dtype=as_text or None)
    if nlev > 1:
        df.columns = pd.MultiIndex.from_tuples(list(df.columns),
                                               names=meta.get("column_names"))
    if meta.get("index_is_datetime"):
        tz = meta.get("index_tz")
        if tz:
            # the file's fixed offset is not the zone; the SIDECAR carries the zone
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
        else:
            df.index = pd.to_datetime(df.index)
    df.index.name = meta.get("index_name")
    for col in df.columns:
        want = meta.get("dtypes", {}).get(_col_key(col))
        if want and str(df[col].dtype) != want:
            try:
                df[col] = df[col].astype(want)
            except (TypeError, ValueError):
                pass
    return df


def _dump(obj: Any, folder: Path) -> dict[str, Any]:
    """Write an artefact into `folder` and return its sidecar metadata."""
    meta: dict[str, Any] = {}
    if isinstance(obj, Bars):
        meta["kind"] = "bars"
        meta["frame"] = _dump_frame(obj.frame, folder / "frame.csv")
        meta["schema"] = {"adjustment": obj.adjustment.value, "calendar": obj.calendar,
                          "tz": obj.tz, "interval": obj.interval,
                          "bar_label": obj.bar_label, "currency": obj.currency,
                          "half_open": bool(obj.half_open)}
        if obj.actions is not None:
            meta["actions"] = _dump_frame(obj.actions, folder / "actions.csv")
        if obj.listings is not None:
            meta["listings"] = _dump_frame(obj.listings, folder / "listings.csv")
    elif isinstance(obj, Fundamentals):
        meta["kind"] = "fundamentals"
        meta["frame"] = _dump_frame(obj.frame, folder / "frame.csv")
        meta["schema"] = {"tz_of_record": obj.tz_of_record}
    elif isinstance(obj, Macro):
        meta["kind"] = "macro"
        meta["frame"] = _dump_frame(obj.frame, folder / "frame.csv")
        meta["schema"] = {}
    elif isinstance(obj, pd.Series):
        meta["kind"] = "series"
        meta["series_name"] = None if obj.name is None else str(obj.name)
        meta["frame"] = _dump_frame(obj.to_frame(name="value"), folder / "frame.csv")
        meta["schema"] = {}
    elif isinstance(obj, pd.DataFrame):
        meta["kind"] = "frame"
        meta["frame"] = _dump_frame(obj, folder / "frame.csv")
        meta["schema"] = {}
    else:
        raise TypeError(f"the cache stores Bars, Fundamentals, Macro, DataFrame or "
                        f"Series, not {type(obj).__name__}")
    return meta


def _load(folder: Path, meta: dict[str, Any], prov: Provenance) -> Any:
    kind = meta["kind"]
    frame = _load_frame(folder / "frame.csv", meta["frame"])
    sch = meta.get("schema", {})
    if kind == "bars":
        actions = (_load_frame(folder / "actions.csv", meta["actions"])
                   if "actions" in meta else None)
        listings = (_load_frame(folder / "listings.csv", meta["listings"])
                    if "listings" in meta else None)
        return Bars(frame=frame, adjustment=Adjustment(sch["adjustment"]),
                    calendar=sch["calendar"], tz=sch["tz"], interval=sch["interval"],
                    bar_label=sch["bar_label"], currency=sch["currency"],
                    provenance=prov, actions=actions, listings=listings,
                    half_open=bool(sch.get("half_open", True)))
    if kind == "fundamentals":
        return Fundamentals(frame=frame, provenance=prov,
                            tz_of_record=sch.get("tz_of_record", "UTC"))
    if kind == "macro":
        return Macro(frame=frame, provenance=prov)
    if kind == "series":
        s = frame["value"]
        return s.rename(meta.get("series_name"))
    return frame


def _content_of(obj: Any) -> Any:
    """The array a Provenance hash covers - the frame, for the three schema objects."""
    return obj.frame if isinstance(obj, (Bars, Fundamentals, Macro)) else obj


# -------------------------------------------------------------------------------- Cache
class Cache:
    """Content-addressed, vintage-keeping, and it never repairs anything it stored."""

    def __init__(self, root: str | Path, *, keep_vintages: bool = True) -> None:
        self.root = Path(root)
        self.keep_vintages = bool(keep_vintages)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------ keys
    def key(self, request: dict, decl: "declare.Declaration | str") -> str:
        """The address of one request. Scrubbed first, so a key can never encode a key."""
        name = decl if isinstance(decl, str) else decl.name
        payload = {"adapter": str(name).split(":", 1)[0], "request": scrub(dict(request))}
        return content_hash(payload)[:32]

    def _folder(self, key: str) -> Path:
        return self.root / key

    def vintages(self, key: str) -> list[Path]:
        folder = self._folder(key)
        if not folder.is_dir():
            return []
        return sorted((p for p in folder.iterdir() if p.is_dir() and (p / META).exists()),
                      key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1)

    # ------------------------------------------------------------------------- read
    def get(self, key: str) -> tuple[Any, Provenance] | None:
        """The latest vintage stored under `key`, or None."""
        vs = self.vintages(key)
        if not vs:
            return None
        return self._read(vs[-1])

    def get_vintage(self, key: str, n: int) -> tuple[Any, Provenance] | None:
        vs = self.vintages(key)
        if not (0 <= n < len(vs)):
            return None
        return self._read(vs[n])

    def _read(self, folder: Path) -> tuple[Any, Provenance]:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
        prov = Provenance.from_dict(meta["provenance"])
        return _load(folder, meta, prov), prov

    # ------------------------------------------------------------------------ write
    def put(self, obj: Any, prov: Provenance, *,
            acknowledge_paid_tier: bool = False,
            decl: "declare.Declaration | None" = None,
            key: str | None = None) -> str:
        """Store an artefact and return its key.

        Refuses when the source declares `cache_policy="no-persist"`. Passing
        `acknowledge_paid_tier=True` is the caller stating, on the record, that their plan
        permits retention - the library cannot check a plan, so it makes the claim
        explicit instead of guessing.
        """
        if decl is None and declare.registered(prov.source):
            decl = declare.lookup(prov.source)
        if decl is not None and decl.cache_policy == "no-persist" \
                and not acknowledge_paid_tier:
            raise CachePolicyError(
                f"{decl.name} declares cache_policy='no-persist': its terms forbid "
                f"retaining this data in any persistent or durable storage, so writing it "
                f"to disk would put a free-tier user in breach. If your plan permits "
                f"retention, say so explicitly:\n"
                f"    cache.put(obj, prov, acknowledge_paid_tier=True)\n"
                f"terms: {decl.terms_url}")

        # `key` is passed only by refetch(), which must land the new vintage NEXT TO the
        # old one even if the adapter's normalisation of the request has since changed
        key = key or self.key(prov.request, decl.name if decl is not None else prov.source)
        folder = self._folder(key)
        existing = self.vintages(key)
        if existing and not self.keep_vintages:
            for p in existing:
                shutil.rmtree(p)
            existing = []
        n = (int(existing[-1].name[1:]) + 1) if existing else 0
        target = folder / f"v{n}"
        target.mkdir(parents=True, exist_ok=True)
        meta = _dump(obj, target)
        meta["provenance"] = prov.to_dict()
        meta["key"] = key
        meta["written_at"] = utc_now()
        meta["cache_policy"] = decl.cache_policy if decl is not None else "unstated"
        meta["includes_delisted"] = bool(decl.includes_delisted) if decl is not None else None
        meta["terms_url"] = decl.terms_url if decl is not None else prov.terms_url
        (target / META).write_text(canonical_json(meta), encoding="utf-8")
        return key

    # --------------------------------------------------------------------- manifest
    def manifest(self) -> pd.DataFrame:
        """One row per stored artefact - the table `result_manifest` wants and the one
        `python -m fin_skills.data manifest` prints."""
        rows = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for i, v in enumerate(self.vintages(folder.name)):
                meta = json.loads((v / META).read_text(encoding="utf-8"))
                prov = meta["provenance"]
                fmeta = meta.get("frame", {})
                span = self._span(v, meta)
                rows.append({
                    "key": meta.get("key", folder.name), "vintage": i,
                    "source": prov.get("source", ""), "kind": meta.get("kind", ""),
                    "retrieved_at": prov.get("retrieved_at", ""),
                    "content_sha256": prov.get("content_sha256", "")[:16],
                    "rows": (fmeta.get("shape") or [0, 0])[0],
                    "cols": (fmeta.get("shape") or [0, 0])[1],
                    "span": span,
                    "adjustment": meta.get("schema", {}).get("adjustment", ""),
                    "includes_delisted": meta.get("includes_delisted"),
                    "cache_policy": meta.get("cache_policy", ""),
                    "terms_url": meta.get("terms_url", ""),
                })
        cols = ["key", "vintage", "source", "kind", "retrieved_at", "content_sha256",
                "rows", "cols", "span", "adjustment", "includes_delisted",
                "cache_policy", "terms_url"]
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _span(folder: Path, meta: dict[str, Any]) -> str:
        try:
            df = _load_frame(folder / "frame.csv", meta["frame"])
        except (OSError, ValueError, KeyError):
            return ""
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            return f"{df.index.min().date()}..{df.index.max().date()}"
        return f"{len(df)} rows"

    # ----------------------------------------------------------------------- verify
    def verify(self) -> list[Finding]:
        """Re-hash everything on disk. A mismatch means the file changed under you."""
        out: list[Finding] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for v in self.vintages(folder.name):
                try:
                    obj, prov = self._read(v)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    out.append(Finding("error", f"unreadable artefact: {exc}",
                                       str(v.relative_to(self.root))))
                    continue
                actual = content_hash(_content_of(obj))
                where = str(v.relative_to(self.root))
                if actual != prov.content_sha256:
                    out.append(Finding(
                        "error",
                        f"content hash mismatch: stored {prov.content_sha256[:16]}, "
                        f"recomputed {actual[:16]} - the bytes changed after they were "
                        f"written, so this artefact is not what its provenance claims",
                        where))
                else:
                    out.append(Finding("info", f"{prov.source} {actual[:12]} verified",
                                       where))
        return out

    # ---------------------------------------------------------------------- refetch
    def refetch(self, key: str, adapter: Any, *,
                acknowledge_paid_tier: bool = False) -> Divergence:
        """Pull the same request again and COMPARE, keeping both vintages.

        `adapter` is anything with a `replay(request)` method (every adapter here has
        one) or a plain callable taking the normalised request. The stored vintage is
        never overwritten: a changed series is evidence about the vendor, and deleting
        the old copy destroys the only proof.
        """
        found = self.get(key)
        if found is None:
            raise KeyError(f"nothing cached under {key!r}")
        old_obj, old_prov = found
        request = dict(old_prov.request)

        replay = getattr(adapter, "replay", None)
        result = replay(request) if callable(replay) else adapter(request)
        if isinstance(result, tuple):
            new_obj, new_prov = result
        else:
            new_obj = result
            new_prov = getattr(new_obj, "provenance", None)
        if new_prov is None:
            raise TypeError("refetch needs a Provenance: return (obj, prov) or an object "
                            "carrying .provenance")

        self.put(new_obj, new_prov, acknowledge_paid_tier=acknowledge_paid_tier, key=key)
        return diverge(key, old_obj, old_prov, new_obj, new_prov)

    def __repr__(self) -> str:
        n = sum(len(self.vintages(p.name)) for p in self.root.iterdir() if p.is_dir())
        return f"Cache({str(self.root)!r}, {n} artefact vintage(s))"


# --------------------------------------------------------------------------- comparison
def _comparable(obj: Any, ticker: str | None = None) -> tuple[pd.Series, str]:
    """The one series a divergence is measured on, and the ticker it belongs to.

    A Bars series is put on the same index `convert.to_bundle()` uses, so a Divergence's
    `other` can be dropped into a bundle built from the same panel and still overlap it.
    """
    if isinstance(obj, Bars):
        from fin_skills.data.convert import _to_guard_index               # noqa: PLC0415
        wide = obj.field("close")
        name = ticker or (wide.columns[0] if wide.shape[1] else "")
        return _to_guard_index(obj, wide[name].astype(float)), str(name)
    if isinstance(obj, Macro):
        return obj.latest(), obj.series_ids[0] if obj.series_ids else ""
    if isinstance(obj, Fundamentals):
        f = obj.frame.sort_values("available_at")
        return f.set_index("available_at")["value"].astype(float), ""
    if isinstance(obj, pd.Series):
        return obj.astype(float), str(obj.name or "")
    if isinstance(obj, pd.DataFrame):
        col = ticker or obj.columns[0]
        return obj[col].astype(float), str(col)
    raise TypeError(f"cannot compare a {type(obj).__name__}")


def diverge(key: str, old_obj: Any, old_prov: Provenance, new_obj: Any,
            new_prov: Provenance, *, ticker: str | None = None) -> Divergence:
    """Measure the difference between two vintages of one artefact."""
    old_s, name = _comparable(old_obj, ticker)
    new_s, _ = _comparable(new_obj, ticker)
    common = old_s.dropna().index.intersection(new_s.dropna().index)
    if len(common):
        a, b = old_s.loc[common], new_s.loc[common]
        rel = (b - a) / a.replace(0.0, np.nan)
        changed = rel.abs() > 1e-9
        n_changed = int(changed.sum())
        first = common[changed.to_numpy(na_value=False)][0] if n_changed else None
        max_bps = float(rel.abs().max() * 1e4) if n_changed else 0.0
    else:
        n_changed, first, max_bps = 0, None, 0.0

    actions = None
    for src in (old_obj, new_obj):
        if isinstance(src, Bars) and src.actions is not None:
            from fin_skills.data.convert import actions_table               # noqa: PLC0415
            actions = actions_table(src, ticker=name or None)
            break
    return Divergence(key=key, old=old_prov, new=new_prov, n_changed=n_changed,
                      first_changed=first, max_abs_bps=max_bps,
                      old_values=old_s.rename("close"), new_values=new_s.rename("other"),
                      actions=actions, ticker=name)


__all__ = ["Cache", "CachePolicyError", "Divergence", "diverge"]
