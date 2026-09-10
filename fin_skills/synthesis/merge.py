"""Combining sources of the SAME kind that disagree - with the pick written down.

`reconcile_sources` answers "why do these two price series differ": convention, corporate
action, or bad data. It does not produce a series. This module is the step after: given
several vendors' versions of one number, produce ONE series, and leave a record of every
place they disagreed and which one was taken.

The idiom it replaces is `a.combine_first(b)`. `combine_first` resolves by NULLITY: it
takes `a` wherever `a` is not NaN and `b` elsewhere. That is not a precedence rule, because
a vendor that prints a WRONG number is not printing a null - a stale repeat of yesterday's
close, a decimal in the wrong place and a genuinely correct quote are all "not NaN". So:

  * whichever frame you happen to write first wins every overlapping disagreement, and
    the two orderings of the same two vendors give different answers with no error;
  * the disagreements themselves are never reported. A silent pick is not a merge, it is
    a coin flip with a deterministic seed.

Two refusals, both of which are about a merge that CANNOT be correct rather than one that
happens to be wrong:

  * **Declared `Adjustment` differs.** A raw series and an `ANCHORED_PRESENT` one are not
    two measurements of one quantity; splicing them puts the entire split/dividend factor
    into a single day's return. Re-adjust first (`Bars.readjust`) and merge afterwards.
  * **Symbology is unresolved, or the entities differ.** A ticker is a slot in an
    exchange's namespace and is reassigned after a delisting. Merging two vendors on a
    ticker merges whatever each of them meant by it on each date, which is a different
    question from whether their numbers agree. Resolve to a permanent id first.

Every number this module produces is a `MergeResult`; `MergeResult.to_facts()` turns it
into `timeline.Fact`s that carry the per-source facts as `inputs`, so the availability
rule and `Dossier.explain()` keep working across the merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from fin_skills.data.provenance import Provenance
from fin_skills.data.schema import Adjustment
from fin_skills.synthesis.timeline import Entity, Fact, Timeline, as_entity

#: how a disagreement beyond the threshold is handled
RESOLUTIONS: tuple[str, ...] = ("record", "raise", "drop")


class MergeRefused(ValueError):
    """A merge that cannot be correct, refused before it produces a number."""


# ---------------------------------------------------------------------- source series
@dataclass(frozen=True)
class SourceSeries:
    """One vendor's version of ONE series, with the three declarations a merge needs.

    `available_lag` is how long after an observation's own timestamp that vendor's value
    became knowable - zero for a real-time feed, a session for a vendor that publishes
    after the close, longer for a redistributor. It is what `to_facts()` puts in the
    `available_at` clock, so the merge cannot quietly make a slow source look fast.
    """

    name: str
    values: pd.Series
    adjustment: Adjustment
    entity: Entity
    available_lag: pd.Timedelta = pd.Timedelta(0)
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("SourceSeries needs a non-empty name")
        if not isinstance(self.values, pd.Series):
            raise TypeError(f"SourceSeries.values must be a Series, got "
                            f"{type(self.values).__name__}")
        if not isinstance(self.values.index, pd.DatetimeIndex):
            raise TypeError(f"{self.name}: values must be indexed by a DatetimeIndex")
        if not self.values.index.is_monotonic_increasing:
            raise ValueError(f"{self.name}: values index is unsorted; every downstream "
                             f"guard assumes sorted time series")
        if self.values.index.has_duplicates:
            raise ValueError(f"{self.name}: values index has duplicate timestamps; two "
                             f"prints on one label is a vendor merge error, not data")
        object.__setattr__(self, "adjustment", Adjustment(self.adjustment))
        object.__setattr__(self, "entity", as_entity(self.entity))
        object.__setattr__(self, "available_lag", pd.Timedelta(self.available_lag))

    def available_at(self, ts: Any) -> pd.Timestamp:
        return pd.Timestamp(ts) + self.available_lag

    def __repr__(self) -> str:
        return (f"SourceSeries({self.name!r}, n={len(self.values)}, "
                f"adjustment={self.adjustment.value!r}, entity={self.entity.key!r}, "
                f"lag={self.available_lag})")


# ----------------------------------------------------------------------- disagreement
@dataclass(frozen=True)
class Disagreement:
    """One timestamp where the sources did not agree, and what was taken instead.

    This record is the whole difference between a merge and a coin flip. `rejected` keeps
    the numbers that were NOT used, so the decision can be re-examined without refetching.
    """

    field: str
    entity: str
    at: pd.Timestamp
    chosen: str
    chosen_value: float
    rejected: tuple[tuple[str, float], ...]
    spread_bps: float
    rule: str = "precedence"

    def __str__(self) -> str:
        others = ", ".join(f"{n}={v:.6g}" for n, v in self.rejected)
        return (f"{self.at.date()} {self.field}[{self.entity}] {self.spread_bps:.1f}bps: "
                f"took {self.chosen}={self.chosen_value:.6g} over {others} ({self.rule})")


@dataclass(frozen=True)
class MergePolicy:
    """The declared source ranking and the threshold beyond which sources 'disagree'.

    `precedence` is a RANKING, not a preference order discovered at runtime: it is the
    caller stating, before seeing the data, which source is authoritative for this
    instrument. Every source in the merge must appear in it, so a vendor cannot enter a
    result without anyone having ranked it.
    """

    precedence: tuple[str, ...]
    tol_bps: float = 10.0
    on_disagreement: str = "record"
    require_resolved_entity: bool = True

    def __post_init__(self) -> None:
        order = tuple(str(p) for p in self.precedence)
        if not order:
            raise ValueError("MergePolicy needs a non-empty precedence ranking; a merge "
                             "with no declared ranking resolves by argument order, which "
                             "is what combine_first already does")
        if len(set(order)) != len(order):
            raise ValueError(f"precedence lists a source twice: {order}")
        object.__setattr__(self, "precedence", order)
        if float(self.tol_bps) < 0:
            raise ValueError("tol_bps must be >= 0")
        object.__setattr__(self, "tol_bps", float(self.tol_bps))
        if self.on_disagreement not in RESOLUTIONS:
            raise ValueError(f"on_disagreement must be one of {list(RESOLUTIONS)}, got "
                             f"{self.on_disagreement!r}")

    def rank(self, name: str) -> int:
        return self.precedence.index(name)


# ------------------------------------------------------------------------ merge result
@dataclass(frozen=True)
class MergeResult:
    """The merged series plus everything needed to defend it."""

    field: str
    entity: Entity
    values: pd.Series
    chosen: pd.Series
    disagreements: tuple[Disagreement, ...]
    adjustment: Adjustment
    policy: MergePolicy
    sources: tuple[SourceSeries, ...] = dc_field(default=(), repr=False)
    n_overlap: int = 0
    n_dropped: int = 0

    @property
    def agreement_rate(self) -> float:
        """Share of timestamps where every present source agreed within the threshold."""
        if not self.n_overlap:
            return float("nan")
        return 1.0 - len(self.disagreements) / float(self.n_overlap)

    def taken_from(self) -> pd.Series:
        """How many timestamps each source supplied."""
        return self.chosen.value_counts().sort_index()

    # ------------------------------------------------------------------- to facts
    def to_facts(self, *, kind: str = "price", at: Any = None) -> list[Fact]:
        """One combined Fact per timestamp, carrying the per-source facts as `inputs`.

        The combined fact's clock is `max` over the sources that were actually present at
        that timestamp - so a value that only a slow redistributor had is knowable when
        THAT vendor published it, not when the fast one printed.
        """
        stamps = self.values.index if at is None else pd.DatetimeIndex([pd.Timestamp(at)])
        by_name = {s.name: s for s in self.sources}
        dis_at: dict[pd.Timestamp, list[Disagreement]] = {}
        for d in self.disagreements:
            dis_at.setdefault(d.at, []).append(d)
        out: list[Fact] = []
        for ts in stamps:
            if ts not in self.values.index:
                continue
            val = self.values.loc[ts]
            if pd.isna(val):
                continue
            inputs = []
            for name in self.policy.precedence:
                src = by_name.get(name)
                if src is None or ts not in src.values.index:
                    continue
                v = src.values.loc[ts]
                if pd.isna(v):
                    continue
                inputs.append(Fact(field=self.field, entity=src.entity, value=float(v),
                                   period_end=ts, filed_at=ts,
                                   available_at=src.available_at(ts), kind=kind,
                                   source=src.name, provenance=src.provenance))
            if not inputs:                                          # pragma: no cover
                continue
            latest = max(f.available_at for f in inputs)
            out.append(Fact(field=self.field, entity=self.entity, value=float(val),
                            period_end=ts, filed_at=ts, available_at=latest, kind=kind,
                            source=f"merge({'>'.join(self.policy.precedence)})",
                            inputs=tuple(inputs),
                            disagreements=tuple(dis_at.get(ts, ())),
                            note=f"adjustment={self.adjustment.value}"))
        return out

    def to_timeline(self, timeline: Timeline | None = None, **kwargs: Any) -> Timeline:
        tl = timeline if timeline is not None else Timeline()
        return tl.extend(self.to_facts(**kwargs))

    # --------------------------------------------------------------------- display
    def report(self, limit: int = 5) -> str:
        lines = [
            f"merge {self.field}[{self.entity.key}] on {self.adjustment.value}: "
            f"{int(self.values.notna().sum())}/{len(self.values)} timestamps filled",
            f"  precedence   : {' > '.join(self.policy.precedence)} "
            f"(threshold {self.policy.tol_bps:.1f} bps)",
            f"  taken from   : " + ", ".join(f"{k}={v}" for k, v in
                                             self.taken_from().items()),
            f"  overlap      : {self.n_overlap} timestamp(s) with 2+ sources present",
            f"  disagreements: {len(self.disagreements)} recorded "
            f"({100.0 * (1.0 - self.agreement_rate):.2f}% of overlap)"
            if self.n_overlap else "  disagreements: 0 (no overlap)",
        ]
        if self.n_dropped:
            lines.append(f"  dropped      : {self.n_dropped} timestamp(s) "
                         f"(on_disagreement='drop')")
        for d in self.disagreements[:limit]:
            lines.append(f"    {d}")
        if len(self.disagreements) > limit:
            lines.append(f"    ... {len(self.disagreements) - limit} more")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


# ------------------------------------------------------------------------ the refusals
def check_mergeable(sources: Sequence[SourceSeries], policy: MergePolicy) -> None:
    """Raise `MergeRefused` for the two merges that cannot be correct. Pure; no output."""
    if len(sources) < 2:
        raise MergeRefused("a merge needs at least two sources; with one, say so and use "
                           "it directly rather than calling this")
    unranked = [s.name for s in sources if s.name not in policy.precedence]
    if unranked:
        raise MergeRefused(
            f"source(s) {unranked} are not in the declared precedence "
            f"{list(policy.precedence)}. Rank every source before the merge runs: a vendor "
            f"that enters a result without anyone ranking it is a vendor nobody chose.")
    names = [s.name for s in sources]
    if len(set(names)) != len(names):
        raise MergeRefused(f"two sources share a name: {names}")

    adjustments = {s.adjustment for s in sources}
    if len(adjustments) > 1:
        detail = ", ".join(f"{s.name}={s.adjustment.value}" for s in sources)
        raise MergeRefused(
            f"declared Adjustment differs across sources ({detail}). These are not two "
            f"measurements of one quantity: splicing them puts the whole corporate-action "
            f"factor into a single day's return. Re-adjust to ONE convention first "
            f"(fin_skills.data.schema.Bars.readjust, which needs the actions table) and "
            f"merge the re-adjusted series.")

    if policy.require_resolved_entity:
        unresolved = [f"{s.name}->{s.entity.key}" for s in sources if not s.entity.is_resolved]
        if unresolved:
            raise MergeRefused(
                f"symbology is unresolved for {unresolved}. A ticker is a slot in an "
                f"exchange's namespace, not an entity: it is reassigned after a delisting "
                f"and it changes on a rename, so merging on it merges whatever each vendor "
                f"meant by it on each date. Resolve to a permanent id (cik, figi, permno, "
                f"isin) first, or pass require_resolved_entity=False and say why.")
    keys = {s.entity.key for s in sources}
    if len(keys) > 1:
        detail = ", ".join(f"{s.name}->{s.entity.key}" for s in sources)
        raise MergeRefused(
            f"the sources describe different entities ({detail}). Merging them would splice "
            f"two issuers' histories into one series; the join is not a disagreement to "
            f"resolve, it is a symbology boundary the caller has not crossed.")


# ------------------------------------------------------------------------------ merge
def merge_series(sources: Sequence[SourceSeries], policy: MergePolicy, *,
                 field: str = "close") -> MergeResult:
    """Merge same-kind series by DECLARED precedence, recording every disagreement.

    At each timestamp: take the highest-ranked source that has a number there; compare it
    with every other source that also has one; and when the spread exceeds
    `policy.tol_bps`, record a `Disagreement` (or raise, or drop the timestamp).
    """
    srcs = list(sources)
    check_mergeable(srcs, policy)
    srcs.sort(key=lambda s: policy.rank(s.name))

    index = srcs[0].values.index
    for s in srcs[1:]:
        index = index.union(s.values.index)
    index = pd.DatetimeIndex(index).sort_values()

    cols = {s.name: pd.to_numeric(s.values.reindex(index), errors="coerce") for s in srcs}
    wide = pd.DataFrame(cols, index=index)[[s.name for s in srcs]]
    present = wide.notna()
    n_present = present.sum(axis=1)

    values = np.full(len(index), np.nan)
    chosen: list[Any] = [None] * len(index)
    disagreements: list[Disagreement] = []
    n_dropped = 0
    n_overlap = int((n_present >= 2).sum())
    tol = policy.tol_bps / 1e4

    block = wide.to_numpy(dtype=float)
    order = [s.name for s in srcs]
    for i in range(len(index)):
        row = block[i]
        ok = np.flatnonzero(~np.isnan(row))
        if ok.size == 0:
            continue
        pick = int(ok[0])                       # sources are already in precedence order
        take = float(row[pick])
        values[i] = take
        chosen[i] = order[pick]
        if ok.size < 2:
            continue
        denom = abs(take) if abs(take) > 0 else 1.0
        rel = np.abs(row[ok] - take) / denom
        spread = float(rel.max())
        if spread <= tol:
            continue
        rejected = tuple((order[j], float(row[j])) for j in ok if j != pick)
        d = Disagreement(field=field, entity=srcs[0].entity.key, at=index[i],
                         chosen=order[pick], chosen_value=take, rejected=rejected,
                         spread_bps=spread * 1e4,
                         rule="precedence" if policy.on_disagreement != "drop" else "drop")
        if policy.on_disagreement == "raise":
            raise MergeRefused(
                f"sources disagree by {spread * 1e4:.1f} bps at {index[i].date()} "
                f"(threshold {policy.tol_bps:.1f}): {d}. on_disagreement='raise' was "
                f"asked for; use 'record' to take the ranked source and keep the evidence.")
        disagreements.append(d)
        if policy.on_disagreement == "drop":
            values[i] = np.nan
            chosen[i] = None
            n_dropped += 1

    merged = pd.Series(values, index=index, name=field)
    taken = pd.Series(chosen, index=index, name="source", dtype=object)
    return MergeResult(field=field, entity=srcs[0].entity, values=merged, chosen=taken,
                       disagreements=tuple(disagreements), adjustment=srcs[0].adjustment,
                       policy=policy, sources=tuple(srcs), n_overlap=n_overlap,
                       n_dropped=n_dropped)


# --------------------------------------------------------------------------- auditing
def unrecorded_disagreements(facts: Iterable[Fact], tol_bps: float = 10.0
                             ) -> list[tuple[Fact, float]]:
    """Merged facts whose inputs disagree beyond `tol_bps` with NO Disagreement recorded.

    The audit the `synthesis_integrity` guard runs: it recomputes the spread from the
    facts' own inputs, so a merge that resolved silently cannot hide behind its output.
    """
    out: list[tuple[Fact, float]] = []
    for f in facts:
        numeric = []
        for parent in f.inputs:
            if parent.field != f.field:
                continue
            try:
                numeric.append(float(parent.value))
            except (TypeError, ValueError):
                continue
        if len(numeric) < 2:
            continue
        arr = np.asarray(numeric, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            continue
        ref = float(f.value) if _is_number(f.value) else float(arr[0])
        denom = abs(ref) if abs(ref) > 0 else 1.0
        spread = float(arr.max() - arr.min()) / denom * 1e4
        if spread > tol_bps and not f.disagreements:
            out.append((f, spread))
    return out


def _is_number(value: Any) -> bool:
    try:
        return not np.isnan(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["Disagreement", "MergePolicy", "MergeRefused", "MergeResult", "RESOLUTIONS",
           "SourceSeries", "check_mergeable", "merge_series", "unrecorded_disagreements"]
