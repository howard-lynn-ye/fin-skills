"""The three contracts: Bars, Fundamentals, Macro - and the Adjustment enum they turn on.

Nothing here is inferred. Every field that has a library in the wild whose default is the
opposite of another's is a constructor argument with NO default, so the mistake surfaces as
a TypeError at construction instead of as a number in a result table:

    yfinance auto_adjust=True          vs  yahooquery adj_ohlc=False
    akshare adjust="" (raw)            vs  the qfq the ecosystem assumes
    pandas resample label="left"       vs  a close-stamped bar
    yfinance `end` exclusive           vs  every vendor whose `end` is inclusive

The schemas are derived from this repository's own trap tables - market-data-sourcing,
market-data-engineering and research-integrity-guards - and from the vendors' own
documentation. They are not derived from any other library's normalised schema.

Three things this layer will not do, ever: fill a gap, repair a price, or guess a
convention. A missing session is NaN and stays NaN, because a filled hole is
indistinguishable from data once it is in a file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from fin_skills.api.base import Finding
from fin_skills.data.provenance import Provenance, content_hash

OHLCV = ("open", "high", "low", "close", "volume")


class Adjustment(str, Enum):
    """Which corporate-action convention a price series is on.

    The English labels are not interchangeable across vendors, so the enum carries the
    behavioural question research-integrity-guards asks instead of the label.
    """

    RAW = "raw"                      # as quoted; splits and dividends are visible jumps
    BACK = "back"                    # anchored at the START (A-share hfq). History never
                                     # rewritten, so a cached copy stays valid.
    FORWARD = "forward"              # anchored at the PRESENT (A-share qfq, Yahoo
                                     # auto_adjust=True). Every new action rewrites history.
    RAW_PLUS_FACTORS = "raw+factors"  # raw prices plus a separate cumulative factor series
    UNKNOWN = "unknown"              # legal, and it POISONS the run: downstream guards
                                     # downgrade to warnings rather than pretend to know

    @property
    def rewrites_history(self) -> bool:
        """True for FORWARD.

        The behavioural test, not the label: when a new split or dividend occurs, does
        yesterday's stored value change? If yes, the series is not reproducible and the
        same query run a month later returns different numbers.
        """
        return self is Adjustment.FORWARD

    @property
    def is_declared(self) -> bool:
        """False only for UNKNOWN - the value that turns every guard into a warning."""
        return self is not Adjustment.UNKNOWN

    def __str__(self) -> str:
        return self.value


# --------------------------------------------------------------------------- helpers
def _as_adjustment(value: Any) -> Adjustment:
    if isinstance(value, Adjustment):
        return value
    try:
        return Adjustment(str(value))
    except ValueError as exc:
        raise ValueError(f"adjustment must be one of "
                         f"{[a.value for a in Adjustment]}, got {value!r}") from exc


def _require_columns(frame: pd.DataFrame, required: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what}.frame must be a DataFrame, got "
                        f"{type(frame).__name__}")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing required column(s) {missing}; a row without "
                         f"them is not point-in-time and this schema will not carry it")


def stack_fields(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the (field, ticker) MultiIndex frame `Bars` wants from wide per-field panels.

        stack_fields({"close": close_df, "volume": vol_df})
    """
    if not fields:
        raise ValueError("stack_fields needs at least one field")
    out = pd.concat(fields.values(), axis=1,
                    keys=[str(k).lower() for k in fields])
    out.columns = pd.MultiIndex.from_tuples(list(out.columns), names=["field", "ticker"])
    return out.sort_index(axis=1)


def _actions_frame(actions: pd.DataFrame) -> pd.DataFrame:
    a = actions.copy()
    for col in ("date", "ratio"):
        if col not in a.columns:
            raise ValueError(f"actions needs a {col!r} column")
    a["date"] = pd.to_datetime(a["date"])
    a["ratio"] = pd.to_numeric(a["ratio"], errors="coerce")
    if "kind" not in a.columns:
        a["kind"] = "split"
    if "ticker" not in a.columns:
        a["ticker"] = pd.NA
    return a.sort_values("date").reset_index(drop=True)


def _cumfactor(index: pd.DatetimeIndex, acts: pd.DataFrame, *, forward: bool) -> pd.Series:
    """The multiplicative factor turning a raw series into an adjusted one.

    `ratio` is the multiplicative PRICE factor of the event, whatever its kind: 2.0 for a
    2-for-1 split (the raw quote halves on the ex-date), prev_close/(prev_close - div) for
    a cash dividend. That is the same meaning `core.adjustment_check` gives the column.

    back-adjusted  P(t) = raw(t) * prod(ratio_i for ex_i <= t)     anchored at the START
    forward        P(t) = raw(t) / prod(ratio_i for ex_i >  t)     anchored at the PRESENT
    """
    f = pd.Series(1.0, index=index, dtype=float)
    for _, a in acts.iterrows():
        r = float(a["ratio"])
        if not np.isfinite(r) or r <= 0:
            continue
        if forward:
            # only bars STRICTLY BEFORE the ex-date are rescaled: on the ex-date the raw
            # quote has already taken the drop, so its forward-adjusted value is itself
            f = f.where(index >= a["date"], f / r)     # scale history DOWN
        else:
            f = f.where(index < a["date"], f * r)      # scale the present UP
    return f


# ------------------------------------------------------------------------------- Bars
@dataclass(frozen=True)
class Bars:
    """One instrument or one panel of instruments, on ONE declared calendar.

    `adjustment`, `calendar`, `tz`, `interval`, `bar_label` and `currency` have no
    defaults on purpose. Guessing any of them is how a backtest ends up trading a split.
    """

    frame: pd.DataFrame              # MultiIndex columns (field, ticker); fields are OHLCV
    adjustment: Adjustment
    calendar: str                    # "XNYS" | "XSHG" | "24/7" | "observed"
    tz: str                          # exchange-local zone of the LABELS
    interval: str                    # "1d" | "1h" | "1m"
    bar_label: str                   # "close" | "open" - which end of the interval it names
    currency: str
    provenance: Provenance
    actions: pd.DataFrame | None = None     # date, ratio, kind[, ticker, pay_date]
    listings: pd.DataFrame | None = None    # ticker, start_date, end_date, reason
    half_open: bool = True                  # [start, end) - yfinance's `end` is exclusive

    # ------------------------------------------------------------------ construction
    def __post_init__(self) -> None:
        object.__setattr__(self, "adjustment", _as_adjustment(self.adjustment))
        f = self.frame
        if not isinstance(f, pd.DataFrame):
            raise TypeError(f"Bars.frame must be a DataFrame, got {type(f).__name__}")
        if not isinstance(f.columns, pd.MultiIndex) or f.columns.nlevels != 2:
            raise ValueError("Bars.frame needs MultiIndex columns (field, ticker); "
                             "fin_skills.data.schema.stack_fields() builds one")
        if not isinstance(f.index, pd.DatetimeIndex):
            raise TypeError("Bars.frame must be indexed by a DatetimeIndex")
        if not f.index.is_monotonic_increasing:
            raise ValueError("Bars.frame index is unsorted; every guard downstream "
                             "assumes sorted time series")
        if f.index.has_duplicates:
            raise ValueError("Bars.frame index has duplicate timestamps; two bars on one "
                             "label is a vendor merge error, not data")
        if self.bar_label not in ("open", "close"):
            raise ValueError("bar_label must be 'open' or 'close' - which end of the "
                             "interval the index names")
        if f.index.tz is not None and str(f.index.tz) != self.tz:
            raise ValueError(f"index tz {str(f.index.tz)!r} contradicts the declared "
                             f"tz {self.tz!r}")
        if self.actions is not None:
            object.__setattr__(self, "actions", _actions_frame(self.actions))
        if self.listings is not None:
            _require_columns(self.listings, ("ticker", "start_date"), "Bars.listings")
        cols = f.columns.set_names(["field", "ticker"])
        object.__setattr__(self, "frame", f.set_axis(cols, axis=1))

    # ------------------------------------------------------------------- accessors
    @property
    def tickers(self) -> list[str]:
        return list(dict.fromkeys(self.frame.columns.get_level_values("ticker")))

    @property
    def fields(self) -> list[str]:
        return list(dict.fromkeys(self.frame.columns.get_level_values("field")))

    def field(self, name: str) -> pd.DataFrame:
        """One field as a wide panel, dates x tickers."""
        if name not in self.fields:
            raise KeyError(f"no field {name!r}; have {self.fields}")
        return self.frame[name]

    def close(self, ticker: str | None = None) -> pd.Series:
        """One instrument's close. With no ticker, the only one - never an arbitrary pick."""
        wide = self.field("close")
        if ticker is None:
            if wide.shape[1] != 1:
                raise ValueError(f"close() needs a ticker: this panel holds "
                                 f"{wide.shape[1]} of them")
            ticker = wide.columns[0]
        return wide[ticker].rename(ticker)

    def ohlcv(self, ticker: str | None = None) -> pd.DataFrame:
        """One instrument's OHLCV frame - the shape the `bars` slot wants."""
        if ticker is None:
            names = self.tickers
            if len(names) != 1:
                raise ValueError(f"ohlcv() needs a ticker: this panel holds {len(names)}")
            ticker = names[0]
        cols = [f for f in OHLCV if f in self.fields]
        out = pd.concat({f: self.frame[(f, ticker)] for f in cols}, axis=1)
        return out[cols]

    def alive(self) -> pd.DataFrame:
        """Per ticker: first and last observation, and whether it ends before the panel.

        The survivorship tell, computed from the data rather than believed from a
        vendor's blurb: a decade-long panel in which nothing ends early is a
        current-snapshot screen, not a clean universe.
        """
        wide = self.field("close")
        rows = []
        end = wide.index.max()
        for t in wide.columns:
            s = wide[t].dropna()
            rows.append({"ticker": t,
                         "first": s.index.min() if len(s) else pd.NaT,
                         "last": s.index.max() if len(s) else pd.NaT,
                         "n_obs": int(len(s)),
                         "ends_early": bool(len(s) and s.index.max() < end)})
        return pd.DataFrame(rows).set_index("ticker")

    # ------------------------------------------------------------------ adjustment
    def readjust(self, to: Adjustment | str) -> "Bars":
        """Recompute the price fields onto another convention, from `actions`.

        Raises when `actions` is None. A convention with no event table cannot be checked,
        and an uncheckable convention is worse than a raw price - it looks authoritative.
        """
        to = _as_adjustment(to)
        if self.actions is None or not len(self.actions):
            raise ValueError(
                "readjust() needs an actions table (date, ratio[, kind]); without one the "
                "conversion cannot be verified, and an unverifiable convention is worse "
                "than a raw price. Fetch actions from the adapter and try again.")
        if self.adjustment is Adjustment.UNKNOWN:
            raise ValueError("cannot readjust from UNKNOWN: there is no basis to convert "
                             "from. Determine the convention first "
                             "(fin_skills.api.check(close=..., actions=...)).")
        if to is Adjustment.UNKNOWN:
            raise ValueError("readjust(UNKNOWN) would throw away a known convention")
        if to is self.adjustment:
            return self

        price_fields = [f for f in self.fields if f in ("open", "high", "low", "close")]
        out = self.frame.copy()
        for t in self.tickers:
            acts = self.actions
            if "ticker" in acts.columns and acts["ticker"].notna().any():
                acts = acts[(acts["ticker"] == t) | acts["ticker"].isna()]
            acts = acts[(acts["date"] >= self.frame.index.min())
                        & (acts["date"] <= self.frame.index.max())]
            if not len(acts):
                continue
            to_raw = self._factor_to_raw(self.adjustment, self.frame.index, acts)
            from_raw = self._factor_from_raw(to, self.frame.index, acts)
            for f in price_fields:
                if (f, t) in out.columns:
                    out[(f, t)] = out[(f, t)] * to_raw * from_raw
        return replace(self, frame=out, adjustment=to,
                       provenance=self.provenance.with_content(out))

    @staticmethod
    def _factor_from_raw(to: Adjustment, index: pd.DatetimeIndex,
                         acts: pd.DataFrame) -> pd.Series:
        if to is Adjustment.BACK:
            return _cumfactor(index, acts, forward=False)
        if to is Adjustment.FORWARD:
            return _cumfactor(index, acts, forward=True)
        return pd.Series(1.0, index=index)          # RAW and RAW_PLUS_FACTORS store raw

    @classmethod
    def _factor_to_raw(cls, frm: Adjustment, index: pd.DatetimeIndex,
                       acts: pd.DataFrame) -> pd.Series:
        return 1.0 / cls._factor_from_raw(frm, index, acts)

    # ------------------------------------------------------------------- calendar
    def align(self, sessions: Iterable) -> "Bars":
        """Reindex onto a declared session list. A bar on a NON-session raises; a session
        with no bar becomes NaN and stays NaN.

        The asymmetry is deliberate. An extra bar means the calendar declaration is wrong,
        which is a bug. A hole means the vendor had nothing, which is information.
        """
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).sort_values()
        if self.frame.index.tz is not None and idx.tz is None:
            idx = idx.tz_localize(self.frame.index.tz)
        extra = self.frame.index.difference(idx)
        if len(extra):
            raise ValueError(
                f"{len(extra)} bar(s) fall on non-sessions of calendar {self.calendar!r} "
                f"(first {extra[0].date()}); either the calendar declaration is wrong or "
                f"the vendor returned a holiday bar")
        out = self.frame.reindex(idx)
        return replace(self, frame=out, provenance=self.provenance.with_content(out))

    def to_panel(self, sessions=None):
        """Hand the panel to `fin_skills.engine`.

        The engine is imported lazily - the data layer does not depend on it, and
        importing one must not drag in the other - and only the arguments its Panel
        actually names are passed. `adjustment` is the exception: a Panel that cannot
        carry the convention is refused rather than handed data, because a convention
        dropped at this boundary is a convention nobody downstream can check.
        """
        import inspect                                                   # noqa: PLC0415
        try:
            from fin_skills import engine                                # noqa: PLC0415
        except ImportError as exc:                                       # pragma: no cover
            raise ImportError("fin_skills.engine is not available in this install; "
                              "Bars.to_panel() needs it") from exc
        b = self.align(sessions) if sessions is not None else self
        candidates = {"frame": b.frame, "adjustment": b.adjustment.value,
                      "calendar": b.calendar, "tz": b.tz, "interval": b.interval,
                      "bar_label": b.bar_label, "currency": b.currency,
                      "actions": b.actions, "listings": b.listings}
        params = inspect.signature(engine.Panel).parameters
        if "adjustment" not in params:                                   # pragma: no cover
            raise TypeError("engine.Panel does not take an adjustment; this layer will "
                            "not hand it a price panel whose convention it cannot carry")
        return engine.Panel(**{k: v for k, v in candidates.items() if k in params})

    # -------------------------------------------------------------------- identity
    def validate(self) -> list[Finding]:
        from fin_skills.data.validate import validate_bars                # noqa: PLC0415
        return validate_bars(self)

    def fingerprint(self) -> str:
        """Content hash of the arrays PLUS the declarations.

        Two panels with identical numbers but different declared adjustments are not the
        same artefact, and treating them as one is exactly how a qfq series ends up
        cached under a hfq key.
        """
        h = hashlib.sha256()
        h.update(content_hash(self.frame).encode("ascii"))
        for part in (self.adjustment.value, self.calendar, self.tz, self.interval,
                     self.bar_label, self.currency, str(bool(self.half_open))):
            h.update(b"\0" + part.encode("utf-8"))
        if self.actions is not None and len(self.actions):
            h.update(b"\0actions\0" + content_hash(self.actions).encode("ascii"))
        return h.hexdigest()

    def as_data_source(self):
        return self.provenance.as_data_source(self.adjustment.value)

    def __repr__(self) -> str:
        return (f"Bars({self.frame.shape[0]}x{len(self.tickers)} {self.interval} "
                f"{self.adjustment.value} {self.calendar} {self.currency}, "
                f"actions={0 if self.actions is None else len(self.actions)})")


# ----------------------------------------------------------------------- Fundamentals
FUNDAMENTAL_COLUMNS = ("entity_id", "entity_scheme", "period_start", "period_end",
                       "filed_at", "acceptance_at", "available_at", "form", "accn",
                       "tag", "value", "unit", "is_amendment")

_PIT_KEY = ("entity_id", "tag", "period_start", "period_end")


@dataclass(frozen=True)
class Fundamentals:
    """Point-in-time BY CONSTRUCTION: no row exists without all three dates.

    research-integrity-guards: `period_end` is a 30-90 day look-ahead, `filed_at` still
    admits a post-close leak, and `acceptance_at` is the truth - while the SEC's own two
    APIs disagree about its timezone (the Submissions API's `acceptanceDateTime` is UTC,
    the Financial Statement Data Sets' `sub.txt.accepted` is Eastern). All three are
    stored; only `available_at` is ever USED.
    """

    frame: pd.DataFrame
    provenance: Provenance
    tz_of_record: str = "UTC"

    def __post_init__(self) -> None:
        _require_columns(self.frame, FUNDAMENTAL_COLUMNS, "Fundamentals")
        f = self.frame.copy()
        for col in ("period_start", "period_end", "filed_at", "acceptance_at",
                    "available_at"):
            f[col] = pd.to_datetime(f[col], errors="coerce")
        if f["available_at"].isna().any():
            raise ValueError("every Fundamentals row needs an available_at; a row whose "
                             "knowability date is unknown cannot be used point-in-time")
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        object.__setattr__(self, "frame", f.reset_index(drop=True))

    # ------------------------------------------------------------------ point in time
    def as_of(self, ts, *, forms: tuple[str, ...] = (),
              entities: Sequence[str] | None = None) -> pd.DataFrame:
        """The latest vintage KNOWN AS OF `ts`. One row per (entity, tag, period).

        `sort_values('available_at').groupby(key).tail(1)` - never
        `drop_duplicates(keep='last')`, which ignores the vintage filter entirely and
        silently selects the restatement, and never `groupby().last()`, which takes the
        last NON-NULL value of each column independently and can assemble a row that was
        never filed in that shape.
        """
        ts = pd.Timestamp(ts)
        f = self.frame
        if forms:
            f = f[f["form"].isin(tuple(forms))]
        if entities is not None:
            f = f[f["entity_id"].isin(list(entities))]
        f = f[f["available_at"] <= ts]
        if not len(f):
            return f.reset_index(drop=True)
        return (f.sort_values(["available_at", "accn"], kind="mergesort")
                 .groupby(list(_PIT_KEY), sort=False, dropna=False).tail(1)
                 .sort_values(["entity_id", "tag", "period_end"], kind="mergesort")
                 .reset_index(drop=True))

    def vintages(self, tag: str, period_end, *, entity: str | None = None) -> pd.DataFrame:
        """Every version of one number, with the date each became knowable."""
        pe = pd.Timestamp(period_end)
        f = self.frame[(self.frame["tag"] == tag) & (self.frame["period_end"] == pe)]
        if entity is not None:
            f = f[f["entity_id"] == entity]
        return f.sort_values("available_at", kind="mergesort").reset_index(drop=True)

    def to_daily(self, sessions: Iterable, tag: str, *,
                 entities: Sequence[str] | None = None) -> pd.DataFrame:
        """The value knowable on each session, as a wide panel (sessions x entity).

        An as-of join, not a fill: a session before the first filing is NaN and stays NaN.
        """
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))).sort_values()
        f = self.frame[self.frame["tag"] == tag]
        if entities is not None:
            f = f[f["entity_id"].isin(list(entities))]
        out = pd.DataFrame(index=idx, columns=sorted(f["entity_id"].unique()), dtype=float)
        for eid, g in f.groupby("entity_id", sort=False):
            g = g.sort_values("available_at", kind="mergesort")
            # for each session, the latest vintage of the latest period known by then
            known = g[["available_at", "period_end", "value"]].copy()
            vals = []
            for ts in idx:
                seen = known[known["available_at"] <= ts]
                if not len(seen):
                    vals.append(np.nan)
                    continue
                latest_period = seen["period_end"].max()
                row = seen[seen["period_end"] == latest_period].iloc[-1]
                vals.append(float(row["value"]))
            out[eid] = vals
        out.index.name = "session"
        return out

    def restatements(self) -> pd.DataFrame:
        """Periods whose value changed across vintages: a PIT and a naive study will
        disagree for every row here."""
        rows = []
        for key, g in self.frame.groupby(list(_PIT_KEY), sort=False, dropna=False):
            g = g.sort_values("available_at", kind="mergesort")
            vals = g["value"].dropna().to_numpy()
            if len(vals) < 2 or np.allclose(vals, vals[0], rtol=1e-12, atol=0.0):
                continue
            rows.append({"entity_id": key[0], "tag": key[1], "period_start": key[2],
                         "period_end": key[3], "n_vintages": int(len(g)),
                         "first_available": g["available_at"].iloc[0],
                         "last_available": g["available_at"].iloc[-1],
                         "original_val": float(vals[0]), "restated_val": float(vals[-1])})
        return pd.DataFrame(rows)

    def to_facts(self, *, entity: str | None = None,
                 tag: str | None = None) -> list[dict]:
        """The SEC companyfacts `units` list shape `pit_fundamentals` wants.

        `filed` is mapped from `available_at`, not from `filed_at`: available_at is the
        date the number could actually be traded on, which is the date the guard should
        compare against.
        """
        f = self.frame
        if entity is not None:
            f = f[f["entity_id"] == entity]
        if tag is not None:
            f = f[f["tag"] == tag]
        out = []
        for _, r in f.sort_values("available_at", kind="mergesort").iterrows():
            out.append({
                "start": None if pd.isna(r["period_start"])
                else r["period_start"].strftime("%Y-%m-%d"),
                "end": r["period_end"].strftime("%Y-%m-%d"),
                "val": float(r["value"]), "accn": str(r["accn"]),
                "fy": r.get("fy"), "fp": r.get("fp"), "form": str(r["form"]),
                "filed": r["available_at"].strftime("%Y-%m-%d"),
            })
        return out

    def validate(self) -> list[Finding]:
        from fin_skills.data.validate import validate_fundamentals          # noqa: PLC0415
        return validate_fundamentals(self)

    def fingerprint(self) -> str:
        return content_hash(self.frame)

    def as_data_source(self):
        return self.provenance.as_data_source("PIT, available_at<=as_of")

    def __repr__(self) -> str:
        return (f"Fundamentals({len(self.frame)} rows, "
                f"{self.frame['entity_id'].nunique()} entities, "
                f"{self.frame['tag'].nunique()} tags, tz_of_record={self.tz_of_record!r})")


# ------------------------------------------------------------------------------- Macro
MACRO_COLUMNS = ("series_id", "obs_date", "value", "realtime_start", "realtime_end",
                 "sa_flag")


@dataclass(frozen=True)
class Macro:
    """Vintage-aware: a macro series is an (obs_date x realtime_start) MATRIX, not a line.

    FRED's own documented example: 2013Q4 GDP was 17102.5 on 2014-01-30, then 17080.7 on
    2014-02-28, then 17089.6 on 2014-03-27. Backtesting the current series trades on
    figures published up to a decade later.
    """

    frame: pd.DataFrame
    provenance: Provenance

    def __post_init__(self) -> None:
        _require_columns(self.frame, MACRO_COLUMNS, "Macro")
        f = self.frame.copy()
        for col in ("obs_date", "realtime_start", "realtime_end"):
            f[col] = pd.to_datetime(f[col], errors="coerce")
        if f["realtime_start"].isna().any():
            raise ValueError("every Macro row needs a realtime_start; the observation "
                             "date is not the date the number was published")
        f["value"] = pd.to_numeric(f["value"], errors="coerce")
        object.__setattr__(self, "frame", f.reset_index(drop=True))

    @property
    def series_ids(self) -> list[str]:
        return sorted(self.frame["series_id"].unique())

    def as_of(self, ts, *, series_id: str | None = None) -> pd.Series:
        """The vintage current at `ts`, DEDUPLICATED - one value per obs_date.

        This is what `fredapi.get_series_as_of_date()` is documented to return and does
        not: its implementation is `df[df['realtime_start'] <= as_of_date]`, a DataFrame
        with one row per revision, so treating it as a series double-counts every revised
        observation. The `.groupby(...).tail(1)` below is the step its docstring omits.
        """
        ts = pd.Timestamp(ts)
        f = self.frame
        if series_id is not None:
            f = f[f["series_id"] == series_id]
        elif f["series_id"].nunique() > 1:
            raise ValueError(f"as_of() needs a series_id: this Macro holds "
                             f"{self.series_ids}")
        f = f[f["realtime_start"] <= ts]
        if not len(f):
            return pd.Series(dtype=float, name=series_id)
        out = (f.sort_values(["realtime_start"], kind="mergesort")
                .groupby("obs_date", sort=True).tail(1)
                .sort_values("obs_date", kind="mergesort"))
        return out.set_index("obs_date")["value"].rename(
            series_id or str(f["series_id"].iloc[0]))

    def latest(self, *, series_id: str | None = None) -> pd.Series:
        """The fully-revised series - DISPLAY ONLY.

        The returned Series carries `.attrs['research_use'] = 'DISPLAY ONLY'`, because
        every value in it may have been published after the date it is indexed at.
        """
        f = self.frame
        if series_id is not None:
            f = f[f["series_id"] == series_id]
        elif f["series_id"].nunique() > 1:
            raise ValueError(f"latest() needs a series_id: this Macro holds "
                             f"{self.series_ids}")
        out = (f.sort_values(["realtime_start"], kind="mergesort")
                .groupby("obs_date", sort=True).tail(1)
                .sort_values("obs_date", kind="mergesort"))
        s = out.set_index("obs_date")["value"].rename(
            series_id or str(f["series_id"].iloc[0]))
        s.attrs["research_use"] = "DISPLAY ONLY"
        return s

    def vintages(self, obs_date, *, series_id: str | None = None) -> pd.DataFrame:
        """Every published value of one observation, with its publication date."""
        d = pd.Timestamp(obs_date)
        f = self.frame[self.frame["obs_date"] == d]
        if series_id is not None:
            f = f[f["series_id"] == series_id]
        return f.sort_values("realtime_start", kind="mergesort").reset_index(drop=True)

    def validate(self) -> list[Finding]:
        from fin_skills.data.validate import validate_macro                 # noqa: PLC0415
        return validate_macro(self)

    def fingerprint(self) -> str:
        return content_hash(self.frame)

    def as_data_source(self):
        return self.provenance.as_data_source("vintage, realtime_start<=as_of")

    def __repr__(self) -> str:
        return (f"Macro({len(self.frame)} rows, series={self.series_ids}, "
                f"{self.frame['obs_date'].nunique()} observation dates)")


__all__ = ["Adjustment", "Bars", "FUNDAMENTAL_COLUMNS", "Fundamentals", "MACRO_COLUMNS",
           "Macro", "OHLCV", "stack_fields"]
