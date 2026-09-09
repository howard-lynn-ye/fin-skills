"""Seeded synthetic data for the fin_skills.data tests. No network, no files, no vendors.

Two datasets of each kind: one CLEAN, on which the four data-only guards run green, and
one CORRUPT, built by breaking exactly one thing so the matching guard runs red. The
corruptions are the real ones this repository has recorded, not arbitrary noise:

  * a survivor-only panel - the dead names silently dropped, the listing table left behind
  * a RAW series relabelled as adjusted - the split still in it, the label saying otherwise
  * a restatement selected by drop_duplicates(keep="last") instead of the vintage on file
  * a fat-finger print on a quiet day, which reconciliation must call a DATA ERROR

tests/ is on sys.path during collection, so test modules import this as
`from _data_fixtures import ...`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.schema import (Adjustment, Bars, Fundamentals, Macro, stack_fields)

SEED = 20260909
TZ = "America/New_York"


def _prov(source: str = "yfinance", content=None, **request):
    # library_version says out loud that these are not real vendor bytes, while the
    # source keeps the adapter shape so Cache.put() resolves the right Declaration
    return make_provenance(source, library_version="0.0.0-synthetic",
                           request={"method": "bars", **request},
                           content=content if content is not None else pd.DataFrame())


# ------------------------------------------------------------------------------ prices
def raw_panel(n_names: int = 60, years: int = 8, n_dead: int = 20, seed: int = SEED):
    """A RAW, tick-aligned panel where `n_dead` names actually stop trading.

    Returns (frame, actions, listings). Prices are rounded to the cent, because tick
    alignment is what `detect_convention`'s anchor test reads: real quotes are struck in
    cents, and dividing them by a split ratio produces values that are not.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=252 * years, tz=TZ)
    names = [f"T{i:03d}" for i in range(n_names)]

    value = 40.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, (len(idx), n_names)), axis=0))
    volume = rng.integers(1_000_000, 40_000_000, (len(idx), n_names)).astype("int64")

    # two splits on one name, so the convention has something to be checked against
    split_at = [idx[int(len(idx) * 0.15)], idx[int(len(idx) * 0.45)]]
    acts = pd.DataFrame({"date": split_at, "ratio": [7.0, 3.0],
                         "kind": ["split", "split"], "ticker": [names[0], names[0]]})
    upto = np.ones(len(idx))
    for d, r in zip(acts["date"], acts["ratio"]):
        upto *= np.where(idx >= d, r, 1.0)
    value[:, 0] = value[:, 0] / upto                       # the raw quote takes the drop

    close = pd.DataFrame(np.round(value, 2), index=idx, columns=names)
    vol = pd.DataFrame(volume, index=idx, columns=names)

    # `n_dead` names die at spread-out dates; everything after is NaN and stays NaN
    dead_at = {}
    for i, t in enumerate(names[len(names) - n_dead:] if n_dead else []):
        pos = int(len(idx) * (0.25 + 0.65 * (i + 0.5) / n_dead))
        dead_at[t] = idx[pos]
        close.loc[close.index > idx[pos], t] = np.nan
        vol.loc[vol.index > idx[pos], t] = 0

    listings = pd.DataFrame([
        {"ticker": t, "start_date": idx[0].tz_localize(None),
         "end_date": (dead_at[t].tz_localize(None) if t in dead_at else pd.NaT),
         "reason": "delisted" if t in dead_at else ""}
        for t in names])

    frame = stack_fields({"open": np.round(close * 0.995, 2),
                          "high": np.round(close * 1.01, 2),
                          "low": np.round(close * 0.99, 2),
                          "close": close, "volume": vol.astype("int64")})
    return frame, acts, listings


def clean_bars(**kw) -> Bars:
    """A BACK-adjusted panel (anchored at the start, history never rewritten) with its
    actions and listings - the shape survivorship_audit and adjustment_check both pass on."""
    frame, acts, listings = raw_panel(**kw)
    raw = Bars(frame=frame, adjustment=Adjustment.RAW, calendar="XNYS", tz=TZ,
               interval="1d", bar_label="close", currency="USD",
               provenance=_prov(content=frame, symbols=list(frame.columns.levels[1])),
               actions=acts, listings=listings)
    return raw.readjust(Adjustment.BACK)


def survivor_only_bars(**kw) -> Bars:
    """The same panel with the dead names simply absent - what a today's-constituents
    screen returns - and the listing table left in place.

    That is the CONFIRMED survivorship shape rather than the statistical one: the exchange
    says those names were delisted inside the sample and the universe does not contain
    them, so the audit does not have to infer anything.
    """
    frame, acts, listings = raw_panel(**kw)
    close = frame["close"]
    survivors = [c for c in close.columns if close[c].notna().all()]
    fields = {f: frame[f][survivors] for f in ("open", "high", "low", "close")}
    fields["volume"] = frame["volume"][survivors].astype("int64")
    live = stack_fields(fields)
    return Bars(frame=live, adjustment=Adjustment.BACK, calendar="XNYS", tz=TZ,
                interval="1d", bar_label="close", currency="USD",
                provenance=_prov(content=live), actions=acts, listings=listings)


def mislabelled_bars(**kw) -> Bars:
    """RAW prices carrying the BACK label. The split is still a jump in the series and the
    declaration says it has been adjusted away - the exact failure `adjustment_check`
    exists to catch."""
    frame, acts, listings = raw_panel(**kw)
    return Bars(frame=frame, adjustment=Adjustment.BACK, calendar="XNYS", tz=TZ,
                interval="1d", bar_label="close", currency="USD",
                provenance=_prov(content=frame), actions=acts, listings=listings)


def split_ticker(bars: Bars) -> str:
    """The one name in the fixture panel that has corporate actions."""
    return str(bars.actions["ticker"].iloc[0])


# ------------------------------------------------------------------------ fundamentals
#: FY2022 Q3 revenue: filed once, then restated DOWN in an amendment three months later.
#: The vintage on file at 2022-12-31 is 1.00e9; drop_duplicates(keep="last") picks 0.94e9.
RESTATEMENT_AS_OF = pd.Timestamp("2022-12-31")
ORIGINAL_VAL = 1_000_000_000.0
RESTATED_VAL = 940_000_000.0


def fundamentals_frame() -> pd.DataFrame:
    rows = [
        # FY2022 Q3, as originally reported in the Q3 10-Q (accepted 2022-11-03 18:05 ET,
        # after the close, so it was first tradeable on 2022-11-04)
        dict(period_start="2022-07-01", period_end="2022-09-30", value=ORIGINAL_VAL,
             accn="0000000-22-001", form="10-Q", filed_at="2022-11-03",
             acceptance_at="2022-11-03T22:05:00Z", available_at="2022-11-04",
             is_amendment=False),
        # ...restated DOWN in an amended 10-Q filed three months later
        dict(period_start="2022-07-01", period_end="2022-09-30", value=RESTATED_VAL,
             accn="0000000-23-001", form="10-Q/A", filed_at="2023-02-14",
             acceptance_at="2023-02-14T14:02:00Z", available_at="2023-02-14",
             is_amendment=True),
        # an earlier quarter nobody restated, so the two paths agree on it
        dict(period_start="2022-04-01", period_end="2022-06-30", value=880_000_000.0,
             accn="0000000-22-000", form="10-Q", filed_at="2022-08-04",
             acceptance_at="2022-08-04T13:30:00Z", available_at="2022-08-04",
             is_amendment=False),
        # a quarter filed AFTER the as-of date: not on file at all on 2022-12-31
        dict(period_start="2022-10-01", period_end="2022-12-31", value=1_100_000_000.0,
             accn="0000000-23-002", form="10-Q", filed_at="2023-02-02",
             acceptance_at="2023-02-02T21:40:00Z", available_at="2023-02-03",
             is_amendment=False),
    ]
    for r in rows:
        r.update(entity_id="0000320193", entity_scheme="cik", tag="Revenues", unit="USD")
    return pd.DataFrame(rows)


def clean_fundamentals() -> Fundamentals:
    frame = fundamentals_frame()
    return Fundamentals(frame=frame, provenance=_prov("edgar", content=frame),
                        tz_of_record="UTC")


# -------------------------------------------------------------------------------- macro
#: FRED's own documented example: 2013Q4 GDP was 17102.5, then 17080.7, then 17089.6.
GDP_VINTAGES = [("2014-01-30", 17102.5), ("2014-02-28", 17080.7), ("2014-03-27", 17089.6)]


def macro_frame() -> pd.DataFrame:
    rows = []
    for rt, val in GDP_VINTAGES:
        rows.append(dict(series_id="GDP", obs_date="2013-10-01", value=val,
                         realtime_start=rt, realtime_end=pd.NaT, sa_flag="SAAR"))
    # a neighbouring quarter with two vintages, so as_of() has more than one date to dedupe
    rows.append(dict(series_id="GDP", obs_date="2014-01-01", value=17044.0,
                     realtime_start="2014-04-30", realtime_end=pd.NaT, sa_flag="SAAR"))
    rows.append(dict(series_id="GDP", obs_date="2014-01-01", value=17016.0,
                     realtime_start="2014-05-29", realtime_end=pd.NaT, sa_flag="SAAR"))
    return pd.DataFrame(rows)


def clean_macro() -> Macro:
    frame = macro_frame()
    return Macro(frame=frame, provenance=_prov("fred", content=frame))


# ----------------------------------------------------------------------- two vintages
def rewritten_vintage(bars: Bars, ticker: str, factor: float = 1.0 / 1.004) -> Bars:
    """The same series after a new dividend, as a FORWARD-adjusted vendor would return it:
    every historical value rescaled by one constant, which is what makes a cached copy
    disagree with a live pull."""
    frame = bars.frame.copy()
    for f in ("open", "high", "low", "close"):
        if (f, ticker) in frame.columns:
            frame[(f, ticker)] = frame[(f, ticker)] * factor
    return Bars(frame=frame, adjustment=bars.adjustment, calendar=bars.calendar,
                tz=bars.tz, interval=bars.interval, bar_label=bars.bar_label,
                currency=bars.currency, provenance=bars.provenance.with_content(frame),
                actions=bars.actions, listings=bars.listings)


def bad_tick_vintage(bars: Bars, ticker: str, pos: int | None = None,
                     factor: float = 1.35) -> Bars:
    """One fat-finger print on a quiet day, far from any corporate action. Reconciliation
    must call this a DATA ERROR rather than a convention difference."""
    if pos is None:
        pos = int(len(bars.frame) * 0.75)      # after the last split, nowhere near it
    frame = bars.frame.copy()
    col = ("close", ticker)
    values = frame[col].to_numpy(dtype=float).copy()
    values[pos] *= factor
    frame[col] = values
    return Bars(frame=frame, adjustment=bars.adjustment, calendar=bars.calendar,
                tz=bars.tz, interval=bars.interval, bar_label=bars.bar_label,
                currency=bars.currency, provenance=bars.provenance.with_content(frame),
                actions=bars.actions, listings=bars.listings)


class StubAdapter:
    """An adapter that returns a prepared object instead of calling anything.

    `Cache.refetch()` asks for `replay(request)`; this hands back the next vintage in the
    queue, so a refetch can be tested without a vendor, a key or a socket.
    """

    def __init__(self, *vintages) -> None:
        self.queue = list(vintages)
        self.requests: list[dict] = []

    def replay(self, request: dict):
        self.requests.append(dict(request))
        obj = self.queue.pop(0)
        return obj, obj.provenance
