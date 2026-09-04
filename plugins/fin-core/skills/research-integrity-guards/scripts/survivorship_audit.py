#!/usr/bin/env python3
"""Audit a backtest universe for survivorship bias, and price the damage in bps/yr.

WHY: you screen tickers *today* -- from yfinance, financedatabase, or SEC
company_tickers.json -- then backtest ten years. Every company that went to zero is
absent, so what you actually tested is "the list of things that survived". The bias is
largest exactly where a strategy claims to add value (distressed, small-cap, mean
reversion into weakness). Nothing in the backtest errors out; the Sharpe just comes back
too good.

The tell is mechanical and needs no external data: in a bias-free panel a meaningful
fraction of names STOP having data before the sample ends. If a decade of history
contains zero names that end early, the universe was built from a current snapshot.
Zero delistings is not a clean dataset -- it is a missing one.

Usage:
    from survivorship_audit import audit_universe, survivorship_inflation

    a = audit_universe(prices_df, listings=listing_table, expected_annual_delist_rate=0.05)
    print(a.report())
    if a.verdict.startswith("SURVIVOR-ONLY"):
        raise SystemExit("refusing to backtest on a survivor-only universe")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------------------
# core measurements
# --------------------------------------------------------------------------------------
def last_observation(prices: pd.DataFrame) -> pd.Series:
    """Last date each column has a real value. This is the delisting proxy."""
    return prices.apply(lambda s: s.last_valid_index())


def first_observation(prices: pd.DataFrame) -> pd.Series:
    return prices.apply(lambda s: s.first_valid_index())


def dead_names(prices: pd.DataFrame, min_gap_days: int = 30,
               sample_end: pd.Timestamp | None = None) -> pd.Index:
    """Names whose data stops >= min_gap_days before the sample end.

    A short gap is a data outage; a name that simply never comes back is a candidate
    delisting. We deliberately do NOT require a listing table here -- the whole point is
    to have a check that works on whatever panel you were handed.
    """
    end = pd.Timestamp(sample_end) if sample_end is not None else prices.index.max()
    last = last_observation(prices).dropna()
    cutoff = end - pd.Timedelta(days=min_gap_days)
    return pd.Index([c for c, t in last.items() if pd.Timestamp(t) < cutoff])


def _ew_cagr(prices: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> float:
    """Equal-weight, daily-rebalanced CAGR over whatever names are alive each day.

    fill_method=None matters: forward-filling a delisted name's last price would
    manufacture a flat, riskless tail for a company that no longer exists.
    """
    rets = prices.pct_change(fill_method=None)
    ew = rets.mean(axis=1, skipna=True).dropna()
    if len(ew) < 2:
        return float("nan")
    years = len(ew) / periods_per_year
    return float((1.0 + ew).prod() ** (1.0 / years) - 1.0)


def survivorship_inflation(prices: pd.DataFrame, min_gap_days: int = 30,
                           periods_per_year: int = TRADING_DAYS) -> dict:
    """Return difference between the full panel and the survivors-only subset.

    This is the bias, measured rather than assumed -- but only measurable when the panel
    still CONTAINS its dead names. On a survivor-only panel the number is unknowable from
    the data itself, which is precisely why the audit has to flag the panel instead.

    Caveat encoded in the number: a delisted name's terminal loss is only captured to the
    extent its price path declined before the data stopped. Bankruptcies that gap to zero
    need a recovery rate from the delisting table; without one this UNDERSTATES the bias.
    """
    dead = dead_names(prices, min_gap_days=min_gap_days)
    survivors = prices.columns.difference(dead)
    full = _ew_cagr(prices, periods_per_year)
    if len(dead) == 0:
        return {"measurable": False, "n_names": int(prices.shape[1]), "n_dead": 0,
                "full_universe_cagr": full, "survivors_only_cagr": full,
                "inflation_ann": None, "inflation_bps": None,
                "reason": "panel contains no names that end early -- nothing to compare against"}
    surv = _ew_cagr(prices[survivors], periods_per_year)
    return {"measurable": True, "n_names": int(prices.shape[1]), "n_dead": int(len(dead)),
            "full_universe_cagr": full, "survivors_only_cagr": surv,
            "inflation_ann": float(surv - full), "inflation_bps": float((surv - full) * 1e4),
            "reason": ""}


# --------------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------------
@dataclass
class SurvivorshipAudit:
    n_names: int
    sample_start: str
    sample_end: str
    years: float
    n_ended_early: int
    frac_ended_early: float
    expected_frac_ended_early: float
    n_started_late: int
    n_missing_delisted: int
    missing_delisted: list[str]
    n_prelisting_data: int
    prelisting_names: list[str]
    inflation: dict
    verdict: str
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        infl = self.inflation
        if infl.get("measurable"):
            infl_line = (f"{infl['inflation_bps']:.0f} bps/yr "
                         f"(full {infl['full_universe_cagr']:+.2%} -> "
                         f"survivors-only {infl['survivors_only_cagr']:+.2%})")
        else:
            infl_line = f"NOT MEASURABLE FROM THIS PANEL - {infl.get('reason', '')}"
        lines = [
            "SURVIVORSHIP AUDIT",
            f"  names               : {self.n_names}",
            f"  sample              : {self.sample_start} -> {self.sample_end} "
            f"({self.years:.1f} yr)",
            f"  ended early         : {self.n_ended_early} "
            f"({self.frac_ended_early:.1%}) vs {self.expected_frac_ended_early:.1%} expected "
            f"at the assumed delisting rate",
            f"  started late (IPOs) : {self.n_started_late}   "
            f"(late starts are NOT evidence of bias)",
            f"  delisted-but-absent : {self.n_missing_delisted}"
            + (f"  {self.missing_delisted[:8]}" if self.missing_delisted else ""),
            f"  pre-listing data    : {self.n_prelisting_data}"
            + (f"  {self.prelisting_names[:8]}" if self.prelisting_names else ""),
            f"  implied inflation   : {infl_line}",
            f"  VERDICT             : {self.verdict}",
        ]
        lines += [f"    ! {n}" for n in self.notes]
        return "\n".join(lines)


def audit_universe(universe: pd.DataFrame | Sequence[str] | Iterable[str],
                   listings: pd.DataFrame | None = None,
                   sample_start: str | pd.Timestamp | None = None,
                   sample_end: str | pd.Timestamp | None = None,
                   min_gap_days: int = 30,
                   expected_annual_delist_rate: float = 0.05) -> SurvivorshipAudit:
    """Audit a price panel (or a bare ticker list) for survivorship bias.

    universe : DataFrame of prices (dates x tickers), or a list of tickers.
    listings : optional table with columns ticker, listing_date, delisting_date.
               Turns the statistical tell into hard evidence: a name the exchange
               delisted inside the sample that is simply absent from your universe.
    expected_annual_delist_rate : your prior for the fraction of listed names that leave
               the tape each year. 0.05 is a placeholder, NOT a verified constant --
               estimate it from your own listing table and pass it in.
    """
    notes: list[str] = []
    prices: pd.DataFrame | None = None
    if isinstance(universe, pd.DataFrame):
        prices = universe.sort_index()
        tickers = list(prices.columns)
        start = pd.Timestamp(sample_start) if sample_start is not None else prices.index.min()
        end = pd.Timestamp(sample_end) if sample_end is not None else prices.index.max()
    else:
        tickers = list(universe)
        if sample_start is None or sample_end is None:
            raise ValueError("a bare ticker list needs explicit sample_start and sample_end")
        start, end = pd.Timestamp(sample_start), pd.Timestamp(sample_end)
        notes.append("no price panel supplied - only the listing-table cross-check ran")

    years = max((end - start).days / 365.25, 1e-9)
    expected_frac = 1.0 - (1.0 - expected_annual_delist_rate) ** years

    n_early = n_late = 0
    if prices is not None:
        dead = dead_names(prices, min_gap_days=min_gap_days, sample_end=end)
        n_early = len(dead)
        first = first_observation(prices).dropna()
        n_late = sum(1 for t in first.values
                     if pd.Timestamp(t) > start + pd.Timedelta(days=min_gap_days))
    frac_early = n_early / len(tickers) if tickers else 0.0

    # hard evidence, if a listing table is available
    missing: list[str] = []
    prelisting: list[str] = []
    if listings is not None and len(listings):
        lt = listings.copy()
        for col in ("listing_date", "delisting_date"):
            if col in lt.columns:
                lt[col] = pd.to_datetime(lt[col], errors="coerce")
        held = set(tickers)
        if "delisting_date" in lt.columns:
            gone = lt[(lt["delisting_date"].notna())
                      & (lt["delisting_date"] >= start) & (lt["delisting_date"] <= end)]
            missing = sorted(t for t in gone["ticker"] if t not in held)
        if prices is not None and "listing_date" in lt.columns:
            first = first_observation(prices)
            ldate = lt.set_index("ticker")["listing_date"]
            # data before the listing date is backfilled history for an entity that did
            # not trade yet -- the mirror image of survivorship, and just as fatal
            prelisting = sorted(
                t for t in prices.columns
                if t in ldate.index and pd.notna(first.get(t)) and pd.notna(ldate[t])
                and pd.Timestamp(first[t]) < ldate[t] - pd.Timedelta(days=1)
            )

    infl = (survivorship_inflation(prices, min_gap_days=min_gap_days)
            if prices is not None else
            {"measurable": False, "reason": "no price panel supplied"})

    # verdict, strongest evidence first
    if missing:
        verdict = (f"SURVIVOR-BIASED (confirmed): {len(missing)} names the listing table "
                   f"delisted inside the sample are absent from the universe")
    elif prices is None:
        verdict = "UNTESTABLE: supply a price panel or a delisting table"
    elif n_early == 0 and years >= 1.0:
        verdict = (f"SURVIVOR-ONLY SNAPSHOT (red flag): not one of {len(tickers)} names "
                   f"ends early in {years:.1f} yr; ~{expected_frac:.0%} should have")
        notes.append("results from this universe are an UPPER BOUND, not an estimate")
    elif frac_early < 0.5 * expected_frac:
        verdict = (f"LIKELY SURVIVOR-BIASED: {frac_early:.1%} end early vs "
                   f"{expected_frac:.1%} expected - partial delisting coverage")
    else:
        verdict = (f"PLAUSIBLY BIAS-FREE: {frac_early:.1%} end early, consistent with "
                   f"{expected_frac:.1%} expected")
    if prelisting:
        notes.append(f"{len(prelisting)} names carry data before their listing date "
                     f"(backfilled history / identifier reuse)")
    if prices is not None and n_early and not infl.get("measurable"):
        notes.append("inflation unmeasurable despite dead names - check min_gap_days")

    return SurvivorshipAudit(
        n_names=len(tickers), sample_start=str(start.date()), sample_end=str(end.date()),
        years=years, n_ended_early=n_early, frac_ended_early=frac_early,
        expected_frac_ended_early=expected_frac, n_started_late=n_late,
        n_missing_delisted=len(missing), missing_delisted=missing,
        n_prelisting_data=len(prelisting), prelisting_names=prelisting,
        inflation=infl, verdict=verdict, notes=notes,
    )


# --------------------------------------------------------------------------------------
def _synthetic_panel(n_names: int = 200, years: int = 10, seed: int = 11
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A panel where weak names actually die, plus the matching listing table."""
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    idx = pd.bdate_range("2015-01-02", periods=n)
    mu = rng.normal(0.04, 0.18, n_names) / TRADING_DAYS      # wide cross-section of drift
    sig = rng.uniform(0.20, 0.55, n_names) / np.sqrt(TRADING_DAYS)
    shocks = rng.normal(mu, sig, size=(n, n_names))
    px = 100.0 * np.exp(np.cumsum(shocks, axis=0))

    cols = [f"N{i:03d}" for i in range(n_names)]
    frame = pd.DataFrame(px, index=idx, columns=cols)
    rows = []
    for j, c in enumerate(cols):
        # delisting rule: first time the name loses 80% of its value, it leaves the tape
        below = np.where(frame[c].to_numpy() < 20.0)[0]
        if len(below):
            k = int(below[0])
            frame.iloc[k + 1:, j] = np.nan
            rows.append({"ticker": c, "listing_date": idx[0],
                         "delisting_date": idx[min(k, n - 1)]})
        else:
            rows.append({"ticker": c, "listing_date": idx[0], "delisting_date": pd.NaT})
    return frame, pd.DataFrame(rows)


if __name__ == "__main__":
    full, listing_table = _synthetic_panel()

    # exactly what a screen against a current snapshot hands you: the same history,
    # with every name that stopped trading quietly dropped
    alive_today = [c for c in full.columns if pd.notna(full[c].iloc[-1])]
    survivor_only = full[alive_today]

    print("=== A. bias-free panel (dead names retained) ===")
    print(audit_universe(full, listings=listing_table).report())

    print("\n=== B. same data, screened from a current snapshot ===")
    print(audit_universe(survivor_only, listings=listing_table).report())

    print("\n=== C. panel B again, with NO listing table to check against ===")
    print(audit_universe(survivor_only).report())

    infl = survivorship_inflation(full)
    print(f"\nSame strategy, same dates: the survivor-only panel pays "
          f"{infl['inflation_bps']:.0f} bps/yr more "
          f"({infl['full_universe_cagr']:+.2%} -> {infl['survivors_only_cagr']:+.2%}) "
          f"and cannot measure that gap from its own data. C flags it from the tell alone; "
          f"B proves it. 'Zero names end early' is a defect, not a clean dataset.")
