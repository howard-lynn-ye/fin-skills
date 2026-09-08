"""Guard: survivorship bias in a price panel (research-integrity-guards / survivorship_audit.py)."""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.core.survivorship_audit import audit_universe


@register
class SurvivorshipGuard(Guard):
    """Audit a backtest universe for survivorship bias and price the damage in bps/yr.

    Inputs
        prices                      : DataFrame of prices, dates x tickers, with NaN
                                      after a name stops trading. A bare list of tickers
                                      is accepted with sample_start/sample_end, but can
                                      only be cross-checked against `listings`.
        listings                    : optional table with ticker, listing_date,
                                      delisting_date - turns the tell into evidence.
        sample_start, sample_end    : override the panel's own span.
        min_gap_days                : a name whose data ends this long before the end is
                                      a candidate delisting. Default 30.
        expected_annual_delist_rate : your prior; 0.05 is a placeholder, not a constant.

    Fails when the universe is confirmed or likely survivor-biased, when a decade of
    history contains no name that ends early (a current-snapshot screen), or when it
    cannot be tested at all - "could not prove it is clean" fails closed.
    """

    name = "survivorship_audit"
    skill = "research-integrity-guards"
    summary = "Flags a survivor-only universe: zero delistings in a long panel is a missing dataset, not a clean one."
    wraps = ("fin_skills.core.survivorship_audit.audit_universe",
             "fin_skills.core.survivorship_audit.survivorship_inflation")
    required = ("prices",)
    optional = ("listings", "sample_start", "sample_end", "min_gap_days",
                "expected_annual_delist_rate")

    def check(self, prices: pd.DataFrame | Sequence[str], listings: pd.DataFrame | None = None,
              sample_start: str | pd.Timestamp | None = None,
              sample_end: str | pd.Timestamp | None = None, min_gap_days: int = 30,
              expected_annual_delist_rate: float = 0.05) -> Outcome:
        out = Outcome()
        if isinstance(prices, pd.DataFrame):
            if not isinstance(prices.index, pd.DatetimeIndex):
                raise TypeError("prices must be indexed by date (DatetimeIndex)")
        elif isinstance(prices, (list, tuple, pd.Index)):
            if not all(isinstance(t, str) for t in prices):
                raise TypeError("a bare universe must be a list of ticker strings")
            if sample_start is None or sample_end is None:
                raise TypeError("a bare ticker list needs sample_start and sample_end")
        else:
            raise TypeError("prices must be a DataFrame (dates x tickers) or a list of tickers")
        if listings is not None:
            if not isinstance(listings, pd.DataFrame) or "ticker" not in listings.columns:
                raise TypeError("listings must be a DataFrame with a 'ticker' column")

        audit = audit_universe(prices, listings=listings, sample_start=sample_start,
                               sample_end=sample_end, min_gap_days=min_gap_days,
                               expected_annual_delist_rate=expected_annual_delist_rate)
        out.note(verdict=audit.verdict, n_names=audit.n_names, years=audit.years,
                 n_ended_early=audit.n_ended_early, frac_ended_early=audit.frac_ended_early,
                 expected_frac_ended_early=audit.expected_frac_ended_early,
                 n_missing_delisted=audit.n_missing_delisted,
                 missing_delisted=audit.missing_delisted,
                 n_prelisting_data=audit.n_prelisting_data, inflation=audit.inflation,
                 report=audit.report())

        v = audit.verdict
        if v.startswith("SURVIVOR-BIASED"):
            out.error(v, where="listings")
        elif v.startswith("SURVIVOR-ONLY"):
            out.error(v, where="panel")
        elif v.startswith("LIKELY"):
            out.error(v, where="panel")
        elif v.startswith("UNTESTABLE"):
            out.error(v + " - an unproven universe is treated as a biased one", where="panel")
        else:
            out.info(v, where="panel")
        infl = audit.inflation
        if infl.get("measurable"):
            out.info(f"survivors-only subset inflates CAGR by {infl['inflation_bps']:.0f} bps/yr",
                     where="inflation")
        for n in audit.notes:
            out.warning(n, where="notes")
        return out
