#!/usr/bin/env python3
"""The risk-free argument means different things in quantstats and empyrical.

quantstats' `rf=` is an ANNUAL rate, de-annualised geometrically before use.
empyrical's `risk_free=` is a PER-PERIOD rate, subtracted from every return raw.
The two arguments sit next to identically-named functions, so passing the same
0.05 to both is the normal mistake, not an exotic one.

Nothing raises. The only symptom is the number, and the number is so extreme it
reads as a data bug rather than an argument bug — which is why people chase the
data instead of the call.

Run:  python rf_convention.py
Both libraries are optional. Without them this reproduces the same arithmetic
from the definitions, so the demonstration still runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = 252


def _sharpe_annual_rf(r: pd.Series, rf_annual: float, periods: int = PERIODS) -> float:
    """quantstats' convention: de-annualise the rate geometrically, then subtract."""
    per_period = (1.0 + rf_annual) ** (1.0 / periods) - 1.0
    excess = r - per_period
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))


def _sharpe_per_period_rf(r: pd.Series, rf_per_period: float, periods: int = PERIODS) -> float:
    """empyrical's convention: subtract the rate from every period as given."""
    excess = r - rf_per_period
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))


def demo(seed: int = 0, n: int = PERIODS * 4) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    r = pd.Series(rng.normal(0.0005, 0.01, n),
                  index=pd.bdate_range("2021-01-01", periods=n))

    out = {
        "qs_rf_0.05_annual": _sharpe_annual_rf(r, 0.05),
        "ep_rf_0.05_raw": _sharpe_per_period_rf(r, 0.05),
        "ep_rf_0.05_over_252": _sharpe_per_period_rf(r, 0.05 / PERIODS),
        "no_rf": _sharpe_per_period_rf(r, 0.0),
    }

    # If the real libraries are installed, verify the reference implementations
    # above actually match them rather than asserting that they do.
    try:
        import warnings

        warnings.filterwarnings("ignore")
        import empyrical as ep
        import quantstats as qs

        out["LIVE_qs.stats.sharpe(rf=0.05)"] = float(qs.stats.sharpe(r, rf=0.05))
        out["LIVE_ep.sharpe_ratio(risk_free=0.05)"] = float(ep.sharpe_ratio(r, risk_free=0.05))
        out["LIVE_ep.sharpe_ratio(risk_free=0.05/252)"] = float(
            ep.sharpe_ratio(r, risk_free=0.05 / PERIODS))
    except ImportError:
        pass
    return out


if __name__ == "__main__":
    res = demo()
    print("Same 0.05 passed to two libraries, one 4-year return series (seed=0)\n")
    for k, v in res.items():
        flag = "  <- reads as a data bug, is an argument bug" if v < -10 else ""
        print(f"  {k:<44} {v:>12.6f}{flag}")

    live = {k: v for k, v in res.items() if k.startswith("LIVE_")}
    if live:
        qs_v = res["LIVE_qs.stats.sharpe(rf=0.05)"]
        ep_v = res["LIVE_ep.sharpe_ratio(risk_free=0.05)"]
        ep_c = res["LIVE_ep.sharpe_ratio(risk_free=0.05/252)"]
        print(f"\n  verified against the installed libraries:")
        print(f"    empyrical with the annual rate passed raw is {ep_v:,.2f} —"
              f" {abs(ep_v / qs_v):,.0f}x the correct magnitude")
        print(f"    dividing by {PERIODS} brings it to {ep_c:.6f}, near quantstats' {qs_v:.6f}")
        print(f"    the residual {abs(ep_c - qs_v):.6f} is quantstats de-annualising"
              f" GEOMETRICALLY, not by division — so /252 is close but still not its answer")
    else:
        print("\n  quantstats/empyrical not installed — reference implementations only")

    print("\n  Rule: quantstats/ffn/PyPortfolioOpt take an ANNUAL rate."
          "\n        empyrical/pyfolio-reloaded take a PER-PERIOD rate."
          "\n        A Sharpe below about -10 is almost always this, not the strategy.")
