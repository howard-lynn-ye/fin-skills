#!/usr/bin/env python3
"""Three continuous contracts from the same futures chain, and only one of them is returns.

A futures contract expires. To get a series longer than one contract you weld the chain
together, and there are three ways to do it. They produce three different series, all of
them plausible-looking, and the choice decides whether `pct_change()` on the result is the
P&L you would have made or a number with no meaning at all.

This builds a backwardated market (deferred cheaper than near, the crude-oil case), stitches
it three ways, scores each against the true P&L of actually holding and rolling the front
contract, and shows the back-adjusted series walking through zero into negative prices.

    python examples/futures_roll.py        (offline, seeded, a few seconds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.futures_fx.continuous_contract import (compare_methods, safe_returns,  # noqa: E402
                                                       stitch, true_roll_return)

SEED = 20260909
YEARS = 8                # long enough for the back-adjusted series to reach zero
N_CONTRACTS = 29         # quarterly, so 28 rolls
CARRY = 0.22             # +22%/yr of BACKWARDATION: a contract T years out is that cheaper
ROLL_LEAD = 5            # roll this many business days before expiry


def pct(x: float) -> str:
    """Percentages that stay in their column even when the arithmetic detonates."""
    if not np.isfinite(x):
        return "n/a"
    return f"{x:,.1%}" if abs(x) < 100 else f"{x:.1e}"


def chain(seed: int = SEED) -> tuple[pd.DataFrame, list[pd.Timestamp], pd.Series]:
    """A quarterly chain: rows are dates, columns are contracts ordered NEAR -> FAR.

    Every contract prices off one spot path, discounted for its own time to expiry, so the
    term structure is a clean, persistent backwardation and nothing about the result is an
    artefact of independent noise per contract.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=252 * YEARS)
    spot = pd.Series(60.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, len(dates)))),
                     index=dates, name="spot")

    expiries = [dates[i] for i in range(120, len(dates), 63)][:N_CONTRACTS]
    cols, prices = [], {}
    for k, exp in enumerate(expiries):
        tte = pd.Series((exp - dates).days / 365.0, index=dates)
        px = spot * np.exp(-CARRY * tte.clip(lower=0.0))       # far = cheaper
        alive = (dates <= exp) & (dates >= exp - pd.Timedelta(days=400))
        name = f"C{k:02d}"
        cols.append(name)
        prices[name] = px.where(alive)
    contracts = pd.DataFrame(prices, index=dates)[cols]

    # one roll per adjacent pair, a few days before the near leg expires, and only on a
    # date where BOTH legs print - _validate rejects anything else, for good reason.
    rolls = [dates[dates.searchsorted(exp) - ROLL_LEAD] for exp in expiries[:-1]]
    return contracts, rolls, spot


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * 78)


def main() -> int:
    contracts, rolls, spot = chain()
    truth_pct = true_roll_return(contracts, rolls)
    truth_pts = true_roll_return(contracts, rolls, in_points=True)

    unadj = stitch(contracts, rolls, "unadjusted")
    diff = stitch(contracts, rolls, "difference")
    ratio = stitch(contracts, rolls, "ratio")

    rule("1. ONE CHAIN, THREE SERIES")
    print(f"  {contracts.shape[1]} quarterly contracts over {len(contracts):,} business days; "
          f"{len(rolls)} rolls")
    print(f"  spot {spot.iloc[0]:.1f} -> {spot.iloc[-1]:.1f}; term structure {CARRY:.0%}/yr "
          f"BACKWARDATED")
    print(f"\n  {'method':<13}{'first':>10}{'last':>10}{'min':>10}{'max':>10}   "
          f"what the levels are")
    print("  " + "-" * 74)
    for name, s, meaning in (("unadjusted", unadj, "prices that printed, with a gap at each roll"),
                             ("difference", diff, "a P&L path anchored on today's contract"),
                             ("ratio", ratio, "a return path; the levels are not tradeable")):
        print(f"  {name:<13}{s.iloc[0]:>10.2f}{s.iloc[-1]:>10.2f}{s.min():>10.2f}"
              f"{s.max():>10.2f}   {meaning}")

    rule("2. WHICH OPERATOR REPRODUCES THE TRUE P&L?")
    print("  Truth = hold the front contract, sell it and buy the next at every roll.")
    print("  Units: 'ret' = return units (0.01 = 1%), 'pts' = price points.")
    print(f"\n  {'series':<13}{'operator':<14}{'max abs error vs truth':>24}   verdict")
    print("  " + "-" * 74)
    trials = [
        ("ratio", "pct_change()", ratio.pct_change(fill_method=None), truth_pct, "ret"),
        ("difference", "diff()", diff.diff(), truth_pts, "pts"),
        ("difference", "pct_change()", diff.pct_change(fill_method=None), truth_pct, "ret"),
        ("unadjusted", "pct_change()", unadj.pct_change(fill_method=None), truth_pct, "ret"),
    ]
    for name, op, got, want, unit in trials:
        err = (got.reindex(want.index) - want).abs()
        err = err.replace([np.inf, -np.inf], np.nan)
        worst = float(err.max())
        verdict = "EXACT" if worst < 1e-9 else "WRONG"
        print(f"  {name:<13}{op:<14}{worst:>21.6f} {unit:<3}  {verdict}")
    print("  " + "-" * 74)
    print("  Two operators are exact and two are not, and the two that are exact belong to")
    print("  different series. There is no single 'continuous contract'; there is a series")
    print("  and the operator it was built for.")

    rule("3. WHAT THE WRONG PAIRING COSTS")
    table = compare_methods(contracts, rolls)
    print(f"  {'method':<13}{'ann. return':>13}{'error vs truth':>16}{'sign flips':>12}"
          f"{'worst 1-day':>13}")
    print("  " + "-" * 74)
    for m, row in table.iterrows():
        err = (f"{row['err_vs_truth_pp']:.1f}pp" if np.isfinite(row["err_vs_truth_pp"])
               else "not finite")
        print(f"  {m:<13}{pct(row['ann_return']):>12}{err:>16}"
              f"{int(row['sign_flip_days']):>12}{pct(row['worst_1d']):>13}")
    print("  " + "-" * 74)
    print(f"  truth: {table.attrs['truth']['ann_return']:>6.1%} a year over "
          f"{table.attrs['n_days']:,} days")

    rule("4. UNDER BACKWARDATION THE BACK-ADJUSTED SERIES GOES NEGATIVE")
    negative = int((diff < 0).sum())
    print(f"  Back-adjustment ADDS each roll gap (P_new - P_old) to all older prices. In")
    print(f"  backwardation that gap is negative, so history is dragged down: {negative:,} of")
    print(f"  {len(diff):,} days are below zero, low point {diff.min():.2f}.")
    cross = int(np.argmin(np.abs(diff.to_numpy())))
    window = slice(max(0, cross - 2), cross + 3)
    print(f"\n  {'date':<12}{'difference':>12}{'pct_change()':>14}{'TRUE return':>13}   verdict")
    print("  " + "-" * 74)
    bad = diff.pct_change(fill_method=None)
    for d in diff.index[window]:
        dr, tr = bad.get(d, np.nan), truth_pct.get(d, np.nan)
        if not (np.isfinite(dr) and np.isfinite(tr)):
            verdict = "not a number"
        elif np.sign(dr) != np.sign(tr):
            verdict = "SIGN FLIPPED"
        else:
            verdict = f"{abs(dr / tr):,.0f}x too big" if abs(dr) > 5 * abs(tr) else "ok"
        print(f"  {d:%Y-%m-%d}{diff[d]:>12.2f}{pct(dr):>14}{tr:>13.2%}   {verdict}")
    print("  " + "-" * 74)
    print(f"  np.log() is NaN on all {int((diff <= 0).sum()):,} of those days, so every "
          f"log-return,")
    print("  volatility and z-score built on this series is NaN or nonsense.")

    rule("5. THE LIBRARY REFUSES INSTEAD OF RETURNING FLOATS")
    for name, series in (("difference", diff), ("unadjusted", unadj), ("ratio", ratio)):
        try:
            r = safe_returns(series)
            print(f"  safe_returns({name:<11}) -> {len(r):,} returns, valid")
        except ValueError as exc:
            print(f"  safe_returns({name:<11}) -> refused: {str(exc).split(':')[1].strip()}")

    rule("TAKEAWAY")
    print("  Build the series for the question. Dollar P&L, margin and stops read the")
    print("  DIFFERENCE-adjusted series with .diff(); returns, volatility and Sharpe read the")
    print("  RATIO-adjusted series with .pct_change(); the UNADJUSTED series is the only one")
    print("  whose levels are real prices. Cross those pairings and pct_change() on the")
    print(f"  unadjusted chain lands {abs(table.loc['unadjusted', 'err_vs_truth_pp']):.0f}pp a "
          f"year from truth, while pct_change() on the")
    print(f"  back-adjusted one flips the sign of "
          f"{int(table.loc['difference', 'sign_flip_days']):,} days and annualises to a number")
    print("  that is not finite - all of it in floats that looked perfectly fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
