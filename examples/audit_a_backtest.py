#!/usr/bin/env python3
"""Audit a backtest with one Bundle and one check() call.

A research run leaves artefacts behind: returns, turnover, the bars a signal saw, the
signal function itself, the price panel, a regime label. Put them in a Bundle under the
shared vocabulary and every guard that can read them runs, in one call. The guards you
cannot run are as useful as the ones you can -- coverage() names them and says which single
missing slot would unlock each.

This run has two planted defects. We look at coverage, run the guards, read the failures,
fix both, and run again.

    python examples/audit_a_backtest.py        (offline, seeded, a few seconds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.api import Bundle, check, registry                          # noqa: E402

SEED = 20260909
N_DAYS = 500
N_NAMES = 12


def synthetic_run(seed: int = SEED) -> dict:
    """One small research run. Everything here is what a real pipeline would already have."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=N_DAYS)

    # --- a price panel where one name stops trading: NaN after it delists, not dropped.
    steps = rng.normal(0.0003, 0.013, (N_DAYS, N_NAMES))
    prices = pd.DataFrame(80.0 * np.exp(np.cumsum(steps, axis=0)), index=dates,
                          columns=[f"N{i:02d}" for i in range(N_NAMES)])
    prices.iloc[300:, prices.columns.get_loc("N07")] = np.nan       # N07 leaves the tape

    # --- one instrument's bars, the input a signal function actually sees
    close = prices["N00"]
    bars = pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.008,
                         "low": close * 0.992, "close": close,
                         "volume": rng.integers(2_000, 9_000, N_DAYS)}, index=dates)

    # --- the strategy: modest edge, 9% one-way turnover a day
    returns = pd.Series(rng.normal(0.00055, 0.0075, N_DAYS), index=dates, name="strategy")
    turnover = pd.Series(np.clip(rng.lognormal(np.log(0.09), 0.3, N_DAYS), 0, 1), index=dates)

    # --- a regime label with eight episodes, and the market the strategy trades in
    episodes = np.repeat(["calm", "turbulent"] * 4, N_DAYS // 8 + 1)[:N_DAYS]
    regime = pd.Series(episodes, index=dates)
    underlying = prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    return {"dates": dates, "prices": prices, "close": close, "bars": bars,
            "returns": returns, "turnover": turnover, "regime": regime,
            "underlying": underlying, "position": pd.Series(1.0, index=dates)}


def honest_signal(df: pd.DataFrame) -> pd.Series:
    """A 20-bar mean of the close. Reads bar t and earlier, and nothing else."""
    return df["close"].rolling(20).mean()


def peeking_signal(df: pd.DataFrame) -> pd.Series:
    """DEFECT 1. The same mean, shifted so bar t reads bar t+1. One character, whole result."""
    return df["close"].rolling(20).mean().shift(-1)


def rule(title: str) -> None:
    print("\n" + title)
    print("=" * 78)


def report_lines(report) -> None:
    for r in report:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.guard:<20} {r.elapsed_s:6.3f}s")
        for f in r.errors:
            print(f"        -> {f}")
    for name in report.rejected:
        print(f"  SKIP  {name:<20}  refused: either/or inputs, none of them given")


def main() -> int:
    run = synthetic_run()

    # ---------------------------------------------------------------- the defective bundle
    # DEFECT 1: the signal function peeks one bar ahead.
    # DEFECT 2: the price panel was filtered to names still trading at the end, so the one
    #           that delisted has been deleted from history rather than left as NaN.
    survivors = [c for c in run["prices"].columns if run["prices"][c].notna().iloc[-1]]
    bad = Bundle(returns=run["returns"], turnover=run["turnover"],
                 bars=run["bars"], signal_fn=peeking_signal,
                 prices=run["prices"][survivors],
                 dates=run["dates"], underlying_returns=run["underlying"],
                 position=run["position"], regime_labels=run["regime"])

    rule("1. COVERAGE - what can this run be checked with, and what is one slot away?")
    cov = bad.coverage()
    print(f"  ready ({len(cov.ready)}): {', '.join(cov.ready)}")
    print(f"  not ready: {len(cov.missing)} guard(s) lack inputs this run does not have")
    print("\n  one slot away:")
    for slot, guards in cov.unlocks().items():
        print(f"    + {slot:<20} would unlock {', '.join(guards)}")
    print("\n  Coverage is the honest denominator. 'All checks passed' means nothing until")
    print("  you know how many checks could run at all.")

    # ---------------------------------------------------------------- run them
    rule("2. CHECK - every guard whose inputs are present, one call")
    report = check(bad)
    print(f"  {report.summary().splitlines()[0]}\n")
    report_lines(report)
    print("\n  Two guards declare no REQUIRED slot because they take either/or inputs, so")
    print("  coverage lists them as ready and they refuse at the door instead. A refusal is")
    print("  recorded, never raised: one bad input cannot abort the other guards.")

    rule("3. WHAT THE TWO FAILURES MEAN")
    print("  assert_causal      re-ran the signal on truncated history. The value at bar t")
    print("                     changed when bars after t were removed, so it read them.")
    print("  survivorship_audit found no name whose series ends before the panel does. In")
    print("                     500 days across 12 names, nothing failing is not luck.")

    # ---------------------------------------------------------------- fix and re-run
    rule("4. FIXED - the same bundle with the two defects repaired")
    good = bad.with_(signal_fn=honest_signal, prices=run["prices"])
    fixed = check(good)
    print(f"  {fixed.summary().splitlines()[0]}\n")
    report_lines(fixed)

    failed_before = sorted(r.guard for r in report.failed)
    failed_after = sorted(r.guard for r in fixed.failed)

    rule("TAKEAWAY")
    print("  Two lines changed: `.shift(-1)` came off the signal, and the delisted name went")
    print(f"  back into the panel. {len(failed_before)} guards went FAIL -> PASS: "
          f"{', '.join(failed_before)}.")
    print(f"  Of the {len(registry())} guards in the registry, {len(fixed)} could read this "
          f"run, and all {len(fixed)} now pass.")
    print("  A Bundle is the fit(X) of this library: assemble the run once, and every check")
    print("  that can read it runs - including the ones you would not have thought to write.")
    assert not failed_after, failed_after
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
