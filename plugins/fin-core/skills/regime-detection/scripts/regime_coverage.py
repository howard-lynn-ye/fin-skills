#!/usr/bin/env python3
"""Report which regimes a test period contained, and what the strategy did in each.

`research-integrity-guards/scripts/result_manifest.py` refuses to render a result card whose
`regimes_covered` list is empty ("no regime coverage stated - a single bull quarter is not a
backtest"). The field is a list of strings, so the gate can be satisfied with anything. This
script produces strings that actually carry the evidence:

    <label> [<how the label was defined>]: <n obs> obs / <n episodes> episodes, <first>..<last>,
    asset <ann. return>, strategy Sharpe <x> maxDD <y>

and prints the per-regime table those strings summarise. Two things it insists on:

  * say whether each label is an EX-ANTE rule (a threshold fixed before the test period) or an
    EX-POST label (fitted on, or read off, the test period). Ex-post labels are fine for
    DESCRIBING coverage; they are not fine as an input to the strategy being described.
  * report episodes next to observations. A regime that appears once is one observation of
    that regime no matter how many days it lasted; the episode-clustered standard error below
    is the honest uncertainty of a per-regime number.

Run:  python regime_coverage.py     (numpy / pandas; fixed seed; ~1 s)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .regime_lookahead import N, PERIODS, TEST_START, simulate
except ImportError:
    from regime_lookahead import N, PERIODS, TEST_START, simulate

SEED = 0
RATIO = 2.0
VOL_WINDOW = 21
VOL_CUT_ANN = 0.15          # fixed before the test period: >= 15% annualised = "high vol"
DD_LIMIT = 0.10
SMA = 200


# ----------------------------------------------------------------------------- labels -------
def episodes(flag: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs where flag is True, as (start, end) index pairs, end exclusive."""
    out, start = [], None
    for i, f in enumerate(np.r_[flag, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def regime_table(dates: pd.DatetimeIndex, asset: np.ndarray, strat: np.ndarray,
                 pos: np.ndarray, labels: pd.Series, how: str) -> tuple[pd.DataFrame, list[str]]:
    rows, strings = [], []
    for lab in pd.unique(labels):
        m = (labels == lab).to_numpy()
        eps = episodes(m)
        a, s_, p_ = asset[m], strat[m], pos[m]
        n = int(m.sum())
        sd = s_.std(ddof=1) if n > 1 else np.nan
        sharpe = float(s_.mean() / sd * np.sqrt(PERIODS)) if sd and sd > 0 else float("nan")
        eq = np.cumprod(1 + s_)
        maxdd = float((eq / np.maximum.accumulate(eq) - 1).min()) if n else float("nan")
        idx = np.flatnonzero(m)
        # the strategy's outcome per EPISODE of this regime: that count is the sample size for
        # any claim of the form "the strategy survives this kind of period"
        ep_ret = np.array([np.prod(1 + strat[b:e]) - 1 for b, e in eps])
        in_mkt = float((p_ != 0).mean())
        row = {"label": lab, "n_obs": n, "share": n / len(m), "n_episodes": len(eps),
               "longest_ep": max((e - b for b, e in eps), default=0),
               "first": dates[idx[0]].date(), "last": dates[idx[-1]].date(),
               "asset_ann": float(a.mean() * PERIODS), "strat_sharpe": sharpe,
               "strat_maxdd": maxdd, "in_mkt": in_mkt,
               "hit_rate": float((s_[p_ != 0] > 0).mean()) if (p_ != 0).any() else float("nan"),
               "ep_ret_mean": float(ep_ret.mean()), "ep_ret_min": float(ep_ret.min()),
               "ep_ret_max": float(ep_ret.max()),
               "ep_tstat": float(ep_ret.mean() / ep_ret.std(ddof=1) * np.sqrt(len(ep_ret)))
               if len(ep_ret) > 1 and ep_ret.std(ddof=1) > 0 else float("nan"),
               "worst_ep_share": float(ep_ret.min() / ep_ret[ep_ret < 0].sum())
               if (ep_ret < 0).any() else float("nan")}
        rows.append(row)
        strings.append(f"{lab} [{how}]: {n} obs / {len(eps)} episodes, {row['first']}..{row['last']},"
                       f" asset {row['asset_ann']:+.1%}/yr, strategy Sharpe {sharpe:.2f}"
                       f" (in market {in_mkt:.0%}) maxDD {maxdd:.1%}")
    return pd.DataFrame(rows).set_index("label"), strings


# ----------------------------------------------------------------------------- demo ---------
if __name__ == "__main__":
    t0 = time.time()
    lo, hi = TEST_START, N
    r_all, s_all = simulate(N, SEED, RATIO)
    dates_all = pd.bdate_range("2016-01-04", periods=N)
    price = np.exp(np.cumsum(r_all))

    # A toy strategy to report on: long when yesterday's close is above its 200-day average.
    sma = pd.Series(price).rolling(SMA).mean().shift(1).to_numpy()
    pos = (np.r_[np.nan, price[:-1]] > sma).astype(float)
    strat_all = pos * r_all

    # Ex-ante regime rules (every threshold here is a number fixed before looking at the window)
    rv_ann = pd.Series(r_all).rolling(VOL_WINDOW).std(ddof=1).shift(1).to_numpy() * np.sqrt(PERIODS)
    vol_lab = np.where(rv_ann >= VOL_CUT_ANN, f"high-vol (>= {VOL_CUT_ANN:.0%} ann, {VOL_WINDOW}d)",
                       f"low-vol (< {VOL_CUT_ANN:.0%} ann, {VOL_WINDOW}d)")
    peak = np.maximum.accumulate(price)
    dd_lab = np.where(np.r_[False, (price / peak - 1 < -DD_LIMIT)[:-1]],
                      f"drawdown (> {DD_LIMIT:.0%} below peak)", "at-or-near-peak")

    dates, r, strat = dates_all[lo:hi], r_all[lo:hi], strat_all[lo:hi]
    print(f"Test period {dates[0].date()}..{dates[-1].date()} ({hi - lo} obs); strategy = long when"
          f" close[t-1] > {SMA}d SMA; DGP from regime_lookahead.py (ratio {RATIO}, seed {SEED})")
    all_strings: list[str] = []
    axes = [("vol", pd.Series(vol_lab[lo:hi]), "ex-ante rule"),
            ("drawdown", pd.Series(dd_lab[lo:hi]), "ex-ante rule"),
            ("true DGP regime", pd.Series(np.where(s_all[lo:hi] == 1, "turbulent", "calm")),
             "ex-post label: known only because the data are simulated")]
    tabs = {}
    for axis, labels, how in axes:
        tab, strings = regime_table(dates, r, strat, pos[lo:hi], labels, how)
        tabs[axis] = tab
        all_strings += strings
        print(f"\n=== Regime axis: {axis} ({how}) ===")
        cols = ["n_obs", "share", "n_episodes", "longest_ep", "first", "last", "asset_ann",
                "strat_sharpe", "strat_maxdd", "in_mkt", "hit_rate"]
        print(tab[cols].to_string(float_format=lambda x: f"{x:8.3f}"))

    print("\n=== regimes_covered = [  <- paste into ResultCard(...) ===")
    for s_ in all_strings:
        print(f'    "{s_}",')
    print("]")

    print("\n=== One crisis is one observation: the strategy's outcome per EPISODE ===")
    print("  (a per-regime Sharpe is computed from n_obs days, but 'it survives this kind of"
          " period' is a claim about n_episodes things)")
    for axis in tabs:
        for lab, row in tabs[axis].iterrows():
            print(f"  {lab[:44]:<44} episodes={int(row.n_episodes):3d}  return per episode:"
                  f" mean {row.ep_ret_mean:+.2%} min {row.ep_ret_min:+.2%} max {row.ep_ret_max:+.2%}"
                  f"  t-stat over episodes {row.ep_tstat:5.2f}"
                  + (f"  worst episode = {row.worst_ep_share:.0%} of all episode losses"
                     if np.isfinite(row.worst_ep_share) else ""))

    # If the gate itself is importable from the sibling skill, show it accepting the list.
    gate = Path(__file__).resolve().parents[2] / "research-integrity-guards" / "scripts"
    if (gate / "result_manifest.py").exists():
        sys.path.insert(0, str(gate))
        from result_manifest import (CostModel, DataSource, ResultCard, Split, TrialCount,
                                     Universe)

        def card(regimes):
            return ResultCard(
                strategy_id="demo-sma-trend", universe=Universe("simulated", "2026-09-08", True, 1,
                                                                 "single simulated asset"),
                data=[DataSource("regime_lookahead.simulate", "2026-09-08", "none")],
                split=Split("fixed window", "0D", "0D", "obs 0-755", "obs 756-2519"),
                costs=CostModel(0.0, 0.0, "none"), trials=TrialCount(1, "n/a"),
                metrics={"annualization": PERIODS, "rf_convention": "rf=0"},
                cost_curve={0: 1.0, 10: 0.9, 20: 0.8}, benchmark={"capm_alpha": 0.0},
                falsifier="demo", regimes_covered=regimes)

        before = [p for p in card([]).problems() if "regime" in p]
        after = [p for p in card(all_strings).problems() if "regime" in p]
        print(f"\n=== result_manifest.py gate ===")
        print(f"  with regimes_covered=[]        : {before}")
        print(f"  with the {len(all_strings)} strings above: {after or 'no regime problem'}")
        print("  (the other gate problems in this demo card - trial count 1 and so on - are"
              " deliberately left alone; they belong to backtest-validation)")
    print(f"\ntotal runtime {time.time() - t0:.1f}s")
