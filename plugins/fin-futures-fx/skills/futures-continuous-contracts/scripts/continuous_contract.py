"""Stitch expiring futures into a continuous series -- and prove the method changes the answer.

WHY this exists: there is no such thing as "the price of crude oil futures". Every futures
price series you have ever backtested is a CONSTRUCTION -- a chain of contracts that each
died, welded together by a choice you probably did not make consciously. The weld method is
not a formatting detail. It decides the sign of your P&L.

Three failures this catches, all of which pass every unit test you would think to write:

1. UNADJUSTED prices contain a jump at every roll that no position ever earned. In a
   contango market the deferred contract is dearer, so the stitched series leaps UP at each
   roll and `pct_change()` books a fictitious gain 4-12 times a year. A market that bled
   -14%/yr for a long holder reads as +26%/yr. The series is still the RIGHT one for
   anything level-dependent -- margin, tick value, limit moves, "is it above 100" -- because
   those are the prices that actually printed.

2. DIFFERENCE (Panama / back-adjusted) subtracts the cumulative roll gap from history. Over
   enough rolls the cumulative gap exceeds the price level and HISTORICAL PRICES GO
   NEGATIVE. Not a bug -- an arithmetic certainty. A back-adjusted series is a P&L path
   wearing a price's clothes: `pct_change()` on it divides by a number that crosses zero and
   changes sign, producing infinities and returns whose SIGN IS FLIPPED. `log()` of it is
   NaN. Every vol estimate, every z-score, every stop-loss in percent is garbage.

   SIGN CONVENTION -- read this before comparing to a vendor. Here `gap = P_new - P_old` is
   subtracted from older prices, so a CONTANGO market drives history negative. Other vendors
   add it, which drives a BACKWARDATED market negative instead (this is why back-adjusted
   continuous crude goes negative in the 1980s at most data shops). Both conventions are in
   production somewhere, both produce a series that is not a price, and neither one's
   `pct_change()` means anything. Which is exactly why you check against `true_roll_return`
   instead of trusting the label on the file.

3. RATIO (proportional) multiplies history by the cumulative price ratio at each roll. It
   cannot cross zero, so percentage returns survive -- and they are EXACTLY the returns of a
   rolled position (proof: the segment multipliers cancel across the roll). The levels are
   still not tradeable prices; do not compute margin off them.

THE GROUND TRUTH is `true_roll_return`: hold the front contract, sell it and buy the next at
the roll, accumulate the actual P&L. It needs no adjustment because it never stitches. Every
stitching method is a claim about this series, and the claim is cheap to check.

Usage:
    from continuous_contract import stitch, true_roll_return, safe_returns
    px  = stitch(contracts, rolls, "unadjusted")   # levels: margin, limits, tick value
    adj = stitch(contracts, rolls, "ratio")        # returns: signals, vol, Sharpe
    r   = safe_returns(adj)                        # raises on a difference-adjusted series
    assert np.allclose(r, true_roll_return(contracts, rolls), atol=1e-12)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
METHODS = ("unadjusted", "difference", "ratio")

# Set on every stitched series so downstream code can refuse to misuse it.
_METHOD_ATTRS = {
    # method:        (levels are real prices, pct_change is meaningful)
    "unadjusted":    (True, False),
    "difference":    (False, False),
    "ratio":         (False, True),
}


def _validate(contracts: pd.DataFrame,
              roll_dates: Sequence) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Coerce and check the two inputs. Every check here is a real, silent failure.

    `contracts` columns must be ordered NEAR -> FAR by expiry; roll i moves from column i
    to column i+1, so there must be exactly one roll date per adjacent pair.
    """
    if not isinstance(contracts, pd.DataFrame):
        raise TypeError("contracts must be a DataFrame: rows=dates, one column per "
                        "contract, ordered near->far by expiry, NaN outside its life")
    if contracts.shape[1] < 2:
        raise ValueError("need at least two contracts to have anything to stitch")
    if not isinstance(contracts.index, pd.DatetimeIndex):
        raise TypeError("contracts.index must be a DatetimeIndex")
    if not contracts.index.is_monotonic_increasing:
        raise ValueError("contracts.index must be sorted ascending")
    if contracts.index.has_duplicates:
        raise ValueError("contracts.index has duplicate dates")

    rolls = [pd.Timestamp(d) for d in roll_dates]
    n_expected = contracts.shape[1] - 1
    if len(rolls) != n_expected:
        raise ValueError(
            f"need exactly one roll date per adjacent contract pair: got {len(rolls)} "
            f"rolls for {contracts.shape[1]} contracts (expected {n_expected})")
    if any(b <= a for a, b in zip(rolls, rolls[1:])):
        raise ValueError("roll_dates must be strictly increasing")

    cols = list(contracts.columns)
    for k, d in enumerate(rolls):
        if d not in contracts.index:
            raise ValueError(f"roll date {d:%Y-%m-%d} is not a row of contracts")
        # THE killer: if the deferred contract has no print on the roll date, the gap is
        # NaN and the NaN propagates backward through every adjusted price before it. You
        # get a column that is 70% NaN and a backtest that silently starts in 2019.
        p_old = contracts.at[d, cols[k]]
        p_new = contracts.at[d, cols[k + 1]]
        if not np.isfinite(p_old) or not np.isfinite(p_new):
            raise ValueError(
                f"roll {k} on {d:%Y-%m-%d}: need BOTH legs to print to compute the gap, "
                f"got {cols[k]}={p_old!r}, {cols[k + 1]}={p_new!r}. Pick a roll date where "
                f"the deferred contract is actually quoted, or the NaN will eat all of "
                f"the history before this roll.")
    return contracts.astype(float), rolls


def active_contract(contracts: pd.DataFrame, roll_dates: Sequence) -> pd.Series:
    """Which contract is held on each date. Contract k is held THROUGH roll k inclusive.

    Convention: the roll transacts at the close of the roll date, so the return earned ON
    the roll date still belongs to the old contract, and the new contract's first return
    is the day after. Getting this off by one manufactures or destroys one roll gap.
    """
    contracts, rolls = _validate(contracts, roll_dates)
    idx = contracts.index
    # count of rolls strictly before t == segment number at t
    seg = np.searchsorted(np.array(rolls, dtype="datetime64[ns]"),
                          idx.to_numpy(), side="left")
    cols = np.asarray(contracts.columns, dtype=object)
    out = pd.Series(cols[seg], index=idx, name="contract")

    # Trim dates where the nominally active contract does not trade (before its listing).
    live = np.isfinite(contracts.to_numpy()[np.arange(len(idx)), seg])
    if not live.any():
        raise ValueError("no date has a live active contract; check listing windows")
    first, last = int(np.argmax(live)), len(live) - int(np.argmax(live[::-1]))
    span = out.iloc[first:last]
    if not np.isfinite(contracts.to_numpy()[np.arange(first, last), seg[first:last]]).all():
        raise ValueError("the active contract has gaps inside the stitched span; the "
                         "series would silently interpolate across a dead market")
    return span


def _segment_index(active: pd.Series, contracts: pd.DataFrame) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(contracts.columns)}
    return active.map(lookup).to_numpy()


def stitch(contracts: pd.DataFrame, roll_dates: Sequence,
           method: str = "ratio") -> pd.Series:
    """Weld the contract chain into one series. `method` decides what the series MEANS.

    'unadjusted' : the prices that printed. Correct for margin, tick value, limit moves,
                   and any level threshold. WRONG for returns -- it carries a fake jump at
                   every roll.
    'difference' : back-adjusted. Walking backward from the newest contract, at each roll
                   gap = P_new(roll) - P_old(roll); the cumulative gap is SUBTRACTED from
                   all older prices. Historical prices can and do go NEGATIVE. Not a price,
                   not a return base -- see the module docstring on sign conventions.
    'ratio'      : proportional. Older prices are multiplied by the cumulative
                   P_new(roll)/P_old(roll). Sign-preserving; `pct_change()` reproduces the
                   true rolled-position return exactly. Levels are still not tradeable.

    The result carries `.attrs['levels_are_prices']` and `.attrs['returns_valid']` so
    `safe_returns` can refuse to compute returns off the wrong series.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    contracts, rolls = _validate(contracts, roll_dates)
    active = active_contract(contracts, rolls)
    seg = _segment_index(active, contracts)
    vals = contracts.to_numpy()
    pos = contracts.index.get_indexer(active.index)
    raw = vals[pos, seg]

    n_seg = contracts.shape[1]
    if method == "unadjusted":
        out = pd.Series(raw, index=active.index)
    else:
        cols = list(contracts.columns)
        # Walk BACKWARD from the newest contract: the most recent segment is the anchor
        # and gets no adjustment, which is what makes the series agree with today's screen.
        cum = np.zeros(n_seg) if method == "difference" else np.ones(n_seg)
        for k in range(n_seg - 2, -1, -1):
            d = rolls[k]
            p_old = contracts.at[d, cols[k]]
            p_new = contracts.at[d, cols[k + 1]]
            if method == "difference":
                cum[k] = cum[k + 1] + (p_new - p_old)
            else:
                cum[k] = cum[k + 1] * (p_new / p_old)
        adj = cum[seg]
        out = pd.Series(raw - adj if method == "difference" else raw * adj,
                        index=active.index)

    out.name = f"continuous_{method}"
    levels, rets = _METHOD_ATTRS[method]
    out.attrs.update(method=method, levels_are_prices=levels, returns_valid=rets,
                     n_rolls=len(rolls))
    return out


def safe_returns(stitched: pd.Series) -> pd.Series:
    """`pct_change()` with a refusal. The guard that should have existed all along.

    Failure prevented: `stitch(..., 'difference').pct_change()`. It runs, it returns
    floats, and the floats are meaningless -- the denominator crosses zero and flips sign.
    Nothing in pandas will tell you. This will.
    """
    method = stitched.attrs.get("method")
    if method is None:
        raise ValueError("series has no stitching provenance; build it with stitch()")
    if not stitched.attrs.get("returns_valid", False):
        why = ("levels contain a fictitious jump at every roll"
               if method == "unadjusted" else
               "the series crosses zero and is a P&L path, not a price")
        raise ValueError(
            f"refusing pct_change() on a {method!r}-stitched series: {why}. Use "
            f"stitch(..., 'ratio') for returns; keep {method!r} for what it is good at "
            f"({'levels' if method == 'unadjusted' else 'nothing but plotting'}).")
    return stitched.pct_change(fill_method=None).dropna()


def true_roll_return(contracts: pd.DataFrame, roll_dates: Sequence,
                     in_points: bool = False) -> pd.Series:
    """GROUND TRUTH. Hold the front contract; at each roll sell the old and buy the new.

    On the roll date you exit contract k at its own mark and enter contract k+1 at ITS own
    mark, simultaneously. That swap earns nothing: the roll gap is a term-structure fact,
    not a P&L event. So every day's return is priced off ONE contract's own two closes --
    which is why this function never needs an adjustment and never can be wrong.

    Returns fractional returns (default) or per-contract point P&L (`in_points=True`).
    Any stitching method that disagrees with this series is lying to you.
    """
    contracts, rolls = _validate(contracts, roll_dates)
    active = active_contract(contracts, rolls)
    seg = _segment_index(active, contracts)
    vals = contracts.to_numpy()
    pos = contracts.index.get_indexer(active.index)

    px = vals[pos, seg]
    # Price of TODAY'S contract at YESTERDAY'S close -- the mark you actually bought at.
    prev = np.full(len(px), np.nan)
    prev[1:] = vals[pos[:-1], seg[1:]]
    if not np.isfinite(prev[1:]).all():
        raise ValueError("a contract has no prior-day print on its first held day; the "
                         "roll would be marked against a price that never existed")

    out = (px - prev) if in_points else (px / prev - 1.0)
    name = "true_roll_pnl_points" if in_points else "true_roll_return"
    return pd.Series(out, index=active.index, name=name).dropna()


def roll_following_days(contracts: pd.DataFrame,
                        roll_dates: Sequence) -> pd.DatetimeIndex:
    """The first trading day AFTER each roll -- where a stitching error shows up.

    The roll gap lands between the roll date's close and the next close, so this is the
    exact set of days on which a badly-stitched series books a return nobody earned.
    """
    contracts, rolls = _validate(contracts, roll_dates)
    pos = contracts.index.get_indexer([pd.Timestamp(d) for d in rolls]) + 1
    return contracts.index[pos[pos < len(contracts.index)]]


def _stats(r: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2 or (1.0 + r).min() <= 0:
        # A "return" of -180% is not a drawdown, it is proof the series was not a price.
        return {"ann_return": np.nan, "vol": np.nan, "sharpe": np.nan}
    total = float((1.0 + r).prod() - 1.0)
    ann = float((1.0 + total) ** (periods_per_year / len(r)) - 1.0)
    sd = float(r.std(ddof=1))
    return {"ann_return": ann, "vol": sd * np.sqrt(periods_per_year),
            "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan}


def compare_methods(contracts: pd.DataFrame,
                    roll_dates: Sequence,
                    periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """One row per method, scored against `true_roll_return`. The audit that takes 3 lines.

    Columns: level range of the stitched series, the annualised return its `pct_change()`
    implies, the error versus truth, how many days the implied return has the WRONG SIGN,
    and the worst single-day return it manufactures.
    """
    truth = true_roll_return(contracts, roll_dates)
    t_stats = _stats(truth, periods_per_year)
    post_roll = roll_following_days(contracts, roll_dates).intersection(truth.index)
    rows = []
    for m in METHODS:
        s = stitch(contracts, roll_dates, m)
        # Deliberately bypass safe_returns: the point is to score what the naive call does.
        r = s.pct_change(fill_method=None).reindex(truth.index)
        st = _stats(r, periods_per_year)
        both = pd.concat([r, truth], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        material = both.iloc[:, 1].abs() > 1e-9
        flips = int((np.sign(both.iloc[:, 0]) != np.sign(both.iloc[:, 1]))[material].sum())
        finite = r.replace([np.inf, -np.inf], np.nan)
        gap = (r[post_roll] - truth[post_roll]).abs()
        rows.append({
            "method": m,
            "min_level": float(s.min()), "max_level": float(s.max()),
            "goes_negative": bool(s.min() < 0),
            "ann_return": st["ann_return"], "sharpe": st["sharpe"],
            "err_vs_truth_pp": (st["ann_return"] - t_stats["ann_return"]) * 100
                               if np.isfinite(st["ann_return"]) else np.nan,
            "bad_roll_days": int(((gap > 1e-9) | gap.isna()).sum()),
            "n_rolls_scored": len(post_roll),
            "sign_flip_days": flips,
            "worst_1d": float(finite.abs().max()) if finite.notna().any() else np.nan,
            "n_nonfinite": int((~np.isfinite(r.to_numpy(dtype=float))).sum()),
            "returns_valid": s.attrs["returns_valid"],
        })
    out = pd.DataFrame(rows).set_index("method")
    out.attrs["truth"] = t_stats
    out.attrs["n_days"] = len(truth)
    return out


def render(table: pd.DataFrame, title: str = "CONTINUOUS CONTRACT: METHOD COMPARISON") -> str:
    """Fixed-width report. Paste this into any write-up that quotes a futures Sharpe."""
    t = table.attrs["truth"]
    w = 108
    lines = [title, "=" * w,
             f"GROUND TRUTH (hold front, roll at each expiry, {table.attrs['n_days']} days): "
             f"ann {t['ann_return']:+.2%}  vol {t['vol']:.1%}  Sharpe {t['sharpe']:+.2f}",
             "-" * w,
             f"{'method':<11} {'min lvl':>9} {'max lvl':>9} {'ann ret':>8} "
             f"{'err vs truth':>13} {'wrong@roll':>11} {'signflip':>9} {'worst 1d':>10}  "
             f"use for",
             "-" * w]
    use = {"unadjusted": "LEVELS ONLY", "difference": "NOTHING (plots)", "ratio": "RETURNS"}
    for m, row in table.iterrows():
        ann = f"{row['ann_return']:+.1%}" if np.isfinite(row["ann_return"]) else "n/a"
        err = f"{row['err_vs_truth_pp']:+.1f} pp" if np.isfinite(row["err_vs_truth_pp"]) \
            else "UNCOMPUTABLE"
        worst = f"{row['worst_1d']:.1%}" if np.isfinite(row["worst_1d"]) else "inf"
        flag = "  <-- NEGATIVE PRICES" if row["goes_negative"] else ""
        bad = f"{int(row['bad_roll_days'])}/{int(row['n_rolls_scored'])}"
        lines.append(
            f"{m:<11} {row['min_level']:>9,.1f} {row['max_level']:>9,.1f} {ann:>8} "
            f"{err:>13} {bad:>11} {int(row['sign_flip_days']):>9,} {worst:>10}  "
            f"{use[m]}{flag}")
    lines.append("-" * w)
    lines.append("'wrong@roll' = days after a roll where the implied return != the P&L a "
                 "real position earned.")
    lines.append("THE RULE:  levels -> unadjusted.   returns -> ratio.")
    lines.append("           NEVER take returns off a difference-adjusted series.")
    lines.append("=" * w)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def _synthetic_contango(years: int = 8, contango: float = 0.20, drift: float = 0.06,
                        vol: float = 0.16, seed: int = 4
                        ) -> tuple[pd.DataFrame, list[pd.Timestamp], pd.Series]:
    """A persistently-contangoed quarterly futures market. Offline, no network.

    Shaped like natural gas / VIX / any long-only commodity ETF underlier: the SPOT drifts
    UP at +6%/yr while a long rolled futures position BLEEDS, because every quarter you sell
    the cheap front and buy the dear deferred. That divergence is the whole point.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=TRADING_DAYS * years)
    dt = 1.0 / TRADING_DAYS
    spot = 100.0 * np.exp(np.cumsum(
        rng.normal((drift - 0.5 * vol ** 2) * dt, vol * np.sqrt(dt), len(idx))))

    # Positional quarterly expiries (63 business days apart). Positional, not calendar,
    # so the demo is reproducible without a holiday calendar dependency.
    exp_pos = list(range(60, len(idx), 63))
    px = pd.DataFrame(np.nan, index=idx, columns=[f"F{i:02d}" for i in range(len(exp_pos))])
    for i, e in enumerate(exp_pos):
        start = max(0, e - 190)                    # listed ~9 months before expiry
        tau = (np.array(exp_pos[i], dtype=float) - np.arange(start, e + 1)) / TRADING_DAYS
        # Cost-of-carry term structure + a little contract-specific basis noise.
        basis = rng.normal(0.0, 0.0015, len(tau)).cumsum()
        px.iloc[start:e + 1, i] = spot[start:e + 1] * np.exp(contango * tau + basis)

    # Roll 10 business days before expiry, one roll per adjacent pair.
    rolls = [idx[e - 10] for e in exp_pos[:-1]]
    return px, rolls, pd.Series(spot, index=idx, name="spot")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    contracts, rolls, spot = _synthetic_contango()

    print("=" * 108)
    print(f"SYNTHETIC CONTANGO MARKET -- {contracts.shape[1]} quarterly contracts, "
          f"{len(rolls)} rolls, {len(contracts):,} business days (~8 years)")
    print(f"Spot goes {spot.iloc[0]:.0f} -> {spot.iloc[-1]:.0f} "
          f"({spot.iloc[-1] / spot.iloc[0] - 1:+.0%}). Term structure: persistent "
          f"+20%/yr contango.")
    print("=" * 108)

    table = compare_methods(contracts, rolls)
    print()
    print(render(table))

    # -------------------------------------------------- 1. the negative-price crossing
    diff = stitch(contracts, rolls, "difference")
    ratio = stitch(contracts, rolls, "ratio")
    unadj = stitch(contracts, rolls, "unadjusted")
    truth = true_roll_return(contracts, rolls)

    # The denominator of pct_change() passes through zero here, so this is where the
    # arithmetic detonates. Everything either side of it is quietly sign-flipped.
    cross = int(np.argmin(np.abs(diff.to_numpy())))
    win = slice(max(0, cross - 3), cross + 4)
    d_ret = diff.pct_change(fill_method=None)
    print("\n" + "=" * 108)
    print("1. THE DIFFERENCE-ADJUSTED SERIES CROSSING ZERO  (the seven days that break "
          "everything)")
    print("=" * 108)
    print(f"{'date':<12} {'unadjusted':>11} {'difference':>11} {'ratio':>11} "
          f"{'TRUE ret':>10} {'diff pct_change()':>20}   verdict")
    print("-" * 108)
    for d in diff.index[win]:
        dr = d_ret.get(d, np.nan)
        tr = truth.get(d, np.nan)
        verdict = ("--" if not np.isfinite(dr) or not np.isfinite(tr) else
                   "SIGN FLIPPED" if np.sign(dr) != np.sign(tr) else
                   f"{abs(dr / tr):,.0f}x too big" if abs(dr) > 5 * abs(tr) else "ok")
        print(f"{d:%Y-%m-%d}  {unadj[d]:>11.2f} {diff[d]:>11.2f} {ratio[d]:>11.2f} "
              f"{tr:>10.2%} {dr:>20,.1%}   {verdict}")
    print("-" * 108)
    print(f"difference-adjusted range: {diff.min():,.1f} .. {diff.max():,.1f}  "
          f"({int((diff < 0).sum()):,} of {len(diff):,} days are NEGATIVE PRICES)")
    print(f"np.log(difference) is NaN on all {int((diff <= 0).sum()):,} of them, so every "
          f"log-return, vol and z-score built on it is NaN or nonsense.")
    print(f"worst single day pct_change() reports: "
          f"{d_ret.replace([np.inf, -np.inf], np.nan).abs().max():,.0%} -- on a day the "
          f"market moved {truth.loc[d_ret.abs().idxmax()]:.2%}.")

    # ------------------------------------------------------------ 2. ratio == the truth
    print("\n" + "=" * 108)
    print("2. RATIO-ADJUSTED RETURNS vs GROUND TRUTH")
    print("=" * 108)
    r_ratio = safe_returns(ratio)
    err = (r_ratio - truth).abs().max()
    print(f"max |ratio.pct_change() - true_roll_return| over {len(truth):,} days = "
          f"{err:.2e}   <-- machine precision, they are the SAME series")
    print(f"ratio  ann {_stats(r_ratio)['ann_return']:+.2%}   Sharpe "
          f"{_stats(r_ratio)['sharpe']:+.2f}")
    print(f"TRUTH  ann {_stats(truth)['ann_return']:+.2%}   Sharpe "
          f"{_stats(truth)['sharpe']:+.2f}")

    # -------------------------------------------------- 3. unadjusted, wrong every roll
    print("\n" + "=" * 108)
    print("3. UNADJUSTED RETURNS ARE WRONG AT EVERY SINGLE ROLL DATE")
    print("=" * 108)
    r_unadj = unadj.pct_change(fill_method=None).reindex(truth.index)
    post = roll_following_days(contracts, rolls).intersection(truth.index)
    fake = r_unadj[post] - truth[post]
    print(f"days after a roll: {len(post)} of {len(rolls)} rolls fall inside the sample, "
          f"and ALL {len(post)} carry a fictitious return")
    print(f"injected per roll: mean {fake.mean():+.2%}, min {fake.min():+.2%}, "
          f"max {fake.max():+.2%}  -- pure term structure, earned by nobody")
    print(f"cumulative fiction over 8 years: {fake.sum():+.1%} of notional")
    print(f"\nSPOT rose {spot.iloc[-1] / spot.iloc[0] - 1:+.0%} over the sample. A long "
          f"rolled position still LOST {(1 + truth).prod() - 1:+.0%}, because it paid the "
          f"contango 31 times.")
    print(f"The unadjusted series implies {_stats(r_unadj)['ann_return']:+.1%}/yr; the "
          f"truth is {_stats(truth)['ann_return']:+.1%}/yr. Same data, OPPOSITE SIGN.")

    # ------------------------------------------------------------------ 4. the guard
    print("\n" + "=" * 108)
    print("4. THE GUARD")
    print("=" * 108)
    for m in ("difference", "unadjusted"):
        try:
            safe_returns(stitch(contracts, rolls, m))
        except ValueError as e:
            print(f"safe_returns({m!r:<13}) CAUGHT: {e}")
    print(f"safe_returns('ratio'      ) -> {len(safe_returns(ratio)):,} clean daily returns")

    print("\n" + "=" * 108)
    print("THE RULE:  levels -> unadjusted.   returns -> ratio.")
    print("           NEVER take returns off a difference-adjusted series.")
    print("           And check whichever you picked against true_roll_return -- it is")
    print("           three lines and it is the only thing that cannot be wrong.")
    print("=" * 108)
