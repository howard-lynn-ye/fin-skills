#!/usr/bin/env python3
"""Identify which price-adjustment convention a series uses, and reconcile two sources.

WHY: adjustment defaults disagree across libraries and flip between releases --
`yf.download()` adjusts, `yahooquery.history()` does not, Alpha Vantage's free endpoint is
unadjusted. Nothing warns you. A raw series carries a -86% "return" on Apple's 7:1 split
date, which every momentum, drawdown and stop-loss rule will happily trade. And when two
sources disagree you have to know whether you are looking at an adjustment convention
(harmless for returns) or a genuine bad tick (not harmless at all).

The maths worth knowing before you argue about names: back- and forward-adjusted series
are the SAME series scaled by one global constant -- the total cumulative action factor.
Returns are therefore identical and it does not matter which you use; price LEVELS differ
by that constant, so anything level-based (a $5 penny-stock filter, share-count sizing,
reconciling against a live quote) is silently wrong under the wrong convention.

Definitions used here are behavioural, because vendors use the words inconsistently:
  raw              -- as quoted; jumps by the split ratio on the action date.
  back-adjusted    -- anchored at the PRESENT: history is divided by later factors, so
                      every new corporate action REWRITES the whole history.
  forward-adjusted -- anchored at the START: history never changes; new data is scaled up.
`detect_convention_from_vintages` decides this the only truly definitive way -- by
comparing two pulls of the same series taken either side of a new corporate action.

Usage:
    from adjustment_check import detect_convention, reconcile, reconcile_report

    print(detect_convention(px, actions)["convention"])
    print(reconcile_report(reconcile(yf_close, yq_close, actions)))
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TICK = 0.01           # US equity quote increment; adjusted prices almost never land on it
_CLEAN_HI = 0.90      # a segment this tick-aligned was left untouched by the adjuster
_CLEAN_LO = 0.60


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _actions_frame(actions: pd.DataFrame | Sequence[tuple]) -> pd.DataFrame:
    """Accept a DataFrame(date, ratio[, kind]) or a list of (date, ratio) pairs."""
    if isinstance(actions, pd.DataFrame):
        a = actions.copy()
    else:
        a = pd.DataFrame(list(actions), columns=["date", "ratio"])
    a["date"] = pd.to_datetime(a["date"])
    a["ratio"] = pd.to_numeric(a["ratio"])
    if "kind" not in a.columns:
        a["kind"] = "split"
    return a.sort_values("date").reset_index(drop=True)


def _tick_clean_fraction(s: pd.Series, tick: float = TICK) -> float:
    """Fraction of values sitting exactly on the quote increment.

    Real quotes are struck in cents. Dividing them by a 7:1 factor produces values that
    almost never land back on a cent, so tick alignment marks the segment the adjuster
    left alone -- which is the anchor, which is the convention.
    """
    v = s.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return float("nan")
    frac = np.abs(v / tick - np.round(v / tick))
    return float(np.mean(frac < 1e-6))


def _segment_bounds(index: pd.DatetimeIndex, acts: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The head (before the first action) and tail (after the last action) segments."""
    first, last = acts["date"].iloc[0], acts["date"].iloc[-1]
    return index < first, index >= last


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------
def detect_convention(prices: pd.Series,
                      actions: pd.DataFrame | Sequence[tuple],
                      raw_reference: pd.Series | None = None,
                      jump_tol: float = 0.15) -> dict:
    """Classify a single price series as raw / back-adjusted / forward-adjusted / unknown.

    Two independent tests, strongest first:
      1. JUMP -- does the close-to-close move on each action date match the split ratio?
         Only a raw series does that; both adjusted conventions are continuous there.
      2. ANCHOR -- which end of the series is still tick-aligned? Back-adjustment divides
         history (destroying alignment early, preserving it at the tail); forward
         adjustment leaves history alone.
    `raw_reference` (any unadjusted quote overlapping the index -- a live price, an
    exchange close) upgrades the anchor test from inference to measurement.
    """
    px = prices.dropna().sort_index()
    acts = _actions_frame(actions)
    acts = acts[(acts["date"] >= px.index.min()) & (acts["date"] <= px.index.max())]
    if px.empty or acts.empty:
        return {"convention": "unknown", "confidence": "none",
                "reason": "no corporate action inside the sample - nothing to test against"}

    # --- 1. jump test -------------------------------------------------------------
    logret = np.log(px).diff()
    hits, checked = 0, 0
    for _, a in acts.iterrows():
        pos = px.index.searchsorted(a["date"], side="left")
        if pos == 0 or pos >= len(px):
            continue
        checked += 1
        # a raw 7:1 split shows log(1/7); a day's market move cannot fake that
        if abs(float(logret.iloc[pos]) - np.log(1.0 / a["ratio"])) < jump_tol:
            hits += 1
    if checked and hits == checked:
        return {"convention": "raw", "confidence": "high",
                "reason": f"{hits}/{checked} action dates show a price jump matching the ratio",
                "cum_factor": float(acts["ratio"].prod())}

    head_mask, tail_mask = _segment_bounds(px.index, acts)
    head_clean = _tick_clean_fraction(px[head_mask])
    tail_clean = _tick_clean_fraction(px[tail_mask])
    base = {"jump_hits": f"{hits}/{checked}", "head_tick_clean": head_clean,
            "tail_tick_clean": tail_clean, "cum_factor": float(acts["ratio"].prod())}

    # --- 2. anchor test against a known raw quote ---------------------------------
    if raw_reference is not None:
        ref = raw_reference.dropna()
        common = px.index.intersection(ref.index)
        if len(common):
            rel = (px.loc[common] / ref.loc[common]).astype(float)
            head_ok = bool(np.isclose(rel[common < acts["date"].iloc[0]], 1.0, rtol=1e-6).all()) \
                if (common < acts["date"].iloc[0]).any() else False
            tail_ok = bool(np.isclose(rel[common >= acts["date"].iloc[-1]], 1.0, rtol=1e-6).all()) \
                if (common >= acts["date"].iloc[-1]).any() else False
            if tail_ok and not head_ok:
                return {**base, "convention": "back-adjusted", "confidence": "high",
                        "reason": "matches the raw reference at the tail only - "
                                  "history was rewritten, present is the anchor"}
            if head_ok and not tail_ok:
                return {**base, "convention": "forward-adjusted", "confidence": "high",
                        "reason": "matches the raw reference at the head only - "
                                  "history preserved, later data scaled"}

    # --- 3. anchor test from tick alignment ---------------------------------------
    if head_clean <= _CLEAN_LO and tail_clean >= _CLEAN_HI:
        return {**base, "convention": "back-adjusted", "confidence": "high",
                "reason": f"history is off-tick ({head_clean:.0%} aligned) while the tail is "
                          f"on-tick ({tail_clean:.0%}) - past prices were divided"}
    if head_clean >= _CLEAN_HI and tail_clean <= _CLEAN_LO:
        return {**base, "convention": "forward-adjusted", "confidence": "high",
                "reason": f"tail is off-tick ({tail_clean:.0%}) while history is on-tick "
                          f"({head_clean:.0%}) - later prices were scaled"}
    if head_clean >= _CLEAN_HI and tail_clean >= _CLEAN_HI:
        # continuous across the action but tick-aligned throughout: multiplying cents by an
        # integer ratio preserves alignment, dividing by one does not, so the untouched
        # anchor must be the head
        return {**base, "convention": "forward-adjusted", "confidence": "medium",
                "reason": "continuous at every action yet tick-aligned throughout - "
                          "consistent with multiplying history forward by integer ratios"}
    return {**base, "convention": "unknown", "confidence": "none",
            "reason": f"continuous at the actions but tick alignment is ambiguous "
                      f"(head {head_clean:.0%}, tail {tail_clean:.0%}) - supply raw_reference "
                      f"or two vintages"}


def detect_convention_from_vintages(before: pd.Series, after: pd.Series,
                                    new_action_date: str | pd.Timestamp,
                                    tol: float = 1e-6) -> dict:
    """The definitive test: two pulls of the same series either side of a NEW action.

    Re-pull a series after a split lands. If the values you already stored for dates
    BEFORE the split have changed, the vendor back-adjusts and every cached file you hold
    is now on a different scale from the live one.
    """
    d = pd.Timestamp(new_action_date)
    common = before.dropna().index.intersection(after.dropna().index)
    hist = common[common < d]
    if len(hist) == 0:
        return {"convention": "unknown", "reason": "no shared history before the new action"}
    rel = (after.loc[hist] / before.loc[hist]).astype(float)
    changed = float(np.abs(rel - 1.0).max())
    if changed <= tol:
        return {"convention": "forward-adjusted", "history_rewritten": False,
                "max_history_change": changed,
                "reason": "history identical after a new action - cached files stay valid"}
    return {"convention": "back-adjusted", "history_rewritten": True,
            "max_history_change": changed, "implied_factor": float(np.median(rel)),
            "reason": f"every historical value moved by ~{np.median(rel):.4g}x - "
                      f"any stored copy is now on the old scale"}


# --------------------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------------------
def reconcile(a: pd.Series, b: pd.Series,
              actions: pd.DataFrame | Sequence[tuple] | None = None,
              tol_bps: float = 10.0, window_days: int = 3) -> dict:
    """Compare two sources and say WHY they differ: convention, action, or data error.

    The discriminator is the ratio a/b, not the raw difference. An adjustment disagreement
    makes that ratio piecewise constant with steps only at corporate actions; a bad tick
    makes it step at an unrelated date (and usually step straight back). A globally
    constant ratio is a pure convention difference and leaves returns untouched.
    """
    common = a.dropna().index.intersection(b.dropna().index)
    if len(common) == 0:
        return {"verdict": "NO OVERLAP", "n_common": 0, "steps": pd.DataFrame()}
    av, bv = a.loc[common].astype(float), b.loc[common].astype(float)
    ratio = av / bv
    rel = ratio - 1.0
    tol = tol_bps / 1e4

    diverge = rel.abs() > tol
    log_ratio = np.log(ratio)
    step_size = log_ratio.diff()
    step_dates = common[(step_size.abs() > tol).to_numpy(na_value=False)]

    acts = _actions_frame(actions) if actions is not None else pd.DataFrame(
        columns=["date", "ratio", "kind"])
    rows = []
    for d in step_dates:
        near = acts[(acts["date"] - d).abs() <= pd.Timedelta(days=window_days)] \
            if len(acts) else acts
        rows.append({
            "date": d.date(),
            "ratio_before": float(ratio.shift(1).loc[d]),
            "ratio_after": float(ratio.loc[d]),
            "attribution": (f"{near['kind'].iloc[0]} {near['ratio'].iloc[0]:g}:1 on "
                            f"{near['date'].iloc[0].date()}") if len(near) else
                           "UNEXPLAINED -> data error",
        })
    steps = pd.DataFrame(rows)

    unexplained = int(steps["attribution"].str.startswith("UNEXPLAINED").sum()) if len(steps) else 0
    # identical returns with different levels is the signature of the two adjusted
    # conventions, which differ only by the global cumulative factor
    ret_gap = float((av.pct_change(fill_method=None)
                     - bv.pct_change(fill_method=None)).abs().max(skipna=True))

    if not diverge.any():
        verdict = f"IDENTICAL within {tol_bps:g} bps"
    elif len(steps) == 0:
        verdict = (f"ADJUSTMENT CONVENTION ONLY: constant {float(ratio.iloc[0]):.6g}x level "
                   f"offset, returns match to {ret_gap:.2e} - safe for returns, wrong for "
                   f"any price-level rule")
    elif unexplained == 0:
        verdict = (f"ADJUSTMENT DIFFERENCE: all {len(steps)} step(s) land on corporate "
                   f"actions - not a data error")
    else:
        verdict = (f"DATA ERROR: {unexplained} of {len(steps)} step(s) have no corporate "
                   f"action within {window_days}d")
    return {"verdict": verdict, "n_common": int(len(common)),
            "n_divergent_dates": int(diverge.sum()),
            "first_divergence": str(common[diverge.to_numpy()][0].date()) if diverge.any() else "",
            "max_abs_rel_diff": float(rel.abs().max()),
            "max_abs_return_diff": ret_gap, "n_unexplained_steps": unexplained, "steps": steps}


def reconcile_report(res: dict) -> str:
    lines = [
        f"  overlap        : {res['n_common']} dates, "
        f"{res['n_divergent_dates']} divergent (first {res['first_divergence'] or 'n/a'})",
        f"  max |level diff|: {res['max_abs_rel_diff']:.4%}   "
        f"max |return diff|: {res['max_abs_return_diff']:.2e}",
        f"  VERDICT        : {res['verdict']}",
    ]
    if len(res["steps"]):
        for _, r in res["steps"].iterrows():
            lines.append(f"    step {r['date']}  ratio {r['ratio_before']:.4g} -> "
                         f"{r['ratio_after']:.4g}   [{r['attribution']}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def _synthetic(seed: int = 3) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Raw + both adjusted variants of one instrument with two splits."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=750)
    value = 40.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, len(idx))))  # economic value
    splits = pd.DataFrame({"date": pd.to_datetime(["2022-11-15", "2023-09-05"]),
                           "ratio": [7.0, 3.0], "kind": ["split", "split"]})

    upto = np.ones(len(idx))       # cumulative ratio of actions at or before each date
    after = np.ones(len(idx))      # cumulative ratio of actions strictly after each date
    for d, r in zip(splits["date"], splits["ratio"]):
        upto *= np.where(idx >= d, r, 1.0)
        after *= np.where(idx < d, r, 1.0)

    raw = pd.Series(np.round(value / upto, 2), index=idx)   # quoted in cents, splits visible
    back = pd.Series(raw.to_numpy() / after, index=idx)     # anchored at the present
    fwd = pd.Series(raw.to_numpy() * upto, index=idx)       # anchored at the start
    return raw, back, fwd, splits


if __name__ == "__main__":
    raw, back, fwd, splits = _synthetic()

    print("=== detect_convention on three variants of the SAME instrument ===")
    for name, s in [("raw (as quoted)", raw), ("back-adjusted", back), ("forward-adjusted", fwd)]:
        d = detect_convention(s, splits)
        print(f"  {name:<18} -> {d['convention']:<16} [{d['confidence']}]  {d['reason']}")

    print("\n  with one known unadjusted quote to anchor against, the same call measures it:")
    for name, s in [("back-adjusted", back), ("forward-adjusted", fwd)]:
        d = detect_convention(s, splits, raw_reference=raw)
        print(f"  {name:<18} -> {d['convention']:<16} [{d['confidence']}]  {d['reason']}")

    print("\n=== the definitive two-vintage test (re-pull after a NEW split) ===")
    new_date = raw.index[-1]
    for label, series in [("vendor back-adjusts", back), ("vendor forward-adjusts", fwd)]:
        repull = series / 2.0 if label.startswith("vendor back") else series.copy()
        v = detect_convention_from_vintages(series, repull, new_date)
        print(f"  {label:<24} -> {v['convention']:<16} {v['reason']}")

    print("\n=== reconcile: raw vs back-adjusted (two libraries, different defaults) ===")
    print(reconcile_report(reconcile(raw, back, splits)))

    print("\n=== reconcile: back-adjusted vs forward-adjusted ===")
    print(reconcile_report(reconcile(back, fwd, splits)))

    print("\n=== reconcile: back-adjusted vs the same series with one bad tick ===")
    corrupt = back.copy()
    corrupt.iloc[400] *= 1.35                       # a fat-finger print on a quiet day
    print(reconcile_report(reconcile(back, corrupt, splits)))

    print("\nSame tool, three answers: a split explains the first two mismatches and nothing")
    print("explains the third. Without the attribution step all three look like 'the sources")
    print("disagree', and the usual response - trusting whichever one looks smoother - keeps")
    print("the bad tick and throws away the correctly adjusted series.")
