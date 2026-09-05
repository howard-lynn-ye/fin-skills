#!/usr/bin/env python3
"""The same fills score differently against every benchmark. Pick one in advance.

Transaction-cost analysis has no single number. An execution that beats VWAP by
4 bps can lose 30 bps against the price when the decision was made, and both
statements are arithmetically correct. Which one you quote decides whether the
desk looks good, which is exactly why the benchmark has to be chosen before the
trade rather than after it.

This script prices that gap on one synthetic order, and shows the two ways VWAP
is gameable:

    1. trade only when the market comes to you (participate in the cheap prints)
    2. trade slowly enough that your own volume dominates the benchmark

Neither requires lying. Both produce a genuinely better VWAP number and a worse
outcome for the fund.

Run:  python benchmark_choice.py

numpy only.
"""
from __future__ import annotations

import numpy as np

BPS = 1e4


def make_day(seed: int = 7, n: int = 390, drift_bps: float = 120.0,
             vol_bps: float = 4.0, s0: float = 100.0):
    """One session of minute bars that drifts AGAINST a buyer, by construction.

    The drift is imposed deterministically and the noise is added around it, so
    the sign of the demonstration does not depend on the draw. An earlier version
    sampled the drift inside the noise and the realised path went the other way,
    which silently inverted every conclusion below it.
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, drift_bps / BPS, n)          # imposed, not sampled
    noise = np.cumsum(rng.normal(0.0, vol_bps / BPS, n))
    noise -= np.linspace(0.0, noise[-1], n)               # pin the endpoints
    px = s0 * np.exp(trend + noise)
    # U-shaped volume: heavy at the open and close
    t = np.linspace(0, 1, n)
    vol = 1.0 + 3.0 * (np.exp(-8 * t) + np.exp(-8 * (1 - t)))
    vol = vol * rng.lognormal(0, 0.25, n)
    return px, vol


def benchmarks(px, vol, decision_px):
    return {
        "decision (arrival)": decision_px,
        "interval VWAP": float(np.sum(px * vol) / np.sum(vol)),
        "interval TWAP": float(px.mean()),
        "close": float(px[-1]),
        "open": float(px[0]),
    }


def cost_bps(avg_fill: float, bench: float, side: int = 1) -> float:
    """Positive = the execution cost money against that benchmark. side +1 buy."""
    return side * (avg_fill - bench) / bench * BPS


def schedule_twap(px, vol, shares):
    w = np.ones(len(px))
    return w / w.sum() * shares


def schedule_vwap(px, vol, shares):
    return vol / vol.sum() * shares


def schedule_opportunistic(px, vol, shares, quantile=0.4):
    """Trade only in the cheapest 40% of minutes. Great VWAP, terrible timing."""
    thresh = np.quantile(px, quantile)
    mask = (px <= thresh).astype(float)
    if mask.sum() == 0:
        mask = np.ones_like(px)
    w = mask * vol
    return w / w.sum() * shares


def schedule_front(px, vol, shares, frac=0.15):
    """Finish early. Bad VWAP by construction, best decision-price capture."""
    k = max(1, int(len(px) * frac))
    w = np.zeros(len(px))
    w[:k] = vol[:k]
    return w / w.sum() * shares


def run(name, sched, px, vol, decision_px, shares):
    fills = sched(px, vol, shares)
    avg = float(np.sum(fills * px) / np.sum(fills))
    b = benchmarks(px, vol, decision_px)
    done_by = int(np.argmax(np.cumsum(fills) >= shares * 0.999)) + 1
    return {
        "name": name, "avg_fill": avg, "done_by_min": done_by,
        **{k: cost_bps(avg, v) for k, v in b.items()},
    }


if __name__ == "__main__":
    px, vol = make_day()
    decision_px = float(px[0])          # the price when the PM said "buy"
    shares = 100_000

    print(f"One buy order, {shares:,} shares, over a 390-minute session.")
    print(f"Decision price {decision_px:.4f}, close {px[-1]:.4f} "
          f"({(px[-1] / decision_px - 1) * BPS:+.1f} bps drift against the buyer)\n")

    rows = [
        run("TWAP schedule", schedule_twap, px, vol, decision_px, shares),
        run("VWAP schedule", schedule_vwap, px, vol, decision_px, shares),
        run("Opportunistic (cheap prints only)", schedule_opportunistic, px, vol,
            decision_px, shares),
        run("Front-loaded (15% of session)", schedule_front, px, vol, decision_px, shares),
    ]

    cols = ["decision (arrival)", "interval VWAP", "interval TWAP", "close"]
    print(f"  {'schedule':<34}" + "".join(f"{c:>20}" for c in cols) + f"{'done by':>9}")
    print("  " + "-" * (34 + 20 * len(cols) + 9))
    for r in rows:
        print(f"  {r['name']:<34}"
              + "".join(f"{r[c]:>+19.1f}" for c in cols)
              + f"{r['done_by_min']:>8}m")
    print("\n  positive = cost, negative = the execution beat that benchmark")

    vw = next(r for r in rows if r["name"].startswith("VWAP"))
    fro = next(r for r in rows if r["name"].startswith("Front"))
    drift = (px[-1] / decision_px - 1) * BPS

    print(f"\n  THE POINT. The VWAP schedule scores {vw['interval VWAP']:+.1f} bps against VWAP -")
    print(f"  a textbook-perfect execution by that metric - while costing the fund")
    print(f"  {vw['decision (arrival)']:+.1f} bps against the price when the decision was made.")
    print(f"\n  It matched the market's own schedule through a {drift:+.0f} bps adverse move,")
    print(f"  so it ate the move in full and the metric reported zero. **VWAP cannot")
    print(f"  tell you whether trading at that pace was a good idea.** It only tells")
    print(f"  you whether you looked like everyone else while you did it.")
    print(f"\n  Finishing in {fro['done_by_min']}m instead scores {fro['decision (arrival)']:+.1f} bps "
          f"on the decision price,")
    print(f"  {vw['decision (arrival)'] - fro['decision (arrival)']:.1f} bps better for the fund.")

    best_vwap = min(rows, key=lambda r: r["interval VWAP"])["name"]
    best_arr = min(rows, key=lambda r: r["decision (arrival)"])["name"]
    print(f"\n  ! Honest caveat: on THIS path the two benchmarks agree on the winner")
    print(f"    (VWAP: {best_vwap}; arrival: {best_arr}),")
    print(f"    because the drift is monotone, so trading early wins on every measure.")
    print(f"    The benchmarks diverge when the path is not monotone. The claim here")
    print(f"    is NOT that they always rank differently - it is that a zero VWAP")
    print(f"    score is compatible with any amount of shortfall, which the VWAP row")
    print(f"    above demonstrates on its own.")

    # ---- gaming demonstration 2: your own volume dilutes the benchmark -------
    print(f"\n\n  VWAP is not exogenous - your own prints are IN the benchmark.")
    print(f"  Hold the execution fixed (the front-loaded schedule) and vary only")
    print(f"  how large it is relative to the session:\n")
    print(f"    {'participation':>14}{'your avg fill':>16}{'VWAP incl. own':>17}"
          f"{'measured cost':>16}")
    base_vol = vol / vol.sum() * 1_000_000
    for part in (0.01, 0.05, 0.20, 0.50):
        qty = base_vol.sum() * part / (1.0 - part)
        own = schedule_front(px, vol, qty)                 # NOT proportional to vol
        avg = float(np.sum(own * px) / np.sum(own))
        blended = float(np.sum(px * (base_vol + own)) / np.sum(base_vol + own))
        print(f"    {part:>13.0%}{avg:>16.4f}{blended:>17.4f}"
              f"{cost_bps(avg, blended):>+15.1f}")
    print(f"\n    The fills never change - only the benchmark moves toward them.")
    print(f"    ! If the schedule is PROPORTIONAL to volume the measured cost is")
    print(f"    identically zero at every participation, because VWAP(base+own)")
    print(f"    equals VWAP(base) exactly. That is an algebraic identity, not a")
    print(f"    result, and it is the trivial way to score zero on this metric.")

    print(f"\n\n  Rule: quote implementation shortfall against the DECISION price."
          f"\n        VWAP measures whether you matched the market's own schedule;"
          f"\n        it cannot tell you whether trading at all was a good idea, and"
          f"\n        it is gameable in two directions by the desk being measured.")
