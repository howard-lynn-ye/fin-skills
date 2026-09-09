#!/usr/bin/env python3
"""Triple-barrier labels against fixed-horizon labels - what the path says that the horizon hides.

Advances in Financial Machine Learning (Lopez de Prado 2018), chapter 3. A fixed-horizon label
asks "was price higher 20 bars later"; a triple-barrier label asks "which of profit-taking, stop
loss and the holding-period limit came FIRST". They disagree whenever the path went through a
barrier and came back, and the disagreement is not noise: a model trained on fixed-horizon labels
is being told that a trade which drew down through its stop and recovered was a winner.

Measured here on one seeded series with clustered volatility:

  1. the barrier geometry: `getEvents`-style volatility-scaled barriers vs fixed-width ones, and
     how the label distribution moves across volatility terciles;
  2. the label distribution, triple barrier against fixed horizon;
  3. the fraction of fixed-horizon labels that are wrong about the path - the trade was stopped
     out before the horizon - and the profit-and-loss that costs;
  4. the maximum adverse excursion the fixed-horizon backtest assumes you sat through;
  5. `pt_sl` asymmetry: the label distribution is a design choice, not a property of the market;
  6. the arithmetic-vs-log return mismatch between mlfinpy's Snippet 3.2 touch test and its
     Snippet 3.9 relabelling, which silently converts horizontal touches into vertical ones.

Bars are integer indices rather than timestamps, so the whole script is numpy - the book's
version keys everything by DatetimeIndex, which changes nothing about the arithmetic.

Run:  python triple_barrier.py     (numpy; no optional libraries; seed 0; ~4 s)
"""
from __future__ import annotations

import math
import time

import numpy as np

SEED = 0
N_BARS = 6000
EVENT_STEP = 5           # sample an event every 5 bars (the book uses a CUSUM filter; see the
                         # structural-breaks skill - the labelling arithmetic is the same)
VOL_SPAN = 100           # mlfinpy get_daily_vol's default lookback (AFML Snippet 3.1, page 44)
VERT_BARS = 20           # the vertical barrier: maximum holding period
PT_SL = (4.0, 4.0)       # profit-taking and stop-loss multiples of the target
WARMUP = 300             # bars before the first event, so the EWMA vol is not warming up
# stochastic volatility: log-vol AR(1)
PHI_VOL = 0.99
SIGMA_LOGVOL = 0.08
VOL_MEAN = 0.01          # 1 % per bar on average


# ------------------------------------------------------------------------ data --------------
def stochastic_vol_prices(n_bars: int = N_BARS, seed: int = SEED, phi: float = PHI_VOL,
                          sig_lv: float = SIGMA_LOGVOL, vol_mean: float = VOL_MEAN):
    """Log price with clustered volatility: sigma_t from a log-AR(1), returns sigma_t * z_t.

    Returns (close, sigma_true). No drift, so a symmetric barrier pair is a fair coin on the
    horizontal barriers and every asymmetry below comes from the geometry, not from alpha.
    """
    rng = np.random.default_rng(seed)
    lv = np.empty(n_bars)
    lv[0] = math.log(vol_mean)
    e = rng.normal(0.0, sig_lv, n_bars)
    for t in range(1, n_bars):
        lv[t] = math.log(vol_mean) + phi * (lv[t - 1] - math.log(vol_mean)) + e[t]
    sigma = np.exp(lv)
    ret = sigma * rng.standard_normal(n_bars)
    return 100.0 * np.exp(np.cumsum(ret)), sigma


def ewma_vol(close: np.ndarray, span: int = VOL_SPAN) -> np.ndarray:
    """Causal EWMA standard deviation of one-bar arithmetic returns, span `span`.

    mlfinpy 0.1.2 `get_daily_vol` (AFML Snippet 3.1, page 44) is
    `(close / close.shift(1) - 1).ewm(span=lookback).std()` with lookback=100 - arithmetic
    returns, pandas' adjusted EWM, unbiased standard deviation. This reproduces that on an
    integer bar index. vol[t] uses returns through t only.
    """
    close = np.asarray(close, dtype=float)
    r = np.full(close.shape[0], np.nan)
    r[1:] = close[1:] / close[:-1] - 1.0
    alpha = 2.0 / (span + 1.0)
    out = np.full(close.shape[0], np.nan)
    sw = sw2 = swx = swx2 = 0.0          # adjust=True: weights (1-alpha)^i, never renormalised
    n = 0
    for t in range(1, close.shape[0]):
        decay = 1.0 - alpha
        sw, sw2 = sw * decay + 1.0, sw2 * decay * decay + 1.0
        swx = swx * decay + r[t]
        swx2 = swx2 * decay + r[t] * r[t]
        n += 1
        if n < 2:
            continue
        mean = swx / sw
        var_biased = swx2 / sw - mean * mean
        denom = 1.0 - sw2 / (sw * sw)    # pandas' bias correction for adjusted EWM variance
        if denom > 0:
            out[t] = math.sqrt(max(var_biased / denom, 0.0))
    return out


# ------------------------------------------------------------------------ barriers ----------
def first_touch(close: np.ndarray, t0: int, vert: int, pt_level: float, sl_level: float,
                side: float = 1.0) -> tuple[int, str]:
    """First bar in (t0, t0+vert] whose signed return crosses a horizontal barrier.

    `pt_level` and `sl_level` are signed cumulative-return levels (pt > 0, sl < 0), compared
    against `(close[t] / close[t0] - 1) * side` - arithmetic returns, strict inequality, exactly
    as mlfinpy 0.1.2 `triple_barriers` (AFML Snippet 3.2, page 45) does. Returns
    (touch_bar, which) with which in {"pt", "sl", "vert"}.
    """
    end = min(t0 + vert, close.shape[0] - 1)
    for t in range(t0 + 1, end + 1):
        r = (close[t] / close[t0] - 1.0) * side
        if pt_level is not None and r > pt_level:
            return t, "pt"
        if sl_level is not None and r < sl_level:
            return t, "sl"
    return end, "vert"


def triple_barrier_events(close: np.ndarray, t_events: np.ndarray, trgt: np.ndarray,
                          pt_sl: tuple[float, float] = PT_SL, vert: int = VERT_BARS,
                          side: np.ndarray | None = None, min_ret: float = 0.0) -> dict:
    """Label each event by which barrier is touched first.

    Returns arrays keyed by the surviving events: t0, t1 (touch bar), trgt, touched, label, ret.
    Without `side` the label is the sign of the barrier: +1 pt, -1 sl, 0 vertical (AFML Snippet
    3.9, page 55). With `side` it is a META-label in {0, 1}: 1 when the side was right.

    `min_ret` reproduces `get_events`' filter, which DROPS every event whose target is below it.
    """
    close = np.asarray(close, dtype=float)
    t_events = np.asarray(t_events, dtype=int)
    keep = np.isfinite(trgt[t_events]) & (trgt[t_events] > min_ret)
    t0s = t_events[keep]
    tg = trgt[t0s]
    sides = np.ones(t0s.shape[0]) if side is None else np.asarray(side, dtype=float)[keep]
    pt = pt_sl[0] * tg if pt_sl[0] > 0 else np.full(tg.shape, np.nan)
    sl = -pt_sl[1] * tg if pt_sl[1] > 0 else np.full(tg.shape, np.nan)

    t1 = np.empty(t0s.shape[0], dtype=int)
    touched = np.empty(t0s.shape[0], dtype="<U4")
    for i, t0 in enumerate(t0s):
        t1[i], touched[i] = first_touch(close, int(t0), vert,
                                        None if np.isnan(pt[i]) else float(pt[i]),
                                        None if np.isnan(sl[i]) else float(sl[i]),
                                        float(sides[i]))
    ret = (close[t1] / close[t0s] - 1.0) * sides
    if side is None:
        label = np.where(touched == "pt", 1.0, np.where(touched == "sl", -1.0, 0.0))
    else:
        label = np.where(touched == "pt", 1.0, 0.0)
        label[ret <= 0] = 0.0            # get_bins: a losing bet is a 0 whatever it touched
    return {"t0": t0s, "t1": t1, "trgt": tg, "touched": touched, "label": label, "ret": ret,
            "side": sides, "pt": pt, "sl": sl}


def fixed_horizon_labels(close: np.ndarray, t0s: np.ndarray, horizon: int = VERT_BARS,
                         threshold: np.ndarray | float = 0.0) -> dict:
    """Label by the sign of the return `horizon` bars later; 0 inside +/- threshold.

    AFML section 3.2 (page 43) calls this "the fixed-time horizon method" and criticises it for
    exactly the reason section 3 of this script measures.
    """
    close = np.asarray(close, dtype=float)
    t0s = np.asarray(t0s, dtype=int)
    t1 = np.minimum(t0s + horizon, close.shape[0] - 1)
    ret = close[t1] / close[t0s] - 1.0
    tau = np.broadcast_to(np.asarray(threshold, dtype=float), ret.shape)
    label = np.where(ret > tau, 1.0, np.where(ret < -tau, -1.0, 0.0))
    return {"t0": t0s, "t1": t1, "ret": ret, "label": label}


def excursions(close: np.ndarray, t0s: np.ndarray, horizon: int = VERT_BARS,
               side: np.ndarray | float = 1.0) -> dict:
    """Maximum adverse and favourable excursion of each event over the fixed horizon."""
    close = np.asarray(close, dtype=float)
    t0s = np.asarray(t0s, dtype=int)
    sides = np.broadcast_to(np.asarray(side, dtype=float), t0s.shape)
    mae = np.empty(t0s.shape[0])
    mfe = np.empty(t0s.shape[0])
    for i, t0 in enumerate(t0s):
        end = min(int(t0) + horizon, close.shape[0] - 1)
        path = (close[int(t0) + 1:end + 1] / close[int(t0)] - 1.0) * sides[i]
        mae[i] = float(path.min()) if path.size else 0.0
        mfe[i] = float(path.max()) if path.size else 0.0
    return {"mae": mae, "mfe": mfe}


def path_conflicts(close: np.ndarray, ev: dict, horizon: int = VERT_BARS,
                   threshold: np.ndarray | float = 0.0) -> dict:
    """How often the fixed-horizon label disagrees with what the path actually did.

    The default `threshold=0` is the usual fixed-horizon label: the SIGN of the return `horizon`
    bars later. For each event it is compared against the triple-barrier outcome over the same
    horizon and the same barriers, counting the horizon "winners" that were stopped out on the
    way. Also returns the profit-and-loss of following each label set.
    """
    t0s = ev["t0"]
    fh = fixed_horizon_labels(close, t0s, horizon, threshold)
    exc = excursions(close, t0s, horizon)
    stopped = exc["mae"] < ev["sl"]                   # the long stop was crossed at some point
    ran_up = exc["mfe"] > ev["pt"]
    fh_up = fh["label"] > 0
    fh_dn = fh["label"] < 0
    # what you actually realise if you honour the stop: the barrier return, not the horizon one
    realised = ev["ret"].copy()
    return {
        "fh_label": fh["label"], "fh_ret": fh["ret"], "tb_label": ev["label"],
        "stopped": stopped, "ran_up": ran_up,
        "n": int(t0s.shape[0]),
        "fh_up_but_stopped": int(np.sum(fh_up & stopped)),
        "fh_dn_but_ran_up": int(np.sum(fh_dn & ran_up)),
        "wrong_about_path": int(np.sum((fh_up & stopped) | (fh_dn & ran_up))),
        "label_disagree": int(np.sum(fh["label"] != ev["label"])),
        "mae": exc["mae"], "mfe": exc["mfe"],
        "pnl_fixed_horizon": float(np.mean(fh["ret"] * np.sign(fh["label"]))),
        "pnl_with_stop": float(np.mean(realised * np.sign(fh["label"]))),
    }


def pandas_ewma_vol(close: np.ndarray, span: int = VOL_SPAN) -> np.ndarray:
    """The same thing through pandas, so `ewma_vol` is checked rather than assumed to match."""
    import pandas as pd

    s = pd.Series(np.asarray(close, dtype=float))
    return (s / s.shift(1) - 1.0).ewm(span=span).std().to_numpy()


def barrier_touched_relabel(ret_arith: np.ndarray, trgt: np.ndarray,
                            pt: float, sl: float) -> np.ndarray:
    """mlfinpy 0.1.2 `barrier_touched` (AFML Snippet 3.9, page 55), on the LOG return.

    It re-derives the label from the return at the touch time with
    `ret > np.log(1 + target) * pt` / `ret < -np.log(1 + target) * sl`, where `ret` is the LOG
    return - while `triple_barriers` decided the touch with the ARITHMETIC return against
    `pt * target`. The two agree at pt = sl = 1 and diverge as the multiples grow.
    """
    r_log = np.log1p(ret_arith)
    up = np.log1p(trgt) * pt
    dn = -np.log1p(trgt) * sl
    return np.where((ret_arith > 0) & (r_log > up), 1.0,
                    np.where((ret_arith < 0) & (r_log < dn), -1.0, 0.0))


def distribution(label: np.ndarray) -> dict:
    n = label.shape[0]
    return {v: float(np.mean(label == v)) for v in (-1.0, 0.0, 1.0)} | {"n": n}


# ------------------------------------------------------------------------ demo --------------
def _fmt_dist(d: dict) -> str:
    return (f"-1 {d[-1.0]:6.1%}   0 {d[0.0]:6.1%}   +1 {d[1.0]:6.1%}   (n={d['n']})")


def _main() -> None:
    t_all = time.time()
    close, sigma = stochastic_vol_prices()
    vol = ewma_vol(close)
    t_events = np.arange(WARMUP, N_BARS - VERT_BARS - 1, EVENT_STEP)

    print(f"=== 1. Volatility-scaled barriers (AFML Snippet 3.1 p.44 + 3.6 p.50) ===")
    print(f"  T={N_BARS} bars, log-vol AR(1) phi={PHI_VOL}: realised per-bar vol ranges "
          f"{sigma.min():.4f} to {sigma.max():.4f} ({sigma.max() / sigma.min():.1f}x)")
    print(f"  own EWMA vol vs pandas (close/close.shift(1)-1).ewm(span={VOL_SPAN}).std(): "
          f"max abs diff {np.nanmax(np.abs(vol - pandas_ewma_vol(close))):.2e}")
    print(f"  EWMA(span={VOL_SPAN}) target vol at the events: "
          f"{np.nanmin(vol[t_events]):.4f} to {np.nanmax(vol[t_events]):.4f}; "
          f"{len(t_events)} events every {EVENT_STEP} bars, vertical barrier {VERT_BARS} bars")
    ev = triple_barrier_events(close, t_events, vol, PT_SL, VERT_BARS)
    fixed_trgt = np.full_like(vol, float(np.nanmedian(vol[t_events])))
    ev_fixed = triple_barrier_events(close, t_events, fixed_trgt, PT_SL, VERT_BARS)
    terc = np.quantile(vol[ev["t0"]], [1 / 3, 2 / 3])
    grp = np.digitize(vol[ev["t0"]], terc)
    print("  fraction of events resolving on the VERTICAL barrier, by volatility tercile:")
    print("    tercile   vol-scaled barriers   fixed-width barriers")
    for g, name in enumerate(("low ", "mid ", "high")):
        m = grp == g
        print(f"      {name}          {np.mean(ev['touched'][m] == 'vert'):8.1%}"
              f"              {np.mean(ev_fixed['touched'][m] == 'vert'):8.1%}")
    lo, hi = grp == 0, grp == 2
    sp_scaled = abs(np.mean(ev["touched"][lo] == "vert") - np.mean(ev["touched"][hi] == "vert"))
    sp_fixed = abs(np.mean(ev_fixed["touched"][lo] == "vert")
                   - np.mean(ev_fixed["touched"][hi] == "vert"))
    print(f"    low-to-high spread: vol-scaled {sp_scaled:.1%}, fixed-width {sp_fixed:.1%} "
          f"({sp_fixed / max(sp_scaled, 1e-12):.1f}x)")

    print(f"\n=== 2. Label distributions, same events, same {VERT_BARS}-bar horizon ===")
    fh_tau = fixed_horizon_labels(close, ev["t0"], VERT_BARS, PT_SL[0] * ev["trgt"])
    fh_zero = fixed_horizon_labels(close, ev["t0"], VERT_BARS, 0.0)
    print(f"  triple barrier pt_sl={PT_SL}:                    {_fmt_dist(distribution(ev['label']))}")
    print(f"  fixed horizon, threshold = the same pt barrier:  {_fmt_dist(distribution(fh_tau['label']))}")
    print(f"  fixed horizon, threshold = 0 (sign only):        {_fmt_dist(distribution(fh_zero['label']))}")

    print(f"\n=== 3. How often the fixed-horizon label is wrong about the path ===")
    pc = path_conflicts(close, ev, VERT_BARS)
    n = pc["n"]
    print(f"  events: {n}")
    print(f"  fixed-horizon label disagrees with the triple-barrier label: {pc['label_disagree']}"
          f" ({pc['label_disagree'] / n:.1%})")
    print(f"  labelled +1 by the horizon but the STOP was crossed first or on the way: "
          f"{pc['fh_up_but_stopped']} ({pc['fh_up_but_stopped'] / n:.1%})")
    print(f"  labelled -1 by the horizon but the PROFIT target was crossed on the way: "
          f"{pc['fh_dn_but_ran_up']} ({pc['fh_dn_but_ran_up'] / n:.1%})")
    print(f"  either way, the horizon label describes a path you could not have held: "
          f"{pc['wrong_about_path']} ({pc['wrong_about_path'] / n:.1%})")
    print("  and that fraction is set by how tight the stop is:")
    print("    stop (sigma)   1.0     2.0     3.0     4.0")
    frac = []
    for m in (1.0, 2.0, 3.0, 4.0):
        e_m = triple_barrier_events(close, t_events, vol, (m, m), VERT_BARS)
        frac.append(path_conflicts(close, e_m, VERT_BARS)["wrong_about_path"] / n)
    print("    wrong path   " + "".join(f"{f:7.1%} " for f in frac))
    gap = pc["pnl_fixed_horizon"] - pc["pnl_with_stop"]
    print("  a PERFECT model of the fixed-horizon labels - the ceiling any such model can reach -")
    print(f"    scored the way it was trained (hold to the horizon, no stop): "
          f"{pc['pnl_fixed_horizon']:+.5f} per event")
    print(f"    scored on the path, with the same barriers honoured:          "
          f"{pc['pnl_with_stop']:+.5f} per event")
    print(f"    the gap is {gap:+.5f} per event, {gap / pc['pnl_fixed_horizon']:.0%} of the "
          f"apparent profit, and it is drawdown you were not allowed to hold")
    print("    (symmetric pt_sl, so a short's barrier return is the long's negated)")

    print(f"\n=== 4. The drawdown the fixed-horizon backtest assumes you sat through ===")
    up = pc["fh_label"] > 0
    mae = pc["mae"][up]
    sigma_units = mae / ev["trgt"][up]
    through_stop = float(np.mean(mae < ev["sl"][up]))
    print(f"  among the {int(up.sum())} events the sign-only horizon label calls +1, the worst "
          f"drawdown before the horizon:")
    print(f"    median {np.median(mae):.4f}, 10th percentile {np.quantile(mae, 0.10):.4f}, "
          f"worst {mae.min():.4f}")
    print(f"    in target-volatility units: median {np.median(sigma_units):.2f} sigma, "
          f"worst {np.min(sigma_units):.2f} sigma")
    print(f"    through the {PT_SL[1]:.0f}-sigma stop before the horizon: {through_stop:.1%}; "
          f"through a 1-sigma stop: {np.mean(sigma_units < -1.0):.1%}")

    print(f"\n=== 5. pt_sl is a design choice, and it sets the class balance ===")
    print("    pt_sl        label distribution                          mean barrier return")
    for pt_sl in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0), (1.0, 0.0)):
        e = triple_barrier_events(close, t_events, vol, pt_sl, VERT_BARS)
        print(f"  {str(pt_sl):<12} {_fmt_dist(distribution(e['label']))}   "
              f"{np.mean(e['ret']):+.5f}")
    print("  the series has zero drift by construction, so none of this is alpha - it is the")
    print("  geometry. A label distribution is not evidence about the market.")

    print(f"\n=== 6. Arithmetic vs log: Snippet 3.2 touches, Snippet 3.9 relabels ===")
    for pt_sl in ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)):
        e = triple_barrier_events(close, t_events, vol, pt_sl, VERT_BARS)
        re_lab = barrier_touched_relabel(e["ret"], e["trgt"], pt_sl[0], pt_sl[1])
        horiz = e["touched"] != "vert"
        flipped = int(np.sum(horiz & (re_lab == 0.0)))
        print(f"  pt_sl={pt_sl}: {int(horiz.sum())} horizontal touches, of which "
              f"{flipped} ({flipped / max(int(horiz.sum()), 1):.1%}) are relabelled 0 "
              f"(vertical) by the Snippet 3.9 log test")
    tg = 0.02
    print(f"  why: at target {tg:.0%} and pt=2, Snippet 3.2 fires at r > {2 * tg:.4f} while "
          f"Snippet 3.9 needs r > {math.expm1(2 * math.log1p(tg)):.4f} - a touch inside that "
          f"band becomes a vertical-barrier label")

    print("\nRule: label by which barrier is hit first, scale the barriers by the volatility at"
          " the event, and never train on a horizon label whose path went through your stop.")
    print(f"total runtime {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    _main()
