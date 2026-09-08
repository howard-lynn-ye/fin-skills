#!/usr/bin/env python3
"""Rule-based regime detectors: what each one lags, leaks, and costs, measured on known regimes.

Every regime method is a trade-off between detection delay and false alarms, and every one has
a place where the future can creep in. This script runs the common non-model detectors on the
same synthetic two-regime data as regime_lookahead.py (so the true regime is known) and prints,
for each: label accuracy, median delay into and out of turbulence, false alarms, and the Sharpe
of "long unless flagged". Each detector is run in its honest form (everything computed from data
through t-1) and, where people usually leak, in the leaky form too, so the size of the leak is a
number rather than a warning.

Detectors:
  realized-vol threshold      21-day trailing vol vs a fixed level, an expanding-window
                              percentile (honest), or a full-sample percentile (leaks)
  turbulence index            Kritzman-Li Mahalanobis distance of a 3-asset return vector,
                              mean/covariance from an expanding window (honest) or the full
                              sample (leaks); flagged above an expanding 90th percentile
  trend/vol quadrant          price above/below its 200-day average x vol above/below its
                              expanding median; "turbulent" = downtrend AND high vol
  drawdown state              more than 10% below the running peak
  Markov switching            predicted probability, full-sample parameters (from
                              regime_lookahead.py), for comparison on the same data

Run:  python regime_methods.py      (numpy / pandas / statsmodels; fixed seed; ~10 s)
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

try:
    from .regime_lookahead import (MU, N, P_STAY, PERIODS, SIG_CALM, TEST_START, calm_index,
                                   detection, fit_ms, label_accuracy, strategy_stats)
except ImportError:
    from regime_lookahead import (MU, N, P_STAY, PERIODS, SIG_CALM, TEST_START, calm_index,
                                  detection, fit_ms, label_accuracy, strategy_stats)

SEED = 0
RATIO = 2.0
K_ASSETS = 3
CORR = (0.3, 0.8)          # pairwise correlation in calm / turbulent regime
VOL_WINDOW = 21
MIN_HISTORY = 252          # expanding-window statistics start after one year
TURB_SMOOTH = 10
SMA = 200
DD_LIMIT = 0.10
PCTL_VOL = 0.75
PCTL_TURB = 0.90


def simulate_multi(n: int, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """K correlated assets sharing one regime path; correlation and vol both switch."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < P_STAY[s[t - 1]] else 1 - s[t - 1]
    chol = []
    for k in range(2):
        sig = SIG_CALM * (ratio if k == 1 else 1.0)
        c = np.full((K_ASSETS, K_ASSETS), CORR[k]) + (1 - CORR[k]) * np.eye(K_ASSETS)
        chol.append(np.linalg.cholesky(c) * sig)
    z = rng.standard_normal((n, K_ASSETS))
    r = np.empty((n, K_ASSETS))
    for t in range(n):
        r[t] = MU[s[t]] + chol[s[t]] @ z[t]
    return r, s


def trailing_vol(r: np.ndarray, w: int) -> np.ndarray:
    """std of r[t-w..t-1]: known at the close of t-1, usable for day t."""
    return pd.Series(r).rolling(w).std(ddof=1).shift(1).to_numpy()


def expanding_quantile(x: np.ndarray, q: float, min_n: int) -> np.ndarray:
    """q-quantile of x[..t-1]: past only."""
    return pd.Series(x).expanding(min_n).quantile(q).shift(1).to_numpy()


def turbulence(r: np.ndarray, min_n: int, full_sample: bool) -> np.ndarray:
    """Mahalanobis distance of r[t] from the mean/covariance of r[..t-1] (or the full sample)."""
    n = len(r)
    d = np.full(n, np.nan)
    if full_sample:
        mu, cov = r.mean(axis=0), np.cov(r, rowvar=False)
        inv = np.linalg.inv(cov)
        diff = r - mu
        return np.einsum("ij,jk,ik->i", diff, inv, diff)
    for t in range(min_n, n):
        past = r[:t]
        mu, cov = past.mean(axis=0), np.cov(past, rowvar=False)
        diff = r[t] - mu
        d[t] = diff @ np.linalg.solve(cov, diff)
    return d


def delays_into(s: np.ndarray, turb_flag: np.ndarray, lo: int, hi: int, into: int) -> float:
    """Median days from a true switch INTO regime `into` until the flag agrees."""
    out = []
    sw = [t for t in range(max(lo, 1), hi) if s[t] != s[t - 1]]
    for i, t0 in enumerate(sw):
        if s[t0] != into:
            continue
        t_end = sw[i + 1] if i + 1 < len(sw) else hi
        hit = next((t for t in range(t0, t_end) if turb_flag[t] == into), None)
        out.append(hit - t0 if hit is not None else t_end - t0)   # missed: count full episode
    return float(np.median(out)) if out else float("nan")


def evaluate(name: str, uses: str, turb_flag: np.ndarray, r: np.ndarray, s: np.ndarray,
             lo: int, hi: int) -> dict:
    flag = np.nan_to_num(turb_flag).astype(int)
    p_calm = 1.0 - flag
    det = detection(s, p_calm, lo, hi)
    st = strategy_stats(r[lo:hi], (1 - flag[lo:hi]).astype(float))
    return {"detector": name, "uses": uses,
            "acc": label_accuracy(s, p_calm, lo, hi),
            "flag_share": float(flag[lo:hi].mean()),
            "into_turb": delays_into(s, flag, lo, hi, 1),
            "into_calm": delays_into(s, flag, lo, hi, 0),
            "false_alarms": det["false_alarms"], "missed": det["missed"],
            "sharpe": st["sharpe"], "maxdd": st["maxdd"], "switches": st["switches"]}


if __name__ == "__main__":
    t_all = time.time()
    lo, hi = TEST_START, N
    R, s = simulate_multi(N, SEED, RATIO)
    r = R[:, 0]                                    # the asset we trade
    logp = np.cumsum(r)
    price = np.exp(logp)
    print(f"DGP: {N} obs x {K_ASSETS} assets, calm mu={MU[0]} sig={SIG_CALM} corr={CORR[0]},"
          f" turbulent mu={MU[1]} sig={SIG_CALM * RATIO} corr={CORR[1]}, p_stay={P_STAY}, seed={SEED}")
    print(f"Evaluation window obs {lo}..{hi - 1}; true turbulent share there ="
          f" {(s[lo:hi] == 1).mean():.3f}; every flag is computed from data through t-1 unless"
          f" marked LEAKS")

    rows = []
    rows.append(evaluate("buy & hold", "nothing", np.zeros(N), r, s, lo, hi))
    rows.append(evaluate("oracle lagged 1 day", "true regime at t-1", np.r_[0, s[:-1]], r, s, lo, hi))

    rv = trailing_vol(r, VOL_WINDOW)
    theta_fixed = float(np.sqrt(SIG_CALM * SIG_CALM * RATIO))      # geometric midpoint of the
    rows.append(evaluate(f"vol > fixed {theta_fixed:.4f}", "r[..t-1]; threshold set in advance",
                         rv > theta_fixed, r, s, lo, hi))           # two TRUE vols: a cheat
    theta_round = 0.20 / np.sqrt(PERIODS)
    rows.append(evaluate("vol > 20% annualised", "r[..t-1]; round-number threshold",
                         rv > theta_round, r, s, lo, hi))
    rows.append(evaluate(f"vol > expanding p{int(PCTL_VOL * 100)}", "r[..t-1]; threshold from past",
                         rv > expanding_quantile(rv, PCTL_VOL, MIN_HISTORY), r, s, lo, hi))
    rows.append(evaluate(f"vol > full-sample p{int(PCTL_VOL * 100)}", "LEAKS: threshold uses all r",
                         rv > np.nanquantile(rv, PCTL_VOL), r, s, lo, hi))

    d_past = turbulence(R, MIN_HISTORY, full_sample=False)
    d_full = turbulence(R, MIN_HISTORY, full_sample=True)
    for name, d, uses in (("turbulence, expanding cov", d_past, "R[..t-1] for mean/cov and threshold"),
                          ("turbulence, full-sample cov", d_full, "LEAKS: mean/cov from all R")):
        sm = pd.Series(d).rolling(TURB_SMOOTH).mean().shift(1).to_numpy()
        thr = expanding_quantile(sm, PCTL_TURB, MIN_HISTORY)
        rows.append(evaluate(name, uses, sm > thr, r, s, lo, hi))

    sma = pd.Series(price).rolling(SMA).mean().shift(1).to_numpy()
    down = np.r_[np.nan, price[:-1]] < sma
    high_vol = rv > expanding_quantile(rv, 0.5, MIN_HISTORY)
    rows.append(evaluate(f"price < {SMA}d SMA", "price[..t-1]", down, r, s, lo, hi))
    rows.append(evaluate("downtrend AND high vol", "quadrant: SMA x expanding median vol",
                         down & high_vol, r, s, lo, hi))
    peak = np.maximum.accumulate(price)
    in_dd = np.r_[False, (price / peak - 1.0 < -DD_LIMIT)[:-1]]
    rows.append(evaluate(f"drawdown > {int(DD_LIMIT * 100)}%", "price[..t-1]", in_dd, r, s, lo, hi))

    t0 = time.time()
    res, llfs = fit_ms(r)
    p_pred = np.asarray(res.predicted_marginal_probabilities)[:, calm_index(res)]
    rows.append(evaluate("Markov switching, predicted", "r[..t-1]; params from ALL r",
                         p_pred < 0.5, r, s, lo, hi))
    ms_time = time.time() - t0

    quad = pd.crosstab(pd.Series(np.where(down[lo:hi], "down", "up"), name="trend"),
                       pd.Series(np.where(np.nan_to_num(high_vol[lo:hi]), "high vol", "low vol"),
                                 name="vol"), normalize=True)

    tab = pd.DataFrame(rows).set_index("detector")
    print("\n=== Detectors on the same regimes (delays in days, medians; missed switches count"
          " the whole episode) ===")
    cols = ["acc", "flag_share", "into_turb", "into_calm", "false_alarms", "missed",
            "sharpe", "maxdd", "switches"]
    print(tab[cols].to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\n  uses:")
    for name, uses in tab["uses"].items():
        print(f"    {name:<30} {uses}")

    print("\n=== Leak sizes (leaky minus honest, same detector) ===")
    for a, b in ((f"vol > full-sample p{int(PCTL_VOL * 100)}", f"vol > expanding p{int(PCTL_VOL * 100)}"),
                 ("turbulence, full-sample cov", "turbulence, expanding cov")):
        print(f"  {a:<30} vs {b:<28} Sharpe {tab.loc[a, 'sharpe'] - tab.loc[b, 'sharpe']:+.3f},"
              f" acc {tab.loc[a, 'acc'] - tab.loc[b, 'acc']:+.3f}")
    print("  (small here because the simulated vol level is stationary; on a series whose vol"
          " level trends, a full-sample percentile is set by years the strategy had not seen)")

    print(f"\n=== Trend/vol quadrant occupancy in the window ===")
    print(quad.to_string(float_format=lambda x: f"{x:6.3f}"))
    print(f"\n  Markov fit: llf per start {np.round(llfs, 2)}, {ms_time:.1f}s")
    dofs = [f"vol window {VOL_WINDOW}", f"vol percentile {PCTL_VOL}", "the fixed threshold",
            f"turbulence smoothing {TURB_SMOOTH}", f"turbulence percentile {PCTL_TURB}",
            f"SMA {SMA}", f"drawdown {DD_LIMIT}", f"min history {MIN_HISTORY}",
            "the 0.5 cut on the Markov probability"]
    print(f"\n  Researcher degrees of freedom used above: {len(dofs)}, each of which is a trial: "
          + ", ".join(dofs) + ".")
    print(f"\ntotal runtime {time.time() - t_all:.1f}s")
