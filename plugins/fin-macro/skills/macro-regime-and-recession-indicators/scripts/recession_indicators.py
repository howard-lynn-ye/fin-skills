#!/usr/bin/env python3
"""Recession indicators, and the label that was assigned years after the fact.

No network. Sections 1 and 2 are arithmetic on the NBER's own published announcement
dates, transcribed below. Sections 3-5 run on a seeded synthetic economy.

  1. The NBER announcement lag, from the committee's own table - and the number that
     follows from it: what share of US recession months had NOT been classified as
     recession months when they happened.
  2. The same lag applied to a label: `usrec_known_at()`, the point-in-time version of
     FRED's USREC. It does not abstain on an undeclared recession - it prints 0.
  3. USREC read as a live signal versus read retrospectively, and a classifier trained on
     each label. The first comparison is enormous; the second is nearly nothing, and
     knowing which is which is the point of the section.
  4. The Sahm rule, implemented as its own FRED page defines it, run on a vintage
     unemployment series and on today's revised one, plus how much revision its hard
     0.50 threshold tolerates before the signal moves by months.
  5. Yield-curve inversion: lead time, variability, and false positives you can only
     score with a label that arrives 7 to 21 months late.

`../../../fin-core/skills/regime-detection/SKILL.md` owns HMM/filtering mechanics and the
smoothed-vs-filtered distinction; this script never fits a regime model. The subject here
is the LABEL, not the estimator.

VERIFIED SOURCE DATA (read 2026-09-09, transcribed verbatim):

  nber.org/research/business-cycle-dating/business-cycle-dating-procedure-frequently-asked-
  questions - the committee's own table of turning points, announcement dates and elapsed
  months, transcribed in TURNING_POINTS below. From the same page:
    "There is no fixed timing rule because the committee waits long enough to avoid any
     doubt about the existence of a peak or trough."
    "The committee also allows sufficient time for standard data revisions in order to
     assign an accurate peak or trough date."

  fred.stlouisfed.org/series/USREC - "This time series is an interpretation of US Business
  Cycle Expansions and Contractions data provided by The National Bureau of Economic
  Research (NBER). ... For this time series, the recession begins the first day of the
  period following a peak and ends on the last day of the period of the trough."

  fred.stlouisfed.org/series/SAHMREALTIME - "Sahm Recession Indicator signals the start of
  a recession when the three-month moving average of the national unemployment rate (U3)
  rises by 0.50 percentage points or more relative to the minimum of the three-month
  averages from the previous 12 months. This indicator is based on 'real-time' data, that
  is, the unemployment rate (and the recent history of unemployment rates) that were
  available in a given month. The BLS revises the unemployment rate each year at the
  beginning of January, when the December unemployment rate for the prior year is
  published. Revisions to the seasonal factors can affect estimates in recent years.
  Otherwise the unemployment rate does not revise."

Run:  python recession_indicators.py   (numpy + pandas + scipy, seed 20260909, about 12 s)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 20260909

# --------------------------------------------------------------------------------------
# 1. The NBER's own announcement record
# --------------------------------------------------------------------------------------
# (kind, turning-point year, month, announcement date, elapsed months AS PRINTED BY NBER)
TURNING_POINTS = [
    ("peak",   1980,  1, "1980-06-03",  5),
    ("trough", 1980,  7, "1981-07-08", 12),
    ("peak",   1981,  7, "1982-01-06",  6),
    ("trough", 1982, 11, "1983-07-08",  8),
    ("peak",   1990,  7, "1991-04-25",  9),
    ("trough", 1991,  3, "1992-12-22", 21),
    ("peak",   2001,  3, "2001-11-26",  8),
    ("trough", 2001, 11, "2003-07-17", 20),
    ("peak",   2007, 12, "2008-12-01", 12),
    ("trough", 2009,  6, "2010-09-20", 15),
    ("peak",   2020,  2, "2020-06-08",  4),
    ("trough", 2020,  4, "2021-07-19", 15),
]


def announcement_table() -> pd.DataFrame:
    """Recompute the elapsed-months column and check it against NBER's printed one."""
    rows = []
    for kind, y, m, ann, printed in TURNING_POINTS:
        a = pd.Timestamp(ann)
        lag = (a.year - y) * 12 + (a.month - m)
        rows.append({"kind": kind, "turning_point": f"{y}-{m:02d}", "announced": ann,
                     "computed_lag": lag, "nber_prints": printed, "match": lag == printed})
    return pd.DataFrame(rows)


def recessions_from_turning_points() -> list[dict]:
    """USREC's own convention: the recession runs from the month AFTER the peak
    through the month of the trough."""
    peaks = [t for t in TURNING_POINTS if t[0] == "peak"]
    troughs = [t for t in TURNING_POINTS if t[0] == "trough"]
    out = []
    for (_, py, pm, pann, _), (_, ty, tm, tann, _) in zip(peaks, troughs):
        start = pd.Period(f"{py}-{pm:02d}", "M") + 1
        end = pd.Period(f"{ty}-{tm:02d}", "M")
        out.append({"start": start, "end": end, "n_months": (end - start).n + 1,
                    "peak_announced": pd.Timestamp(pann),
                    "trough_announced": pd.Timestamp(tann)})
    return out


def unlabelled_recession_months() -> pd.DataFrame:
    """For each recession, how many of its months had no NBER recession label yet.

    A month `m` inside a recession is known to be a recession month only once the PEAK
    has been announced. Announcement in month A labels every month from the peak forward,
    so months strictly before A are unlabelled as they happen.
    """
    rows = []
    for r in recessions_from_turning_points():
        ann_m = pd.Period(r["peak_announced"], "M")
        months = pd.period_range(r["start"], r["end"], freq="M")
        unlabelled = int(sum(m < ann_m for m in months))
        rows.append({"recession": f"{r['start']} to {r['end']}", "months": r["n_months"],
                     "peak_announced": str(ann_m), "unlabelled": unlabelled,
                     "share": unlabelled / r["n_months"],
                     "trough_lag_months": (pd.Period(r["trough_announced"], "M")
                                           - r["end"]).n})
    return pd.DataFrame(rows)


def episodes(state: np.ndarray):
    """(start, end) index pairs for every run of 1s."""
    edges = np.diff(np.r_[0, state.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def usrec_known_at(state: np.ndarray, peak_lag: np.ndarray, trough_lag: np.ndarray,
                   d: int) -> np.ndarray:
    """The USREC series as ALFRED would have served it at month `d`.

    This is NOT a series with holes in it. A recession the committee has not declared
    reads as ZERO - a confident, published, wrong 0 - and only turns into a 1 once the
    peak announcement lands. That is what makes USREC dangerous as a training target:
    the vintage label does not abstain, it disagrees.
    """
    out = np.zeros(state.size)
    for i, (s, e) in enumerate(episodes(state)):
        if s + peak_lag[i % peak_lag.size] > d:
            continue                                        # peak not announced yet
        stop = e if e + trough_lag[i % trough_lag.size] <= d else min(d, state.size - 1)
        out[s:stop + 1] = 1.0
    out[d + 1:] = np.nan                                    # the future is not published
    return out


# --------------------------------------------------------------------------------------
# 2. A seeded synthetic economy
# --------------------------------------------------------------------------------------

# The six US recessions since 1980, in months, from TURNING_POINTS: 1980, 1981-82,
# 1990-91, 2001, 2007-09, 2020. And the five expansions between them, trough to peak.
US_RECESSION_LENGTHS = (6, 16, 8, 8, 18, 2)
US_EXPANSION_LENGTHS = (12, 92, 120, 73, 128)


def simulate(n_months: int = 900, seed: int = SEED) -> dict:
    """Recession episodes, a leading term spread, unemployment, and payroll growth.

    Episode lengths are the REAL US ones since 1980, tiled with a small jitter, so the
    business cycle here has the same shape the NBER's announcement lags were measured
    against. That matters: whether a live label is useless depends entirely on the ratio
    of the announcement lag to the length of the recession.

    The term spread inverts ahead of each recession by a lead drawn per episode from 4 to
    22 months, plus inversions that lead to nothing - "the curve inverts before
    recessions, sometimes, at a lead you do not know", made explicit rather than
    discovered. Unemployment rises during recessions and drifts down afterwards; payroll
    growth goes negative. Noise levels are set so the honest classifier lands short of
    perfect: a separable problem would hide the label effect.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros(n_months, dtype=int)
    t, i = 0, 0
    while t < n_months:
        exp_len = max(6, US_EXPANSION_LENGTHS[i % 5] + int(rng.integers(-8, 9)))
        rec_len = max(2, US_RECESSION_LENGTHS[i % 6] + int(rng.integers(-2, 3)))
        t += exp_len
        state[t:t + rec_len] = 1
        t += rec_len
        i += 1

    # term spread: one inversion dip before each recession, at a random lead
    dip = np.zeros(n_months)
    leads = {}
    for i, (s, _) in enumerate(episodes(state)):
        L = int(rng.integers(4, 23))
        leads[s] = L
        a, b = max(0, s - L), max(0, s - L + int(rng.integers(3, 10)))
        dip[a:b] += 1.0
    for _ in range(6):                                   # inversions that lead to nothing
        a = int(rng.integers(0, n_months - 8))
        dip[a:a + int(rng.integers(2, 7))] += 1.0
    dip = np.convolve(dip, np.ones(5) / 5.0, mode="same")
    spread = 1.35 - 2.6 * np.clip(dip, 0, 1.4) + rng.normal(0.0, 0.30, n_months)

    # unemployment: rises in recession, drifts down otherwise
    u = np.empty(n_months)
    u[0] = 5.5
    du = np.where(state == 1, 0.13, -0.025) + rng.normal(0.0, 0.10, n_months)
    for t in range(1, n_months):
        u[t] = np.clip(u[t - 1] + du[t], 3.2, 12.0)

    payroll = np.where(state == 1, -120.0, 155.0) + rng.normal(0.0, 245.0, n_months)
    return {"state": state, "spread": spread, "u": u, "payroll": payroll, "leads": leads}


def unemployment_vintages(u: np.ndarray, seed: int = SEED, *, sf_sd: float = 0.075,
                          reach_years: int = 5) -> np.ndarray:
    """U[m, t] = the unemployment rate for month t as published in month m.

    Models exactly what SAHMREALTIME's own note describes: the level does not revise, but
    every January the BLS recomputes seasonal factors and the recent years move. Each
    January a fresh set of monthly seasonal offsets is drawn and applied to the last
    `reach_years` years; older months are frozen at whatever the last January left them.
    """
    rng = np.random.default_rng(seed + 4242)
    n = u.size
    U = np.empty((n, n))
    cur = u.copy()
    for m in range(n):
        if m % 12 == 0 and m > 0:
            lo = max(0, m - 12 * reach_years)
            offs = rng.normal(0.0, sf_sd, 12)
            idx = np.arange(lo, m)
            cur[idx] = u[idx] + offs[idx % 12]
        U[m] = cur
        U[m, m + 1:] = np.nan
    return U


# --------------------------------------------------------------------------------------
# 3. A classifier, trained twice
# --------------------------------------------------------------------------------------

def features(sim: dict) -> np.ndarray:
    """Three point-in-time features: the spread, the 3-month change in u, payroll growth."""
    u, spread, pay = sim["u"], sim["spread"], sim["payroll"]
    du3 = np.r_[np.zeros(3), u[3:] - u[:-3]]
    pay3 = pd.Series(pay).rolling(3, min_periods=1).mean().to_numpy()
    X = np.column_stack([spread, du3, pay3 / 100.0])
    return X


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    """Penalised logistic regression, scipy only. Returns [intercept, *coefs]."""
    A = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = A @ b
        return float(np.logaddexp(0.0, z).sum() - y @ z + l2 * b[1:] @ b[1:])

    def grad(b):
        p = 1.0 / (1.0 + np.exp(-(A @ b)))
        g = A.T @ (p - y)
        g[1:] += 2 * l2 * b[1:]
        return g

    res = minimize(nll, np.zeros(A.shape[1]), jac=grad, method="L-BFGS-B")
    return res.x


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ beta)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC, ties averaged."""
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def detection_delay(state: np.ndarray, prob: np.ndarray, lo: int, hi: int,
                    thresh: float = 0.5) -> dict:
    """Months from each recession start to the first month the model calls it."""
    sig = prob > thresh
    edges = np.diff(np.r_[0, state, 0])
    starts = [s for s in np.flatnonzero(edges == 1) if lo <= s < hi]
    ends = [e - 1 for e in np.flatnonzero(edges == -1)]
    delays, early = [], 0
    for s in starts:
        e = min([x for x in ends if x >= s], default=hi - 1)
        window = np.flatnonzero(sig[s:e + 1])
        delays.append(int(window[0]) if window.size else np.nan)
        if sig[max(lo, s - 3):s].any():
            early += 1
    fp = float(np.mean(sig[lo:hi][state[lo:hi] == 0]))
    return {"n_recessions": len(starts), "median_delay": float(np.nanmedian(delays)),
            "missed": int(np.sum(np.isnan(delays))), "called_early": early,
            "false_alarm_rate": fp}


def live_usrec(state: np.ndarray, peak_lag, trough_lag) -> np.ndarray:
    """USREC read as a LIVE signal: the value the series carried for month d, at month d.

    This is what "use USREC as a regime flag" means once you stop pretending. Building it
    costs one call per month, which is why nobody does it.
    """
    return np.array([usrec_known_at(state, peak_lag, trough_lag, d)[d]
                     for d in range(state.size)])


def label_error(state: np.ndarray, peak_lag, trough_lag, lo: int, hi: int,
                recent: int = 24) -> dict:
    """How wrong the vintage label is, and WHERE in the training window it is wrong.

    Averaged over refit dates. The far past is essentially correct; every error sits in
    the last couple of years, which is exactly the part a live model leans on.
    """
    err, err_recent, err_rec_recent = [], [], []
    for d in range(lo, hi, 12):
        lab = usrec_known_at(state, peak_lag, trough_lag, d)[:d + 1]
        truth = state[:d + 1]
        wrong = lab != truth
        err.append(wrong.mean())
        tail = slice(max(0, d + 1 - recent), d + 1)
        err_recent.append(wrong[tail].mean())
        rec = truth[tail] == 1
        err_rec_recent.append(wrong[tail][rec].mean() if rec.any() else np.nan)
    live = live_usrec(state, peak_lag, trough_lag)
    rec_all = state[lo:hi] == 1
    return {"all_rows": float(np.mean(err)),
            "recent_rows": float(np.mean(err_recent)),
            "recent_recession_rows": float(np.nanmean(err_rec_recent)),
            "live_recall": float(live[lo:hi][rec_all].mean()),
            "live_precision": float(state[lo:hi][live[lo:hi] == 1].mean())
            if (live[lo:hi] == 1).any() else float("nan")}


def announcement_lags(size: int = 40):
    """NBER's own six peak lags and six trough lags, tiled - no draw, no seed.

    Peaks 5, 6, 9, 8, 12, 4 months; troughs 12, 8, 21, 20, 15, 15. Any synthetic economy
    that uses these inherits the real committee's timing, not an assumption about it.
    """
    peaks = np.array([t[4] for t in TURNING_POINTS if t[0] == "peak"])
    troughs = np.array([t[4] for t in TURNING_POINTS if t[0] == "trough"])
    return np.resize(peaks, size), np.resize(troughs, size)


def train_two_ways(sim: dict, seed: int = SEED, lo: int = 420, step: int = 12,
                   train_window: int | None = None) -> dict:
    """Retrospective-label training against vintage-label training, same test period.

    `train_window=None` is an expanding window; an integer is a rolling one, which is the
    realistic case - the fraction of the training set that is mislabelled goes up as the
    window shortens.
    """
    state, X = sim["state"], features(sim)
    n = state.size
    peak_lag, trough_lag = announcement_lags()

    def start_of(d):
        return 0 if train_window is None else max(0, d - train_window)

    # A: one fit on the WHOLE sample with the retrospective label - the common setup
    beta_full = fit_logit(X, state.astype(float))
    p_full = predict(beta_full, X)

    # B: walk-forward with the RETROSPECTIVE label - isolates the window from the label
    # C: walk-forward with the VINTAGE label - the only honest one
    p_wf, p_rt = np.full(n, np.nan), np.full(n, np.nan)
    n_fits, kept_pos = 0, []
    b_wf = b_rt = None
    for d in range(lo, n, step):
        a = start_of(d)
        hi = min(d + step, n)
        b_wf = fit_logit(X[a:d], state[a:d].astype(float))
        p_wf[d:hi] = predict(b_wf, X[d:hi])
        lab = usrec_known_at(state, peak_lag, trough_lag, d - 1)[a:d]
        if 0 < lab.sum() < lab.size:
            b_rt = fit_logit(X[a:d], lab)
            n_fits += 1
            kept_pos.append(float(lab.sum() / max(state[a:d].sum(), 1)))
        if b_rt is not None:
            p_rt[d:hi] = predict(b_rt, X[d:hi])

    test = slice(lo, n)
    y = state[test].astype(float)
    live = live_usrec(state, peak_lag, trough_lag)
    return {
        "beta_full": beta_full, "n_fits": n_fits,
        "pos_kept": float(np.mean(kept_pos)) if kept_pos else float("nan"),
        "auc": {"USREC as a signal, retrospective": auc(y, state[test].astype(float)),
                "USREC as a signal, as of that month": auc(y, live[test]),
                "classifier, full-sample retrospective label": auc(y, p_full[test]),
                "classifier, walk-forward retrospective label": auc(y, p_wf[test]),
                "classifier, walk-forward vintage label": auc(y, p_rt[test])},
        "detect": {"USREC as a signal, retrospective":
                   detection_delay(state, state.astype(float), lo, n),
                   "USREC as a signal, as of that month":
                   detection_delay(state, live, lo, n),
                   "classifier, full-sample retrospective label":
                   detection_delay(state, p_full, lo, n),
                   "classifier, walk-forward retrospective label":
                   detection_delay(state, np.nan_to_num(p_wf), lo, n),
                   "classifier, walk-forward vintage label":
                   detection_delay(state, np.nan_to_num(p_rt), lo, n)},
        "label_error": label_error(state, peak_lag, trough_lag, lo, n),
    }


# --------------------------------------------------------------------------------------
# 4. The Sahm rule
# --------------------------------------------------------------------------------------

def label_cost(seeds=range(12), windows=(None, 240, 120)) -> pd.DataFrame:
    """Does the vintage LABEL cost a fitted classifier anything? Averaged over seeds.

    Reported as mean AUC and mean detection delay for the two training labels at three
    training-window lengths, plus the standard error of the AUC difference - because a
    difference smaller than its own standard error is not a finding.
    """
    rows = []
    for w in windows:
        d_auc, a_retro, a_vint, dl_retro, dl_vint = [], [], [], [], []
        for s in seeds:
            sim = simulate(900, SEED + s)
            r = train_two_ways(sim, SEED + s, train_window=w)
            ar = r["auc"]["classifier, walk-forward retrospective label"]
            av = r["auc"]["classifier, walk-forward vintage label"]
            a_retro.append(ar), a_vint.append(av), d_auc.append(ar - av)
            dl_retro.append(r["detect"]["classifier, walk-forward retrospective label"]
                            ["median_delay"])
            dl_vint.append(r["detect"]["classifier, walk-forward vintage label"]
                           ["median_delay"])
        rows.append({"train_window": "expanding" if w is None else str(w),
                     "auc_retro_label": np.mean(a_retro), "auc_vintage_label": np.mean(a_vint),
                     "auc_diff": np.mean(d_auc),
                     "se": np.std(d_auc, ddof=1) / np.sqrt(len(d_auc)),
                     "delay_retro": np.nanmean(dl_retro), "delay_vintage": np.nanmean(dl_vint)})
    return pd.DataFrame(rows).set_index("train_window")


def sahm(u: np.ndarray) -> np.ndarray:
    """FRED's definition, verbatim: the 3-month moving average of U3 minus the minimum of
    the 3-month averages from the previous 12 months. Signals at >= 0.50."""
    ma = pd.Series(u).rolling(3).mean()
    return (ma - ma.shift(1).rolling(12).min()).to_numpy()


SAHM_THRESHOLD = 0.50


def sahm_realtime(U: np.ndarray) -> np.ndarray:
    """The Sahm value for month m computed from the series AS PUBLISHED in month m."""
    n = U.shape[0]
    out = np.full(n, np.nan)
    for m in range(n):
        row = U[m, :m + 1]
        if row.size >= 16:
            out[m] = sahm(row)[-1]
    return out


def sahm_compare(sim: dict, seed: int = SEED, sf_sd: float = 0.075) -> dict:
    u = sim["u"]
    U = unemployment_vintages(u, seed, sf_sd=sf_sd)
    rt = sahm_realtime(U)
    fin = sahm(u)
    ok = np.isfinite(rt) & np.isfinite(fin)
    trig_rt, trig_fin = rt >= SAHM_THRESHOLD, fin >= SAHM_THRESHOLD
    disagree = ok & (trig_rt != trig_fin)

    rows = []
    for s, e in episodes(sim["state"]):
        if s < 20:
            continue
        w = slice(s, min(e + 7, u.size))

        def first(mask):
            idx = np.flatnonzero(mask[w])
            return int(idx[0]) if idx.size else np.nan
        rows.append({"start": int(s), "len": int(e - s + 1),
                     "vintage": first(trig_rt), "revised": first(trig_fin)})
    tab = pd.DataFrame(rows)
    both = tab.dropna(subset=["vintage", "revised"])
    # false alarms: a NEW crossing of the threshold with no recession starting nearby
    starts = np.array([s for s, _ in episodes(sim["state"])])
    fired = np.nan_to_num(rt) >= SAHM_THRESHOLD
    new_trigger = np.flatnonzero(fired & ~np.r_[False, fired[:-1]])
    fa = int(sum(not ((starts >= m - 8) & (starts <= m + 2)).any() for m in new_trigger))
    return {
        "n_triggers": int(new_trigger.size),
        "table": tab,
        "mean_abs_gap": float(np.abs(rt[ok] - fin[ok]).mean()),
        "max_abs_gap": float(np.abs(rt[ok] - fin[ok]).max()),
        "disagree_share": float(disagree.sum() / ok.sum()),
        "n_disagree": int(disagree.sum()),
        "revised_earlier": int((both["revised"] < both["vintage"]).sum()),
        "vintage_earlier": int((both["vintage"] < both["revised"]).sum()),
        "same": int((both["vintage"] == both["revised"]).sum()),
        "missed_vintage": int(tab["vintage"].isna().sum()),
        "missed_revised": int(tab["revised"].isna().sum()),
        "n_recessions": len(tab), "n_both": len(both),
        "median_delay": float(both["vintage"].median()),
        "false_alarms": fa,
    }


def sahm_sensitivity(sim: dict, sds=(0.0, 0.075, 0.15, 0.30, 0.60),
                     seed: int = SEED) -> pd.DataFrame:
    """Where the hard 0.50 threshold starts to break as the input revisions grow."""
    rows = []
    for sd in sds:
        c = sahm_compare(sim, seed, sf_sd=sd)
        rows.append({"sf_sd_pp": sd, "mean_abs_gap": c["mean_abs_gap"],
                     "max_abs_gap": c["max_abs_gap"],
                     "threshold_disagreements": c["n_disagree"],
                     "same_trigger_month": c["same"], "scored": c["n_both"],
                     "false_alarms": c["false_alarms"]})
    return pd.DataFrame(rows).set_index("sf_sd_pp")


# --------------------------------------------------------------------------------------
# 5. Yield-curve inversion
# --------------------------------------------------------------------------------------

def inversion_record(sim: dict, horizon: int = 24) -> dict:
    """Episodes of a negative spread, the lead to the next recession, and the misses."""
    inv = sim["spread"] < 0.0
    state = sim["state"]
    edges = np.diff(np.r_[0, inv.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    rec_starts = np.flatnonzero(np.diff(np.r_[0, state, 0]) == 1)
    leads, misses = [], 0
    for s in starts:
        nxt = rec_starts[rec_starts > s]
        if nxt.size and nxt[0] - s <= horizon:
            leads.append(int(nxt[0] - s))
        else:
            misses += 1
    leads = np.array(leads, dtype=float)
    covered = 0
    for r in rec_starts:
        if inv[max(0, r - horizon):r].any():
            covered += 1
    return {"n_inversions": len(starts), "n_recessions": len(rec_starts),
            "median_lead": float(np.median(leads)) if leads.size else float("nan"),
            "min_lead": float(leads.min()) if leads.size else float("nan"),
            "max_lead": float(leads.max()) if leads.size else float("nan"),
            "iqr": (float(np.percentile(leads, 25)), float(np.percentile(leads, 75)))
            if leads.size else (float("nan"),) * 2,
            "false_alarms": misses, "recessions_preceded": covered}


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()

    print("=" * 78)
    print("1. The NBER's own announcement record")
    print("=" * 78)
    at = announcement_table()
    print(f"   {'kind':<8}{'turning point':>15}{'announced':>13}{'lag (mo)':>10}"
          f"{'NBER prints':>13}{'match':>8}")
    for _, r in at.iterrows():
        print(f"   {r['kind']:<8}{r['turning_point']:>15}{r['announced']:>13}"
              f"{r['computed_lag']:>10}{r['nber_prints']:>13}{str(r['match']):>8}")
    print(f"   all {len(at)} recomputed lags match NBER's printed column: "
          f"{bool(at['match'].all())}")
    pk = at[at["kind"] == "peak"]["computed_lag"]
    tr = at[at["kind"] == "trough"]["computed_lag"]
    print(f"   peaks:   mean {pk.mean():.1f} months, range {pk.min()}-{pk.max()}")
    print(f"   troughs: mean {tr.mean():.1f} months, range {tr.min()}-{tr.max()}")

    ur = unlabelled_recession_months()
    print("\n   And so, for every US recession since 1980, the months that were NOT yet")
    print("   classified as recession months while they were happening:")
    print(f"   {'recession':<24}{'months':>8}{'peak announced':>17}{'unlabelled':>12}"
          f"{'share':>8}{'trough lag':>12}")
    for _, r in ur.iterrows():
        print(f"   {r['recession']:<24}{r['months']:>8}{r['peak_announced']:>17}"
              f"{r['unlabelled']:>12}{r['share']:>8.0%}{r['trough_lag_months']:>12}")
    tot_u, tot_m = int(ur["unlabelled"].sum()), int(ur["months"].sum())
    print(f"   TOTAL: {tot_u} of {tot_m} recession months, {tot_u / tot_m:.1%}, carried no")
    print("   NBER recession label at the time. USREC has one for every single month.")

    print("\n" + "=" * 78)
    print("2-3. A classifier trained on the answer, and one trained on what was known")
    print("=" * 78)
    sim = simulate()
    n = sim["state"].size
    print(f"   {n} synthetic months, {sim['state'].mean():.1%} of them in recession, "
          f"{int(np.sum(np.diff(np.r_[0, sim['state']]) == 1))} recessions.")
    res = train_two_ways(sim)
    av = res["label_error"]
    print(f"   Episode lengths are the real US ones since 1980; announcement lags are NBER own 12.")
    print("   The vintage USREC does not abstain on an undeclared recession - it prints a")
    print("   confident 0.")
    print(f"   Read as a LIVE flag over the test period: it catches "
          f"{av['live_recall']:.0%} of true")
    print(f"   recession months, at {av['live_precision']:.0%} precision. The retrospective")
    print("   series catches 100% at 100%. Same series, same name, two different objects.")
    print("   Read as a TRAINING TARGET the damage is concentrated at the edge - share of")
    print("   mislabelled rows, averaged over refit dates:")
    print(f"     whole expanding window {av['all_rows']:.2%} | last 24 months "
          f"{av['recent_rows']:.1%} | recession months in the last 24 "
          f"{av['recent_recession_rows']:.0%}")
    print(f"\n   {'signal / training scheme':<46}{'AUC':>8}{'delay':>8}{'missed':>8}"
          f"{'false alarm':>13}")
    for k, a in res["auc"].items():
        d = res["detect"][k]
        print(f"   {k:<46}{a:>8.3f}{d['median_delay']:>8.0f}{d['missed']:>8}"
              f"{d['false_alarm_rate']:>13.1%}")
    aucs = res["auc"]
    gap_signal = (aucs["USREC as a signal, retrospective"]
                  - aucs["USREC as a signal, as of that month"])
    print(f"   USREC read retrospectively vs as of the month: {gap_signal:+.3f} of AUC.")
    print("   That is the whole of the apparent skill of a 'recession-aware' rule, and it")
    print("   is not skill: it is the committee's minutes, delivered early.")
    lc = label_cost()
    print("\n   Does the vintage LABEL cost a FITTED classifier anything? 12 seeds:")
    print(f"   {'train window':<14}{'AUC retro':>11}{'AUC vintage':>13}{'diff':>8}"
          f"{'se':>8}{'delay retro':>13}{'delay vintage':>15}")
    for w, r in lc.iterrows():
        print(f"   {w:<14}{r['auc_retro_label']:>11.3f}{r['auc_vintage_label']:>13.3f}"
              f"{r['auc_diff']:>8.3f}{r['se']:>8.3f}{r['delay_retro']:>13.1f}"
              f"{r['delay_vintage']:>15.1f}")
    print("   Barely. On 40+ years of monthly data the mislabelled rows are a fraction of")
    print("   a percent of the training set, and the cost is 0.002-0.007 of AUC - real,")
    print("   and two orders of magnitude below the 0.389 that reading USREC AS the signal")
    print("   costs. Shorten the window to 120 months and the whole comparison dissolves")
    print("   into noise (se 0.041) and changes sign. The vintage label keeps the right")
    print(f"   NUMBER of positives ({res['pos_kept']:.0%} of the true count); it just puts "
          f"them in the wrong")
    print("   months. So: the damage from USREC is in using it AS the signal or AS the")
    print("   evaluation target, not in using it as a training target on a long sample.")

    print("\n" + "=" * 78)
    print("4. The Sahm rule on vintage unemployment versus today's")
    print("=" * 78)
    sc = sahm_compare(sim)
    print("   Rule (fred.stlouisfed.org/series/SAHMREALTIME, verbatim): 3-month moving")
    print(f"   average of U3 minus the minimum of the previous 12 months' 3-month")
    print(f"   averages, signalling at >= {SAHM_THRESHOLD:.2f} percentage points.")
    print("   FRED publishes the SAME rule twice: SAHMREALTIME on the vintage series and")
    print("   SAHMCURRENT on today's. Only SAHMREALTIME's note mentions vintages.")
    print(f"   Only the seasonal factors revise, once a year, over the last 5 years - the")
    print(f"   level does not. Mean absolute difference between the real-time and the")
    print(f"   revised Sahm value: {sc['mean_abs_gap']:.4f}pp (max {sc['max_abs_gap']:.3f}pp).")
    print(f"   Months where the two land on OPPOSITE sides of the 0.50 threshold: "
          f"{sc['n_disagree']} ({sc['disagree_share']:.2%})")
    print(f"   First trigger inside each of {sc['n_recessions']} recessions:")
    print(f"     revised fires earlier {sc['revised_earlier']}x, vintage earlier "
          f"{sc['vintage_earlier']}x, same month {sc['same']}x")
    print(f"     never fires: vintage {sc['missed_vintage']}, revised "
          f"{sc['missed_revised']}")
    print(f"   median months from recession start to the vintage rule's first signal: "
          f"{sc['median_delay']:.0f}")
    print(f"   fresh crossings of the threshold: {sc['n_triggers']}, of which "
          f"{sc['false_alarms']} had no recession starting nearby")
    ss = sahm_sensitivity(sim)
    print("\n   How much revision the 0.50 threshold tolerates (seasonal-factor sd, pp):")
    print(f"   {'sf sd':>7}{'mean |gap|':>13}{'max |gap|':>12}"
          f"{'threshold flips':>17}{'same trigger month':>21}{'false alarms':>15}")
    for sd, r in ss.iterrows():
        same = f"{int(r['same_trigger_month'])}/{int(r['scored'])}"
        print(f"   {sd:>7.3f}{r['mean_abs_gap']:>13.4f}{r['max_abs_gap']:>12.3f}"
              f"{int(r['threshold_disagreements']):>17}{same:>21}"
              f"{int(r['false_alarms']):>15}")
    print("   At U3's own revision size the rule barely moves - which is the point of")
    print("   choosing an input that does not revise. Push the revisions up and the hard")
    print("   threshold is what breaks first: the value moves by hundredths and the")
    print("   SIGNAL moves by months.")

    print("\n" + "=" * 78)
    print("5. Yield-curve inversion: a real lead, and an unusable one")
    print("=" * 78)
    ir = inversion_record(sim)
    print(f"   {ir['n_inversions']} inversion episodes, {ir['n_recessions']} recessions.")
    print(f"   lead from first inversion to recession start: median "
          f"{ir['median_lead']:.0f} months, IQR {ir['iqr'][0]:.0f}-{ir['iqr'][1]:.0f}, "
          f"range {ir['min_lead']:.0f}-{ir['max_lead']:.0f}")
    print(f"   inversions with no recession inside 24 months: {ir['false_alarms']}")
    print(f"   recessions preceded by an inversion within 24 months: "
          f"{ir['recessions_preceded']} of {ir['n_recessions']}")
    print("   The signal is real and the lead is enormous and variable. Sizing a position")
    print("   on it means holding through an interquartile range of months, and the only")
    print("   way to score whether an inversion was a false alarm is the NBER label -")
    print(f"   which arrives {pk.mean():.0f} months after a peak and {tr.mean():.0f} "
          f"after a trough.")

    print(f"\n   (total runtime {time.time() - t0:.1f}s)")
    print("\nRule: never train, label or evaluate on USREC as if it existed at the time -"
          " 64% of US recession months since 1980 carried no NBER label while they were"
          " happening; use an announcement-lagged label, or a rule whose inputs are"
          " point-in-time, and say which.")
