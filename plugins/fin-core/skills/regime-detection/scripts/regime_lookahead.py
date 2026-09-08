#!/usr/bin/env python3
"""Smoothed regime probabilities use the future. Filtered ones use today. Predicted ones trade.

A Markov-switching fit hands back three probability series for every date t:

    smoothed   P(S_t | r_1..r_N)     two-sided: conditions on the WHOLE sample
    filtered   P(S_t | r_1..r_t)     one-sided, but includes today's return r_t
    predicted  P(S_t | r_1..r_{t-1}) what you could actually have known at the close of t-1

The regime plot everyone shows is the smoothed one, because it is the cleanest. A strategy that
switches on it has been told the crash was coming. The filtered series is what most people
reach for as "past only", and it still leaks one day: to know P(S_t | r_t) you need r_t, and r_t
is the return you are about to trade. On top of both, the parameters themselves were estimated
on the full sample. The honest signal is the predicted probability under parameters fitted
only on data before the decision (the walk-forward row below).

This script measures the size of each leak on synthetic two-regime data, where the true regime
is known, and reports:

  1. the strategy ladder  (buy&hold, oracle, smoothed, filtered, predicted, walk-forward)
  2. detection delay after a true switch, and false alarms, for each series
  3. how the gap behaves as regimes get harder to separate, and as episodes get shorter
  4. local optima that report converged=True, label switching, the transition-matrix
     orientation, and what happens when you fit on price levels instead of returns

The Hamilton filter and Kim smoother are re-implemented here in numpy (a dozen lines each)
and checked against statsmodels' output, so the timing semantics above are verified rather
than assumed. Estimation needs statsmodels; without it the script runs the timing comparison
at the true parameters and says so.

Run:  python regime_lookahead.py        (numpy / pandas / statsmodels; fixed seeds; ~80 s)
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

try:
    # Import at module level: statsmodels arms its own "always" warning filters on import,
    # which would override a later catch_warnings block if the import happened inside it.
    import statsmodels
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    HAVE_SM = True
except ImportError:                                  # pragma: no cover
    HAVE_SM = False

SEED = 0
N = 2520                    # 10 years of daily returns
PERIODS = 252
MU = (0.0006, -0.0010)      # daily mean: calm ~ +15%/yr, turbulent ~ -25%/yr
SIG_CALM = 0.008            # daily vol in calm regime ~ 12.7% annualised
P_STAY = (0.99, 0.96)       # expected regime durations 100 d (calm) and 25 d (turbulent)
TEST_START = 756            # the first 3 years are the initial estimation window
REFIT_EVERY = 252           # walk-forward: re-estimate parameters once a year
MAIN_RATIO = 2.0
SWEEP_SEEDS = (0, 1, 2)
# (label, turbulent/calm vol ratio, p_stay turbulent, turbulent daily mean)
SWEEP = (
    ("ratio 1.25, 25d episodes", 1.25, 0.96, -0.0010),
    ("ratio 1.5,  25d episodes", 1.50, 0.96, -0.0010),
    ("ratio 2.0,  25d episodes", 2.00, 0.96, -0.0010),
    ("ratio 3.0,  25d episodes", 3.00, 0.96, -0.0010),
    ("ratio 2.0,   7d crashes ", 2.00, 0.85, -0.0040),
)
# Two optimiser starts per fit: statsmodels' default start, and its random-start search.
# Section 4 shows why one is not enough.
STARTS = ((0, 1), (5, 1))   # (search_reps, rng seed)


# ----------------------------------------------------------------------------- data ---------
def simulate(n: int, seed: int, sig_ratio: float, p_stay=P_STAY, mu=MU
             ) -> tuple[np.ndarray, np.ndarray]:
    """Two-regime Markov-switching returns. Returns (returns, true regime: 0 calm, 1 turbulent)."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay[s[t - 1]] else 1 - s[t - 1]
    sig = (SIG_CALM, SIG_CALM * sig_ratio)
    r = rng.normal(np.asarray(mu)[s], np.asarray(sig)[s])
    return r, s


# ----------------------------------------------------------------------------- reference ----
def hamilton_filter(y: np.ndarray, P: np.ndarray, mu: np.ndarray, sig2: np.ndarray,
                    pi0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P is COLUMN-stochastic, P[to, from]. Returns (predicted, filtered), each (n, k).

    predicted[t] = P(S_t | y_0..y_{t-1});  filtered[t] = P(S_t | y_0..y_t).
    """
    n = len(y)
    pred = np.empty((n, len(mu)))
    filt = np.empty_like(pred)
    p = np.asarray(pi0, dtype=float)
    for t in range(n):
        pred[t] = p
        lik = np.exp(-0.5 * (y[t] - mu) ** 2 / sig2) / np.sqrt(2 * np.pi * sig2)
        joint = p * lik
        filt[t] = joint / joint.sum()
        p = P @ filt[t]
    return pred, filt


def kim_smoother(P: np.ndarray, pred: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """smoothed[t] = P(S_t | y_0..y_{N-1}): a BACKWARD pass, so every value sees the future."""
    sm = np.empty_like(filt)
    sm[-1] = filt[-1]
    for t in range(len(filt) - 2, -1, -1):
        sm[t] = filt[t] * (P.T @ (sm[t + 1] / pred[t + 1]))
    return sm


def true_param_probs(y: np.ndarray, sig_ratio: float, p_stay=P_STAY, mu=MU) -> dict:
    """The three series at the TRUE parameters: timing effects only, no estimation error."""
    P = np.array([[p_stay[0], 1 - p_stay[1]], [1 - p_stay[0], p_stay[1]]])
    pi0 = np.array([1 - p_stay[1], 1 - p_stay[0]]) / (2 - p_stay[0] - p_stay[1])   # ergodic
    pred, filt = hamilton_filter(y, P, np.asarray(mu), np.array([SIG_CALM, SIG_CALM * sig_ratio]) ** 2,
                                 pi0)
    return {"smoothed": kim_smoother(P, pred, filt)[:, 0], "filtered": filt[:, 0],
            "predicted": pred[:, 0]}


# ----------------------------------------------------------------------------- model --------
def _model(y: np.ndarray):
    return MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)


def fit_ms(y: np.ndarray, starts=STARTS):
    """Fit from several starts, keep the highest log-likelihood. Returns (result, llf per start)."""
    fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for reps, rs in starts:
            fits.append(_model(y).fit(search_reps=reps, rng=np.random.default_rng(rs), disp=False))
    llfs = [float(f.llf) for f in fits]
    return fits[int(np.argmax(llfs))], llfs


def calm_index(res) -> int:
    """The regime with the smaller variance. NEVER assume it is regime 0 (see section 4)."""
    names = list(res.model.param_names)
    s2 = [res.params[names.index(f"sigma2[{k}]")] for k in range(2)]
    return int(np.argmin(s2))


def p_calm_series(res, calm: int) -> dict[str, np.ndarray]:
    return {
        "smoothed": np.asarray(res.smoothed_marginal_probabilities)[:, calm],
        "filtered": np.asarray(res.filtered_marginal_probabilities)[:, calm],
        "predicted": np.asarray(res.predicted_marginal_probabilities)[:, calm],
    }


def verify_reference(res) -> dict[str, float]:
    """Max abs difference between the numpy recursions and statsmodels, at the fitted params."""
    names = list(res.model.param_names)
    P = np.asarray(res.regime_transition)[:, :, 0]
    mu = np.array([res.params[names.index(f"const[{k}]")] for k in range(2)])
    sig2 = np.array([res.params[names.index(f"sigma2[{k}]")] for k in range(2)])
    pred, filt = hamilton_filter(np.asarray(res.model.endog).ravel(), P, mu, sig2,
                                 np.asarray(res.initial_probabilities))
    sm = kim_smoother(P, pred, filt)
    return {"predicted": float(np.abs(pred - res.predicted_marginal_probabilities).max()),
            "filtered": float(np.abs(filt - res.filtered_marginal_probabilities).max()),
            "smoothed": float(np.abs(sm - res.smoothed_marginal_probabilities).max())}


def walk_forward_predicted(y: np.ndarray, start: int, step: int) -> tuple[np.ndarray, int]:
    """P(calm at t | r_1..r_{t-1}) with parameters estimated ONLY on data before the last refit."""
    n = len(y)
    out = np.full(n, np.nan)
    n_fits = 0
    for k in range(start, n, step):
        res, _ = fit_ms(y[:k])
        n_fits += 1
        calm = calm_index(res)
        end = min(k + step, n)
        # Run the (causal) filter forward through the next block with the frozen parameters.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = _model(y[:end]).filter(res.params)
        out[k:end] = np.asarray(fr.predicted_marginal_probabilities)[k:end, calm]
    return out, n_fits


# ----------------------------------------------------------------------------- metrics ------
def strategy_stats(r: np.ndarray, pos: np.ndarray) -> dict[str, float]:
    ret = pos * r
    sd = ret.std(ddof=1)
    sharpe = float(ret.mean() / sd * np.sqrt(PERIODS)) if sd > 0 else float("nan")
    eq = np.cumprod(1.0 + ret)
    maxdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    cagr = float(eq[-1] ** (PERIODS / len(ret)) - 1.0)
    return {"sharpe": sharpe, "cagr": cagr, "maxdd": maxdd,
            "in_mkt": float(pos.mean()), "switches": int((np.diff(pos) != 0).sum())}


def switch_points(s: np.ndarray, lo: int, hi: int) -> list[int]:
    return [t for t in range(max(lo, 1), hi) if s[t] != s[t - 1]]


def detection(s: np.ndarray, p_calm: np.ndarray, lo: int, hi: int) -> dict[str, float]:
    """Signed delay (days) until the series first puts P>0.5 on the NEW true regime.

    Negative = the series flagged the new regime BEFORE it happened (only possible if it saw
    the future, or by a lucky false alarm). 'missed' = the regime ended before detection.
    """
    sw = switch_points(s, lo, hi)
    delays, missed = [], 0
    for i, t0 in enumerate(sw):
        new = s[t0]
        q = p_calm if new == 0 else 1.0 - p_calm           # P(new regime)
        t_end = sw[i + 1] if i + 1 < len(sw) else hi
        t_prev = sw[i - 1] if i > 0 else lo
        if q[t0] > 0.5:                                     # already flagged: how early?
            lead = 0
            while t0 - lead - 1 >= t_prev and q[t0 - lead - 1] > 0.5:
                lead += 1
            delays.append(-lead)
            continue
        hit = next((t for t in range(t0, t_end) if q[t] > 0.5), None)
        if hit is None:
            missed += 1
        else:
            delays.append(hit - t0)
    d = np.asarray(delays, dtype=float)
    flagged = (p_calm[lo:hi] > 0.5).astype(int)
    n_flag_switches = int((np.diff(flagged) != 0).sum())
    return {
        "n_switches": len(sw),
        "median_delay": float(np.median(d)) if len(d) else float("nan"),
        "mean_delay": float(d.mean()) if len(d) else float("nan"),
        "share_anticipated": float((d < 0).mean()) if len(d) else float("nan"),
        "missed": missed,
        "false_alarms": max(n_flag_switches - len(sw), 0),
    }


def label_accuracy(s: np.ndarray, p_calm: np.ndarray, lo: int, hi: int) -> float:
    return float(((p_calm[lo:hi] > 0.5) == (s[lo:hi] == 0)).mean())


def sharpe_of(r, p_calm, lo, hi) -> float:
    return strategy_stats(r[lo:hi], (p_calm[lo:hi] > 0.5).astype(float))["sharpe"]


# ----------------------------------------------------------------------------- sections -----
def ladder(r: np.ndarray, s: np.ndarray, lo: int, hi: int) -> tuple[pd.DataFrame, dict]:
    info: dict = {}
    probs = {f"{k} @ TRUE params": v for k, v in true_param_probs(r, MAIN_RATIO).items()}
    if HAVE_SM:
        res, llfs = fit_ms(r)
        calm = calm_index(res)
        probs.update(p_calm_series(res, calm))
        probs["walk-forward"], n_fits = walk_forward_predicted(r, lo, REFIT_EVERY)
        info = {"res": res, "calm": calm, "n_fits": n_fits, "llfs": llfs,
                "ref_check": verify_reference(res)}

    signals = [  # (name, position series, what it conditions on, probability key)
        ("buy & hold", np.ones(len(r)), "nothing", None),
        ("oracle (true regime today)", (s == 0).astype(float), "TRUE regime at t: not available",
         None),
        ("oracle lagged 1 day", np.r_[1.0, (s[:-1] == 0)], "true regime at t-1: best causal", None),
    ]
    uses = {"smoothed": "r_1..r_N", "filtered": "r_1..r_t", "predicted": "r_1..r_{t-1}"}
    for k, u in uses.items():
        signals.append((f"{k} @ TRUE params", None, f"{u} + true params (no estimation)",
                        f"{k} @ TRUE params"))
    if HAVE_SM:
        for k, u in uses.items():
            signals.append((k if k != "filtered" else "filtered (same day)", None,
                            f"{u} + full-sample params", k))
        signals.append(("walk-forward predicted", None, "r_1..r_{t-1} + params fitted before t",
                        "walk-forward"))
    rows = []
    for name, pos, use, key in signals:
        if key is not None:
            pos = np.nan_to_num(probs[key] > 0.5).astype(float)
        row = {"signal": name, "uses": use, **strategy_stats(r[lo:hi], pos[lo:hi])}
        if key is not None:
            det = detection(s, probs[key], lo, hi)
            row.update(label_acc=label_accuracy(s, probs[key], lo, hi),
                       median_delay=det["median_delay"], anticipated=det["share_anticipated"],
                       missed=det["missed"], false_alarms=det["false_alarms"])
        rows.append(row)
    info["probs"] = probs
    return pd.DataFrame(rows).set_index("signal"), info


def separation_sweep(lo: int, hi: int) -> pd.DataFrame:
    rows = []
    for label, ratio, p11, mu1 in SWEEP:
        for seed in SWEEP_SEEDS:
            p_stay, mu = (P_STAY[0], p11), (MU[0], mu1)
            r, s = simulate(N, seed, ratio, p_stay, mu)
            tp = true_param_probs(r, ratio, p_stay, mu)
            row = {"config": label, "seed": seed,
                   "buy_hold": strategy_stats(r[lo:hi], np.ones(hi - lo))["sharpe"],
                   "oracle_lag1": strategy_stats(r[lo:hi], np.r_[1.0, (s[:-1] == 0)][lo:hi])["sharpe"],
                   "gap_true_params": sharpe_of(r, tp["smoothed"], lo, hi)
                   - sharpe_of(r, tp["predicted"], lo, hi),
                   "turb_share": float((s[lo:hi] == 1).mean())}
            if HAVE_SM:
                res, llfs = fit_ms(r)
                pr = p_calm_series(res, calm_index(res))
                sh = {k: sharpe_of(r, v, lo, hi) for k, v in pr.items()}
                det_f, det_s = detection(s, pr["filtered"], lo, hi), detection(s, pr["smoothed"], lo, hi)
                row.update(smoothed=sh["smoothed"], filtered=sh["filtered"], predicted=sh["predicted"],
                           gap=sh["smoothed"] - sh["predicted"],
                           gap_fp=sh["filtered"] - sh["predicted"],
                           acc_smoothed=label_accuracy(s, pr["smoothed"], lo, hi),
                           acc_predicted=label_accuracy(s, pr["predicted"], lo, hi),
                           delay_filtered=det_f["median_delay"], delay_smoothed=det_s["median_delay"],
                           false_alarms_filt=det_f["false_alarms"], missed_filt=det_f["missed"],
                           llf_default_start=llfs[0], llf_search_start=llfs[1],
                           starts_disagree=abs(llfs[0] - llfs[1]) > 1e-3)
            rows.append(row)
    return pd.DataFrame(rows)


def fit_pathologies(r: np.ndarray, s: np.ndarray, main_llfs: list[float],
                    sweep: pd.DataFrame) -> None:
    print("\n=== 4. Things the fit does that nothing warns you about ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = _model(r).fit(search_reps=0, disp=False)
        search = _model(r).fit(search_reps=5, rng=np.random.default_rng(1), disp=False)
        flat = _model(r).fit(start_params=np.array([0.5, 0.5, 0.0, 0.0, r.var(), r.var()]),
                             disp=False)
        swapped = _model(r).fit(search_reps=20, rng=np.random.default_rng(7), disp=False)

    def s2(res):
        names = list(res.model.param_names)
        return [float(res.params[names.index(f"sigma2[{k}]")]) for k in range(2)]

    print("  local optima    : same data, two optimiser starts, both report converged=True")
    for name, res in (("default start (search_reps=0)", default),
                      ("random-start search (search_reps=5)", search)):
        print(f"    {name:<38} llf={res.llf:9.3f} converged={res.mle_retvals['converged']}"
              f"  sigma={np.sqrt(s2(res)[0]):.5f}/{np.sqrt(s2(res)[1]):.5f}"
              f"  expected durations={np.round(res.expected_durations, 1)} d")
    print(f"    llf gap {abs(default.llf - search.llf):.1f}; the loser has a 'regime' whose expected"
          f" duration is about one day -- that is an outlier detector, not a regime")
    n_dis = int(sweep["starts_disagree"].sum())
    worst = float((sweep["llf_default_start"] - sweep["llf_search_start"]).abs().max())
    n_search_wins = int((sweep["llf_search_start"] > sweep["llf_default_start"] + 1e-3).sum())
    print(f"    across the {len(sweep)} sweep fits the two starts disagreed {n_dis} times"
          f" (max llf gap {worst:.1f}); the random-start search won {n_search_wins} of them")
    print(f"  'converged' fit : start from equal regimes -> converged={flat.mle_retvals['converged']}"
          f"  llf={flat.llf:.3f}  vs best llf={max(main_llfs):.3f}  (gap {max(main_llfs) - flat.llf:.1f})")
    print(f"                    params {np.round(flat.params, 5)}  <- one regime, twice")

    a, b = default, swapped
    print(f"  label switching : fit A llf={a.llf:.3f} sigma2[0]={s2(a)[0]:.2e} sigma2[1]={s2(a)[1]:.2e}"
          f" -> calm is regime {int(np.argmin(s2(a)))}")
    print(f"                    fit B llf={b.llf:.3f} sigma2[0]={s2(b)[0]:.2e} sigma2[1]={s2(b)[1]:.2e}"
          f" -> calm is regime {int(np.argmin(s2(b)))}")
    print(f"                    same likelihood: {abs(a.llf - b.llf) < 1e-3}; same index for 'calm':"
          f" {int(np.argmin(s2(a))) == int(np.argmin(s2(b)))}")

    P = np.asarray(a.regime_transition)[:, :, 0]
    fp = np.asarray(a.filtered_marginal_probabilities)
    pp = np.asarray(a.predicted_marginal_probabilities)
    err = np.abs(pp[1:] - fp[:-1] @ P.T).max()
    print(f"  transition mat  : regime_transition[:, :, 0] column sums = {np.round(P.sum(axis=0), 6)}"
          f" (COLUMN-stochastic: P[to, from])")
    print(f"                    max |predicted[t] - P @ filtered[t-1]| = {err:.1e}")

    logp = np.cumsum(r)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = _model(logp).fit(search_reps=5, rng=np.random.default_rng(1), disp=False)
    lab = np.asarray(pr.smoothed_marginal_probabilities).argmax(axis=1)
    acc_true = max((lab == s).mean(), (lab != s).mean())
    above = (logp > np.median(logp)).astype(int)
    acc_lvl = max((lab == above).mean(), (lab != above).mean())
    print(f"  fit on PRICES   : log-price fit converged={pr.mle_retvals['converged']}"
          f" in {time.time() - t0:.1f}s; expected durations = {np.round(pr.expected_durations, 0)} d")
    print(f"                    label agrees with TRUE regime {acc_true:.1%} (chance = 50%),"
          f" with 'price above its full-sample median' {acc_lvl:.1%}")


# ----------------------------------------------------------------------------- main ---------
if __name__ == "__main__":
    t_all = time.time()
    lo, hi = TEST_START, N
    print(f"statsmodels {statsmodels.__version__ if HAVE_SM else 'NOT INSTALLED'},"
          f" numpy {np.__version__}, pandas {pd.__version__}")
    print(f"DGP: {N} daily obs, calm mu={MU[0]} sig={SIG_CALM}, turbulent mu={MU[1]}"
          f" sig={SIG_CALM * MAIN_RATIO} (ratio {MAIN_RATIO}), p_stay={P_STAY} (expected durations"
          f" {1 / (1 - P_STAY[0]):.0f} and {1 / (1 - P_STAY[1]):.0f} d); seed={SEED}")
    print(f"Evaluation window: obs {lo}..{hi - 1} ({hi - lo} days); strategy = long when"
          f" P(calm) > 0.5 else flat; Sharpe = mean/sd*sqrt({PERIODS}), rf=0")

    r, s = simulate(N, SEED, MAIN_RATIO)
    tab, info = ladder(r, s, lo, hi)
    if HAVE_SM:
        chk = info["ref_check"]
        print(f"\nnumpy Hamilton filter / Kim smoother vs statsmodels at the fitted parameters:"
              f" max abs diff predicted {chk['predicted']:.1e}, filtered {chk['filtered']:.1e},"
              f" smoothed {chk['smoothed']:.1e}")
        print(f"\n=== 1. Strategy ladder (main fit converged={info['res'].mle_retvals['converged']},"
              f" llf per start={np.round(info['llfs'], 3)}, walk-forward refits={info['n_fits']}) ===")
    else:
        print("\n=== 1. Strategy ladder (statsmodels not installed: TRUE-parameter rows only) ===")
    cols = ["sharpe", "cagr", "maxdd", "in_mkt", "switches", "label_acc", "median_delay",
            "anticipated", "missed", "false_alarms"]
    print(tab[cols].to_string(float_format=lambda x: f"{x:8.3f}"))
    print("\n  uses:")
    for name, uses in tab["uses"].items():
        print(f"    {name:<28} {uses}")

    print("\n=== 2. The gaps (Sharpe, then max drawdown in percentage points) ===")
    smt, prt = tab.loc["smoothed @ TRUE params"], tab.loc["predicted @ TRUE params"]
    print(f"  at TRUE params, smoothed - predicted : Sharpe {smt.sharpe - prt.sharpe:+.3f},"
          f" maxDD {100 * (smt.maxdd - prt.maxdd):+.1f} pp  <- timing only")
    if HAVE_SM:
        sm, fi, pr_, wf = (tab.loc[k] for k in ("smoothed", "filtered (same day)", "predicted",
                                                  "walk-forward predicted"))
        print(f"  smoothed - predicted   : Sharpe {sm.sharpe - pr_.sharpe:+.3f},"
              f" maxDD {100 * (sm.maxdd - pr_.maxdd):+.1f} pp")
        print(f"  smoothed - filtered    : Sharpe {sm.sharpe - fi.sharpe:+.3f},"
              f" maxDD {100 * (sm.maxdd - fi.maxdd):+.1f} pp")
        print(f"  filtered - predicted   : Sharpe {fi.sharpe - pr_.sharpe:+.3f}  <- the one-day leak")
        print(f"  predicted - walk-fwd   : Sharpe {pr_.sharpe - wf.sharpe:+.3f}  <- in-sample parameters")
        print(f"  smoothed - walk-fwd    : Sharpe {sm.sharpe - wf.sharpe:+.3f},"
              f" maxDD {100 * (sm.maxdd - wf.maxdd):+.1f} pp  <- the whole look-ahead")
        print(f"  detection delay (median days after a true switch): smoothed {sm.median_delay:+.0f},"
              f" filtered {fi.median_delay:+.0f}, predicted {pr_.median_delay:+.0f},"
              f" walk-forward {wf.median_delay:+.0f}")
        print(f"  smoothed flagged the new regime BEFORE it started in {sm.anticipated:.0%} of"
              f" switches; filtered {fi.anticipated:.0%}, predicted {pr_.anticipated:.0%}")
        print(f"  label accuracy: smoothed {sm.label_acc:.3f}, filtered {fi.label_acc:.3f},"
              f" predicted {pr_.label_acc:.3f}, walk-forward {wf.label_acc:.3f}")

    print(f"\n=== 3. Sweep over regime separation and episode length, seeds {SWEEP_SEEDS},"
          f" full-sample parameters ===")
    sw = separation_sweep(lo, hi)
    spec = {"buy_hold": ("buy_hold", "median"), "oracle_lag1": ("oracle_lag1", "median"),
            "turb_share": ("turb_share", "median"),
            "gap_TRUE": ("gap_true_params", "median")}
    if HAVE_SM:
        spec.update(smoothed=("smoothed", "median"), filtered=("filtered", "median"),
                    predicted=("predicted", "median"), gap_med=("gap", "median"),
                    gap_min=("gap", "min"), gap_max=("gap", "max"), gap_fp=("gap_fp", "median"),
                    acc_sm=("acc_smoothed", "median"), acc_pred=("acc_predicted", "median"),
                    delay_filt=("delay_filtered", "median"), delay_sm=("delay_smoothed", "median"),
                    false_alarms=("false_alarms_filt", "median"), missed=("missed_filt", "median"))
    agg = sw.groupby("config", sort=False).agg(**spec)
    print(agg.to_string(float_format=lambda x: f"{x:6.2f}"))
    print("  (medians over seeds; gap = smoothed minus predicted Sharpe, gap_TRUE the same at the"
          " true parameters, gap_fp = filtered minus predicted; delays in days; false alarms and"
          " missed switches per 1764-day window)")

    if HAVE_SM:
        fit_pathologies(r, s, info["llfs"], sw)
    print(f"\ntotal runtime {time.time() - t_all:.1f}s")
