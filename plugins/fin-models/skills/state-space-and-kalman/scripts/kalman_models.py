#!/usr/bin/env python3
"""A time-varying beta by Kalman filter and RTS smoother, and why the smoothed beta looks ahead.

The regression y_t = alpha + beta_t x_t + e_t with beta_t a random walk (local level) or a
random walk with a drifting slope (local linear trend) is a linear Gaussian state-space model.
The Kalman filter hands back three estimates of beta_t for every date, and they are not
interchangeable:

    predicted   E[beta_t | y_1..y_{t-1}]   what you could have used to hedge day t
    filtered    E[beta_t | y_1..y_t]       needs day t's own return - fine at the close of t
    smoothed    E[beta_t | y_1..y_N]       a backward pass over the WHOLE sample

The smoothed path is the one every tutorial plots, because it is the cleanest. A hedge or a
spread built on it has been told what beta will be. This script measures, on a seeded series
with a known break in beta:

  1. that the numpy filter / smoother reproduce statsmodels' KalmanSmoother (when installed);
  2. the lead of the smoothed estimate over the filtered one - cross-correlation with the
     future truth, how far it had moved BEFORE the break, and how many days before / after
     the break each estimate crosses the midpoint;
  3. the Sharpe and residual variance of a beta hedge built on each estimate, against the
     oracle that knows the true beta. The smoothed and same-day-filtered hedges sit BELOW the
     residual-variance floor of the true beta. That cannot happen live; it is the tell.

The noise variance h is concentrated out of the likelihood, so the local-level fit is a
one-dimensional search over the signal-to-noise ratio q/h and the trend fit a two-dimensional
one - the standard trick, and the reason the demo runs in seconds rather than minutes.

Run:  python kalman_models.py      (numpy / scipy; statsmodels optional; fixed seeds; ~10 s)
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy import optimize

SEED = 0
N = 1500
BREAK_AT = 750
BETA_BEFORE, BETA_AFTER = 1.0, 0.5
ALPHA = 0.0003            # daily alpha of y, about 7.6 %/yr, the thing the hedge is meant to isolate
SIG_X = 0.010             # daily sd of the market return
SIG_EPS = 0.005           # idiosyncratic noise
SIG_BETA = 0.002          # random-walk innovation of beta per day (main path)
SWEEP_SIG_BETA = (0.0, 0.002, 0.005, 0.010)
SWITCH_EVERY = 60         # kind='switch': beta flips every 60 days, faster than the filter follows
SWITCH_PERIODS = (250, 120, 60, 30)
PERIODS = 252
MAX_LAG = 40
BREAK_WINDOW = 150        # +-days around the break used for the windowed cross-correlation
ROLL_WINDOW = 250
PRIOR_SCALE = 1e5         # prior variance of alpha and beta = PRIOR_SCALE * h (diffuse)
SLOPE_PRIOR_SCALE = 1.0   # prior variance of the slope = SLOPE_PRIOR_SCALE * h


# ----------------------------------------------------------------------------- data ---------
def simulate(n: int = N, seed: int = SEED, kind: str = "break", sig_beta: float = SIG_BETA,
             switch_every: int = SWITCH_EVERY) -> dict[str, np.ndarray]:
    """y_t = ALPHA + beta_t x_t + e_t, with `eps` (the idiosyncratic shock) returned too.

    kind='break':  beta steps from 1.0 to 0.5 at BREAK_AT, once.
    kind='drift':  beta declines linearly across the whole sample.
    kind='switch': beta alternates between 1.0 and 0.5 every `switch_every` days - the case
                   where the state moves faster than a causal filter can follow it.
    In every case beta also wanders as a random walk of daily sd `sig_beta` around that base.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "break":
        base = np.where(t < BREAK_AT, BETA_BEFORE, BETA_AFTER)
    elif kind == "drift":
        base = BETA_BEFORE + (BETA_AFTER - BETA_BEFORE) * t / (n - 1)
    elif kind == "switch":
        if switch_every < 1:
            raise ValueError("switch_every must be at least 1 day")
        base = np.where((t // switch_every) % 2 == 0, BETA_BEFORE, BETA_AFTER)
    else:
        raise ValueError("kind must be 'break', 'drift' or 'switch'")
    beta = base + np.cumsum(rng.normal(0.0, sig_beta, n))
    x = rng.normal(0.0, SIG_X, n)
    eps = rng.normal(0.0, SIG_EPS, n)
    return {"y": ALPHA + beta * x + eps, "x": x, "beta": beta, "eps": eps}


# ----------------------------------------------------------------------------- models -------
def local_level_system(x: np.ndarray, q_beta: float, h: float, intercept: bool = True) -> dict:
    """State (alpha, beta) with beta a random walk and alpha constant; Z_t = [1, x_t]."""
    x = np.asarray(x, dtype=float)
    if intercept:
        Z = np.column_stack([np.ones_like(x), x])
        T = np.eye(2)
        Q = np.diag([0.0, q_beta])
    else:
        Z, T, Q = x[:, None], np.eye(1), np.array([[q_beta]])
    return {"Z": Z, "T": T, "Q": Q, "H": float(h)}


def local_linear_trend_system(x: np.ndarray, q_beta: float, q_slope: float, h: float,
                              intercept: bool = True) -> dict:
    """State (alpha, beta, slope): beta_t = beta_{t-1} + slope_{t-1} + eta_t, slope a random walk."""
    x = np.asarray(x, dtype=float)
    if intercept:
        Z = np.column_stack([np.ones_like(x), x, np.zeros_like(x)])
        T = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        Q = np.diag([0.0, q_beta, q_slope])
    else:
        Z = np.column_stack([x, np.zeros_like(x)])
        T = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.diag([q_beta, q_slope])
    return {"Z": Z, "T": T, "Q": Q, "H": float(h)}


def kalman_filter(y: np.ndarray, Z: np.ndarray, T: np.ndarray, Q: np.ndarray, H: float,
                  a0: np.ndarray, P0: np.ndarray) -> dict:
    """Kalman filter for y_t = Z_t a_t + e_t (var H), a_t = T a_{t-1} + eta_t (cov Q).

    a0, P0 are the mean and covariance of a_1 BEFORE seeing y_1 - the same convention as
    statsmodels' `initialize_known`. Returns predicted a_{t|t-1}, filtered a_{t|t}, their
    covariances, innovations v_t, their variances F_t, and the exact Gaussian log-likelihood
    from the prediction-error decomposition.
    """
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    T = np.asarray(T, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n, k = Z.shape
    if y.shape != (n,):
        raise ValueError("y must be 1-D with one row of Z per observation")
    a = np.asarray(a0, dtype=float).reshape(k).copy()
    P = np.asarray(P0, dtype=float).reshape(k, k).copy()
    identity_T = bool(np.array_equal(T, np.eye(k)))
    pred = np.empty((n, k))
    pred_cov = np.empty((n, k, k))
    filt = np.empty((n, k))
    filt_cov = np.empty((n, k, k))
    v = np.empty(n)
    F = np.empty(n)
    Tt = T.T
    for t in range(n):
        pred[t] = a
        pred_cov[t] = P
        z = Z[t]
        Pz = P @ z
        v[t] = y[t] - z @ a
        F[t] = z @ Pz + H
        K = Pz / F[t]
        a = a + K * v[t]
        P = P - np.outer(K, Pz)
        filt[t] = a
        filt_cov[t] = P
        if identity_T:
            P = P + Q
        else:
            a = T @ a
            P = T @ P @ Tt + Q
    llf = float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(F) + v ** 2 / F))
    return {"predicted": pred, "predicted_cov": pred_cov, "filtered": filt,
            "filtered_cov": filt_cov, "v": v, "F": F, "llf": llf}


def rts_smoother(T: np.ndarray, kf: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel: a_{t|N} = a_{t|t} + J_t (a_{t+1|N} - a_{t+1|t}), a BACKWARD pass."""
    pred, pred_cov, filt, filt_cov = (kf[k] for k in ("predicted", "predicted_cov", "filtered",
                                                        "filtered_cov"))
    T = np.asarray(T, dtype=float)
    n, k = filt.shape
    sm = np.empty_like(filt)
    sm_cov = np.empty_like(filt_cov)
    sm[-1], sm_cov[-1] = filt[-1], filt_cov[-1]
    for t in range(n - 2, -1, -1):
        J = filt_cov[t] @ T.T @ np.linalg.pinv(pred_cov[t + 1])
        sm[t] = filt[t] + J @ (sm[t + 1] - pred[t + 1])
        sm_cov[t] = filt_cov[t] + J @ (sm_cov[t + 1] - pred_cov[t + 1]) @ J.T
    return sm, sm_cov


def concentrated_llf(kf_unit: dict) -> tuple[float, float]:
    """Given a filter run with H = 1 and Q, P0 expressed relative to h, the MLE of h is
    mean(v^2 / F) and the maximised log-likelihood is -0.5 (n log 2pi + n log h + sum log F + n)."""
    v, F = kf_unit["v"], kf_unit["F"]
    n = len(v)
    h = float(np.mean(v ** 2 / F))
    llf = -0.5 * (n * math.log(2.0 * math.pi) + n * math.log(h) + float(np.sum(np.log(F))) + n)
    return h, llf


def _prior(k: int, beta_index: int, slope_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    a0 = np.zeros(k)
    a0[beta_index] = 1.0
    P0 = np.eye(k) * PRIOR_SCALE
    if slope_index is not None:
        P0[slope_index, slope_index] = SLOPE_PRIOR_SCALE
    return a0, P0


def _finish(y, x, system_fn, ratios, h, a0, P0star, beta_index, extra) -> dict:
    s = system_fn(x, *[r * h for r in ratios], h)
    P0 = P0star * h
    kf = kalman_filter(y, s["Z"], s["T"], s["Q"], s["H"], a0, P0)
    sm, sm_cov = rts_smoother(s["T"], kf)
    out = {"h": float(h), "llf": kf["llf"], "system": s, "a0": a0, "P0": P0, "kf": kf,
           "smoothed": sm, "smoothed_cov": sm_cov, "beta_index": beta_index}
    out.update(extra)
    return out


LOG_Q_LO, LOG_Q_HI = math.log(1e-8), math.log(1e3)
N_GRID = 41


def fit_local_level(y: np.ndarray, x: np.ndarray, intercept: bool = True,
                    search: str = "grid") -> dict:
    """Maximum likelihood for (q_beta, h) over log(q_beta / h), h concentrated out.

    🚨 The profile likelihood in q is NOT unimodal. Whenever beta moves in steps there is a
    second, flat optimum at q -> 0 - "beta is constant" - and a bounded scalar search started
    on the whole interval can settle there and report `success=True` with a straight-line beta.
    `search="grid"` (the default) evaluates a coarse log-spaced grid first and refines around
    the best cell; `search="bounded"` is the naive scipy call, kept so the demo can show the
    failure. The demo's section 5 measures a 74.6-nat gap between the two on one series.
    """
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    k, bi = (2, 1) if intercept else (1, 0)
    a0, P0star = _prior(k, bi)

    def nll(log_ratio):
        s = local_level_system(x, math.exp(float(log_ratio)), 1.0, intercept)
        return -concentrated_llf(kalman_filter(y, s["Z"], s["T"], s["Q"], 1.0, a0, P0star))[1]

    if search == "bounded":
        res = optimize.minimize_scalar(nll, bounds=(LOG_Q_LO, LOG_Q_HI), method="bounded",
                                       options={"xatol": 1e-5})
        best_x, best_f, n_evals, ok = float(res.x), float(res.fun), int(res.nfev), bool(res.success)
    elif search == "grid":
        grid = np.linspace(LOG_Q_LO, LOG_Q_HI, N_GRID)
        vals = np.array([nll(g) for g in grid])
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, N_GRID - 1)]
        res = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-5})
        best_x, best_f = (float(res.x), float(res.fun)) if res.fun <= vals[i] else \
                         (float(grid[i]), float(vals[i]))
        n_evals, ok = N_GRID + int(res.nfev), bool(res.success)
    else:
        raise ValueError("search must be 'grid' or 'bounded'")

    ratio = math.exp(best_x)
    s = local_level_system(x, ratio, 1.0, intercept)
    h, _ = concentrated_llf(kalman_filter(y, s["Z"], s["T"], s["Q"], 1.0, a0, P0star))
    return _finish(y, x, lambda xx, q, hh: local_level_system(xx, q, hh, intercept), (ratio,),
                   h, a0, P0star, bi, {"q_beta": ratio * h, "converged": ok,
                                       "n_evals": n_evals, "profile_llf": -best_f,
                                       "search": search})


TREND_STARTS = ((1e-1, 1e-5), (1e-3, 1e-3), (1e1, 1e-7), (1e-6, 1e-1))


def fit_local_linear_trend(y: np.ndarray, x: np.ndarray, intercept: bool = True) -> dict:
    """Maximum likelihood for (q_beta, q_slope, h): L-BFGS-B on the two ratios to h, from four
    starts. One start is not enough for the same reason the 1-D search needs a grid."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    k, bi = (3, 1) if intercept else (2, 0)
    a0, P0star = _prior(k, bi, slope_index=bi + 1)

    def nll(theta):
        s = local_linear_trend_system(x, math.exp(theta[0]), math.exp(theta[1]), 1.0, intercept)
        return -concentrated_llf(kalman_filter(y, s["Z"], s["T"], s["Q"], 1.0, a0, P0star))[1]

    lo, hi = math.log(1e-10), math.log(1e3)
    best, n_evals = None, 0
    for start in TREND_STARTS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(nll, np.log(start), method="L-BFGS-B",
                                    bounds=[(lo, hi), (lo, hi)], options={"maxiter": 200})
        n_evals += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    rq, rs = np.exp(best.x)
    s = local_linear_trend_system(x, rq, rs, 1.0, intercept)
    h, _ = concentrated_llf(kalman_filter(y, s["Z"], s["T"], s["Q"], 1.0, a0, P0star))
    return _finish(y, x, lambda xx, q, qs, hh: local_linear_trend_system(xx, q, qs, hh, intercept),
                   (rq, rs), h, a0, P0star, bi,
                   {"q_beta": rq * h, "q_slope": rs * h, "converged": bool(best.success),
                    "n_evals": n_evals, "profile_llf": float(-best.fun)})


def beta_series(fit: dict) -> dict[str, np.ndarray]:
    """The three beta estimates aligned to t. 'predicted' is what a hedge for day t can use."""
    bi = fit["beta_index"]
    return {"predicted": fit["kf"]["predicted"][:, bi], "filtered": fit["kf"]["filtered"][:, bi],
            "smoothed": fit["smoothed"][:, bi]}


# ----------------------------------------------------------------------------- statsmodels --
def statsmodels_check(y: np.ndarray, system: dict, a0: np.ndarray, P0: np.ndarray) -> dict | None:
    """Run statsmodels' KalmanSmoother on the same system; max abs differences. None if absent."""
    try:
        import statsmodels
        from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
    except ImportError:
        return None
    Z, T, Q, H = system["Z"], system["T"], system["Q"], system["H"]
    n, k = Z.shape
    ks = KalmanSmoother(k_endog=1, k_states=k, k_posdef=k)
    ks.bind(np.asarray(y, dtype=float)[None, :])
    ks["design"] = np.ascontiguousarray(Z.T).reshape(1, k, n)
    ks["transition"] = T
    ks["selection"] = np.eye(k)
    ks["state_cov"] = Q
    ks["obs_cov"] = np.array([[H]])
    ks.initialize_known(np.asarray(a0, dtype=float), np.asarray(P0, dtype=float))
    res = ks.smooth()
    kf = kalman_filter(y, Z, T, Q, H, a0, P0)
    sm, _ = rts_smoother(T, kf)
    return {"version": statsmodels.__version__,
            "predicted": float(np.abs(res.predicted_state[:, :n].T - kf["predicted"]).max()),
            "filtered": float(np.abs(res.filtered_state.T - kf["filtered"]).max()),
            "smoothed": float(np.abs(res.smoothed_state.T - sm).max()),
            "llf": float(abs(res.llf - kf["llf"]))}


def recursive_ls_check(y: np.ndarray, x: np.ndarray) -> dict | None:
    """statsmodels' RecursiveLS is the OTHER time-varying-beta tool, and its `smoothed` series
    is not time-varying at all: the model holds the coefficients FIXED, so the smoothed
    estimate of a constant is the full-sample OLS answer painted across every date. Returns
    the spread of each series and the distance from the full-sample slope. None without
    statsmodels.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.recursive_ls import RecursiveLS
    except ImportError:
        return None
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RecursiveLS(y, sm.add_constant(x)).fit()
    filt = np.asarray(res.recursive_coefficients.filtered, dtype=float)[1]
    smoo = np.asarray(res.recursive_coefficients.smoothed, dtype=float)[1]
    xc = x - x.mean()
    ols = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    return {"filtered": filt, "smoothed": smoo, "ols": ols,
            "smoothed_ptp": float(np.ptp(smoo)), "filtered_ptp": float(np.ptp(filt[ROLL_WINDOW:])),
            "smoothed_minus_ols": float(np.abs(smoo - ols).max()),
            "n_predicted": int(len(res.states.predicted)), "nobs": int(res.nobs)}


# ----------------------------------------------------------------------------- metrics ------
def lead_lag_profile(est: np.ndarray, truth: np.ndarray, max_lag: int = MAX_LAG,
                     window: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """corr(est_t, truth_{t+k}) for k in -max_lag..max_lag. A positive best k means the estimate
    at t is aligned with the truth k days LATER - it knew where beta was going; a negative one
    means it is that many days behind.

    `window` scores only t in [lo, hi). Over a long flat stretch every lag correlates about
    equally and the argmax is noise, so the honest measurement is a window around the move.
    """
    est, truth = np.asarray(est, dtype=float), np.asarray(truth, dtype=float)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(len(lags))
    n = len(est)
    lo, hi = (0, n) if window is None else window
    lo, hi = max(lo, max_lag), min(hi, n - max_lag)
    if hi - lo < 3 * max_lag:
        raise ValueError("window too short for this max_lag")
    idx = np.arange(lo, hi)
    for i, k in enumerate(lags):
        out[i] = np.corrcoef(est[idx], truth[idx + k])[0, 1]
    return lags, out


def best_alignment(est: np.ndarray, truth: np.ndarray, max_lag: int = MAX_LAG,
                   window: tuple[int, int] | None = None) -> int:
    lags, c = lead_lag_profile(est, truth, max_lag, window)
    return int(lags[int(np.argmax(c))])


def noise_absorption(x: np.ndarray, eps: np.ndarray, beta_true: np.ndarray,
                     beta_used: np.ndarray) -> dict[str, float]:
    """Why a look-ahead hedge beats the true-beta floor, as an exact variance decomposition.

    The hedged return is  resid_t = alpha + eps_t + (beta_true_t - beta_used_t) x_t, so

        var(resid) / var(eps) - 1 = var(mis) / var(eps)  +  2 cov(mis, eps) / var(eps)
                                    ------ mis-hedge ---    ---- noise absorbed ----

    with mis_t = (beta_true_t - beta_used_t) x_t. The first term is non-negative and is the
    price of not knowing beta. The second is zero for any beta chosen without seeing eps_t,
    and NEGATIVE for one fitted to y_t: the estimate has regressed away part of the day's own
    idiosyncratic return, which is what pushes the residual below a floor set by the truth.
    """
    x, eps = np.asarray(x, dtype=float), np.asarray(eps, dtype=float)
    mis = (np.asarray(beta_true, dtype=float) - np.asarray(beta_used, dtype=float)) * x
    ve = float(np.var(eps, ddof=1))
    cross = 2.0 * float(np.cov(mis, eps, ddof=1)[0, 1])
    rho = float(np.corrcoef(mis, eps)[0, 1])
    n = len(eps)
    return {"mishedge_term": float(np.var(mis, ddof=1)) / ve, "noise_term": cross / ve,
            "total": (float(np.var(mis, ddof=1)) + cross) / ve,
            "corr_mis_eps": rho, "n": n,
            "t_stat": rho * math.sqrt((n - 2) / max(1.0 - rho ** 2, 1e-12))}


def break_crossing(est: np.ndarray, break_at: int, before: float, after: float,
                   search: int = 150) -> int | None:
    """Days relative to the break at which the estimate first crosses the midpoint between the
    old and new level, looking from `search` days before. Negative = crossed BEFORE the break."""
    mid = 0.5 * (before + after)
    lo, hi = max(0, break_at - search), min(len(est), break_at + search)
    seg = np.asarray(est[lo:hi], dtype=float)
    crossed = seg < mid if after < before else seg > mid
    if not crossed.any():
        return None
    return int(np.argmax(crossed)) + lo - break_at


def share_moved_before(est: np.ndarray, break_at: int, before: float, after: float) -> float:
    """How much of the step the estimate had already taken on the day BEFORE the break."""
    return float((before - est[break_at - 1]) / (before - after))


def hedge_stats(y: np.ndarray, x: np.ndarray, beta_used: np.ndarray, periods: int = PERIODS
                ) -> dict[str, float]:
    """Hedged return h_t = y_t - beta_used_t * x_t. Sharpe = mean/sd*sqrt(periods), rf = 0."""
    h = np.asarray(y, dtype=float) - np.asarray(beta_used, dtype=float) * np.asarray(x, dtype=float)
    sd = h.std(ddof=1)
    return {"sharpe": float(h.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan"),
            "resid_var": float(h.var(ddof=1)), "mean": float(h.mean())}


def rolling_ols_beta(y: np.ndarray, x: np.ndarray, window: int = ROLL_WINDOW) -> np.ndarray:
    """OLS slope over the trailing window ending at t-1 (NaN until the window is full)."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    for t in range(window, n):
        xs, ys = x[t - window:t], y[t - window:t]
        xc = xs - xs.mean()
        out[t] = float(np.dot(xc, ys - ys.mean()) / np.dot(xc, xc))
    return out


def ladder(d: dict[str, np.ndarray], fit: dict, start: int = ROLL_WINDOW) -> dict[str, dict]:
    """Hedge outcomes over t >= start for every beta source, oracle first."""
    y, x, beta = d["y"], d["x"], d["beta"]
    b = beta_series(fit)
    xc = x - x.mean()
    ols_full = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    sources = {
        "oracle: true beta_t": beta,
        "smoothed (t|N)": b["smoothed"],
        "filtered (t|t), same day": b["filtered"],
        "predicted (t|t-1), causal": b["predicted"],
        "full-sample OLS (constant)": np.full(len(y), ols_full),
        f"rolling {ROLL_WINDOW}-day OLS, causal": rolling_ols_beta(y, x, ROLL_WINDOW),
    }
    out = {}
    for name, bu in sources.items():
        st = hedge_stats(y[start:], x[start:], bu[start:])
        st["beta_rmse"] = float(np.sqrt(np.mean((bu[start:] - beta[start:]) ** 2)))
        out[name] = st
    return out


def sweep(sig_betas=SWEEP_SIG_BETA, seed: int = SEED, start: int = ROLL_WINDOW) -> list[dict]:
    """The same measurements as beta moves faster: crossing days, share moved before the break,
    residual variance relative to the true-beta floor, and the smoothed minus causal Sharpe."""
    rows = []
    for sb in sig_betas:
        d = simulate(N, seed, "break", sig_beta=sb)
        fit = fit_local_level(d["y"], d["x"])
        b = beta_series(fit)
        lad = ladder(d, fit, start)
        floor = lad["oracle: true beta_t"]["resid_var"]
        rows.append({"sig_beta": sb, "q_beta_hat": fit["q_beta"],
                     "cross_smoothed": break_crossing(b["smoothed"], BREAK_AT, BETA_BEFORE, BETA_AFTER),
                     "cross_predicted": break_crossing(b["predicted"], BREAK_AT, BETA_BEFORE, BETA_AFTER),
                     "moved_smoothed": share_moved_before(b["smoothed"], BREAK_AT, BETA_BEFORE, BETA_AFTER),
                     "moved_predicted": share_moved_before(b["predicted"], BREAK_AT, BETA_BEFORE, BETA_AFTER),
                     "resid_smoothed_vs_floor": lad["smoothed (t|N)"]["resid_var"] / floor - 1.0,
                     "resid_filtered_vs_floor": lad["filtered (t|t), same day"]["resid_var"] / floor - 1.0,
                     "resid_predicted_vs_floor": lad["predicted (t|t-1), causal"]["resid_var"] / floor - 1.0,
                     "sharpe_oracle": lad["oracle: true beta_t"]["sharpe"],
                     "sharpe_smoothed": lad["smoothed (t|N)"]["sharpe"],
                     "sharpe_predicted": lad["predicted (t|t-1), causal"]["sharpe"],
                     "sharpe_rolling": lad[f"rolling {ROLL_WINDOW}-day OLS, causal"]["sharpe"]})
    return rows


SMOOTHED, FILTERED, PREDICTED = ("smoothed (t|N)", "filtered (t|t), same day",
                                 "predicted (t|t-1), causal")
ORACLE = "oracle: true beta_t"
ROLLING = f"rolling {ROLL_WINDOW}-day OLS, causal"


def switch_sweep(periods=SWITCH_PERIODS, seed: int = SEED, start: int = ROLL_WINDOW) -> list[dict]:
    """The look-ahead is worth little when beta breaks once and a lot when it breaks constantly.

    beta flips between 1.0 and 0.5 every `p` days; the causal filter needs ~20 days to follow a
    flip, so as p falls it spends most of the sample mid-transition while the smoother, reading
    backwards, does not.
    """
    rows = []
    for p in periods:
        d = simulate(N, seed, "switch", switch_every=p)
        fit = fit_local_level(d["y"], d["x"])
        lad = ladder(d, fit, start)
        floor = lad[ORACLE]["resid_var"]
        row = {"switch_every": p, "q_beta_hat": fit["q_beta"],
               "resid_sm": lad[SMOOTHED]["resid_var"] / floor - 1.0,
               "resid_pred": lad[PREDICTED]["resid_var"] / floor - 1.0,
               "rmse_sm": lad[SMOOTHED]["beta_rmse"], "rmse_pred": lad[PREDICTED]["beta_rmse"]}
        for key, name in (("oracle", ORACLE), ("sm", SMOOTHED), ("filt", FILTERED),
                          ("pred", PREDICTED), ("roll", ROLLING)):
            row[f"sharpe_{key}"] = lad[name]["sharpe"]
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- demo --------
if __name__ == "__main__":
    t_all = time.time()
    d = simulate(N, SEED, "break")
    print(f"DGP: y = {ALPHA} + beta_t x + e, x ~ N(0, {SIG_X}), e ~ N(0, {SIG_EPS}); beta random walk"
          f" (sd {SIG_BETA}/day) around {BETA_BEFORE} until t={BREAK_AT}, then {BETA_AFTER};"
          f" N={N}, seed={SEED}")

    fit = fit_local_level(d["y"], d["x"])
    print(f"\n=== 1. Local-level fit: q_beta={fit['q_beta']:.2e} (the DGP's random-walk part is"
          f" {SIG_BETA ** 2:.2e}; the step at t={BREAK_AT} is absorbed into q), h={fit['h']:.2e}"
          f" (true {SIG_EPS ** 2:.2e}), llf={fit['llf']:.2f}, converged={fit['converged']} in"
          f" {fit['n_evals']} likelihood evaluations ===")
    chk = statsmodels_check(d["y"], fit["system"], fit["a0"], fit["P0"])
    if chk is not None:
        print(f"  statsmodels {chk['version']} KalmanSmoother on the same system: max abs diff"
              f" predicted {chk['predicted']:.1e}, filtered {chk['filtered']:.1e},"
              f" smoothed {chk['smoothed']:.1e}, |llf diff| {chk['llf']:.1e}")
    else:
        print("  statsmodels not installed: numpy recursions unchecked on this machine")

    b = beta_series(fit)
    win = (BREAK_AT - BREAK_WINDOW, BREAK_AT + BREAK_WINDOW)
    print(f"\n=== 2. Which estimate knows the future? corr(beta_est[t], beta_true[t+k]) over the"
          f" {2 * BREAK_WINDOW} days around the break, k in +-{MAX_LAG} ===")
    print("  (positive best k = the estimate at t matches the truth k days LATER: it leads)")
    for name in ("smoothed", "filtered", "predicted"):
        lags, c = lead_lag_profile(b[name], d["beta"], window=win)
        k = int(lags[int(np.argmax(c))])
        c10 = c[list(lags).index(10)]
        c_10 = c[list(lags).index(-10)]
        cross = break_crossing(b[name], BREAK_AT, BETA_BEFORE, BETA_AFTER)
        moved = share_moved_before(b[name], BREAK_AT, BETA_BEFORE, BETA_AFTER)
        print(f"  {name:<10} best k {k:+3d} d (corr {c.max():.3f}); corr at k=+10 {c10:.3f},"
              f" at k=-10 {c_10:.3f}; crosses the midpoint {cross:+d} d from the break;"
              f" on the day BEFORE the break it had already moved {moved:.0%} of the step")

    lad = ladder(d, fit)
    print(f"\n=== 3. A beta hedge on each estimate, t >= {ROLL_WINDOW} (Sharpe of y - beta x,"
          f" rf=0; alpha / noise implies {ALPHA / SIG_EPS * math.sqrt(PERIODS):.2f} in expectation) ===")
    for name, st in lad.items():
        print(f"  {name:<30} Sharpe {st['sharpe']:6.2f}   resid var {st['resid_var']:.3e}"
              f"   beta RMSE {st['beta_rmse']:.4f}")
    floor = lad["oracle: true beta_t"]["resid_var"]
    sm, fi, pr = (lad[k] for k in ("smoothed (t|N)", "filtered (t|t), same day",
                                   "predicted (t|t-1), causal"))
    print(f"  residual variance relative to the true-beta floor: smoothed {sm['resid_var'] / floor - 1:+.1%},"
          f" filtered same-day {fi['resid_var'] / floor - 1:+.1%}, predicted {pr['resid_var'] / floor - 1:+.1%}")
    print(f"  smoothed minus causal Sharpe: {sm['sharpe'] - pr['sharpe']:+.2f}")
    print("  where the deficit comes from - var(resid)/var(eps) - 1 = mis-hedge + noise absorbed:")
    sl = slice(ROLL_WINDOW, None)
    abs_t = {}
    for name, bu in (("smoothed (t|N)", b["smoothed"]), ("filtered (t|t)", b["filtered"]),
                     ("predicted (t|t-1)", b["predicted"]),
                     (f"rolling {ROLL_WINDOW}d OLS", rolling_ols_beta(d["y"], d["x"], ROLL_WINDOW))):
        na = noise_absorption(d["x"][sl], d["eps"][sl], d["beta"][sl], bu[sl])
        abs_t[name] = na["t_stat"]
        print(f"     {name:<20} mis-hedge {na['mishedge_term']:+.2%}   noise absorbed"
              f" {na['noise_term']:+.2%}   total {na['total']:+.2%}"
              f"   corr(mis-hedge, eps) {na['corr_mis_eps']:+.3f} (t {na['t_stat']:+.1f})")
    causal = max(abs(abs_t["predicted (t|t-1)"]), abs(abs_t[f"rolling {ROLL_WINDOW}d OLS"]))
    peeking = max(abs(abs_t["smoothed (t|N)"]), abs(abs_t["filtered (t|t)"]))
    print(f"     the middle column is zero in expectation for any beta chosen without seeing"
          f" eps_t: the causal rows peak at |t| = {causal:.1f}, the two that used y_t reach"
          f" |t| = {peeking:.1f}.")

    print(f"\n=== 3b. The same, as beta moves faster (sd of the daily beta innovation; seed {SEED}) ===")
    print("  sig_beta  q_hat     cross sm/pred (d)  moved before break sm/pred   resid vs floor sm/filt/pred"
          "   Sharpe oracle/sm/pred/rolling")
    for r in sweep():
        print(f"  {r['sig_beta']:<8.3f}  {r['q_beta_hat']:.1e}   {r['cross_smoothed']:+4d} / {r['cross_predicted']:+4d}"
              f"        {r['moved_smoothed']:5.0%} / {r['moved_predicted']:5.0%}"
              f"          {r['resid_smoothed_vs_floor']:+7.2%} / {r['resid_filtered_vs_floor']:+7.2%} /"
              f" {r['resid_predicted_vs_floor']:+7.2%}   {r['sharpe_oracle']:.2f} / {r['sharpe_smoothed']:.2f}"
              f" / {r['sharpe_predicted']:.2f} / {r['sharpe_rolling']:.2f}")

    llt = fit_local_linear_trend(d["y"], d["x"])
    d2 = simulate(N, SEED + 1, "drift")
    fit_ll2 = fit_local_level(d2["y"], d2["x"])
    llt2 = fit_local_linear_trend(d2["y"], d2["x"])
    print("\n=== 4. Local level vs local linear trend (RMSE of the causal predicted beta, t >="
          f" {ROLL_WINDOW}; the likelihoods are not compared - the slope state carries its own"
          " prior, so the fits are not nested) ===")
    for label, dd, f_ll, f_llt in (("break DGP", d, fit, llt), ("linear-drift DGP", d2, fit_ll2, llt2)):
        e_ll = np.sqrt(np.mean((beta_series(f_ll)["predicted"][ROLL_WINDOW:] - dd["beta"][ROLL_WINDOW:]) ** 2))
        e_llt = np.sqrt(np.mean((beta_series(f_llt)["predicted"][ROLL_WINDOW:] - dd["beta"][ROLL_WINDOW:]) ** 2))
        print(f"  {label:<18} local level {e_ll:.4f}   local linear trend {e_llt:.4f}"
              f"   (trend fit: q_slope={f_llt['q_slope']:.1e}, i.e. a constant drift in beta;"
              f" {f_llt['n_evals']} evaluations)")
    chk2 = statsmodels_check(d2["y"], llt2["system"], llt2["a0"], llt2["P0"])
    if chk2 is not None:
        print(f"  statsmodels check on the 3-state trend model: predicted {chk2['predicted']:.1e},"
              f" filtered {chk2['filtered']:.1e}, smoothed {chk2['smoothed']:.1e}, |llf| {chk2['llf']:.1e}")

    rls = recursive_ls_check(d["y"], d["x"])
    print("\n=== 4b. statsmodels RecursiveLS: its 'smoothed' coefficients are not time-varying ===")
    if rls is None:
        print("  statsmodels not installed; skipped")
    else:
        print(f"  recursive_coefficients.filtered: spread {rls['filtered_ptp']:.4f} over t >="
              f" {ROLL_WINDOW} - a real expanding-window estimate")
        print(f"  recursive_coefficients.smoothed: spread {rls['smoothed_ptp']:.1e}, and"
              f" max |smoothed - full-sample OLS| = {rls['smoothed_minus_ols']:.1e}. The model"
              f" holds beta FIXED, so smoothing a constant returns the whole sample's answer,"
              f" once per date.")
        print(f"  and res.states.predicted has {rls['n_predicted']} rows for {rls['nobs']}"
              f" observations - the extra one is the forecast past the end of the sample.")

    d_sw = simulate(N, SEED, "switch", switch_every=SWITCH_EVERY)
    naive = fit_local_level(d_sw["y"], d_sw["x"], search="bounded")
    good = fit_local_level(d_sw["y"], d_sw["x"], search="grid")
    print(f"\n=== 5. The profile likelihood in q is not unimodal (beta flipping every"
          f" {SWITCH_EVERY} days, seed {SEED}) ===")
    for label, f in (("scipy bounded search", naive), ("coarse grid + refine", good)):
        pb = beta_series(f)["predicted"][ROLL_WINDOW:]
        rmse = float(np.sqrt(np.mean((pb - d_sw["beta"][ROLL_WINDOW:]) ** 2)))
        print(f"  {label:<22} q_beta={f['q_beta']:.2e}  llf={f['profile_llf']:.2f}"
              f"  converged={f['converged']}  {f['n_evals']:>3} evaluations;"
              f"  predicted beta sd {pb.std(ddof=1):.4f} (true {d_sw['beta'][ROLL_WINDOW:].std(ddof=1):.4f}),"
              f" RMSE {rmse:.4f}")
    print(f"  the naive search reports success on a likelihood {good['profile_llf'] - naive['profile_llf']:.1f}"
          f" nats worse, with q_beta {naive['q_beta'] / good['q_beta']:.1e} x the right value: a beta"
          f" that has stopped tracking. Nothing warns, and `converged` is True either way.")

    print(f"\n=== 5b. When the look-ahead actually pays: beta flips 1.0 <-> {BETA_AFTER} every p days"
          f" (seed {SEED}, t >= {ROLL_WINDOW}) ===")
    print("  p (days)  q_hat     beta RMSE sm/pred   resid vs floor sm/pred   Sharpe oracle/sm/filt/pred/roll")
    for r in switch_sweep():
        print(f"  {r['switch_every']:>8}  {r['q_beta_hat']:.1e}   {r['rmse_sm']:.4f} / {r['rmse_pred']:.4f}"
              f"      {r['resid_sm']:+7.2%} / {r['resid_pred']:+7.2%}      {r['sharpe_oracle']:.2f}"
              f" / {r['sharpe_sm']:.2f} / {r['sharpe_filt']:.2f} / {r['sharpe_pred']:.2f}"
              f" / {r['sharpe_roll']:.2f}")

    print("\nRule: hedge and trade on the PREDICTED beta; plot the smoothed one if you like, but a"
          " backtest on it has read the future.")
    print(f"total runtime {time.time() - t_all:.1f}s")
