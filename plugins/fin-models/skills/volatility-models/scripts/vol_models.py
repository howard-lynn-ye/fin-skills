#!/usr/bin/env python3
"""GARCH(1,1) by hand and checked against arch; range-based realized variance; HAR-RV.

Three things a volatility model can get silently wrong, each measured here on seeded data:

  1. SCALE. `arch` was written for returns in PERCENT. Hand it decimals and it warns
     (`DataScaleWarning`, verified in arch/univariate/base.py::_check_scale) but by default
     does NOT rescale: the fit proceeds on numbers a hundred times too small. Pass
     `rescale=True` and it multiplies y by a power of ten and reports the factor in
     `res.scale`; every conditional variance and forecast then comes back in the rescaled
     units and has to be divided out again. Section 2 measures what each path returns.
  2. RANGE ESTIMATORS. Parkinson, Garman-Klass, Rogers-Satchell and Yang-Zhang use the day's
     high and low and are several times more efficient than close-to-close - on the process
     they assume. Two of them assume zero drift, three of them assume no overnight gap, and
     all of them assume the high and low are observed continuously. Section 3 measures the
     efficiency AND the bias on simulated GBM with a known variance.
  3. HAR-RV (Corsi 2009): an OLS on the daily, 5-day and 22-day averages of realized variance.
     It is a strong forecasting baseline; the script fits it, checks the coefficients against
     arch's HARX, and scores it out of sample against the naive forecast.

The GARCH(1,1) estimator is a NumPy/SciPy re-implementation of what arch does - the same
backcast (0.94-weighted mean of the first 75 squared residuals), the same recursion, the same
Gaussian likelihood, the same SLSQP optimiser - so the skill works without arch and the
comparison, when arch is importable, is a check on the implementation rather than an
assumption about it.

Run:  python vol_models.py        (numpy / scipy; arch optional; fixed seeds; ~5-10 s)
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy import optimize, signal

SEED = 0
PERIODS = 252
LN2 = math.log(2.0)

# GARCH(1,1) data-generating process, in PERCENT units: unconditional variance
# omega / (1 - alpha - beta) = 1.0, i.e. a daily vol of 1 % (~15.9 % annualised).
TRUE_GARCH = {"mu": 0.03, "omega": 0.02, "alpha": 0.08, "beta": 0.90}
N_GARCH = 3000
GARCH_NAMES = ("mu", "omega", "alpha", "beta")

# Simulated OHLC for the range estimators: 50 years of 21-day windows.
GBM_DAYS = 12_600
STEPS_PER_DAY = 390
WINDOW = 21
ANNUAL_VOL = 0.20
SCENARIOS = (
    # label, annual drift, share of the daily variance that happens overnight
    ("zero drift, no overnight gap", 0.0, 0.0),
    ("drift 50 %/yr, no overnight gap", 0.5, 0.0),
    ("drift 200 %/yr, no overnight gap", 2.0, 0.0),
    ("zero drift, 30 % of variance overnight", 0.0, 0.3),
)
DISC_DAYS = 4200                      # 200 windows for the discretisation experiment
DISC_STEPS = (26, 78, 390, 3900)      # 15-minute, 5-minute, 1-minute, 6-second observation

# HAR-RV: a log-volatility AR(1) with 78 five-minute returns per day.
HAR_DAYS = 3000
HAR_STEPS = 78
HAR_TRAIN = 2000
HAR_LAGS = (1, 5, 22)


# ----------------------------------------------------------------------------- GARCH -------
def simulate_garch11(n: int, mu: float, omega: float, alpha: float, beta: float,
                     seed: int = SEED, burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian GARCH(1,1) returns and the true conditional variance, after `burn` warm-up draws."""
    if not (omega > 0 and alpha >= 0 and beta >= 0 and alpha + beta < 1):
        raise ValueError("need omega > 0, alpha, beta >= 0 and alpha + beta < 1")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    s2 = np.empty(n + burn)
    e = np.empty(n + burn)
    s2[0] = omega / (1.0 - alpha - beta)
    e[0] = math.sqrt(s2[0]) * z[0]
    for t in range(1, n + burn):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = math.sqrt(s2[t]) * z[t]
    return mu + e[burn:], s2[burn:]


def arch_backcast(resids: np.ndarray) -> float:
    """arch's VolatilityProcess.backcast: 0.94^i-weighted mean of the first min(75, n) e^2."""
    resids = np.asarray(resids, dtype=float)
    tau = min(75, resids.shape[0])
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(resids[:tau] ** 2 * w))


def garch11_variance(resids: np.ndarray, omega: float, alpha: float, beta: float,
                     backcast: float | None = None) -> np.ndarray:
    """arch's GARCH recursion for p = q = 1, o = 0 (recursions_python.py::garch_recursion_python):

        sigma2[0] = omega + alpha * backcast + beta * backcast
        sigma2[t] = omega + alpha * e[t-1]^2 + beta * sigma2[t-1]

    Vectorised as an IIR filter; `test_models_vol_models.py` checks it against the loop.
    """
    resids = np.asarray(resids, dtype=float)
    if backcast is None:
        backcast = arch_backcast(resids)
    e2 = resids ** 2
    x = omega + alpha * np.r_[backcast, e2[:-1]]
    sigma2, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * backcast])
    return sigma2


def garch11_loglik(params, r: np.ndarray, backcast: float | None = None) -> float:
    """Gaussian log-likelihood of a constant-mean GARCH(1,1); params = (mu, omega, alpha, beta)."""
    mu, omega, alpha, beta = params
    e = np.asarray(r, dtype=float) - mu
    s2 = garch11_variance(e, omega, alpha, beta, backcast)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return -np.inf
    return float(-0.5 * np.sum(np.log(2.0 * np.pi) + np.log(s2) + e ** 2 / s2))


def fit_garch11(r: np.ndarray, ftol: float = 1e-10) -> dict:
    """Constant-mean Gaussian GARCH(1,1) by SLSQP under arch's bounds and alpha + beta < 1.

    The backcast is computed once from the starting-value residuals (r - mean) and held fixed
    during the optimisation, exactly as arch's `fit` does, and the omega bound is arch's
    `(1e-8 * v, 10 * v)` with `v = mean(e^2)` (GARCH.bounds, arch/univariate/volatility.py).
    arch's own start is a grid search over `itertools.product(alphas, gammas, abg)` with
    `alphas = gammas = [0.01, 0.05, 0.1, 0.2]` and `abg = [0.5, 0.7, 0.9, 0.98]` - 64
    likelihood evaluations over 16 distinct points once o = 0 makes the gamma axis inert
    (GARCH.starting_values). We use four of the same points and keep the highest likelihood.
    """
    r = np.asarray(r, dtype=float).ravel()
    if r.ndim != 1 or r.shape[0] < 50:
        raise ValueError("need a 1-D series of at least 50 observations")
    mu0 = float(r.mean())
    e0 = r - mu0
    v = float(e0.var())
    bc = arch_backcast(e0)

    def nll(p):
        ll = garch11_loglik(p, r, bc)
        return -ll if np.isfinite(ll) else 1e10

    bounds = [(None, None), (1e-8 * v, 10.0 * v), (0.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: 1.0 - p[2] - p[3]}]
    best = None
    for alpha0, ab0 in ((0.05, 0.90), (0.10, 0.98), (0.20, 0.70), (0.01, 0.50)):
        x0 = np.array([mu0, (1.0 - ab0) * v, alpha0, ab0 - alpha0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(nll, x0, method="SLSQP", bounds=bounds, constraints=cons,
                                    options={"maxiter": 500, "ftol": ftol})
        if best is None or res.fun < best.fun:
            best = res
    params = dict(zip(GARCH_NAMES, (float(x) for x in best.x)))
    e = r - params["mu"]
    return {"params": params, "llf": float(-best.fun), "converged": bool(best.success),
            "backcast": bc,
            "sigma2": garch11_variance(e, params["omega"], params["alpha"], params["beta"], bc),
            "persistence": params["alpha"] + params["beta"]}


def garch11_forecast_variance(sigma2_last: float, e_last: float, omega: float, alpha: float,
                              beta: float, horizon: int) -> np.ndarray:
    """Analytic h-step variance forecasts, h = 1..horizon (the same recursion arch uses)."""
    out = np.empty(horizon)
    out[0] = omega + alpha * e_last ** 2 + beta * sigma2_last
    for h in range(1, horizon):
        out[h] = omega + (alpha + beta) * out[h - 1]
    return out


def annualise_vol(daily_variance: float, scale: float = 1.0) -> float:
    """sqrt(daily variance) / scale * sqrt(252): `scale` is the factor the data were multiplied by
    before fitting (100 for percent), so the answer is a decimal annual vol."""
    return math.sqrt(daily_variance) / scale * math.sqrt(PERIODS)


def arch_crosscheck(r_pct: np.ndarray, own: dict) -> dict | None:
    """Fit the same model with arch three ways; None when arch is not importable."""
    try:
        import arch
        from arch import arch_model
        from arch.utility.exceptions import DataScaleWarning
    except ImportError:
        return None
    out = {"version": arch.__version__}

    def fit(y, **kw):
        am = arch_model(y, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="normal", **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = am.fit(disp="off", show_warning=False)     # _check_scale runs inside fit
        scale_msgs = [str(c.message) for c in caught if issubclass(c.category, DataScaleWarning)]
        p = {"mu": float(res.params["mu"]), "omega": float(res.params["omega"]),
             "alpha": float(res.params["alpha[1]"]), "beta": float(res.params["beta[1]"])}
        # arch's own starting values, read AFTER fit so they sit on the data the optimiser saw
        # (fit rescales y in place when rescale=True): the mean start, then the 4 x 4 grid
        # search in GARCH.starting_values (alphas 0.01..0.2 x persistence 0.5..0.98).
        sv_mean = np.asarray(am.starting_values(), dtype=float)
        sv_vol = np.asarray(am.volatility.starting_values(am.resids(sv_mean)), dtype=float)
        start = {"mu": float(sv_mean[0]), "omega": float(sv_vol[0]), "alpha": float(sv_vol[1]),
                 "beta": float(sv_vol[2])}
        f = np.asarray(res.forecast(horizon=22).variance)[-1]
        cond_var = np.asarray(res.conditional_volatility, dtype=float) ** 2
        opt = res.optimization_result
        return {"params": p, "start": start,
                "moved_from_start": {k: p[k] - start[k] for k in GARCH_NAMES},
                "llf": float(res.loglikelihood), "scale": float(res.scale),
                "converged": int(res.convergence_flag) == 0,
                "n_iter": int(getattr(opt, "nit", -1)), "n_fev": int(getattr(opt, "nfev", -1)),
                "scale_warning": scale_msgs[0].splitlines()[0] if scale_msgs else "",
                "cond_var": cond_var, "forecast_var": f}

    out["percent"] = fit(r_pct)
    out["decimal_default"] = fit(r_pct / 100.0)                 # rescale=None: warns, no rescale
    out["decimal_rescale_true"] = fit(r_pct / 100.0, rescale=True)
    out["decimal_rescale_false"] = fit(r_pct / 100.0, rescale=False)
    # A decimal fit of the SAME model should reach llf_percent + T ln(100) (the Jacobian of
    # y / 100); anything below that is a worse optimum, not a different scale.
    out["decimal_llf_deficit"] = (out["decimal_default"]["llf"]
                                  - (out["percent"]["llf"] + len(r_pct) * math.log(100.0)))
    # The decimal fits' variance path, put back into percent^2 and compared with the percent fit.
    dec = out["decimal_default"]["cond_var"] * 1e4
    pct = out["percent"]["cond_var"]
    out["decimal_vs_percent_cond_var"] = {
        "max_rel_diff": float(np.abs(dec / pct - 1.0).max()),
        "rmse_rel": float(np.sqrt(np.mean((dec / pct - 1.0) ** 2)))}

    # Is the recursion the same? Evaluate OUR variance path at ARCH's percent-fit parameters.
    p = out["percent"]["params"]
    e = r_pct - p["mu"]
    ours = garch11_variance(e, p["omega"], p["alpha"], p["beta"], arch_backcast(r_pct - r_pct.mean()))
    out["max_abs_cond_var_diff_at_arch_params"] = float(np.abs(ours - out["percent"]["cond_var"]).max())
    out["llf_at_arch_params_ours"] = garch11_loglik([p[k] for k in GARCH_NAMES], r_pct,
                                                    arch_backcast(r_pct - r_pct.mean()))
    out["param_diff"] = {k: own["params"][k] - p[k] for k in GARCH_NAMES}
    return out


# ----------------------------------------------------------------------------- ranges ------
def simulate_ohlc(days: int, steps: int, annual_vol: float, annual_drift: float = 0.0,
                  overnight_share: float = 0.0, seed: int = SEED) -> dict:
    """Geometric Brownian motion observed on `steps` intraday increments per day, with an
    overnight jump between the previous close and the open carrying `overnight_share` of the
    day's variance. Returns O, H, L, C, the previous close, and the true daily variance."""
    if not 0.0 <= overnight_share < 1.0:
        raise ValueError("overnight_share must be in [0, 1)")
    rng = np.random.default_rng(seed)
    var_day = annual_vol ** 2 / PERIODS
    mu_day = annual_drift / PERIODS
    var_intra, var_night = var_day * (1.0 - overnight_share), var_day * overnight_share
    incr = (mu_day * (1.0 - overnight_share) / steps
            + math.sqrt(var_intra / steps) * rng.standard_normal((days, steps)))
    night = mu_day * overnight_share + math.sqrt(var_night) * rng.standard_normal(days)
    path = np.cumsum(incr, axis=1)                       # intraday log path relative to the open
    day_move = path[:, -1]
    log_open = np.cumsum(night) + np.r_[0.0, np.cumsum(day_move)[:-1]]
    log_close = log_open + day_move
    log_high = log_open + np.maximum(path.max(axis=1), 0.0)
    log_low = log_open + np.minimum(path.min(axis=1), 0.0)
    log_prev_close = np.r_[0.0, log_close[:-1]]
    return {"O": np.exp(log_open), "H": np.exp(log_high), "L": np.exp(log_low),
            "C": np.exp(log_close), "C_prev": np.exp(log_prev_close), "true_var": var_day}


def ohlc_components(O, H, L, C, C_prev) -> dict[str, np.ndarray]:
    """u = ln(H/O), d = ln(L/O), c = ln(C/O), o = ln(O/C_prev), r = ln(C/C_prev)."""
    O, H, L, C, C_prev = (np.asarray(a, dtype=float) for a in (O, H, L, C, C_prev))
    if np.any(H < np.maximum(O, C)) or np.any(L > np.minimum(O, C)):
        raise ValueError("high must be >= max(open, close) and low <= min(open, close)")
    return {"u": np.log(H / O), "d": np.log(L / O), "c": np.log(C / O),
            "o": np.log(O / C_prev), "r": np.log(C / C_prev)}


def _windows(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = a.shape[0] // n
    if m == 0:
        raise ValueError("series shorter than one window")
    return a[: m * n].reshape(m, n)


def rv_close_to_close(r, n: int) -> np.ndarray:
    """Sample variance (ddof=1) of the n close-to-close log returns in each window."""
    return _windows(r, n).var(axis=1, ddof=1)


def rv_parkinson(u, d, n: int) -> np.ndarray:
    """Parkinson (1980): mean of ln(H/L)^2 over the window, divided by 4 ln 2."""
    return _windows((np.asarray(u) - np.asarray(d)) ** 2, n).mean(axis=1) / (4.0 * LN2)


def rv_garman_klass(u, d, c, n: int) -> np.ndarray:
    """Garman-Klass (1980): mean of 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2."""
    u, d, c = (np.asarray(a) for a in (u, d, c))
    return _windows(0.5 * (u - d) ** 2 - (2.0 * LN2 - 1.0) * c ** 2, n).mean(axis=1)


def rv_rogers_satchell(u, d, c, n: int) -> np.ndarray:
    """Rogers-Satchell (1991): mean of ln(H/C) ln(H/O) + ln(L/C) ln(L/O); drift-independent."""
    u, d, c = (np.asarray(a) for a in (u, d, c))
    return _windows(u * (u - c) + d * (d - c), n).mean(axis=1)


def yang_zhang_k(n: int) -> float:
    """k = 0.34 / (1.34 + (n + 1) / (n - 1)), Yang-Zhang (2000) with alpha = 1.34."""
    if n < 2:
        raise ValueError("Yang-Zhang needs at least two days per window")
    return 0.34 / (1.34 + (n + 1.0) / (n - 1.0))


def rv_yang_zhang(o, c, u, d, n: int) -> np.ndarray:
    """Yang-Zhang (2000): var(overnight) + k var(open-to-close) + (1 - k) Rogers-Satchell."""
    k = yang_zhang_k(n)
    so2 = _windows(o, n).var(axis=1, ddof=1)
    sc2 = _windows(c, n).var(axis=1, ddof=1)
    return so2 + k * sc2 + (1.0 - k) * rv_rogers_satchell(u, d, c, n)


EST_NAMES = ("close-to-close", "Parkinson", "Garman-Klass", "Rogers-Satchell", "Yang-Zhang")


def all_estimators(comp: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {"close-to-close": rv_close_to_close(comp["r"], n),
            "Parkinson": rv_parkinson(comp["u"], comp["d"], n),
            "Garman-Klass": rv_garman_klass(comp["u"], comp["d"], comp["c"], n),
            "Rogers-Satchell": rv_rogers_satchell(comp["u"], comp["d"], comp["c"], n),
            "Yang-Zhang": rv_yang_zhang(comp["o"], comp["c"], comp["u"], comp["d"], n)}


def efficiency_table(est: dict[str, np.ndarray], true_var: float) -> dict[str, dict[str, float]]:
    """Per estimator: mean / true variance (1.0 = unbiased), the Monte Carlo standard error of
    that ratio, Var(close-to-close) / Var(est), and RMSE relative to close-to-close.

    `se` is what says whether a bias ratio of 0.98 is bias or sampling noise: it is
    sd(estimate) / sqrt(#windows) / true_var, so a ratio within about 2 se of 1.0 is noise.
    """
    base = est["close-to-close"].var(ddof=1)
    return {name: {"bias_ratio": float(v.mean() / true_var),
                   "se": float(v.std(ddof=1) / math.sqrt(v.shape[0]) / true_var),
                   "efficiency": float(base / v.var(ddof=1)),
                   "rmse_ratio": float(np.sqrt(np.mean((v - true_var) ** 2))
                                       / np.sqrt(np.mean((est["close-to-close"] - true_var) ** 2)))}
            for name, v in est.items()}


# ----------------------------------------------------------------------------- HAR-RV ------
def simulate_realized_variance(days: int, steps: int, seed: int = SEED, phi: float = 0.98,
                               sd_logvol: float = 0.30, mean_annual_vol: float = 0.20
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Daily realized variance (sum of squared intraday returns) from a log-vol AR(1)."""
    rng = np.random.default_rng(seed)
    m = math.log(mean_annual_vol / math.sqrt(PERIODS))
    lv = np.empty(days)
    lv[0] = m
    innov = rng.standard_normal(days) * sd_logvol * math.sqrt(1.0 - phi ** 2)
    for t in range(1, days):
        lv[t] = m + phi * (lv[t - 1] - m) + innov[t]
    true_var = np.exp(2.0 * lv)
    r = np.sqrt(true_var / steps)[:, None] * rng.standard_normal((days, steps))
    return (r ** 2).sum(axis=1), true_var


def har_design(rv: np.ndarray, lags=HAR_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """Rows t = max(lags)..n-1: target rv[t]; regressors 1 and, for each k in lags, the mean of
    rv[t-1..t-k]. Corsi (2009) with (1, 5, 22) = daily, weekly, monthly; arch's HARX
    lags=[1, 5, 22] builds the same columns."""
    rv = np.asarray(rv, dtype=float)
    L = max(lags)
    n = rv.shape[0]
    if n <= L + 1:
        raise ValueError("series too short for the longest HAR lag")
    cs = np.r_[0.0, np.cumsum(rv)]
    t = np.arange(L, n)
    cols = [np.ones(n - L)]
    for k in lags:
        cols.append((cs[t] - cs[t - k]) / k)          # mean of rv[t-k .. t-1]
    return np.column_stack(cols), rv[L:]


def har_fit(rv: np.ndarray, lags=HAR_LAGS) -> np.ndarray:
    X, y = har_design(rv, lags)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def har_forecast(coef: np.ndarray, rv_hist: np.ndarray, lags=HAR_LAGS) -> float:
    """One-step-ahead forecast from the history up to and including its last element."""
    rv_hist = np.asarray(rv_hist, dtype=float)
    feats = [1.0] + [rv_hist[-k:].mean() for k in lags]
    return float(np.dot(coef, feats))


def qlike(forecast: np.ndarray, realized: np.ndarray) -> float:
    """QLIKE loss: mean of realized/forecast - ln(realized/forecast) - 1 (Patton 2011)."""
    f, r = np.asarray(forecast, dtype=float), np.asarray(realized, dtype=float)
    return float(np.mean(r / f - np.log(r / f) - 1.0))


def har_out_of_sample(rv: np.ndarray, train: int, lags=HAR_LAGS) -> dict:
    """Fit once on rv[:train]; forecast every later day one step ahead; compare with naive and
    the trailing 22-day mean on MSE and QLIKE."""
    coef = har_fit(rv[:train], lags)
    idx = np.arange(train, len(rv))
    har = np.array([har_forecast(coef, rv[:t], lags) for t in idx])
    naive = rv[idx - 1]
    mean22 = np.array([rv[t - 22:t].mean() for t in idx])
    actual = rv[idx]
    out = {"coef": coef, "n_test": len(idx)}
    for name, f in (("HAR", har), ("naive (yesterday)", naive), ("22-day mean", mean22)):
        out[name] = {"mse": float(np.mean((f - actual) ** 2)), "qlike": qlike(np.maximum(f, 1e-12), actual)}
    return out


def arch_har_crosscheck(rv: np.ndarray, lags=HAR_LAGS) -> dict | None:
    try:
        from arch.univariate import ConstantVariance, HARX
    except ImportError:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = HARX(rv, lags=list(lags), volatility=ConstantVariance()).fit(disp="off")
    arch_coef = np.asarray(res.params)[: len(lags) + 1]
    own = har_fit(rv, lags)
    return {"arch": arch_coef, "own": own, "names": list(res.params.index[: len(lags) + 1]),
            "max_abs_diff": float(np.abs(arch_coef - own).max())}


# ----------------------------------------------------------------------------- demo --------
def _fmt(p: dict) -> str:
    return "  ".join(f"{k}={p[k]:.5f}" for k in GARCH_NAMES)


if __name__ == "__main__":
    t_all = time.time()
    import scipy
    try:
        import arch as _arch
        arch_ver = _arch.__version__
    except ImportError:
        arch_ver = "NOT INSTALLED"
    print(f"arch {arch_ver}, numpy {np.__version__}, scipy {scipy.__version__}; seed {SEED}")

    # ---- 1. GARCH(1,1) recovery -----------------------------------------------------------
    r_pct, s2_true = simulate_garch11(N_GARCH, seed=SEED, **TRUE_GARCH)
    own = fit_garch11(r_pct)
    print(f"\n=== 1. GARCH(1,1) by maximum likelihood, T={N_GARCH}, returns in PERCENT ===")
    print(f"  true      {_fmt(TRUE_GARCH)}  persistence={TRUE_GARCH['alpha'] + TRUE_GARCH['beta']:.3f}")
    print(f"  own MLE   {_fmt(own['params'])}  persistence={own['persistence']:.3f}"
          f"  llf={own['llf']:.3f}  converged={own['converged']}")
    err = {k: own["params"][k] - TRUE_GARCH[k] for k in GARCH_NAMES}
    print("  error     " + "  ".join(f"{k}={err[k]:+.5f}" for k in GARCH_NAMES))
    se_mu = r_pct.std(ddof=1) / math.sqrt(N_GARCH)
    print(f"  the mean is the parameter that does not converge: mu is off by {err['mu']:+.5f} ="
          f" {err['mu'] / se_mu:+.2f} standard errors of the sample mean (se {se_mu:.5f}), while"
          f" alpha and beta land within {max(abs(err['alpha']), abs(err['beta'])) / 0.98:.1%} of the"
          f" true persistence")
    rmse_path = math.sqrt(np.mean((own["sigma2"] - s2_true) ** 2))
    print(f"  conditional variance path vs truth: RMSE {rmse_path:.4f} on a mean variance of"
          f" {s2_true.mean():.3f}; corr {np.corrcoef(own['sigma2'], s2_true)[0, 1]:.4f}")

    chk = arch_crosscheck(r_pct, own)
    if chk is not None:
        a = chk["percent"]
        print(f"  arch {chk['version']}  {_fmt(a['params'])}  llf={a['llf']:.3f}  scale={a['scale']:.0f}"
              f"  converged={a['converged']}")
        print("  own - arch: " + "  ".join(f"{k}={chk['param_diff'][k]:+.2e}" for k in GARCH_NAMES)
              + f"  llf {own['llf'] - a['llf']:+.2e}")
        print(f"  our recursion at arch's parameters: max |sigma2 - arch.conditional_volatility**2|"
              f" = {chk['max_abs_cond_var_diff_at_arch_params']:.1e}; our llf at arch's params"
              f" {chk['llf_at_arch_params_ours']:.6f} vs arch {a['llf']:.6f}")
    else:
        print("  arch not installed: the numpy estimator stands alone (pip install arch to cross-check)")

    # ---- 2. Scaling ------------------------------------------------------------------------
    print("\n=== 2. The same returns in decimals: what arch does with them ===")
    if chk is not None:
        dd, dt, df, pc = (chk[k] for k in ("decimal_default", "decimal_rescale_true",
                                          "decimal_rescale_false", "percent"))
        print(f"  percent fit  : SLSQP took {pc['n_iter']} iterations ({pc['n_fev']} function evaluations);"
              f" arch's starting grid point was alpha={pc['start']['alpha']:.2f} beta={pc['start']['beta']:.2f}"
              f" and the fit moved alpha by {pc['moved_from_start']['alpha']:+.4f},"
              f" beta by {pc['moved_from_start']['beta']:+.4f}")
        print(f"  rescale=None (default): warning -> \"{dd['scale_warning']}\"; res.scale={dd['scale']:.0f}")
        print(f"     {_fmt(dd['params'])}  llf={dd['llf']:.3f}  converged={dd['converged']}"
              f"  iterations={dd['n_iter']} evaluations={dd['n_fev']}")
        print(f"     starting grid point alpha={dd['start']['alpha']:.2f} beta={dd['start']['beta']:.2f}"
              f" omega={dd['start']['omega']:.3e}; the fit moved "
              + ", ".join(f"{k} by {dd['moved_from_start'][k]:+.1e}" for k in GARCH_NAMES)
              + " - it reported the starting values as the optimum")
        print(f"     likelihood in comparable units, decimal llf - (percent llf + T ln 100):"
              f" {chk['decimal_llf_deficit']:+.3f} nats - a worse fit, flagged converged")
        print(f"  rescale=True : res.scale={dt['scale']:.0f}; parameters come back in the RESCALED units")
        print(f"     {_fmt(dt['params'])}  llf={dt['llf']:.3f}  iterations={dt['n_iter']}")
        print(f"  rescale=False: no warning, fit in decimals; res.scale={df['scale']:.0f}")
        print(f"     {_fmt(df['params'])}  llf={df['llf']:.3f}  converged={df['converged']}"
              f"  iterations={df['n_iter']} evaluations={df['n_fev']}")
        cv = chk["decimal_vs_percent_cond_var"]
        print(f"  conditional variance path, decimal-default fit x 1e4 vs percent fit: max relative"
              f" difference {cv['max_rel_diff']:.1%}, RMS {cv['rmse_rel']:.1%}")
        print(f"  persistence alpha+beta: percent {pc['params']['alpha'] + pc['params']['beta']:.4f},"
              f" decimal default {dd['params']['alpha'] + dd['params']['beta']:.4f},"
              f" decimal rescale=True {dt['params']['alpha'] + dt['params']['beta']:.4f},"
              f" decimal rescale=False {df['params']['alpha'] + df['params']['beta']:.4f}")
        # Annualised vol from the 1-step and 22-step forecasts, converted CORRECTLY from each fit
        print("  1-step / 22-step annualised vol forecast, each converted with its own res.scale:")
        for name, d, scale in (("percent fit", pc, 100.0), ("decimal default", dd, 1.0),
                               ("decimal rescale=True", dt, 100.0), ("decimal rescale=False", df, 1.0)):
            f = d["forecast_var"]
            print(f"     {name:<22} {annualise_vol(f[0], scale):.4f} / {annualise_vol(f[-1], scale):.4f}"
                  f"   (raw forecast variance {f[0]:.3e}, res.scale={d['scale']:.0f})")
        wrong = annualise_vol(dt["forecast_var"][0], 1.0)
        print(f"  the trap: rescale=True returns variance in (100 x return)^2 units; read it as decimals"
              f" and the 1-step annualised vol is {wrong:.2f}, i.e. {wrong:.0%} - {wrong / annualise_vol(dt['forecast_var'][0], 100.0):.0f}x")
        wrong2 = annualise_vol(pc["forecast_var"][0], 1.0)
        print(f"  the same mistake on a percent fit gives {wrong2:.2f} ({wrong2:.0%}); the true value is"
              f" {annualise_vol(s2_true[-1], 100.0):.4f}")
    else:
        own_dec = fit_garch11(r_pct / 100.0)
        print("  arch not installed; the numpy estimator on decimals (it does not warn either):")
        print(f"     {_fmt(own_dec['params'])}  llf={own_dec['llf']:.3f}  omega is 1e4 x smaller,"
              f" alpha/beta unchanged: {own_dec['params']['alpha'] + own_dec['params']['beta']:.4f}"
              f" vs {own['persistence']:.4f} in percent")

    # ---- 3. Range estimators -----------------------------------------------------------------
    print(f"\n=== 3. Range-based realized variance on simulated GBM: {GBM_DAYS} days, {STEPS_PER_DAY}"
          f" steps/day, {WINDOW}-day windows ({GBM_DAYS // WINDOW} windows), annual vol {ANNUAL_VOL:.0%} ===")
    print("  bias = mean estimate / true variance (1.000 = unbiased); efficiency = Var(close-to-close) /"
          " Var(estimator)")
    for label, drift, night in SCENARIOS:
        ohlc = simulate_ohlc(GBM_DAYS, STEPS_PER_DAY, ANNUAL_VOL, drift, night, seed=SEED)
        comp = ohlc_components(ohlc["O"], ohlc["H"], ohlc["L"], ohlc["C"], ohlc["C_prev"])
        tab = efficiency_table(all_estimators(comp, WINDOW), ohlc["true_var"])
        print(f"  -- {label}")
        for name, row in tab.items():
            print(f"     {name:<16} bias {row['bias_ratio']:.3f} +/- {row['se']:.3f}"
                  f"   efficiency {row['efficiency']:5.2f}"
                  f"   RMSE vs close-to-close {row['rmse_ratio']:.2f}")

    print(f"  -- discretisation: every estimator's bias vs intraday steps per day"
          f" ({DISC_DAYS} days, zero drift, no gap). Each formula is unbiased for the")
    print(f"     CONTINUOUS path its paper assumes, so the column has to walk to 1.000 as the"
          f" high and low are observed more often.")
    print("     steps/day  " + "".join(f"{n:>17}" for n in EST_NAMES))
    for steps in DISC_STEPS:
        ohlc = simulate_ohlc(DISC_DAYS, steps, ANNUAL_VOL, 0.0, 0.0, seed=SEED)
        comp = ohlc_components(ohlc["O"], ohlc["H"], ohlc["L"], ohlc["C"], ohlc["C_prev"])
        est = all_estimators(comp, WINDOW)
        print(f"     {steps:>9}  "
              + "".join(f"{est[n].mean() / ohlc['true_var']:>17.3f}" for n in EST_NAMES))

    # ---- 4. HAR-RV -------------------------------------------------------------------------
    rv, true_v = simulate_realized_variance(HAR_DAYS, HAR_STEPS, seed=SEED)
    oos = har_out_of_sample(rv, HAR_TRAIN)
    print(f"\n=== 4. HAR-RV (Corsi 2009): OLS on the {HAR_LAGS} lag averages, fit on {HAR_TRAIN} days,"
          f" scored on the next {oos['n_test']} ===")
    print("  coefficients (const, daily, weekly, monthly): "
          + ", ".join(f"{c:.6g}" for c in oos["coef"])
          + f"   (they sum to {oos['coef'][1:].sum():.4f}; mean RV {rv[:HAR_TRAIN].mean():.3e})")
    hc = arch_har_crosscheck(rv[:HAR_TRAIN])
    if hc is not None:
        print(f"  arch HARX(lags=[1, 5, 22]) {hc['names']}: "
              + ", ".join(f"{c:.6g}" for c in hc["arch"]) + f"; max |diff| {hc['max_abs_diff']:.1e}")
    for name in ("HAR", "naive (yesterday)", "22-day mean"):
        print(f"  {name:<18} MSE {oos[name]['mse']:.3e}   QLIKE {oos[name]['qlike']:.4f}")
    print(f"  HAR / naive: MSE {oos['HAR']['mse'] / oos['naive (yesterday)']['mse']:.3f},"
          f" QLIKE {oos['HAR']['qlike'] / oos['naive (yesterday)']['qlike']:.3f}")

    print("\nRule: fit GARCH on returns x 100, divide every variance by res.scale**2 before annualising,"
          " and never use a range estimator on bars with overnight gaps unless it is Yang-Zhang.")
    print(f"total runtime {time.time() - t_all:.1f}s")
