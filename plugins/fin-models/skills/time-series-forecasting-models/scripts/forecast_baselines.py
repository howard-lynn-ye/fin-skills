#!/usr/bin/env python3
"""Forecast baselines, rolling-origin evaluation, MASE, and the Diebold-Mariano test.

Four things a forecasting result can get silently wrong, each measured here on seeded data:

  1. THE LEVEL. R^2 of a price on its own lag is 0.99 and means nothing: the naive forecast
     "tomorrow = today" reproduces it. The same model scored on RETURNS has an R^2 near zero.
     Section 1 measures both on the SAME series, plus the lag-1 signature that says a "good"
     price forecast is just yesterday's price.
  2. THE BASELINE. naive, seasonal-naive, drift and mean are the four forecasts everything is
     supposed to beat. Section 3 runs a rolling-origin walk-forward over two DGPs - a daily
     price and a seasonal series - and scores MASE, so a model that loses to `y[t-1]` is
     visible immediately.
  3. THE SCALE. MASE (Hyndman-Koehler 2006) divides by the IN-SAMPLE naive MAE, so it is
     comparable across series and its interpretation is fixed: MASE > 1 means worse than the
     training-set naive. Section 2 checks the definition's own identity - the in-sample naive
     forecast has MASE exactly 1.
  4. THE TEST. Two forecasts differing on MSE is not evidence. Section 4 runs Diebold-Mariano
     with the Harvey-Leybourne-Newbold small-sample correction, verifies the statistic against
     scipy's one-sample t-test in the h = 1 case where they must coincide up to a known factor,
     and measures the test's SIZE under the null by Monte Carlo.

statsmodels is optional: ARIMA and ETS are imported inside their functions and the demo prints
the baselines and the DM test without them.

Run:  python forecast_baselines.py    (numpy / scipy; statsmodels optional; seeds; ~41 s)
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy import stats

SEED = 0
PERIODS = 252

# --- DGP A: a daily log-price random walk with drift. Nothing is forecastable but the drift.
N_PRICE = 1300
MU_DAILY = 0.0003              # ~7.6 %/yr
SIG_DAILY = 0.010
PRICE_TRAIN = 1000

# --- DGP B: a monthly series with a trend and a 12-period seasonal. Genuinely forecastable.
N_SEASON = 240
SEASON_M = 12
SEASON_TREND = 0.2
SEASON_AMP = 10.0
SEASON_NOISE = 2.0
SEASON_TRAIN = 168             # 14 years in, 6 years out

HORIZON = 1
DM_SIMS = 1000
DM_T = 250
ARIMA_ORDER = (1, 0, 1)        # on returns / on the seasonal level after differencing
REFIT_EVERY = 250              # walk-forward: refit the ARIMA this often, filter in between


# ----------------------------------------------------------------------------- data ---------
def simulate_price(n: int = N_PRICE, seed: int = SEED, mu: float = MU_DAILY,
                   sigma: float = SIG_DAILY, p0: float = 100.0) -> dict[str, np.ndarray]:
    """A geometric random walk. `price` is the level, `ret` the log return (ret[0] = 0)."""
    rng = np.random.default_rng(seed)
    r = np.r_[0.0, rng.normal(mu, sigma, n - 1)]
    return {"price": p0 * np.exp(np.cumsum(r)), "ret": r}


def simulate_seasonal(n: int = N_SEASON, m: int = SEASON_M, seed: int = SEED,
                      trend: float = SEASON_TREND, amp: float = SEASON_AMP,
                      noise: float = SEASON_NOISE) -> np.ndarray:
    """level = 100 + trend * t + amp * sin(2 pi t / m) + noise. Deterministic apart from noise,
    so seasonal-naive, ETS and ARIMA all have something real to find."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 100.0 + trend * t + amp * np.sin(2.0 * math.pi * t / m) + rng.normal(0.0, noise, n)


# ----------------------------------------------------------------------------- section 1 ----
def r2_of_lag(y: np.ndarray, lag: int = 1) -> float:
    """R^2 of the OLS regression y_t = a + b y_{t-lag}. On a price this is ~1 and says nothing."""
    y = np.asarray(y, dtype=float)
    if lag < 1 or len(y) <= lag + 2:
        raise ValueError("need lag >= 1 and a longer series")
    a, b = y[lag:], y[:-lag]
    X = np.column_stack([np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)
    resid = a - X @ coef
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1.0 - float(resid @ resid) / ss_tot)


def out_of_sample_r2(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray) -> float:
    """1 - SSE(forecast) / SSE(benchmark) - the only R^2 that means anything out of sample
    (Campbell-Thompson form). Negative means the forecast is worse than the benchmark."""
    actual, forecast, benchmark = (np.asarray(a, dtype=float)
                                   for a in (actual, forecast, benchmark))
    sse_f = float(np.sum((actual - forecast) ** 2))
    sse_b = float(np.sum((actual - benchmark) ** 2))
    return float(1.0 - sse_f / sse_b) if sse_b > 0 else float("nan")


def lag_signature(forecast: np.ndarray, actual: np.ndarray, max_lag: int = 5) -> dict:
    """corr(forecast_t, actual_{t-k}) for k = 0..max_lag. A price 'forecast' whose strongest
    correlation is at k = 1 is reproducing yesterday's value and calling it a prediction."""
    forecast, actual = np.asarray(forecast, dtype=float), np.asarray(actual, dtype=float)
    out = {}
    for k in range(max_lag + 1):
        f = forecast[k:] if k else forecast
        a = actual[: len(actual) - k] if k else actual
        out[k] = float(np.corrcoef(f, a)[0, 1])
    return out


# ----------------------------------------------------------------------------- baselines ----
def f_naive(train: np.ndarray, h: int) -> np.ndarray:
    """y_hat[T+i] = y[T] for every i."""
    return np.full(h, float(np.asarray(train, dtype=float)[-1]))


def f_seasonal_naive(train: np.ndarray, h: int, m: int) -> np.ndarray:
    """y_hat[T+i] = y[T + i - m * ceil(i / m)]: the same point in the most recent full cycle."""
    train = np.asarray(train, dtype=float)
    if m < 1 or len(train) < m:
        raise ValueError("need at least one full season of history")
    idx = [len(train) + i - m * math.ceil((i + 1) / m) for i in range(h)]
    return train[np.asarray(idx)]


def f_drift(train: np.ndarray, h: int) -> np.ndarray:
    """The line through the first and last observation, extrapolated (Hyndman's drift method)."""
    train = np.asarray(train, dtype=float)
    if len(train) < 2:
        raise ValueError("drift needs two observations")
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def f_mean(train: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(np.asarray(train, dtype=float).mean()))


BASELINES = {
    "naive": lambda tr, h, m: f_naive(tr, h),
    "seasonal naive": lambda tr, h, m: f_seasonal_naive(tr, h, m),
    "drift": lambda tr, h, m: f_drift(tr, h),
    "mean": lambda tr, h, m: f_mean(tr, h),
}


# ----------------------------------------------------------------------------- scoring ------
def scaling_factor(train: np.ndarray, m: int = 1) -> float:
    """MASE denominator: the MAE of the (seasonal) naive forecast IN SAMPLE,
    mean(|y[t] - y[t-m]|) over the training set. Hyndman-Koehler (2006)."""
    train = np.asarray(train, dtype=float)
    if len(train) <= m:
        raise ValueError("training series shorter than the seasonal period")
    d = np.abs(train[m:] - train[:-m])
    return float(d.mean())


def mase(actual: np.ndarray, forecast: np.ndarray, scale: float) -> float:
    if not scale > 0:
        raise ValueError("MASE scaling factor must be positive")
    return float(np.mean(np.abs(np.asarray(actual, dtype=float)
                                - np.asarray(forecast, dtype=float))) / scale)


def score(actual: np.ndarray, forecast: np.ndarray, scale: float) -> dict[str, float]:
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    e = actual - forecast
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mase": mase(actual, forecast, scale)}


def rolling_origin(y: np.ndarray, forecast_fn, start: int, horizon: int = HORIZON
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Walk forward one observation at a time. At each origin T in [start, n - horizon] the
    function sees y[:T] and nothing else, and is scored on y[T + horizon - 1].

    Returns (forecasts, actuals), both of length n - horizon + 1 - start.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if start < 2 or start > n - horizon:
        raise ValueError("start must leave at least two training points and one test point")
    fc, act = [], []
    for T in range(start, n - horizon + 1):
        f = np.asarray(forecast_fn(y[:T], horizon), dtype=float)
        if f.shape != (horizon,):
            raise ValueError("forecast_fn must return exactly `horizon` values")
        fc.append(f[-1])
        act.append(y[T + horizon - 1])
    return np.asarray(fc), np.asarray(act)


# ----------------------------------------------------------------------------- DM test ------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "squared",
                    harvey: bool = True) -> dict[str, float]:
    """Diebold-Mariano (1995) with the Harvey-Leybourne-Newbold (1997) small-sample correction.

    d_t = L(e1_t) - L(e2_t); the statistic is d_bar / sqrt(V_hat(d_bar)) with V_hat the
    Newey-West long-run variance truncated at h - 1 lags (so at h = 1 it is just gamma_0 / T).
    A NEGATIVE statistic means forecast 1 has the smaller loss. `harvey=True` multiplies by
    sqrt((T + 1 - 2h + h(h-1)/T) / T) and refers the result to a t distribution with T - 1
    degrees of freedom instead of the normal.

    ⚠️ The null is EQUAL EXPECTED LOSS, not "the models are the same"; and it assumes the loss
    differential is covariance-stationary, which two nested models estimated on the same data
    can violate. Comparing more than two forecasts needs a multiple-comparison procedure
    (`arch.bootstrap.SPA` / `StepM` / `MCS`), not a table of pairwise DM tests.
    """
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    if e1.shape != e2.shape or e1.ndim != 1:
        raise ValueError("e1 and e2 must be 1-D arrays of the same length")
    if h < 1:
        raise ValueError("h must be at least 1")
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")
    T = len(d)
    if T <= h:
        raise ValueError("need more observations than the forecast horizon")
    dbar = float(d.mean())
    dc = d - dbar
    gamma = [float(dc @ dc) / T]
    for k in range(1, h):
        gamma.append(float(dc[k:] @ dc[:-k]) / T)
    v = (gamma[0] + 2.0 * sum(gamma[1:])) / T
    if not v > 0:
        return {"stat": float("nan"), "p_value": float("nan"), "dbar": dbar, "df": T - 1}
    stat = dbar / math.sqrt(v)
    if harvey:
        stat *= math.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    p = float(2.0 * stats.t.sf(abs(stat), df=T - 1)) if harvey else \
        float(2.0 * stats.norm.sf(abs(stat)))
    return {"stat": float(stat), "p_value": p, "dbar": dbar, "df": T - 1}


def dm_matches_scipy_t_test(e1: np.ndarray, e2: np.ndarray) -> dict[str, float]:
    """At h = 1 with harvey=False, DM is a one-sample t-test on the loss differential using the
    POPULATION variance, so it must equal scipy's `ttest_1samp` statistic times sqrt(T/(T-1)).
    Computed both ways here rather than asserted."""
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dm = diebold_mariano(e1, e2, h=1, harvey=False)["stat"]
    t_stat = float(stats.ttest_1samp(d, 0.0).statistic)
    return {"dm": dm, "scipy_t": t_stat, "ratio": dm / t_stat,
            "expected_ratio": math.sqrt(T / (T - 1.0)),
            "abs_error": abs(dm / t_stat - math.sqrt(T / (T - 1.0)))}


def dm_size_under_the_null(sims: int = DM_SIMS, T: int = DM_T, seed: int = SEED,
                           alpha: float = 0.05) -> dict[str, float]:
    """Two forecasts with the SAME expected loss: the rejection rate should be `alpha`."""
    rng = np.random.default_rng(seed)
    p = np.empty(sims)
    for i in range(sims):
        e = rng.standard_normal((T, 2))
        p[i] = diebold_mariano(e[:, 0], e[:, 1], h=1)["p_value"]
    return {"size": float(np.mean(p < alpha)), "alpha": alpha, "sims": sims, "T": T,
            "se": float(math.sqrt(alpha * (1 - alpha) / sims))}


# ----------------------------------------------------------------------------- statsmodels --
def arima_walk_forward(y: np.ndarray, start: int, order=ARIMA_ORDER, trend=None,
                       refit_every: int = REFIT_EVERY, horizon: int = HORIZON,
                       seasonal_order=(0, 0, 0, 0)) -> dict | None:
    """One-step-ahead ARIMA forecasts under a walk-forward that never sees the future.

    The model is refit every `refit_every` origins and EXTENDED in between with
    `res.append(new_obs, refit=False)` - ✅ statsmodels' documented way to add observations
    without re-estimating (MLEResults.append(endog, exog=None, refit=False, fit_kwargs=None)).
    Returns None when statsmodels is not importable.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None
    y = np.asarray(y, dtype=float)
    n = len(y)
    fc, act = [], []
    res = None
    n_fits = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T in range(start, n - horizon + 1):
            if res is None or (T - start) % refit_every == 0:
                res = ARIMA(y[:T], order=order, seasonal_order=seasonal_order,
                            trend=trend).fit()
                n_fits += 1
            elif len(res.model.endog) < T:
                res = res.append(y[len(res.model.endog):T], refit=False)
            fc.append(float(np.asarray(res.forecast(steps=horizon))[-1]))
            act.append(float(y[T + horizon - 1]))
    return {"forecast": np.asarray(fc), "actual": np.asarray(act), "n_fits": n_fits,
            "order": order, "trend": res.model.trend}


SMOOTHING_KEYS = ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend")


def ets_walk_forward(y: np.ndarray, start: int, seasonal_periods: int | None = None,
                     trend: str | None = "add", seasonal: str | None = None,
                     horizon: int = HORIZON, refit_every: int = 1) -> dict | None:
    """Holt-Winters exponential smoothing over a rolling origin. None without statsmodels.

    🚨 `ExponentialSmoothing` has no `append`, so there is no cheap honest way to carry a fit
    forward. With `refit_every > 1` this re-applies the last ESTIMATED smoothing parameters to
    the longer series with `optimized=False`, which is a DIFFERENT estimator - the initial
    states are re-derived by the heuristic rather than by the likelihood. The demo measures
    what that costs. `refit_every=1` (the default) refits at every origin and is the honest
    comparison; nothing here ever sees data past the origin either way.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")
    y = np.asarray(y, dtype=float)
    n = len(y)
    fc, act = [], []
    n_fits = 0
    fixed: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T in range(start, n - horizon + 1):
            model = ExponentialSmoothing(y[:T], trend=trend, seasonal=seasonal,
                                         seasonal_periods=seasonal_periods,
                                         initialization_method="estimated")
            if (T - start) % refit_every == 0:
                res = model.fit()
                fixed = {k: float(res.params[k]) for k in SMOOTHING_KEYS
                         if k in res.params and np.isfinite(res.params[k])}
                n_fits += 1
            else:
                res = model.fit(optimized=False, **fixed)
            fc.append(float(np.asarray(res.forecast(horizon))[-1]))
            act.append(float(y[T + horizon - 1]))
    return {"forecast": np.asarray(fc), "actual": np.asarray(act), "n_fits": n_fits}


def arima_trend_defaults() -> dict | None:
    """✅ statsmodels/tsa/arima/model.py: `trend=None` becomes 'c' when the model is NOT
    integrated and 'n' when it is, so an ARIMA(p, 1, q) has no drift term unless you ask."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return None
    rng = np.random.default_rng(SEED)
    steps = rng.normal(0.05, 0.10, 400)             # drift 0.05/step, se of the mean 0.005
    y = np.cumsum(steps)
    out = {"sample_drift": float(steps.mean())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for label, order, trend in (("d=0 on the level", (1, 0, 0), None),
                                    ("d=1, default trend", (1, 1, 0), None),
                                    ("d=1, trend='t'", (1, 1, 0), "t")):
            res = ARIMA(y, order=order, trend=trend).fit()
            f = np.asarray(res.forecast(steps=50))
            out[label] = {"trend": res.model.trend, "params": list(res.model.param_names),
                          "f1": float(f[0]), "f50": float(f[-1]),
                          "slope": float((f[-1] - f[0]) / 49.0)}
    out["true_slope"] = 0.05
    # ARIMA(0, 1, 0) with the default trend='n' has no parameters at all and its one-step
    # forecast is the last observation: it IS the naive forecast, to machine precision.
    price = simulate_price(600, SEED)["price"]
    diffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for T in (300, 400, 500, 600):
            res = ARIMA(price[:T], order=(0, 1, 0)).fit()
            diffs.append(abs(float(np.asarray(res.forecast(steps=1))[0]) - price[T - 1]))
    out["naive_identity"] = {"max_abs": float(max(diffs)), "n": len(diffs)}
    return out


# ----------------------------------------------------------------------------- demo --------
def _table(rows: dict[str, dict[str, float]], keys=("mase", "rmse", "mae")) -> None:
    for name, s in rows.items():
        print(f"     {name:<26} " + "   ".join(f"{k.upper()} {s[k]:.4f}" for k in keys))


if __name__ == "__main__":
    t_all = time.time()
    try:
        import statsmodels
        sm_ver = statsmodels.__version__
    except ImportError:
        sm_ver = "NOT INSTALLED"
    print(f"statsmodels {sm_ver}, numpy {np.__version__}, seed {SEED}")

    # ---- 1. The random-walk illusion --------------------------------------------------------
    d = simulate_price()
    price, ret = d["price"], d["ret"]
    print(f"\n=== 1. The same series, two framings ({N_PRICE} days of a random walk with drift"
          f" {MU_DAILY} and vol {SIG_DAILY}) ===")
    print(f"  R^2 of price_t on price_(t-1):  {r2_of_lag(price):.6f}")
    print(f"  R^2 of return_t on return_(t-1): {r2_of_lag(ret[1:]):.6f}")
    te = slice(PRICE_TRAIN, N_PRICE)
    naive_price = price[PRICE_TRAIN - 1:N_PRICE - 1]
    print(f"  the naive forecast 'tomorrow = today' on the last {N_PRICE - PRICE_TRAIN} days:"
          f" in-sample-style R^2 vs the mean benchmark"
          f" {out_of_sample_r2(price[te], naive_price, np.full(N_PRICE - PRICE_TRAIN, price[:PRICE_TRAIN].mean())):.6f}")
    print(f"  the SAME forecast against the naive benchmark (out-of-sample R^2):"
          f" {out_of_sample_r2(price[te], naive_price, naive_price):.6f} - by construction zero,"
          f" which is the honest score of 'tomorrow = today'")
    sig = lag_signature(naive_price, price[te])
    print("  corr(forecast_t, actual_(t-k)) for k = 0..5: "
          + ", ".join(f"k={k}: {v:.4f}" for k, v in sig.items()))
    print(f"  the signature of a price 'forecast' that is really a lag: the correlation at k=1"
          f" ({sig[1]:.4f}) is the largest, not k=0 ({sig[0]:.4f})")
    ret_naive = np.zeros(N_PRICE - PRICE_TRAIN)
    train_mean = np.full(N_PRICE - PRICE_TRAIN, ret[1:PRICE_TRAIN].mean())
    up = float(np.mean(ret[te] > 0))
    print(f"  the SAME model as a RETURN forecast predicts exactly zero every day: out-of-sample"
          f" R^2 against the training mean return"
          f" {out_of_sample_r2(ret[te], ret_naive, train_mean):.6f}, and it has no direction at"
          f" all - the share of up days is {up:.3f}, so 'long every day' is the accuracy any"
          f" directional claim has to beat")

    # ---- 2. MASE ----------------------------------------------------------------------------
    print("\n=== 2. MASE (Hyndman-Koehler 2006) and its own identity ===")
    tr = price[:PRICE_TRAIN]
    sc = scaling_factor(tr, 1)
    in_sample_naive_mase = mase(tr[1:], tr[:-1], sc)
    print(f"  scaling factor (in-sample naive MAE, m=1): {sc:.6f}")
    print(f"  MASE of the in-sample naive forecast: {in_sample_naive_mase:.10f}"
          f" - exactly 1 by construction; that is what makes MASE readable")
    print(f"  MASE of the naive forecast on the HELD-OUT days: "
          f"{mase(price[te], naive_price, sc):.4f} (not 1 - the scale of the moves changed)")

    # ---- 3. Rolling-origin walk-forward ----------------------------------------------------
    print(f"\n=== 3. Rolling-origin walk-forward, horizon {HORIZON}, one origin per step ===")
    for label, y, start, m, order, seas_order, refit, ets_kw in (
            (f"DGP A: daily price, origins {PRICE_TRAIN}..{N_PRICE - 1}",
             price, PRICE_TRAIN, 5, (1, 1, 1), (0, 0, 0, 0), REFIT_EVERY,
             {"seasonal_periods": None, "seasonal": None}),
            (f"DGP B: monthly seasonal (m={SEASON_M}), origins {SEASON_TRAIN}..{N_SEASON - 1}",
             simulate_seasonal(), SEASON_TRAIN, SEASON_M, (1, 0, 0), (1, 0, 0, SEASON_M), 24,
             {"seasonal_periods": SEASON_M, "seasonal": "add"})):
        print(f"  -- {label}")
        sc_l = scaling_factor(y[:start], 1)
        rows, errs = {}, {}
        for name, fn in BASELINES.items():
            fc, act = rolling_origin(y, lambda tr_, h_, _f=fn, _m=m: _f(tr_, h_, _m), start)
            rows[name] = score(act, fc, sc_l)
            errs[name] = act - fc
        ar = arima_walk_forward(y, start, order=order, seasonal_order=seas_order,
                                refit_every=refit)
        if ar is not None:
            rows[f"ARIMA{order}x{seas_order}"] = score(ar["actual"], ar["forecast"], sc_l)
            errs["ARIMA"] = ar["actual"] - ar["forecast"]
        et = ets_walk_forward(y, start, **ets_kw)
        if et is not None:
            rows["ETS (Holt-Winters)"] = score(et["actual"], et["forecast"], sc_l)
            errs["ETS"] = et["actual"] - et["forecast"]
        _table(rows)
        if ar is not None:
            print(f"     (ARIMA trend={ar['trend']!r}, refit {ar['n_fits']}x then extended with"
                  f" append(refit=False); ETS refit {et['n_fits']}x, one per origin)")
            dense = arima_walk_forward(y, start, order=order, seasonal_order=seas_order,
                                       refit_every=max(refit // 5, 1))
            print(f"     refit cadence is a choice, not a detail: the same ARIMA refit"
                  f" {dense['n_fits']}x instead of {ar['n_fits']}x scores MASE"
                  f" {score(dense['actual'], dense['forecast'], sc_l)['mase']:.4f}"
                  f" against {rows[f'ARIMA{order}x{seas_order}']['mase']:.4f}")
            stale = ets_walk_forward(y, start, refit_every=refit, **ets_kw)
            print(f"     and ETS with its smoothing parameters frozen between {refit}-origin"
                  f" refits ({stale['n_fits']} fits) scores MASE"
                  f" {score(stale['actual'], stale['forecast'], sc_l)['mase']:.4f} against"
                  f" {rows['ETS (Holt-Winters)']['mase']:.4f} refit at every origin - freezing"
                  f" them is a different estimator, not a cheaper one")
        # ---- 4. DM against the naive, on this DGP
        base = errs["naive"]
        print("     Diebold-Mariano against the naive forecast (negative = the model wins,"
              " squared loss, h=1, HLN corrected):")
        for name in [k for k in errs if k != "naive"]:
            dm = diebold_mariano(errs[name], base)
            better = "better" if dm["stat"] < 0 else "worse "
            print(f"        {name:<20} DM {dm['stat']:+7.3f}  p {dm['p_value']:.4f}  ({better}"
                  f" than naive, {'significant' if dm['p_value'] < 0.05 else 'NOT significant'}"
                  f" at 5 %)")

    # ---- 4b. Verifying the DM statistic -----------------------------------------------------
    print("\n=== 4. Is the Diebold-Mariano statistic right? ===")
    rng = np.random.default_rng(SEED)
    ea = rng.standard_normal(400)
    eb = rng.standard_normal(400) * 1.1
    chk = dm_matches_scipy_t_test(ea, eb)
    print(f"  at h=1, harvey=False, DM is a one-sample t-test on the loss differential with the"
          f" POPULATION variance:")
    print(f"     DM {chk['dm']:.6f}   scipy ttest_1samp {chk['scipy_t']:.6f}"
          f"   ratio {chk['ratio']:.10f}   sqrt(T/(T-1)) {chk['expected_ratio']:.10f}"
          f"   |difference| {chk['abs_error']:.1e}")
    e1 = rng.standard_normal(300)
    e2 = e1 + rng.standard_normal(300) * 0.3
    a, b = diebold_mariano(e1, e2), diebold_mariano(e2, e1)
    print(f"  antisymmetry: DM(e1, e2) {a['stat']:+.6f} = -DM(e2, e1) {b['stat']:+.6f};"
          f" same p-value {a['p_value']:.6f} / {b['p_value']:.6f}")
    sz = dm_size_under_the_null()
    print(f"  size under the null (two forecasts with equal expected loss, {sz['sims']} runs of"
          f" T={sz['T']}): rejects {sz['size']:.1%} at the {sz['alpha']:.0%} level"
          f" (+-{2 * sz['se']:.1%} at two standard errors)")

    # ---- 5. The ARIMA trend default ---------------------------------------------------------
    td = arima_trend_defaults()
    print("\n=== 5. statsmodels ARIMA: an integrated model has no drift by default ===")
    if td is None:
        print("  statsmodels not installed; skipped")
    else:
        print(f"  a random walk with drift {td['true_slope']} per step (realised sample drift"
              f" {td['sample_drift']:+.5f}), 50-step forecasts:")
        default_key, trend_key = "d=1, default trend", "d=1, trend='t'"
        for label in ("d=0 on the level", default_key, trend_key):
            r = td[label]
            print(f"     {label:<20} trend={r['trend']!r:<5} params {r['params']}"
                  f"  forecast slope {r['slope']:+.5f}")
        print(f"  ARIMA(1,1,0) with the default trend forecasts a FLAT line (slope"
              f" {td[default_key]['slope']:+.5f}) on a series that drifts"
              f" {td['sample_drift']:+.5f} per step in sample; trend='t' recovers"
              f" {td[trend_key]['slope']:+.5f}")
        if td.get("naive_identity") is not None:
            ni = td["naive_identity"]
            print(f"  and ARIMA(0,1,0) with the default trend IS the naive forecast: max"
                  f" |forecast - last observation| over {ni['n']} origins = {ni['max_abs']:.1e}")

    print("\nRule: score every forecast against the naive one with MASE and a Diebold-Mariano"
          " test, and never report an R^2 on a price level.")
    print(f"total runtime {time.time() - t_all:.1f}s")
