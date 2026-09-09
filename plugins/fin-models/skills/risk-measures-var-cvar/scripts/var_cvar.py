"""Value-at-risk and expected shortfall: four estimators, two backtests, and the scaling trap.

WHY this exists: "the 99 % VaR is 2.3 %" is not a fact until three things are stated - the
estimator, the horizon, and how the number was scaled to that horizon. The estimators below
disagree exactly in the tail you care about, a rolling window passes the exception COUNT
while failing the exception TIMING, and sqrt(10) x (1-day VaR) is wrong in both directions
depending on what the data are doing. Every number is measured on a seeded GARCH(1,1) series
with Student-t innovations, whose conditional quantile is known exactly because we simulated it.

SIGN CONVENTION: every VaR and ES here is a POSITIVE LOSS; `losses = -returns`. quantstats
and empyrical return negative numbers for the same quantity; say which you mean.

What the module provides:

    var_es_historical        empirical quantile of losses and the mean beyond it
    var_es_normal            mu + sigma z, and the closed-form normal expected shortfall
    var_es_cornish_fisher    the third-order skew/kurtosis expansion of the quantile, and the
                             expected shortfall of that expansion in closed form
    cornish_fisher_is_monotone  the expansion stops being a quantile function at large kurtosis
    var_es_evt               peaks-over-threshold with a generalized Pareto tail
                             (scipy.stats.genpareto, loc fixed at 0), McNeil-Frey VaR and ES
    kupiec_pof               unconditional coverage LR test, chi2(1)
    kupiec_via_binomial      the same statistic through scipy's binomial log-pmf, as a check
    christoffersen_independence, conditional_coverage   Markov independence LR, chi2(1); joint chi2(2)
    backtest_size_under_null the measured size of all three tests on iid Bernoulli hits
    simulate_garch_t         the DGP, returning the true conditional sigma_t
    rolling_var              one-step-ahead VaR from a rolling window, seven methods
    horizon_var_from_state   h-day VaR by simulation from a given conditional variance
    scale_var_sqrt_time      the guard: refuses sqrt(h) scaling unless the assumption is asserted
    discrete_var_es          exact VaR / ES of a discrete distribution (the subadditivity example)

Run:  python var_cvar.py     (numpy / scipy; fixed seeds; about 10 s)
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy import stats
from scipy.special import xlogy

LEVEL = 0.99
HORIZON = 10
WINDOW = 500
N_TEST = 2000
LAMBDA = 0.94
SEED = 0
GARCH = {"alpha": 0.08, "beta": 0.90, "nu": 5.0, "uncond_vol": 0.01}


# ------------------------------------------------------------------ estimators ---------------
def losses(returns: np.ndarray) -> np.ndarray:
    return -np.asarray(returns, float)


def var_es_historical(loss: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    """Empirical quantile (numpy's default linear interpolation) and the mean of losses at
    or beyond it."""
    loss = np.asarray(loss, float)
    var = float(np.quantile(loss, level))
    return var, float(loss[loss >= var].mean())


def var_es_normal(loss: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    loss = np.asarray(loss, float)
    mu, sd = loss.mean(), loss.std(ddof=1)
    z = stats.norm.ppf(level)
    return float(mu + sd * z), float(mu + sd * stats.norm.pdf(z) / (1.0 - level))


def cornish_fisher_z(z, skew: float, exkurt: float):
    """z + (z^2 - 1) S/6 + (z^3 - 3z) K/24 - (2z^3 - 5z) S^2/36, K = EXCESS kurtosis."""
    z = np.asarray(z, float)
    return (z + (z ** 2 - 1.0) * skew / 6.0 + (z ** 3 - 3.0 * z) * exkurt / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * skew ** 2 / 36.0)


def cornish_fisher_is_monotone(skew: float, exkurt: float,
                               z_range: tuple[float, float] = (-3.5, 3.5)) -> bool:
    """Is the expansion still a quantile function over z_range?

    At skew 0 the derivative is 1 + (3z^2 - 3) K/24, minimised at z = 0 where it is 1 - K/8:
    the closed-form break is EXCESS KURTOSIS 8. Beyond it the "99 % VaR" can be smaller than
    the 97.5 % one, which is not a rounding problem - it is not a quantile any more.
    """
    z = np.linspace(z_range[0], z_range[1], 1401)
    return bool(np.all(np.diff(cornish_fisher_z(z, skew, exkurt)) > 0))


def var_es_cornish_fisher(loss: np.ndarray, level: float = LEVEL) -> tuple[float, float, dict]:
    """Cornish-Fisher VaR, and the expected shortfall of the expanded quantile function in
    closed form: with z_cf(z) = a0 + a1 z + a2 z^2 + a3 z^3 and phi the normal density,
    ES = mu + sd [a0 + a2 + (a1 + a2 z_p + a3 (z_p^2 + 2)) phi(z_p) / (1 - p)]."""
    loss = np.asarray(loss, float)
    mu, sd = loss.mean(), loss.std(ddof=1)
    s, k = float(stats.skew(loss)), float(stats.kurtosis(loss))
    z = stats.norm.ppf(level)
    phi = stats.norm.pdf(z)
    a0, a1, a2, a3 = -s / 6.0, 1.0 - k / 8.0 + 5.0 * s ** 2 / 36.0, s / 6.0, k / 24.0 - s ** 2 / 18.0
    var = mu + sd * cornish_fisher_z(z, s, k)
    es = mu + sd * (a0 + a2 + (a1 + a2 * z + a3 * (z ** 2 + 2.0)) * phi / (1.0 - level))
    return float(var), float(es), {"skew": s, "exkurt": k, "monotone": cornish_fisher_is_monotone(s, k)}


def fit_gpd(loss: np.ndarray, threshold: float) -> tuple[float, float, int]:
    """Generalized Pareto fit to the exceedances over `threshold` with loc fixed at 0.
    scipy's genpareto has density (1 + c x)^(-1 - 1/c) / scale: c is the tail index xi."""
    exc = np.asarray(loss, float)
    exc = exc[exc > threshold] - threshold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xi, _, beta = stats.genpareto.fit(exc, floc=0)
    return float(xi), float(beta), int(len(exc))


def var_es_evt(loss: np.ndarray, level: float = LEVEL,
               threshold_quantile: float = 0.90) -> tuple[float, float, dict]:
    """Peaks-over-threshold VaR and ES (McNeil-Frey form):
    VaR = u + beta/xi [((n/N_u)(1 - p))^-xi - 1],  ES = VaR/(1 - xi) + (beta - xi u)/(1 - xi)."""
    if level <= threshold_quantile:
        raise ValueError("level must lie beyond the threshold quantile")
    loss = np.asarray(loss, float)
    u = float(np.quantile(loss, threshold_quantile))
    xi, beta, n_u = fit_gpd(loss, u)
    tail_prob = (1.0 - level) * len(loss) / n_u
    if abs(xi) < 1e-10:
        var = u - beta * math.log(tail_prob)
    else:
        var = u + beta / xi * (tail_prob ** (-xi) - 1.0)
    es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi) if xi < 1.0 else math.inf
    return float(var), float(es), {"xi": xi, "beta": beta, "threshold": u, "n_exceedances": n_u}


def gpd_es_numeric(var: float, xi: float, beta: float, threshold: float) -> float:
    """E[L | L > VaR] by numerical integration of the fitted tail, to check the closed form."""
    y0 = var - threshold
    return threshold + stats.genpareto.expect(lambda y: y, args=(xi,), loc=0.0, scale=beta,
                                              lb=y0, conditional=True)


# ------------------------------------------------------------------ backtests ----------------
def kupiec_pof(n_exceptions: int, n_obs: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood ratio, chi2 with 1 degree of freedom.

    LR_uc = -2 ln[ (1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x) ]. The binomial coefficient is the
    same under both hypotheses and cancels, so this equals
    -2 (binom.logpmf(x, n, p) - binom.logpmf(x, n, x/n)); `kupiec_via_binomial` checks that.
    """
    x, n = int(n_exceptions), int(n_obs)
    ll0 = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1.0 - x / n) + xlogy(x, x / n)
    lr = float(-2.0 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, 1))


def kupiec_via_binomial(n_exceptions: int, n_obs: int, p: float) -> float:
    """The same statistic through scipy's binomial log-pmf: an independent path to the number."""
    x, n = int(n_exceptions), int(n_obs)
    return float(-2.0 * (stats.binom.logpmf(x, n, p) - stats.binom.logpmf(x, n, x / n)))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float, dict]:
    """Christoffersen (1998) independence test: a first-order Markov chain on the hit
    sequence against a constant hit probability. n_ij counts transitions i -> j."""
    h = np.asarray(hits).astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p0, p1):
        return xlogy(n00, 1.0 - p0) + xlogy(n01, p0) + xlogy(n10, 1.0 - p1) + xlogy(n11, p1)

    lr = float(-2.0 * (ll(pi, pi) - ll(pi01, pi11)))
    return lr, float(stats.chi2.sf(lr, 1)), {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
                                             "pi01": pi01, "pi11": pi11, "pi": pi}


def conditional_coverage(hits: np.ndarray, p: float) -> dict:
    """LR_cc = LR_uc + LR_ind, chi2 with 2 degrees of freedom."""
    h = np.asarray(hits).astype(int)
    lr_uc, p_uc = kupiec_pof(int(h.sum()), len(h), p)
    lr_ind, p_ind, counts = christoffersen_independence(h)
    lr_cc = lr_uc + lr_ind
    return {"n_exceptions": int(h.sum()), "expected": p * len(h), "n_obs": len(h),
            "lr_uc": lr_uc, "p_uc": p_uc, "lr_ind": lr_ind, "p_ind": p_ind,
            "lr_cc": lr_cc, "p_cc": float(stats.chi2.sf(lr_cc, 2)), **counts}


def backtest_size_under_null(p: float = 0.01, n_obs: int = 500, n_seq: int = 10000,
                             alpha: float = 0.05, seed: int = 0) -> dict:
    """Rejection rates of the three tests on iid Bernoulli(p) hit sequences: the size."""
    rng = np.random.default_rng(seed)
    hits = (rng.random((n_seq, n_obs)) < p).astype(int)
    a, b = hits[:, :-1], hits[:, 1:]
    n00 = ((a == 0) & (b == 0)).sum(1)
    n01 = ((a == 0) & (b == 1)).sum(1)
    n10 = ((a == 1) & (b == 0)).sum(1)
    n11 = ((a == 1) & (b == 1)).sum(1)
    x = hits.sum(1)
    lr_uc = -2.0 * (xlogy(n_obs - x, 1 - p) + xlogy(x, p)
                    - xlogy(n_obs - x, 1 - x / n_obs) - xlogy(x, x / n_obs))
    with np.errstate(divide="ignore", invalid="ignore"):
        pi01 = np.where(n00 + n01 > 0, n01 / (n00 + n01), 0.0)
        pi11 = np.where(n10 + n11 > 0, n11 / (n10 + n11), 0.0)
        pi = (n01 + n11) / (n_obs - 1)
    l0 = xlogy(n00, 1 - pi) + xlogy(n01, pi) + xlogy(n10, 1 - pi) + xlogy(n11, pi)
    l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr_ind = -2.0 * (l0 - l1)
    c1, c2 = stats.chi2.ppf(1 - alpha, 1), stats.chi2.ppf(1 - alpha, 2)
    return {"uc": float((lr_uc > c1).mean()), "ind": float((lr_ind > c1).mean()),
            "cc": float((lr_uc + lr_ind > c2).mean()), "nominal": alpha, "n_seq": n_seq,
            "n_obs": n_obs, "p": p}


# ------------------------------------------------------------------ the DGP ------------------
def t_quantile_unit(level: float, nu: float) -> float:
    """Quantile of a Student-t rescaled to unit variance."""
    return float(stats.t.ppf(level, nu) * math.sqrt((nu - 2.0) / nu))


def t_es_unit(level: float, nu: float) -> float:
    """Expected shortfall of the unit-variance Student-t: c (nu + q^2)/(nu - 1) f(q)/(1 - p)."""
    q = stats.t.ppf(level, nu)
    return float(math.sqrt((nu - 2.0) / nu) * (nu + q ** 2) / (nu - 1.0) * stats.t.pdf(q, nu) / (1.0 - level))


def simulate_garch_t(n: int, alpha: float = GARCH["alpha"], beta: float = GARCH["beta"],
                     nu: float = GARCH["nu"], uncond_vol: float = GARCH["uncond_vol"],
                     seed: int = SEED, burn: int = 1000,
                     sigma2_0: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) with unit-variance Student-t innovations. Returns (returns, sigma) where
    sigma[t] is the conditional volatility of day t, known at the close of t-1."""
    rng = np.random.default_rng(seed)
    omega = uncond_vol ** 2 * (1.0 - alpha - beta)
    z = (rng.standard_t(nu, n + burn) * math.sqrt((nu - 2.0) / nu)).tolist()
    s2 = uncond_vol ** 2 if sigma2_0 is None else float(sigma2_0)
    r, sig = [], []
    for zt in z:
        s = math.sqrt(s2)
        rt = s * zt
        r.append(rt)
        sig.append(s)
        s2 = omega + alpha * rt * rt + beta * s2
    return np.array(r[burn:]), np.array(sig[burn:])


def horizon_var_from_state(sigma2_0: float, h: int, level: float, n_paths: int = 200_000,
                           alpha: float = GARCH["alpha"], beta: float = GARCH["beta"],
                           nu: float = GARCH["nu"], uncond_vol: float = GARCH["uncond_vol"],
                           seed: int = 1) -> float:
    """h-day loss quantile (sum of daily returns) by simulation from conditional variance sigma2_0."""
    rng = np.random.default_rng(seed)
    omega = uncond_vol ** 2 * (1.0 - alpha - beta)
    c = math.sqrt((nu - 2.0) / nu)
    s2 = np.full(n_paths, float(sigma2_0))
    cum = np.zeros(n_paths)
    for _ in range(h):
        r = np.sqrt(s2) * rng.standard_t(nu, n_paths) * c
        cum += r
        s2 = omega + alpha * r ** 2 + beta * s2
    return float(np.quantile(-cum, level))


def ewma_variance_path(x: np.ndarray, lam: float = LAMBDA, init: float | None = None) -> np.ndarray:
    """s2[t] = forecast variance for period t from x[:t]; s2[0] = init or x[0]^2."""
    x = np.asarray(x, float)
    s2 = np.empty(len(x) + 1)
    s2[0] = x[0] ** 2 if init is None else init
    for t in range(len(x)):
        s2[t + 1] = lam * s2[t] + (1.0 - lam) * x[t] ** 2
    return s2


METHODS = ("historical", "normal", "cornish_fisher", "evt", "ewma_normal", "ewma_fhs", "oracle")


def rolling_var(loss: np.ndarray, window: int, level: float, method: str, refit_every: int = 10,
                lam: float = LAMBDA, sigma: np.ndarray | None = None,
                nu: float | None = None) -> np.ndarray:
    """One-step-ahead VaR for days window..T-1, each from the previous `window` losses.

    'ewma_normal' / 'ewma_fhs' use the RiskMetrics variance path (normal quantile, or the
    empirical quantile of the standardized losses in the window: filtered historical
    simulation). 'evt' refits the GPD every `refit_every` days. 'oracle' uses the true
    conditional sigma and the Student-t quantile: what a perfect conditional model would do.
    """
    loss = np.asarray(loss, float)
    n = len(loss)
    out = np.full(n - window, np.nan)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method.startswith("ewma"):
        s2 = ewma_variance_path(loss, lam, init=float(loss[:window].var()))
    z = stats.norm.ppf(level)
    cached = None
    for i, t in enumerate(range(window, n)):
        win = loss[t - window:t]
        if method == "historical":
            out[i] = np.quantile(win, level)
        elif method == "normal":
            out[i] = win.mean() + win.std(ddof=1) * z
        elif method == "cornish_fisher":
            out[i] = var_es_cornish_fisher(win, level)[0]
        elif method == "evt":
            if i % refit_every == 0:
                cached = var_es_evt(win, level)[0]
            out[i] = cached
        elif method == "ewma_normal":
            out[i] = math.sqrt(s2[t]) * z
        elif method == "ewma_fhs":
            zstd = win / np.sqrt(s2[t - window:t])
            out[i] = math.sqrt(s2[t]) * np.quantile(zstd, level)
        elif method == "oracle":
            out[i] = sigma[t] * t_quantile_unit(level, nu)
    return out


def scale_var_sqrt_time(var_1d: float, horizon: int, *, iid_normal_zero_mean: bool = False) -> float:
    """sqrt(h) scaling, refused unless the assumption that makes it exact is asserted."""
    if not iid_normal_zero_mean:
        raise ValueError(
            "sqrt(h) scaling of a 1-day VaR is exact only for iid normal zero-mean returns. Under "
            "fat tails it overstates the h-day quantile and under volatility clustering it "
            "understates it (both measured in this module's demo). Simulate the horizon or use a "
            "horizon model; pass iid_normal_zero_mean=True to assert the assumption explicitly.")
    return float(var_1d) * math.sqrt(horizon)


def discrete_var_es(values: np.ndarray, probs: np.ndarray, level: float) -> tuple[float, float]:
    """Exact VaR (smallest v with P(L <= v) >= level) and the Rockafellar-Uryasev expected
    shortfall of a discrete loss distribution, the atom at VaR split fractionally."""
    values, probs = np.asarray(values, float), np.asarray(probs, float)
    order = np.argsort(values)
    v, p = values[order], probs[order]
    cdf = np.cumsum(p)
    var = float(v[np.searchsorted(cdf, level - 1e-12, side="left")])
    above = v > var
    es = ((p[above] * v[above]).sum() + (1.0 - level - p[above].sum()) * var) / (1.0 - level)
    return var, float(es)


# ------------------------------------------------------------------ demo ---------------------
if __name__ == "__main__":
    t0 = time.time()
    g = GARCH
    nu = g["nu"]
    print("=" * 100)
    print(f"DGP: GARCH(1,1) alpha {g['alpha']} beta {g['beta']}, unit-variance Student-t({nu:.0f}) innovations,"
          f" unconditional vol {g['uncond_vol']:.0%}/day; losses are positive numbers")
    print("=" * 100)
    r, sig = simulate_garch_t(WINDOW + N_TEST, seed=SEED)
    loss = losses(r)
    r_long, _ = simulate_garch_t(1_000_000, seed=99)
    loss_long = losses(r_long)
    true_var, true_es = var_es_historical(loss_long, LEVEL)

    # ---------------------------------------------------------------- 1. unconditional estimators
    print(f"\n1. UNCONDITIONAL {LEVEL:.0%} 1-DAY VaR / ES ON THE {len(loss):,}-DAY SAMPLE vs THE TRUTH"
          f" (a 1,000,000-day path of the same DGP)")
    print(f"{'estimator':<34} {'VaR %':>7} {'ES %':>7} {'VaR err':>8} {'ES err':>8}")
    print("-" * 100)
    h_var, h_es = var_es_historical(loss, LEVEL)
    n_var, n_es = var_es_normal(loss, LEVEL)
    c_var, c_es, c_info = var_es_cornish_fisher(loss, LEVEL)
    e_var, e_es, e_info = var_es_evt(loss, LEVEL)
    for name, v, e in (("historical", h_var, h_es), ("parametric normal", n_var, n_es),
                       ("Cornish-Fisher", c_var, c_es), ("EVT POT, u = 90th pct", e_var, e_es),
                       ("TRUE (1e6-day simulation)", true_var, true_es)):
        print(f"{name:<34} {100 * v:>7.3f} {100 * e:>7.3f} {v / true_var - 1:>+8.1%} {e / true_es - 1:>+8.1%}")
    print("-" * 100)
    print(f"sample skew {c_info['skew']:+.2f}, excess kurtosis {c_info['exkurt']:.2f}; Cornish-Fisher quantile"
          f" function monotone: {c_info['monotone']}")
    k_break = next(k for k in np.arange(0.0, 20.0, 0.05) if not cornish_fisher_is_monotone(0.0, k))
    print(f"with zero skew the Cornish-Fisher expansion stops being monotone at excess kurtosis"
          f" {k_break:.2f} on a 0.05 grid; the closed form is 8 exactly (derivative 1 - K/8 at"
          f" z = 0). A Student-t(4.5) has excess kurtosis 12, so a daily equity series is past"
          f" it, and the expansion then returns a 99 % 'VaR' below the 97.5 % one.")
    conv = []
    for n_w in (len(loss), 500, 100):
        w = loss[-n_w:]
        hv, he = var_es_historical(w, LEVEL)
        rv, re_ = discrete_var_es(w, np.full(n_w, 1.0 / n_w), LEVEL)
        conv.append(f"n={n_w}: VaR {100 * hv:.4f}/{100 * rv:.4f}, ES {100 * he:.4f}/{100 * re_:.4f}")
    print("historical ES conventions (mean of losses at or beyond the interpolated quantile /"
          " Rockafellar-Uryasev")
    print("with the atom at VaR split fractionally), as % of notional: " + "; ".join(conv) + ".")
    print("They agree to four decimals on every window here and the VaR differs only in the"
          " fourth; state which you")
    print("used anyway, because a library that reports the other one will not tie out with"
          " yours to the digit.")
    print(f"GPD tail index xi = {e_info['xi']:.3f}, scale {e_info['beta']:.5f},"
          f" {e_info['n_exceedances']} exceedances; closed-form ES {100 * e_es:.4f} % vs"
          f" numerical integration of the fitted tail"
          f" {100 * gpd_es_numeric(e_var, e_info['xi'], e_info['beta'], e_info['threshold']):.4f} %")
    print(f"true Student-t({nu:.0f}) conditional quantile at 99%: {t_quantile_unit(LEVEL, nu):.3f} sigma"
          f" (normal 2.326); conditional ES {t_es_unit(LEVEL, nu):.3f} sigma (normal 2.665)")

    # ---------------------------------------------------------------- 2. rolling backtest
    print(f"\n2. ROLLING 1-DAY {LEVEL:.0%} VaR, {WINDOW}-DAY WINDOW, {N_TEST:,} TEST DAYS:"
          f" exceptions and the backtests")
    print(f"{'method':<22} {'exceptions':>10} {'expected':>8} {'Kupiec p':>9} {'indep. p':>9} {'cc p':>7} "
          f"{'n11':>4} {'pi11':>6} {'mean VaR %':>10}")
    print("-" * 100)
    test_loss = loss[WINDOW:]
    labels = {"historical": "historical", "normal": "normal", "cornish_fisher": "Cornish-Fisher",
              "evt": "EVT POT (refit 10d)", "ewma_normal": "EWMA(0.94) normal",
              "ewma_fhs": "EWMA(0.94) FHS", "oracle": "oracle: true sigma_t x t-q"}
    bt, mean_var = {}, {}
    for m in METHODS:
        v = rolling_var(loss, WINDOW, LEVEL, m, sigma=sig, nu=nu)
        hits = (test_loss > v).astype(int)
        res = conditional_coverage(hits, 1.0 - LEVEL)
        bt[m], mean_var[m] = res, float(v.mean())
        print(f"{labels[m]:<22} {res['n_exceptions']:>10} {res['expected']:>8.0f}"
              f" {res['p_uc']:>9.3f} {res['p_ind']:>9.3f} {res['p_cc']:>7.3f} {res['n11']:>4}"
              f" {res['pi11']:>6.2f} {100 * v.mean():>10.3f}")
    print("-" * 100)
    print("Reading it: the parametric normal fails on COUNT (36 against 20 expected, Kupiec p"
          " 0.001) because the")
    print("innovations are fat-tailed; the EWMA(0.94) filter fixes the TIMING but its normal"
          " quantile still fails the")
    print("count (32, p 0.013); only filtered historical simulation - the EWMA scale with the"
          " EMPIRICAL quantile of")
    print("the standardized residuals - passes everything (21 exceptions, Kupiec p 0.824, cc p"
          " 0.455). Cornish-Fisher")
    print(f"errs the other way: it holds {mean_var['cornish_fisher'] / mean_var['oracle'] - 1:.0%}"
          f" more capital than the oracle's mean VaR, for 15 exceptions instead of 20.")
    print("AND: the independence test rejects NOTHING here, although pi11 = P(exception |"
          " exception yesterday) runs")
    print("4 to 7 times the unconditional 1 %. With about 20 exceptions in 2,000 days there are"
          " one or two consecutive")
    print("pairs to test on, and section 3 measures the test's actual size at 1.4 % against a"
          " nominal 5 %. A passed")
    print("independence test at the 99 % level is close to no evidence at all.")

    # ---------------------------------------------------------------- 3. size of the tests
    size = backtest_size_under_null()
    print(f"\n3. SIZE OF THE TESTS UNDER THE NULL: {size['n_seq']:,} iid Bernoulli({size['p']})"
          f" sequences of {size['n_obs']} days, nominal {size['nominal']:.0%}")
    print(f"   rejection rates: Kupiec {size['uc']:.1%}, independence {size['ind']:.1%},"
          f" conditional coverage {size['cc']:.1%}  (discrete hit counts make the chi2"
          f" approximation coarse at 1 %)")
    lr_a, _ = kupiec_pof(28, 2000, 0.01)
    print(f"   the Kupiec statistic through two paths: this module {lr_a:.10f}, scipy's binomial"
          f" log-pmf {kupiec_via_binomial(28, 2000, 0.01):.10f}"
          f" (difference {abs(lr_a - kupiec_via_binomial(28, 2000, 0.01)):.1e})")
    lr_exact, p_exact = kupiec_pof(20, 2000, 0.01)
    lr_reg, p_reg, _ = christoffersen_independence(np.tile([1, 0, 0, 0, 0], 100))
    lr_blk, p_blk, c_blk = christoffersen_independence(np.tile([1, 1, 0, 0], 100))
    print(f"   LR_uc is exactly 0 at its null (x = p*n): {abs(lr_exact):.1e}, p {p_exact:.3f}.")
    print(f"   the independence test fires on REGULARITY too: one exception every fifth day"
          f" gives LR_ind {lr_reg:.1f}, p {p_reg:.3f} - 'independence rejected' does not tell"
          f" you the hits were clustered.")
    print(f"   and it is blind whenever pi01 = pi11: the block pattern 1,1,0,0 repeated has"
          f" pi01 = {c_blk['pi01']:.2f} = pi11 = {c_blk['pi11']:.2f}, so LR_ind = {lr_blk:.1e},"
          f" p {p_blk:.3f} - a perfectly predictable sequence passes. The Markov alternative"
          f" only looks one day back.")

    # ---------------------------------------------------------------- 4. sqrt(h)
    print(f"\n4. sqrt({HORIZON}) x 1-DAY VaR vs THE TRUE {HORIZON}-DAY VaR at {LEVEL:.0%}"
          f" (loss = sum of daily returns)")
    print(f"{'returns':<48} {'1-day VaR %':>11} {'sqrt(h) x':>10} {'true h-day %':>12} {'error':>8}")
    print("-" * 100)
    rng = np.random.default_rng(5)
    n_blocks = len(loss_long) // HORIZON
    rows = []
    normal = rng.normal(0.0, g["uncond_vol"], n_blocks * HORIZON)
    v1 = float(np.quantile(-normal, LEVEL))
    vh = float(np.quantile(-normal.reshape(-1, HORIZON).sum(1), LEVEL))
    rows.append(("iid normal, 1 %/day", v1, vh))
    tt = rng.standard_t(nu, n_blocks * HORIZON) * math.sqrt((nu - 2) / nu) * g["uncond_vol"]
    v1 = float(np.quantile(-tt, LEVEL))
    vh = float(np.quantile(-tt.reshape(-1, HORIZON).sum(1), LEVEL))
    rows.append((f"iid Student-t({nu:.0f}), 1 %/day (fat tails only)", v1, vh))
    vh = float(np.quantile(loss_long[:n_blocks * HORIZON].reshape(-1, HORIZON).sum(1), LEVEL))
    rows.append(("GARCH-t, unconditional (clustering + fat tails)", true_var, vh))
    for mult in (0.5, 1.0, 2.0):
        s2_0 = (mult * g["uncond_vol"]) ** 2
        v1 = mult * g["uncond_vol"] * t_quantile_unit(LEVEL, nu)
        vh = horizon_var_from_state(s2_0, HORIZON, LEVEL)
        rows.append((f"GARCH-t, conditional, sigma_t = {mult:.1f} x long-run", v1, vh))
    for name, v1, vh in rows:
        scaled = scale_var_sqrt_time(v1, HORIZON, iid_normal_zero_mean=True)
        print(f"{name:<48} {100 * v1:>11.3f} {100 * scaled:>10.3f} {100 * vh:>12.3f} {scaled / vh - 1:>+8.1%}")
    print("-" * 100)
    print("The sign of the error is not fixed: fat tails alone make sqrt(h) OVERSTATE (the sum of"
          " h draws is closer")
    print("to normal than one draw is), while starting from a quiet day makes it UNDERSTATE,"
          " because volatility")
    print("mean-reverts upward over the horizon. 'Conservative' is not a property of sqrt(h)"
          " scaling.")
    print(f"\n   the same error as the horizon grows, at {LEVEL:.0%}:")
    print(f"   {'state':<44} " + " ".join(f"{f'h={h}':>9}" for h in (5, 10, 25, 60)))
    for mult in (0.5, 2.0):
        s2_0 = (mult * g["uncond_vol"]) ** 2
        v1 = mult * g["uncond_vol"] * t_quantile_unit(LEVEL, nu)
        errs = []
        for h in (5, 10, 25, 60):
            vh = horizon_var_from_state(s2_0, h, LEVEL, n_paths=100_000)
            errs.append(v1 * math.sqrt(h) / vh - 1.0)
        print(f"   {f'GARCH-t, sigma_t = {mult:.1f} x long-run':<44} "
              + " ".join(f"{e:>+9.1%}" for e in errs))
    errs = []
    for h in (5, 10, 25, 60):
        n_b = len(tt) // h
        vh = float(np.quantile(-tt[:n_b * h].reshape(-1, h).sum(1), LEVEL))
        errs.append(float(np.quantile(-tt, LEVEL)) * math.sqrt(h) / vh - 1.0)
    print(f"   {f'iid Student-t({nu:.0f}), no clustering':<44} "
          + " ".join(f"{e:>+9.1%}" for e in errs))
    try:
        scale_var_sqrt_time(true_var, HORIZON)
    except ValueError as e:
        print(f"\nthe guard: scale_var_sqrt_time(var, 10) -> ValueError: {str(e)[:88]}...")

    # ---------------------------------------------------------------- 5. subadditivity
    print("\n5. VaR IS NOT SUBADDITIVE, ES IS: two independent positions, each loses 1.0 with probability 2 %")
    lvl = 0.975
    va, ea = discrete_var_es([0.0, 1.0], [0.98, 0.02], lvl)
    vs, es_ = discrete_var_es([0.0, 1.0, 2.0], [0.98 ** 2, 2 * 0.98 * 0.02, 0.02 ** 2], lvl)
    print(f"   {lvl:.1%} VaR: each {va:.3f}, sum of the two {2 * va:.3f}, the combined book {vs:.3f}"
          f"  -> VaR(A+B) > VaR(A) + VaR(B): diversifying LOOKS riskier")
    print(f"   {lvl:.1%} ES : each {ea:.3f}, sum {2 * ea:.3f}, combined {es_:.3f}  -> ES(A+B) <= ES(A) + ES(B)")

    print("\n" + "=" * 100)
    print("THE RULE:  name the estimator, the horizon and the scaling with every VaR; backtest the timing of")
    print("           exceptions, not only their count; never sqrt(h) a fat-tailed, clustered series;")
    print("           optimize and report ES, quote VaR only when a regulator asks for it.")
    print("=" * 100)
    print(f"total runtime {time.time() - t0:.1f}s")
