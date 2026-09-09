#!/usr/bin/env python3
"""Monte Carlo that converges to the right number: variance reduction, LSM, QMC, and bias.

Four things, each verified against something that is not another Monte Carlo:

1. VARIANCE REDUCTION on a discretely monitored arithmetic-average Asian call. Antithetic
   variates, a control variate (the GEOMETRIC-average Asian, which is exactly lognormal and
   therefore has a closed form - Kemna & Vorst 1990) and stratified sampling, each with a
   measured variance-reduction factor at the same path budget.

2. LONGSTAFF & SCHWARTZ (2001), "Valuing American Options by Simulation: A Simple
   Least-Squares Approach", Review of Financial Studies 14(1), 113-147, read in full at
   galton.uchicago.edu/~mykland/346W07/Longstaff.pdf. Two things are reproduced from it:

     * the eight-path worked example of section 1. The paper prints the two regressions,
       "E[Y|X] = -1.070 + 2.983X - 1.813X^2" at t=2 and "E[Y|X] = 2.038 - 3.335X + 1.356X^2"
       at t=1, the resulting stopping rule, and "a value of .1144 for the American put ...
       roughly twice the value of .0564 for the European put". All of it comes back exactly.
     * rows of its Table 1: an American put, K=40, r=.06, "exercisable 50 times per year",
       "100,000 (50,000 plus 50,000 antithetic) paths", basis "a constant and the first three
       Laguerre polynomials".

   And two traps the paper itself names:

     * footnote 9, verbatim: "if the American option were valued by taking the maximum of the
       immediate exercise value and the estimated continuation value, and discounting this
       value back, the resulting American option value could be severely upward biased. This
       bias arises since the maximum operator is convex; measurement error in the estimated
       continuation value results in the maximum operator being upward biased." Measured.
     * section 1: "We use only in-the-money paths since it allows us to better estimate the
       conditional expectation function in the region where exercise is relevant and
       significantly improves the efficiency of the algorithm." Measured.

   Cross-checked against a CRR binomial tree, which is the reference used by
   ../../../fin-models/skills/option-pricing-models/SKILL.md.

3. QUASI-MONTE CARLO with scipy.stats.qmc.Sobol. Source-verified in the installed scipy
   1.13.0: the constructor is Sobol(d, *, scramble=True, bits=None, seed=None,
   optimization=None) - scrambling is ON by default, "If True, use LMS+shift scrambling" -
   "Max dimensionality is 21201", and the class carries the warning "Sobol' sequences are a
   quadrature rule and they lose their balance properties if one uses a sample size that is
   not a power of 2, or skips the first point, or thins the sequence". Calling .random(5)
   raises UserWarning: "The balance properties of Sobol' points require n to be a power of 2."

   !! An UNSCRAMBLED Sobol sequence starts at exactly (0, 0, ...), and norm.ppf(0) = -inf.
      The demo measures what that does to a price.

4. THE BIAS THE STANDARD ERROR CANNOT SEE. A down-and-out call has a closed form under
   CONTINUOUS monitoring; simulate it with m monitoring dates and you price a different
   contract, because the path can dip under the barrier and come back between two dates. More
   paths shrink the error bar and do nothing to that. Broadie, Glasserman & Kou's continuity
   correction - shift the barrier to H*exp(-beta*sigma*sqrt(T/m)) for a down barrier, with
   beta = -zeta(1/2)/sqrt(2*pi) = 0.5826 - removes most of it, and the demo verifies the
   constant against scipy.special.zeta.

Run:  python monte_carlo.py       (numpy + scipy, fixed seed, about 50 s)
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import special, stats
from scipy.stats import qmc

SEED = 20260909
# Broadie, Glasserman & Kou's continuity-correction constant, -zeta(1/2)/sqrt(2*pi).
#
# !! scipy.special.zeta(0.5, 1) is NAN. The Hurwitz form does not carry the analytic
#    continuation below 1; scipy.special.zetac - "Riemann zeta function minus 1 ... For x < 1
#    the analytic continuation is computed" - does, and zetac(0.5) + 1 = -1.4603545088095866.
#    A nan here does not raise: it flows into exp(), then into a barrier level, and only
#    surfaces as a comparison that is mysteriously False.
ZETA_HALF = float(special.zetac(0.5) + 1.0)
BGK_BETA = float(-ZETA_HALF / math.sqrt(2.0 * math.pi))

# Longstaff & Schwartz (2001) section 1: the eight sample paths, K = 1.10, r = 6%, three
# exercise dates. The paper prints only the in-the-money X and Y columns; this matrix is the
# unique one consistent with all of them, and reproducing the paper's two regressions and its
# .1144 from it is the check that it is right.
LS_PATHS = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34],
])
LS_COEF_T2 = (-1.070, 2.983, -1.813)      # the paper's printed regression at time 2
LS_COEF_T1 = (2.038, -3.335, 1.356)       # ... and at time 1
LS_AMERICAN, LS_EUROPEAN = 0.1144, 0.0564

# Longstaff & Schwartz (2001) Table 1, K=40, r=.06, 50 exercise points per year:
# (S, sigma, T) -> (finite-difference American, closed-form European, simulated American, s.e.)
LS_TABLE1 = {
    (36, 0.20, 1): (4.478, 3.844, 4.472, 0.010),
    (40, 0.20, 1): (2.314, 2.066, 2.313, 0.009),
    (44, 0.20, 1): (1.110, 1.017, 1.118, 0.007),
    (36, 0.40, 2): (8.508, 7.700, 8.488, 0.024),
}


# ----------------------------------------------------------------------------- the basics
def _nd(x):
    return stats.norm.cdf(x)


def bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes-Merton European call, the reference for everything below."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K and sigma must be positive")
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    return float(S * math.exp(-q * T) * _nd(d1) - K * math.exp(-r * T) * _nd(d1 - v))


def bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return bs_call(S, K, T, r, q, sigma) - S * math.exp(-q * T) + K * math.exp(-r * T)


def mc_stats(payoffs) -> tuple:
    """(mean, standard error). The standard error is over the INDEPENDENT unit - for
    antithetic sampling that is the PAIR average, which is what the callers pass in."""
    x = np.asarray(payoffs, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 samples")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


# ---------------------------------------------------------------- Asian options and the CV
def geometric_asian_call(S: float, K: float, T: float, r: float, q: float, sigma: float,
                         m: int) -> float:
    """Exact price of a DISCRETELY monitored geometric-average Asian call (Kemna-Vorst 1990).

    log G = log S0 + (1/m) * sum_i [(r - q - sigma^2/2)*t_i + sigma*W(t_i)] is normal, with
    variance (sigma^2/m^2) * sum_i sum_j min(t_i, t_j), so the price is a Black-Scholes form
    in that normal. This is the control variate for the arithmetic average, and its closed
    form is what makes the control variate free.
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    t = np.arange(1, m + 1) * (T / m)
    mu = math.log(S) + (r - q - 0.5 * sigma ** 2) * float(t.mean())
    var = (sigma ** 2 / m ** 2) * float(np.minimum.outer(t, t).sum())
    sd = math.sqrt(var)
    d1 = (mu - math.log(K) + var) / sd
    return float(math.exp(-r * T) * (math.exp(mu + 0.5 * var) * _nd(d1) - K * _nd(d1 - sd)))


def _asian_averages(z: np.ndarray, S: float, T: float, r: float, q: float,
                    sigma: float) -> tuple:
    """(arithmetic average, geometric average) along paths built from standard normals `z`
    of shape (n_paths, m)."""
    m = z.shape[1]
    dt = T / m
    logs = math.log(S) + np.cumsum((r - q - 0.5 * sigma ** 2) * dt
                                   + sigma * math.sqrt(dt) * z, axis=1)
    paths = np.exp(logs)
    return paths.mean(axis=1), np.exp(logs.mean(axis=1))


def asian_mc(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.25, m=12, n=100_000,
             method: str = "plain", seed: int = SEED) -> dict:
    """Arithmetic-average Asian call by Monte Carlo. `method` is one of

      plain        i.i.d. normals
      antithetic   each path paired with its negation; the PAIR AVERAGE is the sample unit
      control      the geometric Asian as a control variate, beta by least squares
      stratified   the terminal Brownian value stratified into n equiprobable bins, with a
                   Brownian bridge filling in the path - the classic 1-D stratification

    Every method draws the same number of PATHS, so the variance-reduction factors compare
    like with like. Returns the price, the standard error and the sample unit count.
    """
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)
    if method == "plain":
        z = rng.standard_normal((n, m))
        a, _ = _asian_averages(z, S, T, r, q, sigma)
        unit = disc * np.maximum(a - K, 0.0)
    elif method == "antithetic":
        half = n // 2
        z = rng.standard_normal((half, m))
        a1, _ = _asian_averages(z, S, T, r, q, sigma)
        a2, _ = _asian_averages(-z, S, T, r, q, sigma)
        unit = 0.5 * disc * (np.maximum(a1 - K, 0.0) + np.maximum(a2 - K, 0.0))
    elif method == "control":
        z = rng.standard_normal((n, m))
        a, g = _asian_averages(z, S, T, r, q, sigma)
        y = disc * np.maximum(a - K, 0.0)
        x = disc * np.maximum(g - K, 0.0)
        beta = float(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1))
        unit = y - beta * (x - geometric_asian_call(S, K, T, r, q, sigma, m))
    elif method == "stratified":
        # stratify the TERMINAL Brownian value, then fill the path with a Brownian bridge.
        # TWO draws per stratum, so the within-stratum variance is estimable: with one draw
        # per stratum a stratified estimator has NO honest standard error, and the naive
        # sample sd reports the plain-MC number because the marginal law is unchanged.
        k = 2
        n_str = n // k
        u = (np.repeat(np.arange(n_str), k) + rng.random(n_str * k)) / n_str
        w_T = stats.norm.ppf(u) * math.sqrt(T)
        nn = w_T.size
        z = rng.standard_normal((nn, m))
        dt = T / m
        ramp = (np.arange(1, m + 1) / m)[None, :]
        w = np.cumsum(math.sqrt(dt) * z, axis=1)
        w = w - w[:, [-1]] * ramp + w_T[:, None] * ramp             # Brownian bridge to w_T
        logs = math.log(S) + (r - q - 0.5 * sigma ** 2) * dt * np.arange(1, m + 1) + sigma * w
        a = np.exp(logs).mean(axis=1)
        pay = disc * np.maximum(a - K, 0.0)
        cells = pay.reshape(n_str, k)
        price = float(pay.mean())
        se = float(math.sqrt(float(cells.var(ddof=1, axis=1).sum() / k) / n_str ** 2))
        return {"method": method, "price": price, "se": se, "units": int(pay.size),
                "paths": int(pay.size)}
    else:
        raise ValueError("method must be plain, antithetic, control or stratified")
    price, se = mc_stats(unit)
    return {"method": method, "price": price, "se": se, "units": int(unit.size), "paths": n}


def asian_reference(n_blocks: int = 5, block: int = 200_000, seed: int = SEED, **kw) -> tuple:
    """A high-accuracy reference price for the arithmetic Asian, by control variate in blocks.

    Independent of any Sobol construction, so it can referee the QMC comparison without the
    reference and the estimator sharing points.
    """
    vals = np.array([asian_mc(n=block, method="control", seed=seed + 5_000 * i, **kw)["price"]
                     for i in range(n_blocks)])
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_blocks))


def variance_reduction_table(n: int = 100_000, **kw) -> list:
    """Every method at the same PATH budget, with the reduction factor in VARIANCE terms."""
    base = asian_mc(n=n, method="plain", **kw)
    rows = []
    for meth in ("plain", "antithetic", "control", "stratified"):
        r = asian_mc(n=n, method=meth, **kw)
        r["vrf"] = (base["se"] / r["se"]) ** 2
        rows.append(r)
    return rows


# --------------------------------------------------------------- Longstaff-Schwartz LSM
def design_matrix(x: np.ndarray, basis: str = "poly", K: float = 1.0) -> np.ndarray:
    """Regression basis. `poly` is a constant, X and X^2 in the moneyness X = S/K.

    `laguerre` is Longstaff & Schwartz's own choice - a constant plus the first three WEIGHTED
    Laguerre polynomials, exp(-X/2)*L_k(X). `laguerre_raw` uses the same functions on the raw
    stock price instead of the moneyness, which is where they underflow: at S = 40,
    exp(-S/2) = 2e-9, so all three columns are numerically zero next to the constant.
    """
    if basis == "poly":
        z = x / K
        return np.column_stack([np.ones_like(z), z, z ** 2])
    if basis in ("laguerre", "laguerre_raw"):
        z = x / K if basis == "laguerre" else x
        w = np.exp(-z / 2.0)
        return np.column_stack([np.ones_like(z), w, w * (1.0 - z),
                                w * (1.0 - 2.0 * z + z ** 2 / 2.0)])
    raise ValueError("basis must be poly, laguerre or laguerre_raw")


def _ols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares by the normal equations, with an SVD fallback when they are ill
    conditioned. The 4x4 Gram matrix makes this an order of magnitude faster than lstsq on a
    50,000-row design, and LSM solves one of these per exercise date.

    The condition check is not decoration: the `laguerre_raw` basis of `design_matrix` really
    is near-singular, and it must degrade the same way lstsq would rather than return noise.
    """
    G = A.T @ A
    if np.linalg.cond(G) < 1e10:
        try:
            return np.linalg.solve(G, A.T @ y)
        except np.linalg.LinAlgError:
            pass
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def lsm_price(paths: np.ndarray, K: float, r: float, dt: float, itm_only: bool = True,
              rule: str = "cashflow", basis: str = "poly", scale: float | None = None,
              coefs: list | None = None) -> float:
    """Longstaff-Schwartz for an American PUT on `paths` (n_paths x (n_dates+1), column 0 = t0).

    `rule="cashflow"` is the paper's algorithm: regress the realised discounted cash flow that
    follows on the current state, compare it with the intrinsic value, and carry the CASH FLOW
    backwards. `rule="maxvalue"` is footnote 9's mistake - carry max(intrinsic, fitted
    continuation) backwards instead, which is upward biased because the max is convex and the
    fitted value carries estimation error.

    `itm_only=False` regresses on every path, which the paper explicitly does not do.
    `scale` divides the state before it enters the basis; it defaults to K, which is what
    keeps the weighted Laguerre functions from underflowing. Longstaff & Schwartz's printed
    section-1 regressions are in the RAW state, so that example passes scale=1.

    If `coefs` is a list it is filled with the per-date regression coefficients.
    """
    if rule not in ("cashflow", "maxvalue"):
        raise ValueError("rule must be cashflow or maxvalue")
    n_paths, n_col = paths.shape
    n_ex = n_col - 1
    if n_ex < 1:
        raise ValueError("paths must have at least one exercise date")
    sc = K if scale is None else scale
    df = math.exp(-r * dt)
    value = np.maximum(K - paths[:, -1], 0.0)            # value carried back from t+1
    for j in range(n_ex - 1, 0, -1):
        value = value * df
        s = paths[:, j]
        intrinsic = np.maximum(K - s, 0.0)
        mask = (intrinsic > 0.0) if itm_only else np.ones(n_paths, dtype=bool)
        cont = np.zeros(n_paths)
        if mask.sum() > paths.shape[1]:
            A = design_matrix(s[mask], basis, sc)
            beta = _ols(A, value[mask])
            cont[mask] = A @ beta
            if coefs is not None:
                coefs.append((j, beta))
        ex = mask & (intrinsic > cont)
        if rule == "cashflow":
            value = np.where(ex, intrinsic, value)
        else:
            value = np.where(mask, np.maximum(intrinsic, cont), value)
    return float(np.mean(value * df))


def ls_paper_example() -> dict:
    """The eight-path example of Longstaff & Schwartz (2001) section 1, reproduced."""
    coefs: list = []
    price = lsm_price(LS_PATHS, 1.10, 0.06, 1.0, itm_only=True, scale=1.0, coefs=coefs)
    european = float(np.mean(np.maximum(1.10 - LS_PATHS[:, -1], 0.0)) * math.exp(-0.06 * 3))
    by_date = {j: b for j, b in coefs}
    return {"price": price, "european": european,
            "coef_t2": tuple(by_date[2]), "coef_t1": tuple(by_date[1]),
            "paper_price": LS_AMERICAN, "paper_european": LS_EUROPEAN,
            "paper_t2": LS_COEF_T2, "paper_t1": LS_COEF_T1}


def gbm_paths(S: float, T: float, r: float, sigma: float, n_paths: int, n_steps: int,
              seed: int, antithetic: bool = True) -> np.ndarray:
    """Exact lognormal GBM paths - no discretisation bias to confuse the LSM bias with."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if antithetic:
        z = np.vstack([z, -z])
    logs = math.log(S) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z,
                                   axis=1)
    return np.column_stack([np.full(z.shape[0], S), np.exp(logs)])


def lsm_american_put(S: float, K: float, T: float, r: float, sigma: float,
                     n_paths: int = 100_000, per_year: int = 50, seed: int = SEED,
                     basis: str = "laguerre", itm_only: bool = True,
                     rule: str = "cashflow", out_of_sample: bool = False) -> float:
    """LS (2001) Table 1's setting: 50 exercise points per year, 100,000 antithetic paths.

    `out_of_sample=True` runs the paper's own diagnostic - estimate the stopping rule on one
    set of paths and value it on a different set. That is the honest estimate; the in-sample
    one is the one the paper recommends "in order to minimize computational time".
    """
    n_steps = int(round(per_year * T))
    paths = gbm_paths(S, T, r, sigma, n_paths, n_steps, seed)
    if not out_of_sample:
        return lsm_price(paths, K, r, T / n_steps, itm_only, rule, basis)
    fresh = gbm_paths(S, T, r, sigma, n_paths, n_steps, seed + 1)
    return _lsm_apply(paths, fresh, K, r, T / n_steps, itm_only, basis)


def _lsm_apply(train: np.ndarray, test: np.ndarray, K: float, r: float, dt: float,
               itm_only: bool, basis: str) -> float:
    """Fit the regressions on `train`, then value `test` under the resulting stopping rule."""
    n_ex = train.shape[1] - 1
    df = math.exp(-r * dt)
    v_tr = np.maximum(K - train[:, -1], 0.0)
    betas: dict = {}
    for j in range(n_ex - 1, 0, -1):
        v_tr = v_tr * df
        s = train[:, j]
        intrinsic = np.maximum(K - s, 0.0)
        mask = (intrinsic > 0.0) if itm_only else np.ones(train.shape[0], dtype=bool)
        cont = np.zeros(train.shape[0])
        if mask.sum() > train.shape[1]:
            A = design_matrix(s[mask], basis, K)
            beta = _ols(A, v_tr[mask])
            betas[j] = beta
            cont[mask] = A @ beta
        v_tr = np.where(mask & (intrinsic > cont), intrinsic, v_tr)
    v_te = np.maximum(K - test[:, -1], 0.0)
    for j in range(n_ex - 1, 0, -1):
        v_te = v_te * df
        s = test[:, j]
        intrinsic = np.maximum(K - s, 0.0)
        mask = (intrinsic > 0.0) if itm_only else np.ones(test.shape[0], dtype=bool)
        cont = np.zeros(test.shape[0])
        if j in betas:
            A = design_matrix(s[mask], basis, K)
            cont[mask] = A @ betas[j]
        v_te = np.where(mask & (intrinsic > cont), intrinsic, v_te)
    return float(np.mean(v_te * df))


def crr_american_put(S: float, K: float, T: float, r: float, q: float, sigma: float,
                     steps: int = 2000, ex_every: int = 1) -> float:
    """Cox-Ross-Rubinstein tree with early exercise - the independent reference for LSM.

    The same tree that ../../../fin-models/skills/option-pricing-models/ documents, including
    its own oscillation with the parity of `steps`; 2000 steps is well inside the tolerance
    the comparison here needs.

    `ex_every` restricts exercise to every k-th step, which is what makes the comparison with
    LSM fair: Longstaff & Schwartz price an option "exercisable 50 times per year", not a
    continuously exercisable one, and the difference is worth more than the Monte Carlo error.
    """
    if steps < 1 or ex_every < 1 or steps % ex_every:
        raise ValueError("steps must be at least 1 and a multiple of ex_every")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1): the tree is too coarse")
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    v = np.maximum(K - S * u ** (2 * j - steps), 0.0)
    for i in range(steps - 1, -1, -1):
        v = disc * (p * v[1:i + 2] + (1 - p) * v[0:i + 1])
        if i % ex_every == 0:
            s = S * u ** (2 * np.arange(i + 1) - i)
            v = np.maximum(v, K - s)
    return float(v[0])


def bermudan_tree_put(S: float, K: float, T: float, r: float, sigma: float,
                      per_year: int = 50, refine: int = 40) -> float:
    """A CRR tree on exactly Longstaff & Schwartz's contract: `per_year` exercise dates a year,
    with `refine` tree steps between consecutive exercise dates."""
    n_ex = int(round(per_year * T))
    return crr_american_put(S, K, T, r, 0.0, sigma, steps=n_ex * refine, ex_every=refine)


# ------------------------------------------------------------------- quasi-Monte Carlo
def sobol_normals(d: int, m_pow: int, scramble: bool = True, seed: int = SEED) -> np.ndarray:
    """2^m_pow standard normals in d dimensions from a Sobol sequence.

    `random_base2` is used rather than `random`, because scipy's own warning says the balance
    properties need a power-of-two sample size. An unscrambled sequence starts at exactly
    (0, ..., 0), whose normal inverse is -inf; that row is left in on purpose so the caller
    can see it.
    """
    eng = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    return stats.norm.ppf(eng.random_base2(m_pow))


def sobol_first_point(d: int = 4) -> dict:
    """What an unscrambled Sobol sequence hands you as its first point, and what ppf does."""
    plain = qmc.Sobol(d=d, scramble=False).random(2)
    scram = qmc.Sobol(d=d, scramble=True, seed=SEED).random(2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qmc.Sobol(d=2, scramble=True, seed=SEED).random(5)
        msgs = [str(w.message) for w in caught]
    return {"unscrambled_first": plain[0], "unscrambled_ppf": stats.norm.ppf(plain[0]),
            "scrambled_first": scram[0], "non_power_of_two_warning": msgs}


def asian_qmc(m_pow: int, S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.25, m=12,
              n_scrambles: int = 16, seed: int = SEED) -> tuple:
    """Randomised QMC: `n_scrambles` independent scrambles of 2^m_pow Sobol points.

    The mean over scrambles is the estimate; their spread is the error bar. A SINGLE QMC run
    has no error estimate at all - the points are deterministic, so there is nothing to take a
    variance of. That is the practical price of QMC and the reason randomisation exists.
    """
    disc = math.exp(-r * T)
    vals = []
    for i in range(n_scrambles):
        z = sobol_normals(m, m_pow, scramble=True, seed=seed + i)
        a, _ = _asian_averages(z, S, T, r, q, sigma)
        vals.append(disc * np.maximum(a - K, 0.0).mean())
    v = np.array(vals)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(n_scrambles))


def qmc_vs_mc(powers=(10, 12, 14, 16), n_reps: int = 8, seed: int = SEED, **kw) -> list:
    """RMS error against a high-accuracy reference, for plain MC and scrambled Sobol, at the
    same budget. The fitted slope of log(error) on log(N) is the convergence rate.

    The reference is the CONTROL-VARIATE estimator, not a big QMC run: a QMC reference would
    share its low-discrepancy points with the thing being measured and flatter it.
    """
    ref, _ = asian_reference(seed=seed + 999, **kw)
    rows = []
    for p in powers:
        n = 2 ** p
        mc_err = np.array([asian_mc(n=n, method="plain", seed=seed + 31 * i, **kw)["price"] - ref
                           for i in range(n_reps)])
        qm_err = []
        for i in range(n_reps):
            z = sobol_normals(kw.get("m", 12), p, scramble=True, seed=seed + 77 * i)
            a, _ = _asian_averages(z, kw.get("S", 100.0), kw.get("T", 1.0), kw.get("r", 0.05),
                                   kw.get("q", 0.0), kw.get("sigma", 0.25))
            price = math.exp(-kw.get("r", 0.05) * kw.get("T", 1.0)) * \
                np.maximum(a - kw.get("K", 100.0), 0.0).mean()
            qm_err.append(price - ref)
        rows.append({"n": n, "mc_rmse": float(np.sqrt((mc_err ** 2).mean())),
                     "qmc_rmse": float(np.sqrt((np.array(qm_err) ** 2).mean())), "ref": ref})
    for key in ("mc", "qmc"):
        lg_n = np.log([r["n"] for r in rows])
        lg_e = np.log([r[f"{key}_rmse"] for r in rows])
        slope = float(np.polyfit(lg_n, lg_e, 1)[0])
        for r in rows:
            r[f"{key}_slope"] = slope
    return rows


# ------------------------------------------------------ the bias the error bar cannot see
def down_and_out_call(S: float, K: float, H: float, T: float, r: float, q: float,
                      sigma: float) -> float:
    """CONTINUOUSLY monitored down-and-out call, no rebate, for H < K and H < S.

    c_do = c - c_di with
        lam  = (r - q + sigma^2/2) / sigma^2
        y    = log(H^2/(S*K)) / (sigma*sqrt(T)) + lam*sigma*sqrt(T)
        c_di = S*exp(-q*T)*(H/S)^(2*lam)*N(y) - K*exp(-r*T)*(H/S)^(2*lam-2)*N(y-sigma*sqrt(T))
    """
    if not 0.0 < H < min(S, K):
        raise ValueError("this form needs 0 < H < min(S, K)")
    v = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma ** 2) / sigma ** 2
    y = math.log(H * H / (S * K)) / v + lam * v
    c_di = (S * math.exp(-q * T) * (H / S) ** (2 * lam) * _nd(y)
            - K * math.exp(-r * T) * (H / S) ** (2 * lam - 2) * _nd(y - v))
    return float(bs_call(S, K, T, r, q, sigma) - c_di)


def barrier_mc(S=100.0, K=100.0, H=90.0, T=1.0, r=0.05, q=0.0, sigma=0.25, m=52,
               n=200_000, seed: int = SEED) -> dict:
    """Discretely monitored down-and-out call: the barrier is checked at m dates only.

    Antithetic, exact lognormal steps (so there is NO Euler bias to confuse this with), and
    O(n) memory - the path is stepped forward and only a knocked-out flag is kept.
    """
    if m < 1 or n < 2:
        raise ValueError("need m >= 1 and n >= 2")
    rng = np.random.default_rng(seed)
    dt = T / m
    drift, vol = (r - q - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt)
    half = n // 2
    s = np.full((2, half), float(S))
    alive = np.ones((2, half), dtype=bool)
    for _ in range(m):
        z = rng.standard_normal(half)
        s = s * np.exp(drift + vol * np.array([z, -z]))
        alive &= s > H
    pay = math.exp(-r * T) * np.where(alive, np.maximum(s - K, 0.0), 0.0)
    unit = pay.mean(axis=0)                              # the antithetic PAIR is the unit
    price, se = mc_stats(unit)
    return {"m": m, "paths": n, "price": price, "se": se}


def bgk_corrected(S=100.0, K=100.0, H=90.0, T=1.0, r=0.05, q=0.0, sigma=0.25,
                  m: int = 52) -> float:
    """The continuous formula at the shifted barrier H*exp(-beta*sigma*sqrt(T/m)) - Broadie,
    Glasserman & Kou's continuity correction for a DOWN barrier."""
    return down_and_out_call(S, K, H * math.exp(-BGK_BETA * sigma * math.sqrt(T / m)),
                             T, r, q, sigma)


# ---------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    print("=" * 92)
    print("MONTE CARLO METHODS  --  variance reduction, Longstaff-Schwartz, QMC, and bias")
    print("=" * 92)

    # ---- 1. the rate, and the three variance reductions
    print("\n1. The standard error and its 1/sqrt(N) rate, on an arithmetic Asian call")
    print("   S=K=100, T=1, r=5%, sigma=25%, 12 monitoring dates")
    print(f"   {'paths':>10}{'price':>11}{'std err':>10}{'ratio to previous':>20}")
    prev = None
    for n in (10_000, 40_000, 160_000, 640_000):
        d = asian_mc(n=n, method="plain")
        rat = "" if prev is None else f"{prev / d['se']:>20.3f}"
        print(f"   {n:>10}{d['price']:>11.4f}{d['se']:>10.4f}{rat}")
        prev = d["se"]
    print("   4x the paths halves the error bar, three times over. That is the whole rate:")
    print("   to cut the error by 10 you pay 100x, and no amount of it touches a bias.")
    print("\n   Variance reduction at a fixed 100,000-PATH budget:")
    print(f"   {'method':<14}{'price':>11}{'std err':>10}{'variance reduction':>21}"
          f"{'sample units':>14}")
    rows = variance_reduction_table(100_000)
    for rrow in rows:
        print(f"   {rrow['method']:<14}{rrow['price']:>11.5f}{rrow['se']:>10.5f}"
              f"{rrow['vrf']:>20.1f}x{rrow['units']:>14}")
    geo = geometric_asian_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 12)
    print(f"   The control variate is the GEOMETRIC-average Asian, worth {geo:.5f} in closed")
    print("   form (Kemna-Vorst): it is exactly lognormal, correlates ~0.999 with the")
    print("   arithmetic payoff, and costs one extra exponential per path.")
    print("   ! Antithetic sampling halves the number of independent units, so its error bar")
    print("     must be computed over PAIR AVERAGES. Treating 100,000 antithetic draws as")
    print("     100,000 independent samples understates the error and flatters the method.")

    # ---- 2. Longstaff-Schwartz
    print("\n2. Longstaff & Schwartz (2001), reproduced - the eight-path example of section 1")
    ex = ls_paper_example()
    print(f"   regression at t=2:  {ex['coef_t2'][0]:+.3f} {ex['coef_t2'][1]:+.3f} X "
          f"{ex['coef_t2'][2]:+.3f} X^2     paper: "
          f"{ex['paper_t2'][0]:+.3f} {ex['paper_t2'][1]:+.3f} X {ex['paper_t2'][2]:+.3f} X^2")
    print(f"   regression at t=1:  {ex['coef_t1'][0]:+.3f} {ex['coef_t1'][1]:+.3f} X "
          f"{ex['coef_t1'][2]:+.3f} X^2     paper: "
          f"{ex['paper_t1'][0]:+.3f} {ex['paper_t1'][1]:+.3f} X {ex['paper_t1'][2]:+.3f} X^2")
    print(f"   American {ex['price']:.4f} (paper .1144)   European {ex['european']:.4f} "
          f"(paper .0564)")
    print("   Both regressions and both prices, from a price matrix the paper never prints -")
    print("   it is pinned down by its own X and Y columns, and this is the proof.")

    print("\n   ... and rows of its Table 1 (K=40, r=6%, 50 exercise points a year,")
    print("   100,000 antithetic paths, a constant and three weighted Laguerre polynomials):")
    print(f"   {'S':>4}{'sigma':>7}{'T':>4}{'LSM here':>11}{'Bermudan tree':>15}"
          f"{'paper FD':>10}{'paper LSM':>11}{'(s.e.)':>9}{'LSM - FD':>10}{'cont. Amer.':>13}")
    for (S0, sg, T0), (fd, eu, sim, se) in LS_TABLE1.items():
        mine = lsm_american_put(S0, 40.0, float(T0), 0.06, sg, 100_000, 50, basis="laguerre")
        tree = bermudan_tree_put(S0, 40.0, float(T0), 0.06, sg)
        cont = crr_american_put(S0, 40.0, float(T0), 0.06, 0.0, sg, 2000)
        print(f"   {S0:>4}{sg:>7.2f}{T0:>4}{mine:>11.4f}{tree:>15.4f}{fd:>10.3f}"
              f"{sim:>11.3f}{se:>9.3f}{mine - fd:>+10.4f}{cont:>13.4f}")
    print("   The tree column prices EXACTLY the paper's contract - 50 exercise dates a year,")
    print("   40 tree steps between consecutive ones - and lands on its finite-difference")
    print("   column, so three unrelated methods meet in the same place.")
    print("   ! The last column is the same tree with exercise allowed at all 2000 steps. The")
    print("     gap is a different CONTRACT, not an error, and it is several times the paper's")
    print("     own Monte Carlo standard error. Compare a Bermudan LSM against a Bermudan")
    print("     tree, or the comparison measures the exercise schedule instead of the method.")

    print("\n   TRAP 2a - the paper's footnote 9: valuing by max(intrinsic, FITTED continuation)")
    print("   instead of carrying the realised cash flow back. 'The maximum operator is convex;")
    print("   measurement error in the estimated continuation value results in the maximum")
    print("   operator being upward biased.'")
    print("   Same paths, same seed, same regressions - only the value carried backwards")
    print("   changes, so the last two columns are the rule and nothing else:")
    print(f"   {'S':>4}{'sigma':>7}{'T':>4}{'cash flow':>12}{'max-value':>11}{'tree':>10}"
          f"{'max - cash':>12}{'max - tree':>12}")
    for (S0, sg, T0) in ((36, 0.20, 1), (40, 0.20, 1), (44, 0.20, 1)):
        good = lsm_american_put(S0, 40.0, float(T0), 0.06, sg, 40_000, 50, basis="laguerre")
        bad = lsm_american_put(S0, 40.0, float(T0), 0.06, sg, 40_000, 50, basis="laguerre",
                               rule="maxvalue")
        tree = bermudan_tree_put(S0, 40.0, float(T0), 0.06, sg)
        print(f"   {S0:>4}{sg:>7.2f}{T0:>4}{good:>12.4f}{bad:>11.4f}{tree:>10.4f}"
              f"{bad - good:>+12.4f}{bad - tree:>+12.4f}")
    print("   The 'max - tree' column is what a user sees; 'max - cash' is the same 40,000")
    print("   paths priced two ways, so it is the bias with the Monte Carlo noise removed.")
    print("   Always POSITIVE, at every strike: max(a, b_hat) with a noisy b_hat is upward")
    print("   biased even when b_hat is unbiased, and there is no path count that fixes it")
    print("   because it is the estimator, not the sample.")

    print("\n   TRAP 2b - regressing on ALL paths, and the basis that underflows")
    print(f"   {'variant':<34}{'price':>10}{'tree':>10}{'error':>10}")
    tree36 = bermudan_tree_put(36, 40.0, 1.0, 0.06, 0.20)
    ins = None
    for label, kwargs in (("in-the-money paths only (paper)", {}),
                          ("all paths", {"itm_only": False}),
                          ("polynomial basis in S/K", {"basis": "poly"}),
                          ("Laguerre on the RAW S, not S/K", {"basis": "laguerre_raw"})):
        p = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, 100_000, 50,
                             **{"basis": "laguerre", **kwargs})
        ins = p if ins is None else ins
        print(f"   {label:<34}{p:>10.4f}{tree36:>10.4f}{p - tree36:>+10.4f}")
    oos = lsm_american_put(36, 40.0, 1.0, 0.06, 0.20, 100_000, 50, basis="laguerre",
                           out_of_sample=True)
    print(f"   in-sample {ins:.4f} against out-of-sample {oos:.4f} (the paper's own diagnostic,")
    print(f"   its Table 2): a gap of {oos - ins:+.4f}, which is what the paper reports too -")
    print("   'the in-sample and out-of-sample values are virtually identical'.")
    print("   ! LSM's stopping rule is estimated, so it is suboptimal, so the OUT-OF-SAMPLE")
    print("     value is a lower bound on the true price IN EXPECTATION. It is not a bound on")
    print("     any single run: at these path counts the Monte Carlo noise is several times")
    print("     the bias, which is exactly why the paper ran the diagnostic five times per")
    print("     row. An LSM number above the tree is not evidence of anything by itself.")

    # ---- 3. QMC
    print("\n3. Quasi-Monte Carlo with a scrambled Sobol sequence")
    sf = sobol_first_point()
    print(f"   unscrambled Sobol's first point: {np.array2string(sf['unscrambled_first'], precision=3)}"
          f"  ->  norm.ppf = {np.array2string(sf['unscrambled_ppf'], precision=3)}")
    print(f"   scrambled first point:          "
          f"{np.array2string(sf['scrambled_first'], precision=3)}")
    print(f"   .random(5) warns: {sf['non_power_of_two_warning']}")
    print("   ! scramble=True is the DEFAULT (scipy 1.13.0, 'If True, use LMS+shift")
    print("     scrambling'), and turning it off hands you a -inf on the very first path.")
    print("     Use random_base2(m) for 2^m points; slicing or thinning the sequence throws")
    print("     away the balance property that is the entire reason to use it.")
    print(f"   {'N = 2^p':>10}{'MC rmse':>12}{'QMC rmse':>12}{'QMC advantage':>16}")
    qrows = qmc_vs_mc()
    for rrow in qrows:
        print(f"   {rrow['n']:>10}{rrow['mc_rmse']:>12.5f}{rrow['qmc_rmse']:>12.5f}"
              f"{rrow['mc_rmse'] / rrow['qmc_rmse']:>15.1f}x")
    print(f"   fitted convergence rate: MC N^{qrows[0]['mc_slope']:.2f}, "
          f"QMC N^{qrows[0]['qmc_slope']:.2f}  (theory: -0.50 and up to -1)")
    p, se = asian_qmc(14, n_scrambles=16)
    print(f"   randomised QMC at 2^14 x 16 scrambles: {p:.5f} +/- {se:.5f}")
    print("   ! A single QMC run has NO error estimate - the points are deterministic, so")
    print("     there is nothing to take a variance of. The error bar above comes from")
    print("     independent SCRAMBLES, which is what randomised QMC is for.")

    # ---- 4. the bias the error bar cannot see
    print("\n4. TRAP - the standard error is an error bar on the ESTIMATOR, not on the model")
    exact = down_and_out_call(100.0, 100.0, 90.0, 1.0, 0.05, 0.0, 0.25)
    vanilla = bs_call(100.0, 100.0, 1.0, 0.05, 0.0, 0.25)
    print(f"   A down-and-out call, S=K=100, H=90, T=1, r=5%, sigma=25%. CONTINUOUS monitoring")
    print(f"   has a closed form: {exact:.5f}, against the vanilla call's {vanilla:.5f}.")
    print("   Simulate it by checking the barrier at m dates - exact lognormal steps, so there")
    print("   is no Euler bias anywhere - and the paths that dip under H between two dates and")
    print("   come back are never knocked out:")
    print(f"   {'m':>6}{'paths':>10}{'MC price':>11}{'std err':>10}{'error':>10}"
          f"{'error / se':>12}{'BGK target':>13}{'left':>7}")
    brows = []
    for m_mon, n_p in ((12, 150_000), (52, 150_000), (52, 600_000), (252, 150_000)):
        d = barrier_mc(m=m_mon, n=n_p)
        d["target"] = bgk_corrected(m=m_mon)
        brows.append(d)
        print(f"   {m_mon:>6}{n_p:>10}{d['price']:>11.5f}{d['se']:>10.5f}"
              f"{d['price'] - exact:>+10.5f}{(d['price'] - exact) / d['se']:>12.1f}"
              f"{d['target']:>16.5f}{(d['price'] - d['target']) / d['se']:>+6.1f}")
    a52, b52, c252 = brows[1], brows[2], brows[3]
    print("   Columns 5 and 6 are the trap. Read the two m = 52 rows against each other:")
    print(f"   4x the paths cuts the error bar {a52['se']:.4f} -> {b52['se']:.4f}, and the price stays")
    print(f"   {a52['price'] - exact:.2f} / {b52['price'] - exact:.2f} too high, so the bias goes from "
          f"{(a52['price'] - exact) / a52['se']:.0f} to {(b52['price'] - exact) / b52['se']:.0f} standard")
    print("   errors. MORE paths make the reported error bar a WORSE description of how wrong")
    print(f"   the number is. Monitoring dates are what fix it: m = 12 -> 252 cuts the bias")
    print(f"   {brows[0]['price'] - exact:.2f} -> {c252['price'] - exact:.2f}, a factor of "
          f"{(brows[0]['price'] - exact) / (c252['price'] - exact):.1f} against the "
          f"{math.sqrt(252 / 12):.1f} that 1/sqrt(m) predicts.")
    print("   The last column is Broadie-Glasserman-Kou: the continuous formula at the barrier")
    print(f"   H*exp(-beta*sigma*sqrt(T/m)) with beta = -zeta(1/2)/sqrt(2*pi) = {BGK_BETA:.6f},")
    print(f"   from zeta(1/2) = {ZETA_HALF:.13f}. The number after it is the remaining error in")
    print("   standard errors: the correction turns a 40-sigma bias into about one sigma, out")
    print("   of one closed-form evaluation and no extra paths at all.")
    print(f"   ! scipy.special.zeta(0.5, 1) returns {float(special.zeta(0.5, 1))} - the Hurwitz form carries no")
    print("     analytic continuation below 1. scipy.special.zetac(0.5) + 1 does. A nan there")
    print("     does not raise; it flows through exp() into a barrier level and surfaces much")
    print("     later as a comparison that is inexplicably False.")

    print("\nRule: a standard error measures how much the ESTIMATOR wobbles and is blind to"
          " whatever the estimator converges to - so report the discretisation separately,"
          " and reduce variance only after the thing you are converging to is right.")
