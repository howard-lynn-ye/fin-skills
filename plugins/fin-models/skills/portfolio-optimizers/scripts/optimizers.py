"""Portfolio optimizers - mean-variance, Black-Litterman, risk parity, minimum CVaR - and the two
measurements that should travel with any weight vector they produce.

WHY this exists: an optimizer returns the weights that are best for the INPUTS. Nobody has
the inputs. The two facts that follow are the ones a backtest never shows you:

  1. Mean-variance weights are a steep function of expected returns. Perturb every asset's
     mu by one basis point and the optimizer trades a measurable fraction of the book. That
     is estimation error being converted into turnover, and it is what "the optimizer is an
     error maximizer" means in numbers.
  2. Out of sample, 1/N beats the sample-based mean-variance rule until the estimation
     window is implausibly long - the effect DeMiguel, Garlappi and Uppal (2009) named. The
     demo measures the crossover window on a simulated universe where 1/N is NOT optimal:
     the true tangency portfolio's Sharpe is 1.88x 1/N's, and the demo prints both population
     numbers so the setup is checked rather than claimed.

Every estimator here is implemented from its formula on numpy / scipy; PyPortfolioOpt is
optional and, when importable, its `BlackLittermanModel` numbers are compared and printed.
It is imported inside `pypfopt_bl_check` and scikit-learn is never imported in this process,
because on some toolchains `import sklearn` before `import osqp` (a cvxpy dependency, and so
a PyPortfolioOpt one) crashes the interpreter - measured in the sibling skill
`../../covariance-and-risk-models/scripts/risk_model.py`.

What the module provides:

    mean_variance            max mu'w - (gamma/2) w'Sigma w with a budget and optional
                             long-only / box bounds (scipy SLSQP); closed form when unbounded
    tangency, min_variance   the two closed-form corners
    black_litterman          reverse-optimized prior, posterior mean in two algebraically equal
                             forms (checked), posterior covariance Sigma + M, default Omega =
                             diag(tau P Sigma P') as in He-Litterman / PyPortfolioOpt
    risk_parity              equal (or budgeted) risk contributions by Newton's method on the
                             convex form 0.5 w'Sigma w - sum b_i log w_i
    min_cvar_lp              Rockafellar-Uryasev minimum-CVaR as a linear program
                             (scipy.optimize.linprog, HiGHS), checked against the sample CVaR
    perturbation_turnover    the 1-bp sensitivity; assert_stable_weights is the guard
    oos_backtest             rolling-window out-of-sample returns of a dict of strategies
    ledoit_wolf_identity     a self-contained copy of the shrinkage used by the demo

HRP is deliberately absent: `../../../fin-libraries/skills/lib-pyportfolioopt/scripts/weight_traps.py`
implements it and measures its prices-for-returns trap.

Run:  python optimizers.py     (numpy / scipy; PyPortfolioOpt optional; fixed seeds; about 10 s)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

PERIODS = 12
SEED = 0


# ------------------------------------------------------------------ mean-variance ------------
def mean_variance_closed_form(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 1.0,
                              budget: float = 1.0) -> np.ndarray:
    """argmax mu'w - (gamma/2) w'Sigma w  s.t. 1'w = budget, no other constraint."""
    mu = np.asarray(mu, float)
    ones = np.ones(len(mu))
    s_mu = np.linalg.solve(Sigma, mu)
    s_1 = np.linalg.solve(Sigma, ones)
    lam = (ones @ s_mu / risk_aversion - budget) / (ones @ s_1)
    return s_mu / risk_aversion - lam * s_1


def mean_variance(mu: np.ndarray, Sigma: np.ndarray, risk_aversion: float = 1.0,
                  long_only: bool = True, budget: float = 1.0,
                  bounds: list[tuple] | None = None, w0: np.ndarray | None = None) -> np.ndarray:
    """max mu'w - (gamma/2) w'Sigma w  s.t. 1'w = budget, w >= 0 if long_only, or `bounds`."""
    mu, Sigma = np.asarray(mu, float), np.asarray(Sigma, float)
    n = len(mu)
    if not long_only and bounds is None:
        return mean_variance_closed_form(mu, Sigma, risk_aversion, budget)
    if bounds is None:
        bounds = [(0.0, None)] * n
    start = np.full(n, budget / n) if w0 is None else np.asarray(w0, float)
    res = minimize(lambda w: -(mu @ w) + 0.5 * risk_aversion * w @ Sigma @ w, start,
                   jac=lambda w: -mu + risk_aversion * Sigma @ w, bounds=bounds, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - budget,
                                 "jac": lambda w: np.ones(n)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"SLSQP did not converge: {res.message}")
    return res.x


def tangency(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Sigma^-1 mu / |1' Sigma^-1 mu|: the maximum-Sharpe portfolio, unconstrained.

    The ABSOLUTE value in the denominator is DeMiguel et al.'s convention and it matters: when
    1' Sigma^-1 mu is negative, dividing by the signed sum flips every weight and silently
    turns the portfolio inside out. Dividing by |.| keeps the direction and lets the weights
    sum to -1, which is visible. Check the sign of w.sum() before you trade it.
    """
    x = np.linalg.solve(Sigma, np.asarray(mu, float))
    return x / abs(x.sum())


def min_variance(Sigma: np.ndarray, long_only: bool = False) -> np.ndarray:
    Sigma = np.asarray(Sigma, float)
    if not long_only:
        x = np.linalg.solve(Sigma, np.ones(len(Sigma)))
        return x / x.sum()
    return mean_variance(np.zeros(len(Sigma)), Sigma, 1.0, long_only=True)


# ------------------------------------------------------------------ Black-Litterman ----------
def implied_returns(Sigma: np.ndarray, w_mkt: np.ndarray, risk_aversion: float,
                    rf: float = 0.0) -> np.ndarray:
    """Reverse optimization: pi = delta Sigma w_mkt (+ rf), the market's implied excess return."""
    w = np.asarray(w_mkt, float)
    return risk_aversion * np.asarray(Sigma, float) @ (w / w.sum()) + rf


def black_litterman(Sigma: np.ndarray, P: np.ndarray, Q: np.ndarray, w_mkt: np.ndarray | None = None,
                    pi: np.ndarray | None = None, tau: float = 0.05,
                    omega: np.ndarray | None = None, risk_aversion: float = 1.0,
                    rf: float = 0.0) -> dict:
    """Posterior expected returns and covariance given K views P mu = Q + eps, eps ~ N(0, Omega).

    Prior: mu ~ N(pi, tau Sigma). Posterior mean, two equal forms:
        pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P pi)                (solve form)
        [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 pi + P' Omega^-1 Q] (precision form)
    Posterior covariance of returns: Sigma + M, M = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
    (He and Litterman; what PyPortfolioOpt's bl_cov returns).
    Default Omega = diag(diag(tau P Sigma P')). With neither pi nor w_mkt this raises: a zero
    prior is not a prior.
    """
    Sigma = np.asarray(Sigma, float)
    P = np.atleast_2d(np.asarray(P, float))
    Q = np.asarray(Q, float).ravel()
    if pi is None:
        if w_mkt is None:
            raise ValueError("give pi or w_mkt: a zero prior is not a prior (PyPortfolioOpt warns and"
                             " uses zeros when pi=None; that is an opinion, not equilibrium)")
        pi = implied_returns(Sigma, w_mkt, risk_aversion, rf)
    pi = np.asarray(pi, float).ravel()
    tS = tau * Sigma
    if omega is None:
        omega = np.diag(np.diag(P @ tS @ P.T))
    omega = np.asarray(omega, float)
    A = P @ tS @ P.T + omega
    mu_solve = pi + tS @ P.T @ np.linalg.solve(A, Q - P @ pi)
    M = np.linalg.inv(np.linalg.inv(tS) + P.T @ np.linalg.solve(omega, P))
    mu_prec = M @ (np.linalg.solve(tS, pi) + P.T @ np.linalg.solve(omega, Q))
    Sigma_bl = Sigma + M
    Sigma_bl_solve = Sigma + tS - tS @ P.T @ np.linalg.solve(A, P @ tS)
    return {"pi": pi, "mu_bl": mu_solve, "Sigma_bl": Sigma_bl, "omega": omega, "tau": tau,
            "mean_form_gap": float(np.max(np.abs(mu_solve - mu_prec))),
            "cov_form_gap": float(np.max(np.abs(Sigma_bl - Sigma_bl_solve))),
            "w_implied": np.linalg.solve(risk_aversion * Sigma, mu_solve - rf)}


def pypfopt_bl_check(Sigma: np.ndarray, w_mkt: np.ndarray, P: np.ndarray, Q: np.ndarray,
                     tau: float, risk_aversion: float, names: list[str]) -> dict | None:
    """Compare with PyPortfolioOpt's BlackLittermanModel. None when it is not installed."""
    try:
        from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
    except ImportError:
        return None
    S = pd.DataFrame(Sigma, index=names, columns=names)
    caps = pd.Series(np.asarray(w_mkt, float), index=names)
    pi_lib = market_implied_prior_returns(caps, risk_aversion, S, risk_free_rate=0.0)
    mine = black_litterman(Sigma, P, Q, w_mkt=w_mkt, tau=tau, risk_aversion=risk_aversion)
    bl = BlackLittermanModel(S, pi=pi_lib, P=np.asarray(P, float), Q=np.asarray(Q, float), tau=tau)
    return {"max_abs_pi_diff": float(np.max(np.abs(pi_lib.to_numpy() - mine["pi"]))),
            "max_abs_mu_diff": float(np.max(np.abs(bl.bl_returns().to_numpy() - mine["mu_bl"]))),
            "max_abs_cov_diff": float(np.max(np.abs(bl.bl_cov().to_numpy() - mine["Sigma_bl"]))),
            "max_abs_omega_diff": float(np.max(np.abs(np.asarray(bl.omega) - mine["omega"])))}


# ------------------------------------------------------------------ risk parity --------------
def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """w_i (Sigma w)_i / sigma_p: contributions to portfolio volatility, summing to sigma_p."""
    w, Sigma = np.asarray(w, float), np.asarray(Sigma, float)
    return w * (Sigma @ w) / np.sqrt(w @ Sigma @ w)


def risk_parity(Sigma: np.ndarray, budget: np.ndarray | None = None, max_iter: int = 200,
                tol: float = 1e-12) -> np.ndarray:
    """Weights whose risk contributions are proportional to `budget` (equal by default).

    Newton's method on the convex problem min 0.5 w'Sigma w - sum_i b_i log w_i, whose
    stationary point satisfies w_i (Sigma w)_i = b_i; the weights are then normalized to
    sum to one (risk-contribution shares are scale-free).
    """
    Sigma = np.asarray(Sigma, float)
    n = len(Sigma)
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, float) / np.sum(budget)
    w = 1.0 / np.sqrt(np.diag(Sigma))
    w /= np.sqrt(w @ Sigma @ w)
    for _ in range(max_iter):
        g = Sigma @ w - b / w
        if np.max(np.abs(g)) < tol:
            break
        step = np.linalg.solve(Sigma + np.diag(b / w ** 2), g)
        alpha = 1.0
        while np.any(w - alpha * step <= 0.0):
            alpha *= 0.5
        w = w - alpha * step
    return w / w.sum()


# ------------------------------------------------------------------ minimum CVaR -------------
def sample_cvar(loss: np.ndarray, level: float) -> float:
    """Rockafellar-Uryasev CVaR of an equiprobable sample: the mean of the worst (1-level)
    fraction of losses, the atom at the VaR split fractionally."""
    loss = np.sort(np.asarray(loss, float))[::-1]
    k = (1.0 - level) * len(loss)
    m = int(np.floor(k))
    tail = loss[:m].sum()
    if k > m:
        tail += (k - m) * loss[m]
    return float(tail / k)


def min_cvar_lp(R: np.ndarray, level: float = 0.95, long_only: bool = True, budget: float = 1.0,
                min_return: float | None = None) -> dict:
    """min_{w, alpha, z} alpha + 1/((1-level) T) sum_t z_t
       s.t. z_t >= -r_t'w - alpha, z_t >= 0, 1'w = budget, w >= 0 (long_only), mean(R) w >= min_return.
    At the optimum alpha is the VaR and the objective the CVaR of the portfolio's sample losses."""
    R = np.asarray(R, float)
    n_obs, n = R.shape
    c = np.concatenate([np.zeros(n), [1.0], np.full(n_obs, 1.0 / ((1.0 - level) * n_obs))])
    A_ub = np.hstack([-R, -np.ones((n_obs, 1)), -np.eye(n_obs)])
    b_ub = np.zeros(n_obs)
    if min_return is not None:
        A_ub = np.vstack([A_ub, np.concatenate([-R.mean(axis=0), [0.0], np.zeros(n_obs)])])
        b_ub = np.append(b_ub, -min_return)
    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(n_obs)])[None, :]
    bounds = ([(0.0, None) if long_only else (None, None)] * n + [(None, None)]
              + [(0.0, None)] * n_obs)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[budget], bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed: {res.message}")
    w = res.x[:n]
    return {"w": w, "var": float(res.x[n]), "cvar": float(res.fun), "n_iter": int(res.nit),
            "cvar_check": sample_cvar(-(R @ w), level)}


# ------------------------------------------------------------------ stability ----------------
def perturbation_turnover(mu: np.ndarray, Sigma: np.ndarray, solver, bp: float = 1.0,
                          n_draws: int = 20, seed: int = SEED) -> dict:
    """Turnover 0.5 sum |w(mu + d) - w(mu)| for d = +-bp basis points per asset (random signs),
    plus the number of weights that change sign. `solver(mu, Sigma) -> w`."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float)
    base = solver(mu, Sigma)
    turns, flips = [], []
    for _ in range(n_draws):
        d = rng.choice([-1.0, 1.0], size=len(mu)) * bp * 1e-4
        w = solver(mu + d, Sigma)
        turns.append(0.5 * np.abs(w - base).sum())
        flips.append(int(((w > 1e-9) != (base > 1e-9)).sum()))
    return {"mean_turnover": float(np.mean(turns)), "max_turnover": float(np.max(turns)),
            "mean_sign_flips": float(np.mean(flips)), "gross_leverage": float(np.abs(base).sum()),
            "bp": bp}


def assert_stable_weights(mu: np.ndarray, Sigma: np.ndarray, solver, bp: float = 1.0,
                          max_turnover: float = 0.05, **kw) -> dict:
    """The guard: raise when a 1-bp change in expected returns moves more than max_turnover
    of the book. Returns the perturbation report otherwise."""
    rep = perturbation_turnover(mu, Sigma, solver, bp=bp, **kw)
    if rep["mean_turnover"] > max_turnover:
        raise ValueError(
            f"a {bp:.0f} bp perturbation of mu turns over {rep['mean_turnover']:.1%} of the book "
            f"(limit {max_turnover:.0%}): the weights are estimation error, not a view. Shrink mu "
            f"and Sigma, add constraints, or use a mu-free rule (min-variance, risk parity, 1/N).")
    return rep


# ------------------------------------------------------------------ shrinkage (local copy) ---
def ledoit_wolf_identity(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage to mu I, the sklearn estimator, on the ddof=1 sample covariance.
    The full estimator family lives in fin_skills.models.risk_model; this copy keeps the
    optimizer skill self-contained."""
    X = np.asarray(X, float)
    n_obs, n = X.shape
    Xc = X - X.mean(axis=0)
    Sb = Xc.T @ Xc / n_obs
    mu = np.trace(Sb) / n
    delta = ((Sb - mu * np.eye(n)) ** 2).sum() / n
    X2 = Xc ** 2
    beta = min(((X2.T @ X2).sum() / n_obs - (Sb ** 2).sum()) / (n * n_obs), delta)
    s = 0.0 if beta == 0 else beta / delta
    S = Xc.T @ Xc / (n_obs - 1)
    return (1.0 - s) * S + s * (np.trace(S) / n) * np.eye(n), float(s)


# ------------------------------------------------------------------ out-of-sample ------------
def simulate_universe(n_assets: int = 25, n_months: int = 2280, seed: int = SEED,
                      factor_mean: float = 0.005, factor_vol: float = 0.045,
                      alpha_vol: float = 0.002) -> dict:
    """Monthly returns from a one-factor model with cross-sectional alpha.

    Betas uniform on [0.5, 1.5], idiosyncratic vol uniform on [10 %, 30 %] a year, and a fixed
    per-asset alpha drawn N(0, alpha_vol) once. The alpha is what makes 1/N SUB-optimal here:
    with alpha_vol = 0 the true tangency portfolio's Sharpe is within 1.3 % of 1/N's and the
    comparison is degenerate. At the default the tangency Sharpe is about 1.8x 1/N's, so a
    shortfall out of sample is estimation error and nothing else. `population_sharpe` prints
    both, so the claim is measured rather than asserted.
    """
    rng = np.random.default_rng(seed)
    beta = rng.uniform(0.5, 1.5, n_assets)
    idio = rng.uniform(0.10, 0.30, n_assets) / np.sqrt(PERIODS)
    alpha = rng.normal(0.0, alpha_vol, n_assets)
    f = rng.normal(factor_mean, factor_vol, n_months)
    e = rng.normal(0.0, 1.0, (n_months, n_assets)) * idio
    mu = alpha + beta * factor_mean
    Sigma = np.outer(beta, beta) * factor_vol ** 2 + np.diag(idio ** 2)
    R = alpha[None, :] + beta[None, :] * f[:, None] + e
    return {"R": R, "mu": mu, "Sigma": Sigma, "beta": beta, "alpha": alpha}


def sharpe(r: np.ndarray, periods: int = PERIODS) -> float:
    r = np.asarray(r, float)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def population_sharpe(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                      periods: int = PERIODS) -> float:
    """The Sharpe ratio a weight vector earns at the TRUE moments - no sampling error."""
    w, mu, Sigma = np.asarray(w, float), np.asarray(mu, float), np.asarray(Sigma, float)
    return float(mu @ w / np.sqrt(w @ Sigma @ w) * np.sqrt(periods))


def max_population_sharpe(mu: np.ndarray, Sigma: np.ndarray, periods: int = PERIODS) -> float:
    """sqrt(mu' Sigma^-1 mu) x sqrt(periods): the ceiling the tangency portfolio attains."""
    mu = np.asarray(mu, float)
    return float(np.sqrt(mu @ np.linalg.solve(np.asarray(Sigma, float), mu)) * np.sqrt(periods))


def oos_backtest(R: np.ndarray, window: int, strategies: dict, n_oos: int | None = None) -> pd.DataFrame:
    """Each month t >= window: estimate on R[t-window:t], hold the weights over R[t].
    `strategies` maps name -> fn(R_window, state) -> w; `state` carries the previous weights."""
    R = np.asarray(R, float)
    n_obs = len(R)
    start = window if n_oos is None else n_obs - n_oos
    if start < window:
        raise ValueError("not enough history for the window and n_oos")
    out = {name: np.empty(n_obs - start) for name in strategies}
    state: dict = {}
    for i, t in enumerate(range(start, n_obs)):
        est = R[t - window:t]
        for name, fn in strategies.items():
            w = fn(est, state.get(name))
            state[name] = w
            out[name][i] = w @ R[t]
    return pd.DataFrame(out, index=np.arange(start, n_obs))


def _strategies(long_only_gamma: float | None = 3.0) -> dict:
    def mv_sample(est, _):
        return tangency(est.mean(0), np.cov(est.T))

    def mv_lw(est, _):
        return tangency(est.mean(0), ledoit_wolf_identity(est)[0])

    def minvar_sample(est, _):
        return min_variance(np.cov(est.T))

    def minvar_lw(est, _):
        return min_variance(ledoit_wolf_identity(est)[0])

    strategies = {"1/N": lambda est, _: np.full(est.shape[1], 1.0 / est.shape[1]),
                  "mean-variance (sample)": mv_sample, "mean-variance (Ledoit-Wolf)": mv_lw,
                  "min-variance (sample)": minvar_sample, "min-variance (Ledoit-Wolf)": minvar_lw}
    if long_only_gamma is not None:
        strategies["mean-variance long-only (sample)"] = (
            lambda est, prev: mean_variance(est.mean(0), np.cov(est.T), long_only_gamma,
                                            long_only=True, w0=prev))
    return strategies


# ------------------------------------------------------------------ demo ---------------------
BL_NAMES = ["A bonds", "B US equity", "C intl equity", "D EM equity", "E commodities"]
BL_VOLS = np.array([0.05, 0.15, 0.17, 0.22, 0.14])
BL_CORR = np.array([[1.00, 0.10, 0.05, 0.00, -0.05],
                    [0.10, 1.00, 0.75, 0.60, 0.20],
                    [0.05, 0.75, 1.00, 0.65, 0.25],
                    [0.00, 0.60, 0.65, 1.00, 0.30],
                    [-0.05, 0.20, 0.25, 0.30, 1.00]])
BL_CAPS = np.array([40.0, 30.0, 15.0, 5.0, 10.0])
BL_P = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],      # C returns 7 % (absolute)
                 [0.0, -1.0, 0.0, 1.0, 0.0]])    # D beats B by 2 % (relative)
BL_Q = np.array([0.07, 0.02])
BL_DELTA, BL_TAU = 2.5, 0.05


if __name__ == "__main__":
    t0 = time.time()
    # ---------------------------------------------------------------- 1. Black-Litterman
    print("=" * 100)
    print(f"1. BLACK-LITTERMAN: 5 assets, delta = {BL_DELTA}, tau = {BL_TAU},"
          f" default Omega = diag(tau P Sigma P')")
    print("=" * 100)
    Sigma = np.outer(BL_VOLS, BL_VOLS) * BL_CORR
    assert np.linalg.eigvalsh(Sigma).min() > 0
    w_mkt = BL_CAPS / BL_CAPS.sum()
    bl = black_litterman(Sigma, BL_P, BL_Q, w_mkt=w_mkt, tau=BL_TAU, risk_aversion=BL_DELTA)
    w_bl = bl["w_implied"]
    print(f"{'asset':<16} {'w_mkt':>7} {'prior pi':>9} {'posterior':>10} {'change':>8}"
          f" {'prior vol':>9} {'post vol':>9} {'w_bl':>7}")
    print("-" * 100)
    for i, name in enumerate(BL_NAMES):
        print(f"{name:<16} {w_mkt[i]:>7.1%} {bl['pi'][i]:>9.2%} {bl['mu_bl'][i]:>10.2%}"
              f" {bl['mu_bl'][i] - bl['pi'][i]:>+8.2%} {np.sqrt(Sigma[i, i]):>9.2%}"
              f" {np.sqrt(bl['Sigma_bl'][i, i]):>9.2%} {w_bl[i]:>7.1%}")
    print("-" * 100)
    print(f"views: C = {BL_Q[0]:.0%} absolute (prior {bl['pi'][2]:.2%});"
          f" D - B = {BL_Q[1]:+.0%} (prior {bl['pi'][3] - bl['pi'][1]:+.2%})."
          f" Omega diag = {np.array2string(np.diag(bl['omega']), precision=6)}")
    print(f"solve form vs precision form of the posterior mean: max |diff| {bl['mean_form_gap']:.2e};"
          f" covariance forms {bl['cov_form_gap']:.2e}")
    print(f"the posterior moves assets the views never named (A, E) through the covariance; w_bl sums to"
          f" {w_bl.sum():.3f} because w = (delta Sigma)^-1 mu is unnormalized")
    chk = pypfopt_bl_check(Sigma, w_mkt, BL_P, BL_Q, BL_TAU, BL_DELTA, BL_NAMES)
    if chk is None:
        print("PyPortfolioOpt not installed here: BlackLittermanModel cross-check skipped")
    else:
        print(f"PyPortfolioOpt cross-check: max |diff| prior {chk['max_abs_pi_diff']:.2e},"
              f" posterior mean {chk['max_abs_mu_diff']:.2e}, bl_cov"
              f" {chk['max_abs_cov_diff']:.2e}, omega {chk['max_abs_omega_diff']:.2e}")
    try:
        black_litterman(Sigma, BL_P, BL_Q)
    except ValueError as e:
        print(f"the guard: black_litterman(Sigma, P, Q) with no prior -> ValueError: {str(e)[:70]}...")

    # ---------------------------------------------------------------- 2. instability
    print("\n" + "=" * 100)
    print("2. MEAN-VARIANCE WEIGHTS UNDER A 1-BASIS-POINT PERTURBATION OF mu"
          " (25 assets, 120 months of estimates)")
    print("=" * 100)
    uni = simulate_universe(seed=SEED)
    est = uni["R"][:120]
    mu_hat, S_hat = est.mean(0), np.cov(est.T)
    S_lw, s_lw = ledoit_wolf_identity(est)
    print(f"cond(sample Sigma) = {np.linalg.cond(S_hat):.1f}, cond(Ledoit-Wolf) = {np.linalg.cond(S_lw):.1f}"
          f" (shrinkage {s_lw:.3f}); monthly mu_hat ranges {mu_hat.min():.2%} .. {mu_hat.max():.2%}")
    print(f"{'solver':<44} {'gross lev':>9} {'turnover/bp':>12} {'max':>8} {'sign flips':>10}")
    print("-" * 100)
    solvers = {
        "tangency, sample Sigma": (lambda m, S: tangency(m, S), S_hat),
        "tangency, Ledoit-Wolf Sigma": (lambda m, S: tangency(m, S), S_lw),
        "mean-variance long-only gamma=3, sample": (lambda m, S: mean_variance(m, S, 3.0), S_hat),
        "mean-variance long-only gamma=3, Ledoit-Wolf": (lambda m, S: mean_variance(m, S, 3.0), S_lw),
        "min-variance long-only (mu-free)": (lambda m, S: min_variance(S, long_only=True), S_hat),
    }
    reports = {}
    for name, (fn, S) in solvers.items():
        rep = perturbation_turnover(mu_hat, S, fn)
        reports[name] = rep
        print(f"{name:<44} {rep['gross_leverage']:>9.2f} {rep['mean_turnover']:>12.2%}"
              f" {rep['max_turnover']:>8.2%} {rep['mean_sign_flips']:>10.1f}")
    print("-" * 100)
    se_mu = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    print(f"For scale: the standard error of each 120-month mean is"
          f" {1e4 * se_mu.min():.0f} to {1e4 * se_mu.max():.0f} bp/month (median"
          f" {1e4 * np.median(se_mu):.0f}). The 1 bp perturbation above is"
          f" {np.median(se_mu) / 1e-4:.0f}x SMALLER than the noise already in mu_hat.")
    try:
        assert_stable_weights(mu_hat, S_hat, lambda m, S: tangency(m, S))
    except ValueError as e:
        print(f"the guard: assert_stable_weights(tangency) -> ValueError: {str(e)[:80]}...")

    # ---------------------------------------------------------------- 3. risk parity
    print("\n" + "=" * 100)
    print("3. RISK PARITY (equal risk contribution) vs 1/N and inverse-vol, 25-asset sample covariance")
    print("=" * 100)
    w_rp = risk_parity(S_hat)
    rc = risk_contributions(w_rp, S_hat)
    share = rc / rc.sum()
    w_iv = 1.0 / np.sqrt(np.diag(S_hat))
    w_iv /= w_iv.sum()
    for name, w in (("risk parity", w_rp), ("1/N", np.full(25, 1 / 25)), ("inverse vol", w_iv)):
        s = risk_contributions(w, S_hat)
        s /= s.sum()
        print(f"{name:<14} risk shares min {s.min():.4f} max {s.max():.4f}  weights min"
              f" {w.min():.3f} max {w.max():.3f}"
              f"  ann vol {np.sqrt(w @ S_hat @ w * PERIODS):.2%}")
    print(f"risk parity: max |share - 1/N| = {np.max(np.abs(share - 1 / 25)):.1e}")

    # ---------------------------------------------------------------- 4. min-CVaR LP
    print("\n" + "=" * 100)
    print("4. MINIMUM CVaR AS A LINEAR PROGRAM (Rockafellar-Uryasev), 10 assets x 500 months,"
          " level 95 %")
    print("=" * 100)
    R10 = simulate_universe(n_assets=10, n_months=500, seed=3)["R"]
    lp = min_cvar_lp(R10, level=0.95)
    w_mv = min_variance(np.cov(R10.T), long_only=True)
    print(f"LP objective (CVaR) {lp['cvar']:.6f} = sample CVaR of the LP portfolio {lp['cvar_check']:.6f}"
          f" (|diff| {abs(lp['cvar'] - lp['cvar_check']):.1e}); alpha = VaR {lp['var']:.6f};"
          f" HiGHS iterations {lp['n_iter']}")
    print(f"min-CVaR weights: {np.array2string(lp['w'], precision=3)}")
    print(f"min-variance long-only has CVaR {sample_cvar(-(R10 @ w_mv), 0.95):.6f} on the same sample"
          f" ({sample_cvar(-(R10 @ w_mv), 0.95) / lp['cvar'] - 1:+.1%})")

    # ---------------------------------------------------------------- 5. out of sample
    print("\n" + "=" * 100)
    print("5. OUT OF SAMPLE: 1/N vs OPTIMIZED, 25 assets, 120-month rolling window,"
          " 360 test months, 6 seeds")
    print("=" * 100)
    n_oos, seeds, window = 360, (0, 1, 2, 3, 4, 5), 120
    pop = []
    for s in seeds:
        u = simulate_universe(seed=s)
        eqw = np.full(len(u["mu"]), 1.0 / len(u["mu"]))
        pop.append((max_population_sharpe(u["mu"], u["Sigma"]),
                    population_sharpe(eqw, u["mu"], u["Sigma"])))
    pop = np.array(pop)
    print(f"POPULATION Sharpe at the TRUE moments, mean over {len(seeds)} seeds: tangency"
          f" {pop[:, 0].mean():.3f} vs 1/N {pop[:, 1].mean():.3f}"
          f" ({pop[:, 0].mean() / pop[:, 1].mean():.2f}x). 1/N is NOT optimal on this universe,")
    print("so everything the optimizer gives up below is estimation error, not a rigged DGP.")
    table = {}
    for s in seeds:
        u = simulate_universe(seed=s)
        oos = oos_backtest(u["R"], window, _strategies(3.0), n_oos=n_oos)
        for name in oos:
            table.setdefault(name, []).append(sharpe(oos[name]))
        table.setdefault("true tangency (oracle weights)", []).append(
            sharpe(u["R"][-n_oos:] @ tangency(u["mu"], u["Sigma"])))
    print(f"\n{'strategy':<36} {'OOS Sharpe (mean of 6 seeds)':>28} {'worst':>8} {'best':>8}")
    print("-" * 100)
    for name, vals in table.items():
        a = np.array(vals)
        print(f"{name:<36} {a.mean():>28.3f} {a.min():>8.2f} {a.max():>8.2f}")
    print("-" * 100)
    print("window sweep, sample-based mean-variance vs 1/N (same 360 test months, mean over"
          " seeds):")
    sweep = {}
    for M in (60, 120, 240, 480, 960, 1920):
        vals = []
        for s in seeds:
            u = simulate_universe(seed=s)
            oos = oos_backtest(u["R"], M, {"mv": _strategies(None)["mean-variance (sample)"],
                                           "1/N": _strategies(None)["1/N"]}, n_oos=n_oos)
            vals.append((sharpe(oos["mv"]), sharpe(oos["1/N"])))
        sweep[M] = np.mean(vals, axis=0)
    cross = next((M for M, (mv, eq) in sweep.items() if mv >= eq), None)
    print("   " + "  ".join(f"M={M}: mv {mv:.2f} vs 1/N {eq:.2f}" for M, (mv, eq) in sweep.items()))
    print(f"   first window where sample mean-variance beats 1/N:"
          f" {cross if cross else 'none up to 1,920 months'}. DeMiguel, Garlappi and Uppal's"
          f" abstract: for 25 assets")
    print("   the estimation window needed is more than 3,000 months, and for 50 assets more"
          " than 6,000, 'although in")
    print("   practice these parameters are estimated using 120 months of data'.")
    print("HRP: see fin-libraries/lib-pyportfolioopt/scripts/weight_traps.py - not reimplemented"
          " here.")

    print("\n" + "=" * 100)
    print("THE RULE:  an optimizer maximizes the error in its inputs. Report turnover per basis point of mu and")
    print("           the out-of-sample Sharpe against 1/N with every weight vector; if you cannot defend mu,")
    print("           use a mu-free rule (min-variance, risk parity, 1/N) and say so.")
    print("=" * 100)
    print(f"total runtime {time.time() - t0:.1f}s")
