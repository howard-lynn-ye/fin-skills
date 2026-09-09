"""Covariance and risk models, and the two ways a sample covariance lies to an optimizer.

WHY this exists: `np.cov(returns.T)` is an unbiased estimate of every entry and a terrible
estimate of the matrix. Two failures follow, and both look like success in-sample:

  1. With more assets than observations (N > T) the sample covariance has rank at most T-1.
     It is singular. A mean-variance or minimum-variance optimizer inverts it anyway, lands in
     the null space, and reports a portfolio with almost no risk and enormous gross leverage.
  2. Even with N < T, the optimizer picks weights that exploit the estimation error: the
     variance it PREDICTS for its own portfolio is biased low, and the variance you REALIZE
     out of sample is higher. The ratio realized / predicted is the quantity to report.

The estimators here are the usual repair kit, each implemented from its formula and checked
against the library that ships it when that library is importable:

    sample_cov               centred, ddof selectable
    ledoit_wolf              Ledoit-Wolf shrinkage to a scaled identity (the sklearn
                             estimator, checked to machine precision when sklearn is present)
                             or to the constant-correlation target (the "Honey, I shrunk the
                             sample covariance matrix" estimator, in the form PyPortfolioOpt
                             codes it)
    ewma_cov                 RiskMetrics exponentially weighted covariance, lambda 0.94 daily /
                             0.97 monthly, checked against pandas' ewm(adjust=False)
    effective_days           RiskMetrics Eq. [5.26], K = ln(tolerance)/ln(lambda): the effective
                             sample size behind an EWMA, and the reason it is not invertible
    pca_factor_cov           k principal components plus a diagonal residual
    n_factors_above_mp       how many correlation eigenvalues clear the Marchenko-Pastur edge
    barra_cov                a fundamental (Barra-style) factor covariance: cross-sectional
                             regressions on exposures, factor covariance, specific variance
    gmv_weights, bias_ratio  the minimum-variance portfolio and its realized / predicted ratio
    check_invertible         the guard: refuses a covariance a solver should not invert
    probe                    runs one library check in a fresh interpreter, because importing
                             scikit-learn and cvxpy into the same process crashes it here

Run:  python risk_model.py     (numpy / pandas; scikit-learn and PyPortfolioOpt optional and
                                each checked in its own subprocess; fixed seeds; about 10 s)
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = 252
# RiskMetrics Technical Document, Fourth Edition (December 1996), Sec. 5.3.2 and Table 5.9:
# "the decay factor for the daily data set is 0.94, and the decay factor for the monthly data
# set is 0.97". Chosen by minimising the RMSE of the variance forecast on 480+ series and then
# taking a forecast-accuracy-weighted average of the per-series optima (Eqs. [5.30]-[5.35]).
LAMBDA_DAILY = 0.94
LAMBDA_MONTHLY = 0.97
SEED = 0


# --------------------------------------------------------------- one library per interpreter --
# On this machine `import sklearn` followed by `import osqp` - which cvxpy, and therefore
# PyPortfolioOpt, imports - terminates the interpreter with a Windows access violation. A
# segfault cannot be caught, so every library cross-check below runs in a FRESH process and the
# demo reports the return code instead of dying with it. Section 5 measures the crash itself.
_PROBE_HEAD = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parent)!r}); "
               f"import {Path(__file__).stem} as rm\n")


def probe(code: str, timeout: float = 180.0) -> tuple[int, str]:
    """Run `code` in a fresh interpreter; return (returncode, stdout). 0 means it survived."""
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return -1, f"could not start a probe: {type(exc).__name__}"
    return r.returncode, (r.stdout or "").strip()


# ------------------------------------------------------------------ estimators ---------------
def sample_cov(X: np.ndarray, ddof: int = 1) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - ddof)


def ledoit_wolf(X: np.ndarray, target: str = "identity", ddof: int = 0) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage: (1 - s) S + s F with s estimated from the data.

    target="identity": F = mu I, mu = tr(S)/N, with s from Ledoit & Wolf (2004a) exactly as
    sklearn.covariance.ledoit_wolf_shrinkage codes it (centred data, S divided by T). ddof only
    rescales the matrix that is shrunk; the intensity is scale-invariant.
    target="constant_correlation": F keeps the sample variances and replaces every
    correlation by their average, with the intensity of Ledoit & Wolf (2004b) in the form
    PyPortfolioOpt's `_ledoit_wolf_constant_correlation` uses: S with the requested ddof in
    the sample and pi terms, the biased second moment inside the theta term (ddof=1
    reproduces PyPortfolioOpt, ddof=0 the authors' MATLAB covCor).
    Returns (Sigma, shrinkage).
    """
    X = np.asarray(X, float)
    n_obs, n = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (n_obs - ddof)
    if target == "identity":
        Sb = Xc.T @ Xc / n_obs
        mu_b = np.trace(Sb) / n
        delta = ((Sb - mu_b * np.eye(n)) ** 2).sum() / n
        X2 = Xc ** 2
        beta = ((X2.T @ X2).sum() / n_obs - (Sb ** 2).sum()) / (n * n_obs)
        beta = min(beta, delta)
        shrink = 0.0 if beta == 0 else beta / delta
        mu = np.trace(S) / n
        return (1.0 - shrink) * S + shrink * mu * np.eye(n), float(shrink)
    if target == "constant_correlation":
        var = np.diag(S)[:, None]
        std = np.sqrt(var)
        r_bar = ((S / (std @ std.T)).sum() - n) / (n * (n - 1))
        F = r_bar * (std @ std.T)
        np.fill_diagonal(F, var.ravel())
        y = Xc ** 2
        pi_mat = (y.T @ y) / n_obs - 2.0 * (Xc.T @ Xc) * S / n_obs + S ** 2
        pi_hat = pi_mat.sum()
        term1 = ((Xc ** 3).T @ Xc) / n_obs
        help_ = Xc.T @ Xc / n_obs
        term2 = np.diag(help_)[:, None] * S
        term3 = help_ * var
        term4 = var * S
        theta = term1 - term2 - term3 + term4
        np.fill_diagonal(theta, 0.0)
        rho_hat = np.diag(pi_mat).sum() + r_bar * (((1.0 / std) @ std.T) * theta).sum()
        gamma_hat = ((S - F) ** 2).sum()
        kappa = (pi_hat - rho_hat) / gamma_hat
        shrink = max(0.0, min(1.0, kappa / n_obs))
        return shrink * F + (1.0 - shrink) * S, float(shrink)
    raise ValueError("target must be 'identity' or 'constant_correlation'")


def ewma_cov(X: np.ndarray, lam: float = LAMBDA_DAILY, init: np.ndarray | None = None,
             demean: bool = False) -> np.ndarray:
    """RiskMetrics recursion S_t = lam S_{t-1} + (1 - lam) x_{t-1} x_{t-1}', returned for T+1.

    RiskMetrics assumes a zero mean, so returns are not demeaned unless asked. The recursion
    starts from the second-moment matrix of the first 20 rows unless `init` is given; with
    lambda 0.94 the start is forgotten within a few half-lives (11.2 days each).
    """
    X = np.asarray(X, float)
    if demean:
        X = X - X.mean(axis=0)
    if init is None:
        n0 = min(len(X), 20)
        S = X[:n0].T @ X[:n0] / n0
    else:
        S = np.array(init, float)
    for x in X:
        S = lam * S + (1.0 - lam) * np.outer(x, x)
    return S


def ewma_variance_path(x: np.ndarray, lam: float = LAMBDA_DAILY,
                       init: float | None = None) -> np.ndarray:
    """sigma2[t] = forecast variance for period t built from x[:t]; sigma2[0] = init or x[0]^2."""
    x = np.asarray(x, float)
    s2 = np.empty(len(x) + 1)
    s2[0] = x[0] ** 2 if init is None else init
    for t in range(len(x)):
        s2[t + 1] = lam * s2[t] + (1.0 - lam) * x[t] ** 2
    return s2


def half_life(lam: float) -> float:
    return float(np.log(0.5) / np.log(lam))


def effective_days(lam: float, tolerance: float = 0.01) -> float:
    """RiskMetrics' own effective sample size, Eq. [5.26]: K = ln(tolerance) / ln(lambda).

    The number of days whose weights account for all but `tolerance` of the total. Table 5.7 of
    the Technical Document tabulates it; at the 1 % tolerance level it gives 74 days for
    lambda 0.94 and 151 for 0.97, which the demo reproduces.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lambda must be strictly between 0 and 1")
    return float(np.log(tolerance) / np.log(lam))


def span_to_lambda(span: float) -> float:
    """pandas ewm(span=s) has alpha = 2/(s+1); the RiskMetrics lambda is 1 - alpha."""
    return 1.0 - 2.0 / (span + 1.0)


def pca_factor_cov(X: np.ndarray, k: int, ddof: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma = B B' + D with B the top-k principal components scaled by sqrt(eigenvalue) and
    D the diagonal of what they leave. Returns (Sigma, B, all eigenvalues descending)."""
    S = sample_cov(X, ddof)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    B = vecs[:, :k] * np.sqrt(np.maximum(vals[:k], 0.0))
    common = B @ B.T
    D = np.maximum(np.diag(S) - np.diag(common), 1e-12)
    return common + np.diag(D), B, vals


def marchenko_pastur_edge(n_assets: int, n_obs: int, sigma2: float = 1.0) -> float:
    """Largest eigenvalue of a pure-noise covariance with variance sigma2: (1 + sqrt(N/T))^2."""
    return sigma2 * (1.0 + np.sqrt(n_assets / n_obs)) ** 2


def n_factors_above_mp(X: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Eigenvalues of the correlation matrix above the Marchenko-Pastur edge with sigma2 = 1.
    With real factors present the residual variance is below 1, so the edge is conservative."""
    X = np.asarray(X, float)
    vals = np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]
    edge = marchenko_pastur_edge(X.shape[1], X.shape[0])
    return int((vals > edge).sum()), vals, edge


def barra_cov(R: np.ndarray, exposures: np.ndarray, weights: np.ndarray | None = None,
              lam: float | None = None, ddof: int = 1) -> dict:
    """Fundamental factor covariance. exposures is (N, K) static or (T, N, K).

    Each period: f_t = (B' W B)^-1 B' W r_t (W = diag(weights) for a cap-weighted regression,
    identity by default), u_t = r_t - B f_t. F = cov(f) (EWMA with `lam` if given),
    D = diag(var(u_i)), Sigma = B_T F B_T' + D. Returns the pieces.
    """
    R = np.asarray(R, float)
    B = np.asarray(exposures, float)
    n_obs, n = R.shape
    static = B.ndim == 2
    k = B.shape[-1]
    f = np.empty((n_obs, k))
    U = np.empty((n_obs, n))
    for t in range(n_obs):
        Bt = B if static else B[t]
        if weights is None:
            BtW = Bt.T
        else:
            BtW = Bt.T * np.asarray(weights if np.ndim(weights) == 1 else weights[t], float)
        f[t] = np.linalg.solve(BtW @ Bt, BtW @ R[t])
        U[t] = R[t] - Bt @ f[t]
    F = sample_cov(f, ddof) if lam is None else ewma_cov(f, lam, demean=True)
    D = np.diag(U.var(axis=0, ddof=ddof))
    B_last = B if static else B[-1]
    return {"Sigma": B_last @ F @ B_last.T + D, "F": F, "D": D, "factor_returns": f,
            "specific": U}


# ------------------------------------------------------------------ portfolio side -----------
def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Minimum-variance weights Sigma^-1 1 / 1' Sigma^-1 1 (pseudo-inverse if singular)."""
    ones = np.ones(len(Sigma))
    try:
        x = np.linalg.solve(Sigma, ones)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(Sigma) @ ones
    return x / x.sum()


def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    return float(w @ Sigma @ w)


def bias_ratio(w: np.ndarray, Sigma_hat: np.ndarray, Sigma_realized: np.ndarray) -> float:
    """Realized / predicted portfolio variance for weights optimized on Sigma_hat."""
    return portfolio_variance(w, Sigma_realized) / portfolio_variance(w, Sigma_hat)


def condition_report(Sigma: np.ndarray, n_obs: int | None = None) -> dict:
    vals = np.linalg.eigvalsh(Sigma)
    return {"n": len(Sigma), "n_obs": n_obs, "rank": int(np.linalg.matrix_rank(Sigma)),
            "cond": float(np.linalg.cond(Sigma)), "min_eig": float(vals.min()),
            "max_eig": float(vals.max())}


def check_invertible(Sigma: np.ndarray, n_obs: int | None = None, max_cond: float = 1e8) -> dict:
    """Refuse a covariance an optimizer should not invert. The guard for the N > T trap.

    Raises ValueError when the matrix is rank-deficient or when its condition number exceeds
    max_cond. `n_obs` is only used to explain WHY: a sample covariance from n_obs observations
    has rank at most n_obs - 1. It must not by itself be the test, or the guard would also
    refuse the shrunk matrix that fixes the problem - which is full rank and well conditioned
    from the same short sample. Returns the condition report, with `short_sample` set when
    n_obs <= N, so a caller can insist the matrix was regularized.
    """
    rep = condition_report(Sigma, n_obs)
    n = rep["n"]
    rep["short_sample"] = short = n_obs is not None and n_obs <= n
    why = (f" (a sample covariance from {n_obs} observations has rank at most {n_obs - 1})"
           if short else "")
    if rep["rank"] < n:
        raise ValueError(f"covariance is rank {rep['rank']} < {n}: singular{why}; shrink it or"
                         f" impose a factor structure before inverting")
    if rep["cond"] > max_cond:
        raise ValueError(f"condition number {rep['cond']:.2e} exceeds {max_cond:.0e}{why}; the"
                         f" optimizer will amplify estimation error by that factor")
    return rep


# ------------------------------------------------------------------ synthetic data -----------
def simulate_factor_returns(n_assets: int = 100, n_obs: int = 250, n_factors: int = 3,
                            seed: int = SEED) -> dict:
    """Daily returns from a k-factor model with heterogeneous loadings and idiosyncratic vol."""
    rng = np.random.default_rng(seed)
    f_vol = np.array([0.010, 0.006, 0.004])[:n_factors]
    B = np.column_stack([rng.normal(1.0, 0.3, n_assets)] +
                        [rng.normal(0.0, 0.5, n_assets) for _ in range(n_factors - 1)])
    i_vol = rng.uniform(0.008, 0.020, n_assets)
    Sigma_true = B @ np.diag(f_vol ** 2) @ B.T + np.diag(i_vol ** 2)
    f = rng.normal(0.0, 1.0, (n_obs, n_factors)) * f_vol
    e = rng.normal(0.0, 1.0, (n_obs, n_assets)) * i_vol
    return {"R": f @ B.T + e, "Sigma_true": Sigma_true, "B": B, "f_vol": f_vol, "i_vol": i_vol}


def estimators(R: np.ndarray, B_exposures: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The candidate covariance estimates for one sample, by name."""
    out = {"sample": sample_cov(R),
           "ledoit_wolf identity": ledoit_wolf(R, "identity", ddof=1)[0],
           "ledoit_wolf const-corr": ledoit_wolf(R, "constant_correlation", ddof=1)[0],
           "ewma lambda=0.94": ewma_cov(R, LAMBDA_DAILY),
           "pca 3 factors": pca_factor_cov(R, 3)[0]}
    if B_exposures is not None:
        out["barra-style (exposures known)"] = barra_cov(R, B_exposures)["Sigma"]
    return out


# ------------------------------------------------------------------ probe payloads -----------
SKLEARN_PROBE = _PROBE_HEAD + '''
import numpy as np
try:
    from sklearn.covariance import LedoitWolf
except ImportError:
    print("scikit-learn is not installed here; the identity-target formula is unchecked")
else:
    R = rm.simulate_factor_returns(30, 300, seed=5)["R"]
    lw = LedoitWolf().fit(R)
    mine, s_mine = rm.ledoit_wolf(R, "identity", ddof=0)
    print(f"max |cov diff| {np.max(np.abs(lw.covariance_ - mine)):.2e}; shrinkage "
          f"{lw.shrinkage_:.6f} vs {s_mine:.6f} (diff {abs(lw.shrinkage_ - s_mine):.1e})")
'''

PYPFOPT_PROBE = _PROBE_HEAD + '''
import numpy as np, pandas as pd
try:
    from pypfopt.risk_models import CovarianceShrinkage
except ImportError:
    print("PyPortfolioOpt is not installed here; the constant-correlation formula is unchecked")
else:
    R = rm.simulate_factor_returns(30, 300, seed=5)["R"]
    cs = CovarianceShrinkage(pd.DataFrame(R), returns_data=True, frequency=1)
    theirs = cs.ledoit_wolf("constant_correlation").to_numpy()
    mine, s_mine = rm.ledoit_wolf(R, "constant_correlation", ddof=1)
    print(f"max |cov diff| {np.max(np.abs(theirs - mine)):.2e}; delta {cs.delta:.6f} vs "
          f"{s_mine:.6f} (diff {abs(cs.delta - s_mine):.1e})")
'''


# ------------------------------------------------------------------ demo ---------------------
if __name__ == "__main__":
    t0 = time.time()
    ann = np.sqrt(PERIODS)
    print("=" * 100)
    print("1. N > T: THE SAMPLE COVARIANCE IS SINGULAR AND THE OPTIMIZER FINDS THE HOLE")
    print("   100 assets from a 3-factor DGP; minimum-variance weights from each estimate;"
          " 'true' uses the DGP covariance")
    print("=" * 100)
    print(f"{'T':>5} {'estimator':<22} {'rank':>5} {'cond':>10} {'gross lev':>10} {'max|w|':>8}"
          f" {'pred vol':>9} {'true vol':>9} {'true/pred var':>14}")
    print("-" * 100)
    n_assets = 100
    for n_obs in (60, 250, 1000):
        sim = simulate_factor_returns(n_assets, n_obs, seed=1)
        for name, Sig in (("sample", sample_cov(sim["R"])),
                          ("ledoit_wolf identity", ledoit_wolf(sim["R"], ddof=1)[0])):
            rep = condition_report(Sig, n_obs)
            w = gmv_weights(Sig)
            pred = portfolio_variance(w, Sig)
            pv = np.sqrt(max(pred, 0.0)) * ann
            tv = np.sqrt(portfolio_variance(w, sim["Sigma_true"])) * ann
            print(f"{n_obs:>5} {name:<22} {rep['rank']:>5} {rep['cond']:>10.2e}"
                  f" {np.abs(w).sum():>10.1f} {np.abs(w).max():>8.2f} {pv:>8.2%} {tv:>8.2%}"
                  f" {bias_ratio(w, Sig, sim['Sigma_true']):>14.3g}")
    print("-" * 100)
    print("At T=60 the sample covariance is rank 59 for 100 assets: the minimum-variance solution"
          " lands in its null")
    print("space, PREDICTS 0.00 % annualized vol, and delivers 243 %. The bias ratio is negative"
          " because the predicted")
    print("variance is negative - the arithmetic is meaningless, not merely optimistic.")
    sim60 = simulate_factor_returns(n_assets, 60, seed=1)
    for name, Sig in (("sample", sample_cov(sim60["R"])),
                      ("ledoit_wolf", ledoit_wolf(sim60["R"], ddof=1)[0])):
        try:
            rep = check_invertible(Sig, n_obs=60)
            print(f"check_invertible({name}, n_obs=60): ok, rank {rep['rank']}, cond"
                  f" {rep['cond']:.2e}, short_sample={rep['short_sample']}")
        except ValueError as e:
            print(f"check_invertible({name}, n_obs=60): ValueError: {e}")

    # ---------------------------------------------------------------- 2. predicted vs realized
    print("\n" + "=" * 100)
    print("2. PREDICTED vs REALIZED VARIANCE OF THE OPTIMIZED PORTFOLIO (the Michaud bias)")
    print("   50 assets, 250 in-sample days, minimum-variance weights; realized = next 250 days;"
          " median [min, max] over 20 seeds")
    print("=" * 100)
    n_assets, n_in, n_out, n_seeds = 50, 250, 250, 20
    table: dict[str, list] = {}
    shrink_id, shrink_cc = [], []
    for s in range(n_seeds):
        sim = simulate_factor_returns(n_assets, n_in + n_out, seed=100 + s)
        R_in, R_out = sim["R"][:n_in], sim["R"][n_in:]
        S_out = sample_cov(R_out)
        shrink_id.append(ledoit_wolf(R_in, "identity", ddof=1)[1])
        shrink_cc.append(ledoit_wolf(R_in, "constant_correlation", ddof=1)[1])
        for name, Sig in estimators(R_in, sim["B"]).items():
            w = gmv_weights(Sig)
            pred = portfolio_variance(w, Sig)
            table.setdefault(name, []).append((np.sqrt(pred) * ann,
                                               np.sqrt(portfolio_variance(w, sim["Sigma_true"])) * ann,
                                               np.sqrt(portfolio_variance(w, S_out)) * ann,
                                               portfolio_variance(w, sim["Sigma_true"]) / pred,
                                               portfolio_variance(w, S_out) / pred))
    print(f"{'estimator':<30} {'pred vol':>9} {'true vol':>9} {'OOS vol':>9}"
          f" {'true/pred':>20} {'OOS/pred':>20}")
    print("-" * 100)
    for name, rows in table.items():
        a = np.array(rows)
        med = np.median(a, axis=0)
        print(f"{name:<30} {med[0]:>8.2%} {med[1]:>8.2%} {med[2]:>8.2%}"
              f" {med[3]:>6.2f} [{a[:, 3].min():.2f}, {a[:, 3].max():.2f}]"
              f" {med[4]:>6.2f} [{a[:, 4].min():.2f}, {a[:, 4].max():.2f}]")
    print("-" * 100)
    print(f"Ledoit-Wolf intensity, median over seeds: identity target {np.median(shrink_id):.3f},"
          f" constant-correlation target {np.median(shrink_cc):.3f}")
    print("Reading it: the sample-covariance portfolio realizes MORE variance than it promised;"
          " every structured estimate narrows the gap. Report this ratio, not the in-sample frontier.")

    # ---------------------------------------------------------------- 3. EWMA
    print("\n" + "=" * 100)
    print("3. RISKMETRICS EWMA")
    print("=" * 100)
    print(f"half-life of lambda 0.94 = {half_life(0.94):.2f} periods; 0.97 = {half_life(0.97):.2f};"
          f" pandas ewm(span=180) is lambda = {span_to_lambda(180):.5f}, half-life"
          f" {half_life(span_to_lambda(180)):.1f}")
    print(f"RiskMetrics Eq. [5.26] effective days K = ln(0.01)/ln(lambda):"
          f" lambda 0.94 -> {effective_days(0.94):.1f}, lambda 0.97 -> {effective_days(0.97):.1f}"
          f" (Table 5.7 of the Technical Document prints 74 and 151 at the 1 % tolerance level)")
    n_a, n_o = 50, 250
    sim_e = simulate_factor_returns(n_a, n_o, seed=11)
    c_ewma = np.linalg.cond(ewma_cov(sim_e["R"], LAMBDA_DAILY))
    c_samp = np.linalg.cond(sample_cov(sim_e["R"]))
    print(f"N={n_a}, T={n_o}: condition number of the EWMA(0.94) covariance {c_ewma:.2e} against"
          f" {c_samp:.2e} for the sample covariance.")
    print(f"It is nominally full rank, but it carries the estimation error of a"
          f" {effective_days(0.94):.0f}-day sample: T_eff/N = {effective_days(0.94) / n_a:.1f}"
          f" against {n_o / n_a:.1f} for the equally weighted estimate. That is why the"
          f" minimum-variance")
    print("portfolio built on it in section 2 predicted 3.47 % vol and realized 16.70 %. EWMA at"
          " 0.94 is a one-day")
    print("FORECAST, not a matrix to invert; for optimization use a longer lambda, or shrink it,"
          " or impose factors.")
    x = simulate_factor_returns(1, 2000, 1, seed=7)["R"][:, 0]
    s2 = ewma_variance_path(x, 0.94)
    pd_adj_false = pd.Series(x ** 2).ewm(alpha=1 - 0.94, adjust=False).mean().to_numpy()
    pd_adj_true = pd.Series(x ** 2).ewm(alpha=1 - 0.94, adjust=True).mean().to_numpy()
    print(f"max |recursion - pandas ewm(alpha=0.06, adjust=False)| over 2,000 days:"
          f" {np.max(np.abs(s2[1:] - pd_adj_false)):.2e}")
    print(f"pandas' default is adjust=True: relative gap to the recursion at day 10"
          f" {abs(pd_adj_true[9] / pd_adj_false[9] - 1):.1%}, day 50"
          f" {abs(pd_adj_true[49] / pd_adj_false[49] - 1):.2%}, day 2000"
          f" {abs(pd_adj_true[-1] / pd_adj_false[-1] - 1):.1e}")
    print("PyPortfolioOpt's exp_cov(span=180) uses ewm(span=180) with pandas' adjust=True default on demeaned"
          " products: a much slower decay than RiskMetrics, and not the recursion.")

    # ---------------------------------------------------------------- 4. PCA / MP
    print("\n" + "=" * 100)
    print("4. PCA FACTOR COVARIANCE AND THE MARCHENKO-PASTUR EDGE")
    print("=" * 100)
    for n_a, n_o in ((100, 1000), (50, 250)):
        sim = simulate_factor_returns(n_a, n_o, seed=3)
        k, vals, edge = n_factors_above_mp(sim["R"])
        print(f"N={n_a:>3}, T={n_o:>4}: correlation eigenvalues above the MP edge"
              f" {edge:.2f}: {k} (DGP has 3); top four eigenvalues"
              f" {np.array2string(vals[:4], precision=2)}")

    # ---------------------------------------------------------------- 5. library cross-checks
    print("\n" + "=" * 100)
    print("5. LIBRARY CROSS-CHECKS, EACH IN ITS OWN INTERPRETER")
    print("=" * 100)
    for label, code in (
        ("sklearn.covariance.LedoitWolf (identity target)", SKLEARN_PROBE),
        ("pypfopt CovarianceShrinkage (constant correlation)", PYPFOPT_PROBE),
    ):
        rc, out = probe(code)
        if rc == 0 and out:
            print(f"{label}: {out}")
        else:
            print(f"{label}: probe exited {rc} with no result - see the import-order trap below")
    rc_bad, out_bad = probe("try:\n import sklearn.covariance\n import osqp\n print('survived')\n"
                            "except ImportError as e:\n print('missing', e.name)")
    rc_good, out_good = probe("try:\n import osqp\n import sklearn.covariance\n print('survived')\n"
                              "except ImportError as e:\n print('missing', e.name)")
    print(f"\nIMPORT ORDER: 'import sklearn.covariance' then 'import osqp' -> returncode {rc_bad}"
          f" ({out_bad or 'no output'});")
    print(f"              'import osqp' then 'import sklearn.covariance' -> returncode {rc_good}"
          f" ({out_good or 'no output'}).")
    print("osqp is a cvxpy dependency, so PyPortfolioOpt, Riskfolio-Lib and skfolio all pull it"
          " in. Where the")
    print("returncodes differ, the interpreter dies on the second import and no try/except can"
          " catch it: import the")
    print("cvxpy side first, or run each library in its own process as this script does.")

    print("\n" + "=" * 100)
    print("THE RULE:  N/T decides the estimator. Shrink or impose a factor structure before you invert,")
    print("           and report realized / predicted variance out of sample, not the in-sample frontier.")
    print("=" * 100)
    print(f"total runtime {time.time() - t0:.1f}s")
