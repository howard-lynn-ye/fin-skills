"""Zero curves, Nelson-Siegel/Svensson, and the Vasicek / CIR / Hull-White bond formulas.

WHY this exists: a "5-year zero rate" is not a number until four conventions are attached to
it (day count, compounding frequency, the instrument it was stripped from, the interpolation
between pillars), and every library and every counterparty picks its own defaults. The traps
here are the ones that survive a glance:

  * COMPOUNDING AND DAY COUNT. The same discount factor quoted as an annually-compounded
    30/360 rate, a continuously-compounded ACT/365 rate, or an ACT/360 money-market rate
    differs by basis points -- measured below. A curve built in one convention and consumed
    in another is off by that much at every pillar, and nothing raises.

  * NELSON-SIEGEL LAMBDA. The decay parameter lambda is weakly identified: with it free, a
    least-squares fit moves it across a wide range from one resample of the same curve to the
    next, and the betas move with it. Diebold and Li (2006, J. Econometrics 130, 337-364)
    fixed lambda = 0.0609 with maturity in MONTHS, "the value that maximizes the loading on
    the medium-term factor at exactly 30 months"; in years that is 0.0609 x 12 = 0.7308. The
    demo measures the spread of a free lambda against a fixed one, and re-derives where the
    curvature loading actually peaks (it is 29.4 months, not 30 -- the paper rounded).

  * CIR SIMULATION. The Euler scheme for dr = kappa (theta - r) dt + sigma sqrt(r) dW takes
    sqrt of a negative rate the moment a step overshoots zero; a plain implementation returns
    NaN paths, a "fixed" one silently flips the sign. Full truncation (Lord, Koekkoek and
    van Dijk 2010) keeps the scheme defined; the demo counts the paths that would have died
    and measures the bias of each scheme against the closed-form bond price.

The closed forms are verified against QuantLib 1.43 when it is importable (Vasicek,
CoxIngersollRoss and HullWhite discountBond; NelsonSiegelFitting / SvenssonFitting evaluated
from parameters; InterestRate.equivalentRate; Thirty360 vs Actual365Fixed year fractions).

Usage:
    from term_structure import bootstrap_from_par, zero_rate, fit_nelson_siegel, \
        vasicek_bond, cir_bond, hull_white_bond, simulate_cir
"""
from __future__ import annotations

import math
from datetime import date
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

DIEBOLD_LI_LAMBDA_PER_MONTH = 0.0609            # Diebold & Li (2006), tau in months
DIEBOLD_LI_LAMBDA_PER_YEAR = 0.0609 * 12.0      # the same decay with tau in years
DIEBOLD_LI_MATURITIES_MONTHS = (3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108,
                                120)            # Diebold & Li (2006) Table 1
COMPOUNDINGS = ("continuous", "annual", "semiannual", "simple")
DAY_COUNTS = ("ACT/365", "ACT/360", "30/360")


# ------------------------------------------------------------------ bootstrap
def bootstrap_from_par(par_rates: Sequence[float]) -> np.ndarray:
    """Discount factors P(0, n), n = 1..N years, from par rates of annual-coupon bonds.

    A par bond pays c each year and 100 at maturity and is worth 100, so
        1 = c * sum_{i<=n} P_i + P_n   =>   P_n = (1 - c * sum_{i<n} P_i) / (1 + c).
    Exact, no interpolation, no library: the defining identity is `par_bond_price` == 100.
    """
    dfs: list[float] = []
    for c in par_rates:
        if c <= -1.0:
            raise ValueError("par rate below -100%")
        dfs.append((1.0 - c * sum(dfs)) / (1.0 + c))
        if dfs[-1] <= 0.0:
            raise ValueError("bootstrapped discount factor is not positive: inconsistent par "
                             "rates")
    return np.array(dfs)


def par_bond_price(coupon: float, dfs: Sequence[float], n: int | None = None) -> float:
    """Price per 100 of an annual-coupon bond maturing in year n on a discount curve."""
    d = np.asarray(dfs, dtype=float)
    n = d.size if n is None else n
    return 100.0 * (coupon * d[:n].sum() + d[n - 1])


def zero_rate(df: float, t: float, compounding: str = "continuous") -> float:
    """Zero rate implied by a discount factor over t years, in the named compounding."""
    if df <= 0.0 or t <= 0.0:
        raise ValueError("df and t must be positive")
    if compounding == "continuous":
        return -math.log(df) / t
    if compounding == "annual":
        return df ** (-1.0 / t) - 1.0
    if compounding == "semiannual":
        return 2.0 * (df ** (-1.0 / (2.0 * t)) - 1.0)
    if compounding == "simple":
        return (1.0 / df - 1.0) / t
    raise ValueError(f"compounding must be one of {COMPOUNDINGS}")


def discount_factor(z: float, t: float, compounding: str = "continuous") -> float:
    if compounding == "continuous":
        return math.exp(-z * t)
    if compounding == "annual":
        return (1.0 + z) ** (-t)
    if compounding == "semiannual":
        return (1.0 + z / 2.0) ** (-2.0 * t)
    if compounding == "simple":
        return 1.0 / (1.0 + z * t)
    raise ValueError(f"compounding must be one of {COMPOUNDINGS}")


def zero_curve_from_discounts(dfs: Sequence[float], times: Sequence[float],
                              compounding: str = "continuous") -> np.ndarray:
    return np.array([zero_rate(d, t, compounding) for d, t in zip(dfs, times)])


def year_fraction(d0: date, d1: date, convention: str = "ACT/365") -> float:
    """ACT/365 (Fixed), ACT/360, or 30/360 Bond Basis (ISDA 2006 4.16(f): D1 -> 30 if 31;
    D2 -> 30 if 31 and D1 is 30 or 31)."""
    if convention == "ACT/365":
        return (d1 - d0).days / 365.0
    if convention == "ACT/360":
        return (d1 - d0).days / 360.0
    if convention == "30/360":
        dd0 = min(d0.day, 30)
        dd1 = 30 if (d1.day == 31 and dd0 == 30) else d1.day
        return (360 * (d1.year - d0.year) + 30 * (d1.month - d0.month) + (dd1 - dd0)) / 360.0
    raise ValueError(f"convention must be one of {DAY_COUNTS}")


def convention_table(df: float, d0: date, d1: date,
                     base: tuple[str, str] = ("30/360", "annual")) -> list[dict[str, float]]:
    """One discount factor, every (day count, compounding) pair, and the bp gap to the base."""
    rows = []
    z_base = zero_rate(df, year_fraction(d0, d1, base[0]), base[1])
    for dc in DAY_COUNTS:
        t = year_fraction(d0, d1, dc)
        for comp in ("annual", "semiannual", "continuous", "simple"):
            z = zero_rate(df, t, comp)
            rows.append({"day_count": dc, "compounding": comp, "t": t, "zero": z,
                         "bp_vs_base": (z - z_base) * 1e4})
    return rows


# ------------------------------------------------------------------ Nelson-Siegel / Svensson
def ns_loadings(t, lam: float) -> np.ndarray:
    """Columns: level 1, slope (1 - e^{-lt})/(lt), curvature (1 - e^{-lt})/(lt) - e^{-lt}."""
    t = np.asarray(t, dtype=float)
    x = lam * t
    slope = np.where(x > 1e-12, (1.0 - np.exp(-x)) / np.where(x > 1e-12, x, 1.0), 1.0)
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(t), slope, curv])


def nelson_siegel(t, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """y(t) = b0 + b1 (1-e^{-lt})/(lt) + b2 [(1-e^{-lt})/(lt) - e^{-lt}]. Same form (and the
    same parameter order b0, b1, b2, lambda) as QuantLib's NelsonSiegelFitting."""
    return ns_loadings(t, lam) @ np.array([beta0, beta1, beta2])


def svensson(t, beta0: float, beta1: float, beta2: float, beta3: float, lam1: float,
             lam2: float) -> np.ndarray:
    """Svensson (1994): Nelson-Siegel plus b3 [(1-e^{-l2 t})/(l2 t) - e^{-l2 t}]."""
    return nelson_siegel(t, beta0, beta1, beta2, lam1) + beta3 * ns_loadings(t, lam2)[:, 2]


def fit_nelson_siegel(t, y, lam: float | None = None,
                      lam_bounds: tuple[float, float] = (0.01, 5.0)) -> dict[str, object]:
    """Fit by OLS on the loadings when lambda is fixed (Diebold-Li's method), otherwise by
    nonlinear least squares over (b0, b1, b2, lambda)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if lam is not None:
        X = ns_loadings(t, lam)
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = X @ betas - y
        return {"betas": tuple(float(b) for b in betas), "lam": float(lam), "fixed": True,
                "rmse": float(np.sqrt(np.mean(res ** 2)))}
    lam0 = 0.5 * (lam_bounds[0] + lam_bounds[1])
    best = None
    for l0 in np.geomspace(max(lam_bounds[0] * 1.5, 1e-3), lam_bounds[1] * 0.7, 5):
        b0 = fit_nelson_siegel(t, y, l0)["betas"]
        sol = least_squares(lambda p: ns_loadings(t, p[3]) @ p[:3] - y, [*b0, l0],
                            bounds=([-np.inf] * 3 + [lam_bounds[0]],
                                    [np.inf] * 3 + [lam_bounds[1]]),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    res = ns_loadings(t, best.x[3]) @ best.x[:3] - y
    return {"betas": tuple(float(b) for b in best.x[:3]), "lam": float(best.x[3]),
            "fixed": False, "rmse": float(np.sqrt(np.mean(res ** 2))), "lam0": lam0}


def fit_svensson(t, y, lam_bounds: tuple[float, float] = (0.01, 5.0)) -> dict[str, object]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def resid(p):
        return svensson(t, *p) - y

    best = None
    for l1, l2 in ((0.3, 3.0), (0.7, 0.1), (1.5, 0.3), (0.2, 1.0)):
        b = fit_nelson_siegel(t, y, l1)["betas"]
        sol = least_squares(resid, [*b, 0.0, l1, l2],
                            bounds=([-np.inf] * 4 + [lam_bounds[0]] * 2,
                                    [np.inf] * 4 + [lam_bounds[1]] * 2),
                            xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    return {"params": tuple(float(v) for v in best.x),
            "rmse": float(np.sqrt(np.mean(resid(best.x) ** 2)))}


def curvature_loading_peak() -> float:
    """x* maximising (1 - e^{-x})/x - e^{-x}; the curvature loading peaks at tau = x*/lambda."""
    res = minimize_scalar(lambda x: -((1.0 - math.exp(-x)) / x - math.exp(-x)),
                          bounds=(0.5, 5.0), method="bounded", options={"xatol": 1e-12})
    return float(res.x)


def diebold_li_lambda_facts() -> dict[str, float]:
    x_star = curvature_loading_peak()
    return {"x_star": x_star,
            "lambda_exact_for_30_months": x_star / 30.0,
            "lambda_paper": DIEBOLD_LI_LAMBDA_PER_MONTH,
            "peak_months_at_paper_lambda": x_star / DIEBOLD_LI_LAMBDA_PER_MONTH,
            "lambda_paper_per_year": DIEBOLD_LI_LAMBDA_PER_YEAR}


def synthetic_yields(seed: int = 0, noise_bp: float = 5.0,
                     betas: tuple[float, float, float] = (6.0, -2.0, -1.5),
                     lam: float = DIEBOLD_LI_LAMBDA_PER_MONTH
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nelson-Siegel yields in percent at the Diebold-Li maturities (months) plus iid noise."""
    rng = np.random.default_rng(seed)
    t = np.array(DIEBOLD_LI_MATURITIES_MONTHS, dtype=float)
    clean = nelson_siegel(t, *betas, lam)
    return t, clean, clean + rng.normal(0.0, noise_bp / 100.0, t.size)


def lambda_stability(n_boot: int = 200, seed: int = 0, noise_bp: float = 5.0
                     ) -> dict[str, object]:
    """Residual bootstrap of one noisy curve, fitted with lambda free and lambda fixed."""
    t, clean, y = synthetic_yields(seed, noise_bp)
    rng = np.random.default_rng(seed + 1)
    base_free = fit_nelson_siegel(t, y)
    base_fixed = fit_nelson_siegel(t, y, DIEBOLD_LI_LAMBDA_PER_MONTH)
    resid = y - nelson_siegel(t, *base_free["betas"], base_free["lam"])
    lams, b2_free, b2_fixed, b1_free, b1_fixed = [], [], [], [], []
    for _ in range(n_boot):
        yb = nelson_siegel(t, *base_free["betas"], base_free["lam"]) + rng.choice(resid, t.size)
        fr = fit_nelson_siegel(t, yb)
        fx = fit_nelson_siegel(t, yb, DIEBOLD_LI_LAMBDA_PER_MONTH)
        lams.append(fr["lam"])
        b2_free.append(fr["betas"][2])
        b1_free.append(fr["betas"][1])
        b2_fixed.append(fx["betas"][2])
        b1_fixed.append(fx["betas"][1])
    lams = np.array(lams)
    return {"base_free": base_free, "base_fixed": base_fixed, "n_boot": n_boot,
            "lam_p05": float(np.percentile(lams, 5)), "lam_p95": float(np.percentile(lams, 95)),
            "lam_std": float(lams.std(ddof=1)), "lam_min": float(lams.min()),
            "lam_max": float(lams.max()),
            "beta2_std_free": float(np.std(b2_free, ddof=1)),
            "beta2_std_fixed": float(np.std(b2_fixed, ddof=1)),
            "beta1_std_free": float(np.std(b1_free, ddof=1)),
            "beta1_std_fixed": float(np.std(b1_fixed, ddof=1)),
            "rmse_free": base_free["rmse"], "rmse_fixed": base_fixed["rmse"]}


def svensson_collinearity(t, lam1: float, ratios: Sequence[float] = (2.0, 1.2, 1.05, 1.01)
                          ) -> list[tuple[float, float]]:
    """Condition number of the Svensson loading matrix as lambda2 -> lambda1: the two
    curvature columns become identical and beta2, beta3 are no longer separately identified."""
    t = np.asarray(t, dtype=float)
    out = []
    for ratio in ratios:
        X = np.column_stack([ns_loadings(t, lam1), ns_loadings(t, lam1 * ratio)[:, 2]])
        out.append((ratio, float(np.linalg.cond(X))))
    return out


# ------------------------------------------------------------------ short-rate bonds
def vasicek_bond(r: float, tau: float, a: float, b: float, sigma: float) -> float:
    """Vasicek (1977): dr = a (b - r) dt + sigma dW.
        B = (1 - e^{-a tau}) / a,
        A = exp[(b - sigma^2/(2 a^2)) (B - tau) - sigma^2 B^2 / (4 a)],   P = A e^{-B r}.
    QuantLib's Vasicek::A carries the same expression with an extra lambda*sigma/a term
    (market price of risk, default 0)."""
    if a <= 0.0:
        raise ValueError("a must be positive")
    B = (1.0 - math.exp(-a * tau)) / a
    A = math.exp((b - 0.5 * sigma * sigma / (a * a)) * (B - tau)
                 - 0.25 * sigma * sigma * B * B / a)
    return A * math.exp(-B * r)


def cir_bond(r: float, tau: float, kappa: float, theta: float, sigma: float) -> float:
    """Cox, Ingersoll and Ross (1985): dr = kappa (theta - r) dt + sigma sqrt(r) dW.
        h = sqrt(kappa^2 + 2 sigma^2),
        A = [2 h e^{(kappa + h) tau / 2} / (2 h + (kappa + h)(e^{h tau} - 1))]^{2 kappa theta / sigma^2},
        B = 2 (e^{h tau} - 1) / (2 h + (kappa + h)(e^{h tau} - 1)),   P = A e^{-B r}.
    Same expressions as QuantLib's CoxIngersollRoss::A and ::B (parameter order there is
    r0, theta, k, sigma)."""
    h = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    e = math.exp(h * tau) - 1.0
    denom = 2.0 * h + (kappa + h) * e
    A = (2.0 * h * math.exp(0.5 * (kappa + h) * tau) / denom) ** (2.0 * kappa * theta
                                                                    / (sigma * sigma))
    B = 2.0 * e / denom
    return A * math.exp(-B * r)


def hull_white_bond(t: float, T: float, r_t: float, a: float, sigma: float,
                    P0: Callable[[float], float], f0: Callable[[float], float]) -> float:
    """Hull-White one factor: dr = (theta(t) - a r) dt + sigma dW fitted to P0(.).
        B = (1 - e^{-a (T - t)}) / a,
        ln A = ln(P0(T)/P0(t)) + B f0(t) - sigma^2 (1 - e^{-2 a t}) B^2 / (4 a),   P = A e^{-B r_t},
    with f0(t) the initial instantaneous forward. QuantLib's HullWhite::A writes the last term
    as 0.25 sigma^2 B(t,T)^2 B(0, 2t) with B(0,2t) = (1 - e^{-2at})/a: the same quantity."""
    B = (1.0 - math.exp(-a * (T - t))) / a
    lnA = (math.log(P0(T) / P0(t)) + B * f0(t)
           - 0.25 * sigma * sigma * (1.0 - math.exp(-2.0 * a * t)) * B * B / a)
    return math.exp(lnA - B * r_t)


def affine_ab(model: str, tau: float, **kw) -> tuple[float, float]:
    """(A, B) of the affine form P = A e^{-B r} for 'vasicek' or 'cir'.

    Separated from the price so A and B can be checked individually rather than only through
    their product: B = -d ln P / dr exactly (P is affine in r in the exponent), and A = P e^{B r}
    at any r. The demo does both numerically.
    """
    if model == "vasicek":
        a, b, sigma = kw["a"], kw["b"], kw["sigma"]
        if a <= 0.0:
            raise ValueError("a must be positive")
        B = (1.0 - math.exp(-a * tau)) / a
        A = math.exp((b - 0.5 * sigma * sigma / (a * a)) * (B - tau)
                     - 0.25 * sigma * sigma * B * B / a)
        return A, B
    if model == "cir":
        kappa, theta, sigma = kw["kappa"], kw["theta"], kw["sigma"]
        h = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
        e = math.exp(h * tau) - 1.0
        denom = 2.0 * h + (kappa + h) * e
        A = (2.0 * h * math.exp(0.5 * (kappa + h) * tau) / denom) ** (2.0 * kappa * theta
                                                                     / (sigma * sigma))
        return A, 2.0 * e / denom
    raise ValueError("model must be 'vasicek' or 'cir'")


def deterministic_bond(r: float, tau: float, speed: float, level: float) -> float:
    """The sigma -> 0 limit both models must collapse to: r(s) solves dr = speed(level - r) ds,
    so int_0^tau r ds = level*tau + (r - level) * B(tau) with the SAME B, and
    P = exp(-that). An independent check of A and B together that uses neither."""
    B = (1.0 - math.exp(-speed * tau)) / speed
    return math.exp(-(level * tau + (r - level) * B))


def simulate_cir(r0: float, kappa: float, theta: float, sigma: float, T: float,
                 steps: int = 120, paths: int = 10_000, seed: int = 0,
                 scheme: str = "euler") -> dict[str, object]:
    """Euler paths of CIR and the Monte Carlo bond price E[exp(-int r dt)].

    'euler'           : sqrt(r) of a negative r is NaN and the path is dead from then on.
    'full_truncation' : Lord, Koekkoek & van Dijk (2010): use max(r, 0) in the drift, the
                        diffusion and the discounting; r itself may dip below zero.
    Returns the count of steps that went negative, the NaN paths, and the MC bond price with
    its standard error, to compare with `cir_bond`.
    """
    if scheme not in ("euler", "full_truncation"):
        raise ValueError("scheme must be 'euler' or 'full_truncation'")
    rng = np.random.default_rng(seed)
    dt = T / steps
    r = np.full(paths, r0)
    integral = np.zeros(paths)
    negative_steps = 0
    with np.errstate(invalid="ignore"):
        for _ in range(steps):
            z = rng.standard_normal(paths)
            if scheme == "euler":
                r_use = r
            else:
                r_use = np.maximum(r, 0.0)
            integral += r_use * dt
            r = r + kappa * (theta - r_use) * dt + sigma * np.sqrt(r_use) * math.sqrt(dt) * z
            negative_steps += int(np.sum(r < 0.0))
    disc = np.exp(-integral)
    n_nan = int(np.sum(~np.isfinite(disc)))
    finite = disc[np.isfinite(disc)]
    return {"negative_steps": negative_steps, "nan_paths": n_nan,
            "mc_price": float(finite.mean()) if finite.size else float("nan"),
            "mc_se": float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1
            else float("nan"), "paths": paths, "steps": steps}


# ------------------------------------------------------------------ optional cross-checks
def quantlib_cross_checks(par_rates: Sequence[float], dfs: Sequence[float], d0: date,
                          d1: date, ns: tuple[float, float, float, float],
                          sv: tuple[float, ...]) -> dict[str, object] | None:
    try:
        import QuantLib as ql
    except ImportError:
        return None
    today = ql.Date(d0.day, d0.month, d0.year)
    ql.Settings.instance().evaluationDate = today
    out: dict[str, object] = {"version": ql.__version__}
    dcq = ql.Actual365Fixed()

    def flat(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dcq))

    # Vasicek is (r0, SPEED, level, sigma); CoxIngersollRoss is (r0, LEVEL, speed, sigma).
    # The middle two are in the opposite order between the two classes, and both accept the
    # swap silently - record what each mistake costs.
    out["vasicek"] = ql.Vasicek(0.05, 0.1, 0.05, 0.01).discountBond(1.0, 5.0, 0.04)
    out["vasicek_swapped"] = ql.Vasicek(0.05, 0.05, 0.1, 0.01).discountBond(1.0, 5.0, 0.04)
    out["cir"] = ql.CoxIngersollRoss(0.05, 0.05, 0.3, 0.1).discountBond(1.0, 5.0, 0.04)
    out["cir_swapped"] = ql.CoxIngersollRoss(0.05, 0.3, 0.05, 0.1).discountBond(1.0, 5.0, 0.04)
    out["hull_white"] = ql.HullWhite(flat(0.05), 0.1, 0.01).discountBond(1.0, 5.0, 0.04)
    ir = ql.InterestRate(0.05, dcq, ql.Compounded, ql.Annual)
    out["five_pct_annual_as_continuous"] = ir.equivalentRate(ql.Continuous, ql.Annual, 5.0).rate()
    dq1 = ql.Date(d1.day, d1.month, d1.year)
    out["yf_act365"] = ql.Actual365Fixed().yearFraction(today, dq1)
    out["yf_act360"] = ql.Actual360().yearFraction(today, dq1)
    out["yf_30360"] = ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(today, dq1)
    curve = ql.FittedBondDiscountCurve(today, ql.NelsonSiegelFitting(), ql.Array(list(ns)),
                                       today + ql.Period(30, ql.Years), dcq)
    out["ns_disc_5y"] = curve.discount(5.0)
    curve_sv = ql.FittedBondDiscountCurve(today, ql.SvenssonFitting(), ql.Array(list(sv)),
                                          today + ql.Period(30, ql.Years), dcq)
    out["sv_disc_5y"] = curve_sv.discount(5.0)
    # reprice the par bonds on the bootstrapped curve with QuantLib's bond engine
    dc30 = ql.Thirty360(ql.Thirty360.BondBasis)
    dates = [today] + [today + ql.Period(n, ql.Years) for n in range(1, len(dfs) + 1)]
    disc_curve = ql.YieldTermStructureHandle(ql.DiscountCurve(dates, [1.0, *dfs], dc30))
    engine = ql.DiscountingBondEngine(disc_curve)
    prices = []
    for n, c in enumerate(par_rates, 1):
        sched = ql.Schedule(today, today + ql.Period(n, ql.Years), ql.Period(ql.Annual),
                            ql.NullCalendar(), ql.Unadjusted, ql.Unadjusted,
                            ql.DateGeneration.Backward, False)
        bond = ql.FixedRateBond(0, 100.0, sched, [c], dc30)
        bond.setPricingEngine(engine)
        prices.append(bond.cleanPrice())
    out["par_bond_prices"] = prices
    return out


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    W = 96
    print("=" * W)
    print("TERM STRUCTURE MODELS -- bootstrap, conventions, Nelson-Siegel, short-rate bonds")
    print("=" * W)

    # ------------------------------------------------------------- 1. bootstrap
    par = [0.030, 0.035, 0.040, 0.044, 0.047]
    dfs = bootstrap_from_par(par)
    times = np.arange(1, 6, dtype=float)
    print("\n1. BOOTSTRAP FROM PAR RATES (annual coupons)")
    print(f"   {'year':>4} {'par':>7} {'DF':>12} {'zero cont':>10} {'zero annual':>12} "
          f"{'par bond reprice':>17}")
    for n, (c, d, t) in enumerate(zip(par, dfs, times), 1):
        print(f"   {n:>4} {c:>7.3%} {d:>12.8f} {zero_rate(d, t):>10.5%} "
              f"{zero_rate(d, t, 'annual'):>12.5%} {par_bond_price(c, dfs, n):>17.10f}")
    worst = max(abs(par_bond_price(c, dfs, n) - 100.0) for n, c in enumerate(par, 1))
    print(f"   every par bond reprices to 100 on its own curve: worst |price - 100| = {worst:.1e}")
    back = zero_curve_from_discounts(dfs, times, "annual")
    print(f"   zero rates from the DFs and back: max |DF - DF(zero)| = "
          f"{max(abs(discount_factor(z, t, 'annual') - d) for z, t, d in zip(back, times, dfs)):.1e}")

    # ------------------------------------------------------------- 2. conventions
    d0, d1 = date(2026, 9, 8), date(2031, 9, 8)
    df5 = float(dfs[-1])
    print(f"\n2. THE SAME 5y DISCOUNT FACTOR {df5:.8f} ({d0} -> {d1}) AS A ZERO RATE, PER CONVENTION")
    print(f"   base: 30/360, annual (what the annual-coupon bootstrap actually produced)")
    print(f"   {'day count':>9} {'compounding':>12} {'T (years)':>10} {'zero rate':>10} "
          f"{'bp vs base':>11}")
    rows = convention_table(df5, d0, d1)
    for row in rows:
        print(f"   {row['day_count']:>9} {row['compounding']:>12} {row['t']:>10.6f} "
              f"{row['zero']:>10.5%} {row['bp_vs_base']:>+11.2f}")
    bp = {(r_["day_count"], r_["compounding"]): r_["bp_vs_base"] for r_ in rows}
    print(f"   compounding alone (30/360 annual -> continuous): {bp[('30/360', 'continuous')]:+.2f} bp")
    print(f"   day count alone (30/360 -> ACT/365, annual):     {bp[('ACT/365', 'annual')]:+.2f} bp")
    print(f"   day count alone (30/360 -> ACT/360, annual):     {bp[('ACT/360', 'annual')]:+.2f} bp")
    print(f"   both (ACT/360 continuous vs 30/360 annual):       {bp[('ACT/360', 'continuous')]:+.2f} bp")

    # ------------------------------------------------------------- 3. Nelson-Siegel
    facts = diebold_li_lambda_facts()
    print("\n3. NELSON-SIEGEL: DIEBOLD-LI'S LAMBDA, AND WHAT A FREE LAMBDA DOES")
    print(f"   curvature loading (1-e^-x)/x - e^-x peaks at x* = {facts['x_star']:.6f}")
    print(f"   lambda = {facts['lambda_paper']} (per month) puts the peak at "
          f"{facts['peak_months_at_paper_lambda']:.2f} months; exactly 30 months needs "
          f"lambda = {facts['lambda_exact_for_30_months']:.4f}")
    print(f"   in YEARS the same decay is lambda = 0.0609 x 12 = {facts['lambda_paper_per_year']:.4f}"
          f" (feeding 0.0609 to a curve in years puts the hump at "
          f"{facts['x_star'] / 0.0609:.0f} YEARS)")
    stab = lambda_stability(n_boot=200, seed=0, noise_bp=5.0)
    bf, bx = stab["base_free"], stab["base_fixed"]
    print(f"   synthetic curve: NS(6.0, -2.0, -1.5, lambda 0.0609) in percent at the 17 "
          f"Diebold-Li maturities + 5 bp noise, seed 0")
    print(f"   one fit, lambda free : betas ({bf['betas'][0]:.3f}, {bf['betas'][1]:.3f}, "
          f"{bf['betas'][2]:.3f}) lambda {bf['lam']:.4f} rmse {bf['rmse'] * 100:.2f} bp")
    print(f"   one fit, lambda fixed: betas ({bx['betas'][0]:.3f}, {bx['betas'][1]:.3f}, "
          f"{bx['betas'][2]:.3f}) lambda {bx['lam']:.4f} rmse {bx['rmse'] * 100:.2f} bp")
    print(f"   {stab['n_boot']} residual-bootstrap refits:")
    print(f"     lambda free : lambda 5th-95th pct [{stab['lam_p05']:.4f}, {stab['lam_p95']:.4f}], "
          f"range [{stab['lam_min']:.4f}, {stab['lam_max']:.4f}], std {stab['lam_std']:.4f}")
    print(f"     beta2 std   : free {stab['beta2_std_free']:.3f}  vs  fixed "
          f"{stab['beta2_std_fixed']:.3f}  ({stab['beta2_std_free'] / stab['beta2_std_fixed']:.1f}x)")
    print(f"     beta1 std   : free {stab['beta1_std_free']:.3f}  vs  fixed "
          f"{stab['beta1_std_fixed']:.3f}  ({stab['beta1_std_free'] / stab['beta1_std_fixed']:.1f}x)")
    print(f"     fit quality : rmse free {stab['rmse_free'] * 100:.2f} bp vs fixed "
          f"{stab['rmse_fixed'] * 100:.2f} bp -- the free lambda buys almost nothing")
    t_m, clean, y = synthetic_yields(0, 5.0)
    sv_fit = fit_svensson(t_m, y)
    print(f"   Svensson on the same curve: rmse {sv_fit['rmse'] * 100:.2f} bp, params "
          f"{tuple(round(v, 3) for v in sv_fit['params'])}")
    print("   Svensson loading-matrix condition number as lambda2 -> lambda1: "
          + ", ".join(f"ratio {r_:.2f}: {c:.1e}" for r_, c in svensson_collinearity(t_m, 0.0609)))

    # ------------------------------------------------------------- 4. short-rate bonds
    print("\n4. VASICEK / CIR / HULL-WHITE ZERO-COUPON BONDS, P(1, 5) with r(1) = 4%")
    vas = vasicek_bond(0.04, 4.0, 0.1, 0.05, 0.01)
    cir = cir_bond(0.04, 4.0, 0.3, 0.05, 0.1)
    hw = hull_white_bond(1.0, 5.0, 0.04, 0.1, 0.01, lambda T: math.exp(-0.05 * T),
                         lambda t: 0.05)
    print(f"   Vasicek(a=0.1, b=5%, sigma=1%)       {vas:.12f}")
    print(f"   CIR(kappa=0.3, theta=5%, sigma=10%)  {cir:.12f}   Feller 2 kappa theta = "
          f"{2 * 0.3 * 0.05:.3f} vs sigma^2 = {0.1 ** 2:.3f}")
    print(f"   Hull-White(flat 5%, a=0.1, sigma=1%) {hw:.12f}")

    print("   A(t,T) and B(t,T) checked SEPARATELY, not only through their product:")
    h_r = 1e-6
    models = (("Vasicek", "vasicek", vasicek_bond, dict(a=0.1, b=0.05, sigma=0.01), 0.1, 0.05),
              ("CIR    ", "cir", cir_bond, dict(kappa=0.3, theta=0.05, sigma=0.1), 0.3, 0.05))
    for label, model, price, kw, speed, level in models:
        A, B = affine_ab(model, 4.0, **kw)
        b_num = -(math.log(price(0.04 + h_r, 4.0, *kw.values()))
                  - math.log(price(0.04 - h_r, 4.0, *kw.values()))) / (2.0 * h_r)
        a_from_p = price(0.04, 4.0, *kw.values()) * math.exp(B * 0.04)
        print(f"     {label} B={B:.12f}  -dlnP/dr={b_num:.12f}  |diff| {abs(B - b_num):.1e}   "
              f"A={A:.12f}  P e^(Br)={a_from_p:.12f}  |diff| {abs(A - a_from_p):.1e}")
    print("   sigma -> 0 must collapse to the deterministic ODE integral "
          "exp(-(level*tau + (r-level)*B)) -- and one of them stops:")
    for label, model, price, kw, speed, level in models:
        det = deterministic_bond(0.04, 4.0, speed, level)
        row = []
        for sg in (1e-2, 1e-4, 1e-6, 1e-9):
            small = dict(kw)
            small["sigma"] = sg
            row.append(f"{sg:g}: {price(0.04, 4.0, *small.values()) - det:+.1e}")
        print(f"     {label} P(sigma) - deterministic {det:.12f}   " + "   ".join(row))
    print("   -> Vasicek's A is an exponential and stays exact. CIR's A is (.)^(2 kappa theta "
          "/ sigma^2): at sigma=1e-9 that")
    print("      exponent is 3.0e+16 on a base one ulp below 1, and cir_bond returns 712.06 "
          "instead of 0.838. Small-sigma CIR")
    print("      is a floating-point trap, not a modelling one -- the useful range bottoms out "
          "around sigma=1e-4 here.")

    # ------------------------------------------------------------- 5. CIR simulation
    print("\n5. CIR EULER SIMULATION: sqrt of a negative rate")
    for label, (kap, th, sg) in (("Feller holds   ", (0.5, 0.03, 0.15)),
                                 ("Feller violated", (0.5, 0.03, 0.20))):
        exact = cir_bond(0.03, 5.0, kap, th, sg)
        eu = simulate_cir(0.03, kap, th, sg, 5.0, 60, 20_000, 0, "euler")
        ft = simulate_cir(0.03, kap, th, sg, 5.0, 60, 20_000, 0, "full_truncation")
        print(f"   {label} kappa={kap} theta={th} sigma={sg}: 2 kappa theta={2 * kap * th:.3f} "
              f"sigma^2={sg * sg:.3f}; exact P(0,5)={exact:.6f}")
        print(f"     plain Euler, monthly steps, 20,000 paths: {eu['negative_steps']:,} negative "
              f"steps, {eu['nan_paths']:,} NaN paths ({eu['nan_paths'] / eu['paths']:.1%}); "
              f"MC on the survivors {eu['mc_price']:.6f} (bias {eu['mc_price'] - exact:+.6f}, "
              f"se {eu['mc_se']:.6f})")
        print(f"     full truncation                          : {ft['negative_steps']:,} negative "
              f"steps, {ft['nan_paths']:,} NaN paths; MC {ft['mc_price']:.6f} "
              f"(bias {ft['mc_price'] - exact:+.6f}, se {ft['mc_se']:.6f})")

    # ------------------------------------------------------------- 6. QuantLib
    print("\n6. CROSS-CHECK AGAINST QUANTLIB")
    ns_params = (0.05, -0.02, 0.01, 0.5)
    sv_params = (0.05, -0.02, 0.01, 0.02, 0.5, 0.1)
    live = quantlib_cross_checks(par, dfs, d0, d1, ns_params, sv_params)
    if live is None:
        print("   QuantLib not installed - the closed forms above stand on the reference "
              "implementation")
    else:
        print(f"   QuantLib {live['version']}")
        print(f"   Vasicek.discountBond(1,5,0.04)         {live['vasicek']:.12f}  mine {vas:.12f}  "
              f"|diff| {abs(live['vasicek'] - vas):.1e}")
        print(f"   CoxIngersollRoss.discountBond(1,5,.04) {live['cir']:.12f}  mine {cir:.12f}  "
              f"|diff| {abs(live['cir'] - cir):.1e}")
        print(f"   HullWhite.discountBond(1,5,0.04)       {live['hull_white']:.12f}  mine {hw:.12f}  "
              f"|diff| {abs(live['hull_white'] - hw):.1e}")
        print(f"   THE MIDDLE TWO ARGUMENTS ARE IN THE OPPOSITE ORDER BETWEEN THE TWO CLASSES:")
        print(f"     Vasicek(r0, SPEED=0.1, level=0.05, sigma)  {live['vasicek']:.12f}   "
              f"swapped -> {live['vasicek_swapped']:.12f} "
              f"({live['vasicek_swapped'] - live['vasicek']:+.6f}, no error)")
        print(f"     CoxIngersollRoss(r0, LEVEL=0.05, speed=0.3, sigma)  {live['cir']:.12f}   "
              f"swapped -> {live['cir_swapped']:.12f} "
              f"({live['cir_swapped'] - live['cir']:+.6f}, no error)")
        mine_ns = math.exp(-nelson_siegel(np.array([5.0]), *ns_params)[0] * 5.0)
        mine_sv = math.exp(-svensson(np.array([5.0]), *sv_params)[0] * 5.0)
        print(f"   NelsonSiegelFitting from parameters, P(5) {live['ns_disc_5y']:.12f}  mine "
              f"{mine_ns:.12f}  |diff| {abs(live['ns_disc_5y'] - mine_ns):.1e}")
        print(f"   SvenssonFitting from parameters,     P(5) {live['sv_disc_5y']:.12f}  mine "
              f"{mine_sv:.12f}  |diff| {abs(live['sv_disc_5y'] - mine_sv):.1e}")
        print(f"   InterestRate 5% annual -> continuous   {live['five_pct_annual_as_continuous']:.10f}"
              f"  mine {math.log(1.05):.10f}")
        print(f"   yearFraction {d0}->{d1}: ACT/365 {live['yf_act365']:.6f} (mine "
              f"{year_fraction(d0, d1, 'ACT/365'):.6f}), ACT/360 {live['yf_act360']:.6f} (mine "
              f"{year_fraction(d0, d1, 'ACT/360'):.6f}), 30/360 {live['yf_30360']:.6f} (mine "
              f"{year_fraction(d0, d1, '30/360'):.6f})")
        print(f"   DiscountingBondEngine on the bootstrapped DFs, par bonds clean prices: "
              + ", ".join(f"{p:.8f}" for p in live["par_bond_prices"]))

    print("\n" + "=" * W)
    print("RULE: a zero rate is (day count, compounding, instrument) -- carry all three; fix")
    print("      Nelson-Siegel lambda (0.0609/month = 0.7308/year) unless you can show it is")
    print("      identified; never take sqrt(r) of an Euler CIR step without truncation.")
    print("=" * W)
