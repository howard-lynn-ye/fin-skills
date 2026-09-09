"""Option pricing models -- closed form, tree, Heston, SABR, Monte Carlo -- and where each breaks.

WHY this exists: every model here has a textbook formula, and every formula has a place where
the textbook implementation returns a plausible wrong number without raising:

  * Heston (1993) wrote the characteristic function with g = (b - rho*sigma*i*phi + d) /
    (b - rho*sigma*i*phi - d) and exp(+d*tau). Under the principal branch of the complex log
    that expression is DISCONTINUOUS in phi once the maturity is long enough, so the
    inversion integral silently integrates across a jump. Albrecher, Mayer, Schoutens and
    Tistaert (2007, "The little Heston trap") showed the algebraically identical form with
    1/g and exp(-d*tau) is continuous. QuantLib's AnalyticHestonEngine calls the safe form
    `Gatheral` and the original `BranchCorrection` (it repairs the original with a rotation
    count). This file implements BOTH and measures the maturity at which the original breaks.

  * A Cox-Ross-Rubinstein tree converges to Black-Scholes by OSCILLATING: even step counts
    sit on one side of the limit, odd counts on the other, so "use more steps" buys much less
    than it seems to. Averaging n and n+1 steps, or Richardson-extrapolating 2*C(2n) - C(n),
    removes most of the oscillation for a European payoff. Measured below.

  * A Monte Carlo standard error falls with the number of PATHS (as 1/sqrt(N)) and does not
    fall with the number of TIME STEPS. Steps remove the DISCRETISATION BIAS of an Euler
    scheme, which the standard error does not see. Confusing the two is how a "converged"
    Monte Carlo is wrong by a quarter of a vol point with a tiny reported error bar.

  * The SABR (Hagan et al. 2002) formula has a 0/0 at the money: z / x(z) with z -> 0. The
    literal formula returns NaN exactly at K = F; implementations switch to a series there,
    and the switch threshold is a choice that differs between libraries.

Every number the SKILL.md quotes is printed by the demo below. QuantLib is optional: when it
is importable every model is cross-checked against it and the comparison is printed rather
than assumed.

Usage:
    from option_models import bsm_price, crr_price, crr_smoothed, heston_price, \
        sabr_implied_vol, mc_european
    heston_price(100, 100, 10.0, 0.03, 0.0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3,
                 rho=-0.7)                                   # Albrecher form, the default
    heston_price(..., formulation="heston1993")              # the original: breaks at long T
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

FORMULATIONS = ("albrecher", "heston1993")


# ------------------------------------------------------------------ Black-Scholes-Merton
def _flag(flag: str) -> str:
    f = flag.strip().lower()[:1]
    if f not in ("c", "p"):
        raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
    return f


def bsm_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
              flag: str = "c") -> float:
    """Black-Scholes-Merton (1973) European price with a continuous dividend yield q.

    The dividend yield enters as exp(-q*T) on the spot and in d1. `vollib.black_scholes`
    has no q at all (see lib-vollib); this is the Merton form, equal to `black_scholes_merton`.
    """
    f = _flag(flag)
    if T <= 0.0:
        intrinsic = S - K if f == "c" else K - S
        return max(intrinsic, 0.0)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    if f == "c":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


# ------------------------------------------------------------------ CRR binomial tree
def crr_price(S: float, K: float, T: float, r: float, q: float, sigma: float,
              steps: int = 200, flag: str = "c", american: bool = False,
              prob: str = "textbook") -> float:
    """Cox-Ross-Rubinstein (1979) tree: u = exp(sigma*sqrt(dt)), d = 1/u.
    `american=True` takes max(continuation, exercise) at every node. Vectorised backward
    induction, O(steps^2).

    Two probabilities are in circulation for the SAME u and d, and they differ at O(dt):
      'textbook'  p = (exp((r-q)*dt) - d) / (u - d)                     -- CRR (1979)
      'quantlib'  p = 1/2 + (r - q - sigma^2/2)*dt / (2*sigma*sqrt(dt)) -- what QuantLib's
                  BinomialVanillaEngine(process, "crr", n) actually computes. Reproduced to
                  3e-13 in the demo; the textbook probability misses QuantLib by 1.2e-4 at
                  n=100. Both converge to Black-Scholes, so neither is wrong -- but if you
                  are benchmarking a tree against QuantLib, match the probability first.

    fin_skills.core.option_lifecycle.crr is the same tree used for assignment accounting;
    this copy exists so the skill script runs standalone and so `crr_smoothed` can wrap it.
    """
    f = _flag(flag)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if prob not in ("textbook", "quantlib"):
        raise ValueError("prob must be 'textbook' or 'quantlib'")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    if prob == "textbook":
        p = (math.exp((r - q) * dt) - d) / (u - d)
    else:
        p = 0.5 + 0.5 * (r - q - 0.5 * sigma * sigma) * dt / (sigma * math.sqrt(dt))
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0,1): dt too large for "
                         f"sigma={sigma}, r-q={r - q}; use more steps")
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d ** j
    V = np.maximum(ST - K, 0.0) if f == "c" else np.maximum(K - ST, 0.0)
    for i in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if american:
            Si = S * u ** (i - j[: i + 1]) * d ** j[: i + 1]
            ex = np.maximum(Si - K, 0.0) if f == "c" else np.maximum(K - Si, 0.0)
            V = np.maximum(V, ex)
    return float(V[0])


def crr_smoothed(S: float, K: float, T: float, r: float, q: float, sigma: float,
                 steps: int = 200, flag: str = "c", american: bool = False,
                 method: str = "average") -> float:
    """Two fixes for the odd/even oscillation of the CRR tree.

    'average'    : 0.5 * (C(n) + C(n+1)) -- the two parities straddle the limit.
    'richardson' : 2 * C(2n) - C(n)      -- cancels the O(1/n) error term.
    Both are measured in the demo against the Black-Scholes limit.
    """
    if method == "average":
        return 0.5 * (crr_price(S, K, T, r, q, sigma, steps, flag, american)
                      + crr_price(S, K, T, r, q, sigma, steps + 1, flag, american))
    if method == "richardson":
        return (2.0 * crr_price(S, K, T, r, q, sigma, 2 * steps, flag, american)
                - crr_price(S, K, T, r, q, sigma, steps, flag, american))
    raise ValueError("method must be 'average' or 'richardson'")


def crr_convergence(S: float, K: float, T: float, r: float, q: float, sigma: float,
                    flag: str = "p", steps: Sequence[int] = (50, 51, 100, 101, 200, 201, 400,
                                                             401, 800, 801)
                    ) -> list[tuple[int, float, float]]:
    """(n, CRR European price, error vs Black-Scholes) for each step count."""
    ref = bsm_price(S, K, T, r, q, sigma, flag)
    return [(n, p, p - ref) for n in steps
            for p in (crr_price(S, K, T, r, q, sigma, n, flag, american=False),)]


# ------------------------------------------------------------------ Heston
def _gauss_legendre_grid(phi_max: float, n_panels: int, order: int = 8
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes/weights on [0, phi_max]. Fixed and deterministic, so a
    discontinuous integrand produces a wrong number rather than an adaptive-quadrature
    warning: that wrong number IS the measurement this file exists to make."""
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0.0, phi_max, n_panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def heston_log_cf_terms(phi: np.ndarray, tau: float, r: float, q: float, kappa: float,
                        theta: float, sigma: float, rho: float, j: int,
                        formulation: str = "albrecher") -> tuple[np.ndarray, np.ndarray]:
    """C_j(phi, tau) and D_j(phi, tau) of Heston (1993) eq. (17)-(18) with lambda = 0, so that
    f_j = exp(C_j + D_j * v0 + i*phi*ln S). j=1 uses u=+1/2, b=kappa-rho*sigma; j=2 uses
    u=-1/2, b=kappa.

    'heston1993': g = (b - rho*sigma*i*phi + d)/(b - rho*sigma*i*phi - d), exp(+d*tau).
    'albrecher' : g = (b - rho*sigma*i*phi - d)/(b - rho*sigma*i*phi + d) = 1/g_1993,
                  exp(-d*tau). Identical algebra, different branch behaviour of the log.
    QuantLib calls these BranchCorrection (plus a rotation-count repair) and Gatheral.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"formulation must be one of {FORMULATIONS}, got {formulation!r}")
    if j not in (1, 2):
        raise ValueError("j must be 1 or 2")
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * sigma if j == 1 else kappa
    a = kappa * theta
    iphi = 1j * phi
    xi = b - rho * sigma * iphi
    # The 1993 form overflows exp(+d*tau) at long maturities and returns NaN: that is part
    # of the measurement, so the numpy warnings are silenced rather than the outcome hidden.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        d = np.sqrt(xi * xi - sigma * sigma * (2.0 * u * iphi - phi * phi))
        if formulation == "heston1993":
            g = (xi + d) / (xi - d)
            edt = np.exp(d * tau)
            C = (r - q) * iphi * tau + (a / sigma ** 2) * (
                (xi + d) * tau - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
            D = ((xi + d) / sigma ** 2) * (1.0 - edt) / (1.0 - g * edt)
        else:
            g = (xi - d) / (xi + d)
            edt = np.exp(-d * tau)
            C = (r - q) * iphi * tau + (a / sigma ** 2) * (
                (xi - d) * tau - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
            D = ((xi - d) / sigma ** 2) * (1.0 - edt) / (1.0 - g * edt)
    return C, D


def heston_price(S: float, K: float, T: float, r: float, q: float, v0: float, kappa: float,
                 theta: float, sigma: float, rho: float, flag: str = "c",
                 formulation: str = "albrecher", phi_max: float | None = None,
                 n_panels: int | None = None) -> float:
    """Heston (1993) semi-analytic European price:
        call = S e^{-qT} P1 - K e^{-rT} P2,
        P_j  = 1/2 + (1/pi) * int_0^inf Re[ e^{-i phi ln K} f_j(phi) / (i phi) ] dphi,
    integrated on a fixed composite Gauss-Legendre grid (no adaptive quadrature, so the
    'heston1993' branch-cut discontinuity shows up as a wrong price, not a warning).
    Put via put-call parity.
    """
    f = _flag(flag)
    if T <= 0.0:
        return max(S - K if f == "c" else K - S, 0.0)
    # The integrand decays roughly like exp(-c*phi*sqrt(T)); scale the range with 1/sqrt(T)
    # so very short maturities are still resolved (measured against QuantLib in the demo).
    if phi_max is None:
        phi_max = min(4000.0, 200.0 / math.sqrt(min(T, 1.0)))
    if n_panels is None:
        n_panels = int(min(8000, max(400, phi_max * 2)))
    phi, w = _gauss_legendre_grid(phi_max, n_panels)
    lnS, lnK = math.log(S), math.log(K)
    P = []
    for j in (1, 2):
        C, D = heston_log_cf_terms(phi, T, r, q, kappa, theta, sigma, rho, j, formulation)
        with np.errstate(over="ignore", invalid="ignore"):
            cf = np.exp(C + D * v0 + 1j * phi * lnS)
            integrand = np.real(np.exp(-1j * phi * lnK) * cf / (1j * phi))
        P.append(0.5 + float(np.dot(w, integrand)) / math.pi)
    call = S * math.exp(-q * T) * P[0] - K * math.exp(-r * T) * P[1]
    if f == "c":
        return call
    return call - S * math.exp(-q * T) + K * math.exp(-r * T)


def heston_trap_scan(S: float, K: float, r: float, q: float, v0: float, kappa: float,
                     theta: float, sigma: float, rho: float,
                     maturities: Sequence[float] = (0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30)
                     ) -> list[dict[str, float]]:
    """Price the same call with both formulations at each maturity. The Albrecher form is
    the reference; the 1993 form's error is the measurement."""
    rows = []
    for T in maturities:
        alb = heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho, "c", "albrecher")
        old = heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho, "c", "heston1993")
        rows.append({"T": float(T), "albrecher": alb, "heston1993": old, "error": old - alb})
    return rows


def heston_kappa_scan(S: float, K: float, T: float, r: float, q: float, v0: float,
                      theta: float, sigma: float, rho: float,
                      kappas: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 3.0, 6.0, 12.0)
                      ) -> list[dict[str, float]]:
    """The second axis of the trap: at a FIXED maturity, raise kappa*theta (the coefficient
    a/sigma^2 that multiplies the offending log) and the 1993 form starts crossing branches.
    Same comparison as heston_trap_scan, with T held and kappa varied."""
    rows = []
    for kappa in kappas:
        alb = heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho, "c", "albrecher")
        old = heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho, "c", "heston1993")
        rows.append({"kappa": float(kappa), "kappa_theta": kappa * theta,
                     "a_over_sigma2": kappa * theta / sigma ** 2,
                     "albrecher": alb, "heston1993": old, "error": old - alb,
                     "jumps": float(heston_cf_discontinuities(T, r, q, kappa, theta, sigma,
                                                              rho, "heston1993"))})
    return rows


def heston_cf_discontinuities(T: float, r: float, q: float, kappa: float, theta: float,
                              sigma: float, rho: float, formulation: str,
                              phi_max: float = 100.0, n: int = 20001) -> int:
    """Count jumps larger than pi in the phase of the log term between adjacent phi points --
    the direct fingerprint of a branch-cut crossing. Zero means the log is continuous."""
    phi = np.linspace(1e-6, phi_max, n)
    total = 0
    for j in (1, 2):
        C, _ = heston_log_cf_terms(phi, T, r, q, kappa, theta, sigma, rho, j, formulation)
        total += int(np.sum(np.abs(np.diff(np.imag(C))) > math.pi))
    return total


def heston_first_nonfinite_phi(T: float, r: float, q: float, kappa: float, theta: float,
                               sigma: float, rho: float, formulation: str = "heston1993",
                               phi_max: float = 200.0, n: int = 2001) -> float | None:
    """Smallest phi on a uniform grid at which C_1 stops being finite -- the 1993 form's
    exp(+d*tau) overflows to inf there and the integrand becomes NaN. None if it never does."""
    phi = np.linspace(1e-6, phi_max, n)
    C, _ = heston_log_cf_terms(phi, T, r, q, kappa, theta, sigma, rho, 1, formulation)
    bad = np.flatnonzero(~np.isfinite(C))
    return float(phi[bad[0]]) if bad.size else None


# ------------------------------------------------------------------ SABR
def sabr_implied_vol(K: float, F: float, T: float, alpha: float, beta: float, nu: float,
                     rho: float) -> float:
    """Hagan, Kumar, Lesniewski, Woodward (2002) eq. (2.17a)-(2.17c) lognormal implied vol.

    At the money the literal formula is z/x(z) with z -> 0 (0/0). Below |z| ~ sqrt(eps) it
    uses the series 1 - rho*z/2 + (2 - 3 rho^2) z^2 / 12, the same branch QuantLib's
    unsafeSabrVolatility takes, so the two agree at K = F as well as away from it.
    """
    if K <= 0.0 or F <= 0.0:
        raise ValueError("SABR lognormal vol needs positive strike and forward")
    if alpha <= 0.0 or not 0.0 <= beta <= 1.0 or nu < 0.0 or not -1.0 < rho < 1.0:
        raise ValueError("SABR parameters out of range: alpha>0, 0<=beta<=1, nu>=0, |rho|<1")
    one_b = 1.0 - beta
    logM = math.log(F / K)
    fk_pow = (F * K) ** (0.5 * one_b)
    denom = fk_pow * (1.0 + one_b ** 2 / 24.0 * logM ** 2 + one_b ** 4 / 1920.0 * logM ** 4)
    z = (nu / alpha) * fk_pow * logM
    if abs(z) < 1e-7:
        multiplier = 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0
    else:
        x = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        multiplier = z / x
    term = (one_b ** 2 / 24.0 * alpha ** 2 / (F * K) ** one_b
            + 0.25 * rho * beta * nu * alpha / fk_pow
            + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
    return alpha / denom * multiplier * (1.0 + term * T)


def sabr_atm_vol(F: float, T: float, alpha: float, beta: float, nu: float, rho: float) -> float:
    """Hagan et al. (2002) eq. (2.18), the K = F special case, written out separately so the
    general formula's ATM branch can be checked against it."""
    one_b = 1.0 - beta
    term = (one_b ** 2 / 24.0 * alpha ** 2 / F ** (2.0 * one_b)
            + 0.25 * rho * beta * nu * alpha / F ** one_b
            + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
    return alpha / F ** one_b * (1.0 + term * T)


# ------------------------------------------------------------------ Monte Carlo
def mc_european(S: float, K: float, T: float, r: float, q: float, sigma: float,
                flag: str = "c", n_paths: int = 100_000, n_steps: int = 1,
                antithetic: bool = True, scheme: str = "log", seed: int = 0
                ) -> tuple[float, float]:
    """European price and its standard error by Monte Carlo.

    scheme='log'  : exact lognormal increments -- no discretisation bias at ANY step count.
    scheme='euler': S_{t+dt} = S_t (1 + (r-q) dt + sigma sqrt(dt) Z) -- biased, bias -> 0
                    as n_steps grows. This is the scheme most first implementations use.
    antithetic    : each Z is paired with -Z; the estimator is the mean of pair averages
                    and the standard error is computed over the PAIRS, which is the only
                    honest way to report it (treating 2N correlated draws as independent
                    understates the error).
    Returns (price, standard_error). The seed makes every run reproducible.
    """
    f = _flag(flag)
    if scheme not in ("log", "euler"):
        raise ValueError("scheme must be 'log' or 'euler'")
    rng = np.random.default_rng(seed)
    n = n_paths // 2 if antithetic else n_paths
    dt = T / n_steps
    disc = math.exp(-r * T)

    def terminal(z: np.ndarray) -> np.ndarray:
        if scheme == "log":
            drift = (r - q - 0.5 * sigma * sigma) * dt
            return S * np.exp(np.sum(drift + sigma * math.sqrt(dt) * z, axis=1))
        s = np.full(z.shape[0], S)
        for k in range(n_steps):
            s = s * (1.0 + (r - q) * dt + sigma * math.sqrt(dt) * z[:, k])
        return s

    z = rng.standard_normal((n, n_steps))
    st = terminal(z)
    pay = np.maximum(st - K, 0.0) if f == "c" else np.maximum(K - st, 0.0)
    if antithetic:
        st2 = terminal(-z)
        pay2 = np.maximum(st2 - K, 0.0) if f == "c" else np.maximum(K - st2, 0.0)
        pay = 0.5 * (pay + pay2)
    pay = disc * pay
    return float(pay.mean()), float(pay.std(ddof=1) / math.sqrt(len(pay)))


# ------------------------------------------------------------------ optional cross-checks
def quantlib_cross_checks(S: float, K: float, T: float, r: float, q: float, sigma: float,
                          heston: dict[str, float], sabr: dict[str, float],
                          r_heston: float = 0.03, q_heston: float = 0.0,
                          maturities: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 30.0)
                          ) -> dict[str, object] | None:
    """Price the same instruments in QuantLib 1.43 and return its numbers, or None.
    r/q/sigma feed the Black-Scholes process (tree, MC); r_heston/q_heston the Heston one."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    today = ql.Date(8, 9, 2026)
    ql.Settings.instance().evaluationDate = today          # BEFORE any curve (lib-quantlib)
    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()

    def flat(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(today, rate, dc))

    out: dict[str, object] = {"version": ql.__version__}
    spot = ql.QuoteHandle(ql.SimpleQuote(S))
    hp = ql.HestonProcess(flat(r_heston), flat(q_heston), spot, heston["v0"], heston["kappa"],
                          heston["theta"], heston["sigma"], heston["rho"])
    model = ql.HestonModel(hp)
    gl = ql.AnalyticHestonEngine_Integration.gaussLaguerre(192)
    engines = {
        "default_ctor": ql.AnalyticHestonEngine(model),
        "Gatheral": ql.AnalyticHestonEngine(model, ql.AnalyticHestonEngine.Gatheral, gl),
        "BranchCorrection": ql.AnalyticHestonEngine(
            model, ql.AnalyticHestonEngine.BranchCorrection, gl),
    }
    rows = []
    for Tm in maturities:
        expiry = today + ql.Period(int(round(Tm * 365)), ql.Days)
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                               ql.EuropeanExercise(expiry))
        row = {"T": dc.yearFraction(today, expiry)}
        for name, eng in engines.items():
            opt.setPricingEngine(eng)
            row[name] = opt.NPV()
        rows.append(row)
    out["heston"] = rows

    out["sabr"] = {Ks: ql.sabrVolatility(Ks, sabr["F"], sabr["T"], sabr["alpha"],
                                          sabr["beta"], sabr["nu"], sabr["rho"])
                   for Ks in sabr["strikes"]}

    bs = ql.BlackScholesMertonProcess(
        spot, flat(q), flat(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, cal, sigma, dc)))
    expiry = today + ql.Period(int(round(T * 365)), ql.Days)
    put = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K), ql.EuropeanExercise(expiry))
    am = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Put, K),
                          ql.AmericanExercise(today, expiry))
    crr_rows = {}
    for n in (100, 101, 800, 801):
        put.setPricingEngine(ql.BinomialVanillaEngine(bs, "crr", n))
        crr_rows[n] = put.NPV()
    out["crr_european_put"] = crr_rows
    am.setPricingEngine(ql.QdFpAmericanEngine(bs))
    out["american_put_qdfp"] = am.NPV()
    put.setPricingEngine(ql.AnalyticEuropeanEngine(bs))
    out["bs_put"] = put.NPV()

    call = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                            ql.EuropeanExercise(expiry))
    mc = {}
    for anti in (False, True):
        for steps in (1, 52):
            call.setPricingEngine(ql.MCEuropeanEngine(
                bs, "pseudorandom", timeSteps=steps, requiredSamples=100_000,
                antitheticVariate=anti, seed=42))
            mc[(anti, steps)] = (call.NPV(), call.errorEstimate())
    out["mc"] = mc
    out["mc_samples"] = 100_000
    return out


def payoff_std(S: float, K: float, T: float, r: float, q: float, sigma: float,
               flag: str = "c", n_paths: int = 400_000, antithetic: bool = False,
               seed: int = 7) -> float:
    """Standard deviation of one discounted payoff (or of one antithetic PAIR average), so a
    library's reported error estimate can be turned back into the sample count it used:
    N = (std / std_err)^2."""
    price, se = mc_european(S, K, T, r, q, sigma, flag, n_paths, 1, antithetic, "log", seed)
    n = n_paths // 2 if antithetic else n_paths
    return se * math.sqrt(n)


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
    W = 96
    print("=" * W)
    print("OPTION PRICING MODELS -- every number below is produced here, not quoted")
    print(f"S={S} K={K} T={T} r={r} q={q} sigma={sigma}")
    print("=" * W)

    # ------------------------------------------------------------- 1. BSM + parity
    call = bsm_price(S, K, T, r, q, sigma, "c")
    put = bsm_price(S, K, T, r, q, sigma, "p")
    parity = call - put - (S * math.exp(-q * T) - K * math.exp(-r * T))
    print(f"\n1. Black-Scholes-Merton  call {call:.10f}  put {put:.10f}  "
          f"put-call parity residual {parity:+.2e}")
    assert abs(parity) < 1e-12

    # ------------------------------------------------------------- 2. CRR oscillation
    print("\n2. CRR CONVERGENCE OSCILLATES WITH THE PARITY OF THE STEP COUNT (European put)")
    print(f"   Black-Scholes limit {put:.6f}")
    print(f"   {'steps':>6} {'CRR':>12} {'error':>12}")
    conv = crr_convergence(S, K, T, r, q, sigma, "p")
    for n, p, e in conv:
        print(f"   {n:>6} {p:>12.6f} {e:>+12.6f}")
    e100 = dict((n, e) for n, _, e in conv)
    avg = crr_smoothed(S, K, T, r, q, sigma, 100, "p", method="average")
    rich = crr_smoothed(S, K, T, r, q, sigma, 100, "p", method="richardson")
    rich_odd = crr_smoothed(S, K, T, r, q, sigma, 101, "p", method="richardson")
    print(f"   n=100 error {e100[100]:+.6f}, n=101 error {e100[101]:+.6f}: opposite signs.")
    print(f"   n=800 error {e100[800]:+.6f} is only {abs(e100[100] / e100[800]):.1f}x smaller "
          f"than n=100 -- 64x the work.")
    print(f"   average(100,101)          = {avg:.6f}  error {avg - put:+.6f}  "
          f"({abs(e100[100] / (avg - put)):.0f}x better than n=100 alone)")
    print(f"   richardson 2C(200)-C(100) = {rich:.6f}  error {rich - put:+.6f}  "
          f"({abs(e100[100] / (rich - put)):.0f}x better than n=100 alone)")
    print(f"   richardson 2C(202)-C(101) = {rich_odd:.6f}  error {rich_odd - put:+.6f}  "
          f"({abs(e100[101] / (rich_odd - put)):.2f}x better than n=101 alone -- i.e. WORSE: "
          f"Richardson needs both legs on the SAME parity)")
    am_eu = crr_price(S, K, T, r, q, sigma, 800, "p", american=False)
    am = crr_price(S, K, T, r, q, sigma, 800, "p", american=True)
    print(f"   American put, 800 steps: {am:.6f} vs European {am_eu:.6f} "
          f"(early-exercise premium {am - am_eu:.6f})")

    # ------------------------------------------------------------- 3. Heston trap
    hest = dict(v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
    print("\n3. THE LITTLE HESTON TRAP -- the 1993 formulation vs the Albrecher (2007) form")
    print(f"   Heston params {hest}, r={0.03}, q={0.0}, ATM call, S=K=100")
    print(f"   {'T':>5} {'albrecher':>14} {'heston1993':>14} {'error':>14}  log-branch jumps")
    scan = heston_trap_scan(S, K, 0.03, 0.0, **hest)
    first_break = None
    for row in scan:
        jumps = heston_cf_discontinuities(row["T"], 0.03, 0.0, hest["kappa"], hest["theta"],
                                          hest["sigma"], hest["rho"], "heston1993")
        wrong = not (abs(row["error"]) < 1e-6)          # NaN counts as wrong
        if first_break is None and wrong:
            first_break = row["T"]
        print(f"   {row['T']:>5.1f} {row['albrecher']:>14.8f} {row['heston1993']:>14.8f} "
              f"{row['error']:>+14.8f}  {jumps:>4}{'  <-- WRONG' if wrong else ''}")
    alb_jumps = max(heston_cf_discontinuities(Tm, 0.03, 0.0, hest["kappa"], hest["theta"],
                                              hest["sigma"], hest["rho"], "albrecher")
                    for Tm in (1.0, 10.0, 30.0))
    print(f"   Albrecher form: {alb_jumps} branch jumps at T=1, 10, 30. First maturity where "
          f"the 1993 form is wrong by >1e-6: T={first_break}")
    finite = [x for x in scan if math.isfinite(x["error"])]
    worst = max(finite, key=lambda x: abs(x["error"]))
    n_nan = sum(1 for x in scan if not math.isfinite(x["heston1993"]))
    print(f"   worst finite 1993-form error in the scan: {worst['error']:+.4f} at "
          f"T={worst['T']:.0f} ({abs(worst['error']) / worst['albrecher']:.1%} of the price); "
          f"NaN (exp(d*tau) overflowed) at {n_nan} of the {len(scan)} maturities")
    ovf = heston_first_nonfinite_phi(20.0, 0.03, 0.0, hest["kappa"], hest["theta"],
                                     hest["sigma"], hest["rho"])
    ovf_alb = heston_first_nonfinite_phi(20.0, 0.03, 0.0, hest["kappa"], hest["theta"],
                                         hest["sigma"], hest["rho"], "albrecher")
    print(f"   the NaN, located: at T=20 the 1993 C_1 stops being finite at phi={ovf:.1f} "
          f"(exp(+d*tau) overflows); the Albrecher form's exp(-d*tau) never does "
          f"({'finite everywhere' if ovf_alb is None else f'phi={ovf_alb:.1f}'})")

    # -------------------------------------------------- 3b. the second axis: kappa*theta
    print("\n3b. THE SAME TRAP AT FIXED T=2, DRIVEN BY kappa*theta INSTEAD OF MATURITY")
    print(f"   {'kappa':>6} {'kappa*theta':>12} {'a/sigma^2':>10} {'albrecher':>13} "
          f"{'heston1993':>13} {'error':>13}  jumps")
    kscan = heston_kappa_scan(S, K, 2.0, 0.03, 0.0, hest["v0"], hest["theta"], hest["sigma"],
                              hest["rho"])
    for row in kscan:
        flag_k = "  <-- WRONG" if abs(row["error"]) > 1e-6 else ""
        print(f"   {row['kappa']:>6.2f} {row['kappa_theta']:>12.3f} "
              f"{row['a_over_sigma2']:>10.3f} {row['albrecher']:>13.8f} "
              f"{row['heston1993']:>13.8f} {row['error']:>+13.8f}  {int(row['jumps']):>4}"
              f"{flag_k}")
    ok_k = [r["kappa"] for r in kscan if abs(r["error"]) <= 1e-6]
    bad_k = [r["kappa"] for r in kscan if abs(r["error"]) > 1e-6]
    worst_k = max(kscan, key=lambda x: abs(x["error"]))
    print(f"   T is held at 2y throughout: the 1993 form is exact up to kappa={max(ok_k):.2f} "
          f"and wrong from kappa={min(bad_k):.2f} on. Worst error {worst_k['error']:+.4f} "
          f"({abs(worst_k['error']) / worst_k['albrecher']:.1%}) at kappa={worst_k['kappa']:.2f}.")
    print("   Maturity is not the only axis -- a fast, high-long-run-variance calibration "
          "breaks the 1993 form at ordinary maturities.")

    # ------------------------------------------------------------- 4. SABR
    sabr = dict(F=100.0, T=1.0, alpha=0.25, beta=0.6, nu=0.4, rho=-0.25,
                strikes=(70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0))
    print("\n4. SABR (Hagan 2002, eq. 2.17a-c) implied vols, F=100, T=1, "
          f"alpha={sabr['alpha']} beta={sabr['beta']} nu={sabr['nu']} rho={sabr['rho']}")
    my_sabr = {Ks: sabr_implied_vol(Ks, sabr["F"], sabr["T"], sabr["alpha"], sabr["beta"],
                                    sabr["nu"], sabr["rho"]) for Ks in sabr["strikes"]}
    atm_218 = sabr_atm_vol(sabr["F"], sabr["T"], sabr["alpha"], sabr["beta"], sabr["nu"],
                           sabr["rho"])
    print("   " + "  ".join(f"K={Ks:.0f}: {v:.6f}" for Ks, v in my_sabr.items()))
    print(f"   ATM from the general formula {my_sabr[100.0]:.12f}; from eq. (2.18) "
          f"{atm_218:.12f}; |diff| {abs(my_sabr[100.0] - atm_218):.1e}")
    print(f"   alpha=0.25 but the ATM vol is {my_sabr[100.0]:.4f}: alpha is NOT the ATM vol "
          f"for beta<1 (derivatives-pricing section 5)")

    # ------------------------------------------------------------- 5. Monte Carlo
    print("\n5. MONTE CARLO: paths shrink the standard error, steps shrink the Euler bias")
    print(f"   exact call {call:.6f}")
    print(f"   {'block':>6} {'paths':>8} {'steps':>6} {'scheme':>6} {'antithetic':>10} "
          f"{'price':>10} {'std err':>9} {'price-exact':>12}")
    mc_rows = []
    for n_paths in (10_000, 40_000, 160_000):
        p, se = mc_european(S, K, T, r, q, sigma, "c", n_paths, 1, False, "log", seed=1)
        mc_rows.append(("paths", n_paths, 1, "log", False, p, se))
    for steps in (1, 12, 52):
        p, se = mc_european(S, K, T, r, q, sigma, "c", 40_000, steps, False, "log", seed=1)
        mc_rows.append(("steps", 40_000, steps, "log", False, p, se))
    p, se = mc_european(S, K, T, r, q, sigma, "c", 40_000, 1, True, "log", seed=1)
    mc_rows.append(("anti", 40_000, 1, "log", True, p, se))
    for steps in (1, 4, 16, 64):
        p, se = mc_european(S, K, T, r, q, sigma, "c", 400_000, steps, True, "euler", seed=2)
        mc_rows.append(("euler", 400_000, steps, "euler", True, p, se))
    for block, n_paths, steps, scheme, anti, p, se in mc_rows:
        print(f"   {block:>6} {n_paths:>8} {steps:>6} {scheme:>6} {str(anti):>10} {p:>10.4f} "
              f"{se:>9.4f} {p - call:>+12.4f}")
    se_paths = [row[6] for row in mc_rows if row[0] == "paths"]
    se_steps = [row[6] for row in mc_rows if row[0] == "steps"]
    se_anti = [row[6] for row in mc_rows if row[0] == "anti"][0]
    euler = [(row[2], row[5] - call, row[6]) for row in mc_rows if row[0] == "euler"]
    print(f"   4x the paths: std err {se_paths[0]:.4f} -> {se_paths[1]:.4f} -> {se_paths[2]:.4f} "
          f"(ratios {se_paths[0] / se_paths[1]:.2f}, {se_paths[1] / se_paths[2]:.2f}; "
          f"1/sqrt(N) predicts 2.00)")
    print(f"   52x the steps at fixed paths: std err {se_steps[0]:.4f} -> {se_steps[2]:.4f} "
          f"(ratio {se_steps[0] / se_steps[2]:.2f}: steps do NOT reduce the error bar)")
    print(f"   antithetic at the same 40,000 paths: std err {se_steps[0]:.4f} -> {se_anti:.4f} "
          f"({se_steps[0] / se_anti:.2f}x smaller)")
    print(f"   Euler bias with 400,000 antithetic paths: "
          + ", ".join(f"{s} step{'s' if s > 1 else ''}: {b:+.4f} (se {e:.4f})"
                      for s, b, e in euler))
    print(f"   -> with 1 Euler step the bias is {abs(euler[0][1]) / euler[0][2]:.0f} standard "
          f"errors: a 'converged' error bar around a wrong number.")

    # ------------------------------------------------------------- 6. QuantLib
    print("\n6. CROSS-CHECK AGAINST QUANTLIB")
    live = quantlib_cross_checks(S, K, T, r, q, sigma, hest, sabr, r_heston=0.03, q_heston=0.0)
    if live is None:
        print("   QuantLib not installed -- the reference implementations above stand alone;")
        print("   install QuantLib to see every model verified against AnalyticHestonEngine,")
        print("   sabrVolatility, BinomialVanillaEngine, QdFpAmericanEngine, MCEuropeanEngine.")
    else:
        print(f"   QuantLib {live['version']}")
        print(f"   {'T':>5} {'ql Gatheral':>14} {'ql default':>14} {'ql BranchCorr':>14} "
              f"{'mine albrecher':>15} {'|diff|':>9} {'mine 1993':>14}")
        worst_alb = 0.0
        for row in live["heston"]:
            mine = heston_price(S, K, row["T"], 0.03, 0.0, flag="c", **hest)
            old = heston_price(S, K, row["T"], 0.03, 0.0, flag="c", formulation="heston1993",
                               **hest)
            diff = abs(mine - row["Gatheral"])
            worst_alb = max(worst_alb, diff)
            print(f"   {row['T']:>5.1f} {row['Gatheral']:>14.8f} {row['default_ctor']:>14.8f} "
                  f"{row['BranchCorrection']:>14.8f} {mine:>15.8f} {diff:>9.1e} {old:>14.8f}")
        print(f"   Albrecher form vs QuantLib Gatheral: worst |diff| {worst_alb:.1e} out to 30y."
              f" QuantLib's BranchCorrection form (the 1993 form with a rotation-count repair)")
        print(f"   agrees too; the UNREPAIRED 1993 form in this file does not.")
        worst_sabr = max(abs(my_sabr[Ks] - live["sabr"][Ks]) for Ks in sabr["strikes"])
        print(f"   SABR: worst |mine - ql.sabrVolatility| over 7 strikes = {worst_sabr:.1e}")
        print(f"   CRR tree, European put -- QuantLib's \"crr\" is NOT the textbook probability:")
        print(f"   {'n':>5} {'ql BinomialVanilla':>19} {'p=(e^(r-q)dt-d)/(u-d)':>23} "
              f"{'diff':>10} {'p=1/2+drift/2sd':>17} {'diff':>10}")
        worst_qlprob = 0.0
        for n, v in live["crr_european_put"].items():
            tb = crr_price(S, K, T, r, q, sigma, n, "p")
            qp = crr_price(S, K, T, r, q, sigma, n, "p", prob="quantlib")
            worst_qlprob = max(worst_qlprob, abs(qp - v))
            print(f"   {n:>5} {v:>19.10f} {tb:>23.10f} {tb - v:>+10.1e} {qp:>17.10f} "
                  f"{qp - v:>+10.1e}")
        print(f"   -> QuantLib's CRR uses pu = 1/2 + (r-q-sigma^2/2)dt / (2 sigma sqrt(dt)), "
              f"reproduced to {worst_qlprob:.1e}. Both converge; benchmark like for like.")
        print(f"   American put: CRR 800 steps {am:.6f} vs QdFpAmericanEngine "
              f"{live['american_put_qdfp']:.6f} (diff {am - live['american_put_qdfp']:+.6f})")
        mcq = live["mc"]
        print(f"   MCEuropeanEngine, requiredSamples=100,000: std err {mcq[(False, 1)][1]:.4f} "
              f"at 1 step, {mcq[(False, 52)][1]:.4f} at 52 steps; antithetic "
              f"{mcq[(True, 1)][1]:.4f} -- the same two facts, in QuantLib's own error estimate")
        sd_one = payoff_std(S, K, T, r, q, sigma, "c", antithetic=False)
        sd_pair = payoff_std(S, K, T, r, q, sigma, "c", antithetic=True)
        n_plain = (sd_one / mcq[(False, 1)][1]) ** 2
        n_anti = (sd_pair / mcq[(True, 1)][1]) ** 2
        print(f"   payoff std {sd_one:.3f}, antithetic-pair std {sd_pair:.3f}: QuantLib's error "
              f"estimates imply {n_plain:,.0f} payoffs plain and {n_anti:,.0f} PAIRS antithetic")
        print(f"   -> with antitheticVariate=True, requiredSamples counts pairs: the antithetic "
              f"run evaluated about {2 * n_anti:,.0f} paths, not 100,000")

    print("\n" + "=" * W)
    print("RULE: price Heston with the Albrecher/Gatheral form, average CRR odd and even steps,")
    print("      and report Monte Carlo bias and standard error separately -- steps fix one,")
    print("      paths fix the other.")
    print("=" * W)
