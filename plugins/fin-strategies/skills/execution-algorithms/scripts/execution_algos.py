#!/usr/bin/env python3
"""Execution schedules, Almgren-Chriss in closed form, and the shortfall nobody reports.

Three things, on one seeded session and on the Almgren-Chriss paper's own parameters:

  1. VWAP / TWAP / POV schedules against a synthetic intraday volume profile, and what a
     participation-rate cap actually does to the tail of an order;
  2. Almgren & Chriss (2000) in closed form - kappa, the trajectory, and the expected cost
     and variance - reproducing the paper's own printed numbers, and comparing the optimal
     trajectory with TWAP at a stated risk aversion;
  3. Perold's implementation-shortfall decomposition, with the opportunity term that a
     fill-only analysis cannot see because unfilled shares leave no fill record.

The two traps:

  * benchmarking an order against the VWAP the order's own impact moved. This is NOT the
    dilution effect that execution-cost-analysis measures (fills held fixed, benchmark
    drifting toward them) - here the PRICES move, for the trader and for everyone else in
    the session, and the SHARE of the true cost the benchmark absorbs grows with
    participation: 5% of it at 1%, 30% of it at 25%;
  * "Almgren-Chriss beats TWAP". At any positive risk aversion it does not, on expected
    cost - it deliberately pays MORE to buy less variance, and TWAP is the paper's own
    minimum-expected-cost trajectory.

ALMGREN & CHRISS (2000), "Optimal Execution of Portfolio Transactions". Every formula below
was read in the December 2000 preprint (smallake.kr/wp-content/uploads/2016/03/optliq.pdf),
equation numbers as printed there:

  (6)  g(v) = gamma * v                                   permanent impact, linear
  (7)  h(n_k/tau) = eps*sgn(n_k) + (eta/tau)*n_k          temporary impact, linear
  (5)  V(x) = sigma^2 * sum_k tau * x_k^2
  (8)  E(x) = 0.5*gamma*X^2 + eps*sum_k |n_k| + (eta~/tau)*sum_k n_k^2
       with  eta~ = eta - 0.5*gamma*tau        (unnumbered, immediately after (8))
  (9)  linear trajectory: n_k = X/N, x_k = (N-k)*X/N
  (10) E = 0.5*gamma*X^2 + eps*X + (eta - 0.5*gamma*tau) * X^2/T
  (11) V = (1/3)*sigma^2*X^2*T*(1 - 1/N)*(1 - 1/(2N))
  (16) (1/tau^2)*(x_{j-1} - 2 x_j + x_{j+1}) = kappa~^2 * x_j
       with kappa~^2 = lambda*sigma^2 / eta~, and (2/tau^2)*(cosh(kappa*tau) - 1) = kappa~^2
  (17) x_j = sinh(kappa*(T - t_j)) / sinh(kappa*T) * X
  (18) n_j = 2*sinh(kappa*tau/2) / sinh(kappa*T) * cosh(kappa*(T - t_{j-1/2})) * X
  (19) kappa ~ kappa~ + O(tau^2) ~ sqrt(lambda*sigma^2 / eta) + O(tau)
  (20) E(X) = 0.5*gamma*X^2 + eps*X
              + eta~ * X^2 * tanh(kappa*tau/2)
                * (tau*sinh(2*kappa*T) + 2*T*sinh(kappa*tau))
                / (2 * tau^2 * sinh^2(kappa*T))
       V(X) = 0.5*sigma^2*X^2
              * (tau*sinh(kappa*T)*cosh(kappa*(T - tau)) - T*sinh(kappa*tau))
                / (sinh^2(kappa*T) * sinh(kappa*tau))
  section 2.3: the half-life theta = 1/kappa, "the amount of time it takes to deplete the
       portfolio by a factor of e" - and it is INDEPENDENT of the execution time T.

Table 1 (p. 25), the paper's test case, verbatim: S0 = 50 $/share; X = 10^6 share; T = 5
days; N = 5; sigma = 0.95 ($/share)/day^(1/2); alpha = 0.02 ($/share)/day; eps = 0.0625
$/share; gamma = 2.5e-7 $/share^2; eta = 2.5e-6 ($/share)/(share/day); lambda_u = 1e-6 /$;
lambda_v = 1.645.

Run:  python execution_algos.py     (numpy + pandas only, fixed seed, a few seconds)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260909
BPS = 1e4

# Almgren-Chriss (2000) Table 1, p. 25 - the paper's own test case, quoted above.
AC_TABLE1 = {
    "S0": 50.0, "X": 1e6, "T": 5.0, "N": 5, "sigma": 0.95, "alpha": 0.02,
    "eps": 0.0625, "gamma": 2.5e-7, "eta": 2.5e-6, "lam_u": 1e-6, "lam_v": 1.645,
}


# ============================================================ Part 1: intraday schedules
def simulate_session(n_bins: int = 390, seed: int = SEED, s0: float = 50.0,
                     day_volume: float = 5e6, vol_bps: float = 3.0,
                     drift_bps: float = 0.0) -> dict:
    """One session of minute bars: a U-shaped volume profile and an arithmetic mid path.

    Arithmetic (not geometric) because that is Almgren-Chriss's own price model, eq. (1).
    The drift is imposed deterministically rather than sampled, so no conclusion below
    depends on which way one draw happened to go.
    """
    if n_bins < 10 or day_volume <= 0:
        raise ValueError("need at least 10 bins and positive volume")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_bins)
    shape = 1.0 + 3.0 * (np.exp(-8.0 * t) + np.exp(-8.0 * (1.0 - t)))   # U-shape
    volume = shape * rng.lognormal(0.0, 0.25, n_bins)
    volume = volume / volume.sum() * day_volume
    steps = rng.normal(0.0, s0 * vol_bps / BPS, n_bins)
    steps -= steps.mean()                       # pin the endpoint so drift is exactly imposed
    mid = s0 + np.cumsum(steps) + np.linspace(0.0, s0 * drift_bps / BPS, n_bins)
    return {"mid": mid, "volume": volume, "n_bins": n_bins, "s0": s0,
            "day_volume": float(volume.sum())}


def twap_schedule(shares: float, n_bins: int) -> np.ndarray:
    """Equal size per bin. Ignores volume entirely - that is the whole definition."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    return np.full(n_bins, shares / n_bins, dtype=float)


def vwap_schedule(shares: float, volume_forecast: np.ndarray) -> np.ndarray:
    """Proportional to the FORECAST volume profile. A realised profile is not available
    in advance, which is the practical difference between this and a VWAP benchmark."""
    v = np.asarray(volume_forecast, dtype=float)
    if v.ndim != 1 or v.size < 1 or np.any(v < 0) or v.sum() <= 0:
        raise ValueError("volume_forecast must be a non-negative 1-D array with positive sum")
    return shares * v / v.sum()


def pov_schedule(shares: float, volume: np.ndarray, rate: float) -> np.ndarray:
    """Trade `rate` of each bin's volume until the order is done. Returns the filled sizes.

    A POV order does not promise completion: if the day is thin, the tail is simply not
    filled. `pov_shortfall` below is that unfilled quantity, and it is the input to the
    opportunity term of the implementation shortfall.
    """
    v = np.asarray(volume, dtype=float)
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    want = rate * v
    done = np.cumsum(want)
    fill = np.where(done <= shares, want, np.maximum(0.0, shares - (done - want)))
    return fill


def pov_shortfall(shares: float, volume: np.ndarray, rate: float) -> float:
    """Shares a POV order at `rate` fails to complete within the session."""
    return float(max(0.0, shares - pov_schedule(shares, volume, rate).sum()))


def fill_prices(mid: np.ndarray, own: np.ndarray, volume: np.ndarray, side: int = 1,
                gamma: float = 2.5e-7, eta: float = 2.5e-6, day_len: float = 1.0):
    """Prices with the order's OWN impact, in the Almgren-Chriss linear form.

    Permanent, eq. (6): the mid is displaced by gamma * (cumulative own shares) and stays
    displaced - so every later print in the session, yours and everybody else's, is at the
    moved price. Temporary, eq. (7): the trader additionally pays eta * (own rate) on their
    own fills only. `side=+1` is a buy (impact pushes the price up and costs the buyer).

    Returns (mid_with_permanent_impact, own_fill_price).
    """
    own = np.asarray(own, dtype=float)
    tau = day_len / len(mid)
    perm = side * gamma * np.cumsum(own)
    moved = np.asarray(mid, dtype=float) + perm
    temp = side * eta * (own / tau)
    return moved, moved + temp


def session_vwap(price: np.ndarray, volume: np.ndarray) -> float:
    v = np.asarray(volume, dtype=float)
    return float(np.sum(np.asarray(price, dtype=float) * v) / v.sum())


def cost_bps(avg_fill: float, benchmark: float, side: int = 1) -> float:
    """Positive = the execution cost money against that benchmark. side=+1 buy."""
    return side * (avg_fill - benchmark) / benchmark * BPS


def vwap_self_contamination(session: dict, participations, side: int = 1,
                            gamma: float = 2.5e-7, eta: float = 2.5e-6) -> pd.DataFrame:
    """How much of the order's true cost the realised session VWAP hides.

    For each participation rate: run the SAME (volume-proportional) schedule, let it move
    the market through eq. (6) and eq. (7), then compare the average fill against
      * the realised session VWAP, which contains the impact the order caused, and
      * the counterfactual VWAP of the same session with the order absent.
    The gap between those two costs is the bias, and it is one-signed.
    """
    mid, base_vol = session["mid"], session["volume"]
    clean_vwap = session_vwap(mid, base_vol)
    rows = []
    for part in participations:
        if not 0.0 < part < 1.0:
            raise ValueError("participation must be in (0, 1)")
        qty = base_vol.sum() * part / (1.0 - part)
        own = vwap_schedule(qty, base_vol)
        moved, fills = fill_prices(mid, own, base_vol, side, gamma, eta)
        avg = float(np.sum(fills * own) / own.sum())
        # everyone else prints at the permanently moved mid; the trader prints at `fills`
        all_px = np.concatenate([moved, fills])
        all_vol = np.concatenate([base_vol, own])
        rows.append({
            "participation": part, "shares": qty, "avg_fill": avg,
            "realised_vwap": session_vwap(all_px, all_vol),
            "clean_vwap": clean_vwap,
            "cost_vs_realised": cost_bps(avg, session_vwap(all_px, all_vol), side),
            "cost_vs_clean": cost_bps(avg, clean_vwap, side),
        })
    out = pd.DataFrame(rows)
    out["hidden_bps"] = out["cost_vs_clean"] - out["cost_vs_realised"]
    return out


# ====================================================== Part 2: Almgren & Chriss (2000)
def eta_tilde(eta: float, gamma: float, tau: float) -> float:
    """AC (2000), unnumbered display after eq. (8):  eta~ = eta - 0.5*gamma*tau."""
    return eta - 0.5 * gamma * tau


def ac_kappa(lam: float, sigma: float, eta: float, gamma: float, tau: float) -> dict:
    """kappa~ and the exact kappa of AC (2000) section 2.2.

    kappa~^2 = lambda*sigma^2 / eta~                      (unnumbered, after eq. 16)
    (2/tau^2) * (cosh(kappa*tau) - 1) = kappa~^2          (unnumbered, after eq. 16)
    kappa ~ sqrt(lambda*sigma^2/eta)                      (eq. 19, the small-tau form)

    lambda < 0 (a risk-SEEKING trader, the paper's trajectory C) makes kappa~^2 negative and
    kappa imaginary; the difference equation then has the trigonometric solution, which
    `ac_trajectory` handles with sin/cos. lambda = 0 gives the linear trajectory of eq. (9).
    """
    et = eta_tilde(eta, gamma, tau)
    if et <= 0:
        raise ValueError("eta~ = eta - gamma*tau/2 must be positive for a convex problem")
    kt2 = lam * sigma ** 2 / et
    c = 1.0 + 0.5 * kt2 * tau ** 2
    if c > 1.0:
        kappa = float(np.arccosh(c) / tau)
    elif c == 1.0:
        kappa = 0.0
    else:
        if c < -1.0:
            raise ValueError("lambda is too negative for a real trajectory at this tau")
        kappa = float(np.arccos(c) / tau)       # imaginary kappa; magnitude of the frequency
    return {"eta_tilde": et, "kappa_tilde_sq": kt2,
            "kappa_tilde": float(np.sqrt(abs(kt2))) * (1.0 if kt2 >= 0 else -1.0),
            "kappa": kappa, "trigonometric": c < 1.0,
            "kappa_eq19": float(np.sqrt(abs(lam) * sigma ** 2 / eta)),
            "half_life": float("inf") if kappa == 0 else 1.0 / kappa}


def ac_trajectory(X: float, T: float, N: int, kappa: float, trigonometric: bool = False):
    """AC (2000) eq. (17) and (18): holdings x_j, j = 0..N, and the trade list n_j, j = 1..N.

    kappa = 0 is the linear trajectory of eq. (9), which is the limit of (17).
    """
    if N < 1 or T <= 0:
        raise ValueError("need N >= 1 and T > 0")
    tau = T / N
    t = np.arange(N + 1) * tau
    if kappa == 0.0:
        x = (T - t) / T * X
    elif trigonometric:                                     # imaginary kappa: sinh -> sin
        x = np.sin(kappa * (T - t)) / np.sin(kappa * T) * X
    else:
        x = np.sinh(kappa * (T - t)) / np.sinh(kappa * T) * X
    n = -np.diff(x)
    return x, n


def ac_trade_list(X: float, T: float, N: int, kappa: float) -> np.ndarray:
    """AC (2000) eq. (18) written out directly, as a cross-check on -diff(eq. 17)."""
    tau = T / N
    t_half = (np.arange(1, N + 1) - 0.5) * tau
    if kappa == 0.0:
        return np.full(N, X / N)
    return (2.0 * np.sinh(0.5 * kappa * tau) / np.sinh(kappa * T)
            * np.cosh(kappa * (T - t_half)) * X)


def direct_cost(x: np.ndarray, T: float, sigma: float, eta: float, gamma: float,
                eps: float) -> tuple[float, float]:
    """AC (2000) eq. (8) and eq. (5) evaluated on ANY holdings path. The reference.

    E(x) = 0.5*gamma*X^2 + eps*sum|n_k| + (eta~/tau)*sum n_k^2      (8)
    V(x) = sigma^2 * sum_k tau * x_k^2                              (5)

    The closed forms below must reproduce these exactly; the demo checks that they do.
    """
    x = np.asarray(x, dtype=float)
    N = x.size - 1
    tau = T / N
    n = -np.diff(x)
    et = eta_tilde(eta, gamma, tau)
    E = 0.5 * gamma * x[0] ** 2 + eps * np.abs(n).sum() + et / tau * np.sum(n ** 2)
    V = sigma ** 2 * tau * np.sum(x[1:] ** 2)        # k = 1..N, the holdings held over each step
    return float(E), float(V)


def linear_cost(X: float, T: float, N: int, sigma: float, eta: float, gamma: float,
                eps: float) -> tuple[float, float]:
    """AC (2000) eq. (10) and eq. (11): the constant-rate (TWAP) trajectory in closed form."""
    tau = T / N
    E = 0.5 * gamma * X ** 2 + eps * X + (eta - 0.5 * gamma * tau) * X ** 2 / T
    V = (1.0 / 3.0) * sigma ** 2 * X ** 2 * T * (1.0 - 1.0 / N) * (1.0 - 1.0 / (2.0 * N))
    return float(E), float(V)


def ac_cost(X: float, T: float, N: int, kappa: float, sigma: float, eta: float,
            gamma: float, eps: float) -> tuple[float, float]:
    """AC (2000) eq. (20): E(X) and V(X) of the optimal trajectory, in closed form."""
    tau = T / N
    if kappa == 0.0:
        return linear_cost(X, T, N, sigma, eta, gamma, eps)
    et = eta_tilde(eta, gamma, tau)
    sh_T, sh_tau = np.sinh(kappa * T), np.sinh(kappa * tau)
    E = (0.5 * gamma * X ** 2 + eps * X
         + et * X ** 2 * np.tanh(0.5 * kappa * tau)
         * (tau * np.sinh(2.0 * kappa * T) + 2.0 * T * sh_tau)
         / (2.0 * tau ** 2 * sh_T ** 2))
    V = (0.5 * sigma ** 2 * X ** 2
         * (tau * sh_T * np.cosh(kappa * (T - tau)) - T * sh_tau)
         / (sh_T ** 2 * sh_tau))
    return float(E), float(V)


def eq16_residual(x: np.ndarray, T: float, kappa_tilde_sq: float) -> float:
    """Max |LHS - RHS| of AC eq. (16) on the interior points. Zero iff (17) solves (16)."""
    x = np.asarray(x, dtype=float)
    N = x.size - 1
    tau = T / N
    lhs = (x[:-2] - 2.0 * x[1:-1] + x[2:]) / tau ** 2
    return float(np.max(np.abs(lhs - kappa_tilde_sq * x[1:-1])))


def efficient_frontier(lams, X: float, T: float, N: int, sigma: float, eta: float,
                       gamma: float, eps: float) -> pd.DataFrame:
    """(E, V) of the optimal trajectory for a list of risk aversions - AC section 2.1."""
    rows = []
    for lam in lams:
        k = ac_kappa(lam, sigma, eta, gamma, T / N)
        x, _ = ac_trajectory(X, T, N, k["kappa"], k["trigonometric"])
        E, V = direct_cost(x, T, sigma, eta, gamma, eps)
        rows.append({"lambda": lam, "kappa": k["kappa"], "half_life": k["half_life"],
                     "E": E, "sqrt_V": np.sqrt(V), "U": E + lam * V,
                     "frac_done_half": float(1.0 - x[len(x) // 2] / X)})
    return pd.DataFrame(rows)


# ============================================ Part 3: Perold's implementation shortfall
def implementation_shortfall(decision_px: float, arrival_px: float, avg_fill_px: float,
                             final_px: float, target_qty: float, filled_qty: float,
                             fees: float = 0.0, side: int = 1) -> dict:
    """The four-way decomposition, in currency, on a side-signed convention.

        delay        = filled   * (arrival  - decision)     the research-to-desk handoff
        execution    = filled   * (avg_fill - arrival)      the algorithm
        opportunity  = unfilled * (final    - decision)     the shares never bought
        fees         = explicit
        total        = delay + execution + opportunity + fees

    Positive = a cost to the fund. The identity is exact by construction and the returned
    `identity_error` proves it on every call. `bps` are on the notional the decision
    intended, target_qty * decision_px, so the components are additive in bps too.

    ATTRIBUTION. The paper-portfolio framing, the decision-time bid-ask midpoint as the
    paper price, and charging the unfilled shares are Perold (1988), "The Implementation
    Shortfall: Paper Versus Reality", JPM 14(3), 4-9. His own decomposition has TWO terms,
    execution and opportunity, with commissions folded into the net transaction price;
    there is no arrival price and no separate delay term in his math. The FOUR-way split
    above - commission, price impact, timing/delay, opportunity - is Wagner & Edwards
    (1993), "Best Execution", Financial Analysts Journal 49(1), 65-71, p. 67. It is a
    refinement of Perold's execution term, since delay + execution = filled * (avg_fill -
    decision), not a contradiction of it. Calling it "Perold's four-way decomposition" is
    the common miscitation.
    """
    if target_qty <= 0 or filled_qty < 0 or filled_qty > target_qty:
        raise ValueError("need 0 <= filled_qty <= target_qty and target_qty > 0")
    if side not in (1, -1):
        raise ValueError("side must be +1 (buy) or -1 (sell)")
    unfilled = target_qty - filled_qty
    delay = side * filled_qty * (arrival_px - decision_px)
    execution = side * filled_qty * (avg_fill_px - arrival_px)
    opportunity = side * unfilled * (final_px - decision_px)
    total = delay + execution + opportunity + fees
    notional = target_qty * decision_px
    paper = side * target_qty * (final_px - decision_px)
    real = side * filled_qty * (final_px - avg_fill_px) - fees
    out = {"delay": delay, "execution": execution, "opportunity": opportunity,
           "fees": fees, "total": total, "fill_rate": filled_qty / target_qty,
           "identity_error": abs(total - (paper - real))}
    out.update({f"{k}_bps": v / notional * BPS
                for k, v in list(out.items())[:5]})
    return out


# ---------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    print("=" * 90)
    print("EXECUTION ALGORITHMS  --  schedules, Almgren-Chriss (2000) in closed form, shortfall")
    print("=" * 90)

    # ---- 1. schedules on one seeded session
    ses = simulate_session(drift_bps=60.0)
    mid, vol = ses["mid"], ses["volume"]
    qty = 250_000.0
    print(f"\n1. One session: {ses['n_bins']} minute bins, {ses['day_volume']:,.0f} shares "
          f"of base volume,")
    print(f"   mid {mid[0]:.4f} -> {mid[-1]:.4f} ({(mid[-1] / mid[0] - 1) * BPS:+.0f} bps against "
          f"a buyer). Order: {qty:,.0f} shares.")
    print(f"   {'schedule':<26}{'avg fill':>11}{'vs arrival':>12}{'vs clean VWAP':>15}"
          f"{'done by':>9}{'unfilled':>10}")
    scheds = {"TWAP": twap_schedule(qty, ses["n_bins"]),
              "VWAP (perfect forecast)": vwap_schedule(qty, vol),
              "POV 10%": pov_schedule(qty, vol, 0.10),
              "POV 3%": pov_schedule(qty, vol, 0.03)}
    for name, own in scheds.items():
        _, fills = fill_prices(mid, own, vol)
        avg = float(np.sum(fills * own) / own.sum())
        done = int(np.argmax(np.cumsum(own) >= own.sum() * 0.999)) + 1
        print(f"   {name:<26}{avg:>11.4f}{cost_bps(avg, mid[0]):>+11.1f}"
              f"{cost_bps(avg, session_vwap(mid, vol)):>+15.1f}{done:>8}m"
              f"{qty - own.sum():>10,.0f}")
    print("   'clean VWAP' is the session VWAP the market would have had with the order")
    print("   absent - not a number a desk can compute. Part 2 is about the one it can.")
    print("   POV is the only one of the four that can fail to finish, and the 3% cap leaves")
    print("   a real tail behind. That tail is not free - it is the opportunity term in part 5.")

    # ---- 2. TRAP 1 - the VWAP your own impact moved
    print("\n2. TRAP 1 - benchmarking against the VWAP your own order moved")
    tab = vwap_self_contamination(ses, (0.01, 0.05, 0.10, 0.25))
    tab["hidden_share"] = tab["hidden_bps"] / tab["cost_vs_clean"]
    print(f"   {'participation':>14}{'avg fill':>11}{'realised VWAP':>15}{'clean VWAP':>12}"
          f"{'measured cost':>15}{'TRUE cost':>12}{'hidden':>9}{'% of true hidden':>18}")
    for _, r in tab.iterrows():
        print(f"   {r['participation']:>13.0%}{r['avg_fill']:>11.4f}{r['realised_vwap']:>15.4f}"
              f"{r['clean_vwap']:>12.4f}{r['cost_vs_realised']:>+14.1f}"
              f"{r['cost_vs_clean']:>+12.1f}{r['hidden_bps']:>+9.1f}"
              f"{r['hidden_share']:>17.0%}")
    print("   Both costs rise with participation, so the benchmark is not simply flat - what")
    print("   grows is the SHARE of the true cost the benchmark absorbs: 5% of it at 1%")
    print("   participation, 30% of it at 25%. Permanent impact moves the mid for every")
    print("   other print in the session, so this is not only your own prints diluting the")
    print("   average: the whole benchmark has been pushed toward you.")
    print("   ! Distinct from the dilution effect in execution-cost-analysis section 2, where")
    print("     the fills are held FIXED and only the weighting changes. Both are one-signed")
    print("     the same way, and they compound.")
    print("   ! The linear impact of eq. (7) is the paper's own weakest assumption - AC write")
    print("     that in the temporary term 'we would expect nonlinear effects to be most")
    print("     important, and the approximation (7) to be most doubtful'. Read the direction")
    print("     of this table, not its magnitudes, and never extrapolate it past ~25%.")

    # ---- 3. Almgren-Chriss on the paper's own Table 1
    p = AC_TABLE1
    X, T, N, sig, eta, gam, eps = (p["X"], p["T"], p["N"], p["sigma"], p["eta"],
                                   p["gamma"], p["eps"])
    tau = T / N
    k = ac_kappa(p["lam_u"], sig, eta, gam, tau)
    print(f"\n3. Almgren-Chriss (2000), reproducing the paper's Table 1 test case")
    print(f"   S0={p['S0']:.0f}, X={X:,.0f}, T={T:.0f}d, N={N}, sigma={sig}, eps={eps}, "
          f"gamma={gam:g}, eta={eta:g}, lambda={p['lam_u']:g}")
    print(f"   eta~ = eta - gamma*tau/2 = {k['eta_tilde']:.6g}   "
          f"kappa~^2 = lambda*sigma^2/eta~ = {k['kappa_tilde_sq']:.6g}")
    print(f"   exact kappa from cosh(kappa*tau) = 1 + kappa~^2 tau^2/2 : {k['kappa']:.4f}/day")
    print(f"   eq. (19) small-tau form sqrt(lambda*sigma^2/eta)        : {k['kappa_eq19']:.4f}/day")
    print(f"   PAPER (p. 24): 'we have from (19) that for the optimal strategy, kappa ~ 0.6/day,")
    print(f"   so kappa*T ~ 3'   ->  here kappa*T = {k['kappa'] * T:.2f}   half-life 1/kappa = "
          f"{k['half_life']:.2f} days")
    bh = sig * np.sqrt(T)
    print(f"   PAPER (p. 24): holding the position untraded has sigma*sqrt(T) = 2.12 $/share")
    print(f"   and sqrt(V) = $2.12M   ->  here {bh:.4f} $/share, ${bh * X / 1e6:.2f}M")

    x_ac, n_ac = ac_trajectory(X, T, N, k["kappa"])
    print(f"\n   eq. (17) holdings x_j : " + "  ".join(f"{v:,.0f}" for v in x_ac))
    print(f"   eq. (18) trade list n_j: " + "  ".join(f"{v:,.0f}" for v in n_ac))
    print(f"   -diff(eq. 17) vs eq. (18) written out : max abs diff "
          f"{np.max(np.abs(n_ac - ac_trade_list(X, T, N, k['kappa']))):.3e}")
    print(f"   eq. (17) substituted back into eq. (16): max residual "
          f"{eq16_residual(x_ac, T, k['kappa_tilde_sq']):.3e}")
    E_cf, V_cf = ac_cost(X, T, N, k["kappa"], sig, eta, gam, eps)
    E_d, V_d = direct_cost(x_ac, T, sig, eta, gam, eps)
    print(f"   eq. (20) closed form  vs  eq. (8)/(5) summed on the trajectory:")
    print(f"     E  {E_cf:>18,.2f}  vs {E_d:>18,.2f}   rel diff {abs(E_cf - E_d) / E_d:.3e}")
    print(f"     V  {V_cf:>18,.4g}  vs {V_d:>18,.4g}   rel diff {abs(V_cf - V_d) / V_d:.3e}")
    x_lin, _ = ac_trajectory(X, T, N, 0.0)
    E_l, V_l = linear_cost(X, T, N, sig, eta, gam, eps)
    E_ld, V_ld = direct_cost(x_lin, T, sig, eta, gam, eps)
    print(f"   eq. (10)/(11) closed form vs the same sums on the linear trajectory:")
    print(f"     E  {E_l:>18,.2f}  vs {E_ld:>18,.2f}   rel diff {abs(E_l - E_ld) / E_ld:.3e}")
    print(f"     V  {V_l:>18,.4g}  vs {V_ld:>18,.4g}   rel diff {abs(V_l - V_ld) / V_ld:.3e}")
    E_bad, _ = ac_cost(X, T, N, k["kappa"], sig, eta + 0.5 * gam * tau, gam, eps)
    print(f"   and the cost of writing eta where the paper writes eta~ in eq. (20):")
    print(f"     E {E_bad:>19,.2f}  vs {E_cf:>18,.2f}   {(E_bad / E_cf - 1) * 100:+.2f}% at these")
    print(f"     parameters, and the gap grows with gamma*tau/eta")

    # ---- 4. TRAP 2 - "Almgren-Chriss beats TWAP"
    print(f"\n4. TRAP 2 - 'Almgren-Chriss beats TWAP'. At lambda = {p['lam_u']:g} it does not,"
          f" on cost")
    lam = p["lam_u"]
    print(f"   {'trajectory':<34}{'E[cost] $':>14}{'E in bps':>10}{'sqrt(V) $':>14}"
          f"{'U = E + lam*V':>16}")
    for name, (E_, V_) in (("TWAP / linear, eq. (9)-(11)", (E_l, V_l)),
                           ("Almgren-Chriss, eq. (17)/(20)", (E_cf, V_cf))):
        print(f"   {name:<34}{E_:>14,.0f}{E_ / (X * p['S0']) * BPS:>10.1f}"
              f"{np.sqrt(V_):>14,.0f}{E_ + lam * V_:>16,.0f}")
    print(f"   AC costs {E_cf - E_l:+,.0f} MORE in expectation and "
          f"{np.sqrt(V_l) - np.sqrt(V_cf):,.0f} LESS in standard deviation.")
    print(f"   It wins only on U = E + lambda*V, by {(E_l + lam * V_l) - (E_cf + lam * V_cf):,.0f},")
    print(f"   which is the objective it was derived to minimise. TWAP IS the minimum-")
    print(f"   EXPECTED-cost trajectory in this model - the paper calls eq. (9) 'minimum")
    print(f"   impact'. Quoting an AC schedule as cheaper than TWAP inverts the result.")
    print(f"\n   The frontier, and the half-life theta = 1/kappa (section 2.3):")
    print(f"   {'lambda':>12}{'kappa /day':>12}{'half-life d':>13}{'E[cost] $':>14}"
          f"{'sqrt(V) $':>13}{'% done by T/2':>15}")
    for _, r in efficient_frontier([2e-6, 1e-6, 2e-7, 0.0, -2e-7], X, T, N, sig, eta,
                                   gam, eps).iterrows():
        print(f"   {r['lambda']:>12.1e}{r['kappa']:>12.4f}{r['half_life']:>13.2f}"
              f"{r['E']:>14,.0f}{r['sqrt_V']:>13,.0f}{r['frac_done_half']:>14.0%}")
    print("   lambda = 2e-6, 0 and -2e-7 are the paper's trajectories A, B and C (Figure 1).")
    print("   B (lambda = 0) is exactly the linear trajectory; C is risk-SEEKING, so kappa is")
    print("   imaginary and the solution turns trigonometric - it postpones selling.")
    print("   ! theta = 1/kappa does not depend on T. Doubling the deadline does not slow an")
    print("     AC trade down; it just leaves more of the window unused.")

    # ---- 5. implementation shortfall, and the term nobody has the data for
    print("\n5. Perold's implementation shortfall, on the POV 3% order from part 1")
    own3 = pov_schedule(qty, vol, 0.03)
    _, f3 = fill_prices(mid, own3, vol)
    filled = float(own3.sum())
    avg3 = float(np.sum(f3 * own3) / filled)
    decision = float(mid[0]) - 0.02              # the PM decided before the desk saw it
    isr = implementation_shortfall(decision_px=decision, arrival_px=float(mid[0]),
                                   avg_fill_px=avg3, final_px=float(mid[-1]),
                                   target_qty=qty, filled_qty=filled,
                                   fees=0.0005 * filled)
    print(f"   decision {decision:.4f}  arrival {mid[0]:.4f}  avg fill {avg3:.4f}  "
          f"close {mid[-1]:.4f}")
    print(f"   target {qty:,.0f} shares, filled {filled:,.0f} ({isr['fill_rate']:.1%})")
    print(f"   {'component':<24}{'$':>14}{'bps of intended notional':>28}")
    for comp in ("delay", "execution", "opportunity", "fees", "total"):
        print(f"   {comp:<24}{isr[comp]:>14,.0f}{isr[comp + '_bps']:>27.1f}")
    print(f"   identity (paper - real) - total = {isr['identity_error']:.3e}")
    fill_only = isr["execution_bps"]
    print(f"   A fill-only TCA reports {fill_only:.1f} bps - the execution row - because the")
    print(f"   delay and opportunity terms have no fill record to be computed from. The whole")
    print(f"   order actually cost {isr['total_bps']:.1f} bps, and "
          f"{isr['opportunity_bps'] / isr['total_bps']:.0%} of that is the "
          f"{qty - filled:,.0f} shares")
    print(f"   the 3% cap never bought. Lowering the participation rate does not lower the")
    print(f"   cost; it moves the cost into the column your TCA cannot see.")

    print("\nRule: pick the benchmark before the trade and make it exogenous, quote"
          " implementation shortfall against the decision price with the unfilled shares"
          " charged, and read Almgren-Chriss as buying variance reduction with expected cost.")
