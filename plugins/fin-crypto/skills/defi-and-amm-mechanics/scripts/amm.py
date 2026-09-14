#!/usr/bin/env python3
"""Automated market makers: the invariant, price impact, impermanent loss, and what a range does.

Six demonstrations, each printing the numbers quoted in ../SKILL.md:

  1. invariant   - x*y = k, the closed-form price impact, and the identity that the EXECUTED
                   price moves half as far in log terms as the pool price
  2. fees        - 30 bp on the input is not 30 bp on the output; the protocol split; and
                   sqrt(k) as the fee accumulator, checked against a simulated tape
  3. IL          - the closed form 2*sqrt(r)/(1+r) reproduced exactly from reserve arithmetic
  4. IL vs fees  - the break-even daily volume-to-TVL ratio at several volatilities, so the
                   answer to "do fees cover impermanent loss" is a number
  5. range       - concentrated liquidity: capital efficiency 1/(1 - 1/sqrt(m)), the time in
                   range that pays for it, and the same amplification applied to the loss
  6. the honest  - a sandwich extracts almost exactly the slippage tolerance you set; gas, MEV
                   frequency and priority auctions are NOT measurable offline and are named

Seeded synthetic paths; the formulas are read from the protocols' own whitepapers (cited in the
constants). No network at run time, numpy/pandas only, fixed seed, no file writes, ASCII output.

Run:  python amm.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260910
W = 96

# ----------------------------------------------------------------------------------------------
# Source-verified constants - read 2026-09-10 in the protocols' own whitepapers
# ----------------------------------------------------------------------------------------------

# Uniswap v2 Core, Adams / Zinsmeister / Robinson, March 2020, app.uniswap.org/whitepaper.pdf
#   sec 1: "Traders pay a 30-basis-point fee on trades, which goes to liquidity providers",
#          "maintaining the invariant that the product of the reserves cannot decrease"
#   sec 2.2 eq (1): p_t = r_t^a / r_t^b            (the marginal price, fees excluded)
#   sec 2.4: "Uniswap v2 includes a 0.05% protocol fee that can be turned on and off" -
#          "traders will continue to pay a 0.30% fee on all trades; 83.3% of that fee (0.25% of
#          the amount traded) will go to liquidity providers, and 16.6% of that fee (0.05% of
#          the amount traded) will go to the feeTo address", taken as a 1/6 cut
#   sec 2.4 eq (4): f_1,2 = 1 - sqrt(k_1)/sqrt(k_2)   (accumulated fees as a share of liquidity)
#   sec 2.4 footnote 5: because the fee is charged on the INPUT, the fee relative to the
#          WITHDRAWN amount is 1/(1 - 0.003) - 1 = 3/997
V2 = dict(fee=0.0030, protocol_cut=1.0 / 6.0, lp_share=0.0025, protocol_share=0.0005,
          fee_on="input", withdrawn_fee_fraction=(3.0, 997.0))

# Uniswap v3 Core, Adams / Zinsmeister / Salem / Keefer / Robinson, March 2021,
#   app.uniswap.org/whitepaper-v3.pdf
#   sec 2 eq (2.1): x * y = k
#   sec 2: "The amount of liquidity provided can be measured by the value L, which is equal to
#          sqrt(k)", and eq (2.2): (x + L/sqrt(p_b)) * (y + L*sqrt(p_a)) = L^2
#   sec 2: "When the price exits a position's range, the position's liquidity is no longer
#          active, and no longer earns fees."
#   sec 3.1: "The initial fee tiers and tick spacings supported are 0.05% ..., 0.30% ..., and 1%"
#   sec 3.2.1: fees "are no longer" continuously deposited as liquidity - "fee earnings are
#          stored separately and held as the tokens in which the fees are paid", so a v3
#          position's fees do NOT compound while a v2 position's do
V3 = dict(fee_tiers=(0.0005, 0.0030, 0.0100), liquidity="L = sqrt(k)",
          out_of_range="no liquidity, no fees", fees_compound=False)


# ----------------------------------------------------------------------------------------------
# 1. The invariant and price impact
# ----------------------------------------------------------------------------------------------
def cp_swap(x: float, y: float, dx: float, fee: float = V2["fee"]) -> dict[str, float]:
    """Sell `dx` of token X into an (x, y) constant-product pool. Fee is charged on the INPUT."""
    if x <= 0 or y <= 0 or dx < 0 or not 0.0 <= fee < 1.0:
        raise ValueError("reserves and dx must be non-negative and the fee a fraction")
    g = 1.0 - fee
    dy = g * dx * y / (x + g * dx)
    x1, y1 = x + dx, y - dy
    return {"dy": dy, "x1": x1, "y1": y1, "spot_before": y / x, "spot_after": y1 / x1,
            "exec_price": dy / dx if dx > 0 else y / x, "k_before": x * y, "k_after": x1 * y1}


def cp_impact(u: float, fee: float = 0.0) -> dict[str, float]:
    """Closed form for a trade of size `u = dx / x`, as ratios to the pre-trade spot price.

    exec / spot = (1 - fee) / (1 + (1 - fee) * u)     the price the trader actually gets
    pool / spot = 1 / ((1 + u) * (1 + (1-fee)*u))     post-trade reserve price
    At fee = 0 the first is 1/(1+u): the EXECUTED price moves exactly half as far in log terms
    as the pool price does.
    """
    if u < 0 or not 0.0 <= fee < 1.0:
        raise ValueError("u must be non-negative and fee a fraction")
    g = 1.0 - fee
    pool_ratio = 1.0 / ((1.0 + u) * (1.0 + g * u))
    return {"u": u, "exec_ratio": g / (1.0 + g * u), "pool_ratio": pool_ratio,
            "slippage": g / (1.0 + g * u) - 1.0, "pool_move": pool_ratio - 1.0}


def swap_by_slices(x: float, y: float, dx: float, slices: int = 200_000,
                   fee: float = 0.0) -> float:
    """Walk the curve in slices; path independence holds only with zero fee and no other flow."""
    if slices < 1:
        raise ValueError("slices must be positive")
    cp_swap(x, y, dx, fee)  # validate before computing the first slice
    step = dx / slices
    g = 1.0 - fee
    xs, ys, out = x, y, 0.0
    for _ in range(slices):
        d = g * step * ys / (xs + g * step)
        out += d
        xs, ys = xs + step, ys - d
    return out


# ----------------------------------------------------------------------------------------------
# 2. Fees
# ----------------------------------------------------------------------------------------------
def fee_on_withdrawn(fee: float = V2["fee"]) -> float:
    """A fee charged on the INPUT is a larger fraction of the OUTPUT: 1/(1-fee) - 1."""
    if not 0.0 <= fee < 1.0:
        raise ValueError("fee must be a fraction")
    return 1.0 / (1.0 - fee) - 1.0


def fee_growth_from_k(k0: float, k1: float) -> float:
    """Uniswap v2 eq (4): accumulated fees as a share of the pool, `1 - sqrt(k0)/sqrt(k1)`."""
    if k0 <= 0 or k1 <= 0:
        raise ValueError("k must be positive")
    return 1.0 - np.sqrt(k0) / np.sqrt(k1)


def simulate_tape(x: float = 1_000_000.0, y: float = 1_000_000.0, n_trades: int = 4000,
                  trade_frac: float = 0.002, fee: float = V2["fee"],
                  seed: int = SEED) -> dict[str, float]:
    """Random two-sided flow through one pool; returns the realised fee growth two ways."""
    rng = np.random.default_rng(seed)
    xs, ys = x, y
    k0 = xs * ys
    collected_x = 0.0
    for _ in range(n_trades):
        size = trade_frac * (xs if rng.random() < 0.5 else ys) * rng.lognormal(0.0, 0.5)
        if rng.random() < 0.5:
            d = cp_swap(xs, ys, size, fee)
            collected_x += fee * size
            xs, ys = d["x1"], d["y1"]
        else:
            d = cp_swap(ys, xs, size, fee)               # the other direction, same function
            ys, xs = d["x1"], d["y1"]
    k1 = xs * ys
    return {"k0": k0, "k1": k1, "fee_growth": fee_growth_from_k(k0, k1),
            "sqrt_k_growth": float(np.sqrt(k1 / k0) - 1.0), "trades": n_trades}


# ----------------------------------------------------------------------------------------------
# 3-4. Impermanent loss
# ----------------------------------------------------------------------------------------------
def il_closed_form(price_ratio: float) -> float:
    """LP value / hold value - 1 for a full-range constant-product pool. Always <= 0."""
    r = np.asarray(price_ratio, dtype=float)
    if np.any(r <= 0):
        raise ValueError("price ratio must be positive")
    return float(2.0 * np.sqrt(r) / (1.0 + r) - 1.0) if np.ndim(r) == 0 \
        else 2.0 * np.sqrt(r) / (1.0 + r) - 1.0


def il_from_reserves(x0: float, y0: float, price_ratio: float) -> float:
    """The same number from reserve arithmetic: rebalance the pool to the new price, revalue."""
    if x0 <= 0 or y0 <= 0 or price_ratio <= 0:
        raise ValueError("reserves and price ratio must be positive")
    k, p0 = x0 * y0, y0 / x0
    p1 = p0 * price_ratio
    x1, y1 = np.sqrt(k / p1), np.sqrt(k * p1)               # arbitrage restores y/x = p1
    return (x1 * p1 + y1) / (x0 * p1 + y0) - 1.0


def il_vs_fees(annual_vol: float, days: int = 30, fee: float = V2["fee"],
               n_paths: int = 20_000, seed: int = SEED) -> dict[str, float]:
    """Mean realised IL over `days`, and the daily volume-to-TVL ratio that pays for it."""
    if annual_vol <= 0 or days < 1 or n_paths < 1 or not 0 < fee < 1:
        raise ValueError("vol, days and n_paths must be positive and fee in (0, 1)")
    rng = np.random.default_rng(seed + 3)
    dt = days / 365.0
    r = np.exp(rng.normal(-0.5 * annual_vol ** 2 * dt, annual_vol * np.sqrt(dt), n_paths))
    il = 2.0 * np.sqrt(r) / (1.0 + r) - 1.0
    mean_il = float(il.mean())
    return {"annual_vol": annual_vol, "days": days, "mean_il": mean_il,
            "median_il": float(np.median(il)), "worst_decile_il": float(np.quantile(il, 0.10)),
            "breakeven_daily_vol_to_tvl": -mean_il / (fee * days),
            "fee": fee, "paths": n_paths}


# ----------------------------------------------------------------------------------------------
# 5. Concentrated liquidity
# ----------------------------------------------------------------------------------------------
def capital_efficiency(m: float) -> float:
    """Capital multiple of a symmetric range [p/m, p*m] against the full curve: 1/(1 - 1/sqrt(m))."""
    if m <= 1.0:
        raise ValueError("m must exceed 1")
    return 1.0 / (1.0 - 1.0 / np.sqrt(m))


def cl_amounts(liquidity: float, p: float, pa: float, pb: float) -> tuple[float, float]:
    """Uniswap v3 eq (2.2) solved for the real reserves of a position at price `p`."""
    if not 0.0 < pa < pb or p <= 0 or liquidity <= 0:
        raise ValueError("need 0 < pa < pb, a positive price and positive liquidity")
    s, sa, sb = np.sqrt(np.clip(p, pa, pb)), np.sqrt(pa), np.sqrt(pb)
    return liquidity * (1.0 / s - 1.0 / sb), liquidity * (s - sa)


def cl_value(liquidity: float, p: float, pa: float, pb: float) -> float:
    x, y = cl_amounts(liquidity, p, pa, pb)
    return x * p + y


def cl_loss_vs_hold(m: float, price_ratio: float, p0: float = 1.0,
                    liquidity: float = 1.0) -> float:
    """A range position's value against holding its own opening basket, at `price_ratio`."""
    pa, pb = p0 / m, p0 * m
    x0, y0 = cl_amounts(liquidity, p0, pa, pb)
    p1 = p0 * price_ratio
    return cl_value(liquidity, p1, pa, pb) / (x0 * p1 + y0) - 1.0


def time_in_range(m: float, annual_vol: float, days: int = 30, steps_per_day: int = 24,
                  n_paths: int = 4000, seed: int = SEED) -> float:
    """Fraction of observations a GBM spends inside [1/m, m], starting at 1."""
    if m <= 1.0 or annual_vol <= 0 or min(days, steps_per_day, n_paths) < 1:
        raise ValueError("m must exceed 1 and vol, days, steps and paths be positive")
    rng = np.random.default_rng(seed + 5)
    n = days * steps_per_day
    dt = 1.0 / (365.0 * steps_per_day)
    p = np.exp(np.cumsum(rng.normal(-0.5 * annual_vol ** 2 * dt,
                                    annual_vol * np.sqrt(dt), (n_paths, n)), axis=1))
    return float(((p >= 1.0 / m) & (p <= m)).mean())


# ----------------------------------------------------------------------------------------------
# 6. Sandwich
# ----------------------------------------------------------------------------------------------
def sandwich(x: float, y: float, dx_victim: float, tolerance: float,
             fee: float = V2["fee"]) -> dict[str, float]:
    """A limit-saturating front-run size; not a profit optimizer or execution prediction.

    The attacker buys first, the victim executes at the worse price, the attacker sells back.
    Everything here is the constant-product formula; what this file cannot model is whether the
    attacker gets the ordering (see the printed caveat).
    """
    if not 0.0 < tolerance < 1.0 or dx_victim <= 0:
        raise ValueError("tolerance must be a fraction in (0, 1) and victim size positive")
    quoted = cp_swap(x, y, dx_victim, fee)["dy"]
    floor = quoted * (1.0 - tolerance)

    def victim_out(dx_a: float) -> float:
        a = cp_swap(x, y, dx_a, fee)
        return cp_swap(a["x1"], a["y1"], dx_victim, fee)["dy"]

    lo, hi = 0.0, dx_victim
    while victim_out(hi) > floor and hi < 1e6 * dx_victim:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if victim_out(mid) > floor:
            lo = mid
        else:
            hi = mid
    dx_a = 0.5 * (lo + hi)

    a = cp_swap(x, y, dx_a, fee)
    v = cp_swap(a["x1"], a["y1"], dx_victim, fee)
    back = cp_swap(v["y1"], v["x1"], a["dy"], fee)             # sell the Y back for X
    profit = back["dy"] - dx_a
    return {"front_run": dx_a, "front_run_over_victim": dx_a / dx_victim,
            "victim_quoted": quoted, "victim_filled": v["dy"],
            "victim_shortfall": v["dy"] / quoted - 1.0, "tolerance": tolerance,
            "attacker_profit_x": profit, "profit_over_victim_size": profit / dx_victim,
            "capture_of_tolerance": profit / dx_victim / tolerance}


THE_RULE = "RULE: verify pool accounting exactly; estimate LP fees and realized execution from actual inputs."


def main() -> None:                                          # noqa: C901 - a printed report
    print("=" * W)
    print("AMM MECHANICS - constant product, concentrated liquidity, and what an LP is short")
    print("=" * W)

    # ---- 1. the invariant
    print("\n1. x*y = k, AND PRICE IMPACT IS A CLOSED FORM")
    x0 = y0 = 1_000_000.0
    print(f"   {'trade / reserve':>16}{'exec vs spot':>14}{'pool moves':>13}"
          f"{'exec = half of pool (log)':>27}")
    for u in (0.001, 0.01, 0.05, 0.10, 0.50):
        c = cp_impact(u)
        half = np.log(c["exec_ratio"]) / np.log(c["pool_ratio"])
        print(f"   {u:>16.1%}{c['slippage']:>14.4%}{c['pool_move']:>13.4%}{half:>27.6f}")
    exact = cp_swap(x0, y0, 50_000.0, fee=0.0)["dy"]
    sliced = swap_by_slices(x0, y0, 50_000.0, slices=200_000, fee=0.0)
    print(f"   closed form vs walking the curve in 200,000 slices, dx = 5% of the reserve:")
    print(f"   {exact:.10f} vs {sliced:.10f}   |diff| {abs(exact - sliced):.2e} "
          f"({abs(exact - sliced) / exact:.1e} relative)")
    print("   ! With ZERO fees and no intervening flow, one trade and 200,000 slices have the")
    print("     SAME fill. With fees retained in reserves, splitting changes the result; this is")
    print("     not a claim about live routing or execution schedules.")

    # ---- 2. fees
    print("\n2. FEES - 30 bp on the input is not 30 bp on the output")
    num, den = V2["withdrawn_fee_fraction"]
    print(f"   Source-verified (Uniswap v2 Core, sec 1 and 2.4): a {V2['fee']:.2%} fee on the "
          f"INPUT, of which")
    print(f"   {V2['lp_share']:.2%} of the amount traded goes to LPs and {V2['protocol_share']:.2%}"
          f" to the protocol when the {V2['protocol_cut']:.3f} cut is on.")
    print(f"   Relative to the WITHDRAWN amount the same fee is {num:.0f}/{den:.0f} = "
          f"{fee_on_withdrawn():.6%}, not {V2['fee']:.4%}.")
    tape = simulate_tape()
    print(f"   Fee accumulator, v2 eq (4): f = 1 - sqrt(k0)/sqrt(k1) over {tape['trades']:,} "
          f"seeded trades")
    print(f"   k grew {tape['k1'] / tape['k0'] - 1:.4%}; eq (4) reads "
          f"{tape['fee_growth']:.4%} of the pool, and sqrt(k) grew "
          f"{tape['sqrt_k_growth']:.4%}")
    print(f"   identity 1 - sqrt(k0/k1) vs sqrt(k1/k0) - 1: "
          f"{abs(tape['fee_growth'] - tape['sqrt_k_growth'] / (1 + tape['sqrt_k_growth'])):.2e}")
    print(f"   Historical v3 whitepaper initial fee tiers: "
          f"{', '.join(f'{t:.2%}' for t in V3['fee_tiers'])}.")
    print("   And (v3 sec 3.2.1) v3 fees are held separately in the tokens they were paid in --")
    print("   a v2 position's fees compound into its liquidity, a v3 position's do NOT.")

    # ---- 3. impermanent loss
    print("\n3. IMPERMANENT LOSS - the closed form, reproduced from the reserves")
    print(f"   {'price ratio':>12}{'closed form':>14}{'from reserves':>16}{'|diff|':>11}")
    worst = 0.0
    for r in (0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0):
        a, b = il_closed_form(r), il_from_reserves(1_000_000.0, 1_000_000.0, r)
        worst = max(worst, abs(a - b))
        print(f"   {r:>12.2f}{a:>14.4%}{b:>16.4%}{abs(a - b):>11.2e}")
    print(f"   worst |difference| over the table: {worst:.2e} -- IL = 2*sqrt(r)/(1+r) - 1 is an")
    print("   identity, not a fit, and it is symmetric in r and 1/r: a halving and a doubling")
    print(f"   both cost {il_closed_form(2.0):.4%}.")

    # ---- 4. IL against fees
    print("\n4. DO FEES COVER IT? THE BREAK-EVEN IS A NUMBER, NOT AN OPINION")
    print(f"   30 days, {V2['fee']:.2%} fee, 20,000 seeded terminal prices per row:")
    print(f"   {'annual vol':>11}{'mean IL':>10}{'median IL':>11}{'worst decile':>14}"
          f"{'break-even daily volume / TVL':>31}")
    rows = []
    for v in (0.30, 0.60, 1.00, 1.50):
        f = il_vs_fees(v)
        rows.append(f)
        print(f"   {v:>11.0%}{f['mean_il']:>10.3%}{f['median_il']:>11.3%}"
              f"{f['worst_decile_il']:>14.3%}{f['breakeven_daily_vol_to_tvl']:>31.2%}")
    print("   ! For small moves IL grows approximately with variance; the toy fee estimate is")
    print("     linear in assumed volume. These break-even ratios use mean IL and fixed TVL,")
    print("     not observed volume, joint price-flow dynamics, gas or protocol fees.")
    print("   ! The median is far kinder than the mean: IL is a left tail, and a pool that looks")
    print("     fine most months is the same pool as the one that does not.")

    # ---- 5. concentrated liquidity
    print("\n5. AN IN-RANGE POSITION AMPLIFIES LOSS; FEES ALSO DEPEND ON FLOW AND COMPETING LIQUIDITY")
    print(f"   Source-verified (v3 sec 2): {V3['liquidity']}, real reserves from eq (2.2), and")
    print(f"   out of range the position has {V3['out_of_range']}.")
    print(f"   {'range':>18}{'capital x':>11}{'in range':>10}{'fee mult':>10}"
          f"{'loss +20%':>11}{'x full':>8}{'x full at +2%':>15}")
    full20, full02 = il_closed_form(1.2), il_closed_form(1.02)
    for m in (1.05, 1.10, 1.25, 2.00, 4.00):
        eff = capital_efficiency(m)
        tir = time_in_range(m, 0.60)
        loss = cl_loss_vs_hold(m, 1.2)
        small = cl_loss_vs_hold(m, 1.02) / full02
        print(f"   {f'[1/{m:.2f}, {m:.2f}]':>18}{eff:>11.1f}{tir:>10.0%}{eff * tir:>10.1f}"
              f"{loss:>11.2%}{loss / full20:>8.1f}{small:>15.1f}")
    print(f"   (the full-range position loses {full20:.2%} at +20% and {full02:.4%} at +2%)")
    print("   The last column is the point: for a move small enough to stay inside the range the")
    print("   loss amplification IS the capital multiple, to the decimal.")
    print("   ! The capital multiple is 1/(1 - 1/sqrt(m)). It exactly scales in-range divergence")
    print("     loss against the initial basket. Fee scaling needs assumptions about flow. Once")
    print("     price leaves it the position is 100% in one asset and earns nothing at all.")
    print("   ! 'Fee multiple' is capital x time in range, an illustration that assumes")
    print("     fee flow per unit of active liquidity is unchanged. It is not a proven bound;")
    print("     pool liquidity and fee-generating volume change with the price.")

    # ---- 6. what cannot be measured here
    print("\n6. THE FILL IS NOT A CLOSED FORM - gas, ordering, and your own slippage tolerance")
    print(f"   {'tolerance':>11}{'front-run / victim':>20}{'victim shortfall':>18}"
          f"{'attacker take':>15}{'of tolerance':>14}")
    for tol in (0.001, 0.005, 0.01, 0.05):
        s = sandwich(1_000_000.0, 1_000_000.0, 20_000.0, tol)
        print(f"   {tol:>11.2%}{s['front_run_over_victim']:>20.2f}{s['victim_shortfall']:>18.2%}"
              f"{s['profit_over_victim_size']:>15.3%}{s['capture_of_tolerance']:>14.0%}")
    print("   ! This example solves the front-run size that pushes output to the stated limit.")
    print("     It does not optimize attacker profit or include gas and ordering competition.")
    print("     The fraction captured depends on the pool, victim size and fee.")
    print("   NOT MEASURABLE OFFLINE from these synthetic inputs:")
    print("     - gas: the fee is denominated in the chain's own token and moves with congestion")
    print("     - ordering: whether the attacker gets the slot is an auction, not arithmetic")
    print("     - frequency: historical transactions and traces can identify many sandwiches")
    print("     - routing: a real fill is split across pools and versions by an aggregator")
    print("   Historical gas, transaction traces and routing data can support a richer replay;")
    print("   this offline model has none of those inputs and cannot estimate realized fills.")

    print("\n" + "=" * W)
    print(THE_RULE)
    print("=" * W)


if __name__ == "__main__":
    main()
