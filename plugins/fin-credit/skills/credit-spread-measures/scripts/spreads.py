"""Credit spread measures -- G, I, Z, asset-swap, discount margin and OAS -- and what each one
is measured AGAINST.

WHY this exists: "the spread" is six different numbers on the same bond and a quote almost never
says which one it is. The differences are small enough to look like rounding and large enough to
lose the trade.

  * G-spread   bond YTM minus the government PAR (benchmark) yield at the same maturity.
               THE TRAP: subtracting the government ZERO rate instead gives a DIFFERENT number,
               because a par yield and a zero rate are not the same thing on a sloped curve.
  * I-spread   bond YTM minus the interpolated par SWAP rate at the same maturity.
  * Z-spread   the parallel add-on to the whole zero curve that reprices the bond exactly.
               Unlike G and I it uses every point of the curve, not one point.
  * ASW        the par/par asset-swap spread: the margin over the floating index that turns the
               bond into a par floater. It is (par - price) amortised over the swap annuity, so
               it depends on how far the PRICE is from 100 in a way the Z-spread does not.
  * DM         discount margin, the floating-rate cousin of the Z-spread. It equals the quoted
               margin only when the FRN trades at par.
  * OAS        the spread over a calibrated short-rate lattice with the embedded option exercised
               optimally. On a BULLET bond the OAS is the Z-spread. On a CALLABLE bond the gap
               between the two IS the option cost -- a Z-spread quoted on a callable is not an OAS.

Everything is closed form or a small recombining lattice; numpy and scipy only, no network, no
files, fixed inputs. QuantLib, if importable, is used only as an independent cross-check inside
`quantlib_cross_check`.

Usage:
    from spreads import g_spread_trap, spread_table, oas_vs_zspread
    g_spread_trap()["naive_bp"]        # 143.48 -- the wrong one
    oas_vs_zspread()["option_cost_bp"]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------- the fixed curves
# A deliberately STEEP government zero curve, annual compounding, t = 1 .. 10 years. The 1-5y
# section is what makes the G-spread trap visible: par and zero diverge fast on a steep curve.
GOVT_ZEROS_PCT = (3.00, 3.50, 4.00, 4.50, 5.00, 5.20, 5.35, 5.45, 5.50, 5.55)
# Swap zero curve = government + a widening swap spread, in basis points, same tenors.
SWAP_SPREAD_BP = (10.0, 14.0, 18.0, 22.0, 26.0, 28.0, 30.0, 31.0, 32.0, 33.0)

# The reference corporate bond: 5 years, 3% annual coupon, priced at a 150 bp Z-spread.
BOND_COUPON = 3.0
BOND_MATURITY = 5
TRUE_Z_BP = 150.0


def govt_zeros(n: int = 10) -> np.ndarray:
    """Government zero rates as decimals for t = 1 .. n."""
    return np.array(GOVT_ZEROS_PCT[:n]) / 100.0


def swap_zeros(n: int = 10) -> np.ndarray:
    """Swap zero rates as decimals for t = 1 .. n."""
    return govt_zeros(n) + np.array(SWAP_SPREAD_BP[:n]) / 1e4


# --------------------------------------------------------------------------- curve arithmetic
def discount_factors(zeros: Sequence[float], spread: float = 0.0) -> np.ndarray:
    """Annual-compounded discount factors for t = 1 .. len(zeros), with a parallel `spread`."""
    z = np.asarray(zeros, dtype=float) + spread
    t = np.arange(1, len(z) + 1, dtype=float)
    if np.any(z <= -1.0):
        raise ValueError("a zero rate at or below -100% has no discount factor")
    return (1.0 + z) ** (-t)


def bond_price(coupon: float, maturity: int, zeros: Sequence[float], spread: float = 0.0,
               redemption: float = 100.0) -> float:
    """Price per 100 of an annual-coupon bullet bond off a zero curve plus `spread`."""
    if maturity < 1 or maturity > len(zeros):
        raise ValueError(f"maturity {maturity} is outside the curve (1..{len(zeros)})")
    df = discount_factors(zeros, spread)[:maturity]
    cf = np.full(maturity, coupon, dtype=float)
    cf[-1] += redemption
    return float(cf @ df)


def price_from_yield(coupon: float, maturity: int, y: float, redemption: float = 100.0) -> float:
    """Price per 100 from a single annual-compounded yield."""
    t = np.arange(1, maturity + 1, dtype=float)
    cf = np.full(maturity, coupon, dtype=float)
    cf[-1] += redemption
    return float(cf @ (1.0 + y) ** (-t))


def yield_to_maturity(price: float, coupon: float, maturity: int,
                      redemption: float = 100.0) -> float:
    """Annual-compounded YTM. brentq over a wide bracket, so it fails loudly rather than drifting."""
    if price <= 0.0:
        raise ValueError("price must be positive")
    f = lambda y: price_from_yield(coupon, maturity, y, redemption) - price   # noqa: E731
    return float(brentq(f, -0.90, 5.0, xtol=1e-14, rtol=1e-15, maxiter=200))


def par_yields(zeros: Sequence[float]) -> np.ndarray:
    """Par (benchmark) yield at each maturity: the annual coupon that prices the bond at 100."""
    df = discount_factors(zeros)
    ann = np.cumsum(df)
    return (1.0 - df) / ann


def z_spread(price: float, coupon: float, maturity: int, zeros: Sequence[float],
             redemption: float = 100.0) -> float:
    """The parallel add-on to the zero curve that reprices the bond. Decimal, not basis points."""
    f = lambda s: bond_price(coupon, maturity, zeros, s, redemption) - price   # noqa: E731
    return float(brentq(f, -0.20, 2.0, xtol=1e-14, rtol=1e-15, maxiter=200))


def asset_swap_spread(price_dirty: float, coupon: float, maturity: int,
                      zeros: Sequence[float], redemption: float = 100.0) -> float:
    """Par/par asset-swap spread over the floating index, off the SWAP zero curve.

    The package costs 100; the bond costs `price_dirty`. The swap absorbs the difference, so
    the spread is (PV of the bond's fixed flows on the swap curve - price) / swap annuity.
    """
    df = discount_factors(zeros)[:maturity]
    pv = bond_price(coupon, maturity, zeros, 0.0, redemption)
    annuity = float(np.sum(df))
    return (pv - price_dirty) / annuity / 100.0


# --------------------------------------------------------------------------- the G-spread trap
@dataclass(frozen=True)
class SpreadSet:
    price: float
    ytm: float
    govt_zero: float
    govt_par: float
    swap_par: float
    g_spread_bp: float
    naive_bp: float
    i_spread_bp: float
    z_spread_bp: float
    z_over_swap_bp: float
    asw_bp: float


def spread_table(coupon: float = BOND_COUPON, maturity: int = BOND_MATURITY,
                 z_bp: float = TRUE_Z_BP) -> SpreadSet:
    """Every spread measure on ONE bond, built from a known Z-spread so the price is exact."""
    gz, sz = govt_zeros(), swap_zeros()
    price = bond_price(coupon, maturity, gz, z_bp / 1e4)
    y = yield_to_maturity(price, coupon, maturity)
    gpar = float(par_yields(gz)[maturity - 1])
    spar = float(par_yields(sz)[maturity - 1])
    gzero = float(gz[maturity - 1])
    return SpreadSet(
        price=price, ytm=y, govt_zero=gzero, govt_par=gpar, swap_par=spar,
        g_spread_bp=(y - gpar) * 1e4,
        naive_bp=(y - gzero) * 1e4,
        i_spread_bp=(y - spar) * 1e4,
        z_spread_bp=z_spread(price, coupon, maturity, gz) * 1e4,
        z_over_swap_bp=z_spread(price, coupon, maturity, sz) * 1e4,
        asw_bp=asset_swap_spread(price, coupon, maturity, sz) * 1e4,
    )


def g_spread_trap(coupon: float = BOND_COUPON, maturity: int = BOND_MATURITY,
                  z_bp: float = TRUE_Z_BP) -> dict[str, float]:
    """The three numbers a desk would all call 'the spread' on the same bond."""
    s = spread_table(coupon, maturity, z_bp)
    return {"price": s.price, "ytm_pct": s.ytm * 100.0,
            "g_spread_bp": s.g_spread_bp, "naive_bp": s.naive_bp, "z_spread_bp": s.z_spread_bp,
            "naive_error_bp": s.naive_bp - s.g_spread_bp,
            "z_minus_g_bp": s.z_spread_bp - s.g_spread_bp}


def slope_sensitivity(slopes_bp: Sequence[float] = (0.0, 25.0, 50.0, 75.0, 100.0),
                      base_pct: float = 5.0, coupon: float = BOND_COUPON,
                      maturity: int = BOND_MATURITY,
                      z_bp: float = TRUE_Z_BP) -> list[dict[str, float]]:
    """How the G-spread trap scales with curve slope: zeros = base + slope * (t - maturity)/1e4.

    The 5-year zero is pinned at `base_pct` in every case, so only the SHAPE changes.
    """
    out = []
    for sl in slopes_bp:
        t = np.arange(1, maturity + 1, dtype=float)
        z = base_pct / 100.0 + sl / 1e4 * (t - maturity)
        price = bond_price(coupon, maturity, z, z_bp / 1e4)
        y = yield_to_maturity(price, coupon, maturity)
        gpar = float(par_yields(z)[maturity - 1])
        out.append({"slope_bp_per_year": sl, "govt_par_pct": gpar * 100.0,
                    "g_spread_bp": (y - gpar) * 1e4,
                    "naive_bp": (y - z[maturity - 1]) * 1e4,
                    "error_bp": (y - z[maturity - 1]) * 1e4 - (y - gpar) * 1e4})
    return out


# --------------------------------------------------------------------------- discount margin
def frn_price(index_rate: float, margin: float, dm: float, maturity: float,
              freq: int = 4, redemption: float = 100.0) -> float:
    """Price of a floater whose forward index is flat at `index_rate`, discounted at index + dm."""
    n = int(round(maturity * freq))
    tau = 1.0 / freq
    disc = 1.0
    pv = 0.0
    for i in range(1, n + 1):
        disc /= 1.0 + (index_rate + dm) * tau
        cf = (index_rate + margin) * tau * 100.0
        if i == n:
            cf += redemption
        pv += cf * disc
    return pv


def discount_margin(price: float, index_rate: float, margin: float, maturity: float,
                    freq: int = 4, redemption: float = 100.0) -> float:
    """Solve for the discount margin that reprices the floater. Decimal, not basis points."""
    f = lambda d: frn_price(index_rate, margin, d, maturity, freq, redemption) - price  # noqa: E731
    return float(brentq(f, -0.20, 2.0, xtol=1e-14, rtol=1e-15, maxiter=200))


def dm_table(prices: Sequence[float] = (102.0, 100.0, 98.5, 95.0),
             index_rate: float = 0.04, margin: float = 0.0120, maturity: float = 5.0,
             freq: int = 4) -> list[dict[str, float]]:
    """Quoted margin vs discount margin vs the straight-line approximation, by price."""
    out = []
    for p in prices:
        dm = discount_margin(p, index_rate, margin, maturity, freq)
        approx = margin + (100.0 - p) / 100.0 / maturity
        out.append({"price": p, "quoted_margin_bp": margin * 1e4, "dm_bp": dm * 1e4,
                    "approx_bp": approx * 1e4, "approx_error_bp": (approx - dm) * 1e4})
    return out


# --------------------------------------------------------------------------- the OAS lattice
def calibrate_tree(zeros: Sequence[float], sigma: float) -> list[np.ndarray]:
    """Lognormal one-factor short-rate lattice, annual steps, 50/50 branching, forward induction.

    r(i, j) = a_i * exp(2 j sigma), j = 0 .. i. Each a_i is solved so the lattice reprices the
    zero-coupon bond maturing at i + 1 EXACTLY -- that repricing is the only test a calibration
    needs, and it is asserted in the demo.
    """
    z = np.asarray(zeros, dtype=float)
    n = len(z)
    target = discount_factors(z)
    rates: list[np.ndarray] = []
    q = np.array([1.0])                       # Arrow-Debreu prices at step 0
    for i in range(n):
        mult = np.exp(2.0 * sigma * np.arange(i + 1))

        def df_at(a: float, _m=mult, _q=q) -> float:
            return float(np.sum(_q / (1.0 + a * _m)))

        a = brentq(lambda x: df_at(x) - target[i], 1e-8, 5.0, xtol=1e-15, rtol=1e-15, maxiter=300)
        r = a * mult
        rates.append(r)
        nxt = np.zeros(i + 2)
        step = q / (1.0 + r)
        nxt[:-1] += 0.5 * step
        nxt[1:] += 0.5 * step
        q = nxt
    return rates


def lattice_bond_price(rates: list[np.ndarray], coupon: float, maturity: int, spread: float,
                       call_price: float | None = None, first_call: int | None = None,
                       redemption: float = 100.0) -> float:
    """Roll a bullet or callable annual-coupon bond back through the lattice at `spread`.

    At a call date the issuer pays `call_price` plus the coupon, so the decision is taken on the
    EX-COUPON continuation value. `first_call=None` (or `call_price=None`) prices the bullet.
    """
    if maturity > len(rates):
        raise ValueError("the lattice is shorter than the bond")
    v = np.full(maturity + 1, redemption + coupon, dtype=float)
    for i in range(maturity - 1, -1, -1):
        r = rates[i] + spread
        v = 0.5 * (v[1:] + v[:-1]) / (1.0 + r)
        if i > 0:
            if call_price is not None and first_call is not None and i >= first_call:
                v = np.minimum(v, call_price)
            v = v + coupon
    return float(v[0])


def lattice_oas(price: float, rates: list[np.ndarray], coupon: float, maturity: int,
                call_price: float | None = None, first_call: int | None = None,
                redemption: float = 100.0) -> float:
    """Solve for the constant spread over every lattice node that reproduces `price`."""
    def f(s: float) -> float:
        return lattice_bond_price(rates, coupon, maturity, s, call_price, first_call,
                                  redemption) - price
    return float(brentq(f, -0.20, 1.0, xtol=1e-14, rtol=1e-15, maxiter=300))


def oas_vs_zspread(coupon: float = 6.5, maturity: int = 10, call_price: float = 100.0,
                   first_call: int = 5, true_oas_bp: float = 80.0,
                   sigma: float = 0.20) -> dict[str, float]:
    """One callable bond, both numbers. The gap between them is the option cost."""
    gz = govt_zeros(maturity)
    rates = calibrate_tree(gz, sigma)
    price = lattice_bond_price(rates, coupon, maturity, true_oas_bp / 1e4,
                               call_price, first_call)
    bullet = lattice_bond_price(rates, coupon, maturity, true_oas_bp / 1e4)
    zs = z_spread(price, coupon, maturity, gz) * 1e4
    oas = lattice_oas(price, rates, coupon, maturity, call_price, first_call) * 1e4
    # the identity check: on the SAME bond without the call, OAS must be the Z-spread
    z_bullet = z_spread(bullet, coupon, maturity, gz) * 1e4
    oas_bullet = lattice_oas(bullet, rates, coupon, maturity) * 1e4
    return {"callable_price": price, "bullet_price": bullet,
            "z_spread_bp": zs, "oas_bp": oas, "option_cost_bp": zs - oas,
            "option_value_points": bullet - price,
            "bullet_z_bp": z_bullet, "bullet_oas_bp": oas_bullet,
            "bullet_gap_bp": oas_bullet - z_bullet}


def option_cost_by_vol(vols: Sequence[float] = (0.0, 0.10, 0.20, 0.30),
                       coupon: float = 6.5, maturity: int = 10, call_price: float = 100.0,
                       first_call: int = 5, true_oas_bp: float = 80.0
                       ) -> list[dict[str, float]]:
    """The option cost is a function of the vol you assumed. That is the second OAS trap."""
    gz = govt_zeros(maturity)
    out = []
    for s in vols:
        rates = calibrate_tree(gz, s)
        price = lattice_bond_price(rates, coupon, maturity, true_oas_bp / 1e4,
                                   call_price, first_call)
        zs = z_spread(price, coupon, maturity, gz) * 1e4
        oas = lattice_oas(price, rates, coupon, maturity, call_price, first_call) * 1e4
        out.append({"sigma": s, "price": price, "z_spread_bp": zs, "oas_bp": oas,
                    "option_cost_bp": zs - oas})
    return out


def oas_repricing_error(sigma: float = 0.20, maturity: int = 10) -> float:
    """Max |lattice zero-bond price - curve discount factor| over the calibration. Should be ~0."""
    gz = govt_zeros(maturity)
    rates = calibrate_tree(gz, sigma)
    target = discount_factors(gz)
    worst = 0.0
    for m in range(1, maturity + 1):
        got = lattice_bond_price(rates, 0.0, m, 0.0, redemption=1.0)
        worst = max(worst, abs(got - target[m - 1]))
    return worst


# --------------------------------------------------------------------------- QuantLib check
def quantlib_cross_check(coupon: float = BOND_COUPON, maturity: int = BOND_MATURITY,
                         z_bp: float = TRUE_Z_BP) -> dict[str, float] | None:
    """Independent Z-spread from QuantLib's BondFunctions.zSpread, or None if it is absent."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    today = ql.Date(9, 9, 2026)
    ql.Settings.instance().evaluationDate = today          # a stale global gives NPV exactly 0.0
    cal, dc = ql.NullCalendar(), ql.Thirty360(ql.Thirty360.BondBasis)
    gz = govt_zeros(maturity)
    dates = [today] + [today + ql.Period(int(t), ql.Years) for t in range(1, maturity + 1)]
    dfs = [1.0] + list(discount_factors(gz)[:maturity])
    curve = ql.DiscountCurve(dates, dfs, dc, cal)
    curve.enableExtrapolation()
    sched = ql.Schedule(today, today + ql.Period(maturity, ql.Years), ql.Period(ql.Annual),
                        cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    bond = ql.FixedRateBond(0, 100.0, sched, [coupon / 100.0], dc)
    price = bond_price(coupon, maturity, gz, z_bp / 1e4)
    # QuantLib 1.43 takes a BondPrice, not a bare float: the clean/dirty flag is part of the type.
    quote = ql.BondPrice(price, ql.BondPrice.Clean)
    zs = ql.BondFunctions.zSpread(bond, quote, curve, dc, ql.Compounded, ql.Annual, today)
    y = bond.bondYield(quote, dc, ql.Compounded, ql.Annual)
    return {"ql_version": ql.__version__, "ql_z_spread_bp": zs * 1e4, "ql_ytm_pct": y * 100.0,
            "mine_z_spread_bp": z_spread(price, coupon, maturity, gz) * 1e4,
            "mine_ytm_pct": yield_to_maturity(price, coupon, maturity) * 100.0}


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    W = 96
    print("=" * W)
    print("CREDIT SPREAD MEASURES -- G, I, Z, ASW, DM and OAS on the same bonds")
    print("=" * W)

    s = spread_table()
    print(f"\n1. ONE BOND, SIX SPREADS  ({BOND_MATURITY}y {BOND_COUPON:.1f}% annual corporate, "
          f"priced at a {TRUE_Z_BP:.0f} bp Z-spread)")
    print(f"   price {s.price:.6f}   YTM {s.ytm * 100:.6f}%")
    print(f"   government 5y ZERO {s.govt_zero * 100:.6f}%   government 5y PAR "
          f"{s.govt_par * 100:.6f}%   swap 5y PAR {s.swap_par * 100:.6f}%")
    print(f"   {'measure':<34}{'value bp':>12}  measured against")
    rows = [("G-spread (YTM - govt PAR)", s.g_spread_bp, "one point of the government par curve"),
            ("'YTM - govt ZERO'  <-- WRONG", s.naive_bp, "one point of the government ZERO curve"),
            ("I-spread (YTM - par swap)", s.i_spread_bp, "one point of the par swap curve"),
            ("Z-spread (over govt zeros)", s.z_spread_bp, "the whole government zero curve"),
            ("Z-spread (over swap zeros)", s.z_over_swap_bp, "the whole swap zero curve"),
            ("ASW par/par (over swap)", s.asw_bp, "the swap annuity, amortising 100 - price")]
    for label, val, against in rows:
        print(f"   {label:<34}{val:>12.2f}  {against}")
    print(f"   -> the G-spread trap: {s.naive_bp:.2f} bp against the ZERO rate is "
          f"{s.naive_bp - s.g_spread_bp:+.2f} bp from the")
    print(f"      G-spread {s.g_spread_bp:.2f} bp. Same bond, same day, no error raised.")
    print(f"   -> ASW {s.asw_bp:.2f} bp vs Z-over-swap {s.z_over_swap_bp:.2f} bp = "
          f"{s.asw_bp - s.z_over_swap_bp:+.2f} bp, because the bond is "
          f"{100 - s.price:.2f} points below par.")

    print("\n2. THE TRAP SCALES WITH THE SLOPE (5y zero pinned at 5.00% in every row)")
    print(f"   {'slope bp/yr':>12}{'govt 5y par %':>16}{'G-spread bp':>14}"
          f"{'YTM - zero bp':>16}{'error bp':>11}")
    for row in slope_sensitivity():
        print(f"   {row['slope_bp_per_year']:>12.0f}{row['govt_par_pct']:>16.4f}"
              f"{row['g_spread_bp']:>14.2f}{row['naive_bp']:>16.2f}{row['error_bp']:>+11.2f}")
    print("   On a FLAT curve the two agree exactly. The error is the curve, not the credit.")

    print("\n3. DISCOUNT MARGIN: the quoted margin is the DM only at par")
    print("   5y quarterly FRN, forward index flat at 4.00%, quoted margin 120 bp")
    print(f"   {'price':>8}{'quoted bp':>12}{'DM bp':>10}{'DM - quoted':>14}"
          f"{'straight-line bp':>19}{'its error bp':>14}")
    for row in dm_table():
        print(f"   {row['price']:>8.2f}{row['quoted_margin_bp']:>12.2f}{row['dm_bp']:>10.2f}"
              f"{row['dm_bp'] - row['quoted_margin_bp']:>+14.2f}{row['approx_bp']:>19.2f}"
              f"{row['approx_error_bp']:>+14.2f}")

    print("\n4. A Z-SPREAD ON A CALLABLE BOND IS NOT AN OAS")
    err = oas_repricing_error()
    print(f"   lattice: 10 annual steps, lognormal, sigma = 20%. Calibration repricing error "
          f"{err:.2e}")
    o = oas_vs_zspread()
    print(f"   10y 6.5% corporate, callable at 100 from year 5")
    print(f"   bullet price {o['bullet_price']:.6f}   callable price {o['callable_price']:.6f}"
          f"   call option worth {o['option_value_points']:.6f} points")
    print(f"   Z-spread {o['z_spread_bp']:.2f} bp   OAS {o['oas_bp']:.2f} bp   "
          f"OPTION COST {o['option_cost_bp']:.2f} bp")
    print(f"   identity check on the SAME bond with the call removed: Z {o['bullet_z_bp']:.4f} bp "
          f"vs OAS {o['bullet_oas_bp']:.4f} bp, gap {o['bullet_gap_bp']:+.4f} bp")
    print(f"   {'sigma':>8}{'price':>12}{'Z-spread bp':>14}{'OAS bp':>10}{'option cost bp':>17}")
    for row in option_cost_by_vol():
        print(f"   {row['sigma']:>8.0%}{row['price']:>12.6f}{row['z_spread_bp']:>14.2f}"
              f"{row['oas_bp']:>10.2f}{row['option_cost_bp']:>17.2f}")
    print("   The OAS is the only one of the six that depends on a VOLATILITY you chose.")

    q = quantlib_cross_check()
    print("\n5. QUANTLIB CROSS-CHECK")
    if q is None:
        print("   QuantLib is not installed -- every number above is still produced by this file.")
    else:
        print(f"   QuantLib {q['ql_version']}  BondFunctions.zSpread = "
              f"{q['ql_z_spread_bp']:.6f} bp   this file {q['mine_z_spread_bp']:.6f} bp   "
              f"diff {q['ql_z_spread_bp'] - q['mine_z_spread_bp']:+.2e} bp")
        print(f"   QuantLib bondYield = {q['ql_ytm_pct']:.6f}%   this file "
              f"{q['mine_ytm_pct']:.6f}%   diff "
              f"{q['ql_ytm_pct'] - q['mine_ytm_pct']:+.2e} pp")

    print("\n" + "=" * W)
    print("THE RULE: a spread is a pair -- the number AND the curve it was measured against. "
          "'YTM minus")
    print("the government zero' is not the G-spread, and a Z-spread on a callable bond is not "
          "an OAS.")
    print("=" * W)
