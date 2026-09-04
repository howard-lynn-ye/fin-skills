"""Option-Greek unit normaliser. Two libraries price the same option and disagree by 100x.

WHY this exists — the numbers below are verified library behaviour, not folklore:

  * `py_vollib` / `py_lets_be_rational` report **vega per 1 VOL POINT** (a 0.01 move in
    sigma), **theta per CALENDAR DAY** (annual theta / 365) and **rho per 1% RATE** (a
    0.01 move in r).
  * `QuantLib` (and `financepy`) report **vega and rho in ABSOLUTE units** (per 1.00 of
    vol, per 1.00 of rate) and **theta ANNUALISED** (per year).

So for one plain ATM call the two libraries print vega 0.375 and 37.524, theta -0.0176
and -6.414, rho 0.532 and 53.232. Price, delta and gamma are IDENTICAL in both. That is
exactly why the bug survives review: you spot-check three of the six numbers, they agree,
you ship. Then the vega P&L on a one-vol-point move is off by a factor of 100 and the
theta decay on an overnight hold is off by 365.

Nothing raises. Nothing warns. The greek dict has no units attached, so a risk report
that mixes a vollib vega with a QuantLib rho is arithmetically valid and completely wrong.

The fix is to (1) tag every greek vector with the convention it came from, (2) convert
through one canonical absolute convention, and (3) never hand-multiply a greek by a bump
— pass absolute bumps to `pnl()` and let it scale them into the greek's own units.

Usage:
    from greeks_convention import CONVENTIONS, convert, sanity_check, pnl

    g_ql = {"price": 10.45, "delta": 0.637, "gamma": 0.0188,
            "vega": 37.524, "theta": -6.414, "rho": 53.232}
    g_vl = convert(g_ql, "quantlib", "vollib")     # -> vega 0.375, theta -0.0176
    sanity_check(g_vl, flag="c", moneyness=1.0)    # raises if a sign is impossible
    pnl(g_ql, "quantlib", d_vol=0.01, d_years=1 / 365)   # bumps are ALWAYS absolute
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

Greeks = dict[str, float]

# Greeks whose numeric value depends on the reporting convention.
SCALED_GREEKS: tuple[str, ...] = ("vega", "theta", "rho")
# Greeks that are convention-free across QuantLib / vollib / financepy. These are the
# ones that agree, which is precisely why a mismatch passes a casual eyeball check.
UNSCALED_GREEKS: tuple[str, ...] = ("price", "delta", "gamma")


@dataclass(frozen=True)
class Convention:
    """How one library scales vega / theta / rho relative to absolute units.

    `*_scale` multiplies an ABSOLUTE greek to produce this library's printed number:
        vega_absolute * vega_scale = vega_as_this_library_prints_it
    """

    name: str
    vega_scale: float
    theta_scale: float
    rho_scale: float
    vega_unit: str
    theta_unit: str
    rho_unit: str
    note: str = ""

    def scale_of(self, greek: str) -> float:
        return {"vega": self.vega_scale, "theta": self.theta_scale,
                "rho": self.rho_scale}[greek]

    def describe(self) -> str:
        return (f"{self.name:<28} vega: {self.vega_unit:<22} "
                f"theta: {self.theta_unit:<22} rho: {self.rho_unit}")


_ABSOLUTE = dict(
    vega_scale=1.0, theta_scale=1.0, rho_scale=1.0,
    vega_unit="per 1.00 vol (absolute)",
    theta_unit="per YEAR (annualised)",
    rho_unit="per 1.00 rate (absolute)",
)
# vollib divides vega by 100, theta by 365 and rho by 100 relative to absolute units.
_PER_POINT_PER_CALENDAR_DAY = dict(
    vega_scale=1.0 / 100.0, theta_scale=1.0 / 365.0, rho_scale=1.0 / 100.0,
    vega_unit="per 1 vol POINT (/100)",
    theta_unit="per CALENDAR day (/365)",
    rho_unit="per 1% rate (/100)",
)

CONVENTIONS: dict[str, Convention] = {
    "absolute": Convention(name="absolute", note="the canonical pivot", **_ABSOLUTE),
    "quantlib": Convention(
        name="quantlib",
        note="QuantLib AnalyticEuropeanEngine: vega/rho absolute, thetaPerDay() exists "
             "but theta() is annualised",
        **_ABSOLUTE),
    "financepy": Convention(
        name="financepy", note="financepy matches QuantLib's absolute/annualised units",
        **_ABSOLUTE),
    "vollib": Convention(
        name="vollib",
        note="py_vollib.black_scholes.greeks.analytical: vega/100, theta/365, rho/100",
        **_PER_POINT_PER_CALENDAR_DAY),
    "per_point_per_day": Convention(
        name="per_point_per_day", note="generic name for the vollib convention",
        **_PER_POINT_PER_CALENDAR_DAY),
    "per_point_per_trading_day": Convention(
        name="per_point_per_trading_day",
        vega_scale=1.0 / 100.0, theta_scale=1.0 / 252.0, rho_scale=1.0 / 100.0,
        vega_unit="per 1 vol POINT (/100)",
        theta_unit="per TRADING day (/252)",
        rho_unit="per 1% rate (/100)",
        note="risk systems that decay on the trading calendar; theta is 365/252 = 1.45x "
             "the calendar-day number for the SAME option"),
}


def get_convention(conv: str | Convention) -> Convention:
    if isinstance(conv, Convention):
        return conv
    try:
        return CONVENTIONS[conv.strip().lower()]
    except KeyError:
        raise KeyError(
            f"unknown greek convention {conv!r}. Known: {sorted(CONVENTIONS)}. "
            f"If your library is not listed, price one option with sigma bumped by 0.01 "
            f"and see whether its vega matches the price change (per-point) or 100x it "
            f"(absolute) -- do NOT guess."
        ) from None


def convert(greeks: Mapping[str, float], frm: str | Convention,
            to: str | Convention, strict: bool = True) -> Greeks:
    """Re-express a greek dict from one library's convention in another's.

    price/delta/gamma pass through untouched — they are the same number in every library
    here, which is why a mismatch is invisible until it hits P&L.

    `strict=True` (default) REFUSES unknown keys. Second-order vol greeks (vanna, volga,
    charm, veta) carry vol and/or time units too, and their scaling is NOT the same as
    vega's — silently passing them through would create a subtler version of this bug.
    """
    src, dst = get_convention(frm), get_convention(to)
    out: Greeks = {}
    for key, value in greeks.items():
        k = key.lower()
        if k in UNSCALED_GREEKS:
            out[key] = float(value)
        elif k in SCALED_GREEKS:
            absolute = float(value) / src.scale_of(k)
            out[key] = absolute * dst.scale_of(k)
        elif strict:
            raise KeyError(
                f"convert() does not know the units of {key!r}. Handled: "
                f"{UNSCALED_GREEKS + SCALED_GREEKS}. Cross greeks (vanna/volga/charm) "
                f"scale by a PRODUCT of vol and time factors, not by vega's factor; "
                f"convert them explicitly or pass strict=False to carry them unchanged "
                f"(and wrong)."
            )
        else:
            out[key] = value
    return out


def to_absolute(greeks: Mapping[str, float], frm: str | Convention,
                strict: bool = True) -> Greeks:
    """Canonical form: vega per 1.00 vol, theta per year, rho per 1.00 rate."""
    return convert(greeks, frm, "absolute", strict=strict)


def sanity_check(greeks: Mapping[str, float], flag: str,
                 moneyness: float | None = None,
                 *, raise_on_fail: bool = True) -> list[str]:
    """Assert the sign invariants a LONG vanilla option cannot violate.

    Hard invariants (violating one means a sign error, a short position mislabelled long,
    or greeks glued together from two different pricers):
        theta < 0, gamma > 0, vega > 0,
        call delta in [0, 1], put delta in [-1, 0],
        call rho > 0, put rho < 0.

    Known exception, deliberately NOT special-cased: a deep in-the-money EUROPEAN put on a
    non-dividend payer can have positive theta (the discounted intrinsic value grows as
    expiry nears). If you legitimately hit that, you are outside the regime this guard is
    for — check the sign by hand rather than deleting the assert.

    `moneyness` = S/K drives soft checks only; they are RETURNED as warnings, never raised,
    because dividends and rates can legitimately push delta across the naive boundary.

    Returns the list of soft warnings. Raises ValueError on any hard violation.
    """
    f = flag.strip().lower()[:1]
    if f not in ("c", "p"):
        raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")

    hard: list[str] = []
    g = {k.lower(): float(v) for k, v in greeks.items()}

    if "theta" in g and not g["theta"] < 0:
        hard.append(f"theta={g['theta']:+.6g} is not negative; a long option decays. "
                    f"Sign flip, or a short position labelled long.")
    if "gamma" in g and not g["gamma"] > 0:
        hard.append(f"gamma={g['gamma']:+.6g} is not positive; long convexity is positive.")
    if "vega" in g and not g["vega"] > 0:
        hard.append(f"vega={g['vega']:+.6g} is not positive; a long option is long vol.")
    if "delta" in g:
        d = g["delta"]
        if f == "c" and not (0.0 <= d <= 1.0):
            hard.append(f"call delta={d:+.6g} outside [0, 1].")
        if f == "p" and not (-1.0 <= d <= 0.0):
            hard.append(f"put delta={d:+.6g} outside [-1, 0]. A put delta reported as "
                        f"positive usually means the library returns |delta| or the "
                        f"call's delta.")
    if "rho" in g:
        if f == "c" and not g["rho"] > 0:
            hard.append(f"call rho={g['rho']:+.6g} is not positive.")
        if f == "p" and not g["rho"] < 0:
            hard.append(f"put rho={g['rho']:+.6g} is not negative.")

    if hard and raise_on_fail:
        raise ValueError("greek sanity check FAILED:\n  - " + "\n  - ".join(hard))

    soft: list[str] = list(hard) if not raise_on_fail else []
    if moneyness is not None and "delta" in g:
        d, m = g["delta"], float(moneyness)
        if f == "c" and m > 1.05 and d < 0.5:
            soft.append(f"ITM call (S/K={m:.3f}) with delta {d:.4f} < 0.5 — possible only "
                        f"with a large dividend yield or deeply negative carry.")
        if f == "c" and m < 0.95 and d > 0.5:
            soft.append(f"OTM call (S/K={m:.3f}) with delta {d:.4f} > 0.5 — check carry.")
        if f == "p" and m < 0.95 and d > -0.5:
            soft.append(f"ITM put (S/K={m:.3f}) with delta {d:.4f} > -0.5 — check carry.")
    return soft


def pnl(greeks: Mapping[str, float], conv: str | Convention, *,
        d_spot: float = 0.0, d_vol: float = 0.0, d_years: float = 0.0,
        d_rate: float = 0.0) -> dict[str, float]:
    """First-order P&L attribution. Bumps are ALWAYS in absolute units; the convention
    scaling is applied here so you never hand-multiply a greek by a bump again.

        d_spot  : currency move in the underlying
        d_vol   : ABSOLUTE vol move (0.01 == one vol point == 1%)
        d_years : ABSOLUTE time elapsed in YEARS (1/365 == one calendar day)
        d_rate  : ABSOLUTE rate move (0.0001 == one basis point)

    A greek reported in convention units already contains its own divisor, so the matching
    bump is `bump / scale` — e.g. a vollib vega (scale 1/100) pairs with d_vol/0.01 = "how
    many vol points", and a QuantLib vega (scale 1.0) pairs with d_vol itself. Getting this
    pairing wrong by hand is the 100x error.
    """
    c = get_convention(conv)
    g = {k.lower(): float(v) for k, v in greeks.items()}
    delta_pnl = g.get("delta", 0.0) * d_spot
    gamma_pnl = 0.5 * g.get("gamma", 0.0) * d_spot ** 2
    vega_pnl = g.get("vega", 0.0) * (d_vol / c.vega_scale)
    theta_pnl = g.get("theta", 0.0) * (d_years / c.theta_scale)
    rho_pnl = g.get("rho", 0.0) * (d_rate / c.rho_scale)
    total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
    return {"delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl, "vega_pnl": vega_pnl,
            "theta_pnl": theta_pnl, "rho_pnl": rho_pnl, "total": total}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes(S: float, K: float, T: float, r: float, q: float, sigma: float,
                  flag: str = "c") -> Greeks:
    """Generalised Black-Scholes greeks in the ABSOLUTE convention (the pivot).

    Present so the demo can prove the reference numbers rather than assert them, and so a
    library's output can be checked against a second opinion with zero dependencies.
    """
    f = flag.strip().lower()[:1]
    if f not in ("c", "p"):
        raise ValueError(f"flag must be 'c' or 'p', got {flag!r}")
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive for the analytic greeks")
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf1 = _norm_pdf(d1)
    if f == "c":
        price = S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
        delta = df_q * _norm_cdf(d1)
        theta = (-S * df_q * pdf1 * sigma / (2 * sqrt_t)
                 - r * K * df_r * _norm_cdf(d2) + q * S * df_q * _norm_cdf(d1))
        rho = K * T * df_r * _norm_cdf(d2)
    else:
        price = K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)
        delta = -df_q * _norm_cdf(-d1)
        theta = (-S * df_q * pdf1 * sigma / (2 * sqrt_t)
                 + r * K * df_r * _norm_cdf(-d2) - q * S * df_q * _norm_cdf(-d1))
        rho = -K * T * df_r * _norm_cdf(-d2)
    return {"price": price, "delta": delta, "gamma": df_q * pdf1 / (S * sigma * sqrt_t),
            "vega": S * df_q * pdf1 * sqrt_t, "theta": theta, "rho": rho}


def report(greeks: Mapping[str, float], conv: str | Convention) -> str:
    """One block of text carrying the greeks AND the units they are in."""
    c = get_convention(conv)
    g = {k.lower(): float(v) for k, v in greeks.items()}
    lines = [f"convention: {c.name}  ({c.note})"]
    for k in ("price", "delta", "gamma", "vega", "theta", "rho"):
        if k not in g:
            continue
        unit = {"vega": c.vega_unit, "theta": c.theta_unit, "rho": c.rho_unit}.get(
            k, "convention-free")
        lines.append(f"  {k:<6} {g[k]:>16.8f}   {unit}")
    return "\n".join(lines)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Verified Black-Scholes reference: S=K=100, T=1y (ACT/365), r=5%, q=0, sigma=20%.
    # ---------------------------------------------------------------------------------
    S = K = 100.0
    T, r, q, sigma = 1.0, 0.05, 0.0, 0.20

    QUANTLIB_CALL: Greeks = {
        "price": 10.45058357, "delta": 0.63683065, "gamma": 0.01876202,
        "vega": 37.52403469, "theta": -6.41402755, "rho": 53.23248155,
    }
    VOLLIB_CALL: Greeks = {
        "price": 10.45058357, "delta": 0.63683065, "gamma": 0.01876202,
        "vega": 0.37524035, "theta": -0.01757268, "rho": 0.53232482,
    }

    print("=" * 78)
    print("1. THE SAME OPTION, PRICED TWICE. price/delta/gamma AGREE; vega/theta/rho DO NOT")
    print("=" * 78)
    print(report(QUANTLIB_CALL, "quantlib"))
    print(report(VOLLIB_CALL, "vollib"))
    print("\nThree of six numbers match exactly. That is the trap: the spot-check passes.")

    # Second opinion with zero dependencies -- the reference numbers are DERIVED, not
    # copied, so this file cannot drift away from the pricers it claims to normalise.
    mine = black_scholes(S, K, T, r, q, sigma, "c")
    for key, ref in QUANTLIB_CALL.items():
        assert abs(mine[key] - ref) < 1e-6, (key, mine[key], ref)
    print("closed-form check: local Black-Scholes reproduces all 6 QuantLib numbers "
          "to <1e-6.")

    # -------------------------------------------------------------- 2. ROUND TRIP
    print("\n" + "=" * 78)
    print("2. ROUND TRIP quantlib -> vollib -> quantlib REPRODUCES BOTH SIDES EXACTLY")
    print("=" * 78)
    as_vollib = convert(QUANTLIB_CALL, "quantlib", "vollib")
    back = convert(as_vollib, "vollib", "quantlib")
    print(f"{'greek':<7}{'quantlib in':>16}{'-> vollib':>16}{'published vollib':>18}"
          f"{'-> back':>16}")
    for key in ("price", "delta", "gamma", "vega", "theta", "rho"):
        print(f"{key:<7}{QUANTLIB_CALL[key]:>16.8f}{as_vollib[key]:>16.8f}"
              f"{VOLLIB_CALL[key]:>18.8f}{back[key]:>16.8f}")
        assert abs(as_vollib[key] - VOLLIB_CALL[key]) < 5e-9, key
        assert abs(back[key] - QUANTLIB_CALL[key]) < 1e-12, key
    print("\nboth directions reproduce the published numbers to <5e-9. ratios: "
          f"vega x{QUANTLIB_CALL['vega'] / VOLLIB_CALL['vega']:.1f}, "
          f"theta x{QUANTLIB_CALL['theta'] / VOLLIB_CALL['theta']:.1f}, "
          f"rho x{QUANTLIB_CALL['rho'] / VOLLIB_CALL['rho']:.1f}")

    trading_day = convert(QUANTLIB_CALL, "quantlib", "per_point_per_trading_day")
    print(f"\nsame option, per_point_per_trading_day theta = {trading_day['theta']:.8f} "
          f"vs calendar-day {VOLLIB_CALL['theta']:.8f}  (365/252 = 1.45x, a third "
          f"convention that also silently 'works')")

    # ------------------------------------------------------ 3. THE 100x P&L ERROR
    print("\n" + "=" * 78)
    print("3. THE P&L: vol +1 point, one calendar day passes, rates +1bp")
    print("=" * 78)
    d_vol, d_years, d_rate = 0.01, 1.0 / 365.0, 0.0001

    ql_pnl = pnl(QUANTLIB_CALL, "quantlib", d_vol=d_vol, d_years=d_years, d_rate=d_rate)
    vl_pnl = pnl(VOLLIB_CALL, "vollib", d_vol=d_vol, d_years=d_years, d_rate=d_rate)
    print(f"correct, greeks tagged 'quantlib': total = {ql_pnl['total']:+.8f}")
    print(f"correct, greeks tagged 'vollib'  : total = {vl_pnl['total']:+.8f}")
    # 1e-7, not 0: the published vollib numbers are quoted to 8dp, so they carry ~3e-9 of
    # rounding that the /100 scaling magnifies. Agreement to 7dp IS the point.
    assert abs(ql_pnl["total"] - vl_pnl["total"]) < 1e-7
    print("identical to 7dp, as they must be -- the option does not care which library "
          "you used.")

    # The actual field bug: QuantLib greeks multiplied by vollib-style bumps ("1 vol
    # point", "1 day", "1 percent") because the greek dict carried no units.
    naive_vega = QUANTLIB_CALL["vega"] * 1.0        # "one vol point"
    naive_theta = QUANTLIB_CALL["theta"] * 1.0      # "one day"
    naive_rho = QUANTLIB_CALL["rho"] * 0.01         # "one bp = 0.01 percent"
    naive_total = naive_vega + naive_theta + naive_rho
    true_vega = ql_pnl["vega_pnl"]
    print(f"\nmismatched (QuantLib greeks x vollib-style bumps):")
    print(f"  vega  P&L {naive_vega:+14.8f}   correct {true_vega:+.8f}   "
          f"WRONG BY {naive_vega / true_vega:.0f}x")
    print(f"  theta P&L {naive_theta:+14.8f}   correct {ql_pnl['theta_pnl']:+.8f}   "
          f"WRONG BY {naive_theta / ql_pnl['theta_pnl']:.0f}x")
    print(f"  rho   P&L {naive_rho:+14.8f}   correct {ql_pnl['rho_pnl']:+.8f}   "
          f"WRONG BY {naive_rho / ql_pnl['rho_pnl']:.0f}x")
    print(f"  total     {naive_total:+14.8f}   correct {ql_pnl['total']:+.8f}")
    assert round(naive_vega / true_vega) == 100
    assert round(naive_theta / ql_pnl["theta_pnl"]) == 365
    print("\nOn a 1,000-lot book that is a hedge sized 100x too large, from arithmetic "
          "that never raised.")

    # -------------------------------------------------------- 4. SANITY INVARIANTS
    print("\n" + "=" * 78)
    print("4. SIGN INVARIANTS: what a long option can never do")
    print("=" * 78)
    print("long call, vollib units :", sanity_check(VOLLIB_CALL, "c", moneyness=S / K),
          "(no warnings)")
    put = black_scholes(S, K, T, r, q, sigma, "p")
    print(f"long put  , absolute    : delta={put['delta']:+.6f} theta={put['theta']:+.6f} "
          f"rho={put['rho']:+.6f} ->", sanity_check(put, "p", moneyness=S / K))

    for label, bad in [
        ("theta reported as a positive 'decay per day'", {**VOLLIB_CALL, "theta": 0.01757268}),
        ("put greeks with |delta| instead of delta", {**put, "delta": +0.36316935}),
        ("a SHORT position pasted into a long report", {k: -v for k, v in put.items()}),
    ]:
        try:
            sanity_check(bad, "p" if "put" in label or "SHORT" in label else "c")
        except ValueError as e:
            print(f"\nCAUGHT [{label}]:\n  {str(e).splitlines()[1].strip()}")

    print("\n" + "=" * 78)
    print("Tag the convention at the boundary. Convert once. Never hand-scale a bump.")
    print("=" * 78)
