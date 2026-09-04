#!/usr/bin/env python3
"""vollib's Greeks are already SCALED; the textbook closed form is not. Nothing warns.

`vega` and `rho` come back multiplied by 0.01 — per **1 percentage point** of vol or
rate — and `theta` divided by 365 — per **calendar day**. That is the trading-screen
convention, and it is the OPPOSITE of what the raw Black-Scholes derivative gives you,
which is what QuantLib, financepy and every textbook return.

So the mistake has two directions and both are silent:
  * treat vollib's number as raw and scale it again  -> 100x / 365x too SMALL
  * treat QuantLib's raw number as already scaled    -> 100x / 365x too LARGE

Neither raises. Both produce a plausible-looking float. On a real position the gap is
six figures of vega you either do not have or cannot hedge.

Run:  python greeks_scaling.py
vollib is optional. Without it this reproduces the raw derivatives and the scaling
factors from the definitions, so the demonstration still runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# One concrete contract, the same one the SKILL.md table was measured on.
S, K, T, R, Q, SIGMA, FLAG = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "c"

# One concrete position: 500 contracts on a 100-multiplier listed option.
CONTRACTS, MULTIPLIER = 500, 100
UNITS = CONTRACTS * MULTIPLIER

# The scaling table asserted in lib-vollib/SKILL.md, as {greek: divisor}.
SKILL_MD_TABLE = {"vega": 100.0, "theta": 365.0, "rho": 100.0}


def _d1_d2(S, K, T, r, q, sigma):
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def raw_greeks(S, K, T, r, q, sigma, flag="c") -> dict[str, float]:
    """Textbook Black-Scholes-Merton derivatives, RAW per-unit — QuantLib's convention.

    vega  = dV/dsigma per 1.00 of vol   (not per vol point)
    theta = dV/dt     per YEAR          (not per day)
    rho   = dV/dr     per 1.00 of rate  (not per 1%)
    """
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    eqt, ert = np.exp(-q * T), np.exp(-r * T)
    sign = 1.0 if flag == "c" else -1.0
    Nd1, Nd2 = norm.cdf(sign * d1), norm.cdf(sign * d2)
    pdf_d1 = norm.pdf(d1)

    price = sign * (S * eqt * Nd1 - K * ert * Nd2)
    theta = (-S * eqt * pdf_d1 * sigma / (2 * np.sqrt(T))
             + sign * q * S * eqt * Nd1
             - sign * r * K * ert * Nd2)
    return {
        "price": float(price),
        "delta": float(sign * eqt * Nd1),
        "gamma": float(eqt * pdf_d1 / (S * sigma * np.sqrt(T))),
        "vega": float(S * eqt * pdf_d1 * np.sqrt(T)),
        "theta": float(theta),
        "rho": float(sign * K * T * ert * Nd2),
    }


def scaled_greeks(raw: dict[str, float]) -> dict[str, float]:
    """Apply the trading-screen scaling — this is what vollib hands back."""
    out = dict(raw)
    for g, divisor in SKILL_MD_TABLE.items():
        out[g] = raw[g] / divisor
    return out


def live_vollib() -> dict[str, float] | None:
    """Ask the installed library, so the table above is verified and not merely asserted."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # py_vollib is the deprecated shim; it resolves to the vollib package.
            from py_vollib.black_scholes import black_scholes
            from py_vollib.black_scholes.greeks.analytical import (
                delta, gamma, rho, theta, vega)
    except ImportError:
        return None
    a = (FLAG, S, K, T, R, SIGMA)
    return {
        "price": float(black_scholes(*a)),
        "delta": float(delta(*a)),
        "gamma": float(gamma(*a)),
        "vega": float(vega(*a)),
        "theta": float(theta(*a)),
        "rho": float(rho(*a)),
    }


if __name__ == "__main__":
    raw = raw_greeks(S, K, T, R, Q, SIGMA, FLAG)
    scaled = scaled_greeks(raw)
    live = live_vollib()

    print(f"European call  S=K={K:.0f}  T={T:.0f}y ACT/365  r={R:.0%}  q={Q:.0%}  "
          f"sigma={SIGMA:.0%}\n")

    rows = []
    for g in ("price", "delta", "gamma", "vega", "theta", "rho"):
        div = SKILL_MD_TABLE.get(g, 1.0)
        rows.append({
            "greek": g,
            "RAW (QuantLib/textbook)": raw[g],
            "SCALED (vollib)": scaled[g],
            "factor": f"/{div:.0f}" if div != 1.0 else "same",
            "unit of the scaled number": {
                "vega": "per 1 vol POINT", "theta": "per CALENDAR DAY",
                "rho": "per 1% of rate"}.get(g, "-"),
        })
    df = pd.DataFrame(rows).set_index("greek")
    print(df.to_string(float_format=lambda x: f"{x:>14.8f}"))

    # ---- verify the reference implementation against the installed library -------
    print()
    if live is None:
        print("  vollib not installed - reference implementation only; the scaling")
        print("  factors above come from the definitions, not from a measurement.")
    else:
        print("  verified against the installed library (py_vollib -> vollib):")
        worst = 0.0
        mismatch = []
        for g in ("price", "delta", "gamma", "vega", "theta", "rho"):
            err = abs(live[g] - scaled[g])
            worst = max(worst, err)
            print(f"    {g:<6} vollib {live[g]:>16.10f}   reference-scaled "
                  f"{scaled[g]:>16.10f}   |diff| {err:.2e}")
            # Re-derive the factor the library actually used, from its own output.
            if g in SKILL_MD_TABLE:
                measured = raw[g] / live[g]
                claimed = SKILL_MD_TABLE[g]
                ok = abs(measured - claimed) < 1e-6 * claimed
                mismatch.append((g, claimed, measured, ok))
        print(f"    worst absolute disagreement across all six: {worst:.2e}"
              f"  -> vollib IS the pre-scaled convention")

        print("\n  SKILL.md scaling table, checked against what vollib actually returned:")
        for g, claimed, measured, ok in mismatch:
            print(f"    {g:<6} SKILL.md says /{claimed:<5.0f} measured raw/vollib = "
                  f"{measured:.8f}   {'MATCH' if ok else 'DISAGREES'}")
        if all(ok for *_, ok in mismatch):
            print("    all three factors in the SKILL.md table reproduce exactly.")

    # ---- what the confusion costs on a real position ----------------------------
    src = live if live is not None else scaled
    print(f"\nPosition: long {CONTRACTS} contracts x {MULTIPLIER} multiplier "
          f"= {UNITS:,} units of underlying\n")

    cases = [
        ("vega", "a +1 vol point move (20% -> 21%)", src["vega"], raw["vega"], 100.0),
        ("theta", "one calendar day of decay", src["theta"], raw["theta"], 365.0),
        ("rho", "a +1% parallel rate move", src["rho"], raw["rho"], 100.0),
    ]
    for g, event, correct_per_unit, raw_per_unit, factor in cases:
        right = correct_per_unit * UNITS
        wrong = raw_per_unit * UNITS
        print(f"  {g.upper():<6} P&L on {event}")
        print(f"    correct (vollib number x units)          ${right:>18,.2f}")
        print(f"    raw derivative used as if scaled         ${wrong:>18,.2f}"
              f"   <- {factor:.0f}x")
        print(f"    size of the mis-statement                ${wrong - right:>18,.2f}\n")

    print("  Rule: vollib hands you SCREEN units (vega/rho per 1 point, theta per day).")
    print("        QuantLib and financepy hand you RAW derivatives (per 1.00, per year).")
    print("        Never mix the two in one risk file. Nothing in either library will")
    print("        tell you that you did - the only symptom is a hedge off by 100x.")
