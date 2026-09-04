#!/usr/bin/env python3
"""QuantLib prices against THREE dates, and nothing checks that they agree.

`ql.Settings.instance().evaluationDate` is a process-global. A term structure built
with a fixed reference date (`ql.FlatForward(refDate, ...)`) freezes its own anchor at
construction. The instrument carries a third date, its exercise date. The engine uses
all three without ever comparing them:

  * expiry <= evaluationDate            -> NPV() returns exactly 0.0, silently.
    This is the default state of a fresh process: evaluationDate is the OS clock, so
    every historical valuation is "expired" until you set it.
  * curve reference != evaluationDate   -> NPV() returns a plausible WRONG number.
    This is the ordering bug: the curves were built, THEN the evaluation date moved.
    The engine measures variance and discounting from the CURVE's reference date, so
    the option silently prices with the wrong time to expiry.

A zero in a book of thousands of positions reads as "worthless option", not "wrong
date", and the wrong-but-nonzero case does not even read as wrong.

Run:  python npv_zero.py
QuantLib is optional. Without it the reference model below reproduces all four
numbers from ACT/365 date arithmetic, so the demonstration still runs.
"""
from __future__ import annotations

from datetime import date

import numpy as np
from scipy.stats import norm

S, K, R, Q, SIGMA = 100.0, 100.0, 0.05, 0.0, 0.20

# Scenario 1: a historical valuation. Both dates are in the past, so a fresh process
# whose evaluationDate is the OS clock is already past this option's expiry.
HIST_VAL, HIST_EXP = date(2020, 1, 15), date(2021, 1, 15)

# Scenario 2: the ordering bug. STALE is whatever evaluationDate held when the curves
# were constructed; VAL is where the analyst moved it afterwards.
STALE, VAL, EXP = date(2026, 9, 4), date(2026, 6, 4), date(2027, 6, 4)


def act365(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.0


def bs_call(S, K, t, r, q, sigma) -> float:
    if t <= 0.0:
        return max(S - K, 0.0)
    v = sigma * np.sqrt(t)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * t) / v
    return float(S * np.exp(-q * t) * norm.cdf(d1)
                 - K * np.exp(-r * t) * norm.cdf(d1 - v))


def reference_npv(eval_date: date, curve_ref: date, expiry: date, spot: float = S) -> float:
    """What QuantLib's AnalyticEuropeanEngine actually does, in eight lines.

    Note which date governs which step - that IS the trap:
      the EXPIRY test is against evaluationDate,
      the TIME used for variance and discounting is measured from the CURVE reference.
    """
    if expiry <= eval_date:
        return 0.0                                   # expired event, dropped silently
    t = act365(curve_ref, expiry)                    # <- NOT measured from eval_date
    return bs_call(spot, K, t, R, Q, SIGMA)


def live_quantlib() -> dict[str, object] | None:
    """Build the same option four ways in the real library and report what it returns."""
    try:
        import QuantLib as ql
    except ImportError:
        return None

    dc, cal = ql.Actual365Fixed(), ql.NullCalendar()

    def qd(d: date) -> "ql.Date":
        return ql.Date(d.day, d.month, d.year)       # day-FIRST, unlike datetime

    def fixed_process(ref: "ql.Date", spot: float = S):
        """Curves anchored to a fixed date - they do NOT follow evaluationDate."""
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            ql.YieldTermStructureHandle(ql.FlatForward(ref, Q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(ref, R, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref, cal, SIGMA, dc)))

    def floating_process():
        """0 settlement days + a calendar - the reference date TRACKS evaluationDate."""
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(0, cal, Q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(0, cal, R, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(0, cal, SIGMA, dc)))

    def priced(process, expiry: date, strike: float = K):
        o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike),
                             ql.EuropeanExercise(qd(expiry)))
        o.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return o

    out: dict[str, object] = {"version": ql.__version__}

    # --- 1. never set the evaluation date: it is the OS clock -------------------
    out["default_eval_date"] = str(ql.Settings.instance().evaluationDate)
    out["forgot"] = float(priced(fixed_process(qd(HIST_VAL)), HIST_EXP).NPV())

    # --- 2. set it, but only AFTER the curves were built ------------------------
    ql.Settings.instance().evaluationDate = qd(STALE)
    stale_process = fixed_process(ql.Settings.instance().evaluationDate)   # anchored now
    ql.Settings.instance().evaluationDate = qd(VAL)                       # moved after
    out["curve_ref_after_move"] = str(stale_process.riskFreeRate().referenceDate())
    out["eval_after_move"] = str(ql.Settings.instance().evaluationDate)
    out["wrong_order"] = float(priced(stale_process, EXP).NPV())
    out["wrong_order_yf"] = float(dc.yearFraction(qd(STALE), qd(EXP)))

    # --- 3a. the fix: evaluation date FIRST, then build -------------------------
    ql.Settings.instance().evaluationDate = qd(VAL)
    out["right_order"] = float(priced(fixed_process(qd(VAL)), EXP).NPV())
    out["right_order_yf"] = float(dc.yearFraction(qd(VAL), qd(EXP)))

    # --- 3b. the belt-and-braces fix: curves that track evaluationDate ----------
    ql.Settings.instance().evaluationDate = qd(STALE)
    fl = floating_process()                     # built while the date is still stale
    ql.Settings.instance().evaluationDate = qd(VAL)      # ... and it follows anyway
    out["floating_ref"] = str(fl.riskFreeRate().referenceDate())
    out["floating"] = float(priced(fl, EXP).NPV())

    # --- 4. the boundary: expiry ON the evaluation date, 20 points in the money -
    ql.Settings.instance().evaluationDate = qd(VAL)
    itm = priced(fixed_process(qd(VAL), spot=120.0), VAL)
    out["expiry_today"] = float(itm.NPV())
    ql.Settings.instance().includeReferenceDateEvents = True
    itm.recalculate()
    out["expiry_today_included"] = float(itm.NPV())
    ql.Settings.instance().includeReferenceDateEvents = False

    # --- 5. the "throws unhelpfully" half of the trap ---------------------------
    try:
        empty = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)), ql.YieldTermStructureHandle(),
            ql.YieldTermStructureHandle(), ql.BlackVolTermStructureHandle())
        priced(empty, EXP).NPV()
        out["empty_handle"] = "no error (unexpected)"
    except RuntimeError as e:
        out["empty_handle"] = str(e).strip().splitlines()[-1]
    return out


if __name__ == "__main__":
    live = live_quantlib()
    ref_forgot = reference_npv(date.today(), HIST_VAL, HIST_EXP)
    ref_wrong = reference_npv(VAL, STALE, EXP)
    ref_right = reference_npv(VAL, VAL, EXP)
    ref_expiry_today = reference_npv(VAL, VAL, VAL, spot=120.0)

    print("European call  S=K=100  r=5%  q=0%  sigma=20%  ACT/365\n")
    print(f"  1. FORGOT to set evaluationDate")
    print(f"       curve reference {HIST_VAL}   expiry {HIST_EXP}")
    print(f"       evaluationDate defaults to the OS clock, which is past the expiry")
    print(f"       NPV = {ref_forgot:.10f}   <- exactly zero, no warning, no exception\n")

    print(f"  2. Set evaluationDate AFTER building the curves")
    print(f"       curve reference {STALE}   evaluationDate {VAL}   expiry {EXP}")
    print(f"       the engine measured {act365(STALE, EXP):.10f}y of variance and")
    print(f"       discounting, not {act365(VAL, EXP):.10f}y - the curve anchor never moved")
    print(f"       NPV = {ref_wrong:.10f}   <- plausible, and wrong by "
          f"{abs(ref_wrong / ref_right - 1):.2%}\n")

    print(f"  3. Set evaluationDate FIRST, then build")
    print(f"       curve reference {VAL}   evaluationDate {VAL}   expiry {EXP}")
    print(f"       NPV = {ref_right:.10f}   <- correct\n")

    print(f"  4. Expiry ON the evaluation date, spot 120 vs strike 100")
    print(f"       NPV = {ref_expiry_today:.10f}   <- 20 points in the money, "
          f"priced at zero\n")

    if live is None:
        print("  QuantLib not installed - reference model only. The four numbers above")
        print("  come from ACT/365 date arithmetic plus the closed form, which is")
        print("  exactly what the engine does once you know which date governs which step.")
    else:
        print(f"  verified against the installed library (QuantLib {live['version']}):")
        print(f"    default evaluationDate in this process : {live['default_eval_date']}")
        pairs = [
            ("1 forgot to set it", live["forgot"], ref_forgot),
            ("2 set after building", live["wrong_order"], ref_wrong),
            ("3 set before building", live["right_order"], ref_right),
            ("4 expiry == eval date", live["expiry_today"], ref_expiry_today),
        ]
        for label, got, exp in pairs:
            print(f"    {label:<22} QuantLib {got:>16.10f}   reference {exp:>16.10f}"
                  f"   |diff| {abs(got - exp):.2e}")
        print(f"\n    after the move, curve reference is still "
              f"{live['curve_ref_after_move']} while evaluationDate is "
              f"{live['eval_after_move']}")
        print(f"    QuantLib's own yearFraction confirms it: {live['wrong_order_yf']:.10f}y"
              f" priced instead of {live['right_order_yf']:.10f}y")
        print(f"\n    fix A - set evaluationDate first          NPV "
              f"{live['right_order']:.10f}")
        print(f"    fix B - FlatForward(0, calendar, ...)     NPV "
              f"{live['floating']:.10f}"
              f"  (reference tracked to {live['floating_ref']})")
        print(f"    boundary - includeReferenceDateEvents=True gives "
              f"{live['expiry_today_included']:.10f} instead of "
              f"{live['expiry_today']:.10f}")
        print(f"    empty handle raises only at pricing time: "
              f"{live['empty_handle']!r}")

    print("\n  Rule: set ql.Settings.instance().evaluationDate BEFORE constructing any")
    print("        term structure, and assert it afterwards. Prefer curves built with")
    print("        (0, calendar) so their reference date follows the global. An NPV of")
    print("        exactly 0.0 is a date bug until proven otherwise.")
