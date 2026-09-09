"""Day count, accrued interest, clean vs dirty, and the settlement lag that moves both.

WHY this exists: "accrued interest" is not a number until four things are attached to it --
the day-count convention, the coupon schedule the convention is measured against, the
settlement date, and the face amount. Libraries default all four, and the defaults disagree.

The traps measured below, all silent:

  * ACT/ACT ICMA WITHOUT ITS SCHEDULE. ACT/ACT (ICMA / ISMA / "Bond") is defined as
    days / (period_days * frequency) -- it CANNOT be evaluated from two dates alone, because
    the denominator is the coupon period, not the year. QuantLib's ActualActual(ISMA) accepts
    two dates anyway. Its own source (ql/time/daycounters/actualactual.cpp, read 2026-09-09)
    dispatches `case ISMA: case Bond: if (!schedule.empty()) ISMA_Impl else Old_ISMA_Impl`,
    and Old_ISMA_Impl guesses the frequency with lround(12 * days / 365) months. On a
    105-day stub of a SEMIANNUAL bond it infers a QUARTERLY period and returns exactly 0.25.

  * THE 30/360 FAMILY IS NOT ONE CONVENTION. 30/360 US, 30/360 Bond Basis, 30E/360 and
    30E/360 ISDA differ in how they treat the 31st and the last day of February. QuantLib's
    thirty360.cpp holds SIX implementations behind NINE enum constants, and Thirty360.USA is
    NOT Thirty360.BondBasis.

  * CLEAN vs DIRTY. Bonds quote clean. Cash settles dirty. The gap is the accrued, and it is
    the largest single number in this file.

  * SETTLEMENT LAG. T+1 vs T+2 is one more day of accrued at the same clean price.

Everything is computed twice: once by the reference implementation in this file (numpy only)
and, when QuantLib is importable, again by QuantLib 1.43 inside `quantlib_cross_checks`.

Usage:
    from conventions import year_fraction, accrued_interest, convention_table
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from typing import Iterable, Sequence

# The nine conventions this module implements. QuantLib's nine Thirty360 constants map onto
# the last five (measured in `quantlib_cross_checks`): USA -> 30/360 US; ISMA, BondBasis and
# NASD -> 30/360 BondBasis; European and EurobondBasis -> 30E/360; ISDA and German ->
# 30E/360 ISDA; Italian -> 30/360 Italian.
CONVENTIONS = ("ACT/ACT ICMA", "ACT/ACT ISDA", "ACT/365F", "ACT/360",
               "30/360 US", "30/360 BondBasis", "30E/360", "30E/360 ISDA", "30/360 Italian")
THIRTY_FLAVOURS = ("30/360 US", "30/360 BondBasis", "30E/360", "30E/360 ISDA", "30/360 Italian")


# ------------------------------------------------------------------ calendar helpers
def is_leap(year: int) -> bool:
    return calendar.isleap(year)


def last_day_of_month(d: date) -> int:
    return calendar.monthrange(d.year, d.month)[1]


def is_last_of_february(d: date) -> bool:
    return d.month == 2 and d.day == last_day_of_month(d)


def add_business_days(d: date, n: int, holidays: Iterable[date] = ()) -> date:
    """Move n business days forward (n > 0) or back (n < 0), skipping weekends and holidays."""
    hol = set(holidays)
    step = 1 if n >= 0 else -1
    left = abs(int(n))
    cur = d
    while left:
        cur = cur + timedelta(days=step)
        if cur.weekday() < 5 and cur not in hol:
            left -= 1
    return cur


# ------------------------------------------------------------------ 30/360 family
def thirty_360_days(d0: date, d1: date, flavour: str = "30/360 BondBasis",
                    termination: date | None = None) -> int:
    """Day count under one 30/360 flavour. The flavours differ ONLY on the 31st and on
    the last day of February, which is why they agree on most date pairs and not on yours.

      30/360 US          ISDA 2006 4.16(f): the February end-of-month rules apply
      30/360 BondBasis   the same WITHOUT the February rules (this is QuantLib's BondBasis)
      30E/360            Eurobond basis: 31 -> 30 on both legs, February untouched
      30E/360 ISDA       ISDA 2006 4.16(h): last day of the month -> 30 on both legs, except
                         an end date that is the TERMINATION date falling in February
      30/360 Italian     European, plus: a February date after the 27th becomes the 30th
    """
    if flavour not in THIRTY_FLAVOURS:
        raise ValueError(f"unknown 30/360 flavour {flavour!r}; use one of {THIRTY_FLAVOURS}")
    y0, m0, dd0 = d0.year, d0.month, d0.day
    y1, m1, dd1 = d1.year, d1.month, d1.day

    if flavour == "30/360 US":
        if is_last_of_february(d0) and is_last_of_february(d1):
            dd1 = 30
        if is_last_of_february(d0):
            dd0 = 30
        if dd1 == 31 and dd0 >= 30:
            dd1 = 30
        if dd0 == 31:
            dd0 = 30
    elif flavour == "30/360 BondBasis":
        if dd1 == 31 and dd0 >= 30:
            dd1 = 30
        if dd0 == 31:
            dd0 = 30
    elif flavour == "30E/360":
        dd0 = min(dd0, 30)
        dd1 = min(dd1, 30)
    elif flavour == "30E/360 ISDA":
        if dd0 == last_day_of_month(d0):
            dd0 = 30
        if dd1 == last_day_of_month(d1) and not (d1 == termination and d1.month == 2):
            dd1 = 30
    else:                                                     # 30/360 Italian
        if m0 == 2 and dd0 > 27:
            dd0 = 30
        if m1 == 2 and dd1 > 27:
            dd1 = 30
        dd0 = min(dd0, 30)
        dd1 = min(dd1, 30)
    return 360 * (y1 - y0) + 30 * (m1 - m0) + (dd1 - dd0)


# ------------------------------------------------------------------ ACT/ACT
def act_act_isda(d0: date, d1: date) -> float:
    """ACT/ACT ISDA (2006 definitions 4.16(b)): each calendar year gets its own denominator."""
    if d1 < d0:
        raise ValueError("d1 must not precede d0")
    total = 0.0
    for y in range(d0.year, d1.year + 1):
        lo = max(d0, date(y, 1, 1))
        hi = min(d1, date(y + 1, 1, 1))
        if hi > lo:
            total += (hi - lo).days / (366.0 if is_leap(y) else 365.0)
    return total


def icma_frequency_guess(d0: date, d1: date) -> int:
    """The frequency QuantLib INFERS when ActualActual(ISMA) is handed no reference period.

    Reproduces actualactual.cpp: months = round(12 * (refEnd - refStart) / 365); the year
    fraction of a whole such period is months/12. A 105-day stub rounds to 3 months.
    """
    months = int(round(12.0 * (d1 - d0).days / 365.0))
    if months == 0:
        months = 12
    return months


def act_act_icma(d0: date, d1: date, ref_start: date | None = None,
                 ref_end: date | None = None, freq: int = 2) -> float:
    """ACT/ACT ICMA: days / (days in the COUPON PERIOD * coupons per year).

    With `ref_start`/`ref_end` this is the real definition. Without them it reproduces
    QuantLib's fallback -- the two dates become their own reference period and the frequency
    is guessed -- which is the trap, not the convention.
    """
    if ref_start is None or ref_end is None:
        months = icma_frequency_guess(d0, d1)
        period = months / 12.0
        return period * (d1 - d0).days / max((d1 - d0).days, 1)
    if not (ref_start <= d0 <= d1 <= ref_end):
        raise ValueError("ACT/ACT ICMA needs ref_start <= d0 <= d1 <= ref_end")
    return (d1 - d0).days / ((ref_end - ref_start).days * float(freq))


# ------------------------------------------------------------------ the public entry point
def year_fraction(d0: date, d1: date, convention: str = "ACT/ACT ICMA", *,
                  ref_start: date | None = None, ref_end: date | None = None,
                  freq: int = 2, termination: date | None = None) -> float:
    """Year fraction between two dates under any of `CONVENTIONS`."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; use one of {CONVENTIONS}")
    if d1 < d0:
        raise ValueError("d1 must not precede d0")
    if convention == "ACT/ACT ICMA":
        return act_act_icma(d0, d1, ref_start, ref_end, freq)
    if convention == "ACT/ACT ISDA":
        return act_act_isda(d0, d1)
    if convention == "ACT/365F":
        return (d1 - d0).days / 365.0
    if convention == "ACT/360":
        return (d1 - d0).days / 360.0
    return thirty_360_days(d0, d1, convention, termination) / 360.0


def accrued_interest(coupon_rate: float, prev_coupon: date, settle: date, next_coupon: date,
                     convention: str = "ACT/ACT ICMA", freq: int = 2, face: float = 100.0,
                     *, use_schedule: bool = True) -> float:
    """Accrued interest per `face`. `use_schedule=False` drops the ICMA reference period."""
    ref = dict(ref_start=prev_coupon, ref_end=next_coupon) if use_schedule else {}
    yf = year_fraction(prev_coupon, settle, convention, freq=freq, **ref)
    return face * coupon_rate * yf


def dirty_price(clean: float, accrued: float) -> float:
    return clean + accrued


def settlement_amount(clean: float, accrued: float, face: float) -> float:
    """Cash that actually moves. Face is the notional, clean and accrued are per 100."""
    return face * (clean + accrued) / 100.0


# ------------------------------------------------------------------ tables
def convention_table(coupon_rate: float, prev_coupon: date, settle: date, next_coupon: date,
                     freq: int = 2, face: float = 1_000_000.0) -> list[dict]:
    """Accrued on ONE bond under every convention, plus the ICMA no-schedule fallback."""
    rows = []
    base = accrued_interest(coupon_rate, prev_coupon, settle, next_coupon, "ACT/ACT ICMA", freq)
    for conv in CONVENTIONS:
        per100 = accrued_interest(coupon_rate, prev_coupon, settle, next_coupon, conv, freq)
        rows.append({"convention": conv,
                     "year_fraction": year_fraction(
                         prev_coupon, settle, conv,
                         ref_start=prev_coupon, ref_end=next_coupon, freq=freq),
                     "accrued_per_100": per100,
                     "cash_per_face": face * per100 / 100.0,
                     "vs_icma_per_face": face * (per100 - base) / 100.0})
    bad = accrued_interest(coupon_rate, prev_coupon, settle, next_coupon, "ACT/ACT ICMA",
                           freq, use_schedule=False)
    rows.append({"convention": "ACT/ACT ICMA (no schedule)",
                 "year_fraction": act_act_icma(prev_coupon, settle),
                 "accrued_per_100": bad,
                 "cash_per_face": face * bad / 100.0,
                 "vs_icma_per_face": face * (bad - base) / 100.0})
    return rows


def thirty_360_table(d0: date, d1: date, coupon_rate: float = 0.05,
                     face: float = 1_000_000.0) -> list[dict]:
    """The 30/360 flavours on one date pair, and the cash spread between them."""
    rows = [{"flavour": f, "days": thirty_360_days(d0, d1, f),
             "year_fraction": thirty_360_days(d0, d1, f) / 360.0,
             "accrued_per_100": 100.0 * coupon_rate * thirty_360_days(d0, d1, f) / 360.0}
            for f in THIRTY_FLAVOURS]
    lo = min(r["accrued_per_100"] for r in rows)
    for r in rows:
        r["cash_vs_lowest"] = face * (r["accrued_per_100"] - lo) / 100.0
    return rows


def settlement_lag_table(coupon_rate: float, prev_coupon: date, trade: date, next_coupon: date,
                         lags: Sequence[int] = (0, 1, 2, 3), freq: int = 2,
                         face: float = 1_000_000.0,
                         holidays: Iterable[date] = ()) -> list[dict]:
    """The same clean price on T+0..T+3: only the accrued moves, and it moves every time."""
    rows = []
    base = None
    for n in lags:
        s = add_business_days(trade, n, holidays) if n else trade
        per100 = accrued_interest(coupon_rate, prev_coupon, s, next_coupon, "ACT/ACT ICMA", freq)
        base = per100 if base is None else base
        rows.append({"lag": n, "settle": s, "accrued_per_100": per100,
                     "cash_per_face": face * per100 / 100.0,
                     "vs_t0_per_face": face * (per100 - base) / 100.0})
    return rows


# ------------------------------------------------------------------ QuantLib
def quantlib_cross_checks(prev_coupon: date, settle: date, next_coupon: date,
                          maturity: date, coupon_rate: float = 0.05,
                          pair: tuple[date, date] | None = None) -> dict | None:
    """Every number above, recomputed by QuantLib 1.43. Returns None if it is not installed."""
    try:
        import QuantLib as ql
    except ImportError:
        return None

    def q(d: date):
        return ql.Date(d.day, d.month, d.year)

    out: dict = {"version": ql.__version__}

    # --- the nine Thirty360 constants on one date pair
    a, b = pair if pair else (prev_coupon, settle)
    consts = [("USA", ql.Thirty360.USA), ("BondBasis", ql.Thirty360.BondBasis),
              ("European", ql.Thirty360.European),
              ("EurobondBasis", ql.Thirty360.EurobondBasis),
              ("Italian", ql.Thirty360.Italian), ("German", ql.Thirty360.German),
              ("ISMA", ql.Thirty360.ISMA), ("ISDA", ql.Thirty360.ISDA),
              ("NASD", ql.Thirty360.NASD)]
    thirty = {}
    for nm, c in consts:
        dc = ql.Thirty360(c, q(maturity)) if c == ql.Thirty360.ISDA else ql.Thirty360(c)
        thirty[nm] = dc.dayCount(q(a), q(b))
    out["thirty360"] = thirty
    out["thirty360_distinct"] = sorted(set(thirty.values()))

    # --- ActualActual ISMA with and without the reference period
    dc = ql.ActualActual(ql.ActualActual.ISMA)
    out["isma_with_ref"] = dc.yearFraction(q(prev_coupon), q(settle),
                                           q(prev_coupon), q(next_coupon))
    out["isma_no_ref"] = dc.yearFraction(q(prev_coupon), q(settle))
    out["isda_yf"] = ql.ActualActual(ql.ActualActual.ISDA).yearFraction(q(prev_coupon), q(settle))
    out["afb_yf"] = ql.ActualActual(ql.ActualActual.AFB).yearFraction(q(prev_coupon), q(settle))
    out["act365f_yf"] = ql.Actual365Fixed().yearFraction(q(prev_coupon), q(settle))
    out["act360_yf"] = ql.Actual360().yearFraction(q(prev_coupon), q(settle))

    # --- a real FixedRateBond, accrued under each day count
    ql.Settings.instance().evaluationDate = q(settle)
    sched = ql.Schedule(q(prev_coupon), q(maturity), ql.Period(ql.Semiannual),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                        ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
    day_counts = [("ACT/ACT ICMA", ql.ActualActual(ql.ActualActual.ISMA, sched)),
                  ("ACT/ACT ISDA", ql.ActualActual(ql.ActualActual.ISDA)),
                  ("ACT/365F", ql.Actual365Fixed()), ("ACT/360", ql.Actual360()),
                  ("30/360 BondBasis", ql.Thirty360(ql.Thirty360.BondBasis))]
    bond_accrued = {}
    for nm, d in day_counts:
        bond_accrued[nm] = ql.FixedRateBond(1, 100.0, sched, [coupon_rate], d) \
            .accruedAmount(q(settle))
    out["bond_accrued"] = bond_accrued
    # the same bond with the schedule-free ISMA day counter: FixedRateBond supplies the
    # reference period itself, so this path is SAFE. That is worth knowing precisely.
    out["bond_accrued_no_schedule"] = ql.FixedRateBond(
        1, 100.0, sched, [coupon_rate], ql.ActualActual(ql.ActualActual.ISMA)) \
        .accruedAmount(q(settle))
    yr = ql.InterestRate(coupon_rate, ql.ActualActual(ql.ActualActual.ISMA, sched),
                         ql.Compounded, ql.Semiannual)
    yr_bad = ql.InterestRate(coupon_rate, ql.ActualActual(ql.ActualActual.ISMA),
                             ql.Compounded, ql.Semiannual)
    bond = ql.FixedRateBond(1, 100.0, sched, [coupon_rate],
                            ql.ActualActual(ql.ActualActual.ISMA, sched))
    out["mod_duration_with_schedule"] = ql.BondFunctions.duration(bond, yr, ql.Duration.Modified)
    out["mod_duration_no_schedule"] = ql.BondFunctions.duration(bond, yr_bad,
                                                                ql.Duration.Modified)

    # --- where the fallback DOES bite: a curve, or any bare two-date year fraction
    curve_bad = ql.FlatForward(q(prev_coupon), coupon_rate,
                               ql.ActualActual(ql.ActualActual.ISMA),
                               ql.Compounded, ql.Semiannual)
    curve_ok = ql.FlatForward(q(prev_coupon), coupon_rate, ql.Actual365Fixed(),
                              ql.Compounded, ql.Semiannual)
    out["curve_t_isma"] = curve_bad.timeFromReference(q(settle))
    out["curve_t_act365"] = curve_ok.timeFromReference(q(settle))
    out["curve_df_isma"] = curve_bad.discount(q(settle))
    out["curve_df_act365"] = curve_ok.discount(q(settle))
    out["compound_factor_isma"] = yr_bad.compoundFactor(q(prev_coupon), q(settle))
    out["compound_factor_act365"] = ql.InterestRate(
        coupon_rate, ql.Actual365Fixed(), ql.Compounded,
        ql.Semiannual).compoundFactor(q(prev_coupon), q(settle))
    return out


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    W = 92
    PREV, SETTLE, NEXT = date(2026, 1, 15), date(2026, 4, 30), date(2026, 7, 15)
    MAT = date(2036, 1, 15)
    CPN, FACE = 0.05, 1_000_000.0

    print("=" * W)
    print("BOND CONVENTIONS AND ACCRUED -- a 5% semiannual bond, 1,000,000 face")
    print(f"coupon period {PREV} -> {NEXT} ({(NEXT - PREV).days} days), settle {SETTLE} "
          f"({(SETTLE - PREV).days} days in)")
    print("=" * W)

    print("\n1. ACCRUED UNDER EVERY CONVENTION (per 100, and the cash on 1,000,000 face)")
    print(f"   {'convention':<28}{'year fraction':>15}{'per 100':>12}{'cash':>16}"
          f"{'vs ICMA':>14}")
    for r in convention_table(CPN, PREV, SETTLE, NEXT, 2, FACE):
        print(f"   {r['convention']:<28}{r['year_fraction']:>15.8f}"
              f"{r['accrued_per_100']:>12.6f}{r['cash_per_face']:>16,.2f}"
              f"{r['vs_icma_per_face']:>+14,.2f}")
    rows = convention_table(CPN, PREV, SETTLE, NEXT, 2, FACE)
    good = rows[0]["cash_per_face"]
    bad = rows[-1]["cash_per_face"]
    print(f"   -> ACT/ACT ICMA WITH the schedule {good:,.2f}; the same day counter handed only "
          f"two dates {bad:,.2f}")
    print(f"   -> gap {good - bad:+,.2f} on 1,000,000 face, and nothing raises")
    print(f"   -> the fallback guesses {icma_frequency_guess(PREV, SETTLE)} months from a "
          f"{(SETTLE - PREV).days}-day stub, i.e. QUARTERLY on a SEMIANNUAL bond, so the year "
          f"fraction is exactly {act_act_icma(PREV, SETTLE):.8f}")

    print("\n2. THE 30/360 FAMILY IS FIVE NAMED RULES, NOT ONE")
    for lbl, (a, b) in (("2026-01-15 -> 2026-07-31", (date(2026, 1, 15), date(2026, 7, 31))),
                        ("2026-02-28 -> 2026-08-31", (date(2026, 2, 28), date(2026, 8, 31))),
                        ("2028-02-29 -> 2028-08-31", (date(2028, 2, 29), date(2028, 8, 31)))):
        tt = thirty_360_table(a, b, CPN, FACE)
        days = {r["flavour"]: r["days"] for r in tt}
        spread = max(r["cash_vs_lowest"] for r in tt)
        print(f"   {lbl}  " + "  ".join(f"{k.replace('30/360 ', '').replace('30E/360', 'E'):>10}"
                                        f"={v:3d}" for k, v in days.items()))
        print(f"   {'':<24}  {len(set(days.values()))} distinct answers, "
              f"spread {spread:,.2f} of accrued on 1,000,000 face at a {CPN:.0%} coupon")

    print("\n3. CLEAN vs DIRTY -- the quote is not the cash")
    clean = 98.500
    acc = accrued_interest(CPN, PREV, SETTLE, NEXT, "ACT/ACT ICMA", 2)
    print(f"   clean {clean:.3f}   accrued {acc:.6f}   dirty {dirty_price(clean, acc):.6f}")
    print(f"   settlement on 1,000,000 face = {settlement_amount(clean, acc, FACE):,.2f}, "
          f"of which {FACE * acc / 100.0:,.2f} is accrued")
    print("   -> a P&L built from clean prices alone drops the coupon accrual entirely")

    print("\n4. SETTLEMENT LAG -- same clean price, different cash")
    for r in settlement_lag_table(CPN, PREV, SETTLE, NEXT, (0, 1, 2, 3), 2, FACE):
        print(f"   T+{r['lag']}  settle {r['settle']}  accrued {r['accrued_per_100']:.6f}"
              f"   cash {r['cash_per_face']:>12,.2f}   vs T+0 {r['vs_t0_per_face']:>+10,.2f}")

    print("\n5. CROSS-CHECK AGAINST QUANTLIB")
    live = quantlib_cross_checks(PREV, SETTLE, NEXT, MAT, CPN,
                                 pair=(date(2026, 2, 28), date(2026, 8, 31)))
    if live is None:
        print("   QuantLib not installed - the reference implementation above stands alone")
    else:
        print(f"   QuantLib {live['version']}")
        print("   Thirty360 constants on 2026-02-28 -> 2026-08-31: "
              + ", ".join(f"{k}={v}" for k, v in live["thirty360"].items()))
        print(f"   -> {len(live['thirty360_distinct'])} distinct day counts "
              f"{live['thirty360_distinct']} from NINE named constants; "
              f"Thirty360.USA != Thirty360.BondBasis")
        mine = {f: thirty_360_days(date(2026, 2, 28), date(2026, 8, 31), f)
                for f in THIRTY_FLAVOURS}
        print(f"   mine: US={mine['30/360 US']} BondBasis={mine['30/360 BondBasis']} "
              f"E={mine['30E/360']} E-ISDA={mine['30E/360 ISDA']} "
              f"Italian={mine['30/360 Italian']}  (matches QuantLib's USA / BondBasis / "
              f"European / ISDA / Italian)")
        print(f"   ActualActual(ISMA) WITH ref period  {live['isma_with_ref']:.8f}   "
              f"mine {act_act_icma(PREV, SETTLE, PREV, NEXT):.8f}   "
              f"|diff| {abs(live['isma_with_ref'] - act_act_icma(PREV, SETTLE, PREV, NEXT)):.1e}")
        print(f"   ActualActual(ISMA) WITHOUT it       {live['isma_no_ref']:.8f}   "
              f"mine {act_act_icma(PREV, SETTLE):.8f}   "
              f"|diff| {abs(live['isma_no_ref'] - act_act_icma(PREV, SETTLE)):.1e}")
        print(f"   ActualActual(ISDA) {live['isda_yf']:.8f} (mine "
              f"{act_act_isda(PREV, SETTLE):.8f})   AFB {live['afb_yf']:.8f}   "
              f"ACT/365F {live['act365f_yf']:.8f}   ACT/360 {live['act360_yf']:.8f}")
        print("   FixedRateBond.accruedAmount on a real 10y 5% semiannual bond:")
        for k, v in live["bond_accrued"].items():
            mine_v = accrued_interest(CPN, PREV, SETTLE, NEXT, k, 2)
            print(f"     {k:<18}{v:>12.8f} per 100   mine {mine_v:>12.8f}   "
                  f"|diff| {abs(v - mine_v):.1e}")
        ns = live["bond_accrued_no_schedule"]
        ref = live["bond_accrued"]["ACT/ACT ICMA"]
        print("   WHERE THE SCHEDULE-FREE ISMA COUNTER DOES AND DOES NOT BITE:")
        print(f"     SAFE  FixedRateBond.accruedAmount, counter built with no schedule: "
              f"{ns:.8f} vs {ref:.8f} ({FACE * (ref - ns) / 100.0:+,.2f} on 1,000,000) -- the "
              f"bond passes its own coupon period in")
        print(f"     SAFE  BondFunctions.duration(Modified): "
              f"{live['mod_duration_no_schedule']:.8f} vs "
              f"{live['mod_duration_with_schedule']:.8f}")
        print(f"     BITES yearFraction({PREV}, {SETTLE}) called directly: "
              f"{live['isma_no_ref']:.8f} vs {live['isma_with_ref']:.8f} -> "
              f"{FACE * CPN * (live['isma_with_ref'] - live['isma_no_ref']):+,.2f} of accrued")
        print(f"     BITES InterestRate.compoundFactor(d1, d2): "
              f"{live['compound_factor_isma']:.10f} vs ACT/365F "
              f"{live['compound_factor_act365']:.10f} "
              f"({1e4 * (live['compound_factor_act365'] / live['compound_factor_isma'] - 1):+.1f} "
              f"bp of price)")
        print(f"     BITES FlatForward built on it: t({SETTLE}) = "
              f"{live['curve_t_isma']:.8f} vs ACT/365F {live['curve_t_act365']:.8f}; "
              f"DF {live['curve_df_isma']:.10f} vs {live['curve_df_act365']:.10f} "
              f"({1e4 * (live['curve_df_isma'] / live['curve_df_act365'] - 1):+.1f} bp)")

    print("\n" + "=" * W)
    print("RULE: accrued is (day count, coupon schedule, settlement date) -- ACT/ACT ICMA")
    print("      without its schedule is a guess, the 30/360 name alone is ambiguous, and")
    print("      the quote is clean while the cash is dirty.")
    print("=" * W)
