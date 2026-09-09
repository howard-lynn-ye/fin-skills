"""Compounded-in-arrears SOFR: the index, the averages, and the four ways to be a few basis
points wrong without anything raising.

WHY this exists: a compounded overnight rate is not an average of the fixings. It is a product
of daily accrual factors, each weighted by the number of CALENDAR days that fixing applies for,
on a 360-day year for SOFR and a 365-day year for SONIA and TONA -- and then a lookback,
lockout, payment delay or observation shift moves which fixings are in the window at all.

Four independent errors, all measured below, none of which raises:

  * COMPOUNDED vs SIMPLE AVERAGE. The ARRC and ISDA conventions compound. A simple average of
    the same fixings is lower by roughly r^2 * T / 2.

  * UNWEIGHTED vs DAY-WEIGHTED FIXINGS. A Friday fixing applies for three calendar days, and a
    fixing before a holiday for more. Averaging the business-day fixings equally silently
    reweights every day of the quarter.

  * ACT/360 vs ACT/365. The same compound factor annualized on the wrong basis is off by the
    365/360 factor -- SOFR and ESTR are ACT/360, SONIA and TONA are ACT/365.

  * LOOKBACK WITH vs WITHOUT AN OBSERVATION SHIFT. A "5-day lookback" can mean two different
    windows with two different sets of day weights. Both are in use and both are called a
    lookback.

The NY Fed's own published SOFR Index worked example is reproduced exactly, so the compounding
arithmetic is checked against the administrator and not against itself. QuantLib's
OvernightIndexedCoupon, when importable, checks the lookback / lockout / observation-shift side.

Usage:
    from sofr import compounded_rate, sofr_index_path, observation_window, synthetic_sofr
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Sequence

import numpy as np

# NY Fed, "Additional Information about Reference Rates Administered by the New York Fed"
# (read 2026-09-09): the SOFR Index starts at 1.00000000 on 2018-04-02 and compounds by
# (1 + SOFR * d/360) where d is the number of CALENDAR days that fixing applies for. The
# page's own worked example, verbatim:
NYFED_INDEX_EXAMPLE = (
    # (SOFR value date, SOFR, calendar days applicable, published index after compounding)
    (date(2018, 4, 2), 0.0180, 1, 1.00005000),
    (date(2018, 4, 3), 0.0183, 1, 1.00010084),
    (date(2018, 4, 4), 0.0174, 1, 1.00014917),
    (date(2018, 4, 5), 0.0175, 1, 1.00019779),
    (date(2018, 4, 6), 0.0175, 3, 1.00034365),
)
SOFR_BASIS = 360.0          # SOFR and ESTR
SONIA_BASIS = 365.0         # SONIA and TONA

# QuantLib 1.43 ships a dedicated UnitedStates::SOFR fixing calendar. Read out of it on
# 2026-09-09, these are its 2026 weekday holidays; it is UnitedStates::GovernmentBond PLUS
# Good Friday (2026-04-03), because SIFMA recommends a full close and no SOFR is published.
SOFR_CALENDAR_HOLIDAYS_2026 = (
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 10, 12), date(2026, 11, 11), date(2026, 11, 26), date(2026, 12, 25),
)

# Day count basis by currency. The ACT/360 vs ACT/365 split is the one that silently changes
# the number. Each basis is read out of QuantLib 1.43's own index definitions (see
# `quantlib_index_day_counts`), which is why they are stated here rather than recalled.
RFR_CONVENTIONS = (
    # (rate, currency, administrator, day count basis, published index base value)
    ("SOFR", "USD", "NY Fed", 360.0, "index 1.00000000 from 2018-04-02"),
    ("SONIA", "GBP", "Bank of England", 365.0, "index 100.00000000 from 2018-04-23"),
    ("ESTR", "EUR", "ECB", 360.0, "-"),
    ("TONA", "JPY", "Bank of Japan", 365.0, "-"),
    ("SARON", "CHF", "SIX", 360.0, "-"),
)


# ------------------------------------------------------------------ calendars
def business_days(start: date, end: date, holidays: Iterable[date] = ()) -> list[date]:
    """Business days in [start, end): weekdays that are not holidays."""
    hol = set(holidays)
    out, d = [], start
    while d < end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


def shift_business_days(d: date, n: int, holidays: Iterable[date] = ()) -> date:
    hol = set(holidays)
    step = 1 if n >= 0 else -1
    left, cur = abs(int(n)), d
    while left:
        cur += timedelta(days=step)
        if cur.weekday() < 5 and cur not in hol:
            left -= 1
    return cur


def calendar_day_weights(dates: Sequence[date], period_end: date) -> list[int]:
    """How many calendar days each fixing applies for: to the NEXT business day, or to the end
    of the period for the last one. This is the NY Fed's 'calendar days applicable'."""
    if not dates:
        raise ValueError("no fixing dates")
    nxt = list(dates[1:]) + [period_end]
    w = [(b - a).days for a, b in zip(dates, nxt)]
    if any(x <= 0 for x in w):
        raise ValueError("fixing dates must be strictly increasing and inside the period")
    return w


# ------------------------------------------------------------------ compounding
def compound_factor(rates: Sequence[float], weights: Sequence[int],
                    basis: float = SOFR_BASIS) -> float:
    """prod(1 + r_i * d_i / basis) -- the SOFR Index arithmetic."""
    if len(rates) != len(weights):
        raise ValueError("rates and weights must be the same length")
    f = 1.0
    for r, d in zip(rates, weights):
        f *= 1.0 + r * d / basis
    return f


def compounded_rate(rates: Sequence[float], weights: Sequence[int],
                    basis: float = SOFR_BASIS) -> float:
    """The annualized compounded-in-arrears rate: (prod - 1) * basis / total days."""
    total = float(sum(weights))
    return (compound_factor(rates, weights, basis) - 1.0) * basis / total


def simple_average_rate(rates: Sequence[float], weights: Sequence[int]) -> float:
    """Day-weighted SIMPLE average -- no compounding. Correct arithmetic, wrong convention."""
    total = float(sum(weights))
    return float(sum(r * d for r, d in zip(rates, weights)) / total)


def unweighted_average_rate(rates: Sequence[float]) -> float:
    """The mean of the business-day fixings. Ignores that a Friday counts for three days."""
    return float(np.mean(np.asarray(rates, dtype=float)))


def mixed_basis_rate(rates: Sequence[float], weights: Sequence[int],
                     accrual_basis: float = SOFR_BASIS,
                     annualization_basis: float = SONIA_BASIS) -> float:
    """The basis appears TWICE -- inside every daily accrual factor and again in the
    annualization -- and changing only one of them is the ACT/360-vs-ACT/365 trap.

    Used consistently the two bases nearly cancel; mixed, the error is the full 365/360.
    """
    total = float(sum(weights))
    return ((compound_factor(rates, weights, accrual_basis) - 1.0)
            * annualization_basis / total)


def sofr_index_path(rates: Sequence[float], weights: Sequence[int], start: float = 1.0,
                    basis: float = SOFR_BASIS, dp: int | None = 8) -> list[float]:
    """The published index after each fixing. The NY Fed publishes it to eight decimals."""
    out, idx = [], start
    for r, d in zip(rates, weights):
        idx *= 1.0 + r * d / basis
        out.append(round(idx, dp) if dp is not None else idx)
    return out


def rate_from_index(index_start: float, index_end: float, days: int,
                    basis: float = SOFR_BASIS) -> float:
    """(I_end / I_start - 1) * basis / days -- the NY Fed's own way to get any tenor."""
    if index_start <= 0 or days <= 0:
        raise ValueError("index_start and days must be positive")
    return (index_end / index_start - 1.0) * basis / days


# ------------------------------------------------------------------ observation windows
def observation_window(accrual_start: date, accrual_end: date, holidays: Iterable[date] = (),
                       lookback: int = 0, lockout: int = 0,
                       observation_shift: bool = False) -> dict:
    """Which fixings enter the coupon, and what each one is weighted by.

    lookback (no shift)  the fixing used on accrual day d is the one k business days before d,
                         weighted by the ACCRUAL period's day counts ("rate shift")
    observation shift    the whole window moves back k business days and the OBSERVATION
                         period's own day counts are used
    lockout              the last `lockout` business days all reuse the last observable fixing
    """
    if lookback < 0 or lockout < 0:
        raise ValueError("lookback and lockout must not be negative")
    if observation_shift and lookback == 0:
        raise ValueError("an observation shift needs a lookback")
    if observation_shift:
        obs_start = shift_business_days(accrual_start, -lookback, holidays)
        obs_end = shift_business_days(accrual_end, -lookback, holidays)
        fixing_dates = business_days(obs_start, obs_end, holidays)
        weights = calendar_day_weights(fixing_dates, obs_end)
    else:
        accrual_days = business_days(accrual_start, accrual_end, holidays)
        fixing_dates = [shift_business_days(d, -lookback, holidays) if lookback else d
                        for d in accrual_days]
        weights = calendar_day_weights(accrual_days, accrual_end)
    if lockout:
        if lockout >= len(fixing_dates):
            raise ValueError("lockout is longer than the observation window")
        frozen = fixing_dates[-lockout - 1]
        fixing_dates = fixing_dates[:-lockout] + [frozen] * lockout
    return {"fixing_dates": fixing_dates, "weights": weights,
            "total_days": sum(weights), "n_fixings": len(fixing_dates)}


def rate_for_window(fixings: dict, window: dict, basis: float = SOFR_BASIS) -> float:
    rates = [fixings[d] for d in window["fixing_dates"]]
    return compounded_rate(rates, window["weights"], basis)


# ------------------------------------------------------------------ synthetic data
def synthetic_sofr(start: date, end: date, holidays: Iterable[date] = (), seed: int = 0,
                   level: float = 0.0430, month_end_bump: float = 0.0020,
                   quarter_end_bump: float = 0.0035,
                   policy_date: date | None = None, policy_step: float = 0.0025) -> dict:
    """A seeded SOFR path with the two features that make the window conventions matter:

      * month-end and quarter-end spikes that last exactly ONE business day, so equal
        weighting and calendar-day weighting disagree;
      * a policy step on `policy_date`, so moving the window five business days moves real
        money rather than noise.
    """
    rng = np.random.default_rng(seed)
    days = business_days(start, end, holidays)
    r = level
    out: dict = {}
    for i, d in enumerate(days):
        r += 0.02 * (level - r) + rng.normal(0.0, 0.00004)
        nxt = days[i + 1] if i + 1 < len(days) else end
        bump = 0.0
        if nxt.month != d.month:                 # last business day of the month
            bump = quarter_end_bump if d.month in (3, 6, 9, 12) else month_end_bump
        step = policy_step if (policy_date is not None and d >= policy_date) else 0.0
        out[d] = round(r + bump + step, 4)       # SOFR is published to the nearest basis point
    return out


# ------------------------------------------------------------------ QuantLib
def quantlib_cross_checks(fixings: dict, accrual_start: date, accrual_end: date,
                          holidays: Sequence[date] = ()) -> dict | None:
    """Compare against ql.OvernightIndexedCoupon under Compound and Simple averaging, with and
    without a 5-day lookback and an observation shift. Returns None if QuantLib is absent."""
    try:
        import QuantLib as ql
    except ImportError:
        return None

    def q(d: date):
        return ql.Date(d.day, d.month, d.year)

    ql.Settings.instance().evaluationDate = q(accrual_end)
    index = ql.Sofr(ql.YieldTermStructureHandle())
    cal = index.fixingCalendar()
    index.clearFixings()
    # If the hard-coded holiday list and QuantLib's SOFR fixing calendar ever disagree, the
    # comparison below would be between two different day-weight schemes. Say so instead.
    mismatch = [d for d in fixings if cal.isBusinessDay(q(d)) == (d in set(holidays))]
    for d, r in sorted(fixings.items()):
        if cal.isBusinessDay(q(d)):
            index.addFixing(q(d), r, True)
    out: dict = {"version": ql.__version__, "calendar": cal.name(),
                 "calendar_mismatches": sorted(mismatch)}

    def rate(**kw):
        cpn = ql.OvernightIndexedCoupon(q(accrual_end), 1_000_000.0, q(accrual_start),
                                        q(accrual_end), index, **kw)
        return cpn.rate()

    out["compounded"] = rate()
    out["simple"] = rate(averagingMethod=ql.RateAveraging.Simple)
    out["lookback5"] = rate(lookbackDays=5)
    out["lookback5_shift"] = rate(lookbackDays=5, applyObservationShift=True)
    out["lockout5"] = rate(lockoutDays=5)
    out["day_counts"] = quantlib_index_day_counts()
    return out


def quantlib_index_day_counts() -> dict | None:
    """The day count each RFR index carries in QuantLib's own definitions -- the ACT/360 vs
    ACT/365 split of `RFR_CONVENTIONS`, read from the library rather than recalled."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    h = ql.YieldTermStructureHandle()
    out = {}
    for label, attr in (("SOFR", "Sofr"), ("SONIA", "Sonia"), ("ESTR", "Estr"),
                        ("TONA", "Tonar"), ("SARON", "Saron")):
        if hasattr(ql, attr):
            out[label] = getattr(ql, attr)(h).dayCounter().name()
    return out


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    W = 92
    START, END = date(2026, 4, 1), date(2026, 7, 1)
    HOLIDAYS = list(SOFR_CALENDAR_HOLIDAYS_2026)
    POLICY = date(2026, 6, 18)                            # a 25 bp step inside the quarter
    NOTIONAL = 100_000_000.0
    # SOFR is published one business day in arrears, so a lookback needs fixings from before
    # the accrual period starts.
    FIXINGS = synthetic_sofr(date(2026, 1, 2), END, HOLIDAYS, seed=7, policy_date=POLICY)

    print("=" * W)
    print(f"SOFR COMPOUNDED IN ARREARS -- accrual {START} to {END}, {NOTIONAL:,.0f} notional")
    print("=" * W)

    print("\n1. THE NY FED'S OWN PUBLISHED SOFR INDEX EXAMPLE, REPRODUCED")
    rates = [r for _, r, _, _ in NYFED_INDEX_EXAMPLE]
    wts = [d for _, _, d, _ in NYFED_INDEX_EXAMPLE]
    mine = sofr_index_path(rates, wts, 1.0)
    print(f"   {'value date':<14}{'SOFR':>8}{'cal days':>10}{'NY Fed index':>16}{'this file':>16}"
          f"{'|diff|':>10}")
    worst = 0.0
    for (d, r, w, published), got in zip(NYFED_INDEX_EXAMPLE, mine):
        worst = max(worst, abs(got - published))
        print(f"   {str(d):<14}{r:>8.2%}{w:>10d}{published:>16.8f}{got:>16.8f}"
              f"{abs(got - published):>10.1e}")
    print(f"   -> worst |diff| = {worst:.1e}; the index starts at 1.00000000 on 2018-04-02 and "
          f"compounds by (1 + SOFR x d/360)")
    print(f"   -> note the last row: a FRIDAY fixing carries 3 calendar days, not 1")

    base = observation_window(START, END, HOLIDAYS)
    r_series = [FIXINGS[d] for d in base["fixing_dates"]]
    w_series = base["weights"]
    print(f"\n   the accrual period holds {base['n_fixings']} business-day fixings covering "
          f"{base['total_days']} calendar days")

    print("\n2. FOUR WAYS TO BE A FEW BASIS POINTS WRONG")
    comp = compounded_rate(r_series, w_series, SOFR_BASIS)
    simple = simple_average_rate(r_series, w_series)
    unweighted = unweighted_average_rate(r_series)
    act365 = compounded_rate(r_series, w_series, SONIA_BASIS)
    mixed = mixed_basis_rate(r_series, w_series, SOFR_BASIS, SONIA_BASIS)
    rows = [("compounded in arrears, day-weighted, ACT/360  (CORRECT)", comp),
            ("simple day-weighted average instead of compounding", simple),
            ("compounded but with UNWEIGHTED fixings", compounded_rate(
                r_series, [1] * len(r_series), SOFR_BASIS)),
            ("mean of the fixings (no weights, no compounding)", unweighted),
            ("ACT/365 used CONSISTENTLY (accrual and annualization)", act365),
            ("ACT/360 accrual annualized on 365 (the MIXED basis)", mixed)]
    print(f"   {'method':<54}{'rate':>11}{'bp vs correct':>15}{'per 100mm/qtr':>16}")
    for label, r in rows:
        bp = 1e4 * (r - comp)
        cash = NOTIONAL * (r - comp) * base["total_days"] / SOFR_BASIS
        print(f"   {label:<54}{r:>11.6%}{bp:>+15.2f}{cash:>+16,.0f}")
    print(f"   -> the basis appears TWICE. Changed in both places it nearly cancels "
          f"({1e4 * (act365 - comp):+.2f} bp); changed in one it is the full 365/360 "
          f"({1e4 * (mixed - comp):+.2f} bp)")

    print("\n3. LOOKBACK, LOCKOUT AND OBSERVATION SHIFT ARE FOUR DIFFERENT WINDOWS")
    variants = [("plain in-arrears", dict()),
                ("5-day lookback (rate shift)", dict(lookback=5)),
                ("5-day lookback + observation shift", dict(lookback=5,
                                                            observation_shift=True)),
                ("5-day lockout", dict(lockout=5))]
    print(f"   {'convention':<38}{'fixings':>9}{'cal days':>10}{'rate':>12}{'bp vs plain':>13}"
          f"{'per 100mm':>13}")
    plain = None
    for label, kw in variants:
        win = observation_window(START, END, HOLIDAYS, **kw)
        r = rate_for_window(FIXINGS, win, SOFR_BASIS)
        plain = r if plain is None else plain
        bp = 1e4 * (r - plain)
        cash = NOTIONAL * (r - plain) * win["total_days"] / SOFR_BASIS
        print(f"   {label:<38}{win['n_fixings']:>9}{win['total_days']:>10}{r:>12.6%}"
              f"{bp:>+13.2f}{cash:>+13,.0f}")
    lb = rate_for_window(FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5),
                         SOFR_BASIS)
    lbs = rate_for_window(FIXINGS, observation_window(START, END, HOLIDAYS, lookback=5,
                                                      observation_shift=True), SOFR_BASIS)
    print(f"   -> lookback WITH vs WITHOUT the observation shift: {1e4 * (lbs - lb):+.2f} bp "
          f"({NOTIONAL * (lbs - lb) * base['total_days'] / SOFR_BASIS:+,.0f} per 100mm), and "
          f"BOTH are called a '5-day lookback'")
    print("   -> the shift moves the DAY WEIGHTS as well as the rates; that is the whole "
          "difference")

    print("\n4. THE SOFR INDEX GIVES THE SAME ANSWER, AND IS THE ONLY THING YOU NEED TO STORE")
    idx = sofr_index_path(r_series, w_series, 1.0, SOFR_BASIS, dp=None)
    exact = rate_from_index(1.0, idx[-1], base["total_days"], SOFR_BASIS)
    idx_r = sofr_index_path(r_series, w_series, 1.0, SOFR_BASIS, dp=8)
    rounded = rate_from_index(1.0, idx_r[-1], base["total_days"], SOFR_BASIS)
    print(f"   compounded from the fixings   {comp:.10%}")
    print(f"   from an UNROUNDED index       {exact:.10%}   |diff| {abs(exact - comp):.1e}")
    print(f"   from the PUBLISHED 8-dp index {rounded:.10%}   |diff| {abs(rounded - comp):.1e} "
          f"= {1e4 * abs(rounded - comp):.4f} bp")
    print("   -> two index values and a day count reproduce any tenor; the NY Fed notes that")
    print("      averages built from the rounded index can differ in the fifth decimal place")

    print("\n5. THE SAME ARITHMETIC, FIVE CURRENCIES, TWO DAY-COUNT BASES")
    dcs = quantlib_index_day_counts() or {}
    print(f"   {'rate':<8}{'ccy':>5}{'administrator':>18}{'basis':>9}"
          f"{'QuantLib index day count':>28}{'rate on this path':>20}")
    for name, ccy, admin, basis, pub in RFR_CONVENTIONS:
        r = compounded_rate(r_series, w_series, basis)
        print(f"   {name:<8}{ccy:>5}{admin:>18}{f'ACT/{basis:.0f}':>9}"
              f"{dcs.get(name, '(QuantLib absent)'):>28}{r:>20.6%}")
    for name, ccy, admin, basis, pub in RFR_CONVENTIONS:
        if pub != "-":
            print(f"   {name}: {pub}")
    print(f"   -> a consistent ACT/365 reads {1e4 * (act365 - comp):+.2f} bp against ACT/360 on "
          f"the SAME fixings, but the MIXED basis reads {1e4 * (mixed - comp):+.2f} bp -- the "
          f"fixings never changed, only where the basis was applied")

    print("\n6. CROSS-CHECK AGAINST QUANTLIB")
    live = quantlib_cross_checks(FIXINGS, START, END, HOLIDAYS)
    if live is None:
        print("   QuantLib not installed - the NY Fed example in section 1 is the check")
    else:
        print(f"   QuantLib {live['version']}  (ql.Sofr + OvernightIndexedCoupon)")
        pairs = [("compounded", comp, "compounded in arrears"),
                 ("simple", simple, "simple average"),
                 ("lookback5", lb, "5-day lookback"),
                 ("lookback5_shift", lbs, "5-day lookback + observation shift")]
        for key, mine_r, label in pairs:
            print(f"   {label:<38}QuantLib {live[key]:.8%}   mine {mine_r:.8%}   "
                  f"|diff| {1e4 * abs(live[key] - mine_r):.4f} bp")
        print(f"   {'5-day lockout':<38}QuantLib {live['lockout5']:.8%}")
        print(f"   fixing calendar: {live['calendar']}; disagreements with this file's "
              f"hard-coded 2026 holiday list: {len(live['calendar_mismatches'])}")
        print(f"   -> QuantLib's Compound vs Simple averaging differs by "
              f"{1e4 * (live['compounded'] - live['simple']):+.2f} bp, the same effect as "
              f"section 2")

    print("\n" + "=" * W)
    print("RULE: a compounded RFR is a PRODUCT of (1 + r_i x d_i / basis), weighted by CALENDAR")
    print("      days -- ACT/360 for SOFR and ESTR, ACT/365 for SONIA and TONA -- and a")
    print("      'lookback' names two different windows until you say whether it shifts.")
    print("=" * W)
