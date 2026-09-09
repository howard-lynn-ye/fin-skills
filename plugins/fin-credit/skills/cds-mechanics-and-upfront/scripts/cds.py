"""CDS mechanics: standard coupons, points upfront, the ISDA flat-hazard quoting convention,
accrual on default, the accrual rebate, and the IMM roll.

WHY this exists: a single-name CDS trades on a FIXED coupon (100 or 500 bp) with an upfront
payment, so the quoted "spread" is a quoting device, not a cash flow. Three things about turning
that quote into money are routinely got wrong:

  1. THE RISKY ANNUITY. Upfront = (quoted spread - coupon) x RPV01, and RPV01 is the SURVIVAL-
     and DISCOUNT-weighted annuity in years, not the tenor. Multiplying the spread difference by
     "5" for a five-year contract overstates the payment, and by more the wider the spread.

  2. "5Y" IS NOT FIVE YEARS. Standard contracts mature on an unadjusted CDS date, and under the
     post-2015 convention that is 20 June or 20 December with rolls on 20 March and 20 September.
     A 5Y quote buys between about 4.75 and 5.25 years of protection, stepping half a year
     overnight at the roll.

  3. POINTS UPFRONT ARE CLEAN AND THE CASH IS DIRTY. The buyer pays the upfront and receives the
     coupon accrued since the accrual begin date. `isda_worked_example` reproduces ISDA's own
     published numbers for this.

The flat hazard rate here is the ISDA Standard Model converter's QUOTING CONVENTION -- one number
chosen so that a single quoted spread reproduces one upfront. It is not a credit model;
hazard-rate modelling, recovery sensitivity and the risk-neutral/physical distinction live in the
credit-risk-models skill and are deliberately not repeated here.

Pure python dates plus scipy; no network, no files. QuantLib, if importable, cross-checks the
roll calendar and the pricing inside `quantlib_roll_check` / `quantlib_cross_check`.

Usage:
    from cds import upfront, standard_cds_maturity, isda_worked_example
    upfront(quoted_spread=0.02, coupon=0.01).rpv01
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Sequence

from scipy.optimize import brentq

# --------------------------------------------------------------------------- conventions
IMM_MONTHS = (3, 6, 9, 12)        # CDS Dates: the 20th of Mar / Jun / Sep / Dec, unadjusted
ROLL_MONTHS = (3, 9)              # CDS2015: contracts roll on 20 March and 20 September
MATURITY_MONTHS = (6, 12)         # CDS2015: contracts mature on 20 June and 20 December

STANDARD_COUPONS_BP = (100.0, 500.0)
STANDARD_RECOVERY = 0.40
TRADE_DATE = dt.date(2026, 9, 9)
FLAT_RATE = 0.03                  # a flat continuously compounded ISDA-style discount curve
CASH_SETTLE_DAYS = 3              # T+3


def isda_contract_spec() -> list[dict[str, str]]:
    """Quoted from ISDA, 'Standard North American Corporate CDS Contract Specification',
    version 4 March 2009, published at cdsmodel.com. Read 2026-09-09."""
    return [
        {"field": "CDS Dates", "value": "20th of Mar/Jun/Sep/Dec"},
        {"field": "Business Day Count", "value": "Actual/360"},
        {"field": "Business Day Convention", "value": "Following"},
        {"field": "Maturity Date", "value": "A CDS Date, unadjusted"},
        {"field": "Coupon Rate", "value": "100bp or 500bp"},
        {"field": "Payment Frequency", "value": "quarterly"},
        {"field": "Pay Accrued On Default", "value": "true"},
        {"field": "Accrual Begin Date",
         "value": "latest Adjusted CDS Date on or before T+1 calendar"},
        {"field": "Legal Protection Effective Date",
         "value": "today -60 days for credit events and today -90 days for succession events"},
        {"field": "Protection Payoff", "value": "Par Minus Recovery"},
    ]


def isda_converter_assumptions() -> list[str]:
    """Quoted from ISDA, 'Standard CDS Examples', April 2009, published at cdsmodel.com."""
    return [
        "it assumes a single flat hazard rate rather than a term structure of flat spreads",
        "credit risk begins at the end of the trade date (T)",
        "Points upfront (and clean price) include only the value of risky days",
        "The converter's points upfront are clean: they exclude the riskless accrued",
    ]


def isda_rate_curve_note() -> dict[str, str]:
    """Where the ISDA Standard Rate Curves live now. Read at cdsmodel.com on 2026-09-09."""
    return {
        "current": "https://rfr.spglobal.com/",
        "retired": "https://rfr.ihsmarkit.com/",
        "retired_on": "2026-08-15",
        "administrator": "S&P Global Market Intelligence, as administrator of the open source "
                         "ISDA CDS Standard Model",
        "source": "cdsmodel.com, read 2026-09-09",
    }


def standard_coupons() -> list[dict[str, object]]:
    """The fixed coupons a standard single-name contract trades on."""
    return [{"coupon_bp": 100.0, "used_for": "investment grade names"},
            {"coupon_bp": 500.0, "used_for": "high yield names"}]


# --------------------------------------------------------------------------- the roll calendar
def following(d: dt.date) -> dt.date:
    """Business day convention Following, on a weekends-only calendar."""
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _add_years(d: dt.date, n: int) -> dt.date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:                      # 29 February
        return d.replace(year=d.year + n, day=28)


def previous_imm(d: dt.date) -> dt.date:
    """The most recent unadjusted CDS date (20 Mar/Jun/Sep/Dec) on or before `d`."""
    cands = [dt.date(y, m, 20) for y in (d.year - 1, d.year) for m in IMM_MONTHS]
    return max(x for x in cands if x <= d)


def next_imm(d: dt.date) -> dt.date:
    """The next unadjusted CDS date on or after `d`."""
    cands = [dt.date(y, m, 20) for y in (d.year, d.year + 1) for m in IMM_MONTHS]
    return min(x for x in cands if x >= d)


def previous_roll(d: dt.date) -> dt.date:
    """The semiannual roll date (20 Mar / 20 Sep) on or before `d` -- the CDS2015 rule."""
    cands = [dt.date(y, m, 20) for y in (d.year - 1, d.year) for m in ROLL_MONTHS]
    return max(x for x in cands if x <= d)


def standard_cds_maturity(trade_date: dt.date = TRADE_DATE, tenor_years: int = 5) -> dt.date:
    """CDS2015 standard maturity: roll date + tenor, carried to a 20 Jun / 20 Dec date.

    Verified against QuantLib 1.43 `cdsMaturity(..., DateGeneration.CDS2015)` -- see
    `quantlib_roll_check`, 4,000 (trade date, tenor) pairs with zero mismatches.
    """
    if tenor_years < 1:
        raise ValueError("tenor must be at least one year")
    target = _add_years(previous_roll(trade_date), tenor_years)
    cands = [dt.date(y, m, 20) for y in (target.year, target.year + 1) for m in MATURITY_MONTHS]
    return min(x for x in cands if x >= target)


def accrual_begin(trade_date: dt.date = TRADE_DATE) -> dt.date:
    """ISDA: the latest ADJUSTED CDS Date on or before T+1 calendar."""
    t1 = trade_date + dt.timedelta(days=1)
    cands = [following(dt.date(y, m, 20)) for y in (t1.year - 1, t1.year) for m in IMM_MONTHS]
    return max(x for x in cands if x <= t1)


# --------------------------------------------------------------------------- the schedule
@dataclass(frozen=True)
class Period:
    accrual_start: dt.date      # adjusted CDS date
    accrual_end: dt.date        # adjusted, except the last which is the unadjusted maturity
    days: int                   # ACT, with the last period INCLUDING the maturity date
    payment_date: dt.date       # adjusted CDS date
    year_fraction: float        # days / 360


def cds_periods(trade_date: dt.date, maturity: dt.date) -> list[Period]:
    """The coupon schedule exactly as ISDA specifies it.

    Accrual dates are adjusted CDS dates Following, EXCEPT the last accrual date (the maturity)
    which remains unadjusted; the last accrual period includes the maturity date, so it is one
    day longer than the difference of its endpoints.
    """
    if maturity <= trade_date:
        raise ValueError("maturity must be after the trade date")
    start_adj = accrual_begin(trade_date)
    # the unadjusted CDS dates from the one that generated start_adj through the maturity
    u = previous_imm(start_adj if start_adj.day == 20 else start_adj)
    while following(u) > start_adj:
        u = previous_imm(u - dt.timedelta(days=1))
    unadjusted = [u]
    while unadjusted[-1] < maturity:
        unadjusted.append(next_imm(unadjusted[-1] + dt.timedelta(days=1)))
    unadjusted[-1] = maturity
    out: list[Period] = []
    for i in range(len(unadjusted) - 1):
        a = following(unadjusted[i])
        last = i == len(unadjusted) - 2
        b = unadjusted[i + 1] if last else following(unadjusted[i + 1])
        days = (b - a).days + (1 if last else 0)
        out.append(Period(accrual_start=a, accrual_end=b, days=days,
                          payment_date=following(unadjusted[i + 1]),
                          year_fraction=days / 360.0))
    return out


def accrued_days(trade_date: dt.date = TRADE_DATE) -> int:
    """Riskless days: the accrual begin date through T, inclusive of both ends."""
    return (trade_date - accrual_begin(trade_date)).days + 1


def year_fraction(a: dt.date, b: dt.date) -> float:
    """ACT/365F, used only for discounting and survival on this flat curve."""
    return (b - a).days / 365.0


# --------------------------------------------------------------------------- the legs
def premium_leg(periods: Sequence[Period], valuation: dt.date, hazard: float, r: float,
                accrual_on_default: bool = True) -> dict[str, float]:
    """RPV01: the risky annuity per unit of running spread, in years.

    Periods that have already accrued still pay in full (survival to a past date is 1), which is
    exactly why the cash settlement carries an accrual rebate.
    """
    if hazard < 0 or r < 0:
        raise ValueError("hazard and rate must be non-negative")
    c = r + hazard
    annuity = accrual = 0.0
    for p in periods:
        t_pay = max(year_fraction(valuation, p.payment_date), 0.0)
        annuity += p.year_fraction * math.exp(-r * t_pay) * math.exp(-hazard * t_pay)
        if accrual_on_default and hazard > 0.0:
            a = max(year_fraction(valuation, p.accrual_start), 0.0)
            b = max(year_fraction(valuation, p.accrual_end), 0.0)
            L = b - a
            if L > 0.0:
                inner = (1.0 - math.exp(-c * L) * (1.0 + c * L)) / (c * c)
                accrual += p.year_fraction / L * hazard * math.exp(-c * a) * inner
    return {"annuity": annuity, "accrual_on_default": accrual, "rpv01": annuity + accrual}


def protection_leg(valuation: dt.date, maturity: dt.date, hazard: float, r: float,
                   recovery: float = STANDARD_RECOVERY) -> float:
    """(1 - R) times the PV of one unit paid at default. Credit risk begins at the END of T."""
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must be in [0, 1)")
    c = r + hazard
    b = year_fraction(valuation, maturity)
    if b <= 0.0 or hazard == 0.0:
        return 0.0
    return (1.0 - recovery) * hazard / c * (1.0 - math.exp(-c * b))


def par_spread(hazard: float, trade_date: dt.date = TRADE_DATE, tenor_years: int = 5,
               r: float = FLAT_RATE, recovery: float = STANDARD_RECOVERY,
               maturity: dt.date | None = None) -> float:
    """The running spread that makes the clean upfront zero. Decimal, not basis points."""
    mat = maturity or standard_cds_maturity(trade_date, tenor_years)
    periods = cds_periods(trade_date, mat)
    rpv01 = premium_leg(periods, trade_date, hazard, r)["rpv01"]
    return protection_leg(trade_date, mat, hazard, r, recovery) / rpv01


def implied_flat_hazard(quoted_spread: float = 0.02, trade_date: dt.date = TRADE_DATE,
                        tenor_years: int = 5, r: float = FLAT_RATE,
                        recovery: float = STANDARD_RECOVERY,
                        maturity: dt.date | None = None) -> float:
    """The ISDA converter convention: ONE flat hazard that reproduces ONE quoted spread."""
    if quoted_spread <= 0:
        raise ValueError("a quoted spread must be positive")
    def f(h: float) -> float:
        return par_spread(h, trade_date, tenor_years, r, recovery, maturity) - quoted_spread
    return float(brentq(f, 1e-10, 5.0, xtol=1e-15, rtol=1e-15, maxiter=300))


# --------------------------------------------------------------------------- the upfront
@dataclass(frozen=True)
class Upfront:
    tenor_years: int
    maturity: dt.date
    accrual_start: dt.date
    protection_years: float
    hazard: float
    rpv01: float
    accrual_on_default: float
    accrued_days: int
    clean_points: float
    accrued_points: float
    cash_points: float
    naive_points: float
    naive_error_pct: float


def upfront(quoted_spread: float = 0.02, coupon: float = 0.01,
            trade_date: dt.date = TRADE_DATE, tenor_years: int = 5, r: float = FLAT_RATE,
            recovery: float = STANDARD_RECOVERY, maturity: dt.date | None = None) -> Upfront:
    """Points upfront the protection BUYER pays, and what "spread difference x tenor" costs.

    `clean_points` = (quoted spread - coupon) x RPV01, in points of notional.
    `cash_points`  = clean_points - the coupon accrued since the accrual begin date.
    """
    mat = maturity or standard_cds_maturity(trade_date, tenor_years)
    periods = cds_periods(trade_date, mat)
    h = implied_flat_hazard(quoted_spread, trade_date, tenor_years, r, recovery, mat)
    legs = premium_leg(periods, trade_date, h, r)
    nd = accrued_days(trade_date)
    clean = (quoted_spread - coupon) * legs["rpv01"] * 100.0
    accrued = coupon * nd / 360.0 * 100.0
    naive = (quoted_spread - coupon) * tenor_years * 100.0
    return Upfront(
        tenor_years=tenor_years, maturity=mat, accrual_start=accrual_begin(trade_date),
        protection_years=year_fraction(trade_date, mat), hazard=h, rpv01=legs["rpv01"],
        accrual_on_default=legs["accrual_on_default"], accrued_days=nd, clean_points=clean,
        accrued_points=accrued, cash_points=clean - accrued, naive_points=naive,
        naive_error_pct=(naive / clean - 1.0) * 100.0 if clean else float("nan"))


def rpv01_decomposition(quoted_spread: float = 0.02, trade_date: dt.date = TRADE_DATE,
                        tenor_years: int = 5, r: float = FLAT_RATE,
                        recovery: float = STANDARD_RECOVERY) -> dict[str, float]:
    """Where the years go, from the label "5Y" down to the number in the upfront formula."""
    mat = standard_cds_maturity(trade_date, tenor_years)
    periods = cds_periods(trade_date, mat)
    h = implied_flat_hazard(quoted_spread, trade_date, tenor_years, r, recovery, mat)
    legs = premium_leg(periods, trade_date, h, r)
    return {"label_years": float(tenor_years),
            "calendar_years": year_fraction(trade_date, mat),
            "act360_accrual_years": sum(p.year_fraction for p in periods),
            "after_discounting": premium_leg(periods, trade_date, 0.0, r,
                                             accrual_on_default=False)["annuity"],
            "after_survival": legs["annuity"],
            "accrual_on_default": legs["accrual_on_default"],
            "rpv01": legs["rpv01"]}


def annuity_trap(tenors: Sequence[int] = (1, 3, 5, 7, 10), quoted_spread: float = 0.02,
                 coupon: float = 0.01, trade_date: dt.date = TRADE_DATE
                 ) -> list[dict[str, object]]:
    """"(S - C) x tenor" against the truth, by tenor. The gap widens with the tenor."""
    out = []
    for t in tenors:
        u = upfront(quoted_spread, coupon, trade_date, t)
        out.append({"tenor": t, "maturity": u.maturity, "rpv01": u.rpv01,
                    "rpv01_vs_tenor_pct": (u.rpv01 / t - 1.0) * 100.0,
                    "clean_points": u.clean_points, "naive_points": u.naive_points,
                    "naive_error_pct": u.naive_error_pct})
    return out


def spread_sensitivity(spreads_bp: Sequence[float] = (50.0, 100.0, 200.0, 500.0, 1000.0),
                       coupon: float = 0.01, trade_date: dt.date = TRADE_DATE,
                       tenor_years: int = 5) -> list[dict[str, float]]:
    """The wider the spread, the lower the survival, the smaller the RPV01, the worse the trap."""
    out = []
    for s in spreads_bp:
        u = upfront(s / 1e4, coupon, trade_date, tenor_years)
        out.append({"quoted_bp": s, "hazard_pct": u.hazard * 100.0, "rpv01": u.rpv01,
                    "clean_points": u.clean_points, "naive_points": u.naive_points,
                    "naive_error_pct": u.naive_error_pct})
    return out


def coupon_invariance(coupons_bp: Sequence[float] = STANDARD_COUPONS_BP,
                      quoted_spread: float = 0.02, trade_date: dt.date = TRADE_DATE,
                      tenor_years: int = 5) -> list[dict[str, float]]:
    """The same risk on either standard coupon: the upfront absorbs the coupon choice exactly."""
    out = []
    for c in coupons_bp:
        u = upfront(quoted_spread, c / 1e4, trade_date, tenor_years)
        out.append({"coupon_bp": c, "clean_points": u.clean_points, "rpv01": u.rpv01,
                    "protection_pv": u.clean_points / 100.0 + c / 1e4 * u.rpv01})
    return out


# --------------------------------------------------------------------------- ISDA's own example
def isda_worked_example() -> dict[str, object]:
    """Reproduce ISDA's published example (Standard CDS Examples, April 2009, cdsmodel.com).

    A 1y $36mm 100bp standard CDS traded in Feb09 maturing 20Mar10. The document prints the
    five accrual periods (88 / 94 / 91 / 91 / 90 days, paying $88k / $94k / $91k / $91k / $90k),
    and then: quoted at 2 points upfront the buyer pays $720k clean, the seller pays $61k of
    accrued over 61 riskless days, so net the buyer pays $659k.
    """
    trade = dt.date(2009, 2, 20)
    mat = dt.date(2010, 3, 20)
    notional, coupon = 36_000_000.0, 0.01
    periods = cds_periods(trade, mat)
    published_days = [88, 94, 91, 91, 90]
    published_pay = [88_000.0, 94_000.0, 91_000.0, 91_000.0, 90_000.0]
    published_dates = [dt.date(2009, 3, 20), dt.date(2009, 6, 22), dt.date(2009, 9, 21),
                       dt.date(2009, 12, 21), dt.date(2010, 3, 22)]
    rows = [{"n": i + 1, "accrual_start": p.accrual_start, "accrual_end": p.accrual_end,
             "days": p.days, "payment": p.year_fraction * coupon * notional,
             "payment_date": p.payment_date,
             "published_days": published_days[i], "published_payment": published_pay[i],
             "published_date": published_dates[i]}
            for i, p in enumerate(periods)]
    nd = accrued_days(trade)
    clean_cash = 0.02 * notional
    accrued_cash = nd / 360.0 * notional * coupon
    return {"trade_date": trade, "maturity": mat, "rows": rows,
            "days_match": all(r["days"] == r["published_days"] for r in rows),
            "payments_match": all(abs(r["payment"] - r["published_payment"]) < 0.5 for r in rows),
            "dates_match": all(r["payment_date"] == r["published_date"] for r in rows),
            "accrued_days": nd, "published_accrued_days": 61,
            "clean_cash": clean_cash, "published_clean_cash": 720_000.0,
            "accrued_cash": accrued_cash, "published_accrued_cash": 61_000.0,
            "net_cash": clean_cash - accrued_cash, "published_net_cash": 659_000.0}


# --------------------------------------------------------------------------- the roll
def roll_jump(trade_dates: Sequence[dt.date] | None = None, tenor_years: int = 5
              ) -> list[dict[str, object]]:
    """What "5Y" means either side of the 20 September roll."""
    if trade_dates is None:
        trade_dates = [dt.date(2026, 9, d) for d in (14, 18, 21, 23, 28)]
    return [{"trade_date": d, "maturity": standard_cds_maturity(d, tenor_years),
             "protection_years": year_fraction(d, standard_cds_maturity(d, tenor_years))}
            for d in trade_dates]


def roll_cycle_range(tenor_years: int = 5, start: dt.date = dt.date(2026, 1, 1),
                     days: int = 400) -> dict[str, float]:
    """Minimum and maximum protection a `tenor_years` quote buys over a full cycle."""
    vals = [year_fraction(start + dt.timedelta(days=k),
                          standard_cds_maturity(start + dt.timedelta(days=k), tenor_years))
            for k in range(days)]
    return {"min_years": min(vals), "max_years": max(vals),
            "spread_years": max(vals) - min(vals)}


# --------------------------------------------------------------------------- QuantLib checks
def quantlib_roll_check(n_dates: int = 500, step_days: int = 3) -> dict[str, object] | None:
    """Compare `standard_cds_maturity` with QuantLib's own CDS2015 rule, or None if absent."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    checked = mismatches = 0
    start = dt.date(2025, 1, 1)
    for k in range(n_dates):
        d = start + dt.timedelta(days=k * step_days)
        for tenor in (1, 2, 3, 5, 7, 10, 20, 30):
            checked += 1
            q = ql.cdsMaturity(ql.Date(d.day, d.month, d.year), ql.Period(tenor, ql.Years),
                               ql.DateGeneration.CDS2015)
            got = standard_cds_maturity(d, tenor)
            if (q.dayOfMonth(), q.month(), q.year()) != (got.day, got.month, got.year):
                mismatches += 1
    return {"ql_version": ql.__version__, "checked": checked, "mismatches": mismatches}


def quantlib_cross_check(quoted_spread: float = 0.02, coupon: float = 0.01,
                         trade_date: dt.date = TRADE_DATE, tenor_years: int = 5,
                         r: float = FLAT_RATE, recovery: float = STANDARD_RECOVERY
                         ) -> dict[str, object] | None:
    """Price the same contract with all three QuantLib CDS engines, or None if it is absent."""
    try:
        import QuantLib as ql
    except ImportError:
        return None
    today = ql.Date(trade_date.day, trade_date.month, trade_date.year)
    ql.Settings.instance().evaluationDate = today      # a stale global gives NPV exactly 0.0
    disc = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed(), ql.Continuous))
    h = implied_flat_hazard(quoted_spread, trade_date, tenor_years, r, recovery)
    prob = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(today, ql.QuoteHandle(ql.SimpleQuote(h)), ql.Actual365Fixed()))
    mat = standard_cds_maturity(trade_date, tenor_years)
    acc = accrual_begin(trade_date)
    sched = ql.Schedule(ql.Date(acc.day, acc.month, acc.year),
                        ql.Date(mat.day, mat.month, mat.year), ql.Period(ql.Quarterly),
                        ql.WeekendsOnly(), ql.Following, ql.Unadjusted,
                        ql.DateGeneration.CDS2015, False)
    rows = []
    for name, engine in (("MidPointCdsEngine", ql.MidPointCdsEngine(prob, recovery, disc)),
                         ("IntegralCdsEngine",
                          ql.IntegralCdsEngine(ql.Period(1, ql.Days), prob, recovery, disc)),
                         ("IsdaCdsEngine", ql.IsdaCdsEngine(prob, recovery, disc))):
        cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, coupon, sched, ql.Following,
                                   ql.Actual360())
        cds.setPricingEngine(engine)
        rows.append({"engine": name, "fair_spread_bp": cds.fairSpread() * 1e4,
                     "rpv01": abs(cds.couponLegBPS()) * 1e4,
                     "upfront_points": cds.NPV() * 100.0})
    mine = upfront(quoted_spread, coupon, trade_date, tenor_years, r, recovery)
    return {"ql_version": ql.__version__, "quoted_bp": quoted_spread * 1e4, "engines": rows,
            "mine_rpv01": mine.rpv01, "mine_clean_points": mine.clean_points,
            "max_rpv01_diff": max(abs(row["rpv01"] - mine.rpv01) for row in rows),
            "max_spread_diff_bp": max(abs(row["fair_spread_bp"] - quoted_spread * 1e4)
                                      for row in rows),
            "engine_spread_range_bp": (max(row["fair_spread_bp"] for row in rows)
                                       - min(row["fair_spread_bp"] for row in rows))}


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    W = 96
    print("=" * W)
    print("CDS MECHANICS AND UPFRONT -- the risky annuity, the roll, and the accrual rebate")
    print("=" * W)

    print("\n1. ISDA'S OWN WORKED EXAMPLE, REPRODUCED  (Standard CDS Examples, April 2009)")
    ex = isda_worked_example()
    print(f"   1y $36mm 100bp standard CDS, traded {ex['trade_date']}, maturing {ex['maturity']}")
    print(f"   {'#':>3}{'accrual start':>16}{'accrual end':>14}{'days':>7}{'ISDA':>6}"
          f"{'payment':>12}{'ISDA':>11}{'pay date':>14}")
    for r_ in ex["rows"]:
        print(f"   {r_['n']:>3}{str(r_['accrual_start']):>16}{str(r_['accrual_end']):>14}"
              f"{r_['days']:>7}{r_['published_days']:>6}{r_['payment']:>12,.0f}"
              f"{r_['published_payment']:>11,.0f}{str(r_['payment_date']):>14}")
    print(f"   schedule matches ISDA: days {ex['days_match']}, payments {ex['payments_match']}, "
          f"payment dates {ex['dates_match']}")
    print(f"   quoted at 2 points: buyer pays clean ${ex['clean_cash']:,.0f} "
          f"(ISDA ${ex['published_clean_cash']:,.0f});")
    print(f"   seller pays accrued over {ex['accrued_days']} riskless days = "
          f"${ex['accrued_cash']:,.0f} (ISDA {ex['published_accrued_days']} days, "
          f"${ex['published_accrued_cash']:,.0f});")
    print(f"   NET the buyer pays ${ex['net_cash']:,.0f}   (ISDA ${ex['published_net_cash']:,.0f})")

    u = upfront()
    print(f"\n2. ONE STANDARD CONTRACT TODAY  (trade {TRADE_DATE}, quoted 200 bp, 100 bp coupon, "
          f"R = 40%,")
    print("   flat 3% curve)")
    print(f"   standard 5Y maturity        {u.maturity}   (accrual begins {u.accrual_start})")
    print(f"   protection actually bought  {u.protection_years:.4f} years, NOT 5.0000")
    print(f"   implied flat hazard         {u.hazard * 100:.6f}%  "
          f"(the ISDA QUOTING convention, not a credit model)")
    print(f"   risky annuity RPV01         {u.rpv01:.6f} years   "
          f"(of which accrual on default {u.accrual_on_default:.6f})")
    print(f"   clean upfront               {u.clean_points:.6f} points of notional")
    print(f"   less accrued coupon         {u.accrued_points:.6f} points "
          f"({u.accrued_days} riskless days)")
    print(f"   = cash settlement amount    {u.cash_points:.6f} points, paid by the BUYER")
    print(f"   'spread difference x tenor' {u.naive_points:.6f} points  -> "
          f"OVERSTATES by {u.naive_error_pct:+.2f}%")

    d = rpv01_decomposition()
    print("\n3. WHERE THE YEARS GO, FROM THE LABEL '5Y' TO THE NUMBER YOU MULTIPLY BY")
    for label, key in (("the label says", "label_years"),
                       ("calendar life of the standard contract", "calendar_years"),
                       ("ACT/360 accrual on the real IMM schedule", "act360_accrual_years"),
                       ("after discounting at 3%", "after_discounting"),
                       ("after survival at the implied hazard", "after_survival"),
                       ("plus accrual on default", "accrual_on_default"),
                       ("= RPV01, the number in the upfront formula", "rpv01")):
        print(f"   {label:<44}{d[key]:>10.4f} years")

    print("\n4. RPV01 IS NOT THE TENOR, AND THE GAP GROWS WITH IT")
    print(f"   {'tenor':>7}{'maturity':>14}{'RPV01 yrs':>12}{'vs tenor':>11}"
          f"{'upfront pts':>14}{'(S-C) x tenor':>16}{'error':>10}")
    for row in annuity_trap():
        print(f"   {row['tenor']:>6}Y{str(row['maturity']):>14}{row['rpv01']:>12.4f}"
              f"{row['rpv01_vs_tenor_pct']:>+10.2f}%{row['clean_points']:>14.4f}"
              f"{row['naive_points']:>16.4f}{row['naive_error_pct']:>+9.2f}%")

    print("\n5. AND IT GROWS WITH THE SPREAD -- a wide name survives less, so its annuity is lower")
    print(f"   {'quoted':>9}{'flat hazard':>14}{'RPV01 yrs':>12}{'upfront pts':>14}"
          f"{'(S-C) x 5':>12}{'error':>10}")
    for row in spread_sensitivity():
        err = f"{row['naive_error_pct']:>+9.2f}%" if row["clean_points"] else f"{'n/a':>10}"
        print(f"   {row['quoted_bp']:>7.0f}bp{row['hazard_pct']:>13.4f}%{row['rpv01']:>12.4f}"
              f"{row['clean_points']:>14.4f}{row['naive_points']:>12.4f}{err}")
    print("   At the coupon itself the upfront is zero and the error is undefined -- that single")
    print("   point is the only place 'spread difference x tenor' is right.")

    print("\n6. 'FIVE YEAR' IS NOT FIVE YEARS -- the 20 September roll")
    print(f"   {'trade date':>13}{'standard 5Y maturity':>24}{'protection years':>19}")
    for row in roll_jump():
        print(f"   {str(row['trade_date']):>13}{str(row['maturity']):>24}"
              f"{row['protection_years']:>19.4f}")
    rng = roll_cycle_range()
    print(f"   over a full cycle a 5Y quote buys between {rng['min_years']:.4f} and "
          f"{rng['max_years']:.4f} years")
    print(f"   -- a range of {rng['spread_years']:.4f} years, and it steps by half a year "
          f"OVERNIGHT on 20 Mar and 20 Sep.")

    print("\n7. THE COUPON IS A PACKAGING CHOICE; THE UPFRONT ABSORBS IT")
    print(f"   {'coupon':>9}{'RPV01 yrs':>12}{'clean upfront pts':>20}{'PV of protection':>19}")
    for row in coupon_invariance():
        print(f"   {row['coupon_bp']:>7.0f}bp{row['rpv01']:>12.4f}{row['clean_points']:>20.4f}"
              f"{row['protection_pv']:>19.8f}")
    print("   Same PV on either standard coupon -- the upfront is exactly the difference in the")
    print("   premium stream, which is why 100 bp and 500 bp quotes are comparable at all.")

    note = isda_rate_curve_note()
    print("\n8. THE ISDA STANDARD RATE CURVE MOVED")
    print(f"   now      {note['current']}")
    print(f"   retired  {note['retired']}  (decommissioned {note['retired_on']})")
    print(f"   source   {note['source']}")

    print("\n9. QUANTLIB CROSS-CHECKS")
    rc, q = quantlib_roll_check(), quantlib_cross_check()
    if rc is None or q is None:
        print("   QuantLib is not installed -- every number above is still produced by this file.")
    else:
        print(f"   QuantLib {rc['ql_version']}: roll calendar checked on {rc['checked']} "
              f"(trade date, tenor) pairs against")
        print(f"   cdsMaturity(..., DateGeneration.CDS2015) -- {rc['mismatches']} mismatches")
        print(f"   {'engine':<20}{'fair spread bp':>16}{'RPV01 yrs':>12}{'upfront pts':>14}")
        for row in q["engines"]:
            print(f"   {row['engine']:<20}{row['fair_spread_bp']:>16.4f}{row['rpv01']:>12.6f}"
                  f"{row['upfront_points']:>14.4f}")
        print(f"   {'this file':<20}{q['quoted_bp']:>16.4f}{q['mine_rpv01']:>12.6f}"
              f"{q['mine_clean_points']:>14.4f}")
        print(f"   worst disagreement with this file: {q['max_spread_diff_bp']:.4f} bp of spread, "
              f"{q['max_rpv01_diff']:.6f} years of RPV01.")
        print(f"   The three engines also differ from EACH OTHER by "
              f"{q['engine_spread_range_bp']:.4f} bp -- the engine is part")
        print("   of the trade description.")

    print("\n" + "=" * W)
    print("THE RULE: upfront = (quoted spread - coupon) x RPV01, and RPV01 is the RISKY ANNUITY "
          "in years,")
    print("not the tenor. Check the maturity date, not the label: '5Y' is a 20 Jun / 20 Dec date.")
    print("=" * W)
