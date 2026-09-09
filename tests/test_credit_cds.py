"""fin_skills.credit.cds - standard coupons, points upfront, the risky annuity and the IMM roll.

The documented properties: the schedule rebuilt from ISDA's conventions reproduces ISDA's own
published worked example (day counts, payments, payment dates, and the $720k / $61k / $659k
cash) exactly; RPV01 is the risky annuity so "(S - C) x tenor" overstates the upfront, by more
the longer the tenor and the wider the spread, and is right only where the upfront is zero;
the CDS2015 maturity rule agrees with QuantLib on every date tested; a "5Y" quote buys between
4.75 and 5.25 years and steps half a year at the roll; and the upfront absorbs the coupon
choice so the PV of protection is identical on 100 bp and 500 bp.
"""
from __future__ import annotations

import datetime as dt

import pytest

from conftest import requires
from fin_skills.credit.cds import (CASH_SETTLE_DAYS, STANDARD_COUPONS_BP, STANDARD_RECOVERY,
                                   TRADE_DATE, accrual_begin, accrued_days, annuity_trap,
                                   cds_periods, coupon_invariance, following,
                                   implied_flat_hazard, isda_contract_spec,
                                   isda_converter_assumptions, isda_rate_curve_note,
                                   isda_worked_example, next_imm, par_spread, premium_leg,
                                   previous_imm, previous_roll, protection_leg,
                                   quantlib_cross_check, quantlib_roll_check, roll_cycle_range,
                                   roll_jump, rpv01_decomposition, spread_sensitivity,
                                   standard_cds_maturity, standard_coupons, upfront)


# ------------------------------------------------------------------ ISDA's published example
def test_the_schedule_reproduces_isdas_own_published_table():
    ex = isda_worked_example()
    assert ex["days_match"] and ex["payments_match"] and ex["dates_match"]
    assert [r["days"] for r in ex["rows"]] == [88, 94, 91, 91, 90]
    assert [r["payment"] for r in ex["rows"]] == pytest.approx(
        [88_000.0, 94_000.0, 91_000.0, 91_000.0, 90_000.0], abs=1e-6)
    # the last accrual date is the UNADJUSTED Saturday maturity, and it pays the Monday after
    last = ex["rows"][-1]
    assert last["accrual_end"] == dt.date(2010, 3, 20) and last["accrual_end"].weekday() == 5
    assert last["payment_date"] == dt.date(2010, 3, 22)


def test_the_cash_matches_isda_to_the_dollar():
    ex = isda_worked_example()
    assert ex["accrued_days"] == ex["published_accrued_days"] == 61
    assert ex["clean_cash"] == pytest.approx(720_000.0, abs=0.5)
    assert ex["accrued_cash"] == pytest.approx(61_000.0, abs=0.5)
    assert ex["net_cash"] == pytest.approx(659_000.0, abs=0.5)


def test_the_quoted_conventions_are_carried_verbatim():
    spec = {r["field"]: r["value"] for r in isda_contract_spec()}
    assert spec["CDS Dates"] == "20th of Mar/Jun/Sep/Dec"
    assert spec["Business Day Count"] == "Actual/360"
    assert spec["Coupon Rate"] == "100bp or 500bp"
    assert spec["Maturity Date"] == "A CDS Date, unadjusted"
    assert "T+1" in spec["Accrual Begin Date"]
    assert "-60 days" in spec["Legal Protection Effective Date"]
    assert any("single flat hazard rate" in s for s in isda_converter_assumptions())
    assert {c["coupon_bp"] for c in standard_coupons()} == set(STANDARD_COUPONS_BP)
    note = isda_rate_curve_note()
    assert note["current"] == "https://rfr.spglobal.com/"
    assert "ihsmarkit" in note["retired"] and note["retired_on"] == "2026-08-15"


# ------------------------------------------------------------------ the calendar
def test_following_only_moves_weekends_and_the_imm_helpers_bracket_a_date():
    assert following(dt.date(2010, 3, 20)) == dt.date(2010, 3, 22)      # Saturday
    assert following(dt.date(2009, 3, 20)) == dt.date(2009, 3, 20)      # Friday, untouched
    d = dt.date(2026, 9, 9)
    assert previous_imm(d) == dt.date(2026, 6, 20) and next_imm(d) == dt.date(2026, 9, 20)
    assert previous_roll(d) == dt.date(2026, 3, 20)
    # the accrual begin date is the ADJUSTED CDS date, two days later than the raw one here
    assert accrual_begin(d) == dt.date(2026, 6, 22)
    assert accrued_days(d) == 80


def test_the_cds2015_maturity_is_a_20_june_or_20_december_date():
    for tenor in (1, 3, 5, 7, 10):
        m = standard_cds_maturity(TRADE_DATE, tenor)
        assert (m.month, m.day) in {(6, 20), (12, 20)}
    assert standard_cds_maturity(TRADE_DATE, 5) == dt.date(2031, 6, 20)
    with pytest.raises(ValueError):
        standard_cds_maturity(TRADE_DATE, 0)


def test_five_year_means_between_4_75_and_5_25_years_and_jumps_at_the_roll():
    rows = {r["trade_date"]: r for r in roll_jump()}
    before, after = rows[dt.date(2026, 9, 18)], rows[dt.date(2026, 9, 21)]
    assert before["maturity"] == dt.date(2031, 6, 20)
    assert after["maturity"] == dt.date(2031, 12, 20)
    assert before["protection_years"] == pytest.approx(4.7562, abs=5e-5)
    assert after["protection_years"] == pytest.approx(5.2493, abs=5e-5)
    assert after["protection_years"] - before["protection_years"] > 0.49
    rng = roll_cycle_range()
    assert rng["min_years"] == pytest.approx(4.7534, abs=5e-5)
    assert rng["max_years"] == pytest.approx(5.2548, abs=5e-5)
    assert rng["spread_years"] == pytest.approx(0.5014, abs=5e-5)


def test_the_schedule_is_contiguous_and_ends_on_the_unadjusted_maturity():
    mat = standard_cds_maturity(TRADE_DATE, 5)
    periods = cds_periods(TRADE_DATE, mat)
    assert periods[0].accrual_start == accrual_begin(TRADE_DATE)
    assert periods[-1].accrual_end == mat
    for a, b in zip(periods[:-1], periods[1:]):
        assert a.accrual_end == b.accrual_start, "no gaps and no overlaps"
    assert all(85 <= p.days <= 95 for p in periods)
    with pytest.raises(ValueError):
        cds_periods(TRADE_DATE, TRADE_DATE - dt.timedelta(days=1))


# ------------------------------------------------------------------ the trap
def test_the_upfront_uses_the_risky_annuity_and_the_shortcut_overstates_it():
    u = upfront()
    assert u.maturity == dt.date(2031, 6, 20)
    assert u.protection_years == pytest.approx(4.7808, abs=5e-5)
    assert u.rpv01 == pytest.approx(4.367676, abs=5e-6)
    assert u.clean_points == pytest.approx(4.367676, abs=5e-6)
    assert u.accrued_points == pytest.approx(0.222222, abs=5e-6)
    assert u.cash_points == pytest.approx(4.145454, abs=5e-6)
    assert u.naive_points == 5.0
    assert u.naive_error_pct == pytest.approx(14.48, abs=0.005)
    # the upfront is (S - C) x RPV01 exactly, not approximately
    assert u.clean_points == pytest.approx((0.02 - 0.01) * u.rpv01 * 100.0, abs=1e-12)


def test_rpv01_is_below_the_tenor_because_of_discounting_and_survival():
    d = rpv01_decomposition()
    assert d["act360_accrual_years"] > d["label_years"] > d["calendar_years"]
    assert d["after_discounting"] < d["act360_accrual_years"]
    assert d["after_survival"] < d["after_discounting"]
    assert d["rpv01"] == pytest.approx(d["after_survival"] + d["accrual_on_default"], abs=1e-12)
    assert d["rpv01"] == pytest.approx(4.3677, abs=5e-5)


def test_the_error_grows_with_the_tenor_and_with_the_spread():
    rows = annuity_trap()
    errs = [r["naive_error_pct"] for r in rows]
    assert all(b > a for a, b in zip(errs, errs[1:])), "longer tenor, bigger error"
    assert errs[0] == pytest.approx(1.45, abs=0.005)
    assert errs[-1] == pytest.approx(32.54, abs=0.005)
    srows = spread_sensitivity()
    wide = [r for r in srows if r["quoted_bp"] == 1000.0][0]
    assert wide["naive_error_pct"] == pytest.approx(53.17, abs=0.005)
    assert all(b < a for a, b in zip([r["rpv01"] for r in srows],
                                     [r["rpv01"] for r in srows][1:])), "wider name, lower annuity"
    at_coupon = [r for r in srows if r["quoted_bp"] == 100.0][0]
    assert at_coupon["clean_points"] == pytest.approx(0.0, abs=1e-12)


def test_the_upfront_absorbs_the_coupon_choice_exactly():
    rows = coupon_invariance()
    assert {r["coupon_bp"] for r in rows} == set(STANDARD_COUPONS_BP)
    pvs = [r["protection_pv"] for r in rows]
    assert pvs[0] == pytest.approx(pvs[1], abs=1e-12), "same risk, same PV, either packaging"
    assert rows[0]["clean_points"] > 0 > rows[1]["clean_points"], "500 bp flips the sign"


def test_the_legs_and_the_hazard_inversion_round_trip():
    h = implied_flat_hazard(0.02)
    assert par_spread(h) == pytest.approx(0.02, abs=1e-12)
    mat = standard_cds_maturity(TRADE_DATE, 5)
    periods = cds_periods(TRADE_DATE, mat)
    # with no hazard there is no protection and the annuity is purely a discounted accrual
    assert protection_leg(TRADE_DATE, mat, 0.0, 0.03) == 0.0
    zero = premium_leg(periods, TRADE_DATE, 0.0, 0.03)
    assert zero["accrual_on_default"] == 0.0
    assert zero["rpv01"] > premium_leg(periods, TRADE_DATE, h, 0.03)["rpv01"]
    with pytest.raises(ValueError):
        premium_leg(periods, TRADE_DATE, -0.01, 0.03)
    with pytest.raises(ValueError):
        protection_leg(TRADE_DATE, mat, h, 0.03, recovery=1.0)
    with pytest.raises(ValueError):
        implied_flat_hazard(0.0)


def test_accrual_on_default_raises_the_annuity_but_only_a_little():
    mat = standard_cds_maturity(TRADE_DATE, 5)
    periods = cds_periods(TRADE_DATE, mat)
    h = implied_flat_hazard(0.02)
    with_acc = premium_leg(periods, TRADE_DATE, h, 0.03, accrual_on_default=True)
    without = premium_leg(periods, TRADE_DATE, h, 0.03, accrual_on_default=False)
    assert with_acc["rpv01"] > without["rpv01"]
    assert with_acc["accrual_on_default"] == pytest.approx(0.0184, abs=5e-5)
    assert with_acc["accrual_on_default"] / with_acc["rpv01"] < 0.01


# ------------------------------------------------------------------ determinism and the demo
def test_the_run_is_deterministic():
    assert upfront() == upfront()
    assert isda_worked_example()["net_cash"] == isda_worked_example()["net_cash"]
    assert CASH_SETTLE_DAYS == 3 and STANDARD_RECOVERY == 0.40


def test_the_demo_prints_the_rule_and_stays_ascii(run_main):
    out = run_main("fin_skills.credit.cds")
    assert all(ord(c) < 128 for c in out), "a non-ASCII char dies on a stock Windows console"
    assert "THE RULE:" in out
    assert "RISKY ANNUITY" in out
    assert "rfr.spglobal.com" in out
    for token in ("659,000", "4.367676", "5.000000", "+14.48%", "2031-06-20", "2031-12-20"):
        assert token in out


@requires("QuantLib")
def test_quantlib_agrees_on_the_roll_calendar_and_the_price():
    rc = quantlib_roll_check()
    assert rc is not None and rc["checked"] == 4000 and rc["mismatches"] == 0
    q = quantlib_cross_check()
    assert q is not None and len(q["engines"]) == 3
    assert q["max_spread_diff_bp"] < 0.5, "under half a basis point against three engines"
    assert q["max_rpv01_diff"] < 0.01
    # the engines disagree with EACH OTHER too - small, but not zero
    assert 0.0 < q["engine_spread_range_bp"] < 0.5


def test_the_cross_checks_return_none_rather_than_raising_without_quantlib(monkeypatch):
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "QuantLib":
            raise ImportError("blocked")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert quantlib_roll_check() is None
    assert quantlib_cross_check() is None
