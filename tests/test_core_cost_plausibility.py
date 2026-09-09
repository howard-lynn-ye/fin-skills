"""fin_skills.core.cost_plausibility - is the assumed cost consistent with the order sizes?

The acceptance test is the paper itself: `table3()` must reproduce Almgren, Thum, Hauptmann
& Li (2005) Table 3, which prints whole bps, so agreement to under half a bp is exact
agreement. Everything else here is the arithmetic around it and the guard that wraps it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from fin_skills.api import Bundle
from fin_skills.core.cost_plausibility import (BETA, BETA_SQRT, DEFAULT_DAILY_VOL, ETA, GAMMA,
                                               MAX_PARTICIPATION, TABLE_3, check_cost,
                                               impact_bps, implied_cost_bps, participation_rate,
                                               plausible_book, render, table3)

# One research run's worth of inputs: a 25m book, 8.5% one-way turnover a day over 6.5 names
# whose ADV is 17m, at 2.2% daily vol. Implied impact lands between the two stated costs.
RUN = dict(turnover=0.085, book=2.5e7, adv=1.7e7, n_names=6.5, daily_vol=0.022)


# ------------------------------------------------------------------ the primary source
def test_table3_reproduces_the_papers_own_worked_example():
    got = table3()
    assert list(got.index) == ["IBM", "DRI"]
    for T in (0.1, 0.2, 0.5):
        # the paper prints whole bps; under half a bp is agreement to the printed digit
        assert (got[f"J_T{T}"] - got[f"paper_J_T{T}"]).abs().max() < 0.5, T
    assert (got["I_bps"] - got["paper_I_bps"]).abs().max() < 0.5
    # normalised permanent impact, their I/sigma column, to three decimals as printed
    assert got["I_over_sigma"].round(3).tolist() == TABLE_3["paper_I_over_sigma"].tolist()


def test_the_published_coefficients_are_the_ones_transcribed():
    assert (GAMMA, ETA, BETA, BETA_SQRT) == (0.314, 0.142, 0.6, 0.5)
    # their temporary term at T = X/V is exactly eta * sigma, independent of the name
    at_unit_rate = impact_bps(0.1, 0.0157, exec_horizon=0.1)["temporary_bps"] / 1e4 / 0.0157
    assert at_unit_rate == pytest.approx(ETA)


# ------------------------------------------------------------------ the model
def test_impact_is_increasing_in_participation_and_linear_in_volatility():
    ps = [0.001, 0.005, 0.02, 0.05, 0.1]
    costs = [impact_bps(p, 0.02)["realized_bps"] for p in ps]
    assert costs == sorted(costs) and costs[0] > 0
    assert impact_bps(0.0, 0.02)["realized_bps"] == 0.0
    single = impact_bps(0.03, 0.01)["realized_bps"]
    assert impact_bps(0.03, 0.03)["realized_bps"] == pytest.approx(3.0 * single)


def test_a_shorter_execution_horizon_is_dearer_and_only_the_temporary_part_moves():
    slow = impact_bps(0.05, 0.02, exec_horizon=1.0)
    fast = impact_bps(0.05, 0.02, exec_horizon=0.25)
    assert fast["temporary_bps"] > slow["temporary_bps"]
    assert fast["permanent_bps"] == pytest.approx(slow["permanent_bps"])
    assert fast["realized_bps"] > slow["realized_bps"]


def test_the_folklore_square_root_is_dearer_below_full_participation_and_equal_at_it():
    for p in (0.001, 0.01, 0.1):
        assert (impact_bps(p, 0.02, beta=BETA_SQRT)["temporary_bps"]
                > impact_bps(p, 0.02, beta=BETA)["temporary_bps"])
    assert (impact_bps(1.0, 0.02, beta=BETA_SQRT)["temporary_bps"]
            == pytest.approx(impact_bps(1.0, 0.02, beta=BETA)["temporary_bps"]))


def test_model_inputs_are_validated():
    with pytest.raises(ValueError, match="non-negative"):
        impact_bps(-0.01, 0.02)
    with pytest.raises(ValueError, match="exec_horizon"):
        impact_bps(0.01, 0.02, exec_horizon=0.0)
    with pytest.raises(ValueError, match="must be"):
        participation_rate(0.1, 0.0, 1e7, 5)
    with pytest.raises(ValueError, match="must be"):
        participation_rate(0.1, 1e7, 1e7, 0)


# ------------------------------------------------------------------ the arithmetic
def test_participation_is_the_documented_ratio():
    assert participation_rate(0.10, 1e8, 1e7, 5) == pytest.approx(0.10 * 1e8 / 5 / 1e7)
    assert participation_rate(0.10, 1e8, 1e7, 5) == pytest.approx(0.2)
    # doubling the book doubles it; doubling the names traded halves it
    assert participation_rate(0.1, 2e8, 1e7, 5) == 2 * participation_rate(0.1, 1e8, 1e7, 5)
    assert participation_rate(0.1, 1e8, 1e7, 10) == 0.5 * participation_rate(0.1, 1e8, 1e7, 5)


def test_plausible_book_inverts_the_implied_cost():
    stated = 6.0
    b = plausible_book(stated, RUN["turnover"], RUN["adv"], RUN["n_names"], RUN["daily_vol"])
    assert implied_cost_bps(RUN["turnover"], b, RUN["adv"], RUN["n_names"],
                            RUN["daily_vol"]) == pytest.approx(stated, abs=1e-4)
    assert plausible_book(0.0, RUN["turnover"], RUN["adv"], RUN["n_names"]) == 0.0
    # a bigger allowance buys a bigger book
    assert b < plausible_book(60.0, RUN["turnover"], RUN["adv"], RUN["n_names"],
                              RUN["daily_vol"])


# ------------------------------------------------------------------ the verdict
def test_a_cost_below_the_impact_floor_is_rejected_and_one_above_it_is_not():
    cheap = check_cost(cost_bps=2.0, **RUN)
    honest = check_cost(cost_bps=20.0, **RUN)
    assert cheap.implied_bps == pytest.approx(honest.implied_bps)     # same trade, same impact
    assert not cheap.ok and honest.ok
    assert "below" in cheap.reasons[0] and cheap.shortfall_bps > 0
    assert honest.shortfall_bps < 0 and honest.status == "ok"
    # the reported book is the size at which the cheap claim would have been true
    assert cheap.plausible_book < cheap.book < honest.plausible_book


def test_participation_over_the_ceiling_fails_however_generous_the_cost():
    v = check_cost(cost_bps=1e4, **{**RUN, "adv": 2.0e5})
    assert v.participation > MAX_PARTICIPATION
    assert not v.ok and "ceiling" in v.reasons[0]
    assert "of ADV" in v.status
    # both gates can bind at once
    both = check_cost(cost_bps=1.0, **{**RUN, "adv": 2.0e5})
    assert len(both.reasons) == 2


def test_the_pieces_add_up_the_way_the_paper_defines_them():
    v = check_cost(cost_bps=2.0, **RUN)
    assert v.implied_bps == pytest.approx(0.5 * v.permanent_bps + v.temporary_bps)
    assert v.order_usd == pytest.approx(RUN["turnover"] * RUN["book"] / RUN["n_names"])
    assert v.participation == pytest.approx(v.order_usd / RUN["adv"])


def test_render_is_ascii_and_shows_both_numbers_and_the_capacity():
    text = render(check_cost(cost_bps=2.0, **RUN))
    assert text.isascii()
    assert "PARTICIPATION" in text and "IMPLIED cost" in text and "STATED cost" in text
    assert "NOT PLAUSIBLE" in text and "becomes plausible at a book of" in text
    assert "PLAUSIBLE." in render(check_cost(cost_bps=20.0, **RUN))


# ------------------------------------------------------------------ the guard
def test_the_guard_passes_the_honest_cost_and_fails_the_tenth_of_it():
    g = api.get("cost_plausibility")
    assert g.skill == "execution-cost-analysis"
    ok = g.run(cost_bps=20.0, **RUN)
    bad = g.run(cost_bps=2.0, **RUN)
    assert ok.passed and not bad.passed
    assert bad.errors and "below" in bad.errors[0].message
    assert bad.summary().isascii() and bad.evidence["table"].isascii()
    assert bad.evidence["implied_bps"] == pytest.approx(ok.evidence["implied_bps"])
    assert bad.evidence["participation"] == pytest.approx(0.085 * 2.5e7 / 6.5 / 1.7e7)


def test_the_guard_accepts_a_scalar_a_series_or_a_panel_of_adv():
    g = api.get("cost_plausibility")
    names = [f"N{i}" for i in range(8)]
    series = pd.Series(1.7e7, index=names)
    panel = pd.DataFrame(1.7e7, index=pd.bdate_range("2024-01-01", periods=30), columns=names)
    no_count = {k: v for k, v in RUN.items() if k not in ("n_names", "adv")}
    scalar = g.run(**{**RUN, "n_names": 8, "cost_bps": 2.0}).evidence["participation"]
    from_series = g.run(**no_count, adv=series, cost_bps=2.0).evidence
    from_panel = g.run(**no_count, adv=panel, cost_bps=2.0).evidence
    # n_names defaults to the width of adv, so all three describe the same trade
    assert from_series["n_names"] == 8 and from_panel["n_names"] == 8
    assert from_series["participation"] == pytest.approx(scalar)
    assert from_panel["participation"] == pytest.approx(scalar)


def test_the_guard_reports_the_thinnest_name_and_the_assumed_volatility():
    g = api.get("cost_plausibility")
    uneven = pd.Series([1.7e7] * 7 + [1.0e5], index=[f"N{i}" for i in range(8)])
    r = g.run(**{**RUN, "adv": uneven, "cost_bps": 40.0})
    assert r.evidence["adv_min"] == 1.0e5 and r.evidence["adv_median"] == 1.7e7
    assert r.evidence["participation_thinnest"] > r.evidence["participation"]
    assert any("thinnest" in f.where for f in r.warnings)
    # no daily_vol supplied -> the stand-in is used and said so
    quiet = g.run(turnover=0.085, book=2.5e7, adv=1.7e7, n_names=6.5, cost_bps=2.0)
    assert quiet.evidence["daily_vol"] == DEFAULT_DAILY_VOL
    assert any("daily_vol" in f.where for f in quiet.findings)


def test_the_guard_warns_outside_the_band_the_model_was_fitted_on():
    g = api.get("cost_plausibility")
    big = g.run(**{**RUN, "book": 8e7, "cost_bps": 100.0})
    assert big.passed and any("model range" in f.where for f in big.warnings)
    tiny = g.run(**{**RUN, "book": 1e5, "cost_bps": 5.0})
    assert tiny.passed and any("below the 0.25%" in f.message for f in tiny.warnings)


def test_the_guard_rejects_inputs_it_cannot_use():
    g = api.get("cost_plausibility")
    with pytest.raises(TypeError, match="n_names must be given"):
        g.run(turnover=0.085, book=2.5e7, adv=1.7e7, cost_bps=2.0)
    with pytest.raises(TypeError, match="book must be positive"):
        g.run(**{**RUN, "book": 0.0, "cost_bps": 2.0})
    with pytest.raises(TypeError, match="cost_bps"):
        g.run(**{**RUN, "cost_bps": -1.0})
    with pytest.raises(TypeError, match="NaN"):
        g.run(**{**RUN, "turnover": pd.Series([0.1, np.nan]), "cost_bps": 2.0})
    with pytest.raises(TypeError):
        g.run(**{**RUN, "cost_bps": 2.0, "bogus": 1})


def test_a_turnover_series_is_reduced_to_its_mean():
    g = api.get("cost_plausibility")
    turn = pd.Series([0.05, 0.085, 0.12])
    r = g.run(**{**RUN, "turnover": turn, "cost_bps": 2.0})
    assert r.evidence["avg_turnover"] == pytest.approx(float(turn.mean()))


# ------------------------------------------------------------------ through the Bundle
def test_the_bundle_reaches_the_guard_through_its_own_slots():
    b = Bundle(turnover=pd.Series([0.08, 0.09]), book=2.5e7, adv=1.7e7,
               n_names=6.5, cost_bps=2.0, daily_vol=0.022)
    assert b.missing_for("cost_plausibility") == []
    report = b.check(guards=["cost_plausibility"])
    assert report.ran == ["cost_plausibility"] and not report.passed
    # without the book the guard is skipped and says so
    thin = b.without("book").coverage(["cost_plausibility"])
    assert thin.missing == {"cost_plausibility": ["book"]}


def test_demo_prints_the_verification_and_the_verdict(run_main):
    out = run_main("fin_skills.core.cost_plausibility")
    assert "VERIFICATION vs the primary source" in out
    assert "largest disagreement with the paper's printed table" in out
    assert "NOT PLAUSIBLE" in out and "CAPACITY" in out
    assert "THE EXPONENT IS 3/5, NOT 1/2" in out
    assert out.isascii()
