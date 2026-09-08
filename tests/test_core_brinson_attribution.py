"""fin_skills.core.brinson_attribution - effects reconcile exactly, and link across periods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.brinson_attribution import (TOTAL, ReconciliationError, _demo_panels,
                                                 brinson_fachler, brinson_hood_beebower,
                                                 brinson_single_period, carino_coefficients,
                                                 carino_link, multi_period_attribution, render)


@pytest.fixture(scope="module")
def panels():
    return _demo_panels()


def test_single_period_effects_sum_to_the_active_return(panels):
    for p in panels:
        eff = brinson_single_period(p)
        active = float(p.wp @ p.rp) - float(p.wb @ p.rb)
        assert eff.attrs["active"] == pytest.approx(active, abs=1e-15)
        assert eff.loc[TOTAL, "total"] == pytest.approx(active, abs=1e-12)
        assert list(eff.columns) == ["allocation", "selection", "interaction", "total"]


def test_fachler_and_bhb_differ_only_in_the_per_sector_allocation_split(panels):
    f = brinson_fachler(panels[0])
    b = brinson_hood_beebower(panels[0])
    sectors = f.index.drop(TOTAL)
    pd.testing.assert_series_equal(f.loc[sectors, "selection"], b.loc[sectors, "selection"])
    pd.testing.assert_series_equal(f.loc[sectors, "interaction"], b.loc[sectors, "interaction"])
    assert not np.allclose(f.loc[sectors, "allocation"], b.loc[sectors, "allocation"])
    assert f.loc[TOTAL, "allocation"] == pytest.approx(b.loc[TOTAL, "allocation"], abs=1e-12)
    assert f.attrs["method"] == "fachler" and b.attrs["method"] == "bhb"


def test_bhb_credits_an_overweight_for_beating_zero_fachler_for_beating_the_index():
    # one sector overweight that LAGS a rising benchmark: BHB says good call, Fachler says bad
    panel = pd.DataFrame({"wp": [0.6, 0.4], "wb": [0.5, 0.5],
                          "rp": [0.02, 0.10], "rb": [0.02, 0.10]}, index=["lagging", "leading"])
    assert brinson_hood_beebower(panel).loc["lagging", "allocation"] > 0
    assert brinson_fachler(panel).loc["lagging", "allocation"] < 0


def test_naive_sum_misses_compounding_and_carino_closes_it(panels):
    per = [brinson_single_period(p) for p in panels]
    naive = sum(float(e.loc[TOTAL, "total"]) for e in per)
    rp = float(np.prod([1 + e.attrs["rp"] for e in per]) - 1)
    rb = float(np.prod([1 + e.attrs["rb"] for e in per]) - 1)
    assert abs(naive - (rp - rb)) > 1e-4                 # the residual is real
    linked = carino_link(per)
    assert linked.loc[TOTAL, "total"] == pytest.approx(rp - rb, abs=1e-12)
    assert linked.attrs["residual"] == pytest.approx(naive - (rp - rb))
    assert linked.attrs["n_periods"] == 4
    pd.testing.assert_frame_equal(linked, multi_period_attribution(panels))


def test_carino_coefficients_handle_the_tie_without_dividing_by_zero():
    k, big_k, rp_tot, rb_tot = carino_coefficients([0.10, 0.05], [0.10, 0.02])
    assert np.isfinite(k).all()
    assert k[0] == pytest.approx(1 / 1.10)               # the 0/0 limit
    assert rp_tot == pytest.approx(1.10 * 1.05 - 1) and rb_tot == pytest.approx(1.10 * 1.02 - 1)
    with pytest.raises(ValueError, match="wipeout"):
        carino_coefficients([-1.0], [0.0])


def test_missing_cash_line_is_refused_not_renormalised(panels):
    broken = panels[0].copy()
    broken.loc["Staples", "wp"] = 0.14                   # a 6% sleeve left out of the table
    with pytest.raises(ReconciliationError, match="weights do not sum to 1"):
        brinson_fachler(broken)


def test_input_validation(panels):
    with pytest.raises(ValueError, match="missing column"):
        brinson_single_period(panels[0].drop(columns="rb"))
    nan = panels[0].copy()
    nan.loc["Tech", "rp"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        brinson_single_period(nan)
    with pytest.raises(ValueError, match="method"):
        brinson_single_period(panels[0], method="whatever")
    with pytest.raises(ValueError, match="different sector list"):
        carino_link([brinson_single_period(panels[0]),
                     brinson_single_period(panels[1].rename(index={"Tech": "Tech2"}))])
    with pytest.raises(ValueError):
        carino_link([])


def test_render_is_in_basis_points_and_names_the_sectors(panels):
    text = render(multi_period_attribution(panels))
    assert "Tech" in text and "Carino-linked" in text and "bps" in text


def test_demo_runs_to_its_conclusion(run_main):
    out = run_main("fin_skills.core.brinson_attribution")
    assert "ReconciliationError" in out and "floating-point" in out
