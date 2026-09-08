"""fin_skills.libraries.weight_traps - HRP on prices is an inverse-share-price portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import has_module, requires
from fin_skills.libraries.weight_traps import (NAMES, ann_vol, demo, hrp_weights, make_panel,
                                               quasi_diag_order)


@pytest.fixture(scope="module")
def result():
    return demo()


def test_panel_is_seeded_and_penny_is_the_riskiest_cheapest_name():
    rets, prices = make_panel(seed=0)
    rets2, prices2 = make_panel(seed=0)
    pd.testing.assert_frame_equal(rets, rets2)
    pd.testing.assert_frame_equal(prices, prices2)
    assert list(rets.columns) == NAMES
    vol = rets.std(ddof=1) * np.sqrt(252)
    assert vol.idxmax() == "PENNY" and prices.iloc[0].idxmin() == "PENNY"


def test_both_weight_vectors_look_fine_and_only_one_is_meaningful(result):
    for key in ("w_returns", "w_prices"):
        w = result[key]
        assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and set(w.index) == set(NAMES)
    assert result["w_prices"]["PENNY"] > 0.5                       # buys the cheapest ticker
    assert result["w_returns"]["PENNY"] < result["w_returns"].median()   # correct HRP underweights it
    assert result["l1"] > 1.0 and result["max_shift"] > 0.5


def test_price_based_weights_are_riskier_under_the_true_covariance(result):
    assert result["vol_prices"] > result["vol_equal"] > result["vol_returns"]
    rets, _ = make_panel(seed=0)
    assert ann_vol(result["w_equal"], rets.cov()) == pytest.approx(result["vol_equal"])


def test_the_documented_guard_separates_returns_from_prices(result):
    assert result["guard_returns"] is True and result["guard_prices"] is False
    assert result["dollar_var"]["MEGA"] > 1000 * result["dollar_var"]["PENNY"]


def test_quasi_diagonal_order_is_a_permutation_and_differs_between_the_inputs(result):
    rets, prices = make_panel(seed=0)
    assert sorted(quasi_diag_order(rets)) == sorted(NAMES)
    assert result["order_returns"] != result["order_prices"]
    assert hrp_weights(rets).index.tolist() == sorted(NAMES)


@pytest.mark.skipif(has_module("pypfopt"), reason="PyPortfolioOpt is installed")
def test_demo_has_no_live_keys_without_pypfopt(result):
    assert not any(k.startswith("LIVE_") for k in result)


@requires("pypfopt")
def test_installed_pypfopt_matches_the_reference_hrp_on_both_inputs(result):
    assert result["LIVE_err_returns"] < 1e-9 and result["LIVE_err_prices"] < 1e-9
    assert result["LIVE_prices_input"] == "accepted, no warning"
    assert result["LIVE_numpy_input"].startswith("TypeError")     # the container IS checked
    assert isinstance(result["LIVE_reuse"], str)


def test_demo_prints_the_rule(run_main):
    out = run_main("fin_skills.libraries.weight_traps")
    assert "Rule: assert your input contains negative values before HRPOpt sees it." in out
