"""fin_skills.core.source_merge - the silent pick, the spliced convention, the moved ticker.

Every number asserted here is one the SKILL.md quotes.

Run:  python -m pytest tests/test_core_source_merge.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.core.source_merge import (SEED, TOL_BPS, adjustment_pair, measure,
                                          measure_adjustment, measure_merge,
                                          measure_symbology, momentum_sharpe,
                                          precedence_merge, recycled_ticker,
                                          refuse_adjustment_mismatch,
                                          refuse_unresolved_symbology, return_stats,
                                          two_vendors)


# ------------------------------------------------------------------- the precedence merge
def test_precedence_takes_the_ranked_source_and_records_what_it_rejected():
    idx = pd.bdate_range("2024-01-02", periods=5)
    a = pd.Series([10.0, np.nan, 10.2, 10.3, 10.4], index=idx)
    b = pd.Series([10.0, 11.0, 12.0, 10.3, 10.4], index=idx)
    out = precedence_merge({"A": a, "B": b}, ("A", "B"))
    assert list(out["values"]) == [10.0, 11.0, 10.2, 10.3, 10.4]     # B only on the hole
    assert list(out["chosen"]) == ["A", "B", "A", "A", "A"]
    assert out["n_overlap"] == 4
    dis = out["disagreements"]
    assert len(dis) == 1 and dis.iloc[0]["chosen"] == "A"
    assert dis.iloc[0]["rejected"] == {"B": 12.0}
    assert dis.iloc[0]["spread_bps"] == pytest.approx(1e4 * 1.8 / 10.2, abs=1.0)
    # combine_first in the other order silently returns 12.0 and says nothing
    assert b.combine_first(a).loc[idx[2]] == 12.0


def test_an_unranked_source_is_refused():
    idx = pd.bdate_range("2024-01-02", periods=3)
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    with pytest.raises(ValueError, match="not in the declared precedence"):
        precedence_merge({"A": s, "B": s}, ("A",))


def test_the_two_vendors_are_seeded_and_wrong_in_different_ways():
    v = two_vendors()
    assert len(v["index"]) == 2520
    assert (v["n_gaps"], v["n_stale"], v["n_bad_print"]) == (204, 150, 48)
    assert v["A"].isna().sum() == v["n_gaps"]
    assert v["B"].notna().all()                 # B never admits it does not know
    again = two_vendors()
    pd.testing.assert_series_equal(v["B"], again["B"])


# ------------------------------------------------------------------ 1. the measured pick
@pytest.fixture(scope="module")
def merge() -> dict:
    return measure_merge()


def test_how_often_the_naive_merge_takes_the_wrong_vendors_number(merge):
    assert merge["n_overlap"] == 2316
    assert merge["n_disagreements"] == 171
    assert merge["disagreement_pct"] == pytest.approx(7.4, abs=0.1)
    # b.combine_first(a) overrides the declared ranking on every one of them
    assert merge["n_naive_differs"] == merge["n_disagreements"] == 171
    assert merge["n_naive_wrong"] == 187
    assert merge["naive_wrong_pct"] == pytest.approx(7.4, abs=0.1)
    assert merge["n_records_kept_by_combine_first"] == 0
    assert merge["median_spread_bps"] == pytest.approx(112.0, abs=1.0)
    assert merge["max_spread_bps"] == pytest.approx(622.0, abs=1.0)


def test_the_right_order_gives_the_same_numbers_and_still_no_record(merge):
    # the difference between a precedence merge and a.combine_first(b) is NOT the numbers
    assert merge["same_numbers_as_a_first"] is True
    assert merge["n_gapfill_wrong"] == 16       # A's holes, filled with B's bad prints
    assert merge["n_records_kept_by_combine_first"] == 0


def test_what_the_corrupted_return_series_does(merge):
    truth = merge["stats"]["truth"]
    naive = merge["stats"]["naive b.combine_first(a)"]
    kept = merge["stats"]["A>B merged == a.combine_first"]
    assert truth["autocorr_1"] == pytest.approx(0.012, abs=0.001)
    assert naive["autocorr_1"] == pytest.approx(-0.055, abs=0.001)
    assert truth["reversal_sharpe"] == pytest.approx(-0.38, abs=0.01)
    assert naive["reversal_sharpe"] == pytest.approx(0.32, abs=0.01)
    assert kept["reversal_sharpe"] == pytest.approx(-0.29, abs=0.01)
    assert truth["ann_vol"] == pytest.approx(0.192, abs=0.001)
    assert naive["ann_vol"] == pytest.approx(0.206, abs=0.001)
    # nothing an outlier filter would catch: the artefact hides UNDER the tail checks
    assert naive["n_beyond_10_sigma"] == truth["n_beyond_10_sigma"] == 0
    assert naive["max_abs_ret_bps"] == pytest.approx(688.0, abs=1.0)


def test_return_stats_finds_the_single_bad_print_a_vol_check_would_miss():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=400)
    clean = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, 400))), index=idx)
    spiked = clean.copy()
    spiked.iloc[200] *= 1.03                       # one 3% print, then a snap-back
    base, hit = return_stats(clean, "clean"), return_stats(spiked, "one bad print")
    assert base["label"] == "clean" and hit["label"] == "one bad print"
    assert hit["autocorr_1"] < base["autocorr_1"]  # the snap-back IS negative autocorr
    assert hit["n_beyond_10_sigma"] == base["n_beyond_10_sigma"] == 0
    assert hit["ann_vol"] / base["ann_vol"] < 1.05  # and the vol check does not notice


# --------------------------------------------------------- 2. the mismatched adjustment
def test_splicing_two_conventions_puts_the_split_on_the_wrong_day():
    a = measure_adjustment()
    assert a["step_pct"] == pytest.approx(-49.8, abs=0.1)
    assert a["split_day_ret_pct"] == pytest.approx(-0.5, abs=0.1)   # the REAL split day
    assert a["one_convention_ann_vol"] == pytest.approx(0.192, abs=0.001)
    assert a["spliced_ann_vol"] == pytest.approx(0.249, abs=0.001)
    assert a["vol_inflation"] == pytest.approx(1.29, abs=0.01)
    assert a["n_beyond_10_sigma"] == 1
    assert a["switch_date"] < a["split_date"]
    p = adjustment_pair()
    # each vendor's OWN series is fine; only the splice is broken
    one = p["adjusted_full"].pct_change().dropna()
    assert float(one.abs().max()) < 0.1


def test_the_adjustment_refusal():
    refuse_adjustment_mismatch("raw", "raw")                        # same convention: fine
    with pytest.raises(ValueError, match="declared adjustment differs"):
        refuse_adjustment_mismatch("raw", "anchored_present")
    with pytest.raises(ValueError, match="Re-adjust to ONE convention"):
        refuse_adjustment_mismatch("anchored_start", "anchored_present")


# ------------------------------------------------------ 3. the identifier that moved
def test_a_recycled_ticker_glues_two_issuers_into_one_series():
    s = measure_symbology()
    assert s["step_pct"] == pytest.approx(-83.9, abs=0.1)
    assert s["e1_ann_vol"] == pytest.approx(0.184, abs=0.001)
    assert s["glued_ann_vol"] == pytest.approx(0.327, abs=0.001)
    assert s["vol_inflation"] == pytest.approx(1.77, abs=0.01)
    assert s["n_beyond_10_sigma"] == 1
    assert s["momentum_sharpe_glued"] == pytest.approx(-0.04, abs=0.01)
    assert s["momentum_sharpe_e1"] == pytest.approx(0.86, abs=0.01)
    assert s["momentum_sharpe_e2"] == pytest.approx(0.07, abs=0.01)
    # the artefact does not just add noise: it destroys a signal that is really there
    assert s["momentum_sharpe_glued"] < min(s["momentum_sharpe_e1"], s["momentum_sharpe_e2"])
    r = recycled_ticker()
    assert len(r["E1"]) == len(r["E2"]) == 1260
    assert momentum_sharpe(r["E1"]) == pytest.approx(s["momentum_sharpe_e1"], abs=1e-9)


def test_the_symbology_refusal():
    refuse_unresolved_symbology("cik")                              # a permanent id: fine
    with pytest.raises(ValueError, match="a ticker is a slot"):
        refuse_unresolved_symbology("ticker")


# ----------------------------------------------------------------------------- the demo
def test_measure_is_deterministic():
    assert str(measure()) == str(measure())


def test_the_demo_prints_ascii_and_reproduces_every_number(capsys):
    from fin_skills.core.source_merge import _print_report
    _print_report(measure())
    out = capsys.readouterr().out
    assert out.isascii()
    for expected in ("171", "2316", "187", "-0.055", "0.32", "-49.8", "1.29x",
                     "-83.9", "1.77x", f"{TOL_BPS:.0f} bps", str(SEED)):
        assert expected in out, expected
    assert "RULE:" in out
