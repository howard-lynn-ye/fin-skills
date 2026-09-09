"""fin_skills.macro.vintage_backtest - the vintage A/B, and the as-of read it rests on.

The documented properties: as_of() never returns a value that had not been published; the
synthetic panel reproduces BLS's own mean absolute first-to-third revision; the look-ahead
treatments beat the vintage treatment on average and the vintage one is unaffected by
future publications; the transcribed BLS benchmark table says what the skill says it says.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.macro import vintage_backtest as vb


@pytest.fixture(scope="module")
def panel():
    return vb.make_panel(240, vb.SEED)


def test_bls_tables_are_transcribed_as_the_skill_quotes_them():
    assert vb.CES_REVISION_SUMMARY.loc["2003-present", "sa_3_1"] == 51
    assert vb.CES_REVISION_SUMMARY.loc["1979-2003", "sa_3_1"] == 61
    assert vb.CES_REVISION_SUMMARY.loc["all periods", "sa_3_1"] == 57
    eff = vb.benchmark_2025_effect()
    # BLS Table 1: January 2025 goes from +111k to -48k, a 159k move that flips the sign
    assert eff["sign_flips"] == ["Jan"]
    assert eff["max_abs"] == 159.0 and eff["max_month"] == "Jan"
    assert eff["mean_abs"] == pytest.approx(469 / 12)
    assert eff["n_negative_before"] == 3 and eff["n_negative_after"] == 4


def test_revision_scales_invert_the_mean_absolute_moment():
    s1, s2 = vb.revision_scales(51.0, 33.0, 34.0)
    # nested sampling: the stage variances add up to the first-to-third variance
    target_sd = 51.0 * np.sqrt(np.pi / 2.0)
    assert s1 ** 2 + s2 ** 2 == pytest.approx(target_sd ** 2)
    assert s1 < s2                       # BLS's own 33 < 34 split


def test_panel_is_seeded_and_reproduces_the_bls_revision_magnitudes():
    truth, _, meta = vb.make_panel(480, vb.SEED)
    truth2, _, _ = vb.make_panel(480, vb.SEED)
    assert np.array_equal(truth, truth2)
    assert not np.array_equal(truth, vb.make_panel(480, vb.SEED + 1)[0])
    got = vb.measure_revisions(meta)
    assert got["3-1"] == pytest.approx(51.0, abs=3.0)
    assert got["2-1"] == pytest.approx(33.0, abs=3.0)
    assert got["3-2"] == pytest.approx(34.0, abs=3.0)
    # the benchmark is what takes first-to-final past first-to-third
    assert got["final-1"] > got["3-1"] * 1.2


def test_as_of_never_returns_a_value_that_had_not_been_published(panel):
    _, pan, _ = panel
    for when in (30, 120, 239):
        known = vb.as_of(pan, when)
        for ref, val in known.items():
            rows = pan[(pan["ref"] == ref) & (pan["release"] <= when)]
            assert len(rows) > 0
            # the value is the LATEST published on or before `when`, and nothing later
            assert val == rows.sort_values("release").iloc[-1]["value"]
        later = pan[pan["release"] > when]
        assert not any(np.isclose(known.get(r, np.nan), v)
                       for r, v in zip(later["ref"], later["value"])
                       if r not in known.index)


def test_as_of_first_release_and_latest_are_the_three_corners(panel):
    _, pan, _ = panel
    first, last = vb.first_release(pan), vb.latest(pan)
    assert (first != last).any()
    ref = 100
    hist = pan[pan["ref"] == ref].sort_values("release")
    assert first[ref] == hist.iloc[0]["value"]
    assert last[ref] == hist.iloc[-1]["value"]
    rel0 = int(hist.iloc[0]["release"])
    assert vb.as_of(pan, rel0)[ref] == first[ref]
    assert vb.as_of(pan, 239).get(ref) == last[ref]
    with pytest.raises(KeyError):
        vb.as_of(pan, rel0 - 1)[ref]          # not published yet


def test_vintage_matrix_row_equals_as_of(panel):
    _, pan, _ = panel
    V = vb.vintage_matrix(pan, 240)
    for m in (10, 60, 239):
        row = pd.Series(V[m]).dropna()
        ref = vb.as_of(pan, m)
        assert np.allclose(row.to_numpy(), ref.reindex(row.index).to_numpy())
    # a matrix row can only gain values as time passes, never lose them
    assert (np.isfinite(V[100]) <= np.isfinite(V[200])).all()


def test_the_trap_fires_on_revised_data_and_clears_on_the_vintage():
    tab = vb.run_ab(480, vb.SEED)
    assert set(tab.index) == {"vintage", "revised", "ref-date"}
    assert tab.loc["vintage", "sharpe_gap"] == 0.0
    assert tab.loc["ref-date", "sharpe"] > tab.loc["vintage", "sharpe"]
    assert tab.loc["revised", "sharpe"] > tab.loc["vintage", "sharpe"]
    # the signals genuinely differ - that is the mechanism, not a coincidence
    assert tab.loc["revised", "disagree_vs_vintage"] > 0.05
    assert tab.loc["ref-date", "disagree_vs_vintage"] > tab.loc["revised",
                                                                "disagree_vs_vintage"]


def test_the_vintage_run_is_immune_to_a_later_republication():
    """Rewriting history AFTER the last decision cannot move the honest backtest."""
    truth, pan, _ = vb.make_panel(240, vb.SEED)
    tab = vb.run_ab(240, vb.SEED)
    # a comprehensive restatement published in month 220 of every month up to 100
    extra = pd.DataFrame({"ref": np.arange(101), "release": 220,
                          "value": vb.latest(pan).to_numpy()[:101] + 500.0, "est": 5})
    tampered = pd.concat([pan, extra], ignore_index=True)
    sig_a, _ = vb.treatments(pan, 240)["vintage"]
    sig_b, _ = vb.treatments(tampered, 240)["vintage"]
    ok = np.arange(240) < 220
    assert np.allclose(sig_a[ok][2:], sig_b[ok][2:], equal_nan=True)
    # while the "revised" view of the SAME early months moves by exactly the restatement
    rev_a = vb.treatments(pan, 240)["revised"][0]
    rev_b = vb.treatments(tampered, 240)["revised"][0]
    assert np.allclose(rev_b[26:100] - rev_a[26:100], 500.0)
    assert np.isfinite(tab.loc["vintage", "sharpe"])


def test_timing_look_ahead_dominates_the_value_look_ahead_across_seeds():
    """The documented ordering: ref-date >> revised at payrolls' own revision size."""
    gaps_rev, gaps_ref = [], []
    for k in range(40):
        t = vb.run_ab(480, vb.SEED + k)
        gaps_rev.append(t.loc["revised", "sharpe_gap"])
        gaps_ref.append(t.loc["ref-date", "sharpe_gap"])
    assert np.mean(gaps_ref) > 0.15
    assert np.mean(gaps_ref) > 4 * abs(np.mean(gaps_rev))


def test_the_value_gap_scales_with_revision_size():
    small = [vb.run_ab(240, vb.SEED + k, scale=1.0).loc["revised", "sharpe_gap"]
             for k in range(15)]
    big = [vb.run_ab(240, vb.SEED + k, scale=4.0).loc["revised", "sharpe_gap"]
           for k in range(15)]
    assert np.mean(big) > np.mean(small)


def test_momentum_positions_are_flat_where_the_history_is_missing():
    sig = np.array([np.nan, 1.0, 2.0])
    hist = np.array([[np.nan, np.nan], [np.nan, np.nan], [0.0, 1.0]])
    pos = vb.momentum_positions(sig, hist, window=2)
    assert pos[0] == 0.0 and pos[1] == 0.0 and pos[2] == 1.0


def test_sharpe_convention_is_monthly_annualised():
    r = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert vb.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(12.0))
    assert np.isnan(vb.sharpe(np.array([np.nan, np.nan])))


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.macro.vintage_backtest")
    for head in ("1. What BLS itself publishes", "2. A synthetic vintage panel",
                 "3. as_of()", "4. The A/B", "5. Which look-ahead actually costs you"):
        assert head in out
    assert "51k" in out and "-159" in out
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
