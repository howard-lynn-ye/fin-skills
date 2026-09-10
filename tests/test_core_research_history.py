"""fin_skills.core.research_history - FWER and FDR control over a whole trial ledger.

Two kinds of assertion here. The published fixtures (Harvey-Liu-Zhu's six-test example,
their haircut illustration) are exact. The guarantees - a correction's family-wise error
rate, BH's FDR under positive dependence - are MEASURED on seeded nulls, because that is
the only honest way to check that an implementation controls what it claims to control.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.core.research_history import (METHODS, benjamini_hochberg,
                                              benjamini_yekutieli, bonferroni, by_constant,
                                              compare, correct, from_ledger, haircut_sharpe,
                                              harvey_liu_bhy, holm, independent_pvalue,
                                              p_from_t, panel_pvalues, sharpe_from_p,
                                              strategy_panel, t_from_annual_sharpe,
                                              t_from_sharpe, uncorrected)

T = 1008
ALPHA = 0.05

# Harvey, Liu & Zhu's own six-test worked example, and the adjusted p-values they print.
HLZ_P = np.array([0.005, 0.009, 0.0128, 0.0135, 0.045, 0.06])
HLZ_BONF = [0.0300, 0.0540, 0.0768, 0.0810, 0.2700, 0.3600]
HLZ_HOLM = [0.0300, 0.0450, 0.0512, 0.0512, 0.0900, 0.0900]
HLZ_BHY = [0.0496, 0.0496, 0.0496, 0.0496, 0.0600, 0.0600]


# ------------------------------------------------------------------ published fixtures
def test_reproduces_the_published_six_test_example():
    assert by_constant(6) == pytest.approx(2.45, abs=0.005)
    assert bonferroni(HLZ_P, ALPHA)["adjusted"] == pytest.approx(HLZ_BONF, abs=5e-5)
    assert holm(HLZ_P, ALPHA)["adjusted"] == pytest.approx(HLZ_HOLM, abs=5e-5)
    assert harvey_liu_bhy(HLZ_P, ALPHA)["adjusted"] == pytest.approx(HLZ_BHY, abs=5e-5)
    assert int(bonferroni(HLZ_P, ALPHA)["reject"].sum()) == 1
    assert int(holm(HLZ_P, ALPHA)["reject"].sum()) == 2
    assert int(benjamini_yekutieli(HLZ_P, ALPHA)["reject"].sum()) == 4


def test_harvey_liu_bhy_is_not_standard_benjamini_yekutieli():
    """Their recursion pins the largest adjusted p at its RAW value. On this family the
    rejection sets agree; on a flat family they do not."""
    std = benjamini_yekutieli(HLZ_P, ALPHA)["adjusted"]
    theirs = harvey_liu_bhy(HLZ_P, ALPHA)["adjusted"]
    assert std[:4] == pytest.approx(theirs[:4])
    assert std[-1] > theirs[-1] and std[-1] == pytest.approx(2.45 * 0.06, abs=5e-4)
    flat = np.full(6, 0.04)
    assert int(benjamini_yekutieli(flat, ALPHA)["reject"].sum()) == 0
    assert int(harvey_liu_bhy(flat, ALPHA)["reject"].sum()) == 6


def test_reproduces_the_published_haircut_illustration():
    """SR 0.75 over 20 years, 200 independent tests -> the paper reports 0.32."""
    h = haircut_sharpe(0.75, 20.0, 200, method="independent")
    assert h["t_stat"] == pytest.approx(3.354, abs=0.001)
    assert h["pvalue"] == pytest.approx(0.0008, abs=5e-5)
    assert h["adjusted_pvalue"] == pytest.approx(0.148, abs=0.002)
    assert h["haircut_sharpe"] == pytest.approx(0.32, abs=0.01)
    assert 55.0 < h["haircut_pct"] < 60.0


def test_by_constant_is_the_harmonic_number():
    assert by_constant(1) == 1.0
    assert by_constant(100) == pytest.approx(5.187, abs=0.001)
    assert by_constant(1000) == pytest.approx(np.log(1000) + np.euler_gamma, abs=0.01)
    with pytest.raises(ValueError):
        by_constant(0)


@pytest.mark.parametrize("method", METHODS)
def test_every_correction_is_monotone_and_never_below_the_raw_p(method):
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 200) ** 2
    r = correct(p, method, ALPHA)
    assert r["adjusted"].shape == p.shape
    assert (r["adjusted"] >= p - 1e-12).all()
    assert (r["adjusted"] <= 1.0 + 1e-12).all()
    order = np.argsort(p)
    assert np.all(np.diff(r["adjusted"][order]) >= -1e-12)   # monotone in p
    assert np.array_equal(r["reject"], r["adjusted"] <= ALPHA)


def test_holm_never_rejects_less_than_bonferroni():
    rng = np.random.default_rng(4)
    for _ in range(50):
        p = rng.uniform(0, 1, 40) ** 3
        b = bonferroni(p, ALPHA)["reject"]
        h = holm(p, ALPHA)["reject"]
        assert (h | b == h).all()          # Holm's rejections are a superset


def test_correct_rejects_bad_inputs():
    with pytest.raises(ValueError, match="method must be"):
        correct([0.1], "sidak")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bonferroni([0.5, 1.5])
    with pytest.raises(ValueError, match="finite"):
        bonferroni([0.5, np.nan])
    with pytest.raises(ValueError, match="no p-values"):
        bonferroni([])


# ------------------------------------------------------------------ the guarantees
def test_family_wise_error_rate_holds_on_seeded_nulls():
    """The load-bearing measurement for the FWER procedures: 200 replications of 60 null
    strategies each. Uncorrected must blow up; Bonferroni and Holm must hold at 5%."""
    reps, m = 200, 60
    hits = {k: 0 for k in METHODS}
    for s in range(reps):
        p = panel_pvalues(strategy_panel(np.random.default_rng(20_000 + s), T, m))
        for name in METHODS:
            hits[name] += int(correct(p, name, ALPHA)["reject"].any())
    rates = {k: v / reps for k, v in hits.items()}
    assert rates["uncorrected"] > 0.90, rates
    ceiling = ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps)
    for name in ("bonferroni", "holm", "benjamini_yekutieli"):
        assert rates[name] <= ceiling, (name, rates)
    assert rates["benjamini_yekutieli"] <= rates["benjamini_hochberg"] + 1e-9


def test_benjamini_hochberg_controls_fdr_under_a_common_factor():
    """A shared factor is positive dependence - the PRDS case BH is proved for."""
    reps, m, n_true = 150, 60, 6
    fdp = 0.0
    for s in range(reps):
        p = panel_pvalues(strategy_panel(np.random.default_rng(41_000 + s), T, m,
                                         n_true=n_true, true_sr_annual=1.5, rho=0.6))
        rej = benjamini_hochberg(p, ALPHA)["reject"]
        d = int(rej.sum())
        fdp += (int(rej[n_true:].sum()) / d) if d else 0.0
    assert fdp / reps <= ALPHA + 0.02, fdp / reps


def test_the_corrections_cost_power_in_the_documented_order():
    reps, m, n_true = 120, 60, 6
    power = {k: 0.0 for k in METHODS}
    for s in range(reps):
        p = panel_pvalues(strategy_panel(np.random.default_rng(31_000 + s), T, m,
                                         n_true=n_true, true_sr_annual=1.5))
        for name in METHODS:
            power[name] += int(correct(p, name, ALPHA)["reject"][:n_true].sum()) / n_true
    power = {k: v / reps for k, v in power.items()}
    assert power["uncorrected"] > power["benjamini_hochberg"] > power["holm"], power
    assert power["holm"] >= power["bonferroni"] - 1e-9
    assert power["benjamini_hochberg"] > power["benjamini_yekutieli"], power


@pytest.mark.parametrize("method", ("bonferroni", "holm", "fdr_bh", "fdr_by"))
def test_matches_statsmodels_when_it_is_installed(method):
    sm = pytest.importorskip("statsmodels.stats.multitest")
    mine = {"bonferroni": "bonferroni", "holm": "holm", "fdr_bh": "benjamini_hochberg",
            "fdr_by": "benjamini_yekutieli"}[method]
    p = np.random.default_rng(5).uniform(0, 1, 300) ** 3
    rej, adj, _, _ = sm.multipletests(p, alpha=ALPHA, method=method)
    r = correct(p, mine, ALPHA)
    assert adj == pytest.approx(r["adjusted"], abs=1e-12)
    assert np.array_equal(rej, r["reject"])


# ------------------------------------------------------------------ Sharpe <-> p
def test_sharpe_t_and_p_round_trip():
    assert t_from_sharpe(0.1, 400) == pytest.approx(2.0)
    assert t_from_annual_sharpe(1.0, 4.0) == pytest.approx(2.0)
    p = p_from_t(3.0, 10_000)
    assert p == pytest.approx(0.0027, abs=1e-4)
    assert sharpe_from_p(p, 9.0) == pytest.approx(1.0, abs=0.01)
    assert independent_pvalue(0.05, 10) == pytest.approx(1 - 0.95 ** 10)
    assert independent_pvalue(0.001, 200) < min(1.0, 0.001 * 200)


def test_the_haircut_is_heavier_on_small_sharpes_than_on_large_ones():
    small = haircut_sharpe(0.5, 10.0, 100)
    large = haircut_sharpe(2.0, 10.0, 100)
    assert small["haircut_pct"] > large["haircut_pct"] + 40.0
    assert large["haircut_pct"] < 25.0 and large["survives"]
    assert not haircut_sharpe(1.0, 4.0, 1000)["survives"]
    with pytest.raises(ValueError, match="step procedure"):
        haircut_sharpe(1.0, 10.0, 50, method="benjamini_hochberg")
    with pytest.raises(ValueError, match="method must be"):
        haircut_sharpe(1.0, 10.0, 50, method="sidak")


# ------------------------------------------------------------------ the ledger
def test_from_ledger_uses_the_registered_count_not_the_scored_one(tmp_path):
    from fin_skills.core.trial_ledger import TrialLedger

    led = TrialLedger(tmp_path / "trials.jsonl")
    panel = strategy_panel(np.random.default_rng(55), T, 100, n_true=4, true_sr_annual=2.0)
    per_period = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
    ids = [led.record("grid", {"variant": i}) for i in range(100)]
    for i in range(20):                       # score 20, abandon 80
        led.complete(ids[i], {"sharpe": float(per_period[i]), "n_obs": T})
    for i in range(20, 100):
        led.abandon(ids[i], "shelved")

    hist = from_ledger(led)
    assert hist.m_registered == 100 and hist.m_scored == 20 and hist.n_unscored == 80
    assert hist.padded_pvalues().size == 100
    assert (hist.padded_pvalues()[20:] == 1.0).all()
    assert hist.notes and "still count toward m" in hist.notes[0]
    assert led.summary()["n_trials"] == hist.m_registered      # the SAME ledger's count

    honest = hist.summary(ALPHA)
    scored_only = compare(hist.pvalues, ALPHA)
    for name in ("bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli"):
        assert honest.loc[name, "survivors"] <= scored_only.loc[name, "survivors"]
        assert honest.loc[name, "m"] == 100
    assert honest.loc["uncorrected", "survivors"] == scored_only.loc["uncorrected", "survivors"]

    table = hist.table(ALPHA)
    assert len(table) == 20 and list(table.columns[:3]) == ["sharpe", "n_obs", "pvalue"]
    assert table["pvalue"].is_monotonic_increasing


def test_from_ledger_accepts_a_path_and_raw_records(tmp_path):
    from fin_skills.core.trial_ledger import TrialLedger

    led = TrialLedger(tmp_path / "t.jsonl")
    tid = led.record("s", {"a": 1})
    led.complete(tid, {"sharpe": 0.1, "n_obs": T})
    by_path = from_ledger(tmp_path / "t.jsonl")
    by_records = from_ledger(led.read())
    assert by_path.m_registered == by_records.m_registered == 1
    assert by_path.pvalues == pytest.approx(by_records.pvalues)


def test_compare_lists_every_method_with_what_it_controls():
    tab = compare(HLZ_P, ALPHA)
    assert list(tab.index) == list(METHODS)
    assert tab.loc["uncorrected", "survivors"] == 5      # 0.06 is the only one above 5%
    assert "arbitrary" in tab.loc["benjamini_yekutieli", "controls"].lower()
    assert uncorrected(HLZ_P, ALPHA)["m"] == 6


def test_a_seeded_run_is_deterministic():
    a = panel_pvalues(strategy_panel(np.random.default_rng(1), T, 30))
    b = panel_pvalues(strategy_panel(np.random.default_rng(1), T, 30))
    assert a == pytest.approx(b)
    assert correct(a, "holm")["adjusted"] == pytest.approx(correct(b, "holm")["adjusted"])


@pytest.mark.slow
def test_main_prints_the_measurements_and_the_rule(run_main):
    out = run_main("fin_skills.core.research_history")
    assert out.isascii()
    assert "RULE:" in out
    assert "FAMILY-WISE ERROR RATE" in out and "abandoned" in out
    assert "0.0496" in out                       # the reproduced BHY column
