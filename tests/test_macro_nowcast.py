"""fin_skills.macro.nowcast - a mixed-frequency dynamic factor nowcast on a ragged edge.

Properties under test: the Mariano-Murasawa aggregation is what the model observes; the
ragged view is causal and jagged; the Kalman filter recovers a known factor and its
smoothed pass beats its filtered one; the nowcast's error falls as the quarter fills in
while the benchmarks' does not; DynamicFactorMQ's docstring says what the skill quotes and
its fit recovers the factor up to the sign the docstring warns about.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires
from fin_skills.macro import nowcast as nw


@pytest.fixture(scope="module")
def panel():
    return nw.simulate_panel()


def test_mariano_murasawa_weights_and_quarterly_observation_pattern(panel):
    assert nw.MM_WEIGHTS.tolist() == [1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3]
    assert nw.MM_WEIGHTS.sum() == pytest.approx(3.0)
    q = panel["q_end"]
    assert (q % 3 == 2).all()                       # third month of each quarter
    assert np.all(np.diff(q) == 3)
    y = panel["y"]
    assert np.isfinite(y[q]).all()
    mask = np.ones(y.size, dtype=bool)
    mask[q] = False
    assert np.isnan(y[mask]).all()                  # and nowhere else
    # the observed quarterly value IS the weighted average of monthly growth, plus noise
    g = panel["gdp_monthly"]
    t = int(q[40])
    agg = float(nw.MM_WEIGHTS @ g[t - 4:t + 1][::-1])
    assert abs(y[t] - agg) < 6 * panel["sig_g"]


def test_simulation_is_seeded(panel):
    again = nw.simulate_panel()
    assert np.array_equal(panel["f"], again["f"])
    assert np.allclose(panel["X"], again["X"], equal_nan=True)
    assert not np.array_equal(panel["f"], nw.simulate_panel(seed=nw.SEED + 1)["f"])


def test_ragged_is_causal_and_actually_jagged(panel):
    v = 292
    r = nw.ragged(panel, v)
    assert np.isnan(r["X"][v + 1:]).all() and np.isnan(r["y"][v + 1:]).all()
    avail = np.isfinite(r["X"][v - 3:v + 1]).sum(axis=1)
    assert avail[-1] < avail[0]                     # the edge narrows toward the vintage
    assert len(set(avail.tolist())) > 1             # jagged, not a clean rectangle
    for i, lag in enumerate(nw.PUB_LAGS):
        if lag:
            assert np.isnan(r["X"][v + 1 - lag:v + 1, i]).all()
        else:
            assert np.isfinite(r["X"][v, i])
    # a dropna() over the ragged panel throws most of it away
    keep = np.isfinite(r["X"]).all(axis=1) & np.isfinite(r["y"])
    assert keep.sum() < 0.4 * (v + 1)
    es = nw.edge_shape(r["X"], r["y"], v)
    assert len(es) == 5 and es["indicators_available"].iloc[-1] < es["of"].iloc[-1]


def test_state_space_matrices_carry_the_aggregation(panel):
    T, Q, Z, R = nw.build_ss(panel["phi"], panel["loadings"], panel["noise"],
                             panel["lambda_g"], panel["sig_g"])
    assert T.shape == (nw.NLAG, nw.NLAG) and T[0, 0] == panel["phi"]
    assert np.allclose(T[1:, :-1], np.eye(nw.NLAG - 1))    # companion shift
    assert Q[0, 0] == 1.0 and Q[1:, 1:].sum() == 0.0
    k = panel["loadings"].size
    assert np.allclose(Z[:k, 0], panel["loadings"])
    assert Z[:k, 1:].sum() == 0.0                          # monthlies see only f_t
    assert np.allclose(Z[k], panel["lambda_g"] * nw.MM_WEIGHTS)
    assert R.size == k + 1


def test_the_filter_recovers_a_known_factor_and_the_smoother_beats_it(panel):
    rec = nw.factor_recovery(panel, 292)
    assert rec["corr_filtered"] > 0.9
    assert rec["corr_smoothed"] > rec["corr_filtered"]
    assert rec["rmse_smoothed"] < rec["rmse_filtered"]
    st = rec["state"]
    assert st["filtered"].shape[1] == nw.NLAG
    # the state really is a lag stack: today's f_{t-1} is yesterday's f_t
    sm = st["smoothed"]
    assert np.allclose(sm[1:, 1], sm[:-1, 0], atol=1e-8)


def test_the_edge_revision_is_the_ragged_edge_closing(panel):
    tab = nw.edge_penalty(panel, 292)
    assert len(tab) == 6
    assert tab["n_series"].iloc[-1] < tab["n_series"].iloc[0]
    assert (tab["revision"] != 0).all()
    assert tab["revision"].abs().mean() > 0
    # settling with more data is, on average, closer to the truth
    assert tab["err_settled"].abs().mean() <= tab["err_now"].abs().mean()


def test_the_nowcast_improves_as_the_quarter_fills_in_and_the_benchmarks_do_not(panel):
    ev = nw.evaluation(panel, start_q=90, offsets=(-4, -2, 0), step=4)
    assert list(ev.index) == [-4, -2, 0]
    assert 10 < ev["n"].iloc[0] < 30
    dfm = ev["rmse_dfm"].to_numpy()
    assert (np.diff(dfm) < 0).all()                  # improving as the quarter fills in
    assert dfm[-1] < 0.5 * dfm[0]
    # the mean and the AR(1) never read the monthly flow, so they are flat
    for col in ("rmse_mean", "rmse_ar1"):
        late = ev[col].loc[[-2, 0]].to_numpy()
        assert np.allclose(late, late[0])
    assert (ev["dfm_vs_ar1"] < 1.0).all()            # it beats the AR(1) everywhere
    assert ev.loc[0, "dfm_vs_bridge"] < 1.0
    assert np.isnan(ev.loc[-4, "rmse_bridge"])       # no monthly data before the quarter


def test_nowcast_at_projects_when_the_target_is_past_the_vintage(panel):
    q = int(panel["q_end"][60])
    near = nw.nowcast_at(panel, q, q)
    far = nw.nowcast_at(panel, q, q - 4)
    assert np.isfinite(near) and np.isfinite(far)
    assert abs(near - panel["y"][q]) < abs(far - panel["y"][q]) + 5.0
    bm = nw.benchmarks(panel, q, q)
    assert set(bm) >= {"mean", "ar1", "bridge"}
    assert all(np.isfinite(v) for v in bm.values())


@requires("statsmodels")
def test_dynamicfactormq_docstring_says_what_the_skill_quotes():
    facts = nw.dfmq_facts()
    assert facts["available"] is True
    assert all(facts["claims"].values()), facts["claims"]
    for who in ("Modugno", "Giannone", "Bok", "Mariano"):
        assert who in facts["cited"]
    # the docstring's prose years and its reference years disagree
    assert "2017" in facts["prose_years"] and "2018" in facts["reference_years"]
    assert "2011" in facts["prose_years"] and "2010" in facts["reference_years"]


@requires("statsmodels")
@pytest.mark.slow
def test_dynamicfactormq_recovers_the_factor_up_to_sign_and_scale(panel):
    fit = nw.fit_dfmq(panel, n_months=150, maxiter=40)
    assert fit is not None
    assert fit["corr_factor_vs_truth"] > 0.9
    assert fit["corr_factor_vs_numpy"] > 0.95
    assert abs(fit["phi_hat"] - fit["phi_true"]) < 0.15
    # loadings line up with lambda_i / sd(x_i) after one common scale
    assert fit["loading_corr"] > 0.95
    assert fit["loading_max_rel_err"] < 0.35
    # the sign is free, so a raw loading is not interpretable on its own
    assert np.all(np.sign(fit["loadings_hat"]) == np.sign(fit["loadings_hat"][0]))


@pytest.mark.slow
def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.macro.nowcast")
    for head in ("1. The panel, and the ragged edge",
                 "2. Factor recovery at the true parameters",
                 "3. The nowcast against the benchmarks",
                 "4. statsmodels' DynamicFactorMQ"):
        assert head in out
    assert "0.77" in out and "1.17" in out                  # GDPNow's published accuracy
    flat = " ".join(out.split())
    assert "suspended between September 2021 and September 2023" in flat  # the NY Fed hole
    assert out.strip().splitlines()[-1].startswith("Rule:")
    assert all(ord(c) < 128 for c in out)
