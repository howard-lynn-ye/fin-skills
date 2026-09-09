"""fin_skills.ml.triple_barrier - which barrier came first, and what the horizon label hides.

The properties the SKILL.md claims:

  * the EWMA target volatility is causal and IS pandas' `ewm(span=100).std()` on arithmetic
    returns (mlfinpy get_daily_vol / AFML Snippet 3.1);
  * `first_touch` returns the EARLIEST crossing, uses arithmetic returns and a strict
    inequality, respects `side`, and never looks past the vertical barrier;
  * `min_ret` drops events rather than skipping barriers;
  * a fixed barrier width makes the label distribution move with the volatility regime and a
    volatility-scaled one does not;
  * the sign-only fixed-horizon label disagrees with the path, and the disagreement grows as the
    stop tightens;
  * `pt_sl` sets the class balance on a series with zero drift;
  * Snippet 3.2's arithmetic touch and Snippet 3.9's log relabelling agree at pt = 1 and diverge
    above it.
"""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.ml import triple_barrier as tb


@pytest.fixture(scope="module")
def series():
    close, sigma = tb.stochastic_vol_prices()
    return close, sigma, tb.ewma_vol(close)


@pytest.fixture(scope="module")
def events(series):
    close, _, vol = series
    t_events = np.arange(tb.WARMUP, tb.N_BARS - tb.VERT_BARS - 1, tb.EVENT_STEP)
    return t_events, tb.triple_barrier_events(close, t_events, vol)


def test_the_generator_is_seeded_and_volatility_clusters(series):
    close, sigma, _ = series
    close2, sigma2 = tb.stochastic_vol_prices()
    assert np.array_equal(close, close2) and np.array_equal(sigma, sigma2)
    assert not np.array_equal(close, tb.stochastic_vol_prices(seed=tb.SEED + 1)[0])
    assert close.shape == (tb.N_BARS,) and (close > 0).all()
    lv = np.log(sigma)
    assert np.corrcoef(lv[:-1], lv[1:])[0, 1] > 0.95         # clustered by construction
    assert sigma.max() / sigma.min() > 10


def test_ewma_vol_is_pandas_ewm_std_and_is_causal(series):
    close, _, vol = series
    ref = tb.pandas_ewma_vol(close)
    ok = np.isfinite(ref)
    assert np.max(np.abs(vol[ok] - ref[ok])) < 1e-12
    assert np.isnan(vol[0]) and np.isnan(vol[1])             # needs two returns
    shocked = close.copy()
    shocked[3000] *= 1.5
    v2 = tb.ewma_vol(shocked)
    assert np.array_equal(v2[:3000], vol[:3000], equal_nan=True)
    assert v2[3000] != vol[3000]


def test_first_touch_takes_the_earliest_crossing_and_honours_side():
    close = np.array([100.0, 101.0, 90.0, 130.0, 100.0])
    assert tb.first_touch(close, 0, 4, 0.05, -0.05) == (2, "sl")   # -10 % at bar 2 beats +30 %
    assert tb.first_touch(close, 0, 1, 0.05, -0.05) == (1, "vert")  # vertical barrier binds
    assert tb.first_touch(close, 0, 4, 0.05, None) == (3, "pt")     # stop disabled
    # side = -1 flips which barrier the same path touches
    assert tb.first_touch(close, 0, 4, 0.05, -0.05, side=-1.0) == (2, "pt")
    # strict inequality: exactly on the barrier is not a touch
    flat = np.array([100.0, 105.0, 105.0])
    exact = float(flat[1] / flat[0] - 1.0)
    assert tb.first_touch(flat, 0, 2, exact, -1.0) == (2, "vert")
    assert tb.first_touch(flat, 0, 2, np.nextafter(exact, 0.0), -1.0) == (1, "pt")
    # the scan never runs past the end of the series
    assert tb.first_touch(close, 3, 99, 9.0, -9.0)[0] == 4


def test_events_are_labelled_by_the_barrier_they_touched(series, events):
    close, _, vol = series
    t_events, ev = events
    assert set(np.unique(ev["touched"])) <= {"pt", "sl", "vert"}
    assert np.array_equal(ev["label"], np.where(ev["touched"] == "pt", 1.0,
                                                np.where(ev["touched"] == "sl", -1.0, 0.0)))
    assert (ev["t1"] > ev["t0"]).all()
    assert (ev["t1"] - ev["t0"] <= tb.VERT_BARS).all()
    assert np.allclose(ev["ret"], close[ev["t1"]] / close[ev["t0"]] - 1.0)
    # every pt touch really is above its barrier, every sl touch below
    assert (ev["ret"][ev["touched"] == "pt"] > ev["pt"][ev["touched"] == "pt"]).all()
    assert (ev["ret"][ev["touched"] == "sl"] < ev["sl"][ev["touched"] == "sl"]).all()
    # a vertical resolution stayed inside both barriers at its own end bar
    v = ev["touched"] == "vert"
    assert (ev["ret"][v] <= ev["pt"][v]).all() and (ev["ret"][v] >= ev["sl"][v]).all()


def test_min_ret_drops_events_rather_than_skipping_barriers(series, events):
    close, _, vol = series
    t_events, ev = events
    cut = float(np.nanmedian(vol[t_events]))
    small = tb.triple_barrier_events(close, t_events, vol, min_ret=cut)
    assert small["t0"].shape[0] < ev["t0"].shape[0]
    assert (vol[small["t0"]] > cut).all()
    assert set(small["t0"].tolist()) <= set(ev["t0"].tolist())


def test_meta_labels_are_zero_one_and_zero_whenever_the_bet_lost(series, events):
    close, _, vol = series
    t_events, _ = events
    side = np.where(np.arange(t_events.shape[0]) % 2 == 0, 1.0, -1.0)
    meta = tb.triple_barrier_events(close, t_events, vol, side=side)
    assert set(np.unique(meta["label"])) <= {0.0, 1.0}
    assert (meta["ret"][meta["label"] == 1.0] > 0).all()
    assert np.array_equal(meta["side"], side)


def test_a_fixed_barrier_width_moves_the_label_with_the_volatility_regime(series, events):
    close, _, vol = series
    t_events, ev = events
    fixed = np.full_like(vol, float(np.nanmedian(vol[t_events])))
    ev_fixed = tb.triple_barrier_events(close, t_events, fixed)
    terc = np.quantile(vol[ev["t0"]], [1 / 3, 2 / 3])
    grp = np.digitize(vol[ev["t0"]], terc)
    lo, hi = grp == 0, grp == 2
    spread = lambda e: abs(np.mean(e["touched"][lo] == "vert")
                           - np.mean(e["touched"][hi] == "vert"))
    assert spread(ev_fixed) > 3.0 * spread(ev)               # 7.7x in the demo
    assert np.mean(ev_fixed["touched"][lo] == "vert") > np.mean(ev_fixed["touched"][hi] == "vert")


def test_the_sign_only_horizon_label_has_no_neutral_class(series, events):
    close, _, _ = series
    _, ev = events
    fh = tb.fixed_horizon_labels(close, ev["t0"], tb.VERT_BARS, 0.0)
    assert np.mean(fh["label"] == 0.0) == 0.0
    assert np.mean(ev["label"] == 0.0) > 0.2                 # 41.2 % in the demo
    banded = tb.fixed_horizon_labels(close, ev["t0"], tb.VERT_BARS, tb.PT_SL[0] * ev["trgt"])
    assert np.mean(banded["label"] == 0.0) > np.mean(ev["label"] == 0.0)
    assert (banded["t1"] - banded["t0"] <= tb.VERT_BARS).all()


def test_horizon_labels_describe_paths_that_were_stopped_out(series, events):
    close, _, vol = series
    t_events, ev = events
    pc = tb.path_conflicts(close, ev, tb.VERT_BARS)
    n = pc["n"]
    assert pc["label_disagree"] / n > 0.3                    # 47.1 % in the demo
    assert pc["wrong_about_path"] > 0
    # +1 and -1 are mutually exclusive, so the two counts partition the conflicts exactly
    assert pc["wrong_about_path"] == pc["fh_up_but_stopped"] + pc["fh_dn_but_ran_up"]
    # a tighter stop makes more of the horizon labels untradeable
    fracs = []
    for m in (1.0, 2.0, 4.0):
        e_m = tb.triple_barrier_events(close, t_events, vol, (m, m))
        fracs.append(tb.path_conflicts(close, e_m)["wrong_about_path"] / n)
    assert fracs[0] > fracs[1] > fracs[2]
    assert fracs[0] > 0.3                                    # 43.7 % at a 1-sigma stop
    # honouring the barriers costs the perfect-model ceiling real profit
    assert pc["pnl_fixed_horizon"] > pc["pnl_with_stop"] > 0


def test_excursions_bracket_the_realised_return(series, events):
    close, _, _ = series
    _, ev = events
    exc = tb.excursions(close, ev["t0"], tb.VERT_BARS)
    assert (exc["mfe"] >= exc["mae"]).all()
    assert np.mean(exc["mae"] < 0.0) > 0.8          # a path that never dips is possible, not usual
    fh = tb.fixed_horizon_labels(close, ev["t0"], tb.VERT_BARS, 0.0)
    assert (fh["ret"] <= exc["mfe"] + 1e-12).all()
    assert (fh["ret"] >= exc["mae"] - 1e-12).all()
    short = tb.excursions(close, ev["t0"], tb.VERT_BARS, side=-1.0)
    assert np.allclose(short["mae"], -exc["mfe"]) and np.allclose(short["mfe"], -exc["mae"])


def test_pt_sl_sets_the_class_balance_on_a_driftless_series(series, events):
    close, _, vol = series
    t_events, _ = events
    up = tb.triple_barrier_events(close, t_events, vol, (1.0, 2.0))
    dn = tb.triple_barrier_events(close, t_events, vol, (2.0, 1.0))
    assert np.mean(up["label"] == 1.0) > 0.55                # 61.4 % in the demo
    assert np.mean(dn["label"] == -1.0) > 0.55               # 58.6 %
    no_stop = tb.triple_barrier_events(close, t_events, vol, (1.0, 0.0))
    assert np.mean(no_stop["label"] == -1.0) == 0.0          # the stop is disabled
    assert np.mean(no_stop["label"] == 1.0) > 0.6
    for e in (up, dn, no_stop):
        assert abs(np.mean(e["ret"])) < 0.01                 # no drift, whatever the labels say


def test_the_log_relabelling_agrees_at_pt_one_and_diverges_above_it(series, events):
    close, _, vol = series
    t_events, _ = events
    prev = -1
    for m in (1.0, 2.0, 3.0):
        e = tb.triple_barrier_events(close, t_events, vol, (m, m))
        re_lab = tb.barrier_touched_relabel(e["ret"], e["trgt"], m, m)
        horiz = e["touched"] != "vert"
        flipped = int(np.sum(horiz & (re_lab == 0.0)))
        if m == 1.0:
            assert flipped == 0                              # the two conventions coincide
            assert np.array_equal(re_lab[horiz], e["label"][horiz])
        else:
            assert flipped > prev
        prev = flipped
    assert prev > 0


def test_demo_runs_every_section_and_prints_the_rule(run_main):
    out = run_main("fin_skills.ml.triple_barrier")
    for head in ("=== 1. Volatility-scaled barriers", "=== 2. Label distributions",
                 "=== 3. How often the fixed-horizon label",
                 "=== 4. The drawdown", "=== 5. pt_sl is a design choice",
                 "=== 6. Arithmetic vs log"):
        assert head in out
    assert out.isascii()
    rule = [ln for ln in out.splitlines() if ln.startswith("Rule: ")]
    assert len(rule) == 1 and "first" in rule[0] and "stop" in rule[0]
    assert out.strip().splitlines()[-1].startswith("total runtime")
