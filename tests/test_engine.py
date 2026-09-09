"""fin_skills.engine - the twelve invariants, six of them proven by the library's own guards.

The engine's claim is not speed or realism, it is falsifiability: `assert_causal`,
`adjustment_check`, `survivorship_audit`, `pit_universe`, `cost_curve` and `rf_convention`
are run HERE on the engine's own output, through `to_bundle()`, with no adapter written
for any of them. The bespoke assertions cover the five properties no shipped guard can
see - the fill bar, the turnover identity, the calendar, the participation cap and
determinism - plus the one guard (fold_leak_test) that takes `run` itself as its argument.
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from _helpers import REPO_ROOT
from fin_skills.api import check, get
from fin_skills.engine import (Costs, Execution, Panel, Sessions, Universe, equal_weight,
                               fixed_bps, fractional_kelly, from_weights, invariants,
                               rank_long_short, run, spread_share, square_root_impact,
                               vol_target, volume_share)

N_DAYS, N_NAMES = 220, 6
DEAD, SPLIT = "N04", "N00"
TZ, PPY = "America/New_York", 252


# ------------------------------------------------------------------ the synthetic world
def sessions(n: int = N_DAYS) -> Sessions:
    return Sessions.from_index(pd.bdate_range("2021-01-04", periods=n), tz=TZ,
                               periods_per_year=PPY, name="XNYS")


def raw_frames(n: int = N_DAYS, n_names: int = N_NAMES, seed: int = 11, split: bool = True):
    """Cent-struck RAW quotes: one 3:1 split that really jumps, one name that stops trading."""
    idx = pd.bdate_range("2021-01-04", periods=n)
    cols = [f"N{i:02d}" for i in range(n_names)]
    rng = np.random.default_rng(seed)
    path = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, (n, n_names)), axis=0))
    close = pd.DataFrame(np.round(path, 2), index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(20_000, 90_000, (n, n_names)).astype(float),
                          index=idx, columns=cols)
    split_on = idx[n // 2]
    actions = pd.DataFrame([{"date": split_on, "ticker": SPLIT, "ratio": 3.0, "kind": "split"}])
    if split:
        close.loc[close.index >= split_on, SPLIT] = np.round(
            close.loc[close.index >= split_on, SPLIT] / 3.0, 2)
        volume.loc[volume.index >= split_on, SPLIT] *= 3.0
    end = idx[int(n * 0.6)]
    if DEAD in cols:
        close.loc[close.index > end, DEAD] = np.nan
        volume.loc[volume.index > end, DEAD] = np.nan
    listings = pd.DataFrame([{"ticker": c, "start_date": idx[0],
                              "end_date": end if c == DEAD else pd.NaT} for c in cols])
    return close, volume, actions if split else None, listings


def build_panel(adjustment: str = "back", split: bool = True, **kw) -> Panel:
    close, volume, actions, listings = raw_frames(split=split, **kw)
    p = Panel(open=close.shift(1).fillna(close), high=close * 1.01, low=close * 0.99,
              close=close, volume=volume, sessions=sessions(len(close)), adjustment="raw",
              actions=actions, listings=listings, source="synthetic (tests/test_engine.py)",
              retrieved_at="2026-09-09T00:00Z")
    return p if adjustment == "raw" else p.readjust(adjustment)


def rebuild(base: Panel, close: pd.DataFrame) -> Panel:
    """A panel from a wide close frame - the shape assert_causal perturbs."""
    return Panel(open=close.shift(1).fillna(close), high=close * 1.01, low=close * 0.99,
                 close=close, volume=base.volume, sessions=base.sessions,
                 adjustment=base.adjustment, listings=base.listings, source=base.source,
                 retrieved_at=base.retrieved_at)


def momentum(p: Panel) -> pd.DataFrame:
    lr = np.log(p.close).diff()
    return lr.rolling(40).sum() - lr.rolling(5).sum()


def one_name(n: int = 60, flat_open: bool = True):
    """A single name whose first two closes are equal, so a fully invested book holds no
    cash and buy-and-hold is EXACTLY the asset. Volume is tiny, for the cap tests."""
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(5)
    px = np.round(50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), 2)
    px[1] = px[0]
    close = pd.DataFrame({"AAA": px}, index=idx)
    op = close.copy() if flat_open else close.shift(1).fillna(close)
    vol = pd.DataFrame({"AAA": 10_000.0}, index=idx)
    return Panel(open=op, high=close * 1.001, low=close * 0.999, close=close, volume=vol,
                 sessions=Sessions.from_index(idx, tz=TZ, periods_per_year=PPY, name="XNYS"),
                 adjustment="raw")


def constant(p: Panel) -> pd.DataFrame:
    return pd.DataFrame(1.0, index=p.close.index, columns=p.close.columns)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(PPY))


@pytest.fixture(scope="module")
def full():
    """One realistic run: back-adjusted panel, PIT universe, real costs, a benchmark."""
    p = build_panel("back")
    bench = p.close.pct_change(fill_method=None).mean(axis=1).fillna(0.0).rename("bench")
    return run(p, momentum,
               execution=Execution(fill_at="next_open", signal_lag=1, participation_cap=0.05),
               costs=Costs(commission_bps=0.5, spread_bps=1.0, slippage=square_root_impact(),
                           borrow_bps_annual=50.0, cash_rate=0.05),
               sizing=rank_long_short(quantile=0.25),
               universe=Universe(rebalance="ME", adv_lookback=21, members=p.listings),
               capital=10_000_000.0, benchmark=bench, strict=True)


# ================================================================== I1  causality
def test_engine_is_causal():
    """assert_causal on the ENGINE, not just on a signal: perturb bars >= k, and no cell of
    `weights` before k may move. This is the guard's own falsification, run on run()."""
    base = build_panel("raw", split=False)

    def engine_weights(close: pd.DataFrame) -> pd.DataFrame:
        return run(rebuild(base, close), momentum, sizing=rank_long_short(),
                   universe=Universe(rebalance="ME", max_names=4), strict=False).weights

    engine_weights.__name__ = "run(panel, momentum).weights"
    res = get("assert_causal").run(fn=engine_weights, df=base.close, k=len(base.close) // 2)
    assert res.passed, res.summary()
    assert res.evidence["k"] == len(base.close) // 2


# ================================================================== I2  signal lag
def test_signal_lag_zero_is_rejected():
    with pytest.raises(ValueError, match=r"signal_lag must be >= 1"):
        Execution(signal_lag=0)
    with pytest.raises(ValueError, match="vectorbt"):     # the message names the defect
        Execution(signal_lag=0)
    for kwargs in ({"fill_at": "this_close"}, {"on_delist": "ignore"}, {"on_halt": "guess"},
                   {"participation_cap": 0.0}, {"max_gross": -1.0}):
        with pytest.raises(ValueError):
            Execution(**kwargs)


def test_an_oracle_signal_is_reported_as_a_leak():
    """A perfect oracle passes every ENGINE invariant and fails assert_causal through the
    same bundle - the engine reports the leak instead of laundering it."""
    p = build_panel("back")

    def oracle(q: Panel) -> pd.DataFrame:
        return q.close.pct_change(fill_method=None).shift(-1)

    res = run(p, oracle, sizing=rank_long_short(), strict=True)
    assert res.findings == []                       # the engine itself has nothing to say
    assert sharpe(res.gross_returns) > 5.0          # and the curve is beautiful
    report = res.check(guards=["assert_causal"])
    assert report.ran == ["assert_causal"] and not report.passed
    assert "LOOK-AHEAD" in report[0].errors[0].message


# ================================================================== I3  no same-bar fill
def test_no_same_bar_fill():
    """open[t] is a sentinel 1.5 * close[t-1], so a fill price names its own bar."""
    base = build_panel("raw", split=False)
    op = (base.close.shift(1) * 1.5).fillna(base.close * 1.5)
    p = Panel(open=op, high=base.high, low=base.low, close=base.close, volume=base.volume,
              sessions=base.sessions, adjustment="raw", listings=base.listings)
    res = run(p, momentum, execution=Execution(fill_at="next_open", participation_cap=None),
              sizing=rank_long_short(), strict=True)
    trades = res.fills[res.fills["kind"] == "trade"]
    assert len(trades) > 50
    for row in trades.itertuples(index=False):
        assert row.price == pytest.approx(float(op.loc[row.date, row.ticker]))
        assert row.date == p.sessions.shift(row.decided_on, 1)
        assert row.price != pytest.approx(float(base.close.loc[row.decided_on, row.ticker]))
    assert res.findings == []


# ================================================================== I4  adjustment
def test_forward_adjusted_panel_labelled_back_is_rejected():
    raw = build_panel("raw")
    fwd = raw.readjust("forward")
    mislabelled = Panel(open=fwd.open, high=fwd.high, low=fwd.low, close=fwd.close,
                        volume=fwd.volume, sessions=fwd.sessions, adjustment="back",
                        actions=fwd.actions, listings=fwd.listings)
    with pytest.raises(ValueError, match="I4"):
        run(mislabelled, momentum, strict=True)
    res = run(mislabelled, momentum, strict=False)
    assert [f for f in res.findings if "I4" in f.where]
    assert "forward-adjusted" in str(res.findings[0])
    # the correctly labelled panel passes, and adjustment_check agrees through the bundle
    ok = run(raw.readjust("back"), momentum, strict=True)
    report = ok.check(guards=["adjustment_check"])
    assert report.ran == ["adjustment_check"] and report.passed
    assert report[0].evidence["convention"] == "back-adjusted"


def test_readjust_refuses_without_an_event_table_and_round_trips():
    bare = build_panel("raw", split=False)
    with pytest.raises(ValueError, match="unfalsifiable|cannot be checked"):
        bare.readjust("back")
    raw = build_panel("raw")
    back = raw.readjust("back")
    assert back.adjustment == "back" and back.fingerprint() != raw.fingerprint()
    pd.testing.assert_frame_equal(back.readjust("raw").close, raw.close)
    # returns are convention-invariant; price LEVELS are not - the whole point of the field
    fwd = raw.readjust("forward")
    pd.testing.assert_frame_equal(back.close.pct_change(fill_method=None),
                                  fwd.close.pct_change(fill_method=None))
    assert not np.allclose(back.close[SPLIT].dropna(), fwd.close[SPLIT].dropna())


@pytest.mark.skipif(not (REPO_ROOT / "benchmarks" / "leak_bench.py").is_file(),
                    reason="benchmarks/ not present in this checkout")
def test_readjust_matches_leak_benchs_own_definitions():
    """The conventions are leak_bench's, not new ones: same arithmetic, asserted."""
    name = "leak_bench_for_engine"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "benchmarks" / "leak_bench.py")
    lb = importlib.util.module_from_spec(spec)
    sys.modules[name] = lb          # leak_bench defines dataclasses, which look themselves up
    try:
        spec.loader.exec_module(lb)
    finally:
        sys.modules.pop(name, None)
    raw = build_panel("raw")
    pd.testing.assert_frame_equal(raw.readjust("back").close,
                                  lb.back_adjust(raw.close, raw.actions))
    pd.testing.assert_frame_equal(raw.readjust("forward").close,
                                  lb.forward_adjust(raw.close, raw.actions))


# ================================================================== I5  delisting
def test_delisting_closes_out():
    p = build_panel("back")
    end = pd.Timestamp(p.listings.set_index("ticker").loc[DEAD, "end_date"])
    kw = dict(sizing=equal_weight(long_only=False),
              universe=Universe(rebalance="ME", members=p.listings), strict=True)
    at_last = run(p, momentum, execution=Execution(on_delist="close_at_last"), **kw)
    at_zero = run(p, momentum, execution=Execution(on_delist="zero_recovery"), **kw)

    # 1. the position goes to zero ON the delist session and never returns
    assert at_last.positions.loc[:end, DEAD].abs().max() > 0
    assert float(at_last.positions.loc[end:, DEAD].abs().max()) == 0.0
    closeout = at_last.fills[(at_last.fills["kind"] == "closeout")
                             & (at_last.fills["ticker"] == DEAD)]
    assert len(closeout) == 1 and closeout.iloc[0]["date"] == end

    # 2. the closeout return is BOOKED on that session, at exactly the position's value
    row = closeout.iloc[0]
    assert row["price"] == pytest.approx(float(p.close.ffill().loc[end, DEAD]))
    recovered = -float(row["side"]) * float(row["shares"]) * float(row["price"])
    prev_equity = float(at_last.equity.shift(1)[end])
    assert at_zero.returns[end] < at_last.returns[end]
    assert (at_last.returns[end] - at_zero.returns[end]) == pytest.approx(
        recovered / prev_equity, rel=1e-9)

    # 3. the name is absent from every later universe value
    assert any(DEAD in v for d, v in at_last.universe.items() if d < end)
    assert all(DEAD not in v for d, v in at_last.universe.items() if d > end)

    # 4. survivorship_audit passes through the bundle, listings and all
    report = at_last.check(guards=["survivorship_audit"])
    assert report.ran == ["survivorship_audit"] and report.passed, report.summary()
    assert report[0].evidence["n_ended_early"] == 1


def test_on_delist_raise_refuses_to_guess_a_recovery():
    p = build_panel("back")
    with pytest.raises(RuntimeError, match="stops trading"):
        run(p, momentum, execution=Execution(on_delist="raise"),
            sizing=equal_weight(long_only=False))


# ================================================================== I6  point-in-time
def test_universe_is_point_in_time():
    base = build_panel("raw", split=False)
    uni = Universe(rebalance="ME", adv_lookback=21, max_names=4, members=base.listings)
    k = len(base.close) // 2
    bumped = base.close.copy()
    bumped.iloc[k:, 1] *= 12.0          # one name only: a uniform bump cannot change a ranking
    a, b = uni.build(base), uni.build(rebuild(base, bumped))
    cut = base.close.index[k]
    assert list(a) == list(b)
    assert all(a[d] == b[d] for d in a if d < cut)          # nothing before k moved
    assert any(a[d] != b[d] for d in a if d > cut)          # and it DOES react after
    report = run(base, momentum, universe=uni, strict=True).check(guards=["pit_universe"])
    assert report.ran == ["pit_universe"] and report.passed, report.summary()
    assert report[0].evidence["total_removals"] > 0


# ================================================================== I7  turnover
def test_turnover_identity(full):
    want = full.weights.diff().abs().sum(axis=1)
    want.iloc[0] = float(full.weights.iloc[0].abs().sum())
    assert float((full.turnover - want).abs().max()) < 1e-12
    assert full.turnover.min() >= 0.0


def test_buy_and_hold_reproduces_the_asset():
    p = one_name()
    res = run(p, constant, execution=Execution(participation_cap=None), sizing=equal_weight(),
              costs=Costs(), capital=100_000.0, strict=True)
    assert len(res.fills) == 1 and res.fills.iloc[0]["date"] == p.sessions.index[1]
    got, want = res.gross_returns.iloc[2:], p.close["AAA"].pct_change().iloc[2:]
    assert float((got - want).abs().max()) < 1e-12
    assert float(res.turnover.iloc[1:].abs().max()) < 1e-12
    assert float((res.returns - res.gross_returns).abs().max()) == 0.0   # zero costs
    assert res.findings == []


# ================================================================== I8  costs
def test_cost_curve_is_monotone(full):
    assert float((full.net(0) - full.gross_returns).abs().max()) == 0.0
    assert full.net() is full.returns
    curve = [sharpe(full.net(b)) for b in (0, 5, 10, 20, 50)]
    assert all(a >= b for a, b in zip(curve, curve[1:])), curve
    report = check(full.to_bundle(), guards=["cost_curve"])
    assert report.ran == ["cost_curve"]
    guard_curve = report[0].evidence["curve"]["sharpe"]
    assert guard_curve.is_monotonic_decreasing
    assert report[0].evidence["cost_bps"] == pytest.approx(3.0)   # 2*(0.5 + 1.0 + 0)


def test_round_trip_bps_states_only_what_it_knows():
    assert Costs(commission_bps=0.5, spread_bps=1.0).round_trip_bps() == pytest.approx(3.0)
    assert Costs(slippage=fixed_bps(2.0)).round_trip_bps() == pytest.approx(4.0)
    # an impact model has no context-free number, so it contributes nothing here
    assert Costs(slippage=square_root_impact()).round_trip_bps() == 0.0


# ================================================================== I9  calendar
def test_calendar_is_declared_not_inferred():
    close, volume, _, listings = raw_frames(split=False)
    idx = close.index
    holiday = idx[50]
    short = Sessions.from_index(idx.delete(50), tz=TZ, periods_per_year=PPY, name="XNYS")
    long = close.stack(future_stack=True).rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    for f in ("open", "high", "low"):
        long[f] = long["close"]
    long["volume"] = volume.stack(future_stack=True).to_numpy()
    with pytest.raises(ValueError, match="not sessions on the declared calendar"):
        Panel.from_bars(long, short, adjustment="raw")
    p = Panel.from_bars(long[long["date"] != holiday], short, adjustment="raw",
                        listings=listings)
    res = run(p, momentum, strict=True)
    assert holiday not in res.returns.index
    assert res.returns.index.equals(short.index)
    assert res.weights.index.equals(short.index) and res.fills["date"].isin(short.index).all()
    # a 21-BAR window spans 21 sessions, which is more than 21 calendar days
    assert short.bars_between(short.index[100], short.index[121]) == 21
    assert (short.index[121] - short.index[100]).days > 21


def test_sessions_refuse_what_cannot_be_reproduced():
    idx = pd.bdate_range("2021-01-04", periods=10)
    with pytest.raises(ValueError, match="tz-naive"):
        Sessions.from_index(idx.tz_localize("UTC"), tz=TZ, periods_per_year=PPY)
    with pytest.raises(ValueError, match="sorted and unique"):
        Sessions.from_index(idx[::-1], tz=TZ, periods_per_year=PPY)
    with pytest.raises(ValueError, match="no default"):
        Sessions.from_index(idx, tz="", periods_per_year=PPY)
    with pytest.raises(ValueError, match="periods_per_year"):
        Sessions.from_index(idx, tz=TZ, periods_per_year=0)
    observed = Sessions.from_panel_index(idx, tz="UTC", periods_per_year=365)
    assert observed.name == "observed" and observed.is_session(idx[3])
    assert observed.shift(idx[3], 2) == idx[5]
    with pytest.raises(IndexError):
        observed.shift(idx[-1], 5)


# ================================================================== I10 participation cap
def test_participation_cap_carries_the_remainder():
    p = one_name()
    cap, capital = 0.05, 1_000_000.0
    res = run(p, constant, execution=Execution(participation_cap=cap, allow_partial=True),
              sizing=equal_weight(), capital=capital, strict=True)
    trades = res.fills[res.fills["kind"] == "trade"]
    volume = p.volume["AAA"]
    assert len(trades) > 5                                   # the cap really binds
    for row in trades.itertuples(index=False):
        assert row.shares <= cap * float(volume[row.date]) + 1e-9
    # nothing is silently dropped: the first order's fill plus its remainder IS the order
    requested = capital / float(p.close["AAA"].iloc[0])
    first_fill, first_left = trades.iloc[0], res.unfilled.iloc[0]
    assert first_fill["date"] == first_left["date"] == p.sessions.index[1]
    assert first_fill["shares"] + first_left["shares"] == pytest.approx(requested, rel=1e-9)
    assert first_left["carried_to"] == p.sessions.index[2]
    assert res.findings == []

    assert res.unfilled["carried_to"].notna().all()      # every remainder went somewhere
    lag = [p.sessions.bars_between(r["decided_on"], r["date"]) for _, r in res.fills.iterrows()]
    assert max(lag) > 1                                  # a carried order keeps its own date

    dropped = run(p, constant, execution=Execution(participation_cap=cap, allow_partial=False),
                  sizing=equal_weight(), capital=capital, strict=True)
    assert dropped.unfilled["carried_to"].isna().all()   # and here, none of them did
    assert all(p.sessions.bars_between(r["decided_on"], r["date"]) == 1
               for _, r in dropped.fills.iterrows())

    uncapped = run(p, constant, execution=Execution(participation_cap=None),
                   sizing=equal_weight(), capital=capital, strict=True)
    assert len(uncapped.fills) == 1 and uncapped.unfilled.empty


# ================================================================== I11 cash
def test_cash_is_a_position_and_earns_the_declared_rate(full):
    total = full.held_weights().sum(axis=1) + full.cash / full.equity
    assert float((total - 1.0).abs().max()) < 1e-12
    # a flat book is all cash, and the cash leg earns the DECLARED ANNUAL rate
    p = one_name()
    flat = run(p, lambda q: pd.DataFrame(np.nan, index=q.close.index, columns=q.close.columns),
               costs=Costs(cash_rate=0.05), capital=1_000.0, strict=True)
    assert len(flat.fills) == 0
    assert float((flat.returns.iloc[1:] - 0.05 / PPY).abs().max()) < 1e-15
    report = check(full.to_bundle(), guards=["rf_convention"])
    assert report.ran == ["rf_convention"] and report.passed
    assert report[0].evidence["rf"] == 0.05 and report[0].evidence["periods"] == PPY
    with pytest.raises(ValueError, match="ANNUAL decimal"):
        Costs(cash_rate=5.0)


# ================================================================== I12 determinism
def test_engine_is_deterministic():
    p = build_panel("back")
    kw = dict(sizing=rank_long_short(), universe=Universe(rebalance="ME", members=p.listings))
    a, b = run(p, momentum, **kw), run(p, momentum, **kw)
    pd.testing.assert_frame_equal(a.weights, b.weights)
    pd.testing.assert_frame_equal(a.positions, b.positions)
    pd.testing.assert_frame_equal(a.fills, b.fills)
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert a.panel.fingerprint() == b.panel.fingerprint()
    assert a.universe == b.universe


def test_engine_has_no_shared_state_across_folds():
    """fold_leak_test with `run` as its run_fold: serial, shuffled and thread-pool agree."""
    p = build_panel("back")
    idx = p.sessions.index
    folds = [(0, 70), (55, 130), (110, 200)]

    def run_fold(fold, config):
        lo, hi = fold
        sub = p.restrict(start=idx[lo], end=idx[hi - 1])
        return float(run(sub, momentum, sizing=rank_long_short(), strict=False).returns.sum())

    run_fold.__name__ = "engine_fold"
    report = get("fold_leak_test").run(run_fold=run_fold, folds=folds)
    assert report.passed, report.summary()
    assert report.evidence["shared_state"] == []


# ================================================================== the bundle
def test_check_runs_six_guards_with_no_adapter(full):
    report = check(full.to_bundle())
    assert set(report.ran) == {"adjustment_check", "assert_causal", "cost_curve",
                               "pit_universe", "rf_convention", "survivorship_audit"}
    assert set(report.ran).isdisjoint(report.skipped)
    assert "ran 6 guard(s)" in report.summary() and report.summary().isascii()
    # the five that test an ENGINE property pass; cost_curve tests the STRATEGY, and this
    # one is a 40-bar momentum rule on seeded noise, so it dies below the 3 bps it pays -
    # which is the outcome the design wants a run to end with by default.
    by_name = {r.guard: r for r in report}
    assert all(by_name[g].passed for g in ("adjustment_check", "assert_causal", "pit_universe",
                                           "rf_convention", "survivorship_audit"))
    assert not by_name["cost_curve"].passed
    assert by_name["cost_curve"].evidence["breakeven_bps"] <= 3.0


def test_coverage_names_the_slot_each_missing_guard_needs(full):
    cov = full.to_bundle().coverage()
    assert {"adjustment_check", "assert_causal", "cost_curve", "pit_universe",
            "rf_convention", "survivorship_audit"} <= set(cov.ready)
    unlocks = cov.unlocks()
    assert unlocks["regime_labels"] == ["regime_coverage"]
    assert unlocks["model_returns"] == ["spa_test"]      # because benchmark= was supplied
    # regime_lookahead needs p_calm as well; a run does not produce one
    assert cov.missing["regime_lookahead"] == ["regime_labels", "p_calm"]
    assert "one slot away" in cov.summary() and cov.summary().isascii()


def test_bundle_slots_carry_the_run_the_design_says_they_do(full):
    b = full.to_bundle()
    assert b.returns is full.gross_returns          # the slot doc says GROSS
    assert b.prices is full.panel.close
    assert list(b.rebalance_dates) == sorted(full.universe)
    assert list(b.listings.columns) == ["ticker", "listing_date", "delisting_date"]
    assert list(b.members.columns) == ["ticker", "start_date", "end_date"]
    assert b.periods_per_year == PPY and b.expected == "back-adjusted"
    assert b.bars.shape[1] == 5 and callable(b.signal_fn)
    assert b.signal_fn(b.bars).index.equals(b.bars.index)
    extra = full.to_bundle(regime_labels=pd.Series("calm", index=full.returns.index))
    assert "regime_coverage" in extra.coverage().ready


def test_card_is_prefilled_and_still_refuses_to_render(full):
    from fin_skills.core.result_manifest import TrialCount
    card = full.card("engine-demo-v1", falsifier="Fails if net Sharpe <= 0 out of sample.")
    problems = card.problems()
    assert any("trial count" in p for p in problems)
    with pytest.raises(ValueError, match="REFUSING"):
        card.render()
    text = card.render(strict=False)
    assert full.panel.fingerprint()[:16] in text and "back" in text
    assert len(card.cost_curve) == 5 and "annualization" in card.metrics
    assert card.universe.includes_delisted is True
    assert "capm_alpha" in card.benchmark
    filled = full.card("engine-demo-v1", falsifier="stated",
                       trials=TrialCount(37, "research/trials.jsonl"),
                       regimes_covered=["calm", "turbulent"])
    assert filled.problems() == []
    assert get("result_manifest").run(card=filled).passed


# ================================================================== the parts
def test_panel_from_bars_accepts_long_and_wide_and_hashes_itself():
    close, volume, _, _ = raw_frames(n=30, n_names=3, split=False)
    sess = sessions(30)
    wide = pd.concat({f: (close if f != "volume" else volume)
                      for f in Panel.FIELDS}, axis=1)
    p = Panel.from_bars(wide, sess, adjustment="raw")
    assert list(p.close.columns) == sorted(close.columns) and len(p.close) == 30
    assert len(p.fingerprint()) == 64
    assert p.fingerprint() == Panel.from_bars(wide, sess, adjustment="raw").fingerprint()
    assert p.fingerprint() != Panel.from_bars(wide, sess, adjustment="back").fingerprint()
    assert p.ohlcv("N00").shape == (30, 5)
    assert p.restrict(tickers=["N00"]).close.shape == (30, 1)
    assert p.swap(p.ohlcv("N01"), "N01").close.columns.tolist() == ["N01"]
    assert p.liquidity(5).notna().all().all() and "raw" in repr(p)
    with pytest.raises(ValueError, match="adjustment must be one of"):
        Panel.from_bars(wide, sess, adjustment="adjusted")
    with pytest.raises(ValueError, match="index must BE the declared session index"):
        Panel(open=p.open, high=p.high, low=p.low, close=p.close, volume=p.volume,
              sessions=sessions(29), adjustment="raw")


def test_slippage_models_are_non_negative_and_ordered():
    p = one_name(n=40)
    small = 100.0
    ctx_kw = dict(date=p.sessions.index[-1], tickers=pd.Index(["AAA"]),
                  side=pd.Series(1.0, index=["AAA"]), price=pd.Series(50.0, index=["AAA"]),
                  volume=pd.Series(10_000.0, index=["AAA"]), history=p)
    from fin_skills.engine.costs import SlippageContext
    for model in (fixed_bps(2.0), spread_share(0.5), volume_share(), square_root_impact()):
        out = model(SlippageContext(shares=pd.Series(small, index=["AAA"]), **ctx_kw))
        assert float(out.iloc[0]) >= 0.0
        bigger = model(SlippageContext(shares=pd.Series(small * 10, index=["AAA"]), **ctx_kw))
        assert float(bigger.iloc[0]) >= float(out.iloc[0]) - 1e-12
    # zipline's quadratic: 10% participation -> 1e4 * 0.1 * 0.01 = 10 bps
    q = volume_share(0.1)(SlippageContext(shares=pd.Series(1_000.0, index=["AAA"]), **ctx_kw))
    assert float(q.iloc[0]) == pytest.approx(10.0)


def test_the_five_sizers_produce_the_book_they_claim():
    p = build_panel("back")
    sig = momentum(p)
    idx, cols = p.sessions.index, list(p.close.columns)
    from fin_skills.engine.sizing import SizingContext
    d = idx[-1]
    ctx = SizingContext(d, sig.loc[d], cols, p, pd.Series(0.0, index=cols), 1e6, p.sessions)
    ew = equal_weight()(ctx)
    assert ew.sum() == pytest.approx(1.0) and (ew > 0).all()
    ls = rank_long_short(quantile=0.34)(ctx)
    assert ls.sum() == pytest.approx(0.0) and ls.abs().sum() == pytest.approx(1.0)
    vt = vol_target(annual_vol=0.10, lookback=60)(ctx)
    r = p.close[list(vt.index)].tail(61).pct_change(fill_method=None).fillna(0.0)
    assert float((r @ vt).std(ddof=1) * np.sqrt(PPY)) == pytest.approx(0.10, rel=1e-6)
    fk = fractional_kelly(fraction=0.25, lookback=120, cap=0.5)(ctx)
    assert fk.abs().max() <= 0.5 + 1e-12 and fk.abs().sum() <= 1.0 + 1e-12
    frame = pd.DataFrame(0.2, index=idx[::10], columns=cols)
    assert from_weights(frame)(ctx).equals(pd.Series(0.2, index=cols))
    assert "equal_weight" in equal_weight().__name__


def test_invariants_are_twelve_named_claims_each_with_its_test():
    assert len(invariants.INVARIANTS) == 12
    assert list(invariants.INVARIANTS) == [f"I{i}" for i in range(1, 13)]
    # SEVEN, not the design's six: I12's fold_leak_test is a shipped guard too. Six is the
    # number check(to_bundle()) runs unaided, because fold_leak_test needs run_fold + folds,
    # which are not artefacts of a run.
    proven_by_a_guard = [k for k, v in invariants.INVARIANTS.items() if v[2] != "-"]
    assert proven_by_a_guard == ["I1", "I4", "I5", "I6", "I8", "I11", "I12"]
    assert {invariants.INVARIANTS[k][2] for k in proven_by_a_guard} == {
        "assert_causal", "adjustment_check", "survivorship_audit", "pit_universe",
        "cost_curve", "rf_convention", "fold_leak_test"}
    text = invariants.describe()
    assert text.isascii() and all(v[1] in text for v in invariants.INVARIANTS.values())
    # every named test exists in this file
    source = (REPO_ROOT / "tests" / "test_engine.py").read_text(encoding="utf-8")
    for _, test, _ in invariants.INVARIANTS.values():
        assert f"def {test}(" in source, test
