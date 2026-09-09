"""Tests for fin_skills.api - the unified Guard interface and the conventions module.

Every registered guard gets a synthetic clean input (must PASS) and, where the defect can
be planted, a defect input (must FAIL). Every conventions function is compared against the
skill script it delegates to, on one worked example.

Run:  python -m pytest tests/test_api.py -q
"""
from __future__ import annotations

import contextlib
import functools
import io
import pathlib
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from fin_skills.api import Finding, Guard, GuardResult, conventions as conv

API_DIR = pathlib.Path(api.__file__).resolve().parent

# The guards the design names explicitly; each must exist in the registry.
NAMED = ["assert_causal", "safe_asof", "adjustment_check", "survivorship_audit", "pit_universe",
         "pit_fundamentals", "fold_leak_test", "warmup_probe", "contamination_probe",
         "paper_account_guard", "cost_curve", "trial_ledger", "result_manifest", "spa_test",
         "purge_effect", "regime_coverage"]


def _capture(fn: Callable[[], Any]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


# ======================================================================================
# synthetic inputs, one builder per guard: {"clean": kwargs, "defect": kwargs}
# ======================================================================================
def _bars(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    return pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1e6})


def build_assert_causal() -> dict:
    df = _bars()
    return {"clean": dict(fn=lambda d: d.close.rolling(20).mean(), df=df, k=250, name="sma_20"),
            "defect": dict(fn=lambda d: d.close.shift(-1), df=df, k=250, name="shift_minus_1")}


def build_safe_asof() -> dict:
    ts = pd.Timestamp
    signals = pd.DataFrame({"time": [ts("2024-01-02 09:30:01")], "symbol": ["AAA"], "signal": [1]})
    quotes = pd.DataFrame({"time": [ts("2024-01-02 09:29:59"), ts("2024-01-02 09:30:01"),
                                    ts("2024-01-02 09:30:05")],
                           "symbol": ["AAA"] * 3, "px": [100.0, 101.0, 102.0]})
    base = dict(left=signals, right=quotes, on="time", by="symbol")
    return {"clean": dict(base, tolerance="5min"),
            # pandas' default: the quote stamped at the signal's own timestamp is matched
            "defect": dict(base, tolerance="5min", allow_exact_matches=True),
            "defect_no_tolerance": dict(base),
            "defect_forward": dict(base, tolerance="5min", direction="forward")}


def build_adjustment_check() -> dict:
    from fin_skills.core.adjustment_check import _synthetic
    raw, back, fwd, splits = _synthetic()
    return {"clean": dict(close=back, actions=splits),
            "defect": dict(close=raw, actions=splits),                    # unadjusted split
            "defect_expected": dict(close=back, actions=splits, expected="forward-adjusted")}


def build_reconcile_sources() -> dict:
    from fin_skills.core.adjustment_check import _synthetic
    raw, back, fwd, splits = _synthetic()
    corrupt = back.copy()
    corrupt.iloc[400] *= 1.35
    return {"clean": dict(close=back, actions=splits, other=fwd),
            "defect": dict(close=back, actions=splits, other=corrupt)}


def build_survivorship_audit() -> dict:
    from fin_skills.core.survivorship_audit import _synthetic_panel
    full, listings = _synthetic_panel()
    alive = [c for c in full.columns if pd.notna(full[c].iloc[-1])]
    return {"clean": dict(prices=full, listings=listings),
            "defect": dict(prices=full[alive], listings=listings),
            "defect_no_listings": dict(prices=full[alive])}


def build_pit_universe() -> dict:
    from fin_skills.core.pit_universe import _synthetic_index, rebalance_universe
    prices, members = _synthetic_index()
    rebals = pd.bdate_range(prices.index[0], prices.index[-1], freq="BQE")
    today = sorted(members.loc[members["end_date"].isna(), "ticker"])
    snapshot = {pd.Timestamp(d): [t for t in today if pd.notna(prices[t].asof(pd.Timestamp(d)))]
                for d in rebals}
    return {"clean": dict(universe=rebalance_universe(rebals, members)),
            "clean_members": dict(members=members, rebalance_dates=rebals),
            "defect": dict(universe=snapshot)}


def build_pit_fundamentals() -> dict:
    from fin_skills.core.pit_fundamentals import DEMO_FACTS, naive_latest, pit_facts
    as_of = "2023-01-15"
    return {"clean": dict(facts=DEMO_FACTS, as_of=as_of, used=pit_facts(DEMO_FACTS, as_of)),
            "defect": dict(facts=DEMO_FACTS, as_of=as_of, used=naive_latest(DEMO_FACTS))}


class _Scaler:
    def fit(self, x: np.ndarray) -> "_Scaler":
        self.mean_ = float(np.mean(x))
        self.scale_ = float(np.std(x)) or 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_


def build_fold_leak_test() -> dict:
    data = np.random.default_rng(42).normal(0, 1, 1200)
    data.flags.writeable = False
    folds = [(i * 200, (i + 1) * 200) for i in range(6)]
    shared_scaler = _Scaler().fit(data)
    shared_rng = np.random.default_rng(0)

    def run_fold_leaky(fold: tuple[int, int], config: dict) -> float:
        lo, hi = fold
        z = shared_scaler.transform(data[lo:hi])
        boot = shared_rng.integers(0, len(z), size=(config["n_boot"], len(z)))
        return float(np.mean(z[boot]))

    def run_fold_clean(fold: tuple[int, int], config: dict) -> float:
        lo, hi = fold
        z = _Scaler().fit(data[lo:hi]).transform(data[lo:hi])
        rng = np.random.default_rng(lo)
        boot = rng.integers(0, len(z), size=(config["n_boot"], len(z)))
        return float(np.mean(z[boot]))

    return {"clean": dict(run_fold=run_fold_clean, folds=folds, config={"n_boot": 8}),
            "defect": dict(run_fold=run_fold_leaky, folds=folds, config={"n_boot": 8})}


def build_warmup_probe() -> dict:
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 1500)))

    def ema20(a: np.ndarray) -> np.ndarray:
        return pd.Series(a).ewm(span=20, adjust=False).mean().to_numpy()

    base = dict(indicator={"EMA(20)": ema20}, closes=closes, tol=1e-9, max_probe=500)
    return {"clean": dict(base, history=600),
            "defect": dict(base, history=20)}          # "a 20-period EMA needs 20 bars"


def build_contamination_probe() -> dict:
    return {"clean": dict(cutoff="2024-06-01", test_start="2025-01-01", test_end="2026-01-01"),
            "defect": dict(cutoff="2024-06-01", test_start="2020-01-01", test_end="2024-01-01")}


def build_paper_account_guard() -> dict:
    return {"clean": dict(broker="ib", account_id="DU1234567"),
            "defect": dict(broker="ib", account_id="U1234567", extra={"port": 7497}),
            "defect_alpaca": dict(broker="alpaca", base_url="https://api.alpaca.markets/v2",
                                  extra={"paper": True})}


def build_cost_curve() -> dict:
    rng = np.random.default_rng(11)
    gross = pd.Series(rng.normal(0.001, 0.006, 1000),
                      index=pd.bdate_range("2020-01-01", periods=1000))
    return {"clean": dict(returns=gross, turnover=0.05, cost_bps=10.0),
            "defect": dict(returns=gross, turnover=1.5, cost_bps=10.0)}


def build_cost_plausibility() -> dict:
    # Same 25m book and 6.5-name flow both ways; only the stated cost and the ADV move.
    base = dict(turnover=0.085, book=2.5e7, adv=1.7e7, n_names=6.5, daily_vol=0.022)
    return {"clean": dict(base, cost_bps=20.0),
            "defect": dict(base, cost_bps=2.0),                  # below the impact floor
            "defect_participation": dict(base, adv=2.0e5, cost_bps=500.0)}  # 100%+ of ADV


def build_trial_ledger() -> dict:
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.45, 50).tolist()
    return {"clean": dict(sharpes=[0.5, 0.6, 2.5], best_sharpe=2.5, n_obs=1260),
            "defect": dict(sharpes=noise, best_sharpe=max(noise), n_obs=1260)}


def build_result_manifest() -> dict:
    from fin_skills.core.result_manifest import (CostModel, DataSource, ResultCard, Split,
                                                 TrialCount, Universe)
    complete = ResultCard(
        strategy_id="core-etf-trend-v2",
        universe=Universe("EODHD", "2026-01-02", True, 512, "ADV>1e6 at each rebalance"),
        data=[DataSource("EODHD EOD", "2026-01-02T09:00Z", "backward-adjusted")],
        split=Split("CombinatorialPurgedCV", "20D", "5D", "2010-2019", "2020-2026"),
        costs=CostModel(2.0, 3.0, "sqrt(participation)", 50.0, 0.0, "^IRX"),
        trials=TrialCount(137, "research/trials.jsonl"),
        metrics={"sharpe_net": 0.62, "annualization": 252, "rf_convention": "annual, geometric"},
        cost_curve={0: 1.10, 10: 0.62, 20: 0.15, 50: -0.44},
        benchmark={"name": "SPY", "capm_alpha": -0.002, "alpha_t": -0.31},
        falsifier="Fails if net-of-cost excess return over SPY is <= 0.",
        regimes_covered=["high-vol [ex-ante rule]: 400 obs / 6 episodes"])
    incomplete = ResultCard(
        strategy_id="demo-trend-v1",
        universe=Universe("yfinance", "2026-09-03", False, 5, "hand-picked"),
        data=[DataSource("yfinance", "2026-09-03T12:00Z", "auto_adjust=True")],
        split=Split("walk-forward", "0D", "0D", "2010-2019", "2020-2026"),
        costs=CostModel(1.0, 2.0, "none"), trials=TrialCount(1, ""),
        metrics={"sharpe_net": 1.9}, cost_curve={}, benchmark={}, falsifier="")
    return {"clean": dict(card=complete), "defect": dict(card=incomplete)}


def build_spa_test() -> dict:
    from fin_skills.core.spa_test import _panel
    bench, models, names = _panel(np.random.default_rng(3), edge=0.0012, edge_on=4)
    bench0, models0, _ = _panel(np.random.default_rng(3), edge=0.0)
    return {"clean": dict(benchmark_returns=bench, model_returns=pd.DataFrame(models, columns=names),
                          reps=500, seed=12),
            "defect": dict(benchmark_returns=bench0,
                           model_returns=pd.DataFrame(models0, columns=names), reps=500, seed=12)}


def build_purge_effect() -> dict:
    from fin_skills.libraries.purge_effect import H, kfold, make_dataset, purged_kfold
    x, y = make_dataset(0)
    n = len(y)
    return {"clean": dict(x=x, y=y, horizon=H, splits=list(purged_kfold(n, 5, H, 0))),
            "defect": dict(x=x, y=y, horizon=H, splits=list(kfold(n, 5))),
            "clean_no_splits": dict(x=x, y=y, horizon=H)}


def build_regime_coverage() -> dict:
    rng = np.random.default_rng(1)
    n = 480
    dates = pd.bdate_range("2020-01-01", periods=n)
    asset = rng.normal(0.0003, 0.01, n)
    pos = np.where(np.arange(n) % 40 < 30, 1.0, 0.0)
    strat = pos * asset
    two = np.array((["low-vol"] * 120 + ["high-vol"] * 120) * 2)
    one = np.array(["bull"] * n)
    base = dict(dates=dates, asset=asset, strategy=strat, position=pos)
    return {"clean": dict(base, labels=two, how="ex-ante rule"),
            "defect": dict(base, labels=one, how="ex-post label")}


def build_continuous_contract() -> dict:
    from fin_skills.futures_fx.continuous_contract import _synthetic_contango, stitch
    contracts, rolls, _spot = _synthetic_contango()
    return {"clean": dict(stitched=stitch(contracts, rolls, "ratio"), contracts=contracts,
                          roll_dates=rolls),
            "defect": dict(stitched=stitch(contracts, rolls, "difference")),
            "defect_unadjusted": dict(stitched=stitch(contracts, rolls, "unadjusted"))}


VOLLIB_CALL = {"price": 10.45058357, "delta": 0.63683065, "gamma": 0.01876202,
               "vega": 0.37524035, "theta": -0.01757268, "rho": 0.53232482}


def build_greeks_convention() -> dict:
    return {"clean": dict(greeks=VOLLIB_CALL, flag="c", moneyness=1.0, convention="vollib"),
            "defect": dict(greeks={**VOLLIB_CALL, "theta": 0.01757268}, flag="c")}


def build_brinson_attribution() -> dict:
    from fin_skills.core.brinson_attribution import _demo_panels
    panels = _demo_panels()
    broken = panels[0].copy()
    broken.loc["Staples", "wp"] = 0.14          # a 6% cash sleeve missing from the table
    return {"clean": dict(panel=panels[0]), "clean_multi": dict(panels=panels),
            "defect": dict(panel=broken)}


def build_fx_conventions() -> dict:
    return {"clean": dict(pair=["EURUSD", "USD/JPY", "AUDUSD"]), "defect": dict(pair="JPYUSD")}


def build_ashare_rules() -> dict:
    from fin_skills.china.ashare_rules import daily_limit_pct, limit_price
    prev = 200.0
    up = limit_price(prev, daily_limit_pct("300750", None, "2021-06-01"), "up")
    locked = {"prev_close": prev, "open": up, "high": up, "low": up, "close": up, "volume": 1e6}
    normal = {"prev_close": prev, "open": 202.0, "high": up, "low": 199.0, "close": 239.9,
              "volume": 8.4e6}
    base = dict(code="300750", date="2021-06-01")
    return {"clean": dict(base, bar=normal),
            "defect": dict(base, bar=locked),
            "defect_t_plus_1": dict(base, bar=normal, side="sell",
                                    lots=[{"date": "2021-06-01", "qty": 1000}])}


def build_join_asof_sortedness() -> dict:
    from fin_skills.libraries.join_asof_sortedness import tiny_case
    signals, quotes = tiny_case()
    sorted_signals = signals.sort_values(["symbol", "time"]).reset_index(drop=True)
    return {"clean": dict(left=sorted_signals, right=quotes, on="time", by="symbol", val="px"),
            "defect": dict(left=signals, right=quotes, on="time", by="symbol", val="px")}


def build_npv_zero() -> dict:
    from fin_skills.libraries.npv_zero import EXP, HIST_EXP, HIST_VAL, STALE, VAL
    return {"clean": dict(eval_date=VAL, curve_ref=VAL, expiry=EXP),
            "defect": dict(eval_date=VAL, curve_ref=STALE, expiry=EXP),
            "defect_expired": dict(eval_date="2026-09-08", curve_ref=HIST_VAL, expiry=HIST_EXP)}


def build_weight_traps() -> dict:
    from fin_skills.libraries.weight_traps import make_panel
    rets, prices = make_panel()
    return {"clean": dict(asset_returns=rets), "defect": dict(asset_returns=prices)}


def build_rf_convention() -> dict:
    from fin_skills.libraries.rf_convention import _sharpe_annual_rf, _sharpe_per_period_rf
    r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 1008))
    return {"clean": dict(returns=r, rf=0.05, reported_sharpe=_sharpe_annual_rf(r, 0.05)),
            "defect": dict(returns=r, rf=0.05, reported_sharpe=_sharpe_per_period_rf(r, 0.05))}


def build_leveraged_reset() -> dict:
    from fin_skills.core.leveraged_reset import leveraged_wealth
    r = np.random.default_rng(0).normal(0.0, 0.25 / np.sqrt(252), 252)
    exact = float(leveraged_wealth(r, 3.0)[-1] - 1.0)
    return {"clean": dict(index_returns=r, lev=3.0, modelled_return=exact),
            "defect": dict(index_returns=r, lev=3.0, modelled_return=3.0 * (np.prod(1 + r) - 1))}


def build_regime_lookahead() -> dict:
    from fin_skills.core.regime_lookahead import (MAIN_RATIO, N, TEST_START, simulate,
                                                  true_param_probs)
    r, s = simulate(N, 0, MAIN_RATIO)
    probs = true_param_probs(r, MAIN_RATIO)
    return {"clean": dict(regime=s, p_calm=probs["predicted"], lo=TEST_START),
            "defect": dict(regime=s, p_calm=probs["smoothed"], lo=TEST_START)}


_BUILDERS: dict[str, Callable[[], dict]] = {
    k[len("build_"):]: v for k, v in dict(globals()).items() if k.startswith("build_")}


@functools.lru_cache(maxsize=None)
def synthetic(name: str) -> dict:
    return _BUILDERS[name]()


ALL_GUARDS = [cls.name for cls in api.registry()]


# ======================================================================================
# the registry and the interface
# ======================================================================================
def test_registry_is_non_empty_and_contains_the_named_guards() -> None:
    names = ALL_GUARDS
    assert len(names) >= 16
    missing = [n for n in NAMED if n not in names]
    assert not missing, f"named guards absent from the registry: {missing}"
    for cls in api.registry():
        assert issubclass(cls, Guard)
        assert cls.name and cls.skill and cls.summary and cls.wraps
        assert cls.required or cls.optional


def test_one_module_per_guard_and_every_module_registers() -> None:
    modules = sorted(p.stem for p in (API_DIR / "guards").glob("*.py")
                     if not p.stem.startswith("_"))
    assert modules == ALL_GUARDS


def test_every_guard_has_a_fixture() -> None:
    assert sorted(_BUILDERS) == ALL_GUARDS


def test_wrapped_functions_exist() -> None:
    import importlib
    for cls in api.registry():
        for dotted in cls.wraps:
            parts = dotted.split(".")
            obj: Any = importlib.import_module(".".join(parts[:3]))   # fin_skills.<ns>.<module>
            for part in parts[3:]:                                      # then Class / method
                obj = getattr(obj, part)
            assert callable(obj), dotted
    # only the owner's own module is wrapped: module namespace, then skill listed in it
    for cls in api.registry():
        assert all(w.startswith("fin_skills.") for w in cls.wraps)


@pytest.mark.parametrize("name", ALL_GUARDS)
def test_every_guard_runs_clean_and_passes(name: str) -> None:
    guard = api.get(name)
    assert isinstance(guard, Guard) and guard.name == name
    for label, kwargs in synthetic(name).items():
        if not label.startswith("clean"):
            continue
        result = guard.run(**kwargs)
        assert isinstance(result, GuardResult)
        assert result.guard == name and result.skill == guard.skill
        assert result.passed, f"{name}[{label}] failed on clean input:\n{result.summary()}"
        assert result.errors == []
        assert isinstance(result.evidence, dict) and result.elapsed_s >= 0.0
        assert all(isinstance(f, Finding) for f in result.findings)
        text = result.summary()
        assert text.startswith("PASS") and text.isascii()


@pytest.mark.parametrize("name", ALL_GUARDS)
def test_every_guard_fails_on_the_planted_defect(name: str) -> None:
    guard = api.get(name)
    for label, kwargs in synthetic(name).items():
        if not label.startswith("defect"):
            continue
        result = guard.run(**kwargs)
        assert not result.passed, f"{name}[{label}] passed on a planted defect:\n{result.summary()}"
        assert result.errors, f"{name}[{label}] has no error-level finding"
        text = result.summary()
        assert text.startswith("FAIL") and text.isascii()


def test_the_three_headline_defects_are_named_in_the_findings() -> None:
    r = api.get("assert_causal").run(**synthetic("assert_causal")["defect"])
    assert "LOOK-AHEAD" in r.errors[0].message
    r = api.get("safe_asof").run(**synthetic("safe_asof")["defect"])
    assert "LOOK-AHEAD" in r.errors[0].message and r.evidence["n_matched"] == 1
    r = api.get("adjustment_check").run(**synthetic("adjustment_check")["defect"])
    assert r.evidence["convention"] == "raw" and "RAW" in r.errors[0].message
    clean = api.get("adjustment_check").run(**synthetic("adjustment_check")["clean"])
    assert clean.evidence["convention"] == "back-adjusted"


def test_bad_inputs_raise_typeerror_not_findings() -> None:
    g = api.get("assert_causal")
    df = _bars(50)
    with pytest.raises(TypeError):
        g.run(fn=lambda d: d.close, df=df, k=10, bogus=1)          # unknown keyword
    with pytest.raises(TypeError):
        g.run(df=df)                                                # missing required
    with pytest.raises(TypeError):
        g.run(fn=lambda d: d.close, df=df, k=500)                   # k beyond the frame
    with pytest.raises(TypeError):
        g.run(fn=lambda d: d.close, df="not a frame", k=5)
    # a ValueError raised by the wrapped script surfaces as TypeError, message attached
    c = api.get("cost_curve")
    with pytest.raises(TypeError, match="turnover has"):
        c.run(returns=np.zeros(10), turnover=np.ones(7))
    with pytest.raises(TypeError):
        api.get("contamination_probe").run(cutoff="2024-01-01", test_start="2025-01-01",
                                           test_end="2024-06-01")   # end before start
    with pytest.raises(KeyError):
        api.get("no_such_guard")
    with pytest.raises(ValueError):
        Finding("fatal", "bad severity")


def test_run_all_runs_what_it_can_and_reports_the_rest() -> None:
    inputs: dict[str, Any] = {}
    inputs.update(synthetic("safe_asof")["clean"])
    inputs.update(synthetic("cost_curve")["clean"])
    inputs.update(synthetic("fx_conventions")["clean"])
    inputs.update(synthetic("contamination_probe")["clean"])
    inputs.update(synthetic("paper_account_guard")["clean"])
    report = api.run_all(**inputs)
    expected = sorted(cls.name for cls in api.registry() if cls().accepts(inputs))
    assert sorted(report.ran) == expected
    assert {"safe_asof", "join_asof_sortedness", "cost_curve", "fx_conventions",
            "contamination_probe", "paper_account_guard"} <= set(report.ran)
    assert set(report.skipped) == set(ALL_GUARDS) - set(report.ran)
    assert all(isinstance(r, GuardResult) for r in report)
    assert report.passed and report.failed == []
    assert report.summary().isascii() and "SKIP" in report.summary()
    assert "assert_causal" in report.skipped and report.skipped["assert_causal"] == ["fn", "df"]


def test_input_names_share_one_meaning() -> None:
    names = api.input_names()
    assert set(names["left"]) == {"safe_asof", "join_asof_sortedness"}
    assert "tol" in names and "name" in names
    assert names["returns"] == ["cost_curve", "rf_convention"]


def test_api_sources_are_ascii() -> None:
    for p in API_DIR.rglob("*.py"):
        assert p.read_bytes().isascii(), p


# ======================================================================================
# conventions: same number as the script it delegates to
# ======================================================================================
def test_annualization_delegates_to_perp_mechanics() -> None:
    from fin_skills.crypto import perp_mechanics as pm
    r = pd.Series(np.random.default_rng(0).normal(0.0012, 0.025, 1460))
    assert conv.annualize_sharpe(r, "crypto") == pm.sharpe(r, pm.CRYPTO_DAYS)
    assert conv.annualize_sharpe(r.to_numpy(), 252) == pm.sharpe(r, 252)
    assert conv.annualization_factor("equity") == pm.EQUITY_DAYS == 252
    assert conv.annualization_factor("crypto") == pm.CRYPTO_DAYS == 365
    assert conv.annualization_factor("hourly") == pm.CRYPTO_DAYS * 24
    assert conv.annualization_factor("funding_8h") == pm.CRYPTO_DAYS * 3
    with pytest.raises(ValueError):
        conv.annualization_factor("weekly")


def test_fx_conventions_delegate() -> None:
    from fin_skills.futures_fx import fx_conventions as fx
    assert conv.pip_size("USDJPY") == fx.pip_size("USDJPY") == 0.01
    assert conv.pip_size("EUR/USD") == fx.pip_size("EUR/USD") == 0.0001
    got = conv.pip_value("USDJPY", 100_000, 150.25)
    assert got == fx.pip_value("USDJPY", 100_000, 150.25)
    assert round(got.value_usd, 2) == 6.66                     # the docstring's number
    carry = conv.carry_return(0.98, 0.0475, 0.0025, 365, "AUDUSD")
    assert carry == fx.carry_return(0.98, 0.0475, 0.0025, 365, "AUDUSD")
    assert carry.ret > 0 > carry.points                        # opposite signs, always


def test_crr_and_greeks_delegate() -> None:
    from fin_skills.core import option_lifecycle as opt
    from fin_skills.libraries import greeks_scaling as gs
    for american in (True, False):
        assert conv.crr(100, 90, 0.05, 0.02, 0.2, 30 / 365, american=american) == \
            opt.crr(100, 90, 0.05, 0.02, 0.2, 30 / 365, american=american)
    got = conv.greeks_screen_units(100, 100, 1.0, 0.05, 0.0, 0.20, "c")
    assert got == gs.scaled_greeks(gs.raw_greeks(100, 100, 1.0, 0.05, 0.0, 0.20, "c"))
    assert abs(got["vega"] - VOLLIB_CALL["vega"]) < 1e-6      # vollib's published number
    assert abs(got["theta"] - VOLLIB_CALL["theta"]) < 1e-6


def test_crypto_conventions_match_the_script_output() -> None:
    from fin_skills.crypto import perp_mechanics as pm
    funding = _capture(pm.section_funding)
    m = re.search(r"payment per interval = .* = \$([\d,.]+)", funding)
    assert m and float(m.group(1).replace(",", "")) == \
        conv.funding_payment(100_000, pm.OKX_FUNDING_INTEREST) == 10.0
    assert conv.funding_payment(100_000, pm.OKX_FUNDING_INTEREST, periods=3, side="short") == -30.0

    basis = _capture(pm.section_basis)
    row = next(line for line in basis.splitlines() if line.strip().startswith("BTC-25DEC26"))
    simple_pct, compound_pct = re.findall(r"(-?\d+\.\d+)%", row)[-2:]
    mark, exp = pm.DERIBIT_FUTURES["BTC-25DEC26"]
    days = (pd.Timestamp(exp + "T08:00:00Z") - pm.AS_OF).total_seconds() / 86400
    assert f"{conv.basis_annualized(mark, pm.DERIBIT_INDEX, days):.2%}" == simple_pct + "%"
    assert f"{conv.basis_annualized(mark, pm.DERIBIT_INDEX, days, compound=True):.2%}" == \
        compound_pct + "%"

    inverse = _capture(pm.section_inverse)
    row = next(line for line in inverse.splitlines() if line.strip().startswith("+20%"))
    printed_btc = row.split()[2]
    assert f"{conv.inverse_contract_pnl(100_000, pm.DERIBIT_INDEX, pm.DERIBIT_INDEX * 1.2):+.4f}" \
        == printed_btc
    assert conv.inverse_contract_pnl(100_000, 100.0, 80.0, side="short") > 0

    liq = _capture(pm.section_liquidation)
    row = next(line for line in liq.splitlines() if line.strip().startswith("10x"))
    printed_long = row.split()[1]
    mmr = pm.OKX_TIER1["mmr"]
    assert f"{conv.liquidation_price(1.0, 10, mmr, 'long') - 1:.2%}" == printed_long
    assert conv.liquidation_price(50_000, 5, mmr, "short") == 50_000 * pm.liq_ratio(5, mmr, "short")


def test_stitch_continuous_delegates() -> None:
    from fin_skills.futures_fx import continuous_contract as cc
    contracts, rolls, _ = cc._synthetic_contango()
    got = conv.stitch_continuous(contracts, rolls)
    want = cc.stitch(contracts, rolls, "ratio")
    pd.testing.assert_series_equal(got, want)
    assert got.attrs == want.attrs and got.attrs["returns_valid"] is True
    assert conv.stitch_continuous(contracts, rolls, "difference").attrs["returns_valid"] is False


def test_execution_benchmarks_delegate() -> None:
    from fin_skills.core import benchmark_choice as bc
    px, vol = bc.make_day()
    decision = float(px[0])
    row = bc.run("VWAP", bc.schedule_vwap, px, vol, decision, 100_000)
    avg = row["avg_fill"]
    assert conv.implementation_shortfall(avg, decision) == row["decision (arrival)"]
    costs = conv.benchmark_costs(avg, px, vol, decision)
    for name in ("decision (arrival)", "interval VWAP", "interval TWAP", "close", "open"):
        assert costs[name] == row[name]
    assert abs(costs["interval VWAP"]) < 1e-9        # a VWAP schedule scores zero vs VWAP
