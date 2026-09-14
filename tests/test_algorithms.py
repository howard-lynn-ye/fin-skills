"""Behavioral contracts for suitability routing, adapters and temporal model selection."""
import json
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import fin_skills
from fin_skills.algorithms import (Algorithm, Registry, Request, auto_run, catalog,
                                   default_registry, recommend, run, walk_forward)
from fin_skills.algorithms import core
from fin_skills.tools import call_tool, list_tools
from _helpers import REPO_ROOT
from conftest import has_module


@pytest.fixture
def returns():
    rng = np.random.default_rng(77)
    return pd.DataFrame(rng.normal(size=(160, 3)) * [0.01, 0.02, 0.04],
                        index=pd.date_range("2020-01-01", periods=160), columns=list("ABC"))


def test_catalog_provenance_and_isolation():
    rows = catalog()
    assert len({a["id"] for a in rows}) == len(rows)
    assert {a["status"] for a in rows} <= {"ready", "catalog_only", "missing_dependency"}
    for a in rows:
        assert a["source"] and a["verified_on"] and a["skill"] in fin_skills.names()
    rows[0]["name"] = "changed"
    assert catalog()[0]["name"] != "changed"
    json.dumps(rows)


def test_fresh_registry_extends_without_changing_default():
    reg = default_registry()
    spec = Algorithm("custom", "custom forecast", "forecast", "local", ("series",),
                     ("forecast",), source="local test", priority=100)
    reg.register(spec, lambda data, p: [42] * p.get("horizon", 1))
    assert reg.auto_run("forecast", {"series": [1, 2, 3]})["result"] == [42]
    assert "custom" not in {a["id"] for a in catalog()}
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(spec)


def test_objective_and_capabilities_are_hard_constraints():
    result = recommend(Request("组合优化", ("asset_returns",), 100,
                               objective="min_variance", required_capabilities=("long_only",),
                               allowed_libraries=("fin-skills",)))
    assert result.selected.algorithm.id == "min_variance"
    assert "equal_weight" in result.rejected
    assert "not measured performance" in result.basis


def test_preferences_change_selection_and_explain_why():
    base = Request("forecast", ("series",), 100, allowed_libraries=("fin-skills",))
    assert recommend(base).selected.algorithm.id == "naive"
    result = recommend(replace(base, preferences=("stationary",)))
    assert result.selected.algorithm.id == "mean"
    assert "matches preference: stationary" in result.selected.reasons


def test_small_sample_has_explicit_fallback():
    result = recommend(Request("portfolio", ("asset_returns",), 1))
    assert result.selected.algorithm.id == "equal_weight"
    assert "observations" in str(result.rejected["min_variance"])


def test_unknown_sample_size_is_not_treated_as_enough():
    result = recommend(Request("forecast", ("series",)))
    assert result.selected is None
    assert "n_observations" in str(result.rejected)


def test_catalog_only_never_executes_even_when_package_exists(monkeypatch):
    monkeypatch.setattr(core, "find_spec", lambda name: object())
    reg = Registry()
    reg.register(Algorithm("unimplemented", "Unimplemented", "regime", "local", ("X",),
                           ("regime",), source="test"))
    result = reg.recommend(Request("regime", ("X",), 200))
    assert result.selected is None
    advisory = reg.recommend(Request("regime", ("X",), 200, executable_only=False))
    assert advisory.selected.status == "catalog_only" and not advisory.selected.executable
    with pytest.raises(NotImplementedError, match="catalog_only"):
        reg.run("unimplemented", {"X": np.ones((200, 2))})


def test_missing_dependency_never_silently_changes_backend(monkeypatch):
    monkeypatch.setattr(core, "find_spec", lambda name: None)
    result = recommend(Request("regression", ("X", "y", "X_predict"), 200))
    assert result.selected is None
    with pytest.raises(ImportError, match="scikit-learn"):
        run("ridge", {})


@pytest.mark.parametrize("kwargs,match", [
    ({"preferences": ("typo",)}, "preferences"),
    ({"available_inputs": ("returnz",)}, "available_inputs"),
    ({"allowed_libraries": ("made_up",)}, "allowed_libraries"),
    ({"objective": "maximize_future_profit"}, "objective"),
    ({"required_capabilities": ("no_drawdowns",)}, "required_capabilities"),
])
def test_unknown_routing_constraints_are_not_ignored(kwargs, match):
    values = dict(task="portfolio", available_inputs=("asset_returns",), n_observations=100)
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        recommend(Request(**values))


@pytest.mark.parametrize("kwargs", [
    {"n_observations": True}, {"n_observations": -1}, {"n_observations": 1.2},
    {"max_complexity": "infinite"}, {"executable_only": "false"}, {"preferences": "trend"},
])
def test_request_types_are_validated(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Request("forecast", ("series",), **kwargs)


def test_portfolio_methods_preserve_assets_and_budget(returns):
    for id, parameters in [("equal_weight", {}), ("inverse_volatility", {}),
                            ("min_variance", {}), ("hrp", {"linkage": "single"})]:
        result = run(id, {"asset_returns": returns}, **parameters)
        assert list(result.index) == list(returns.columns)
        assert np.isfinite(result).all() and (result >= -1e-8).all()
        assert result.sum() == pytest.approx(1)
    vol = returns.std()
    weights = run("inverse_volatility", {"asset_returns": returns})
    np.testing.assert_allclose(weights * vol, np.full(3, (weights * vol).iloc[0]))


def test_min_variance_improves_sample_variance(returns):
    w = run("min_variance", {"asset_returns": returns}).to_numpy()
    cov = returns.cov().to_numpy()
    equal = np.full(3, 1 / 3)
    assert w @ cov @ w < equal @ cov @ equal


def test_hrp_requires_linkage(returns):
    with pytest.raises(ValueError, match="linkage"):
        run("hrp", {"asset_returns": returns})


@pytest.mark.parametrize("value", [np.ones((40, 2)) * 100, np.ones((40, 2)) * -2])
def test_price_or_impossible_return_rejected(value):
    with pytest.raises(ValueError, match="decimal simple returns"):
        auto_run("portfolio", {"asset_returns": value})


@pytest.mark.parametrize("value", [[1, np.nan], [1, np.inf], [True, False], ["1", "2"], []])
def test_invalid_numeric_data_rejected(value):
    with pytest.raises(ValueError):
        auto_run("forecast", {"series": value})


def test_time_order_and_duplicate_index_rejected():
    for index in ([2, 1], [1, 1]):
        with pytest.raises(ValueError, match="index"):
            auto_run("forecast", {"series": pd.Series([2, 3], index=index)})


def test_auto_run_uses_actual_observations_and_no_candidate(returns):
    result = auto_run("portfolio", {"asset_returns": returns}, objective="min_variance")
    assert result["selection"]["request"]["n_observations"] == len(returns)
    assert result["selection"]["selected"] == "min_variance"
    blocked = auto_run("regime", {"X": np.ones((200, 2))})
    assert blocked["executed"] is False and blocked["result"] is None
    with pytest.raises(TypeError, match="derives"):
        auto_run("forecast", {"series": [1]}, n_observations=200)


def test_forecast_baselines():
    np.testing.assert_equal(run("naive", {"series": [1, 2, 3]}, horizon=2), [3, 3])
    np.testing.assert_equal(run("drift", {"series": [1, 2, 3]}, horizon=2), [4, 5])
    np.testing.assert_equal(run("mean", {"series": [1, 2, 3]}, horizon=2), [2, 2])
    np.testing.assert_equal(run("seasonal_naive", {"series": [1, 2, 1, 2],
                                                "seasonal_period": 2}, horizon=3), [1, 2, 1])
    with pytest.raises(TypeError, match="unexpected parameters"):
        run("naive", {"series": [1]}, wrong=1)


def test_volatility_and_tail_loss_units():
    returns = np.tile([-0.01, 0.01], 20)
    vol = run("historical_volatility", {"returns": returns})
    assert vol["volatility"] == pytest.approx(np.std(returns, ddof=1))
    ewma = run("ewma_volatility", {"returns": returns}, periods_per_year=4)
    assert ewma["volatility"] == pytest.approx(0.02)
    for id in ("historical_var_es", "normal_var_es"):
        result = run(id, {"returns": returns})
        assert result["expected_shortfall"] >= result["var"] > 0


def test_signal_is_strictly_lagged_and_future_invariant():
    original = pd.Series(np.arange(1, 51, dtype=float))
    changed = original.copy()
    changed.iloc[30:] *= 0.01
    a = run("ma_crossover", {"prices": original})
    b = run("ma_crossover", {"prices": changed})
    pd.testing.assert_series_equal(a.iloc[:31], b.iloc[:31])
    assert a.iloc[:20].isna().all() and a.iloc[20] == 1
    momentum = run("momentum", {"returns": np.tile([-0.01, 0.02], 30)})
    assert momentum.iloc[:20].isna().all()


def test_execution_volume_changes_selection_and_preserves_quantity():
    twap = auto_run("execution", {"shares": 100, "n_bins": 4})
    np.testing.assert_equal(twap["result"], [25] * 4)
    vwap = auto_run("execution", {"shares": 100, "n_bins": 4, "volume_forecast": [1, 3]})
    assert vwap["selection"]["selected"] == "vwap"
    np.testing.assert_equal(vwap["result"], [25, 75])
    with pytest.raises(ValueError):
        run("vwap", {"shares": 100, "volume_forecast": [0, 0]})


def test_option_exercise_style_drives_automatic_selection():
    option = dict(S=100, K=105, T=1, r=0.03, q=0.01, sigma=0.2, flag="p", exercise="european")
    european = auto_run("pricing", {"option": option})
    american = auto_run("pricing", {"option": dict(option, exercise="american")})
    assert european["selection"]["selected"] == "black_scholes"
    assert american["selection"]["selected"] == "american_crr"
    assert american["result"] >= european["result"]
    call = run("black_scholes", {"option": dict(option, flag="c")})
    assert call - european["result"] == pytest.approx(100 * np.exp(-0.01) - 105 * np.exp(-0.03))


def test_walk_forward_selects_known_trend_and_refits():
    report = walk_forward(np.arange(40.), initial_train=10, horizon=5, gap=2)
    assert report["selected"] == "drift" and report["ranking"][0]["score"] == 0
    np.testing.assert_equal(report["forecast"], np.arange(40., 45.))
    assert not report["test_performance_estimated"]
    assert report["unused_validation_tail"] == 3
    for fold in report["folds"]:
        assert fold["train_end"] + 2 == fold["validation_start"]


def test_every_fold_fit_receives_only_its_past():
    registry = Registry()
    seen = []
    def spy(data, parameters):
        seen.append(np.asarray(data["series"]).copy())
        return np.full(parameters["horizon"], data["series"][-1])
    registry.register(Algorithm("spy", "Spy", "forecast", "test", ("series",),
                                ("forecast",), source="test"), spy)
    series = np.arange(25.)
    result = walk_forward(series, initial_train=10, horizon=3, gap=1,
                          candidates=["spy"], registry=registry)
    for sample, fold in zip(seen[:-1], result["folds"]):
        np.testing.assert_equal(sample, series[:fold["train_end"]])
    np.testing.assert_equal(seen[-1], series)


def test_candidate_failing_later_fold_cannot_win_with_partial_score():
    registry = default_registry()
    def fragile(data, parameters):
        if len(data["series"]) > 10:
            raise RuntimeError("training failed")
        return np.arange(10, 10 + parameters["horizon"])
    registry.register(Algorithm("fragile", "Fragile", "forecast", "test", ("series",),
                                ("forecast",), source="test"), fragile)
    report = walk_forward(np.arange(25.), initial_train=10, horizon=3,
                          candidates=["fragile", "naive"], registry=registry)
    assert report["selected"] == "naive"
    assert report["rejected"]["fragile"]["completed_folds"] == 1
    assert len(report["ranking"]) == 1


def test_optional_supervised_adapters_fit_only_training(returns):
    if not has_module("sklearn"):
        pytest.skip("sklearn is not installed")
    _optional_process("""
X = np.random.default_rng(77).normal(size=(160, 3)) * [0.01, 0.02, 0.04]
y = 1 + X[:, 0] * 2
for id in ('ridge', 'random_forest_regression', 'logistic', 'random_forest_classification'):
    target = (y > 1).astype(int) if id in ('logistic', 'random_forest_classification') else y
    data = {'X': X, 'y': target, 'X_predict': X[:2]}
    a = run(id, data)
    b = run(id, dict(data, X_predict=np.vstack([X[:1], [1e6] * 3])))
    np.testing.assert_allclose(a[0], b[0])
""")


def test_feature_misalignment_is_rejected():
    if not has_module("sklearn"):
        pytest.skip("sklearn is not installed")
    X = pd.DataFrame(np.ones((20, 2)), columns=["a", "b"])
    with pytest.raises(ValueError, match="columns"):
        run("ridge", {"X": X, "y": np.arange(20.), "X_predict": X[["b", "a"]]})
    with pytest.raises(ValueError, match="indexes"):
        run("ridge", {"X": X, "y": pd.Series(np.arange(20.), index=np.arange(1, 21)),
                      "X_predict": X})


def _optional_process(body, *args):
    # CVXPY's native extension can crash after other numeric runtimes have loaded on
    # Windows. Exercise the REAL bridge in a fresh interpreter, never hide a nonzero exit.
    prelude = """
import json, socket, sys
import numpy as np
import pandas as pd
from fin_skills.algorithms import run
def refuse(*args, **kwargs):
    raise RuntimeError('network disabled in optimizer test')
socket.socket.connect = socket.create_connection = socket.getaddrinfo = refuse
"""
    result = subprocess.run([sys.executable, "-c", prelude + body, *args],
                            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize("backend", ["pypfopt", "skfolio", "riskfolio"])
@pytest.mark.parametrize("method", ["hrp", "min_variance"])
def test_optional_optimizers_through_real_bridge(backend, method, returns):
    if not has_module(backend):
        pytest.skip(f"{backend} is not installed")
    out = _optional_process("""
backend, method = sys.argv[1:]
returns = pd.DataFrame(np.random.default_rng(77).normal(size=(160, 3)) * [0.01, 0.02, 0.04])
parameters = {'linkage': 'single'} if method == 'hrp' else {}
result = run(f'{backend}_{method}', {'asset_returns': returns}, **parameters)
print(json.dumps({'weights': result.tolist(), 'backend': result.attrs['backend']}))
""", backend, method)
    values = json.loads(out)
    # The existing PyPortfolioOpt bridge returns clean_weights rounded to five places.
    assert sum(values["weights"]) == pytest.approx(1, abs=2e-5)
    assert min(values["weights"]) >= -1e-8 and values["backend"] == backend


def test_optional_arima_returns_finite_forecast():
    if not has_module("statsmodels"):
        pytest.skip("statsmodels is not installed")
    _optional_process("""
series = np.random.default_rng(0).normal(size=60).cumsum()
result = run('arima', {'series': series}, horizon=3)
assert result.shape == (3,) and np.isfinite(result).all()
""")


def test_tools_export_and_call_real_algorithm():
    names = {t["name"] for t in list_tools()}
    assert {"list_algorithms", "recommend_algorithms", "run_algorithm", "auto_algorithm",
            "compare_forecast_algorithms"} <= names
    result = call_tool("auto_algorithm", {"task": "forecast", "data": {
        "series": {"index": [1, 2, 3], "values": [10, 11, 12]}}})
    assert result["executed"] and result["result"] == [12]
    json.dumps(result, allow_nan=False)
    comparison = call_tool("compare_forecast_algorithms", {"series": list(range(30)),
                           "initial_train": 10, "horizon": 5})
    assert comparison["selected"] == "drift"
    recommendations = call_tool("recommend_algorithms", {"request": {
        "task": "forecast", "available_inputs": ["series"], "n_observations": 10}})
    assert recommendations["selected"] == "naive"


def test_tool_result_is_complete_or_rejected_not_silently_truncated():
    result = call_tool("run_algorithm", {"algorithm_id": "naive", "data": {"series": [2]},
                       "parameters": {"horizon": 700}})
    assert len(result["result"]) == 700
    with pytest.raises(ValueError, match="10,000"):
        call_tool("run_algorithm", {"algorithm_id": "twap",
                  "data": {"shares": 100, "n_bins": 10_001}})
