"""Behavioral acceptance checks for preflight, holdout isolation, provenance and models."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fin_skills.algorithms import (Algorithm, Registry, Request, auto_run, default_registry,
                                   fit, load_model, profile_data, research)
from fin_skills.algorithms.research import fingerprint


def test_constant_assets_route_to_equal_weight_and_hrp_requires_linkage():
    out = auto_run("portfolio", {"asset_returns": np.zeros((60, 3))})
    assert out["selection"]["selected"] == "equal_weight"
    assert "nonconstant" in str(out["selection"]["rejected"]["inverse_volatility"])
    np.testing.assert_allclose(out["result"], [1 / 3] * 3)
    R = np.random.default_rng(0).normal(0, .01, (60, 3))
    out = auto_run("portfolio", {"asset_returns": R}, preferences=("robust", "correlated_assets"))
    assert out["selection"]["selected"] != "hrp"
    assert "linkage" in str(out["selection"]["rejected"]["hrp"])


def test_request_cannot_lie_about_its_data():
    reg = default_registry()
    with pytest.raises(ValueError, match="must match"):
        reg.recommend(Request("forecast", ("series",), 100), data={"series": [1, 2]})


def test_holdout_changes_do_not_change_selection_or_validation(tmp_path):
    y = np.arange(60.)
    kwargs = dict(initial_train=20, horizon=5, holdout=10, gap=2,
                  candidates=("naive", "drift", "mean"))
    first = research("forecast", {"series": y}, ledger_path=tmp_path / "trials.jsonl", **kwargs)
    other = y.copy()
    other[-10:] = -1000
    second = research("forecast", {"series": other}, **kwargs)
    assert first.selected == second.selected == "drift"
    assert first.validation["ranking"] == second.validation["ranking"]
    assert first.validation["holdout"]["score"] != second.validation["holdout"]["score"]
    assert first.validation["holdout"]["train_end"] == 48
    for fold in first.validation["folds"]:
        assert fold["train_end"] + 2 == fold["start"]
        assert fold["end"] <= 50
    events = [json.loads(s) for s in (tmp_path / "trials.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["registered", "completed"] * 3
    assert len(first.metadata["trial_ids"]) == 3
    path = first.save(tmp_path / "result.json")
    assert json.loads(path.read_text())["schema_version"] == 1
    with pytest.raises(FileExistsError):
        first.save(path)


def test_no_candidate_and_failed_fold_remain_visible():
    reg = default_registry()
    reg.register(Algorithm("bad", "bad", "forecast", "local", ("series",), ("forecast",),
                           source="test", priority=100), lambda d, p: [np.nan] * p["horizon"])
    report = research("forecast", {"series": np.arange(40.)}, registry=reg,
                      initial_train=10, horizon=5, holdout=5, candidates=("bad", "naive"))
    assert report.selected == "naive" and "bad" in report.validation["rejected"]
    report = research("portfolio", {"asset_returns": np.zeros((40, 3))},
                      constraints={"objective": "min_variance"})
    assert report.status == "failed" and report.selected is None


def test_every_fit_receives_only_the_allowed_prefix():
    reg = Registry()
    seen = []
    def spy(data, p):
        seen.append(list(data["series"]))
        return np.full(p["horizon"], data["series"][-1])
    reg.register(Algorithm("spy", "spy", "forecast", "local", ("series",), ("forecast",),
                           source="test"), spy)
    r = research("forecast", {"series": np.arange(30.)}, registry=reg, initial_train=10,
                 horizon=5, gap=2, holdout=5)
    assert [len(x) for x in seen] == [10, 15, 23]
    assert r.validation["holdout"]["start"] == 25


@pytest.mark.parametrize("task,ids,key", [
    ("portfolio", ("equal_weight", "inverse_volatility"), "asset_returns"),
    ("volatility", ("historical_volatility", "ewma_volatility"), "returns"),
    ("risk", ("historical_var_es", "normal_var_es"), "returns"),
    ("signal", ("momentum",), "returns"),
])
def test_task_specific_validation_and_holdout(task, ids, key):
    rng = np.random.default_rng(19)
    data = rng.normal(0, .01, (100, 3) if task == "portfolio" else 100)
    result = research(task, {key: data}, initial_train=40, horizon=10, holdout=20,
                      candidates=ids, cost_bps=5)
    assert result.status in ("completed", "audit_failed")
    assert np.isfinite(result.validation["holdout"]["score"])
    if task in ("portfolio", "signal"):
        assert result.audit["status"] in ("partial", "failed")
        assert len(result.result["net_returns"]) == 20


def test_costs_reduce_portfolio_returns():
    R = np.random.default_rng(88).normal(.0002, .01, (100, 3))
    kw = dict(initial_train=40, horizon=10, holdout=20, candidates=("equal_weight",),
              metric="negative_mean_return")
    a = research("portfolio", {"asset_returns": R}, cost_bps=0, **kw)
    b = research("portfolio", {"asset_returns": R}, cost_bps=20, **kw)
    assert b.validation["holdout"]["score"] > a.validation["holdout"]["score"]


def test_fingerprint_preserves_middle_values_and_feature_order():
    X = np.zeros((2000, 3))
    Y = X.copy()
    Y[1000, 1] = 1
    assert fingerprint({"X": X}) != fingerprint({"X": Y})
    df = pd.DataFrame(Y, columns=list("abc"))
    assert fingerprint({"X": df}) != fingerprint({"X": df[["b", "a", "c"]]})


@pytest.mark.parametrize("id", ["naive", "mean", "drift", "seasonal_naive"])
def test_native_model_roundtrip_and_repeated_predict(id, tmp_path):
    data = {"series": np.arange(40.)}
    if id == "seasonal_naive":
        data["seasonal_period"] = 5
    model = fit(id, data)
    expected = model.predict(horizon=3)
    path = model.save(tmp_path / "model.zip")
    with pytest.raises(ValueError, match="trusted=True"):
        load_model(path)
    loaded = load_model(path, trusted=True)
    np.testing.assert_equal(loaded.predict(horizon=3), expected)
    np.testing.assert_equal(loaded.predict(horizon=5)[:3], expected)


def test_model_environment_and_checksum_rejection(tmp_path):
    model = fit("naive", {"series": [1., 2.]})
    model.metadata["environment"]["numpy"] = "impossible-version"
    path = model.save(tmp_path / "model.zip")
    with pytest.raises(ValueError, match="environment differs"):
        load_model(path, trusted=True)


def test_supervised_persistence_and_feature_contract(tmp_path):
    pytest.importorskip("sklearn")
    X = pd.DataFrame(np.random.default_rng(5).normal(size=(60, 3)), columns=list("abc"))
    y = X.a * 2 - X.b
    model = fit("ridge", {"X": X, "y": y})
    expected = model.predict(X.iloc[-4:])
    loaded = load_model(model.save(tmp_path / "ridge.zip"), trusted=True)
    np.testing.assert_allclose(expected, loaded.predict(X.iloc[-4:]))
    with pytest.raises(ValueError, match="columns"):
        loaded.predict(X[["b", "a", "c"]])
    result = research("regression", {"X": X, "y": y}, initial_train=20, horizon=5,
                      holdout=10, candidates=("ridge",))
    assert result.validation["holdout"]["score"] < .3


def test_runtime_failure_is_not_silently_replaced():
    reg = default_registry()
    def bad(d, p):
        raise RuntimeError("intentional solver failure")
    reg.register(Algorithm("bad", "bad", "forecast", "local", ("series",), ("forecast",),
                           source="test", priority=100), bad)
    result = research("forecast", {"series": [1., 2., 3.]}, registry=reg)
    assert result.selected == "bad" and result.status == "failed"
    assert "intentional solver failure" in result.metadata["errors"]["bad"]


def test_empty_audit_is_not_passed():
    report = research("forecast", {"series": [1., 2.]}, audit_bundle={}, guards=[])
    assert report.audit["status"] == "partial"


def test_research_json_tools_roundtrip():
    from fin_skills.tools import call_tool
    result = call_tool("research_algorithms", {"task": "forecast", "data": {
        "series": list(range(40))}, "evaluation": {"initial_train": 10, "holdout": 10,
        "horizon": 5, "candidates": ["naive", "drift"]}})
    assert result["selected"] == "drift" and result["validation"]["holdout"]["score"] == 0
    json.dumps(result, allow_nan=False)
    profile = call_tool("profile_algorithm_data", {"task": "forecast", "data": {"series": [1, 2, 3]}})
    assert profile["observations"] == 3


def test_single_class_is_rejected_before_execution():
    data = {"X": np.ones((30, 2)), "y": np.zeros(30), "X_predict": np.ones((2, 2))}
    result = auto_run("classification", data)
    assert not result["executed"]
    assert result["selection"]["rejected"]


def test_unvalidated_forecast_honors_horizon_and_rejects_unused_evaluation_options():
    report = research("forecast", {"series": [1., 2., 3.]}, horizon=5)
    assert len(report.result) == 5
    with pytest.raises(ValueError, match="require temporal"):
        research("forecast", {"series": [1., 2., 3.]}, metric="mae")


@pytest.mark.parametrize("bad", [{"holdout": 5}, {"initial_train": 20, "holdout": 50},
                                 {"initial_train": 20, "metric": "not-a-metric"}])
def test_invalid_evaluation_configuration(bad):
    with pytest.raises(ValueError):
        research("forecast", {"series": np.arange(40.)}, **bad)
