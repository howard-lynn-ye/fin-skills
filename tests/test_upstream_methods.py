"""Real native backends; no mock of a numerical solver, estimator or execution engine."""
import numpy as np
import pandas as pd
import pytest

from fin_skills.model_zoo import create_model, model_catalog, load_model
from fin_skills.model_zoo.upstream_catalog import (TA_FUNCTIONS, SKLEARN, TIME_SERIES,
    RISK_MEASURES, RISK_SPECIAL, FRONTIERS, CVX, COMPARISONS, NEURAL, RL)


def returns():
    rng = np.random.default_rng(891)
    return pd.DataFrame(rng.normal(.001, .01, (100, 3)),
                        index=pd.date_range("2024-01-01", periods=100), columns=list("ABC"))


@pytest.mark.parametrize("id_", TA_FUNCTIONS)
def test_every_talib_function_native_parity_and_prefix(id_):
    talib = pytest.importorskip("talib")
    from talib import abstract
    meta = TA_FUNCTIONS[id_]
    rng = np.random.default_rng(62)
    close = 20 + np.cumsum(rng.normal(0, .2, 140))
    raw = {"open": close - .1, "high": close + .5, "low": close - .5, "close": close,
           "volume": rng.uniform(100, 500, 140), "periods": np.full(140, 10.)}
    inputs = {field: raw.get(field, np.linspace(.2, .8, 140)) for field in meta["inputs"]}
    model = create_model(id_)
    actual = model.run({"inputs": inputs})
    native = abstract.Function(meta["function"])(inputs)
    expected = [native] if len(meta["outputs"]) == 1 else native
    for name, values in zip(meta["outputs"], expected):
        np.testing.assert_allclose(actual["values"][name], values, equal_nan=True)
    prefix = model.run({"inputs": {k: v[:100] for k, v in inputs.items()}})
    np.testing.assert_allclose(prefix["values"], actual["values"].iloc[:100], equal_nan=True)


def test_talib_rejects_missing_misaligned_and_nonfinite_inputs():
    pytest.importorskip("talib")
    with pytest.raises(ValueError, match="missing"):
        create_model("ta_atr").run({"inputs": {"close": np.ones(30)}})
    with pytest.raises(ValueError, match="indices"):
        create_model("ta_atr").run({"inputs": {
            "high": pd.Series(np.ones(30), index=range(30)),
            "low": pd.Series(np.ones(30), index=range(1, 31)), "close": np.ones(30)}})
    with pytest.raises(ValueError, match="nonfinite"):
        create_model("talib_sma").run({"inputs": {"close": [1, np.nan, 2]}})


@pytest.mark.parametrize("id_", SKLEARN)
def test_sklearn_real_fit_predict_and_artifact(id_, tmp_path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.uniform(.1, 1, (60, 3)), columns=list("ABC"))
    task = SKLEARN[id_][2]
    options = {"n_components": 2} if id_ in ("gmm_regime", "ica_factors", "nmf_factors") else {}
    if id_ in ("kmeans_regime", "spectral_regime"):
        options["n_clusters"] = 2
    if id_ == "mlp":
        options.update(hidden_layer_sizes=(8,), max_iter=50)
    model = create_model(id_, **options)
    y = (X.A > .5).astype(int) if task == "classification" else X.A * 2 + X.B
    data = {"X": X} if task in ("clustering", "decomposition") else {"X": X, "y": y}
    if task == "calibration":
        data = {"X": np.linspace(0, 1, 60), "y": np.linspace(0, 1, 60)**2}
    model.fit(data)
    if id_ in ("dbscan_regime", "spectral_regime"):
        assert len(model.run(data)["labels"]) == 60
        with pytest.raises(ValueError, match="transductive"):
            model.predict(X.iloc[-4:])
        return
    heldout = data["X"][-4:]
    predicted = model.predict(heldout)
    assert len(predicted) == 4 and np.isfinite(predicted).all()
    path = tmp_path / (id_ + ".zip")
    model.save(path)
    np.testing.assert_allclose(load_model(path, trusted=True).predict(heldout), predicted)
    if isinstance(heldout, pd.DataFrame):
        with pytest.raises(ValueError, match="columns"):
            model.predict(heldout[list("CBA")])


@pytest.mark.parametrize("id_", TIME_SERIES)
def test_statsmodels_real_methods(id_):
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(281)
    X = rng.normal(size=(160, 3))
    options = {}
    if id_ == "stl_decomposition":
        options["period"] = 12
    if id_ in ("markov_regression", "markov_autoregression"):
        options.update(switching_variance=True, fit_parameters={"maxiter": 500})
        X[:80] *= .2
        X[80:] += 4.
    if id_ == "dynamic_factor":
        options["fit_parameters"] = {"maxiter": 500}
    if id_ == "granger_test":
        X = X[:, :2]
    data = {TIME_SERIES[id_][1]: X[:, 0] if TIME_SERIES[id_][1] == "series" else X}
    model = create_model(id_, **options)
    if TIME_SERIES[id_][0] == "forecast":
        forecast = model.fit(data).predict(horizon=3)
        assert len(forecast) == 3 and np.isfinite(forecast).all()
    else:
        result = model.run(data)
        assert isinstance(result, dict) and result


@pytest.mark.parametrize("id_", [*RISK_MEASURES, *RISK_SPECIAL, *FRONTIERS])
def test_native_optimizers(id_):
    pytest.importorskip("riskfolio")
    pytest.importorskip("pypfopt")
    r = returns()
    data = {"asset_returns": r}
    if id_ == "factor_risk_budget":
        data["factors"] = r[["A", "B"]].rename(columns={"A": "f1", "B": "f2"})
    options = {"portfolio": {"sht": True}} if id_ == "factor_risk_budget" else {}
    result = create_model(id_, **options).run(data)
    weights = np.asarray(result).ravel()
    assert len(weights) == 3 and np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1., abs=1e-4)
    if id_ != "factor_risk_budget":
        assert weights.min() >= -1e-4
    else:
        with pytest.raises(RuntimeError, match="long-only"):
            create_model(id_).run(data)


@pytest.mark.parametrize("id_", CVX)
def test_native_cvx_decisions_parity_and_no_future_dependency(id_, tmp_path):
    pytest.importorskip("cvxportfolio")
    from fin_skills.bridges.cvxportfolio import make_market_data, native_policy
    r = returns()
    r["USDOLLAR"] = 0.
    t = r.index[70]
    h = pd.Series([100., 200., 100., 600.], index=r.columns)
    target = pd.Series([.3, .3, .3, .1], index=r.columns)
    p = {}
    if id_ in ("single_period_optimization", "multi_period_optimization"):
        p = dict(expected_returns=pd.Series([.001, .002, .001], index=list("ABC")),
                 covariance=pd.DataFrame(np.eye(3)*.001, index=list("ABC"), columns=list("ABC")))
    elif id_ in ("periodic_rebalance", "proportional_rebalance", "adaptive_rebalance"):
        p["target"] = target
        p[{"periodic_rebalance": "rebalancing_times", "proportional_rebalance": "target_matching_times",
           "adaptive_rebalance": "tracking_error"}[id_]] = .05 if id_ == "adaptive_rebalance" else [t]
    market = make_market_data(r, cache_dir=tmp_path / "native")
    direct = native_policy(id_, **p).execute(h, market, t=t)
    result = create_model(id_, **p).run({"holdings": h, "market_data": market, "t": t})
    pd.testing.assert_series_equal(result[0], direct[0], atol=1e-4, rtol=1e-5)
    changed = r.copy()
    changed.loc[changed.index >= t, "A"] = .1
    other = make_market_data(changed, cache_dir=tmp_path / "other")
    out = create_model(id_, **p).run({"holdings": h, "market_data": other, "t": t})
    pd.testing.assert_series_equal(result[0], out[0], atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("id_", COMPARISONS)
def test_arch_comparison_real_bootstrap(id_):
    pytest.importorskip("arch")
    rng = np.random.default_rng(12)
    data = dict(losses=rng.normal(size=(90, 3)), benchmark_losses=rng.normal(size=90))
    result = create_model(id_, reps=20).run(data)
    assert result


@pytest.mark.parametrize("id_", ["temporal_cv", "grouped_cv"])
def test_native_cv_splits(id_):
    pytest.importorskip("sklearn")
    X = np.arange(80.).reshape(40, 2)
    data = {"X": X}
    if id_ == "grouped_cv":
        data["groups"] = np.repeat(np.arange(4), 10)
    splits = create_model(id_, n_splits=3).run(data)
    assert len(splits) == 3
    for split in splits:
        assert not set(split["train"]) & set(split["test"])
        if id_ == "temporal_cv":
            assert split["train"].max() < split["test"].min()


@pytest.mark.parametrize("id_", RL)
def test_new_sb3_models_train_predict_save(id_, tmp_path):
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    env = gym.make("CartPole-v1" if id_ in ("dqn", "a2c") else "Pendulum-v1")
    parameters = dict(n_steps=4) if id_ == "a2c" else dict(buffer_size=100, learning_starts=0, batch_size=4)
    try:
        model = create_model(id_, **parameters).fit(env, total_timesteps=8)
        observation, _ = env.reset(seed=7)
        prediction = model.predict(observation)
        assert env.action_space.contains(prediction)
        path = tmp_path / (id_ + ".zip")
        model.save(path)
        np.testing.assert_allclose(load_model(path, trusted=True).predict(observation), prediction)
    finally:
        env.close()


@pytest.mark.parametrize("id_", NEURAL)
def test_new_neuralforecast_models_fit_predict(id_):
    pytest.importorskip("neuralforecast")
    dates = pd.date_range("2024-01-01", periods=32)
    history = pd.DataFrame({"unique_id": "a", "ds": dates, "y": np.sin(np.arange(32.))})
    model = create_model(id_, freq="D", h=2, input_size=8, max_steps=1,
                         batch_size=1, windows_batch_size=8)
    prediction = model.fit(history).predict()
    assert len(prediction) == 2
    assert np.isfinite(prediction.drop(columns=["unique_id", "ds"]).to_numpy()).all()


def test_integrated_catalog_deduplicates():
    from fin_skills.algorithms import get_method, method_coverage
    from fin_skills.tools import call_tool
    from fin_skills.tools.payloads import decode
    card = get_method("ta_atr")
    assert card["status"] == "integrated" and card["model_id"] == "ta_atr"
    assert get_method("model:ta_atr")["id"] == "ta_atr"
    assert get_method("cointegration_pairs")["status"] == "external"


def test_json_native_talib_call():
    pytest.importorskip("talib")
    from fin_skills.tools import call_tool
    result = call_tool("run_model", {"model_id": "talib_sma", "parameters": {"timeperiod": 3},
        "data": {"inputs": {"close": list(range(1, 20))}}})
    assert result["result"]["lookback"] == 2
