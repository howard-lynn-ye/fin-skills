"""Public adapters: actual execution, numerical parity, isolation and persistence."""
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

from fin_skills.model_zoo import create_model, load_model, model_catalog
from fin_skills.model_zoo.feedback import NavMark, IntervalCredit, OptionReceipt


def test_discovery_is_unique_and_operations_are_explicit():
    cards = model_catalog()
    assert len({c["id"] for c in cards}) == len(cards)
    json.dumps(cards, allow_nan=False)
    assert "update" in next(c for c in cards if c["id"] == "fly_memory")["operations"]
    assert all(c["pretrained"] is False for c in cards if c["id"] != "jev")
    jev = next(c for c in cards if c["id"] == "jev")
    assert jev["pretrained"] is True and jev["weights_bundled"] is False
    with pytest.raises(ValueError, match="unknown"):
        model_catalog("typo")
    with pytest.raises(KeyError, match="unknown"):
        create_model("typo")


def test_numerical_covariance_parity_and_json():
    from fin_skills.models.risk_model import ledoit_wolf
    from fin_skills.tools import call_tool
    X = np.random.default_rng(9).normal(0, .01, (80, 3))
    out = create_model("ledoit_wolf_covariance").run({"asset_returns": X})
    np.testing.assert_allclose(out["covariance"], ledoit_wolf(X)[0])
    for name in ("ewma_covariance", "pca_covariance"):
        result = create_model(name).run({"asset_returns": X})
        assert result["covariance"].shape == (3, 3)
        assert np.linalg.eigvalsh(result["covariance"]).min() > 0
    result = call_tool("run_model", {"model_id": "ledoit_wolf_covariance",
        "data": {"asset_returns": X.tolist()}})
    json.dumps(result, allow_nan=False)
    assert call_tool("list_models", {"task": "covariance"})["models"]
    with pytest.raises((TypeError, ValueError), match="stateful"):
        call_tool("run_model", {"model_id": "ppo", "data": {}})


def test_forward_kalman_is_prefix_causal():
    from fin_skills.models.kalman_models import kalman_filter
    data = dict(y=np.arange(12.), Z=np.ones((12, 1)), T=np.eye(1),
                Q=np.eye(1)*.1, H=.5, a0=np.zeros(1), P0=np.eye(1))
    model = create_model("kalman_filter")
    actual = model.run(data)
    np.testing.assert_allclose(actual["filtered"], kalman_filter(**data)["filtered"])
    changed = dict(data, y=np.r_[np.arange(6.), np.ones(6)*999])
    np.testing.assert_array_equal(actual["filtered"][:6], model.run(changed)["filtered"][:6])
    with pytest.raises(ValueError, match="semidefinite"):
        model.run(dict(data, Q=-np.eye(1)))


def test_curve_credit_and_surface_adapters():
    from fin_skills.models.term_structure import nelson_siegel, svensson
    from fin_skills.models.credit_models import merton_equity
    from fin_skills.models.vol_surface import svi_total_variance
    t = np.array([.25, .5, 1., 2., 3., 5., 7., 10., 20.])
    y = nelson_siegel(t, .03, -.02, .01, .4)
    ns = create_model("nelson_siegel", lam=.4).run({"maturities": t, "yields": y})
    np.testing.assert_allclose(ns["betas"], [.03, -.02, .01], atol=1e-10)
    sy = svensson(t, .03, -.02, .01, .005, .4, 2.)
    out = create_model("svensson").run({"maturities": t, "yields": sy})
    assert out["rmse"] < 1e-4
    E, s = merton_equity(150, .25, 100, .03, 1.)
    credit = create_model("merton_credit").run(dict(E=E, sigma_E=s, D=100, r=.03, T=1.))
    assert credit["V"] == pytest.approx(150, abs=1e-6)
    k = np.linspace(-.5, .5, 21)
    w = svi_total_variance(k, .02, .1, -.3, 0., .2)
    surface = create_model("svi_surface", starts=2).run({"log_moneyness": k, "total_variance": w})
    assert surface["rmse_w"] < 1e-5


def test_classic_lifecycle_and_trusted_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    X = np.random.default_rng(8).normal(size=(80, 2))
    model = create_model("ridge").fit({"X": X, "y": X[:, 0]})
    out = model.predict(X[:4])
    path = model.save(tmp_path/"model.zip")
    with pytest.raises(ValueError, match="trusted"):
        load_model(path)
    np.testing.assert_array_equal(load_model(path, trusted=True).predict(X[:4]), out)
    with pytest.raises(FileExistsError):
        model.save(path)


NEURAL_PARAMETERS = {
    "lstm_forecast": dict(encoder_hidden_size=8, encoder_n_layers=1,
                          decoder_hidden_size=8, decoder_layers=1),
    "transformer_forecast": dict(hidden_size=8, n_head=2, encoder_layers=1,
                                 decoder_layers=1, conv_hidden_size=8),
    "patchtst_forecast": dict(hidden_size=8, n_heads=2, encoder_layers=1,
                             linear_hidden_size=8, patch_len=4, stride=2),
    "nhits_forecast": dict(mlp_units=[[8, 8]] * 3),
}


def _panel():
    return pd.DataFrame({"unique_id": "synthetic", "ds": pd.date_range("2024-01-01", periods=40),
                         "y": np.sin(np.arange(40) / 3)})


@pytest.mark.parametrize("name", NEURAL_PARAMETERS)
def test_neural_native_training_and_checkpoint_roundtrip(name, tmp_path):
    pytest.importorskip("neuralforecast")
    from fin_skills.model_zoo.neural import MODELS
    model = create_model(name, freq="D", h=2, input_size=8, max_steps=2,
                         windows_batch_size=8, **NEURAL_PARAMETERS[name]).fit(_panel())
    native = model.forecaster.models[0]
    assert type(native).__name__ == MODELS[name]
    assert len(native.train_trajectories) == 2
    assert all(np.isfinite(loss) for _, loss in native.train_trajectories)
    out = model.predict()
    assert len(out) == 2 and np.isfinite(out[name]).all()
    assert out.ds.min() > _panel().ds.max()
    # Repeated prediction uses fitted weights without another training call.
    pd.testing.assert_frame_equal(out, model.predict())
    path = model.save(tmp_path/(name+".zip"))
    restored = load_model(path, trusted=True)
    pd.testing.assert_frame_equal(restored.predict(), out)
    pd.testing.assert_frame_equal(restored.predict(_panel()), model.predict(_panel()))


@pytest.mark.parametrize("name,env_name", [("ppo", "CartPole-v1"), ("sac", "Pendulum-v1")])
def test_rl_real_backend_training_action_and_roundtrip(name, env_name, tmp_path):
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    env = gym.make(env_name)
    try:
        params = (dict(n_steps=8, batch_size=8, n_epochs=1) if name == "ppo" else
                  dict(buffer_size=100, learning_starts=4, batch_size=4))
        model = create_model(name, seed=11, policy_kwargs={"net_arch": [16]}, **params)
        model.fit(env, total_timesteps=16)
        obs, _ = env.reset(seed=23)
        action = model.predict(obs)
        assert env.action_space.contains(action)
        path = model.save(tmp_path/(name+".zip"))
        np.testing.assert_array_equal(load_model(path, trusted=True).predict(obs), action)
    finally:
        env.close()


def test_sac_rejects_discrete_actions():
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    try:
        with pytest.raises(ValueError, match="Box"):
            create_model("sac").fit(env, total_timesteps=8)
    finally:
        env.close()


def test_fly_mature_loss_duplicate_and_roundtrip(tmp_path):
    pytest.importorskip("fin_skills_fly")
    # Synthetic circuit parameters test the interface, not a fitted biological model.
    p = [np.array([[[0., 0., 0., 10., 2., 3.]]]), np.full((1,1,1), -1.),
         np.full((1,6,2), -1.), np.zeros((1,6,6)), np.full((1,1,3), 1e6),
         np.full((1,1,1), 100.)]
    model = create_model("fly_memory", circuit_parameters=p)
    a = NavMark(pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02"), 1., "fixture")
    b = NavMark(pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-04"), .99, "fixture")
    receipt = OptionReceipt("loss", pd.Timestamp("2025-01-01"),
        (IntervalCredit(a, b, math.log(.99)),), pd.Timestamp("2025-01-04"))
    before = model.predict()
    assert model.update(receipt, now="2025-01-04")["status"] == "pending"
    np.testing.assert_array_equal(model.predict(), before)
    assert model.update(receipt, now="2025-01-05")["target"] < 0
    assert not np.array_equal(model.predict(), before)
    path = model.save(tmp_path/"fly.zip")
    restored = load_model(path, trusted=True)
    np.testing.assert_array_equal(restored.predict(), model.predict())
    assert restored.update(receipt, now="2025-01-05")["status"] == "duplicate"


def test_missing_optional_dependency_is_actionable(monkeypatch):
    import fin_skills.model_zoo.catalog as module
    monkeypatch.setattr(module, "_available", lambda _: False)
    with pytest.raises(ImportError, match=r"fin-skills\[deep\]"):
        create_model("lstm_forecast", freq="D")


def test_artifact_tampering_is_rejected_before_loading(tmp_path):
    import zipfile
    model = create_model("naive").fit({"series": np.arange(10.)})
    original = model.save(tmp_path/"original.zip")
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(tmp_path/"bad.zip", "w") as bad:
        bad.writestr("manifest.json", source.read("manifest.json"))
        bad.writestr("model.pkl", source.read("model.pkl") + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path/"bad.zip", trusted=True)


def test_neural_history_contract_rejects_ambiguous_inputs():
    from fin_skills.model_zoo.neural import NeuralForecastModel
    model = NeuralForecastModel("lstm_forecast", freq="D", input_size=8)
    data = _panel()
    for invalid, message in [(data.iloc[::-1], "ordered"),
                             (data.drop(index=10), "freq"),
                             (data.assign(y=np.nan), "missing"),
                             (data.assign(future_target=0), "exactly")]:
        with pytest.raises(ValueError, match=message):
            model._history(invalid, 9)
    with pytest.raises(ValueError, match="at least"):
        model._history(data.iloc[:3], 9)
    with pytest.raises(ValueError, match="before prediction"):
        model.predict()
