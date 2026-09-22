"""Real upstream integration tests; replayed agent responses are transport fixtures."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest


def _market(n=6):
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame([dict(date=d, tic=t, open=p, close=p, feature=float(i))
        for i, d in enumerate(dates) for t, p in (("AAA", 10.), ("BBB", 20.))])


def test_finrl_next_open_cost_ledger_and_future_isolation():
    pytest.importorskip("stable_baselines3")
    from fin_skills.bridges.finrl import make_finrl_env
    original = _market()
    future = original.copy()
    future.loc[future.date >= future.date.unique()[2], ["open", "close", "feature"]] *= 10
    a = make_finrl_env(original, initial_cash=1000, hmax=10, buy_cost=.01, sell_cost=.01,
                       features=["feature"])
    b = make_finrl_env(future, initial_cash=1000, hmax=10, buy_cost=.01, sell_cost=.01,
                       features=["feature"])
    np.testing.assert_array_equal(a.reset(seed=3)[0], b.reset(seed=3)[0])
    first, second = a.step([1., 0.]), b.step([1., 0.])
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    assert first[4]["fees"] == 1 and first[4]["nav"] == 999
    assert first[1] == pytest.approx(np.log(999 / 1000))
    _, _, _, _, sale = a.step([-1., 0.])
    assert sale["units"] == [0., 0.] and sale["nav"] == 998
    while not a.finished:
        a.step([0., 0.])
    assert np.exp(sum(r["log_return"] for r in a.ledger)) == pytest.approx(a.nav / 1000)
    assert len(a.ledger) == len(original.date.unique()) - 1
    with pytest.raises(RuntimeError, match="finished"):
        a.step([0., 0.])
    a.close(); b.close()


def test_finrl_executes_at_next_open_and_rejects_bad_data():
    from fin_skills.bridges.finrl import make_finrl_env
    data = _market()
    data.loc[(data.date == data.date.unique()[1]) & (data.tic == "AAA"), "open"] = 12.
    env = make_finrl_env(data, initial_cash=1000, hmax=10, buy_cost=0, sell_cost=0)
    obs, _ = env.reset()
    assert obs[1] == 10  # Today's close; next open (12) is hidden.
    info = env.step([1., 0.])[4]
    assert info["cash"] == 880 and info["nav"] == 980
    with pytest.raises(ValueError, match="complete"):
        make_finrl_env(data.iloc[1:])
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan, 0.])
    env.close()


@pytest.mark.parametrize("name", ["ppo", "sac"])
def test_real_rl_policy_trains_in_finrl(name):
    pytest.importorskip("stable_baselines3")
    from fin_skills.bridges.finrl import make_finrl_env
    from fin_skills.model_zoo import create_model
    env = make_finrl_env(_market(), initial_cash=1000)
    params = dict(n_steps=8, batch_size=8, n_epochs=1) if name == "ppo" else dict(
        learning_starts=2, buffer_size=100, batch_size=4)
    try:
        model = create_model(name, policy_kwargs={"net_arch": [8]}, **params)
        model.fit(env, total_timesteps=16)
        obs, _ = env.reset(seed=1)
        assert env.action_space.contains(model.predict(obs))
    finally:
        env.close()


def _qlib_provider(root):
    """Synthetic local Qlib binary fixture, no network or external market data."""
    dates = pd.date_range("2025-01-01", periods=8, freq="B")
    (root / "calendars").mkdir()
    (root / "instruments").mkdir()
    (root / "calendars/day.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in dates))
    (root / "instruments/all.txt").write_text("\n".join(
        f"{s}\t{dates[0]:%Y-%m-%d}\t{dates[-1]:%Y-%m-%d}" for s in ("AAA", "BBB")))
    for ticker, price in (("AAA", 10.), ("BBB", 20.)):
        folder = root / "features" / ticker.lower()
        folder.mkdir(parents=True)
        for field, value in {"open": price, "close": price, "high": price,
                             "low": price, "volume": 100000., "factor": 1., "change": 0.}.items():
            np.r_[0., np.full(len(dates), value)].astype("<f4").tofile(folder / f"{field}.day.bin")
    return dates


def test_qlib_native_strategy_executor_parity_and_timing(tmp_path):
    qlib = pytest.importorskip("qlib")
    from fin_skills.bridges.qlib_strategy import qlib_backtest
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    dates = _qlib_provider(tmp_path)
    qlib.init(provider_uri=str(tmp_path), region="us", expression_cache=None,
              dataset_cache=None, kernels=1, joblib_backend="threading")
    idx = pd.MultiIndex.from_product([dates[:-1], ["AAA", "BBB"]], names=["datetime", "instrument"])
    scores = pd.Series([2., 1.] * (len(dates)-1), index=idx)
    ready = pd.Series(idx.get_level_values("datetime") + pd.Timedelta(hours=16), index=idx)
    settings = dict(start=dates[1], end=dates[-2], account=1000., topk=1, n_drop=1,
                    open_cost=.001, close_cost=.001, min_cost=0.)
    out = qlib_backtest(scores, ready, **settings)
    native, _ = backtest(start_time=dates[1], end_time=dates[-2], account=1000.,
        benchmark=pd.Series(0., index=dates[1:-1]),
        strategy=TopkDropoutStrategy(signal=scores, topk=1, n_drop=1, risk_degree=1.),
        executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True),
        exchange_kwargs=dict(freq="day", deal_price="$open", trade_unit=1,
                             limit_threshold=None, open_cost=.001, close_cost=.001, min_cost=0.))
    pd.testing.assert_frame_equal(out["report"], native["1day"][0])
    assert out["report"]["cost"].sum() > 0 and out["nav"].iloc[-1] < 1
    modified = scores.copy()
    modified.loc[(dates[4], "BBB")] = 1000.
    changed = qlib_backtest(modified, ready, **settings)
    pd.testing.assert_frame_equal(out["report"].loc[:dates[4]], changed["report"].loc[:dates[4]])
    with pytest.raises(ValueError, match="unavailable"):
        qlib_backtest(scores, ready + pd.Timedelta(days=1), **settings)
    from fin_skills.tools.agent import ModelSession
    session = ModelSession({"signals": ("backtest", dict(scores=scores, available_at=ready,
        start=dates[1], end=dates[-2]))})
    output = session.backtest_strategy("signals", topk=1, n_drop=1, account=1000.,
                                      open_cost=.001, close_cost=.001, min_cost=0.)
    json.dumps(output, allow_nan=False)
    assert output["result"]["provenance"]["strategy"] == "TopkDropoutStrategy"


def test_model_session_data_roles_and_lifecycle():
    from fin_skills.tools.agent import ModelSession
    session = ModelSession({"training": ("train", {"series": np.arange(10.)}),
                            "future": ("predict", np.arange(3.))})
    with pytest.raises(ValueError, match="train"):
        session.fit_model("naive", "future", "invalid")
    session.fit_model("drift", "training", "trend")
    assert session.predict_model("trend", horizon=2)["result"] == [10., 11.]
    assert [r["operation"] for r in session.receipts] == ["fit", "predict"]
    with pytest.raises(ValueError, match="new"):
        session.fit_model("naive", "training", "trend")


@pytest.mark.parametrize("chosen,expected", [("naive", [9., 9.]), ("drift", [10., 11.])])
def test_actual_autogen_loop_executes_model_selected_tools(chosen, expected):
    pytest.importorskip("autogen_agentchat")
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from fin_skills.tools.agent import ModelSession, make_tool_agent
    def response(name, arguments, call_id):
        return CreateResult(finish_reason="function_calls", usage=RequestUsage(prompt_tokens=0,
            completion_tokens=0), content=[FunctionCall(id=call_id, name=name,
                arguments=json.dumps(arguments))], cached=False)
    session = ModelSession({"train": ("train", {"series": np.arange(10.)})})
    # Upstream replay client supplies tool selections; this is not live LLM evidence.
    client = ReplayChatCompletionClient([
        response("list_models", {"task": "forecast"}, "1"),
        response("fit_model", {"model_id": chosen, "data_id": "train", "handle": "model"}, "2"),
        response("predict_model", {"handle": "model", "horizon": 2}, "3"), "Done"],
        model_info=dict(vision=False, function_calling=True, json_output=False,
                        family="unknown", structured_output=False))
    agent = make_tool_agent(client, session=session, allowed_tools=["list_models"])
    result = asyncio.run(agent.run(task="Choose a model and forecast two steps."))
    assert result.messages[-1].content == "Done"
    assert session.receipts[-1]["result"] == expected
    assert session.receipts[0]["model_id"] == chosen


def test_autogen_allowlist_and_invalid_tool_schema():
    pytest.importorskip("autogen_agentchat")
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from fin_skills.tools.agent import make_tool_agent
    with pytest.raises(ValueError, match="registered"):
        make_tool_agent(ReplayChatCompletionClient(["unused"]), allowed_tools=["run_shell"])


def test_autogen_bad_call_is_feedback_not_execution():
    pytest.importorskip("autogen_agentchat")
    from autogen_core import FunctionCall
    from autogen_core.models import CreateResult, RequestUsage
    from autogen_ext.models.replay import ReplayChatCompletionClient
    from fin_skills.tools.agent import make_tool_agent
    response = CreateResult(finish_reason="function_calls", cached=False,
        usage=RequestUsage(prompt_tokens=0, completion_tokens=0), content=[FunctionCall(
            id="bad", name="run_model", arguments=json.dumps({"model_id": "naive"}))])
    client = ReplayChatCompletionClient([response, "Input was rejected"], model_info=dict(
        vision=False, function_calling=True, json_output=False, family="unknown", structured_output=False))
    agent = make_tool_agent(client, allowed_tools=["run_model"])
    result = asyncio.run(agent.run(task="Validate a call"))
    failures = [value for message in result.messages if message.type == "ToolCallExecutionEvent"
                for value in message.content]
    assert len(failures) == 1 and failures[0].is_error
    assert "data" in failures[0].content


def test_model_session_rl_rollout_budget_and_bound_environment():
    from fin_skills.tools.agent import ModelSession
    from fin_skills.bridges.finrl import make_finrl_env
    session = ModelSession(environment_factories={"market": lambda: make_finrl_env(_market())},
                           max_training_steps=24)
    with pytest.raises(ValueError, match="budget"):
        session.fit_model("ppo", "market", "too_large", {"n_steps": 16}, total_timesteps=17)
    session.fit_model("ppo", "market", "policy", {"n_steps": 8, "n_epochs": 1}, total_timesteps=16)
    assert session.models["policy"].estimator.num_timesteps == 16
