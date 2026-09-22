"""Financial timing, credit assignment and evaluation-state regression checks."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.fly_reuse.core import activate, Account, Compact, quote, run_window

UPSTREAM = Path(os.environ.get("FLY_REUSE_UPSTREAM", "runs/fly_reuse"))
pytestmark = pytest.mark.skipif(not (UPSTREAM/"stonkfly-lab").is_dir(),
                                reason="Pinned upstream checkouts are required")


@pytest.fixture(autouse=True)
def upstream():
    if (UPSTREAM/"stonkfly-lab").is_dir():
        activate(UPSTREAM)


def bars(n=50):
    return pd.DataFrame({"t": np.arange(n)*86400 + 1704067200,
                         "open": np.full(n, 100.0), "close": np.full(n, 100.0)})


def state(policy, held=False):
    return policy.encode(np.linspace(99, 100, 30), 1704153599.999, held)


def test_same_bar_fill_refused_and_fee_in_return(tmp_path):
    a = Account(tmp_path/"book.sqlite")
    q = quote(100, 1704153600)
    with pytest.raises(ValueError, match="strictly"):
        a.execute(1, q, q.timestamp)
    before = a.nav(q)
    a.execute(1, q, q.timestamp-.001)
    after = a.nav(q)
    assert after < before
    assert len(a.fills) == 1 and a.fills[0]["fee"] > 0
    assert np.isclose(after, a.cash+a.units*float(q.bid))
    a.close()


def test_next_open_and_flat_market_net_loss(tmp_path):
    f = bars()
    f.loc[22, "open"] = 110
    out = tmp_path/"run"
    result = run_window(f, 21, 27, None, out, constant=1)
    fills = json.loads((out/"fills.json").read_text())
    assert len(fills) == 1
    assert fills[0]["fill_time"] == int(f.t.iloc[22])
    assert np.isclose(fills[0]["amount"]/fills[0]["quantity"], float(quote(110, 1).ask))
    assert result["net_return_pct"] < 0


def test_cash_zero_and_future_tail_cannot_change_prefix(tmp_path):
    f = bars()
    p1, p2 = Compact("fly", 11, .1), Compact("fly", 11, .1)
    run_window(f, 21, 45, p1, tmp_path/"a", training=True)
    f.loc[35:, "close"] = 200
    run_window(f, 21, 45, p2, tmp_path/"b", training=True)
    a = json.loads((tmp_path/"a/trace.json").read_text())
    b = json.loads((tmp_path/"b/trace.json").read_text())
    assert a[:14] == b[:14]
    assert run_window(f, 21, 30, None, tmp_path/"cash", constant=0)["net_return_pct"] == 0


def test_profitable_flat_action_strengthens_sell_not_buy():
    policy = Compact("fly", 11, .1, batch_size=1)
    s = state(policy)
    old = {**s, "action": 0, "probability": .5, "gated": False}
    buy, sell = policy.brain.w_buy.copy(), policy.brain.w_sell.copy()
    policy.receive(old, .01, s, 5, True)
    assert policy.brain.w_buy.sum() < buy.sum()
    assert policy.brain.w_sell.sum() > sell.sum()


def test_real_shuffle_and_maturity_check():
    p = Compact("shuffled", 11, .1, batch_size=4)
    s = state(p)
    old = {**s, "action": 1, "probability": .5, "gated": False}
    for i in range(4):
        p.receive(old, .001*i, s, i, True)
    order = p.shuffle_records[0]["order"]
    assert sorted(order) == [0, 1, 2, 3]
    assert all(i != v for i, v in enumerate(order))
    p.receive(old, .01, s, 100, True)
    with pytest.raises(AssertionError, match="Unmatured"):
        p.flush(99)


def test_evaluation_weights_frozen_and_dynamic_state_reset(tmp_path):
    p = Compact("fly", 11, .1)
    run_window(bars(), 21, 45, p, tmp_path/"train", training=True)
    f = p.frozen_copy()
    assert f.weights_hash() == p.weights_hash()
    assert not f.encoder.ret_hist and np.all(f.brain.eligibility == 0)
    result = run_window(bars(), 21, 45, f, tmp_path/"test")
    assert result["initial_weights"] == result["final_weights"]


def test_risk_gate_does_not_erase_opportunity_memory():
    p = Compact("fly_gated", 11, -.1, batch_size=1)
    s = state(p)
    old = p.decide(s, True)
    assert old["gated"] and old["action"] == 0
    before = p.brain.w_buy.copy()
    p.receive(old, -.02, s, 5, True)
    assert np.array_equal(before, p.brain.w_buy)


def test_encoder_uses_market_time(monkeypatch):
    import flylab.odors
    monkeypatch.setattr(flylab.odors, "session_bin", lambda: 0)
    a = state(Compact("fly", 11, .1))
    monkeypatch.setattr(flylab.odors, "session_bin", lambda: 3)
    b = state(Compact("fly", 11, .1))
    assert np.array_equal(a["kc"], b["kc"])
