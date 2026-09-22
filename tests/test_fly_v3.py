import numpy as np
import pandas as pd
import pytest

from benchmarks.library_utility.evaluate import ledger
from benchmarks.verified_memory.fly_v3 import (Account, audit_receipt, observation,
    receipt_for, simulate, targets_for)
from benchmarks.verified_memory.fly_v2 import Eligibility
from benchmarks.verified_memory.model import synthetic


def test_hold_preserves_units_cash_and_trade_age():
    book = Account(0.2, np.array([0.4, 0.2, 0.1, 0.1]), 4)
    old_units = book.units.copy()
    result = book.rebalance(None, np.array([2., 1., 0.5, 3.]), 10, 20)
    assert result == {"fee": 0., "turnover": 0.}
    np.testing.assert_array_equal(book.units, old_units)
    assert book.cash == 0.2 and book.last_trade == 4


@pytest.mark.parametrize("arm", ["fly_v3", "fly_no_hold", "fly_market_only", "fly_no_cost_input",
                                    "fly_hold_advantage", "dense_v3", "linear_v3", "fly_frozen_v3"])
def test_account_matches_independent_scorer_and_strict_timing(arm):
    prices = synthetic(301, 220)
    result = simulate(prices, arm, 11, start=125, end=219)
    scored = ledger(prices, result["actions"], start=125, end=219, cost_bps=5)
    np.testing.assert_allclose(result["nav"], scored["nav"], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(result["daily_turnover"], scored["daily_turnover"], atol=1e-11)
    rewards = [r["net_reward"] for r in result["receipts"] if r["origin"] >= 125]
    assert np.prod(1 + np.asarray(rewards)) == pytest.approx(result["nav"][-1], abs=1e-11)
    assert result["audit"]["early_updates"] == result["audit"]["blocked_receipts"] == 0
    if arm == "fly_no_hold":
        assert all(a["action"] != "hold" for a in result["actions"])


def test_future_prices_cannot_change_prior_actions_or_learning():
    prices = synthetic(9, 200)
    changed = prices.copy()
    changed.iloc[146:] *= np.linspace(0.3, 4, len(prices) - 146)[:, None]
    for arm in ("fly_v3", "fly_hold_advantage"):
        a = simulate(prices, arm, 11, start=100, end=199, stop_before=146)
        b = simulate(changed, arm, 11, start=100, end=199, stop_before=146)
        # A changed pandas memory layout can change reductions at machine epsilon.
        # Discrete decisions remain exact; numerical results use an explicit tolerance.
        assert_close_tree(a["actions"], b["actions"])
        assert_close_tree(a["updates"], b["updates"])
        np.testing.assert_allclose(a["nav"], b["nav"], rtol=1e-12, atol=1e-12)


def assert_close_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_close_tree(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_close_tree(x, y)
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12)
    else:
        assert a == b


def test_flat_price_fee_loss_and_cash_hold():
    px = pd.DataFrame(np.ones((140, 4)), index=pd.bdate_range("2000-01-03", periods=140, tz="UTC"))
    cash = simulate(px, "fly_v3", 1, start=90, end=139, fixed_action=0)
    invested = simulate(px, "fly_v3", 1, start=90, end=139, fixed_action=1)
    assert cash["nav"][-1] == 1 and sum(cash["fees"]) == 0
    assert invested["nav"][-1] == pytest.approx((1 - 0.0005) / (1 + 0.0005))
    assert invested["audit"]["negative_rewards_learned"] > 0


def test_cost_input_ablation_keeps_shape_and_other_state():
    book = Account(0.4, np.array([0.3, 0.2, 0.1, 0.]), 10)
    px = np.ones(4)
    experts = np.array([np.full(4, 0.25), [1., 0, 0, 0], [0., 1, 0, 0], np.full(4, 0.25), np.zeros(4)])
    targets = targets_for(book, px, experts)
    full, fees = observation(np.ones(20), book, px, targets, 20, 5, "full")
    no_cost, _ = observation(np.ones(20), book, px, targets, 20, 5, "no_cost")
    market_only, _ = observation(np.ones(20), book, px, targets, 20, 5, "market_only")
    assert full.shape == no_cost.shape == (34,)
    np.testing.assert_array_equal(full[:-8], no_cost[:-8])
    assert np.all(no_cost[-8:] == 0) and np.all(market_only[20:] == 0)
    assert fees[0] == 0 and max(fees) > 0


def test_receipt_rejects_backdating_and_wrong_arithmetic():
    prices = synthetic(9, 100)
    book = Account.fresh(4, 64)
    active = {"memory": Eligibility(64, np.ones(1), 0, 0.), "fill": 65, "base": 1., "fee": 0.,
              "cash_before": 1., "units_before": [0.] * 4, "cash_after": 1., "units_after": [0.] * 4}
    receipt = receipt_for(active, book, prices.iloc[70].to_numpy(), 70, 5)
    assert audit_receipt(receipt, prices.index)["passed"]
    assert not audit_receipt({**receipt, "claimed_available": 64}, prices.index)["passed"]
    assert not audit_receipt({**receipt, "net_reward": 0.1}, prices.index)["passed"]
    assert receipt["hold_advantage"] == 0


def test_real_loss_is_valid_and_hold_shadow_accounts_for_terminal_fee():
    idx = pd.bdate_range("2000-01-03", periods=100, tz="UTC")
    book = Account(0., np.array([1., 0, 0, 0]), 65)
    active = {"memory": Eligibility(64, np.ones(1), 0, 0.), "fill": 65, "base": 1., "fee": 0.,
              "cash_before": 0., "units_before": [1., 0, 0, 0], "cash_after": 0., "units_after": [1., 0, 0, 0]}
    receipt = receipt_for(active, book, np.full(4, 0.8), 70, 5, terminal=True)
    assert receipt["net_reward"] < -0.2 and receipt["hold_advantage"] == 0
    assert audit_receipt(receipt, idx)["passed"]
