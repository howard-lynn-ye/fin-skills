"""Tests for fin_skills.china.board_lot_guard and BoardLotFeasibilityGuard."""
from __future__ import annotations

import pytest

from fin_skills.api import get
from fin_skills.china.board_lot_guard import evaluate_board_lot_feasibility


def test_board_lot_feasibility_small_capital_distortion():
    # 15,000 RMB trying to allocate 30% to 511010 (~135 RMB/share = 13,500 RMB/lot)
    weights = {"510300": 0.20, "511010": 0.30, "518880": 0.25, "513100": 0.25}
    res = evaluate_board_lot_feasibility(capital=15000.0, target_weights=weights)
    assert not res.is_feasible
    assert res.feasibility_status == "UNFEASIBLE_HIGH_DISTORTION"
    assert res.granularity_distortion > 0.25
    assert res.min_recommended_capital >= 50000.0


def test_board_lot_feasibility_adequate_capital():
    # 200,000 RMB supports clean lot rounding
    weights = {"510300": 0.20, "511010": 0.30, "518880": 0.25, "513100": 0.25}
    res = evaluate_board_lot_feasibility(capital=200000.0, target_weights=weights)
    assert res.is_feasible
    assert res.feasibility_status == "EXCELLENT"
    assert res.granularity_distortion < 0.10


def test_board_lot_guard_api():
    g = get("board_lot_feasibility")
    weights = {"510300": 0.20, "511010": 0.30, "518880": 0.25, "513100": 0.25}

    # Small capital -> error
    res_err = g.run(capital=12000.0, target_weights=weights)
    assert not res_err.passed
    assert any(f.severity == "error" for f in res_err.findings)

    # Large capital -> pass
    res_ok = g.run(capital=150000.0, target_weights=weights)
    assert res_ok.passed
    assert not any(f.severity == "error" for f in res_ok.findings)
