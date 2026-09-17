"""Tests for fin_skills.china.qdii_premium_guard and QDIIPremiumGuard."""
from __future__ import annotations

import pytest

from fin_skills.api import get
from fin_skills.china.qdii_premium_guard import (
    KNOWN_QDII_ETFS,
    calculate_premium_rate,
    evaluate_qdii_order,
    is_qdii_etf,
)


def test_is_qdii_etf_identification():
    assert is_qdii_etf("513100")
    assert is_qdii_etf("513500.SH")
    assert is_qdii_etf("159941.SZ")
    assert is_qdii_etf("513050")
    assert not is_qdii_etf("510300")  # HS300 is domestic
    assert not is_qdii_etf("518880")  # Gold is domestic


def test_calculate_premium_rate():
    # Price 1.10, IOPV 1.00 -> +10% premium
    assert pytest.approx(calculate_premium_rate(1.10, 1.00)) == 0.10
    # Price 0.98, IOPV 1.00 -> -2% discount
    assert pytest.approx(calculate_premium_rate(0.98, 1.00)) == -0.02
    with pytest.raises(ValueError):
        calculate_premium_rate(-1.0, 1.0)
    with pytest.raises(ValueError):
        calculate_premium_rate(1.0, 0.0)


def test_evaluate_qdii_order_circuit_breaker():
    # 513100 trading at 1.15 with IOPV 1.00 (15% premium) -> HARD CIRCUIT
    res = evaluate_qdii_order("513100", price=1.15, iopv=1.00)
    assert res.status == "HARD_CIRCUIT"
    assert res.allowed_weight_factor == 0.0
    assert res.redirect_target == "518880"
    assert "CRITICAL" in res.message


def test_evaluate_qdii_order_derate():
    # 513500 trading at 1.02 with IOPV 1.00 (2.0% premium) -> DERATE 50%
    res = evaluate_qdii_order("513500", price=1.02, iopv=1.00)
    assert res.status == "DERATE_50"
    assert res.allowed_weight_factor == 0.5
    assert res.redirect_target == "518880"
    assert "WARNING" in res.message


def test_evaluate_qdii_order_normal_and_discount():
    # 0.8% premium -> NORMAL
    res1 = evaluate_qdii_order("513100", price=1.008, iopv=1.00)
    assert res1.status == "NORMAL"
    assert res1.allowed_weight_factor == 1.0
    assert res1.redirect_target is None

    # -1.5% discount -> DISCOUNT_OPPORTUNITY
    res2 = evaluate_qdii_order("513100", price=0.985, iopv=1.00)
    assert res2.status == "DISCOUNT_OPPORTUNITY"
    assert res2.allowed_weight_factor == 1.0


def test_qdii_guard_api_integration():
    g = get("qdii_premium")
    # Extreme bubble -> fails check
    res_fail = g.run(code="513100", price=2.00, iopv=1.50)
    assert not res_fail.passed
    assert any(f.severity == "error" for f in res_fail.findings)

    # Moderate premium -> passes with warning
    res_warn = g.run(code="513500", price=1.02, iopv=1.00)
    assert res_warn.passed
    assert any(f.severity == "warning" for f in res_warn.findings)

    # Safe trade -> passes with info
    res_pass = g.run(code="513100", price=1.01, iopv=1.00)
    assert res_pass.passed
    assert not any(f.severity in ("warning", "error") for f in res_pass.findings)
