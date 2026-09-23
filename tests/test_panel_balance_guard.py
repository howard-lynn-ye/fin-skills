"""Unit tests for the 2D (Company x Year) PanelBalanceGuard (`check_panel_balance`)."""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.api import check_panel_balance, get
from fin_skills.core.panel_balance import audit_panel_balance, rebalance_company_year_panel


def test_panel_balance_flags_megacap_and_recency_skew():
    rows = []
    # 1 mega-cap with 400 rows in 2025, 9 mid-caps with 5 rows each across 2018-2024
    for i in range(400):
        rows.append({"symbol": "SH600000", "date": f"2025-06-{(i % 28) + 1:02d}", "title": "Mega-cap news"})
    for s_idx in range(1, 10):
        for yr in range(2018, 2025):
            rows.append({"symbol": f"SZ30000{s_idx}", "date": f"{yr}-05-15", "title": "Mid-cap news"})

    df = pd.DataFrame(rows)
    res = check_panel_balance(df, entity_col="symbol", time_col="date", max_company_gini=0.35, max_year_gini=0.25)
    assert not res.passed
    assert "IMBALANCED PANEL" in res.evidence["verdict"]
    assert res.evidence["company_gini"] > 0.35
    assert res.evidence["year_gini"] > 0.25


def test_rebalance_company_year_panel_restores_balance():
    rows = []
    for s_idx in range(1, 11):
        sym = f"SH60000{s_idx}"
        for yr in range(2018, 2026):
            # Give symbol 1 and year 2025 120 rows, others 12 rows
            n_items = 120 if (s_idx == 1 or yr == 2025) else 12
            for k in range(n_items):
                rows.append({"symbol": sym, "date": f"{yr}-03-{(k % 28) + 1:02d}", "title": "Disclosure"})

    raw_df = pd.DataFrame(rows)
    raw_res = check_panel_balance(raw_df, max_company_gini=0.35, max_year_gini=0.25)
    assert not raw_res.passed

    balanced_df = rebalance_company_year_panel(raw_df, entity_col="symbol", time_col="date", max_per_cell=12)
    bal_res = check_panel_balance(balanced_df, max_company_gini=0.35, max_year_gini=0.25)
    assert bal_res.passed
    assert bal_res.evidence["company_gini"] == pytest.approx(0.0, abs=1e-4)
    assert bal_res.evidence["year_gini"] == pytest.approx(0.0, abs=1e-4)
    assert "normalized_year_balanced_weight" in balanced_df.columns
    assert get("panel_balance").name == "panel_balance"
