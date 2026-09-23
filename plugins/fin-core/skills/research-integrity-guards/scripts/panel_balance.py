#!/usr/bin/env python3
"""Audit and repair 2D (Company x Year) panel imbalance in alternative financial datasets.

WHY: alternative data crawlers (corporate announcements, social media streams, analyst
notes) naturally suffer from two compounding sampling biases:
  1. Cross-Sectional Mega-Cap Dominance: popular tickers or alphabetically early symbols
     accumulate 10x-20x more records than mid/small-cap or newly listed boards (e.g.
     SH600xxx vs. SH688xxx/SZ300xxx), causing models to overfit retail megaphone chatter.
  2. Temporal Recency Inflation: reverse-chronological pagination fills quotas with
     recent years (e.g. 2025-2026) while leaving historical regimes (2018-2022) sparse.

Nothing raises a Python exception, yet cross-sectional Rank IC turns negative out-of-sample
because the model learns density artifacts rather than fundamentals.

Usage:
    from fin_skills.core.panel_balance import audit_panel_balance, rebalance_company_year_panel

    audit = audit_panel_balance(panel_df, entity_col="symbol", time_col="date")
    if not audit.passed:
        balanced_df = rebalance_company_year_panel(panel_df, entity_col="symbol", time_col="date", max_per_cell=36)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


def compute_gini(values: np.ndarray | list[float]) -> float:
    """Compute the Gini coefficient over non-negative counts (0 = uniform, 1 = monopoly)."""
    arr = np.asarray([v for v in values if v >= 0], dtype=np.float64)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


@dataclass
class PanelBalanceAudit:
    verdict: str
    passed: bool
    n_rows: int
    n_entities: int
    n_years: int
    company_gini: float
    year_gini: float
    max_to_median_company_ratio: float
    max_to_median_year_ratio: float
    year_counts: dict[str, int] = field(default_factory=dict)
    top_entities: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            f"PanelBalanceAudit: {self.verdict} (passed={self.passed})",
            f"  rows={self.n_rows}, entities={self.n_entities}, years={self.n_years}",
            f"  company_gini={self.company_gini:.4f} (max/median={self.max_to_median_company_ratio:.2f}x)",
            f"  year_gini={self.year_gini:.4f} (max/median={self.max_to_median_year_ratio:.2f}x)",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def audit_panel_balance(
    panel: pd.DataFrame,
    entity_col: str = "symbol",
    time_col: str = "date",
    max_company_gini: float = 0.35,
    max_year_gini: float = 0.25,
    max_company_ratio: float = 5.0,
) -> PanelBalanceAudit:
    """Audit 2D (Company x Year) concentration in an event or panel DataFrame."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if entity_col not in panel.columns:
        raise TypeError(f"entity_col {entity_col!r} not found in panel columns")
    if time_col not in panel.columns:
        raise TypeError(f"time_col {time_col!r} not found in panel columns")
    if panel.empty:
        return PanelBalanceAudit(
            verdict="EMPTY PANEL",
            passed=False,
            n_rows=0,
            n_entities=0,
            n_years=0,
            company_gini=0.0,
            year_gini=0.0,
            max_to_median_company_ratio=0.0,
            max_to_median_year_ratio=0.0,
            notes=["Panel contains 0 rows."],
        )

    entities = panel[entity_col].astype(str)
    years = panel[time_col].astype(str).str.slice(0, 4)

    comp_vc = entities.value_counts()
    yr_vc = years.value_counts()

    comp_vals = comp_vc.to_numpy(dtype=np.float64)
    yr_vals = yr_vc.to_numpy(dtype=np.float64)

    c_gini = compute_gini(comp_vals)
    y_gini = compute_gini(yr_vals)
    c_ratio = float(np.max(comp_vals) / max(1.0, float(np.median(comp_vals))))
    y_ratio = float(np.max(yr_vals) / max(1.0, float(np.median(yr_vals))))

    notes: list[str] = []
    failed_reasons: list[str] = []

    if c_gini > max_company_gini or c_ratio > max_company_ratio:
        failed_reasons.append(
            f"cross-sectional mega-cap skew (company_gini={c_gini:.4f} > {max_company_gini:.2f}, "
            f"max/median={c_ratio:.2f}x > {max_company_ratio:.1f}x)"
        )
        notes.append(
            "Apply stratified (entity, year) cell capping via rebalance_company_year_panel(df, max_per_cell=36)."
        )

    if y_gini > max_year_gini:
        failed_reasons.append(
            f"temporal recency skew (year_gini={y_gini:.4f} > {max_year_gini:.2f}, "
            f"max/median={y_ratio:.2f}x)"
        )
        notes.append(
            "Apply inverse cell-density weighting (normalized_year_balanced_weight) and deficit-first historical backfill."
        )

    passed = len(failed_reasons) == 0
    verdict = "BALANCED 2D PANEL" if passed else "IMBALANCED PANEL: " + "; ".join(failed_reasons)

    return PanelBalanceAudit(
        verdict=verdict,
        passed=passed,
        n_rows=int(len(panel)),
        n_entities=int(len(comp_vc)),
        n_years=int(len(yr_vc)),
        company_gini=round(c_gini, 4),
        year_gini=round(y_gini, 4),
        max_to_median_company_ratio=round(c_ratio, 2),
        max_to_median_year_ratio=round(y_ratio, 2),
        year_counts={str(k): int(v) for k, v in sorted(yr_vc.items())},
        top_entities={str(k): int(v) for k, v in comp_vc.head(5).items()},
        notes=notes,
    )


def rebalance_company_year_panel(
    panel: pd.DataFrame,
    entity_col: str = "symbol",
    time_col: str = "date",
    max_per_cell: int = 36,
) -> pd.DataFrame:
    """Stratify a panel by (entity, year), cap each cell at `max_per_cell` via uniform
    chronological spacing, and attach `normalized_year_balanced_weight`.
    """
    if panel.empty:
        return panel.copy()
    work = panel.copy()
    work["_year"] = work[time_col].astype(str).str.slice(0, 4)
    chunks = []
    for (_, _), grp in work.groupby([entity_col, "_year"], sort=False):
        raw_cnt = len(grp)
        if raw_cnt > max_per_cell:
            grp_sorted = grp.sort_values(time_col)
            idx = np.linspace(0, raw_cnt - 1, max_per_cell, dtype=int)
            sub = grp_sorted.iloc[idx].copy()
        else:
            sub = grp.copy()
        sub["cell_count_raw"] = raw_cnt
        sub["cell_count_balanced"] = len(sub)
        sub["sample_weight"] = float(max_per_cell) / max(1.0, float(len(sub)))
        chunks.append(sub)
    out = pd.concat(chunks, ignore_index=True)
    yr_sums = out.groupby("_year")["sample_weight"].transform("sum")
    out["normalized_year_balanced_weight"] = out["sample_weight"] / yr_sums.clip(lower=1e-8)
    return out.drop(columns=["_year"])
