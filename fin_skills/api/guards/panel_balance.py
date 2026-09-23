"""Guard: 2D (Company x Year) panel balance in alternative data panels (research-integrity-guards / panel_balance.py)."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register
from fin_skills.core.panel_balance import audit_panel_balance, rebalance_company_year_panel


@register
class PanelBalanceGuard(Guard):
    """Audit a research panel for 2D (Company x Year) concentration skew (mega-cap dominance & recency inflation).

    Inputs
        panel            : DataFrame of event/alternative data records containing entity and timestamp columns.
        entity_col       : column identifying the company/ticker (default 'symbol').
        time_col         : column identifying the observation date/timestamp (default 'date').
        max_company_gini : upper bound on cross-sectional entity Gini coefficient (default 0.35).
        max_year_gini    : upper bound on temporal annual Gini coefficient (default 0.25).
        max_company_ratio: upper bound on max-to-median entity record ratio (default 5.0).
    """

    name = "panel_balance"
    skill = "research-integrity-guards"
    summary = "Flags 2D (Company x Year) concentration skew where mega-caps or recent years monopolize training/evaluation density."
    wraps = (
        "fin_skills.core.panel_balance.audit_panel_balance",
        "fin_skills.core.panel_balance.rebalance_company_year_panel",
    )
    required = ("panel",)
    optional = ("entity_col", "time_col", "max_company_gini", "max_year_gini", "max_company_ratio")

    def check(
        self,
        panel: pd.DataFrame,
        entity_col: str = "symbol",
        time_col: str = "date",
        max_company_gini: float = 0.35,
        max_year_gini: float = 0.25,
        max_company_ratio: float = 5.0,
    ) -> Outcome:
        out = Outcome()
        audit = audit_panel_balance(
            panel=panel,
            entity_col=entity_col,
            time_col=time_col,
            max_company_gini=max_company_gini,
            max_year_gini=max_year_gini,
            max_company_ratio=max_company_ratio,
        )
        out.note(
            verdict=audit.verdict,
            passed=audit.passed,
            n_rows=audit.n_rows,
            n_entities=audit.n_entities,
            n_years=audit.n_years,
            company_gini=audit.company_gini,
            year_gini=audit.year_gini,
            max_to_median_company_ratio=audit.max_to_median_company_ratio,
            max_to_median_year_ratio=audit.max_to_median_year_ratio,
            year_counts=audit.year_counts,
            top_entities=audit.top_entities,
            report=audit.report(),
        )
        if not audit.passed:
            out.error(audit.verdict, where="panel_balance")
            for n in audit.notes:
                out.warning(n, where="remediation")
        else:
            out.info(audit.verdict, where="panel_balance")
        return out


def check_panel_balance(
    panel: pd.DataFrame,
    entity_col: str = "symbol",
    time_col: str = "date",
    max_company_gini: float = 0.35,
    max_year_gini: float = 0.25,
    max_company_ratio: float = 5.0,
) -> GuardResult:
    """Run the `panel_balance` executable guard directly on a DataFrame."""
    return get("panel_balance").run(
        panel=panel,
        entity_col=entity_col,
        time_col=time_col,
        max_company_gini=max_company_gini,
        max_year_gini=max_year_gini,
        max_company_ratio=max_company_ratio,
    )
