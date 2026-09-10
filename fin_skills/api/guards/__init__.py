"""One module per guard. Importing this package registers every guard.

Module -> guard name -> owning skill -> wrapped function(s): see `fin_skills.api.registry()`
and `Guard.wraps`; `python -c "import fin_skills.api as a; [print(g().describe()) for g in a.registry()]"`
prints the whole table.
"""
from __future__ import annotations

from fin_skills.api.guards import (  # noqa: F401  (import for the registration side effect)
    adjustment_check,
    ashare_rules,
    assert_causal,
    brinson_attribution,
    contamination_probe,
    continuous_contract,
    cost_curve,
    cost_plausibility,
    fold_leak_test,
    fx_conventions,
    greeks_convention,
    join_asof_sortedness,
    leveraged_reset,
    npv_zero,
    paper_account_guard,
    pit_fundamentals,
    pit_universe,
    purge_effect,
    reconcile_sources,
    regime_coverage,
    regime_lookahead,
    research_audit,
    result_manifest,
    rf_convention,
    safe_asof,
    spa_test,
    survivorship_audit,
    trial_ledger,
    warmup_probe,
    weight_traps,
)

__all__ = [
    "adjustment_check", "ashare_rules", "assert_causal", "brinson_attribution",
    "contamination_probe", "continuous_contract", "cost_curve", "cost_plausibility",
    "fold_leak_test",
    "fx_conventions", "greeks_convention", "join_asof_sortedness", "leveraged_reset",
    "npv_zero", "paper_account_guard", "pit_fundamentals", "pit_universe", "purge_effect",
    "reconcile_sources", "regime_coverage", "regime_lookahead", "research_audit",
    "result_manifest",
    "rf_convention", "safe_asof", "spa_test", "survivorship_audit", "trial_ledger",
    "warmup_probe", "weight_traps",
]
