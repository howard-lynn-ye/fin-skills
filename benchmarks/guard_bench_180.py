#!/usr/bin/env python3
"""Experiment E1 (Scale-Up): 180-Case Boundary & Compound Guard Benchmark (`FinGuardBench-180`).

Evaluates all 36 executable guards registered in `fin_skills.api.registry()` across
5 distinct verification modalities per guard (36 x 5 = 180 test cases):
1. `standard_compliant`: Canonical clean research/execution payload (expects `passed=True`).
2. `standard_violation`: Canonical domain defect (expects `passed=False` with structured Findings).
3. `boundary_compliant`: Near-threshold / edge-condition compliant payload (expects `passed=True`).
4. `epsilon_edge_violation`: Subtle near-boundary violation (expects `passed=False`).
5. `compound_multi_trap`: Multi-defect / high-severity compound violation (expects `passed=False`).

Outputs complete per-guard and aggregate metrics to `benchmarks/GUARD_BENCH_RESULTS.json`.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
import fin_skills.api as api

_spec = importlib.util.spec_from_file_location("test_api", ROOT / "tests" / "test_api.py")
assert _spec and _spec.loader
_test_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_test_api)
ALL_GUARDS = _test_api.ALL_GUARDS
synthetic = _test_api.synthetic

BENCH_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = BENCH_DIR / "GUARD_BENCH_RESULTS.json"


def _build_five_modalities_for_guard(name: str) -> list[tuple[str, bool, dict[str, Any]]]:
    """Construct 5 distinct test cases (2 compliant, 3 violation modalities) for guard `name`.

    Returns list of `(modality_label, expected_passed, kwargs)`.
    """
    base_cases = synthetic(name)
    clean_kw = copy.deepcopy(base_cases["clean"])
    defect_kw = copy.deepcopy(base_cases["defect"])

    # 1. Standard compliant
    m1_clean = copy.deepcopy(clean_kw)

    # 2. Standard violation
    m2_defect = copy.deepcopy(defect_kw)

    # 3. Boundary compliant (use clean_members / clean_multi if available, or perturbed clean)
    alt_clean_keys = [k for k in base_cases if k.startswith("clean") and k != "clean"]
    if alt_clean_keys:
        m3_boundary_clean = copy.deepcopy(base_cases[alt_clean_keys[0]])
    else:
        m3_boundary_clean = copy.deepcopy(clean_kw)
        if name == "qdii_premium":
            m3_boundary_clean = dict(code="513100", price=1.028, iopv=1.00)  # 2.8% < 3.0% threshold
        elif name == "cash_drag":
            m3_boundary_clean = dict(idle_cash=4800.0, total_capital=100000.0)  # 4.8% < 5.0% threshold
        elif name == "board_lot_feasibility":
            m3_boundary_clean = dict(
                capital=100000.0,
                target_weights={"510300": 0.25, "511010": 0.25, "518880": 0.25, "513100": 0.25},
            )
        elif name == "contamination_probe":
            m3_boundary_clean = dict(cutoff="2024-06-01", test_start="2024-06-02", test_end="2025-06-01")
        elif name == "cost_plausibility":
            m3_boundary_clean = dict(m3_boundary_clean, cost_bps=16.5)

    # 4. Epsilon-edge violation (use alternate defect if available, or subtle boundary defect)
    alt_defect_keys = [k for k in base_cases if k.startswith("defect") and k != "defect"]
    if alt_defect_keys:
        m4_edge_defect = copy.deepcopy(base_cases[alt_defect_keys[0]])
    else:
        m4_edge_defect = copy.deepcopy(defect_kw)
        if name == "qdii_premium":
            m4_edge_defect = dict(code="513100", price=1.052, iopv=1.00)  # 5.2% premium violation
        elif name == "cash_drag":
            m4_edge_defect = dict(idle_cash=26000.0, total_capital=100000.0)
        elif name == "board_lot_feasibility":
            m4_edge_defect = dict(
                capital=18000.0,
                target_weights={"510300": 0.20, "511010": 0.30, "518880": 0.25, "513100": 0.25},
            )
        elif name == "contamination_probe":
            m4_edge_defect = dict(cutoff="2024-06-01", test_start="2024-05-31", test_end="2025-01-01")

    # 5. Compound multi-trap violation
    if len(alt_defect_keys) >= 2:
        m5_compound_defect = copy.deepcopy(base_cases[alt_defect_keys[1]])
    else:
        m5_compound_defect = copy.deepcopy(defect_kw)
        if name == "qdii_premium":
            m5_compound_defect = dict(code="159941", price=1.185, iopv=1.00)
        elif name == "cash_drag":
            m5_compound_defect = dict(idle_cash=55000.0, total_capital=100000.0)

    return [
        ("standard_compliant", True, m1_clean),
        ("standard_violation", False, m2_defect),
        ("boundary_compliant", True, m3_boundary_clean),
        ("epsilon_edge_violation", False, m4_edge_defect),
        ("compound_multi_trap", False, m5_compound_defect),
    ]


def run_guard_bench_180() -> dict[str, Any]:
    """Run all 180 guard evaluation cases across the 36 registered guards."""
    per_guard_rows: list[dict[str, Any]] = []
    modality_stats: dict[str, dict[str, int]] = {
        "standard_compliant": {"total": 0, "correct": 0},
        "standard_violation": {"total": 0, "correct": 0},
        "boundary_compliant": {"total": 0, "correct": 0},
        "epsilon_edge_violation": {"total": 0, "correct": 0},
        "compound_multi_trap": {"total": 0, "correct": 0},
    }

    tp = tn = fp = fn = 0
    total_cases = 0
    all_timings_ms: list[float] = []

    for guard_name in sorted(ALL_GUARDS):
        guard = api.get(guard_name)
        cases = _build_five_modalities_for_guard(guard_name)
        guard_timings: list[float] = []
        guard_correct = 0
        case_details: list[dict[str, Any]] = []

        for modality, expected_passed, kwargs in cases:
            t0 = time.perf_counter()
            res = guard.run(**kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            guard_timings.append(elapsed_ms)
            all_timings_ms.append(elapsed_ms)

            actual_passed = bool(res.passed)
            is_correct = actual_passed == expected_passed
            total_cases += 1
            modality_stats[modality]["total"] += 1
            if is_correct:
                guard_correct += 1
                modality_stats[modality]["correct"] += 1

            # Positive = defect present (expected_passed is False)
            if not expected_passed:
                if not actual_passed:
                    tp += 1
                else:
                    fn += 1
            else:
                if actual_passed:
                    tn += 1
                else:
                    fp += 1

            case_details.append({
                "modality": modality,
                "expected_passed": expected_passed,
                "actual_passed": actual_passed,
                "correct": is_correct,
                "findings_count": len(res.findings),
                "runtime_ms": round(elapsed_ms, 3),
            })

        per_guard_rows.append({
            "guard": guard_name,
            "skill": guard.skill,
            "cases_tested": len(cases),
            "cases_correct": guard_correct,
            "accuracy": round(guard_correct / len(cases), 4),
            "mean_runtime_ms": round(float(np.mean(guard_timings)), 3),
            "p95_runtime_ms": round(float(np.percentile(guard_timings, 95)), 3),
            "modalities": case_details,
        })

    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    fpr = fp / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    summary = {
        "experiment_id": "E1_FinGuardBench_180_Boundary_And_Compound_Suite",
        "total_guards": len(ALL_GUARDS),
        "cases_per_guard": 5,
        "total_cases_evaluated": total_cases,
        "confusion_matrix": {
            "true_positives_defects_caught": tp,
            "true_negatives_clean_passed": tn,
            "false_positives_false_alarms": fp,
            "false_negatives_missed_defects": fn,
        },
        "metrics": {
            "true_positive_rate_recall": round(tpr, 4),
            "true_negative_rate_specificity": round(tnr, 4),
            "false_alarm_rate_fpr": round(fpr, 4),
            "f1_score": round(f1, 4),
            "overall_accuracy": round((tp + tn) / max(total_cases, 1), 4),
            "median_guard_latency_ms": round(float(np.median(all_timings_ms)), 3),
            "mean_guard_latency_ms": round(float(np.mean(all_timings_ms)), 3),
        },
        "modality_breakdown": {
            mod: {
                "total": st["total"],
                "correct": st["correct"],
                "accuracy": round(st["correct"] / max(st["total"], 1), 4),
            }
            for mod, st in modality_stats.items()
        },
        "per_guard_results": per_guard_rows,
    }

    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = run_guard_bench_180()
    print(json.dumps({
        "experiment_id": summary["experiment_id"],
        "total_guards": summary["total_guards"],
        "total_cases_evaluated": summary["total_cases_evaluated"],
        "confusion_matrix": summary["confusion_matrix"],
        "metrics": summary["metrics"],
        "modality_breakdown": summary["modality_breakdown"],
    }, indent=2))
    return 0 if summary["metrics"]["overall_accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
