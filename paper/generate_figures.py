#!/usr/bin/env python3
"""Generate publication-grade vector figures (PDF + 300 DPI PNG) for the NAACL 2026 Short Paper.

Reads directly from:
- `benchmarks/AGENT_STUDY_RESULTS.json`
- `benchmarks/PARITY_AND_COST_RESULTS.json`
- `benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json`

Outputs:
1. `paper/latex_naacl/figs/fig_overview_compliance_and_kol.{pdf,png}`
2. `paper/latex_naacl/figs/fig_guard_scaling_and_clock_replay.{pdf,png}`
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks"
FIGS_DIR = ROOT / "paper" / "latex_naacl" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_figure_1() -> None:
    """Figure 1: FinGuardBench-60 Ablation + Multi-Round Self-Repair (Left) & KOL Paradox (Right)."""
    e2 = json.loads((BENCH_DIR / "AGENT_STUDY_RESULTS.json").read_text(encoding="utf-8"))
    e4 = json.loads((BENCH_DIR / "REAL_WORLD_KOL_AUDIT_RESULTS.json").read_text(encoding="utf-8"))

    cm = e2["condition_metrics"]
    cond_a = cm["Condition_A_No_Library"]
    cond_b = cm["Condition_B_Skills_Text_Only"]
    cond_c = cm["Condition_C_Skills_Plus_Executable_Guards"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.28}
    )

    # Panel (a): FinGuardBench-60 Compliance Rate (%) & Sharpe Inflation Gap
    labels = [
        "Cond A\n(No Lib)",
        "Cond B\n(SKILL.md)",
        "Cond C\n(Pass@1)",
        "Cond C\n(Pass@2)",
        "Cond C\n(Pass@3)",
    ]
    rates = np.array([
        cond_a["compliance_rate"] * 100.0,
        cond_b["compliance_rate"] * 100.0,
        cond_c["pass_at_1_rate"] * 100.0,
        cond_c["pass_at_2_rate"] * 100.0,
        cond_c["pass_at_3_rate"] * 100.0,
    ])
    ci_lo = np.array([
        cond_a["compliance_95_ci"][0] * 100.0,
        cond_b["compliance_95_ci"][0] * 100.0,
        cond_c["pass_at_1_95_ci"][0] * 100.0,
        cond_c["pass_at_2_95_ci"][0] * 100.0,
        cond_c["pass_at_3_95_ci"][0] * 100.0,
    ])
    ci_hi = np.array([
        cond_a["compliance_95_ci"][1] * 100.0,
        cond_b["compliance_95_ci"][1] * 100.0,
        cond_c["pass_at_1_95_ci"][1] * 100.0,
        cond_c["pass_at_2_95_ci"][1] * 100.0,
        cond_c["pass_at_3_95_ci"][1] * 100.0,
    ])
    yerr = np.vstack([rates - ci_lo, ci_hi - rates])

    colors = ["#94a3b8", "#3b82f6", "#0ea5e9", "#10b981", "#059669"]
    x = np.arange(len(labels))
    bars = ax1.bar(x, rates, width=0.58, color=colors, edgecolor="#1e293b", linewidth=0.6,
                   yerr=yerr, capsize=3.0, error_kw={"elinewidth": 0.9, "ecolor": "#1e293b"})

    for bar, val, hi in zip(bars, rates, ci_hi):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            hi + 2.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#0f172a",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 116)
    ax1.set_ylabel("Methodological Compliance (%)")
    ax1.set_title("(a) FinGuardBench-60 & Self-Repair")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (b): Bilingual KOL Tier 5-Day IC vs Mean Followers (k)
    tier_order = [
        ("TIER_1_CORE_ALPHA", "Tier-1 Alpha\n(n=70)"),
        ("TIER_2_SOLID_RESEARCHER", "Tier-2 Solid\n(n=115)"),
        ("TIER_0_ELITE_KOL", "Tier-0 Elite\n(n=31)"),
        ("TIER_NEUTRAL_RETAIL", "Neutral\n(n=1120)"),
        ("TIER_CONTRARIAN_INDICATOR", "Contrarian\n(n=98)"),
    ]
    tier_lookup = {t["tier"]: t for t in e4["recomputed_tier_breakdown"]}
    t_names = [lbl for _, lbl in tier_order]
    ic_vals = [tier_lookup[k]["mean_ic_5d"] for k, _ in tier_order]
    fan_vals_k = [tier_lookup[k]["mean_fans"] / 1000.0 for k, _ in tier_order]

    x2 = np.arange(len(t_names))
    bar_colors = ["#059669" if v > 0.05 else ("#dc2626" if v < -0.05 else "#64748b") for v in ic_vals]
    b2 = ax2.bar(x2, ic_vals, width=0.52, color=bar_colors, edgecolor="#1e293b", linewidth=0.6)
    ax2.axhline(0.0, color="#334155", linewidth=0.8)

    for bar, ic_v, f_k in zip(b2, ic_vals, fan_vals_k):
        offset = 0.014 if ic_v >= 0 else -0.014
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            ic_v + offset,
            f"{ic_v:+.2f}\n({f_k:.0f}k)",
            ha="center",
            va="bottom" if ic_v >= 0 else "top",
            fontsize=6.7,
            fontweight="bold",
            color="#0f172a",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(t_names, fontsize=7.0)
    ax2.set_ylim(-0.34, 0.31)
    ax2.set_ylabel("5-Day Rank IC (Mean Fans)")
    ax2.set_title("(b) KOL Follower Paradox (3.96x)")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"fig_overview_compliance_and_kol.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_figure_2() -> None:
    """Figure 2: 36-Guard Runtime Scaling (Left) & Transaction Cost / Horizon Sensitivity (Right)."""
    e3 = json.loads((BENCH_DIR / "PARITY_AND_COST_RESULTS.json").read_text(encoding="utf-8"))
    e4 = json.loads((BENCH_DIR / "REAL_WORLD_KOL_AUDIT_RESULTS.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.50), gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.28}
    )

    # Panel (a): Guard Scaling Curve (N = 250 to 100,000 rows)
    sc = e3["guard_runtime_scaling"]["scaling_curve"]
    n_rows = [r["n_rows"] for r in sc]
    p50_ms = [r["p50_ms"] for r in sc]
    p95_ms = [r["p95_ms"] for r in sc]
    alpha_exp = e3["guard_runtime_scaling"]["empirical_complexity_exponent_alpha"]

    ax1.plot(n_rows, p50_ms, marker="o", linewidth=1.8, color="#0284c7", label=f"Median (p50, alpha={alpha_exp:.2f})")
    ax1.plot(n_rows, p95_ms, marker="s", linewidth=1.3, linestyle="--", color="#0f172a", label="95th Percentile (p95)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset Rows N (log scale)")
    ax1.set_ylabel("Bundle.check() Latency (ms)")
    ax1.set_title("(a) Sub-Linear Guard Scaling")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)

    # Panel (b): Transaction Cost Sensitivity (5, 10, 20, 30 bps) & Horizon Sharpe
    cs = e4["transaction_cost_sensitivity_matrix"]
    bps = [r["round_trip_cost_bps"] for r in cs]
    sr_guarded = [r["guarded_core_satellite_sharpe"] for r in cs]
    sr_naive = [r["naive_unfiltered_sharpe"] for r in cs]

    x_idx = np.arange(len(bps))
    width = 0.34
    b_g = ax2.bar(x_idx - width / 2, sr_guarded, width=width, color="#059669", edgecolor="#1e293b",
                  linewidth=0.6, label="Guarded Core-Satellite (T+5)")
    b_n = ax2.bar(x_idx + width / 2, sr_naive, width=width, color="#e11d48", edgecolor="#1e293b",
                  linewidth=0.6, label="Naive Unfiltered Social")

    for bar, v in zip(b_g, sr_guarded):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=6.8, fontweight="bold", color="#065f46")
    for bar, v in zip(b_n, sr_naive):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=6.8, fontweight="bold", color="#9f1239")

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([f"{int(b)} bps" for b in bps])
    ax2.set_ylim(0, 2.68)
    ax2.set_xlabel("Round-Trip Transaction Cost Assumption")
    ax2.set_ylabel("OOS Net Sharpe (2023-2026)")
    ax2.set_title("(b) Cost Sensitivity: Guarded vs. Naive")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(pad=0.7)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"fig_guard_scaling_and_clock_replay.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    make_figure_1()
    make_figure_2()
    print(f"Successfully generated Figure 1 and Figure 2 in {FIGS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
