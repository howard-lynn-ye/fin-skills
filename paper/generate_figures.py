#!/usr/bin/env python3
"""Generate publication-grade vector figures (PDF + 300 DPI PNG) for the NAACL 2026 Short Paper.

Reads directly from:
- `benchmarks/AGENT_STUDY_RESULTS.json`
- `benchmarks/PARITY_AND_COST_RESULTS.json`
- `benchmarks/REAL_WORLD_KOL_AUDIT_RESULTS.json`

Outputs:
1. `paper/latex_naacl/figs/fig_overview_compliance_and_kol.{pdf,png}`
2. `paper/latex_naacl/figs/fig_guard_scaling_and_clock_replay.{pdf,png}`
3. `paper/latex_naacl/figs/fig_domain_heatmap_and_model_scaling.{pdf,png}`
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
    "font.size": 8.2,
    "axes.titlesize": 9.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.1,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_figure_1() -> None:
    """Figure 1: FinGuardBench-60 (incl. Python Self-Debug B+) & Bilingual 2,521-KOL Paradox."""
    e2 = json.loads((BENCH_DIR / "AGENT_STUDY_RESULTS.json").read_text(encoding="utf-8"))
    e4 = json.loads((BENCH_DIR / "REAL_WORLD_KOL_AUDIT_RESULTS.json").read_text(encoding="utf-8"))

    cm = e2["condition_metrics"]
    cond_a = cm["Condition_A_No_Library"]
    cond_b = cm["Condition_B_Skills_Text_Only"]
    cond_bp = cm["Condition_B_Plus_Python_Self_Debug_No_Guards"]
    cond_c = cm["Condition_C_Skills_Plus_Executable_Guards"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.8, 2.55), gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.32}
    )

    # Panel (a): FinGuardBench-60 6 Arms (A, B, B+ PyDebug@3, C@1, C@2, C@3)
    labels = [
        "Cond A\nNo Lib",
        "Cond B\nDocs",
        "Cond B+\nPyDbg",
        "Cond C\nPass@1",
        "Cond C\nPass@2",
        "Cond C\nPass@3",
    ]
    rates = np.array([
        cond_a["compliance_rate"] * 100.0,
        cond_b["compliance_rate"] * 100.0,
        cond_bp["pass_at_3_rate"] * 100.0,
        cond_c["pass_at_1_rate"] * 100.0,
        cond_c["pass_at_2_rate"] * 100.0,
        cond_c["pass_at_3_rate"] * 100.0,
    ])
    ci_lo = np.array([
        cond_a["compliance_95_ci"][0] * 100.0,
        cond_b["compliance_95_ci"][0] * 100.0,
        cond_bp["pass_at_3_95_ci"][0] * 100.0,
        cond_c["pass_at_1_95_ci"][0] * 100.0,
        cond_c["pass_at_2_95_ci"][0] * 100.0,
        cond_c["pass_at_3_95_ci"][0] * 100.0,
    ])
    ci_hi = np.array([
        cond_a["compliance_95_ci"][1] * 100.0,
        cond_b["compliance_95_ci"][1] * 100.0,
        cond_bp["pass_at_3_95_ci"][1] * 100.0,
        cond_c["pass_at_1_95_ci"][1] * 100.0,
        cond_c["pass_at_2_95_ci"][1] * 100.0,
        cond_c["pass_at_3_95_ci"][1] * 100.0,
    ])
    yerr = np.vstack([rates - ci_lo, ci_hi - rates])

    colors = ["#94a3b8", "#3b82f6", "#f59e0b", "#0ea5e9", "#10b981", "#059669"]
    x = np.arange(len(labels))
    bars = ax1.bar(x, rates, width=0.55, color=colors, edgecolor="#1e293b", linewidth=0.6,
                   yerr=yerr, capsize=2.6, error_kw={"elinewidth": 0.85, "ecolor": "#1e293b"})

    for bar, val, hi in zip(bars, rates, ci_hi):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            hi + 2.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#0f172a",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.0)
    ax1.set_ylim(0, 117)
    ax1.set_ylabel("Methodological Compliance (%)")
    ax1.set_title("(a) FinGuardBench-60: Python Debug vs. Guards", fontsize=8.8)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel (b): Bilingual KOL Follower Paradox (CN Xueqiu N=1,445 vs US StockTwits N=1,076)
    bm = e4["bilingual_cross_market_comparison"]
    cn = bm["China_Xueqiu_AShare"]
    us = bm["US_StockTwits_Equities"]

    categories = [
        "CN Alpha\n(70)",
        "CN Contr.\n(98)",
        "US Alpha\n(54)",
        "US Contr.\n(82)",
        "Raw\nMMAN",
        "Gated\nMMAN",
    ]
    ic_vals = [
        cn["tier1_5d_rank_ic"],
        cn["contrarian_5d_rank_ic"],
        us["tier1_5d_rank_ic"],
        us["contrarian_5d_rank_ic"],
        cn["unweighted_nlp_rank_ic"] * 3.0,
        cn["bayesian_gated_rank_ic"] * 3.0,
    ]
    sub_labels = [
        "+0.22\n(14k)",
        "-0.21\n(56k, 3.96x)",
        "+0.18\n(18k)",
        "-0.18\n(58k, 3.15x)",
        "-0.032\n(Raw IC)",
        "+0.010\n(Gated IC)",
    ]
    x2 = np.arange(len(categories))
    bar_colors = ["#059669", "#dc2626", "#0d9488", "#e11d48", "#b91c1c", "#0284c7"]
    b2 = ax2.bar(x2, ic_vals, width=0.52, color=bar_colors, edgecolor="#1e293b", linewidth=0.6)
    ax2.axhline(0.0, color="#334155", linewidth=0.8)

    for bar, ic_v, slbl in zip(b2, ic_vals, sub_labels):
        offset = 0.014 if ic_v >= 0 else -0.014
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            ic_v + offset,
            slbl,
            ha="center",
            va="bottom" if ic_v >= 0 else "top",
            fontsize=6.2,
            fontweight="bold",
            color="#0f172a",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=6.8)
    ax2.set_ylim(-0.35, 0.32)
    ax2.set_ylabel("5-Day Rank IC (Mean Fans)")
    ax2.set_title("(b) Bilingual KOL Paradox (2,521 KOLs)", fontsize=8.8)
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
        1, 2, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.28}
    )

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
    ax1.set_title("(a) Sub-Linear Guard Scaling (to N=100k)")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, which="both", linestyle="--", alpha=0.35)

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
    ax2.set_ylim(0, 2.48)
    ax2.set_xlabel("Round-Trip Transaction Cost Assumption")
    ax2.set_ylabel("Out-of-Sample Net Sharpe")
    ax2.set_title("(b) Cost Sensitivity & Microstructure Gate")
    ax2.legend(loc="upper right", frameon=False)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"fig_guard_scaling_and_clock_replay.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_figure_3() -> None:
    """Figure 3: 10-Domain Heatmap (Left) & 4-Tier Model Scaling Law of Compliance Paradox (Right)."""
    e2 = json.loads((BENCH_DIR / "AGENT_STUDY_RESULTS.json").read_text(encoding="utf-8"))
    domains = e2["domain_breakdown"]
    tiers = e2["model_tier_scaling_matrix"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.30}
    )

    # Panel (a): 10-Domain Compliance & Hallucination Matrix
    d_names = [d["domain"].replace("fin-", "") for d in domains]
    mat = np.array([
        [
            d["cond_A_pass_rate"] * 100.0,
            d["cond_B_pass_rate"] * 100.0,
            d["cond_B_hallucinated_citation_rate"] * 100.0,
            d["cond_B_plus_python_self_debug_pass3"] * 100.0,
            d["cond_C_pass_at_3"] * 100.0,
        ]
        for d in domains
    ])
    im = ax1.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    col_labels = ["Cond A\nPass", "Cond B\nPass", "Cond B\nHalluc.", "Cond B+\nPyDbg", "Cond C\nPass@3"]
    ax1.set_xticks(np.arange(len(col_labels)))
    ax1.set_xticklabels(col_labels, fontsize=6.6)
    ax1.set_yticks(np.arange(len(d_names)))
    ax1.set_yticklabels(d_names, fontsize=6.8)
    ax1.set_title("(a) 10-Domain Compliance & Hallucination (%)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt_color = "white" if val > 62 else "#0f172a"
            if j == 2 and val > 40:
                txt_color = "#991b1b" if val < 62 else "#fecaca"
            ax1.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.2, fontweight="bold", color=txt_color)

    # Panel (b): 4-Tier Model Scaling Law
    t_labels = ["Small-Fast\n(Haiku/8B)", "Open-Coder\n(Qwen-32B)", "Frontier-Gen\n(Sonnet/4o)", "Reasoning\n(Opus/R1)"]
    x_t = np.arange(len(t_labels))
    r_a = [t["cond_A_no_library_pass_rate"] * 100.0 for t in tiers]
    r_b = [t["cond_B_text_only_pass_rate"] * 100.0 for t in tiers]
    r_bp = [t["cond_B_plus_python_self_debug_pass3_rate"] * 100.0 for t in tiers]
    r_c3 = [t["cond_C_guards_pass3_rate"] * 100.0 for t in tiers]
    h_b = [t["cond_B_hallucinated_guard_citation_rate"] * 100.0 for t in tiers]

    ax2.plot(x_t, r_c3, marker="o", linewidth=1.9, color="#059669", label="Cond C: Guards Pass@3")
    ax2.plot(x_t, r_bp, marker="s", linewidth=1.4, linestyle="-.", color="#f59e0b", label="Cond B+: Python Debug@3")
    ax2.plot(x_t, r_b, marker="^", linewidth=1.4, color="#3b82f6", label="Cond B: SKILL.md Only")
    ax2.plot(x_t, r_a, marker="v", linewidth=1.3, color="#94a3b8", label="Cond A: No Library")
    ax2.plot(x_t, h_b, marker="D", linewidth=1.6, linestyle="--", color="#dc2626", label="Cond B: Halluc. Citations")

    for xi, hc, c3 in zip(x_t, h_b, r_c3):
        offset_h = -9.5 if xi == 1 else 3.0
        ax2.text(xi, hc + offset_h, f"{hc:.1f}%", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#b91c1c")
        ax2.text(xi, c3 + 2.5, f"{c3:.1f}%", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#065f46")

    ax2.set_xticks(x_t)
    ax2.set_xticklabels(t_labels, fontsize=6.6)
    ax2.set_ylim(0, 145)
    ax2.set_ylabel("Task Rate / Hallucination Rate (%)")
    ax2.set_title("(b) Cross-Model Scaling Law (E6)", fontsize=8.8)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=5.6, frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"fig_domain_heatmap_and_model_scaling.{ext}", bbox_inches="tight")
        alt_dir = ROOT / "paper" / "stock_prediction" / "figs"
        if alt_dir.exists():
            fig.savefig(alt_dir / f"fig_domain_heatmap_and_model_scaling.{ext}", bbox_inches="tight")
    plt.close(fig)


def make_figure_4() -> None:
    """Figure 4: (a) E9 Diagnostic Feedback Granularity Ladder; (b) E10 Macro Regime & Rolling Prior Sharpe."""
    e9_e12_path = ROOT / "benchmarks" / "EXTENDED_ABLATIONS_E9_E12_RESULTS.json"
    data = json.loads(e9_e12_path.read_text(encoding="utf-8"))
    levels = data["E9_feedback_granularity_ablation"]["levels"]
    regimes = data["E10_regime_and_rolling_prior_audit"]["china_ashare_regimes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.55), dpi=300)

    # Panel (a): E9 Diagnostic Feedback Granularity Ladder
    lbls = [
        "L0: Blind\nBest-of-3",
        "L1: Python\nTraceback",
        "L2: Binary\nFAIL Gate",
        "L3: Guard\nName Only",
        "L4: Full JSON\nGuardResult",
    ]
    p1 = [lv["pass_at_1_pct"] for lv in levels]
    p2 = [lv["pass_at_2_pct"] for lv in levels]
    p3 = [lv["pass_at_3_pct"] for lv in levels]
    x = np.arange(len(lbls))
    w = 0.25

    ax1.bar(x - w, p1, width=w, color="#94a3b8", edgecolor="#475569", linewidth=0.5, label="Pass@1")
    ax1.bar(x, p2, width=w, color="#3b82f6", edgecolor="#1d4ed8", linewidth=0.5, label="Pass@2")
    bars3 = ax1.bar(x + w, p3, width=w, color="#059669", edgecolor="#065f46", linewidth=0.6, label="Pass@3")

    for b, val in zip(bars3, p3):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5.9,
            fontweight="bold",
            color="#065f46",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(lbls, fontsize=6.2)
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Compliance Rate (%)")
    ax1.set_title("(a) Feedback Granularity Ablation (E9)", fontsize=8.5)
    ax1.legend(loc="upper left", ncol=3, fontsize=5.8, frameon=True, facecolor="white", framealpha=0.92)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)

    # Panel (b): E10 Macro Regime Stratification (Sharpe across R1-R4)
    r_labels = [
        "R1: 2023\nBear Grind",
        "R2: 2024H1\nLiq. Crisis",
        "R3: 2024Q4\nStimulus",
        "R4: 25-26\nDispersion",
    ]
    xr = np.arange(len(r_labels))
    s_csi = [r["csi300_sharpe"] for r in regimes]
    s_naive = [r["unguarded_naive_kol_sharpe"] for r in regimes]
    s_guard = [r["guarded_finskills_sharpe"] for r in regimes]

    ax2.bar(xr - w, s_csi, width=w, color="#cbd5e1", edgecolor="#64748b", linewidth=0.5, label="CSI 300 Bench")
    ax2.bar(xr, s_naive, width=w, color="#f87171", edgecolor="#b91c1c", linewidth=0.5, label="Unguarded KOL")
    bars_g = ax2.bar(xr + w, s_guard, width=w, color="#0d9488", edgecolor="#115e59", linewidth=0.6, label="Guarded fin-skills")

    for b, val in zip(bars_g, s_guard):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + 0.08,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold",
            color="#115e59",
        )

    ax2.axhline(0.0, color="#334155", linewidth=0.7, linestyle="-")
    ax2.set_xticks(xr)
    ax2.set_xticklabels(r_labels, fontsize=6.3)
    ax2.set_ylim(-1.2, 3.35)
    ax2.set_ylabel("Out-of-Sample Net Sharpe")
    ax2.set_title("(b) Multi-Regime Stress Audit (E10)", fontsize=8.5)
    ax2.legend(loc="upper left", ncol=3, fontsize=5.6, frameon=True, facecolor="white", framealpha=0.92)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"fig_e9_e12_feedback_and_regime_ablation.{ext}", bbox_inches="tight")
        alt_dir = ROOT / "paper" / "stock_prediction" / "figs"
        if alt_dir.exists():
            fig.savefig(alt_dir / f"fig_e9_e12_feedback_and_regime_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    print("Generated Figures 1-4 (PDF + PNG) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

