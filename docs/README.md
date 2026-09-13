# 📚 `fin-skills` Documentation Hub (v2 Branch)

Welcome to the **`fin-skills` Documentation Center** (`v2` branch). This directory contains full operational guides, architecture manuals, and integration handbooks for connecting `fin-skills` (114 Quantitative Finance Agent Skills + 32 Executable Python Backtest Guards) to your local quantitative research pipelines and AI coding agents.

---

## 🧭 Core Documentation Index

| Document | Language | Audience & Purpose | Quick Link |
| :--- | :---: | :--- | :--- |
| **📖 User Manual (`USER_MANUAL_ZH.md`)** | 简体中文 | **Comprehensive Architecture & Resource Connection Manual**.<br>Explains how `fin-skills` operates (Zero-Backend local execution), how it evaluates historical panel data (`assert_causal`, `warmup_probe`, `cost_curve`, `deflated_sharpe`), and how to wire `fin-skills` to local feature tables (`stock_prediction`), MCP servers, and Jetski AI agents. | [View Manual](./USER_MANUAL_ZH.md) |
| **🛠️ Operations Guide (`OPERATIONS_GUIDE_ZH.md`)** | 简体中文 | **Step-by-Step CLI & Daily Operations Playbook**.<br>Provides copy-pasteable terminal commands for environment verification, automated pre-commit/CI data audits, interactive Jetski agent prompts, and `v2` Git branch synchronization workflows. | [View Operations Guide](./OPERATIONS_GUIDE_ZH.md) |
| **🇨🇳 Project Overview (`README_ZH.md`)** | 简体中文 | **Chinese Master README & Full 114-Skill Taxonomy**.<br>Executive summary, Mermaid 3-tier system architecture, project maturity scorecard, `leak_bench` 12-defect detection benchmark, and Chinese translations of all 114 skills across 11 domains. | [View Chinese README](./README_ZH.md) |
| **🌐 English Master README (`../README.md`)** | English | **Canonical English Specification & Guard Reference**.<br>Official agent skills specification, 32 Python API guard signatures (`fin_skills.api`), and third-party skill marketplace catalog. | [View English README](../README.md) |

---

## ⚡ Quick Architecture Summary: How `fin-skills` Works

`fin-skills` is a **pure local, zero-remote-backend** quantitative verification engine and AI skill library:

1. **No External API Dependency**: All 32 Python guards in `fin_skills.api` run 100% in-memory on your local machine (`pandas` / `numpy` / `scipy`), ensuring zero data leakage or cloud telemetry of proprietary alpha signals.
2. **Historical Data Perturbation & Statistical Audits**:
   - **Suffix Mutation (`assert_causal`)**: Mutates future tail rows (`T-k ... T`) and recomputes your signal; raises a hard error if historical values (`< T-k`) change by even `1e-12`.
   - **Warmup Instability (`warmup_probe`)**: Truncates initial history windows to verify where rolling/EWM indicators stabilize.
   - **Cost Break-Even (`cost_curve`)**: Sweeps transaction cost assumptions (`0` to `50 bps`) to find the exact fee threshold where strategy net Sharpe drops to zero.
   - **Multiple Testing Adjustment (`deflated_sharpe`)**: Adjusts observed Sharpe ratios for sample skewness, kurtosis, and total trial count $N$.

---

## 🚀 Quick Verification Command

Run the built-in self-test suite from the repository root to verify that all 32 guards and 79 unit tests pass in your local environment:

```bash
pytest tests/test_api.py -v
python3 scripts/validate.py
```
