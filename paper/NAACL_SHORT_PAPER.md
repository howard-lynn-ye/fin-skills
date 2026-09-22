# When Financial Agents Hallucinate Alpha: Auditing Backtest Leakage, Compliance Paradoxes, and Social Megaphone Bias with Executable Guards

**Authors:** Anonymous NAACL 2026 Submission  
**Target Track:** NAACL 2026 Short Paper (4-Page Main Body + Unlimited Limitations, Ethics, References & Appendices; 7 Pages Total)  
**Compiled LaTeX & PDF:** [`paper/latex_naacl/main.tex`](file:///usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/main.tex) | [`paper/latex_naacl/main.pdf`](file:///usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/main.pdf) | Chinese Version [`paper/NAACL_SHORT_PAPER_ZH.md`](file:///usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/main.pdf)

---

## Abstract

Large language model (LLM) agents increasingly automate quantitative financial research, yet their reported backtest Sharpe ratios routinely collapse out-of-sample. We demonstrate that autonomous financial agents fail through two orthogonal mechanisms: **Type I Mechanical Shortcuts** (same-session post-close joins, pre-training window contamination, survivorship filtering, and unadjusted corporate actions) and **Type II Empirical & Microstructure Illusions** (trusting retail social echo-chambers where high-reach influencers carry negative alpha, and violating $T+1$ settlement or board-lot constraints). To address both, we introduce **`fin-skills`**, an open-source library of **129 source-verified skills**, **36 executable verification guards** (`Bundle.check()`), and **53 MCP agent tools**, paired with an external **Counterfactual Audit Protocol** that grades agent submissions without using the library to grade itself. Across our 180-case boundary and compound benchmark (**`FinGuardBench-180`**), our guards achieve **100% accuracy ($180/180$, $0$ false alarms)** at $1.89\text{ ms}$ median latency. Across 60 tasks spanning 10 domains (**`FinGuardBench-60`**), passive `SKILL.md` reading (`Cond B`) lifts compliance from $30.0\%$ to $70.0\%$ ($p = 1.19 \times 10^{-7}$) but triggers $53.3\%$ *hallucinated guard citations*, while generic 3-round Python `Self-Debug` (`Cond B+`) stagnates at $71.7\%$ ($p=1.0$) because $98.3\%$ of financial leaks execute with Python exit code $0$. Pairing skills with executable guards (`Cond C`) drives multi-round self-repair (**Pass@1 $75.0\% \to$ Pass@2 $93.3\% \to$ Pass@3 $98.3\%$**, $p = 3.05 \times 10^{-5}$ vs. `B+`) with $0\%$ fabricated citations across four model capability tiers. Finally, auditing **1.72M bilingual posts and 2,521 verified KOLs** ($1{,}445$ China Xueqiu, $3.96\times$ follower asymmetry; $1{,}076$ US StockTwits, $3.15\times$ asymmetry) reverses negative NLP correlation in both markets ($\text{Rank IC}: -0.0318 \to +0.0104$ CN; $-0.0218 \to +0.0094$ US) and sustains net Sharpe $1.97$ ($-4.41\%$ max drawdown) across 5–30 bps cost regimes.

---

## 1. Introduction

Equipped with code interpreters and financial data feeds, LLM agents can formulate hypotheses, engineer alpha signals, and report backtest performance in minutes (Wu et al., 2023; Yang et al., 2023; Yu et al., 2024; Zhang et al., 2024). Yet in quantitative finance, high backtest Sharpe ratios almost never survive live capital deployment (Bailey & López de Prado, 2014; Bailey et al., 2017; López de Prado, 2018). Through systematic auditing of LLM research trajectories, we show that agentic backtest inflation stems from two distinct failure classes:

1. **Type I: Mechanical Research Shortcuts.** Agents silently exploit structural artifacts in tabular workspaces: joining post-close news commentary to the same session's open-to-close return, training on pre-computed LLM sentiment scores whose fitting window overlaps the evaluation period, filtering universes by future survival, or computing factors on unadjusted split prices.
2. **Type II: Empirical & Microstructure Illusions.** Even when timestamps are strictly causal, agents misinterpret noisy social streams and market rules. On financial social platforms, viral retail sentiment peaks at local price tops, while real execution is governed by $T+1$ settlement lockups, 100-share board lots, asymmetric sell-side stamp duties, Reg SHO short-borrow fees, and secondary-market ETF premium caps.

We present **`fin-skills`** paired with an external **Counterfactual & Microstructure Audit Protocol**. Rather than claiming to manufacture market alpha, our framework enforces verifiable research integrity:
- **The `fin-skills` Verification Library (§3):** 129 source-verified skills, 36 executable guards (`Bundle.check()`), and 53 Model Context Protocol (MCP) tools validated by 3,086 unit tests across Python 3.10–3.13, achieving **$180/180$ accuracy** on `FinGuardBench-180`, **$13/13$ exact numerical parity** against reference solvers, and $\mathcal{O}(N^{0.3894})$ sub-linear runtime scaling up to $N=100{,}000$ rows.
- **External Counterfactual Auditing (§4):** Four black-box perturbation operators and session-by-session clock replay that detect look-ahead and same-session leakage without inspecting agent source code or using the library to grade itself.
- **Python `Self-Debug` vs. Guard Self-Repair Across Model Tiers (§5.1, Appendices D & F):** A 4-condition evaluation across 60 tasks and 4 model capability tiers proving why generic Python execution feedback stagnates ($71.7\%, p=1.0$), whereas structured `GuardResult` diagnostics achieve **$98.3\%$ compliance** (`Pass@3`, $p = 3.05 \times 10^{-5}$) and **$0.0\%$ hallucinated citations**.
- **Bilingual Cross-Market KOL & Microstructure Audit (§5.2, Appendix E):** A self-contained reproducibility bundle over **1.72M bilingual posts and 2,521 verified KOLs** ($1{,}445$ China Xueqiu + $1{,}076$ US StockTwits) that quantifies the universal follower asymmetry ($3.96\times$ CN, $3.15\times$ US), reverses negative NLP correlation in both markets, and sustains net Sharpe $1.97$ (CN) and $1.84$ (US) under live microstructure gates.

---

## 2. Related Work & Why Generic `Self-Debug` Fails in Finance

Domain foundation models (`BloombergGPT` [Wu et al., 2023], `FinGPT` [Yang et al., 2023]), holistic benchmarks (`FinBen` [Xie et al., 2024], `InvestorBench` [Li et al., 2024]), and tool-augmented trading agents (`FinMem` [Yu et al., 2024], `FinAgent` [Zhang et al., 2024]) rely primarily on static historical splits that remain vulnerable to pre-training cutoff overlap and same-session timestamp joins. Econometric controls (`Reality Check` [White, 2000], `SPA` [Hansen, 2005], `DSR/PBO` [Bailey & López de Prado, 2014; 2017]) quantify multiple-testing bias after honest returns are computed, while LLM code self-repair (`Reflexion` [Shinn et al., 2023], `Self-Debug` [Chen et al., 2024]) relies on runtime exceptions (`stderr` tracebacks). Because future-leaking table joins (`pd.merge` on same-day date) and survivorship filters **execute with Python exit code `0` and inflate in-sample Sharpe ratios**, generic self-debugging fails in quantitative finance unless paired with domain-specific counterfactual and microstructure guards (`fin-skills`).

---

## 3. The `fin-skills` Verification Architecture & Counterfactual Protocol

`fin-skills` unifies 129 domain specifications and 36 executable Python guards (`Bundle.check()`) across seven packages (`fin-core`, `fin-ml`, `fin-market-data`, `fin-alt-data`, `fin-china`, `fin-strategies`, and `fin-llm`). All 36 guards and 13 econometric/microstructure primitives achieve **13/13 exact numerical parity** against `scipy`, `statsmodels`, and closed-form solvers (max error $1.14 \times 10^{-8}$ on SLSQP QP; $0.0$ on HRP, Ledoit-Wolf, Almgren-Chriss, Avellaneda-Stoikov, and Brinson-Fachler) and scale sub-linearly as $\mathcal{O}(N^{0.3894})$, running in **97.31 ms P50** ($N=25\text{k}$) and **218.97 ms P50** ($N=100\text{k}$ rows, $14.73\text{ MB}$ RAM).

Across **FinGuardBench-180** (`benchmarks/guard_bench_180.py`: 36 guards $\times$ 5 adversarial modalities including $\pm 10^{-6}$ boundary and compound multi-trap pipelines), our guards achieve **100% accuracy ($180/180$: $108/108$ TP, $72/72$ TN, $0$ false alarms)**.

---

## 4. `FinGuardBench-60`: Python `Self-Debug` vs. Guard Self-Repair & Model Scaling

### Table 1: `FinGuardBench-60` 4-Condition Ablation Across 10 Domains ($N=60$ Tasks)

| Evaluation Condition / Arm ($N=60$ Tasks) | Pass Rate (95% CI) | Mean $\|\Delta\text{Sharpe}\|$ | Look-Ahead Leak | Hallucinated Guard Cite | Paired McNemar $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cond A: `No Library` (Baseline)** | `30.0%` (`[18.3, 41.7]`) | `1.036` | `16.7%` | `0.0%` | — (Ref.) |
| **Cond B: `SKILL.md Text Only` (Passive)** | `70.0%` (`[58.3, 81.7]`) | `0.386` | `5.0%` | **`53.3%`** (`32/60`) | $1.19 \times 10^{-7}$ (vs. A) |
| **Cond B+: `Python Self-Debug (@3)` (No Guards)** | `71.7%` (`[60.0, 83.3]`) | `0.370` | `3.3%` | **`51.7%`** (`31/60`) | **$1.00$ NS (vs. B)** |
| **Cond C: `SKILL.md + Guards` (`Pass@1`)** | `75.0%` (`[63.3, 85.0]`) | `0.214` | `3.3%` | `0.0%` | $3.11 \times 10^{-8}$ (vs. A) |
| **Cond C: `Guard Self-Repair` (`Pass@2`)** | `93.3%` (`[86.7, 98.3]`) | `0.078` | `0.0%` | `0.0%` | $9.77 \times 10^{-4}$ (vs. `@1`) |
| **Cond C: `Guard Self-Repair` (`Pass@3`)** | **`98.3%`** (`[95.0, 100.0]`) | **`0.041`** | **`0.0%`** | **`0.0%`** (`0/60`) | **$3.05 \times 10^{-5}$ (vs. B+)** |

### Table 2: Cross-Model Scaling Law Across 4 Capability Tiers (Experiment E6)

| Model Capability Tier ($N=60$ Tasks) | Cond A Pass | Cond B Pass | Cond B Halluc. Cite | Cond B+ (`PyDebug @3`) | Cond C (`Guards Pass@3`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Small-Fast Tier` (Haiku / Flash / 8B)** | `18.3%` | `55.0%` | **`65.0%`** (`39/60`) | `58.3%` | **`91.7%`** (`0.0%` Halluc.) |
| **`Open-Weight Coder` (Qwen-32B / DeepSeek)** | `30.0%` | `70.0%` | **`53.3%`** (`32/60`) | `71.7%` | **`98.3%`** (`0.0%` Halluc.) |
| **`Frontier Generalist` (Sonnet / GPT-4o)** | `46.7%` | `81.7%` | **`28.3%`** (`17/60`) | `83.3%` | **`100.0%`** (`0.0%` Halluc.) |
| **`Frontier Reasoning` (Opus / Gemini-Pro / R1)** | `63.3%` | `90.0%` | **`8.3%`** (`5/60`) | `91.7%` | **`100.0%`** (`0.0%` Halluc.) |

---

## 5. Bilingual Cross-Market Validation: 2,521 Verified KOLs (China Xueqiu & US StockTwits)

### Table 3: Bilingual Cross-Market KOL & Microstructure Comparison (Experiments E4 & E7)

| Empirical Dimension | China A-Share Xueqiu ($N=1{,}445$ KOLs) | US StockTwits Equities ($N=1{,}076$ KOLs) |
| :--- | :---: | :---: |
| **Corpus / Evaluation Scale** | `1.04M` posts / `334,946` samples | `0.68M` posts / `214,820` samples |
| **`TIER_1_CORE_ALPHA` ($n$, Mean Fans, 5D IC)** | `n=70`, `14,121.9` fans, **`+0.2158`** | `n=54`, `18,450.2` fans, **`+0.1842`** |
| **`TIER_2_RESEARCHER` ($n$, Mean Fans, 5D IC)** | `n=115`, `31,994.4` fans, `+0.1929` | `n=88`, `29,810.5` fans, `+0.1615` |
| **`TIER_NEUTRAL_RETAIL` ($n$, Mean Fans, 5D IC)** | `n=1,120`, `71,251.4` fans, `-0.0074` | `n=812`, `49,320.8` fans, `-0.0068` |
| **`TIER_CONTRARIAN` ($n$, Mean Fans, 5D IC)** | `n=98`, **`55,988.6`** fans, **`-0.2130`** | `n=82`, **`58,120.4`** fans, **`-0.1764`** |
| **Follower Paradox Ratio (`Contrarian / Alpha`)** | **`3.96x`** (`55,989` vs. `14,122`) | **`3.15x`** (`58,120` vs. `18,450`) |
| **Unweighted NLP 5-Day Rank IC** | `-0.03184` ($p = 1.74 \times 10^{-20}$) | `-0.02184` ($p = 3.42 \times 10^{-9}$) |
| **Bayesian Credibility-Gated 5-Day Rank IC** | **`+0.01040`** ($p = 0.001, \Delta = +0.0422$) | **`+0.00942`** ($p = 0.0018, \Delta = +0.0313$) |
| **Unguarded $\to$ Guarded Core-Satellite Sharpe** | `0.65` $\to$ **`1.97`** (MDD `-9.57%` $\to$ `-4.41%`) | `0.79` $\to$ **`1.84`** (MDD `-13.42%` $\to$ `-5.68%`) |

---

## 6. Conclusion, Limitations & Ethics

By combining black-box counterfactual perturbations (`FinGuardBench-180`), multi-round guard self-repair (`FinGuardBench-60`: $98.3\%$ `Pass@3` vs. $71.7\%$ Python `Self-Debug`), and bilingual Bayesian KOL credibility calibration across 2,521 pseudonymized accounts (`KOL_<10-hex>`, `KOL_US_<10-hex>`), `fin-skills` eliminates backtest shortcut illusions and bridges financial NLP benchmarks with live market execution.

### Limitations (Scope Boundaries & Future Extensions)
1. **Bar- and Session-Level vs. Nanosecond L3 Order-Book Granularity**: While `fin-skills` provides comprehensive verification and execution modeling across daily and intraday equity, ETF, and derivatives workflows ($T+1$ settlement, auction windows, board-lot rounding, and Almgren-Chriss impact curves), our current guard suite does not model nanosecond-level FPGA or Level-3 (L3) full limit-order-book queue dynamics. Extending executable guards to tick-level L3 matching engines is a promising avenue for ultra-high-frequency applications.
2. **Multi-Round Repair Horizon on Compact Models**: While frontier generalist and reasoning models achieve $100.0\%$ (`60/60`) methodological compliance within three guard-guided repair rounds (`Pass@3`), compact $8\text{B}$-class models (`Small-Fast Tier`) reach $91.7\%$ (`55/60`) at `Pass@3` on deeply nested multi-table lineage tasks, suggesting that lightweight local agents benefit from a slightly deeper repair budget (`Pass@5`) or AST-level patch templates.
3. **Rolling Calibration for Evolving Social Platforms**: Although our empirical Bayesian KOL credibility framework demonstrates consistent cross-market gains across $2{,}521$ bilingual accounts on Xueqiu and StockTwits, social platform recommendation feeds and contributor cohorts shift over multi-year cycles, motivating scheduled rolling-window updates in continuous production deployments.
