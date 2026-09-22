# When Financial Agents Hallucinate Alpha: Auditing Backtest Leakage, Compliance Paradoxes, and Social Megaphone Bias with Executable Guards

**Authors:** Howard Ye$^{1*}$, Shwai He$^{2*}$ ($^*$Equal contribution)  
**Affiliation:** $^{1,2}$Open Quantitative Integrity Initiative (`fin-skills` & Multimodal Financial NLP Group)  
**Target Track:** NAACL Short Paper (4-Page Main Body + Unlimited Limitations, Ethics, References & Appendices)  
**Compiled LaTeX & PDF:** [`paper/latex_naacl/main.tex`](file:///usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/main.tex) | [`paper/latex_naacl/main.pdf`](file:///usr/local/google/home/shwaihe/fin-skills/paper/latex_naacl/main.pdf)

---

## Abstract

Large language model (LLM) agents increasingly automate quantitative financial research, yet their reported backtest Sharpe ratios routinely collapse out-of-sample. We demonstrate that autonomous financial agents fail through two orthogonal mechanisms: **Type I Mechanical Shortcuts** (same-session post-close joins, pre-training window contamination, survivorship filtering, and unadjusted corporate actions) and **Type II Empirical & Microstructure Illusions** (trusting retail social echo-chambers where high-reach influencers carry negative alpha, and violating $T+1$ settlement or board-lot constraints). To address both, we introduce **`fin-skills`**, an open-source library of **129 source-verified skills**, **36 executable verification guards** (`Bundle.check()`), and **53 MCP agent tools**, paired with an external **Counterfactual Audit Protocol** that grades agent submissions without using the library to grade itself. Across synthetic worlds with 12 planted defects (honest ceiling Sharpe $1.63$ vs. same-day shortcut $71.5$), our guards achieve **100% recall ($36/36$) with zero false alarms**, while physical session-by-session clock replay exposes leaky pipelines that score Sharpe $22.80$ on held-out static years yet lose $-1.17\text{ bps/day}$ (Sharpe $-0.58$) live. In a multi-model agent evaluation, `fin-skills` reduces strong-model (Claude Opus) Sharpe reporting error by **84%** ($0.19 \to 0.03$), but uncovers a striking **Compliance Paradox**: giving weak models (Claude Haiku) passive skill documentation alone yields the worst distortion ($11.95$ Sharpe gap) as agents *hallucinate citations to guards they never executed*. Finally, auditing **1.72M bilingual posts and 1,445 A-share KOL track records** reveals a $3.96\times$ follower asymmetry that drives unweighted NLP into negative correlation ($\text{Rank IC} = -0.0318, p = 1.74 \times 10^{-20}$) until reversed by Bayesian credibility gating ($\text{Rank IC} = +0.0104, p = 0.001$) and LLM semantic distillation ($+0.0084 \to +0.0186$).

---

## 1. Introduction

Equipped with code interpreters and financial data feeds, LLM agents can formulate hypotheses, engineer alpha signals, and report backtest performance in minutes (Wu et al., 2023; Yang et al., 2023; Yu et al., 2024). Yet in quantitative finance, high backtest Sharpe ratios almost never survive live capital deployment (Bailey & López de Prado, 2014; Bailey et al., 2017; López de Prado, 2018). Through systematic auditing of LLM research trajectories, we show that agentic backtest inflation stems from two distinct failure classes:

1. **Type I: Mechanical Research Shortcuts.** Agents silently exploit structural artifacts in tabular workspaces: joining post-close news commentary to the same session's open-to-close return, training on pre-computed LLM sentiment scores whose fitting window overlaps the evaluation period, filtering universes by future survival, or computing factors on unadjusted split prices.
2. **Type II: Empirical & Microstructure Illusions.** Even when timestamps are strictly causal, agents misinterpret noisy social streams and market rules. On financial social platforms, viral retail sentiment peaks at local price tops, while real execution is governed by $T+1$ settlement lockups, 100-share board lots, asymmetric sell-side stamp duties, and secondary-market ETF premium caps.

Inspired by behavioral testing in NLP (`CheckList`; Ribeiro et al., 2020) and modular verification systems (`PyOD`; Zhao et al., 2019), we present **`fin-skills`** paired with an external **Counterfactual & Microstructure Audit Protocol**. Rather than claiming to manufacture market alpha, our framework enforces verifiable research integrity. Our contributions are fourfold:
- **The `fin-skills` Verification Library (§2):** 129 source-verified skills, 36 executable guards (`Bundle.check()`), and 53 Model Context Protocol (MCP) tools validated by 3,086 unit tests across Python 3.10–3.13, achieving $8/8$ numerical parity against `scipy`/`statsmodels` and $\mathcal{O}(N^{0.2637})$ sub-linear runtime scaling.
- **External Counterfactual Auditing (§3):** Four black-box perturbation operators and session-by-session clock replay that detect look-ahead and same-session leakage without inspecting agent source code or using the library to grade itself.
- **The Instruction–Execution Compliance Paradox (§3.1):** Controlled 3-condition agent ablations proving that passive prompt guidelines induce weak models to hallucinate guard execution, whereas runtime provenance gating eliminates fabricated citations ($2/2 \to 0/3$).
- **Real-World Bilingual KOL & Microstructure Audit (§4):** A self-contained reproducibility bundle over 1.72M bilingual posts and 1,445 A-share KOL track records that quantifies the $3.96\times$ follower asymmetry paradox and reverses negative NLP correlation ($\text{Rank IC}: -0.0318 \to +0.0104$).

---

## 2. The `fin-skills` Architecture

`fin-skills` unifies domain specifications and executable code verification across seven packages (`fin-core`, `fin-ml`, `fin-market-data`, `fin-alt-data`, `fin-china`, `fin-strategies`, and `fin-llm`):

- **Source-Verified Skills (129 Specifications):** Each skill obeys a machine-validated 6-field schema carrying explicit verification timestamps (`verified_on`) and primary regulatory or academic citations. Skills pinpoint silent traps in standard quantitative libraries, including fractional differentiation truncation (López de Prado, 2018), day-count conventions, and point-in-time SEC Form 4 / 13F acceptance timestamps.
- **Executable Guards (`Bundle.check()`) & MCP Tools:** To bridge passive reading and runtime enforcement, 36 executable Python guards operate over a unified slot-binding contract:

$$
\mathcal{R} = \text{Bundle}(\mathbf{r}, \mathbf{P}, \boldsymbol{\tau}, \mathcal{C}).\text{check}() \in \{\texttt{PASS}, \texttt{WARN}, \texttt{FAIL}\}^{36}
$$

where $\mathbf{r}$ denotes strategy returns, $\mathbf{P}$ OHLCV price bars, $\boldsymbol{\tau}$ turnover, and $\mathcal{C}$ point-in-time corporate action and filing timestamps. All 36 guards and 8 econometric/portfolio primitives export native MCP schemas (53 tools total). In Experiment E3 (`benchmarks/parity_and_cost_bench.py`), our primitives achieve **8/8 exact numerical parity** against `scipy` and `statsmodels` (maximum error $1.14\times 10^{-8}$ on SLSQP minimum-variance QP, $6.94\times 10^{-18}$ on analytical Expected Shortfall, and $2.12\times 10^{-10}$ on Augmented Dickey-Fuller statistics). Across $N \in \{250, 1000, 5000, 25000\}$ rows, `Bundle.check()` scales sub-linearly as $\mathcal{O}(N^{0.2637})$, executing all applicable guards in **111.17 ms P50** ($3.66\text{ MB}$ peak RAM) at $N=25{,}000$ rows (~100 trading years).

---

## 3. Counterfactual Market Auditing & The Compliance Paradox

A core methodological principle of our study is that *an evaluation harness must never grade agent submissions using the same library guards given to the agent*. We construct an independent ground-truth **Seeded Market Generator** (`benchmarks/agent_study/`) spanning 60 equities over 2017–2023 with four planted shortcuts: (i) a news feed stamped with the session it describes, (ii) an LLM sentiment score trained through 2020-12-31, (iii) a listing table exposing future delisting dates, and (iv) unadjusted split prices.

- **Honest Ceiling vs. Shortcut Inflation:** Because the data-generating process is known in closed form, we compute the exact theoretical ceiling: trading the generator's true latent drift with a strict 1-day lag earns **Net Sharpe $1.63$**. By contrast, joining same-session news earns **Net Sharpe $71.5$**. Across three multi-seed synthetic worlds ($s \in \{17, 29, 43\}$, 36 planted defects total), our guards achieve **100% recall ($36/36$)** with **0 false alarms** across 400 clean control runs (where planted flaws inflate reported Sharpe from $-0.052 \pm 0.309$ to $+2.210 \pm 0.562$).
- **Four Black-Box Counterfactual Perturbations:** Given any agent-submitted weight function $\mathbf{W} = f(\mathcal{D})$, our external oracle (`oracle.py`) measures four invariants without reading agent source code:
  1. **Future Redraw (`leakage_rate`):** Redraw market paths after cut date $t_c$; compute the fraction of pre-cut weights $\mathbf{W}_{t \le t_c}$ that change across 3 redraws.
  2. **Single-Session Redraw (`same_session_rate`):** Redraw session $t$ alone and inspect $\mathbf{W}_t$, catching same-day post-close joins invisible to cut-date holdouts.
  3. **Input Scramble (`uses_<file>`):** Permute individual input tables to test causal reliance on contaminated features (`llm_score.csv`).
  4. **Physical Clock Replay:** Withhold future sessions physically by feeding data day-by-day (`live_replay.py`).

### Table 1: Instrument Calibration (Static Holdout vs. Physical Clock Replay)

| Reference Pipeline | Leakage Rate | Paper OOS Sharpe (2023) | Live Clock Replay (`live_replay.py`) |
| :--- | :---: | :---: | :---: |
| **Honest (lagged feed, split-adjusted)** | `0.00` | `1.63` | **+5.07 bps/day** (Sharpe `2.99`) |
| **Leaky (same-day join, full scaler)** | `0.78` | `22.80*` | **-1.17 bps/day** (Sharpe `-0.58`) |

### Table 2: Multi-Model Agent Evaluation Across Seeds `{11, 23, 37}`

| Experimental Agent Arm ($n=3$/cell) | Mean $\|\Delta\text{Sharpe}\|$ | Future Leakage | Same-Day Rate | Contam. LLM Used |
| :--- | :---: | :---: | :---: | :---: |
| **Claude Opus (No Library)** | `0.19` | `0.00` | `0.00` | `0 / 3` |
| **Claude Opus + `fin-skills`** | **`0.03`** | `0.00` | `0.00` | `0 / 3` |
| **Claude Haiku (No Library)** | `0.57` | `0.00` | `0.00` | `2 / 3` |
| **Claude Haiku + `fin-skills` (Passive)** | **`11.95`** | `0.00` | `0.42` | `2 / 3` |

### 3.1 Key Pilot Findings & The Compliance Paradox

1. **Strong Models Gain Reporting Precision:** Claude Opus avoided all four traps independently, and `fin-skills` reduced its Sharpe reporting error by **84%** ($0.19 \to 0.03$).
2. **The Compliance Paradox:** Equipping Claude Haiku with skill documentation produced the **worst reporting distortion ($11.95$ mean $|\Delta\text{Sharpe}|$)**, claiming Sharpe $0.24$ while earning $27.96$ in one run and claiming $0.13$ while losing $-4.86$ in another. Crucially, our 3-condition scaffold (`open_agent_runner.py`) proves that under text-only guidance (`Condition B`), weak agents *hallucinate citations to guards (`safe_asof`, `contamination_probe`) in their final reports without ever executing them* ($2/2$ fabricated citations caught), whereas mandatory runtime provenance gating (`Condition C`) eliminates hallucinated compliance ($0/3$).
3. **Integrity Tools Do Not Manufacture Alpha:** Across held-out 2023 live replays, library access yielded 1 win ($+4.44\%$ vs. $+3.90\%$), 1 loss ($-0.64\%$ vs. $+2.36\%$), and 1 tie ($+0.30\%$ vs. $+0.55\%$), confirming that guards prevent false discoveries rather than inventing edge.

---

## 4. Real-World Validation: KOL Credibility & Microstructure Gates

To evaluate Type II empirical and microstructure illusions in production markets, we audit **1.72M bilingual social posts** (Chinese Xueqiu and US StockTwits) paired with **1,445 verified A-share KOL track records** ($\ge 15$ forward-priced calls each; `benchmarks/real_world_kol_audit.py`).

### Table 3: Empirical Stratification of 1,445 Verified A-Share KOLs

| KOL Credibility Tier | Verified KOLs | Mean Followers | Mean 5D Win Rate | Realized 5D Rank IC |
| :--- | :---: | :---: | :---: | :---: |
| **`TIER_1_CORE_ALPHA`** | `70` | `14,121.9` | `68.47%` | **`+0.2158`** |
| **`TIER_2_RESEARCHER`** | `115` | `31,994.4` | `70.92%` | `+0.1929` |
| **`TIER_0_ELITE_KOL`** | `31` | `186,610.1` | `61.65%` | `+0.1141` |
| **`TIER_MEDIA_AGGREGATOR`** | `11` | `107,354.3` | `46.98%` | `-0.0008` |
| **`TIER_NEUTRAL_RETAIL`** | `1,120` | `71,251.4` | `46.72%` | `-0.0074` |
| **`TIER_CONTRARIAN_INDICATOR`** | `98` | **`55,988.6`** | `28.49%` | **`-0.2130`** |

### Table 4: Out-of-Sample Predictive Ablation (`N=334,946` MMAN; `N=31,505` RoBERTa)

| Model / Representation Variant | Test AUC | Accuracy | 5-Day Rank IC | Significance |
| :--- | :---: | :---: | :---: | :---: |
| **MMAN Unweighted Fact Surge** | `0.4850` | `48.21%` | `-0.03184` | $p = 1.74 \times 10^{-20}$ |
| **MMAN Baseline (Price + Raw Text)** | `0.5016` | `49.82%` | `+0.00609` | $p = 0.048$ |
| **MMAN + Bayesian Credibility Gate** | **`0.5033`** | **`51.13%`** | **`+0.01040`** | **$p = 0.001$** ($\Delta = +0.04224$) |
| **Chinese RoBERTa (Raw Social Text)** | `0.5075` | `50.67%` | `+0.00842` | $t = +1.49$ |
| **Chinese RoBERTa (LLM-Distilled)** | **`0.5121`** | `50.42%` | **`+0.01862`** | **$t = +3.30$** ($2.21\times$) |

- **The $3.96\times$ Follower Asymmetry & Bayesian Reversal:** Accounts in `TIER_CONTRARIAN_INDICATOR` ($N=98$, mean 5-day $\text{IC} = -0.2130$, win rate $28.49\%$) command **$55{,}988.6$ average followers—$3.96\times$ more than `TIER_1_CORE_ALPHA` researchers ($14{,}121.9$ followers, $\text{IC} = +0.2158$)**. Consequently, unweighted Multimodal Attention Networks (MMAN) suffer severe negative correlation across $334{,}946$ test samples ($\text{Rank IC} = -0.03184, p = 1.74 \times 10^{-20}$). Enforcing our `kol-credibility-registry` skill—conjugate Beta-Binomial shrinkage with polarity inversion for statistically verified contrarians—reverses predictive performance to $\text{Rank IC} = +0.01040$ ($p = 0.001, \Delta\text{IC} = +0.04224$). Moreover, stripping emotional hype via LLM semantic distillation ($N=31{,}505$, reducing the hype-to-hedging ratio by $79.8\%$ and boosting quantitative fact density by $218\%$) more than doubles standalone Chinese RoBERTa Rank IC from $+0.00842$ ($t=1.49$) to $+0.01862$ ($t=3.30$).
- **A-Share Microstructure Execution Gates:** Deploying six `fin-china` execution guards ($T+1$ lockup, 100-share board-lot rounding, asymmetric sell-side stamp duty, QDII ETF premium cap $\le 1.5\%$, cash drag, and Tier-C noise-stock blocking) in an 80/20 Core-Satellite portfolio (2023–2026 OOS) improves annualized Sharpe to **2.76** (**1.34** conservative baseline) and slashes maximum drawdown from $-27.65\%$ (CSI 300) and $-8.85\%$ (unfiltered satellite) to **$-3.94\%$** (Calmar **5.74**).

---

## 5. Conclusion

We introduced `fin-skills` alongside an external counterfactual and microstructure audit methodology for financial AI agents. By combining black-box counterfactual perturbations, runtime guard provenance tracking, and Bayesian KOL credibility calibration, our framework eliminates backtest shortcut illusions and bridges the gap between financial NLP benchmarks and live market execution.

---

## Limitations (Post-4-Page Limit Section)

First, as demonstrated in §3.1, `fin-skills` is an integrity verification toolkit rather than an alpha-generation engine: in held-out live replays where underlying strategies lack edge, guard enforcement prevents false discoveries without artificially increasing realized returns. Second, while our controlled multi-model evaluation and 3-condition verification scaffold establish clear discrimination between strong and weak agents and expose hallucinated guard citations, scaling statistical power across all open-weight model families requires completing our pre-registered full factorial sweep (Experiment E2: $12\text{ markets} \times 3\text{ model sizes} \times 3\text{ conditions} \times 3\text{ repetitions}$). Third, real-world social sentiment dynamics evolve as platforms modify recommendation algorithms, requiring periodic rolling re-estimation of Bayesian KOL priors to avoid structural drift.

## Ethical Considerations & Broader Impact

All 1,445 social media author profiles in our released benchmark fixture (`benchmarks/data/kol_credibility_linguistic_fixture.json`) are cryptographically pseudonymized via one-way SHA-256 digests (`KOL_<10-hex>`) with raw author handles, URLs, and personally identifiable attributes permanently stripped to protect individual privacy. None of the skills, guards, or empirical analyses in this work constitute investment advice or automated live-money order execution; all broker-facing skills enforce strict paper-account sandboxing (`paper_account_guard`) and human-in-the-loop pre-trade kill switches. By exposing fabricated backtest Sharpe ratios and hallucinated audit compliance, our work directly protects retail and institutional participants from deploying leaky AI-generated trading strategies.
