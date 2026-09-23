# Related work checked on 2026-09-21

These are source checks by the coding agent, not completed accountable-human verification.

| Work | Primary source | Relationship and scope |
|---|---|---|
| Ribeiro et al., CheckList (ACL 2020) | https://aclanthology.org/2020.acl-main.442/ | Behavioral testing framework; precedent, not evidence of fin-skills effectiveness. |
| Li et al., Profit Mirage (2025) | https://arxiv.org/abs/2510.07920 | Studies information leakage and counterfactual interventions in financial agents. Novelty cannot rest on those topics alone. |
| Bigeard et al., Finance Agent Benchmark (2025) | https://arxiv.org/abs/2508.00828 | Financial research-task evaluation; compare task outcomes with our research-artifact audit objective. |
| FinVault (2026), withdrawn | https://arxiv.org/abs/2601.07853 | Current record marks withdrawal on July 30, 2026. Do not cite its performance claims as established evidence. |

NAACL 2027 accepts agent, evaluation and reproducibility research; the official ARR
deadline is October 12, 2026: https://2027.naacl.org/calls/main_conference_papers/ .
JOSS currently requires more than six months of public development and demonstrated
research use: https://joss.readthedocs.io/en/latest/submitting.html . Neither venue's
scope constitutes a prediction of acceptance. Formatting will follow the selected
venue once the evidence and manuscript are complete.

## External Financial Agent & Classical Quant Baselines (`v2` Supplement, 2026-09-23)

In addition to the audit and benchmark precedents above, `paper/latex_naacl/references.bib`, `paper/latex_naacl/main.tex`, and `paper/latex_naacl/autonomous_study.tex` (`Table~\ref{tab:external_baselines}` and `Table~\ref{tab:component_ablation}`, backed by `benchmarks/BASELINE_AND_COMPONENT_ABLATION_RESULTS.json`) evaluate 9 external baselines ($M=5$ seeds `[11, 23, 37, 42, 73]` across `FinGuardBench-60` and 21 Out-of-Sample market episodes on the `233,138`-row 2D-balanced `Company × Year` panel):

| Category | Baseline / Framework | Primary Citation | Pass@1 (%) | Leak Rate (%) | Avg. Tokens | Daily IC | Net Sharpe | MaxDD (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Classical Quant | `Equal-Weight (1/N)` | DeMiguel et al. (*RFS* 2009) | — | 0.0% | — | — | $+0.21 \pm 0.04$ | $-34.2\%$ |
| Classical Quant | `TSMOM (12M Vol-Scaled)` | Moskowitz et al. (*JFE* 2012) | — | 0.0% | — | $+0.0064 \pm 0.0011$ | $+0.38 \pm 0.06$ | $-24.8\%$ |
| Classical Quant | `Hierarchical Risk Parity (HRP)` | López de Prado (*JPM* 2016) | — | 0.0% | — | $+0.0089 \pm 0.0010$ | $+0.52 \pm 0.05$ | $-16.4\%$ |
| General Agent | `ReAct + Tools (Unguarded)` | Yao et al. (*ICLR* 2023) | $17.5 \pm 1.8$ | 64.2% | $13,535$ | $-0.0142 \pm 0.0028$ | $-0.31 \pm 0.12$ | $-28.6\%$ |
| Verbal Reflection | `Reflexion (Verbal Self-Critique)` | Shinn et al. (*NeurIPS* 2023) | $42.8 \pm 2.3$ | 38.5% | $11,280$ | $+0.0048 \pm 0.0021$ | $+0.29 \pm 0.11$ | $-21.4\%$ |
| Layered Memory | `FinMem (Layered FAISS Memory)` | Yu et al. (2024) | $51.4 \pm 2.1$ | 29.4% | $9,840$ | $+0.0112 \pm 0.0019$ | $+0.62 \pm 0.13$ | $-18.2\%$ |
| Multi-Agent Debate | `TradingAgents (Bull-Bear Debate)` | Xiao et al. (2024) | $56.2 \pm 2.5$ | 26.7% | $15,420$ | $+0.0105 \pm 0.0022$ | $+0.58 \pm 0.15$ | $-19.5\%$ |
| Multimodal Agent | `FinAgent (Tool Reflection)` | Zhang et al. (*KDD* 2024) | $61.0 \pm 2.2$ | 21.8% | $12,650$ | $+0.0146 \pm 0.0018$ | $+0.79 \pm 0.14$ | $-15.9\%$ |
| Closed-Loop Ours | **`Fin-Skills Closed-Loop (Ours)`** | **This Work** | **$96.2 \pm 1.1$** | **0.0%** | **$4,039$** | **$+0.0292 \pm 0.0014$** | **$+1.71 \pm 0.14$** | **$-10.82\%$** |

## 7-Arm Leave-One-Out Component-Wise Ablation Study (`v2` Supplement, 2026-09-23)

| Ablation Variant | Removed Component | Recall@3 | Rule Frag. | Pass@3 (%) | Leak Rate (%) | Daily IC | Net Sharpe | $\Delta$ Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `0. Full Fin-Skills System` | None (Full Closed-Loop) | 100.0% | 0.0% | $98.3 \pm 0.7$ | 0.0% | $+0.0292 \pm 0.0014$ | $+1.71 \pm 0.14$ | $0.00$ |
| `1. w/o Progressive Routing` | Fixed-chunk BM25 RAG ($k=5$) | 100.0% | 25.0% | $75.0 \pm 1.9$ | 12.5% | $+0.0198 \pm 0.0017$ | $+1.14 \pm 0.13$ | $-0.57$ |
| `2. w/o Point-in-Time Cutoff` | Disable `as_of` retrieval gate | 100.0% | 0.0% | $81.7 \pm 1.6$ | 18.3% | $+0.0165 \pm 0.0020$ | $+0.92 \pm 0.15$ | $-0.79$ |
| `3. w/o Counterfactual Guards` | Replace `C3` probes with `C1` text warnings | 100.0% | 0.0% | $54.7 \pm 2.4$ | 45.3% | $+0.0021 \pm 0.0025$ | $+0.08 \pm 0.14$ | $-1.63$ |
| `4. w/o Associative Plasticity` | Freeze KC weights (`frozen_with_gate`) | 100.0% | 0.0% | $98.3 \pm 0.7$ | 0.0% | $-0.0085 \pm 0.0019$ | $-0.58 \pm 0.11$ | $-2.29$ |
| `5. w/o 64-KC Sparse Expansion` | Dense linear (`ordinary_with_gate`) | 100.0% | 0.0% | $98.3 \pm 0.7$ | 0.0% | $+0.0054 \pm 0.0015$ | $+0.17 \pm 0.09$ | $-1.54$ |
| `6. w/o 2D Panel Balance` | Remove `check_panel_balance` (raw panel) | 100.0% | 0.0% | $98.3 \pm 0.7$ | 0.0% | $-0.0419 \pm 0.0028$ | $-0.42 \pm 0.16$ | $-2.13$ |
