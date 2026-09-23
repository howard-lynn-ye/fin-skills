# Implementation and Evidence Status (`v2` Unified Post-Merge Closure)

Updated 2026-09-22. This record summarizes the complete implementation, contract-hardening fixes, and verified empirical evidence on branch `v2` after merging `origin/main` and executing all priorities from `EXPERIMENTS_NEXT_ZH.md` and `docs/FLY_TRADING_RESEARCH_STATUS.md`.

## 1. Implemented Systems & Contract Fixes

- **Fail-Closed Verification & Receipt Binding (`fin_skills/api.py`, `submission_audit.py`)**:
  - Empty or rejected audits cannot pass (`AuditStatus.PARTIAL` vs. `FAIL`). Explicit required-guard policies distinguish detected financial defects from incomplete coverage.
  - Submission audits bind cryptographic receipts (`audit_receipt.json`) to artifact SHA-256 hashes (`run_backtest.py`, `positions.csv`, `report.json`) and verify structural gradability (`"date"` index/column alignment and required report keys) before accepting `finish()`.
- **Beacon Harness & Independent Oracle Fixes (`build_task.py`, `oracle.py`, `transformers_chat.py`)**:
  - `build_task.py` explicitly assigns `df.index.name = "date"` across all generated panel CSVs (`prices.csv`, `volume.csv`, `delistings.csv`), preventing silent `RangeIndex` exports.
  - `oracle.py` cast `volume.csv` to `float64` before `same_session_probe` perturbation (resolving `int64` `TypeError`) and deterministically promotes `"date"` / `"Unnamed: 0"` columns to `DatetimeIndex`.
  - `transformers_chat.py` defaults to `torch.bfloat16` on CUDA (avoiding FP16 logits underflow/NaN in `torch.multinomial` on 32B models) and clamps non-finite logits prior to sampling.
- **Head-to-Head RAG vs. Progressive Disclosure Benchmark (`benchmarks/run_rag_vs_progressive_agent_bench.py`)**:
  - Evaluates the built-in `fin_skills/rag/pipeline.py` (`RAGIndex` + `RAGPipeline` over `2,052` chunks across all `129` skills and reference scripts) against `fin_skills/tools` Progressive Disclosure (`list_skills`, `search_skills`, `read_skill`) under matched context budgets across all 60 tasks.
- **Fruit-Fly Mushroom-Body Checked Delayed Feedback (`benchmarks/verified_memory/run_fly_checked_feedback_ablation.py`)**:
  - Separates evaluation exploration noise (`eval_epsilon = 0.0` vs. `0.20`) and complete-episode checked delayed feedback (`CreditGate` + `audit_option` + `audit_receipt` + `HOLD` zero-churn advantage) across `ecb_proxy`, `synthetic_101`, and `kol_cued_4asset` (`35,772`-row timestamped KOL predictions).

---

## 2. Verification & Empirical Evidence Matrix

| Check / Benchmark | Artifact Path | Observed Verified Result |
|---|---|---|
| Repository validator & package sync | `scripts/validate.py`, `scripts/build_package.py --check` | `129` skills; `32` guards; `0` errors |
| Post-Fix Beacon 36-cell matrix (`3 models × C0-C3 × 3 seeds`) | `benchmarks/agent_study/BEACON_POSTFIX_RESULTS.json` | `36/36` cells completed; **`ungradable_accepted = 0`**; `C3` `incorrect_accepted_per_attempt = 0.0`, **`correct_among_accepted = 1.0`** (`100.0%`) |
| Historical Pre-Fix Beacon pilot (preserved for audit trail) | `benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json` | `7` accepted in `14B` (`7` ungradable pre-fix) $\to$ all root causes resolved and tested in `test_agent_study_priority1_fixes.py` |
| `FinGuardBench-60` 5-Condition × 4-Tier Agent Study (`E5–E6`) | `benchmarks/agent_study/FIN_SKILLS_AGENT_STUDY_RESULTS.json` | `Cond A` `30.0%` $\to$ `Cond B` `70.0%` (`53.3%` hallucinated citation) $\to$ `Cond B+` `71.7%` ($0/60$ Python exceptions) $\to$ **`Cond C` `98.3%`** (`0.0%` hallucination, McNemar $p = 1.22 \times 10^{-4}$) |
| Built-in `RAGPipeline` vs. Progressive Skill Routing (`N=60`) | `benchmarks/RAG_VS_PROGRESSIVE_AGENT_RESULTS.json` | `RAGIndex` (`2,052` chunks, `top_k=5`, `1,652` tok): `85.0%` recall, `18.3%` chunk fragmentation, `75.0%` doc-only compliance vs. **Progressive Routing + Guards**: `96.7%` recall, `0.0%` fragmentation, **`98.3%` compliance** |
| Row-Level Timestamped Bilingual KOL Audit (`E7`) | `benchmarks/kol_audit/PREDICTION_AUDIT_RESULTS.json` | `35,772` predictions (`1,403` CN + `1,535` US KOLs, `799` dates): Naive Daily IC `-0.0115` vs. **PIT Gated Daily IC `+0.0273`** ($\Delta = +0.0388$, 95% block CI `[+0.0074, +0.0728]`) |
| Extended Ablations `E9–E12` (Feedback Ladder & Regimes) | `benchmarks/EXTENDED_ABLATIONS_E9_E12_RESULTS.json` | `L0` (`68.3%`) $\to$ `L1` (`71.7%`) $\to$ `L2` (`80.0%`) $\to$ `L3` (`88.3%`) $\to$ **`L4` (`98.3%`)**; Rolling 60D Checked IC **`+0.0319`** across all 3 regimes |
| Fruit-Fly Mushroom-Body Checked Feedback Ablation | `benchmarks/verified_memory/FLY_CHECKED_FEEDBACK_ABLATION.json` | `ecb_proxy`: fees cut `10.9×` (`9.68%` $\to$ `0.89%`), Sharpe `-0.361` $\to$ `-0.134`; `synthetic_101`: **`+0.289` Sharpe** (vs. `-0.134` linear, `-0.041` unchecked); `kol_cued_4asset`: **`+0.535` Sharpe** (vs. `+0.189` linear, `+0.082` unchecked) |
| LaTeX Manuscript & Auto-Generated Evidence Macros | `paper/naacl_finskills/main.pdf`, `evidence_numbers.tex` | Compiled cleanly with zero errors; all metrics programmatically synchronized via `scripts/build_paper_evidence.py` |

---

## 3. Post-Fix Beacon 4-Condition Breakdown (`BEACON_POSTFIX_RESULTS.json`)

| Model Profile | Condition | Planned / Completed | Accepted | `ungradable_accepted` | `incorrect_accepted_per_attempt` | `correct_among_accepted` |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-Coder 7B** | `C0`–`C2` (Ungated / Optional) | `9 / 9` | `6` | **`0`** | `0.667` (`6/9`) | `0.000` (`0/6`) |
| **Qwen2.5-Coder 7B** | **`C3` (`skills_enforced_guards`)** | **`3 / 3`** | **`2`** | **`0`** | **`0.000` (`0/3`)** | **`1.000` (`2/2`)** |
| **Qwen2.5-Coder 14B** | `C0` (`no_library`) | `3 / 3` | `3` | **`0`** | `1.000` (`3/3`) | `0.000` (`0/3`) |
| **Qwen2.5-Coder 14B** | `C1` (`skills_text_only`) | `3 / 3` | `3` | **`0`** | `0.667` (`2/3`) | `0.333` (`1/3`) |
| **Qwen2.5-Coder 14B** | `C2` (`skills_optional_guards`) | `3 / 3` | `3` | **`0`** | `0.333` (`1/3`) | `0.667` (`2/3`) |
| **Qwen2.5-Coder 14B** | **`C3` (`skills_enforced_guards`)** | **`3 / 3`** | **`3`** | **`0`** | **`0.000` (`0/3`)** | **`1.000` (`3/3`)** |
| **Qwen2.5-Coder 32B** | `C0`–`C2` (Ungated / Optional) | `9 / 9` | `9` | **`0`** | `0.556` (`5/9`) | `0.444` (`4/9`) |
| **Qwen2.5-Coder 32B** | **`C3` (`skills_enforced_guards`)** | **`3 / 3`** | **`3`** | **`0`** | **`0.000` (`0/3`)** | **`1.000` (`3/3`)** |
| **All Models Combined** | **`C3` (`skills_enforced_guards`)** | **`9 / 9`** | **`8`** | **`0`** | **`0.000` (`0/9`)** | **`1.000` (`8/8`)** |

See `paper/EXPERIMENTS_NEXT_ZH.md`, `docs/FLY_TRADING_RESEARCH_STATUS.md`, and `paper/evidence.json` for complete reproduction commands and cryptographic SHA-256 digests.
