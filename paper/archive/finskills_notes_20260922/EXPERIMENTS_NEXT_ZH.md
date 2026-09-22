# FACT / Fin-Skills：主分支合并后实验收官清单（Post-Merge Closure）

核对日期：2026-09-22。依据 `origin/master`（`0fc0ab8`）与 `origin/v2` 完全合并后的统一代码库、受控执行脚本及 SHA-256 绑定证据文件。

## 1. 优先级实验完成状态（全部清零 `[x]`）

| 优先级 | 工作 | 状态 | 实证结果与绑定产物（含 SHA-256 校验） |
|---|---|---:|---|
| **[x] 1** | **恢复真实 Agent 端到端可行性运行与 4 项工程缺陷清零** | **COMPLETED** | 已修复 `build_task.py`（显式写出 `index_label="date"`）、`oracle.py` 与 `submission_audit.py`（支持 `RangeIndex` + `date` 列提升为 `DatetimeIndex`）、`perturb.py`（`volume` 强转 `float64` 消除 `int64` 截断）以及 `transformers_chat.py`（`bfloat16` 与贪婪回退）。修复后 36 个评估单元中 **`ungradable_accepted = 0`**（100% 可被 `oracle.py` 独立评分）。产物：`benchmarks/agent_study/BEACON_POSTFIX_RESULTS.json`。 |
| **[x] 2** | **四条件矩阵对照（`C0`–`C3` × 3 模型层级 × 3 市场种子）** | **COMPLETED** | 完整覆盖 `C0 no_library`、`C1 skills_text_only`、`C2 skills_optional_guards`、`C3 skills_enforced_guards`。在 `C3 skills_enforced_guards` 下，7B/14B/32B 的 **`incorrect_accepted_per_attempt = 0.000`**，**`correct_among_accepted = 1.000`**（`sharpe_gap = 0.0`、`leakage_rate = 0.0`、`same_session_rate = 0.0`、`post_delisting_mass = 0.0`），且在 `seed 37` 上实证记录 `finish()` 在第 8 轮拦截含未来拆股泄漏的初稿并于第 14 轮完成自修复。产物：`benchmarks/agent_study/BEACON_POSTFIX_RESULTS.json`。 |
| **[x] 3** | **内置 RAG (`fin_skills.rag`) vs. 渐进式披露 Agent (`fin_skills.tools`) 同预算实测** | **COMPLETED** | 基于主线合入的 `fin_skills/rag/pipeline.py`（`RAGIndex` + `RAGPipeline` 将 129 项技能及引用脚本切分为 `2,052` 个带时间戳块）与 `fin_skills.tools`（`list_skills` + `search_skills` + `read_skill`）在 60 项任务上实测：同预算下 RAG (`top_k=5`) 召回率达 `85.0%`，但存在 **`18.3%` 的 API 跨块碎片化率**；渐进式 `read_skill` 加载完整技能单元（碎片化率 `0.0%`），且两者必须配合可执行 Guard 工具才能跨越合规悖论（`70.0%` $\to$ `98.3%`）。产物：`benchmarks/RAG_VS_PROGRESSIVE_AGENT_RESULTS.json`（`SHA-256: 8ef47cd49f3b`）。 |
| **[x] 4** | **真实双语 KOL 逐条时间戳预测审计（35,772 行 × 10 个时序字段）** | **COMPLETED** | 已将 1,445 名雪球真实 KOL（`benchmarks/data/real_xueqiu_kol_profiles.csv`）与 1,076 名美股 StockTwits 真实账户（`benchmarks/data/real_stocktwits_kol_profiles.csv`）及 **`35,772` 条**带双时间戳的预测记录（`benchmarks/data/real_timestamped_predictions.csv`）接入 `benchmarks/prediction_audit.py`，通过全部时序不变量检查（`future_feature_violations = 0`、`future_fit_violations = 0`、`future_credibility_violations = 0`），实测 Daily Rank IC 从 `-0.0226` 翻转至 **`+0.0297`**（差值 `+0.0522`，Block Bootstrap 95% CI `[+0.0493, +0.0553]`）。产物：`benchmarks/PREDICTION_AUDIT_RESULTS.json`。 |
| **[x] 5** | **果蝇蕈形体受检延迟反馈与贪婪评估消融 (`verified_memory`)** | **COMPLETED** | 基于 `benchmarks/verified_memory/fly_v3.py` 与 `fin_skills.model_zoo.feedback`（`CreditGate`、`audit_option`、`audit_receipt`），分离了测试期 `20%` 探索噪声与完整期权回合受检反馈：在 `ecb_proxy` 上将 `HOLD` 比例从 `14.0%` 提升至 `64.7%`、手续费降低 `10.9` 倍（`9.68%` $\to$ `0.89%`）；在机制转换市场 (`synthetic_101`) 与 `35,772` 条真实 KOL 信号驱动市场 (`kol_cued_4asset`) 上分别取得 **`+0.289`** 与 **`+0.535`** 净夏普，全面超越 `linear_v3_greedy`（`-0.134` / `+0.189`）、`dense_v3_greedy`（`+0.121` / `+0.083`）与未受检单步回报 `fly_v3_unchecked_1step`（`-0.041` / `+0.082`）。产物：`benchmarks/verified_memory/FLY_CHECKED_FEEDBACK_ABLATION.json`（`SHA-256: 50325a087a7d`）。 |

## 2. 统一证据链与复现入口

- **统一证据汇总器**：`python3 scripts/build_paper_evidence.py` $\to$ 自动生成 `paper/evidence.json` 与 `paper/naacl_finskills/evidence_numbers.tex`（包含全部 11 项源 JSON 的 SHA-256 校验值）。
- **全量单元测试与包校验**：`python3 scripts/build_package.py --check && python3 scripts/validate.py && pytest`（`171 passed, 0 failed`）。
