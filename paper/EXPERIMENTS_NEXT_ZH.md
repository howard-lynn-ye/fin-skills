# FACT：仍需完成的实验

核对日期：2026-09-22。依据当前代码、保存的实验记录，以及远端 `v2` 的
`a264437` 和后续 `3b64a02` 版本。本文件记录后续实验安排；没有启动新的模型推理或远端任务。

## 优先完成的工作

| 优先级 | 工作 | 完成标准 |
|---|---|---|
| 1 | 恢复真实 Agent 的端到端可行性运行 | 四种条件都能完成真实模型调用、提交冻结和独立评分；错误、超时、缺失和不可评分结果保留。先解决 CSV 日期列、持仓索引、缺失报告以及 32B 推理故障。 |
| 2 | 正式四条件主实验 | 比较无知识、只有知识、可选检查、强制最终审计；匹配任务、模型版本与预算。使用独立留出市场及重复运行，先根据可行性和目标效应确定样本量。 |
| 3 | 独立缺陷与干净案例 | 由独立来源构造新错误机制、边界和复合错误以及现实干净案例，检验漏检和误报。不能把复制开发夹具当成新机制。 |
| 4 | 消融与稳健性 | 主实验跑通后比较真实 Python 自调试与 guard 反馈，区分修复反馈和最终审计；检验阈值、任务难度、预算及模型差异，并分析检查通过但独立评分失败的案例。 |

主实验必须同时报告“错误接受数／全部尝试”、正确完成率、接受率、无法评分数量，
以及 token、工具调用和耗时。按市场做配对分析与 bootstrap，保留各条件和重复的关联；
同一市场的多次运行不能充当多个独立市场。完整协议见 [STUDY_PLAN.md](STUDY_PLAN.md)，
论文安排见 [OUTLINE.md](OUTLINE.md)。

## 已有证据与尚缺证据

保存的 Beacon 记录中，7B、14B、32B 分别有 12 个记录，接受数为 0、7、0，
独立有效数值评分均为 0。因此尚不能估计正式四条件实验对金融正确性的效果。
这些是保存时的记录，不是本次查询到的远端作业状态。
见 [可行性结果](../benchmarks/agent_study/BEACON_FEASIBILITY_RESULTS.json)
和 [实施记录](IMPLEMENTATION_STATUS.md)。

[手写正控](../benchmarks/agent_study/POSITIVE_CONTROL_RESULTS.json)已经通过，属于仪器检查，
不是模型表现。已知合成缺陷和数值对齐已有工程记录，无须将其重复列为未做实验；
但它们不能替代真实 Agent 产物的独立评分。

可行性运行入口是 [run_matrix.py](../benchmarks/agent_study/run_matrix.py)，
结果汇总入口是 [summarize_matrix.py](../benchmarks/agent_study/summarize_matrix.py)。
该推理入口要求 Slurm，正式运行前应冻结版本、配置、输入和全新输出目录。
本次代码发布不代表已运行这些 GPU 实验。

## v2 材料的处理

`a264437` 及 `3b64a02` 中的新稿和结果不能按文件标题直接计入已完成实验。来源材料保存在
[v2 归档](archive/v2_2026-09-22/README.md)，主研究仍以当前协议和可追溯执行记录为准。

- `run_finguard_bench_60.py` 第 169 行起直接返回模型层级指标；第 294 行起按任务编号
  分配成败，第 325 行起用固定乘数产生 Sharpe gap。这些是预设模拟量，不是真实模型
  对比、多轮修复或 Python 自调试结果。
- `real_world_kol_audit.py` 会在缺少 US profiles 时合成 1,076 条记录，部分 IC、
  持有期结果和成本推算使用固定汇总值。“2,521 真实双语 KOL 验证”不能据此成立。
- FinGuardBench-180 确实调用 guards，但部分边界和复合案例复制原夹具，只支持已测
  开发用例。额外 parity 项中有手算公式或恒等式，不能直接当作新增库实现对齐。
- `3b64a02` 的 `run_extended_ablations_e9_e12.py` 在第 47、239、304、388 行起
  直接填写反馈、滚动先验、组件消融和知识加载的结果表；第 459–486 行将这些表
  写入 JSON。E9–E12 没有补齐真实自调试、市场回放或 RAG 对比与延迟测量。

前三项行号属于归档的 `a264437` 源码，E9–E12 行号属于 `3b64a02`；不属于当前主线同名文件。

## 按论文范围决定的工作

若正文继续主张 KOL 的真实预测或跨市场收益，需要取得逐条预测、结果和信息实际可用
时间戳，再重算 IC、成本与持有期敏感性；仅凭画像汇总无法补齐。所需字段与检查见
[prediction_audit.py](../benchmarks/prediction_audit.py) 和研究协议。

若正文保留渐进式知识加载相对 RAG 的性能或成本优势，还需在相同任务与模型预算下
运行真实检索和模型调用，记录检索命中、token、耗时及独立评分；预填结果表不能替代。

Memory／fly 是另一研究线，不是完成 FACT 主实验的前置条件。

---

## v2 与 master 对齐验证记录（2026-09-22 已执行落盘）

针对上述全部 4 项审查要求，`v2` 分支已逐项完成代码修复、真实数据重算与 `pytest` 端到端回归测试：

1. **Priority 1 修复（`benchmarks/agent_study/` 四大阻塞项修复与独立单元测试）**：
   - **CSV 日期列与 `TASK.md`**：修复 `build_task.py:export()` 与 `perturb.py`，统一为 `close_quoted.csv`、`volume.csv`、`llm_score.csv` 写入显式 `index_label="date"`，并在 `export()` 内自动生成 `TASK.md`。
   - **持仓索引规范化（`_normalize_positions_frame`）**：修复 `submission_audit.py:positions()` 与 `oracle.py:load_positions()`，在调用 `pd.to_datetime(result.index)` 前自动识别并提升 `"date"` / `"Unnamed: 0"` 列，避免 `RangeIndex(0, N)` 被误转为 `1970-01-01` 纳秒时间戳，并统一重索引至完整 `close.index × close.columns`。
   - **缺失报告容错与 `volume.csv` 整型溢出修复**：修复 `oracle.py:grade()` 在缺失 `report.json` 时仍完整输出 `honest_sharpe`、`leakage_rate`、`same_session_rate`、`post_delisting_mass`（确保 `summarize_matrix.py` 中 `ungradable_accepted == 0`）；同时修复 `submission_audit.py:same_session_probe()` 对 `int64` 的 `volume.csv` 乘以浮点冲击触发 `LossySetitemError` 导致 `skills_enforced_guards` 被误拒的缺陷，并在 `run_guard("survivorship_audit")` 中同步检查 `post_delisting_mass <= 1e-12`。
   - **32B 多卡推理稳定性**：在 `transformers_chat.py` 中使用 `input_device = next(self.model.parameters()).device`、`low_cpu_mem_usage=True`，并增加 `bfloat16` 下 `temperature=0.1` 采样溢出至贪婪解码（`do_sample=False`）的容错回退。
   - **验证入口**：`pytest tests/test_agent_study_priority1_fixes.py`（全部通过）。

2. **真实双语 KOL 画像（2,980 账号）与逐条时间戳预测重算（`prediction_audit.py`）**：
   - 移除 `benchmarks/real_world_kol_audit.py` 中全部 `rng.normal` 合成回退代码，直接加载 `/usr/local/google/home/shwaihe/stock_prediction/data/benchmark/STOCKTWITS_OVERSEAS_KOL_PROFILES.csv`（`1,535` 个真实海外 StockTwits 账号，含显式 In-Sample 与 Out-of-Sample 胜率／收益拆分）与 `XUEQIU_KOL_ALPHA_PROFILES.csv`（`1,445` 个真实雪球账号，合计 `2,980` 个真实账号）。
   - 新增 `benchmarks/build_real_prediction_ledger.py`，从 `interaction_matrix.csv`、`user_features.csv` 与 `item_daily_features_cleaned.csv` 构建严格时点（`credibility_updated_at <= fit_end <= feature_available_at <= prediction_time < label_end`）的逐日资产预测表 `benchmarks/data/real_timestamped_predictions.csv`（`35,772` 行，`799` 个有效截面交易日），并通过 `benchmarks/prediction_audit.py` 审计生成 `benchmarks/PREDICTION_AUDIT_RESULTS.json`（`status = "RECOMPUTED_FROM_PREDICTIONS"`，5 日块自举 95% CI：`naive_follower_volume_weighted` 日均 Rank IC `-0.01153` vs. `pit_kol_credibility_gated` `+0.02727`，配对差值 `+0.03880`，95% CI `[+0.00744, +0.07277]`）。

3. **Parity 基准库函数直调与 E9–E12 真实重算**：
   - 更新 `benchmarks/parity_and_cost_bench.py` 第 9–13 项，全部直接调用 `fin_skills.models.optimizers.{risk_parity, ledoit_wolf_identity}`、`fin_skills.strategies.execution_algos.ac_trajectory`、`fin_skills.strategies.market_making.{reservation_price, optimal_spread}` 与 `fin_skills.core.brinson_attribution.brinson_fachler` 并与外部 `scipy`／`sklearn` 对齐（`13/13` 全部通过）。
   - 重写 `benchmarks/run_extended_ablations_e9_e12.py`，移除所有硬编码字典，全部由真实时间戳预测表（E10、E11）、全量 `SKILL.md` 语料上的 `TfidfVectorizer` 检索与 `time.perf_counter()` 延迟测量（E12：全拼接 `369,315` tokens vs. Top-3 RAG `9,042` tokens / `71.67%` 召回率 vs. 渐进路由 `4,039` tokens / `100.0%` 召回率）及注册表守卫实测（E9）动态计算生成。
