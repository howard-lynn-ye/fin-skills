# Paper outline / 论文大纲

Working title: **FACT: A Financial Agent Causal Tester and Verified Execution Engine**
(Former working title: Verified Execution and Autonomous Tool Use for Financial Research Agents)

Updated: 2026-09-21. Working Resource/Benchmark outline for an AI/NLP conference
(e.g., NAACL/ACL). The framework, recorded validation and planned agent comparisons are
distinguished below. The experimental design follows the revised [study protocol][S1].

Shared manuscript: https://www.overleaf.com/project/6aad9a03f27c3c07a182965d

## Central argument / 论文主线

**A financial research agent should be evaluated on whether its conclusions survive
execution and financial checks, as well as whether its code runs.** A program can produce
positions and a plausible Sharpe ratio while using information unavailable at the decision
time, excluding historical failures, or reporting performance inconsistent with its positions.
FACT makes these discrepancies observable through domain guidance, executable checks and
independent evaluation of the final research artifact. [S1] [S3]

Packaged as `fin-skills`, the framework connects a source-dated knowledge resource with a
unified audit interface and tools that agents can call during research. The central experiment
asks whether requiring fresh execution evidence for the final artifact reduces incorrect
accepted conclusions beyond text guidance and optional tools, while preserving useful task
completion. The contribution combines a reusable resource, an execution mechanism and a
controlled benchmark; its empirical value is assessed through the comparisons below. [S1] [S2]

**中文主线：金融 Agent 能生成代码和收益数字，还需要证明这些数字经得起检查。**
FACT 将领域知识、实际执行的检查和独立评测连接起来，使研究者能够发现时间使用错误、
历史股票池偏差和报告与账本不一致等问题。论文进一步研究：相比只提供知识或允许模型自行
调用工具，要求最终研究结果附带有效执行证据，能否减少被接受的错误结论，同时保留正确完成
任务的能力。这里的研究对象是 Agent 完成金融研究的过程及结果，贡献不只在于技能或工具数量。

## Research questions / 研究问题

- **RQ1 — Knowledge Injection & Constraint Verification.** Does a fixed excerpt of
  source-checked domain guidance improve agents' use of financial conventions and the
  correctness of their submitted research? The direct comparison is text only versus the
  common baseline, C1 versus C0.
  *在相同任务和工具条件下，加入领域知识能否帮助 Agent 更正确地使用数据、编写研究代码和报告结果？*
- **RQ2 — Executable Auditing.** Which defects do the checks detect, and does requiring
  final-artifact verification improve outcomes beyond making those checks optional?
  Instrument tests establish detection scope; C2 versus C1 assesses optional tool access,
  and C3 versus C2 assesses the additional final-audit requirement.
  *检查工具能发现哪些错误？模型可以自行检查，与提交前必须通过检查，分别带来什么变化？*
- **RQ3 — Agent Correctness, Completion and Cost.** How do accepted-output correctness,
  useful completion, failure modes and resource use vary across models and conditions?
  These outcomes are measured together so that rejection alone cannot count as improvement.
  *不同模型在结果正确性、任务完成和调用成本之间表现如何？哪些错误可以修复，哪些仍会通过检查？*

## Contributions / 贡献定位

1. **The FACT Knowledge Base (Agent Skills).** The recorded resource contains 129 skills
   across 17 plugins, connecting task and library selection with dated sources, conventions,
   failure modes and executable examples. Skills are documents with different scopes;
   their count is not a count of distinct defects. [S2]
   **金融研究知识资源：** 将工具选择、默认行为和方法注意事项组织成 Agent 可按需使用的知识，
   并记录来源与核查日期。
2. **An Executable Audit Engine.** A common interface exposes 36 registered guards and
   reports findings, execution evidence and missing inputs. Final-audit policies bind checks
   to the submitted artifact and distinguish failed checks from incomplete evidence. [S2] [S3]
   **可执行审计机制：** 将领域要求变成可以运行的检查，记录检查针对哪个版本的代码和数据执行，
   避免把“提到了检查”或“旧版本曾通过”当成最终结果已经验证。
3. **The FACT Leak Detection Benchmark & Empirical Audit.** The benchmark separates
   controlled defect detection from agent-generated research. Four paired conditions
   compare knowledge guidance, optional checks and enforced final verification using an
   independent evaluator and explicit completion and correctness outcomes. [S1] [S3]
   **基准与受控评测：** 一部分实验检验工具能否发现已知错误，另一部分检验 Agent 是否能产出
   正确的研究结果，并比较不同干预的收益与成本。

The fruit-fly memory and trading-utility studies remain separate research directions.
They are not needed to establish this paper's core contributions. The wider data, model and
research capabilities remain documented in the [project overview][S2].

---

## Section Outline / 正文结构规划

### 1. Introduction / 引言

A concrete failure motivates the paper: a news item published after a session is joined
to that session's positions because its timestamp names the session it describes. The
program still runs, but its result uses information unavailable when the decision was made.
The benchmark supplies this timing distinction explicitly. [S3]

FACT addresses the gap at three stages: guidance before implementation, checks during
research, and verification of the final submission. The introduction connects these stages
to the three contributions and asks whether they improve correct task completion under
matched budgets. The magnitude and prevalence of agent errors belong in the results.

*引言从一个能直接理解的研究错误展开，再介绍 FACT 如何帮助 Agent 避免、发现并修复错误，
最后落到“正确完成研究”的评价目标。*

### 2. Related Work / 相关工作

The comparison has three relevant lines of work. CheckList provides a precedent for testing
model behavior beyond aggregate accuracy. Profit Mirage already studies information leakage
in financial agents and introduces FinLake-Bench and the counterfactual FactFin framework.
Finance Agent Benchmark evaluates financial research using SEC filings and an agent tool
harness. These works motivate a comparison of evaluation targets and mechanisms. [R1] [R2] [R3]

FACT's proposed distinction is the combination of reusable financial guidance, checks bound
to final research artifacts, and experiments separating text, voluntary checking and final
enforcement. The related-work comparison considers what each system evaluates, which evidence
supports acceptance, and whether the intervention is separated from final scoring. It avoids
assuming that all other evaluators test only syntax or that financial leakage is a new topic.

*相关工作围绕“评什么、如何检查、什么证据决定结果可信”展开；创新点落在具体机制及其受控比较。*

### 3. The FACT Framework (`fin-skills`) / 框架设计

#### 3.1 Source-Verified Knowledge Layer / 带来源的知识层

Each skill links a task or library to relevant conventions, known failure modes, verification
dates, references and examples. Domain skills support tool selection; library skills explain
particular packages. This structure lets a caller select relevant guidance without loading
the entire collection. The agent experiment uses a fixed curated excerpt, so its text effect
applies to that excerpt and task setting. [S2] [S3]

#### 3.2 The Executable Audit Engine / 可执行检查与覆盖范围

`Bundle` gives research artifacts a shared input vocabulary. Coverage inspection identifies
which checks can run and which inputs are missing; execution returns findings and evidence.
A declared required-check policy distinguishes `PASS`, `FAIL` and `INCOMPLETE`. A passing
subset does not satisfy a policy that also requires missing checks. [S2]

| Financial requirement | Representative checks | Evidence examined |
|---|---|---|
| Information is available before the decision. | `assert_causal`, `safe_asof`, `pit_fundamentals` | Signal dependencies, join direction and information availability. |
| Data preserve relevant asset history. | `survivorship_audit`, `pit_universe`, `adjustment_check` | Listings, delistings, corporate actions and price adjustments. |
| Training and evaluation follow the declared separation. | `fold_leak_test`, `purge_effect`, `contamination_probe` | Transformation fitting, label windows and supplied training-cutoff metadata. |
| Numerical conclusions match their stated assumptions. | Accounting checks, `cost_plausibility`, `result_manifest` | Positions, returns, costs and recorded research inputs. |

Two examples explain the mechanism. `assert_causal` changes later inputs and checks whether
earlier outputs move. `contamination_probe` compares a supplied training cutoff with the test
window; its separate behavioral probe requires caller-supplied questions and model responses.
It does not discover a model's training corpus or establish whether a particular example was
memorized. Here, causal testing means checking temporal dependencies under specified
perturbations. [S2] [S6]

#### 3.3 Final-Artifact Verification / 最终研究结果的执行证据

An execution receipt records the checked action and hashes of the code, data, manifest and
numerical claims. Changing the artifact makes previous receipts stale. A reported check can
be unexecuted, executed and failed, or executed on a different version; those cases remain
distinct. The enforced condition runs its public checks on the final submission before
acceptance. Independent scoring follows submission freezing and supplies no repair feedback.
[S1] [S3]

#### 3.4 Tool Integration and Research Workflow / Agent 接入与研究流程

Python interfaces and JSON/MCP tools connect guidance, research artifacts and checks to agent
workflows. Data provenance, information availability and the reference backtest engine provide
supporting inputs and reproducible accounting. The framework figure follows one research
artifact from task guidance through implementation, checking, revision and final scoring,
with a visibly separate path for the independent evaluator. [S2] [S3]

### 4. The Leak Detection Benchmark / 审计引擎有效性验证

#### 4.1 Tasks and Controlled Defects / 市场构造与错误类型

The recorded synthetic run contains 1,825 trading days and 60 assets, with listings,
delistings, splits, news timing and financial-data revisions. Clean and corrupted variants
exercise 12 planted defect families, including future-dependent signals, same-session joins,
survivor-only universes, shared preprocessing and understated costs. The detection matrix
records which checks apply to each defect and which actually detect it. [S4]

#### 4.2 Detection, False Alarms and Coverage / 检出、误报与覆盖

In that recorded run, 13 guards were tested; every one of the 12 planted defect families
was detected by at least one guard, with no recorded false alarms on the clean control.
These are regression results for the specified cases, rather than a test of all 36 guards.
The manuscript reports the matrix, applicable-check coverage and missed individual checks,
alongside the separate numerical-parity evidence. [S4] [S5]

Broader detection evaluation is planned using independently authored defect mechanisms and
realistic clean cases. It will retain the separation between cases used to develop checks
and cases used to assess generalization, and report case counts and uncertainty explicitly.
[S1]

*本节回答“检查工具在什么条件下有效”。已知错误的检出结果和后续独立测试分开报告，
不能用前者直接替代 Agent 实验。*

### 5. Empirical Agent Evaluation / Agent 研究正确性的受控评测 (Core Experiment)

#### 5.1 Task and Experimental Controls / 任务与控制变量

Agents receive a market workspace and produce `submission.py`, whose `build_positions`
function returns positions, plus `report.json` with reported performance and cost assumptions.
The task's frozen manifest and data define the evaluation window and available information.
All conditions share ordinary accounting inspection, the data, and the basic task interface.
[S3]

Markets, model revisions, inference seeds, temperature and turn/token caps are paired across
conditions; execution order is randomized in the frozen protocol. Actual tokens, elapsed time
and hardware use are reported because equal caps need not imply equal consumption. The
recorded feasibility study uses Qwen2.5-Coder 7B/14B/32B on public development seeds. A broader
study will freeze its model snapshots, independently held-out tasks and sample-size rationale
before evaluation. [S1] [S5]

#### 5.2 Four Conditions / 四组对照

| Condition | Domain guidance | Optional library guards | Required final audit | Main comparison |
|---|---|---|---|---|
| C0 — Common baseline | No library excerpt. | Unavailable. | None. | Reference condition with ordinary accounting feedback. |
| C1 — Text only | Fixed curated excerpt. | Unavailable. | None. | C1 versus C0 measures the excerpt's effect. |
| C2 — Optional guards | Same excerpt. | Available through the study adapters. | None. | C2 versus C1 measures optional-check access. |
| C3 — Enforced audit | Same excerpt. | Available through the same adapters. | Public checks must pass on the final artifact. | C3 versus C2 measures final enforcement. |

The current artifact adapters invoke `assert_causal` and `survivorship_audit`. The mandatory
policy additionally checks accounting and dependencies at three public same-session probes.
This experimental intervention is narrower than the full library. Its receipts and final
checks run before the independent oracle evaluates the frozen submission. [S3]

#### 5.3 Outcomes and Denominators / 指标与统计单位

The primary outcome is **incorrect accepted submissions divided by all attempted submissions**.
It is reported with acceptance, correctness among accepted outputs and the count of correctly
completed tasks. An accepted output is not automatically correct, and an output without a
valid independent grade is not counted as a success. [S1] [S7]

| Outcome | Definition and interpretation |
|---|---|
| Incorrect accepted submissions per attempt | Counts accepted artifacts that fail independent correctness criteria, relative to all attempts in a complete batch. |
| Acceptance rate | Reports how often the workflow accepts a submission, regardless of independent correctness. |
| Correctness among accepted outputs | Reports the fraction of accepted artifacts that meet independent criteria; undefined with no accepted outputs or unresolved accepted grades. |
| Error and completion breakdown | Retains rejected submissions, turn limits, provider failures, missing outputs and ungradable artifacts as separate outcomes. |
| Mechanism and cost measures | Reports future/same-session dependence, Sharpe discrepancy, post-delisting exposure, citation fidelity, repair outcomes, tool calls, tokens and time. |

The current development criteria require finite independent metrics, absolute Sharpe
discrepancy at most 0.05, zero measured future and same-session dependence, and post-delisting
mass at most `1e-12`. The last threshold is the summarizer's numerical tolerance for zero.
Raw metrics and threshold sensitivity accompany the binary result. Public accounting uses
the agent's stated costs; the independent oracle uses fixed costs, so discrepancies under
both conventions must be distinguishable. [S1] [S3] [S7]

Every planned cell remains in the record. The current summarizer leaves primary estimates
unavailable when a batch is incomplete or accepted submissions remain ungradable, instead of
treating missing evidence as zero error. Rejected submissions are independently evaluated
when executable, allowing examination of useful outputs that enforcement might reject.
[S1] [S7]

#### 5.4 Comparative Analysis / 结果比较与解释

The main results follow the research questions: C1 versus C0 for knowledge guidance,
C2 versus C1 for optional checking, and C3 versus C2 for final enforcement. Correctness and
completion appear together, followed by error types, repair behavior and resource costs.
Cases with passing public checks but failed independent evaluation are especially informative
about remaining coverage gaps. Tool-call frequency alone does not establish better reasoning.

Repeated evaluations use paired market-level comparisons, preserving repetitions and
conditions within each sampled market. The analysis reports uncertainty and threshold
sensitivity without treating dates, perturbations and agent runs as interchangeable
independent samples. Small feasibility studies remain descriptive. [S1]

#### 5.5 Current Evidence and Remaining Evaluation / 已有证据与后续评测

The saved feasibility record currently supports an assessment of operational completion:

| Model | Recorded cells | Accepted submissions | Valid independent numerical grades |
|---|---:|---:|---:|
| Qwen2.5-Coder 7B | 12 | 0 | 0 |
| Qwen2.5-Coder 14B | 12 | 7 | 0 |
| Qwen2.5-Coder 32B, initial attempt | 12 infrastructure-failure records | 0 | 0 |

The 14B submissions include malformed or missing artifacts that prevented valid grading;
the initial 32B attempt failed before its first tool call. These records do not yet estimate
an effect on financial correctness. The historical pilot remains a separate exploratory
record; it is not pooled with the revised four-condition protocol. Subsequent results will
be reported regardless of whether guidance or enforcement improves outcomes. [S3] [S5]

*本节的核心是“知识、工具和强制检查分别改变了什么”。当前先导记录和正式比较分开，
不预先写成“大模型表现显著下降”或“FACT 显著提升正确率”。*

### 6. Discussion and Limitations / 讨论与局限

The discussion connects the observed tradeoff between correctness, completion and cost to
the design of research agents: when guidance is sufficient, when voluntary checking helps,
and when external verification changes accepted outcomes. Useful negative or mixed findings
identify where tools, interfaces or acceptance policies need improvement.

**Limitations.** Checks cover declared inputs and tested perturbations; passing them does not
establish universal correctness or profitability. Public development data and a fixed guidance
excerpt constrain generalization, and the current feasibility record lacks valid independent
agent grades. Process confinement protects the host but does not establish resistance to
in-process evaluator tampering. Broader financial tasks and other domains require separate
validation. [S1] [S3] [S5]

### 7. Conclusion / 结论

FACT connects financial knowledge, executable checks and benchmark evaluation around the
research artifact an agent actually submits. This enables analysis of both successful task
completion and the evidence supporting its conclusions. The empirical comparison determines
whether optional tools or required final verification improve that combination, and at what
cost, within the tested setting.

*FACT 将“研究结果是否经得起检查”纳入 Agent 的任务目标与评测。最终结论围绕实测的正确性、
完成率和成本展开，说明哪些机制有效、适用范围是什么。*

---

## Planned Figures and Tables / 图表安排

| Display | Role in the paper |
|---|---|
| Figure 1 — Framework and one concrete research artifact | Shows knowledge, implementation, checks, revision, frozen submission and the separately executed oracle. |
| Table 1 — Resource and experimental coverage | Separates the 129-skill/36-guard resource from the fixed excerpt, 13-guard defect run and narrower agent-study adapters. |
| Table 2 — Defect detection matrix | Reports applicable checks, detected and missed defects, and clean controls with explicit case counts. |
| Table 3 — Main four-condition comparison | Presents acceptance, independent correctness, incorrect accepted outputs, unresolved grades and resource use together. Values await complete scored batches. |

Full skill inventories, schemas, individual guard details, frozen prompts and revisions,
historical pilots, per-run outcomes and additional checks belong in the appendix or repository.

## Evidence and Writing Notes / 来源与写作记录

The revised [study protocol][S1] governs the planned comparisons. Its revision followed an
inspected historical pilot and is not described as preregistration. Local implementation and
results are anchored below; numerical results cited here are existing records, not experiments
rerun for this outline. The related-work summaries use the primary records checked on
2026-09-21. This file remains a working outline rather than a submission-ready manuscript.

- [S1] Revised study protocol: conditions, metrics, pairing and evaluation separation.
- [S2] Project overview: recorded resource inventory, architecture and integration scope.
- [S3] Agent-study protocol: artifact interfaces, public checks, independent scoring and pilot.
- [S4] Synthetic detection record: world size, tested guards and defect matrix.
- [S5] Implementation status: numerical validation and recorded feasibility outcomes.
- [S6] Contamination probe source: declared cutoff overlap and optional behavioral probes.
- [S7] Summary implementation: correctness thresholds, missingness and denominators.

[S1]: STUDY_PLAN.md
[S2]: ../PROJECT_OVERVIEW_ZH.md
[S3]: ../benchmarks/agent_study/README.md
[S4]: ../benchmarks/RESULTS.md
[S5]: IMPLEMENTATION_STATUS.md
[S6]: ../plugins/fin-llm/skills/llm-finance-agents/scripts/contamination_probe.py
[S7]: ../benchmarks/agent_study/summarize_matrix.py
[R1]: https://aclanthology.org/2020.acl-main.442/ "Ribeiro et al. (2020), Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"
[R2]: https://arxiv.org/abs/2510.07920 "Li et al. (2025), Profit Mirage: Revisiting Information Leakage in LLM-based Financial Agents"
[R3]: https://arxiv.org/abs/2508.00828 "Bigeard et al. (2025), Finance Agent Benchmark: Benchmarking LLMs on Real-world Financial Research Tasks"
