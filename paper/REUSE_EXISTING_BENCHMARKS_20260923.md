# 复用公开基准和现成实现的补实验方案

核查日期：2026-09-23。本文记录官方仓库、论文及文档的在线核查，不是新实验结果，
也不代表已完成 Beacon 安装、数据完整性检查或正式推理。已完成的 2,112 回合保持不变。

此前将独立任务、答案与证据标注统一列为“需要外部输入”，范围过大。公开第三方基准
已经能补上一部分证据，不必等待我们重新招募标注者。公开基准不等于模型从未见过的
私有留出；真人效率和新时间段的前瞻评估仍是不同问题。

## 优先复用的资源

| 优先级 / 缺口 | 已核查的资源 | 直接复用什么 | 在本研究中的用途与边界 |
|---|---|---|---|
| P0：真实金融问题和客观答案 | [FinQA，EMNLP 2021](https://github.com/czyssrs/FinQA) | 财报表格/正文、问题、参考程序、执行答案、支持事实及官方评估脚本 | 工具选择与数值执行的外部作者基准；用官方程序评分，不重新自造数值标签 |
| P0：答案和证据标注 | [FinanceBench](https://github.com/patronus-ai/financebench) | 公开的 150 个案例、人工答案/理由/证据页、原 PDF，以及论文的已标注模型输出 | 检索、回答正确性与证据定位；不能把整个 10,231 题库说成已公开，也不能认为旧标注已经覆盖我们新模型的所有自由文本断言 |
| P0：成熟工具调用基线 | [LlamaIndex ReActAgent](https://developers.llamaindex.ai/python/examples/agent/react_agent/) | 实际 ReActAgent、FunctionTool 与工具执行循环 | 在同一模型/信息/预算中比较原生工具、加领域说明、加 FinSkills；必须真的运行该框架，不能将自写循环改名为 LlamaIndex |
| P0：普通经验记忆强基线 | [Reflexion，NeurIPS 2023](https://github.com/noahshinn/reflexion) | 作者的反馈、反思和后续尝试代码及轨迹 | 接入同一金融环境，明确标为金融适配版；原方法不是现成的金融跨公司迁移基准 |
| P1：长文档上下文 | [DocFinQA，ACL 2024](https://aclanthology.org/2024.acl-short.42/)；[Kensho 数据](https://huggingface.co/datasets/kensho/DocFinQA) | 完整财报上下文及问题/程序/答案、既有数据划分 | HiSTrim 对比 full/recency/retrieval；FinQA 与 DocFinQA 有题目来源重合，必须按公司/报告/问题去重，不能冒充两个独立任务集 |
| P1：多轮金融上下文 | [ConvFinQA，EMNLP 2022](https://github.com/czyssrs/ConvFinQA) | 多轮问题链、公开训练/开发标注、评估提交格式 | 测连续对话的状态和计算链；不把同一对话历史记忆当成跨任务经验学习。官方说明测试集答案不公开，不能默认可在本地完整评分 |
| P1：金融专用框架 | [FinRobot 官方仓库](https://github.com/AI4Finance-Foundation/FinRobot) | 已开源的金融 agent、工具与工作流 | 作为独立系统基线；固定具体版本。当前 README 区分 V0/V1 开源与 V2 未开源，不能统称最新版均可复现；示例涉及外部数据 API |
| P1：真实时间序列和分层记忆 | [FinMem 作者仓库](https://github.com/pipiku915/FinMem-LLM-StockTrading)；[InvestorBench，ACL 2025](https://github.com/felis33/INVESTOR-BENCH) | 原有分层记忆、warmup/train-test、交易动作、检查点和评估流程 | 复用框架后核实历史数据覆盖/可获得性。二者原配置依赖 OpenAI embeddings；若统一换成本地编码器，写清这是受控适配，而非原文成绩复现 |
| 可选：统一金融评估入口 | [FinBen / PIXIU](https://github.com/The-FinAI/PIXIU) | 已有金融评估任务与运行框架 | 先复用相关任务和评分模块，不为扩大数字而把全部任务再跑一遍 |

FinQA 官方仓库专门记录过检索格式导致标签泄漏的修复。应固定修复后的版本，并从
模型输入中剥离 `program`、`exe_ans`、`gold_inds` 等评分字段；这不是只把文件名改成
test 就能保证的隔离。采用 train/dev 做适配与校准，冻结后才运行公开 test，记录
模型训练污染未知。官方 private-test 提交通道应另行确认可用性，不先承诺成绩。

## 怎么复用到共同实验中

第一步先接 FinQA 的官方评分器和 FinanceBench 的公开证据标签，跑小型开发校准，
确认模型看到的是问题与可用文档、评分器看到的是答案。现成 LlamaIndex agent 负责
工具调用，所有条件共享相同的底层计算函数、文档和响应预算。FinSkills 的增量是
工具组织、说明或检查机制，不能偷偷增加信息或换更强模型。

第二步将 Reflexion 原有经验流程适配到同一任务环境，保留无记忆、冻结记忆、普通
反思/检索、fly 条件。在训练/开发问题中获得反馈，之后固定经验库，在不重叠报告
的评估问题上观察方法选择与成功率；若另测在线更新，单独预注册反馈何时可见。
不得将后一道测试题的答案提前写入记忆，也不使用跨组共享经验。先核实数据组间
重合，再冻结 task/episode ID；不凭更多随机种子宣称跨任务泛化。

第三步在真实长文档上测 HiSTrim。原来的 8,192 输入约束不能直接容纳 DocFinQA 的
完整报告；应分别定义完整文档长上下文协议与共同候选检索包协议。超过容量的样本
必须计入容量失败/覆盖率，不能悄悄截断后仍称 full context。报告实际保留证据、
答案、prefill/生成时间、显存与 KV，保持共同可见输入和长度上限。

成熟框架的系统对比与上述模块消融分表报告：前者回答整个系统效果，后者才能较清楚
地归因到 FinSkills、记忆或压缩。FinRobot、FinMem、InvestorBench 不应仅因为名称
含 finance 就塞进不支持的任务接口；先使用它们原本支持的任务/模块。

## Jev 的实际接入和替代基线

[TypeSafe 官方 quickstart](https://docs.typesafe.ai/introduction/quickstart)
明确提供 `typesafe-sdk`，从控制台取得 API key，使用认证的 `/v1/systemone`。
[官方模型文档](https://docs.typesafe.ai/models)列出可固定的 `jev-1.13.0`。
因此可以复用官方 SDK，毋须继续手写传输客户端，但 SDK 不会消除凭据要求。
本次没有从官方资料确认无需认证的本地 Jev 权重方案。

检索实验不必因此全停。可使用作者发布的
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
作为本地交叉编码器重排基线，和现有 BM25/E5 在相同候选文档上比较。它是单独的
BGE 基线，不称为 Jev 结果，也不假定能替代 Jev 的全部结构化决策功能。

## 本次排除的捷径与真正剩余的输入

- FinAgentBench 的已核查 arXiv v1 表示接受后将公开数据；本次未确认作者发布的可用
  下载。搜索到同名/近名 Hugging Face 数据不能自动当作这篇论文的数据。
  [论文来源](https://arxiv.org/html/2508.14052v1)。
- FinMem 使用作者 `pipiku915` 仓库作为来源；搜索到的越南市场扩展 fork 不是
  原论文实现的直接替代，不能借用其配置却声称复现作者原版。
- LLM-as-judge 可以作为另列的辅助指标，但不能冒充独立人工内容支持标注。
- 真人省时实验仍需要参与者；公开答案不能替代人实际完成任务的耗时。
- 固定历史基准可以检验历史时间划分，但不能证明模型不知道历史行情；前瞻时间外
  泛化应另行设计。交易比较须统一成本、执行时点、仓位、风险暴露和换手口径。

执行路线调整为“上游数据 + 上游评分器 + 现成 agent/记忆实现 + 最小适配层”。正式
Beacon 批次前固定上游 commit、数据文件 hash、许可证/来源、输入与标签隔离及预算。
下载、安装、校准和推理仍只在 Beacon/RADFM，动态分配节点。当前文档更新不产生
新的提交回执，不改变旧实验，也不宣称上述新方案已经跑通。
