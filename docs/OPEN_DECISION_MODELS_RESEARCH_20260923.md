# 开源 Jev 类模型：选型研究与 Beacon 接入安排

核查日期：2026-09-23。结论：采用可本地运行的开源决策模型路线。
首选 Kev-4B 作为主要候选，Kev-0.8B 作为小模型对照，Laya 英文／多语言版本作为
轻量对照。四个模型已完成真实接入验证与公开路由回归测试；Kev-4B 有小幅改善但
对选项顺序敏感，BM25 保持默认。详见[本项目实测记录](OPEN_DECISION_EXECUTION_20260923.md)，
这些公开回归结果不构成独立金融任务的质量排名。

这里的“开源 Jev”指社区发布的 Jev 类决策模型。论文、模型注册表和结果文件分别使用
Kev、Laya、NanoJev、Verdict 的真实名称和版本。TypeSafe 的托管 Jev 保留为可选接口，
其 API key 不再阻塞开源路线。

## 研究方法和证据

阅读了作者仓库、模型卡、依赖声明、推理实现和评测说明；通过 GitHub/Hugging Face
API 获取固定 commit、模型 revision、文件清单和公开访问状态。模型权重只在 Beacon
下载，本地保存小型源码与元数据，位于 `runs/open-decision-research-20260923-v1/`。
可跟踪的版本清单见 [OPEN_DECISION_MODELS_LOCK.json](OPEN_DECISION_MODELS_LOCK.json)。
作者报告的准确率、延迟和校准效果均不作为本项目实测结果。

## 候选比较

| 候选 | 发布与接口 | 对我们的用途 | 首轮安排 |
|---|---|---|---|
| [Kev](https://github.com/jaredpalmer/kev) | Qwen3.5 基础模型、LoRA 和决策头；支持 choice、score、noul；提供 Python/CUDA 和兼容 System One 的服务 | 工具选择、少量候选的证据判断、是否需要额外计算 | Kev-4B 为主要候选，0.8B 为资源对照 |
| [Laya](https://github.com/NandhaKishorM/laya) | 英文、多语言和专门微调的三个 checkpoint；Python 直接 predict | 短片段重排、少选项路由；中文用多语言版本单独测 | 先验证英文和多语言两个版本 |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | 0.6B 动作决策模型，发布权重、训练数据及训练流程 | 后续训练金融专用决策器的可复用研究实现 | 暂不拿游戏权重充当通用金融模型 |
| [OpenJev/Verdict](https://github.com/Heman10x-NGU/openJev-verdict-2.0) | README 区分通用 Verdict 与 typed-workflow 的 Verdict 2.0 | 后续小型决策器对照 | 先核对每项成绩对应的权重、推理版本和许可证 |
| [nico-martin/open-jev](https://github.com/nico-martin/open-jev) | 浏览器/TypeScript 推理库，可加载 Kev 或 DeBERTa 的 ONNX 导出 | 将来的浏览器部署 | 与当前 Python/Beacon 主实验需求不同 |

同名项目并不共享同一权重。浏览器库的旧 Kev-0.6B/4B 导出，也不能自动代表当前
Qwen3.5 版本的 Kev。比较模型时必须记录基础模型、adapter、决策头和推理代码。

## 为什么优先 Kev

Kev 提供训练、校准、评测和服务代码，结构化接口与现有适配层接近，便于固定输入和
复用结果检查。作者模型卡明确披露领域外退化、选项顺序敏感性和日期计算缺陷。
这些信息让我们能够提前设计对照，但不构成金融任务有效性的证明。
[作者模型卡](https://huggingface.co/jaredpalmer/kev-4b)

具体接入时有四个容易漏掉的问题：

1. **下载 adapter 不等于下载完整模型。** 4B 与 0.8B 仓库包含 LoRA 和 `head.pt`，
   还依赖对应的 Qwen3.5-Base。`adapter_config.json` 的 revision 为 null，完整基础模型
   revision 记录在 `training_config.json`；部署清单必须同时固定。
2. **服务上限不等于训练验证范围。** 源码默认训练状态/分支预算与服务预算不同。
   服务允许更长输入，也必须单独检查证据保留和长上下文正确率。
3. **日期预处理是额外工具。** `KEV_DATE_FACTS` 可以追加日期差事实。主比较先关闭；
   如果开启，应给其他条件相同日期事实，另列消融，不能归因于模型本身。
4. **返回别名不能当作权重身份。** 服务支持兼容别名，响应中的 model 来自请求。
   实验另保存真实 checkpoint revision 和文件 hash。

以上分别依据固定版本的
[模型实现](https://github.com/jaredpalmer/kev/blob/557598fced1dada75dfbf36ed144dce309ac6ceb/kev/model.py)、
[服务实现](https://github.com/jaredpalmer/kev/blob/557598fced1dada75dfbf36ed144dce309ac6ceb/kev/serve.py)
和本地获取的模型配置。模型卡还说明训练使用交叉熵及开发集校准，不能将所有
“Jev 类模型”统一描述为复现了 TypeSafe 的 RLCD 训练。

## Laya 的价值和限制

Laya 英文版适合短英文输入，多语言版是中文候选。发布的模型卡列出英文约 421M、
多语言约 322M 参数。Hugging Face 元数据确认所检查 checkpoint 为公开、非 gated。
代码和模型卡标注 Apache-2.0；这不替代其基础模型和训练数据各自条款的检查。
[英文模型](https://huggingface.co/convaiinnovations/laya)、
[多语言模型](https://huggingface.co/convaiinnovations/laya-multilingual)

源码会截断状态和选项文本。默认英文总预算为 512 tokens，多语言与 typed-decisions
为 1024；其中一部分还用于问题和选项。不能直接把一整篇财报送入后称为完整文档评测。
需要相同候选片段、显式记录原始/实际 token 数、截断位置与是否保留支持证据。
一次列出整个技能目录也可能把选项描述截短；应先用检索产生相同的小候选集。
[输入构造源码](https://github.com/NandhaKishorM/laya/blob/c7527708f9f5220c669d8aa385077cd28d04708a/laya/common.py)

作者报告明确指出：typed-decisions 上的好成绩来自该数据集训练部分的专门微调；
基础版本在这个任务上弱得多。其 Jev 对比还引用第三方成绩，样本与提示未完全匹配。
发布概率在新领域也可能过度自信。因此不采用“已经超过 Jev”“概率天然可靠”作为
选型依据。首先测未经本项目调参的结果，校准只在独立开发部分进行。
[作者基准与限制](https://github.com/NandhaKishorM/laya/blob/c7527708f9f5220c669d8aa385077cd28d04708a/BENCHMARKS.md)

## 为什么另外两个不先做默认模型

NanoJev 的 `unified-games-v1` 明确提供无需登录的完整下载路径，其模型卡说明主要
训练和评测针对 Maze、Snake 和 ViZDoom。它值得复用为可训练控制器，但游戏成功率
不能换算为金融工具选择能力。代码为 MIT，基础模型保留上游许可；权重与数据的
具体再分发条件应随制品核实。[发布说明](https://huggingface.co/C-Tianyu/NanoJev)

Verdict 仓库首页对应两个不同模型；通用 HF checkpoint 与 Verdict 2.0 的 Git LFS
制品不能混用成绩。GitHub API 对所查 commit 的许可识别返回 `NOASSERTION`，虽然
LICENSE 以 Apache 2.0 开头，但文本与完整标准许可证不一致。这里不作法律判断；
在作者澄清和制品核对前不把它作为默认可再分发依赖。
[固定 LICENSE](https://github.com/Heman10x-NGU/openJev-verdict-2.0/blob/bff28567cff463b833bf044f351a8b7945d53e07/LICENSE)

## Beacon 的实际兼容性

本次通过现有凭据隔离入口读取 Beacon 环境，未执行 GPU 推理：Python 3.12.13、
PyTorch 2.13.0+cu130、Transformers 4.57.6、huggingface_hub 0.36.2、safetensors 0.8.0。

Kev 固定 commit 的依赖声明要求 Python 3.12/3.13、torch >=2.6 且 <2.9、
transformers >=5.17 且 <6，并有新的 PEFT 等要求。它与现有环境存在明确版本冲突，
需要独立环境和真实加载检查。不能为了省安装直接修改声明或覆盖现有实验环境。
[依赖文件](https://github.com/jaredpalmer/kev/blob/557598fced1dada75dfbf36ed144dce309ac6ceb/pyproject.toml)

Laya 声明 torch >=2.0、transformers >=4.48 等依赖，现有版本满足声明下限，
但仍需实际验证。`laya.load` 当前没有 revision 参数，因此先按固定 revision 下载到
RADFM，再从本地目录加载，避免隐式使用浮动主分支。
[依赖文件](https://github.com/NandhaKishorM/laya/blob/c7527708f9f5220c669d8aa385077cd28d04708a/pyproject.toml)、
[加载实现](https://github.com/NandhaKishorM/laya/blob/c7527708f9f5220c669d8aa385077cd28d04708a/laya/agent.py)

源码还有两处适配细节：概率输出四舍五入到四位小数，可能超出现有 Jev 契约对
概率和的容差；正式适配器应保留原始概率并明确处理舍入，不能悄悄改变评分。
加载器包含 tokenizer 配置兼容性改写，因此应记录加载前后的文件哈希，区分下载
制品与实际使用制品。本轮 smoke 用已有严格契约检验，若失败就保留失败记录。

本轮提交作业 **1653341**，测试两个 Laya checkpoint 的真实 CUDA 加载以及
choice/score/noul 返回值是否满足库的结构契约。每个模型使用同样的三个公开编写的
短请求，其中包含中文。它不生成金融正确率排名，不验证概率校准；冷启动计时也不
作为速度优势。源码、权重、缓存、临时文件和调度日志均放在：

`/beacon-projects/radfm/wy891/fin-skills-campaign-open-decision-20260923-v1`

提交回执在 `runs/beacon-campaign-open-decision-20260923-v1/deployment-receipt.json`。
作业 1653341 随后 FAILED / 1:0。加载和推理已进入响应验证，但四位小数概率之和
超出现有契约的容差；completion.json 记录 `probabilities must sum to one`。
这验证了真实兼容问题，不表示模型没有推理能力，也不算集成通过。日志还记录英文
checkpoint 某个温度校准桶被上游代码限制到其允许范围，正式实验须记录实际生效设置。

随后提交诊断作业 **1653362** 到同路径前缀的 `-v2` 目录：保留每个原始响应，再
逐项记录现有严格契约通过/失败，使一个响应的不兼容不会遮掉后续情况；另记录
加载前后哈希。模型、问题与严格契约不变，未归一化概率或修改上游响应。v1 保留。
诊断作业已完成：六个原始响应中四个满足严格契约，两个存在上述舍入问题。
随后完成 Laya 和 Kev 的库适配器及真实 GPU 接入验证；作业、制品和结果见
[运行记录](OPEN_DECISION_EXECUTION_20260923.md)。早期失败制品保留。

## 怎样接入我们的库

新增独立模型 ID：`kev`、`laya`；`jev` 继续表示 TypeSafe 服务。模型注册信息明确
本地/托管部署、checkpoint、支持语言、输入预算和可选依赖。基础安装不自动下载权重。
本地模型直接调用 Python 或同作业内的 loopback 服务，无需 TypeSafe key。

Pipeline 的使用方式是：SQLite 按历史时点选记录 → BM25/向量召回 → 决策器选工具
或给片段评分 → 执行确定性计算/审计 → 保存结果与模型版本。自然语言解释仍由生成
模型提供。`noul` 判断“似乎可用”不能代替数据库的时间过滤；模型说“通过”也不能
替代执行器的审计结果。两类本地模型的 Python 工厂与 RAG 回调已经接通，
完整金融 workflow 的质量收益仍需单独验证。

## 优先实验：让选择由本项目证据决定

| 任务 | 配对比较 | 主要指标 |
|---|---|---|
| 技能／工具选择 | 现有 BM25、固定生成模型、Kev-0.8B、Kev-4B、Laya | 工具选择正确率、参数与执行成功率、最终任务正确率 |
| RAG 重排 | 固定召回集合的原顺序、局部决策器评分、常规重排基线 | nDCG/Recall、最终答案、引用实际支持性、增量耗时 |
| 是否升级给大模型 | 固定阈值／规则与校准后的决策器 | 错误率相同时的覆盖率、成本与总完成率 |
| 输入稳健性 | 相同问题的选项顺序、候选数量、干扰文本、证据位置 | 决策翻转、超长拒绝/截断率、关键证据保留率 |
| 中文与英文 | 分开评测语言，固定翻译与候选内容 | 每种语言的正确率和校准，不把翻译题算作独立任务 |

先保留发布模型的原始结果；如需金融微调，训练、校准与最终评测按报告/来源分组，
避免相邻片段跨集。FinQA/FinanceBench 可用于各自适合的外部任务，不能把参考答案
或 gold evidence 作为待比较检索器的输入。金融数值真值来自独立执行；检索排序标签
与答案支持性分开测。

比较时固定候选集合、可用时间、信息量和最大调用预算，记录实际消耗。模型不同的
tokenizer 使“相同 token 数”不一定表示相同内容，因此还保留实际文本和截断信息。
报告加载与下载成本、预热后的延迟分布、吞吐、显存和任务正确率；CUDA 计时须同步，
缓存命中单独分组。没有生成 token 不等于没有计算或输出序列化成本。

论文中的主张应是“本库可组合并实测开源决策模型在金融 workflow 中的效果”。
模型本身来自上游，是否提高任务质量、降低成本，等待冻结实验的结果。
