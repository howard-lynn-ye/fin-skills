# 本地决策模型：Laya 与 Kev

两个适配器都使用本地权重，不需要 TypeSafe API key。它们返回结构化决策，不生成
自由文本，也不替代可执行审计。模型及可选依赖由调用者在专用环境中准备；基础安装
不会自动下载权重。固定版本及选型依据见
[研究报告](OPEN_DECISION_MODELS_RESEARCH_20260923.md)。

## Laya

在安装了固定上游 Laya 的环境中，先取得所需 checkpoint 的完整本地目录。
英文和多语言模型都使用同一个适配器，分别传入各自路径和 revision。
上游加载器可能更新 tokenizer 兼容性配置，建议使用专用工作副本。

```python
from fin_skills.model_zoo import create_model

model = create_model(
    "laya",
    checkpoint="/absolute/path/to/laya-checkpoint",
    revision="1c5edc17a7acd8701df6fc341c0d179f1c62c982",
    device="cuda",
)
result = model.predict(
    "Find the annual report passage that explains revenue.",
    questions={"tool": {
        "type": "choice",
        "instructions": "Which tool should handle the request?",
        "criteria": {"retrieve": "Retrieve document evidence", "calculate": "Calculate a numerical result"},
    }},
)
```

适配器在执行前检查问题、选项和状态是否会被上游截断；会发生截断则明确报错。
它保留 `raw_response`，只在四位小数舍入误差范围内归一化概率并重算期望分数，
在 `serialization` 中记录变化。超出范围的异常响应仍被拒绝。置信度保持原值，
这个处理不校准概率，也不提升模型准确率。

`provenance` 记录调用者声明的 checkpoint revision、实际设备、输入 token 哈希、
上游代码哈希、加载前后配置哈希和实际温度设置。声明 revision 本身不证明文件身份；
实验部署还必须验证下载清单和权重哈希。

## Kev

Kev 当前固定版本与原 Beacon 环境的依赖不同，使用独立环境。需要完整的本地
adapter/head，以及对应 Qwen 基础模型；缺少本地文件时不会回退到在线下载。

```python
model = create_model(
    "kev",
    checkpoint="/absolute/path/to/kev-4b-adapter",
    revision="485ace8703592fcf405488b262449990824cfed1",
    base_checkpoint="/absolute/path/to/qwen35-4b-base",
    base_revision="1001bb4d826a52d1f399e183466143f4da7b741b",
    device="cuda",
    dtype="bfloat16",
)
```

`head.pt` 必须来自可信制品。加载时核对它声明的基础模型 revision；基础模型路径
只在内存中替换为调用者的本地目录，不改写模型文件。适配器不启用日期事实追加或
prefix cache，记录 dtype、温度及代码身份。请求返回相同的 `raw_response` 与舍入
处理记录。`usage.output_tokens=0` 表示没有逐 token 解码；序列化结果的 token 数
单独记录，并不意味着推理没有成本。Kev-0.8B、Kev-4B 与两个 Laya 版本均已
完成真实 GPU 接入测试，证据和局限见[运行记录](OPEN_DECISION_EXECUTION_20260923.md)。

## 连接内置 RAG

```python
from fin_skills.rag import Document, RAGIndex, RAGPipeline

index = RAGIndex.from_documents([
    Document("report", "Revenue was 100 units.", "synthetic:report"),
    Document("office", "The revenue report describes blue office walls.", "synthetic:office"),
])
evidence = RAGPipeline(index, reranker=model.rerank).prepare("What was revenue?", top_k=2)
```

每个候选片段分别进行一次决策推理。`reranking.receipts` 保存每次响应，片段原有
来源和偏移保持不变；新增字段为 `decision_relevance`、`decision_confidence`。
此例只准备证据，没有调用生成模型。大段材料先按任务固定切分；不能为让某模型通过
而在看到测试结果后改变片段。历史可用性仍由数据库/检索的时间过滤控制。

这些模型需要加载有状态的本地运行时，所以只通过 Python 使用；JSON/MCP 的
`list_models` 可以发现它们，`run_model` 不开放任意本地模型文件加载。
