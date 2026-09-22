# 内置 RAG 与模型流水线

更新：2026-09-22。公开入口为 `fin_skills.rag` 和 `fin_skills.model_zoo`。

库中现在有可执行的 RAG 组件：接收文档、分块、检索、可选重排、组装带来源的上下文，
再调用用户提供的文本生成函数。原来的 `fin_skills.find()` 仍用于简单的技能文本搜索。
生成模型和 embedding 模型由调用方选择，导入或构建默认检索索引不会联网。

## 直接使用库内知识

```python
from fin_skills.rag import RAGIndex, RAGPipeline

index = RAGIndex.from_skills(["backtest-validation"], include_references=True)
prepared = RAGPipeline(index).prepare("survivorship universe", top_k=3)
print(prepared["context"])
print(prepared["citations"])
```

省略技能名称会读取全部已打包的技能。来源使用 `fin-skills:技能名/文件名` 标识，
从已安装的包读取，不要求用户保留项目源码目录。`include_references=True` 同时载入参考文件。

`prepare()` 返回 `messages`、`context`、`passages`、`citations` 和 `no_evidence`。
外部 Agent 可以直接使用这些值；无需先接入某一家模型服务。

## 自己的文档与生成函数

```python
import json
from fin_skills.rag import Document, RAGIndex, RAGPipeline

documents = [Document(
    id="retrieval-note",
    text="RAG retrieves source documents and preserves citations.",
    source="team-manual/retrieval",
    metadata={"version": "1"},
    available_at="2026-09-01T10:00:00Z",
)]
index = RAGIndex.from_documents(documents, chunk_size=1200, overlap=200)

def extractive_demo(messages):
    # 离线提取示例，用于验证接线；不是模型生成或质量评测。
    context = json.loads(messages[-1]["content"])["retrieved_context"]
    return context.splitlines()[1] + " [S1]"

pipeline = RAGPipeline(index, generator=extractive_demo)
result = pipeline.answer("source documents", top_k=1)
print(result["answer"])
print(result["citation_check"])
```

生产流程将 `extractive_demo` 换成自己的 `generator(messages) -> str`。
回调接收普通聊天消息列表，可以调用本地模型或已授权的托管模型。
文档也可以是含 `id`、`text` 的字典；支持 `source`、`metadata`、`available_at`。
重复文档 ID 会报错。读取文件、PDF 提取、网页抓取由上游组件完成，RAG 接收其文本结果。
`documents_from_skills(...)` 返回同一种 `Document`，可与用户文档合并建索引。

## 检索、向量与保存

默认 `method="lexical"` 使用 BM25；这是词项检索，不冒充语义检索。
分块按 Unicode 字符数和重叠长度进行，保留文档 ID、来源、元数据、可用时间以及
原文中的 `start`/`end` 字符偏移。检索返回的内容可以逐段核对原文。

语义检索通过显式 `embedder(list[str])` 回调接入。它必须返回每段文本一行、维度一致、
有限且非零的数值矩阵。构建时传入 `embedding_id` 标记坐标空间，再用
`index.search(query, method="vector")` 或 `pipeline.answer(query, method="vector")`。
库计算归一化余弦分数，不安装或下载 embedding 模型。回调的联网行为由调用方控制。
两种检索都只保留分数高于 `min_score` 的命中，默认阈值为零；这些分数不是正确率。

```python
# index 是上文建立的索引；保存路径必须尚不存在。
index.save("research_index.json")
restored = RAGIndex.load("research_index.json")
hits = restored.search("source documents", top_k=3)
```

保存格式为带版本的 JSON，包含文档、分块配置，以及已生成的向量；不保存可执行回调。
向量索引加载后要继续做向量查询，必须重新传入 `embedder` 和匹配的 `embedding_id`。
这个 ID 是调用方的模型版本声明，不是 embedding 模型内容的认证。
当前实现是内存索引，JSON 读写有大小限制，不是分布式向量数据库。

## JEV 和果蝇记忆使用相同入口

```python
from fin_skills.model_zoo import create_model

# TYPESAFE_API_KEY 在运行环境中设置；显式允许发送检索片段给 TypeSafe。
jev = create_model("jev", allow_network=True, model="jev-latest")

# generator 使用上文的离线示例，也可以换成自己的文本生成回调。
pipeline = RAGPipeline(index, generator=extractive_demo, reranker=jev.rerank)
result = pipeline.answer("source documents", top_k=1, fetch_k=3)
```

此处 JEV 指 TypeSafe Jev。接入依据是 2026-09-22 核对的
[官方 API 文档](https://docs.typesafe.ai/api)和
[快速开始](https://docs.typesafe.ai/introduction/quickstart)。
JEV 提供 `choice`、`score`、`noul` 结构化决策，不生成自由文本。
本库的 `jev.rerank()` 将候选片段的相关性作为 `score` 问题提交，保留原始来源，
返回 `jev_relevance`、`jev_confidence`、实际模型标识和 token 用量。
也可单独使用 `jev.predict(state, questions=...)` 做流程选择或分类。

库提供的是 JEV 适配代码，托管服务需要 API key 和访问资格；本库不附带它的模型权重。
`create_model("jev")` 本身不会联网，真实调用须显式开启。
回调 `transport(payload, *, timeout)` 可用于自己的客户端或模拟测试，且自行控制联网。
错误不会静默替换成模拟答案，库也不自动重试请求。

果蝇记忆仍通过 `create_model("fly_memory", circuit_parameters=...)` 创建，
使用独立安装的 GPL `fin-skills-fly` 扩展，保留原来的反馈检查和参数要求。
JEV、果蝇记忆及其他模型可以组合进同一 Python 流水线；不需要把它们伪装成同一种模型。
具体安装与调用见[多模型指南](MODEL_USAGE.md)。

## 来源、上下文预算和时间边界

`max_context_chars` 限制 `context` 的字符数，包括来源标题与分隔符；它不是模型 token
预算，也不包含系统提示和问题文本。只有完整放得下的片段进入上下文，超长片段不会截断。
`fetch_k` 控制重排前的候选数，须不小于 `top_k`；重排器不能修改原文、来源或原始分数，
也不能返回不存在或重复的片段 ID，但可以返回候选子集和新增评分字段。

没有命中时，`answer()` 返回 `status="no_evidence"`、`answer=None`，不调用生成函数。
生成后的 `citation_check` 只检查 `[S1]` 形式的引用标签是否来自本次上下文；
不存在或遗漏的标签会被标记。它不证明引用内容支持答案，也不保证模型不会遵循恶意文档指令。

历史查询可传入带显式时区的 `as_of`。只有 `available_at <= as_of` 的文档参与检索，
缺少可用时间的文档会被排除。BM25 的语料统计也只用当时可用的文档，避免未来文本改变排序。
这些时间来自调用方提供的来源记录，库不能证明其真实性。技能的 `verified_on`
不能代替真实可用时间，因此没有默认把当前技能库当作历史时点的已知信息。

## JSON/MCP 与验证

```python
from fin_skills.tools import call_tool

context = call_tool("retrieve_context", {
    "query": "source documents",
    "documents": [document.to_dict() for document in documents],
    "top_k": 1,
})
```

`retrieve_context` 在 JSON/MCP 中提供离线 BM25 检索和上下文组装。未传 `documents`
和 `skills` 时使用库内知识；显式传 `documents=[]` 表示空语料，不会偷偷加入全库。
向量、重排和生成回调使用 Python API。JEV 无状态调用另由已有的 `run_model` 提供，
参数只接受模型标识、超时和显式联网开关，密钥不放入工具参数。

可直接运行[离线完整示例](../examples/rag_pipeline.py)：

```bash
python examples/rag_pipeline.py
python -m pytest -q tests/test_rag.py tests/test_jev.py tests/test_tools.py tests/test_mcp.py
python -m pytest -q -m slow tests/test_examples.py
python scripts/check_distribution.py
```

示例中的 JEV transport 和文本生成函数均标注为离线测试桩。
这些检查用于验证接口、来源、数据约束和打包，不是 JEV 真实推理、RAG 检索质量、
延迟或金融收益的实证实验。

### 本次验证记录（2026-09-22）

Windows 基础依赖环境使用 Python 3.11.7、NumPy 1.24.0、pandas 2.2.2、SciPy 1.11.4。
实际执行结果如下；独立安装包检查使用的是它自己解析的依赖版本。

| 检查 | 结果 |
|---|---|
| 默认完整 pytest 回归 | 3260 passed，351 skipped，51 deselected；62 warnings |
| 离线示例测试 | 13 passed |
| wheel 与源码包 | 严格元数据检查通过；分别在新环境安装并在仓库外运行，`DISTRIBUTION_OK` |
| 果蝇可选扩展 | 隔离构建并安装 `fin-skills-fly`；实际后端的反馈、重复更新与保存加载测试 1 passed |
| 目录与生成包校验 | `validate.py` 输出 OK；129 个技能，保留 1 条技能发现预算建议警告 |

默认回归中的跳过包含未安装的可选后端及不适用用例，慢测试按默认配置排除。
果蝇检查使用合成电路参数验证接口；JEV 检查使用模拟 transport，没有发送真实推理请求。
检索质量、真实 JEV 重排收益与延迟仍需另行设计带独立标签和固定模型版本的评测。
