# 多模型使用指南

fin-skills 面向直接编写 Python 的用户，也面向自主选择工具的智能体。
统一模型入口是 `fin_skills.model_zoo`；原有 `fin_skills.algorithms` 接口继续可用。
本指南介绍调用与功能验证，不提供收益保证或预训练权重。

Qlib 策略回测、FinRL 交易环境与 AutoGen 自主工具调用见[完整流程接入](REUSED_WORKFLOWS.md)。

## 统一模型入口

```python
from fin_skills.model_zoo import model_catalog, create_model, load_model

cards = model_catalog()  # operations、status、依赖、许可证、输入和限制
model = create_model("ridge", seed=7).fit({"X": X_train, "y": y_train})
prediction = model.predict(X_new)
model.save("ridge_zoo.zip")
restored = load_model("ridge_zoo.zip", trusted=True)
```

每个模型卡的 `operations` 列出支持的操作。`run` 用于直接执行；`fit` / `predict`
用于估计模型；果蝇记忆支持 `predict` / `update`。不会把缺少某种操作的模型假装成通用预测器。
`model_zoo.load_model` 加载本节保存的产物；原 `algorithms.load_model` 继续加载旧格式。

新增入口包括：

| ID | 支持的功能 | 依赖与范围 |
|---|---|---|
| `lstm_forecast`、`transformer_forecast`、`patchtst_forecast`、`nhits_forecast` | 训练、多步预测、保存加载 | NeuralForecast 原生模型；无预训练权重 |
| `ppo`、`sac` | 环境训练、动作预测、保存加载 | Stable-Baselines3；用户提供 Gymnasium 环境 |
| `fly_memory` | 记忆读出、经过检查的完整过程反馈更新、保存加载 | 单独安装 GPL 扩展，显式提供电路参数 |
| `jev` | 结构化决策、概率评分、检索片段重排 | TypeSafe 托管 API；需要 API key；本库包含适配器，不分发权重 |
| `kalman_filter` | 固定参数的前向状态滤波 | 基础依赖；不包含使用未来数据的平滑器 |
| `ledoit_wolf_covariance`、`ewma_covariance`、`pca_covariance` | 协方差估计 | 基础依赖；传入小数收益 |
| `nelson_siegel`、`svensson` | 期限结构拟合 | 基础依赖；横截面曲线拟合不是未来收益预测 |
| `merton_credit` | 结构信用模型反解 | 基础依赖；风险中性违约概率 |
| `svi_surface` | 单个到期日的 SVI 曲面切片拟合 | 基础依赖；拟合不自动证明无套利 |

原算法注册表的模型也能通过 `create_model` 调用。JSON/MCP 新增 `list_models`
和 `run_model`：后者只执行模型卡中 `json_run=True` 的方法，不加载磁盘模型或任意代码。
深度学习、RL、记忆模型的有状态生命周期目前使用 Python API。

## 深度学习与强化学习

在需要这些后端的环境中安装相应 extras：

```bash
python -m pip install -e ".[deep,rl]"
```

深度模型直接复用 [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/)，
输入为恰好包含 `unique_id`、`ds`、`y` 的 pandas DataFrame。`ds` 必须为无时区的日期类型，
各序列时间递增、不重复且符合显式 `freq`；每条训练序列至少有 `input_size + h` 个观测。
交易所节假日需要调用方提供正确的日历，不能把缺失交易日静默补齐。
全部观测须在训练截止时已经可用，评估段留在调用方；当前入口不接收外生变量。
网络结构、优化器和缩放使用上游实现，训练默认在 CPU 上运行。

```python
sequence = create_model("lstm_forecast", freq="B", h=5, input_size=24,
                        max_steps=100, seed=7, encoder_hidden_size=32)
sequence.fit(training_df)
prediction = sequence.predict()  # 训练历史之后的 5 个日历步
# 也可 predict(new_history_df)，使用既有权重对新历史预测。
# 改用其他三个模型时保留数据契约，结构参数按各自上游接口传入。
```

NeuralForecast 的原生保存加载负责权重与配置；本库外层记录依赖版本和摘要。
产物也包含训练历史，使用与训练数据相同的访问权限。`fit` 每次重新训练，
不隐式续训；上游训练会设置随机种子，可能影响进程的随机数状态。

PPO 和 SAC 使用真实的 Stable-Baselines3 后端。模型接口接收环境对象，不把数组伪装成
完整交易环境。环境负责观测、成交、奖励与终止规则；SAC 要求连续动作空间。

```python
policy = create_model("ppo", seed=7, n_steps=128, batch_size=64)
policy.fit(training_env, total_timesteps=4096)
action = policy.predict(observation, deterministic=True)
policy.save("ppo_zoo.zip")
```

保存的 RL 产物用于恢复推理策略，不包含完整重放缓冲或环境状态；这不是精确恢复训练任务。
`fit` 开始一轮新训练。动作不是经纪商订单，接口不会自动交易。

## 果蝇实验扩展

已有记忆方程以独立的 `fin-skills-fly` 包提供，保留 GPL-3.0-or-later 声明和上游来源。
核心 wheel 不包含该扩展代码或拟合参数。安装与使用：

```bash
python -m pip install ./benchmarks/fly_paper
```

```python
from fin_skills_fly import load_parameters
from fin_skills.model_zoo import create_model
from fin_skills.model_zoo.feedback import NavMark, IntervalCredit, OptionReceipt

parameters = load_parameters("published_parameters.mat")  # 用户提供的作者参数文件
memory = create_model("fly_memory", circuit_parameters=parameters)
scores = memory.predict()  # 两个线索的记忆读出，不是收益预测
result = memory.update(receipt, now=decision_time)
```

`receipt` 使用 `OptionReceipt` 表示完整决策过程，各 `IntervalCredit` 连接含时点、来源
和扣费账户净值的 `NavMark`。反馈必须严格早于 `now` 可用，负收益也更新；无效或未到达的
反馈不改变记忆，同一回执只应用一次。独立账本仍需核对净值来源是否真实。
本适配器暴露记忆组件，不把它宣称为完整自主交易策略，也不附带预训练金融权重。

## JEV 结构化决策与检索重排

JEV 与果蝇记忆使用同一个 `create_model` 入口，承担不同的流水线步骤。
这里的 JEV 指 TypeSafe Jev；接口按 2026-09-22 的
[官方 API 文档](https://docs.typesafe.ai/api)接入。它返回 `choice`、`score`、`noul`
三种结构化答案，不生成自由文本。`score` 是各等级的概率加权值，`noul` 是肯定答案的概率。

```python
from fin_skills.model_zoo import create_model

# 先在运行环境设置 TYPESAFE_API_KEY；本次调用会把 state/questions 发给 TypeSafe。
jev = create_model("jev", model="jev-latest", allow_network=True, timeout=30)
result = jev.predict(
    {"question": "Which component should retrieve supporting documents?"},
    questions={
        "component": {
            "type": "choice",
            "instructions": "Choose the component responsible for document retrieval.",
            "criteria": {"rag": "Document retrieval", "memory": "Feedback-driven memory"},
        }
    },
)
print(result["answers"]["component"])
print(result["model"], result["usage"])
```

创建模型和导入库不会联网；实际托管调用需要显式 `allow_network=True`。
也可传入 `transport(payload, *, timeout)` 回调，使用调用方自己的客户端或离线测试桩。
自定义 transport 控制自己的联网行为，库不会把环境中的密钥传给它。
没有自动安装、权重下载或请求重试；请求失败会报错，不会替换成模拟答案。
模型卡的 `ready` 只表示适配器可用，不代表账户已获准访问服务。

`jev.rerank(query, passages, top_k=3)` 接收带 `text` 字段的字典列表，保留原来源字段，
在返回的 `passages` 中增加 `jev_relevance` 与 `jev_confidence`。
空列表直接返回空结果，不发请求。相关性和置信度均不是事实正确性的证明。
完整 RAG 组合见 [RAG 流水线](RAG_PIPELINE.md)。

JSON/MCP 通过 `run_model` 调用同一接口，`data` 包含 `state` 和 `questions`，
`parameters` 仅接受 `model`、`allow_network`、`timeout`。密钥来自服务端环境，
不放在工具参数里。需要可复现版本时应指定供应商支持的固定模型 ID，
并记录响应中的实际 `model`；`jev-latest` 是可变化的别名。

仓库提供 `python scripts/run_jev_probe.py --output runs/jev-probe-plan`，默认只保存两次
公开合成输入的探测计划，不联网。配置环境中的 `TYPESAFE_API_KEY` 后，使用 `--run`
和一个新的输出目录才会调用真实服务；错误、实际模型 ID、用量和耗时分别留存。
该探测只核对连接和接口，不证明检索质量或置信度校准。

需要比较检索增益时，使用 [BM25／Jev 配对实验](../benchmarks/rag_jev/README.md)。
它冻结同一候选集合和上下文预算，在保存推理结果后单独读取相关性标签评分。
内置问题只是公开合成开发样例；正式论文仍需要独立标注的问题集和下游任务评估。

## 选择已有模型

```python
from fin_skills.algorithms import catalog

models = catalog("regression")
for model in models:
    print(model["id"], model["inputs"], model["status"])
```

`ready` 表示适配器存在且依赖可发现，不保证任何输入都能成功运行。
`missing_dependency` 表示需要安装对应可选依赖；库不会自动安装软件。

| 类别 | 当前模型或方法 ID | 使用方式 |
|---|---|---|
| 基础预测 | `naive`, `mean`, `drift`, `seasonal_naive` | `fit` → `predict(horizon=...)`，也可 `run` |
| 统计预测 | `arima`, `auto_arima` | 同上；需要对应可选依赖 |
| 回归 | `ridge`, `random_forest_regression`, `lightgbm_regression`, `xgboost_regression` | `fit` → `predict(X_new)`，也可一次性 `run` |
| 分类 | `logistic`, `random_forest_classification`, `lightgbm_classification`, `xgboost_classification` | 同上；预测输出为类别 |
| 组合配置 | `equal_weight`, `inverse_volatility`, `min_variance`, `hrp` | `run` 返回目标权重 |
| 风险与信号 | 通过 `catalog("risk")`、`catalog("signal")` 查询 | `run` 返回对应估计或信号 |
| 果蝇启发记忆 | `model_zoo` 中的 `fly_memory` | 独立扩展；`predict` / `update`，见上文 |

更完整的输入和依赖说明见[算法目录](ALGORITHMS.md)。
`fit` 当前支持预测、回归和分类，不支持对所有目录条目执行训练。
当前监督模型适配器主要暴露 `seed`，内部超参数并非全部可配置；不能把第三方模型的
任意构造参数直接传给本库。

## 训练一次，多次预测

以下片段使用用户提供的 `X_train`、`y_train`、`X_new`；它们不是内置数据。
训练数据按时间排序，标签在训练截止时必须已经可用。预测 DataFrame 保持训练时的列名和顺序。

```python
from fin_skills.algorithms import fit

model = fit("ridge", {"X": X_train, "y": y_train}, seed=7)
prediction = model.predict(X_new)
```

在满足样本数和依赖要求时，将 `ridge` 换成 `random_forest_regression`、
`lightgbm_regression` 或 `xgboost_regression`，可沿用同样的调用形式。
这是接口兼容，不意味着各模型具有相同适用条件或预测效果。

时间序列预测不需要 `X_new`，使用预测步数：

```python
forecast_model = fit("naive", {"series": training_series})
forecast = forecast_model.predict(horizon=5)
```

## 保存并重新加载

```python
from fin_skills.algorithms import load_model

model.save("ridge_model.zip")
restored = load_model("ridge_model.zip", trusted=True)
prediction = restored.predict(X_new)
```

保存路径必须不存在。模型文件含 pickle，只能对自己创建或来源可信的产物设置
`trusted=True`；校验摘要用于发现损坏，不用于证明来源安全。加载默认检查环境版本和实现摘要。
这套接口复用已训练状态，不是下载预训练金融模型的服务。

## 直接执行方法，或由智能体调用

```python
from fin_skills.algorithms import run

weights = run("hrp", {"asset_returns": historical_returns}, linkage="single")
```

这里得到组合权重，不会下单。预测值、交易信号、目标仓位和实际成交属于不同输出，
调用方需要明确转换规则和成交时点。

智能体可以通过 `list_algorithms` 发现候选，再通过 `run_algorithm` 指定模型、输入和参数。
这些是已有 JSON/MCP 工具；持久化模型的全部训练生命周期尚未作为有状态远程工具暴露。
完整时间顺序验证及模型比较见[研究流程](RESEARCH_WORKFLOW.md)。

## 验证范围

适配器测试覆盖数值对照、上游神经模型训练与日期约束、RL 后端训练和动作接口、保存加载，
以及果蝇记忆的反馈成熟时间、负收益更新和重复回执。运行记录另列于
[模型接入验证](MODEL_INTEGRATION.md)。这些检查验证调用机制，不证明金融预测或盈利能力。
预训练权重分发、完整金融 RL 环境、所有模型的有状态远程服务仍未提供。
