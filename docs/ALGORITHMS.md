# 算法目录、自动选择与运行

`fin_skills.algorithms` 把算法、实现库、适用条件和可执行接口放进同一个注册表。
选择器先检查数据、样本数、目标、依赖和约束，再按明确规则排序，并返回每个选择或排除的原因。
这些排序分数是本库的策略规则，不是收益预测、准确率或经过实证验证的算法优劣。

## 快速开始

```python
import numpy as np
import pandas as pd
from fin_skills.algorithms import catalog, Request, recommend, auto_run, run

returns = pd.DataFrame(
    np.random.default_rng(7).normal(0, 0.01, (250, 3)), columns=["A", "B", "C"]
)

# 查看每个算法的 inputs、objectives、tags、capabilities 和 status。
print(catalog("portfolio"))

# 只有数据概况时，先做推荐。
selection = recommend(Request(
    task="portfolio", available_inputs=("asset_returns",), n_observations=len(returns),
    objective="min_variance", required_capabilities=("long_only",),
))
print(selection.as_dict())

# 有实际数据时，从数据推导样本数，自动选择并执行。
answer = auto_run("portfolio", {"asset_returns": returns}, objective="min_variance")
print(answer["selection"]["selected"])
print(answer["result"])  # 按资产名索引的权重

# 也能指定算法。HRP 要显式选 linkage，避免不同库的默认值不一致。
weights = run("hrp", {"asset_returns": returns}, linkage="single")
```

`auto_run` 没有可执行候选时返回 `executed=False`、`result=None` 和排除理由。
所选算法执行失败会抛出原始异常，不悄悄换成另一种算法。
未知任务、字段、目标、偏好和约束也会报错；支持词汇可从 `catalog(task)` 查询。

`task` 支持 `portfolio / forecast / volatility / risk / signal / regression /
classification / execution / pricing / stat_arb / regime`，也支持对应中文别名：
`组合优化 / 预测 / 波动率 / 风险 / 信号 / 回归 / 分类 / 执行 / 定价 / 统计套利 / 状态识别`。

## 接入范围

目录把“算法 + 实现后端”作为一条记录；同一 HRP 算法的不同库实现分别列出。
记录数请运行 `len(catalog())`，避免文档数字与代码脱节。

| 任务 | 已有执行适配器 | 目录中记录、尚未接执行适配器 |
|---|---|---|
| 组合配置 | 等权、逆波动率、最小方差、HRP；PyPortfolioOpt/skfolio/Riskfolio 的 HRP 与最小方差桥接 | — |
| 时间序列预测 | naive、mean、drift、seasonal naive、statsmodels ARIMA(1,1,0)、StatsForecast AutoARIMA | — |
| 波动率 | 历史波动率、EWMA、arch GARCH、EGARCH | — |
| 尾部风险 | 历史法与正态法 VaR/ES | — |
| 趋势信号 | 时间序列动量、均线交叉 | — |
| 回归与分类 | scikit-learn Ridge、LogisticRegression、随机森林；LightGBM、XGBoost；Qlib 线性模型 | — |
| 执行计划 | TWAP、基于预测成交量的 VWAP | — |
| 期权定价 | 欧式 Black-Scholes-Merton、美式 CRR、QuantLib Heston | — |
| 统计套利 | statsmodels Engle-Granger | — |
| 状态识别 | hmmlearn HMM、ruptures 变点检测 | — |

第三方能力条目在 2026-09-14 对照官方文档核查。来源保存在每条记录的 `source` 中；
例如组合算法见 [PyPortfolioOpt](https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html)、
[skfolio](https://skfolio.org/) 和 [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io/en/latest/)；
预测、波动率与监督学习见 [statsmodels](https://www.statsmodels.org/stable/tsa.html)、
[arch](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html) 和
[scikit-learn](https://scikit-learn.org/stable/supervised_learning.html)。
官方来源证明对应库具有该能力；本库的推荐分数和样本门槛属于实现策略。

`status` 的含义：

- `ready`：执行适配器存在，且 Python 能发现所需模块。这不是二进制兼容性或求解成功的证明。
- `missing_dependency`：适配器存在，但可选库未安装。不会自动下载或安装。
- `catalog_only`：仅记录该算法及其来源，尚未接入统一执行接口；即使安装了库也不会自动运行。

默认 `executable_only=True`。设为 `False` 可用于研究选型，但要检查候选的执行状态。
本机 Windows 测试曾在混合加载数值库时触发 CVXPY 原生组件访问冲突；第三方适配器的行为测试
在独立进程中执行，任何非零退出都算失败。部署中混用监督模型和优化器时，也应先验证该环境的组合，
或把可选后端任务放在独立进程中运行。
复杂度上限、`allowed_libraries`、`required_capabilities` 和目标都是硬筛选条件。
`preferences` 是软偏好；分数为 `priority + 10 × 匹配偏好数 - 复杂度等级`，并用算法 ID 打破平局。
提供实际数据时会依据可用字段和样本量筛选；期权还会依据 `exercise` 选择美式或欧式方法。
不自动推断牛熊市、收益机会或潜在市场状态。

## 输入和参数

数组按时间递增排列。Pandas 索引必须唯一、递增、无缺失；训练字段索引必须严格一致。
不自动填补缺失值或删除行。回归和分类的 `X_predict` 列数及 DataFrame 列顺序必须与 `X` 一致。
预测用的特征和标签必须由调用方保证在当时已经可用。

| 算法组 | 输入字段 | 可选参数或必要说明 |
|---|---|---|
| 组合 | `asset_returns`，时间 × 资产矩阵 | `hrp` 及各后端 HRP 必须传 `linkage`；配置只支持 long-only |
| 预测 | `series`；seasonal naive 另需 `seasonal_period` | `horizon=1`；ARIMA 固定 `(1,1,0)` |
| 波动率 | `returns`，一维收益率 | `periods_per_year=1`；EWMA 另有 `decay=0.94` |
| VaR/ES | `returns` | `confidence=0.95`；输出单位为每期小数损失 |
| 动量 | `returns` | `lookback=20` |
| 均线交叉 | `prices` | `fast=5, slow=20` |
| 回归/分类 | `X, y, X_predict` | `seed=0`；Ridge 和 Logistic 的缩放器只拟合训练数据；分类标签须为数值 |
| TWAP | `shares, n_bins` | 数量非负；只返回计划，不发送订单 |
| VWAP | `shares, volume_forecast` | 成交量必须是事前预测；不会校验预测的来源时间 |
| 定价 | `option` 字典 | 必须包含 `S,K,T,r,q,sigma,flag,exercise`；CRR 支持 `steps=200` |

`option.flag` 为 `c/p`，`exercise` 为 `european/american`；利率、股息率和波动率使用年化小数，
`T` 为年数，股息按连续收益率处理。这些简化适配器要求正的到期时间和波动率。
收益率必须是每期**简单小数收益率**，例如 1% 写作 `0.01`，不能传价格或百分数 `1`。
入口采用保守规则：拒绝低于 -1 的收益率或列均值绝对值大于 0.10 的数据；
这只是输入筛查，不能证明单位正确，也会排除某些极端高收益样本。

信号输出已延后一根 bar；预热期保留 NaN。组合权重只对提供的历史拟合，应该用于之后的收益率。
任何算法的 `min_observations` 都只是路由门槛，不代表统计估计可靠。
样本充足性、交易成本和回测完整性仍可交给现有 `fin_skills.api` 检查。

## 用时间验证选择预测算法

```python
from fin_skills.algorithms import walk_forward

comparison = walk_forward(
    list(range(80)), initial_train=40, horizon=5, gap=2,
    candidates=["naive", "mean", "drift"], metric="mae",
)
print(comparison["selected"])
print(comparison["ranking"])  # 所有完整通过验证的候选及各窗口误差
print(comparison["forecast"]) # 选中方法在全部输入历史上重新拟合后，对未来的预测
```

每个窗口只用此前的数据拟合；预测 `gap + horizon` 步，略过 gap 后评分。
验证窗口不重叠，支持 MAE 和 RMSE。某个算法只要有一个窗口失败，就不能凭剩余窗口的平均分获胜。
失败原因会保留；重新拟合失败也会报告，不替换赢家。窗口边界均为从零开始的位置，右端不包含。
计算限制为最多 20 个候选、200 个完整窗口，且至少需要两个完整窗口。

这里的误差已经参与了选型，因此是**验证误差**。要报告泛化性能，需要另外留出一个未用于选型的后续测试区间。
这个 walk_forward 接口仅比较预测误差。跨任务验证、独立保留测试期、审计和模型保存见[完整研究流程](RESEARCH_WORKFLOW.md)。

## MCP / JSON 工具

现有 MCP 服务和 OpenAI/Anthropic 工具导出会自动包含这组接口：
`list_algorithms`、`recommend_algorithms`、`run_algorithm`、`auto_algorithm`、`compare_forecast_algorithms`、`profile_algorithm_data`、`research_algorithms`。

```python
from fin_skills.tools import call_tool

result = call_tool("auto_algorithm", {
    "task": "execution",
    "data": {"shares": 1000, "volume_forecast": [1, 2, 1]},
    "constraints": {"preferences": ["volume_profile"]},
})
print(result["selection"]["selected"], result["result"])
```

JSON 数组以及现有 Series/Frame payload 都可作为输入。单个结果数组超过 10,000 个值时工具会拒绝返回，
请用 Python API；不会截断权重、信号或预测而假装是完整结果。Python 数组输入上限为每字段 1,000,000 个值。

## 注册新算法库

```python
from fin_skills.algorithms import Algorithm, default_registry

registry = default_registry()
registry.register(
    Algorithm(
        id="my_forecaster", name="My forecaster", task="forecast", library="my-library",
        inputs=("series",), objectives=("forecast",), min_observations=20,
        source="https://example.org/my-library/docs", verified_on="2026-09-14",
    ),
    handler=lambda data, params: [float(data["series"][-1])] * params.get("horizon", 1),
)
result = registry.run("my_forecaster", {"series": list(range(30))}, horizon=3)
```

真实扩展应填写实际的来源和核验日期，并在 handler 中验证自有参数和数据契约。
注册仅作用于当前 registry，不修改全局默认目录；重复 ID 会拒绝。自定义执行器是受信任的 Python 代码，
JSON 接口不接受函数、模块路径或动态代码。新增库只需补记录、执行适配器和对应行为测试，
无需再写一套选择器。新增任务类型需要同时扩展任务词汇及数据契约。

本功能位于手写层 `fin_skills/algorithms/`，已经加入 `build_package.py` 的保留目录。
再生成技能包不会覆盖它；算法内部优先复用已有模型、策略脚本和优化器桥接。

## 第一阶段验证记录（2026-09-14，研究流程扩展之前）

- `python scripts/build_index.py`、`python scripts/build_package.py` 已运行；
  `python scripts/validate.py` 返回 OK，保留已有技能发现预算提示。
- `python -m pytest -q`：2857 通过、29 跳过、46 个 slow 测试默认未选中。
- `python -m pytest -q -m slow tests/test_examples.py`：8 通过。
- PyPortfolioOpt、scikit-learn 和 statsmodels 的已安装适配器使用真实后端验证。
  skfolio、Riskfolio 和 MCP SDK 在本机未安装，对应测试跳过；JSON 工具定义及调用已经验证。

后续工程化改进的范围、最新验证与尚未完成的外部验收见[工程化验收记录](PROFESSIONAL_READINESS.md)。
