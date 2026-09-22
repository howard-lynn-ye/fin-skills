# 原生量化实现接入

2026-09-21 的运行时统计：模型注册表 284 项，研究总目录 346 项。
目录含 51 个策略、37 个决策方法、258 个算法条目；按接口状态划分为
284 个已注册适配器、45 个外部实现、17 个方法参考。
条目包含不同后端及 TA-Lib 函数，不能当作独立交易策略或预训练模型数量。

本次扩充复用 TA-Lib、scikit-learn、statsmodels、Riskfolio-Lib、PyPortfolioOpt、
Cvxportfolio、arch、NeuralForecast 和 Stable Baselines3 的原生实现。
LEAN 与 Hummingbot 使用独立的 Python 引擎桥接。
完整 ID、输入、操作及来源见 [生成目录](../catalog/QUANT_METHODS.md)。

## 统一入口

```bash
pip install 'fin-skills[methods]'
pip install 'fin-skills[deep,rl]'  # 按需安装神经网络与强化学习依赖
```

```python
from fin_skills.model_zoo import create_model, model_catalog, load_model
from fin_skills.algorithms import get_method

# 输入 OHLC 数据须按时间排序；预热 NaN 原样保留。
atr = create_model("ta_atr", timeperiod=14).run({"inputs": ohlc})

model = create_model("elastic_net", alpha=0.01)
model.fit({"X": X_train, "y": y_train})
prediction = model.predict(X_test)
model.save("elastic-net.zip")
restored = load_model("elastic-net.zip", trusted=True)

forecast = create_model("sarimax", order=(1, 0, 0))
forecast.fit({"series": training_series})
future = forecast.predict(horizon=5)
portfolio = create_model("mean_cvar").run({"asset_returns": training_returns})
card = get_method("mean_cvar")
```

输入契约以 `model_catalog()` 和 `get_method()` 为准。`json_run=False` 的方法需要
Python 原生对象；不能通过 JSON 工具伪造环境或引擎。训练与预测方法可由
`ModelSession` 和 AutoGen 工具循环调用，模型自主选择方法。
`trusted=True` 只适用于可信模型文件；版本和源文件摘要不同会使严格加载失败。

标准化仅拟合训练集。DBSCAN 和谱聚类不提供虚构的样本外预测接口。
全样本状态概率、STL 分解和聚类结果标为回顾性分析。优化失败不会被等权组合掩盖；
Riskfolio 因子风险预算若违反声明的只做多限制，会显式报错。
Cvxportfolio 的现金列必须由调用方提供，缓存目录必须显式指定。

## LEAN

```python
from fin_skills.bridges.lean import LeanBacktest

engine = LeanBacktest(engine_dir="/path/to/Lean/Launcher/bin/Release",
    dotnet="/path/to/dotnet", template_config="/path/to/Lean/Launcher/config.json")
result = engine.run(algorithm_file="/path/to/trusted/algorithm.dll",
    algorithm_type="MyAlgorithm", data_folder="/path/to/data",
    work_dir="/path/to/new-run")
```

另安装 `fin-skills[lean]`，并自行构建原生引擎。调用方提供可信算法文件及本地数据，
每次运行使用新输出目录。桥接固定为回测配置，不提交云任务。
已验证上游提交 `b2a01cc15b09c1d448920f4af81c73f8b07ec7d4`、.NET 10.0.401：
`BasicTemplateFrameworkAlgorithm` 完成原生回测并生成 summary。
这证明引擎调用链可用；目录中的期权、执行、风控策略仍须各自实现和验收。

## Hummingbot

`fin_skills.bridges.hummingbot.native_strategy(method_id, **parameters)` 初始化原生策略，
所需参数为上游 `init_params` 的参数。调用方提供连接器、市场元组和时钟。
资金费率策略通过 `funding_strategy(checkout=..., connectors=..., config=...)`
从显式指定的可信上游源码加载。工厂不会启动时钟或创建交易所连接。

已验证上游提交 `2bfaccc48dd49e71a5b6d9b3011808e127dd00cd` 的纯做市策略：
原生模拟交易所的中间价为 100 时，1% 双边价差生成 99/101 报价，数量为 1，
并经过下一次报价刷新。测试见 `tests/test_native_engines.py`。
其他策略工厂尚未完成各自端到端验证，也未验收实盘连接器。

上游当前完整依赖集合存在 urllib3/web3/连接器 SDK 约束冲突。
此次在独立环境从源码构建原生扩展，仅安装模拟路径所需依赖；不能把这次通过解释为
完整 Hummingbot 分发包已经无冲突安装。核心库不强制安装这些交易所依赖。

## 验证范围

Beacon RADFM 上的适配器、目录、工具和复用工作流回归：**358 passed**。
另外分别运行 LEAN 原生样例与 Hummingbot 原生模拟报价测试，各 **1 passed**。
适配器测试包含 TA-Lib 全部登记函数的原生数值对照及前缀不变性、训练/加载一致性、
优化器调用、Cvxportfolio 原生决策对照、神经模型和强化学习的短训练路径。
这里的训练步数仅用于接口验收，不构成收益、泛化或论文性能证据。

原生引擎测试默认在缺少依赖时跳过。LEAN 需设置 `FIN_SKILLS_LEAN_CHECKOUT` 和
`FIN_SKILLS_DOTNET`。测试、缓存及输出均在 `/beacon-projects/radfm/wy891/` 下，
未在 Beacon HOME 安装或运行实验。

合并前另外验证了原生 vectorbt 桥接（23 passed，2 个与当前安装状态相反的测试跳过）、
全部示例（11 passed）和独立果蝇扩展（8 passed）。该环境使用 vectorbt 1.0.0、
Plotly 6.9.0、skfolio 1.3.0；Plotly 7.1.0 删除的 `scattermapbox` 模板字段会使
此版 vectorbt 导入失败，而 Plotly 5 又不满足此版 skfolio 的依赖声明。
