# 模型接入与验证记录

日期：2026-09-21。公开入口：`fin_skills.model_zoo`。
用法见[模型使用指南](MODEL_USAGE.md)，最小示例见
[`examples/model_zoo.py`](../examples/model_zoo.py)。

## 复用已有实现

| 接入内容 | 实际实现 | 本库负责的部分 |
|---|---|---|
| LSTM、VanillaTransformer、PatchTST、NHITS | Nixtla NeuralForecast | 模型发现、规则日历与输入检查、统一创建入口、产物封装 |
| PPO、SAC | Stable-Baselines3 | 显式 Gymnasium 环境接口、动作预测、原生策略保存加载 |
| 既有预测、回归、分类与组合方法 | 原 `fin_skills.algorithms` 注册表和对应后端 | 复用原接口，增加模型卡和统一生命周期 |
| Kalman、协方差、期限结构、信用与 SVI | 仓库既有 `fin_skills.models` 数值模块 | 参数与维度检查、结果接口；没有重写方程 |
| 果蝇记忆 | 既有 GPL `benchmarks/fly_paper` 实现 | 独立安装包、显式参数、成熟反馈与重复更新检查 |

深度模型没有在本库重写网络。NeuralForecast 的模型与原生保存加载文档是适配依据：
[LSTM](https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html)、
[Transformer](https://nixtlaverse.nixtla.io/neuralforecast/models.vanillatransformer.html)、
[PatchTST](https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html)、
[NHITS](https://nixtlaverse.nixtla.io/neuralforecast/models.nhits.html)、
[保存加载](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/save_load_models.html)。
RL 接口对应上游 [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
和 [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)。

## 验证环境与边界

### 上游做法与后续复用边界

进一步查阅官方文档后，建议按组件扩展已有适配层：

| 项目 | 上游已有做法 | 本仓库现状与复用方向 |
|---|---|---|
| Qlib | 预测分数、组合策略与执行器分层 | 已接入原生 `TopkDropoutStrategy` 与 `SimulatorExecutor`，核对可用时间和扣费净值 |
| FinRL | 市场环境、RL 算法与应用分层 | 已接入 0.3.7 股票环境，补充下一开盘成交、收盘反馈和独立账本 |
| AutoGen 工具调用 | 模型根据工具 schema 产生调用，执行后继续推理 | 已接入 `AssistantAgent`，提供会话内模型训练、预测和 Qlib 回测工具；规则推荐作为可选建议 |

来源：[Qlib 策略接口](https://qlib.readthedocs.io/en/latest/component/strategy.html)、
[FinRL 官方仓库](https://github.com/AI4Finance-Foundation/FinRL)、
[FinRL 分层说明](https://finrl.readthedocs.io/en/latest/start/three_layer.html)、
[AutoGen 工具调用](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/agents.html)。
FinRL 分层说明页面标注 0.3.1，只用于解释设计；实际接入仍需对照当前源码验证。
具体接口与验证边界见[完整流程接入](REUSED_WORKFLOWS.md)。这里只覆盖表中组件，
不表示三个框架的全部功能都已集成。

本库应主要维护上游缺少的统一数据约束、可用时间检查、反馈验证与实验记录，
并用消融实验检验这些组件的效果。模型、优化器、交易环境或 Agent 编排已有
合适实现时，先增加薄适配器；兼容性与正确性测试通过后再进入收益实验。

### 已执行环境

执行位置：Beacon 的 `/beacon-projects/radfm/wy891/fin-skills-model-zoo-20260921`。
环境、缓存、日志与临时产物都置于该 RADFM 目录；本机不运行项目训练或测试。

实际环境查询得到 63 个模型/方法条目，全部依赖可发现，58 个 JSON 工具。
这包含数值方法和原注册表，不是 63 个预训练模型，也不代表逐个完成金融实证。
深度学习与 RL 的训练、预测和持久化使用 Python 接口；JSON/MCP 提供模型发现
及无状态数值执行，不执行任意环境对象或加载磁盘 pickle。

| 包 | 实际安装版本 |
|---|---|
| neuralforecast | 3.2.2 |
| stable-baselines3 | 2.9.0 |
| torch | 2.14.0 |
| pytorch-lightning | 2.5.6 |
| pyqlib | 0.9.7 |
| statsforecast | 2.1.1 |
| scikit-learn | 1.9.1 |
| numpy / pandas | 2.5.3 / 2.3.3 |
| scipy（兼容修复后的环境） | 1.16.3 |
| fin-skills-fly | 0.1.0 |

验证使用合成时序、数值对照和 Gymnasium 标准环境。神经模型以少量优化步数
检查真实训练与推理链路，RL 检查动作空间和策略保存加载；没有把这些检查当作
收益实验。果蝇测试显式使用合成电路参数，不冒充作者拟合权重。

目录构建与包生成已运行，`validate.py` 报告 `OK`：129 个技能符合规范。
保留 1 条已有警告：`fin-libraries` 的 24 项超过建议的 20 项发现预算。
重新生成前后的源码内容一致（忽略 CRLF/LF 差异），无需回传修改生成文件。

兼容修复后的最终回归：**219 passed，0 skipped**，耗时 155.26 秒。
范围为下列命令列出的 7 个测试模块，不是仓库全量测试。
其中 `test_model_zoo.py` 包含 16 项，覆盖四个 NeuralForecast 后端的真实训练、
预测日期、重复预测、原生保存加载，PPO/SAC 训练与策略恢复，以及果蝇反馈更新。
日志为该 Beacon 目录下的 `logs/integration-final.log`。
运行保留 115 条上游告警，主要涉及 Lightning 配置与 NumPy 日期转换弃用；未隐藏告警。

基础示例成功执行。核心和 GPL 扩展 wheel 均构建成功，检查确认核心 wheel
包含 `model_zoo`，不包含 `fin_skills_fly` 或 benchmark 代码；扩展包含其许可文件。
将核心 wheel 安装到独立目录后，在源码目录之外成功导入、发现模型并执行预测。
产物保留在 Beacon 的 `wheels/`，完整环境版本记录在 `logs/environment.txt`。

## 可复现命令

在安装所需可选依赖的工作区中执行：

```bash
python -m pip install -e '.[algorithms,qlib,optimizers,deep,rl,dev]'
python -m pip install ./benchmarks/fly_paper
python -m pytest -q tests/test_model_zoo.py tests/test_algorithm_backends.py \
  tests/test_algorithms.py tests/test_algorithm_research.py \
  tests/test_strategy_workflow.py tests/test_tools.py tests/test_episode_credit.py
python scripts/build_index.py
python scripts/build_package.py
python scripts/validate.py
```

可选依赖缺失时部分测试会跳过，报告结果时须保留跳过数。核心包继续保留 MIT
声明；GPL 扩展独立打包，保留上游代码的许可和出处，不包含拟合参数数据。
已训练产物仍需来自可信来源；摘要检查不提供签名或来源认证。

初次回归使用 SciPy 1.18.1，得到 218 通过、1 失败：PyPortfolioOpt 的 HRP
访问了已不存在的 `_LINKAGE_METHODS`。`optimizers` extra 因此限制 `scipy<1.17`，
使用包含该属性的 1.16 系列验证。这是依赖兼容限制，未改写 HRP 或静默替换后端。
依据为运行错误及上游
[PyPortfolioOpt 源码](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/pypfopt/hierarchical_portfolio.py)
与 [SciPy 1.16.3 源码](https://github.com/scipy/scipy/blob/v1.16.3/scipy/cluster/hierarchy.py)。

对论文而言，这次接入属于可复用的系统能力。LSTM、PatchTST、PPO 等算法本身
不是本项目新提出的方法；库对任务成功率、时间约束和交易收益的影响需要另行实验。
