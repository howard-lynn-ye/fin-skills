# 复用 Qlib、FinRL 与 AutoGen

本库负责接口与检查，模型、组合策略、撮合和智能体循环调用对应上游实现。
它们都是研究与模拟接口，不连接券商下单。

## 安装

```bash
python -m pip install -e '.[qlib,rl,agent]'
# FinRL 根包会带入大量行情与券商依赖。本适配器只使用其独立股票环境模块：
python -m pip install --no-deps finrl==0.3.7
```

FinRL 的环境源文件由已安装分发包定位并加载，保留原代码，没有复制或修改上游文件。
当前适配器固定支持 0.3.7；其实际文件摘要在环境的 `backend_sha256` 属性中。
`[rl]` 提供这个模块需要的 Gymnasium、SB3、PyTorch 和绘图依赖。
AutoGen 的服务端模型客户端由调用方提供，需要支持函数调用。

## Qlib：预测分数 → 原生策略 → 原生回测

```python
import qlib
from fin_skills.bridges.qlib_strategy import qlib_backtest

qlib.init(provider_uri="/path/to/your/qlib_data", region="us")
result = qlib_backtest(
    scores, available_at, start="2025-01-02", end="2025-03-31",
    topk=10, n_drop=2, account=100000,
    open_cost=0.001, close_cost=0.001, min_cost=0,
)
print(result["nav"])
```

`scores` 和 `available_at` 为索引完全一致的 Series，MultiIndex 名称严格为
`datetime`、`instrument`。前者是分数，后者是分数实际可用的时间。
日期层使用无时区的交易日标签。分数必须在其标注交易日结束前可用；延迟披露需
调用方按真实可用时间重新归属，不能把未来信息标在过去。

适配器调用 Qlib `TopkDropoutStrategy` 和 `SimulatorExecutor`，读取前一个交易日
的分数，在下一个交易日开盘撮合。`report`、`positions` 与 `indicators` 来自 Qlib，
`net_returns` 扣除报告中的成本，`nav` 再与账户权益逐项核对。
`benchmark=None` 明确转换为零收益参照，避免 Qlib 0.9.7 隐式使用沪深 300。

价格数据、复权、退市样本、交易日历和价格限制依赖调用方的 Qlib 数据与配置。
默认 `trade_unit=1`，`limit_threshold=None`；不表示某个市场的真实制度。
本接口只检查已声明分数的可用时间，不证明上游模型训练与数据来源没有泄漏。
Qlib provider 是进程全局配置，不要在同一进程中并行切换不同 provider。

## FinRL：收盘决策，下一交易日开盘成交

```python
from fin_skills.bridges.finrl import make_finrl_env
from fin_skills.model_zoo import create_model

env = make_finrl_env(
    market_frame, features=["momentum"], initial_cash=100000,
    hmax=100, buy_cost=0.001, sell_cost=0.001,
)
policy = create_model("ppo", n_steps=128, batch_size=64, seed=7)
policy.fit(env, total_timesteps=4096)
observation, info = env.reset(seed=7)
action = policy.predict(observation)
observation, reward, terminated, truncated, info = env.step(action)
```

`market_frame` 包含 `date`、`tic`、`open`、`close` 及所选特征。每个交易日的股票集合
必须完整一致，价格为正，各数值有限；特征必须在该日收盘可用。调用方负责复权一致性。
观察向量按现金、当日收盘价、持仓股数、各特征分组排列；股票按代码排序。

动作范围为每只股票 `[-1, 1]`，FinRL 将动作乘 `hmax` 后转成整数股数，并处理现金、
持仓与费用。适配器隐藏下一开盘价，使用真实下一开盘价撮合，再按成交日收盘价计净值。
奖励为相邻净值的对数收益，包含费用与原持仓的隔夜变化；反馈在该日收盘后可用。
最后一个交易日完成后立即终止，避免额外的空终止步重复发放奖励。

评估读取 `env.ledger` 或 `step()` 的 `info`，其中记录成交股数、现金、费用、持仓、
净值和反馈所属交易日。底层 FinRL 的 `asset_memory` 使用不同估值时点，不能用作
这个适配器的业绩记录。当前不模拟做空、停牌、部分成交、市场冲击或动态股票池。

## AutoGen：由模型决定调用什么工具

```python
from fin_skills.tools.agent import ModelSession, make_tool_agent

session = ModelSession(
    datasets={
        "train": ("train", {"X": X_train, "y": y_train}),
        "test_features": ("predict", X_test),
    },
    environment_factories={"market_train": lambda: make_finrl_env(training_market)},
    max_models=4, max_training_steps=512,
)
agent = make_tool_agent(model_client, session=session)
# 在现有异步程序中运行：
result = await agent.run(task="查看可用模型和数据，选择合适的方法，训练并预测。说明选择理由。")
print(result.messages[-1].content)
print(session.receipts)
```

`model_client` 使用 AutoGen 支持的客户端，并由调用方配置服务地址、模型和凭据。
工具循环来自 AutoGen `AssistantAgent`。本库保留原工具 schema，允许模型选择
`list_models`、`run_model` 等工具；绑定 session 后还提供 `list_datasets`、
`fit_model`、`predict_model`、`backtest_strategy`。
规则推荐只是一个可选工具，不强制决定模型最终选择。

训练和预测通过命名数据与内存模型 handle 关联。预测数据不能通过 `fit_model` 训练。
神经模型训练步数、RL rollout 与模型数量有配置上限；这是调用预算，不是操作系统沙箱。
RL 环境只由调用方预先绑定的工厂创建，模型不能提交环境代码或加载任意磁盘对象。
`backtest_strategy` 使用 `backtest` 角色的数据，其内容必须恰好包括
`scores`、`available_at`、`start`、`end`；调用方先初始化 Qlib。
会话工具按锁串行执行；有依赖的调用应分轮提出。独立研究运行使用新的 agent/session。

所有输入的研究数据仍由调用方负责正确划分。服务端模型会收到任务文本、工具参数与
工具返回值，包括预测和回测结果；绑定数据前应按自己的数据访问要求选择推理服务。
原 benchmark 执行器及其冻结协议保持不变，这个公共接口不追溯更改旧实验结果。

## 验证与来源

新增测试：`tests/test_reused_workflows.py`。覆盖真实 Qlib 回测对照、FinRL 账本、
未来数据扰动、PPO/SAC 训练、数据角色隔离、训练预算及 AutoGen 工具调用。
AutoGen 部分使用官方 ReplayChatCompletionClient，验证调用机制及不同模型选择
确实改变执行结果；它不是在线 LLM 自主选择质量或高收益的实验。

2026-09-21 在 Beacon RADFM 工作区实际执行的整合回归为 **188 passed，0 skipped**，
耗时 47.00 秒，覆盖新增流程、模型接口、工具、既有 bridges、反馈和研究接口测试。
日志位于 `/beacon-projects/radfm/wy891/fin-skills-model-zoo-20260921/logs/reused-workflows-regression.log`。
验证环境：pyqlib 0.9.7、finrl 0.3.7、AutoGen AgentChat/Ext 0.7.5、
Gymnasium 1.3.0、Stable-Baselines3 2.9.0。测试保留上游告警，未隐藏或修改上游代码。

目录与包重新生成后，`validate.py` 报告 `OK`（129 个技能，保留已有的发现预算警告）。
核心 wheel 构建成功，并在源码目录之外完成 Qlib 导入、FinRL 单股票环境步进、
AutoGen 运行及模型服务客户端导入检查。产物与环境记录分别保存在该 Beacon 工作区
的 `wheels-reused/` 和 `logs/reused-environment.txt`。

官方实现依据：[Qlib 策略](https://qlib.readthedocs.io/en/latest/component/strategy.html)、
[FinRL 环境源码](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/env_stock_trading/env_stocktrading.py)、
[AutoGen 工具循环](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/agents.html)。
