# 现成果蝇交易代码：复用决定与源码核查

日期：2026-09-21，America/New_York。范围：公开作者仓库、作者实验说明和本地离线测试。
这是对前一轮文献研究的软件实现补查。来源、固定版本和测试命令见
[reuse_sources.json](reuse_sources.json)。助手已核查列出的材料，人工学术复核尚未完成。

**决定：以 fruit-fly-fund 的离线对照实验框架为工程起点，保留其 Stonkfly 神经后端
作为现成基线；新行为模块优先复用 Stonkfly Lab 的编码与蘑菇体组件。**
我们的工作集中在账户反馈接口、时间划分，以及有行为依据的记忆与行动门控。
不重新编写连接组模拟器，也不先搭一个更大的果蝇网络。

这更新了 [RESEARCH_REPORT.md](RESEARCH_REPORT.md) 的实施顺序：先接通、核对现成基线，
再按需要引入 Bennett、Gkanias 等原作者模型作机制对照。原报告关于行为证据和创新性的
限制仍成立；采用完整连接组作比较对象，不表示它已经是有效的交易学习器。

## 已有项目能提供什么

| 项目 | 作者已实现的内容 | 本项目的取舍 |
|---|---|---|
| [fruit-fly-fund][fund] | 封装 Stonkfly；学习开启／冻结双组、独立账本、留出与重置协议、记录回放 | 首选工程基础；采用后端接口和实验记录，修正下述时间与状态控制 |
| [Stonkfly][stonk] | 连接组导入、LIF 内核、视觉输入、KC→MBON 可塑性、固定解码、检查点 | 直接保留为完整连接组基线；无需自写神经模拟器 |
| [Stonkfly Lab][lab] | 紧凑的虚拟气味编码、稀疏 KC、双 MBON 读出、前次线索快照 | 复用 `odors.py`、`mb.py`；市场奖励与控制实验不能原样采用 |
| [OpenFly][open] | 多种感觉编码、神经活动的拟合读出、训练／验证／测试、期权模拟器 | 借鉴模块边界与评估结构；其 NIFTY 卖跨式规则和报价建模不适合作为我们的默认交易任务 |
| [FlyAlpha][alpha] | 果蝇启发交易脚手架及合成示例 | 保留作相关工作；本轮未运行，暂不增加另一套基础框架 |
| [FlyTV][flytv] | Pine 实现、连接组子网、奖励可塑性与图结构对照 | 参考实验设计及作者负结果；不移植 Pine 界面 |
| [TraderFly][trader] | 公布脉冲回路与交易输出的部分 TypeScript 源码 | 作者声明 all rights reserved，且未公开完整交易执行规则；不纳入代码复用 |

前四个仓库的 LICENSE 正文均已读取，为 MIT；fruit-fly-fund 的 GitHub 自动识别显示
`NOASSERTION`，但正文明确原作者代码为 MIT，并单列 vendored Stonkfly 的许可。
复用时保留原作者署名与许可。连接组数据有独立许可，不能把代码 MIT 当作数据许可。

## 源码核查发现：能复用，但不能直接据此报收益

### Stonkfly 的输入和交易输出仍是工程映射

其 [模型说明][stonk-model] 和 [验证记录][stonk-validation] 明确承认：视觉代理、
左右 DNp20 活动到 BUY／SELL 的映射、PnL 到强化脉冲都含工程假设。
作者曾观察到连续 BUY 建议；图像背景也会显著影响 KC 是否活动。
所以必须同时比较普通特征模型、固定／随机结构和冻结记忆，不能把图规模当作交易优势。

### Stonkfly Lab：奖励归因与对照需要修改

- [`mb.py`][lab-mb] 的正 `valence` 总是增加 BUY 权重、减少 SELL 权重；
  `reinforce` 没有行动参数。[`engine.py`][lab-engine] 只在教师任务中对 SELL 翻转符号。
  因此它的市场模式不能直接解释为“奖励刚才执行的任意行动”。
- `snapshot_eligibility()` 返回当前 KC 快照，不返回类中累积的 eligibility。
  当前代码实现的是上一线索快照更新，不能凭变量名声称验证了多步资格迹。
- [`colony.py`][lab-colony] 的 `shuffle_reward` 是确定性符号取反，仍保留完整的
  线索—结果依赖；它是反向奖励对照，不是打乱对应关系的对照。
- `engine.py` 把成交后的权益存为下一步参考，成交费用已从参考值扣除；
  [`reinforcement.py`][lab-reward] 在默认 `exclude_fees=True` 时又加回前次费用。
  对净收益目标而言，不能直接使用这段奖励计算。

已运行[可重复的函数探针](../../../benchmarks/fly_paper/reuse_probe.py)：
同一气味在一次正反馈后，BUY−SELL 归一化分数从约 0.0081 升至 0.8595；
在价格不变、前后都用成交后权益且关闭 deadband 的算术例子中，奖励仍为 +0.0008。
完整输入与输出见 [reuse_probe_result.json](reuse_probe_result.json)。
这是函数行为检查，不是收益实验；默认 deadband 可能屏蔽该小额奖励。

### fruit-fly-fund：留出不自动等于向未来预测

[`scripts/experiment.sh`][fund-script] 明确先训练最新窗口，再将 exam 向更早的行情移。
[`market.py`][fund-market] 的 offset 实现也按此倒退。
这可以检查未见窗口，却不能当作“用过去训练、预测未来”的金融样本外证据。
我们将使用固定行情快照，显式保证训练结束早于验证、验证结束早于测试，
并核对奖励成熟时间、窗口重叠与检查点来源。

[`starting.py`][fund-start] 修复了恢复检查点后意外解冻的风险，并清零重置权重对应的
memory deviations，值得复用。但它恢复整个检查点后只重置部分字段；其“重置后等同
从未训练大脑”的说明不能推广到所有动态状态。若要把差异归因于长期权重，两个实验组
须从相同动态初态开始，仅载入指定权重；另行检验持续状态的贡献。

作者 README 报告的短窗口实验中，学习开启没有改善留出表现，且早期窗口存在重叠。
本轮没有复跑这些神经市场实验；已运行的账本核对只检查公开记录中的资金算术。
这些结果适合作为工程与失败经验，不能视为已验证盈利模型。

## 已实际运行的检查

均在本地隔离 checkout 中运行，没有启动交易连接或完整连接组训练。

| 固定版本上的命令 | 本轮输出 | 验证范围 |
|---|---|---|
| Stonkfly：`python -m pytest tests/test_neural.py -q` | 6 passed, 1 skipped | 解码、反馈分类、规则时间次序；完整连接组测试跳过 |
| Stonkfly Lab：`python -m pytest -q` | 5 passed | 上游小型教师任务与 broker 测试；不是市场收益 |
| fruit-fly-fund：四个指定测试文件，命令见 JSON | 47 passed | 配置公平性、已发布记录账本算术、检查点控制、行情窗口单元测试 |
| 本仓库 `reuse_probe.py` | 已输出 JSON | 前述紧凑模型与奖励的函数行为 |

本地 Python／依赖版本见 JSON；没有按 Stonkfly 锁定的完整依赖环境安装。
因此通过的是这些选定离线测试，不能外推为生产环境或完整神经运行已通过。

## 接入我们库的具体边界

1. **现成基线保持独立。** 在 fruit-fly-fund 固定版本上跑可重现的双组协议，记录上游
   commit、配置、数据哈希、输入帧哈希及权重变化。日期划分与状态重置使用我们的薄适配层。
2. **共用交易环境。** 我们的库负责行情可见时间、实际成交、费用及净权益核对；
   先让同一订单序列在两边账本核对一致，再比较策略。决策输入只包含当时可见信息。
   现有批量 `signal(panel)` 不足以支持这一闭环，增加逐步账户反馈适配器。
3. **紧凑模型复用现有组件。** 市场机会特征→虚拟气味编码→KC／MBON 记忆；
   当前风险承受能力与可用资本→行动门控；执行结果→绑定原线索与行动的延迟更新。
   输入编码和权重更新来自现有实现，账户奖励、行动归因与状态门控需单独验证。
4. **贡献只测新增部分。** 保留相同编码的普通学习器、冻结记忆、去掉状态门控、真正
   打乱强化对应关系的对照。先测机会暂时中断、失效和恢复，再看固定金融目标下的净收益。
   修正会计或时间顺序带来的改善不能算作果蝇机制增益。

本轮完成了候选源码检查、固定版本下载和上述离线验证，**尚未完成该适配层，也没有
新的市场收益结果**。后续正式计算的代码、环境、数据、缓存、临时文件、日志及输出
统一放在 Beacon `/beacon-projects/radfm/wy891/`；不写入 Beacon HOME。

[fund]: https://github.com/armanbabazadeh6/fruit-fly-fund/tree/56f01f6426a4d6d6293a4e06afcf4a035133a968
[stonk]: https://github.com/nftechie/stonkfly/tree/78ef3e05ab0fa086032098558d893667068944a0
[lab]: https://github.com/paappraiser/stonkfly-lab/tree/09e4529e2e1a135083838abd00ccacfd95c63a29
[open]: https://github.com/marketcalls/openfly/tree/fd4b06ba0d32fa76965672777e5811da7d9171eb
[alpha]: https://github.com/TanaeemDar/FlyAlpha/tree/350d60ad2062464a36b50f8e1db9e68dfffb57f2
[flytv]: https://www.tradingview.com/script/nmnHkA02-FlyTV-a-real-fruit-fly-connectome-trading-on-your-chart/
[trader]: https://github.com/SotoAlt/traderfly-brain/tree/05a61a7dad295369c0ded8f030422047d8f2dca8
[stonk-model]: https://github.com/nftechie/stonkfly/blob/78ef3e05ab0fa086032098558d893667068944a0/docs/model.md
[stonk-validation]: https://github.com/nftechie/stonkfly/blob/78ef3e05ab0fa086032098558d893667068944a0/docs/validation.md
[lab-mb]: https://github.com/paappraiser/stonkfly-lab/blob/09e4529e2e1a135083838abd00ccacfd95c63a29/flylab/mb.py#L62
[lab-engine]: https://github.com/paappraiser/stonkfly-lab/blob/09e4529e2e1a135083838abd00ccacfd95c63a29/flylab/engine.py#L56
[lab-colony]: https://github.com/paappraiser/stonkfly-lab/blob/09e4529e2e1a135083838abd00ccacfd95c63a29/flylab/colony.py#L77
[lab-reward]: https://github.com/paappraiser/stonkfly-lab/blob/09e4529e2e1a135083838abd00ccacfd95c63a29/flylab/reinforcement.py#L14
[fund-script]: https://github.com/armanbabazadeh6/fruit-fly-fund/blob/56f01f6426a4d6d6293a4e06afcf4a035133a968/scripts/experiment.sh#L30
[fund-market]: https://github.com/armanbabazadeh6/fruit-fly-fund/blob/56f01f6426a4d6d6293a4e06afcf4a035133a968/flyvsly/market.py#L177
[fund-start]: https://github.com/armanbabazadeh6/fruit-fly-fund/blob/56f01f6426a4d6d6293a4e06afcf4a035133a968/flyvsly/starting.py#L146
