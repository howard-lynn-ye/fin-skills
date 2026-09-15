# 实盘交易前防御体系与自适应调仓操作指南 (Live Trading & Pre-Trade Guards Guide)

> **发布时间**: 2026-09-15  
> **适用对象**: 面向 A 股/全球 ETF 实盘配置者、量化交易工程师及自主决策 AI 智能体  
> **核心宗旨**: 彻底消灭学术纸面回测与真实世界执行之间的鸿沟，在下单前筑牢资金防线。

---

## 1. 为什么需要“实盘交易前守卫”（Pre-Trade Guards）？

学术量化研究往往局限于**“事后审计（Post-Backtest Audit）”**——在历史回测结束后，用算法检查：你是否偷看了未来（Lookahead）？你的过拟合概率有多高（DSR）？你是否扣除了手续费？

但在真实的交易日 14:30，当你准备打开券商软件挂单时，面对的是全新的、完全不同的**“真实市场陷阱”**：
1. **QDII 额度与溢价陷阱**：纳指或标普被国内散户追捧，二级市场价格可能比基金真实净值（IOPV）贵 5%~15%。一旦盲目追高买入，等套利盘开通溢价暴跌，直接遭遇无谓杀跌。
2. **整数手与资金颗粒度失真**：A 股单笔买入必须是 100 股（1手）。若账户只有 2 万元，而 10 年期国债 ETF 一手就要 1.3 万元（占 65% 仓位），理论上精细的 10% 或 20% 资产配置在物理上根本无法执行。
3. **现金隐形拖累（Cash Drag）**：回测假设闲置现金躺赚 2.0% 无风险利息，而券商默认活期利息只有 0.30%。如果不主动打理，10 年复利凭空蒸发近 20% 的现金部分收益。
4. **频繁调仓手续费磨损（Over-trading Friction）**：如果资产权重偏离 0.5% 就触发调仓，最低 2~5 元的起步佣金会像水滴石穿一样严重腐蚀账户本金。

针对以上四大痛点，`fin-skills` 全面引入了**实盘事前防御体系（Pre-Trade Defense Suite）**。

---

## 2. 三大核心实盘防御守卫详解

### 2.1 `QDIIPremiumGuard` (QDII 实时溢价率与熔断守卫)
* **所属模块**: `fin_skills.china.qdii_premium_guard`
* **API 标识**: `fin_skills.api.get("qdii_premium")`
* **运作原理**:
  实时计算二级市场交易价相较于基金盘中实时参考净值（IOPV）的偏离度：
  $$\text{Premium Rate} = \frac{\text{Price} - \text{IOPV}}{\text{IOPV}}$$
* **三级拦截标准**:
  * $\text{Premium} \le 1.5\%$：**正常交易 (PASS)**，全额执行目标权重；
  * $1.5\% < \text{Premium} \le 3.0\%$：**减半降权 (DERATE_50)**，买入权重压降 50%，剩余 50% 自动重定向至避险资产（黄金 ETF 518880）；
  * $\text{Premium} > 3.0\%$：**硬性熔断拦截 (HARD_CIRCUIT)**，判定为二级市场投机泡沫，禁止买入任何份额，资金 100% 重定向至防御资产。

```python
from fin_skills.api import get

guard = get("qdii_premium")
# 检查纳指 ETF (513100) 盘中现价 1.20 vs IOPV 1.10 (溢价 9.1%)
result = guard.run(code="513100", price=1.20, iopv=1.10)
print(result.passed)  # False (触发硬性熔断)
print(result.findings[0].message)
# "CRITICAL: Nasdaq 100 ETF (513100) trading at extreme premium of 9.09% (> limit 3.0%). Capital redirected to 518880."
```

---

### 2.2 `BoardLotFeasibilityGuard` (100股整数手可行性守卫)
* **所属模块**: `fin_skills.china.board_lot_guard`
* **API 标识**: `fin_skills.api.get("board_lot_feasibility")`
* **运作原理**:
  测算给定账户资金规模下，各资产按 100 股向下/四舍五入取整后产生的**累计权重失真度（Granularity Distortion）**：
  $$\text{Distortion} = \sum_{i} |w_{\text{discrete}, i} - w_{\text{target}, i}|$$
* **评估基准**:
  * $\text{Distortion} \le 15\%$：资金充足，可真实落地；
  * $\text{Distortion} > 25\%$：资金量过小，100 股单手面值过高导致资产配比失真，守卫直接**报错拦截**，并给出当前策略下的**最低建议启动资金**。

```python
from fin_skills.api import get

guard = get("board_lot_feasibility")
weights = {"510300": 0.20, "511010": 0.30, "518880": 0.25, "513100": 0.25}
# 1.5 万元资金尝试配置一手就要 1.35 万元的国债 ETF 511010
result = guard.run(capital=15000.0, target_weights=weights)
print(result.passed)  # False (报错拦截)
# "CRITICAL: Capital 15,000 RMB causes extreme granularity distortion of 33.1%. Minimum recommended capital: 101,000 RMB."
```

---

### 2.3 `CashDragGuard` (闲置现金拖累与隔夜增益守卫)
* **所属模块**: `fin_skills.china.cash_yield_optimizer`
* **API 标识**: `fin_skills.api.get("cash_drag")`
* **运作原理**:
  A 股交易结束后，投资者账户中的可用未投资现金若任由券商结息，年化仅 0.30%。守卫在收盘前自动检测闲置资金，并提供**零违约风险的国债逆回购（GC001 / 204001, R-001 / 131810）或场内货币基金（511880）**部署指令。
* **周四 3 天利息加速器 (Thursday Multiplier)**：
  由于中国结算 T+1 交收规则，**周四下单 1 天期逆回购（GC001），资金在周五早上 09:15 即解冻可用用于买股票，但计息天数享受周五、周六、周日整整 3 天！**守卫会在周四自动亮起加速器提醒。

```python
from fin_skills.api import get

guard = get("cash_drag")
# 10 万元账户中闲置 2.5 万元，且今日是周四
result = guard.run(idle_cash=25000.0, total_capital=100000.0, trade_date="2026-09-17")
print(result.findings[0].message)
# "Action before 15:30: Execute 1-day reverse repo for 25,000 RMB (25 lots). Lending overnight captures 3 days of ~2.10% interest."
```

---

## 3. 持仓状态机与死区增量调仓算法 (`DeadbandRebalancer`)

在实盘中，投资者绝非每次都从零全量建仓，而是持有既存资产，并按月新增定投资金。

```mermaid
flowchart TD
    A["输入当前持仓状态 (my_holdings.json)<br/>+ 今日市场最新价格"] --> B{"计算各资产权重偏离度<br/>Max |w_curr - w_tgt|"}
    B -->|偏离度 <= 5% 死区<br/>且无大额增量现金| C["🟢 判定: HOLD_WITHIN_DEADBAND<br/>保持现状，零交易摩擦"]
    B -->|偏离度 > 5% 死区<br/>或有定投资金流入| D{"是否有新增定投资金?<br/>(new_cash_inflow > 0)"}
    D -->|是 (按月定投)| E["🔵 增量优先平抑 (Inflow-First)<br/>仅将新增现金买入落后资产<br/>完全免除卖出资产手续费"]
    E --> F{"平抑后是否重回死区?"}
    F -->|是| G["✅ 成功平抑: 零卖出换手完成再平衡"]
    F -->|否| H["🟠 全量双向再平衡<br/>按100股整手卖出超配，买入低配"]
    D -->|否| H
```

### 3.1 核心机制
1. **死区节流 (Deadband Throttling)**：
   设置 $\pm 5\%$ 相对死区与 $\pm 2\%$ 绝对死区。当由于日常价格波动导致微小漂移时，系统坚决不发调仓指令，杜绝无效换手。
2. **增量定投优先平抑 (Inflow-First Balancing)**：
   投资者每月存入 3000~10000 元工资定投。算法**优先使用这笔增量现金购买当前配比最落后的资产**。绝大多数情况下，只需单向买入即可将组合拉回均衡线，彻底免除“卖出高配资产支付佣金”的双重摩擦。

---

## 4. 14:30 自动巡检与多端 Webhook 机器人

### 4.1 为什么是 14:30？
A 股交易时间为 09:30-11:30，13:00-15:00。
* 14:30 之前行情多有反复；
* 14:30 之后全天多空搏杀基本定型；
* 距离 15:00 收盘还有 30 分钟，投资者有充裕时间通过手机或电脑进行限价挂单；
* 15:00 收盘后至 15:30，还有 30 分钟专属时间执行 GC001 国债逆回购。

### 4.2 命令行启动与实操

```bash
# 1. 运行每日 14:30 自动巡检（读取现有持仓）
python3 research/production/live_advisor_bot.py --holdings research/production/my_holdings.json

# 2. 发工资日：带 5,000 元新增定投资金运行（自动执行增量优先买入）
python3 research/production/live_advisor_bot.py --holdings research/production/my_holdings.json --inflow 5000

# 3. 盘中向飞书 / 企业微信群机器人推送决策富文本卡片
python3 research/production/live_advisor_bot.py --webhook "https://open.feishu.cn/open-apis/bot/v2/hook/xxxx"

# 4. 确认已在券商完成挂单，自动更新持仓状态文件并自增调仓版本
python3 research/production/live_advisor_bot.py --holdings research/production/my_holdings.json --confirm
```

### 4.3 自动化调度 (Linux Crontab)
编辑 `crontab -e`，添加以下工作日自动任务（周一至周五 14:30 自动执行并推送到飞书）：

```cron
30 14 * * 1-5 /usr/bin/python3 /path/to/fin-skills/research/production/live_advisor_bot.py --holdings /path/to/fin-skills/research/production/my_holdings.json --webhook "https://open.feishu.cn/open-apis/bot/v2/hook/xxxx" >> /tmp/advisor_cron.log 2>&1
```

---

## 5. 总结

| 优化维度 | 传统回测 / 竞品现状 | `fin-skills` 实盘防御标准 |
| :--- | :--- | :--- |
| **QDII 交易** | 默认按 NAV 或收盘价成交，忽视溢价 | 实时 IOPV 比对，>3% 自动硬性熔断并重定向避险 |
| **成交颗粒度** | 假定任意小数股成交（Fractional shares） | 强制 100 股整数手，偏离超标预警并给出最低资金门槛 |
| **现金管理** | 假定现金躺赚 2.0% 无风险收益 | 区分 0.3% 活期与 2.1% 逆回购，周四自动激活 3 天利息加速 |
| **再平衡操作** | 机械按月双向全量卖出/买入 | 带死区节流，工资定投增量优先单向平抑，零卖出磨损 |
| **交付形态** | 仅能在 Jupyter Notebook 中看图表 | 14:30 自动化定时巡检，多端 Webhook 决策卡片直达手机 |

通过以上实盘防御体系，`fin-skills` 正式完成了从学术纸面模型到生产级财富管理决策引擎的进化。
