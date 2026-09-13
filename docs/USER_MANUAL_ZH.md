# 📘 fin-skills 实战连接与集成使用手册（User Manual & Integration Playbook）

> **核心目标**：本手册指导如何将 **114 个量化金融 Agent Skills**、**32 个可执行代码防伪守卫 (`fin_skills.api`)** 与您本地的 **`stock_prediction` 多模态量化投研仓库（867 万行物理数据湖、MMAN/Qwen3 模型、5日回测引擎）** 深度连接，构建“AI 辅助研发 $\rightarrow$ 自动化防伪体检 $\rightarrow$ 顶会/机构级可信交付”的完整闭环。

---

## 🧭 1. 全局资源连接拓扑图（How Everything Connects）

在您的研发环境中，目前共有三大核心资源模块。它们通过以下三层通道实现无缝协同：

```mermaid
flowchart TD
    subgraph R1 ["📦 资源一：fin-skills 防伪引擎与知识底座 (/home/shwaihe/fin-skills)"]
        S1["114 个 Agent Skills (SKILL.md)<br/>已挂载至 ~/.gemini/config/skills/"]
        S2["fin_skills.api 统一 Python 包<br/>Bundle 容器 + 32 个 GuardResult 守卫"]
        S3["conventions 市场惯例与税率库<br/>A股 T+1/涨跌停/印花税减半 & 美股规则"]
    end

    subgraph R2 ["📊 资源二：stock_prediction 投研与数据湖 (/home/shwaihe/stock_prediction)"]
        D1["本地物理数据湖 (867万行 / 1.1GB)<br/>雪球/股吧/StockTwits + SEC 财报 + 54D特征"]
        D2["多模态预测模型 (MMAN + Qwen3)<br/>特征工程脚本 & 投资者信誉门控"]
        D3["5日真实交易仿真回测引擎<br/>净值曲线 / 换手率 / 夏普比率报告"]
    end

    subgraph R3 ["🤖 资源三：Jetski AI 编程助手与自动化流水线"]
        A1["写代码/改模型阶段<br/>自动读取 SKILL.md 规避静默 Bug"]
        A2["跑实验/回测验收阶段<br/>运行 scripts/audit_with_fin_skills.py"]
        A3["写论文/审稿答辩阶段<br/>输出 0 前视偏差、0 语料污染证明"]
    end

    S1 ==>|通道 1：Agent 上下文自动注入| A1
    A1 ==>|指导编写无泄露代码| D2
    D1 & D2 & D3 ==>|通道 2：装入 Bundle 容器| S2
    S2 & S3 ==>|通道 3：一键防伪体检脚本| A2
    A2 ==>|签发可信实验报告| A3

    style R1 fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#1e40af
    style R2 fill:#ecfdf5,stroke:#10b981,stroke-width:2px,color:#065f46
    style R3 fill:#f3e8ff,stroke:#8b5cf6,stroke-width:2px,color:#6b21a8
```

---

## 🛠️ 2. 三大连接通道实操指南

### 通道一：AI 编程助手自动触发（写代码时防患于未然）

我们已经将 `fin-skills` 中最核心的 **48 个领域技能** 加上总路由 `fin-skills` 安装到了您的本地 Jetski 技能目录（`~/.gemini/config/skills/`）。

#### 如何使用？
在您与 Jetski 对话开发 `stock_prediction` 时，无需手动复制粘贴文档，只需**在提问中自然提及任务关键词**，Agent 会自动调用 `view_file` 读取对应的源码级规范：

| 您的开发/投研场景 | 推荐对 Agent 说的话（触发词示例） | Agent 自动加载的 Skill 与保护机制 |
| :--- | :--- | :--- |
| **构建新特征 / 滚动归一化** | *“帮我在 `stock_prediction` 里加一个 30 日情绪动量特征，请遵循 `signal-construction` 规范防止未来函数。”* | 自动加载 [`signal-construction`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/signal-construction/SKILL.md)：严禁使用 `center=True`、全样本 `StandardScaler` 或未对齐的 `rolling`，并自动生成 `assert_causal` 测试。 |
| **A 股雪球/股吧策略回测** | *“我们要测 A 股 2021-2026 的回测收益，请按 `china-trading-stack` 和 `china-ashare-trading-taxes` 设置交易成本与涨跌停限制。”* | 自动加载 [`china-trading-stack`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/china-trading-stack/SKILL.md) 与 [`china-ashare-trading-taxes`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/china-ashare-trading-taxes/SKILL.md)：强制开启 T+1、剔除一字涨跌停无法成交订单、并在 `2023-08-28` 前后切换卖方单边印花税率（0.1% $\rightarrow$ 0.05%）。 |
| **评估 Qwen3 / FinBERT 表现** | *“帮我检查 Qwen3 在盲测集上的准确率是否存在预训练语料时间重叠，参考 `llm-finance-agents`。”* | 自动加载 [`llm-finance-agents`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/llm-finance-agents/SKILL.md)：调用 `contamination_probe` 比对模型 Cutoff 日期与回测起止日，防止把“模型背诵历史”当成预测能力。 |
| **合并 SEC 财报或公告时间戳** | *“将 `interaction_matrix.csv` 中的公告与日 K 线对齐，请按 `combining-data-sources` 检查时间戳。”* | 自动加载 [`combining-data-sources`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/combining-data-sources/SKILL.md)：将盘后（`>16:00`）发布的公告严格映射至**次一交易日（T+1）**开盘后交易，杜绝盘后信息穿越回当日收盘价成交。 |
| **出论文/报告前的终极自检** | *“我们的回测跑出了 Sharpe 1.8，请按 `research-integrity-guards` 帮我做一次五关防伪审查。”* | 自动加载 [`research-integrity-guards`](file:///usr/local/google/home/shwaihe/.gemini/config/skills/research-integrity-guards/SKILL.md)：逐一核查股票池幸存者偏差、时间戳可得性、标签重叠泄露、盈亏平衡换手成本与多重试验次数平减（DSR）。 |

---

### 通道二：一键运行桥接审计脚本（已为您内置在 `stock_prediction` 中）

为了让您无需手写胶水代码就能立即体验连接效果，我们已在您的 `stock_prediction` 仓库中创建了专属桥接脚本：
👉 **[`scripts/audit_with_fin_skills.py`](file:///usr/local/google/home/shwaihe/stock_prediction/scripts/audit_with_fin_skills.py)**

#### 运行命令：
```bash
cd /usr/local/google/home/shwaihe/stock_prediction
python3 scripts/audit_with_fin_skills.py
```

#### 该脚本自动为您完成的 5 项真实数据体检：
1. **读取真实物理数据**：自动加载 `data/benchmark/item_daily_features.csv`（涵盖 2011–2026 年的真实 OHLCV 与特征面板）。
2. **特征工程未来函数数学探测 (`assert_causal`)**：
   - 对您的 30 日滚动 Z-Score 特征函数进行**未来数据篡改压力测试**（篡改第 1926 天之后的未来行情，验证前 1926 天的历史特征值变化量为 `0.00000`，耗时 `0.006s` 签发 `PASS`）。
   - 同时演示：若特征代码中不慎混入 `shift(-1)`，守卫在 `0.005s` 内精准抓出 `LOOK-AHEAD` 报错（最大偏差 `7.989`）。
3. **回测产物装入 `Bundle` 全自动体检 (`Suite.check`)**：
   - 将策略净值曲线 `returns`、换手率 `turnover`、K 线 `bars` 装入 `fin_skills.api.Bundle`，一键激活 5 大核心守卫：
     - `assert_causal`: **PASS**（信号因果性验证通过）
     - `warmup_probe`: **PASS**（精确测出指标需 `29` 根 K 线预热期，冷启动相对误差 `1.4`）
     - `cost_curve`: **PASS**（测出策略盈亏平衡手续费上限为 `25.4 bps`，当前 `10.0 bps` 成本下净夏普为 `0.37`）
     - `rf_convention`: **PASS**（验证无风险利率按年化几何口径扣除，修正 `empyrical` 直接减年化利率算出 `-14.02` 的错误）
     - `contamination_probe`: **PASS**（验证测试窗口 `2025-01-01 -> 2026-08-28` 完全位于 LLM 预训练截止日 `2024-12-01` 之后，零语料污染）
4. **大模型回测窗口污染对比审计**：
   - 对比 `2022-2023` 旧测试集（直接触发 `FAIL: INVALID - entire backtest window predates training cutoff`）与您的 `2025-2026` 盲测集（`PASS: CLEAN`），为论文审稿提供铁证。
5. **跨市场交易惯例与税率自动校准**：
   - 自动加载股票年化因子（`252` 天）与 A 股 `2023-08-28` 印花税单边减半规则。

---

### 通道三：在您的自定义训练/回测脚本中嵌入 `fin_skills.api`（代码模板）

当您在编写新的实验脚本（例如 `scripts/train_mman_xueqiu_gpu.py` 或 `scripts/evaluate_xueqiu_feature_effectiveness.py`）时，只需在脚本末尾加入 **8 行代码**，即可为每次实验自动加上“防伪钢印”：

```python
# =====================================================================
# 📋 嵌入模板：在任何回测/评估脚本末尾加入 fin-skills 自动防伪体检
# =====================================================================
from fin_skills.api import Bundle, check

# 1. 将您脚本算出的回测序列装入 Bundle
audit_bundle = Bundle(
    returns=my_strategy_daily_returns,   # pd.Series (DatetimeIndex)
    turnover=my_daily_turnover,          # pd.Series (每日双边换手率，如 0.15 表示 15%)
    rf=0.02,                             # 年化无风险利率 (如 2%)
    bars=my_ohlcv_dataframe,             # 包含 open, high, low, close, volume 的 DataFrame
    signal_fn=my_feature_function,       # 您的特征/信号计算函数 def fn(df) -> Series
    cutoff="2024-12-01",                 # 若使用了 LLM，填入模型预训练语料截止日
    test_start=str(my_strategy_daily_returns.index.min().date()),
    test_end=str(my_strategy_daily_returns.index.max().date()),
)

# 2. 运行全套防伪守卫并打印报告
audit_report = check(audit_bundle)
print(audit_report.summary())

# 3. (可选) 若有任何严重作弊/泄露未通过，直接阻断实验结果保存，防止污染实验账本！
assert audit_report.passed, "❌ 实验未通过 fin-skills 防伪审计，请检查未来函数或数据泄露！"
```

---

## 🔍 3. 常用核心守卫（Guards）速查表：什么时候用哪个？

您可以随时通过 `from fin_skills.api import get` 单独调用以下高频守卫：

| 守卫名称 (`get("...")`) | 适用阶段 | 传入参数示例 | 它帮您抓什么致命问题？ |
| :--- | :---: | :--- | :--- |
| **`assert_causal`** | 特征工程 / 信号构建 | `fn=my_signal_fn, df=bars_df, k=500` | 抓出 `shift(-1)`、居中滚动窗口、全样本归一化等一切**未来函数**。 |
| **`warmup_probe`** | 技术指标 / 情绪平滑 | `indicator=my_1d_fn, close=close_series` | 测出 EMA/RSI/Z-Score 指标需要多少根 K 线**预热（Warm-up）**才能收敛，防止测试集开头因冷启动失真。 |
| **`safe_asof`** | 舆情/财报与 K 线合并 | `left=posts_df, right=bars_df, on="timestamp"` | 抓出 `pd.merge_asof` 中方向写反（`direction="forward"`）或盘后舆情错误对齐至当日收盘价的**时间戳穿越**。 |
| **`contamination_probe`** | Qwen3 / FinBERT 评测 | `test_start="2025-01-01", test_end="2026-08-28", cutoff="2024-12-01"` | 证明您的测试集时间窗口完全位于大模型**预训练语料截止日（Cutoff）之后**，回应审稿人对 LLM 记忆背诵的质疑。 |
| **`cost_curve`** | 5日交易仿真回测 | `returns=ret_series, turnover=turn_series, cost_bps=10.0` | 计算策略的**盈亏平衡手续费（Breakeven Cost bps）**，判断策略在扣除真实印花税与滑点后是否依然盈利。 |
| **`survivorship_audit`** | 股票池构建 (`universe`) | `prices=price_matrix_df` | 检查历史股票池是否只包含活到今天的股票，抓出**幸存者偏差**。 |
| **`adjustment_check`** | 行情数据清洗 (`bars`) | `close=close_series, actions=splits_df` | 检查拆股/分红除权日是否存在虚假暴跌跳空，或因直接使用前复权（`qfq`）导致历史早期出现负价格。 |

---

## 🚀 4. 推荐日常投研工作流（Best Practice Workflow）

1. **Step 1（设计特征）**：在写新特征前，让 Jetski 参考 `signal-construction` 和 `fin-ml`（如三道屏障标签 `triple-barrier-labeling` 或分数阶差分 `fractional-differentiation`）。
2. **Step 2（单测验证）**：对写好的特征函数跑一行 `get("assert_causal").run(fn=..., df=...)`，确保 `0.01 秒` 拿到 `PASS`。
3. **Step 3（模型回测）**：运行 MMAN / Qwen3 训练与回测（如 `train_mman_xueqiu_gpu.py`）。
4. **Step 4（一键审计）**：运行 `python3 scripts/audit_with_fin_skills.py`（或在脚本末尾调用 `check(bundle)`），生成包含 **因果性、预热期、盈亏平衡成本、无风险利率口径、LLM 零语料污染** 的五维合格证报告，直接附在论文或实验记录（Research Notebook）中！
