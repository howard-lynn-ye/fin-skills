# fin-skills 文档中心 (Documentation Hub)

`fin-skills` 是一个融合了源码级验证知识库、自动化防伪审计引擎、实盘事前交易防御与交互式大类资产决策系统的综合量化投研框架。包含 127 个金融技能、36 个可执行守卫、46 个 JSON/MCP 工具以及完整的 10 年跨周期资产配置引擎。

---

## 🚀 1. 交互式实盘决策看板与生产工具 (Interactive Tools)

| 工具 / 脚本 | 访问方式 / 启动命令 | 功能定位与核心价值 |
| :--- | :--- | :--- |
| **交互式 Web 决策看板** | `http://shwaihe.c.googlers.com:8088/`<br/>或启动：`python3 research/production/web_dashboard.py` | **零门槛可视化交互界面**。支持滑动选择资金（5万~200万）、一键切换策略模式、实时环形图与 100 股整数手买卖指令生成、一键复制券商下单单。 |
| **每日实盘投顾命令行** | `python3 research/production/daily_advisor.py --capital 100000` | 实时拉取东方财富盘中行情，根据目标波动率计算最优配比，规整整百股并评估免印花税手续费。 |
| **14:30 自动巡检与多通道机器人** | `python3 research/production/live_advisor_bot.py --holdings ...` | 盘中 14:30 自动执行 QDII 溢价熔断、死区增量调仓、14:50 GC001 逆回购收益规划，支持飞书/企微 Webhook。 |

---

## 📘 2. 新手与小白友好入门指南 (Beginner & AI Guides)

| 指南文档 | 路径 | 核心内容与解决问题 |
| :--- | :--- | :--- |
| **AI 从业者的量化金融背景认知指南** | [`guides/quantitative_finance_background_for_ai.md`](guides/quantitative_finance_background_for_ai.md) | 从小白和算法工程师视角出发，生动讲解 A 股与海外市场运行规则（T+1、跌停板、印花税减半、小资金整百股陷阱），配有直观流程图与对比表。 |
| **经济金融与量化交易核心术语速查词典** | [`guides/financial_terminology_glossary.md`](guides/financial_terminology_glossary.md) | 涵盖收益风险、因子选股、衍生品、交易制度等四大领域的 24 个高频黑话（Alpha/Beta/夏普/回撤/IC/贴水/点进时等），提供通俗白话与数学公式。 |

---

## 🔬 3. 跨周期实证研究、压力测试与实盘白皮书 (Empirical Research & Whitepapers)

| 研究报告 | 路径 | 核心结论与实证数据 |
| :--- | :--- | :--- |
| **全球大类资产穿越牛熊10年压力测试白皮书** | [`guides/global_asset_allocation_10y_whitepaper.md`](guides/global_asset_allocation_10y_whitepaper.md) | 基于 2015-2026 跨越 10 年（2,800+ 交易日）真实历史点入数据，穿透 2015 股灾、2018 熊市、2020 流动性危机、2022 双杀，实证验证稳健策略最大回撤 9.38%（沪深300为 46.7%）。 |
| **实盘事前交易防御体系与执行指南** | [`guides/live_trading_and_pre_trade_guards.md`](guides/live_trading_and_pre_trade_guards.md) | 详解四大事前交易守卫：QDII 实时溢价率熔断、100 股整数手残差检测、增量调仓死区节流、14:50 GC001 闲置现金收益增益。 |
| **实战交易模拟与多预算实证报告** | [`guides/empirical_trading_simulation_findings.md`](guides/empirical_trading_simulation_findings.md) | 记录 5 万、20 万、100 万不同资金体量在个股池 vs ETF 池上的执行差异与物理可行性边界。 |

---

## 🛠️ 4. 系统集成、采集与操作规范 (System & Operations)

| 文档名称 | 路径 | 适用场景 |
| :--- | :--- | :--- |
| **公共信息采集与跟踪指南** | [`COLLECTION.md`](COLLECTION.md) | 真实新闻/披露数据源、RSS/SEC Form 4/13F/Bluesky 抓取与轮询调度。 |
| **研发里程碑完成审计报告** | [`COMPLETION_AUDIT.md`](COMPLETION_AUDIT.md) | 仓库各项功能的自审记分卡、Guard 覆盖率与测试通过情况。 |
| **系统集成手册 (中文版)** | [`USER_MANUAL_ZH.md`](USER_MANUAL_ZH.md) | 如何在外部系统或现有量化体系中集成调用 `fin_skills.api`。 |
| **生产运维与部署指南 (中文版)** | [`OPERATIONS_GUIDE_ZH.md`](OPERATIONS_GUIDE_ZH.md) | 自动化脚本部署、定时任务配置与工作流监控。 |
| **英文主说明文件** | [`../README.md`](../README.md) | 完整的 127 技能清单、API 规范与架构概览。 |
| **中文主说明文件** | [`../README_ZH.md`](../README_ZH.md) | 中文全景架构、双层解决方案与常见量化错误纠偏。 |

---

> [!TIP]
> 💡 **快速上手建议**：
> 1. 初次接触量化：先阅读 [`guides/quantitative_finance_background_for_ai.md`](guides/quantitative_finance_background_for_ai.md) 建立宏观认知；
> 2. 遇到专业名词：随时查阅 [`guides/financial_terminology_glossary.md`](guides/financial_terminology_glossary.md)；
> 3. 需要实盘决策：启动 Web 看板 `python3 research/production/web_dashboard.py` 或直接在浏览器打开 [`research/production/dashboard.html`](../research/production/dashboard.html)。
