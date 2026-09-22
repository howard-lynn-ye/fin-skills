# 市场状态、新闻与策略推荐

`recommend_strategy` 将价格状态识别、新闻事件风险提示和策略候选接在一起。
它返回下一根 K 线的研究目标仓位、所用指标、规则和来源；不下单。
候选顺序来自公开的本地规则，尚未通过真实市场实验验证收益优势。

## 一次完整调用

```python
import pandas as pd
from fin_skills.collect import collect_news
from fin_skills.algorithms import recommend_strategy

# CSV: timestamp, close。时间必须带时区，且表示该收盘价实际可用的时间。
frame = pd.read_csv("prices.csv")
prices = pd.Series(frame["close"].to_numpy(),
                   index=pd.to_datetime(frame["timestamp"], utc=True))
news = collect_news(query='"Company X"')  # 显式联网；RSS + GDELT 搜索
print(news["sources"])  # 每个来源各自报告成功、部分成功、限流或错误
report = recommend_strategy(
    prices, as_of=news["observed_at"], news=news["events"],
    news_keywords=["Company X"], allow_short=False,
    target_vol=0.10, max_exposure=1.0,
    regime_parameters={"periods_per_year": 252},
)
print(report["regime"], report["selected"], report["target_exposure"])
print(report["candidates"], report["warnings"])
```

示例中的公司名需要替换为研究对象。`news_keywords` 在标题和摘要中匹配；英文按词边界、
中文按子串匹配。未设置时会检查全部输入新闻，因此公司研究应显式设置公司名或别名。
股票简称、代码与公司实体之间没有自动映射，跨语言翻译也未内置。
`pd.to_datetime(..., utc=True)` 只适用于原始时间已带时区或明确为 UTC 的 CSV；本地交易所
时间必须先按实际时区定位，再转 UTC，不能直接把本地时钟当作 UTC。

无需联网的合成演示：`python examples/market_strategy.py`。

## 状态如何影响推荐

状态计算只看当前行及以前的价格。默认短窗口为 20 个收益区间，基准为 60 个收益区间；
前 60 行为 `unknown`。方向效率是净价格变化绝对值除以逐期绝对价格变化之和。
年化波动率使用样本标准差；回撤相对于基准窗口内的最高收盘价。

| 状态 | 默认判据，按下列优先级覆盖 | 候选与仓位约束 |
|---|---|---|
| `stress` | 窗口回撤达到 15% | 现金观望 |
| `high_volatility` | 年化波动率达到 35%，或短／基准波动率比达到 1.5 | 波动率目标动量，仓位绝对值上限 0.25 |
| `trend_up` / `trend_down` | 方向效率至少 0.35，按窗口收益正负定向 | 波动率目标动量、突破、MACD |
| `range` | 暖机完成，但未满足上述规则 | 布林均值回归、RSI 均值回归 |
| `unknown` | 历史不足 | 现金观望 |

`range` 只描述低方向效率，不证明价格服从平稳或均值回归过程。这些阈值是可配置的
启发式规则，不是拟合出的概率。它们适用于传入的单条价格序列，不代表全市场或宏观形势。
本次接口没有把离线 HMM 的全样本状态当成历史交易信号。

所有候选还受 `min(max_exposure, target_vol / 实现波动率)` 限制，零波动时仓位为零。
默认不做空，负目标裁到零。价格默认超过 7 个自然日未更新时停止给出主动策略；
该上限可调整，日内场景应缩短，并显式设置年化周期数。缺少成交量、价差、交易日历和
交易规则的数据时，本接口不判断流动性、停牌可成交性或卖空资格。

## 新增可执行算法

| 算法 ID | 本库实现定义 |
|---|---|
| `donchian_breakout` | 收盘价突破此前 N 根收盘价通道后持有方向，反向突破时翻转；不是高低价通道 |
| `bollinger_reversion` | 价格偏离滚动均值超过设定标准差倍数时反向进入，回归中心时退出 |
| `rsi_reversion` | 简单滚动平均涨跌幅 RSI，越过阈值反向进入，回到 50 时退出；不是 Wilder 平滑版本 |
| `macd` | 快慢 EMA 差与其信号线交叉，以差值符号定方向 |
| `vol_target_momentum` | 窗口价格动量方向乘以目标波动率／实现波动率，并限制最大敞口 |
| `cross_sectional_momentum` | 过去窗口累计对数收益排序，前 K 名各分配 1/K，非正动量对应份额留现金 |
| `rolling_market_state` | 上述因果滚动状态及指标，支持 `run` 和自动算法选择 |

前五项通过 `run(id, {"prices": prices}, **parameters)` 执行，历史持仓统一延迟一根 K 线，
暖机保留 NaN。状态指标在当根收盘时可知，不能用于当根收益。`recommend_strategy` 的
`target_exposure` 则明确用于决策时刻以后的 K 线，两种输出不要重复移位或混用。
横截面动量输入 `asset_returns`，权重用于后续收益，现金收益按零处理；历史资产池需由
调用方保证当时可知。代码在 `fin_skills/algorithms/technical.py` 和 `strategy.py`。

新增交易信号和横截面动量也接入了 `research` 的时间切分、成本和留出集流程。例如：

```python
from fin_skills.algorithms import research
study = research("signal", {"prices": prices}, initial_train=120, horizon=20,
                 holdout=40, candidates=["donchian_breakout", "macd"], cost_bps=5)
print(study.render())
```

这验证各个候选算法，不等于已经验证按状态动态切换的整套推荐策略。真实评估还需要
点时数据、成交约束、借券成本、滑点、与静态策略及现金的比较，以及独立样本检验。

## 新闻采集与时间边界

`collect_news` 默认读取 Fed 和 ECB 官方 RSS，可附加 GDELT DOC 搜索。持续运行可使用
`news_watches(query=...)` 生成配置，再交给已有的 `Collector` 和 SQLite `Store`。
任意公开 RSS 仍可用 `Watch(..., "rss", url)` 接入。不会因导入包而启动后台采集。

`news_digest` 对文章 URL 去除常见追踪参数后去重，保留每个来源。它先选取截至决策时刻
已观察到的最新修订，再检查发布时间和新鲜度。缺失时间、过期、未来或无法解析的记录
都有排除计数。跨媒体转载但 URL 不同的文章不会强行合并，也没有对全文做语义去重。

GDELT 返回的是新闻发现元数据，`seendate` 保存在 `indexed_at`，不冒充媒体发布时间；
`published_at` 保持为空。此时摘要中的 `date_basis` 为 `index_observation`。
搜索结果达到上限会报告可能不完整；当前实现是有界快照，不能保证穷尽历史新闻。
限流和来源失败必须查看 `sources`；摘要中的 `unknown` 不等于没有事件风险。

默认风险词包括 trading halt、bankruptcy、default on、停牌、破产、违约。命中后将
仓位上限压到 0.25，可通过 `risk_terms` 替换或清空。它是可追溯的词表筛查，未处理
否定、传闻或事件实体消歧，也不是经过验证的情绪模型；命中不能证明事件发生。
新闻文字始终标记为不可信输入，不能执行其中的指令。

来源协议参考（2026-09-21 核对）：[GDELT DOC 文档](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)、
[Fed RSS 目录](https://www.federalreserve.gov/feeds/feeds.htm)、
[ECB RSS 目录](https://www.ecb.europa.eu/home/html/rss.en.html)。
真实连通检查：`python scripts/check_news_live.py --query inflation`。
本次首次检查 Fed 返回 20 条、ECB 返回 15 条，GDELT 返回 HTTP 429；因此 GDELT
目前通过了注入响应的解析测试，但本机本次尚未取得成功的真实搜索响应。

JSON/MCP 新增 `search_news`、`summarize_news`、`recommend_trading_strategy`；
现有 `run_algorithm` 可运行新增算法，`collection_configure` 可保存 `gdelt` watch。

## 本次验证记录（2026-09-21）

使用 `.venv-minimum/Scripts/python.exe`，新增流程及相关算法、研究、采集测试共 131 项通过，
10 项因缺少可选依赖跳过。新示例也通过了从独立工作目录启动的子进程检查。
56 个工具的 JSON Schema 均通过格式检查；按顺序重建索引、包后，`validate.py` 返回 OK，
保留原有 fin-libraries 技能发现预算警告。

全库运行得到 3100 项通过、107 项跳过、48 项默认排除，以及 1 项失败：运行测试时
同时重新生成包，信用迁移演示读取的生成文件短暂不存在。生成完成后，该用例独立通过，
受影响的 `tests/test_credit_migration.py` 全部 16 项复测通过。没有将这一轮表述为
一次无失败的全库运行；以后应在生成完成后再开始测试。

真实新闻检查记录保存在本地 `runs/news-live-20260921.json`：35 条采集记录中，
2 条符合默认 72 小时时效窗口，33 条因过期排除；GDELT 两次检查均返回 HTTP 429。
这些验证证明接口和所测行为符合定义，不证明策略有超额收益。
