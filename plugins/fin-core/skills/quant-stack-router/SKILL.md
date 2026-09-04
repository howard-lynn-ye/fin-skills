---
name: quant-stack-router
description: >-
  Entry point for Python quantitative finance and algorithmic trading: picks the right library and
  flags where the model's training prior is stale. Routes to market data, point-in-time SEC and
  macro data, backtesting engines, broker APIs, indicators, factor research, portfolio and risk
  analytics, backtest validation, and derivatives pricing. TRIGGER — read BEFORE writing code
  whenever the task involves stock, ETF, futures, FX, options, crypto or bond data; a backtest,
  strategy, signal, alpha or factor; a broker, order or paper-trading connection; Sharpe, drawdown,
  tearsheet or attribution; portfolio weights; or any of yfinance, pandas-datareader, openbb,
  polygon, databento, EODHD, akshare, tushare, ccxt, edgartools, fredapi, TA-Lib, pandas-ta,
  vectorbt, backtrader, backtesting.py, zipline, nautilus_trader, freqtrade, qlib, ib_insync,
  ib_async, alpaca, alphalens, PyPortfolioOpt, riskfolio, skfolio, quantstats, mlfinlab, QuantLib,
  arch. SKIP only for generic Python with no market, money or trading semantics.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Python quant stack — router

Your training prior on this ecosystem is **stale in ways that silently produce wrong numbers
rather than errors**. Read §1 before writing code. Then jump from §2 to the one skill that owns
your task; do not try to answer from memory.

Every fact in this library carries a verification date. Where a fact is likely to have moved,
re-check the primary source listed in `shared/live-sources.md` rather than trusting the cache.

## 1. Version drift — where your prior is wrong

*(verified 2026-09-03 against the PyPI JSON API and each project's own repo)*

| Area | Stale prior you probably hold | Current reality |
|---|---|---|
| **TA-Lib install** | "needs the C library compiled by hand; no Windows wheels" | **Solved.** 0.7.1 ships 54 wheels incl. `cp311-win_amd64`. `pip install TA-Lib` just works. |
| **QuantLib install** | "a nightmare, build from source" | **Solved.** 1.43 ships `cp39-abi3-win_amd64`; one wheel covers 3.9+. No sdist at all. |
| **IBKR client** | `ib_insync` | **Dead** (last release 2023-07-02). Successor is **`ib_async`** (2.1.0, 2025-12-08). |
| **TD Ameritrade** | `tda-api` | **Dead** (2022-06). TDA absorbed into Schwab → **`schwab-py`**. |
| **Alpaca** | `alpaca-trade-api` | **Deprecated** (2024-01). → **`alpaca-py`**. |
| **`pandas_datareader`** | `pdr.get_data_yahoo(...)` | **Removed in 0.11.0.** Yahoo, Stooq, Tiingo, IEX, Quandl, AlphaVantage, Morningstar, `Options` — all deleted. It is now a **macro-only** library (FRED, Fama-French, OECD, Eurostat, EconDB, BoC). Every equity tutorial using it is broken. |
| **`yfinance` adjustment** | `yf.download()` returns raw OHLC + `Adj Close` | **`auto_adjust=True` since 1.0.** OHLC are already adjusted and **there is no `Adj Close` column** — `df['Adj Close']` raises `KeyError`. |
| **`backtrader`** | "a standard choice" | **Unmaintained** (last release 2023-04-19) **and GPL-3.0**. |
| **`mlfinlab`** | "the AFML reference implementation" | **Gone.** Not installable from PyPI; the GitHub source is stubbed (**every function body is `pass`**); proprietary all-rights-reserved. Use **`RiskLabAI`** + **`purgedcv`**. |
| **`pandas-ta`** | "the pandas-native TA-Lib" | See `signal-construction` — there is a **supply-chain caution** on the current PyPI package. Maintained successor: **`pandas-ta-classic`**. |
| **`openbb`** | MIT | **Relicensed to AGPL-3.0 on 2024-05-14.** Network copyleft. |
| **`backtesting.py`** | permissive | **AGPL-3.0.** |
| **`PyBroker`** / **`vectorbt`** | open source | Both are **Apache-2.0 + Commons Clause** — you may not sell a product or service deriving substantially from them. Not OSI-open-source. |
| **`nautilus_trader`** | "works everywhere" | Requires **Python >=3.12,<3.15**. Will not install on 3.11. |
| **`fracdiff`** | the fractional-differencing package | **Archived** 2023-12; `requires_python <3.10`. Unusable on modern Python. |
| **Microsoft Qlib CN data** | `qlib_data --region cn` downloads it | **Official dataset disabled** ("more restrict data security policy"). Use the community mirror `chenditc/investment_data`. |
| **Alpha Vantage free tier** | 500 requests/day | **25 requests/day.** Effectively a demo. |
| **`polars`** | "a fast Rust dataframe library" | The `polars` wheel is now an **empty 865 KB shim** (`py3-none-any`, no compiled code) that hard-depends on `polars-runtime-32`. **A lockfile listing only `polars` does not pin the engine.** |
| **`polars.join_asof`** | "pandas is the one that silently misjoins" | **Inverted.** `pandas.merge_asof` **raises** on unsorted keys; **`polars.join_asof` does NOT check sortedness when `by=` is given** — silently wrong rows. |
| **SEC XBRL `frames` API** | "a clean cross-section" | **Not point-in-time and cannot be made so** — no `filed` field. 53% of its CY2023 values come from filings made in 2026. Never backtest on it. |

## 2. Quick task reference

Written in the words you would actually use. `->` is a literal path to read next.

**"get stock prices" / "download OHLCV" / "which data source" / yfinance / polygon / databento / EODHD / tiingo / survivorship-free universe / delisted tickers / trading calendar / split & dividend adjustment**
-> `market-data-sourcing` skill. Start with its `plugins/fin-core/skills/market-data-sourcing/references/_decision-table.md`.

**"as-of join" / merge_asof / "join quotes to trades" / Parquet / polars / DuckDB / ArcticDB / "store tick data" / "my timestamps are wrong" / "too big for memory" / "different numbers when I parallelise"**
-> `market-data-engineering` skill. **Its as-of-join section is where look-ahead most often enters a pipeline.**

**"fundamentals" / 10-K / 10-Q / 8-K / EDGAR / XBRL / "point-in-time financials" / "as-of-date fundamentals" / earnings dates / CIK / restatements**
-> `fundamental-and-macro-data` skill, §SEC. **Read its restatement and `acceptanceDateTime` sections before writing any event study.**

**"macro data" / FRED / CPI / GDP / NFP / "revised data" / vintage / ALFRED / World Bank / IMF / Eurostat**
-> `fundamental-and-macro-data` skill, §Macro. GDP is revised for years — `plugins/fin-core/skills/fundamental-and-macro-data/references/fredapi.md` covers the vintage API and its three bugs.

**"backtest this" / "which backtesting library" / vectorbt / zipline / nautilus / LEAN / freqtrade / "my backtest looks too good"**
-> `backtesting-engines` skill. If the backtest looks too good, go to `research-integrity-guards` **first**.

**"connect to IBKR" / TWS / ib_async / place an order / paper trading / Alpaca / Schwab / Tastytrade / "make sure I don't send a live order"**
-> `broker-execution-apis` skill. Its order-safety patterns are mandatory reading before any code that can transmit an order.

**"RSI" / "MACD" / "moving average" / TA-Lib / "which indicator library" / "does this indicator repaint" / streaming indicators**
-> `signal-construction` skill.

**"is my factor any good" / IC / alphalens / quantile returns / Fama-French / Fama-MacBeth / event study / abnormal returns / CAR**
-> `factor-and-timeseries-research` skill.

**"forecast returns" / ARIMA / GARCH / volatility model / Nixtla / sktime / darts / Prophet / time-series foundation model**
-> `factor-and-timeseries-research` skill, §Forecasting. For volatility specifically, `plugins/fin-core/skills/factor-and-timeseries-research/references/arch.md`.

**"optimize portfolio weights" / mean-variance / Black-Litterman / risk parity / HRP / CVaR / efficient frontier / covariance shrinkage**
-> `portfolio-and-risk` skill, §Optimization.

**"Sharpe ratio" / drawdown / tearsheet / quantstats / pyfolio / "annualize returns" / attribution / VaR**
-> `portfolio-and-risk` skill, §Analytics. **Read its metric-correctness audit** — several popular libraries compute these wrong.

**"is this result real" / overfitting / "I tried 200 strategies" / deflated Sharpe / PBO / walk-forward / purged CV / p-hacking / "how many trials"**
-> `backtest-validation` skill. This is the single highest-value skill in the library.

**"price an option" / Greeks / implied vol / vol surface / SABR / QuantLib / yield curve / bond pricing / swap**
-> `derivatives-pricing` skill.

**"look-ahead bias" / "survivorship bias" / "data leakage" / "point-in-time" / "is my backtest honest" / "review my research design"**
-> `research-integrity-guards` skill. Also load this whenever you are about to *report a number*.

**A-share / 沪深 / akshare / tushare / baostock / 复权 / 涨跌停 / T+1 / vnpy / qlib / 北交所**
-> `fin-china` plugin: `china-ashare-data`, `china-trading-stack`.

**crypto / ccxt / Binance / perpetuals / funding rate / freqtrade / hummingbot**
-> `fin-crypto` plugin: `crypto-data-and-execution`.

**"LLM trading agent" / TradingAgents / FinGPT / FinRobot / RD-Agent / "does AI trading work" / finance MCP server**
-> `fin-llm` plugin: `llm-finance-agents` (read its evidence section before building anything), `finance-mcp-servers`.

## 3. Searching this library

Most library detail lives **inline in each skill's own tables**, not in separate files. Some skills
additionally carry `references/*.md` for material too long to inline — a decision table, a
methodology, a single deep library note. `catalog/index.json` lists exactly which, generated from
frontmatter, so check there rather than assuming a file exists.

When the table above does not name your library, grep the skill bodies first:

```bash
# Which skill covers a given package?
grep -ril "riskfolio" plugins/*/skills/*/SKILL.md

# What does it get wrong? (traps are marked with a siren in every skill)
grep -i -B2 -A6 "quantstats" plugins/fin-core/skills/portfolio-and-risk/SKILL.md

# Only some skills have reference files; list them before reading
ls plugins/*/skills/*/references/
```

## 4. Three rules that override any library's defaults

1. **A library's default is not a safe default.** Adjustment mode, CV splitter, risk-free rate, and
   annualization factor all default to something wrong for research in at least one popular
   package. Set them explicitly, always.
2. **Never report a backtest number without its trial count.** How many variants were tried is part
   of the result. See `backtest-validation`.
3. **Code license ≠ data license.** yfinance is Apache-2.0; the Yahoo data it fetches is
   personal-use-only. The permissive license on a scraper grants you nothing about the scraped data.
