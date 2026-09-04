# fin-skills — Agent Skills for Python quantitative finance

**19 [Agent Skills](https://agentskills.io/specification) for Claude Code that tell an LLM which
Python quant-finance library to use, what each one silently gets wrong, and whether a backtest
result is real.** Covers market data, SEC point-in-time fundamentals, backtesting engines, broker
APIs, technical indicators, factor research, portfolio optimization, risk analytics, derivatives
pricing, China A-shares, crypto, and the evidence on LLM trading agents.

Every claim carries a verification date and a marker: ✅ verified at a primary source · ⚠️ secondhand
· ❓ could not verify. Where a library's behaviour was **measured** rather than read — by installing
it and running the comparison — the skill says so.

It exists because the middle of this domain is empty. The ecosystem is saturated at two ends — API
wrappers and knowledge dumps — and nearly vacant at **the methodology that decides whether a result
is real**. `anthropics/skills` contains zero finance skills; the community repos that do exist cover
compliance, crypto execution and bookkeeping well, and research integrity barely at all.

**Keywords:** Claude Code skills · agent skills · quantitative finance · algorithmic trading ·
backtesting · look-ahead bias · survivorship bias · point-in-time data · yfinance · vectorbt ·
QuantLib · akshare · tushare · ccxt · alphalens · deflated Sharpe ratio

## Install

```bash
/plugin marketplace add howard-lynn-ye/fin-skills
/plugin install fin-core@fin-skills
```

Then install only the market plugins you need:

```bash
/plugin install fin-china@fin-skills     # A-share / Greater China
/plugin install fin-crypto@fin-skills    # crypto
/plugin install fin-llm@fin-skills       # LLM agents + the evidence on whether they work
```

> **Raise your skill-listing budget.** Claude Code's default budget is ~1% of the context window
> (~2,000 tokens), and past roughly 20 skills it **silently drops descriptions to name-only** — a
> dropped skill never auto-triggers. `fin-core` alone costs ~2,233 tokens. Set
> `"skillListingBudgetFraction": 0.02` in `~/.claude/settings.json` before installing more than one
> plugin. Check with `/context` (Skills row) or `/doctor`.

## What's here

<!-- BEGIN GENERATED SKILL TABLE -->

| Plugin | Skill | Covers | Refs | Scripts |
|---|---|---|---:|---:|
| `fin-asia` | [`asia-pacific-markets`](plugins/fin-asia/skills/asia-pacific-markets/SKILL.md) | Data and trading for Asia-Pacific outside mainland China. | 0 | 0 |
| `fin-china` | [`china-ashare-data`](plugins/fin-china/skills/china-ashare-data/SKILL.md) | Get China A-share and Greater China market data without the ecosystem's silent traps. | 4 | 0 |
| `fin-china` | [`china-trading-stack`](plugins/fin-china/skills/china-trading-stack/SKILL.md) | Backtest and execute Chinese-market strategies under the rules a Western engine gets wrong. | 3 | 1 |
| `fin-core` | [`backtest-validation`](plugins/fin-core/skills/backtest-validation/SKILL.md) | Decide whether a result survives the number of things you tried. | 3 | 3 |
| `fin-core` | [`backtesting-engines`](plugins/fin-core/skills/backtesting-engines/SKILL.md) | Choose a backtesting engine and know what it silently models wrong. | 6 | 0 |
| `fin-core` | [`broker-execution-apis`](plugins/fin-core/skills/broker-execution-apis/SKILL.md) | Connect to a broker and place orders without accidentally trading live money. | 4 | 1 |
| `fin-core` | [`derivatives-pricing`](plugins/fin-core/skills/derivatives-pricing/SKILL.md) | Price options and fixed income, and get the Greeks and conventions right. | 4 | 1 |
| `fin-core` | [`external-skill-index`](plugins/fin-core/skills/external-skill-index/SKILL.md) | A verified index of every public finance Agent Skill repository — 139 repos, 4,851 SKILL.md files — so you can find what already exists instead of rebuilding it, and avoid the thir | 0 | 0 |
| `fin-core` | [`factor-and-timeseries-research`](plugins/fin-core/skills/factor-and-timeseries-research/SKILL.md) | Judge whether a cross-sectional factor predicts returns, and forecast financial series. | 7 | 1 |
| `fin-core` | [`fundamental-and-macro-data`](plugins/fin-core/skills/fundamental-and-macro-data/SKILL.md) | Company fundamentals and macro series with correct point-in-time semantics. | 3 | 1 |
| `fin-core` | [`market-data-engineering`](plugins/fin-core/skills/market-data-engineering/SKILL.md) | Store, join and parallelize market data you already hold, without corrupting it. | 4 | 2 |
| `fin-core` | [`market-data-sourcing`](plugins/fin-core/skills/market-data-sourcing/SKILL.md) | Choose a market price or reference data vendor and use it without silently corrupting the numbers. | 11 | 1 |
| `fin-core` | [`portfolio-and-risk`](plugins/fin-core/skills/portfolio-and-risk/SKILL.md) | Turn signals into weights, and compute performance metrics that are actually correct. | 9 | 1 |
| `fin-core` | [`quant-stack-router`](plugins/fin-core/skills/quant-stack-router/SKILL.md) | Entry router for Python quantitative finance: names the right library and flags where the model's training prior is stale. | 0 | 0 |
| `fin-core` | [`research-integrity-guards`](plugins/fin-core/skills/research-integrity-guards/SKILL.md) | Second-pass audit that decides whether a finance result is real, applied after the work exists. | 2 | 3 |
| `fin-core` | [`signal-construction`](plugins/fin-core/skills/signal-construction/SKILL.md) | Compute technical indicators and engineered features without leaking the future. | 2 | 2 |
| `fin-crypto` | [`crypto-data-and-execution`](plugins/fin-crypto/skills/crypto-data-and-execution/SKILL.md) | Crypto market data and execution, and how a 24/7 market breaks equity tooling. | 3 | 0 |
| `fin-futures-fx` | [`futures-continuous-contracts`](plugins/fin-futures-fx/skills/futures-continuous-contracts/SKILL.md) | Build and use a futures price series correctly — a continuous contract does not exist in the market, it is stitched, and the stitching method changes your answer. | 0 | 2 |
| `fin-futures-fx` | [`fx-markets`](plugins/fin-futures-fx/skills/fx-markets/SKILL.md) | Trade and backtest FX correctly — quote conventions, pip sizing, and the carry that a spot-only backtest silently omits. | 0 | 1 |
| `fin-libraries` | [`lib-akshare`](plugins/fin-libraries/skills/lib-akshare/SKILL.md) | akshare is the widest free Chinese-market scraper (1,103 public interfaces) and it purges its own PyPI history, so you cannot pin it. | 0 | 0 |
| `fin-libraries` | [`lib-alpaca-py`](plugins/fin-libraries/skills/lib-alpaca-py/SKILL.md) | Alpaca's current Python SDK, which defaults to the paper host but lets url_override silently send live orders from a client that believes it is in the sandbox. | 0 | 0 |
| `fin-libraries` | [`lib-alphalens`](plugins/fin-libraries/skills/lib-alphalens/SKILL.md) | alphalens-reloaded scores cross-sectional factors, and its forward return starts at date t's OWN price - it never lags your factor. | 0 | 0 |
| `fin-libraries` | [`lib-arch`](plugins/fin-libraries/skills/lib-arch/SKILL.md) | The reference GARCH implementation in Python, and the home of SPA/StepM/MCS - which all take LOSSES, so passing returns silently inverts the test and names your worst strategy as t | 0 | 0 |
| `fin-libraries` | [`lib-backtesting-py`](plugins/fin-libraries/skills/lib-backtesting-py/SKILL.md) | Single-asset bar-loop backtester with honest next-open fills, an AGPL-3.0 licence, and an indicator API that computes over the entire series before slicing. | 0 | 0 |
| `fin-libraries` | [`lib-ccxt`](plugins/fin-libraries/skills/lib-ccxt/SKILL.md) | The unified MIT client for 100+ crypto venues - and not a backtester, with an OHLCV endpoint that silently truncates and returns an unclosed final bar. | 0 | 0 |
| `fin-libraries` | [`lib-edgartools`](plugins/fin-libraries/skills/lib-edgartools/SKILL.md) | edgartools is the default free SEC EDGAR client - typed objects for 20+ form types, XBRL statements, no API key - and it 403s on every request until you call set_identity(). | 0 | 0 |
| `fin-libraries` | [`lib-fredapi`](plugins/fin-libraries/skills/lib-fredapi/SKILL.md) | fredapi wraps FRED/ALFRED and is the primary anti-look-ahead tool in macro - and three of its four vintage methods are buggy in source. | 0 | 0 |
| `fin-libraries` | [`lib-freqtrade`](plugins/fin-libraries/skills/lib-freqtrade/SKILL.md) | freqtrade is a live-first crypto bot with the best bias detectors in the field and a backtester that assumes zero slippage always. | 0 | 0 |
| `fin-libraries` | [`lib-ib-async`](plugins/fin-libraries/skills/lib-ib-async/SKILL.md) | The maintained Interactive Brokers Python client - successor to the archived ib_insync - where one digit of the port number is all that separates paper from live. | 0 | 0 |
| `fin-libraries` | [`lib-nautilus-trader`](plugins/fin-libraries/skills/lib-nautilus-trader/SKILL.md) | Event-driven Rust-core engine with the strongest execution modelling in open source, gated to Python 3.12-3.14, where a wrong ts_init silently makes every bar visible one interval  | 0 | 0 |
| `fin-libraries` | [`lib-polars`](plugins/fin-libraries/skills/lib-polars/SKILL.md) | The polars wheel is now an empty 865 KB py3-none-any shim hard-pinned to polars-runtime-32, so a lockfile listing only polars does not pin the engine. | 0 | 0 |
| `fin-libraries` | [`lib-purgedcv`](plugins/fin-libraries/skills/lib-purgedcv/SKILL.md) | The only genuinely sklearn-protocol-compliant purged and embargoed splitter, and the one that refuses to run until you state when each label resolved - understate evaluation_times  | 0 | 0 |
| `fin-libraries` | [`lib-pyportfolioopt`](plugins/fin-libraries/skills/lib-pyportfolioopt/SKILL.md) | Textbook mean-variance and Black-Litterman optimizer whose HRPOpt silently accepts a price matrix where it requires returns and returns plausible garbage. | 0 | 0 |
| `fin-libraries` | [`lib-qlib`](plugins/fin-libraries/skills/lib-qlib/SKILL.md) | Microsoft Qlib (pip name pyqlib, imported as qlib) ships Alpha158/Alpha360 and a default normalizer that leaks your test set into training, silently. | 0 | 0 |
| `fin-libraries` | [`lib-quantlib`](plugins/fin-libraries/skills/lib-quantlib/SKILL.md) | The only broadly-permissive, mature, full-coverage derivatives library in Python, whose global evaluationDate returns an NPV of exactly 0.0 with no warning once it is past expiry. | 0 | 0 |
| `fin-libraries` | [`lib-quantstats`](plugins/fin-libraries/skills/lib-quantstats/SKILL.md) | The tearsheet library whose cagr(rf=...) accepts your risk-free rate and silently discards it - "cagr" sits on an exclusion list inside _prepare_returns, which dispatches on the ca | 0 | 0 |
| `fin-libraries` | [`lib-riskfolio`](plugins/fin-libraries/skills/lib-riskfolio/SKILL.md) | The 26-risk-measure portfolio optimizer whose stateful API optimizes against stale or missing mu and Sigma - with no error - if you forget assets_stats(). | 0 | 0 |
| `fin-libraries` | [`lib-skfolio`](plugins/fin-libraries/skills/lib-skfolio/SKILL.md) | The sklearn-compatible portfolio estimator library whose CombinatorialPurgedCV breaks sklearn's own split() contract - it yields (train, [test_0, ...]), and normal two-variable unp | 0 | 0 |
| `fin-libraries` | [`lib-talib`](plugins/fin-libraries/skills/lib-talib/SKILL.md) | The C reference implementation of technical indicators, where every pure-Python port disagrees during warm-up and none of them say so. | 0 | 0 |
| `fin-libraries` | [`lib-tushare`](plugins/fin-libraries/skills/lib-tushare/SKILL.md) | tushare is the cheapest source of genuinely point-in-time A-share fundamentals, and it sends your token over plaintext HTTP. | 0 | 0 |
| `fin-libraries` | [`lib-vectorbt`](plugins/fin-libraries/skills/lib-vectorbt/SKILL.md) | Vectorized Numba/Rust backtester built for parameter sweeps, whose from_signals fills at the signal's own bar close by default. | 0 | 0 |
| `fin-libraries` | [`lib-vollib`](plugins/fin-libraries/skills/lib-vollib/SKILL.md) | Machine-precision implied volatility with no bracketing, behind a package name restructured in 2026 - py_vollib is now a DEAD SHIM with four files and zero library code, and every  | 0 | 0 |
| `fin-libraries` | [`lib-yfinance`](plugins/fin-libraries/skills/lib-yfinance/SKILL.md) | The default free Yahoo Finance downloader, whose yf.download() now returns pre-adjusted OHLC with no Adj Close column at all. | 0 | 0 |
| `fin-llm` | [`finance-mcp-servers`](plugins/fin-llm/skills/finance-mcp-servers/SKILL.md) | Pick a finance MCP server, and know its licence and blast radius before connecting it. | 0 | 0 |
| `fin-llm` | [`llm-finance-agents`](plugins/fin-llm/skills/llm-finance-agents/SKILL.md) | What the published evidence says about LLM trading agents, and the real status of the frameworks. | 2 | 1 |
| `fin-llm` | [`rl-and-ml-trading`](plugins/fin-llm/skills/rl-and-ml-trading/SKILL.md) | Reinforcement learning and deep learning for trading: what installs, and what the evidence says. | 0 | 0 |

<!-- END GENERATED SKILL TABLE -->

Start at **`quant-stack-router`** — it holds the version-drift table and routes to everything else.
If you only read one other skill, read **`research-integrity-guards`**.

## A sample of what it corrects

Facts verified 2026-09-03/04 that contradict what most models and tutorials still say:

| Common belief | Verified reality |
|---|---|
| TA-Lib needs the C library compiled by hand | **Solved.** 0.7.1 ships 54 wheels including `cp311-win_amd64` |
| QuantLib is a nightmare to install | **Solved.** 1.43 ships `cp39-abi3-win_amd64` — but **no sdist at all** |
| Use `ib_insync` for IBKR | **Dead** since 2023-07; successor is `ib_async` |
| `pdr.get_data_yahoo(...)` | **Removed in pandas-datareader 0.11.0** — it is now macro-only |
| `yf.download()` returns raw OHLC + `Adj Close` | **`auto_adjust=True` since 1.0** — there is no `Adj Close` column |
| vectorbt's `from_signals` is safe out of the box | **Fills at the signal's own bar close** (`price=np.inf`) |
| `empyrical.sharpe_ratio(risk_free=0.05)` means 5% annual | It means **5% per day**. The result is a Sharpe of −65 |
| `quantstats.cagr(rf=...)` uses the risk-free rate | **It silently discards it** — `"cagr"` is on an exclusion list |
| `arch`'s SPA/StepM/MCS take returns | **They take losses.** Pass returns and the test inverts |
| `mlfinlab` implements AFML | **Off PyPI; the GitHub source is stubbed — every function body is `pass`** |
| `rateslib` is open source | **It never was.** Source-available non-commercial + paid commercial licence |
| `py_vollib` is the options library | **A dead shim since 1.0.12** — the real package is `vollib` |
| Moirai/TimesFM weights are Apache | **Moirai is `cc-by-nc-4.0`** on all variants |
| PDT limits your day trading | **PDT was eliminated 2026-06-04** (SEC Release 34-105226) |

Greek scaling, measured against QuantLib on identical inputs: `vollib`'s **vega is 100× smaller**
(per vol point), **theta 365× smaller** (per calendar day), **rho 100× smaller** (per 1% rate).

## Layout

```
plugins/<plugin>/skills/<skill>/
    SKILL.md          the router: task -> file, plus what will silently lie to you
    references/*.md   one file per library — versions, licence, traps, snippets
    scripts/*.py      runnable, tested tools
catalog/index.json    generated from frontmatter; never hand-edited
scripts/validate.py   enforces the 6-field spec + discovery budget + reference integrity
scripts/build_index.py
```

Skills live in category folders on disk, which Claude Code does **not** discover by default
(`.claude/skills/<category>/<skill>/` is not scanned — issue #39138, closed as not planned). The
plugin manifest's explicit `skills` array is the supported escape hatch, which is why this repo ships
as plugins rather than loose skills.

## The runnable parts

| Script | What it does |
|---|---|
| `signal-construction/scripts/assert_causal.py` | Perturbs only future bars and asserts the past did not move. Catches centered windows, negative shifts, full-sample normalization |
| `backtest-validation/scripts/trial_ledger.py` | Append-only trial ledger + Deflated Sharpe using its honest trial count |
| `research-integrity-guards/scripts/result_manifest.py` | A result card that **refuses to render** without universe provenance, a cost curve, a trial count and a falsifier |
| `llm-finance-agents/scripts/contamination_probe.py` | Training-cutoff overlap check + the accuracy-collapse-at-cutoff probe |

```bash
python plugins/fin-core/skills/backtest-validation/scripts/trial_ledger.py
# 50 noise strategies, best Sharpe 0.88, expected max from noise 0.94
# -> "NOT distinguishable from noise"
```

## Trigger accuracy — measured, not asserted

A skill that never fires is worth nothing. Two harnesses measure whether these descriptions
actually get selected:

| Harness | What it measures | Result |
|---|---|---|
| `scripts/eval_triggers.py` | idf-weighted term overlap; catches descriptions competing for the same words | **58/62 = 94%** (was 63% before the rewrite) |
| `scripts/eval_blind.py` | **a model choosing from the descriptions alone**, seeing exactly the discovery-time view | **61/62 = 98%** |

Both run against `evals/queries.jsonl` — 62 realistic queries including Chinese, pasted error
strings (`finrl import fails with ModuleNotFoundError`) and symptom phrasings (`my strategy works
in backtest but loses money live`). The single blind miss routes "how do I avoid survivorship bias"
to `market-data-sourcing` rather than `research-integrity-guards`, which is defensible — that skill
carries the per-vendor delisted-coverage table.

Descriptions follow the pattern Anthropic's own highest-precision skill uses: a `TRIGGER` keyword
list plus a `SKIP` negative override naming the competing skill.

## Contributing / maintaining

```bash
python scripts/validate.py      # spec compliance, budget, reference integrity, frontmatter
python scripts/eval_triggers.py  # do the descriptions actually select correctly?
python scripts/build_index.py   # regenerate catalog/index.json and the table above
```

`eval_triggers.py` scores all 62 queries in `evals/queries.jsonl` against every description and
reports top-1 accuracy plus the top-2 margin. It is a **lexical proxy, not a live model test** —
but the failure it catches is real: a query whose distinctive words match three descriptions
equally is being resolved close to arbitrarily. Current: **58/62 (94%)**, up from 39/62 (63%)
before the descriptions were rewritten with `TRIGGER`/`SKIP` clauses. Treat a thin margin as a
defect even when the top pick is right.

`validate.py` restricts frontmatter to the six spec fields (`name`, `description`, `license`,
`compatibility`, `metadata`, `allowed-tools`). Claude Code accepts more, but any extra key is a hard
error on claude.ai upload and the Skills API, so the portable subset is enforced here.

When a fact goes stale, update the claim **and** its `verified_on` date. A dated wrong answer is
recoverable; an undated one is not.

## Scope

Deliberately **not** covered, because other repos own them: crypto/DeFi execution plumbing and MEV
(`agiprolabs/claude-trading-skills`), RIA compliance and practice ops (`JoelLewis/finance_skills`),
personal bookkeeping and tax (`openaccountant/skills`). Leakage-safe quant ML overlaps with
`ml4t/skills` (Apache-2.0) — that repo is excellent and worth reading alongside this one.

Nothing here is investment advice, and no skill in this repo places an order.

## Licence

MIT for the repo's own content. **Library licences are a separate matter and are recorded per
library** — this domain contains AGPL (`openbb`, `backtesting.py`, `dbnomics`), GPL (`backtrader`,
`freqtrade`, `financepy`, `cvxportfolio`), Commons Clause (`vectorbt`, `lib-pybroker`),
source-available non-commercial (`rateslib`, `RQAlpha`), and packages with **no licence at all**
(`pytdx`, `Ashare`, `ProsusAI/finbert`). Code licence never implies data licence.
