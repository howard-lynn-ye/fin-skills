# fin-skills — Agent Skills for Python quantitative finance

**70 [Agent Skills](https://agentskills.io/specification) for Claude Code that tell an LLM which
Python quant-finance library to use, what each one silently gets wrong, and whether a backtest
result is real.** 46 domain skills, plus 24 optional per-library deep dives you install only if
you want them. Covers market data, SEC point-in-time fundamentals, backtesting engines, broker
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
> dropped skill never auto-triggers. `fin-core` alone costs ~3,957 tokens. Set
> `"skillListingBudgetFraction": 0.03` in `~/.claude/settings.json` before installing more than one
> plugin. Check with `/context` (Skills row) or `/doctor`.

### As a Python package

The same skill texts, plus every guard script as an importable function. No PyPI needed:

```bash
pip install git+https://github.com/howard-lynn-ye/fin-skills
```

```python
import fin_skills
fin_skills.catalog()                           # every skill: name, plugin, description
fin_skills.load("backtest-validation")         # the SKILL.md text
fin_skills.references("options-backtesting")   # its references/, {filename: text}
fin_skills.find("survivorship", "universe")    # skills whose text mentions both

from fin_skills.core.safe_asof import safe_asof                  # the executable guards
from fin_skills.core.assert_causal import assert_causal
from fin_skills.futures_fx.fx_conventions import pip_size, carry_return
from fin_skills.core.option_lifecycle import crr                # a CRR tree, no QuantLib
```

The same checks behind one interface — 28 guards that return a `GuardResult` instead of raising, and
a typed `conventions` module (annualisation, risk-free, pip and liquidation arithmetic), the way
PyOD puts its detectors behind one API. PyOD's uniformity comes from a
uniform data container (every detector is `fit(X)`); here the container is a `Bundle` — the artefacts
of one research run under a fixed vocabulary — and `check()` runs every guard whose inputs are present:

```python
from fin_skills.api import Bundle, check

b = Bundle(returns=strategy_returns, turnover=turn, rf=0.05,          # one slot feeds every guard
           bars=bars, signal_fn=lambda d: d.close.rolling(20).mean(),  # that means the same thing by it
           close=aapl_close, actions=aapl_actions)
print(b.coverage().summary())  # ready: cost_curve, rf_convention, assert_causal, adjustment_check, ...
                               # one slot away: + prices unlocks survivorship_audit
report = check(b)              # every ready guard, one call; skipped ones say what they still need
print(report.summary())

from fin_skills.api import get, Suite, conventions as c              # the per-guard forms

r = get("assert_causal").run(fn=lambda d: d.close.shift(-1), df=bars, k=250)
r.passed, r.summary()          # False, "FAIL: LOOK-AHEAD ... cells before index 250 changed"
Suite("assert_causal", "warmup_probe", "cost_curve").check(b)        # a reusable subset

c.annualization_factor("crypto")            # 365
c.liquidation_price(entry=100, leverage=10, mmr=0.004, side="long")
c.pip_value("USDJPY", notional=100_000, price=150.25).value_usd
```

`fin_skills.api.slots()` lists the vocabulary — which guards each slot reaches — and a `Bundle` rejects
an unknown slot name, a DataFrame where a Series belongs, or an unsorted DatetimeIndex at construction,
with a message naming the slot.

`python -m pytest -q` runs the suite (slow tests are marked and deselected by default).

API reference, one page per module: **https://howard-lynn-ye.github.io/fin-skills/** (built by
`scripts/build_docs.py` with pdoc and published to the `gh-pages` branch).

`fin_skills/` is generated from the skills by `scripts/build_package.py`; the skills stay the
source of truth and `validate.py` fails if the two drift apart. Namespaces follow the plugins that
ship scripts: `core`, `libraries`, `china`, `futures_fx`, `crypto`, `llm` (a plugin without
scripts, such as `fin-asia`, has no namespace; its skill text is still in `fin_skills.load()`). Every
skill script runs standalone too (`python plugins/<plugin>/skills/<skill>/scripts/<name>.py`).

### Federated third-party packs

The marketplace also lists 92 third-party skill packs by their own GitHub source, so they
install through this marketplace without being copied here. They come from vendor-official
repositories (Alpaca, Kraken, Longbridge, OKX, HTX, Pionex, BloFin, CoinStats, Upstox, J-Quants,
Nansen), from Anthropic's own financial-services marketplace and its A-share port, and from the
community: equity research, real estate, tax and accounting, macro, market data for China, Japan
and India, and trading systems. The five largest are `algo-trading-skills` (501 skills), `ftshare-skills` (215), `yuping322-finskills` (107), `vibe-trading-skills` (90) and `doramagic-skills` (82).

Every one installs **disabled**; its description says what it covers, how it handles credentials,
and whether it can place live orders once keys are set. What this repo verified, on the date in
each entry's `metadata.verified_on`, is the licence, that the repository is live and not archived,
that real `SKILL.md` files exist at the pinned commit, and the skill count. **The claims inside
them are not verified by this repo, and none was installed.** Wave-2 entries pin a full commit
sha, so what installs is the tree that was read; wave-1 entries pin only a branch. Federation is
per plugin - a pack's skills cannot be cherry-picked - and several packs are far larger than the
default listing budget, so install deliberately. What was verified and what was left out is in
[`catalog/federation-notes.md`](catalog/federation-notes.md) and
[`catalog/federation-notes-wave2.md`](catalog/federation-notes-wave2.md).

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
| `fin-core` | [`etf-mechanics`](plugins/fin-core/skills/etf-mechanics/SKILL.md) | Why an ETF's price series does not behave like the index it tracks - daily-reset leverage, NAV vs price, distributions, holdings files and fees. | 0 | 1 |
| `fin-core` | [`execution-cost-analysis`](plugins/fin-core/skills/execution-cost-analysis/SKILL.md) | Measure what your execution actually cost instead of assuming a number - implementation shortfall, benchmark choice, impact models, and the gap between the cost you assumed and the | 0 | 1 |
| `fin-core` | [`external-skill-index`](plugins/fin-core/skills/external-skill-index/SKILL.md) | A verified index of every public finance Agent Skill repository — 139 repos, 4,851 SKILL.md files — so you can find what already exists instead of rebuilding it, and avoid the thir | 0 | 0 |
| `fin-core` | [`factor-and-timeseries-research`](plugins/fin-core/skills/factor-and-timeseries-research/SKILL.md) | Judge whether a cross-sectional factor predicts returns, and forecast financial series. | 7 | 1 |
| `fin-core` | [`fundamental-and-macro-data`](plugins/fin-core/skills/fundamental-and-macro-data/SKILL.md) | Company fundamentals and macro series with correct point-in-time semantics. | 3 | 1 |
| `fin-core` | [`intraday-microstructure`](plugins/fin-core/skills/intraday-microstructure/SKILL.md) | Measure the market at the tick level and know when the measure is lying. | 0 | 1 |
| `fin-core` | [`market-data-engineering`](plugins/fin-core/skills/market-data-engineering/SKILL.md) | Store, join and parallelize market data you already hold, without corrupting it. | 4 | 2 |
| `fin-core` | [`market-data-sourcing`](plugins/fin-core/skills/market-data-sourcing/SKILL.md) | Choose a market price or reference data vendor and use it without silently corrupting the numbers. | 11 | 1 |
| `fin-core` | [`options-backtesting`](plugins/fin-core/skills/options-backtesting/SKILL.md) | Options positions end in ways you do not control - live or in a backtest: assignment, expiry settlement, pin risk, multi-leg lifecycle, historical chain assembly, and the margin th | 3 | 1 |
| `fin-core` | [`portfolio-and-risk`](plugins/fin-core/skills/portfolio-and-risk/SKILL.md) | Turn signals into weights, and compute performance metrics that are actually correct. | 9 | 1 |
| `fin-core` | [`quant-stack-router`](plugins/fin-core/skills/quant-stack-router/SKILL.md) | Entry router for Python quantitative finance: names the right library and flags where the model's training prior is stale. | 0 | 0 |
| `fin-core` | [`regime-detection`](plugins/fin-core/skills/regime-detection/SKILL.md) | Detect and label market regimes without letting the labels see the future, and state regime coverage in the form the result gate demands. | 0 | 3 |
| `fin-core` | [`research-integrity-guards`](plugins/fin-core/skills/research-integrity-guards/SKILL.md) | Second-pass audit that decides whether a finance result is real, applied after the work exists. | 2 | 3 |
| `fin-core` | [`signal-construction`](plugins/fin-core/skills/signal-construction/SKILL.md) | Compute technical indicators and engineered features without leaking the future. | 2 | 2 |
| `fin-core` | [`us-market-rules`](plugins/fin-core/skills/us-market-rules/SKILL.md) | US trading rules that decide whether a strategy is executable at all - short-sale restrictions, margin, settlement, day-trading limits, wash sales, and what a data licence lets you | 0 | 0 |
| `fin-crypto` | [`crypto-data-and-execution`](plugins/fin-crypto/skills/crypto-data-and-execution/SKILL.md) | Crypto market data and execution, and how a 24/7 market breaks equity tooling. | 3 | 1 |
| `fin-futures-fx` | [`futures-continuous-contracts`](plugins/fin-futures-fx/skills/futures-continuous-contracts/SKILL.md) | Build and use a futures price series correctly — a continuous contract does not exist in the market, it is stitched, and the stitching method changes your answer. | 0 | 2 |
| `fin-futures-fx` | [`fx-markets`](plugins/fin-futures-fx/skills/fx-markets/SKILL.md) | Trade and backtest FX correctly — quote conventions, pip sizing, and the carry that a spot-only backtest silently omits. | 0 | 1 |
| `fin-libraries` | [`lib-akshare`](plugins/fin-libraries/skills/lib-akshare/SKILL.md) | akshare is the widest free Chinese-market scraper (1,103 public interfaces) and it purges its own PyPI history, so you cannot pin it. | 0 | 0 |
| `fin-libraries` | [`lib-alpaca-py`](plugins/fin-libraries/skills/lib-alpaca-py/SKILL.md) | Alpaca's current Python SDK, which defaults to the paper host but lets url_override silently send live orders from a client that believes it is in the sandbox. | 0 | 0 |
| `fin-libraries` | [`lib-alphalens`](plugins/fin-libraries/skills/lib-alphalens/SKILL.md) | alphalens-reloaded scores cross-sectional factors, and its forward return starts at date t's OWN price - it never lags your factor. | 0 | 0 |
| `fin-libraries` | [`lib-arch`](plugins/fin-libraries/skills/lib-arch/SKILL.md) | The reference GARCH implementation in Python, and the home of SPA/StepM/MCS - which all take LOSSES, so passing returns silently inverts the test and names your worst strategy as t | 0 | 1 |
| `fin-libraries` | [`lib-backtesting-py`](plugins/fin-libraries/skills/lib-backtesting-py/SKILL.md) | Single-asset bar-loop backtester with honest next-open fills, an AGPL-3.0 licence, and an indicator API that computes over the entire series before slicing. | 0 | 0 |
| `fin-libraries` | [`lib-ccxt`](plugins/fin-libraries/skills/lib-ccxt/SKILL.md) | The unified MIT client for 100+ crypto venues - and not a backtester, with an OHLCV endpoint that silently truncates and returns an unclosed final bar. | 0 | 0 |
| `fin-libraries` | [`lib-edgartools`](plugins/fin-libraries/skills/lib-edgartools/SKILL.md) | edgartools is the default free SEC EDGAR client - typed objects for 20+ form types, XBRL statements, no API key - and it 403s on every request until you call set_identity(). | 0 | 0 |
| `fin-libraries` | [`lib-fredapi`](plugins/fin-libraries/skills/lib-fredapi/SKILL.md) | fredapi wraps FRED/ALFRED and is the primary anti-look-ahead tool in macro - and three of its four vintage methods are buggy in source. | 0 | 0 |
| `fin-libraries` | [`lib-freqtrade`](plugins/fin-libraries/skills/lib-freqtrade/SKILL.md) | freqtrade is a live-first crypto bot with the best bias detectors in the field and a backtester that assumes zero slippage always. | 0 | 0 |
| `fin-libraries` | [`lib-ib-async`](plugins/fin-libraries/skills/lib-ib-async/SKILL.md) | The maintained Interactive Brokers Python client - successor to the archived ib_insync - where one digit of the port number is all that separates paper from live. | 0 | 0 |
| `fin-libraries` | [`lib-nautilus-trader`](plugins/fin-libraries/skills/lib-nautilus-trader/SKILL.md) | Event-driven Rust-core engine with the strongest execution modelling in open source, gated to Python 3.12-3.14, where a wrong ts_init silently makes every bar visible one interval  | 0 | 0 |
| `fin-libraries` | [`lib-polars`](plugins/fin-libraries/skills/lib-polars/SKILL.md) | The polars wheel is now an empty 865 KB py3-none-any shim hard-pinned to polars-runtime-32, so a lockfile listing only polars does not pin the engine. | 0 | 1 |
| `fin-libraries` | [`lib-purgedcv`](plugins/fin-libraries/skills/lib-purgedcv/SKILL.md) | The only genuinely sklearn-protocol-compliant purged and embargoed splitter, and the one that refuses to run until you state when each label resolved - understate evaluation_times  | 0 | 1 |
| `fin-libraries` | [`lib-pyportfolioopt`](plugins/fin-libraries/skills/lib-pyportfolioopt/SKILL.md) | Textbook mean-variance and Black-Litterman optimizer whose HRPOpt silently accepts a price matrix where it requires returns and returns plausible garbage. | 0 | 1 |
| `fin-libraries` | [`lib-qlib`](plugins/fin-libraries/skills/lib-qlib/SKILL.md) | Microsoft Qlib (pip name pyqlib, imported as qlib) ships Alpha158/Alpha360 and a default normalizer that leaks your test set into training, silently. | 0 | 0 |
| `fin-libraries` | [`lib-quantlib`](plugins/fin-libraries/skills/lib-quantlib/SKILL.md) | The only broadly-permissive, mature, full-coverage derivatives library in Python, whose global evaluationDate returns an NPV of exactly 0.0 with no warning once it is past expiry. | 0 | 1 |
| `fin-libraries` | [`lib-quantstats`](plugins/fin-libraries/skills/lib-quantstats/SKILL.md) | The tearsheet library whose cagr(rf=...) accepts your risk-free rate and silently discards it - "cagr" sits on an exclusion list inside _prepare_returns, which dispatches on the ca | 0 | 1 |
| `fin-libraries` | [`lib-riskfolio`](plugins/fin-libraries/skills/lib-riskfolio/SKILL.md) | The 26-risk-measure portfolio optimizer whose stateful API optimizes against stale or missing mu and Sigma - with no error - if you forget assets_stats(). | 0 | 0 |
| `fin-libraries` | [`lib-skfolio`](plugins/fin-libraries/skills/lib-skfolio/SKILL.md) | The sklearn-compatible portfolio estimator library whose CombinatorialPurgedCV breaks sklearn's own split() contract - it yields (train, [test_0, ...]), and normal two-variable unp | 0 | 0 |
| `fin-libraries` | [`lib-talib`](plugins/fin-libraries/skills/lib-talib/SKILL.md) | The C reference implementation of technical indicators, where every pure-Python port disagrees during warm-up and none of them say so. | 0 | 0 |
| `fin-libraries` | [`lib-tushare`](plugins/fin-libraries/skills/lib-tushare/SKILL.md) | tushare is the cheapest source of genuinely point-in-time A-share fundamentals, and it sends your token over plaintext HTTP. | 0 | 0 |
| `fin-libraries` | [`lib-vectorbt`](plugins/fin-libraries/skills/lib-vectorbt/SKILL.md) | Vectorized Numba/Rust backtester built for parameter sweeps, whose from_signals fills at the signal's own bar close by default. | 0 | 0 |
| `fin-libraries` | [`lib-vollib`](plugins/fin-libraries/skills/lib-vollib/SKILL.md) | Machine-precision implied volatility with no bracketing, behind a package name restructured in 2026 - py_vollib is now a DEAD SHIM with four files and zero library code, and every  | 0 | 1 |
| `fin-libraries` | [`lib-yfinance`](plugins/fin-libraries/skills/lib-yfinance/SKILL.md) | The default free Yahoo Finance downloader, whose yf.download() now returns pre-adjusted OHLC with no Adj Close column at all. | 0 | 0 |
| `fin-llm` | [`finance-agent-architectures`](plugins/fin-llm/skills/finance-agent-architectures/SKILL.md) | How the mainstream finance agent systems are built, and how to stage a research-to-execution pipeline whose gates are code. | 2 | 1 |
| `fin-llm` | [`finance-mcp-servers`](plugins/fin-llm/skills/finance-mcp-servers/SKILL.md) | Pick a finance MCP server, and know its licence and blast radius before connecting it. | 0 | 0 |
| `fin-llm` | [`llm-finance-agents`](plugins/fin-llm/skills/llm-finance-agents/SKILL.md) | What the published evidence says about LLM trading agents, and the real status of the frameworks. | 2 | 1 |
| `fin-llm` | [`rl-and-ml-trading`](plugins/fin-llm/skills/rl-and-ml-trading/SKILL.md) | Reinforcement learning and deep learning for trading: what installs, and what the evidence says. | 0 | 0 |
| `fin-models` | [`covariance-and-risk-models`](plugins/fin-models/skills/covariance-and-risk-models/SKILL.md) | Estimate a covariance matrix an optimizer can actually invert, and report how much variance it hides. | 0 | 1 |
| `fin-models` | [`credit-risk-models`](plugins/fin-models/skills/credit-risk-models/SKILL.md) | Estimate a default probability and price credit, and keep the two probabilities apart - the risk-neutral one that prices and the physical one that forecasts. | 0 | 1 |
| `fin-models` | [`factor-models`](plugins/fin-models/skills/factor-models/SKILL.md) | Build long-short factor portfolios from a characteristic panel and test the alpha with standard errors that survive serial correlation. | 1 | 1 |
| `fin-models` | [`implied-vol-surface`](plugins/fin-models/skills/implied-vol-surface/SKILL.md) | Build a volatility surface that is not silently arbitrageable - invert prices to implied vols, fit a smile, check butterfly and calendar arbitrage, and interpolate between maturiti | 0 | 1 |
| `fin-models` | [`option-pricing-models`](plugins/fin-models/skills/option-pricing-models/SKILL.md) | Implement an option pricing model correctly - closed form, tree, characteristic function, Monte Carlo - and the four places each silently returns a plausible wrong number. | 0 | 1 |
| `fin-models` | [`portfolio-optimizers`](plugins/fin-models/skills/portfolio-optimizers/SKILL.md) | Turn expected returns and a covariance matrix into weights, and measure what the optimizer did to your estimation error on the way. | 0 | 1 |
| `fin-models` | [`risk-measures-var-cvar`](plugins/fin-models/skills/risk-measures-var-cvar/SKILL.md) | Compute Value-at-Risk and Expected Shortfall by the four estimators that disagree in the tail, and backtest them properly. | 0 | 1 |
| `fin-models` | [`stat-arb-cointegration`](plugins/fin-models/skills/stat-arb-cointegration/SKILL.md) | Screen, test and trade a cointegrated pair without counting the trials wrong, applying the single-series ADF table to a fitted residual, or estimating the hedge ratio on the window | 0 | 1 |
| `fin-models` | [`state-space-and-kalman`](plugins/fin-models/skills/state-space-and-kalman/SKILL.md) | Estimate a time-varying hedge ratio or beta with a Kalman filter, and know which of its three state series you are allowed to trade - the smoothed one has read the whole sample. | 0 | 1 |
| `fin-models` | [`term-structure-models`](plugins/fin-models/skills/term-structure-models/SKILL.md) | Build and fit a yield curve, and price a zero-coupon bond in a short-rate model, without the convention and identification traps. | 0 | 1 |
| `fin-models` | [`time-series-forecasting-models`](plugins/fin-models/skills/time-series-forecasting-models/SKILL.md) | Score a forecast against the baseline it has to beat - naive, seasonal-naive, drift, mean - with MASE, rolling-origin evaluation and a Diebold-Mariano test, instead of an R^2 on a  | 0 | 1 |
| `fin-models` | [`volatility-models`](plugins/fin-models/skills/volatility-models/SKILL.md) | Fit and forecast volatility - GARCH, range-based realized variance, HAR-RV - without the two errors that silently move the answer: the units `arch` expects, and a range estimator u | 0 | 1 |
| `fin-strategies` | [`alpha-combination-and-neutralization`](plugins/fin-strategies/skills/alpha-combination-and-neutralization/SKILL.md) | Score several alphas, combine them, and strip the exposures you did not mean to take. | 0 | 1 |
| `fin-strategies` | [`execution-algorithms`](plugins/fin-strategies/skills/execution-algorithms/SKILL.md) | Build the schedule that works an order - VWAP, TWAP, POV, Almgren-Chriss - and know what each one is optimizing. | 0 | 1 |
| `fin-strategies` | [`market-making-models`](plugins/fin-strategies/skills/market-making-models/SKILL.md) | Quote a two-sided market and survive the inventory - Avellaneda-Stoikov reservation price and optimal spread, and the adverse selection the model does not price. | 0 | 1 |
| `fin-strategies` | [`position-sizing-kelly`](plugins/fin-strategies/skills/position-sizing-kelly/SKILL.md) | Decide how much to bet given an edge - Kelly, fractional Kelly, and volatility targeting - and the drawdown each implies. | 0 | 1 |
| `fin-strategies` | [`trend-following-models`](plugins/fin-strategies/skills/trend-following-models/SKILL.md) | Build a trend-following or time-series-momentum strategy the way the paper defines it, and measure the two look-aheads that flatter its backtest. | 1 | 1 |

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
| `scripts/eval_triggers.py` | idf-weighted term overlap; catches descriptions competing for the same words | **84/108 = 78%** strict top-1, **105/108 = 97%** routed (the pick links to the expected skill) |
| `scripts/eval_blind.py` | **a model choosing from the descriptions alone**, seeing exactly the discovery-time view | **107/108 = 99%** |

Both run against `evals/queries.jsonl` — 108 realistic queries including Chinese, pasted error
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

`eval_triggers.py` scores the 108 queries in `evals/queries.jsonl` against every description and
reports strict top-1 and routed accuracy plus the top-2 margin. It is a **lexical proxy, not a live
model test** — the numbers that matter are in the table above, and the blind eval is the one to
believe — but the failure it catches is real: a query whose distinctive words match three
descriptions equally is being resolved close to arbitrarily. Treat a thin margin as a defect even
when the top pick is right, and re-run both evals after any description change.

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
