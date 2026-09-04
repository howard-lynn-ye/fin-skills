---
name: lib-freqtrade
description: >-
  freqtrade is a live-first crypto bot with the best bias detectors in the field and a backtester
  that assumes zero slippage always. TRIGGER - freqtrade, "freqtrade backtesting", freqtrade
  trade/hyperopt/download-data, lookahead-analysis, recursive-analysis, startup_candle_count,
  IStrategy, populate_indicators, populate_entry_trend, populate_exit_trend, custom_stoploss,
  stoploss_on_exchange, minimal_roi, trailing_stop, VolumePairList, StaticPairList, dry_run,
  dry_run_wallet, config.json, user_data/strategies, FreqAI, freqtrade GPL. Monthly YYYY.M
  releases have renamed the strategy callbacks repeatedly, so remembered method names are usually
  the old ones. SKIP for backtesting-engines, the skill for equity and futures bar engines. SKIP
  when the question is WHICH library to choose rather than how to use this one - that belongs to
  the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# freqtrade

A live-first retail crypto bot that ships **the best bias-detection tooling in the entire domain** —
and an optimistic backtester whose assumptions it is honest enough to publish in full.

| | |
|---|---|
| pip / import | `freqtrade` — a CLI first (`freqtrade trade`, `backtesting`, `hyperopt`); strategies subclass `IStrategy` |
| Version | 2026.8 (2026-08-31) · 98 releases · monthly `YYYY.M` scheme · repo pushed 2026-09-04 |
| Licence | **GPL-3.0** (PyPI `GPLv3` + GitHub `spdx_id` agree) |
| Python | `requires_python >=3.11` · pure-python wheel + sdist — installs cleanly on Windows |
| Status | Very active. 54,007★ with only 23 open issues. Exchange layer is `ccxt`, so every ccxt venue quirk applies |

**GPL-3.0 is the strongest copyleft in this catalog.** A strategy file freqtrade loads at runtime is
one thing; **shipping a product built on freqtrade is another** — distribution obliges you to release
source under the same terms. Internal and personal use is unaffected. If the deliverable is
proprietary, use `jesse` (MIT) or build on `ccxt` directly.

## The trap that costs you money

**Zero slippage. Always.** Every order fills at the requested price provided that price sits inside
the candle's high/low. Real fees are charged (pulled from ccxt); **market impact is exactly zero**. On
thin alts this is the single largest source of phantom P&L.

Its twin: **stoploss exits fill exactly at the stoploss price, even when the candle's low was lower.**
A gap straight through your stop costs you nothing in backtest and a great deal live.

## The rest of the documented assumptions

freqtrade publishes its own backtest assumptions, which is why this list exists at all:

- **Within a candle, the high is assumed to occur before the low** — which is what makes trailing
  stops look good. You cannot know intrabar ordering from OHLC; every bar engine guesses, and
  freqtrade guesses in the direction that flatters trailing stops.
- **Exit signals take priority over stoploss**, because exits are assumed to trigger at the next
  candle's open. A candle that would have stopped you out mid-bar instead exits on your signal.
- **Survivorship via live pair lists.** `VolumePairList` and friends build the universe from **today's**
  listings, so backtesting 2021 on today's pairs misses every token that went to zero. Use a
  `StaticPairList` rebuilt from archived `fetch_markets()` snapshots, or call the result an upper bound.
- **The hyperopt maximum is not a result.** A 2,000-evaluation hyperopt is 2,000 trials for
  Deflated-Sharpe purposes.
- **`stoploss_on_exchange` is off by default.** On, the stop lives **at the exchange** and survives
  your process dying, your VPS rebooting or your network dropping. It is the most important single
  line in retail crypto risk management, and you must opt in.

Other documented behaviour: entries fill at the **open** unless custom pricing applies; ROI exits
compare against the candle high but never price below the low; stoploss is evaluated before ROI;
order is exit signal → stoploss → ROI → trailing stoploss. The project's own closing statement is
that backtesting will never replace dry-run mode, because intrabar ordering is unknowable.

**Every bar engine makes assumptions of this shape. freqtrade is simply the only one that lists
them.** Treat that list as the template for interrogating any other engine.

## The two bias detectors — worth running from another framework

**`freqtrade lookahead-analysis`** runs a baseline backtest, then re-runs it over progressively
sliced windows and compares dataframe columns. Divergence means the indicator saw the future. It
catches `shift(-N)`, raw `.iloc[]` indexing, unrolled full-series aggregations (`.mean()`, `.max()`)
and badly controlled loops. Stated limits: it only checks **triggered** signals (untriggered leaks
escape as false negatives), false-positives on limit orders with custom pricing callbacks, may
falsely flag FreqAI target indicators, and is useless for rarely-signalling strategies.

**`freqtrade recursive-analysis`** varies `startup_candle_count` and reports each indicator's
last-row variance against the base calculation. `-` = converged, `nan%` = insufficient data, a large
percentage = raise your startup candles. **It is the only off-the-shelf tool that measures the
recursive warm-up problem**: EMA/RSI/ADX converge to different values depending on how much history
preceded them, and a backtest sees 5,000 candles where live sees the ~1,000 the exchange returns in
one call. Its stated mechanism — backtesting loads the whole dataframe into memory while live
processes candles sequentially — describes vectorbt, PyBroker and backtesting.py equally well.

## Minimal correct call

```jsonc
{
  "dry_run": true,                 // default in the shipped example config — keep it until deployed
  "dry_run_wallet": 1000,
  "stake_currency": "USDT",
  "trading_mode": "spot",
  "order_types": {
    "stoploss_on_exchange": true,  // OFF by default; turn it on
    "stoploss_on_exchange_interval": 60
  },
  "pairlists": [                   // NOT VolumePairList — that is survivorship bias
    { "method": "StaticPairList" }
  ],
  "exchange": { "ccxt_config": { "enableRateLimit": true } }
}
```

```bash
freqtrade backtesting        --strategy MyStrat --timerange 20210101-20240101
freqtrade lookahead-analysis --strategy MyStrat   # run BEFORE believing any result
freqtrade recursive-analysis --strategy MyStrat
```

Live keys must carry **trade scope only, never withdraw**; `dry_run: true` is not proof you are safe.

## See also

- `../../../fin-crypto/skills/crypto-data-and-execution/SKILL.md` — survivorship, funding, 365 days
- `../../../fin-crypto/skills/crypto-data-and-execution/references/freqtrade.md` — reference card
- `../../../fin-crypto/skills/crypto-data-and-execution/references/ccxt.md` — the exchange layer
- `../../../fin-core/skills/backtesting-engines/references/execution-realism.md` — fill realism

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`backtesting-engines`** (`../../../fin-core/skills/backtesting-engines/SKILL.md`).

