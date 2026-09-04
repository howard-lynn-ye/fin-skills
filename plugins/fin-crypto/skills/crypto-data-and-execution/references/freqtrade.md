# freqtrade

A live-first retail crypto bot that ships **the best bias-detection tooling in the entire domain** —
and an optimistic backtester whose assumptions it is honest enough to publish in full.

| | |
|---|---|
| pip | `freqtrade` · **2026.8 (2026-08-31)** · 98 releases · monthly `YYYY.M` scheme |
| GitHub | `freqtrade/freqtrade` — **54,007★**, only 30 open issues, pushed 2026-09-04 ✅ |
| Licence | 🚨 **GPL-3.0** ✅ (PyPI `GPLv3` + GitHub `spdx_id: GPL-3.0` agree) |
| Python | `requires_python >=3.11` ✅ · pure-python wheel + sdist — installs cleanly on Windows |
| Exchange layer | `ccxt` — every venue quirk in `ccxt.md` applies here |
| Maintenance | ✅ Very active, tiny issue backlog for a 54k-star project |

🚨 **GPL-3.0 is the strongest copyleft in this catalog.** Your strategy file loaded by freqtrade at
runtime is one thing; **shipping a product built on freqtrade is another** — distribution obliges you
to release source under the same terms. Internal and personal use is unaffected. If the deliverable
is proprietary, use `jesse` (MIT) or build on `ccxt` directly.

## 🚨 Traps — what will silently overstate your backtest

freqtrade documents its own backtest assumptions explicitly ✅ [V-DOC]. The dangerous ones:

**1. 🚨 Zero slippage. Always.** Every order fills at the requested price provided that price sits
inside the candle's high/low. Real fees are charged (pulled from ccxt), **market impact is exactly
zero**. On thin alts this is the single largest source of phantom P&L.

**2. 🚨 Stoploss exits fill exactly at the stoploss price — even when the candle's low was lower.**
A gap straight through your stop costs you nothing in backtest and a great deal live.

**3. 🚨 Within a candle, the high is assumed to occur before the low** (this is what makes trailing
stops look good). You cannot know intrabar ordering from OHLC; every bar engine guesses, freqtrade
guesses in the direction that flatters trailing stops.

**4. 🚨 Exit signals take priority over stoploss**, because exits are assumed to trigger at the next
candle's open. A candle that would have stopped you out mid-bar instead exits on your signal.

**5. 🚨 Survivorship via live pair lists.** `VolumePairList` and friends build the universe from
**today's** exchange listings. Backtesting 2021 on today's pairs is not a backtest of 2021 — every
token that went to zero is missing. Use a `StaticPairList` reconstructed from archived
`fetch_markets()` snapshots, or state the result is an upper bound.

**6. 🚨 The hyperopt maximum is not a result.** A 2,000-evaluation hyperopt is 2,000 trials for
Deflated-Sharpe purposes. See `../../../../fin-core/skills/backtest-validation/SKILL.md`.

**7. 🚨 `stoploss_on_exchange` is off by default.** On, the stop lives **at the exchange** and survives
your process dying, your VPS rebooting, or your network dropping. **The most important single line in
retail crypto risk management** — and you must opt in.

Other documented behaviours worth knowing: entries fill at the **open** unless custom pricing applies;
ROI exits compare against the candle high but never price below the low; stoploss is evaluated before
ROI; evaluation order is exit signal → stoploss → ROI → trailing stoploss. The project's own closing
statement is that backtesting **"will never replace running a strategy in dry-run mode"** — because
intrabar ordering is unknowable.

**Every bar engine in this catalog makes assumptions of this shape. freqtrade is simply the only one
that lists them.** Treat that list as the template for interrogating any other engine.

## 🔑 The two bias detectors — worth running even from another framework

**`freqtrade lookahead-analysis`** ✅ — runs a baseline backtest, then re-runs it over progressively
sliced windows and compares dataframe columns. Divergence ⇒ the indicator saw the future. Catches
`shift(-N)`, raw `.iloc[]` indexing, unrolled full-series aggregations (`.mean()`, `.max()`), and
badly controlled loops.
⚠️ Stated limits: only checks **triggered** signals (untriggered leaks escape as false negatives);
false-positives on limit orders with custom pricing callbacks; may falsely flag FreqAI target
indicators; useless for rarely-signalling strategies.

**`freqtrade recursive-analysis`** ✅ — varies `startup_candle_count` and reports each indicator's
last-row variance against the base calculation. `-` = converged, `nan%` = insufficient data, a large
percentage = raise your startup candles. **The only off-the-shelf tool that measures the recursive
warm-up problem**: EMA/RSI/ADX converge to different values depending on how much history preceded
them, and a backtest sees 5,000 candles where live sees the ~1,000 the exchange returns in one call.

Its stated mechanism — backtesting loads the whole dataframe into memory while live processes candles
sequentially — **describes vectorbt, PyBroker and backtesting.py's `Strategy.I` equally well**. Port
the idea: `../../../../fin-core/skills/signal-construction/scripts/assert_causal.py`.

## Minimal safe configuration

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
freqtrade backtesting  --strategy MyStrat --timerange 20210101-20240101
freqtrade lookahead-analysis  --strategy MyStrat   # run BEFORE believing any result
freqtrade recursive-analysis  --strategy MyStrat
```

🚨 Live keys must carry **trade scope only, never withdraw** — see `../SKILL.md` §6. `dry_run: true`
in config is not proof you are safe: confirm no trade-scoped keys are loaded at all.

## Where it fits

- Crypto tool selection, survivorship, funding, 365-day annualization: `../SKILL.md`
- Exchange layer and its caps/pagination/precision traps: `ccxt.md`
- Bar-engine fill realism across every engine:
  `../../../../fin-core/skills/backtesting-engines/references/execution-realism.md`
