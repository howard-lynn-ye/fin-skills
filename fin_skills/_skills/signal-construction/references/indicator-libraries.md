# Technical indicator libraries — measured against TA-Lib

The comparison below was **executed**: TA-Lib python 0.7.1 / C core 0.7.1, pandas 3.0.5, numpy 2.4.6,
500-bar synthetic GBM, seed 7. `maxabs` = max absolute difference over the overlapping non-NaN region.

## Status

| Package | Version | Released | Licence | Verdict |
|---|---|---|---|---|
| **TA-Lib** (`ta-lib-python`) | **0.7.1** | 2026-07-16 | BSD-ish | ✅ reference. **54 wheels incl. `cp311-win_amd64`** |
| **`pandas-ta-classic`** | 0.6.52 | 2026-06-24 | **MIT** | ✅ the maintained successor, 427★, py≥3.10 |
| **`talipp`** | 2.7.0 | 2025-09-09 | MIT | ✅ incremental/streaming |
| `ta` (bukosabino) | 0.11.0 | 2023-11-02 | MIT | ⚠️ pure python, **sdist only**, ~3 yrs stale |
| 🚨 `pandas-ta` | 0.4.71b0 | 2025-09-14 | 🚨 **NONE DECLARED** | 🔴 **do not install — see below** |
| 🔴 `finta` | 1.3 | 2021-04-03 | LGPL-3.0 | **repo ARCHIVED** 2022-07-24 |
| 🔴 `tulipy` | 0.4.0 | 2019-04-11 | LGPL | dead |
| 🔴 `bta-lib` | — | — | — | dead |

## 🚨 pandas-ta — supply-chain caution

- `github.com/twopirllc/pandas-ta` → **HTTP 404** (removed ~June 2025). Corroborated by
  `MerlinR/Pandas-ta-fork`, whose description reads *"This is stock Pandas-ta of the version I had
  prior to GH removal … but was gone before had chance to officially fork."*
- PyPI now lists **only two releases**: 0.4.67b0 (2025-09-03) and 0.4.71b0 (2025-09-14).
  **All prior history, including the widely used 0.3.14b, was deleted.**
- Metadata: **no `license` field and no license classifier**; author `Pandas TA Support
  <support@pandas-ta.dev>`; `requires_python >=3.12`; the `Repository` URL still points at the dead
  404 repo.
- `pandas-ta.dev` — **DNS does not resolve**, and archive.org has **no snapshots**.
- Community issue `xgboosted/pandas-ta-classic#30` documents the maintainer handover, the PyPI wipe,
  and explicit supply-chain concern.

**A package with no licence, wiped release history, a dead upstream repo and a dead homepage is a
supply-chain risk regardless of intent.** Pin to `pandas-ta-classic`, `ta`, or TA-Lib.

## Measured agreement with TA-Lib

| Library | SMA / BBands / CCI / OBV | RSI(14) | EMA(20) | ATR(14) | ADX(14) | MACD |
|---|---|---|---|---|---|---|
| **talipp 2.7.0** | exact | **1.4e-14, exact incl. warm-up start index** | 2.8e-14 | 3.2e-02 warm-up only | — | 0.155 warm-up only |
| **pandas-ta-classic 0.6.52** | ≤3e-11 | **3.6e-14** | 2.8e-14 | 8.9e-16 | 🚨 **4.25**, emits from bar 13 vs TA-Lib's 27 | 0.155 early bars |
| **ta 0.11.0** | identical | 🚨 **3.84** | 🚨 0.363 | 🚨 0.0477, emits **from bar 0** | 🚨 **0.642**, emits **from bar 0** | 🚨 0.378, 8 bars early |

**Ranking for "numerically identical to TA-Lib":**
1. **TA-Lib** (reference)
2. **`talipp`** — closest, and **the only one that also matches TA-Lib's warm-up start index on RSI**
3. **`pandas-ta-classic`** (native, `talib=False`) — matches the core set; **ADX warm-up is the exception**
4. **`ta`** — matches on non-recursive indicators only

### How long `ta`'s RSI(14) takes to converge

`|diff| > 1.0` through **bar 36** · `> 0.1` through **bar 67** · `> 0.01` through **bar 97** ·
`> 1e-6` through **bar 217**.

🚨 **`ta`'s Wilder-smoothed indicators need ~15× the period of burn-in before they agree with
TA-Lib.** On short datasets, or when you restart the calculation per symbol on a short slice, `ta` and
TA-Lib produce **materially different signals** — the same strategy, two different backtests.

> **Universal rule: discard the first ~10–15 × period bars of any recursive indicator (EMA, RSI, ATR,
> ADX, MACD), whatever the library.** `ta`, `pandas-ta-classic` and `talipp` all emit values *before*
> TA-Lib does, computed on partial windows — **silently wrong, not NaN.**

## TA-Lib

✅ **The install folklore is obsolete.** `ta-lib-python` has shipped prebuilt wheels **bundling the
TA-Lib C library** since **v0.6.5 (2025-08-07)**. `pip install TA-Lib` just works on Windows now.
Repo moved from `mrjbq7/ta-lib` to the **`ta-lib` org**.

```python
import talib
rsi = talib.RSI(close, timeperiod=14)
# abstract API works with dict / pandas / polars inputs
from talib import abstract
out = abstract.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
```

**Gotchas:** it takes **numpy float64 arrays**, not Series — pass `.values` and reattach the index
yourself, or use the abstract API. Its "unstable period" behaviour is documented and is the reason
its warm-up start indices differ from every pure-Python port.

## talipp — the streaming one

✅ **Verified: incremental `.add()` produces bit-identical output to batch construction.** That is the
entire reason the library exists, and it is the correct choice when a strategy must run streaming —
**it removes the backtest-vs-live warm-up divergence** described in
`../../backtesting-engines/SKILL.md` §2.3.

## `ta` — usable with one flag, and three verified bugs

Fine for a quick pure-Python feature set **with `fillna=False`**. Bugs isolated by perturbing only
future bars and checking whether early bars changed — see `_repainting.md` for the full detail:

1. `IchimokuIndicator(..., visual=True).ichimoku_a()` — **CONFIRMED LOOK-AHEAD**, 26 early bars
   changed, max |Δ| 35.34, **even with `fillna=False`**.
2. `DPOIndicator(..., fillna=True).dpo()` — **CONFIRMED LOOK-AHEAD**, 11 early bars, max |Δ| 34.43.
3. `_check_fillna()` does `ffill().bfill()` — the **`bfill()` pulls future values backwards** into
   leading NaNs; **PSAR up/down are genuinely affected**.
4. `ichimoku_b()` hardcodes `min_periods=0` — the first 51 Span B values are computed on partial
   windows and are **silently wrong**.
5. `ta.utils.dropna()` **silently deletes every row where any numeric column equals `0.0`** — it will
   destroy zero-volume and zero-return rows.

## Choosing

| Need | Use |
|---|---|
| Reference correctness, C speed | **TA-Lib** |
| **Streaming / live parity with backtest** | **`talipp`** |
| Pandas-native breadth, MIT | **`pandas-ta-classic`** (watch ADX warm-up) |
| Pure Python, no build step | `ta` — **with `fillna=False`**, discard warm-up generously |
| Large panels, fastest | polars-based indicators, or vectorbt's built-ins |

⚠️ **vectorbt's built-in indicators** are convenient but its licence is **Apache-2.0 + Commons
Clause**, and its `from_signals` default fills at the signal's own bar close — see
`../../backtesting-engines/SKILL.md` §2.1.
