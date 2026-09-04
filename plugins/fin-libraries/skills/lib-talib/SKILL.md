---
name: lib-talib
description: >-
  The C reference implementation of technical indicators, where every pure-Python port disagrees
  during warm-up and none of them say so. TRIGGER - import talib, pip install TA-Lib,
  ta-lib-python, talib.RSI, talib.MACD, talib.ATR, talib.ADX, talib.BBANDS, talib.OBV, from talib
  import abstract, talib.get_functions, unstable period, indicator warm-up, an indicator differing
  between two libraries or between backtest and live; "Exception: input array type is not double",
  a failed ta-lib C build or missing ta_libc.h, and the pandas-ta vs pandas-ta-classic vs talipp
  vs ta choice. Memory is stale here: the install pain is solved - 0.7.1 (2026-07-16) ships 54
  prebuilt wheels bundling the C library, including cp311-win_amd64 - while pandas-ta's repo,
  homepage and release history are all gone. SKIP for whether a signal actually predicts returns
  (factor-and-timeseries-research) and for leak-free signal construction generally
  (signal-construction). SKIP for choosing between libraries - that is the domain skill's job.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# TA-Lib

The C reference every other Python indicator library is measured against — and the reason two
libraries computing "the same" RSI hand you two different backtests.

| | |
|---|---|
| pip / import | `pip install TA-Lib` · `import talib` (repo `ta-lib-python`, now the `ta-lib` org) |
| Version | **0.7.1 (2026-07-16)** · C core 0.7.1 · `requires_python >=3.9` |
| Licence | BSD-ish |
| Status | ✅ **54 wheels incl. `cp311-win_amd64`**, bundling the C library since **0.6.5 (2025-08-07)** |

## The trap that costs you money

🚨 **Every pure-Python port emits values *before* TA-Lib does, computed on partial windows —
silently wrong, not NaN.** TA-Lib's documented "unstable period" behaviour is exactly why its warm-up
start indices differ from every port. Swap libraries, or restart the calculation per symbol on a
short slice, and **the same strategy becomes two different backtests**.

Measured — TA-Lib python 0.7.1 / C core 0.7.1, pandas 3.0.5, numpy 2.4.6, 500-bar synthetic GBM,
seed 7; `maxabs` = max absolute difference over the overlapping non-NaN region:

| Library | SMA/BBands/CCI/OBV | RSI(14) | EMA(20) | ATR(14) | ADX(14) | MACD |
|---|---|---|---|---|---|---|
| **talipp 2.7.0** | exact | **1.4e-14, exact incl. warm-up start index** | 2.8e-14 | 3.2e-02 warm-up only | — | 0.155 warm-up only |
| **pandas-ta-classic 0.6.52** | ≤3e-11 | **3.6e-14** | 2.8e-14 | 8.9e-16 | 🚨 **4.25**, emits from bar 13 vs TA-Lib's 27 | 0.155 early bars |
| **ta 0.11.0** | identical | 🚨 **3.84** | 🚨 0.363 | 🚨 0.0477, emits **from bar 0** | 🚨 **0.642**, emits **from bar 0** | 🚨 0.378, 8 bars early |

**How long `ta`'s RSI(14) takes to converge on the same input:** `|diff| > 1.0` through **bar 36** ·
`> 0.1` through **bar 67** · `> 0.01` through **bar 97** · `> 1e-6` through **bar 217**.

> 🚨 **Universal rule: discard the first ~10–15 × period bars of any recursive indicator (EMA, RSI,
> ATR, ADX, MACD), whatever the library.**

## The install folklore is obsolete

"Needs the C library compiled by hand; no Windows wheels" was true and **is not any more**. `ta-lib-python` ships prebuilt wheels **bundling the TA-Lib C library** since v0.6.5. `pip install TA-Lib` just works on Windows. If you are still writing build instructions into a README, that guidance is stale.

## Calling it correctly

**It takes numpy float64 arrays, not Series.** Pass `.values` and reattach the index yourself, or
use the abstract API, which accepts dict / pandas / polars inputs.

```python
import talib
rsi = talib.RSI(close, timeperiod=14)          # close must be float64 ndarray

from talib import abstract
out = abstract.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
```

## Which library for which job

| Package | Version | Licence | Verdict |
|---|---|---|---|
| **TA-Lib** | 0.7.1 (2026-07-16) | BSD-ish | ✅ the reference |
| **`talipp`** | 2.7.0 (2025-09-09) | MIT | ✅ incremental/streaming; closest match, **only one that also matches TA-Lib's RSI warm-up start index** |
| **`pandas-ta-classic`** | 0.6.52 (2026-06-24) | **MIT** | ✅ the maintained successor, 427★, py≥3.10; watch ADX warm-up |
| `ta` (bukosabino) | 0.11.0 (2023-11-02) | MIT | ⚠️ pure python, **sdist only**, ~3 yrs stale — usable with `fillna=False` |
| 🚨 `pandas-ta` | 0.4.71b0 (2025-09-14) | 🚨 **NONE DECLARED** | 🔴 **do not install** |
| 🔴 `finta` | 1.3 (2021) | LGPL-3.0 | repo **ARCHIVED** 2022-07-24 |
| 🔴 `tulipy` / `bta-lib` | 0.4.0 (2019) / — | LGPL / — | dead |

✅ **`talipp`'s incremental `.add()` produces bit-identical output to batch construction.** That is the whole reason it exists, and it is the correct choice when a strategy runs streaming — it removes the backtest-vs-live warm-up divergence.

## 🚨 `pandas-ta` — supply-chain caution

- `github.com/twopirllc/pandas-ta` → **HTTP 404** (removed ~June 2025).
- PyPI now lists **only two releases**, 0.4.67b0 and 0.4.71b0. **All prior history, including the
  widely used 0.3.14b, was deleted.**
- **No `license` field and no license classifier**; `requires_python >=3.12`; the `Repository` URL
  still points at the dead 404 repo.
- `pandas-ta.dev` — **DNS does not resolve**, and archive.org has **no snapshots**.

**A package with no licence, wiped release history, a dead upstream repo and a dead homepage is a
supply-chain risk regardless of intent.** Pin to `pandas-ta-classic`, `ta`, or TA-Lib.

## 🚨 `ta`'s verified bugs, if you must use it

Isolated by perturbing only future bars and checking whether early bars changed:

1. `IchimokuIndicator(..., visual=True).ichimoku_a()` — **CONFIRMED LOOK-AHEAD**, 26 early bars
   changed, max |Δ| 35.34, **even with `fillna=False`**.
2. `DPOIndicator(..., fillna=True).dpo()` — **CONFIRMED LOOK-AHEAD**, 11 early bars, max |Δ| 34.43.
3. `_check_fillna()` does `ffill().bfill()` — the **`bfill()` pulls future values backwards** into
   leading NaNs; **PSAR up/down are genuinely affected**.
4. `ichimoku_b()` hardcodes `min_periods=0` — the first 51 Span B values are computed on partial
   windows and are **silently wrong**.
5. `ta.utils.dropna()` **silently deletes every row where any numeric column equals `0.0`** — it will
   destroy zero-volume and zero-return rows.

## See also

- `../../../fin-core/skills/signal-construction/SKILL.md` — leak-free signal construction, §2 measured agreement
- `../../../fin-core/skills/signal-construction/references/indicator-libraries.md` — the source card
- `../../../fin-core/skills/signal-construction/references/_repainting.md` — the full repainting detail

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`signal-construction`** (`../../../fin-core/skills/signal-construction/SKILL.md`).

