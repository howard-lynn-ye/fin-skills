---
name: lib-qlib
description: >-
  Microsoft Qlib (pip name pyqlib, imported as qlib) ships Alpha158/Alpha360 and a default
  normalizer that leaks your test set into training, silently. TRIGGER - qlib, pyqlib, "pip
  install pyqlib", qlib.init, provider_uri, REG_CN, Alpha158, Alpha360, DataHandlerLP, ZScoreNorm,
  CSZScoreNorm, fit_start_time, fit_end_time, qrun, workflow_config yaml, "qlib_data --region cn",
  LABEL0, "Ref($close, -2)", qlib expression engine, qlib .bin dataset, investment_data qlib_bin.
  Wheels are cp38-cp312 with no sdist so it fails outright on Python 3.13+, packaging lags the
  repo by about a year, and the official China dataset was switched off. SKIP for lib-alphalens,
  which is the skill for scoring a factor you already have. SKIP when the question is WHICH
  library to choose rather than how to use this one - that belongs to the domain skill.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# qlib

Microsoft's AI-oriented quant research platform — expression engine, `Alpha158`/`Alpha360` feature
sets, a 20+ model zoo, `qrun` YAML workflows. **Its default label is leakage-safe; its default
preprocessing is not, and its official dataset is switched off.**

| | |
|---|---|
| pip / import | **`pyqlib`**, imported as `qlib`. The PyPI package named `qlib` is an unrelated abandoned 2018 package |
| Version | 0.9.7 (2025-08-15) while `microsoft/qlib` was pushed 2026-09-02 — **packaging lags the repo by ~12 months** |
| Licence | MIT (PyPI classifier + GitHub SPDX) |
| Python | `requires_python >=3.8.0`, but wheels are **cp38–cp312 only with NO sdist** — `pip install pyqlib` **fails outright on 3.13+** |
| Status | 48,285★, 301 open issues. Windows wheels exist for cp310/311/312 (`win_amd64`) |

## The trap that costs you money

**`ZScoreNorm` leaks your test set into training, and it is silent.** From
`qlib/contrib/data/handler.py`:

```python
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf"},
    {"class": "ZScoreNorm"},      # time-series z-score, fit on [fit_start_time, fit_end_time]
    {"class": "Fillna"},
]
```

`ZScoreNorm` is a **time-series** normalizer fit over `[fit_start_time, fit_end_time]`. Qlib forces
you to pass those (`check_transform_proc` asserts non-None) but does nothing to stop you passing the
whole sample — and every tutorial that sets `fit_end_time` to the end of the data has **leaked
test-set moments into training features**. **Set both to your training segment only.**

(`CSZScoreNorm` in `_DEFAULT_LEARN_PROCESSORS` is cross-sectional, per-date, applied to the *label* —
leakage-free by construction. The dangerous one is the time-series normalizer on features.)

## The default label is correct — do not "improve" it

```python
def get_label_config(self):
    return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
```

That is `close[t+2]/close[t+1] - 1`: signal formed on day `t`'s close, traded on `t+1`, earning the
`t+1`→`t+2` return. **Leakage-safe by construction.** Rewriting it to `Ref($close,-1)/$close-1`
introduces a one-day look-ahead you cannot trade — a very common "fix" that inflates every metric.

## There is no turnkey dataset

**The official China dataset is disabled.** Qlib's README states verbatim that due to a more strict
data security policy, the official dataset is disabled temporarily. The documented
`qlib_data --region cn` command still appears in the README but the backing data is gone. The **US**
bundle is a community-sourced Yahoo dump shipped with an explicit quality warning.

The live China replacement is **`chenditc/investment_data`** — Apache-2.0, 1,440★, daily release
tags, asset `qlib_bin.tar.gz` ≈ 563.8 MB. It merges Tushare + akshare + baostock + Yahoo + historical
Wind/Caihui dumps, cross-validates them, rescales adjustment factors to a common basis, and
explicitly backfills delisted companies — so **delisted names are included**.

Two further data traps: `cn_data` stores **both** a `$factor` field and adjusted prices, so mixing it
with another source double-adjusts with no error, and Qlib ships `check_data_health` precisely
because these dumps have gaps. There are also **no point-in-time fundamentals for China** — nothing
populates 公告日, and `instruments/csi300.txt` membership ranges inherit the dump's quality entirely.

## The expression engine reads from the provider, not from memory

`Ref`, `Mean`, `Std`, `Slope`, `Resi`, `Corr` and the rest resolve against the installed `.bin`
dataset. **You cannot point Alpha158 at an arbitrary pandas DataFrame** — accepting Qlib's binary
format is the real adoption cost. The formulas are in `qlib/contrib/data/loader.py`, so
reimplementing Alpha158 in plain pandas is legitimate and often cheaper. Both feature sets are
`DataHandlerLP` subclasses emitting a fixed expression list — **definitions, not models**.
**Alpha360** = 60 days of 6 raw fields each normalized by the current close/volume → 360 columns, a
raw tensor for sequence models. **Alpha158** = 158 hand-crafted features in four groups: `kbar`
candle-shape ratios, `price`, `volume`, and `rolling` over windows `[5,10,20,30,60]`.

Security note: `qlib/rl` has been frozen since 2023 and **unpickles data** — the only recent commits
are two pickle-deserialization fixes (`#2076`, `#2153`). **Do not load untrusted qlib artifacts.**

## Minimal correct call

```python
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

TRAIN = ("2010-01-01", "2018-12-31")      # normalization must see ONLY this
handler = Alpha158(
    instruments="csi300",
    start_time="2010-01-01", end_time="2022-12-31",
    fit_start_time=TRAIN[0],              # training segment only — NOT the full sample
    fit_end_time=TRAIN[1],
)
df = handler.fetch()                      # plain DataFrame -> sklearn / LightGBM
# label stays Ref($close,-2)/Ref($close,-1)-1 — leave it alone
```

## See also

- `../../../fin-china/skills/china-trading-stack/SKILL.md` §2 — dataset setup and framework choice
- `../../../fin-china/skills/china-trading-stack/references/qlib.md` — the verified reference card
- `../../../fin-china/skills/china-trading-stack/references/_ashare-rules.md` — rules to encode
- `../lib-tushare/SKILL.md` — sourcing the point-in-time fundamentals Qlib will not give you

## Where this sits

This file is the deep dive on **one** library and assumes the choice is already made.
For which library to pick, how it compares with the alternatives, and the traps that span
several of them, the entry point is the domain skill **`china-trading-stack`** (`../../../fin-china/skills/china-trading-stack/SKILL.md`).

