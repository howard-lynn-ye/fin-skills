# qlib

Microsoft's AI-oriented quant research platform — expression engine, `Alpha158`/`Alpha360` feature
sets, a 20+ model zoo, `qrun` YAML workflows and a point-in-time data layer. **Its default label is
leakage-safe; its default preprocessing is not, and its official dataset is switched off.**

| | |
|---|---|
| pip | 🚨 **`pyqlib`**, imported as `qlib`. The PyPI package named `qlib` is an unrelated abandoned 2018 package |
| Version | **0.9.7 (2025-08-15)** ✅ — while `microsoft/qlib` was pushed **2026-09-02**. **Packaging lags the repo by ~12 months** |
| GitHub | `microsoft/qlib` — **48,285★**, 301 open issues ✅ |
| Licence | **MIT** ✅ (PyPI classifier + GitHub SPDX) |
| Python | ⚠️ `requires_python >=3.8.0`, but see the wheel trap below |
| Wheels | 🚨 **cp38–cp312 only, and NO sdist** ✅ — `pip install pyqlib` **fails outright on Python 3.13+** |

Windows wheels exist for cp310/311/312 (`win_amd64`) ✅ — it installs fine on this machine's 3.11.

## 🚨 Traps

**1. 🚨 `ZScoreNorm` leaks your test set into training — the single most common Qlib leak, and it is
silent.** ✅ read from `qlib/contrib/data/handler.py`:

```python
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf"},
    {"class": "ZScoreNorm"},      # time-series z-score, fit on [fit_start_time, fit_end_time]
    {"class": "Fillna"},
]
```

`ZScoreNorm` is a **time-series** normalizer whose mean and std are fit on `[fit_start_time,
fit_end_time]`. Qlib forces you to pass those (`check_transform_proc` asserts non-None) but does
nothing to stop you passing your whole sample — and every tutorial that sets `fit_end_time` to the end
of the data has **leaked test-set moments into training features**. **Set `fit_start_time` /
`fit_end_time` to your training segment only.**

(`CSZScoreNorm` in `_DEFAULT_LEARN_PROCESSORS` is cross-sectional, per-date, applied to the *label* —
leakage-free by construction. The dangerous one is the time-series normalizer on features.)

**2. ✅ The default label is correct — do not "improve" it.** Both handlers define:

```python
def get_label_config(self):
    return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
```

That is `close[t+2]/close[t+1] - 1`: signal formed on day `t`'s close, traded on `t+1`, earning the
`t+1`→`t+2` return. **Leakage-safe by construction.** Rewriting it to `Ref($close,-1)/$close-1`
introduces a one-day look-ahead you cannot trade — a very common "fix" that inflates every metric.

**3. 🚨 The official China dataset is disabled.** Qlib's README states verbatim: *"Due to more
restrict data security policy. The official dataset is disabled temporarily."* The documented
`qlib_data --region cn` command still appears in the README but the backing data is gone. The **US**
bundle is a community-sourced Yahoo dump shipped with an explicit quality warning. **A Qlib workflow
cannot promise a turnkey dataset.**

✅ Live China replacement: **`chenditc/investment_data`** — Apache-2.0, 1,440★, release tags published
**daily**, asset `qlib_bin.tar.gz` ≈ 563.8 MB. It merges Tushare + akshare + baostock + Yahoo +
historical Wind/Caihui dumps, cross-validates them, rescales adjustment factors to a common basis, and
explicitly backfills delisted companies — so **delisted names are included**, which the official
bundle never handled well. Setup command in `../SKILL.md` §2.

**4. 🚨 `cn_data` stores both a `$factor` field and adjusted prices.** Know which convention a given
dump uses before mixing it with any other source, or you will double-adjust and never see an error.
Qlib ships a `check_data_health` script precisely because these dumps have gaps.

**5. 🚨 The RL submodule has been frozen since 2023 and unpickles data.** ✅ `qlib/rl`'s last
functional commits are 2023 (order-execution open-sourcing, data-format refinement); since then only a
Python-version bump and **two pickle-deserialization security fixes** (2025-12 `#2076`, 2026-03
`#2153`, adding a `RestrictedUnpickler`). **Do not load untrusted qlib artifacts.** Note the scoping is
deliberate and sound: `qlib/rl/order_execution` targets **order execution**, not alpha or allocation —
the one RL-in-finance application with a defensible case.

**6. ⚠️ The expression engine reads from the Qlib provider, not from memory.** `Ref`, `Mean`, `Std`,
`Slope`, `Resi`, `Corr`… all resolve against the installed `.bin` dataset. **You cannot point Alpha158
at an arbitrary pandas DataFrame.** Accepting Qlib's binary data format is the real adoption cost. The
formulas are readable in `qlib/contrib/data/loader.py` — reimplementing Alpha158 in plain pandas is a
legitimate option and often the cheaper one.

**7. ⚠️ Alpha158 and Alpha360 are feature-set *definitions*, not models.** ✅ Both are `DataHandlerLP`
subclasses emitting a fixed expression list. **Alpha360** = the last 60 days of 6 raw fields
(open/high/low/close/vwap/volume), each normalized by the current close/volume → 360 columns; a raw
tensor for sequence models, not a hand-crafted alpha set. **Alpha158** = 158 hand-crafted features in
four groups — `kbar` candle-shape ratios, `price`, `volume`, and `rolling` over windows
`[5,10,20,30,60]` with the operator set ROC/MA/STD/BETA/RSQR/RESI/MAX/MIN/QTLU/QTLD/RANK/RSV/IMAX/
IMIN/CORR/CORD/CNTP/SUMP/VMA/VSTD/WVMA/VSUMP, price-scale quantities divided by `$close`.

**8. ⚠️ No point-in-time fundamentals for China.** Qlib has a PIT database *concept* in the data layer,
but nothing populates Chinese 公告日 for you. `instruments/csi300.txt` encodes membership date ranges
whose quality is **entirely inherited from the dump you installed**. Source PIT data separately —
`../../china-ashare-data/references/tushare.md`.

## Minimal correct usage — the leak closed

```python
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

TRAIN = ("2010-01-01", "2018-12-31")      # normalization must see ONLY this
handler = Alpha158(
    instruments="csi300",
    start_time="2010-01-01", end_time="2022-12-31",
    fit_start_time=TRAIN[0],              # 🚨 training segment only — not the full sample
    fit_end_time=TRAIN[1],
)
df = handler.fetch()                      # plain DataFrame → sklearn / LightGBM
# label stays Ref($close,-2)/Ref($close,-1)-1 — leave it alone
```

## Where it fits

- Dataset setup, RQAlpha's licence trap, framework selection: `../SKILL.md` §2
- A-share rules any Qlib backtest must still encode: `_ashare-rules.md`
- Live execution of what you research here: `vnpy.md`
- Feeding it real Chinese data: `../../china-ashare-data/SKILL.md`
- Alpha158 as one of several factor libraries:
  `../../../../fin-core/skills/factor-and-timeseries-research/references/alpha-factor-libraries.md`
