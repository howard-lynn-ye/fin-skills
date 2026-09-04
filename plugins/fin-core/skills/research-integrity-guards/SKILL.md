---
name: research-integrity-guards
description: >-
  Second-pass audit that decides whether a finance result is real, applied after the work exists.
  TRIGGER - about to REPORT, publish or act on a backtest, factor test or model score; a result
  that looks good ("Sharpe 2.5", "beats SPY", "85% accuracy") and needs challenging; asked to
  validate, verify, sanity-check or critique a research design; asked "what should I check".
  Covers five gates: universe survivorship, availability timestamps, label leakage, cost realism,
  trial count. SKIP when the task is to BUILD something rather than judge it - go to the domain
  skill first (market-data-sourcing, backtesting-engines, factor-and-timeseries-research) and
  return here before reporting a number.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Research integrity guards

A quant result is a claim about money. Most published and self-produced results fail for one of
five reasons, in this order of frequency. Work through them in order — an early failure makes the
later checks irrelevant.

**The default posture is disbelief.** A strategy that beats the market is the extraordinary claim;
the burden is on the result, not on the skeptic. If a check below cannot be answered from the
artifacts, the answer is "unknown", and unknown is treated as failed.

## The five-gate audit

| # | Gate | Fails when | Detail |
|---|---|---|---|
| 1 | **Universe** | The ticker list was built from today's constituents | §1 |
| 2 | **Timestamps** | A value is used before it existed | §2 |
| 3 | **Labels & features** | Training saw the test period | §3 |
| 4 | **Costs & fills** | The trade could not have been executed at that price | §4 |
| 5 | **Trials** | The number of variants tried is unreported | §5 |

---

## 1. Universe — survivorship bias

**The failure:** you screen "S&P 500 stocks" or `financedatabase` or `company_tickers.json` today,
then backtest 2010–2026. Every company that went to zero is missing. The bias is largest exactly
where the strategy claims to add value.

**Delisted-security coverage, verified 2026-09-03:**

| Source | Delisted retained? |
|---|---|
| **SEC EDGAR filings** | ✅ Yes — Lehman (6,341 filings), Bear Stearns (3,344), Enron (351) all intact |
| **EODHD** (paid) | ✅ `get_list_of_tickers(code, delisted=1)` — the cheapest genuinely bias-free equity universe |
| **Tiingo** | ⚠️ ticker metadata carries `endDate`; free-tier history retention unverified |
| **FinanceDataReader** | ⚠️ `StockListing('KRX-DELISTING')` — Korea only |
| **CRSP / Norgate / Polygon** (paid) | ✅ |
| **yfinance, yahooquery, stockdex, defeatbeta, financedatabase, financetoolkit(free), Alpha Vantage** | 🚨 **No — all survivorship-biased** |
| **SEC `company_tickers.json`** | 🚨 **No** — a *current* snapshot. Lehman/Bear/Enron all return `tickers: []`. Using it re-introduces the exact bias EDGAR would have let you avoid. |

**The subtler version — identifier instability.** CIKs and tickers are not stable entities.
Verified example, CIK 933136: `WASHINGTON MUTUAL INC` (1995–2006) → failed in the largest US bank
failure → `WMI HOLDINGS` bankruptcy shell → `WMIH CORP` → reverse-merged into **Mr. Cooper Group**
(a mortgage servicer) → acquired by Rocket 2025. `companyfacts` returns **one continuous series**
for all of it. Keying a backtest on CIK creates a "company" that is a failed thrift, then an empty
shell, then a mortgage servicer.

Also: 18% of CIKs map to more than one ticker; JPM's list includes preferred shares (`JPM-PC`) and
**ETNs** (`VYLD`, `AMJB`). A naive ticker→CIK join attaches JPMorgan's income statement to a note.

**Guards:**
- Build the universe from a **dated membership snapshot**, never from a current list.
- Join on `(identifier, date)` against a listing/delisting table, never on identifier alone.
- Check `formerNames`; treat a SIC change as a discontinuity.
- If you cannot get delisted names, **say so in the result** and treat the Sharpe as an upper bound.

## 2. Timestamps — the availability rule

**One rule governs everything:**

> A datum may enter a decision only at `available_at = max(publication_time, retrieval_time,
> processing_completion)`, expressed in a single timezone, compared against the exchange session.

### 2a. Filings: three different dates, only one is availability

- **`period` / `reportDate`** — fiscal period end. The data **does not exist yet**. Using it is a
  30–90 day look-ahead. This is the single most common fundamental-data bug.
- **`filed` / `filingDate`** — the date EDGAR assigns, after applying its cutoff (17:30 ET for
  periodic reports; **22:00 ET for Forms 3/4/5**). Safe-ish but still admits a 90-minute post-close leak.
- **`acceptanceDateTime`** — the true wall-clock moment. **Use this.**

🚨 **Verified timezone inconsistency, undocumented by the SEC:** the Submissions API's
`acceptanceDateTime` is **genuine UTC**, but the Financial Statement Data Sets' `sub.txt.accepted`
is **Eastern Time**. Test that established it: for filings accepted 18:00–21:59 in the stamped
zone, the Submissions API shows 454 same-day / 4 next-day `filingDate` (consistent with UTC),
while FSDS shows 0 same-day / 370 next-day (consistent with ET). **Comparing the two without
converting introduces a silent 4–5 hour error.**

Worked example: Apple's 8-K accepted `2026-07-30T20:30:28Z` = **16:30 ET** — 30 minutes *after* the
close, but stamped `filingDate 2026-07-30`. Marking it available at that day's close trades on
information that did not exist. Apple's earnings 8-Ks cluster at 20:30Z/21:30Z, i.e. systematically
post-close.

**Rule:** convert `acceptanceDateTime` → exchange local time; if ≥ 16:00, the earliest tradeable
bar is the **next session's open**.

### 2b. Restatements

Fundamentals are rewritten after the fact. **The SEC's `companyfacts` genuinely IS point-in-time
reconstructable** — it returns every vintage with a `filed` field. The look-ahead is a *consumption*
bug, not an API limitation:

```python
# WRONG — silently selects the restated value
df.drop_duplicates(subset=['start', 'end'], keep='last')

# RIGHT — latest vintage KNOWN AS OF the decision date
df[df.filed <= as_of].sort_values('filed').groupby(['start', 'end']).last()
```

Scale, measured on Apple alone: **408 `(start, end, form)` groups have differing values across
vintages.** `AccountsPayableCurrent` FY2017 went 49,049M → 44,242M. `AntidilutiveSecurities` FY2019
went 15.5M → 62.0M — that one is the 2020 4-for-1 split retroactively rewriting share counts.
**Every per-share metric in XBRL is silently split-adjusted backwards**, so a point-in-time EPS
study compares split-adjusted denominators to unadjusted prices unless you handle it.

🚨 **The SEC XBRL `frames` API is not point-in-time and cannot be made so** — its records carry no
`filed` and no `form`. Measured: of 3,141 CY2023 revenue values, only **18 (0.6%)** come from a
filing actually made in 2023; **53% come from filings made in 2026**. Never use it for backtesting.
The **Financial Statement Data Sets** quarterly ZIPs are PIT-safe by construction — each contains
only filings made during that quarter.

### 2c. Macro revisions

GDP, payrolls and CPI are revised for years. FRED's own example: 2013Q4 GDP was 17102.5
(2014-01-30) → 17080.7 (2014-02-28) → 17089.6 (2014-03-27). Backtesting on the current series
trades on numbers published up to a decade later.

- `fredapi.get_series_all_releases()` gives per-observation `realtime_start` — **use `realtime_start`
  as the timestamp, not the observation date.** A January figure is published in February.
- 🚨 `fredapi.get_series_as_of_date()` does **not** do what its docstring says — it returns a
  DataFrame with **duplicate `date` rows** (every revision up to the date). You must add
  `.groupby('date').last()` yourself.
- `pandas_datareader`'s FRED reader has **zero** vintage support (verified: no `realtime|vintage|
  alfred|as_of` anywhere in its source). Not research-grade.
- Deepest free vintage source: **Philadelphia Fed Real-Time Data Set** — a 244-vintage matrix,
  1965Q4→2026Q3, plain XLSX download, no key, no wrapper needed.
- Seasonal adjustment **rewrites already-published history** with no new information. Prefer NSA, or
  use the vintage current at each decision date.

### 2d. Price adjustment

Backward-adjusted (hfq) is anchored at the listing date: new dividends only append, history never
changes. Forward-adjusted (qfq) is anchored at *now*: **every new corporate action rescales the
entire history retroactively**, so a qfq price at *t* is a function of events after *t*.

**Use backward-adjusted or raw+factors. Never forward-adjusted.** If you must display qfq,
recompute at render time and never persist it. Details and per-library defaults:
`references/adjustment-conventions.md`.

## 3. Labels and features — leakage

Read `references/leakage-checklist.md` for the full list. The ones that actually bite:

1. **Any CV that shuffles.** `train_test_split`, `KFold(shuffle=True)`, default `cross_val_score`
   on a DataFrame — catastrophic. Even unshuffled `KFold` fails without purging.
2. **Overlapping label windows.** If labels span `[t0, t1]`, adjacent events share
   outcome-determining bars, so your effective N is far below your row count. This is the entire
   reason purging and embargo exist. Use `purgedcv` (it *forces* you to pass `evaluation_times`)
   or `skfolio.model_selection.CombinatorialPurgedCV`.
3. **Purge horizon set to the mean holding period.** It must be the **maximum**.
4. **Embargo omitted.** Purging handles overlap; the embargo handles serial correlation *after*
   the test window. Both are needed. In skfolio these are counted in **observations, not time** —
   with dollar/volume bars a fixed observation count is a wildly varying time span.
5. **Scalers/normalizers fit on the full sample.** Put them in a `Pipeline` so the splitter governs them.
6. **Stationarity parameters fit on the full sample** — e.g. searching fractional `d` by ADF over
   train+test leaks the test distribution into a feature transform.
7. **Volatility targets and event thresholds computed over the whole series** (e.g. `get_daily_vol()`
   then used as both the barrier width and the CUSUM threshold) seeds future volatility into event
   selection. Use expanding/rolling windows.
8. **Meta-label leakage:** if the primary model's `side` is an in-sample fitted value, the secondary
   model learns to trust an artificially good signal. The primary must emit **out-of-sample** sides.
9. **Repainting indicators** — centered moving averages, zigzag, anything using a future bar. See
   `signal-construction`.
10. **Resample label placement.** `pandas.resample()` labels at the bin *start* by default; check
    `label=` and `closed=` on every bar aggregation.

## 4. Costs and fills — could the trade have happened?

- **Signal→order timing.** A signal computed from bar *t*'s close cannot be filled at bar *t*'s
  close. Next open at the earliest. Per-engine behaviour: `backtesting-engines`, §correctness.
- **Unfillable prints.** A bar that closes limit-up generally could not have been bought
  (A-share specifics: `fin-china`). Halted, illiquid, and auction-only bars likewise.
- **Costs to model, in order of how often they are omitted:** spread (not just commission),
  slippage as a function of participation rate, market impact, borrow cost for shorts, financing
  on margin, and the cash leg's interest. Omitting the cash rate alone can flip an alpha estimate
  — regressing raw returns on raw returns lets the intercept absorb the risk-free rate.
- **Capacity.** Report the assumed participation rate. A strategy that needs 30% of ADV is a
  different claim from one that needs 0.3%.
- **Sanity floor:** re-run with 2× your cost assumption. If the result dies, the result *is* the
  cost assumption.

## 5. Trials — multiple testing

**The number of variants you tried is part of the result.** Every barrier multiple, feature set,
lookback, threshold, universe filter and rebalance frequency is a trial — including the ones you
abandoned, and including everything an automated search (RD-Agent, gplearn, grid search) evaluated.

- Maintain a **trial ledger**: every candidate, its parameters, and its result, appended before you
  look at out-of-sample performance.
- **Deflated Sharpe Ratio** needs the *true* trial count `N`, plus the variance of Sharpe across
  trials and return skew/kurtosis. **Under-reported `N` makes DSR a rubber stamp** — the most common
  abuse is reporting `N` = "the 5 models I saved".
- For "is my strategy better than the benchmark after data snooping", the right tools are
  **White's Reality Check / Hansen's SPA / StepM / Model Confidence Set**, all in `arch.bootstrap`.
  `jsharpe` adds explicit FWER/FDR corrections. See `backtest-validation`.
- Peer-reviewed comparison of out-of-sample testing methods (Arian, Norouzi & Seco, *Knowledge-Based
  Systems* 305, 2024) finds **CPCV outperforms K-Fold, Purged K-Fold and especially Walk-Forward** on
  both PBO and DSR, with walk-forward showing weak false-discovery control. *(Note: the lead author
  also authors `RiskLabAI`, so this is peer-reviewed but not disinterested.)*

## 6. What to write down

A result is not reportable without these. Reuse `scripts/result_manifest.py` to emit them.

```
universe_snapshot_id + how membership was determined (and whether delisted names are present)
data sources + retrieval timestamps + adjustment convention
available_at rule actually applied
train/test split definition, purge horizon, embargo
cost model (spread, slippage, impact, borrow, financing, cash rate)
trial count N, and where the ledger lives
the metric, its annualization factor, and its risk-free-rate treatment
what would falsify this result
```

## 7. Honest reporting

- Report the **cost-after, benchmark-relative** number as the headline, not the gross one.
- Separate **alpha** from **beta and style exposure**. A long-equity strategy in a bull market is
  not alpha; regress against the benchmark and common factors before claiming skill.
- State the confidence interval. A 3-year Sharpe of 1.0 has a standard error near 0.58 — it is not
  distinguishable from zero.
- A negative result is a result. "The unconditional baseline did not clear the threshold" is a
  finding, and it is the correct output far more often than a positive one.
