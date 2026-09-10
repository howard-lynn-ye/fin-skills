---
name: fundamental-and-macro-data
description: >-
  Company fundamentals and macro series with correct point-in-time semantics. TRIGGER - 10-K,
  10-Q, 8-K, 13F, Forms 3/4/5, filings, EDGAR, XBRL, accession number, CIK, "which CIK is this
  ticker", ticker-to-CIK mapping, edgartools; parsing an income statement or balance sheet out of
  a filing; revenue, EPS or balance-sheet history as it was known on a past date; restatements;
  earnings dates; or CPI, GDP, payrolls, unemployment, interest rates, FRED, ALFRED, data vintages
  and revisions. Load before joining ANY fundamental or macro series to prices: the obvious join
  is a look-ahead bug, and the SEC frames API cannot be made point-in-time. SKIP for price and
  OHLCV vendors (market-data-sourcing) and Chinese filings (china-ashare-data).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-03"
---

# Fundamentals and macro data

Both halves of this domain share one failure: **the obvious join is a look-ahead bug.** Fundamentals
are published months after the period they describe and are then rewritten; macro series are revised
for years. Sections 2 and 5 are the ones that change your results.

## 1. SEC EDGAR — the two hard rules

```python
HDR = {"User-Agent": "Your Company you@example.com"}   # REQUIRED — no UA → HTTP 403
```
✅ Empirically confirmed: no User-Agent → **403**; any UA → 200. The header is enforced for
*presence*, not content — but sending a fake browser UA is a policy violation and gets IPs blocked.
✅ **Rate limit: 10 requests/second**, quoted from the SEC webmaster FAQ.

| Endpoint | Detail |
|---|---|
| `data.sec.gov/submissions/CIK##########.json` | Zero-padded 10-digit CIK. `filings.recent` caps at ~1,000; older paginate via `filings.files[]` |
| `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | AAPL = **3.8 MB**, 503 us-gaap tags. Fact fields: `start, end, val, accn, fy, fp, form, filed, frame` |
| `data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json` | Single tag, same fact schema |
| `data.sec.gov/api/xbrl/frames/us-gaap/<Tag>/USD/CY2023.json` | 🚨 **not point-in-time — see §2.3** |
| `efts.sec.gov/LATEST/search-index?q=…&forms=10-K` | Full-text search. ✅ **Coverage starts 2001** — it does *not* reach 1994–2000 filings |
| `sec.gov/files/company_tickers.json` | 10,412 rows. 🚨 **A current snapshot — survivorship-biased** |
| `sec.gov/files/dera/data/financial-statement-data-sets/YYYYqQ.zip` | ✅ **70 quarterly ZIPs, 2009Q1 → 2026Q2** |

🚨 **Frames period syntax:** duration facts use `CY2023`/`CY2023Q1`, but **instantaneous** facts
(Assets, StockholdersEquity) need the **`I` suffix** — `CY2023Q4I`. ✅ Verified: `Assets/USD/CY2023Q1`
→ **404**, `CY2023Q4I` → 200. The #1 frames beginner error.

**Library choice:** `edgartools` (MIT, 2,648★, released 2026-09-02 — typed objects for 20+ form
types, XBRL-standardized statements, built-in MCP server) is the default. `sec-edgar-downloader`
downloads raw files only. `Arelle` is the only full XBRL **validator** — use it when correctness
beats convenience. 🚨 `python-xbrl` is **dead** (2016). 🚨 `edgar-crawler` is **GPL-3.0**.

## 2. 🚨 Point-in-time fundamentals

### 2.1 Three dates, only one is availability

- **`period` / `reportDate`** — fiscal period end. **The data does not exist yet.** Using it is a
  30–90 day look-ahead. The most common fundamental-data bug there is.
- **`filed` / `filingDate`** — the date EDGAR assigns after its cutoff (17:30 ET periodic reports;
  **22:00 ET for Forms 3/4/5**). Safe-ish, but still admits a 90-minute post-close leak.
- **`acceptanceDateTime`** — the true wall-clock moment. **Use this.**

🚨 **Verified, undocumented timezone inconsistency:** the Submissions API's `acceptanceDateTime` is
**genuine UTC**; the Financial Statement Data Sets' `sub.txt.accepted` is **Eastern Time**. The test:
for filings stamped 18:00–21:59, Submissions shows 454 same-day / 4 next-day `filingDate`
(consistent with UTC) while FSDS shows 0 / 370 (consistent with ET). **Comparing them without
converting introduces a silent 4–5 hour error.**

Worked example: Apple's 8-K accepted `2026-07-30T20:30:28Z` = **16:30 ET** — 30 minutes *after* the
close, but stamped `filingDate 2026-07-30`. Apple's earnings 8-Ks cluster at 20:30Z/21:30Z, i.e.
**systematically post-close**. Rule: convert to exchange local; if ≥16:00, the earliest tradeable
bar is the **next session's open**.

### 2.2 ✅ `companyfacts` IS point-in-time reconstructable — the bug is yours

It returns **every vintage**, one row per accession that ever reported the period, each with `filed`:

```python
# WRONG — silently selects the restated value
df.drop_duplicates(subset=['start', 'end'], keep='last')

# RIGHT — latest vintage KNOWN AS OF the decision date
df[df.filed <= as_of].sort_values('filed').groupby(['start', 'end']).last()
```

**Scale, measured on Apple alone: 408 `(start, end, form)` groups have differing values across
vintages.** `AccountsPayableCurrent` FY2017: 49,049M → 44,242M. `AntidilutiveSecurities` FY2019:
15.5M → 62.0M — that one is the 2020 4-for-1 split retroactively rewriting share counts. **Every
per-share metric in XBRL is silently split-adjusted backwards**, so a PIT EPS study compares
split-adjusted denominators against unadjusted prices unless you handle it.

### 2.3 🚨 The `frames` API is not point-in-time and cannot be made so

Its records carry **no `filed` and no `form`** — you cannot filter by vintage. Measured on the
CY2023 annual-revenue frame, by extracting the filing year from each `accn`:

| Filing year of the value frames chose | Count |
|---|---|
| 2023 | **18 (0.6%)** |
| 2024 | 406 |
| 2025 | 1,044 |
| **2026** | **1,673 (53%)** |

**Only 18 of 3,141 values come from a filing actually made in 2023; over half come from filings made
three years later.** Never use frames for backtesting. It is a current-state cross-section only.

### 2.4 ✅ The PIT-safe bulk path: Financial Statement Data Sets

70 quarterly ZIPs, ~85 MB each, four TSVs (`sub`, `num`, `pre`, `tag`). **PIT-safe by construction:**
each ZIP contains only filings *made during that quarter* (2026Q1's `filed` range is
20260102–20260331). Stack ZIPs in order and you get true point-in-time with no filtering logic.
`sub.txt` carries `prevrpt` (superseded by a later amendment) and `detail`; `num.txt` has `ddate`,
`qtrs` (0 = instantaneous, 1 = quarterly, 4 = annual), `segments`, `coreg`.

### 2.5 Identity, coverage and calendar traps

- ✅ **EDGAR itself is survivorship-bias-free**: Lehman (6,341 filings), Bear Stearns (3,344),
  Enron (351) are all intact. 🚨 **But `company_tickers.json` is not** — all three return
  `tickers: []`. Building a universe from it re-introduces the exact bias EDGAR would have avoided.
- **18% of CIKs map to >1 ticker.** `GOOGL → [GOOGL, GOOG, GOOGM, GOOGN]`; BAC has 17; JPM's 9
  include **preferred shares** (`JPM-PC`) and **ETNs** (`VYLD`, `AMJB`). A naive ticker→CIK join
  attaches JPMorgan's income statement to a note. Filter via `company_tickers_exchange.json` and
  drop `-P*` suffixes.
- **CIKs are not stable entities** — see `research-integrity-guards` §1 for the verified
  Washington Mutual → Mr. Cooper example.
- **XBRL coverage starts ~2009.** Lehman, Bear Stearns and Enron return **404** on companyfacts —
  they died before the mandate. **Any XBRL study is structurally post-2009 and cannot include the
  GFC's casualties.**
- **Amendments and late filers** (2025): 10-K/A 3,073 · 10-Q/A 1,238 · **NT 10-K 746** · NT 10-Q
  1,171. An `NT 10-K` (Form 12b-25) says the filing will be late — itself a tradeable negative
  signal, and it means the row simply won't exist on schedule. **Handle absence explicitly; do not
  forward-fill.**
- **Fiscal-year misalignment:** 95.8% of 10-Ks use `fye=1231`, 4.2% don't, and `fy` is a *label*:
  Target's `period=20260131` carries `fy=2025`. **Join on `period`, bucket to calendar quarters
  yourself.** `fye` is sometimes internally inconsistent — don't trust it blindly.

## 3. PIT reference snippet

```python
import requests, pandas as pd
HDR = {"User-Agent": "Your Company you@example.com"}

def pit_facts(cik: int, tag: str, as_of: str, taxonomy="us-gaap"):
    """Only data actually filed on or before `as_of`."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
    facts = requests.get(url, headers=HDR, timeout=30).json()["units"]["USD"]
    df = pd.DataFrame(facts)
    df["filed"] = pd.to_datetime(df["filed"])
    df = df[df["filed"] <= pd.Timestamp(as_of)].sort_values("filed")
    return df.groupby(["start", "end"], as_index=False).last()   # latest vintage KNOWN THEN
# <=10 req/s. Never use /api/xbrl/frames/ for this — it has no `filed`.
```

## 4. Macro sources

| Need | Use | Licence |
|---|---|---|
| **US macro with vintages** | **`fredapi`** (needs a free key) | Apache-2.0 |
| **Deepest vintages, back to 1965** | **Philadelphia Fed RTDSM** — plain XLSX, no key, no wrapper | public |
| World Bank | **`wbgapi`** | MIT — prefer over `wbdata` (**GPL-2.0**) |
| IMF | **`imfp` 2.0** | Apache-2.0 (⚠️ PyPI classifier wrongly says MIT) |
| SDMX (ECB, Eurostat, OECD, BIS, ILO, +36 sources) | **`sdmx1`** — *import as `sdmx`* | Apache-2.0 |
| Eurostat | `eurostat`, or `pandas_datareader` | MIT |
| BLS | ✅ **call the JSON endpoint with `requests`** — the `bls` package is dead (2018, GPL-2.0) | — |
| Broadest aggregator (93 providers) | `dbnomics` | 🚨 **AGPL-3.0** |

**Dead or moved:** 🚨 `pandasdmx` (dead 2023 → `sdmx1`) · 🚨 the old IMF endpoint
`dataservices.imf.org` **no longer resolves at all** (✅ verified; new: `api.imf.org/external/sdmx/3.0/`)
· OECD moved to `sdmx.oecd.org` — the old host 301-redirects but **dataset IDs changed shape**, so
hardcoded queries break even though the host resolves.

**BLS API v2** works **without registration** (✅ verified). Registered: 500 queries/day, 50
series/query, 20 years/query. Unregistered v1: 25 / 25 / 10. Registration is free but **must be
renewed annually**.

## 5. 🚨 Macro revisions — the dominant look-ahead

GDP, payrolls and CPI are revised for years. FRED's own example: 2013Q4 GDP was 17102.5 (2014-01-30)
→ 17080.7 (2014-02-28) → 17089.6 (2014-03-27).

| Source | Vintages? |
|---|---|
| **FRED/ALFRED via `fredapi`** | ✅ per-observation `realtime_start` |
| **Philadelphia Fed RTDSM** | ✅ **the gold standard** — a 244-vintage matrix, 1965Q4→2026Q3 |
| DBnomics | ⚠️ dataset-level release codes only (`WEO:2024-10`), not per-observation |
| **`pandas_datareader` FRED** | ❌ **zero vintage support** — verified by source grep. Not research-grade |
| World Bank / BLS / OECD | ❌ current vintage only |

**Use `realtime_start` as the timestamp, not the observation date.** A January figure is published
in February; indexing it at January is a 1–3 month look-ahead.

🚨 **Three verified bugs in `fredapi` 0.5.2** (from source):
1. **`get_series_as_of_date` does not do what its docstring says** — it returns a DataFrame with
   **duplicate `date` rows** (every revision up to the date), not the latest per date. You must add
   `.groupby('date').last()` yourself.
2. **`realtime_end` is silently dropped** — the parse line is commented out. You cannot directly tell
   when a vintage was superseded; infer it from the next `realtime_start`.
3. **Unbounded downloads** — `get_series_as_of_date` and `get_series_first_release` both call
   `get_series_all_releases()` with no realtime bounds (defaults `1776-07-04` → `9999-12-31`). Cache
   it once rather than calling per-date in a loop.

```python
from fredapi import Fred
fred = Fred(api_key='...')
allr = fred.get_series_all_releases('GDP')            # date | realtime_start | value
pit  = (allr[allr.realtime_start <= '2014-06-30']
        .sort_values('realtime_start')
        .groupby('date')['value'].last())             # REQUIRED — as_of_date does not do this
```

```python
import pandas as pd
url = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
       "real-time-data/data-files/xlsx/routputqvqd.xlsx")
rt = pd.read_excel(url, index_col=0)          # rows = obs period, cols = vintage
gdp_as_known_1999q1 = rt['ROUTPUT99Q1'].dropna()
```

**Two more macro traps:** **seasonal adjustment rewrites already-published history** with no new
information (prefer NSA, or use the vintage current at each decision date); and when mixing
quarterly macro with daily prices, **forward-fill from the release date, never the observation
date**, and never `interpolate()` — interpolation is bidirectional and pulls future values backward.

## ❓ Not verified
`fredapi` claims are from **source inspection, not execution** (no API key available) · which FRED
series actually have ALFRED vintages · whether OECD still publishes its revisions dataset under the
new SDMX structure (**assume no vintage support**) · a general DBnomics vintage-enumeration endpoint
(two paths returned 404) · Bank of England / Riksbank SDMX availability · the `acceptanceDateTime`
timezone is an **empirical inference**, not SEC-documented — which is itself the hazard.

## Per-library deep dives

The optional `fin-libraries` plugin carries a dedicated skill for each library below. Load one
only after this skill has told you which library you want:

- **`lib-edgartools`** — edgartools
- **`lib-fredapi`** — fredapi
