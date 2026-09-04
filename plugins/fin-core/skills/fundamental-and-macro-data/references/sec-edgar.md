# SEC EDGAR — APIs, libraries, and the point-in-time rules

All endpoint behaviour below was **verified by live HTTP probe**, not recalled.

## The two hard rules

```python
HDR = {"User-Agent": "Your Company you@example.com"}   # REQUIRED
```
✅ **No User-Agent → HTTP 403.** Any UA → 200 — the header is enforced for *presence*, not content.
Sending a fake browser UA is a policy violation and gets IPs blocked in practice. The SEC's stated
format is `Sample Company Name AdminContact@<domain>.com`.

✅ **Rate limit: 10 requests/second**, quoted from the SEC webmaster FAQ. The penalty is undocumented;
in practice sustained excess yields a temporary IP block.

## Endpoints

| Endpoint | Verified detail |
|---|---|
| `data.sec.gov/submissions/CIK##########.json` | Zero-padded 10-digit CIK. `filings.recent` caps at ~1,000; older paginate via `filings.files[]` (AAPL had 1,244 more in `CIK0000320193-submissions-001.json`) |
| `data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | AAPL = **3.8 MB**, 503 us-gaap tags. Fact fields: `start, end, val, accn, fy, fp, form, filed, frame` |
| `data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json` | Single tag, same fact schema |
| `data.sec.gov/api/xbrl/frames/us-gaap/<Tag>/USD/CY2023.json` | 🚨 **not point-in-time — see below** |
| `efts.sec.gov/LATEST/search-index?q=…&forms=10-K&startdt=&enddt=&from=` | Elasticsearch proxy. ✅ **Coverage starts 2001** (1995 → 2 hits, 2000 → 5, 2001 → 10,000+) — it does **not** reach 1994–2000 filings even though EDGAR holds them |
| `sec.gov/files/company_tickers.json` | 10,412 rows. 🚨 **a current snapshot — survivorship-biased** |
| `sec.gov/files/dera/data/financial-statement-data-sets/YYYYqQ.zip` | ✅ **70 quarterly ZIPs, 2009Q1 → 2026Q2**, ~85 MB each |

🚨 **Frames period syntax:** duration facts use `CY2023` / `CY2023Q1`, but **instantaneous** facts
(Assets, StockholdersEquity) require the **`I` suffix** — `CY2023Q4I`. ✅ Verified:
`Assets/USD/CY2023Q1` → **404**, `CY2023Q4I` → 200. **The #1 frames beginner error.**

## Libraries

| Library | Version (date) | ★ | Licence | Status |
|---|---|---|---|---|
| **edgartools** | **5.56.0** (2026-09-02) | 2,648 | **MIT** | ✅ **the default** — 442 releases, released the day before this audit |
| sec-edgar-downloader | 5.1.0 (2026-02-02) | 717 | MIT | ✅ maintained, narrow scope (downloads raw files only) |
| secedgar | 0.6.0 (2025-05-09) | 1,412 | Apache-2.0 | ⚠️ slowing; last code commit 2025-12-09 |
| sec-parser | 0.58.1 (2024-06-09) | 295 | MIT | ⚠️ **dormant** — last code commit 2025-03-25 |
| 🔴 python-xbrl | 1.1.1 (**2016-12-27**) | 233 | Apache-2.0 | **DEAD — 10 years** |
| datamule | 5.0.2 (2026-07-27) | 555 | MIT | active, 220 releases; freemium CDN mirror (~$1/100k downloads) to bypass SEC rate limits |
| finagg | 2.0.0 (2026-03-22) | 540 | Apache-2.0 | alpha, low cadence |
| **Arelle** (`arelle-release`) | 2.44.7 (2026-09-03) | 232 | **Apache-2.0** (Workiva) | ✅ the only full XBRL **validator** |
| brel-xbrl | 0.8.2a1 (2025-02-26) | 44 | Apache-2.0 (ETH Zurich) | alpha |
| 🚨 edgar-crawler | **not on PyPI** | 545 | 🚨 **GPL-3.0** | GitHub only |
| sec-api (sec-api.io) | 1.0.36 (2026-04-13) | 317 | MIT (SDK) | commercial backend |

⚠️ GitHub reports `NOASSERTION` for Arelle and brel; both LICENSE files were read directly and **both
are genuinely Apache-2.0** (Arelle's just bundles third-party notices after the Apache text).

**`edgartools` is the default:** typed objects for 20+ form types (10-K, 10-Q, 8-K, 13F, Forms 3/4/5,
ADV), XBRL-standardized statements for cross-company comparison, and a built-in MCP server.
**`Arelle` when correctness beats convenience** — calculation and dimension linkbase validation.

## 🚨 Point-in-time: three dates, only one is availability

- **`period` / `reportDate`** — fiscal period end. **The data does not exist yet.** A 30–90 day
  look-ahead. The most common fundamental-data bug there is.
- **`filed` / `filingDate`** — the date EDGAR assigns after its cutoff (**17:30 ET** for periodic
  reports; **22:00 ET for Forms 3/4/5**). Safe-ish, but still admits a 90-minute post-close leak.
- **`acceptanceDateTime`** — the true wall-clock moment. **Use this.**

🚨 **Verified, undocumented timezone inconsistency:** the Submissions API's `acceptanceDateTime` is
**genuine UTC**; the Financial Statement Data Sets' `sub.txt.accepted` is **Eastern Time**.

The test: for filings stamped 18:00–21:59, Submissions shows **454 same-day / 4 next-day**
`filingDate` (consistent with UTC), while FSDS shows **0 same-day / 370 next-day** (consistent with
ET; its hour histogram peaks at 16:00 with 2,310 filings). **Comparing the two without converting
introduces a silent 4–5 hour error.**

Worked example: Apple's 8-K accepted `2026-07-30T20:30:28Z` = **16:30 ET** — 30 minutes *after* the
close, but stamped `filingDate 2026-07-30`. Apple's earnings 8-Ks cluster at 20:30Z/21:30Z, i.e.
**systematically post-close.**

**Rule:** convert to exchange local; if ≥16:00, the earliest tradeable bar is the **next session's open.**

## ✅ companyfacts IS point-in-time — the bug is in how people use it

It returns **every vintage**, one row per accession that ever reported the period, each with `filed`:

```python
# WRONG — silently selects the restated value
df.drop_duplicates(subset=['start', 'end'], keep='last')

# RIGHT — latest vintage KNOWN AS OF the decision date
df[df.filed <= as_of].sort_values('filed').groupby(['start', 'end']).last()
```

**Measured on Apple alone: 408 `(start, end, form)` groups have differing values across vintages.**
`AccountsPayableCurrent` FY2017: 49,049M → 44,242M. `AntidilutiveSecurities` FY2019: 15.5M → 62.0M —
that one is the 2020 4-for-1 split retroactively rewriting share counts. 🚨 **Every per-share metric
in XBRL is silently split-adjusted backwards**, so a PIT EPS study compares split-adjusted
denominators against unadjusted prices unless you handle it.

## 🚨 frames is not point-in-time and cannot be made so

Its records carry **no `filed` and no `form`** — you cannot filter by vintage. Measured on the CY2023
annual-revenue frame by extracting the filing year from each `accn`:

| Filing year of the value frames chose | Count |
|---|---|
| 2023 | **18 (0.6%)** |
| 2024 | 406 |
| 2025 | 1,044 |
| **2026** | **1,673 (53%)** |

**Only 18 of 3,141 values come from a filing actually made in 2023.** Cross-checked from the other
direction: in companyconcept the `"frame":"CY2023"` tag is attached to the **2025-filed** version of
Apple's FY2023 revenue. **Never use frames for backtesting.**

## ✅ The PIT-safe bulk path

**Financial Statement Data Sets** — each quarterly ZIP contains only filings *made during that
quarter* (2026Q1's `filed` range is 20260102–20260331), so stacking them in order gives true
point-in-time with no filtering logic. Four TSVs: `sub` (36 cols), `num`, `pre`, `tag`.
`sub.txt` carries `prevrpt` (superseded by a later amendment — 1 of 6,169 in 2026Q1) and `detail`;
`num.txt` has `ddate`, `qtrs` (0 = instantaneous, 1 = quarterly, 4 = annual), `segments`, `coreg`.

## Identity and coverage traps

- ✅ **EDGAR itself is survivorship-bias-free**: Lehman Brothers Holdings (CIK 806085) **6,341
  filings, latest 2025-08-28**; Bear Stearns (777001) 3,344; Enron (1024401) 351.
  🚨 **But `company_tickers.json` is not** — all three return `tickers: []`. **Building a universe
  from it re-introduces the exact bias EDGAR would have let you avoid.**
- **1,452 of 8,005 CIKs (18%) map to more than one ticker.** `GOOGL → [GOOGL, GOOG, GOOGM, GOOGN]`;
  BAC has **17**; JPM's 9 include **preferred shares** (`JPM-PC`) and **ETNs** (`VYLD`, `AMJB`).
  A naive ticker→CIK join attaches JPMorgan's income statement to a note. Filter via
  `company_tickers_exchange.json` and drop `-P*` suffixes.
- 🚨 **CIKs are not stable entities.** CIK **933136**: `WASHINGTON MUTUAL INC` (1995–2006) → failed in
  the largest US bank failure → `WMI HOLDINGS` bankruptcy shell → `WMIH CORP` → reverse-merged into
  **Mr. Cooper Group** (a mortgage servicer, COOP) → acquired by Rocket 2025. `companyfacts` returns
  **one continuous series** for all of it. **Check `formerNames`; treat a SIC change as a discontinuity.**
- **XBRL coverage starts ~2009.** AAPL's earliest `filed` is 2009-07-22. Lehman, Bear Stearns and
  Enron return **404** on companyfacts — they died before the mandate. **Any XBRL study is
  structurally post-2009 and cannot include the GFC's casualties.**
- **Amendments and late filers** (2025 volumes): 10-K/A 3,073 · 10-Q/A 1,238 · **NT 10-K 746** ·
  NT 10-Q 1,171. An `NT 10-K` (Form 12b-25) announces a late filing — itself a tradeable negative
  signal, and it means the row **will not exist on schedule**. Handle absence explicitly; do not
  forward-fill.
- **Fiscal-year misalignment:** 95.8% of 10-Ks use `fye=1231`; 4.2% do not, and `fy` is a *label* —
  Target's `period=20260131` carries `fy=2025`. **Join on `period`, bucket to calendar quarters
  yourself.** `fye` is sometimes internally inconsistent (`ARMADA ACQUISITION`: `fye=0930` but
  `period=20251231`) — do not trust it blindly.

## PIT reference implementation

```python
import requests, pandas as pd
HDR = {"User-Agent": "Your Company you@example.com"}

def pit_facts(cik: int, tag: str, as_of: str, taxonomy="us-gaap"):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
    facts = requests.get(url, headers=HDR, timeout=30).json()["units"]["USD"]
    df = pd.DataFrame(facts)
    df["filed"] = pd.to_datetime(df["filed"])
    df = df[df["filed"] <= pd.Timestamp(as_of)].sort_values("filed")
    return df.groupby(["start", "end"], as_index=False).last()
# <=10 req/s. Never use /api/xbrl/frames/ for this.
```
