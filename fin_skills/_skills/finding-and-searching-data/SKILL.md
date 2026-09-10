---
name: finding-and-searching-data
description: >-
  Find the identifier before you fetch it, and know which free sources can actually search.
  TRIGGER - "what is the FRED series id for X", "which ticker is this company", "find the CIK
  for", searching for a series, ticker, symbol, contract or filing by name or phrase;
  fred/series/search, company_tickers.json, EDGAR full-text search, efts.sec.gov, ccxt
  load_markets; a guessed identifier returning an empty frame; retrieving filings by form and
  date range; when a document became PUBLIC versus when it was filed; earnings-call transcript
  timestamps. Load BEFORE guessing an identifier - three of the four free searches are current
  snapshots and cannot answer as of a past date. SKIP for resolving or dating an identifier you
  already hold (security-master-and-symbology), for choosing a price vendor
  (market-data-sourcing), for the point-in-time join once the data is yours
  (fundamental-and-macro-data), for auditing a finished result (research-integrity-guards), and
  for crypto venue mechanics (crypto-data-and-execution).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Finding and searching data

Every data client in this ecosystem fetches **given an identifier**. `yf.download("AAPL")`,
`fred.get_series("DGS10")`, `Company("0000320193")` — each needs the string already in hand.
`fin_skills.data`'s Adapter protocol is the same shape: `bars / fundamentals / macro / actions /
universe`, and none of them takes a name.

So the first question of any real task is *what is the identifier*, and the failure mode is
specific: **a wrong identifier does not raise.** `DGS10` exists and `DGS10YR` does not, `ES=F`
works on Yahoo and `ES` does not, and the wrong one comes back as an empty frame that is
indistinguishable from "no data" the moment it lands in a file. Guessing is worse than stopping.

## 1. What each free source can search — and cannot

✅ Verified 2026-09-10 against each vendor's documentation or a live call. `scripts/find_data_source.py`
carries this table as data, plus `where_to_search(need)`.

| Source | Free text is matched against | Returns | As of a past date? |
|---|---|---|---|
| `api.stlouisfed.org/fred/series/search` | series **title**, units, frequency, tags — stemmed | series id | ✅ `realtime_start`/`realtime_end` |
| `sec.gov/files/company_tickers.json` | ticker and company title, locally after one download | ticker, CIK | 🚨 **NO** — no date field exists |
| `efts.sec.gov/LATEST/search-index` | the full text of filings **and their exhibits**, 2001+ | accession, CIK | ⚠️ on `file_date` only |
| `ccxt` `load_markets()` | unified symbol, base, quote, venue market id | market symbol | 🚨 **NO** — today's listing |

**Three of the four cannot answer a historical question at all.** Anything they return is a
current snapshot however you label it — the same survivorship axis `market-data-sourcing` covers
for prices, applied to discovery. `fin_skills.discovery.search()` records that as a warning naming
the source rather than letting the label pass.

**Nothing free searches these**, and saying so is the honest answer: a delisted company by name, a
futures contract code, the ticker an identifier carried in the past, or a security master keyed on
dates. Those are paid (FIGI/OpenFIGI covers part of it, CRSP and Bloomberg the rest).

### FRED — the default search mode does not match the id
🚨 `search_type` defaults to `full_text`, documented as matching *"series attributes title, units,
frequency, and tags by parsing words into stems"*. **The series id is not in that list.** Searching
`DGS10` in the default mode can miss the series named `DGS10`. Pass `search_type="series_id"` for
substring matching with `*` wildcards. ✅ `limit` defaults to 1000 and is capped at 1000;
`order_by` defaults to `search_rank`; `realtime_start`/`realtime_end` default to **today**, so the
default answer is which series exist *now* (fred/series/search docs, read 2026-09-10).

⚠️ The key is required, and FRED's own API-key page says *all users of an application shall use
their own*. No library may ship or proxy one. The rate limit has **no single published shape** —
the v1 errors page says 120 requests/minute, the v2 page says 2/second — so pace at 2/s, which
satisfies both.

### SEC EDGAR — one rate limit, two very different endpoints
✅ **10 requests per second, regardless of the number of machines used** (sec.gov Internet Security
Policy), with a ten-minute block after a breach. Parallelism buys nothing. A declared `User-Agent`
is required on both endpoints or every request 403s; the SEC checks the header for *presence*, not
content, so a fake browser UA is a policy violation, not a workaround.

🚨 **A full-text hit is a DOCUMENT, not a filing.** ✅ Measured 2026-09-10: `q="supply chain
finance"&forms=10-K` returns **707 hits**, and the top one is
`0001739566-24-000054:a20231110utzinsidertrading.htm` — Utz Brands' **EX-19 insider-trading
policy**, an exhibit attached to a 10-K. Its `form` is still `10-K`. Filtering on `form` alone
counts exhibit text as 10-K text; read `file_type`. The hit id is `<accession>:<document>`, so two
hits can be one filing.

🚨 **A full-text hit carries `file_date` and no `acceptanceDateTime` at all.** It is therefore not
point-in-time as returned — see §3.

### ccxt — the market map belongs to one instance
✅ Read in ccxt's own source (`python/ccxt/base/exchange.py`, master at `__version__ = '4.5.78'`,
2026-09-10): `load_markets(reload=False)` returns `self.markets` **with no request** when they are
already loaded; otherwise it calls `fetch_currencies()` **and** `fetch_markets()` — up to two calls
through the instance's leaky bucket. Hold one instance per venue; a fresh `ccxt.binance()` per call
resets the bucket and is how accounts get banned (`crypto-data-and-execution` owns the rest).

✅ `MarketInterface` (`python/ccxt/base/types.py`, same read) carries 42 fields, including ones a
2024-era memory does not have: `subType`, `index`, `stock`, `prediction`, `quanto`, `created`,
`marginModes`, `outcomes`. 🚨 A delisted pair is **absent** from the map, not flagged
`active: False`, so a pair list built from it is survivorship-biased.

## 2. Using it

```python
from fin_skills.discovery import SearchQuery, search, describe

print(describe())                       # every source: what it searches, and what it cannot
report = search(SearchQuery("10-year treasury constant maturity", asset_class="macro"))
print(report.summary())                 # results, plus every source that could NOT run and why
```

`search()` **never raises because a backend is unavailable.** A missing key, a missing library and
a refused socket all land in `report.skipped` with the reason, so the same call works with every
backend absent and the report says so instead of looking empty. Transport is injected
(`transport=(url, headers) -> bytes`), which is how the whole layer is tested offline.

A key is read from the environment variable its capability names, at the moment of the call. No URL
carrying one is stored, returned, or put in a message.

## 3. 🚨 A document has three dates and they are not interchangeable

| Field | What it is | What it costs you |
|---|---|---|
| `period` / `reportDate` | the period the document **covers** | it does not exist yet — a 30-90 day look-ahead |
| `filingDate` | the date the SEC **stamps**, by its own cutoff rules | wrong in **both** directions |
| `acceptanceDateTime` | the wall-clock instant it became **public** | the truth — the only instant of the three |

✅ **Measured 2026-09-10** over 32,686 filings from 80 randomly chosen CIKs (seed 0) in
`company_tickers.json`, via `data.sec.gov/submissions/`:

- **55.70%** were accepted at or after **16:00 ET**.
- **51.17%** were accepted post-close **and still carry that same day as `filingDate`.** Treat
  `filingDate` as "knowable at that day's close" and more than half your document set is a
  same-day look-ahead.
- **5.40%** have an acceptance date (ET) that differs from `filingDate` outright.
- The acceptance hour histogram peaks at **16:00 ET (8,725 filings)** and **17:00 (4,935)** — the
  17:30 ET cutoff for periodic reports is visible in the data.
- Over the 1,836 10-K/10-Q in that sample, `period_end -> filingDate` is a median of **43 days**
  (10-K 73, 10-Q 40), p90 79, max 523.

✅ Both directions, on Apple's own filings (verbatim in `scripts/find_data_source.py`):

| Accession | Accepted (ET) | `filingDate` | Tradeable |
|---|---|---|---|
| `0000320193-25-000077` (8-K) | 2025-10-30 **16:30:35** | 2025-10-30 | 2025-10-31 |
| `0000320193-24-000081` (10-Q) | 2024-08-01 **18:03:34** | 2024-08-**02** | 2024-08-02 |

One field, two opposite errors: the 8-K is stamped a day too early, the 10-Q a day too late. ✅ Over
Apple's 1,000 most recent filings, **896 (89.6%)** were accepted at or after 16:00 ET and **844 of
those** carry that same day as `filingDate`.

⚠️ The two SEC APIs disagree about the timezone of that stamp — the Submissions API's
`acceptanceDateTime` is UTC, the Financial Statement Data Sets' `sub.txt.accepted` is Eastern
(`research-integrity-guards` measured it). Declare the source zone; never assume it.

**The rule:** convert acceptance to exchange-local; at or after the 16:00 close, the earliest
tradeable bar is the **next session's open**. That derived column is the only one a backtest may
join on.

```python
from fin_skills.discovery import EdgarDocuments
edgar = EdgarDocuments(identity="Your Name you@example.com")
docs = edgar.filings(320193, forms=["10-Q", "10-K"], start="2024-01-01")
docs.get("0000320193-25-000077", as_of="2025-10-30")   # LookAheadError: not public yet
docs.get("0000320193-25-000077", as_of="2025-10-31")   # fine
```

A full-text hit is **provisional** — no acceptance stamp exists in the response — so `get()` on one
raises until `edgar.enrich(hits)` fetches the acceptance instants, or you pass
`allow_provisional=True` and record that you accepted `filed_at` as a proxy.

## 4. 🚨 Earnings-call transcripts: the same trap, and no free source

There is no free, licensable, point-in-time transcript source, so this library ships no backend for
one. That is not the interesting part. The interesting part is what every vendor's schema does:

**the `date` field on a transcript is almost always the date of the CALL, not the date the
transcript was published.** Publication lags by hours for an automated feed and by days for a
human-reviewed one. An NLP feature computed from the text and stamped on the vendor's `date` is a
**same-day look-ahead**: it trades the sentiment of a call using words that did not exist until
after the close, and often not until the following session.

The audio is public at the moment of the call. The text is not. Until you have a publication
timestamp from your vendor, treat a transcript's availability as **unknown**, which in this library
means the feature cannot be used point-in-time at all. The measurement to run on your own vendor is
the same shape as §3: take their `date`, find the publication timestamp, report the distribution of
the gap. `fin_skills.discovery.documents.TRANSCRIPTS` carries this text, and
`EdgarDocuments.transcripts()` raises it rather than returning a stub.

## 5. Checklist before you fetch anything

1. **Do you have the identifier, or did you produce it?** If a model produced it, search for it.
2. **Which source can search this at all?** `where_to_search(need)` returns `()` for the needs
   nothing free serves, and `()` is a real answer.
3. **Does the source support an as-of date?** Three of four do not. If it does not, whatever you
   build from it is a current snapshot — say so in the manifest.
4. **For a document, which of the three dates did you get?** If it is not an acceptance instant,
   the document is provisional and cannot carry a point-in-time feature.
5. **Then hand the identifier to `security-master-and-symbology`** to find out whether it meant the
   same thing on your backtest dates.

## Scripts

- `scripts/find_data_source.py` — the source table with every `cannot`, the need-to-source routing
  (including the needs that route nowhere), and `document_dates()` / `lookahead_summary()` over
  Apple's real filing metadata. Offline, no writes.

## ❓ Not verified

The FRED endpoint's parameters come from its documentation page, **not** from a live call — no API
key was available, so the response shape is unexercised against the server. The ccxt claims come
from reading the library's source on GitHub, not from running it. The 10,000-hit saturation of
EDGAR full-text search is inferred from the `hits.total.value` ceiling observed on broad queries,
not from SEC documentation. Whether any transcript vendor exposes a publication timestamp at all is
unverified — every claim in §4 about vendor `date` fields is ⚠️ secondhand and is exactly the thing
to measure on your own feed before trusting it.
