---
name: choosing-a-data-vendor
description: >-
  Decide whether a data source may legally and factually serve a research question, before any
  fetch code is written. TRIGGER - need delisted, survivorship-free or point-in-time data and are
  choosing where to get it; may I store, cache, redistribute or publish what I fetched; is the
  free tier enough and what does a key cost; comparing vendor terms, licences, rate limits or
  paid tiers; "there are no delisted names on the free tier"; picking between yfinance, Tiingo,
  Alpha Vantage, stooq, EODHD, Norgate, CRSP or Polygon. Automated by `python -m fin_skills.data
  advise`. SKIP once the source is chosen and the question is how to CALL it - adjustment,
  timezone, calendar and off-by-one traps (market-data-sourcing); storing, partitioning or as-of
  joining data you already hold (market-data-engineering); EDGAR, XBRL and macro vintages
  (fundamental-and-macro-data); A-share sources and 退市 lists (china-ashare-data); exchange OHLCV
  and venue limits (crypto-data-and-execution).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Choosing a data vendor

Most of the effort in a data decision goes to the wrong axis. Price, coverage breadth and which
Python client is nicest are all tiebreaks. Three flags decide it, and two of them are decided by a
vendor's **terms**, not by its API — so no amount of code changes the answer.

| Flag | The question it answers | What it costs you to get it wrong |
|---|---|---|
| `includes_delisted` | Are the companies that DIED in this data? | Every result is an upper bound, silently |
| `point_in_time` | Can it say what was known at *t*? | You trade on numbers published years later |
| `redistribution` | May you pass on what you fetched? | A licence breach you discover at publication |

Run the filters **before** picking a library. A source that lacks the dead names, cannot answer
"what was known at *t*", or may not be passed on is not fixable downstream.

## 1. Ask, don't guess

```bash
python -m fin_skills.data advise --market US --frequency 1d --history-years 10
python -m fin_skills.data advise --market US --delisted        # exits 3: nothing serves it
python -m fin_skills.data advise --asset-class crypto --market CRYPTO --no-key
python -m fin_skills.data advise --coverage                    # what each source reaches
```

It reads the same `Declaration` table `python -m fin_skills.data adapters` prints, applies your
constraints as **hard filters**, and either ranks what survives — each with what it costs you,
what it *cannot* do, and the licence consequence of your stated intent — or returns nothing and
names the flag that emptied the list. It **exits 3** when nothing serves the request, so a
pipeline can branch on the dead end instead of grepping for it.

## 2. The free tier for delisted US equities is empty

📐 Measured by `scripts/vendor_decision.py`, over the eight adapters in `fin_skills.data`:

- of 8 sources, **1** includes delisted names, **2** are point-in-time, **2** permit any
  redistribution at all;
- of the **6** that serve price bars, **0** include delisted names;
- `advise --delisted` for US equity dailies refuses **4** sources on `includes_delisted=False` —
  yfinance, tiingo, alphavantage, stooq — and returns an empty recommendation.

The one that *does* include delisted names is **EDGAR**, and it serves filings, not prices. That
asymmetry is the whole shape of the problem: a fundamentals study can be made survivorship-free
for free; a price panel cannot.

**Two free half-answers, and what each does not fix.**

- ✅ **Alpha Vantage `LISTING_STATUS`** returns "a list of active or delisted US stocks and ETFs,
  either as of the latest trading day or at a specific time in history", for "any YYYY-MM-DD date
  later than 2010-01-01", on a free key
  ([alphavantage.co/documentation](https://www.alphavantage.co/documentation/), 2026-09-10). That
  is the **membership** half of survivorship — `get("alphavantage").universe("US", "2014-07-10",
  include_delisted=True)`. It names the dead; it does not price them.
- ✅ **EDGAR** keeps the filings of companies that stopped existing, so a point-in-time
  fundamentals panel is survivorship-free by construction.

**What it costs to fix the price half** (each checked at the vendor's own page on 2026-09-10):

| Vendor | What unlocks it | Price |
|---|---|---|
| ✅ **EODHD** | "Delisted Data" first appears on **ALL-IN-ONE**; the $19.99 EOD plan does not carry it, and the free tier is 20 calls/day | $99.99/month |
| ✅ **Norgate Data** | US Stocks **Platinum** (to 1990) and **Diamond** (to 1950) list "Delisted securities" and "Historical index constituents", and are the only tiers the page itself labels "Suitable for backtesting". Silver and Gold are "Current major-exchange-listed securities" | quoted by calculator |
| ⚠️ **CRSP** (via `wrds`) | the academic reference, with delisting returns and historical index membership | institutional; no public price |

If you cannot buy it, the honest move is not to substitute the closest free thing. It is to write
**"upper bound"** on the result — `fin_skills.api.check()`'s `survivorship_audit` will say so for
you, and `research-integrity-guards` §1 is the paragraph to put in the write-up.

## 3. The code licence is not the data licence

Every free price source here ships permissive **code** and prohibits redistribution of the
**data**. Reading only the PyPI classifier gets this backwards every time.

| Source | Client licence | Data terms | Verified |
|---|---|---|---|
| yfinance | Apache-2.0 | redistribution prohibited; Yahoo's own terms describe personal use | ✅ 2026-09-09 |
| akshare | MIT | 🚨 its documentation restricts the **data** to academic research | ✅ 2026-09-09 |
| Tiingo | MIT (`tiingo`) | every plan — free, $30 Power, $50 Commercial — is "Internal Use Only". Redistribution is a separate plan at $250/month (startups) or $500/month (enterprise), and requires the phrase "Data sourced by Tiingo" | ✅ 2026-09-10 |
| Alpha Vantage | MIT (`alpha-vantage`) | ToS §2a: personal, non-commercial; providing access to others is "commercial use" | ✅ 2026-09-10 |
| stooq | Apache-2.0 (`requests`) | terms 5.3: "Redistribution of data found on the website is not allowed without the consent of Stooq"; the download page adds "This data is intended solely for personal use" | ✅ 2026-09-10 |
| FRED | Apache-2.0 (repo LICENSE — 🚨 **PyPI declares none**) | attribution required verbatim; third-party copyright-flagged series may not be redistributed | ✅ 2026-09-09 |
| SEC EDGAR | MIT (`edgartools`) | US government works — public domain | ✅ 2026-09-09 |

🚨 **Two terms restrict what your own code may do, not just what you may publish.**

- **Tiingo forbids keeping it.** ToS 1.6(a), Starter and Trial plans: *"You may not write, save,
  archive, back up, or otherwise retain Tiingo Data in any persistent or durable storage"*, and
  data may be processed *"only transiently in volatile memory or in a temporary, non-persistent
  cache"*. `fin_skills.data.Cache.put()` refuses on it; `advise --store` refuses to recommend it.
- **Polygon/Massive prohibits non-display use**, which is arguably what a backtest is. The
  `non_display_use` field exists to surface that; this library does not opine on whether your use
  qualifies.

## 4. Free is a shape, not a number

The free tiers are not the same *kind* of limit, so "N requests" is not comparable across them.
All ✅ verified at each vendor's own page on the date shown.

| Source | Free tier | Shape |
|---|---|---|
| Alpha Vantage | **25 requests/DAY**, "and unlimited API requests for verified open-source or educational projects". 🚨 A free daily call returns `outputsize=compact` — "only the latest **100 data points**"; `full` (25+ years) is premium, as is `TIME_SERIES_DAILY_ADJUSTED`, so the free daily series is also **unadjusted** | flat daily (2026-09-10) |
| Tiingo | 50/hour, 1,000/day, **500 unique SYMBOLS/month**, 1 GB/month. 🚨 The symbol counter is the one that bites: a 600-name universe cannot be fetched in one calendar month at any request rate, and no response field says how close you are | nested + non-request (2026-09-10) |
| stooq | no key, no account | 🔴 see below (2026-09-10) |
| yfinance, akshare | **no published number exists** | not "unlimited" (2026-09-09) |
| FRED | two published shapes of one magnitude — "120 requests per minute" (v1 errors page) and "2 requests per second" (v2). Pacing at 2/s satisfies both | contradictory (2026-09-09) |
| SEC EDGAR | 10 requests/second *"regardless of the number of machines used to submit requests"*, then a 10-minute wall | per user (2026-09-09) |

**"No published limit" is not "no limit".** `fin_skills.data.ratelimit.Unpublished` is the honest
shape for yfinance, akshare, FRED and stooq: it paces from `X-RateLimit-*` headers and backs off
on 429, and supplies no number of its own. Any "N per hour is safe" figure you find for Yahoo is
folklore.

## 5. 🔴 The keyless default is currently unreachable

stooq is the source everyone reaches for when they want daily equity bars with no key and no
account, and two thirds of that reputation survives checking: there is genuinely no key, and
`AAPL.US` has daily bars in 1984. The third does not.

✅ Verified 2026-09-10: **every** `stooq.com` and `stooq.pl` URL — the CSV endpoint, the quote
page, `/db/h/` — answered **HTTP 200** with an identical **796-byte JavaScript proof-of-work
page** ("This site requires JavaScript to verify your browser") instead of the data. A 200 with an
HTML body is the worst failure shape there is, because `pd.read_csv(url)` does not raise on it: it
parses the challenge markup and returns a DataFrame.

🔴 And there is no client library left. `pandas_datareader.stooq` was the ecosystem's reader;
**pandas-datareader 0.11.0/0.11.1** (2026-06-23/24, the first releases since 0.10.0 of 2021-07-13)
narrowed the package to "macroeconomic, policy, and factor-style data sources". Verified on the
installed 0.11.1: the shipped modules are `bankofcanada, econdb, eurostat, famafrench, fred,
macro, oecd, wb`, and `DataReader(..., "stooq")` raises `NotImplementedError`.

`fin_skills.data`'s stooq adapter checks the body before parsing and raises with the reason; the
advisor refuses it rather than ranking it. Neither ships a way around the challenge.

## 6. Ranking, once three flags have already decided

Among sources that survive the filters, `advise` orders by stated criteria, not by preference:
point-in-time, then survivorship, then **whether a cached copy stays valid**, then whether raw
prices and factors arrive together, then whether a key is needed, then what the terms let you
keep.

The third one is the surprise. A series anchored at the **present** (Yahoo `auto_adjust=True`,
A-share qfq, stooq's default) is rewritten by every new dividend, so the same query next month
returns different numbers and a stored copy silently disagrees with a fresh pull. ✅ Tiingo is the
only source in this survey that returns raw OHLCV, adjusted OHLCV, `divCash` and `splitFactor` in
the same row (2026-09-10), which is why `advise` ranks it above yfinance for a 10-year US daily
panel even though it needs a key. Detail: `../research-integrity-guards/references/adjustment-conventions.md`.

## 7. Reproduce every number here

```bash
python plugins/fin-core/skills/choosing-a-data-vendor/scripts/vendor_decision.py
python -m fin_skills.data advise --market US --delisted   # the empty answer, in full
python -m fin_skills.data adapters --name tiingo          # one declaration, dated
```

`scripts/vendor_decision.py` carries a dated snapshot of the three flags per source so it runs
offline on a bare install, and cross-checks that snapshot against the live `Declaration` objects
whenever `fin_skills` is importable — so the table in §2 cannot drift from the code it came from.

## Where this sits

This skill answers *may I, and can I, use this source at all*. Once the answer is yes:

- **`../market-data-sourcing/SKILL.md`** — how to CALL the library you chose, and the adjustment,
  timezone, `end`-exclusive and calendar traps that corrupt numbers silently.
- **`../market-data-engineering/SKILL.md`** — storing, partitioning and as-of joining data you
  already hold.
- **`../fundamental-and-macro-data/SKILL.md`** — EDGAR, XBRL, CIK mapping and ALFRED vintages.
- **`../research-integrity-guards/SKILL.md`** — §1 is the survivorship paragraph to put in a
  write-up when the answer above was "buy it or label it an upper bound".
- **`../../../fin-china/skills/china-ashare-data/SKILL.md`** — A-share sources and delisting lists.
- **`../../../fin-crypto/skills/crypto-data-and-execution/SKILL.md`** — venue OHLCV and per-venue
  limits.
