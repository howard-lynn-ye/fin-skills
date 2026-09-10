# Alternative data — sources, breakages, and the decay problem

Verified 2026-09-04. Several of the breakages below were reproduced by installing the packages.

## 🚨 Live-verified breakages in the news/scraping stack

| Package | State |
|---|---|
| 🔴 **`newspaper3k`** | **Dies on `lxml.html.clean`** — the PyPI release is from **2018** and lxml split that module out. Use **`newspaper4k`** |
| 🔴 **`pygooglenews`** | 0.1.3 requires **Python ≥3.12**, so pip backtracks to an older release and fails on a **`use_2to3` build error**. Effectively uninstallable |
| 🔴 **`snscrape`** | Dead by **X/Twitter policy**, not by neglect. There is no maintained replacement at that price point |
| 🚨 **`yfinance.sustainability`** | Returns an **empty DataFrame after a silent 404** — it does not raise. Any ESG pipeline built on it produces empty results that look like "no data" |
| ✅ **`trafilatura`** | The maintained extraction library. Use with `newspaper4k` |

## News APIs

| Source | Terms |
|---|---|
| **NewsAPI** | 🚨 Free tier is **development-only** with **1 month of history**; even the **$1,749/mo** tier caps at **5 years** |
| GDELT | Free, enormous, global. Event coding is noisy and the schema is idiosyncratic |
| Tiingo news | Paid; ticker-tagged and curated — a real differentiator at its price |
| Finnhub news | Included in tiers; coverage varies |
| Benzinga / RavenPack | Institutional, priced accordingly |
| **SEC EDGAR** | ✅ **The only unambiguously clean text source** — public domain, point-in-time, survivorship-free |

⚠️ **News archives are survivorship-biased too.** Outlets delete, paywall and re-slug articles; an
archive assembled today under-represents what was actually published, especially for companies that
later failed — precisely the observations a distress model needs.

## Social

**The WSB result is the cleanest demonstration of alpha decay in the literature.** WallStreetBets
recommendations predicted returns **before GameStop**, and predictability was **completely eliminated
afterwards** (*Review of Financial Studies* **37(5) 1409–1459**). That is a **dated structural break**:
a backtest spanning it will show a strong signal that no longer exists, and the break is invisible in
the aggregate statistics.

`praw` (Reddit) remains usable. StockTwits has an API. X/Twitter is effectively closed to research at
retail prices.

## Satellite, consumer and web

**The satellite parking-lot study is the counter-example worth knowing:** 4.8 M observations across 44
retailers, worth **4–5% around earnings**, and it **survived six years — because it was expensive.**
🔑 **The moat was cost, not cleverness.** That is the general rule for alt data: a dataset stops
working roughly when it becomes affordable.

Credit-card panels, app downloads, web traffic (SimilarWeb) and job postings all follow the same
shape — institutional pricing, short histories, and decay on popularization.

🚨 **SEC v. App Annie ($10M, release 2021-176)** is the record of a major alt-data vendor
**misrepresenting its methodology for four years**. Vendor methodology claims are marketing until
independently checked, and the SEC has enforced on exactly that.

## ESG — the number you should quote

**Berg, Kölbel & Rigobon, *Review of Finance* 26(6) 1315–1344**: ESG ratings from major providers
correlate on average **0.54**, range **0.38–0.71**. Decomposition: **56% measurement, 38% scope,
6% weight**.

**Credit ratings, by comparison, correlate ~0.99.**

🔑 **So "ESG score" is not a measurement — it is a provider's opinion.** A backtest on one provider's
scores does not generalize to another's, and a strategy conditioned on ESG is conditioned on a vendor
choice you must disclose.

## Numerai

`numerapi` is the client. 🔑 **The obfuscation is not paranoia — it exists because of vendor
licensing.** Numerai cannot redistribute the underlying data, so features are unnamed and targets are
factor-neutralized.

🚨 **That makes signals structurally non-transferable.** A model that wins on Numerai's obfuscated,
factor-neutral targets tells you very little about a strategy on real, un-neutralized returns — the
thing you would trade has exactly the factor exposure Numerai removed.

## 🚨 The traps that apply to all alt data

1. **Point-in-time availability.** **Most vendors backfill.** A dataset "starting in 2015" often means
   the *history* was computed in 2020 with 2020's methodology and 2020's entity mapping. Ask for the
   vintage, not the coverage window.
2. **Sample-period limits.** Most alt data starts ~2015. **You cannot test it across 2008**, so you
   have no evidence about how it behaves in a crisis — which is when you would most want it.
3. **Decay on popularization.** See WSB and the satellite study. Alpha in alt data is rented, not owned.
4. **Entity mapping.** Matching a credit-card merchant, an app publisher or a job posting to a
   *tradeable security* is where most of the error lives, and vendors rarely expose their mapping
   confidence.
5. **Licensing kills sharing.** Almost all of it prohibits redistribution and derived-work publication,
   which means **your backtest is not reproducible by anyone else** — including a future you without
   the subscription.
6. 🚨 **`openbb` and all ten of its provider extensions are AGPL-3.0-only** — the easiest path to
   several of these sources carries network copyleft.

## The honest summary

Alt data has produced genuine, documented alpha — the satellite study is real. But every documented
case shares three properties: **it was expensive, it was hard to map to securities, and it decayed
when it got cheap.** A free or cheap alt-data source that appears to predict returns in a backtest is
far more likely to have a point-in-time or entity-mapping bug than an edge.
