# tiingo · twelvedata · finnhub-python

The three mid-tier vendors people reach for when yfinance starts 429-ing. All three have real free
tiers, all three have a **licence or quota clause that is not the number on the pricing page**, and
in every case the binding constraint is the one nobody reads.

| | `tiingo` | `twelvedata` | `finnhub-python` |
|---|---|---|---|
| version | **0.16.1 (2025-04-05)** ✅ | **1.4.0 (2026-04-27)** ✅ | **2.4.29 (2026-06-24)** ✅ |
| releases | 25 | 45 | 53 |
| GitHub | `hydrosquall/tiingo-python` **318★**, 35 open, pushed 2025-12-14 | `twelvedata/twelvedata-python` **772★**, 3 open, pushed 2026-08-12 | `Finnhub-Stock-API/finnhub-python` **1,074★**, 42 open, pushed 2026-06-24 |
| Licence | **MIT** ✅ | **MIT** ✅ | **Apache-2.0** ✅ |
| Python | `requires_python` **null** ⚠️ | `>=2.7` with 3.0–3.4 excluded ⚠️ — meaningless | `requires_python` **null** ⚠️ |
| Verdict | ⚠️ **client aging** — no release in 17 months; the *service* is fine | ✅ healthy | ✅ healthy client, ⚠️ stale README |

Verified 2026-09-04 via the PyPI JSON API and the GitHub REST API. All three clients are thin
`requests` wrappers — client health says little about vendor health, and vice versa.

## 🚨 Traps

🚨 **Tiingo's ToS is the trap, not the rate limit.** Verbatim: *"you may only use the data for your
own personal use and you may not display or share the data with another person or organization."*
**Internal use only — you may not display or redistribute.** That rules out a public dashboard, a
newsletter chart, a client report, or a shared Streamlit app, regardless of what you paid.

🚨 **Tiingo's binding free limit is 500 unique symbols per month**, not the request count ⚠️.
The published free tier is 50 requests/hour, 1,000/day, **500 unique symbols/month**, 1 GB/month.
A single loop over a 500-name universe exhausts the *month* on day one while using 500 of your
30,000 monthly requests. Budget symbols, not calls.

🚨 **Twelve Data bills in CREDITS, and credits ≠ requests.** Free tier is **8 credits/minute, 800
credits/day, 3 exchanges** ⚠️. Endpoint cost varies — a batch or a technical-indicator call debits
more than one credit — so "8 requests a minute" is wrong in the expensive direction. The client
exposes `td.api_usage()` ✅ (*"gives an overview of the current API credits consumption"*); poll it
rather than counting your own calls.

🚨 **Twelve Data's free tier covers 3 exchanges.** Not 3 *countries*. A ticker outside them returns
an error or an empty payload, not a clearly-labelled entitlement failure — which reads as "bad
symbol" in a loop and silently drops names from your universe.

🚨 **Finnhub's documented global cap is 30 API calls/second** ✅ — that is an *anti-burst* ceiling
that applies on every plan, not a free-tier quota. ❓ **The per-minute free-tier limit could not be
verified**; Finnhub's pricing page is JS-rendered and did not yield to a fetch. The widely repeated
**"60 calls/minute" figure is unconfirmed** — do not encode it as a constant. Rate-limit by
observing 429s, not by trusting a remembered number.

🚨 **The first example in finnhub-python's README 403s on a free key.** The headline snippet is
`finnhub_client.stock_candles('AAPL', 'D', 1590988249, 1591852249)` — OHLCV candles are a **premium
endpoint**. New users conclude their key is broken. The README also declares *"Package version:
2.4.25"* while PyPI ships **2.4.29** ✅ — it is not tracking releases, so treat its endpoint list as
indicative rather than as an entitlement map.

⚠️ **Finnhub free company news is retention-limited (~1 year)** and redistribution is prohibited.
`company_news` and `quote` are the reliably-free equity endpoints; anything candle- or
fundamentals-shaped needs checking against your plan.

⚠️ **Delisted coverage is undocumented for all three** ❓. Tiingo's ticker metadata carries an
`endDate` field, but free-tier history retention for dead names is unverified. Twelve Data and
Finnhub document nothing. **Do not build a backtest universe from any of them** — see
`_decision-table.md`; EODHD (`eodhd.md`) is the cheap bias-free option.

## Choosing between them

| Need | Pick | Why |
|---|---|---|
| Long daily history (30+ yrs), clean adjustments, personal research | **Tiingo** | Best history depth per free dollar — but internal use only |
| Intraday bars + technical indicators computed server-side, non-US exchanges | **Twelve Data** | Widest endpoint surface; watch the credit arithmetic |
| Company news, filings metadata, estimates, alt-ish data on a free key | **Finnhub** | Broadest *non-price* free surface; candles are paid |
| Anything redistributed, displayed publicly, or backtested on a survivorship-free universe | **none of them** | See `eodhd.md` / `databento.md` |

## Minimal correct calls

```python
import os
# Tiingo — internal use only; count UNIQUE SYMBOLS, not requests
from tiingo import TiingoClient
tiingo = TiingoClient({"api_key": os.environ["TIINGO_API_KEY"], "session": True})
px = tiingo.get_dataframe("AAPL", startDate="2020-01-01", endDate="2024-01-01", frequency="daily")

# Twelve Data — check credits, do not assume 1 request == 1 credit
from twelvedata import TDClient
td = TDClient(apikey=os.environ["TWELVEDATA_API_KEY"])
print(td.api_usage())                               # 🚨 authoritative credit counter
bars = td.time_series(symbol="AAPL", interval="1day", outputsize=5000).as_pandas()

# Finnhub — quote is free; stock_candles (the README's headline) is NOT
import finnhub
fh = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"])
print(fh.quote("AAPL"))                             # free
print(fh.company_news("AAPL", _from="2024-06-01", to="2024-06-10"))   # note `_from`, not `from`
```

`_from` instead of `from` in every Finnhub date-ranged call — `from` is a Python keyword.

## Cross-references

`_decision-table.md` — free-tier limits and delisted coverage for every vendor, side by side ·
`yfinance.md` — the free baseline and its 429 mitigations · `eodhd.md` — the paid bias-free
universe · `openbb.md` — 🚨 an AGPL-3.0 abstraction over these same keys; it does not change their
quotas or their redistribution terms.
