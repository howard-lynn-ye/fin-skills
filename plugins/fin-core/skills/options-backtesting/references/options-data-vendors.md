# Historical options data — vendors, verified

What each source actually gives you and what it costs, as read from the vendor's own page on
**2026-09-05**. Prices move; the date matters more than the number. Where a figure came from a
third-party page it is marked ⚠️.

🚨 **No free live API serves expired contracts.** Yahoo's `date` parameter selects a *currently
listed* expiry, not an as-of date; Tradier documents that history is unavailable for expired
options; the IBKR TWS API returns no bars for them. There is no path from a free live API to an
options backtest. Everything below is about which paid or archival source to use.

## Read in a browser from the vendor page

### ThetaData — `thetadata.net/pricing`

Three tiers, individual use, monthly. History depth is **by tier**, which is the fact most third-party
pages get wrong.

| Tier | Price | History | Notes |
|---|---|---|---|
| Options Value | **$40/mo** | 4 years | 1-minute intervals, 3 request types |
| Options Standard | **$80/mo** | 8 years | tick level, option-chain snapshots, every OPRA NBBO quote |
| Options Pro | **$160/mo** | 12 years | option-root snapshots, stream every option trade |

The $25 / $60 / $200 figures that circulate are wrong. $80 / $160 was right and omits the $40 entry
tier. Greeks are computed per tick against the underlying tick (⚠️ from the research, not re-read).

### Massive (formerly Polygon.io, renamed 2025-10-30) — `massive.com/pricing?product=options`

| Tier | Price | History | Data |
|---|---|---|---|
| Options Basic | **$0** | 2 years | 5 API calls/min, end-of-day only, minute aggregates |
| Options Starter | **$29/mo** | 2 years | unlimited calls, 15-min delayed, real-time greeks and IV, daily open interest, flat files, WebSockets |
| Options Developer | **$79/mo** | 4 years | + trades |
| Options Advanced | **$199/mo** | 5+ years | real-time, + quotes; *non-pros only* |

All tiers "Individual use". This matches the research agent's reading of the vendor page exactly;
the third-party claims ("$79 entry", "no options on free", "history to 2014") were the wrong ones.

### Databento OPRA — `databento.com/pricing`

| Plan | Price | Includes |
|---|---|---|
| Usage-based | pay per GB | historical only; **the per-GB rate is only in the interactive estimator**, not on the page |
| Standard | **$199/mo** | live data, 16+ yrs of L0 history, 1 yr L1, 1 month L2/L3, then per GB |
| Plus | **$1,750/mo** | annual contract; 16+ yrs L1, external distribution |
| Unlimited | **$4,500/mo** | annual contract; 16+ yrs in all schemas |

$125 signup credit, expires 6 months after signup, one set per team. 🚨 **OPRA history has two start
dates** — quotes/NBBO from 2023-03-28, trades and OHLCV from 2013-04-01 (⚠️ from the research). Never
quote it as one depth. No greeks or IV — you compute both.

### Alpha Vantage `HISTORICAL_OPTIONS` — `alphavantage.co/premium/`

- The page states the free key's *"standard API usage limit (25 API requests per day)"*. One call =
  one symbol-date, so **a free-key backfill of one ticker's history is weeks of calendar time**.
- Premium monthly tiers seen in the page source at **$149.99, $199.99, $249.99** (and higher); the
  rate-limit ↔ price mapping lives in a form widget, not page text.
- ⚠️ Still unconfirmed: whether `HISTORICAL_OPTIONS` itself is served on a free key. ⚠️ History from
  2008-01 with full greeks, IV, OI and volume — from the research, not re-read.

## Read through an API

### DoltHub `post-no-preference/options` — free, SQL API

Schema fetched from the repository's own SQL endpoint (default branch is **`master`** — `main` does
not exist):

```
option_chain: date, act_symbol, expiration, strike, call_put, bid, ask,
              vol, delta, gamma, theta, vega, rho
volatility_history
```

- **No open interest column. No volume column.** `vol` is `decimal(5,4)` — four decimals, capped at
  9.9999 — so it is implied volatility.
- 🚨 Without OI or volume you cannot reject illiquid strikes, which the skill's data-quality rule
  requires. **Not usable for a backtest that filters on liquidity.** Quotes and greeks only.

## From the research, not re-read (⚠️)

| Source | History | Greeks / IV | Note |
|---|---|---|---|
| **historicaloptiondata.com** | 2002 | L2+ only | ~$585 / $615 / $865 per year by level; full universe CSV over FTP |
| **optionsdx.com** | 2010 | ✅ | free tier, but ~7 tickers (SPX, VIX, SPY, QQQ, …); has 1-minute intraday |
| **FirstRateData** | 2010 | ✅ | **includes 4,000+ delisted tickers** — the survivorship fix, rare at any price |
| **ORATS** | EOD 2007, 1-min 2020-08 | ✅ | $199–$899/mo; the product is "**Near** EOD", not a true close |
| **CBOE DataShop** | 2012-01 | paid add-on | 🚨 **methodology break 2026-06-22** — quote sizes now captured at last price change; treat as a hard join boundary. Open-Close is Cboe exchanges only, not consolidated OPRA |
| **OptionMetrics / IvyDB** | 1996-01 | ✅ + constant-maturity surface | negotiated; WRDS is the practical route |
| **Intrinio** | up to 10 yrs | ✅ | OPRA exchange fees billed directly if displayed outside your firm |

Re-check any row you intend to pay for; `_reverify.md` records which of these have been re-read.
