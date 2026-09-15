---
name: korea-taiwan-markets
description: >-
  TRIGGER - KRX, KOSPI, KOSDAQ, 韓國 공매도, Korean short-selling ban, TWSE, TPEX, 台股, 漲跌停, limit-up queue, Korea foreign registration, Taiwan ticks, pykrx, FinanceDataReader, FinMind, shioaji. SKIP for Japan (japan-markets), Hong Kong (hong-kong-markets), India (india-markets), ASEAN (asean-markets), regional selection (asia-pacific-markets), and mainland China (china-trading-stack).
license: MIT
compatibility: Python 3.10+; offline scripts require numpy and pandas.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Korea and Taiwan markets

A price at the daily limit is still a price at which trades can occur. It does not prove an
order could fill, and it does not prove that liquidity was absent. Separate exchange price
constraints, executable capacity at the order time, and the historical short-sale regime.

## Sessions, limits and settlement

✅ Sources checked 2026-09-14: [KRX session/settlement page](https://global.krx.co.kr/contents/GLB/06/0602/0602010201/GLB0602010201T1.jsp),
[KRX trading guide](https://global.krx.co.kr/contents/GLB/01/0109/0109000000/guide_to_trading_in_the_korean_stock_market.pdf),
[TWSE mechanism](https://www.twse.com.tw/en/products/system/trading.html), and
[TWSE settlement](https://www.twse.com.tw/en/about/company/service.html).

| Ordinary cash equities | Korea | Taiwan |
|---|---|---|
| Regular session, local time | 09:00–15:30 | 09:00–13:30 |
| Closing call-auction window | 15:20–15:30 | 13:25–13:30 |
| Normal daily price band | ±30% of base price | ±10% of opening auction reference price |
| Normal settlement | T+2 | T+2 |
| Normal board lot | One share | 1,000 shares; odd-lot markets are separate |

The KRX PDF is an older guide: do not copy its old tick or short-disclosure tables as current.
Both session calendars can have special notices; KRX CSAT late opens require explicit dates.
TWSE's after-hours fixed-price orders are not an extra continuous price-discovery session.

✅ [TWSE 2015 notice](https://www.twse.com.tw/staticFiles/news/news/tsecnews/ff80808187bbc39d0187c0ae58c20023.pdf),
checked 2026-09-14: 7% → 10% on 2015-06-01. ⚠️ Korea's 15% → 30% date, 2015-06-15,
is retained from the 2026-09-10 audit; the current guide verifies the latter band but not the
change date. The helper covers ordinary seasoned shares from 2010 onward; IPO, KONEX,
leveraged and foreign-underlying products need their own rules. TWSE's first-five-session
IPO exception is product-specific, not permission to disable limits for every new listing.

🚨 Use published upper/lower limits and the exchange reference price, not an unadjusted
previous close. Tick rounding near tier changes is venue-specific. `twse_tick` gives the six
ordinary-equity tiers; the queue helper requires already-resolved daily limits.

## Measured queue-capacity scenario

✅ `python scripts/korea_taiwan.py`, run 2026-09-14, uses one fixed latent shock path and
carries residual shocks forward when the scenario's band binds. The explicit assumption is
zero offer capacity on upper-bound pressured bars and zero bid capacity on lower-bound bars.
It is an illustration of a queue model, not an estimate of exchange liquidity.

| Scenario band | Momentum signals | Model fills | Rejected | Model fill rate |
|---|---|---|---|---|
| 30% | 213 | 210 | 3 | 98.6% |
| 10% | 229 | 185 | 44 | 80.8% |

The naive model fills every signal. `reachable` returns `None` when depth/capacity is unknown,
including a constant-price limit bar with positive daily volume. The exact same bar with
sufficient supplied offers can pass the buy-capacity check. A known suspension fails; insufficient
opposite-side capacity fails. A passed check remains conditional on the snapshot, order size
and queue assumptions. Do not cast `None` into an observed fill or into proof of no liquidity.

## Korea short-sale regimes and foreign access

✅ FSC sources checked 2026-09-14: [2021 partial resumption](https://fsc.go.kr/eng/pr010101/75291)
allowed only KOSPI200/KOSDAQ150 names from 2021-05-03; other names remained restricted.
[2025 full resumption](https://www.fsc.go.kr/eng/pr010101/84220) began 2025-03-31 with new
controls. The latest broad ban ran 2023-11-06 through 2025-03-30; the
[FSC extension notice](https://fsc.go.kr/eng/pr010101/82465) describes that reform period.
Historical index membership, borrow, account controls, market-maker exemptions and per-stock
restrictions remain separate. Passing a broad-ban gate does not approve a short order.

⚠️ Earlier ban intervals retained from the 2026-09-10 audit: 2008-10-01–2009-05-31,
2011-08-10–2011-11-09, and 2020-03-16–2021-05-02. Financial-stock restrictions lasted through
2013-11-13 after the first ban. The helper exposes `financial_stock` and `index_member`
explicitly and refuses dates before its coverage.

✅ [FSC registration reform](https://www.fsc.go.kr/eng/pr010101/80123), checked 2026-09-14:
prior foreign-investor registration was abolished from 2023-12-14; LEI/passport identification
replaces it. Sector and issuer ownership ceilings still need security-specific checks.
✅ [TWSE foreign-investor update](https://twmonthly.twse.com.tw/front/content/8a81b9818c4c790b018d33acee68001e?date=Feb.+2024),
checked 2026-09-14: paperless registration and prefunding flexibility are distinct from
abolishing registration. Broker/custodian funding deadlines can precede T+2 settlement.

## Data routes and where this sits

Use KRX official data and announcements, with `pykrx` or `FinanceDataReader` as client routes;
use TWSE/TPEX published files and MOPS filings, with FinMind for aggregation and shioaji for
broker access. Confirm present credentials, terms and coverage instead of treating a client
name as a data entitlement. Delisted price history does not establish a complete point-in-time
universe; fundamentals need announcement/revision timestamps.

- [Regional router](../asia-pacific-markets/SKILL.md).
- [China trading](../../../fin-china/skills/china-trading-stack/SKILL.md): related queue shape, different rules.
- [Market data](../../../fin-market-data/skills/market-data-sourcing/SKILL.md): provenance and adjusted prices.
