---
name: japan-markets
description: >-
  TRIGGER - Japanese equities, TSE, JPX, 東証, 前場, 後場, lunch-break bars, TOPIX500 ticks, Nikkei divisor, 空売り規制, J-Quants, e-Stat, yen quotes. SKIP for Hong Kong (hong-kong-markets), India (india-markets), Korea or Taiwan (korea-taiwan-markets), ASEAN (asean-markets), regional selection (asia-pacific-markets), and mainland China (china-trading-stack).
license: MIT
compatibility: Python 3.10+; offline scripts require numpy and pandas.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Japan markets

Use date-specific session, security-master and corporate-action data. The numerical examples
below are synthetic checks, not observations of Japanese investment returns.

## Sessions and the measured trap

✅ Source checked 2026-09-14: [JPX trading hours](https://www.jpx.co.jp/english/equities/trading/domestic/01.html)
gives 09:00–11:30 and 12:30–15:30, with 15:25–15:30 reserved for the closing-auction order period.
The close changed from 15:00 on 2024-11-05; the date is carried from the 2026-09-10 source audit
of [JPX's implementation notice](https://www.jpx.co.jp/english/news/1030/20230920-01.html).

🚨 Do not fabricate executions in lunch or auction order collection. A session grid and an
executed-trade tape are different objects. The script includes the auction window in its nominal
minute grid to isolate the lunch-break error. It does not model auction liquidity or holidays.
Hourly bars are possible with partial bars and explicit labels; they are not prohibited by TSE.
`session_bar_count` deliberately requires a frequency that tiles both session windows exactly.

✅ Measured 2026-09-14 with `python scripts/japan.py`, using its fixed synthetic seed:

| Grid or volatility pipeline | Result |
|---|---|
| Nominal minute slots, before / after the close extension | 300 / 330 |
| Wall-clock slots after extension | 390, an 18.18% overcount |
| Session-aware annualised volatility | 26.01% |
| Forward-filled wall clock with wall-clock annualiser | 26.01%, relative error −0.02% |
| Real observations with wall-clock annualiser | 28.28%, relative error +8.71% |
| Filled observations with session annualiser | 23.92%, relative error −8.04% |
| Inserted zero-return observations | 3,600 |

The volatility run assumes 60 synthetic weekdays, 245 sessions/year, per-minute volatility
0.09%, and a lunch-spanning observation with 15 minutes' variance. These are model inputs.
The near-cancellation when both sides use wall time is possible; fixing only one side creates
a large error. Dropping the break return changes the estimand; it is not automatically a fix.
Use explicit overnight/break-return treatment for an actual volatility model.

## Ticks, limits and settlement

✅ [JPX tick table](https://www.jpx.co.jp/english/equities/trading/domestic/07.html), checked
2026-09-14: the finer table currently covers TOPIX500 constituents, with sub-yen increments at
low prices. The script includes the ordinary and fine tables, with inclusive upper boundaries.
Select the table from dated security metadata; do not infer it from a ticker's current index.
The page announces liquidity-based tick selection from 2027-03-01; the present helper is not
an implementation of that future regime or of every ETF table.
The table's cash-equity quotes and increments are in yen. Keep quote currency and
per-share units explicit; changing an index divisor or an adjusted-price series does not
convert the price into another currency or into a board-lot notional.

✅ [JPX daily limits](https://www.jpx.co.jp/english/equities/trading/domestic/06.html), checked
2026-09-14: Japan **does have daily price limits**, alongside special-quote mechanisms. They
are absolute-yen bands by base-price tier; do not substitute Korea's or Taiwan's percentage.

⚠️ Carried from the 2026-09-10 primary-source audit: cash-equity settlement changed T+3 → T+2
on 2019-07-16 ([JPX settlement programme](https://www.jpx.co.jp/english/equities/clearing-settlement/tplus2-settlement-cycle/index.html)).
`settlement_date` models that transition using caller-supplied settlement holidays. Weekend-only
arithmetic is insufficient for a production cash ledger. The 2022-04-04 Prime/Standard/Growth
restructuring also requires dated segment membership; current labels do not reconstruct history.

## Indices and short selling

✅ [Nikkei methodology](https://indexes.nikkei.co.jp/nkave/archives/file/nikkei_stock_average_guidebook_en.pdf),
checked 2026-09-14: sum adjusted constituent prices, then divide by the published divisor.
Use the price adjustment factor (PAF), including caps where applicable. A plain market-cap
portfolio does not replicate the Nikkei. [JPX's TOPIX page](https://www.jpx.co.jp/english/markets/indices/topix/),
also checked 2026-09-14, specifies free-float-adjusted market capitalisation; its eligibility
and weight adjustments must come from dated index files.

🚨 Corporate actions can change the divisor **or the PAF**. Nikkei's
[December 2023 split notice](https://indexes.nikkei.co.jp/en/nkave/archives/news/20231214E_1.pdf)
explicitly kept the divisor unchanged while changing PAFs. The script's 25.0000 → 24.0782
continuity example is a generic price-index example, not a claim that every Nikkei split
changes its divisor. This output was reproduced on 2026-09-14.

✅ [JPX short-sale restrictions](https://www.jpx.co.jp/english/equities/trading/regulations/02.html),
checked 2026-09-14: the trigger is a fall of at least 10% from the exchange's daily **base price**,
which can differ from unadjusted previous close. It persists through the next trading day.
On an uptick a short sale at the last price may be permitted; on a downtick the price must be
higher. Reporting/disclosure thresholds are 0.2%/0.5%. The helper tests trigger crossing only,
using decimal prices; it does not implement trigger persistence, borrow or exemptions.

## Data routes

✅ [JPX's J-Quants launch terms](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20230403-01.html),
read 2026-09-14, document a free plan with two years of data and a 12-week delay. This is a dated
plan statement, not a current subscription entitlement. Check the account's present plan and
response timestamps before calling data recent. `jquants-api-client` is the library route;
announcement timestamps and revisions matter for point-in-time fundamentals.

✅ [e-Stat API guide](https://www.e-stat.go.jp/api/api/api/index.php/en/api-info/api-guide),
checked 2026-09-14: an account and application ID are required. Preserve statistical release
and revision dates separately from the observation period. e-Stat is a statistics route, not
an equity execution feed.

## Where this sits

- [Regional router](../asia-pacific-markets/SKILL.md): cross-market source and calendar checks.
- [China trading](../../../fin-china/skills/china-trading-stack/SKILL.md): compare session and fill conventions.
- [Market data](../../../fin-market-data/skills/market-data-sourcing/SKILL.md): adjustments and dated joins.
