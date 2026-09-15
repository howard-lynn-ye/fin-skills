---
name: india-markets
description: >-
  TRIGGER - NSE, BSE, Nifty, Sensex, upper circuit, lower circuit, price bands, Indian STT, index derivatives contract size, T+1 rollout, T+0, Muhurat, bhavcopy, kiteconnect, Zerodha. SKIP for Japan (japan-markets), Hong Kong (hong-kong-markets), Korea or Taiwan (korea-taiwan-markets), ASEAN (asean-markets), regional selection (asia-pacific-markets), and mainland China (china-trading-stack).
license: MIT
compatibility: Python 3.10+; offline scripts require numpy and pandas.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# India markets

Check security/date-specific bands and contract metadata before interpreting a price as a
possible fill. Settlement, tax, and closing-price construction also change within a backtest.

## Price bands and the measured trap

✅ [NSE price bands](https://www.nseindia.com/static/products-services/equity-market-price-bands)
(source audit 2026-09-10): cash securities use assigned 2%, 5%, 10% or 20% bands. Derivatives
underlyings do not have that static band but have an operating range; unlimited-price fills
do not follow from the absence of a static band. Use the dated security file and intraday
band updates. Ticks differ by security and regime: the demo's 0.05 is configurable, not a
universal current equity tick.

✅ Measured 2026-09-14 with `scripts/india.py`, fixed seed and 2,500 synthetic sessions:

| Assumed band | Unconstrained candidate opens outside band | Selected gap-down candidates outside band |
|---|---|---|
| 2% | 157 / 2,500 (6.28%) | 48 / 48 (100.00%) |
| 5% | 67 / 2,500 (2.68%) | 33 / 48 (68.75%) |
| 10% | 30 / 2,500 (1.20%) | 12 / 48 (25.00%) |
| 20% | 1 / 2,500 (0.04%) | 1 / 48 (2.08%) |

The candidate-price generator deliberately has no price bands; these are **latent proposed
fills, not observed NSE opening prints**. The selected sample has a 26x larger breach rate
under the 5% scenario. The audit demonstrates rejection, not strategy profitability: a signal
conditioned on a candidate opening price does not imply that price was observable or executable.
`clip_to_band` is a diagnostic projection, never permission to fill at the band edge.

⚠️ Sliding-band implementation retained from the 2026-09-10 audit of NSE/FAOP/64995,
[2024-11-08 circular](https://nsearchives.nseindia.com/content/circulars/FAOP64995.pdf),
effective 2024-11-18: both edges move in the price-movement direction, and orders outside the
new range are cancelled. The helper demonstrates upward flexing geometry, not a live trigger:
trade-participation criteria, cooling-off and actual messages must be supplied. It must not
increase the band solely because a timer elapsed.

## Market-wide breakers and sessions

✅ [NSE circuit-breaker schedule](https://www.nseindia.com/products-services/equity-market-circuit-breakers)
(source audit 2026-09-10): either Nifty 50 or Sensex can trigger the market-wide mechanism.
The 10% row changes at 13:00/14:30; the 15% row changes at 13:00/14:00. At 20% trading stops
for the day. A reopening uses a pre-open auction. Measured 2026-09-14: at 14:15, a 10% trigger
produces a 15-minute halt while a 15.5% trigger closes the market for the day.
The helper models an initial trigger, not repeated intraday trigger state.

⚠️ NSE/CMTR/74466 and NSE/CMTR/74969 session parameters were read in the 2026-09-10 audit:
CAS-eligible cash names stop continuous trading at 15:15 and use a 15:15–15:35 closing auction
from 2026-08-03; noneligible names retain a 15:30 continuous close. `cash_session` keys this
choice to both date and eligibility. The pre-open remains 09:00–09:15, with restructured
order-entry subperiods from 2026-09-07. Sources: [NSE circular archive](https://www.nseindia.com/resources/exchange-communication-circulars),
[NSE FAQs](https://www.nseindia.com/static/resources/faqs). The archive was rechecked on
2026-09-14 and contains [CAS data from 2026-08-03](https://www.nseindia.com/static/reports/closing-auction-session-historical-data);
the detailed circular parameters above remain explicitly attributed to the earlier audit.
Special/Muhurat sessions require that year's exchange notice; weekday filtering is insufficient.

## Settlement and taxes

⚠️ T+1 rollout dates retained from the 2026-09-10 audit: 2022-02-25 first tranche,
2023-01-27 completion ([NSE/CMTR/54992](https://nsearchives.nseindia.com/content/circulars/CMTR54992.pdf)).
`settlement_lag` refuses dates inside the rollout unless given the security's tranche date.
Optional T+0 began as a beta on 2024-03-28; it is not the default for every cash trade.
Holiday arithmetic is caller supplied and does not establish prefunding deadlines.

✅ [NSE/FATAX/73524](https://nsearchives.nseindia.com/content/circulars/FATAX73524.pdf),
checked 2026-09-14, reports enacted Finance Act 2026 rates effective 2026-04-01:
futures sale 0.05%; option sale 0.15% of premium; exercised options 0.15% on the applicable
exercise tax base. Delivery stock purchase/sale remains 0.1% each. The script also retains
the 2024-10-01 and preceding rate regimes; do not project its earliest table indefinitely
backward. Broker costs, GST, stamp duty and security/product exemptions are separate inputs.

✅ Measured: sale of INR 12,000 option premium gives INR 18 STT. Using INR 1,500,000 strike
notional instead would make the result 125x too large.

## Index derivatives: contract introduction is not trade date

✅ [SEBI exchange-traded derivatives chapter](https://www.sebi.gov.in/sebi_data/commondocs/dec-2024/RE_Chapter%205%20-%20Exchange%20Traded%20Derivatives%20FINAL_1_p.pdf),
checked 2026-09-14, applies the INR 15–20 lakh review range to **index derivatives**, with
a minimum INR 15 lakh at introduction, for contracts introduced after the November 2024
change. It does not triple every stock-derivative lot or all existing contracts on one date.
The published lot depends on review prices, expiry and corporate actions.

`fno_lot_size` is a theoretical lower-bound illustration for new index contracts. Its date
selects the **introduction regime**; its result is not an exchange contract master. The
minimum units and rounding constraint cannot reproduce every published lot. Use actual
instrument files for backtesting or orders. The same 2024 reform also limited weekly index
expiries to one benchmark per exchange (prior source audit 2026-09-10,
SEBI/HO/MRD/TPD-1/P/CIR/2024/132); use expiry-specific listings.

## Data routes and where this sits

Use NSE/BSE UDiFF bhavcopy and dated security/contract files for batch research; use an entitled
broker feed for live updates. `kiteconnect`, `nsepython` and `jugaad-data` are distinct API/client
routes, not interchangeable licences or uptime guarantees. A working website endpoint is not
proof of a documented stable API. Preserve source timestamps and blocked-request errors;
do not relabel stale downloads as real-time data.

- [Regional router](../asia-pacific-markets/SKILL.md).
- [Mainland trading](../../../fin-china/skills/china-trading-stack/SKILL.md): do not copy A-share assumptions.
- [Market data](../../../fin-market-data/skills/market-data-sourcing/SKILL.md): adjustment and source provenance.
