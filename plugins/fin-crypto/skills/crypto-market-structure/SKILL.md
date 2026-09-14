---
name: crypto-market-structure
description: >-
  TRIGGER - crypto annualisation, 365 versus 252, weekend returns, 24/7 market, calendar-day
  rolling windows, exchange daily close timezone, cross-venue price dispersion, no consolidated
  tape, venue outage, auto-deleveraging, ADL, socialised losses, stablecoin depeg or quote-currency
  conversion. SKIP for token events and delisting universes (crypto-token-events), perpetual
  funding and mark-price liquidation (perpetuals-and-funding), AMM pools and LP fees
  (defi-and-amm-mechanics), and choosing exchange clients or fetching data
  (crypto-data-and-execution).
license: MIT
compatibility: Python 3.10+ with numpy and pandas; uses fin_skills.api.conventions when available.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Crypto market structure

Name the venue, instrument, quote currency and bar boundary before comparing prices or returns.
Continuous crypto trading is a calendar convention; it does not guarantee continuous liquidity,
uninterrupted exchange access or a single settlement convention for every derivative.

All measurements below were reproduced on 2026-09-14 by `scripts/market_structure.py`, seed
20260910. They use synthetic prices and are not empirical estimates of venue behavior.

## Annualization follows the observation frequency

Use `fin_skills.api.conventions.annualization_factor()` and `annualize_sharpe()` when the
package is installed. The standalone script falls back only when `fin_skills` itself is absent;
it does not silently mask broken package imports.

For ordinary calendar-day crypto returns, the library's annualization convention is 365.
It uses 8,760 for hourly observations and 1,095 for eight-hour funding observations. A weekday
strategy still needs weekend portfolio returns accounted for if risk is measured in calendar
time. Choosing 252 merely because an equity library defaults to it is not a conversion.

✅ Measured on the same 1,460 calendar-day rows:

| Statistic | Factor 365 | Factor 252 |
|---|---|---|
| Sharpe, mean / sample SD times square-root factor | 0.8432 | 0.7006 |
| Annualized volatility | 65.81% | 54.69% |

Their ratio is `sqrt(252/365) = 0.8309`; the 252 result is 16.9% smaller for this positive
Sharpe. For negative Sharpe the magnitude shrinks rather than performance becoming worse.
For a series with 252 observations per year, applying 365 inflates magnitude by 20.4%.
Dropping weekends alone does not produce that frequency: the script's weekday subset has
no exchange-holiday filter, so its 252/365 comparison is sensitivity analysis, not a claim
that either factor matches the subset's actual observations per year.
These are conventional scaling identities, not proof that returns are independent or that
square-root scaling is a valid forecast at longer horizons.

## Rolling rows are not calendar durations

✅ Measured: on this synthetic daily series, 252-row and 365-row momentum calculations disagree
on sign in 17.7% of 1,095 common observations; correlation is 0.7791. A 252-row lookback on an
unbroken calendar-day series does not span a year. Missing observations require explicit
coverage checks; silently filling outages with zero returns changes both the signal and risk.

✅ Measured: with an assumed desk watching 13:00–21:00 UTC on weekdays, 1,926 of 2,000 hourly
paths cross a 5% downside level. Of those first crossings, 71.3% occur outside desk hours.
This is a homogeneous-volatility example, not evidence that real activity is uniform by hour.
Price jumps and local reopening gaps remain possible even in a continuously traded asset.

## There is no single venue-independent price or daily close

✅ Measured: five synthetic venue feeds of one latent asset over 730 days have median cross-
venue price range 21.4 bp, 99th percentile 43 bp and maximum 48 bp. A lagged thirty-day momentum
rule produces a 15.4 percentage-point return spread across those feeds before transaction costs.
Noise near the entry threshold changes trades; these are not executable arbitrage profits.

The same hourly path sampled at different daily boundaries produces Sharpe values from −1.2856
to −1.2473. Both the intervals and sample endpoints differ. Align UTC opening/closing instants,
quote currency, candle completion, instrument type and venue before calling a difference a
premium. Historical OKX and Deribit boundary examples in the script retain the original
**2026-09-08** observation date; confirm the chosen endpoint's current bar convention.

## Venue failure, socialized losses and ADL

✅ Source-verified 2026-09-14, web text: [Bybit's ADL rules](https://www.bybit.com/en/help-center/article/Auto-Deleveraging-ADL)
describe automatic reduction of opposing positions and settlement at a bankruptcy price. The
current trigger conditions include insurance-fund drawdown thresholds, so **fund exhaustion is
not a universal prerequisite**. ADL is position reduction; a generic account-wide haircut or
socialized-loss allocation is a distinct mechanism. Read the venue's instrument and account
rules rather than treating all of them as the same fee.

✅ Measured illustration: a short entered at 30,000 and marked at 21,000 shows a 30% gain.
If it is instead force-closed at an assumed 27,000, the gain is 10%: two-thirds of that marked
gain is given up. This is a settlement-price example, not an ADL probability or a venue forecast.

✅ Measured: with assumed mean four-hour outages, the simulated fifth-percentile price move
while unable to act is −2.39%. Summing only negative moves over six assumed annual outages gives
−3.33%; **that is not expected annual drag**, because favorable moves have been discarded.
The function exposes the unconditional mean separately. Real outages can coincide with stress,
so independent synthetic returns are insufficient to price exchange exposure.

## Stablecoin depegs change the unit of account and collateral value

For an asset worth `P_USD` and a stablecoin worth `S_USD`, the stablecoin-quoted price is
`P_USD / S_USD`. An apparent asset gain can therefore come from the quote currency falling.
Convert cash balances, fees, funding and collateral with the same timestamped exchange rate.
A depeg can reduce borrowing capacity or create a financing need; spot conversion alone cannot
estimate actual borrow rates, liquidation haircuts or funding costs.

✅ Measured: the synthetic six-day depeg reaches a stablecoin value of 0.917 USD and introduces
a 9.0% peak quote-conversion premium. After recovery the window's conversion effect is 0.00%,
yet annualized tracking error between USD and stablecoin returns is 1,284 bp. High daily
correlation, 0.9805 here, does not remove intraperiod collateral risk.

✅ Source-verified 2026-09-14, web text: [Circle's March 2023 notice](https://www.circle.com/pressroom/3-3-billion-of-usdc-reserve-risk-removed-dollar-de-peg-closes)
reported $3.3 billion of USDC reserves at Silicon Valley Bank, about 8% of total reserves, and
subsequent availability of those deposits. This is a dated banking event; the script's depeg
path is not fitted to the historical USDC price.

## Scripts and routing

Run `python plugins/fin-crypto/skills/crypto-market-structure/scripts/market_structure.py`.
Functions cover annualization, rolling windows, off-desk breaches, cross-venue dispersion,
sampling-hour effects, hypothetical outages/forced closes and stablecoin conversion. They
perform no network access, credential handling or order submission.

- `../crypto-data-and-execution/SKILL.md` — clients, candles and exchange access.
- `../crypto-token-events/SKILL.md` — historical universes and token identity.
- `../perpetuals-and-funding/SKILL.md` — funding, margin and liquidation triggers.
- `../defi-and-amm-mechanics/SKILL.md` — LP exposure when a stablecoin leg changes value.
- `../../../fin-core/skills/backtest-validation/SKILL.md` — return definitions and Sharpe limits.
- `../../../fin-core/skills/broker-execution-apis/SKILL.md` — execution state and reconciliation.
