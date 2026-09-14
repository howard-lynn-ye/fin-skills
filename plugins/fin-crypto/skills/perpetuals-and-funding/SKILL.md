---
name: perpetuals-and-funding
description: >-
  TRIGGER - perpetual swap, perp funding, funding interval, funding history, mark versus index
  versus last price, liquidation threshold, maintenance margin, inverse contract, cash-and-carry,
  basis to dated futures, open interest versus volume, why a perp backtest disagrees with cash
  P&L. SKIP for token swaps and delistings (crypto-token-events), AMM impact and LP fees
  (defi-and-amm-mechanics), calendars, outages and stablecoin depegs (crypto-market-structure),
  exchange clients and data fetching (crypto-data-and-execution), and rolling dated contracts
  (futures-continuous-contracts).
license: MIT
compatibility: Python 3.10+ with numpy and pandas; seeded offline demonstrations only.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Perpetuals and funding

Keep three ledgers distinct: trade P&L, funding cash flows, and margin/liquidation state.
A last-trade candle supplies none of the other two automatically.

## Funding is cash on the settlement notional

✅ Source-verified 2026-09-14, web text: [Bybit funding-fee documentation](https://www.bybit.com/en/help-center/article/Funding-fee-calculation)
uses position value times funding rate; positive funding transfers from longs to shorts and
negative funding reverses the direction. For a linear contract, position value uses quantity
and mark price. Inverse contracts require the venue's coin-denominated valuation convention.

For a fixed linear quantity `q`, accumulate `q * mark_at_settlement * rate` in the cash account.
For shorts the funding sign reverses. If exposure changes, align the quantity held at the actual
settlement time; do not charge a rate simply because a bar exists. Separately record settlement
currency and any USD conversion.

✅ Measured 2026-09-14 by `scripts/perpetuals.py`, seed 20260910:

| Example | Result |
|---|---|
| Synthetic fixed $100,000 notional, 1,095 eight-hour settlements | $9,587 paid |
| Funding as a share of notional / initial margin at 10x | 9.59% / 95.9% |
| Fixed-quantity long, seeded moving settlement mark: price-only return | +47.67% |
| Same quantity and marks, accumulated funding cash debited | +36.82% |
| Funding cash divided by initial capital | 10.84% |

🚨 The two rows use **different exposure policies**. Fixed-notional funding sums to the rate
sum; fixed-quantity funding changes with the mark. The latter now uses a cash ledger rather
than compounding `(1 + price_return - funding_rate)` and calling it a buy-and-hold position.
These examples do not simulate survival of a leveraged margin account, collateral interest,
fees or position resizing.

The seeded generator combines a historical clamp-shaped formula with a configurable cap. Its
default clamp and cap were drawn from different venues; it is **not a current exchange replica**.
The resulting funding rate is a scenario assumption, not an empirical expected return.

## Intervals belong to instruments

✅ Source-verified 2026-09-14, web text: [Bybit instruments-info](https://bybit-exchange.github.io/docs/v5/market/instrument)
defines `fundingInterval` in minutes per instrument. Archive interval changes together with
realized funding. `rate * 24 / interval_hours * 365` is a simple annualized quotation, not a
compounded or predictable return.

✅ Measured 2026-09-14: on the same uncapped premium path the illustrative four-hour rate is
half the eight-hour rate. Multiplying a four-hour rate by three settlements per day therefore
understates the annual quotation by a factor of two; a one-hour rate by a factor of eight.
The demo's differing-length histories produce annual rates of 9.59%, 9.40% and 9.06%, respectively;
they are not the same sampled history. Historical OKX instrument counts in the script remain
**2026-09-08 snapshots**, not current counts.

## Index, mark and last are different inputs

✅ Source-verified 2026-09-14, web text: [Bybit mark-price rules](https://www.bybit.com/en/help-center/article/Mark-Price-Calculation-Perpetual-Expiry-Contracts)
use mark price for liquidation and describe a median construction, a moving basis and fallback
criteria. The page permits selection-rule changes in volatile markets. Fetch the actual mark
series when replaying a venue; do not claim a synthetic reconstruction is exchange truth.

✅ Measured 2026-09-14: 6,000 seeded paths, each six hours with ten-second observations. The
script uses its **historical median-mark model** with a thirty-second basis average, not the
current Bybit implementation. With assumed maintenance margin 0.4% and 50x leverage:

| Trigger model | Trigger hits | Last-trade hits | Trigger only | Last only |
|---|---|---|---|---|
| Median mark | 2,717 | 3,033 | 3 | 319 |
| Hypothetical index-only trigger | 2,699 | 3,033 | 37 | 371 |

“Trigger only” means the trigger crossed while last never crossed anywhere in the simulated
path. Those counts prove the disagreement can occur; they do not estimate its live frequency.
Sampling misses intrabar breaches, and actual liquidation fills depend on venue rules and depth.

For an isolated linear position, constant maintenance fraction `m`, leverage `L` and no fees:
`long_liq/entry = (1 - 1/L)/(1 - m)` and
`short_liq/entry = (1 + 1/L)/(1 + m)`.

✅ Measured 2026-09-14: at the assumed 10x and 0.4% maintenance fraction, the long level is
−9.64% from entry. Paying 0.01% every eight hours for thirty days consumes 0.90% of notional
and moves that level to −8.73% even with a flat mark. Tiered margin, collateral haircuts,
liquidation fees and cross-margin positions require more than this isolated-position formula.

## Dated basis and open interest

`annualised_basis()` computes simple or compounded annual quotations from `F/S` and time to
expiry. `deribit_curve()` preserves the **2026-09-08 13:25 UTC** historical constants recorded in
the original project. Its 3.10%–4.83% simple annualized curve is historical arithmetic; the
synthetic perp path is not a contemporaneous market comparison. A dated spot/futures hedge
requires financing, custody, fees and sufficient margin to survive until expiry; future perp
funding is unknown. Contract chains and roll yield belong to
`../../../fin-futures-fx/skills/futures-continuous-contracts/SKILL.md`.

🚨 Volume cannot uniquely identify position creation: opening on both sides increases OI,
closing on both sides decreases it, and an opening/closing transfer leaves it unchanged.
Aggregate price and OI do not identify traders' motives or prove “new money” versus covering.

✅ Measured 2026-09-14: the seeded tape has volume 366,697, net OI change +17,753 and signed
volume/OI-change correlation −0.0005. Initial OI is a separate balance so simulated closes cannot
create negative outstanding interest. These correlation values belong to this random tape.

## Scripts and routing

Run `python plugins/fin-crypto/skills/perpetuals-and-funding/scripts/perpetuals.py`.
Pure functions cover funding histories/cash costs, fixed-quantity accounting, annualized rates,
liquidation thresholds, mark/last path comparisons, dated basis and OI accounting. The script
never connects to a venue, touches credentials or places orders.

- `../crypto-data-and-execution/SKILL.md` — venue clients and funding-history collection.
- `../crypto-token-events/SKILL.md` — underlying migrations and contract delistings.
- `../defi-and-amm-mechanics/SKILL.md` — AMM-based liquidity; some on-chain perp venues instead
  use order books, so inspect the actual design.
- `../crypto-market-structure/SKILL.md` — annualization, stablecoin collateral and ADL.
- `../../../fin-core/skills/execution-cost-analysis/SKILL.md` — liquidation impact.
- `../../../fin-libraries/skills/lib-ccxt/SKILL.md` — per-exchange funding capabilities.
