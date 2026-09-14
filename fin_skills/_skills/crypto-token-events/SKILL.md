---
name: crypto-token-events
description: >-
  TRIGGER - token swap, redenomination, migration, same ticker changed units, hard fork,
  airdropped fork coin, rebase, elastic supply, balance changed but price did not, wrapped
  or bridged token, delisted pair, reconstruct a historical top-N crypto universe,
  survivorship bias, a -90% day caused by a token conversion. SKIP for funding and liquidation
  (perpetuals-and-funding), AMM pools and LP losses (defi-and-amm-mechanics), 24/7 calendars,
  outages and stablecoin depegs (crypto-market-structure), and exchange clients and OHLCV
  fetching (crypto-data-and-execution).
license: MIT
compatibility: Python 3.10+ with numpy and pandas; all demonstrations run offline.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# Crypto token events

A ticker is not a permanent economic identity. Store venue, chain, contract address, unit and
observation time alongside price. Keep event records separately so adjustments can be rebuilt.
This corrects the former router's shorthand that crypto has “no corporate actions”: analogous
balance and unit events exist even when the exchange client provides no adjustment history.

## Rebases: multiply price by the holder's balance index

✅ Source-verified 2026-09-14, web text: [Ampleforth's protocol documentation](https://docs.ampleforth.org/learn/about-the-ampleforth-protocol)
describes proportional daily supply adjustments through a global scalar. The current rule uses
a sigmoid and adjustable parameters. The script's linear feedback rule is an **illustrative
model**, not an implementation of today's AMPL policy.

🚨 Rebase returns satisfy `(1 + holder_return) = (1 + price_return) * (1 + balance_return)`.
Use the holder's actual balance factor, not total token supply when issuance is dilutive or
custody does not pass the adjustment through. Rebases mechanically change balances; traded
prices can also change in response to market demand.

✅ Measured 2026-09-14 by `scripts/token_events.py`, seed 20260910, synthetic 730-day history:

| Measure | Result |
|---|---|
| Price-only return | +2.40% |
| Balance-index return | +136.82% |
| Holder-value return | +142.49% |
| Return difference | 140.09 percentage points |
| Price-only / holder Sharpe | 0.2608 / 1.1973 |
| Maximum daily identity residual | 4.44e-16 |

The direction and size are sample-specific; the multiplicative accounting identity is general.

## Redenominations: match economic units before joining tickers

🚨 A conversion to more units can resemble a crash even when the ticker stays unchanged.
Store `new_per_old`, the venue's effective timestamp, old/new contract identifiers and the
crediting policy. Express all prices in old units by multiplying post-event prices, or in new
units by dividing pre-event prices. Both give the same adjusted returns.

✅ Measured 2026-09-14: the seeded 500-day, 1-old-for-10-new example has a raw return of −89.11%
versus a holder return of +8.86%. Its raw worst day is −90.73%; adjusted and holder returns agree
with maximum residual 0.00e+00. This is synthetic conversion arithmetic, not observed STRAX P&L.

⚠️ Historical examples retained from the recovered 2026-09-10 source review are
[STRAX to STRAX](https://www.binance.com/en/support/announcement/binance-will-support-the-stratis-strax-token-swap-and-redenomination-plan-2e6eb8e644664a6ca7b6659a226c41b0)
and [GAL to G](https://www.binance.com/en/support/announcement/binance-will-support-the-galxe-gal-token-swap-redenomination-and-rebranding-to-gravity-g-27449abd481b473493b9c058350b4e2e).
Their constants retain that date in the script. They were **not newly source-verified** during
this completion pass; do not use historical notice timing as a current exchange rule.

## Delistings: reconstruct the universe as of the decision date

`top_n_as_of()` selects only pairs trading on the requested day. Ranking uses trailing volume
observed since listing; nonexistent pre-listing volume and future rows cannot enter the rank.
It returns fewer names if fewer are available. A current symbol list cannot reconstruct missing
historical membership without an archive or a point-in-time data supplier.

✅ Measured 2026-09-14, synthetic 500-pair starting universe, three years, top 50:

| Assumed volume-dependent annual hazard | Old top-50 still trading | Current top-50 present at start | Starting cohort return | Survivors-only return |
|---|---|---|---|---|
| 10% | 78% | 62% | 56.0% | 80.4% |
| 20% | 66% | 58% | 59.6% | 111.0% |
| 35% | 52% | 56% | 32.8% | 81.7% |

The middle row has 33 old members still trading and 29 current members that existed at the start.
Those are **different denominators**; 58% is not the fraction of the historical universe retained.
The simulated return difference is 51.4 percentage points. Delisted positions are held in cash
at their last print for this example: no claim is made that this was executable or bounds actual
recovery. Hazards and price processes are assumptions, and this experiment does not establish
crypto's empirical attrition rate or prove it is higher than equities'.

## Wrapped assets and forks

✅ Source-verified 2026-09-14, web text: [WBTC](https://wbtc.network/) describes BTC held in custody
and mint/burn access restricted to approved institutions. A backing ratio does not guarantee
that every holder can redeem directly at par. Different bridge and wrapper designs have
different custody, contract and redemption risks; premiums as well as discounts are possible.

✅ Measured 2026-09-14: the synthetic wrapper example has median absolute basis 2.7 bp, worst
basis −515 bp and annual tracking error 576 bp, despite daily return correlation 0.9955. These
are stress-example outputs, not WBTC history or estimated exploit probabilities.

✅ Source-verified 2026-09-14, web text: the [Ethereum Foundation's DAO fork notice](https://blog.ethereum.org/2016/07/20/hard-fork-completed)
identifies block 1,920,000 on 2016-07-20 and discusses replay risks. For portfolio accounting,
record whether the custodian actually credited the fork asset and when it became tradable.

✅ Measured 2026-09-14: `fork_split(100, 0.10)` allocates 90 to one chain and 10 to the other,
with 0% holder return and −10% ticker-only return. **Value conservation is an input to this
illustration.** Real fork prices need not sum to the pre-fork price, and not every hard fork
creates a separate marketable asset.

## Scripts and event records

Run `python plugins/fin-crypto/skills/crypto-token-events/scripts/token_events.py`.
Functions cover rebase accounting, redenominations, point-in-time selection, survivorship,
wrapper tracking error and an illustrative fork allocation. They use no network, credentials
or order API. For real records retain `kind`, `venue`, `chain`, contract identifiers,
`symbol_before`, `symbol_after`, `effective_ts`, `observed_at`, conversion/balance factors,
source URL, crediting policy and terminal-price provenance.

## Where this sits

- `../crypto-data-and-execution/SKILL.md` — clients, historical data collection and order safety.
- `../perpetuals-and-funding/SKILL.md` — funding and derivative contract units; underlying token
  events can require contract adjustments too.
- `../defi-and-amm-mechanics/SKILL.md` — rebasing assets held inside pools.
- `../crypto-market-structure/SKILL.md` — venue survival and stablecoin quote conversion.
- `../../../fin-core/skills/research-integrity-guards/SKILL.md` — point-in-time universes.
- `../../../fin-core/skills/backtesting-engines/SKILL.md` — corporate-action adjustment support.
- `../../../fin-libraries/skills/lib-ccxt/SKILL.md` — market identifiers and OHLCV limitations.
