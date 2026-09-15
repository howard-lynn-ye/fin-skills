---
name: defi-and-amm-mechanics
description: >-
  TRIGGER - AMM, constant-product pool, x*y=k, Uniswap v2 or v3, concentrated liquidity,
  LP position, impermanent loss, divergence loss, fee income versus IL, pool price impact,
  out-of-range liquidity, DEX quote versus fill, sandwich, MEV or gas. SKIP for funding,
  mark price and liquidation (perpetuals-and-funding), rebases and token swaps
  (crypto-token-events), calendars, venue outages and stablecoin depegs
  (crypto-market-structure), centralised exchange clients (crypto-data-and-execution), and
  quoting on an order book (market-making-models).
license: MIT
compatibility: Python 3.10+ with numpy and pandas; no network, keys or orders.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Read Bash(python:*)
---

# DeFi and AMM mechanics

Pool arithmetic can validate a quote. Realized execution also needs transaction ordering,
actual reserves, fees and routing. Do not apply constant-product formulas to a different
invariant merely because the venue calls itself an AMM.

✅ Source-verified 2026-09-14 through extracted PDF text: the
[Uniswap v2 whitepaper](https://app.uniswap.org/whitepaper.pdf) describes the reserve invariant,
input fee and optional protocol fee. The
[Uniswap v3 whitepaper](https://app.uniswap.org/whitepaper-v3.pdf) describes concentrated ranges,
separate fee balances and inactive liquidity outside a range. Those are historical protocol
specifications, not a claim that today's deployed pools use only their initial fee tiers.

All measurements below were reproduced on 2026-09-14 by `scripts/amm.py`, seed 20260910.
The simulations use continuous quantities and ignore on-chain integer rounding.

## Constant-product impact: account for the fee retained in reserves

For reserves `(x, y)`, selling `dx` of X, fee `f`, `g = 1-f`, and `u = dx/x`:

```
dy = g * dx * y / (x + g * dx)
x_after = x + dx
y_after = y - dy
execution_price / initial_spot = g / (1 + g*u)
post_trade_spot / initial_spot = 1 / ((1 + u) * (1 + g*u))
```

🚨 `1/(1+u)^2` is the pool-price ratio **only at zero fee**. The full input enters reserves,
while the fee-reduced input determines output. `cp_impact()` is tested against the reserve
changes in `cp_swap()` at nonzero fees as well as at zero.

✅ Measured: a zero-fee trade equal to 5% of the input reserve executes 4.7619% below the initial
spot and moves the reserve price down 9.2971%. Its average execution price is the geometric
mean of the initial and final reserve prices. Walking the curve in 200,000 slices returns
47,619.0476190473 units versus the closed form 47,619.0476190476: relative difference 6.7e-15.

That slicing identity assumes zero fees and no other flow. Retained fees, arbitrage between
slices, block ordering and routing change the result. It is not a live execution-scheduling rule.

## Impermanent loss compares with holding the same starting basket

For a full-range, equal-value constant-product position without fees, rebalanced to a new
relative price `r = P1/P0`:

```
IL = LP_value / hold_value - 1 = 2*sqrt(r)/(1+r) - 1
```

✅ Measured independently from the ending reserves: a halving and a doubling both give
−5.7191%; a fourfold move gives −20.0000%; a tenfold move gives −42.5040%. The largest absolute
closed-form/reserve difference in the printed table is 1.11e-16. The formula is symmetric in
`r` and `1/r` and is zero at the starting price.

This is underperformance against holding, not necessarily a negative absolute return. A
negative convexity description can be useful; claiming the position is exactly a sold straddle
is not an identity. Exiting realizes the current difference, while subsequent price reversals
would have changed an unclosed position's difference.

## Fees and a conditional break-even calculation

✅ Source-verified 2026-09-14, v2 whitepaper §2.4: fee-related growth can be expressed through
`sqrt(k)`. V3 §3.2.1 keeps accrued fees separately, so they are not automatically reinvested in
active liquidity. An auto-compounding wrapper is a separate strategy and needs separate costs.

✅ Measured: with the illustrative fee 0.30%, 4,000 synthetic two-sided trades increase `k` by
2.7829%. `1 - sqrt(k0/k1)` is 1.3630%, and `sqrt(k1/k0)-1` is 1.3819%; converting between those
denominators agrees to 1.21e-16. This demo excludes protocol-fee LP-token dilution.

`il_vs_fees()` draws 20,000 terminal prices over thirty days, then solves the **approximation**
`fee * daily_volume / TVL * days = -mean_IL`:

| Assumed annual volatility | Mean IL | Median IL | 10th-percentile IL | Break-even daily volume / TVL |
|---|---|---|---|---|
| 30% | −0.093% | −0.043% | −0.250% | 1.03% |
| 60% | −0.370% | −0.170% | −1.002% | 4.11% |
| 100% | −1.024% | −0.471% | −2.775% | 11.38% |
| 150% | −2.287% | −1.081% | −6.221% | 25.41% |

🚨 These are scenario break-evens with fixed TVL, no gas, no protocol cut and no joint model of
price and fee-generating flow. They are not empirical thresholds for profitable pools. Expected
IL grows approximately with variance for small moves; the exact expression is not globally
quadratic. Use consistent value denominators and actual fee ownership for a live LP analysis.

## Concentrated liquidity: narrow ranges amplify exposure

For a symmetric range `[p0/m, p0*m]`, the capital multiple at entry relative to the full curve
at the same liquidity `L` is `1/(1 - 1/sqrt(m))`. `cl_amounts()` clips the reserve calculation at
range boundaries. Outside the range one token balance is zero and the position supplies no
active liquidity until price returns.

✅ Measured: range `[1/1.10, 1.10]` has a capital multiple of 21.5. For an in-range +2% price
move its divergence loss relative to its own starting basket is amplified by that same factor.
At +20%, which is outside this range, its loss is −6.87% versus −0.41% for full range; the
in-range multiplier cannot be extrapolated past the boundary.

✅ Measured: across 4,000 thirty-day paths with assumed 60% annual volatility, the same range
is active at 62% of observations. The printed fee-multiple illustration is capital multiple
times active-time share, 13.2. This assumes constant fee flow per unit of active liquidity;
actual fees depend on competing liquidity and where volume occurs. It is neither a guaranteed
return nor a rigorous upper bound. A range order can reverse if price crosses back before
liquidity is removed.

## Quote versus realized execution

✅ Measured: in the script's pool with reserves 1,000,000 of each token, victim input 20,000,
fee 0.30% and slippage tolerance 1%, a simulated sandwich can push victim output to the limit.
The resulting attacker profit is 0.862% of victim input. The function solves the
**limit-saturating size**, not globally optimal profit. The fraction captured changes with
reserves, trade size, fees, ordering and costs; it is not a universal 86% rule.

The script cannot infer live gas or sandwich frequency from synthetic prices. Historical
receipts, transaction traces, mempool observations and aggregator routes can support a richer
replay; it would need those actual inputs. Gas is paid even when some transactions revert,
ordering determines whether an attack executes, and a multi-pool route changes the reserve
arithmetic. Report unmeasured costs separately rather than inventing realized performance.

## Scripts and routing

Run `python plugins/fin-crypto/skills/defi-and-amm-mechanics/scripts/amm.py`.
Public functions cover swaps, fee-aware impact, sliced execution, fee growth, full-range IL,
conditional break-even volume, concentrated reserves and a hypothetical sandwich. Everything
runs offline without credentials or order submission.

- `../crypto-data-and-execution/SKILL.md` — venue clients and data sourcing.
- `../perpetuals-and-funding/SKILL.md` — perpetual funding; check whether the actual on-chain
  venue uses an AMM or an order book before borrowing this model.
- `../crypto-token-events/SKILL.md` — wrappers, rebases and token-unit changes inside pools.
- `../crypto-market-structure/SKILL.md` — stablecoin quote conversion and counterparty events.
- `../../../fin-strategies/skills/market-making-models/SKILL.md` — order-book quoting.
- `../../../fin-microstructure/skills/limit-order-book-models/SKILL.md` — passive queue fills.
- `../../../fin-strategies/skills/execution-algorithms/SKILL.md` — order scheduling.
- `../../../fin-core/skills/execution-cost-analysis/SKILL.md` — realized execution costs.
