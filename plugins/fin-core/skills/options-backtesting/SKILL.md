---
name: options-backtesting
description: >-
  Backtest an options strategy without inventing the P&L - assignment, expiry settlement, pin risk,
  multi-leg lifecycle, historical chain assembly, and the margin that decides whether the position
  fits. TRIGGER - backtest a covered call, cash-secured put, wheel, credit spread, iron condor,
  butterfly, calendar, diagonal, straddle, strangle, PMCC; short option assigned, early exercise,
  exercise by exception, expires in the money, pin risk, pinned at the strike; historical option
  chain, options history, chain panel, OSI symbol, adjusted option, non-standard deliverable;
  0DTE, weeklies, third Friday, AM vs PM settlement, cash settled index options; option margin,
  naked margin requirement, portfolio margin, SPAN, buying power reduction; "my options backtest
  returns look too good"; optopsy, optionlab, an options backtesting library. SKIP for pricing a
  single option or fitting a vol surface (derivatives-pricing) and for futures rolls
  (futures-continuous-contracts).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# Options backtesting

**An equity position ends when you sell it. An options position can end five ways, and you control
one of them.**

| How it ends | Who decides |
|---|---|
| You close it | you |
| Expires worthless | nobody — clean |
| Expires ITM, exercised by exception | the clearing house, by default |
| **Assigned early** | **the long side, and you find out afterwards** |
| **Pinned at the strike** | **unknown until assignment notices arrive** |

A backtest that computes `max(S − K, 0)` at expiry has modelled rows two and three and silently
assumed the others never happen. This skill is about what that costs.

For **pricing** a single option, Greeks scaling, and vol-surface fitting, go to
`../derivatives-pricing/SKILL.md` — including its §6, which carries the measured evidence that
`lastPrice` is unusable (54.6% of a live SPY call chain quoted outside its own bid-ask).

## 1. 🚨 Early assignment is not an edge case on a dividend payer

✅ Measured with `scripts/option_lifecycle.py`, verified against QuantLib 1.43 to **9.73e-06**.
American vs European call, S=100, K=90, 30 days:

| Dividend yield | European | American | Early-exercise premium |
|---|---|---|---|
| **0%** | 10.4275 | 10.4275 | **0.0000** |
| 2% | 10.2677 | 10.2677 | 0.0000 |
| 5% | 10.0290 | 10.0509 | 0.0219 |
| 10% | 9.6343 | **10.0000 = intrinsic** | 0.3657 |
| 20% | 8.8569 | **10.0000 = intrinsic** | 1.1431 |

🔑 **At q=0 the premium is exactly zero** — a non-dividend-paying American call is never exercised
early. That is why this trap is invisible until your universe contains a payer, and then it is
everywhere.

🚨 **At q≥10% the American value collapses to intrinsic: the option should be exercised
immediately.** European pricing says 8.8569 against a true 10.00 — **understating it by 11.4%**.

**The rule a backtest needs:** a rational holder exercises a call early to capture a dividend
whenever the call's **remaining time value is less than the dividend**. In the script's covered-call
example that is time value **0.4586 against a dividend of 1.20** — assignment is not a risk there,
it is the expected outcome.

## 2. 🚨 The assignment error changes sign with the path

This is why it cannot be calibrated away with a haircut. ✅ Same script — a covered call assigned at
the ex-dividend date, against a backtest that keeps the position open:

| Expiry price | Naive backtest | With assignment | Error |
|---|---|---|---|
| 110.00 | 8,700 | 7,500 | **+1,200 overstates** |
| 103.00 | 6,700 | 7,500 | −800 understates |
| **96.00** | **−300** | **+7,500** | **−7,800 understates** |

**In the last row the backtest reports a $300 loss on a position that actually made $7,500.** The
shares were called away before the drawdown; the backtest kept holding them down.

A backtest carrying an already-assigned position forward is not paying a fixed cost. It is trading
a position that does not exist, and the result is noise with no stable sign — which is far worse for
a research process than a known bias.

## 3. 🚨 Pin risk — one cent decides your Monday

✅ Short 10 calls struck at 100, at the closing bell:

| Settle | vs strike | Clearing default | Shares Monday |
|---|---|---|---|
| 99.99 | −0.01 | expires | 0 |
| 100.00 | 0.00 | expires | 0 |
| **100.01** | **+0.01** | **auto-exercised** | **−1,000** |

⚠️ **And auto-exercise is a default, not a guarantee.** A holder may submit contrary instructions —
choosing not to exercise an ITM option, or exercising one that is OTM — so the short side's exposure
is genuinely unknown until notices arrive.

**A backtest cannot resolve pin risk, and should not pretend to.** Treat any position open at expiry
within a cent of the strike as an unmodelled exposure and report how often it happened. A strategy
whose returns depend on that resolution going your way has not been backtested.

## 4. Multi-leg positions are one position, not N options

A spread's legs do not fill together, do not get assigned together, and do not margin separately.

- **Legging risk.** A four-leg iron condor filled at four separate mids is a price nobody could have
  traded. Model the spread's own bid-ask, or state that you assumed simultaneous mid fills.
- **Partial assignment.** The short leg of a vertical can be assigned while the long leg is still
  open, which converts a defined-risk position into a stock position overnight. This is the single
  most common way a "defined risk" backtest understates its worst day.
- **Early close of one leg** changes the margin requirement of the whole position, not a quarter of
  it.

## 5. Margin decides whether the position was ever possible

Three different regimes, and a backtest that assumes the wrong one is sizing an account that cannot
exist:

| Regime | Applies to | Shape |
|---|---|---|
| **Reg T** | most retail accounts | Fixed formulas per strategy; naked short options carry a percentage-of-underlying requirement |
| **Portfolio margin** | accounts above a broker-set equity threshold | Risk-based, from a stress of the underlying |
| **SPAN** | futures options | Scenario grid set by the clearing house |

⚠️ **This library does not yet carry verified per-strategy margin formulas or the current broker
thresholds.** Treat the table above as the shape of the problem and get the numbers from your
broker's own margin documentation. Reg T basics that do apply across equities are in
`../us-market-rules/SKILL.md` §4.

**A short-premium backtest with no margin model has not established that the strategy is
executable**, only that it would have been profitable if size were free.

## 6. Assembling a historical chain panel

The join is the work, and it is where look-ahead enters.

- **One row per (underlying, expiry, strike, right, timestamp).** Anything flatter loses the
  information you need.
- **Join the underlying at the same timestamp as the quote**, not the daily close. An option quoted
  at 15:59 against a 16:00 close has a built-in edge that is not real. This is the same
  `merge_asof` direction problem as `../market-data-engineering/SKILL.md` — and here a wrong-side
  match produces a plausible IV rather than an obvious error.
- **Compute your own IV from the mid**, per `../derivatives-pricing/SKILL.md` §6.
- **Expiry conventions change T and therefore every Greek.** AM-settled index options settle on the
  opening prints of the constituents, not a traded price; PM-settled ones settle at the close. An
  off-by-one on expiry date is an off-by-one on every number downstream.
- **Survivorship applies to underlyings.** A chain panel built from today's listed names has already
  dropped every company that was acquired or delisted — see `../research-integrity-guards/SKILL.md`.

## 7. 🚨 Option symbols are not tickers

The OSI symbol encodes root, expiry, right and strike, so **the same economic contract changes
symbol across a corporate action**. After a split or a special dividend the deliverable is adjusted
and the contract is re-flagged — the strike no longer means 100 shares.

**Joining option data across a corporate action on symbol alone silently mixes two different
contracts.** An adjusted option carrying a non-standard deliverable will price as nonsense against a
standard model, and the error looks like a vol anomaly rather than a data bug.

⚠️ The precise OSI field layout, the adjusted-option flagging conventions, and the clearing house's
exercise-by-exception threshold are **not yet verified in this library** — confirm against the OCC's
own contract adjustment memos before relying on any of them.

## 8. Scripts

`scripts/option_lifecycle.py` — the early-exercise premium table, the pin-risk boundary, and the
sign-flipping assignment error. Carries its own CRR binomial tree, so it runs and makes its point
with QuantLib absent; where QuantLib is importable every value is checked against it.
