---
name: options-backtesting
description: >-
  Options positions end in ways you do not control - live or in a backtest. assignment, expiry
  settlement, pin risk, multi-leg lifecycle, historical chain assembly, and the margin that
  decides whether the position fits. TRIGGER - backtest a covered call, cash-secured put, wheel,
  credit spread, iron condor, butterfly, calendar, diagonal, straddle, strangle, PMCC; short
  option assigned, early exercise, exercise by exception, expires in the money, pin risk, pinned
  at the strike; historical option chain, options history, chain panel, OSI symbol, adjusted
  option, non-standard deliverable; 0DTE, weeklies, third Friday, AM vs PM settlement, cash
  settled index options; option margin, naked margin requirement, portfolio margin, SPAN, buying
  power reduction; "my options backtest returns look too good"; optopsy, optionlab, an options
  backtesting library. SKIP for pricing a single option or fitting a vol surface
  (derivatives-pricing) and for futures rolls (futures-continuous-contracts).
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

## 0. 🚨 Almost nothing in Python backtests options correctly

✅ **Fetched from the PyPI JSON API on 2026-09-04.** Of every package on PyPI (the full index, 884,814
names, was swept), **exactly one models assignment, exercise, settlement and margin: QuantConnect's
LEAN.** One more is a genuine multi-leg backtester that explicitly disclaims assignment. Everything
else is a pricer, a payoff plotter, or a live-broker SDK.

| Package | Latest | Multi-leg | Assignment | Margin | What it actually is |
|---|---|---|---|---|---|
| **LEAN** (`lean` 1.0.229, 2026-08-28, Apache-2.0) | ✅ | ✅ | ✅ **position-group** | ✅ | The only complete answer |
| **`optopsy` 2.3.0** (2026-03-04) | ✅ 38 strategies | 🚨 **explicitly disclaimed** | ❌ | ❌ | Best pure-Python option, know the limit |
| `optionlab` 1.8.5 (2026-08-10) | ✅ legs | ❌ | ❌ | ❌ | Single-date P&L / probability calculator |
| `backtrader` (last release **2023-04-19**) | ❌ | ❌ | ❌ | ❌ | 🔴 No options support at all |
| `vectorbt` 1.1.0 | ❌ | ❌ | ❌ | ❌ | No option-contract model |
| `mibian` 0.1.3 (**2016-03-12**) | ❌ | ❌ | ❌ | ❌ | 🔴 Dead ten years |
| `opstrat`, `Optlib`, `wallstreet` | payoff only | ❌ | ❌ | ❌ | 🔴 Dead or pricing-only |

🚨 **`optopsy` is AGPL-3.0-or-later, not MIT.** The widely-cited `michaelchu/optopsy` was MIT and has
been dead since 2021-06-04; the live project is `goldspanlabs/optopsy` and its own README states
AGPL-3.0. **AGPL's network clause reaches SaaS use** — this is load-bearing if you deploy. ⚠️ One
search source said GPL-3.0 instead; re-read the `LICENSE` file before relying on either.

✅ A keyword scan of `optopsy`'s own README: `margin` **0 hits**, `exercise` **0**, `settle` **0**,
`American`/`European` **0**, `multiplier` **0**. `assignment` appears once, in a disclaimer. Its data
contract has no OSI symbol, no multiplier and no deliverable field — **assignment modelling is
structurally impossible there, not merely unimplemented.**

🔑 **LEAN's margin is position-group based, and only if you submit the combo.** A long ITM call
(~$29,150) plus a short OTM call (~$101,499) margins at **$0 as a submitted spread** and as **naked
legs otherwise**. ⚠️ Secondhand from QuantConnect's docs. Its `DefaultOptionAssignmentModel` scans
hourly, considering American exercise within 4 days of expiry.

🔴 **`pip install py_vollib` is now wrong.** ✅ `py-vollib` 1.0.12 is an **empty shim** — its own
description says it "intentionally contains no library code". Use **`vollib`**. And
`py_vollib_vectorized` (last released **2021-02-28**) works by monkey-patching the now-deprecated
namespace. See `../../fin-libraries/skills/lib-vollib/SKILL.md`.

🔴 **`quantconnect-lean` on PyPI is a squatted placeholder** (v0.1.0, 2020, "reserved for future
use"). The real CLI is `lean`. **`optionsuite` is not on PyPI at all** (404) — it is GitHub-only.

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

⚠️ **This library does not carry verified per-strategy margin formulas or current broker
thresholds** — get those from your broker's own margin documentation. Reg T basics that apply across
equities are in `../us-market-rules/SKILL.md` §4.

🔑 **The one structural fact worth encoding**: margin is a property of the *position group*, not of
the legs. LEAN's numbers above make it concrete — the same two contracts require **$0** as a
submitted spread and **~$130,649** as separately-submitted legs. A backtest that sums per-leg
requirements is not conservative, it is modelling a different account.

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

### 🚨 No free live API serves expired contracts

⚠️ **Yahoo, Tradier and the IBKR TWS API all return zero history for expired options.** Yahoo's
`date` parameter selects **which currently-listed expiry to return — it is not an as-of date**, and
expired contracts drop out of the response entirely. Tradier's own documentation states historical
options data is unavailable for expired options.

**So there is no path from a free live API to an options backtest.** That is the structural fact;
everything below is about which paid source to buy.

| Source | History from | Greeks / IV | Note |
|---|---|---|---|
| **Alpha Vantage** `HISTORICAL_OPTIONS` | **2008-01** | ✅ full greeks + IV + OI | ⚠️ The cheapest API-accessible full history with greeks. One call = one symbol-date, so bulk pulls are painful |
| **historicaloptiondata.com** | 2002 | L2+ only | ⚠️ ~$585-865/yr flat files, full universe |
| **optionsdx.com** | 2010 | ✅ | ⚠️ Free tier, but only ~7 tickers (SPX, SPY, QQQ, VIX…) — has 1-minute intraday |
| **FirstRateData** | 2010 | ✅ | ⚠️ Includes **4,000+ delisted tickers** — rare at any price, and the survivorship fix |
| **ORATS** | EOD 2007, 1-min 2020-08 | ✅ it is the product | ⚠️ Product is "**Near** EOD", not a true close |
| **CBOE DataShop** | 2012-01 | paid add-on, not default | 🚨 **Methodology break 2026-06-22** — quote sizes now captured at last price change; silently discontinuous against earlier data. Open-Close is **Cboe exchanges only, not consolidated OPRA** |
| **OptionMetrics / IvyDB** | 1996-01 | ✅ + constant-maturity surface | ⚠️ Negotiated pricing; WRDS is the practical route |
| **Databento** OPRA | 🚨 **two dates**: quotes 2023-03-28, trades 2013-04-01 | ❌ you compute both | ⚠️ Never quote it as one history depth |

🚨 **Polygon.io is now Massive** (renamed 2025-10-30). ⚠️ Most third-party pages describing its
options tiers and history depth contradict the vendor's own pricing page — check the source, not a
blog.

⚠️ Every figure in this table is secondhand and vendor terms move. **Re-check pricing and start
dates before committing**, and treat the DataShop methodology break as a hard join boundary rather
than a footnote.

## 7. 🚨 Option symbols are not tickers

The OSI symbol encodes root, expiry, right and strike, so **the same economic contract changes
symbol across a corporate action**. After a split or a special dividend the deliverable is adjusted
and the contract is re-flagged — the strike no longer means 100 shares.

**Joining option data across a corporate action on symbol alone silently mixes two different
contracts.** An adjusted option carrying a non-standard deliverable will price as nonsense against a
standard model, and the error looks like a vol anomaly rather than a data bug.

**The OSI symbol is 21 characters, in four fixed-width fields:**

| Field | Width | Content |
|---|---|---|
| Root | **6** | left-justified, **space-padded right** |
| Expiration | **6** | `YYMMDD` |
| Type | **1** | `C` or `P` |
| Strike | **8** | strike × 1000, zero-padded left — 3 implied decimals |

```
"SPY   " + "260116" + "P" + "00452500"   ->   SPY   260116P00452500
   6           6        1        8        =   21
```

`AAPL  240119C00190000` decodes to AAPL, 2024-01-19, call, strike **190.000**. ⚠️ Many vendors strip
the root padding (`SPY260116P00452500`, 18 chars) — that is a **vendor variant, not OSI**, and
round-tripping requires knowing which your file uses.

⚠️ **Everything in the rest of this section is secondhand** — retrieved from OCC and Cboe documents
via search rather than read from the source PDFs, because network access failed mid-verification.
**Re-fetch before hard-coding any of it**, and pin a hashed copy: the OCC Rules PDF is live-updated
and returned two different Last-Modified dates for one URL.

### Three traps that break cross-time joins

⚠️ **Adjusted roots do not increment on re-adjustment, and suffixes are recycled.** `MSFT` becomes
`MSFT1` on its first adjustment and **stays `MSFT1`** through later ones, with the strike and
deliverable changing underneath. A freed suffix goes to the next class that needs one. **So
root + strike + expiry is not a stable key across time.** Key on an OCC-side identity plus an
as-of date, and carry the deliverable as a first-class field.

⚠️ **`GOOG7` / `AAPL7` were mini options, not adjusted contracts.** The `7` suffix marked 10-share
minis from 2013-03-18; `8` marked corporate/quarterly series. These collide with the 1-9 adjusted
suffix space, so a regex that reads a trailing digit as "adjusted" mislabels them.

⚠️ **Expiration moved from Saturday to Friday on 2015-02-01.** Standard contracts legally expired
the Saturday after the third Friday before that. Pre-2015 `YYMMDD` fields may hold the Saturday in
one dataset and the Friday in another — which silently breaks DTE arithmetic and symbol
reconstruction across the boundary.

### The multiplier stays 100 while the deliverable diverges

That is the dangerous shape. After a 3:2 split the deliverable becomes 150 shares but the multiplier
is still 100; after a special cash dividend it becomes 100 shares **plus a fixed cash amount**.
⚠️ Real OCC examples: `BABA1` = 100 ADS **+ ~$66.00 cash**; `CRES1` = 100 ADS + cash-in-lieu + **2
IRSA GDS**.

🚨 **So `notional = 100 × price` stays right while `intrinsic = max(S − K, 0)` goes wrong.** The
error looks like a volatility anomaly rather than a data bug, which is why it survives review.

## 7b. ⚠️ Exercise by exception is $0.01 × multiplier

⚠️ Secondhand, from Cboe Regulatory Circular RG08-73: a position **in the money by $0.01 or more**
is automatically exercised, **for all account types since the June 2008 expiration**.

🚨 **Two claims to stop repeating:**

- **"$0.05" is stale** — that was the October 2006 value.
- **"$0.01 per contract" is a units error** that appears in Cboe and OIC material itself. OCC Rule
  1804(b) states the index threshold as **$1.00 per contract**, and $0.01 per contract applies to
  *One Multiplier Options*, which are 1/100th the size. **Encode `threshold = $0.01 × multiplier`.**

⚠️ And it is not absolute: when a corporate action is unresolved at expiry, OCC **removes a class
from exercise-by-exception**, and then expiring positions are not exercised *regardless of how far
in the money they are*. **A deep-ITM option can expire worthless.** Your broker's threshold may also
differ from OCC's.

## 7c. 🚨 SPY and SPX options are not the same instrument class

⚠️ Secondhand, from Cboe product specifications:

| Product | Style | Settlement | Last trading day |
|---|---|---|---|
| **SPY, QQQ, IWM** | **American** | **Physical** — you deliver shares | expiration **Friday** |
| **SPX** (third Friday) | European | **Cash**, settled at **SET** (A.M. Special Opening Quotation) | **Thursday** |
| SPXW (weeklies, 0DTE) | European | Cash, at the P.M. close | expiry day |
| VIX | European | Cash, at VRO | **Tuesday** |

**Same exposure, different instrument.** SPY options are early-assignable — notably around SPY's
quarterly ex-dividend, which is exactly §1's trap — and trade a day longer than standard SPX. Any
code path treating them as one class is broken.

⚠️ **SET is not Friday's open.** Each constituent is priced at its own first trade on its primary
exchange, so the quote finalises only after every name has printed, and it routinely differs from
both Thursday's close and the 9:30 index print — **most on the volatile expirations you care
about.**

## 7d. Assignment is random at the top, and the deadline is not 4:00 pm

⚠️ Secondhand, from OCC and FINRA Rule 2360 material. Assignment is **two-stage**: OCC allocates to
clearing members **randomly**, then the member allocates to customers by FIFO, random, or another
approved method. **Model assignment as a probability, not a rule** — and note that two random stages
mean a small account's realised rate can deviate far from the aggregate.

🚨 **The 4:00 → 5:30 pm window is the whole of pin risk.** Moneyness for exercise-by-exception is set
off the **closing price**, but holders have until **5:30 pm ET** to submit instructions. So an option
**OTM at the close can still be exercised**, and one ITM by $0.01 or more can be **abandoned**. For
`|S_close − K|` small, assignment is a draw driven by the after-hours move, not by the close.

## 8. What still needs a second source

The ⚠️ markers above are a queue, not a permanent hedge. `references/_reverify.md` lists exactly
which claims need re-fetching and why, ranked — the `optopsy` LICENSE file, a hashed copy of the
OCC Rules PDF, and the ThetaData pricing conflict are the top three. It also records what is
already citation-grade so nobody re-does it.

## 9. Scripts

`scripts/option_lifecycle.py` — the early-exercise premium table, the pin-risk boundary, and the
sign-flipping assignment error. Carries its own CRR binomial tree, so it runs and makes its point
with QuantLib absent; where QuantLib is importable every value is checked against it.
