---
name: fx-markets
description: >-
  Trade and backtest FX correctly — quote conventions, pip sizing, and the carry that a spot-only
  backtest silently omits. TRIGGER - forex, FX, currency pair, EURUSD, USDJPY, GBPUSD, AUDUSD,
  USDCHF, USDCAD, NZDUSD; pip, pipette, lot sizing on a currency pair; carry trade, swap points,
  rollover, interest rate parity, covered or uncovered parity; NDF, forward points, T+2 value date;
  "there is no official FX close"; DukasCopy or free tick FX; forex-python or a similar package.
  SKIP for crypto pairs, which have funding rather than swap (crypto-data-and-execution), and for
  FX options and vol surfaces (derivatives-pricing).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# FX markets

FX breaks three assumptions an equity toolchain makes: **there is no consolidated tape, the
quote convention is not uniform, and a position earns or pays interest every night.**

## 1. 🚨 A spot-only backtest is missing the carry

A currency position is two interest rates. Holding long AUD/short USD earns the AUD rate and pays
the USD rate every night — the **swap** or **rollover**. That is not a fee; for a carry trade it is
the entire thesis.

✅ Demonstrated in `scripts/fx_conventions.py`: a long AUDUSD position over the sample returns
**spot only −2.97%/yr (Sharpe −0.27) — a loser** — and **total +1.34%/yr (Sharpe +0.19) — a winner**,
on the same price series. **The sign of the result flips.**

**If your FX backtest computes `pct_change()` on a spot series and stops, it is not a backtest of a
position anyone can hold.** Add `carry_return(spot, r_base, r_quote, days)`.

✅ **Covered interest parity verified live to 0.001%** across two CME contracts. The size of the
omission: leaving FX rollover out **overstates a long-EURUSD backtest by 1.30%/yr — or 3.91%/yr of
equity at 3x leverage**, which is where most FX strategies actually run.

⚠️ Retail rollover is not the interbank differential — brokers mark it up, often asymmetrically, so
the carry you actually receive is smaller than parity implies and the carry you pay is larger. Model
the broker's published swap rates, not the policy rates, when the strategy is carry-dependent.

## 2. 🚨 Pip size is not uniform — and getting it wrong is a 100× error

**JPY pairs quote to 2 decimal places (pip = 0.01). Almost everything else quotes to 4
(pip = 0.0001).**

✅ Demonstrated: sizing a position for a fixed pip risk on USDJPY with the 4-decimal assumption
produces a notional of **$15,025,000 instead of $150,250 — 100× oversized**, turning a budgeted
$1,000 loss into **$100,000**.

This is the FX analogue of the Greeks scaling error in
`../../../fin-core/skills/derivatives-pricing/SKILL.md` §4: an arithmetic mistake that raises no
exception and is invisible in the number itself.

Note also **pipettes** — most venues now quote a fifth decimal (third for JPY), so "the last digit"
is a tenth of a pip, not a pip.

## 3. Quote conventions

`EURUSD` means **units of USD per 1 EUR** — EUR is the base. The market convention is not "always
USD first":

| Quoted as XXX/USD (USD is the quote) | Quoted as USD/XXX (USD is the base) |
|---|---|
| EURUSD, GBPUSD, AUDUSD, NZDUSD | USDJPY, USDCHF, USDCAD |

**A long EURUSD is long EUR and short USD. A long USDJPY is long USD and short JPY.** Treating the
pair symbol as "the asset" and going long it means opposite USD exposure depending on which side of
that table you are on — and the sign error survives every plot.

**Crosses** (EURGBP, AUDJPY) are usually synthesized from two USD legs, so their spreads are wider
than the majors and their tick data is often reconstructed rather than observed.

## 4. 🚨 There is no consolidated tape and no official close

FX is over-the-counter. **Every venue has its own price, and "the" closing price does not exist.**
Consequences that break research:

- Two data sources will disagree, and neither is wrong. Reconciling them is not a data-quality task.
- Daily bars depend on an arbitrary cut — 17:00 New York is the most common convention, but a
  vendor using 00:00 UTC produces different daily returns from the same underlying market.
- **The WM/Refinitiv 16:00 London fix is the closest thing to an official benchmark**, and it is a
  fixing window, not a print. If your strategy trades "the close", say which close.
- Backtests that assume a single global price cannot model the venue selection a real execution
  would face.

## 4b. 🚨 Free FX bars are quantised, and futures invert the convention

✅ **Yahoo's hourly EURUSD is quantised to 1.34 pips — 6.7x the real 0.20-pip spread.** Measured:
**44 distinct values across 116 bars**, all exactly representable in float32, with `1/p` landing on
a 1e-4 grid. There is **no bid/ask and volume is all zeros**. A spread or microstructure study on it
is measuring the storage format, not the market.

🚨 **CME FX futures invert the spot convention for JPY, CAD and CHF.** `6J` quotes **0.00645** while
`USDJPY` quotes **156.10** — reciprocals. Joining a futures series to a spot series without
inverting produces a perfectly plausible, entirely wrong correlation.

✅ **DukasCopy free tick FX is real** — verified end-to-end: **8,717 ticks in one hour** with genuine
bid/ask and a **median spread of 0.20 pips**. Two gotchas: **the month in the URL is zero-indexed**
(January is `00`), and **findatapy's base URL now 301-redirects**.

## 5. Value dates and settlement

Spot is **T+2** for most pairs, with **T+1 for USDCAD** (and a few others). That matters for carry
accrual — the rollover is charged on the value date, so a Wednesday position typically accrues
**three days** of swap to cover the weekend.

**NDFs** (non-deliverable forwards) exist for restricted currencies — KRW, TWD, INR, BRL, CNY
offshore conventions differ from CNH. A "USDCNY" series may be onshore, offshore, or an NDF, and
those are different instruments with different prices.

## 6. Data

🔑 **Free tick-level FX exists**: DukasCopy via `findatapy` (`freq='tick'`, `fields=['bid','ask']`)
is the only free tick source in this catalogue —
see `../../../fin-core/skills/market-data-sourcing/references/findatapy.md`.

For point-in-time interest rates to compute carry honestly, use FRED/ALFRED vintages
(`../../../fin-core/skills/fundamental-and-macro-data/references/fredapi.md`) — policy rates are
revised and republished, and using today's rate history to compute yesterday's carry is the same
look-ahead as any other macro series.

🚨 **`forex-python` is alive but serves ECB *reference* rates**, and the ECB itself states:
*"Using the rates for transaction purposes is strongly discouraged."* They are a daily 16:00 CET
fixing for accounting, **not a tradeable price**. Backtesting execution on them is not a backtest.

🔴 **Dead FX/broker clients:** `v20`, `fxcmpy`, `forexconnect`, `oandapyV20` have all rotted.
`ib_insync` is archived — use `ib_async`.

⚠️ Prefer a real vendor plus your own conventions layer over a wrapper that hides which side of §3
you are on.

## 7. Scripts

`scripts/fx_conventions.py` — `parse_pair`, `is_inverted`, `pip_size`, `pip_value`,
`notional_for_pip_risk`, `carry_return`, `total_return`. Its demo is the spot-vs-total sign flip and
the JPY 100× sizing error.
