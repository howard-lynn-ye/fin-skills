---
name: corporate-actions-processing
description: >-
  TRIGGER - spin-off, spinoff, rights issue, TERP, merger, exchange ratio, reverse split,
  cash in lieu, ticker change, symbol recycling, special dividend, parent price dropped on
  ex-date, 分拆, 配股, 并购. Reconcile event cash and share entitlements with position returns.
  SKIP for ordinary split/dividend price-adjustment detection (market-data-sourcing),
  stale prices and OHLC validation (data-quality-validation), calendars
  (trading-calendars-and-sessions), and survivorship-free universes (research-integrity-guards).
license: MIT
compatibility: Python with numpy and pandas; optional calendar libraries are checked when installed.
metadata:
  version: "0.1.0"
  verified_on: "2026-09-14"
allowed-tools: Bash(python:*)
---

# Corporate actions processing

Carry dated events and position entitlements alongside prices. A return-adjusted price series
can represent event value, but does not tell an execution or accounting system which shares,
rights and cash the investor owns afterward.

## Separate a missing event from an adjustment convention

🚨 Do not claim that every adjusted series books a spin-off as a loss. Treatment depends on
the data provider and return convention. ✅ Primary-source verification, 2026-09-14:
[MSCI Corporate Events Methodology, February 2026, section 2.8.1](https://www.msci.com/indexes/documents/methodology/0_MSCI_Corporate_Events_Methodology_20260210.pdf)
applies a price adjustment factor to the parent on the spin-off ex-date. That disproves a
universal inability to adjust spin-offs; it does not establish another vendor's coverage.

✅ Measured 2026-09-14 by [corporate_actions.py](scripts/corporate_actions.py), fixed synthetic
examples that deliberately omit events from the quoted-price feed:

| Event omitted from feed | Price-only return | Holder return | Holder minus price, percentage points |
|---|---:|---:|---:|
| Spin-off | -30.0000% | 0.0000% | 30.0000 |
| Discounted rights issue | -13.3333% | 0.0000% | 13.3333 |
| Reverse split | 900.0000% | 0.0000% | -900.0000 |

`spin_off_gap()` also computes an explicit adjustment: its ex-date return is 0.0000%.
Subsequent returns still depend on whether distributed shares are kept, sold or reinvested.
The demo includes SpinCo at its first close and has no market movement at the distribution
instant. Real first-close movement is not a pure corporate-action adjustment.

`rights_issue_gap()` assumes freely saleable rights at theoretical value, a fully subscribed
discounted issue and no fees. Subscribing requires an external cash contribution; the helper
subtracts it from proceeds. Letting rights expire, nontransferable rights and above-market
subscriptions need different models. `reverse_split_gap()` assumes fractional entitlements
are cashed at the theoretical ex-price. Actual event terms can use different prices or dates.

## Preserve the position, identity and event clock

✅ Reference accounting identities, executed 2026-09-14:

- Spin-off: value is parent shares plus distributed shares. Store both identifiers, the
  entitlement ratio, ex-date, distribution date and valuation policy.
- Rights issue: retain rights quantity, subscription price, deadline, transferability and the
  holder's election; subtract new contributed cash from investment profit.
- Reverse split: update quantity along with price. Keep fractional entitlement and cash-in-lieu
  treatment explicit. Ordinary split-ratio detection belongs to market-data-sourcing.
- Merger: close the old position using cash and/or acquiring shares on the relevant event date.
  A missing quote alone cannot distinguish consideration received from a loss.
- Ticker change: maintain dated symbol-to-identifier mappings. A reused ticker is not proof of
  issuer continuity. Special dividends require the cash entitlement and adjustment policy.

✅ Measured 2026-09-14: `merger_treatments()` illustrates a cash-merger panel under an explicitly
specified daily equal-weight model: keeping consideration, dropping the name and treating it
as worthless yield different results. It is not a buy-and-hold account simulator.
`ticker_change_cost()` shows that splitting one instrument at a rename loses 252 available
momentum-signal sessions with the configured lookback. `ticker_recycling()` produces a false
-60.68% seam return when different companies share a symbol.

## Investigate rather than invent adjustments

`unexplained_jumps(close, actions, tol)` reports large moves and whether an event date is in
the supplied feed. A date match is not proof that the event explains the entire move. No match
is not proof of data corruption; news can cause a genuine large return.

✅ Measured 2026-09-14: the default screen finds 2 of the 3 inserted events and misses the
rights issue. An empty feed proves no coverage, not the absence of events. Lower thresholds
increase flags on the event-free heavy-tailed comparison too. Never manufacture an adjustment
factor from a jump alone. Run `python scripts/corporate_actions.py` to reproduce the examples;
all inputs are synthetic, offline and deterministic.
