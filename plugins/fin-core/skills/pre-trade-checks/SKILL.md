---
name: pre-trade-checks
description: >-
  Check a proposed order before a human sends it - this skill checks orders and never sends one.
  Fat finger, participation, session and auction windows, duplicate and replay, exposure limits,
  and a kill switch that fails everything closed. TRIGGER - pre-trade risk checks, fat finger,
  sanity-check an order before sending, "did I type an extra zero", notional cap, max order size,
  size against ADV, participation cap, price through the market, limit far from the last trade,
  duplicate client order id, idempotency, retry loop resending orders, kill switch, is the venue
  open, opening or closing auction, half day early close, gross net per-name or sector exposure
  limit, reconciling a position with the broker, SEC Rule 15c3-5, erroneous order controls. SKIP
  for connecting to a broker and proving the session is paper (broker-execution-apis), for
  measuring fills you already have (execution-cost-analysis), for the schedule that works an order
  (execution-algorithms), and for A-share price limits (china-trading-stack).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Pre-trade checks

**This skill checks orders. It does not send them, and neither does anything in this library.**
`scripts/pre_trade.py` takes a *proposed* instruction and returns a verdict; the person still
presses send. The module is read-only by construction in the same sense as
`fin_skills.bridges.execution`, and by the same test: `tests/test_bridges_execution.py` greps
both files — plus the guard — for 49 order-sending method names across ccxt, ib_async and
alpaca-py in three normalisations, and poisons each source in turn to prove the grep can fail.

Everything marked ✅ Measured is printed by `scripts/pre_trade.py` — numpy + pandas, **seed
20260910**, runs in **3.0 s**, no network, no file writes. The fixture is **40 proposals over 12
names against a 10,000,000 book**, three of them deliberately wrong.

> **The rule: a pre-trade check is a function of a PROPOSAL and ASSEMBLED facts that returns a
> verdict, and an absent fact is a block, not a pass.**

## 1. ✅ The law asks for four things in one sentence, not one thing

✅ Source-verified in **17 CFR 240.15c3-5**, the Market Access Rule (read 2026-09-10 at
`law.cornell.edu/cfr/text/17/240.15c3-5`; source credit **[75 FR 69825, Nov. 15, 2010]**).
Paragraph **(c)(1)(ii)** requires controls reasonably designed to

> "Prevent the entry of erroneous orders, by rejecting orders that exceed appropriate price or
> size parameters, on an order-by-order basis or over a short period of time, or that indicate
> duplicative orders."

That is **four** separate requirements — a price parameter, a size parameter, a *rate* over a
short window, and duplicates — and they fail independently. Paragraph **(c)(1)(i)** adds credit
and capital thresholds "more finely-tuned by sector, security, or otherwise".

| The sentence says | This file calls it | What it alone can see |
|---|---|---|
| price parameters | `check_fat_finger` | a transposed digit at any size |
| size parameters | `check_fat_finger`, `check_participation` | a decimal slip at any price |
| over a short period of time | `check_duplicate` | a retry loop with fresh ids each time |
| duplicative orders | `check_duplicate` | the same `client_order_id` twice |
| by sector, security | `check_exposure` | the name and sector caps |

Engineering context, **not legal advice**. This is a US broker-dealer obligation; it is quoted
here because it is the clearest published statement of what the checks have to be, not because
running them discharges anyone's compliance duty.

## 2. You cannot get a green light without assembling the facts

```python
from fin_skills.core.pre_trade import ProposedOrder, TradingState, Caps, check_order

v = check_order(ProposedOrder("AAPL", "buy", 10_000, 190.0, "cid-1", ts), state)
v.allowed        # one boolean
v.findings       # every Finding: check, severity (block/warn/info), code, message, evidence
v.missing        # the facts that were ABSENT - the other half of why allowed is False
print(v.render())
```

`TradingState` has one field per fact and **every one defaults to `None`**, which means ABSENT.

🚨 **`allowed` is False when a fact is missing, not only when a check fails.** An unknown ADV is
not a pass. A calendar nobody declared is not a pass. A broker position nobody read back is not a
pass. Absence is the exact state a system is in right before it lets through the thing everyone
later agrees should have been stopped, so absence gets the same verdict as a breach — and the
verdict names the fact.

✅ Measured — the same clean ticket, one fact removed at a time:

| state | `allowed` | `missing` |
|---|---|---|
| every fact assembled | **True** | — |
| `adv_shares = None` | **False** | `adv_shares[AA01]` |
| `broker_positions = None` | **False** | `broker_positions` |
| `sessions = None` | **False** | `sessions` |
| `already_sent = None` | **False** | `already_sent` |

An **empty** blotter is a fact ("nothing has gone out yet") and passes; `None` is not a fact and
blocks. The same distinction runs through every field.

| check | required facts | blocks on |
|---|---|---|
| `check_kill_switch` | `kill` | tripped |
| `check_session` | `now`, `sessions`, `venue` | closed, not a session, lunch break |
| `check_fat_finger` | `last_trade`, `day_low/high`, `adv_shares`, `equity`, 4 caps | notional, book fraction, ADV multiple, price through |
| `check_participation` | `adv_shares`, `last_trade`, `max_participation`, `pov_cap` | over the ADV cap |
| `check_duplicate` | `already_sent`, `burst_window_s`, `burst_max` | duplicate id, replay burst |
| `check_exposure` | `positions`, `marks`, `equity`, `broker_positions`, `sectors`, 4 caps | unreconciled, gross, net, name, sector |

⚠️ One deliberate exception, stated so it can be argued with: `daily_vol` is **optional**. Absent,
the participation verdict still stands on ADV alone and the *cost consequence* is reported as
"not priced" at warn level rather than blocking.

## 3. ✅ Three deliberate mistakes, and which check saw each

✅ Measured. Of the 40 proposals, **3 are blocked and 37 clear** — zero false positives at the
stated caps:

| ticket | what is wrong | blocked by | the number |
|---|---|---|---|
| `slip-qty` | 10x quantity: 431,800 AA04 instead of 43,180 | `notional_cap`, `notional_vs_book`, `qty_vs_adv`, **`over_adv_cap`** | 8,765,540 notional = **87.66% of the book**, **0.28x ADV** |
| `slip-px` | two digits transposed: 246.71 for a 218.33 last trade | **`price_through_last`** (+ a `price_outside_day_range` warning) | **+13.00%** through, and **4.39x the day's range** outside it |
| `replay` | a retry loop reused the id and sent three more like it | **`duplicate_id`**, **`replay_burst`** | 4 near-identical tickets inside **30 s**; this one makes 5 |

🔑 **The decimal slip lands in the thinnest name in the universe, and that is what makes it
dangerous.** The same keystroke in the most liquid name is 2.8% of ADV and invisible. `slip-qty`
is the only one of the three that any size threshold can see.

🔑 **Nothing about the `replay` ticket is wrong.** It is correctly sized (1.16% of the book),
correctly priced, inside every cap. Only its *multiplicity* is wrong, and multiplicity is not a
magnitude — §4 measures what it costs to catch it with a size cap anyway.

## 4. ✅ What each threshold would have caught, on its own

✅ Measured, one threshold at a time with every other size and price cap widened to infinity.
`planted` is how many of the three it caught; `clean` is how many of the other 37 it stopped.

| `max_adv_multiple` | 0.02 | 0.05 | 0.10 | **0.25** | 0.30 | 0.50 |
|---|---|---|---|---|---|---|
| planted / clean | 1 / 0 | 1 / 0 | 1 / 0 | **1 / 0** | 0 / 0 | 0 / 0 |

| `max_price_through` | 0.005 | 0.01 | 0.02 | 0.05 | **0.10** | 0.20 |
|---|---|---|---|---|---|---|
| planted / clean | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | **1 / 0** | 0 / 0 |

| `max_notional` | 250k | 500k | **750k** | 1.5m | 5m | 10m |
|---|---|---|---|---|---|---|
| planted / clean | 1 / **11** | 1 / 0 | **1 / 0** | 1 / 0 | 1 / 0 | 0 / 0 |

| `max_book_fraction` | 0.010 | 0.025 | 0.05 | **0.075** | 0.25 | 1.00 |
|---|---|---|---|---|---|---|
| planted / clean | **3 / 31** | 1 / 11 | 1 / 0 | **1 / 0** | 1 / 0 | 0 / 0 |

Three things this measures that an argument cannot:

1. 🚨 **No price threshold ever catches the size slip and no size threshold ever catches the price
   typo.** Every row above tops out at `planted = 1`. They are different failures; a system with
   one "fat finger check" has one of them.
2. 🚨 **The only way to catch the replay with a size cap is to stop the book.** ✅ Measured: a
   `max_notional` set just under the replay's own 116,363 catches it and **blocks 29 of the clean
   37 as well**; the `max_book_fraction = 0.010` row is the same trade in the sweep — `planted =
   3`, `clean = 31`. The duplicate window catches that ticket at zero cost, because it is asking
   a different question.
3. **Tightening past the point where `clean` leaves zero buys nothing.** From 750k down to 250k,
   `max_notional` catches the same one mistake and adds 11 false positives.

⚠️ These counts are properties of *this* seeded blotter, not of your flow. The reusable part is
the shape of the exercise: plant the mistakes you are actually afraid of, then sweep.

## 5. ✅ The session comes from a DECLARED calendar, never the clock

`check_session` reads `state.now` and `state.sessions` and nothing else about time. There is **no
call to `datetime.now()`, `Timestamp.now()`, `date.today()` or `time.time()` anywhere in the
module** — proved by an AST walk in `tests/test_core_pre_trade.py` (a grep would trip over the
docstrings that mention them to say they are unused) and by a behavioural test that replaces the
module's `datetime` import and `pd.Timestamp.now` with objects that raise, then asserts the
verdict is byte-identical.

🚨 The reason is the one `../market-data-sourcing/SKILL.md` documents: `exchange_calendars`
derives `GLOBAL_DEFAULT_START` / `GLOBAL_DEFAULT_END` from `pd.Timestamp.now()` **at import**, so
anything built from its defaults answers a different question tomorrow. `fin_skills.engine.spec`'s
`Sessions` refuses to have a default for exactly this reason, and this check consumes it.

✅ Source-verified at **nyse.com, "Holidays & Trading Hours"** (read 2026-09-10): core session
**9:30 a.m. – 4:00 p.m. ET**; **Core Open Auction at 9:30 a.m. ET**; **Closing Auction at 4:00
p.m. ET** with a **Closing Imbalance Period 3:50 – 4:00 p.m. ET**; early closes at **1:00 p.m.
ET** — in 2026 **Jul 3, Nov 27 and Dec 24**. ⚠️ The 9:30–9:31 opening *window* in `NYSE_HOURS` is
this file's representation of "the opening print"; NYSE publishes the auction time, not a range.

⚠️ Secondhand: `SSE_HOURS` (09:15–09:25 call auction, 09:30–11:30 and 13:00–14:57 continuous,
14:57–15:00 closing auction, lunch 11:30–13:00) is taken from this repo's own
`../../../fin-china/skills/china-trading-stack/references/_ashare-rules.md` and was not re-read at
the exchange on 2026-09-10.

✅ Measured, same clean ticket, only `now` changing:

| declared `now` (UTC) | ET | codes |
|---|---|---|
| 2026-09-10 14:05 | 10:05 | `open` |
| 2026-09-10 13:15 | 09:15 | 🚨 `outside_hours` |
| 2026-09-10 13:30:30 | 09:30 | ⚠️ `opening_auction`, `open` |
| 2026-09-10 19:55 | 15:55 | ⚠️ `closing_auction`, `open` |
| 2026-09-10 20:05 | 16:05 | 🚨 `outside_hours` |
| 2026-09-12 14:05 | Sat | 🚨 `not_a_session` |
| **2026-11-27 18:05** | **13:05** | ⚠️ `half_day` + 🚨 `outside_hours` |
| 2026-11-25 18:05 | 13:05 | `open` — the *same clock time*, a normal session |

🔑 **The last two rows are the whole point of a declared calendar.** 13:05 ET is inside the
session on 25 November and past the close on 27 November, and no amount of reasoning about the
current time can tell them apart. `Sessions.half_days` can.

✅ Source-verified at **luldplan.com** (read 2026-09-10), which is why a 10% price-through
threshold is not arbitrary: the **Limit Up-Limit Down** price band for a **Tier 1** NMS stock
above $3.00 is **5%** (20% for $0.75–$3.00, the lesser of $0.15 or 75% below $0.75), **Tier 2**
above $3.00 is **10%**, and the bands **double in the last 25 minutes** of the session for all
Tier 1 names and for Tier 2 names below $3.00. A Tier 1 ticket 10% through the last trade is
outside a band twice its width — it could not have printed even if it had been meant. ⚠️
Secondhand and **not** modelled here: LULD Amendment 18 is reported to have removed the
9:30–9:45 doubling.

## 6. The kill switch is one boolean and a reason string

```python
KillSwitch(tripped=True, reason="cumulative loss -142,500.00 breached the stated limit of -100,000.00")
kill = trip_if(cum_loss=-142_500.0, max_loss=100_000.0)   # or error_rate=, or connected=False
```

No timers, no auto-reset, no severity ladder, no partial trip. Whoever reads it at 09:31 with
money on the line needs to know in one glance whether it is on and why; every additional feature
is a second thing they have to reason about. A tripped switch with an empty reason is a
`ValueError` at construction.

✅ Measured: with the switch tripped, **the same clean ticket that was allowed above returns
`allowed=False` with 6 blocks from 6 checks, all carrying one identical message** — and it stays
blocked for a `TradingState()` with no facts in it at all. Each check function fails closed on
its own, so there is no path that reaches a green light by calling one of them directly.

## 7. Exposure, and why the reconciliation runs first

`check_exposure` computes the *resulting* position and tests it against stated `gross_cap`,
`net_cap`, `name_cap` and per-sector caps. Before any of that, it reconciles your book against
the broker's.

✅ Measured — the broker reporting 300 fewer shares than your book:

```
BLOCK exposure/unreconciled: your book says -100 AA01 and the broker says -400 - a +300 share
disagreement, above the stated tolerance of 0.
```

🚨 **A resulting position computed from a book the venue does not share is arithmetic on a
fiction**, and the caps below it would all be quoting that fiction to four decimal places. The
broker's side comes in read-only from `../broker-execution-apis/SKILL.md`'s bridge:

```python
from fin_skills.bridges.execution import read_execution
from fin_skills.core.pre_trade import broker_positions_from
rec = read_execution("ib", client=ib)                 # reads; cannot send
state = TradingState(..., broker_positions=broker_positions_from(rec.positions))
```

The cost consequence of a participation rate is imported from
`../execution-cost-analysis/scripts/cost_plausibility.py` — Almgren, Thum, Hauptmann & Li (2005) —
and never re-derived here; a test asserts the number is that function's own output. ✅ Measured
on `slip-qty`: 28.00% of ADV over one session is **71.1 bps** of modelled impact, about **62,313**
on that ticket, and the check says out loud that at 28% of ADV the model is being extrapolated
well past the 0.25%-to-a-few-percent band it was fitted on.

## 8. What the script gives you

`scripts/pre_trade.py` — pure functions, numpy + pandas only, ASCII, no network, no file writes.

| Function | Does |
|---|---|
| `ProposedOrder`, `TradingState`, `Caps`, `KillSwitch` | the four things you assemble; `Caps()` has no default threshold anywhere |
| `check_order(proposal, state[, checks]) -> OrderVerdict` | all six checks, one `allowed` boolean, `missing` |
| `check_fat_finger` / `check_participation` / `check_session` / `check_duplicate` / `check_exposure` / `check_kill_switch` | the six, individually; each fails closed on a tripped switch |
| `trip_if(cum_loss=, max_loss=, error_rate=, max_error_rate=, connected=)` | evaluate stated conditions once; pure, reads no clock |
| `VenueHours`, `NYSE_HOURS`, `SSE_HOURS`, `LULD_TIER1_BAND` | §5's declared clocks and cited bands |
| `broker_positions_from(frame)`, `empty_blotter()` | the bridge's positions frame, and the fact that nothing has gone out |
| `seeded_blotter()`, `PLANTED`, `threshold_sweep(...)` | §3 and §4, reproducibly |
| `EXAMPLE_CAPS` | one filled-in set with the provenance of every number — copy it and argue with it |

`fin_skills.api.get("pre_trade")` wraps `check_order` in the guard interface. It is **deliberately
absent from the JSON tool surface** (`fin_skills.tools.schema.excluded()` names it and says why):
the facts a pre-trade gate rests on have to be assembled by the caller from things it *read*, and
a model composing them into a JSON blob is the failure the guard exists to prevent.

## Where this sits

- `../broker-execution-apis/SKILL.md` — the connection, and 🚨 `paper_account_guard`, which must
  pass **before** any of this: proving a session is paper is a different question from whether
  the order is sane, and only one of them is about the socket.
- `../execution-cost-analysis/SKILL.md` — the cost *after* the fills exist, and the
  `cost_plausibility` impact model this skill calls rather than copies.
- `../../../fin-strategies/skills/execution-algorithms/SKILL.md` — when §4 says a ticket needs 2.8
  sessions at a 10% participation of volume, that skill is where the schedule gets built.
- `../us-market-rules/SKILL.md` — whether the trade is permitted at all: SSR, Reg T, PDT, T+1.
- `../../../fin-china/skills/china-trading-stack/SKILL.md` — 🚨 the A-share checks this file does
  not do: price limits, limit-locked bars, suspension and T+1 sellability.
- `../portfolio-and-risk/SKILL.md` — where gross, net and sector limits come from before anyone
  states them as `Caps`.
- `../research-integrity-guards/SKILL.md` — the same posture applied to research: a guard that
  fails closed on a missing fact rather than passing quietly.
