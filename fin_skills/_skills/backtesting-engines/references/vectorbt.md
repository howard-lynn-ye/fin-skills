# vectorbt

The fastest thing in this category for parameter sweeps — and the owner of **the single most
important footgun in the domain**, which is a default, not a bug.

| | |
|---|---|
| pip | `vectorbt` · **1.1.0 (2026-07-05)**; **1.0.0 landed 2026-04-22** |
| GitHub | `polakowo/vectorbt` — 8,978★, 121 open issues, pushed 2026-08-02 |
| Licence | 🚨 **Apache-2.0 + Commons Clause** (read in `LICENSE.md`) — **not OSI open source** |
| Status | ✅ **revived.** The OSS edition was widely believed frozen behind PRO. It is not |

🔑 **v1.0 is a significant revival**, not a maintenance bump: an optional **Rust engine**
(`pip install vectorbt[rust]`) with auto-dispatch between Numba and Rust per call, plus `FlexArray`
zero-copy broadcast. 1.1.0 added Python 3.14 / pandas 3 / NumPy 2.4 support.

⚠️ **0.28.x → 1.0.0 is a breaking rewrite.** Pinned code will not carry over unchanged.

## 🚨 Licence — Commons Clause

The Commons Clause is an addendum that removes the right to **sell** the software. Verbatim effect:
**you may not sell a product or service whose value derives substantially from vectorbt, including
hosting or support services.** Internal research and personal use are unaffected. A SaaS is not.

**VectorBT PRO** is closed source and subscription-based. ⚠️ Public pricing circa 2026 was
**$25/month, or $20/month billed annually; lifetime access from a $150 floor with monthly payments
credited at 50%** — **[UNVERIFIED]**, taken from a search result rather than a direct fetch, and
pricing pages change. PRO adds chunking, data ingest, richer records and pattern search, and **most
PRO tutorials assume it**, which is a frequent source of "this example doesn't work" on the OSS build.

## 🚨 The same-bar fill default

`Order.price` defaults to `np.inf`, and vectorbt's own docstring in
`vectorbt/portfolio/enums.py` says:

> *"If `-np.inf`, replaced by the current open (if available) or the previous close (≈ the current
> open in crypto). If `np.inf`, replaced by the current close."*

and `from_orders`/`from_signals` resolve `if price is None: price = np.inf`. Therefore:

```python
pf = vbt.Portfolio.from_signals(close, entries, exits)   # fills at close[t] — the signal's own bar
```

Combined with the equally default idiom `fast_ma.ma_crossed_above(slow_ma)` — also computed on
`close[t]` — **this is textbook same-bar execution.** The close is not knowable until the bar is over.
Free money in backtest, nothing in live.

**Every vectorbt tutorial showing a beautiful equity curve without shifting is showing a biased
result.** The library will never warn you.

### Fixes, in order of preference

```python
# 1. Fill at the next open, with signals shifted so the open is AFTER the signal bar
pf = vbt.Portfolio.from_signals(
    close,
    entries.vbt.signals.fshift(1),
    exits.vbt.signals.fshift(1),
    price=open_,
)

# 2. Shift the signals only (still fills at close, but a later one)
# 3. price=-np.inf  -> current open; correct ONLY if the signal is also open-based
```

## What it models correctly

Percentage and fixed fees · percentage slippage · min/max size · size granularity · **partial fills
by cash** (`allow_partial`) · **order rejection probability** (`reject_prob`) · cash sharing across a
group · long/short/both directions · SL/TP/trailing stops with configurable `stop_entry_price` /
`stop_exit_price`.

## What it fakes or ignores

**No order book. No queue position. No latency. No partial-fill-by-liquidity** (only by cash).
**No corporate actions. No borrow cost. No margin call. No concept of an asset's lifetime** — a frame
built from today's index members is silently survivorship-biased and vectorbt cannot tell.

Stops are evaluated against OHLC within a bar with a **configurable but crude** conflict mode
(`stop_conflict_mode="exit"` by default) — intrabar high/low ordering is guessed, as in every bar
engine.

## Where it is the right tool

**Searching a parameter space.** Whole 2-D grids of parameters simulate as one array operation; it is
orders of magnitude faster than any event-driven engine for that job.

**The discipline that makes it safe:** sweep with vectorbt, then **re-run the survivor in an
event-driven engine** (nautilus_trader, zipline-reloaded, LEAN) and check the numbers agree. When they
disagree, vectorbt is usually the optimistic one — and the gap is your bias estimate.

🚨 And remember that the sweep itself is a **trial count**. A 10,000-combination grid has
`n_trials = 10,000` for Deflated Sharpe purposes — see `../../backtest-validation/SKILL.md`.
