# findatapy

Two capabilities nothing else on this list has: **free tick-level FX from DukasCopy** and
**ALFRED point-in-time macro vintages**. Both are worth real money. The author's own README calls the
package *"a highly experimental alpha project"*.

| | |
|---|---|
| pip | `findatapy` · **0.1.42 (2026-03-20)** · cadence 0.1.39/0.1.40 (2025-03-08) → 0.1.41 (2026-01-02) → 0.1.42 |
| GitHub | `cuemacro/findatapy` — **2,114★**, 221 forks, 28 open issues, pushed 2026-07-02 |
| Licence | **Apache-2.0** ✅ (GitHub SPDX; PyPI `license` field reads "Apache 2.0", no classifier ⚠️) |
| Python | `requires_python` is **null** ⚠️ — no declared floor |
| Maintenance | ⚠️ **Single-author, slow cadence, self-described alpha.** Alive, not abandoned |

⚠️ **Version numbering is a warning in itself.** After ~10 years it is still `0.1.x`. The author, Saeed
Amen (`cuemacro`), also maintains `finmarketpy` and `chartpy`; findatapy is the market-data layer of
that trio.

## 🔑 Capability 1 — free tick FX from DukasCopy

The only library here that hands you **genuine tick-level FX for free**, through the same
`MarketDataRequest` interface as everything else:

```python
md_request = MarketDataRequest(start_date="...", finish_date="...",
                               category="fx", fields=["bid", "ask"], freq="tick",
                               data_source="dukascopy", tickers=["EURUSD"])
```

⚠️ **What "free tick FX" is and is not.** DukasCopy is a *retail broker's own* feed. It is a real
quote stream, not a synthetic one, and it is excellent for microstructure practice, execution-cost
modelling and vol estimation. It is **one venue's view of an OTC market with no consolidated tape** —
your bids and asks are DukasCopy's, not the interbank market's. Do not treat it as a reference price
for valuation or as a substitute for a prime broker's fills.

⚠️ **Historical downloader fragility.** The changelog carries repeated entries like "Fixed Dukascopy
downloader" and "Added 404 error for downloading from Dukascopy" — the scraper breaks when the
upstream file layout moves, exactly as `yfinance.md` describes for Yahoo. Budget for that.

## 🔑 Capability 2 — ALFRED vintages: real point-in-time macro

**This is the rarer of the two.** FRED gives you *today's* value of a macro series, silently
incorporating every subsequent revision. **ALFRED** (ArchivaL FRED) gives you the series **as it was
published on a past date** — the actual number a strategy could have seen.

```python
md_request = MarketDataRequest(start_date="year", category="fx",
                               data_source="alfred", tickers=["AUDJPY"])
```

🚨 **Why this is a correctness issue, not a nicety.** US nonfarm payrolls, GDP and industrial
production are revised for *years* after first print, sometimes by more than the surprise that moved
markets. A macro strategy backtested on FRED's current vintage is trading on numbers that did not
exist at the trade date — the macro equivalent of look-ahead bias, and it is invisible because the
series looks perfectly well-formed. **findatapy is the only library in `_decision-table.md` offering
point-in-time macro at all.**

⚠️ ALFRED coverage is per-series: not every FRED series has vintages, and vintage depth varies.
❓ I have not verified how findatapy behaves when you request a vintage a series does not have —
check for a silent fallback to the current vintage before trusting it.

## 🚨 Trap 1 — the Redis caching layer errors on Windows

✅ Verbatim from the README:

> "You might often get an error like the below, when you are downloading market data with findatapy,
> and you don't have Redis installed: `Couldn't push MarketDataRequest`"

> "Redis is available for Linux. There is also an unsupported (older) Windows version available, which
> I've found works fine, although it lacks some functionality of later Redis versions."

**The failure is loud but non-fatal:** without Redis, caching fails and findatapy **always** re-fetches
externally. So the practical Windows cost is not a crash — it is that every request hits the vendor,
which matters a great deal when the vendor is rate-limited or when you are pulling tick FX.

**Treat the noise as expected on Windows**, and do not let a `Couldn't push MarketDataRequest` in the
logs send you debugging a data problem that does not exist.

## 🚨 Trap 2 — "alpha project" is the author's own assessment, not a disclaimer

✅ Verbatim from the README:

> "Please bear in mind at present findatapy is currently a highly experimental alpha project and isn't
> yet fully documented"

Read literally and act on it: **pin the exact version**, snapshot anything you pull, and write your own
assertions on the returned frame's shape, index monotonicity and timezone. Do not build an
unsupervised production pipeline on it. The two unique capabilities justify using it as a **research
data acquisition tool**, not as infrastructure.

## Trap 3 — equities are survivorship-biased

⚠️ findatapy's equity paths inherit their sources' universes and carry **no delisted coverage** (see
`_decision-table.md`). Its value is FX tick and ALFRED macro; do not reach for it for an equity
backtest universe. For that, see `eodhd.md`.

## Minimal correct call

```python
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest

market = Market(market_data_generator=MarketDataGenerator())

# ALFRED = point-in-time vintages. Plain 'fred' gives you today's revised numbers,
# which is look-ahead bias in a well-formed disguise.
req = MarketDataRequest(
    start_date="2015-01-01", finish_date="2024-01-01",
    category="fx", data_source="alfred", tickers=["AUDJPY"],
    freq="daily",
)
df = market.fetch_market(req)

# A "Couldn't push MarketDataRequest" line on Windows means Redis is absent:
# caching is off and every call re-fetches. Not an error in your data.
assert df is not None and df.index.is_monotonic_increasing
```

## Related

- `_decision-table.md` — where findatapy sits against every other free source.
- `eodhd.md` — the survivorship-free equity universe findatapy does not provide.
- `../../fundamental-and-macro-data/` — restated-vs-point-in-time is the same trap in fundamentals.
