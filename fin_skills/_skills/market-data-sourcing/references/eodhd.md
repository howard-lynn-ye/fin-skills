# eodhd

The only affordable source of a **survivorship-free equity universe** — and the one whose PyPI package
is **21 months behind its own repository**, with the maintainers admitting no commit corresponds to
what pip installs.

| | |
|---|---|
| pip | `eodhd` · **1.0.32 (2024-12-18)** — 🚨 stale, see below |
| Repo main | **1.4.0**, tagged **2026-08-28** ✅ (`v1.4.0`, commit "ci(release): first release workflow, publishing 1.4.0 to PyPI via OIDC") |
| GitHub | `EodHistoricalData/EODHD-APIs-Python-Financial-Library` — **90★**, 22 forks, 8 open issues, pushed 2026-08-28 |
| Licence | **MIT** ✅ (PyPI classifier + GitHub) |
| Python | `requires_python` is **null** ⚠️ — no floor declared; pip will install it anywhere |
| Maintenance | ⚠️ **Repo active, PyPI abandoned for 20 months.** Vendor-maintained client for a paid API |

## 🔑 Why it is on this list at all — the survivorship-free universe

✅ Verified in source on main, `eodhd/apiclient.py`:

```python
def get_list_of_tickers(self, code: str, delisted: int = 0, include_delisted: bool = False):
    # delisted: 0 = listed only, 1 = delisted only. Ignored if include_delisted=True.
    # include_delisted: True -> returns both listed and delisted tickers  (NEW in 1.4.0)
```

`get_list_of_tickers(code, delisted=1)` returns **only the delisted names for an exchange**. Union it
with `delisted=0` (or use `include_delisted=True` on main) and you have a **point-in-time-complete
listing universe** — the thing every free source silently withholds.

**This is the decisive capability.** Per `_decision-table.md`, every other cheap source —
yfinance, yahooquery, stockdex, defeatbeta, financedatabase, financetoolkit, Alpha Vantage,
findatapy equities — shows you **only what is listed today**. A screen backtested on a today-snapshot
universe never buys Lehman, Enron, Bear Stearns or any of the thousands of quieter failures, and its
Sharpe is fiction.

🚨 **But `delisted=1` is a paid endpoint.** The free tier does not include it. ✅ The free tier is
**20 calls/day** (plus a 500-call welcome bonus) with **1 year of history** and **no delisted data** —
i.e. the free tier gives you exactly the survivorship-biased universe you came here to escape. The
survivorship-free universe is a paid feature; budget for it or use CRSP/Norgate instead.

## 🚨 Trap 1 — PyPI is 21 months behind main, and there is no commit for what you installed

This is the trap that wastes a day. ✅ Verified at both primary sources — the published package is
`1.0.32 (2024-12-18)`; repo main is `1.4.0 (2026-08-28)`.

Their own `CHANGELOG.md`, ✅ verbatim:

> "Twenty months of work reached `main` without being published, so this release covers all of it
> rather than a single feature."

and, more alarming:

> "`setup.py` never carried `1.0.32` — it went 1.0.31 → 1.3.2 → 1.4.0 — so there is **no commit or tag
> in this repository corresponding to the last published package**."

**Consequences:**

- **You cannot read the source of what pip gave you.** Browsing the repo to understand `1.0.32`
  behaviour shows you code that was never in it. Every "the docs say it does X" conclusion is unsound.
- **Bugs fixed on main are still live in the installed package.** Two examples from the 1.4.0
  changelog, both ✅ verified and both real in `1.0.32`:
  - 🚨 **WebSocket control frames were treated as market data.** Nothing filtered them, so the
    `{"status_code": 200, "message": "Authorized"}` ack *and* a `{"status": 403}` auth denial were
    stored, printed, and **fed to the candle builder**. A wrong API key produced a stream of nonsense
    rather than an error — and auth failures arrive over the socket *after* a successful handshake, so
    opening the connection never proved the token was good.
  - 🚨 **European symbols could not be subscribed at all.** The validator was `^[A-z0-9-$]{1,48}$`,
    which contains no dot, so `GSK.LSE`, `SAP.XETRA` and `ASML.AS` were rejected client-side before a
    socket opened. (`[A-z]` also admits the ASCII punctuation between `Z` and `a`.)
- Everything added in 1.4.0 is **absent from pip**: the seven extra WebSocket markets, options,
  index components, `include_delisted`, Exchange Details v2, Fundamentals v1.1.

**Install from git**, pinned to a tag:

```
pip install "eodhd @ git+https://github.com/EodHistoricalData/EODHD-APIs-Python-Financial-Library@v1.4.0"
```

⚠️ 1.4.0 is *tagged* but was published under a brand-new, first-ever CI workflow — check PyPI before
assuming the git pin is still necessary.

## 🚨 Trap 2 — the delisted list is a *listing* universe, not a *price* history

`get_list_of_tickers(delisted=1)` gives you the **symbols**. You still have to pull each one's price
history and fundamentals separately, on the same paid plan, respecting the rate limit. Two follow-on
hazards:

- **Ticker reuse.** A delisted ticker is recycled to a different company. Join on the exchange-scoped
  code plus the delisting date, never on the bare symbol.
- ⚠️ **Completeness is unverified.** ❓ I have no primary-source confirmation of how far back the
  delisted list runs per exchange, or whether it covers pre-electronic-era names. Validate against a
  known list of failures for your exact universe and period before trusting it.

## Minimal correct call

```python
from eodhd import APIClient

client = APIClient(api_key="...")   # delisted data requires a PAID plan; free tier = 20 calls/day

listed   = client.get_list_of_tickers("US", delisted=0)
delisted = client.get_list_of_tickers("US", delisted=1)   # 🔑 the whole point; paid-only
universe = listed + delisted                              # survivorship-free listing universe

# On repo main (>=1.4.0) this is one call:
# universe = client.get_list_of_tickers("US", include_delisted=True)
#
# pip's 1.0.32 predates include_delisted, predates the WebSocket control-frame fix,
# and rejects any symbol containing a dot on the WS API. Install from the v1.4.0 tag.
```

## Related

- `_decision-table.md` — the full delisted-coverage matrix and every source's free-tier limit.
- `yfinance.md` — the survivorship-biased default, and why `YFTickerMissingError` *guesses* delisting.
- `openbb.md` — an abstraction that does **not** normalize survivorship characteristics across
  providers, so it cannot substitute for this.
