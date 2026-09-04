# openbb

One normalized Python schema over ~30 vendors. The convenience is real; the two costs are a
**network-copyleft licence** and an abstraction that **hides exactly the differences that decide
whether a backtest is honest**.

| | |
|---|---|
| pip | `openbb` · **4.7.2 (2026-05-26)** |
| GitHub | `OpenBB-finance/OpenBB` — **72,668★**, 7,504 forks, 115 open issues, pushed 2026-07-30 |
| Licence | 🚨 **AGPL-3.0-only** ✅ (PyPI `info.license` = `AGPL-3.0-only`; classifier = "GNU Affero General Public License v3"). GitHub reports `NOASSERTION` — ignore it, PyPI metadata is explicit |
| Python | `>=3.10,<4` · wheel + sdist |
| Maintenance | ✅ Healthy and actively developed |
| `openbb-terminal` | 🚨 **404 on PyPI** ✅ — removed, not renamed. The Terminal is a separate commercial product now |

## 🚨 Trap 1 — AGPL-3.0 is network copyleft, and it was not always this way

✅ Verified at the primary source: PR **#6415, "Update the license of the code in this repo to AGPL"**,
merged **2024-05-14** by `piiq`.

**openbb was MIT before that date and is AGPL-3.0-only after it.** Consequences:

- **AGPL's §13 triggers on network use, not distribution.** If your users interact with openbb over a
  network — a Streamlit dashboard, an internal FastAPI research service, a Slack bot, an MCP server —
  the AGPL requires you to offer *those users* the corresponding source of your whole combined work.
  Merely "not shipping a binary" does not exempt you.
- 🚨 **The MCP server extension inherits it.** `openbb_platform/extensions/mcp_server/` is a
  first-party extension inside the same AGPL repo. Wiring an LLM agent to openbb over MCP is precisely
  the network-interaction case AGPL is written for.
- ⚠️ **A pre-2024-05-14 pin does not launder it.** Old MIT-licensed versions exist, but they are years
  stale and you cannot take later fixes without taking the AGPL.
- This is a **legal review question, not an engineering preference.** For a proprietary fund codebase,
  treat openbb as a non-starter until counsel signs off.

Every `openbb-*` provider extension is in the same repo and carries the same licence:
✅ `openbb-benzinga` 1.6.1, `openbb-intrinio` 1.6.1, `openbb-fmp` 1.6.1, `openbb-tiingo` 1.6.1,
`openbb-polygon` 1.5.1, `openbb-nasdaq` 1.6.3, `openbb-sec` 1.6.7, `openbb-yfinance` 1.6.3,
`openbb-finviz` 1.5.1, `openbb-biztoc` 1.6.1.

## 🚨 Trap 2 — the normalized schema hides what actually matters

This is the silent-wrong-numbers trap. A uniform `obb.equity.price.historical(...)` returns the same
column names regardless of provider. **It does not make the numbers comparable.** The abstraction
normalizes *shape*, never *semantics*:

| Hidden behind the same call | Why it silently corrupts results |
|---|---|
| **Adjustment convention** | Whether OHLC are split-adjusted, split-and-dividend-adjusted, or raw differs per provider. Switching provider silently changes every return in your panel. Compare this to `yfinance.md`'s `auto_adjust` history — the same hazard, but here it is invisible because you never see the flag. |
| **Corporate-action handling** | Back-adjustment method, dividend timing (ex-date vs pay-date), and spin-off treatment are provider decisions surfaced nowhere in the schema. |
| **Delisted coverage** | ⚠️ **Provider-dependent, and the abstraction does not normalize survivorship characteristics.** A universe built through openbb may or may not be survivorship-biased and the call site gives you no way to tell. |
| **History depth & retention** | Free tiers truncate silently; the schema returns a short frame, not an error. |
| **Timestamp convention** | Exchange-local vs UTC, and bar-label convention, vary. |

**The rule: always pin `provider=` explicitly, and treat a provider change as a data migration —
re-run your validation suite, do not assume continuity.** Never rely on the default provider, which
is a config value that can differ between machines and between openbb releases.

## Where openbb is genuinely worth it

- **Breadth of one API surface** across equities, options, crypto, macro, news and filings — good for
  exploratory work and for LLM tool-calling, where a uniform schema is the whole point.
- ✅ The genuinely free, genuinely redistributable providers are **`openbb-sec`** (EDGAR — US
  government, public domain) and **`openbb-yfinance`** (⚠️ Yahoo scraping, against Yahoo's ToS — see
  `yfinance.md`).
- ⚠️ **Code licence ≠ data licence, twice over.** AGPL governs openbb's code; each provider's data
  carries its own separate terms (Benzinga's redistribution terms are strict; Tiingo's are
  personal-use).

## Minimal correct call

```python
from openbb import obb

# provider= is NEVER optional in research code: the default is a config value that
# varies by machine and by release, and providers differ in adjustment convention,
# corporate-action handling and delisted coverage behind this identical signature.
df = obb.equity.price.historical(
    symbol="AAPL",
    start_date="2020-01-01",
    end_date="2024-01-01",
    provider="yfinance",      # pin it; record it in your run manifest
    interval="1d",
).to_df()

# This universe's survivorship characteristics are a property of the provider,
# not of openbb. openbb does not normalize them and will not warn you.
```

## Verdict

⚠️ **Ship only with a documented licence caveat.** Excellent for exploration, notebooks, and
open-source or internal-non-networked work. For a proprietary, network-served production system,
AGPL-3.0-only is a blocking constraint that no amount of engineering discipline removes.

## Related

- `_decision-table.md` — the delisted-security axis and free-tier limits across all sources.
- `yfinance.md` — the underlying free source for `openbb-yfinance`, and its adjustment-default history.
- `eodhd.md` — the one cheap survivorship-free universe, which openbb's abstraction cannot give you.
