# vnpy

The de-facto standard framework for live Chinese-market trading — MIT, 45k stars, the broadest gateway
coverage in the ecosystem — and it **ships no market data of its own**. Every datafeed is a paid
third party you must supply.

| | |
|---|---|
| pip | `vnpy` · **4.4.0 (2026-05-14)** · 29 releases |
| GitHub | `vnpy/vnpy` — **45,111★**, 34 open issues, pushed 2026-09-01 ✅ |
| Licence | **MIT** ✅ (PyPI classifier + GitHub SPDX agree) — genuinely permissive, unlike RQAlpha |
| Python | **`requires_python >=3.10`** ✅ · core is a pure-python wheel + sdist |
| Architecture | Event engine + `MainEngine` + pluggable **gateways** (brokers) and **apps** (strategies, recorder, risk) |
| Maintenance | ✅ Active, small backlog for its size |

**vnpy is many PyPI distributions, not one.** The core `vnpy` package contains no broker and no data.
Everything real is a separate `vnpy_*` install.

## 🚨 Traps

**1. 🚨 It is a framework, not a data source.** Every datafeed adapter wraps a third-party, mostly paid
service — ✅ all verified live on PyPI, all MIT, all `requires_python >=3.10`:

| Adapter | Provider | Cost | Latest ✅ |
|---|---|---|---|
| `vnpy_xt` | 迅投研 | Paid | 1.4.6 (2025-10-18) |
| `vnpy_rqdata` | 米筐 RiceQuant | Paid | 3.4.7.8 (2026-04-16) |
| `vnpy_tushare` | Tushare | 积分 | 1.4.21.0 (2025-06-11) |
| `vnpy_tqsdk` | 天勤 | Paid | 3.8.6.0 (2025-10-02) |
| `vnpy_wind` / `vnpy_ifind` | 万得 / 同花顺 | Paid | 1.1.0 (2025-06-11) |
| `vnpy_udata` / `vnpy_tinysoft` | 恒生 / 天软 | Paid | ⚠️ **2023 — stale** |

✅ **`vnpy_datayes` (通联) does not exist on PyPI** — HTTP 404 on the JSON API. If a tutorial or doc
lists it, that doc is wrong.

Config is uniform: `SETTINGS["datafeed.name"]` / `["datafeed.username"]` / `["datafeed.password"]`.
⚠️ **For RQData these are not your ricequant.com website login** — they are separate API credentials.

**2. 🚨 `vnpy_ctp` is a compiled extension with a Windows-only wheel.** ✅ 6.7.11.4 (2026-03-29) ships
`win_amd64.whl` + sdist and **nothing else** — no manylinux, no macOS. On Linux you build from source
against the broker's C++ CTP API. Plan the deployment target before writing the strategy; the pure
Python apps (`vnpy_ctastrategy` 1.4.1, `vnpy_datarecorder` 1.1.1) install anywhere.

**3. 🚨 SimNow and production differ only by broker ID and front address.** A config flag is not proof
of which one you reached. **Assert on a server-returned account field** after login, and fail closed:

```python
acct = main_engine.get_all_accounts()[0]
assert acct.accountid.startswith(SIMNOW_PREFIX), f"REFUSING: {acct.accountid} is not simulation"
```

**4. 🚨 The CTA backtester will not encode A-share rules for you.** T+1 settlement, board-dependent
涨跌停, ST's ±5%, 停牌, the 90-minute lunch break and 印花税 on the sell side only are all your
responsibility. ❓ The precise fill and cost model of `vnpy_ctastrategy` was not source-verified here —
**read it before trusting a number**, and check every rule in `_ashare-rules.md` against it.

**5. ⚠️ The lunch break and 夜盘 break naive bar handling.** A normal equity session is exactly **240
minute bars** across 09:30–11:30 and 13:00–15:00; futures 夜盘 means a trading day starts the previous
calendar evening. Roll windows over **bar index, never wall clock**. `../SKILL.md` §3.

**6. ⚠️ Version coupling across the `vnpy_*` fleet.** Each gateway and app versions independently, and
two of the datafeed adapters have not shipped since 2023. Pin the whole set together and re-test as a
unit; a core upgrade can silently orphan an adapter.

## 🔑 The under-appreciated capability: build your own tick history for free

**`vnpy_datarecorder`** ✅ (1.1.1, 2025-06-16, MIT) records live tick and bar data from a *connected
trading gateway* into a local database. With a broker CTP account you therefore have a legal,
free, ToS-clean tick feed — no vendor subscription, no scraping, and the data is exactly what your
execution path saw.

The catch is that it only records forward from the day you start. Start it before you need it.

## Minimal safe wiring

```python
from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.setting import SETTINGS
from vnpy_ctp import CtpGateway
from vnpy_datarecorder import DataRecorderApp

SETTINGS["datafeed.name"] = "rqdata"        # separate paid credentials, NOT the website login
SETTINGS["datafeed.username"] = "..."
SETTINGS["datafeed.password"] = "..."

ee = EventEngine()
me = MainEngine(ee)
me.add_gateway(CtpGateway)                  # win_amd64 wheel only; build from sdist on Linux
me.add_app(DataRecorderApp)                 # free tick history from your own connection
me.connect(ctp_setting, "CTP")
# then assert on a server-returned account id before any order is sent
```

## Where it fits

- Framework selection, Qlib's dataset situation, order safety: `../SKILL.md`
- The machine-checkable A-share rule list your strategy must satisfy: `_ashare-rules.md`
- Research pipeline that feeds it: `qlib.md`
- Where the data comes from: `../../china-ashare-data/SKILL.md`, and specifically
  `../../china-ashare-data/references/tushare.md` for the `vnpy_tushare` adapter's 积分 tiers
- Broker-agnostic order-safety patterns:
  `../../../../fin-core/skills/broker-execution-apis/SKILL.md` §3
