# Schwab, Tastytrade, Robinhood, and FIX

Everything after Interactive Brokers and Alpaca. One of these has a constraint that **rules out
unattended automation entirely**, and it is the one most people reach for.

| Project | PyPI | Version | Released | ★ | Licence | Verdict |
|---|---|---|---|---:|---|---|
| `alexgolec/schwab-py` | `schwab-py` | 1.5.1 | 2025-06-30 | 468 | MIT | ⚠️ **stalled**, 88 open issues |
| `alexgolec/tda-api` | `tda-api` | 1.6.0 | **2022-06-07** | 1,323 | MIT | 🔴 **DEAD** |
| `tastyware/tastytrade` | `tastytrade` | 13.2.3 | 2026-08-07 | 256 | MIT | ✅ active, small |
| `jmfernandes/robin_stocks` | `robin_stocks` | 3.4.0 | 2025-05-18 | 2,120 | MIT | ⚠️ limping, **322 open issues** |
| `robinhood-unofficial/pyrh` | `pyrh` | 2.1.2 | **2023-03-04** | 1,789 | MIT | 🔴 dead |
| `quickfix/quickfix` | `quickfix` | 1.16.0 | 2026-05-09 | 1,987 | Custom QuickFIX | ⚠️ **sdist only** |
| `da4089/simplefix` | `simplefix` | 1.0.17 | 2023-09-12 | 255 | MIT | ✅ stable, **no session layer** |

---

## 🚨 Schwab — the 7-day wall

**The defining constraint, and it is not a library problem:** Schwab's **refresh token hard-expires
after 7 days**, and **nothing can extend it**.

- Access tokens last **30 minutes**; `schwab-py` refreshes them automatically into `token_path`.
- The **refresh** token cannot be rolled over, re-issued programmatically, or kept alive by activity.
  After ~7 days you must complete an **interactive browser login** again.
- Schwab enforces this server-side, and it can fire slightly sooner or later than exactly 7 days.

**Therefore: fully unattended long-running Schwab automation is not possible under the current OAuth
model.** Not "awkward" — not possible. Any Schwab design must budget for a human at a browser every
week, or it will halt mid-week with an expired grant. Encode this before writing a line of code.

🚨 **There is no Schwab sandbox.** No paper environment, no test host. Every credential you hold is a
live credential, which inverts the safety pattern that works at IB and Alpaca
(`./interactive-brokers.md`, `./alpaca.md`). Assert on the **account hash** returned by the accounts
endpoint and gate orders behind an environment variable, because there is no server-side flag that
proves you are safe.

OAuth uses a **loopback callback**, commonly `https://127.0.0.1:8182`.

⚠️ **Maintenance:** `schwab-py` 1.5.1 shipped 2025-06-30; ✅ the last commit on `main` is
**2025-08-04** ("Update patreon info") — roughly **13 months** with no substantive change — while
**88 issues stay open** and new ones keep arriving. The library still works; the backlog does not
shrink. Single-maintainer risk, same shape as `ib_async` in `./interactive-brokers.md`, but with a
longer stall.

🔴 **`tda-api` is dead and cannot be revived.** Last release 2022-06-07, repo untouched since
2024-06-16. **TD Ameritrade was absorbed into Schwab and the API it wrapped no longer exists.** Any
tutorial, StackOverflow answer, or LLM-generated snippet naming `tda-api` is targeting a dead
endpoint. This is the single most common stale recommendation in US retail broker automation.

---

## ✅ Tastytrade — small but healthy

`tastytrade` 13.2.3 (2026-08-07), MIT, Python **`>=3.11`** with classifiers through 3.14, **3 open
issues**, released the same day as its last commit. A 256-star project with a 3-issue backlog is a
different risk profile from a 2,120-star project with 322 — **stars measure attention, open issues
measure debt, and only the second one predicts whether your bug gets fixed.**

Strong for **options** specifically. The trade-off is bus factor: a small library maintained well is
a bet on one maintainer continuing.

---

## 🚨 Robinhood — recommend against

`robin_stocks` and `pyrh` both drive **reverse-engineered private endpoints**. That produces four
problems at once:

1. **No paper trading.** None. There is no safe environment in which to test an order path.
2. **322 open issues** on `robin_stocks` — the endpoints move and the library follows late.
3. **MFA / device-token friction** on every session.
4. 🚨 **Account lockouts are the reported failure mode.** Not an error response — a locked account.

`pyrh` is dead: PyPI 2.1.2 dates to **2023-03-04**, repo last pushed **2024-08-08**.

Use for personal, read-only tinkering at most. Anything that submits an order belongs at a broker
with a documented API and a paper environment.

---

## FIX — `quickfix` vs `simplefix`

🚨 **They solve different problems and the names do not tell you which.**

| | `quickfix` | `simplefix` |
|---|---|---|
| What it is | The C++ QuickFIX engine with Python bindings | **Message parsing/encoding only** |
| Session layer | ✅ Full state machine, logon/logout, heartbeats | 🚨 **None** |
| Sequence numbers | ✅ Message store, gap fill, resend, recovery | 🚨 **None** |
| Reconnection | ✅ | 🚨 **None** |
| Install | 🚨 **sdist only — no wheels at all.** A C++ toolchain is required on every platform | ✅ `py2.py3-none-any` wheel |
| Licence | Custom QuickFIX licence (GitHub reports `NOASSERTION`) | ✅ MIT |

✅ Verified from the PyPI file list: `quickfix` 1.16.0 ships **one file, `quickfix-1.16.0.tar.gz`**.
There is no wheel for any platform. `pip install quickfix` compiles, and it fails on any machine
without a working C++ build environment.

**Choose `simplefix`** when the venue or a broker-supplied gateway owns the session and you only need
to build and read FIX messages. **Choose `quickfix`** when you must be a FIX **initiator** —
because a session layer is not something you write in an afternoon, and getting sequence-number
recovery wrong means duplicate orders after a disconnect.

⚠️ Reality check: **FIX access at retail brokers is essentially nonexistent.** This is
institutional / prime-broker territory. If you are evaluating FIX for a retail account, the answer
is almost certainly a REST API instead.

---

## The pattern that survives all of them

No paper environment (Schwab, Robinhood) means the gate has to live in **your** code:

```python
import os
LIVE = os.environ.get("TRADING_LIVE") == "1"    # env beats config: config files get committed

def submit(order):
    if not LIVE:
        return log.info("DRY RUN %s", order)
    assert account_hash == EXPECTED_HASH, "REFUSING: unexpected account"
    return client.place_order(account_hash, order)
```

Plus deterministic client order IDs, a hard daily notional cap in code, startup reconciliation that
**halts** on mismatch rather than auto-flattening, and a kill switch you have actually triggered
once. See `./alpaca.md` for the idempotency pattern and `./_broker-matrix.md` for the full metadata.
