# A-share rules, as code

The machine-checkable version of the rules a Western backtest engine gets wrong. Encode these or the
backtest is fiction.

## Price limits by board

```python
from datetime import date

def daily_limit_pct(code: str, name: str, d: date) -> float:
    """Return the daily price-limit fraction for an A-share on a given date.

    `name` matters: ST status is a NAME change, not a code change, so the limit for
    the same ticker changes mid-history with no identifier change.
    """
    is_st = name.startswith(("ST", "*ST", "S*ST", "SST"))
    if code.startswith(("300", "301")):          # 创业板 ChiNext
        return 0.05 if is_st else 0.20           # ChiNext ST is still 5%
    if code.startswith("688"):                   # 科创板 STAR — no ST regime
        return 0.20
    if code.startswith(("8", "43", "87")):       # 北交所 BSE
        return 0.30
    return 0.05 if is_st else 0.10               # 主板 Main
```

⚠️ ChiNext moved to ±20% in **August 2020**; before that it was ±10%. If your backtest spans that
date, the function must take `d` into account. New listings have their own day-1 rules (ChiNext /
STAR / BSE are unlimited on the first day).

🚨 Do not derive limits as `prev_close × (1 ± pct)` and compare exactly — exchanges round to the
tick, so a bar can sit one tick inside the computed bound and still be limit-locked. Prefer a vendor
that supplies `high_limit` / `low_limit` (jqdatasdk, rqdatac, Wind, Choice) or use akshare's limit
pools (`stock_zt_pool_em`, `stock_zt_pool_dtgc_em`, `stock_zt_pool_zbgc_em`).

## Fill eligibility

```python
def can_buy(bar, limit_pct) -> bool:
    """A bar locked at limit-up generally could not have been bought."""
    if bar.volume == 0:                  # suspended, or no trading
        return False
    if bar.paused:                       # explicit flag where available
        return False
    up = round(bar.prev_close * (1 + limit_pct), 2)
    return not (bar.close >= up and bar.high == bar.low)   # locked at limit all day
```

Filling at close on limit-up days is **the second-largest source of fake alpha in A-share research**,
after survivorship.

## T+1 settlement

```python
def sellable_qty(position, today):
    """Shares bought today cannot be sold until the next session. Cash is T+0."""
    return sum(lot.qty for lot in position.lots if lot.trade_date < today)
```

**Any intraday-reversal strategy on A-shares is unimplementable.** Futures are **T+0** — a mixed
equity/futures backtest needs two settlement models, not one.

## Sessions

```
09:15–09:25  集合竞价 (opening auction)
09:30–11:30  morning continuous
11:30–13:00  LUNCH — no trading
13:00–14:57  afternoon continuous
14:57–15:00  closing auction (SZ; SH since 2018)
```

Exactly **240 minute bars** in a normal session. Timezone **Asia/Shanghai, no DST**.

🚨 **Never roll over wall-clock time.** `df.resample('1min')` or `df.rolling('30min')` silently spans
the 90-minute lunch gap. Roll over **bar index**:

```python
df["ma20"] = df.close.rolling(20).mean()          # correct: 20 BARS
df["ma20"] = df.close.rolling("20min").mean()     # WRONG: spans the lunch break
```

## Futures night session (夜盘)

Runs ~21:00 to 23:00 / 01:00 / 02:30 depending on contract, which means **a futures trading day
starts the previous calendar evening**. Mapping night bars to the wrong trading date is a classic
off-by-one. `tqsdk` and `rqdatac` (`get_trading_hours`, `night_trading_at_first_date`) handle it;
scrapers do not.

## Costs

```python
def round_trip_cost(notional_buy, notional_sell, commission_rate=0.00025, min_commission=5.0):
    """Stamp duty is SELL-SIDE ONLY — a symmetric cost model ported from US equities is wrong."""
    stamp = notional_sell * 0.0005          # 印花税, sell side only (rate has changed over time)
    transfer = (notional_buy + notional_sell) * 0.00001   # 过户费
    comm = sum(max(n * commission_rate, min_commission) for n in (notional_buy, notional_sell))
    return stamp + transfer + comm
```

⚠️ The stamp-duty rate has changed historically (it was halved to 0.05% in August 2023) — key it to
date if your backtest spans the change. The per-order commission **minimum** matters a lot for small
orders and is frequently omitted.

## Universe

```python
# tushare: list_status defaults to 'L' — omit it and your universe is silently survivorship-biased
live = pro.stock_basic(list_status="L", fields="ts_code,name,list_date,delist_date")
dead = pro.stock_basic(list_status="D", fields="ts_code,name,list_date,delist_date")
universe = pd.concat([live, dead])
```

Delisting in China went from rare to common after the 2020 退市新规, so **the survivorship bias is
time-varying** — worse than a constant bias, because it changes the apparent regime.

🚨 **Codes are reused and businesses are replaced.** Delisted codes have been reassigned, and 借壳上市
(reverse-merger backdoor listings) keep a code while replacing the entire underlying company: the
price series is continuous, the company is not. **Join on `(code, date)` against a listing table,
never on code alone.** Only `tushare` and `rqdatac` ship identifier converters
(`rqdatac.id_convert`); prefix conventions differ across libraries
(`sh600000` / `600000.SH` / `600000.XSHG` / `sh.600000`).

## 北交所 (BSE)

The weakest coverage link across every source. `rqdatac` requires `enable_bjse=True` — **default
False**, so BSE is silently absent unless you opt in. Verify BSE explicitly before claiming a "full
A-share" universe.
