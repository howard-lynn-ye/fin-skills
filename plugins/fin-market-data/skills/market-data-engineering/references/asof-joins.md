# As-of joins across engines

The operation that decides whether your backtest saw the future. Every engine implements it
differently, and three of them default to look-ahead.

## The semantics, side by side

| Engine | Call | Default direction | Checks sortedness? | Exact-match control |
|---|---|---|---|---|
| **pandas** | `merge_asof(l, r, on=, by=)` | backward (`<=`) | ✅ **raises** `ValueError: left keys must be sorted` | `allow_exact_matches=True` (default) |
| **polars** | `join_asof(other, on=, by=)` | backward | 🚨 **NOT when `by=` is given** | `strategy=` / no direct equivalent |
| **DuckDB** | `ASOF JOIN … ON l.t >= r.t` | you write it | n/a (SQL) | you write `>` vs `>=` |
| **DuckDB** | `ASOF JOIN … USING (t)` | 🚨 **forces `>=`** | n/a | 🚨 **no override** |
| **QuestDB** | `ASOF JOIN` | `<=` | n/a | **`LT JOIN`** for strictly earlier |
| **ClickHouse** | `ASOF JOIN` | `<=` | n/a | `<` supported in the ON clause |

## 🚨 polars is now the unsafe one

The folklore says "pandas silently gives garbage on unsorted input". **That is out of date.**
✅ `pandas.merge_asof` **raises**. ✅ `polars.join_asof`'s own docstring states the in-memory engine
*"cannot check the sortedness if 'by' groups are provided"* — **which is exactly the per-symbol quote
join everyone writes.** Unsorted input gives silently wrong rows with no error.

## 🚨 They mean different things by "sorted"

**pandas** (`merge_asof` docstring):
> *"Both DataFrames must be first sorted by the merge key in ascending order before calling this
> function. **Sorting by any additional 'by' grouping columns is not required.**"*

**polars** (`join_asof` docstring):
> *"Both DataFrames must be sorted by the `on` key (**within each `by` group, if specified**)."*

- pandas wants **one global ascending sort by `time`**, ignoring `symbol`.
- polars wants the frame sorted by `time` **inside each `symbol` group** — i.e. `sort(["symbol","time"])`.

✅ Executed: a frame sorted `["symbol","time"]` (valid polars input) is **rejected by pandas**:

```python
L = pd.DataFrame({"time": pd.to_datetime(["09:30:00","09:30:05",    # sym A
                                          "09:30:01","09:30:06"]),  # sym B
                  "sym": ["A","A","B","B"], "s": [1,2,3,4]})
pd.merge_asof(L, R, on="time", by="sym")
# ValueError: left keys must be sorted
```

After a global `sort_values("time")` it works and gives the correct per-symbol result.

🔑 **A global sort by `on` satisfies both engines.** Do that unconditionally and the portability
problem disappears.

🔑 **The real asymmetry:** data sorted by *neither* is **always caught by pandas and never by
polars**. Porting a working pandas join to polars can therefore start silently returning wrong rows.

## 🚨 The exact-match trap — executed

```python
quotes = pd.DataFrame({
    "time": pd.to_datetime(["2024-01-02 09:30:00",
                            "2024-01-02 09:30:01",
                            "2024-01-02 09:30:02"]),
    "bid":  [100.00, 101.00, 102.00]})
signal = pd.DataFrame({"time": pd.to_datetime(["2024-01-02 09:30:01"]), "sig": [1]})

pd.merge_asof(signal, quotes, on="time")                          # bid = 101.00  LOOK-AHEAD
pd.merge_asof(signal, quotes, on="time", allow_exact_matches=False)  # bid = 100.00  CORRECT
```

**Where the tie comes from:**
1. **Bar data.** Signal from the `09:30` bar joined to that bar's close = trading at the close you
   used to decide. Compounded by `resample`'s intraday **`label="left"`** default (unchanged in
   pandas 3.0, verified in source): **the bar labelled `09:30` contains data through `09:34:59`.**
2. **Same-timestamp quote revisions.** Feeds stamp the quote update caused by your trade with the
   same microsecond — matching it means you filled at the post-impact price you created.
3. **Daily fundamentals.** An earnings release stamped `2024-02-01` joined to a `2024-02-01` signal
   assumes you traded on news released that day, often pre-open.

🚨 **The tell is a Sharpe of 2–4, not 40** — good but not absurd, which is exactly the range that
survives a review.

## 🚨 DuckDB's convenient syntax opts you into look-ahead

```sql
-- USING shorthand: forces >= , with NO way to override
SELECT * FROM signals ASOF JOIN quotes USING (time);

-- explicit ON: you control the operator
SELECT * FROM signals s ASOF JOIN quotes q ON s.time > q.time;   -- strictly prior
```

🚨 **Plain `ASOF JOIN` also silently DROPS unmatched left rows.** Use **`ASOF LEFT JOIN`** unless you
genuinely want the inner-join behaviour — otherwise your signal count quietly shrinks and the
survivors are the ones that happened to have a prior quote.

## QuestDB has the cleanest design

- **`ASOF JOIN`** — `<=`
- **`LT JOIN`** — strictly earlier, as a first-class join type rather than a flag
- **`SPLICE JOIN`** — full outer temporal join
- **a `TOLERANCE` clause** — a first-class maximum staleness bound

**`TOLERANCE` is the feature everyone else makes you hand-roll.** Without a staleness bound, an
as-of join will happily match a signal to a quote from three days earlier across a market closure,
and nothing warns you.

⚠️ **ClickHouse** restricts `ASOF JOIN` to the `hash` and `full_sorting_merge` algorithms only.

## The checklist

- [ ] Both frames **globally sorted by the join key** (satisfies pandas and polars alike).
- [ ] **`allow_exact_matches=False`** (or `>` in SQL) whenever the right side is information you react to.
- [ ] A **tolerance / maximum staleness** is set. An unbounded as-of join matches across weekends,
      halts and delistings.
- [ ] **`ASOF LEFT JOIN`** in DuckDB unless you intend to drop unmatched rows — and assert the row
      count afterwards.
- [ ] Timezones are **identical and explicit** on both sides. A tz-naive and a tz-aware key either
      raise or, worse, compare as UTC.
- [ ] The `by` key has the **same dtype** on both sides — a `category` vs `object` symbol column
      silently matches nothing.
- [ ] After the join, **assert the output row count equals the left row count** (for a left join).
      Silent row loss is the most common as-of bug after look-ahead.

## The generic safety test

```python
# An as-of join must never let a future value reach a past row.
joined = pd.merge_asof(sig.sort_values("time"), qt.sort_values("time"),
                       on="time", by="sym",
                       allow_exact_matches=False,
                       tolerance=pd.Timedelta("5min"))
assert (joined["quote_time"] < joined["time"]).all(), "as-of join matched a non-prior quote"
assert len(joined) == len(sig), "as-of join dropped rows"
```

Carrying the right-hand timestamp through the join and asserting on it is the cheapest possible
guard, and it catches every variant above.
