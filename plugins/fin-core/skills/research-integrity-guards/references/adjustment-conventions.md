# Price adjustment conventions

Three conventions exist. Two are safe for research; one is a look-ahead bug that most libraries
choose by default.

## The three

**Raw / unadjusted (不复权)** — the price that actually traded. Correct for anything that depends on
the real price level: price limits, tick sizes, lot sizing, margin, option strikes, round-lot
constraints. **Wrong for computing returns across a split or dividend.**

**Backward-adjusted (后复权, hfq)** — anchored at the **listing date**: `adjusted = raw × factor`.
A new corporate action only appends a new factor; **every historical value is unchanged.** The
series is append-only, stable and reproducible. **This is the research default.**

**Forward-adjusted (前复权, qfq)** — anchored at the **most recent date**:
`adjusted = raw × factor / factor(anchor)`.

## Why forward-adjusted is a look-ahead bug

The anchor is "now". Every time a new dividend or split occurs, `factor(anchor)` changes and **the
entire historical series is rescaled retroactively.** Two consequences:

1. **Non-reproducibility.** The same query, same start date, run a month apart, returns different
   historical prices. A cached qfq series silently disagrees with a fresh pull.
2. **Look-ahead.** A qfq price at time *t* is a function of every corporate action between *t* and
   the anchor. It could not have been computed at *t*. The per-event leak is small, but it
   systematically biases anything sensitive to dividend or split timing.

**Rule: use backward-adjusted, or raw plus the factor series. Never persist forward-adjusted
prices.** If a chart needs qfq, recompute it at render time.

## The worst variant

`tushare`'s qfq is anchored to **the last date of your query window**, not to today
(`data['adj_factor'] / fcts['adj_factor'][0]`, where `fcts` is newest-first). **The same bar has
different values depending on what `end_date` you asked for.** Two backtests over different windows
disagree on overlapping history.

## Library defaults — verified from source

| Library | Call | Default | Safe? |
|---|---|---|---|
| yfinance | `yf.download()` | `auto_adjust=True` (since 1.0) | ✅ adjusted — but **no `Adj Close` column** |
| yfinance | `Ticker.history()` | `auto_adjust=True` (since 0.1.26) | ✅ |
| yahooquery | `.history()` | **`adj_ohlc=False`** | ⚠️ **unadjusted — opposite of yfinance** |
| tiingo | `.get_dataframe([list])` | `metric_name='adjClose'` | ⚠️ adjusted close only, no OHLC |
| Alpha Vantage (free) | daily | unadjusted (adjusted is premium) | 🚨 split-corrupted |
| akshare | `stock_zh_a_hist` | `adjust=""` = raw | ✅ |
| baostock | `query_history_k_data_plus` | `adjustflag='3'` = raw | ✅ |
| tushare | `pro_bar` | `adj=None` = raw | ✅ |
| **efinance** | `get_quote_history` | **`fqt=1` = qfq** | 🚨 |
| **adata** | `get_market` | **`adjust_type=1` = qfq**, and only qfq is implemented | 🚨🚨 |
| **jqdatasdk** | `get_price` | **`fq='pre'` = qfq** | 🚨 |
| mootdx | `Quotes.bars` + `xdxr` | raw + raw corporate actions | ✅✅ best |

## The consistency rule

**Use the same convention for the signal and for the execution price.** A backtest that computes a
moving average on adjusted prices and then fills at the raw price (or vice versa) manufactures
returns at every corporate action. Engines without an adjustments database (vectorbt,
backtesting.py, bt, PyBroker) cannot detect this — the frame you hand them is the truth as far as
they are concerned.

## Cross-source reconciliation

Prices from two sources will disagree if their adjustment conventions differ. Before concluding a
source is wrong:

1. Check both adjustment settings explicitly — do not trust the defaults to match.
2. Compare on a window containing **no** corporate action; if they agree there, the difference is
   adjustment, not data quality.
3. Compare the corporate-action series themselves (dividends and splits), not just the prices.

## What to record

Whatever you store, record alongside it: the convention, the anchor date if forward-adjusted, the
source, and the retrieval timestamp. An adjusted price without its convention is not a number you
can reuse.
