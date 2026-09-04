"""Measure how much history a recursive indicator needs before its value stops moving.

WHY this exists — this is the backtest-vs-live divergence nobody instruments:

A recursive indicator (EMA, Wilder RSI/ATR/ADX, MACD, any `ewm`) has infinite memory. Its
value at bar t depends, with geometrically decaying weight, on EVERY bar before t —
including the seed. The backtest computes it over five years of history. Live, the bot
starts with whatever candles the exchange handed back on the last REST call. Same code,
same parameters, different amount of preceding data, DIFFERENT NUMBER. The signal fires a
bar early, or not at all, and the live equity curve quietly detaches from the backtest.

There is no exception raised, no NaN, no warning. The only mainstream framework that ships
a detector for it is freqtrade (`startup_candle_count`, checked by its own analysis
command); everywhere else the warmup length is a number someone guessed once. The usual
guess — "a 20-period EMA needs 20 bars" — is wrong by an order of magnitude.

What this does: recompute the indicator on progressively longer prefixes of the SAME data
and find the first prefix length after which the last N values stop changing. That number
is your `startup_candle_count`: the bars of history you must fetch and then DISCARD before
the indicator is safe to trade.

Rules of thumb this reproduces: an SMA converges in exactly its window (it is a finite
filter and has no memory beyond it), while an EMA needs roughly 10-15x its period and a
Wilder-smoothed indicator — whose decay factor is (n-1)/n rather than the EMA's
(n-1)/(n+1) — needs even more.

Usage:
    from warmup_probe import warmup_bars, warmup_report
    n = warmup_bars(lambda a: ema(a, 20), closes, tol=1e-9)   # -> bars to discard
    print(warmup_report({"EMA(20)": lambda a: ema(a, 20)}, closes,
                        periods={"EMA(20)": 20}).to_string(index=False))
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

Indicator = Callable[[np.ndarray], "Sequence[float] | np.ndarray | pd.Series"]

NOT_CONVERGED = -1
_SCALE_FLOOR = 1e-12


def _as_array(x: object) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def error_curve(fn: Indicator, data: Sequence[float] | np.ndarray,
                max_probe: int | None = None, n_eval: int = 10,
                step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Relative disagreement vs the full-history value, as a function of prefix length.

    For each probe length p, `fn` is recomputed on the window that starts exactly p bars
    before the evaluation window, and the last `n_eval` values are compared against the
    same values computed over ALL available history. The error is relative to the
    reference magnitude so indicators on different scales (price-level EMA vs 0-100 RSI)
    are directly comparable in one table.

    Returned once and reused for every tolerance, because the expensive part is the
    O(max_probe) recomputation, not the comparison.
    """
    x = _as_array(data)
    if x.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contains NaN/inf; the probe cannot distinguish a warmup "
                         "artefact from a hole in the input")
    if n_eval < 1:
        raise ValueError("n_eval must be >= 1")
    if max_probe is None:
        # Leave the reference itself a healthy margin of extra history.
        max_probe = max(1, (len(x) - n_eval) // 2)
    if max_probe + n_eval > len(x):
        raise ValueError(
            f"need at least max_probe + n_eval = {max_probe + n_eval} bars of data, have "
            f"{len(x)}. The reference must itself be computed on MORE history than the "
            f"longest probe, or you are comparing two equally-unconverged numbers."
        )

    ref = _as_array(fn(x))
    if len(ref) != len(x):
        raise ValueError(
            f"indicator returned {len(ref)} values for {len(x)} bars. `fn` must be "
            f"length-preserving (pad the warmup with NaN) or the last-N comparison is "
            f"misaligned."
        )
    ref = ref[-n_eval:]
    if not np.all(np.isfinite(ref)):
        raise ValueError("the full-history reference values are not finite; the data is "
                         "too short for this indicator")
    scale = max(float(np.max(np.abs(ref))), _SCALE_FLOOR)

    probes = np.arange(0, max_probe + 1, step, dtype=int)
    errs = np.empty(len(probes), dtype=float)
    for i, p in enumerate(probes):
        window = x[len(x) - n_eval - int(p):]
        vals = _as_array(fn(window))[-n_eval:]
        # A short window that cannot even produce a value counts as maximally wrong,
        # not as a skipped probe -- NaN here IS the failure being measured.
        errs[i] = (np.inf if not np.all(np.isfinite(vals))
                   else float(np.max(np.abs(vals - ref))) / scale)
    return probes, errs


def warmup_bars(fn: Indicator, data: Sequence[float] | np.ndarray,
                tol: float = 1e-9, max_probe: int | None = None,
                n_eval: int = 10, step: int = 1) -> int:
    """Bars of history the indicator needs before its output stops depending on the start.

    Returns the smallest probe length after which the relative error NEVER again exceeds
    `tol` (not merely the first dip below it — a decaying oscillation can cross zero and
    come back). Returns `NOT_CONVERGED` (-1) if the error is still above `tol` at
    `max_probe`, which means "more than max_probe", not "zero".

    `tol` is RELATIVE to the reference magnitude. Useful anchors:
        1e-6   agreement to 6 significant figures  (fine for a threshold crossing)
        1e-9   agreement to 9                      (safe default)
        1e-12  effectively float64-identical       (reproduces the 10-15x rule of thumb)
    """
    if tol <= 0:
        raise ValueError("tol must be positive; exact float equality is not a warmup "
                         "criterion, it is a coin flip on the last ULP")
    probes, errs = error_curve(fn, data, max_probe=max_probe, n_eval=n_eval, step=step)
    over = np.flatnonzero(errs > tol)
    if len(over) == 0:
        return int(probes[0])
    last = int(over[-1])
    if last == len(probes) - 1:
        return NOT_CONVERGED
    return int(probes[last + 1])


def warmup_report(fns: Mapping[str, Indicator], data: Sequence[float] | np.ndarray,
                  tolerances: Sequence[float] = (1e-6, 1e-9, 1e-12),
                  periods: Mapping[str, int] | None = None,
                  max_probe: int | None = None, n_eval: int = 10,
                  step: int = 1) -> pd.DataFrame:
    """Warmup table over several indicators and tolerances.

    The tolerance column matters as much as the indicator: "how many bars" is not a
    property of the indicator alone, it is a property of the indicator AND how much
    disagreement you are willing to trade on. Reporting one number without the tolerance
    is how the "20 bars for a 20-EMA" folklore survives.
    """
    rows: list[dict[str, object]] = []
    for name, fn in fns.items():
        probes, errs = error_curve(fn, data, max_probe=max_probe, n_eval=n_eval, step=step)
        row: dict[str, object] = {"indicator": name}
        period = None if periods is None else periods.get(name)
        if period:
            row["period"] = period
        for tol in tolerances:
            over = np.flatnonzero(errs > tol)
            if len(over) == 0:
                bars: int = int(probes[0])
            elif int(over[-1]) == len(probes) - 1:
                bars = NOT_CONVERGED
            else:
                bars = int(probes[int(over[-1]) + 1])
            row[f"bars@{tol:g}"] = bars
            if period:
                row[f"x_period@{tol:g}"] = (float("nan") if bars == NOT_CONVERGED
                                            else round(bars / period, 1))
        row["err@0bars"] = float(errs[0])
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Indicators implemented inline on purpose: TA-Lib and `ta` are exactly the libraries
    # whose warmup you cannot inspect, and the whole point is that this probe works on a
    # black-box callable.
    # ---------------------------------------------------------------------------------
    def sma(x: np.ndarray, n: int) -> np.ndarray:
        """FINITE impulse response: bar t depends on bars t-n+1..t and nothing earlier."""
        s = pd.Series(x, dtype=float)
        return s.rolling(n, min_periods=n).mean().to_numpy()

    def ema(x: np.ndarray, n: int) -> np.ndarray:
        """INFINITE impulse response, alpha = 2/(n+1), seeded with the first observation
        (pandas' `ewm(adjust=False)` convention, and TA-Lib's after its SMA seed)."""
        alpha = 2.0 / (n + 1.0)
        out = np.empty(len(x), dtype=float)
        if len(x) == 0:
            return out
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        return out

    def rsi_wilder(x: np.ndarray, n: int = 14) -> np.ndarray:
        """Wilder smoothing: alpha = 1/n, i.e. decay (n-1)/n -- SLOWER than an n-EMA's
        (n-1)/(n+1). Same nominal 'period', much longer memory. Seeded with the simple
        mean of the first n changes, which is what the start dependence rides in on."""
        out = np.full(len(x), np.nan, dtype=float)
        if len(x) <= n:
            return out
        d = np.diff(x)
        gain = np.where(d > 0, d, 0.0)
        loss = np.where(d < 0, -d, 0.0)
        avg_g = float(np.mean(gain[:n]))
        avg_l = float(np.mean(loss[:n]))
        for i in range(n, len(x)):
            if i > n:
                avg_g = (avg_g * (n - 1) + gain[i - 1]) / n
                avg_l = (avg_l * (n - 1) + loss[i - 1]) / n
            out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
        return out

    # Offline synthetic prices: a fixed-seed GBM. No network, no data files.
    rng = np.random.default_rng(7)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, 1500)))

    fns: dict[str, Indicator] = {
        "SMA(20)": lambda a: sma(a, 20),
        "EMA(20)": lambda a: ema(a, 20),
        "EMA(50)": lambda a: ema(a, 50),
        "RSI_wilder(14)": lambda a: rsi_wilder(a, 14),
    }
    periods = {"SMA(20)": 20, "EMA(20)": 20, "EMA(50)": 50, "RSI_wilder(14)": 14}

    print("=" * 88)
    print("WARMUP PROBE  --  1500 synthetic closes, last 10 values compared")
    print("=" * 88)
    table = warmup_report(fns, closes, periods=periods, max_probe=700, n_eval=10)
    print(table.to_string(index=False))

    print("\n'bars@tol'    = history to fetch AND DISCARD before the indicator is tradable")
    print("'x_period@tol'= that count as a multiple of the indicator's nominal period")
    print("'err@0bars'   = relative error with NO warmup at all (what live actually does")
    print("                on a cold start, if you never set startup_candle_count)")

    sma_bars = warmup_bars(fns["SMA(20)"], closes, tol=1e-12, max_probe=700)
    ema_bars = warmup_bars(fns["EMA(20)"], closes, tol=1e-12, max_probe=700)
    rsi_bars = warmup_bars(fns["RSI_wilder(14)"], closes, tol=1e-12, max_probe=700)
    assert sma_bars <= 20, sma_bars
    assert 10 * 20 <= ema_bars <= 15 * 20, ema_bars

    print("\n" + "=" * 88)
    print("2. THE FOLKLORE, TESTED")
    print("=" * 88)
    print(f"SMA(20)        converges in {sma_bars:>4} bars = {sma_bars / 20:.1f}x period "
          f"-- finite filter: 19 prior bars + the bar itself IS the 20-bar window, "
          f"and not one bar more")
    print(f"EMA(20)        converges in {ema_bars:>4} bars = {ema_bars / 20:.1f}x period "
          f"-- the documented 10-15x rule of thumb, confirmed")
    print(f"RSI_wilder(14) converges in {rsi_bars:>4} bars = {rsi_bars / 14:.1f}x period "
          f"-- WORSE, because Wilder's alpha=1/n decays slower than an EMA's 2/(n+1)")
    print("\nA '14-period RSI' fed 100 candles is not a 14-period RSI. It is a different "
          "\nindicator that happens to share the name.")

    # ------------------------------------------------------ 3. THE DIVERGENCE, IN PRICE
    print("\n" + "=" * 88)
    print("3. WHAT THE DIVERGENCE LOOKS LIKE ON THE LAST BAR (the one you trade)")
    print("=" * 88)
    full_rsi = rsi_wilder(closes, 14)[-1]
    print(f"{'candles fetched':>16}{'RSI(14) on last bar':>22}{'error vs full history':>24}")
    for fetched in (30, 50, 100, 200, 400, 700, 1500):
        v = rsi_wilder(closes[-fetched:], 14)[-1]
        print(f"{fetched:>16}{v:>22.10f}{abs(v - full_rsi):>24.2e}")
    cold = rsi_wilder(closes[-30:], 14)[-1]
    print(f"\nA bot restarted with 30 candles reads RSI {cold:.4f}; the backtest reads "
          f"{full_rsi:.4f}.")
    print(f"Against an overbought threshold of 70 those two disagree about whether to "
          f"sell: {cold > 70} vs {full_rsi > 70}."
          if (cold > 70) != (full_rsi > 70) else
          f"Same side of 70 here -- but the gap is {abs(cold - full_rsi):.4f} RSI points, "
          f"and any threshold inside that band flips the trade.")

    print("\n" + "=" * 88)
    print("Fetch warmup + lookback. Discard the warmup. Assert the discard in live code.")
    print("=" * 88)
