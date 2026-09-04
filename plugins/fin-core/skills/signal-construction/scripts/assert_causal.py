"""Detect look-ahead in any feature function. Six lines of logic; catches almost everything.

The test: perturb ONLY rows at/after index k, recompute, and assert nothing before k moved.
If an early value changed, it depended on future data.

Usage:
    from assert_causal import assert_causal, scan_indicators
    assert_causal(lambda d: ta.momentum.RSIIndicator(d.close).rsi(), df, k=200)
"""
from __future__ import annotations

import pandas as pd


def assert_causal(fn, df: pd.DataFrame, k: int, tol: float = 1e-9, name: str = "") -> None:
    """Raise AssertionError if fn(df) rows [0, k) depend on rows [k, end).

    fn must be a pure function of the frame, returning a Series or DataFrame
    indexed like df. k should sit well past any warm-up period.
    """
    a = fn(df)
    df2 = df.copy()
    num = df2.select_dtypes("number").columns
    df2.loc[df2.index[k:], num] = df2.loc[df2.index[k:], num] * 2.0
    b = fn(df2)

    a0, b0 = a.iloc[:k], b.iloc[:k]
    delta = (a0.fillna(-9e99) - b0.fillna(-9e99)).abs()
    bad = delta > tol
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        worst = float(delta.to_numpy()[bad.to_numpy()].max())
        raise AssertionError(
            f"LOOK-AHEAD in {name or getattr(fn, '__name__', 'fn')}: "
            f"{n_bad} cells before index {k} changed when only rows >= {k} were perturbed "
            f"(max |delta| = {worst:.6g})"
        )


def warmup_bars(fn, df: pd.DataFrame, tol: float = 1e-9) -> int | None:
    """First index at which fn's output stops depending on how much history preceded it.

    Recomputes fn on progressively longer prefixes and compares the tail. This is the
    unstable-period / recursive-warm-up problem: EMA, RSI, ATR, ADX and MACD converge to
    different values depending on how many bars came before. Backtest sees thousands of
    bars; live often sees ~1000. Discard at least this many bars in both.
    """
    full = fn(df)
    n = len(df)
    for cut in range(50, n, 25):
        part = fn(df.iloc[:cut])
        common = part.index.intersection(full.index)[-10:]
        if len(common) == 0:
            continue
        d = (part.loc[common].fillna(-9e99) - full.loc[common].fillna(-9e99)).abs()
        if float(d.to_numpy().max()) <= tol:
            return cut
    return None


def scan_indicators(fns: dict, df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """Run assert_causal over a dict of {name: fn} and report pass/fail as a frame."""
    k = k or (len(df) // 2)
    rows = []
    for name, fn in fns.items():
        try:
            assert_causal(fn, df, k, name=name)
            rows.append({"indicator": name, "causal": True, "error": ""})
        except AssertionError as e:
            rows.append({"indicator": name, "causal": False, "error": str(e)})
        except Exception as e:  # a broken indicator is not a leakage result
            rows.append({"indicator": name, "causal": None, "error": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(7)
    n = 500
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    d = pd.DataFrame({"open": close.shift(1).bfill(), "high": close * 1.01,
                      "low": close * 0.99, "close": close, "volume": 1e6})

    checks = {
        "sma_20 (causal)":        lambda x: x.close.rolling(20).mean(),
        "centered_20 (LEAKS)":    lambda x: x.close.rolling(20, center=True).mean(),
        "shift_minus_1 (LEAKS)":  lambda x: x.close.shift(-1),
        "zscore_fullsample (LEAKS)": lambda x: (x.close - x.close.mean()) / x.close.std(),
        "zscore_expanding (causal)": lambda x: (x.close - x.close.expanding().mean())
                                               / x.close.expanding().std(),
    }
    print(scan_indicators(checks, d).to_string(index=False))
