"""Guard: recursive-indicator warm-up (signal-construction / warmup_probe.py)."""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, require_int
from fin_skills.core.warmup_probe import NOT_CONVERGED, error_curve, warmup_bars


@register
class WarmupGuard(Guard):
    """How many bars an indicator needs before its value stops depending on where history began.

    Inputs
        indicator : callable ndarray -> length-preserving ndarray/Series, or a
                    {name: callable} mapping to probe several at once.
        closes    : 1-D price array with no NaN.
        tol       : RELATIVE tolerance (1e-6 threshold-safe, 1e-9 default, 1e-12 exact).
        max_probe : longest prefix tried. Default (len - n_eval) // 2.
        n_eval    : trailing values compared. Default 10.
        step      : probe step. Default 1.
        history   : bars your live process actually fetches (freqtrade's
                    startup_candle_count). If given, the guard fails when the indicator
                    needs more than that.

    Fails when the indicator has not converged within max_probe, or when it needs more
    history than `history`. Evidence carries the bars needed and the relative error a
    cold start (zero warm-up) would trade on.
    """

    name = "warmup_probe"
    skill = "signal-construction"
    summary = "Measures the warm-up an EMA/RSI/ATR needs; fails if live fetches fewer bars than that."
    wraps = ("fin_skills.core.warmup_probe.warmup_bars",
             "fin_skills.core.warmup_probe.error_curve")
    required = ("indicator", "closes")
    optional = ("tol", "max_probe", "n_eval", "step", "history")

    def check(self, indicator: Callable[..., object] | Mapping[str, Callable[..., object]],
              closes: Sequence[float] | np.ndarray, tol: float = 1e-9,
              max_probe: int | None = None, n_eval: int = 10, step: int = 1,
              history: int | None = None) -> Outcome:
        out = Outcome()
        x = as_1d(closes, "closes")
        if history is not None:
            history = require_int(history, "history", minimum=1)
        if isinstance(indicator, Mapping):
            fns = dict(indicator)
            if not fns or not all(callable(f) for f in fns.values()):
                raise TypeError("indicator mapping must be {name: callable}")
        elif callable(indicator):
            fns = {getattr(indicator, "__name__", "indicator"): indicator}
        else:
            raise TypeError("indicator must be callable or a {name: callable} mapping")

        results: dict[str, dict[str, float | int]] = {}
        for label, fn in fns.items():
            probes, errs = error_curve(fn, x, max_probe=max_probe, n_eval=n_eval, step=step)
            bars = warmup_bars(fn, x, tol=tol, max_probe=max_probe, n_eval=n_eval, step=step)
            cold = float(errs[0])
            results[label] = {"warmup_bars": bars, "max_probe": int(probes[-1]),
                              "err_at_zero_bars": cold}
            if bars == NOT_CONVERGED:
                out.error(f"still not converged to tol={tol:g} after {int(probes[-1])} bars; "
                          f"a cold start is off by {cold:.3g} relative", where=label)
            elif history is not None and bars > history:
                out.error(f"needs {bars} bars to converge to tol={tol:g} but live fetches "
                          f"{history}; the live value is a different indicator", where=label)
            else:
                out.info(f"needs {bars} bars at tol={tol:g}; cold-start error {cold:.3g} relative",
                         where=label)
        out.note(results=results, tol=tol, history=history, n_bars=int(len(x)))
        return out
