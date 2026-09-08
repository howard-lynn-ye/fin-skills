"""Guard: a regime series that saw the future (regime-detection / regime_lookahead.py).

Smoothed Markov-switching probabilities condition on the WHOLE sample; filtered ones
include today's return; only the predicted series is tradeable. The script's
`detection` measures the tell on any regime series: how long after a true switch the
series first flags the new regime, and how often it flagged it BEFORE the switch.

Measured on the script's reference DGP (4 seeds, true parameters, 1764-day window):
the smoothed series has a median delay of 0 or -1 days, zero false alarms and flags
42-64% of switches before they happen; the causal predicted series has a median delay
of 3-4 days, 10-37 false alarms and anticipates 8-16% of switches by luck. The default
thresholds sit between those bands.
"""
from __future__ import annotations

import numpy as np

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import as_1d, require_int, require_number
from fin_skills.core.regime_lookahead import detection, label_accuracy


@register
class RegimeLookaheadGuard(Guard):
    """Does the regime probability flag switches before they happen? Then it used the future.

    Inputs
        regime          : integer regime labels per period (0 = calm/first regime,
                          1 = the other), the truth or a proxy such as a drawdown state.
        p_calm          : the series the strategy trades on: P(regime 0) per period.
        lo, hi          : evaluation window [lo, hi). Default the whole series.
        max_anticipated : share of switches flagged before they happened that fails
                          the guard. Default 0.25 (measured: causal 0.08-0.16,
                          smoothed 0.42-0.64 on the reference DGP).

    Fails when the median detection delay is <= 0 days (the series reacts no later
    than the switch itself, which a causal filter cannot do) or when more than
    `max_anticipated` of the switches were flagged in advance.
    """

    name = "regime_lookahead"
    skill = "regime-detection"
    summary = "Fails a regime-probability series that flags regime switches before they happen (smoothed / in-sample)."
    wraps = ("fin_skills.core.regime_lookahead.detection",
             "fin_skills.core.regime_lookahead.label_accuracy")
    required = ("regime", "p_calm")
    optional = ("lo", "hi", "max_anticipated")

    def check(self, regime: object, p_calm: object, lo: int = 0, hi: int | None = None,
              max_anticipated: float = 0.25) -> Outcome:
        out = Outcome()
        s = as_1d(regime, "regime")
        p = as_1d(p_calm, "p_calm")
        if len(s) != len(p):
            raise TypeError(f"regime ({len(s)}) and p_calm ({len(p)}) must have equal length")
        if not set(np.unique(s)).issubset({0.0, 1.0}):
            raise TypeError("regime must be 0/1 labels")
        if np.nanmin(p) < 0.0 or np.nanmax(p) > 1.0:
            raise TypeError("p_calm must be probabilities in [0, 1]")
        s = s.astype(int)
        p = np.nan_to_num(p)
        lo = require_int(lo, "lo", minimum=0)
        hi = len(s) if hi is None else require_int(hi, "hi", minimum=lo + 2)
        if hi > len(s):
            raise TypeError(f"hi={hi} exceeds the series length {len(s)}")
        max_anticipated = require_number(max_anticipated, "max_anticipated")

        det = detection(s, p, lo, hi)
        acc = label_accuracy(s, p, lo, hi)
        out.note(**det, label_accuracy=acc, lo=lo, hi=hi)
        if det["n_switches"] == 0:
            out.warning("no regime switch inside the window: nothing to time", where="switches")
            return out
        delay = det["median_delay"]
        share = det["share_anticipated"]
        if np.isfinite(delay) and delay <= 0:
            out.error(f"median detection delay {delay:+.0f} day(s): the series reacts no later "
                      f"than the switch itself, which needs the switch day's own data or the "
                      f"future (smoothed probabilities, or parameters fitted on the whole "
                      f"sample)", where="delay")
        if share > max_anticipated:
            out.error(f"{share:.0%} of {det['n_switches']} switches were flagged BEFORE they "
                      f"happened (threshold {max_anticipated:.0%}); a causal series can only "
                      f"do that by a lucky false alarm", where="anticipation")
        if out.passed:
            out.info(f"median delay {delay:+.0f} day(s), {share:.0%} anticipated, "
                     f"{det['false_alarms']} false alarms, label accuracy {acc:.3f}",
                     where="timing")
        return out
