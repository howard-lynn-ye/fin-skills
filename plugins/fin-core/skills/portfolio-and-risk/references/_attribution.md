# Attribution — the ecosystem gap you have to fill yourself

⚠️ **No mature Brinson-attribution library exists in Python.** `pyfolio-reloaded` ships a
`perf_attrib` module for *factor* attribution given a loadings frame, and `alphalens-reloaded`
evaluates a single cross-sectional signal — but nothing implements sector/allocation-selection
attribution or multi-period linking. Single-period Brinson is short enough to write.

## Brinson-Fachler (single period, sector level)

```python
# w_p, w_b : portfolio / benchmark sector weights (each sums to 1)
# r_p, r_b : portfolio / benchmark sector returns
R_b         = (w_b * r_b).sum()                     # total benchmark return
allocation  = ((w_p - w_b) * (r_b - R_b)).sum()     # sector-tilt effect
selection   = (w_b * (r_p - r_b)).sum()             # within-sector stock picking
interaction = ((w_p - w_b) * (r_p - r_b)).sum()
total       = allocation + selection + interaction  # == (w_p*r_p).sum() - R_b
```

**Brinson-Hood-Beebower** differs only in that allocation uses `(w_p - w_b) * r_b` with no `-R_b`
recentring. BF is generally preferred because recentring makes the allocation term measure the tilt
against the *benchmark's own* average, which is what a sector bet actually is.

Always assert `total == portfolio_return - benchmark_return` to within floating-point error. If it
does not close, your weights and returns are measured over different intervals — the usual cause is
mixing beginning-of-period weights with a return computed over a rebalance.

## Multi-period linking — the part that is genuinely hard

Single-period effects **do not sum** across periods, because returns compound. You need a smoothing
algorithm — **Carino**, **Menchero**, or **GRAP** — that distributes the residual so the linked
effects reconcile to the total active return.

⚠️ **None of the Python libraries implement any of them.** If you need multi-period attribution,
either implement Carino (the simplest: a logarithmic scaling coefficient per period) or report
single-period attributions separately and state that they do not sum.

## Factor attribution

For "how much of my return came from market, size, value, momentum":

1. Pull factor returns from the Ken French data library (`pandas_datareader.famafrench` works —
   ✅ its parser was fixed in 0.11.0; anything pulled with 0.10.0 should be re-pulled).
2. Regress portfolio excess returns on factor returns with **HAC (Newey-West)** standard errors —
   `statsmodels.OLS(...).fit(cov_type='HAC', cov_kwds={'maxlags': L})`.
3. The intercept is your alpha. **Report its t-statistic, not just its value.**

🚨 **Regress excess returns on excess returns.** Regressing raw on raw lets the intercept absorb the
risk-free rate, which manufactures alpha. This is a common enough error that it is worth asserting
in code that the risk-free rate has been subtracted from both sides.

`pyfolio-reloaded.perf_attrib` handles the bookkeeping if you already have a loadings DataFrame; it
does not source factors for you.

## Trade-level attribution

`pyfolio.round_trips` decomposes P&L by round trip — useful for separating "the signal was right" from
"the execution was bad", which the factor view cannot distinguish. Look at hold time, win rate and
P&L per trade before concluding a strategy has alpha; a high win rate with negative expectancy is a
sizing problem, not a signal problem.

## Reporting

Attribution is only meaningful against a **stated, investable benchmark** chosen in advance. Picking
the benchmark after seeing results is the same error as picking the event window after seeing
results — and it counts as a trial.
