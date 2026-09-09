# Examples

Three runnable scripts, in the order to read them. Each one is self-contained: seeded
synthetic data, no network, numpy/pandas plus the installed `fin_skills` package, ASCII
output, under five seconds, under 200 lines. Each ends with a `TAKEAWAY` block that says what
you were supposed to see.

```bash
pip install -e .                                   # or set PYTHONPATH to the repo root
python examples/audit_a_backtest.py
python examples/point_in_time_fundamentals.py
python examples/futures_roll.py
```

`tests/test_examples.py` runs all three in a subprocess and asserts their **conclusions**, not
just their exit codes, so they cannot rot quietly. It is marked `slow` (~10 s of subprocesses),
so run it with `python -m pytest -q -m slow tests/test_examples.py`; CI runs it in its `-m slow`
job.

| Example | What it shows | Runs |
|---|---|---|
| [`audit_a_backtest.py`](audit_a_backtest.py) | The whole API in one call: put a research run in a `Bundle`, read `coverage()` to see which checks can run at all, `check()` to run them, then fix the two planted defects and watch them go green | ~3 s |
| [`point_in_time_fundamentals.py`](point_in_time_fundamentals.py) | The same fundamentals joined to the same prices two ways — latest vintage on an exact stamp, versus filed-date vintage on a backward as-of with a tolerance — and what the difference is worth in Sharpe | ~4 s |
| [`futures_roll.py`](futures_roll.py) | One futures chain stitched three ways, which return operator reproduces true dollar P&L, and the back-adjusted series walking through zero into negative prices under backwardation | ~1 s |

## 1. `audit_a_backtest.py`

A `Bundle` is this library's `fit(X)`: name the artefacts a research run already has — returns,
turnover, the bars a signal saw, the signal function, the price panel, a regime label — and
every guard that can read them runs in one `check()` call.

The example prints the **coverage report first**, on purpose. "All checks passed" is
meaningless until you know how many checks could run; `coverage()` lists what is ready, and
`unlocks()` names the single missing slot that would enable each of the others. Then it runs
the guards on a run with two planted defects (a signal function that reads one bar ahead, and
a price panel filtered to names still trading at the end), reads the two failures, repairs
both with `bundle.with_(...)`, and re-runs clean.

## 2. `point_in_time_fundamentals.py`

A vendor's fundamentals table is keyed by the period a number *describes*. Joining on that key
makes two mistakes at once: the number was not public until it was filed weeks later, and the
value stored today is the latest vintage, restatements included.

The world is seeded so that each quarter's true surprise drives a 60-day drift, which makes
the leak measurable rather than rhetorical. The wrong join reports more than double the
Sharpe of the right one, and the two signals agree on the sign of barely half the cells — it
is not a better version of the same idea. `fin_skills.core.safe_asof.safe_merge_asof` does the
right join (mandatory tolerance, `allow_exact_matches=False`), and the `safe_asof` guard
flags the wrong one without needing the backtest at all.

## 3. `futures_roll.py`

Welding an expiring chain into one series has three standard answers and they are three
different series. The example scores each against the true P&L of holding and rolling the
front contract, and finds exactly two exact pairings: `.diff()` on the difference-adjusted
series gives dollar P&L, `.pct_change()` on the ratio-adjusted series gives returns. Crossing
those pairings is silent — the floats look fine.

It runs on a **backwardated** market, where back-adjustment subtracts a positive gap from all
older history and drives it below zero. On that series `pct_change()` flips sign either side
of the crossing and `log()` is NaN for more than half the sample, so
`continuous_contract.safe_returns` refuses to compute returns from it at all.

## Where to go next

- The guards behind these examples:
  `python -c "import fin_skills.api as a; [print(g().describe()) for g in a.registry()]"`,
  or `fin_skills.api.slots()` for the vocabulary a Bundle accepts.
- Whether the guards catch what they claim: `benchmarks/README.md` and `benchmarks/RESULTS.md`.
- The knowledge each guard came from: `fin_skills.load("<skill>")` returns the SKILL.md text.
