# Roadmap

What is being built, what is planned, and what this repo deliberately does not do. Dates are
targets, not promises, and nothing here promises adoption. The roadmap changes through
issues: propose an item with the "New skill proposal" form, or challenge one with a
reproduction.

## In progress: September 2026

- **`fin-models`** - 12 skills, one per model family, as named in the plugin's marketplace
  entry: factor models, covariance and risk models, VaR/CVaR and their backtests, portfolio
  optimizers, volatility models, state-space and Kalman, cointegration and statistical
  arbitrage, time-series forecasting, option pricing and implied-volatility surfaces,
  term-structure and credit models. Each says which Python implementation is verified, the
  pitfalls that produce wrong numbers, and ships an executable script that reproduces the
  claim.
- **`fin-strategies`** - 5 skills, one per family, as named in its marketplace entry: trend
  following and time-series momentum, alpha combination and neutralization, execution
  algorithms (VWAP, TWAP, POV, Almgren-Chriss), market-making models, position sizing and
  Kelly. Each with an executable script and the traps that flatter a backtest.

  Both plugins are registered in `.claude-plugin/marketplace.json` with an empty `skills`
  array (commit 93c2216) so that the agents writing them in parallel share one merge base.
  Until the skills land, the two entries install nothing.
- **Re-run the blind trigger evaluation** (`scripts/eval_blind.py`) once the 17 new
  descriptions are in the listing; the current 107/108 was measured on 53 skills.
- **First tagged release, `v0.1.0`**, with `CHANGELOG.md`, `CITATION.cff` and the leak
  benchmark (`benchmarks/`) committed alongside it.

## Planned: October to December 2026

- **Continuous integration.** A GitHub Actions workflow running `build_index.py` (tree must
  stay clean), `build_package.py --check`, `validate.py`, `pytest`, `check_scripts.py` and
  `benchmarks/leak_bench.py --quick` on Python 3.10 to 3.13, Ubuntu and Windows, plus a
  minimum-dependency job at the `pyproject.toml` floors. Blocked on one account step: the
  token used here lacks the `workflow` scope, so nothing under `.github/workflows/` can be
  pushed until the maintainer grants it.
- **PyPI** through trusted publishing (`pip install fin-skills`; the name was free on
  2026-09-08), then a conda-forge recipe.
- **A Zenodo DOI** minted on the first archived GitHub release, recorded in `CITATION.cff`.
- **Examples.** Three worked scripts: audit a backtest with `Bundle` and `check()`, a
  point-in-time fundamentals join, and a futures roll into a continuous contract.
- **A structured docs site** - install and requirements, getting started, the skill catalogue,
  the guard catalogue, the benchmark, the changelog, how to cite - with the existing pdoc API
  reference mounted under it.
- **The one documented benchmark gap.** `cost_curve` checks whether a strategy survives at the
  cost you stated, not whether the stated cost is plausible; a guard that checks a cost
  assumption against traded volume is the next script to write (see `benchmarks/README.md`).

## Later: 2027

- **A reference backtesting engine and a unified data layer** that produce `Bundle` objects
  directly, so every guard runs on a run's own artefacts without hand-assembly.
- **Bridges** that build a `Bundle` from the outputs of vectorbt-, qlib-, QuantLib- and
  ccxt-style tools, so the guards audit results those tools produced.
- **New plugins** for fixed income and credit, macro, and alternative data, at the same
  granularity as the existing market plugins.

## Out of scope today

Unchanged from the README's "Scope" section: crypto/DeFi execution plumbing and MEV
(`agiprolabs/claude-trading-skills` owns it), RIA compliance and practice operations
(`JoelLewis/finance_skills`), personal bookkeeping and tax (`openaccountant/skills`).
Leakage-safe quant ML overlaps with `ml4t/skills`, which is worth reading alongside this one.

Nothing here is investment advice, and no skill in this repo places an order. That does not
change with any item above.
