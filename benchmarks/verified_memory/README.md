# Verified-feedback online trading feasibility study

This is a new development experiment, separate from the existing LLM agent study.
No whole-brain emulation, LLM training, live broker, or profitability claim is made.
The candidate is a full-information online regressor over five portfolio experts.
It learns every expert's standalone payoff after that payoff becomes available.
This is not bandit feedback or a biological implementation of dopamine plasticity.

## Frozen design

Compare linear normalized LMS, dense random expansion plus LMS, and a mushroom-body-
inspired top-k sparse expansion plus LMS. Dense and sparse use identical projection
matrices and readout sizes. All use the same learning rate, payoff clipping, softmax
temperature, data, expert portfolios, execution lag, and cost convention. There is no
hyperparameter search in this pilot. Sparse activity does not imply fewer parameters.

Cross these with unchecked updates, output-only audit, and verified updates. The
output-only condition must produce exactly the unchecked trades; it only changes
whether the final record is accepted. Verified updates defer backdated feedback until
a corrected mature revision arrives and reject miscomputed payoffs. Losses remain
valid learning signals. Every arm is scored, including rejected output-only runs.

The actual fin-skills `synthesis_integrity` guard checks feedback input lineage.
A new local adapter recomputes standalone payoff accounting separately. The actual
`assert_causal` guard checks feature construction. `TrialLedger` records all cells
before evaluation. The independent final cash/share ledger is imported from the
existing library-utility benchmark; it imports no fin-skills code and supplies no
learning feedback. Audit receipts are policy-independent and can be cached across
arms; this does not give the learner access to the scorer.

Three streams are evaluated separately: clean, known backdated feedback every seventh
origin, and a known incorrect momentum-expert payoff every seventh origin. Corrupt
streams are fault-injection tests, not evidence of forecasting alpha. Detection of
these authored faults cannot establish general detection ability. Receipts depend on
trusted simulator input clocks; forged source provenance is outside this experiment.

Synthetic paths use recurring positive/negative/null autocorrelation regimes and three
independent market seeds. The existing ECB fixture is a separate historical proxy:
foreign-currency prices in EUR, zero cash interest, no carry, indicative reference rates,
no bid/ask or executable fills. It was already inspected in this repository. Its
2016-2025 evaluation suffix is historical development data, not an unseen holdout.
Earlier observations warm up the online learner; thereafter it continues to update
only as labels mature. Synthetic evaluation uses the last 40% of each path.

Decisions occur every five observations, execute at the next observation, and labels
cover the following five observations. Same-timestamp labels cannot update a decision.
Targets include standalone entry and liquidation fees at five basis points per leg;
actual portfolio costs use its own turnover. Scores at zero, five and twenty basis
points replay fixed actions, not policies retrained at each cost. Neither reward nor
score is a Sharpe-optimization objective.

Primary feasibility measures: invalid/early updates, retained negative targets,
accepted-output rate, and successful completion. Economic measures are descriptive:
net proxy return, drawdown, turnover and paired daily return differences. Initialization
seeds on one market are not independent market replications. Contrasts average those
seeds by date before block resampling. No model is selected from the evaluation results.
Future confirmatory work requires independently constructed defects, new data, a
frozen power analysis and broader baseline selection.

## Beacon

All source, protocol, fixtures, temporary files, cache, pytest output, scheduler logs
and results live in a new `/beacon-projects/radfm/wy891/fin-skills-memory-*` directory.
The previous RADFM Python environment is read-only. No HOME environment override or
HOME output is used. This CPU pilot needs no model download or GPU allocation.
`deploy.py` copies only an explicit source/fixture allowlist and refuses an existing
remote directory. The protocol records hashes before the first cell is evaluated.

Run locally with `python benchmarks/verified_memory/run.py --output runs/memory-pilot`.
The output directory must not already exist. Tests use small mechanism fixtures and
do not choose hyperparameters. The full pilot's numerical results are produced on Beacon.

Algorithmic starting points (not claims of a new biological learning rule):
- https://arxiv.org/abs/2107.07617
- https://github.com/paappraiser/stonkfly-lab
- https://github.com/nftechie/stonkfly
