# Fruit-fly trading reuse experiment

This adapter uses existing implementations rather than rebuilding a neural simulator:

| Source | Pinned commit | Reused components |
|---|---|---|
| [fruit-fly-fund](https://github.com/armanbabazadeh6/fruit-fly-fund) | `56f01f6426a4d6d6293a4e06afcf4a035133a968` | NeuralBackend, replay clock/guard, public Kraken data adapter |
| [Stonkfly](https://github.com/nftechie/stonkfly), vendored by fruit-fly-fund | `78ef3e05ab0fa086032098558d893667068944a0` | Full retained graph, visual adapter, memory rule, fixed decoder, Decimal ledger and paper broker |
| [Stonkfly Lab](https://github.com/paappraiser/stonkfly-lab) | `09e4529e2e1a135083838abd00ccacfd95c63a29` | OdorEncoder and MushroomBody, imported unchanged |
| fin-skills | Code hashes in deployment receipt | Costs and trade_charge for independent fee/cash/position reconciliation |

These upstream projects use MIT code licenses. Their source checkouts and license files remain
separate; no upstream implementation is represented as our original work. Connectome data terms
are separate from code licensing. See the upstream release metadata and attribution files.

The new code implements the interface between these components: completed-bar observations,
next-open orders, actual account feedback, financial credit assignment, risk gating and temporal
evaluation. The critic, target exposures and risk threshold are engineering hypotheses. They
are not claimed to be identified fruit-fly cells or biologically validated equations.

## Protocol fixed before the run

- Fetch one public daily BTC-USDC snapshot through fruit-fly-fund's Kraken adapter. Drop the
  still-forming daily candle, require a continuous series, and hash the saved snapshot.
- Keep the first 21 observations for feature warmup. Split the remaining observations 60/20/20
  in increasing time into train, validation and test. Never train on the validation or test
  windows. No model or threshold selection uses their returns in this first experiment.
- Use fixed modeling assumptions: 8 bps commission and 2.5 bps half-spread per fill. These are
  not claimed to be a live exchange fee tier or historical observed spreads. Capital is 10
  USDC and the upstream maximum order is 10 USDC. Upstream minimums and execution vetoes stay
  active; residual inventory and veto counts are reported.
- Decide at the completed daily close, execute at the next daily open, and receive the net
  marked-to-bid equity change at the next completed close. A 1 ms ordering offset distinguishes
  the close event from the following open at their shared daily boundary. This is a bar-level
  execution convention, not a latency or intrabar order-book model.
- Include actual spread and commission in account feedback. Check each fill against our
  library's cost function and independent cash/quantity arithmetic. The terminal position
  remains marked to bid; returns do not include an assumed terminal liquidation fee.
- Freeze weights for evaluation. Create fresh dynamic state for each evaluation window and
  copy only learned parameters/efficacies. Reset accounts between windows. Validation is
  reported but is not carried into test training or selection.

## Compact model

The opportunity memory retains upstream sparse KC expansion and opposing MBON weights. The
encoder's position and partner channels are suppressed, and its session channel is replaced
with market UTC time. Lag history restarts at each evaluation boundary; supplied prices remain
past-only. This is a deliberate separation of opportunity cues from account-dependent action.

Actions are target LONG or FLAT, not repeated unconditional buys. The upstream guard can veto
an order and an order cap can make complete liquidation take multiple bars. Learning uses the
actual portfolio outcome, including inherited exposure; this is not a counterfactual estimate
of the isolated causal contribution of one fill.

An ordinary TD critic sees the sparse code, a bias and actual inventory. Its discount is 0.95,
reward is scaled by 100, TD error is clipped to [-1, 1], and critic step size is 0.05. The actor
uses `(action - probability_long) * TD_error`: a rewarded FLAT choice reduces the BUY preference.
Upstream `MushroomBody.reinforce` applies this signed value with eta 0.01. This replaces the
upstream market mode's action-independent positive-reward-to-BUY mapping.

Trainable arms update on identical batches of 16 completed transitions. The shuffled arm
deranges reward-to-transition pairing within each matured batch, preserving the batch's reward
multiset. It does not invert reward signs or use future batches. A one-element final batch
cannot be shuffled and retains its reward. Batching is an engineering control, not a claim
about biological eligibility duration. Decision probabilities are recorded in memory until
feedback; later batch updates use those behavior probabilities.

The ordinary actor uses the same sparse input and critic but a linear preference update
(step size 0.1), not the MBON plasticity rule. Its initial preference is zero. The compact fly's
readout and initialization differ, so this is a practical comparator, not a parameter-matched
proof that a biological rule alone caused a difference.

The optional risk gate sets the target to FLAT when 20-day realized volatility exceeds its
training-set 80th percentile. A gated observation does not update the opportunity-memory
actor, though its financial critic can update. This is a risk-budget proxy. It is not the
claim that cash, loss or volatility is literally hunger. The threshold is fitted on the
training window and is not an online estimate during training.

Arms: fly, fly with gate, frozen fly, frozen fly with gate, shuffled feedback, ordinary actor,
ordinary actor with gate; seeds 11, 23, 37, 53 and 71. Cash and buy-and-hold use the same paper execution guard.
Five seeds describe algorithmic variability on one path, not five independent market samples.

The frozen-with-gate control was added on 2026-09-22; it is absent from the archived v1 results.
It keeps initial actor and critic weights unchanged while applying the same volatility gate
as the learned arm. `--paired-gate` runs only `fly_gated` and `frozen_gated`, using the same
seeds, costs and chronological splits, and requires an existing market snapshot. Neither
validation nor test updates weights in either arm. No result for this new comparison has
been recorded yet.

`beacon_gate_control.sh NEW_RADFM_ROOT PRIOR_RADFM_ROOT` is the dedicated scheduled-job
entry point. Stage the current source under `NEW_RADFM_ROOT/source` first. It reuses the
prior pinned upstream checkouts, Python environment and `data/kraken-daily.json`, verifies
the two upstream commit IDs, and puts new results and caches under the new root. Both roots
must be distinct direct `fin-skills-fly-*` children of `/beacon-projects/radfm/wy891`.
Submit with absolute `--chdir`, `--output` and `--error` paths under the new root; no job has
been submitted by adding this entry point. The script runs the real upstream integration
checks before the paired comparison, so local skipped tests are not a remote pass.

## Full graph pilot

The full-graph pilot uses the final 201 bars (21 warmup plus 180 split 60/20/20). It compares
upstream plastic and frozen networks with the same visual input and fixed decoder. Only learned
efficacies transfer into fresh frozen evaluation brains. It retains upstream pulse scheduling
and local neural eligibility, rather than pretending to implement the compact TD actor.

The pilot measures whether the reused full graph runs, changes memory and produces account
outcomes. Its test interval differs from the longer compact comparison; do not rank their
headline returns together. It is a development check, not a generalization or profitability
claim. Neither architecture is described as a complete model of fly foraging habits.

## Run and inspect

All formal artifacts belong below `/beacon-projects/radfm/wy891/fin-skills-fly-reuse-*`, including
the environment, datasets, build cache, temporary files, logs and results. Beacon HOME is not
an output location. `beacon_prepare.sh` prepares pinned upstreams and verifies the graph;
submit it as a CPU Slurm job with account `angliece`, partition/QoS `scavenger`, and set
`FLY_REUSE_ROOT`. The first preparation used an archive of the already-reviewed checkouts;
its receipt preserves the archive hash. Later code snapshots are immutable.

After preparation, from this repository:

```text
python benchmarks/fly_reuse/deploy.py --credential-file <local credential path> --remote-root <RADFM root> --receipt <new local receipt.json>
```

Deployment submits compact and dependent full-graph jobs. It refuses existing source snapshots
and receipts. For a failed submission or run, inspect job state and preserve the failure before
retrying; do not overwrite a result directory. A remote runtime needs the library's scipy
dependency as well as upstream dependencies. `beacon_prepare.sh` includes it.

Local checks with the pinned source checkouts available:

```text
python -m pytest benchmarks/fly_reuse/test_core.py -q
```

Every output window has a SQLite ledger, fills, decision/return trace and summary. The top-level
protocol records dates, settings, source hashes and runtime. `results.json` is written only on
completion; a partial file is not a completed result. The experiment uses public development
data, and test returns will be disclosed; subsequent tuning on those returns must not be
reported as new unseen evidence.
