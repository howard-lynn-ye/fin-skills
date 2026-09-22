# Published mushroom-body model reproduction and optional memory extension

This standalone research directory reproduces numerical outputs from Huang, Luo,
Woo et al., *Nature* (2024), DOI `10.1038/s41586-024-07819-w`.
It is not a trading controller or a whole-brain model.

Upstream: https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model

Pinned commit: `5d7c08a9a88f923169a0c3008aca68af421e9a7f`.

The upstream model and the translations/derived harnesses here are
**GPL-3.0-or-later**, copyright 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer
for the original model. Python translation, causal adaptation and execution
harnesses were added 2026-09-21. See `COPYING.txt` and module provenance.
This GPL extension is installed and distributed separately from the MIT
`fin_skills` core package. The optional `fly_memory` adapter imports it when
explicitly created; no upstream source or data is vendored into the core package.
A local pinned checkout is in ignored
`runs/fly-paper-reproduction-20260921/upstream`.

## Optional installation and reuse

From the repository root, install separately with
`pip install ./benchmarks/fly_paper`, then use
`fin_skills.model_zoo.create_model("fly_memory", circuit_parameters=...)`.
Supply explicit parameter matrices or use `fin_skills_fly.load_parameters()` with
a trusted published parameter file. No pretrained parameters are bundled.

`model.py` preserves the offline published simulator. `online.py` changes the clock
to use observed conditioning events and provides a non-mutating readout. Financial
reinforcement and biological-time scaling are engineering hypotheses, not a claim
of biological or trading validation. `fin_skills.model_zoo` requires matured,
single-use feedback receipts before updating the memory. See the
[model usage guide](../../docs/MODEL_USAGE.md) and
[integration record](../../docs/MODEL_INTEGRATION.md) for the reusable interface;
the numerical reproduction and separate development assays are described below.

## Frozen numerical scope

- Fig. 5c / Extended Data Fig. 10j: both fitted model variants, both odor pairs,
  all 10,000 supplied parameter vectors, all six imaging times. The native
  harness calls the author's unmodified parameter and simulation functions.
- Central-parameter trajectories additionally compare every experimental bout's
  odor adaptation, KC activity, effective weights and neuronal response.
- Fig. 5e: complete published point-estimate grid, intact and removed feedback.
- Elementwise agreement requires `atol=1e-8`, `rtol=1e-9`, equal array shapes
  and matching nonfinite masks. Native and Python output arrays are retained.
- The source spreadsheet supplies means and SEMs; fit errors are descriptive
  on the original fitting data. No refitting or held-out prediction is claimed.

`run_native.m` omits plotting and spreadsheet loading from the original example;
it exports raw arrays using the original model functions. The exact data mapping
is performed separately in `run_port.py`. The original confidence-interval
quantiles use midpoint plotting positions, corresponding to NumPy `hazen`.

The author's ten-step solver, matrix orientation, negative effective weights,
odor adaptation and three-hour retention transition are preserved. The latter
depends on the last training bout in the entire supplied protocol; this is an
**offline simulator**, not an admissible online financial state update. A later
online adapter must explicitly define a causal clock and cannot quietly claim
to remain an exact reproduction.

## Execution

Local MATLAB R2024a exited before producing a reference. Beacon has no preloaded
MATLAB/Octave module. An attempted Apptainer image unpack failed on RADFM NFS
extended attributes. Both failures are retained, not counted as model failures.
The fallback installs conda-forge Octave under the RADFM experiment directory;
`OCTAVE_HOME` must point to that prefix. It executes the original MATLAB
functions using Octave, so this is not a same-runtime MATLAB reproduction.

All Beacon runtime files, package caches, temporary files, histories, output and
scheduler logs stay under `/beacon-projects/radfm/wy891/fin-skills-memory-paper-*`.
HOME is neither used as an output target nor reassigned. Initialization files and
Octave command history are disabled. Deployment uses the existing pinned host
key and writes an explicit source-hash manifest before submission.

Deployment phases are `prepare`, `run` (native only), `port` (comparison/plots),
`bridge` (interface assays), then `control` (known-answer sequential tasks).
`--stage-name` selects a fresh retry directory;
`--reproduction-stage` selects the completed comparison whose gate the bridge
must check. Python spreadsheet/plot dependencies are installed with `--no-deps`
into the stage directory, preserving the shared NumPy/SciPy runtime. The initial
combined job completed native work but failed on missing `openpyxl`; a separate
comparison attempt found missing `matplotlib`, then an Octave metadata-format
issue. The successful attempt and all failures are listed in `RESULTS.md`.

Run the narrow numerical and online-interface guardrails with:

```powershell
python -m pytest -q benchmarks/fly_paper/test_model.py benchmarks/fly_paper/test_online.py
```

These tests do not establish profitable trading or independent paper replication.

After native execution, run:

```text
python benchmarks/fly_paper/run_port.py --upstream <checkout> --native <native-output> --output <new-output>
```

The remaining transition to trading is gated on these comparisons. Positive and
negative return learning, long-horizon credit assignment, persistence/termination,
and fin-skills feedback checks require separate tests; numerical replay alone
does not validate those engineering additions.

## Separate interface development

`online.py` is explicitly an adaptation: its retention clock resets after an
observed conditioning event, and probe reads do not mutate memory. It preserves
the original activation, recurrent response and plasticity equations, but its
schedule is causal. A single-training protocol test checks numerical agreement
where the native and causal clocks share the same anchor.

`run_bridge.py` is a forced-exposure interface assay. It uses the central fitted
three-module parameters, a fixed return normalization, and the full option
log-wealth targets from `benchmarks/verified_memory/episode_credit.py`. It checks
positive delayed payoff, negative net payoff, a larger valid loss, a neutral
condition and repeated sign reversals. Its development protocol is written
before each run. These are known-answer cases, not independent held-out evidence.

The signed return-to-punishment mapping, non-mutating normalized MBON readout,
conditioning durations and clock reset are engineering choices. Readout scores
are not calibrated expected returns. The assay learns from supplied outcomes;
it does **not** yet choose when to enter, hold or exit. Do not present it as a
foraging-policy benchmark, market backtest, or evidence of financial advantage.

`episode_credit.py` uses the real `synthesis_integrity` library guard. It requires
positive cost-inclusive NAV marks, contiguous account intervals, valid lineage
and strict feedback maturity, and allows each option receipt to be consumed once.
It cannot authenticate invented source NAV values; independent execution
reconciliation remains the caller's responsibility. Learning from corrected
revisions after an already-consumed receipt is outside this minimal interface.

The interface job requires the frozen native-reproduction result to pass first,
and stages a separate immutable source snapshot under `bridge/` in RADFM.
It cannot overwrite the source or results of the native replay.

`run_control.py` adds an ordinary Monte Carlo scalar critic and reinforces the
chosen action's odor channel with return minus the critic baseline. This is an
engineered actor/critic, not a claim that the original biological model implements
financial RL. Independent circuits index simple observed states. True HOLD,
next-observation execution, fee accounting, positive cash constraints and mature
suffix-return credit are checked against an independent enumerating evaluator.
Each fixed deterministic task trains separately; these are development checks,
not generalization, market or novelty evidence. See `RESULTS.md` for both arms,
all seeds, completed jobs and the limits of the comparison.
