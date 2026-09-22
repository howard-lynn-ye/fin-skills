# Fruit-fly workstream: circuit mechanisms, version 2

The first pilot tested random expansion plus a generic full-information LMS readout.
Its negative trading result is not a sufficient test of mushroom-body learning.
This version keeps the fruit-fly workstream and tests additional circuit mechanisms.

## What is implemented

The PN-to-KC layer uses a fixed sparse random expansion of nonnegative input channels.
An APL-inspired winner-take-all operation retains a small active KC population. Each
candidate action has opponent approach/avoidance output pools. A signed reinforcement
prediction error depresses avoidance synapses after better-than-predicted outcomes
and approach synapses after worse-than-predicted outcomes. Plasticity affects the
chosen action compartment and the KCs active at its originating decision. A saved,
decaying eligibility vector assigns delayed outcomes to that original activity.

This is an engineered rate-based approximation. It is not a FlyWire/MaleCNS graph,
spiking simulation, full-brain emulation, or exact reproduction of a published model.
The output pools, chosen-action mapping, normalized features, bounded weights, trace
time scale, reward scaling, epsilon-greedy policy and background recovery are explicit
engineering choices. We make no claim to have discovered a new biological rule.

Motif sources, consulted 2026-09-21:
- Shen, Dasgupta and Navlakha, *Algorithmic insights on continual learning from fruit
  flies*: sparse expansion, inhibition and compartment-specific learning/freezing.
  https://arxiv.org/html/2107.07617v2
- Bennett, Philippides and Nowotny, *Learning with reinforcement prediction errors in
  a model of the Drosophila mushroom body*: opponent output valence and modeled RPE.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8105414/
The second source presents an RPE model; it does not establish that all biological
dopamine signals are RPEs. Our simple depression rule differs from its complete model.

## Controlled comparisons

`fly_trace` is the candidate. `dense_trace` removes KC inhibition while preserving
the number of output parameters and the same local update rule. `fly_no_trace` applies
the delayed signal to the current KC activity rather than the saved origin activity;
it retains the originating action compartment and prediction. `fly_frozen` disables
all learning. `fly_absolute` removes the predicted-value subtraction from the dopamine
signal. `linear_trace` is a simpler chosen-action linear delta-rule control. The linear
control has fewer parameters; it is not a parameter-matched biological ablation.

Fixed settings are declared in `fly_v2.PARAMETERS`. They are not selected by returns
from the previous pilot or tuned against this run. Normalized code magnitude and the
learning rate are shared across sparse/dense variants. All variants use the same
exploration policy and budget. Opponent saturation is reported rather than hidden.

First test delayed cue acquisition, retention of an earlier cue group during learning
of another, and adaptation after a reward reversal. Correct choices earn +1; incorrect
choices earn -0.3. All models see the same cue sequence; rewards depend on their chosen
actions. Probe scores account for ties and do not train the model or consume its RNG.
These authored tasks establish mechanism operation, not scientific benchmark replication.

Then reuse the previous synthetic paths and historical ECB proxy with frozen settings.
The environment constructs potential expert payoffs, but passes the learner only the
chosen expert's reward. This differs from version 1's all-expert supervision; comparisons
to version 1 are descriptive, not a controlled architecture contrast. A payoff still
means standalone expert entry/hold/liquidation under the fixed cost model, not the
incremental reward of the final continuing portfolio. That modeling limitation remains.

All variants run with clean feedback and verified updates. The candidate additionally
runs paired checked/unchecked conditions on clean and known-corrupt streams. The
library's lineage guard, accounting adapter, source hashes, trial ledger, independent
final scorer and causal timing contract are retained. Gating clean identical feedback
is expected to leave actions unchanged; no clean-data economic advantage is claimed
for the gate itself. Five initialization seeds are repeats, not independent markets.

This is exploratory follow-up after observing version 1. The old runs remain intact.
The runner records source hashes and the full cell list before any results. All Beacon
outputs use a new RADFM directory. No live trading or external data download is used.
