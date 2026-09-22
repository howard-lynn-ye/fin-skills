# First real-price backtest of the reproduced memory actor

Frozen on 2026-09-21 before inspecting market returns. This is an exploratory
development backtest, using the same previously inspected ECB fixture as v3.

- Data: USD, JPY, GBP and CHF reference rates inverted to EUR-denominated assets;
  2005-2015 chronological training, 2016-2025 evaluation with continued online learning.
- Five exploration seeds: 11, 23, 37, 53, 71. One pass per seed, no seed selection.
- Candidate: the same two-channel `CausalMemory` actor and scalar critic as the
  completed control assay. Published central neural parameters remain fixed.
- Observable state: fixed bins of average normalized 5/20/60-observation trends,
  holding status and holding age. No future-fit thresholds or absolute date lookup.
- Decisions every five observations: wait/hold, or enter/exit. Entry buys an equal
  basket. HOLD retains shares without rebalancing. A fixed 60-observation holding
  limit closes positions to bound feedback delay; forced exits are not learned actions.
- Credit: chosen actions receive their complete entry-to-exit suffix log-NAV return,
  including actual entry/exit fees. Waiting receives its realized cash return. Losses
  are not clipped. Feedback is consumed once, strictly after it becomes available.
- Primary fees: 5 bps of traded notional; orders execute at the next observation.
  Terminal liquidation is charged. An independent ledger checks every evaluated NAV.
- Comparison: ordinary Monte Carlo value learner with the same state/actions,
  exploration (0.2), learning rate (0.1), costs and chronology; equal-basket buy-and-hold
  and cash. v3 is historical context only because its state/action space differed.
- Reporting: all five seeds, arithmetic mean cumulative net return, minimum and
  maximum, primary-cost NAV traces; 0/20 bps replays use the frozen 5 bps actions.
- No carry, funding, executable quotes or real trades. This is a price-only proxy,
  not a total-return FX investment. Biological time remains conditioning exposure,
  not market clock time. No parameter search is part of this run.

Local financial and causal checks: 21 passed. Future-price perturbations preserve
all preceding actions, feedback and NAV. Price arrays use a fixed contiguous layout
to make floating-point reduction order deterministic across dataframe copies.

Deployment is under `/beacon-projects/radfm/wy891/fin-skills-memory-paper-20260921-mamba/market/`.
The first `beacon` partition submission was rejected at its association submission
limit; no job was created. Job **1616977** was subsequently accepted by the permitted
`scavenger` partition/QOS using the same frozen source snapshot. Requeue is disabled
to preserve immutable outputs if preempted. All logs, caches and temporary files
remain under the RADFM experiment root.
