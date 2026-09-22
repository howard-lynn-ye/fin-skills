# Library utility pilot: completed results

Recorded 2026-09-21 from the completed Beacon artifacts. The fixed experiment produced
mixed effects: full-library access reduced losses for the 14B model, but worsened the
7B model's mean result. It does not establish that library access improves financial
returns, nor does it demonstrate a high-return trading model. No model was trained.

The [research review and protocol](LIBRARY_UTILITY_RESEARCH.md) describe the intervention
and its limits. These are indicative FX price-only proxy returns, excluding interest and
carry, under assumed proportional costs. They are not executable investment returns.

## Completion and provenance

- Remote root: `/beacon-projects/radfm/wy891/fin-skills-utility-20260921-v1`.
- Slurm `1614147`: `COMPLETED`, exit `0:0`, elapsed `00:04:11`, NVIDIA L40S.
- Instrument tests: `13 passed in 2.69s` in `logs/tests-1614147.txt`.
- Two Qwen2.5-Coder backbones, three conditions, two seeds: 12 completed trials.
  All 48 decisions were valid; none used the invalid-output no-rebalance fallback.
- Initial source archive SHA256:
  `b0519d22f764d76c646af7bd562229e4e104e705dc81eefb40cedb5c8bb56a04`.
- Protocol digest in `results/protocol_sha256.json`:
  `1133b8c41a94ff135380bbbf854458422b28c332b2ea49a5d095cd888b755eea`.
- SHA256 of completed `results/summary.json`:
  `4f2359e27a792241900466a42e3885697c7af239a2e83549d9871494b4531bbd`.
- Initial decision uses history through 2024-12-31; evaluation ends 2025-05-02 and
  contains 84 reference observations. The fixture assigns 20:00 UTC timestamps;
  these are accounting conventions, not independently observed publication times.
- Raw prompts, model responses, tool receipts, frozen actions and ledgers remain in RADFM.
  This document reports aggregates read from those artifacts. No raw run download occurred.

## Primary result at 5 bps

Returns and drawdowns below are percentages; changes between returns are percentage points.
Means average the two inference seeds on the same market path, not independent markets.
`full_library` means the selected skill excerpts plus actual supplied API outputs. It does
not mean every module in the library was used or that the model autonomously chose tools.

| Model | Condition | Mean proxy return | Mean maximum drawdown | Mean risky exposure | Mean turnover |
|---|---|---:|---:|---:|---:|
| 7B | No library | -2.498890% | -3.168316% | 1.000000 | 2.761087 |
| 7B | Skill text | -3.240580% | -3.996918% | 1.000000 | 2.019586 |
| 7B | Text and executed library tools | -2.953700% | -3.320026% | 0.997514 | 4.992962 |
| 14B | No library | -3.634233% | -4.597846% | 1.000000 | 2.599239 |
| 14B | Skill text | -3.204928% | -4.120525% | 1.000000 | 2.591391 |
| 14B | Text and executed library tools | -3.185417% | -3.966039% | 1.000000 | 2.408818 |

Full-library minus no-library: **-0.454810 percentage points for 7B** and
**+0.448816 percentage points for 14B**. The incremental full-library advantage over
skill text was +0.286880 points for 7B and +0.019511 points for 14B. In this pilot,
most of the 14B difference was already present in the text condition.

### Every seed, without selecting favorable runs

| Model | Seed | No library | Skill text | Full library | Full minus no library |
|---|---:|---:|---:|---:|---:|
| 7B | 11 | -1.236130% | -3.240580% | -2.953700% | -1.717570 pp |
| 7B | 23 | -3.761650% | -3.240580% | -2.953700% | +0.807950 pp |
| 14B | 11 | -3.634233% | -3.204928% | -3.185417% | +0.448816 pp |
| 14B | 23 | -3.634233% | -3.204928% | -3.185417% | +0.448816 pp |

The repeated-seed results are identical in several conditions. Repetition does not
create independent evidence of robustness. Every reported paired date-block 95% interval
includes zero. Those intervals concern mean daily return differences, not cumulative
return differences; they are descriptive for a short correlated path, not confirmatory
tests. Full-library intervals, in unscaled daily-return units, are:

| Model | Seed | Mean daily difference | Descriptive 95% interval |
|---|---:|---:|---|
| 7B | 11 | -0.000207337 | [-0.000488202, 0.000006667] |
| 7B | 23 | 0.000097783 | [-0.000160945, 0.000296447] |
| 14B | 11 | 0.000054343 | [-0.000029117, 0.000135426] |
| 14B | 23 | 0.000054343 | [-0.000029117, 0.000135426] |

## Controls and cost sensitivity

At 5 bps, zero-interest cash returned 0%; equal-weight buy-and-hold returned -3.210711%
with -3.969710% maximum drawdown. The library's deterministic rule policy returned
-1.077280% with -1.280846% maximum drawdown, but its mean risky exposure was only
0.293704. Its smaller loss cannot be attributed to better asset selection without a
risk-matched control. Cash outperformed every tested policy on this particular path.

The following rescores use the same frozen decisions; models were not rerun for each cost.
All entries are mean cumulative proxy returns over the same two seeds.

| Model | Condition | 0 bps | 5 bps | 20 bps |
|---|---|---:|---:|---:|
| 7B | No library | -2.364624% | -2.498890% | -2.900554% |
| 7B | Skill text | -3.142799% | -3.240580% | -3.533330% |
| 7B | Full library | -2.711046% | -2.953700% | -3.677990% |
| 14B | No library | -3.508887% | -3.634233% | -4.009289% |
| 14B | Skill text | -3.079404% | -3.204928% | -3.580519% |
| 14B | Full library | -3.068717% | -3.185417% | -3.534674% |

## Resource use and interpretation

The run consumed 347,268 prompt tokens and 2,338 completion tokens. Per four-decision
trial, the following are means over seeds. Generation limits were matched; actual
prompt size and consumed compute were not. Slurm elapsed time includes tests and loading.

| Model | Condition | Mean prompt tokens | Mean completion tokens |
|---|---|---:|---:|
| 7B | No library | 22,717 | 195 |
| 7B | Skill text | 29,949 | 187.5 |
| 7B | Full library | 34,151 | 188 |
| 14B | No library | 22,721 | 200 |
| 14B | Skill text | 29,945 | 201 |
| 14B | Full library | 34,151 | 197.5 |

Inspection of the seed-11 full-library actions found that 7B initially described its
allocation as inverse volatility, then switched to concentrated positions. Its turnover
was greater than either other 7B condition. The 14B reasons described diversification,
and its allocations remained fully invested. These are observations of generated actions
and stated reasons, not proof of an internal reasoning mechanism or causal mediation.

This study establishes that a bounded same-model ablation can produce valid, independently
scored outputs using real library API receipts. It does not yet validate a positive
economic-utility contribution. In particular, it does not evaluate historical news,
executable prices, autonomous tool selection, a trained selector, or generalization beyond
one short FX path and one model family. Public-data pretraining contamination remains
unresolved. No window, seed or model was removed because its result was unfavorable.

The next confirmatory study needs the data and controls listed in the protocol: permissible
point-in-time data, separate development and untouched evaluation periods, independent
markets, risk-matched and competing-tool controls, and a frozen prospective evaluation.
Any policy or training change motivated by this pilot belongs to development and must be
evaluated on new untouched data. Neither the 14B gain nor the 7B loss should be generalized
to the whole library from this experiment.
