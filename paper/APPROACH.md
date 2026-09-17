# Paper approach - notes for discussion

Status: draft for Howard and Shwai to agree on before anything is written.
Written 2026-09-17 against `744e7f5`. Every number below was produced by a script in this
repository on that commit; nothing here is an estimate unless it says so.

---

## 1. What kind of paper this is

The proposal is a **library paper**: the contribution is `fin-skills` itself, and the
experiments exist to show the library does what it claims. It is deliberately *not* a
benchmark paper - "here is a new benchmark for LLM research agents" is a different paper with
a different bar, and writing it would push the library into a supporting role.

Three contributions, in the order they should appear:

1. **The library.** 129 skills whose claims each carry a verification date and a source
   marker, 36 executable guards behind one `Bundle.check()` contract, 53 JSON/MCP tools an
   agent can call, and a collection layer for public disclosures.
2. **An audit method that does not use the library to grade itself.** Re-run a submitted
   pipeline against modified copies of its own market: redraw the future after a cut date,
   redraw one session at a time, scramble one input file at a time, and replay session by
   session with only the data that existed at each decision. See `benchmarks/agent_study/`.
3. **What that method measures.** Guard recall on planted defects, and a controlled study of
   research agents with and without the library.

## 2. Where it can be submitted, and what gates each venue

Checked 2026-09-17:

| Venue | Length | Gate | Earliest for us |
|---|---|---|---|
| arXiv preprint | any | none | now |
| ACL/NAACL **System Demonstrations** | 4 pages | a working system; no adoption requirement | NAACL 2027 demo CFP not yet posted; historically ~3-4 months before the conference (June 2027) |
| JOSS | short | repository public **> 6 months**, evidence of research use | ~March 2027 |
| JMLR MLOSS (where PyOD published) | 4 pages | evidence of an **active user community** | depends on adoption |
| NAACL 2027 **main track** | 4 or 8 | a research contribution; a library description alone is a standard reject | ARR deadline 2026-10-12 - only if we frame it as method + findings |

PyOD's own path was arXiv first (January 2019), JMLR the same year. That ordering works for
us: preprint now, demo track next spring, JOSS once the six months are up.

## 3. What exists today

**The library**, on `master`, CI green across Linux/Windows/macOS and Python 3.10-3.13:
129 skills validated against the 6-field Agent Skills spec, 36 guards, 53 tools,
3,082 tests passing, 124/124 skill scripts running clean, 82% coverage.

**Guard recall** (`benchmarks/guard_bench.py`): a synthetic world with 12 planted research
defects, repeated at three independent seeds. All 12 caught in every world, zero false
alarms on the clean data, zero guard errors.

**The agent study** (`benchmarks/agent_study/`, new). A seeded market is exported as a
workspace with four rewarded shortcuts: a news feed stamped with the session it describes, an
LLM score whose training window covers the fitting period, a listing table with delisting
dates, and quoted prices with splits still in them. The honest ceiling is measured, not
assumed - trading the generator's own latent drift with a one-day lag earns **net Sharpe
1.63**; the wrong-side join earns **71.5**. So a high number is itself evidence.

Instrument calibration, on two reference submissions we wrote ourselves:

| Reference | future leakage | live replay, 25 sessions |
|---|---:|---:|
| honest (lagged feed, adjusted prices) | 0.00 | +5.07 bps/day, Sharpe 2.99 |
| leaky (same-session feed, full-sample scaler) | 0.78 | **-1.17 bps/day, Sharpe -0.58** |

The leaky one earns 22.8 on paper in a year it never saw - hiding the future does not punish
it, because the session's own commentary exists in every copy of the market. Only the
session-by-session replay, which withholds data the way the clock does, takes it away. That
distinction is worth a paragraph in the paper.

**Pilot, 12 agent runs** (3 markets x Claude Opus/Haiku x with/without the library, one run
per cell, full results in `benchmarks/agent_study/PILOT_RESULTS.json`):

| Arm | mean abs. Sharpe gap | future leakage | same-session dependence | used the contaminated LLM score |
|---|---:|---:|---:|---:|
| Opus, no library | 0.19 | 0.00 | 0.00 | 0/3 |
| Opus + fin-skills | 0.03 | 0.00 | 0.00 | 0/3 |
| Haiku, no library | 0.57 | 0.00 | 0.00 | 2/3 |
| Haiku + fin-skills | 11.95 | 0.00 | 0.42 | 2/3 |

Three readings, all n=3 and all provisional:

- The strong model did not take any of the four shortcuts. The library's visible effect there
  is on *reporting*: claimed minus recomputed Sharpe of 0.03 with it against 0.19 without.
- The weak model's failures were not look-ahead. They were trusting a field whose training
  window covers the fitting period, and holding names that had already delisted.
- **The weak model plus the full toolkit was the worst arm in the study.** One run claimed
  0.24 while its own positions earned 27.96; another claimed 0.13 against -4.86. It also
  cited guards it had not run. If this survives a larger sample it is the most interesting
  sentence in the paper, and it is not a flattering one.

**Held-out year, six runs.** Data truncated at 2022-12-30, 2023 withheld, then replayed
session by session over the first 40 sessions of 2023:

| Run | claimed | live Sharpe | live cumulative |
|---|---:|---:|---:|
| s11 no library | 2.31 | 5.39 | +3.90% |
| s11 + library | 1.94 | 5.87 | +4.44% |
| s23 no library | 2.06 | 2.51 | +2.36% |
| s23 + library | 2.40 | -0.72 | -0.64% |
| s37 no library | 3.40 | 0.55 | +0.55% |
| s37 + library | 3.29 | 0.29 | +0.30% |

At this model strength the library makes no visible difference to realised return: one win,
one loss, one tie, over a 40-session window whose standard error swamps the differences.
That is the honest current state, and we should not dress it up.

## 4. The experiments we propose to finish

All of this runs on Beacon with open-weight models, so cost is GPU hours rather than API
tokens, and an outside reader can reproduce it.

**E1 - guard recall.** Done. Reuse as-is.

**E2 - the agent study, at scale.** 12 markets x 3 model sizes x 3 conditions x 3 repetitions.
The third condition matters: *no library* / *skills text only* / *skills + executable guards*.
Only the difference between the last two isolates what the guards add, and a reviewer will ask
for exactly that. Metrics: shortcut incidence, claimed-minus-recomputed gap, and realised
return under session-by-session replay. Engineering still needed: a minimal agent scaffold
(shell + file tools, roughly 200 lines) so an open model can drive the task.

**E3 - parity and cost.** Adapter outputs against PyPortfolioOpt / skfolio / statsmodels, and
guard runtime against data size. Half a day, no model calls.

**A prediction, recorded before the runs** so it cannot be adjusted afterwards: shortcut
incidence rises as the model gets weaker; skills-only improves reporting fidelity modestly;
skills-plus-guards improves it most; and **realised return does not differ across conditions**,
because the library does not produce alpha. If that last one holds, the paper says so - the
claim is that it stops you believing a false result, not that it makes you money.

## 5. Possibilities, and what each costs

- **P1 - pure library paper.** Sections: introduction, design, engineering, validation (E1 +
  a compact E2), limitations. Target arXiv now, demo track in spring, JOSS in March. Lowest
  risk. The weakness is that "we built a library" carries the paper, and the validation has
  to do real work.
- **P2 - library plus evaluation methodology**, in the shape of CheckList (ACL 2020): the
  tool and the way to test with it are presented together, and the evidence is that users
  equipped with it produce better-audited work. This is P1 with E2 promoted, and it is the
  only framing that could hold up in a main track.
- **P3 - benchmark paper.** Strongest novelty, wrong emphasis for what we want to publish,
  and it needs a much larger study. Recommend against for now.
- **P4 - two papers.** P1/P2 now; the benchmark later, once E2 has run at full scale and we
  can report confidence intervals. They cite each other.

Recommendation: **P2, written so it degrades gracefully into P1** if E2 does not scale in
time.

## 6. Open questions for Shwai

1. **Scope.** Do the A-share production tools (`research/production/`: daily advisor, paper
   trading, Core-Satellite, webhook delivery) belong in this paper? My view is no - they are a
   separate application and the paper's spine is research integrity - but they are your work
   and the call is yours.
2. **Authorship** - order, affiliations, and who is corresponding.
3. **Beacon** - which open-weight models are available and what GPU budget we can use. That
   fixes the size of E2.
4. **Length and venue** - 4-page preprint and demo track, or 8 pages aimed at a main track?
5. **The KOL work.** The embedded profiles are now pseudonymous and marked as an illustrative
   fixture, because the statistics came from the external `stock_prediction` audit and no
   script here reproduces them. If that audit's data and code can travel with the paper, this
   could become a real contribution; if not, it stays out.
6. **Prediction registry** - are you willing to freeze the section 4 prediction in the
   repository before the runs? It costs nothing and it is exactly the discipline the library
   argues for.

---

### 中文摘要

这份文档是给 Shwai 的讨论稿，主要问六件事，在第 6 节：实盘那套工具要不要写进论文、署名、Beacon
上有哪些模型和多少 GPU、写 4 页还是 8 页、KOL 数据的出处能不能一起公开、以及愿不愿意在跑实验前把
预期结果先冻结在仓库里。

其余内容：我们主张写一篇**库的论文**而不是基准论文（第 1 节）；各个投稿去处的门槛（第 2 节）；
现在已经有的结果（第 3 节，含守卫召回、12 次代理试点、6 次实盘重放，诚实上限 1.63 对作弊 71.5）；
还要补的实验（第 4 节，放 Beacon 上用开源模型跑，并提前写死预期结果）；四种可能的写法与推荐
（第 5 节，推荐 P2，必要时退回 P1）。
