# Roadmap

Updated 2026-09-22. The acceptance record for the recovered Claude work and current
implementation is [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md).

## Next priorities: library quality and the paper

The next milestone is a reproducible library release with a matching manuscript. RAG,
collection storage, skills and model adapters should form workflows that another researcher
can install, run and inspect. Adding another catalog entry is secondary to that goal.

| Priority | Work | Acceptance criterion |
|---|---|---|
| P0: compatibility | Diagnose the failing current-dependency and slow-test CI jobs; retain the passing minimum-dependency and distribution checks. | Every advertised supported environment passes its required checks, or the documented support range is narrowed with a specific reason. |
| P0: manuscript delivery | Keep the outline, LaTeX implementation description and evidence status consistent; deliver each manuscript revision to GitHub and the existing Overleaf project. | A reviewed diff, successful LaTeX build, verified remote file contents and a record connecting the GitHub revision to the Overleaf revision. See [paper/SYNC.md](paper/SYNC.md). |
| P1: user workflows | Extend the existing examples into tested, installed-package workflows: skill retrieval; collected records to timestamp-aware RAG; and model execution with checked feedback. | Each documented command runs outside the source tree; outputs retain sources, timestamps and configuration. Missing dependencies, credentials, empty retrieval and unavailable feedback produce explicit outcomes. |
| P1: API and support | Define stability for public imports, JSON schemas and saved formats; document model operations, dependency groups and compatibility/migration rules. | Users can distinguish adapter availability, installed backends and tested service access; a saved artifact can be restored in the declared supported environment. |
| P2: empirical evidence | Repair real-agent artifact grading first, then run the paired audit study and retrieval/tool-use comparisons. Evaluate Jev and fruit-fly learning on their own tasks. | Frozen inputs and model revisions, complete per-attempt records, independent scoring, paired uncertainty, and measured quality/completion/cost; no benefit inferred from interface tests. |

As checked on 2026-09-22, mainline CI run `35754566972` completed with failures in
several current-dependency test jobs and extended validation. Minimum dependencies,
Windows/Ubuntu packaging, Python 3.10 test jobs and the optional-backend jobs passed.
This is the status of that run, not a diagnosis or a claim about a later revision.

Paper work proceeds from the library architecture and interfaces to reproducible examples,
then measured comparisons and the resulting discussion. The existing
[experiment plan](paper/EXPERIMENTS_NEXT_ZH.md) records the evidence still needed.

## Implemented

- Importable skill texts and executable modules, a shared `Bundle`/guard API, JSON tools
  and an optional MCP server.
- Built-in [RAG pipelines](docs/RAG_PIPELINE.md), a shared model registry with a hosted
  Jev adapter, and an independently installed fruit-fly memory extension. These are
  implementation capabilities; their empirical benefits remain separate questions.
- Market data and discovery adapters, a reference research engine, synthesis, pre-trade
  checks and read-only bridges to external research outputs.
- Model and strategy families, fixed income, credit, macro, tax and alternative data.
- Deeper Japan, Hong Kong, India, Korea/Taiwan and ASEAN market skills; token events,
  perpetual funding, AMM mechanics and crypto market structure.
- Trading sessions, corporate-action processing and data-quality validation, including
  the `data_quality` API guard.
- [Public information collection](docs/COLLECTION.md): RSS/Atom, bounded static crawling,
  SEC Form 4/13F, House PTR and Bluesky. SQLite checkpoints, revisions, retry scheduling
  and an alert outbox support repeated polling. Operators choose and start their watches.
- **Pre-Trade Defense Guard Suite (2026-09-15)**: 36 registered API guards, adding:
  - `qdii_premium`: Real-time IOPV secondary market premium monitoring, soft derating (1.5%), and hard circuit breaker (3.0%) capital redirection to safe-haven assets (`518880`).
  - `board_lot_feasibility`: A-share 100-share integer lot constraints and portfolio capital granularity distortion index.
  - `cash_drag`: Detection of unmanaged demand deposit cash drag, GC001 / R-001 overnight sweep instructions, and Thursday 3-day interest multiplier optimization.
- **Persistent Portfolio State & Inflow-First Deadband Rebalancing**:
  - `PortfolioState`: JSON state serialization, real-time position valuation, and transaction logging.
  - `DeadbandRebalancer`: 5% relative drift deadband, salary deposit inflow-first allocation (zero-sell-side friction).
- **Automated 14:30 Daily Production Advisor & Multi-Channel Webhook Bot**:
  - `research/production/live_advisor_bot.py`: End-to-end 14:30 daily inspection with rich interactive card payloads for Feishu, WeChat Work, DingTalk, and terminal.
  - Comprehensive user guide: [docs/guides/live_trading_and_pre_trade_guards.md](docs/guides/live_trading_and_pre_trade_guards.md).
- Worked examples, an API documentation site, the v0.1.0 release and active CI across
  Ubuntu, Windows and macOS, including minimum-dependency and packaging jobs. The
  original release predates this completion work; CI coverage does not imply all jobs pass.

## Remaining validation and distribution

- Refresh blind skill-selection measurements whenever the listing changes. Historical
  scores apply to their saved listing, not automatically to an expanded catalog.
- Resolve the failed jobs in the active CI matrix before claiming a supported release.
  Windows, Python 3.10 and minimum-dependency jobs now exist; the older
  `ci/validate.yml` template is not evidence that they remain unimplemented.
- Publish to PyPI after the maintainer configures a trusted publisher and GitHub `pypi`
  environment. A checked wheel is not a published package.
- Link Zenodo and archive a release before adding a DOI to `CITATION.cff`.
- Directory submissions already opened await third-party review. The human-only
  awesome-claude-code form has its own eligibility requirements; do not duplicate PRs.

## Ongoing maintenance

Source API changes, vendor licenses, market-rule effective dates and third-party skills
need dated rechecks. New exchanges, sources and distribution channels such as conda-forge
can be added as concrete requirements arise. Source-specific credentials and operating a
continuous collector are deployment choices, not capabilities enabled by importing Python.

No module in this completion executes live orders. Public-disclosure tracking observes
published reports and statements, with their delays and scope, not private trading accounts.
