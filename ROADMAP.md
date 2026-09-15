# Roadmap

Updated 2026-09-14. The acceptance record for the recovered Claude work and current
implementation is [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md).

## Implemented

- Importable skill texts and executable modules, a shared `Bundle`/guard API, JSON tools
  and an optional MCP server.
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
- Worked examples, an API documentation site, the v0.1.0 release and active CI on Ubuntu
  with Python 3.11–3.13. The original release predates this completion work.

## Remaining validation and distribution

- Refresh blind skill-selection measurements whenever the listing changes. Historical
  scores apply to their saved listing, not automatically to an expanded catalog.
- Extend active CI to Windows, Python 3.10 and minimum dependency versions using the
  reviewed `ci/validate.yml` template. The current token lacks `workflow` scope; existing
  CI remains active. See [ci/README.md](ci/README.md).
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
