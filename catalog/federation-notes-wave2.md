# Federation notes - wave 2

Eighty-four new marketplace entries drawn from 60 upstream repositories, ready to append to
`plugins` in `.claude-plugin/marketplace.json`. Same mechanism as
[`catalog/federation-notes.md`](../../catalog/federation-notes.md) - nothing is copied into this
repository, every entry installs disabled, and the description of each says in its first sentence
that its claims are not verified here. Verified **2026-09-09**.

## What changed since wave 1

Wave 1 pinned nothing: its eight external entries name a repository and, for `okx-agent-skills`, a
branch. **Every wave-2 entry pins both `ref` and a full 40-hex `sha`**, so what a user installs is
the tree that was read on the verification date rather than whatever the upstream default branch
holds later. The cost is that entries go stale silently; the benefit is that the `verified_on` date
in `metadata` now describes an immutable object. Re-pinning is a routine maintenance task - re-run
`wave2_reverify.py` and regenerate.

## What was verified for each pack, on 2026-09-09

Live, read-only, through `gh api`:

1. the repository exists, is **not archived**, is not a fork, and its owner and default branch;
2. its licence SPDX id (all 60 are MIT, MIT-0, Apache-2.0 or ISC - `report.py`'s permissive list);
3. `pushed_at` and star count;
4. the 40-hex HEAD commit SHA of the default branch;
5. the full recursive git tree **at that SHA**, from which every `SKILL.md` under each entry's
   declared paths was counted - the `skill_count` in each entry's `metadata` is that count, not a
   README claim;
6. `.claude-plugin/plugin.json` / `marketplace.json` manifests where present;
7. the README, and a sample of up to fourteen `SKILL.md` files per entry, parsed for frontmatter and
   scanned for credential-shaped file paths, API-key terms, order-placement terms and safety terms.

**Nothing inside the skills was verified.** No claim in any of these packs has been reproduced, and
no pack was installed - installation writes to `~/.claude` and clones third-party code.

Credential handling was checked mechanically: every tracked file matching `.env`, `credentials`,
`secrets`, `api_keys`, `*.pem` or `*keystore*` was listed, and the small ones were read. **No
committed secret was found in any of the 60 repositories.** The only tracked non-example `.env` in
the set is `dashboard/.env.production` in `mnemox-ai/tradememory-protocol`, whose entire content is
`VITE_USE_MOCK=true`. Everything else is a `.env.example`, a `.env.sample`, or a CI/tooling config.
Every pack that needs credentials takes them from environment variables, a vendor CLI login, or the
OS keychain.

## The 84 entries

| Entry | Source | ref | sha (pinned) | Skills | Licence | Upstream pushed |
|---|---|---|---|---:|---|---|
| `a-share-skill` | `shouldnotappearcalm/a-share-skill` | `main` | `84946232ef36...` | 5 | MIT | 2026-06-24 |
| `ai-asset-pricing` | `alexander-m-dickerson/ai-asset-pricing` | `main` | `64aa61cb217f...` | 38 | MIT | 2026-04-19 |
| `ai-berkshire` | `xbtlin/ai-berkshire` | `main` | `d9c124b73a02...` | 22 | MIT | 2026-09-07 |
| `ai-realestate` | `zubair-trabzada/ai-realestate-claude` | `main` | `d435ddd36346...` | 14 | MIT | 2026-04-29 |
| `aicoin-coinos-skills` | `aicoincom/coinos-skills` | `main` | `2a22bd53d710...` | 6 | MIT | 2026-08-29 |
| `algo-trading-skills` | `himanshuj16/algo-trading-skills` | `main` | `101ee444712e...` | 501 | Apache-2.0 | 2026-09-07 |
| `alpaca-skills` | `alpacahq/alpaca-skills` | `main` | `39111abee6b6...` | 14 | Apache-2.0 | 2026-09-08 |
| `alpha-stack` | `tmcga/alpha-stack` | `main` | `eefe76525242...` | 47 | MIT | 2026-05-13 |
| `alphaear-skills` | `RKiding/Awesome-finance-skills` | `main` | `853f09b4d0ba...` | 10 | Apache-2.0 | 2026-03-29 |
| `alphagbm-skills` | `alphagbm/skills` | `main` | `6ecea742ed80...` | 30 | MIT | 2026-07-09 |
| `blofin-skills-hub` | `blofin/blofin-skills-hub` | `main` | `64f4a0ba6b43...` | 9 | Apache-2.0 | 2026-03-19 |
| `china-stock-research-skills` | `spikehongg/china-stock-research-skills` | `main` | `d49f1f360dea...` | 7 | MIT | 2026-03-09 |
| `coinmarketcap-skills` | `opencmc/skills-for-ai-agents-by-coinmarketcap` | `main` | `d4c95dbc9762...` | 8 | MIT | 2026-03-02 |
| `coinstats-cli` | `coinstatshq/coinstats-cli` | `main` | `2231157bc3dd...` | 12 | MIT | 2026-03-10 |
| `cpa-skills` | `adoptai/cpa-skills` | `dev` | `7a592d3a8800...` | 23 | MIT | 2026-09-09 |
| `crypto-com-agent-trading` | `crypto-com/crypto-agent-trading` | `main` | `92e80c552731...` | 2 | Apache-2.0 | 2026-08-29 |
| `deneux-cre-skills` | `sasha-deneux/claude-skills-cre` | `main` | `0b10514d0ee1...` | 10 | MIT | 2026-07-15 |
| `doramagic-skills` | `tangweigang-jpg/doramagic-skills` | `main` | `5df4e00b215d...` | 82 | MIT-0 | 2026-04-25 |
| `duan-yongping-skill` | `kangarooking/duan-yongping-skill` | `main` | `4498bed1e6e1...` | 15 | MIT | 2026-04-17 |
| `eastmoney-miaoxiang` | `meission/eastmoney` | `master` | `0e2dfce23b56...` | 3 | MIT | 2026-03-19 |
| `foundational-research-skills` | `foundationalresearch/skills` | `main` | `6f70fa1ce8d8...` | 14 | MIT | 2026-03-03 |
| `fs-equity-research` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/equity-research` | `main` | `69cbc81467a5...` | 9 | Apache-2.0 | 2026-08-25 |
| `fs-financial-analysis` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/financial-analysis` | `main` | `69cbc81467a5...` | 13 | Apache-2.0 | 2026-08-25 |
| `fs-fund-admin` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/fund-admin` | `main` | `69cbc81467a5...` | 6 | Apache-2.0 | 2026-08-25 |
| `fs-investment-banking` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/investment-banking` | `main` | `69cbc81467a5...` | 9 | Apache-2.0 | 2026-08-25 |
| `fs-lseg` | `anthropics/financial-services` &rarr; `plugins/partner-built/lseg` | `main` | `69cbc81467a5...` | 8 | Apache-2.0 | 2026-08-25 |
| `fs-operations` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/operations` | `main` | `69cbc81467a5...` | 2 | Apache-2.0 | 2026-08-25 |
| `fs-private-equity` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/private-equity` | `main` | `69cbc81467a5...` | 10 | Apache-2.0 | 2026-08-25 |
| `fs-spglobal` | `anthropics/financial-services` &rarr; `plugins/partner-built/spglobal` | `main` | `69cbc81467a5...` | 3 | Apache-2.0 | 2026-08-25 |
| `fs-wealth-management` | `anthropics/financial-services` &rarr; `plugins/vertical-plugins/wealth-management` | `main` | `69cbc81467a5...` | 6 | Apache-2.0 | 2026-08-25 |
| `fscn-china-finance` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/china-finance` | `main` | `59e97ee66833...` | 31 | Apache-2.0 | 2026-06-09 |
| `fscn-fund-admin` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/fund-admin` | `main` | `59e97ee66833...` | 6 | Apache-2.0 | 2026-06-09 |
| `fscn-investment-banking` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/investment-banking` | `main` | `59e97ee66833...` | 10 | Apache-2.0 | 2026-06-09 |
| `fscn-operations` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/operations` | `main` | `59e97ee66833...` | 2 | Apache-2.0 | 2026-06-09 |
| `fscn-private-equity` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/private-equity` | `main` | `59e97ee66833...` | 9 | Apache-2.0 | 2026-06-09 |
| `fscn-wealth-management` | `jwangkun/claude-for-financial-services-cn` &rarr; `vertical-plugins/wealth-management` | `main` | `59e97ee66833...` | 5 | Apache-2.0 | 2026-06-09 |
| `ftshare-skills` | `FTShare-Lab/FTShare-skills` | `main` | `7181b63f8f04...` | 215 | MIT | 2026-09-07 |
| `fund-investment-guide` | `taxueseek/fund-investment-guide` | `main` | `7a0c994db232...` | 9 | MIT | 2026-09-03 |
| `gajetoso-financeskills` | `GAJETOso/financeskills` | `main` | `862774c15d83...` | 48 | MIT | 2026-09-01 |
| `geeksfino-finskills` | `geeksfino/finskills` | `main` | `8722415a68db...` | 30 | Apache-2.0 | 2026-03-05 |
| `himself65-data-providers` | `himself65/finance-skills` &rarr; `plugins/data-providers` | `main` | `0a5759bca1ea...` | 6 | MIT | 2026-08-27 |
| `himself65-market-analysis` | `himself65/finance-skills` &rarr; `plugins/market-analysis` | `main` | `0a5759bca1ea...` | 11 | MIT | 2026-08-27 |
| `htx-skills-hub` | `htx-exchange/htx-skills-hub` | `main` | `3f0b3e34c491...` | 17 | MIT | 2026-06-08 |
| `investorskills` | `questflowai/investorskills` | `main` | `2844d7809a73...` | 63 | MIT | 2026-08-23 |
| `jquants-cli` | `J-Quants/jquants-cli` | `main` | `6dfdf94303e9...` | 1 | MIT | 2026-08-27 |
| `kraken-cli` | `krakenfx/kraken-cli` | `main` | `aa56e5976be5...` | 59 | MIT | 2026-08-07 |
| `krzysu-market-skills` | `krzysu/market-skills` | `main` | `31e8ee1888aa...` | 45 | MIT | 2026-09-03 |
| `langalpha-research` | `ginlix-ai/LangAlpha` &rarr; `plugins/langalpha_research` | `main` | `2a1b537c5152...` | 16 | Apache-2.0 | 2026-09-08 |
| `llmquant-skills` | `LLMQuant/skills` | `master` | `1918237467c2...` | 18 | MIT | 2026-05-30 |
| `longbridge-skills` | `longbridge/skills` | `main` | `03c5fde151fb...` | 13 | MIT | 2026-08-27 |
| `macro-skills` | `fatfingererr/macro-skills` | `master` | `13e538802829...` | 37 | MIT | 2026-02-03 |
| `marian2js-trading-skills` | `marian2js/trading-skills` | `main` | `f1ae7d481154...` | 18 | MIT | 2026-03-16 |
| `mt-metatrader-platform` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/metatrader-platform` | `main` | `44fd4c790ed2...` | 3 | MIT | 2026-08-19 |
| `mt-metatrader-research` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/metatrader-research` | `main` | `44fd4c790ed2...` | 3 | MIT | 2026-08-19 |
| `mt-mql-developer` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/mql-developer` | `main` | `44fd4c790ed2...` | 1 | MIT | 2026-08-19 |
| `mt-mql5-development` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/mql5-development` | `main` | `44fd4c790ed2...` | 10 | MIT | 2026-08-19 |
| `mt-trading-fundamentals` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/trading-fundamentals` | `main` | `44fd4c790ed2...` | 1 | MIT | 2026-08-19 |
| `mt-trading-web-systems` | `algotradingspace-dev/metatrader-skills` &rarr; `plugins/trading-web-systems` | `main` | `44fd4c790ed2...` | 1 | MIT | 2026-08-19 |
| `nansen-cli` | `nansen-ai/nansen-cli` | `main` | `782e33d30fbf...` | 34 | MIT | 2026-09-09 |
| `octagon-skills` | `OctagonAI/skills` | `main` | `51e938c4d086...` | 67 | MIT | 2026-06-05 |
| `okx-agent-trade-kit` | `okx/agent-trade-kit` | `github-main` | `c4e54501e6f3...` | 10 | MIT | 2026-09-07 |
| `openalgo-execution-skills` | `marketcalls/openalgo-execution-skills` | `main` | `1b8eceeb404f...` | 7 | MIT | 2026-04-28 |
| `openclaw-data-china-stock` | `shaoxing-xie/openclaw-data-china-stock` | `main` | `6361207818a7...` | 7 | MIT | 2026-05-05 |
| `pionex-skills` | `pionex-official/pionex-skills` | `main` | `60e42c95de2c...` | 12 | MIT | 2026-07-24 |
| `polymarket-paper-trader` | `agent-next/polymarket-paper-trader` | `main` | `e4263988a8a0...` | 1 | MIT | 2026-08-15 |
| `poorvith-skills-finance` | `poorvith-mp/skills-finance` | `main` | `54a310adb09e...` | 24 | MIT | 2026-09-06 |
| `quant-review-skills` | `shakeebshaan/claude-code-quant-skills` | `main` | `6b39f8feb4a7...` | 6 | MIT | 2026-04-17 |
| `ruflo-neural-trader` | `ruvnet/ruflo` &rarr; `plugins/ruflo-neural-trader` | `main` | `498a23879984...` | 9 | MIT | 2026-09-09 |
| `serenity-skill` | `haskaomni/serenity-skill` | `main` | `dedcf8f9ca8b...` | 6 | MIT | 2026-07-15 |
| `shinkoku` | `kazukinagata/shinkoku` | `main` | `607574d5e5ce...` | 24 | MIT | 2026-09-09 |
| `skills-il-tax-and-finance` | `skills-il/tax-and-finance` | `master` | `0cc2c0498d56...` | 42 | MIT | 2026-09-09 |
| `stock-sdk-mcp` | `chengzuopeng/stock-sdk-mcp` | `main` | `6b57f40b3c42...` | 5 | ISC | 2026-05-24 |
| `tradememory-plugin` | `mnemox-ai/tradememory-protocol` &rarr; `tradememory-plugin` | `master` | `e9ca736e40ca...` | 3 | MIT | 2026-09-08 |
| `trading212-api` | `trading212-labs/agent-skills` &rarr; `plugins/trading212-api` | `master` | `edd9dc870b16...` | 1 | MIT | 2026-09-08 |
| `upstox-skills` | `upstox/upstox-skills` | `master` | `1ae72b1a59e8...` | 1 | MIT | 2026-06-09 |
| `vibe-trading-skills` | `HKUDS/Vibe-Trading` | `main` | `a44ed6e807e6...` | 90 | MIT | 2026-09-09 |
| `vp-re-appraisal-valuation` | `reggiechan74/vp-real-estate` &rarr; `plugins/appraisal-valuation` | `main` | `6199968b2837...` | 6 | Apache-2.0 | 2026-05-15 |
| `vp-re-common-utilities` | `reggiechan74/vp-real-estate` &rarr; `plugins/common-utilities` | `main` | `6199968b2837...` | 7 | Apache-2.0 | 2026-05-15 |
| `vp-re-expropriation-law` | `reggiechan74/vp-real-estate` &rarr; `plugins/expropriation-law` | `main` | `6199968b2837...` | 9 | Apache-2.0 | 2026-05-15 |
| `vp-re-infrastructure-corridor-ops` | `reggiechan74/vp-real-estate` &rarr; `plugins/infrastructure-corridor-ops` | `main` | `6199968b2837...` | 10 | Apache-2.0 | 2026-05-15 |
| `vp-re-leasing-commercial` | `reggiechan74/vp-real-estate` &rarr; `plugins/leasing-commercial` | `main` | `6199968b2837...` | 24 | Apache-2.0 | 2026-05-15 |
| `vp-re-tenancies-residential` | `reggiechan74/vp-real-estate` &rarr; `plugins/tenancies-residential` | `main` | `6199968b2837...` | 3 | Apache-2.0 | 2026-05-15 |
| `wshobson-quantitative-trading` | `wshobson/agents` &rarr; `plugins/quantitative-trading` | `main` | `a30778f8c4e6...` | 2 | MIT | 2026-09-07 |
| `yuping322-finskills` | `yuping322/finskills` | `main` | `dcce68f4bfae...` | 107 | Apache-2.0 | 2026-02-21 |

## Where one entry does not equal one repository

**Two monorepos are federated one plugin directory at a time**, because a single entry for either
would blow the ~20-skill discovery budget many times over and there is no way to exclude part of a
flat `skills/` directory once it is loaded.

- `anthropics/financial-services` (118 `SKILL.md` in the tree) becomes **nine** entries: the seven
  `plugins/vertical-plugins/*` plugins plus the two partner-built ones, `lseg` and `spglobal` - 66
  skills in total, the largest of them 13.
- `jwangkun/claude-for-financial-services-cn` (188 in the tree) becomes **six** entries from
  `vertical-plugins/*` - 63 skills, the largest 31.
- `reggiechan74/vp-real-estate` becomes six, `algotradingspace-dev/metatrader-skills` six, and
  `himself65/finance-skills` two, following each repository's own `plugins/` split.

**Three large single-directory packs stay as one entry**, with the count stated in the description
because there is nothing to split:

| Entry | Skills | About the ~20-skill budget |
|---|---:|---|
| `algo-trading-skills` | 501 | ~25x. Upstream's own marketplace splits these into per-domain plugins, but each of those uses `source: "./"` with an explicit `skills` list - a shape that only works from inside that marketplace. Federating the repository exposes all 501. |
| `ftshare-skills` | 215 | ~10x. A catalogue of one-endpoint skills; enable for a session, not as a standing install. |
| `yuping322-finskills` | 107 | ~5x. |
| `vibe-trading-skills` | 90 | ~4.5x. |
| `doramagic-skills` | 82 | ~4x. |
| `octagon-skills` | 67 | ~3.4x. |
| `investorskills` | 63 | ~3x. |
| `kraken-cli` | 59 | ~3x. |
| `gajetoso-financeskills`, `alpha-stack`, `krzysu-market-skills`, `skills-il-tax-and-finance`, `macro-skills`, `nansen-cli`, `fscn-china-finance`, `alphagbm-skills`, `poorvith-skills-finance`, `shinkoku`, `cpa-skills`, `vp-re-leasing-commercial` | 48 down to 24 | 1.2x-2.4x |

All of them install disabled, so the budget is the user's to spend; the descriptions say the number
rather than hiding it.

## Caveats per pack

Beyond what each description already says.

### Skills paths

Most entries need no `skills` field: the loader scans `skills/` under the source by default, and
paths in `skills` only *add* to that scan - they cannot exclude anything. Sixteen entries name paths
because their skills live elsewhere, and two of those exploit the additive semantics to take a
subset:

- `marian2js-trading-skills` lists seven of the eight subdirectories under `skills/` and leaves out
  `skills/live-trade`, whose eToro skill places real orders. This is the only place in wave 2 where
  a live-order capability is excluded rather than declared - 18 of 19 skills.
- `ftshare-skills` lists `./ftshare-market-data/sub-skills` only, leaving the repository's router
  skill out - 215 of 216.
- `polymarket-paper-trader` lists `./.claude/skills` (1 skill) and leaves the older `./skill`
  directory, which holds a deprecated alias, out.
- `alpaca-skills` lists `./skills/trading-api` and `./skills/broker-api` because `skills/` itself
  holds only those groups plus a `templates/` placeholder - 14 skills.
- `htx-skills-hub` lists `./skills/htx` for the same reason.

**The one unverified path form.** `skills-il-tax-and-finance` (42 skills) and `a-share-skill` (5)
keep their skill directories at the repository root, so both entries use `"skills": ["./"]`. The
documented semantics - a path is "a directory containing `<name>/SKILL.md`" - make this correct, and
`scripts/validate.py` accepts it, but `catalog/federation-notes.md` already lists a bare `"./"` as
untested at install and this wave does not change that. If it fails, the fallback is to list the 42
(or 5) directories individually and accept the second reading of the docs. `crypto-com-agent-trading`
has the same root-level layout but needs no `skills` field: its own `plugin.json` declares
`"skills": "./"`.

### Frontmatter that Claude Code accepts and claude.ai does not

The Agent Skills spec allows six frontmatter fields. Twenty-three entries sample skills carrying
extra keys. They load in Claude Code; they hard-error on claude.ai upload, the Skills API and
`package_skill.py`. This matters only if a user tries to lift a skill out of a federated pack and
upload it - the federated install itself is unaffected.

| Entry | Non-spec frontmatter keys seen in the sampled skills |
|---|---|
| `ai-asset-pricing` | `user_invocable` |
| `ai-realestate` | `author`, `command`, `output`, `skill`, `tags`, `triggers`, `version` |
| `aicoin-coinos-skills` | `required_environment_variables` |
| `alphagbm-skills` | `globs` |
| `blofin-skills-hub` | `auto_activate`, `version` |
| `coinmarketcap-skills` | `user-invocable` |
| `crypto-com-agent-trading` | `user-invocable` |
| `deneux-cre-skills` | `version` |
| `duan-yongping-skill` | `related_skills`, `source_book`, `source_chapter`, `tags` |
| `eastmoney-miaoxiang` | `credentials`, `required_env_vars` |
| `htx-skills-hub` | `auth`, `auth_required`, `risk`, `risk_level`, `version` |
| `investorskills` | `invest` |
| `kraken-cli` | `version` |
| `krzysu-market-skills` | `version` |
| `llmquant-skills` | `category`, `input_data_source` |
| `openalgo-execution-skills` | `argument-hint` |
| `openclaw-data-china-stock` | `author`, `tags`, `triggers`, `version` |
| `polymarket-paper-trader` | `version` |
| `poorvith-skills-finance` | `deprecated`, `group` |
| `ruflo-neural-trader` | `argument-hint` |
| `skills-il-tax-and-finance` | `version` |
| `stock-sdk-mcp` | `author`, `requires`, `tags`, `version` |
| `vibe-trading-skills` | `category` |

### Live orders

Eleven entries carry the FEDERATE\* flag and say so in their first two sentences, following the
`okx-agent-skills` precedent: `trading212-api`, `upstox-skills`, `crypto-com-agent-trading`,
`aicoin-coinos-skills`, `nansen-cli`, `okx-agent-trade-kit`, `pionex-skills`, `blofin-skills-hub`,
`htx-skills-hub`, `krzysu-market-skills`, `openalgo-execution-skills`.

Three further entries can reach a real account and are **not** flagged FEDERATE\*, because the
upstream default is paper and the promotion to live is an explicit act by the user - the fact is
still stated in each description: `kraken-cli` (paper by default, live once keys are set),
`alpaca-skills` (separate paper and live keys), `longbridge-skills` (read-only except the portfolio
skill, which places orders behind a confirmation).

`trading212-api` is the sharp one: `T212_ENV` **defaults to `live`**, so it is the only pack in this
wave that reaches a real account without an opt-in. Its description leads with that.

### Upstreams that have gone quiet

Idle since before 2026-06, so content may have drifted from the market it describes - regulatory,
tax and market-structure packs are the ones to check before relying on: `macro-skills` (2026-02-03),
`yuping322-finskills` (2026-02-21), `coinmarketcap-skills` (2026-03-02), `foundational-research-skills`
(2026-03-03), `geeksfino-finskills` (2026-03-05), `china-stock-research-skills` (2026-03-09),
`coinstats-cli` (2026-03-10), `marian2js-trading-skills` (2026-03-16), `blofin-skills-hub` and
`eastmoney-miaoxiang` (2026-03-19), `alphaear-skills` (2026-03-29), `quant-review-skills` and
`duan-yongping-skill` (2026-04-17), `ai-asset-pricing` (2026-04-19), `doramagic-skills` (2026-04-25),
`openalgo-execution-skills` (2026-04-28), `ai-realestate` (2026-04-29), `openclaw-data-china-stock`
(2026-05-05), `alpha-stack` (2026-05-13), `vp-re-*` (2026-05-15), `stock-sdk-mcp` (2026-05-24),
`llmquant-skills` (2026-05-30), `octagon-skills` (2026-06-05).

### Drift found between the sweep and re-verification

Nine values moved in the 24 hours between the first pass and re-verification. Two changed a skill
count, so the entries carry the new number.

| Repository | Field | 2026-09-08 sweep | 2026-09-09 re-verify |
|---|---|---|---|
| `HKUDS/Vibe-Trading` | pushed_at | 2026-09-08 | 2026-09-09 |
| `kazukinagata/shinkoku` | pushed_at | 2026-03-21 | 2026-09-09 |
| `skills-il/tax-and-finance` | pushed_at | 2026-09-01 | 2026-09-09 |
| `skills-il/tax-and-finance` | SKILL.md count | 40 | 42 |
| `mnemox-ai/tradememory-protocol` | pushed_at | 2026-08-11 | 2026-09-08 |
| `ruvnet/ruflo` | pushed_at | 2026-09-08 | 2026-09-09 |
| `adoptai/cpa-skills` | pushed_at | 2026-07-30 | 2026-09-09 |
| `adoptai/cpa-skills` | SKILL.md count | 22 | 23 |
| `nansen-ai/nansen-cli` | pushed_at | 2026-09-08 | 2026-09-09 |

## Dropped

**No repository was dropped at re-verification.** All 60 still exist, none is archived or a fork,
every licence is unchanged and permissive, every declared skill path still holds `SKILL.md` files at
the pinned SHA, and no repository handles credentials in a committed plaintext file. What follows is
what was deliberately left out inside repositories that *were* taken, plus the earlier exclusions
that still stand.

| Dropped | Why |
|---|---|
| `anthropics/financial-services` `plugins/agent-plugins/*` (10 plugins, 51 skills: earnings-reviewer, gl-reconciler, kyc-screener, market-researcher, meeting-prep-agent, model-builder, month-end-closer, pitch-agent, statement-auditor, valuation-reviewer) | Sub-agent packs whose skills largely restate the vertical plugins already taken. Nineteen entries from one repository is more marketplace noise than the incremental content justifies. Counted and verified; available if wanted. |
| `jwangkun/...-cn` `agent-plugins/*` (4 plugins) | Same reason, and each of the four reports the same 31 skills as `vertical-plugins/china-finance` - the directories appear to re-expose one skill set, so federating them would quadruple-count. |
| `marian2js/trading-skills` `skills/live-trade` (1 skill) | The eToro skill places real orders; excluding it keeps the entry read-only, which is what the rest of the pack is for. |
| `FTShare-Lab/FTShare-skills` root router skill | Redundant next to the 215 sub-skills, and taking it would require a bare `"./"` path. |
| `agent-next/polymarket-paper-trader` `./skill` (2 skills) | Deprecated alias of the `.claude/skills` skill. |
| `himself65/finance-skills` `plugins/social-readers`, `plugins/ui-tools` | Out of scope for a finance library. |
| `mnemox-ai/tradememory-protocol` `.skills/strategy-validator` | The most interesting skill in the repository (deflated Sharpe, walk-forward, regime, CPCV) sits outside the plugin directory, so `git-subdir` cannot reach it. Worth watching - if upstream moves it into `tradememory-plugin/`, re-pin. |
| `alpacahq/alpaca-skills` `skills/templates` | Placeholder, no `SKILL.md`. |
| `poorvith-mp/skills-finance` - 12 redirect stubs | They load (they are real `SKILL.md` files with a `deprecated` key) and cannot be excluded from a flat `skills/`; the entry states that half of its 24 skills are stubs rather than pretending otherwise. |
| The nine repositories in `catalog/federation-notes.md` "Considered and left out" | Unchanged: `JoelLewis/finance_skills`, `ajeeshworkspace/indian-trading-skills`, `ml4t/skills`, `gauss314/skills`, `openaccountant/skills`, `alirezarezvani/claude-skills`, `prof-little-bear/cc-equity-research`, `BaggaT236/AI-Trading-Skills`, and the platform-locked group. |
| 416 SKIP repositories | Categorised in `r1_external_skills.md`. The two largest buckets are 152 with no usable licence and 73 with no `SKILL.md` at all. |

## CONVERT - worth lifting into a native skill, not federatable as it stands

Eight repositories hold content this library wants but in a shape that cannot be a marketplace
entry. For each: what to take, and what makes it unfederatable.

| Repository | Licence / stars | What is worth converting | Why it is not skill-shaped as it stands |
|---|---|---|---|
| [YichengYang-Ethan/oracle3](https://github.com/YichengYang-Ethan/oracle3) | Apache-2.0, 253 | **The highest-value item in this list.** A prediction-market pricing engine: Wang transform with MLE-fitted coefficients, model Greeks and Kelly sizing, paper-traded on Kalshi, Polymarket and DFlow, backed by a working paper (Yang 2026, SSRN). `fin-models` has no event-contract pricing family at all. | Five unique skills, duplicated across `./skills`, `./.claude/` and `./.agent/`; the live-ops skill is bound to the app's credential flow. Convert the pricing method, cite the paper, drop the ops. |
| [simonlin1212/a-stock-data](https://github.com/simonlin1212/a-stock-data) | Apache-2.0, 9,737 | An endpoint map for 60 A-share endpoints across 22 zero-auth sources (mootdx, Tencent, Baidu, 东财/同花顺/iwencai research reports, 龙虎榜, 解禁, margin, F10), with unit tests that extract the code out of the `SKILL.md` itself. Fold the map into `china-ashare-data` and the `lib-akshare` fallback list. | A single root `SKILL.md` - `catalog/federation-notes.md` already rules that shape out. |
| [monarchjuno/tradingcodex](https://github.com/monarchjuno/tradingcodex) | Apache-2.0, 368 | Anti-overfitting review checklists (leakage, snooping, costs, capacity), a data-QC gate, point-in-time replay memory, forecast records, and an order approve/submit separation. The closest external match to `research-integrity-guards`. | Skill paths are nested inside `workspace_templates/modules/.../.agents/skills` and bound to `$tcx-` commands; there is no plugin directory to point at. |
| [ericwang915/valueclaw](https://github.com/ericwang915/valueclaw) | MIT, 8 | Fixed-income templates `fin-core` names as a gap: bond YTM, duration and convexity, yield-curve inversion off FRED, credit spreads, FX carry, options flow, plus Dalio risk-parity and Marks-cycle frameworks. | 97 templates inside `value_claw/templates/skills/**` - an application's template tree, not a plugin; the IB template can place orders. |
| [HammerGPT/Hyper-Alpha-Arena](https://github.com/HammerGPT/Hyper-Alpha-Arena) | Apache-2.0, 1,160 | An 86-factor crypto library with IC/ICIR scoring and a 69-function expression engine - a crypto counterpart to the alpha zoos in `vibe-trading-skills`. | The nine skills under `backend/skills` are the application's own operator prompts; the factor library is Python inside the app, and execution runs through the app's exchange keys. |
| [tigersking520/stock-analysis-skill](https://github.com/tigersking520/stock-analysis-skill) | MIT, 33 | A falsification-first equity research playbook: 五步法, 财报逆向分析/排雷, valuation-overreach checks, market-cap back-solve, expectation gap, next-quarter verification lines. | One root `SKILL.md`. |
| [lisonevf/finance-skills](https://github.com/lisonevf/finance-skills) | MIT, 2 | `code-review-finance` - a review pass over research code for look-ahead, leakage, decimal handling and survivorship. A good shape for a `research-integrity-guards` companion. | Ten skills mixed with three Chinese discretionary trading systems and a TDX-protocol data skill; the useful part is one skill in a pack whose rest is out of scope. |
| [PatrickJS/awesome-cursorrules](https://github.com/PatrickJS/awesome-cursorrules) | CC0-1.0, 40,747 | `rules/alpha-skills-quant-factor-research.mdc`, the only finance rule in the canonical Cursor rules list. | A Cursor `.mdc` rules file, not a skill - and its content derives from `VernonOY/alpha-skills`, which this library already cites. Lowest priority of the eight. |

## Runtime behaviour still not verified

Unchanged from wave 1, plus one addition:

- whether an entry `name` that differs from the upstream `plugin.json` `name` loads cleanly (most of
  wave 2 renames, e.g. `deneux-cre-skills` for a repository whose manifest says `claude-skills-cre`);
- whether a bare `"./"` skills path resolves (two entries depend on it);
- whether `defaultEnabled: false` plus a pinned `sha` interact as expected on `plugin update`;
- installation itself, which was not exercised.

## Downstream edits this wave requires

Appending the 84 entries to `plugins` in `.claude-plugin/marketplace.json` is not the whole change.
Three other places in the repository assert a number that this wave falsifies:

1. **`plugins/fin-core/skills/external-skill-index/SKILL.md`** - its final section is headed
   "Eight of these install through this marketplace already" and carries an eight-row table dated
   2026-09-08. With this wave it becomes 92 entries from 68 upstream repositories. The section
   heading, the intro sentence, the table and the closing paragraph ("Why only eight, and why not
   the rest of the 139") all move. The 84 new rows will not fit the existing one-row-per-pack
   format inside a skill that is already 8,131 chars against a ~20,000-char guidance ceiling - the
   table should collapse to one row per upstream repository with a pointer to
   `catalog/federation-notes.md`, or move into `references/`.
2. **The same skill's frontmatter description** claims "139 repos, 4,851 SKILL.md files" and "44 of
   the 139 declare no usable licence". This sweep examined 484 repositories fully and thousands
   more mechanically; those figures now describe an older, smaller index.
3. **`catalog/external-skills.json`** (139 rows, snapshot dated 2026-09-04) and
   `catalog/federation-notes.md` (verified 2026-09-08) both predate this wave. The federation notes
   need this file merged into them or referenced from them; the catalog needs the 60 new upstream
   repositories, or an explicit statement that it is the wave-1 snapshot.

`scripts/validate.py` will pass on the marketplace change alone - its marketplace block checks
shape, not the prose in a skill - so nothing here is caught automatically. That is the argument for
doing the index edit in the same commit.
