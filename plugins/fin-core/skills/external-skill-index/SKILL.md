---
name: external-skill-index
description: >-
  A verified index of every public finance Agent Skill repository — 139 repos, 4,851 SKILL.md files
  — so you can find what already exists instead of rebuilding it, and avoid the third that is
  legally unusable. TRIGGER - looking for an existing skill, plugin or marketplace for anything
  financial; "is there already a skill for X"; choosing between competing finance skill packs;
  before writing a new finance skill; checking whether a skill repo's licence permits use; or asked
  what the Claude/agent finance skill ecosystem contains. Also load before recommending any
  third-party finance skill repository, because 44 of the 139 declare no usable licence and 12
  advertise skills while shipping none.
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-04"
---

# External finance skill index

Every field below came from a live `gh api repos/<o>/<r>` call on 2026-09-04. Skill counts are
literal `SKILL.md` matches in each repo's git tree, not README claims. The machine-readable catalog
is **`../../../../catalog/external-skills.json`** (139 rows) — grep it rather than guessing:

```bash
python -c "import json;d=json.load(open('catalog/external-skills.json',encoding='utf-8'));\
print('\n'.join(f\"{r['full_name']:<45}{r['stars']:>6}★ {r['license'] or 'NONE':<12}{r['skill_count']:>4} skills\" \
for r in d['repos'] if 'backtesting' in r['tags']))"
```

## 1. The size of the ecosystem

| | |
|---|---|
| Repositories indexed | **139** |
| Raw `SKILL.md` files | **4,851** |
| After removing mirrors and vendored copies | **4,556** |
| Finance-dominant | **3,127** — the fairest single number |

⚠️ **Volume is not depth.** `quantskills` alone is 230 repos and 1,094 skills, but **803 of those
(73%) sit in three auto-generated factor repos.**

## 2. 🚨 A third of it is legally unusable

| Licence | Repos |
|---|---|
| MIT | 76 |
| Apache-2.0 | 14 |
| MIT-0 | 1 |
| **🚨 NONE** | **33** |
| **🚨 NOASSERTION** | **11** |
| GPL-3.0 | 2 (+ the `quantskills` org, 192 of 230) |
| AGPL-3.0 | 1 |
| archived | 2 |

**91 safe (65%) · 44 legally unusable (32%).** No licence is **more restrictive than GPL** — under
Berne it means all rights reserved, so you may not copy, adapt or redistribute it at all.

🚨 **And it hits exactly the repos you would reach for first:**

| Repo | Stars | Licence |
|---|---|---|
| `okx/onchainos-skills` | 329★ | **NONE** |
| `lzwme/finance-quant-skills` | 321★ | **NONE** |
| `ALAGENT-HKU/x2strategy` | 270★ | **NONE** |
| `marketcalls/vectorbt-backtesting-skills` | 200★ | **NONE** |

**Check the licence before you copy a single file.** Stars are not permission.

## 3. 🚨 Twelve repos advertise skills and ship none

**~1,333 stars sit on repos with zero `SKILL.md` files**, including
`quant-sentiment-ai/claude-equity-research` (709★ — actually a slash-command plugin) and
`angieruiz17/claude-fintech-skills` (144★ — a TypeScript app). The word "skills" in a repo name is
marketing, not a format claim. The catalog records `skill_count: 0` for each.

## 4. 🚨 Neither awesome-list has a finance section

- **`ComposioHQ/awesome-claude-skills`** — 74,466★, and its only finance-adjacent entry is an
  **"Invoice Organizer"**.
- **`travisvn/awesome-claude-skills`** — 14,966★, **nothing**.
- **`anthropics/skills`** — 20 skills, **zero finance**, confirmed.

**There is no curated finance section anywhere.** If you are looking for a discovery route, you
would be creating that section, not joining it.

## 5. Where to send someone instead of duplicating

Of 139 repos, **only 9 overlap quantitative research methodology**, 3 of those are GPL or
unlicensed, and the permissive-and-still-maintained remainder holds **13 stars in total**.

| Need | Go to | Licence |
|---|---|---|
| Leakage-safe quant ML, ML-pipeline framing | **`ml4t/skills`** (Stefan Jansen, 61 skills) | Apache-2.0 |
| Crypto/DeFi/Solana execution, MEV, prediction markets | `agiprolabs/claude-trading-skills` (68) | MIT |
| RIA compliance, KYC/AML, Reg BI, GIPS, practice ops | `JoelLewis/finance_skills` (91) | MIT |
| Personal/SMB bookkeeping and tax | `openaccountant/skills` (44) | MIT |
| ⚠️ A-share factor mining | `quantskills/*` | **GPL-3.0**, Pandadata-locked |

⚠️ **`ml4t/skills` is the closest peer and worth reading.** Two structural notes, verified: it has
**no `.claude-plugin/` manifest** while using category directories, so Claude Code will not discover
its skills at `.claude/skills/<category>/<skill>/`; and it uses non-spec frontmatter keys
(`when_to_use`, `dependencies`) that hard-error on claude.ai upload. Its *content* is strong; its
*packaging* targets a different client.

## 6. The three gaps nobody covers

1. **Research integrity with any traction.** 12 repos tag it; the credible ones have **<15 stars or
   are GPL**. Meanwhile **26 repos ship discretionary TA** — ICT/SMC, Chan Theory, and one bundling
   Vedic astrology with options Greeks.
2. **Fixed income, rates and derivatives-pricing correctness.** `fixed-income` appears on **2 of
   139** repos, `derivatives` on 3. **Nothing** covers day counts, calendars, curve bootstrapping,
   OIS discounting, or the QuantLib traps.
3. **Point-in-time and survivorship-safe fundamental data, vendor by vendor.** 21 repos wrap
   market-data APIs; **not one documents *when* the data was knowable.** `point-in-time` appears in
   exactly 2 places, neither vendor-specific.

## 7. How to use this index

- **Before writing a new skill**, grep the catalog for its tags. If something permissive already
  covers it, cite it — this library's position is that duplicating a good MIT skill is waste.
- **Before recommending a repo**, check `license` and `skill_count` in the catalog. A third of the
  ecosystem fails one of those two.
- **`relation`** on each row is `complementary` (80), `overlapping` (9) or `avoid` (50), with the
  reason in `caution`.
- The catalog is a **snapshot dated 2026-09-04**. Star counts and licences change; re-verify before
  acting on a licence claim.
