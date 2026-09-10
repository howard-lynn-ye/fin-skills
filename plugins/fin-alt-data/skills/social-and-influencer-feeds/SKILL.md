---
name: social-and-influencer-feeds
description: >-
  What social data you can legally and practically get in 2026, and what the part you can
  get does to a backtest. TRIGGER - Twitter or X API, tweet sentiment, cashtag, StockTwits,
  Reddit API, PRAW, r/wallstreetbets, Pushshift, scraping social media for a trading signal,
  "which accounts should I follow", influencer or finfluencer track records, "backtest a
  sentiment signal", social sentiment score, deleted tweets, "is scraping legal", hiQ v
  LinkedIn, robots.txt and terms of service for a data pipeline, "the free tier of the X
  API". Ships NO scraper and no credential path, on purpose. SKIP for congressional trades
  (congressional-trading-disclosures), for corporate insiders and Form 4 (insider-form-4),
  for fund holdings and 13F (institutional-13f), for LLM agents that read news
  (llm-finance-agents), and for the general survivorship and availability gates
  (research-integrity-guards).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-10"
---

# Social and influencer feeds

**This skill is mostly about what you cannot do, and the honest version of that is short:
in 2026 there is no free read access to X, Reddit's own help centre says academic research
through its Data API is a policy violation, and StockTwits' developer docs return 404.**
What you *can* get is a sample that has already been filtered by deletion and by your own
attention, and both filters point the same way - toward a backtest that looks better than
the live signal.

Everything marked ✅ Measured is printed by `scripts/social_feeds.py` (numpy + pandas, seed
20260910, under 2 s, no network, **no scraper, no API client, no credential path**). Its
posts are **synthetic and seeded**. Everything marked ✅ source-verified was read at the URL
given, on **2026-09-10**; 🔴 marks a page that could not be reached from here at all.

## 1. Access, checked live on 2026-09-10

### X / Twitter

🚨 **The tier model is gone.** ✅ `docs.x.com/x-api/getting-started/pricing`: *"The X API
uses pay-per-usage pricing. No subscriptions-pay only for what you use."* There is **no
Free, Basic or Pro tier anywhere in the current docs.**

| item | value, ✅ read 2026-09-10 |
|---|---|
| Posts: Read | **$0.005 per resource**, charged *"per resource returned in the response"* |
| User: Read | $0.010 · Following/Followers: Read $0.010 · Like/Mute/Block: Read $0.001 |
| Owned Reads (your own app's data) | *"$0.001 per resource (1,000 resources for $1)."* |
| Post: Create | $0.015 per request; **$0.200** with a URL |
| **Monthly cap** | *"Pay-per-usage plans are capped at 3 million Post reads per monthly billing cycle."* Above that: Enterprise. |
| Deduplication | same resource re-read inside a 24-hour UTC window is free - X calls it *"a soft guarantee"* |
| Empty balance | *"API requests will be blocked until you add credits"* |

🔴 **Could not be reached:** `developer.x.com/en/portal/products` and
`developer.x.com/en/products/x-api` return **HTTP 402** to a fetch and redirect to a login
wall in a browser; `developer.x.com/en/developer-terms/agreement-and-policy` likewise
**402**. ⚠️ **So whether legacy Free/Basic/Pro subscriptions still exist for grandfathered
accounts is UNVERIFIED.** The quotes below are from `docs.x.com/developer-terms/
restricted-use-cases`, which is X's current docs host, not from the Agreement itself.

**Redistribution**, ✅ verbatim:

- *"You may only distribute up to a total of 1,500,000 Post IDs to a single entity within a
  **30 day period**"* - per 30 days, not per day.
- Hydrated content: *"via non-automated means"* only, *"up to 50,000 hydrated public Post
  Objects and/or User Objects per recipient, per day"*, and *"should not make this data
  publicly available (for example, as an attachment to a blog post or in a public Github
  repository)."*
- Academic institutions may redistribute **unlimited** Post IDs / User IDs for
  *"non-commercial research"*.

🔑 **So a reproducible social dataset is a list of IDs, and every replicator has to re-buy
the content.** That is a reproducibility constraint, not a formatting preference.

**The three provisions that bite a finance pipeline**, ✅ verbatim:

- 🚨 *"Never derive or infer, or store derived or inferred, information about a X user's:
  ... **Negative financial status or condition**"*, and *"Credit or insurance risk
  analyses"* is a listed prohibited use.
- 🔑 The carve-out everything else rests on: *"Aggregate analysis of X content that does not
  store any personal data (for example, user IDs, usernames, and other identifiers) is
  permitted"*, subject to the rest of the Agreement. **Aggregate, and do not store
  identifiers.**
- 🚨 *"X prohibits any use of the X APIs and/or X Content to fine-tune or train a foundation
  or frontier model with the exception of Grok."*

⚠️ The Academic Research product track is **absent from the 2026-09-10 docs** - searched,
not found. "Retired in 2023" is secondhand and was not verified. v1.1
`statuses/user_timeline` is still listed in the endpoint map with **no** deprecation marker,
described as *"Limited support; use v2 for new projects."*

### Reddit

| item | ✅ read 2026-09-10, `support.reddithelp.com` |
|---|---|
| Rate limit | *"100 queries per minute (QPM) per OAuth client id"*, averaged *"over a time window (currently 10 minutes)"* |
| 🚨 Unauthenticated | *"Traffic not using OAuth or login credentials will be **blocked**, and the default rate limit will not apply."* Blocked, not throttled. |
| 🚨 **Academic research** | *"Can I perform research using Reddit developer tools and services? **No.**"* The only sanctioned route is **Reddit For Researchers**; using the API for academic research *"is a violation of our policies."* |
| Commercial use | *"You cannot use any Reddit developer tools and services for commercial purposes without first getting our permission"*; *"we'll require a contract"*; the listed examples include *"Services, research, or data access for fees"* |
| Model training | *"You may not use content on Reddit as an input for any model training without explicit consent from Reddit."* |
| **Deletion** | *"You must remove any user content in your possession that has been deleted from Reddit."* 48 hours is a *"strongly recommend"* sweep interval, **not a grace period**, and retention *"even if disassociated, de-identified or anonymized"* violates. |
| Pushshift | reinstated for *"verified Reddit moderators"* with *"use of Pushshift ... limited to moderation use cases only"*. **No public, researcher or developer path exists.** |

🔴 **`redditinc.com/policies/data-api-terms` could not be reached from this environment.**
Everything above is the help centre - official, but **not the contract**. If exact Terms
language is load-bearing for you, open it yourself.

🔑 **The deletion requirement is the same fact as §4's bias, seen from the other side.** You
are contractually obliged to destroy exactly the observations whose absence biases your
backtest. There is no configuration in which you legally hold a complete sample.

### StockTwits

✅ `stocktwits.com/developers` returns **HTTP 200** and renders a page titled *"Stocktwits
Subscriptions"* - consumer plans (Ad Free $85/yr, Edge $229.50/yr), **no endpoint
documentation, no self-serve key registration**. The only API route is an Enterprise plan
whose button reads *"Get In Touch"*. ✅ `api.stocktwits.com` returns **HTTP 302** to
`stocktwits.com`; the historical docs path `api.stocktwits.com/developers/docs` returns
**HTTP 404**.

### ⚠️ hiQ Labs v. LinkedIn - two layers, pointing opposite ways

- **CFAA layer:** 9th Cir. No. 17-16783, decided **2022-04-18**, **31 F.4th 1180**. The
  Supreme Court vacated the first panel opinion and remanded in light of *Van Buren v.
  United States*, 141 S. Ct. 1648 (2021); on remand the panel again affirmed the
  **preliminary injunction**. 🚨 **The panel held hiQ *"raised a serious question"*** whether
  the CFAA's *"without authorization"* concept reaches a bot refused access to data that
  needs no authorization generally. **That is a preliminary-injunction standard, not a
  merits holding that scraping public data is lawful.** Writing it as a flat holding - which
  most summaries do - overstates it.
- **Contract layer:** N.D. Cal. No. 17-cv-03301-EMC (Judge Edward M. Chen). November 2022
  summary judgment: **hiQ breached LinkedIn's User Agreement.** Consent judgment entered
  **2022-12-08**: **$500,000**, and a permanent injunction barring hiQ from code for data
  collection from LinkedIn *"using any of the data, source code, or algorithms developed at
  hiQ"*, requiring it to **delete that code** and all LinkedIn member profile data. ⚠️ These
  terms are quoted from a law-firm client alert, not from the docket - secondhand.

🔑 **hiQ won the CFAA question and lost the company on the user agreement.** The lesson for a
pipeline is that "not a crime" and "not a breach of contract" are different questions, and
the second one is the one that ends projects. **This library ships no scraper for any of
these sources, and will not.**

## 2. ✅ Measured - what reading costs

Arithmetic on X's own published prices, $0.005 per post read:

| budget | posts/year | USD/year | USD/month | % of the monthly cap |
|---|---|---|---|---|
| 1,000/day | 365,000 | **$1,825** | $152 | 1.0% |
| 10,000/day | 3,650,000 | $18,250 | $1,521 | 10.0% |
| 100,000/day | 36,500,000 | $182,500 | $15,208 | 100.0% |
| **at the 3M/month cap** | 36,000,000 | **$180,000** | **$15,000** | 100% |

🔑 A cashtag-level sentiment panel across a few hundred names is a five-figure annual line
item at the low end and hits the Enterprise wall before it becomes a market-wide dataset.
Budget it before you design the study.

## 3. The three timestamps

A post has **created_at**, the moment your collector could actually **see** it, and
**deleted_at**. A historical pull shows you the first and never the third, because the post
is not there either.

✅ Measured on the seeded panel (`INDEX_LAG_PMF` sets it): **55% of posts indexed the same
day**, mean **0.69 days**, p95 2, max 5. And **17.5% of posts are deleted**. Names are drawn
by attention rather than uniformly: **39 effective names out of 200**, which is why a post
count is not a bet count.

🚨 **Nobody in the panel of §4 and §5 has any skill.** Every call is a coin flip. Everything
those two sections show is manufactured by the sampling alone.

## 4. 🚨 Measured - deletion survivorship

A post whose call aged badly is deleted more often than one that aged well. A live collector
stored it; a historical pull cannot see it. 6 seeds:

| sample | hit rate | Sharpe | sd | % of posts |
|---|---|---|---|---|
| **every post** (what a live collector had) | **49.9%** | **-0.103** | 0.440 | 100% |
| **surviving posts** (what a historical pull returns) | **54.3%** | **2.733** | 0.440 | 82.5% |

✅ Measured: **a coin flip reads as 54.3% once the losers are gone - +4.4 points from
nothing.** As a function of how much more often a *wrong* post is deleted:

| extra deletion on a wrong call | apparent hit rate | true | apparent Sharpe |
|---|---|---|---|
| **+0%** (the control) | 49.8% | 50.0% | **-0.052** |
| +5% | 51.3% | 50.0% | 0.931 |
| **+15%** | **54.3%** | 50.0% | **2.901** |
| +30% | 59.8% | 50.0% | 6.187 |

🔑 **At +0% the bias vanishes, which is the control: deletion only matters when it is
correlated with the outcome. It always is.** And note the size - four points of hit rate
backtests at a Sharpe most people would not question. 🚨 **That is what volume does. 20,000
posts is enough statistical power to turn a small selection bias into a large, stable,
entirely fake edge, and social data arrives in millions.** Volume is not a defence against a
sampling bias; it is what makes one look like a discovery.

## 5. 🚨 Measured - the accounts you can name today

Same zero-skill panel. Top decile of accounts by realised call performance, 6 seeds:

| treatment | Sharpe | sd | vs all accounts |
|---|---|---|---|
| all accounts, no selection | -0.058 | 0.608 | - |
| **selected out-of-sample** (first half, measured on the second) | **-0.242** | 0.399 | -0.184 |
| **selected in-sample** (whole sample, measured on it) | **3.100** | 0.167 | **+3.159** |

🔑 **Nobody in this panel can predict anything, and in-sample selection manufactures a
Sharpe of 3.1.** That is what an influencer panel is: the accounts you can name today are
the ones that were right, and their track record is the reason you can name them. The
out-of-sample split is the only test worth running on such a list, and here it correctly
finds nothing.

⚠️ Selecting on the *first half* is still a trial, and the same argument applies to a list
of managers (`../institutional-13f/SKILL.md`) or members of Congress
(`../congressional-trading-disclosures/SKILL.md`). Log it in
`../../../fin-core/skills/backtest-validation/scripts/trial_ledger.py`.

## 6. ✅ Measured - the timestamp A/B

This one needs real skill in the panel or there is nothing for the lag to destroy: 25% of
accounts, a 6-point edge, a 5-day horizon. 6 seeds:

| key | Sharpe | sd | gap | retained |
|---|---|---|---|---|
| **created_at** (look-ahead) | **1.270** | 0.484 | +0.257 | - |
| **indexed_at** (honest) | **1.014** | 0.456 | - | **79.8%** |

🔑 The lag is **under half a day** and it costs a fifth of the signal, because the horizon is
five days. **It is the same arithmetic as the 45-day congressional lag against a 21-day
information half-life, at a hundredth of the scale** - what matters is `lag / horizon`, and
a social signal has the shortest horizon in this plugin.

## 7. Traps

- 🚨 **Backtesting on a historical pull and trading live.** §4. The two samples are not the
  same object, and the difference is signed in your favour in the backtest.
- 🚨 **Keying on `created_at`.** §6. Use the timestamp your own collector wrote.
- 🚨 **Picking the accounts after seeing their record.** §5, +3.159 of Sharpe from nothing.
- 🚨 **A vendor's "historical social sentiment" file.** Ask two questions before anything
  else: *was this collected in real time or pulled later*, and *what is the deletion policy
  for records whose source post was removed*. If the answer to the first is "pulled later",
  §4 applies to the whole file.
- 🚨 **Bot and coordinated-campaign contamination.** A cashtag stream is a target, not an
  observation. Volume spikes are the most manipulable input in this plugin, and a
  pump-and-dump generates exactly the pattern a naive sentiment signal buys.
- 🚨 **Counting posts as independent.** §3: 39 effective names out of 200. Post counts
  overstate the number of bets by an order of magnitude.
- ⚠️ **Storing identifiers.** X's aggregate carve-out is conditioned on *not* storing user
  IDs or usernames; Reddit requires deletion propagation that anonymisation does not cure.
  A design that stores a per-account panel is the design that breaks both.
- ⚠️ **Assuming a free tier because a tutorial from 2023 used one.** §1: there is no free
  tier in X's current docs, and Reddit blocks unauthenticated traffic outright.
- 🚨 **Treating hiQ as permission.** §1: it was a preliminary-injunction ruling on the CFAA,
  and the same plaintiff then lost on contract and paid $500,000 and deleted its algorithms.

## 8. Scripts and where this sits

`scripts/social_feeds.py` - the whole access position as `ACCESS`, with BLOCKED and ABSENT
rows marked; `X_PRICES_USD`, `read_cost()` and `cost_table()`; `HIQ`; `make_panel()` with
three timestamps, `attention_weights()` / `effective_names()`, `index_lag_summary()`;
`survivorship()` and `survivorship_grid()`; `panel_selection()`; `timestamp_ab()`.
numpy + pandas, seed 20260910, under 2 s, no network, no file writes, **no scraper**.

- `../congressional-trading-disclosures/SKILL.md` - the same lag argument at 45 days, and
  the same in-sample selection trap applied to members rather than accounts.
- `../insider-form-4/SKILL.md` - the same argument where the timestamp is a regulated
  filing, so the lag is knowable rather than a property of your collector.
- `../institutional-13f/SKILL.md` - selection over managers, and what a partial view of a
  book does to a reconstruction.
- `../../../fin-core/skills/research-integrity-guards/SKILL.md` §1, §2 and §5 - survivorship,
  availability and trial count; this skill is all three at once in one dataset.
- `../../../fin-llm/skills/llm-finance-agents/SKILL.md` - if the plan is to hand posts to a
  model rather than to a sentiment score, the evidence on whether that survives costs and
  leakage lives there.
- `../../../fin-core/skills/backtest-validation/SKILL.md` - a Sharpe of 2.7 from a coin flip
  is the whole reason the deflated-Sharpe machinery exists.
- `../../../fin-core/skills/signal-construction/scripts/assert_causal.py` - the mechanical
  check that a feature at time t uses nothing after t.
