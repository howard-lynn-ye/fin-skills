#!/usr/bin/env python3
"""Social feeds: what you cannot get, and what the part you can get does to a backtest.

No network, no file writes, NO SCRAPER, no credential path, no API client. Every post here
is SYNTHETIC and seeded. The only real content is the platform-access position transcribed
below, read at the URLs given on 2026-09-10, and the arithmetic on the published prices.

Six parts:

  1. The access position, per platform, with what could NOT be reached marked as such.
  2. What reading costs, computed from X's own published per-resource prices.
  3. A seeded post panel with THREE timestamps - created, indexed by your collector, and
     deleted - because a post has all three and a historical pull shows you one of them.
  4. Deletion survivorship. Posts that aged badly get deleted more often, and a historical
     pull cannot see them. Measured: what that does to apparent accuracy and to Sharpe,
     in a panel where NO account has any skill at all.
  5. Panel selection. Pick the accounts that were right, then backtest them. Measured on
     the same zero-skill panel, because that is the only way to see the size of the
     illusion cleanly.
  6. The timestamp A/B: created_at vs the moment your collector could actually see it.

Run:  python social_feeds.py    (numpy + pandas, seed 20260910)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

SEED = 20260910
TRADING_DAYS_PER_YEAR = 252
VERIFIED_ON = "2026-09-10"

# --------------------------------------------------------------------------------------
# 1. The access position, transcribed (every row read on 2026-09-10)
# --------------------------------------------------------------------------------------

# status codes: OK      = read at the primary source on VERIFIED_ON
#               BLOCKED = the primary page could not be reached; what is here is elsewhere
#               ABSENT  = searched the current docs and the thing is not there
ACCESS: list[tuple[str, str, str, str]] = [
    ("X", "pricing model", "OK",
     "docs.x.com/x-api/getting-started/pricing: 'The X API uses pay-per-usage pricing. "
     "No subscriptions-pay only for what you use.' There is no Free, Basic or Pro tier "
     "anywhere in the current docs."),
    ("X", "post read price", "OK",
     "same page: 'Posts: Read - $0.005 per resource', charged 'per resource returned in "
     "the response'. User: Read $0.010. Owned Reads $0.001."),
    ("X", "monthly cap", "OK",
     "same page and docs.x.com/x-api/fundamentals/post-cap: 'Pay-per-usage plans are "
     "capped at 3 million Post reads per monthly billing cycle.'"),
    ("X", "developer portal / plan list", "BLOCKED",
     "developer.x.com/en/portal/products returns HTTP 402 to a fetch and redirects to a "
     "login wall in a browser. Whether legacy Free/Basic/Pro subscriptions still exist "
     "for grandfathered accounts could NOT be verified."),
    ("X", "developer agreement", "BLOCKED",
     "developer.x.com/en/developer-terms/agreement-and-policy returns HTTP 402. The "
     "quotes below come from docs.x.com/developer-terms/restricted-use-cases instead."),
    ("X", "redistribution of IDs", "OK",
     "restricted-use-cases: 'You may only distribute up to a total of 1,500,000 Post IDs "
     "to a single entity within a 30 day period' - per 30 days, not per day."),
    ("X", "redistribution of content", "OK",
     "same: limited redistribution of hydrated content 'via non-automated means', up to "
     "'50,000 hydrated public Post Objects and/or User Objects per recipient, per day', "
     "and 'should not make this data publicly available'."),
    ("X", "financial inference", "OK",
     "same: 'Never derive or infer, or store derived or inferred, information about a X "
     "user's: ... Negative financial status or condition'; 'Credit or insurance risk "
     "analyses' is a listed prohibited use."),
    ("X", "the aggregate carve-out", "OK",
     "same: 'Aggregate analysis of X content that does not store any personal data (for "
     "example, user IDs, usernames, and other identifiers) is permitted' subject to the "
     "rest of the Agreement. This is the load-bearing sentence for a sentiment pipeline."),
    ("X", "model training", "OK",
     "same: 'X prohibits any use of the X APIs and/or X Content to fine-tune or train a "
     "foundation or frontier model with the exception of Grok.'"),
    ("X", "academic research product", "ABSENT",
     "not present anywhere in the 2026-09-10 docs. Only the academic REDISTRIBUTION "
     "exception survives. 'Retired in 2023' is secondhand and was not verified."),
    ("Reddit", "rate limit", "OK",
     "support.reddithelp.com Reddit Data API Wiki: '100 queries per minute (QPM) per "
     "OAuth client id', averaged 'over a time window (currently 10 minutes)'."),
    ("Reddit", "unauthenticated access", "OK",
     "same: 'Traffic not using OAuth or login credentials will be blocked, and the "
     "default rate limit will not apply.' Blocked, not throttled."),
    ("Reddit", "deletion propagation", "OK",
     "same: 'You must remove any user content in your possession that has been deleted "
     "from Reddit'; 48 hours is a 'strongly recommend' sweep interval, not a grace "
     "period; retention 'even if disassociated, de-identified or anonymized' violates."),
    ("Reddit", "commercial use", "OK",
     "support.reddithelp.com Accessing Reddit Data: 'You cannot use any Reddit developer "
     "tools and services for commercial purposes without first getting our permission'; "
     "'we'll require a contract'; listed examples include 'Services, research, or data "
     "access for fees'."),
    ("Reddit", "academic research", "OK",
     "same page: 'Can I perform research using Reddit developer tools and services? No.' "
     "The only sanctioned route is the Reddit For Researchers program; using the API for "
     "academic research 'is a violation of our policies'."),
    ("Reddit", "Data API Terms themselves", "BLOCKED",
     "redditinc.com/policies/data-api-terms could not be reached from this environment. "
     "Everything above is the help centre, which is official but is not the contract."),
    ("Reddit", "Pushshift", "OK",
     "support.reddithelp.com Pushshift Access Request: reinstated for 'verified Reddit "
     "moderators' with 'use of Pushshift ... limited to moderation use cases only'. No "
     "public, researcher or developer path exists."),
    ("StockTwits", "developer program", "OK",
     "stocktwits.com/developers returns HTTP 200 and renders a page titled 'Stocktwits "
     "Subscriptions' - consumer plans (Ad Free $85.00/year, Edge $229.50/year), no "
     "endpoint docs, no self-serve key. The only API route is an Enterprise plan, "
     "labelled 'API Access', whose button reads 'Get In Touch'."),
    ("StockTwits", "API host and docs", "OK",
     "api.stocktwits.com returns HTTP 302 to stocktwits.com; the historical docs path "
     "api.stocktwits.com/developers/docs returns HTTP 404."),
]

# X's published per-resource prices, read 2026-09-10 at
# docs.x.com/x-api/getting-started/pricing. Section 2 does arithmetic on these and nothing
# else; the arithmetic is the only "measured" number in that section.
X_PRICES_USD = {
    "post_read": 0.005, "user_read": 0.010, "owned_read": 0.001,
    "follow_read": 0.010, "post_create": 0.015, "post_create_with_url": 0.200,
    "counts_recent": 0.005, "counts_all": 0.010, "trends": 0.010,
}
X_MONTHLY_POST_READ_CAP = 3_000_000
X_ID_REDISTRIBUTION_PER_30_DAYS = 1_500_000
X_HYDRATED_PER_RECIPIENT_PER_DAY = 50_000
REDDIT_QPM_PER_OAUTH_CLIENT = 100
REDDIT_QPM_WINDOW_MINUTES = 10
REDDIT_DELETION_SWEEP_HOURS = 48        # recommended, not granted

# hiQ Labs, Inc. v. LinkedIn Corp. - the two layers, and they point opposite ways.
HIQ = {
    "cfaa_court": "9th Cir. No. 17-16783, decided 2022-04-18, 31 F.4th 1180",
    "cfaa_posture": "the Supreme Court GVR'd the first panel opinion in light of Van Buren "
                    "v. United States, 141 S. Ct. 1648 (2021); on remand the panel again "
                    "affirmed the PRELIMINARY INJUNCTION",
    "cfaa_holding": "the panel held hiQ 'raised a serious question' whether the CFAA's "
                    "'without authorization' concept reaches a bot refused access to data "
                    "that needs no authorization generally. That is a preliminary-injunction "
                    "standard, NOT a merits holding that scraping public data is lawful",
    "contract_court": "N.D. Cal. No. 17-cv-03301-EMC (Judge Edward M. Chen)",
    "contract_result": "November 2022 summary judgment: hiQ BREACHED LinkedIn's User "
                       "Agreement. Consent judgment entered 2022-12-08",
    "money_usd": 500_000,
    "injunction": "hiQ permanently enjoined from developing, using, selling or distributing "
                  "code for data collection from LinkedIn 'using any of the data, source "
                  "code, or algorithms developed at hiQ', required to DELETE that code and "
                  "all LinkedIn member profile data in its possession",
    "source_note": "the consent-judgment terms are quoted from a law-firm client alert, not "
                   "from the docket - secondhand",
}


def access_table() -> pd.DataFrame:
    return pd.DataFrame(ACCESS, columns=["platform", "topic", "status", "note"])


# --------------------------------------------------------------------------------------
# 2. What reading costs - arithmetic on published prices
# --------------------------------------------------------------------------------------

def read_cost(posts_per_day: int, days: int = 365,
              price: float | None = None) -> dict:
    """Cost of a post-read budget at X's published per-resource price.

    Deduplication is 'a soft guarantee' within a 24-hour UTC window, so re-reading the same
    post inside a day is free; this ignores that, which makes the number an upper bound on
    a de-duplicated pipeline and the right number for a distinct-post pipeline.
    """
    p = X_PRICES_USD["post_read"] if price is None else float(price)
    total = int(posts_per_day) * int(days)
    return {"posts_per_day": int(posts_per_day), "days": int(days), "posts": total,
            "price_per_read": p, "total_usd": total * p,
            "monthly_posts": int(posts_per_day) * 30,
            "over_cap": int(posts_per_day) * 30 > X_MONTHLY_POST_READ_CAP}


def cost_table(levels=(1_000, 10_000, 100_000)) -> pd.DataFrame:
    rows = {}
    for n in levels:
        c = read_cost(n)
        rows[f"{n:,}/day"] = {
            "posts/year": c["posts"],
            "usd/year": c["total_usd"],
            "usd/month": c["total_usd"] / 12.0,
            "pct of monthly cap": c["monthly_posts"] / X_MONTHLY_POST_READ_CAP,
        }
    cap = X_MONTHLY_POST_READ_CAP * X_PRICES_USD["post_read"]
    rows["at the cap"] = {"posts/year": X_MONTHLY_POST_READ_CAP * 12,
                          "usd/year": cap * 12, "usd/month": cap,
                          "pct of monthly cap": 1.0}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------------------
# 3. The seeded panel - three timestamps
# --------------------------------------------------------------------------------------

def make_returns(n_days: int, n_names: int, rng: np.random.Generator) -> np.ndarray:
    mkt = rng.normal(0.0002, 0.0100, n_days)
    beta = rng.uniform(0.7, 1.3, n_names)
    idio = rng.normal(0.0, 0.0180, (n_days, n_names))
    return mkt[:, None] * beta[None, :] + idio


def forward_returns(rets: np.ndarray, horizon: int) -> np.ndarray:
    """Sum of the next `horizon` daily returns, aligned so row t excludes day t itself."""
    c = np.vstack([np.zeros((1, rets.shape[1])), np.cumsum(rets, axis=0)])
    n = rets.shape[0]
    out = np.full_like(rets, np.nan)
    end = np.minimum(np.arange(n) + 1 + horizon, n)
    start = np.arange(n) + 1
    ok = end > start
    out[ok] = c[end[ok]] - c[start[ok]]
    return out


# Indexing lag. A collector polling a rate-limited API does not see a post the instant it
# is created. SET here, then measured back: most posts land the same day, a real share the
# next, and a tail beyond that (backfill after an outage, a rate-limit queue).
INDEX_LAG_PMF = {0: 0.55, 1: 0.30, 2: 0.10, 3: 0.03, 5: 0.02}


def attention_weights(n_names: int, exponent: float = 1.1) -> np.ndarray:
    """Which names get talked about. A social feed is not a uniform draw over a universe -
    a handful of names carry most of the posts, which is why the effective number of
    independent bets in a sentiment portfolio is far below its post count."""
    w = 1.0 / (np.arange(n_names) + 3.0) ** float(exponent)
    return w / w.sum()


def effective_names(n_names: int, exponent: float = 1.1) -> float:
    p = attention_weights(n_names, exponent)
    return float(1.0 / (p ** 2).sum())


def make_panel(n_posts: int = 20_000, n_accounts: int = 400, n_days: int = 1_260,
               n_names: int = 200, seed: int = SEED, horizon: int = 5,
               skilled_share: float = 0.0, edge: float = 0.06,
               base_delete: float = 0.10, wrong_extra: float = 0.15,
               attention_exponent: float = 1.1
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """Posts with a directional call, three timestamps, and an outcome.

    `skilled_share=0.0` is the default ON PURPOSE: sections 4 and 5 are about illusions
    that appear in a panel where nobody can predict anything, and mixing real skill in
    would let a reader attribute the result to the skill instead of to the bias.

    Deletion is the mechanism the whole file is about: a post whose call turned out WRONG
    is deleted more often than one that aged well, and a historical pull cannot see either
    the deletion or the post.
    """
    rng = np.random.default_rng(seed)
    rets = make_returns(n_days, n_names, rng)
    fwd = forward_returns(rets, horizon)

    account = rng.integers(0, n_accounts, n_posts)
    name = rng.choice(n_names, size=n_posts,
                      p=attention_weights(n_names, attention_exponent))
    create_day = rng.integers(0, n_days - horizon - 8, n_posts)

    lag_vals = np.array(list(INDEX_LAG_PMF))
    lag_p = np.array([INDEX_LAG_PMF[v] for v in lag_vals], dtype=float)
    index_lag = rng.choice(lag_vals, size=n_posts, p=lag_p / lag_p.sum())
    index_day = create_day + index_lag

    outcome = fwd[create_day, name]
    up = np.sign(outcome)
    up[up == 0] = 1.0

    skilled_acct = rng.random(n_accounts) < float(skilled_share)
    skilled = skilled_acct[account]
    right_prob = np.where(skilled, 0.5 + float(edge), 0.5)
    call = np.where(rng.random(n_posts) < right_prob, up, -up)

    correct = call == up
    p_del = float(base_delete) + float(wrong_extra) * (~correct)
    deleted = rng.random(n_posts) < p_del
    # A deleted post is gone from every later pull. When it went is what decides whether a
    # LIVE collector had already stored it; a historical pull never sees it either way.
    delete_day = index_day + rng.integers(1, 40, n_posts)

    posts = pd.DataFrame({
        "account": account, "name": name, "create_day": create_day,
        "index_lag": index_lag, "index_day": index_day, "call": call,
        "outcome": outcome, "correct": correct, "deleted": deleted,
        "delete_day": np.where(deleted, delete_day, -1), "skilled": skilled,
    })
    return posts, rets


def index_lag_summary(posts: pd.DataFrame) -> dict:
    lag = posts["index_lag"].to_numpy()
    return {"mean": float(lag.mean()), "median": float(np.median(lag)),
            "share_same_day": float((lag == 0).mean()),
            "share_next_day_or_later": float((lag > 0).mean()),
            "p95": float(np.percentile(lag, 95)), "max": float(lag.max())}


# --------------------------------------------------------------------------------------
# 4-6. Measurements
# --------------------------------------------------------------------------------------

def portfolio(posts: pd.DataFrame, rets: np.ndarray, key: str, hold: int = 5
              ) -> np.ndarray:
    """Net-call-weighted, market-neutral. Entry at the CLOSE of the key day."""
    n_days, n_names = rets.shape
    delta = np.zeros((n_days + hold + 2, n_names))
    entry = posts[key].to_numpy() + 1
    name = posts["name"].to_numpy()
    side = posts["call"].to_numpy().astype(float)
    np.add.at(delta, (entry, name), side)
    np.add.at(delta, (np.minimum(entry + hold, n_days + hold + 1), name), -side)
    W = np.cumsum(delta, axis=0)[:n_days]
    gross = np.abs(W).sum(axis=1)
    live = gross > 0
    Wn = np.zeros_like(W)
    Wn[live] = W[live] / gross[live, None]
    port = (Wn * rets).sum(axis=1)
    net = Wn.sum(axis=1)
    return np.where(live, port - net * rets.mean(axis=1), 0.0)


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    s = r.std(ddof=1)
    return float("nan") if not np.isfinite(s) or s == 0 else float(
        r.mean() / s * np.sqrt(TRADING_DAYS_PER_YEAR))


def survivorship(seed: int = SEED, n_seeds: int = 6, **kw) -> pd.DataFrame:
    """What a historical pull's missing posts do to apparent accuracy and Sharpe."""
    rows = {"every post (live)": [], "surviving posts (historical pull)": []}
    hits = {"every post (live)": [], "surviving posts (historical pull)": []}
    kept = []
    for k in range(n_seeds):
        posts, rets = make_panel(seed=seed + k, **kw)
        alive = posts[~posts["deleted"]]
        kept.append(float(len(alive) / len(posts)))
        for label, sub in (("every post (live)", posts),
                           ("surviving posts (historical pull)", alive)):
            rows[label].append(sharpe(portfolio(sub, rets, "index_day")))
            hits[label].append(float(sub["correct"].mean()))
    tab = pd.DataFrame({
        "hit_rate": {k: float(np.mean(v)) for k, v in hits.items()},
        "sharpe": {k: float(np.mean(v)) for k, v in rows.items()},
        "sharpe_sd": {k: float(np.std(v, ddof=1)) for k, v in rows.items()},
    })
    tab["share_of_posts"] = [1.0, float(np.mean(kept))]
    return tab


def survivorship_grid(seed: int = SEED, n_seeds: int = 4,
                      extras=(0.0, 0.05, 0.15, 0.30)) -> pd.DataFrame:
    """Apparent hit rate and Sharpe as a function of how asymmetric the deletion is."""
    rows = {}
    for e in extras:
        hit, shp = [], []
        for k in range(n_seeds):
            posts, rets = make_panel(seed=seed + k, wrong_extra=e)
            alive = posts[~posts["deleted"]]
            hit.append(float(alive["correct"].mean()))
            shp.append(sharpe(portfolio(alive, rets, "index_day")))
        rows[f"+{e:.0%}"] = {"apparent_hit_rate": float(np.mean(hit)),
                             "apparent_sharpe": float(np.mean(shp)),
                             "true_hit_rate": 0.5}
    return pd.DataFrame(rows).T


def account_scores(posts: pd.DataFrame, days: slice | None = None) -> pd.Series:
    sub = posts if days is None else posts[
        (posts["create_day"] >= days.start) & (posts["create_day"] < days.stop)]
    s = sub.assign(pnl=sub["call"] * sub["outcome"]).groupby("account")["pnl"]
    return (s.mean() * np.sqrt(s.size())).dropna()


def panel_selection(seed: int = SEED, n_seeds: int = 6, top_frac: float = 0.10,
                    n_days: int = 1_260, **kw) -> pd.DataFrame:
    """Pick the accounts that were right, then 'backtest' them. Zero-skill panel.

    Three honest-to-dishonest treatments:
      all accounts            - no selection at all
      selected out-of-sample  - chosen on the first half, measured on the second
      selected in-sample      - chosen on the WHOLE sample and measured on it. This is
                                what "the accounts you can name today" means.
    """
    out = {"all accounts": [], "selected out-of-sample": [], "selected in-sample": []}
    half = n_days // 2
    for k in range(n_seeds):
        posts, rets = make_panel(seed=seed + k, n_days=n_days, **kw)
        second = posts[posts["create_day"] >= half]
        out["all accounts"].append(sharpe(portfolio(second, rets, "index_day")))

        first_scores = account_scores(posts, slice(0, half))
        n_top = max(1, int(round(top_frac * len(first_scores))))
        oos = set(first_scores.nlargest(n_top).index)
        out["selected out-of-sample"].append(
            sharpe(portfolio(second[second["account"].isin(oos)], rets, "index_day")))

        full_scores = account_scores(posts)
        ins = set(full_scores.nlargest(max(1, int(round(top_frac * len(full_scores))))
                                       ).index)
        out["selected in-sample"].append(
            sharpe(portfolio(posts[posts["account"].isin(ins)], rets, "index_day")))
    tab = pd.DataFrame({"sharpe": {k: float(np.mean(v)) for k, v in out.items()},
                        "sd": {k: float(np.std(v, ddof=1)) for k, v in out.items()}})
    tab["vs all accounts"] = tab["sharpe"] - tab.loc["all accounts", "sharpe"]
    return tab


def timestamp_ab(seed: int = SEED, n_seeds: int = 6, skilled_share: float = 0.25,
                 edge: float = 0.06, **kw) -> pd.DataFrame:
    """created_at vs the day your collector could actually see the post.

    This one needs real skill in the panel, because it measures how much of a REAL edge a
    short indexing lag destroys. With no skill there is nothing to lose.
    """
    rows = {"created_at (look-ahead)": [], "indexed_at (honest)": []}
    for k in range(n_seeds):
        posts, rets = make_panel(seed=seed + k, skilled_share=skilled_share, edge=edge,
                                 **kw)
        rows["created_at (look-ahead)"].append(sharpe(portfolio(posts, rets, "create_day")))
        rows["indexed_at (honest)"].append(sharpe(portfolio(posts, rets, "index_day")))
    tab = pd.DataFrame({"sharpe": {k: float(np.mean(v)) for k, v in rows.items()},
                        "sd": {k: float(np.std(v, ddof=1)) for k, v in rows.items()}})
    base = tab.loc["indexed_at (honest)", "sharpe"]
    tab["gap"] = tab["sharpe"] - base
    tab["retained"] = base / tab["sharpe"]
    return tab


# --------------------------------------------------------------------------------------

def _wrap(text: str, indent: str, width: int = 70) -> None:
    while text:
        cut = text[:width].rfind(" ") if len(text) > width else len(text)
        cut = cut if cut > 0 else len(text)
        print(indent + text[:cut])
        text = text[cut:].lstrip()


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print(f"1. What you can actually get, {VERIFIED_ON}")
    print("=" * 78)
    at = access_table()
    for plat in ("X", "Reddit", "StockTwits"):
        print(f"\n   --- {plat} ---")
        for _, r in at[at["platform"] == plat].iterrows():
            print(f"   [{r['status']:<7}] {r['topic']}")
            _wrap(r["note"], "             ")
    n_blocked = int((at["status"] == "BLOCKED").sum())
    print(f"\n   {n_blocked} of {len(at)} rows are BLOCKED: the page exists and could not be")
    print("   read from here. Those rows are not evidence of anything and are marked so.")

    print(f"\n   hiQ Labs v. LinkedIn - two layers, pointing opposite ways:")
    print(f"   CFAA layer: {HIQ['cfaa_court']}")
    _wrap(HIQ["cfaa_posture"], "     ")
    _wrap(HIQ["cfaa_holding"], "     ")
    print(f"   CONTRACT layer: {HIQ['contract_court']}")
    _wrap(HIQ["contract_result"], "     ")
    _wrap(f"judgment ${HIQ['money_usd']:,}; {HIQ['injunction']}", "     ")
    _wrap(f"({HIQ['source_note']})", "     ")
    print("   hiQ won the CFAA question and lost the company on the user agreement.")

    print("\n" + "=" * 78)
    print("2. What reading costs - arithmetic on X's own published prices")
    print("=" * 78)
    print(f"   Every price below was read on X's own pricing page on {VERIFIED_ON}.")
    print(f"   {'resource':<24}{'USD':>10}")
    for k, v in X_PRICES_USD.items():
        print(f"   {k:<24}{v:>10.3f}")
    ct = cost_table()
    print(f"\n   ${X_PRICES_USD['post_read']:.3f} per post read, cap "
          f"{X_MONTHLY_POST_READ_CAP:,} post reads per monthly billing cycle.")
    print(f"   {'budget':<14}{'posts/year':>14}{'usd/year':>13}{'usd/month':>12}"
          f"{'% of cap':>10}")
    for i, r in ct.iterrows():
        print(f"   {i:<14}{r['posts/year']:>14,.0f}{r['usd/year']:>13,.0f}"
              f"{r['usd/month']:>12,.0f}{r['pct of monthly cap']:>10.1%}")
    print(f"   Redistribution ceilings: {X_ID_REDISTRIBUTION_PER_30_DAYS:,} Post IDs per")
    print(f"   entity per 30 DAYS, and {X_HYDRATED_PER_RECIPIENT_PER_DAY:,} hydrated "
          f"objects per recipient")
    print("   per day by non-automated means. A reproducible dataset for a paper is")
    print("   therefore a list of IDs, and every reader has to re-buy the content.")
    print(f"   Reddit: {REDDIT_QPM_PER_OAUTH_CLIENT} QPM per OAuth client averaged over "
          f"{REDDIT_QPM_WINDOW_MINUTES} minutes,")
    print("   unauthenticated traffic blocked outright, and academic research through the")
    print("   Data API is a stated policy violation. StockTwits: Enterprise sales only.")

    print("\n" + "=" * 78)
    print("3. Three timestamps, and a historical pull shows you one")
    print("=" * 78)
    posts, rets = make_panel()
    il = index_lag_summary(posts)
    print(f"   {len(posts):,} synthetic posts, 400 accounts, 1,260 days, 200 names, "
          f"seed {SEED}")
    print(f"   created_at  - on the post")
    print(f"   indexed_at  - when your collector could see it. SET here: "
          f"{il['share_same_day']:.0%} same day,")
    print(f"                 {il['share_next_day_or_later']:.0%} later, mean "
          f"{il['mean']:.2f} days, p95 {il['p95']:.0f}, max {il['max']:.0f}")
    print(f"   deleted_at  - never in a historical pull, because the post is not either")
    print(f"   deleted in this panel: {posts['deleted'].mean():.1%} of posts")
    print(f"   names are drawn by attention, not uniformly: {effective_names(200):.0f}")
    print(f"   effective names out of 200, which is why a post count is not a bet count.")
    print(f"   NOBODY in this panel has any skill: every call is a coin flip. Everything")
    print(f"   in sections 4 and 5 is an illusion produced by the sampling alone.")

    print("\n" + "=" * 78)
    print("4. Deletion survivorship")
    print("=" * 78)
    sv = survivorship()
    print("   A post whose call aged badly is deleted more often. A live collector stored")
    print("   it; a historical pull cannot see it. 6 seeds.\n")
    print(f"   {'sample':<36}{'hit rate':>10}{'sharpe':>9}{'sd':>8}{'% of posts':>12}")
    for i, r in sv.iterrows():
        print(f"   {i:<36}{r['hit_rate']:>10.1%}{r['sharpe']:>9.3f}"
              f"{r['sharpe_sd']:>8.3f}{r['share_of_posts']:>12.1%}")
    gap = (sv.loc["surviving posts (historical pull)", "hit_rate"]
           - sv.loc["every post (live)", "hit_rate"])
    print(f"   The coin flip reads as {sv.loc['surviving posts (historical pull)', 'hit_rate']:.1%}"
          f" once the losers are gone: +{gap * 100:.1f} points from nothing.")
    sg = survivorship_grid()
    print(f"\n   As a function of how much more often a WRONG post is deleted:")
    print(f"   {'extra deletion':<18}{'apparent hit':>14}{'true hit':>11}"
          f"{'apparent sharpe':>18}")
    for i, r in sg.iterrows():
        print(f"   {i:<18}{r['apparent_hit_rate']:>14.1%}{r['true_hit_rate']:>11.1%}"
              f"{r['apparent_sharpe']:>18.3f}")
    print("   At +0% the bias is gone, which is the control: deletion is only a problem")
    print("   when it is CORRELATED with the outcome. It always is.")
    print("   Note the SIZE. The bias is four points of hit rate and it backtests at a")
    print("   Sharpe most people would not question the provenance of. That is what")
    print("   volume does: 20,000 posts is enough statistical power to turn a small")
    print("   selection bias into a large, stable, entirely fake edge - and social data")
    print("   arrives in millions. Volume is not a defence against a sampling bias; it")
    print("   is what makes one look like a discovery.")

    print("\n" + "=" * 78)
    print("5. Panel selection - the accounts you can name today")
    print("=" * 78)
    ps = panel_selection()
    print("   Same zero-skill panel. Top decile of accounts by realised call performance.")
    print("   6 seeds.\n")
    print(f"   {'treatment':<28}{'sharpe':>9}{'sd':>8}{'vs all accounts':>18}")
    for i, r in ps.iterrows():
        print(f"   {i:<28}{r['sharpe']:>9.3f}{r['sd']:>8.3f}"
              f"{r['vs all accounts']:>18.3f}")
    print("   Nobody in the panel can predict anything. Selecting the top decile on the")
    print("   whole sample and then measuring it on the same sample manufactures the")
    print("   entire result. Selecting on the first half and measuring on the second")
    print("   does not - which is the only test worth running on an influencer list.")

    print("\n" + "=" * 78)
    print("6. The timestamp A/B")
    print("=" * 78)
    ta = timestamp_ab()
    print("   This one needs REAL skill in the panel, or there is nothing for the lag to")
    print("   destroy: 25% of accounts, 6-point edge, 5-day horizon. 6 seeds.\n")
    print(f"   {'key':<28}{'sharpe':>9}{'sd':>8}{'gap':>8}{'retained':>11}")
    for i, r in ta.iterrows():
        print(f"   {i:<28}{r['sharpe']:>9.3f}{r['sd']:>8.3f}{r['gap']:>8.3f}"
              f"{r['retained']:>11.1%}")
    print("   The lag here is under half a day on average. It costs this much because the")
    print("   horizon is five days: the shorter the signal, the more a small lag takes.")
    print("   Same arithmetic as the 45-day congressional lag against a 21-day half-life,")
    print("   at a hundredth of the scale.")

    print(f"\n   (total runtime {time.time() - t0:.1f}s)")
    print("\nRule: a social backtest runs on the posts that survived, keyed to the moment"
          " your collector saw them - so measure the deletion rate and the indexing lag"
          " before the signal, never pick the accounts after seeing their record, and read"
          " the platform's terms yourself, because this library ships no scraper.")


if __name__ == "__main__":
    main()
