"""Which adapter can actually serve this request - and what to do when none of them can.

    from fin_skills.data.advise import Need, recommend
    print(recommend(Need(asset_class="equity", market="US", delisted=True)).report())

The function this module automates is the one an agent otherwise guesses at: it reads the
same `Declaration` table `adapters()` prints, applies the user's stated constraints as HARD
filters, and either returns a ranked list or returns nothing and says why.

Returning nothing is a first-class answer here, not a failure mode. The single most common
request in this domain - *delisted US equity daily bars on a free tier* - has no answer at
all among free sources, and a recommender that quietly hands back the least-bad option
converts a licensing-and-coverage dead end into a survivorship-biased backtest that looks
fine. So when the filters empty the list, `Recommendation.empty_reason` says which flag
did it, `partial` names the adapters that serve PART of the request, and `paid` names the
vendors that do serve it, each with the page and date the claim was verified at.

Three flags decide almost every case, and they are the three a normalised vendor schema
hides:

    includes_delisted   whether a result is an estimate or an UPPER BOUND
    point_in_time       whether the source can answer "what was known at t?"
    redistribution      whether you may pass on what you fetched

Everything else - price, coverage breadth, convenience - is a tiebreak between options
that have already survived those three.

WHERE THE NUMBERS COME FROM. Every per-adapter fact this module ranks on is read from the
adapter's own `Declaration`, which carries the date it was verified. The only knowledge
that lives HERE is `COVERAGE` (which asset classes and markets an adapter reaches, and how
deep its history goes) and `PAID`, and each row of both carries its own source string. A
test asserts every registered adapter has a `COVERAGE` row, so a new adapter cannot be
silently invisible to the advisor.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

#: report width. The output is read in a terminal, so it wraps rather than trailing off.
WIDTH = 94


def _wrap(text: str, first: str, rest: str) -> list[str]:
    return textwrap.wrap(str(text), width=WIDTH, initial_indent=first,
                         subsequent_indent=rest) or [first.rstrip()]

from fin_skills.data import declare
from fin_skills.data.schema import Adjustment

#: what a caller may ask for. Deliberately coarse: the axes below decide the answer, and a
#: finer taxonomy would invite a match that the licence or the coverage does not support.
ASSET_CLASSES = ("equity", "etf", "index", "fx", "futures", "crypto", "macro",
                 "fundamentals")
MARKETS = ("US", "CN", "HK", "JP", "UK", "PL", "HU", "GLOBAL", "CRYPTO")
METHODS = ("bars", "fundamentals", "macro", "universe", "actions")


# --------------------------------------------------------------------------- the request
@dataclass(frozen=True)
class Need:
    """What the caller needs, in the vocabulary the declarations can be filtered on.

    `delisted`, `point_in_time`, `redistribute` and `store` are the four that turn into
    hard refusals, because each maps onto a flag a vendor's terms or coverage decides and
    no amount of code can work around.
    """

    asset_class: str = "equity"
    market: str = "US"
    frequency: str = "1d"
    history_years: float = 1.0
    method: str = "bars"          # what you actually want to call
    delisted: bool = False        # delisted names must be present
    point_in_time: bool = False   # must answer "what was known at t?"
    redistribute: bool = False    # the result will be passed on to someone else
    store: bool = False           # the result will be written to disk
    key_ok: bool = True           # willing to register for an API key
    paid_ok: bool = False         # willing to pay a vendor

    def __post_init__(self) -> None:
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"asset_class must be one of {list(ASSET_CLASSES)}, got "
                             f"{self.asset_class!r}")
        if str(self.market).upper() not in MARKETS:
            raise ValueError(f"market must be one of {list(MARKETS)}, got {self.market!r}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}, got {self.method!r}")
        object.__setattr__(self, "market", str(self.market).upper())
        if float(self.history_years) < 0:
            raise ValueError("history_years cannot be negative")

    def describe(self) -> str:
        want = [f"{self.asset_class} / {self.market}", f"{self.method}()",
                f"interval {self.frequency}", f"{self.history_years:g}y of history"]
        for flag, label in ((self.delisted, "delisted names REQUIRED"),
                            (self.point_in_time, "point-in-time REQUIRED"),
                            (self.redistribute, "will be REDISTRIBUTED"),
                            (self.store, "will be STORED on disk"),
                            (not self.key_ok, "no API key"),
                            (self.paid_ok, "paid vendors acceptable")):
            if flag:
                want.append(label)
        return ", ".join(want)


# ------------------------------------------------------------------------- the coverage
@dataclass(frozen=True)
class Coverage:
    """What an adapter reaches, beyond what its `Declaration` already states.

    `max_history_years` is None where the vendor publishes no number - which is not the
    same as "unlimited", and is why this field never causes a refusal when it is None.
    """

    adapter: str
    asset_classes: tuple[str, ...]
    markets: tuple[str, ...]
    serves: tuple[str, ...]
    max_history_years: float | None
    source: str
    #: set when the endpoint was checked and did not answer a script. An adapter whose
    #: transport is dead is REFUSED rather than ranked, because "free and keyless" is not
    #: a recommendation if the request comes back as an interstitial.
    unreachable: str = ""

    def __post_init__(self) -> None:
        bad = [m for m in self.serves if m not in METHODS]
        if bad:
            raise ValueError(f"{self.adapter}: serves {bad} which are not methods")

    @property
    def reachable(self) -> bool:
        return not self.unreachable


#: One row per registered adapter. Each `source` is where the row was checked, and when.
COVERAGE: dict[str, Coverage] = {
    "yfinance": Coverage(
        "yfinance", ("equity", "etf", "index", "fx", "futures"), ("US", "GLOBAL"),
        ("bars", "actions"), None,
        "Yahoo publishes no coverage or history figure; Declaration dated 2026-09-09. "
        "Yahoo does quote crypto pairs, but this adapter declares calendar=XNYS and "
        "tz=America/New_York, so it cannot hand back a 24/7 series without lying about "
        "the calendar - ccxt is the crypto path"),
    "akshare": Coverage(
        "akshare", ("equity", "index", "macro"), ("CN",),
        ("bars",), None,
        "akshare publishes no coverage or history figure; Declaration dated 2026-09-09"),
    "ccxt": Coverage(
        "ccxt", ("crypto",), ("CRYPTO",),
        ("bars",), None,
        "history is the VENUE's, not ccxt's, and differs per exchange; Declaration "
        "dated 2026-09-09"),
    "fred": Coverage(
        "fred", ("macro",), ("US", "GLOBAL"),
        ("macro",), None,
        "history is per series; ALFRED vintages start when the series was first "
        "published; Declaration dated 2026-09-09"),
    "edgar": Coverage(
        "edgar", ("fundamentals",), ("US",),
        ("fundamentals",), None,
        "EDGAR full-text coverage starts in the 1990s and varies by form; "
        "Declaration dated 2026-09-09"),
    "tiingo": Coverage(
        "tiingo", ("equity", "etf"), ("US", "CN"),
        ("bars", "actions"), 60.0,
        "'History: 60+ Years; Data going back from 1962' - "
        "tiingo.com/products/end-of-day-stock-price-data, verified 2026-09-10"),
    "alphavantage": Coverage(
        "alphavantage", ("equity", "etf"), ("US", "GLOBAL"),
        ("bars", "universe"), 0.4,
        "the FREE key gets outputsize=compact, 'only the latest 100 data points' "
        "(~0.4y); outputsize=full is premium - alphavantage.co/documentation, "
        "verified 2026-09-10"),
    "stooq": Coverage(
        "stooq", ("equity", "etf", "index", "fx", "futures", "macro"),
        ("US", "PL", "UK", "JP", "HK", "HU", "GLOBAL"),
        ("bars",), 40.0,
        "AAPL.US has daily bars in 1984 on stooq.com/q/d/, and /db/h/ ships world, us, "
        "uk, jp, hk, pl, hu and macro archives - verified 2026-09-10",
        unreachable=(
            "on 2026-09-10 every stooq.com and stooq.pl URL - the CSV endpoint, the quote "
            "page and /db/h/ - answered HTTP 200 with an identical 796-byte JavaScript "
            "proof-of-work page, 'This site requires JavaScript to verify your browser', "
            "instead of the data. A 200 with an HTML body is the worst shape there is: "
            "pd.read_csv(url) parses it rather than raising. There is also no client "
            "library left - pandas-datareader 0.11.x dropped the stooq reader and "
            "DataReader(..., 'stooq') raises NotImplementedError")),
}


# ------------------------------------------------------------- paid vendors, when needed
@dataclass(frozen=True)
class PaidOption:
    """A vendor that does serve what no free adapter can, and where that was checked."""

    vendor: str
    unlocks: tuple[str, ...]      # which Need flags it satisfies
    what: str
    price: str
    source: str
    verified: bool = True

    def lines(self, indent: str = "    ") -> list[str]:
        mark = "" if self.verified else "  (SECONDHAND - not verified at a vendor page)"
        out = _wrap(f"{self.vendor}: {self.what}", indent, indent + "  ")
        out += _wrap(self.price, indent + "  price : ", indent + "          ")
        out += _wrap(self.source + mark, indent + "  source: ", indent + "          ")
        return out


PAID: tuple[PaidOption, ...] = (
    PaidOption(
        "EODHD", ("delisted",),
        "'Delisted Data' first appears on the ALL-IN-ONE package; the $19.99 EOD "
        "Historical Data plan does not carry it, and the free tier is 20 calls/day",
        "$99.99/month (ALL-IN-ONE)",
        "eodhd.com/pricing, verified 2026-09-10"),
    PaidOption(
        "Norgate Data", ("delisted", "point_in_time"),
        "US Stocks Platinum (history to 1990) and Diamond (to 1950) list 'Delisted "
        "securities' and 'Historical index constituents' and are the only tiers the page "
        "itself labels 'Suitable for backtesting'; Silver and Gold are 'Current "
        "major-exchange-listed securities' only",
        "priced through a currency/term calculator, so no figure is quoted here",
        "norgatedata.com/prices.php, verified 2026-09-10"),
    PaidOption(
        "CRSP (via WRDS)", ("delisted", "point_in_time"),
        "the academic reference for survivorship-free US equity history with delisting "
        "returns and historical index membership",
        "institutional subscription; no public price list",
        "named by this repository's market-data-sourcing skill; NOT verified at crsp.org",
        verified=False),
    PaidOption(
        "Tiingo redistribution plan", ("redistribute",),
        "the only route to redistribution rights on Tiingo data; every other Tiingo tier, "
        "including $30 Power and $50 Commercial, is licensed 'Internal Use Only'",
        "$250/month for startups, $500/month for enterprise",
        "tiingo.com/products/end-of-day-stock-price-data, verified 2026-09-10"),
    PaidOption(
        "Alpha Vantage premium", ("history",),
        "unlocks outputsize=full (25+ years) and TIME_SERIES_DAILY_ADJUSTED, both of "
        "which the free key cannot reach at any request rate",
        "from $49.99/month (75 requests/min)",
        "alphavantage.co/premium, verified 2026-09-10"),
)


# ---------------------------------------------------------------------------- the answer
@dataclass(frozen=True)
class Refusal:
    adapter: str
    reason: str
    flag: str                     # which axis of the Need did it

    def __str__(self) -> str:
        return f"{self.adapter:<14} {self.reason}"


@dataclass(frozen=True)
class Option:
    """One adapter that CAN serve the request, with its price and its limits."""

    adapter: str
    decl: declare.Declaration
    score: tuple
    cost: tuple[str, ...]
    cannot: tuple[str, ...]
    licence: tuple[str, ...]

    def report(self, indent: str = "  ") -> str:
        pad = indent + "     "
        out = [f"{indent}{self.adapter}  ({self.decl.library}, "
               f"{self.decl.library_license}, verified {self.decl.verified_on})"]
        for label, items in (("costs you", self.cost), ("CANNOT", self.cannot),
                             ("licence", self.licence)):
            for c in items:
                out += _wrap(c, f"{pad}{label:<9} : ", pad + "            ")
        return "\n".join(out)


@dataclass(frozen=True)
class Recommendation:
    need: Need
    ranked: tuple[Option, ...]
    refused: tuple[Refusal, ...]
    partial: tuple[tuple[str, str], ...] = ()
    empty_reason: str = ""
    paid: tuple[PaidOption, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.ranked

    @property
    def best(self) -> Option | None:
        return self.ranked[0] if self.ranked else None

    def names(self) -> list[str]:
        return [o.adapter for o in self.ranked]

    def to_frame(self) -> pd.DataFrame:
        """The ranked options as a table. Empty - with columns - when nothing serves it."""
        rows = [{"rank": i + 1, "adapter": o.adapter,
                 "key": o.decl.key_env_var if o.decl.requires_key else "-",
                 "rate_limit": o.decl.rate_limit.describe(),
                 "delisted": o.decl.includes_delisted, "PIT": o.decl.point_in_time,
                 "redistribution": o.decl.redistribution,
                 "cache": o.decl.cache_policy,
                 "rewrites_history": o.decl.rewrites_history}
                for i, o in enumerate(self.ranked)]
        return pd.DataFrame(rows, columns=["rank", "adapter", "key", "rate_limit",
                                           "delisted", "PIT", "redistribution", "cache",
                                           "rewrites_history"])

    def report(self) -> str:
        lines = _wrap(self.need.describe(), "NEED: ", "      ") + ["=" * WIDTH]
        if self.ranked:
            lines.append(f"{len(self.ranked)} adapter(s) can serve this, best first:")
            lines.append("")
            for i, o in enumerate(self.ranked, start=1):
                body = o.report().splitlines()
                lines.append(f"  {i}. {body[0].strip()}")
                lines += body[1:]
                lines.append("")
        else:
            lines.append("NO ADAPTER IN THIS LIBRARY CAN SERVE THIS REQUEST.")
            lines.append("")
            lines += _wrap(self.empty_reason, "  ", "  ")
            lines.append("")
            if self.partial:
                lines.append("  What IS available, and how far it gets you:")
                for name, what in self.partial:
                    lines += _wrap(f"{name}: {what}", "    ", "      ")
                lines.append("")
        if self.paid:
            lines.append("  Vendors that do serve it, and where that was checked:")
            for p in self.paid:
                lines += p.lines()
            lines.append("")
        if self.refused:
            lines.append(f"Refused ({len(self.refused)}), and why:")
            for r in self.refused:
                lines += _wrap(r.reason, f"  {r.adapter:<14} ", " " * 17)
        return "\n".join(lines).encode("ascii", "replace").decode("ascii")

    def __str__(self) -> str:
        return self.report()


# --------------------------------------------------------------------------- the filters
def _cost_lines(d: declare.Declaration) -> tuple[str, ...]:
    out = [f"${d.key_env_var}, which you register for yourself ({d.key_sharing})"
           if d.requires_key else "no key and no account"]
    out.append(d.rate_limit.describe())
    out.append(d.free_tier)
    return tuple(out)


def _cannot_lines(d: declare.Declaration, cov: Coverage) -> tuple[str, ...]:
    out: list[str] = []
    if not d.includes_delisted:
        out.append("includes_delisted=False - survivor-only, so every result from it is "
                   "an UPPER BOUND, not an estimate")
    if not d.point_in_time:
        out.append("point_in_time=False - it cannot answer 'what was known at t?'")
    if d.rewrites_history:
        out.append(f"adjustment_default={d.adjustment_default.value} rewrites history - "
                   f"the same query next month returns different numbers")
    if cov.max_history_years is not None:
        out.append(f"reaches about {cov.max_history_years:g} year(s) of history "
                   f"({cov.source})")
    else:
        out.append(f"publishes no history depth ({cov.source})")
    return tuple(out)


def _licence_lines(d: declare.Declaration, need: Need) -> tuple[str, ...]:
    out: list[str] = []
    if need.redistribute:
        out.append(f"you said the result will be REDISTRIBUTED and this source declares "
                   f"redistribution={d.redistribution}")
    else:
        out.append(f"redistribution={d.redistribution} - the code licence "
                   f"({d.library_license}) is not the data licence")
    if need.store:
        out.append(f"you said it will be STORED and cache_policy={d.cache_policy}")
    if d.non_display_use == "prohibited":
        out.append("non_display_use=prohibited - the terms forbid non-display use, which "
                   "arguably describes a backtest")
    if d.attribution:
        out.append(f"attribution required, verbatim: {d.attribution}")
    out.append(f"terms: {d.terms_url}")
    return tuple(out)


def _refuse(need: Need, d: declare.Declaration, cov: Coverage) -> Refusal | None:
    """The hard filters, in the order a human would apply them. First failure wins."""
    if need.method not in cov.serves:
        return Refusal(d.name, f"does not serve {need.method}() - it serves "
                               f"{list(cov.serves)}", "method")
    if need.asset_class not in cov.asset_classes:
        return Refusal(d.name, f"no {need.asset_class} coverage - it serves "
                               f"{list(cov.asset_classes)}", "asset_class")
    if need.market not in cov.markets:
        return Refusal(d.name, f"no {need.market} coverage - it serves "
                               f"{list(cov.markets)}", "market")
    if d.interval_support and need.frequency not in d.interval_support:
        return Refusal(d.name, f"interval {need.frequency!r} is not in "
                               f"{list(d.interval_support)}", "frequency")
    if need.delisted and not d.includes_delisted:
        return Refusal(d.name, "includes_delisted=False - it returns today's names, so a "
                               "universe built from it has no dead ones in it",
                       "delisted")
    if need.point_in_time and not d.point_in_time:
        return Refusal(d.name, "point_in_time=False - it serves the latest values, not "
                               "what was on the wire at t", "point_in_time")
    if need.redistribute and d.redistribution == "prohibited":
        return Refusal(d.name, f"redistribution=prohibited ({d.terms_url}) and you said "
                               f"the result will be passed on", "redistribute")
    if need.store and d.cache_policy == "no-persist":
        return Refusal(d.name, "cache_policy=no-persist - its terms forbid retaining the "
                               "data in any persistent or durable storage", "store")
    if not need.key_ok and d.requires_key:
        return Refusal(d.name, f"needs ${d.key_env_var} and you said no key", "key_ok")
    # checked AFTER the caller's own flags, so a request for delisted names is answered
    # with "includes_delisted=False" by every source that lacks them - a complete story -
    # rather than losing one of them to a transport problem that is not the point
    if not cov.reachable:
        return Refusal(d.name, f"VERIFIED UNREACHABLE: {cov.unreachable}", "unreachable")
    if (cov.max_history_years is not None
            and float(need.history_years) > cov.max_history_years):
        return Refusal(d.name, f"reaches about {cov.max_history_years:g}y and you asked "
                               f"for {float(need.history_years):g}y ({cov.source})",
                       "history_years")
    return None


def _score(need: Need, d: declare.Declaration, cov: Coverage) -> tuple:
    """Ranking among options that already passed every hard filter. Lower sorts first.

    The order is stated rather than tuned: correctness axes before convenience axes, and
    convenience axes before breadth. Nothing here can promote an adapter past a filter.
    """
    return (
        0 if d.point_in_time else 1,                    # can it answer "known at t?"
        0 if d.includes_delisted else 1,                # is it survivorship-free
        0 if not d.rewrites_history else 1,             # is a cached copy still valid
        0 if d.adjustment_default in (Adjustment.RAW_PLUS_FACTORS,) else 1,
        0 if not d.requires_key else 1,                 # can you run it right now
        0 if d.cache_policy != "no-persist" else 1,     # may you keep what you fetched
        {"public-domain": 0, "attribution": 1, "prohibited": 2}[d.redistribution],
        -(cov.max_history_years or 0.0),                # deeper history first
        d.name,                                         # stable, so the order is testable
    )


def _partial(need: Need, refused: Sequence[Refusal]) -> tuple[tuple[str, str], ...]:
    """What an adapter that cannot serve the WHOLE request still gets you.

    A dead end is more useful when it names the half that is available, and every entry
    here is a capability the refused adapter genuinely has - not a consolation prize.
    """
    out: list[tuple[str, str]] = []
    by_flag = {r.flag: r for r in refused}
    names = {r.adapter for r in refused}
    if need.delisted and "alphavantage" in names and need.market == "US":
        out.append((
            "alphavantage",
            "its LISTING_STATUS endpoint IS free and IS point-in-time: 'a list of active "
            "or delisted US stocks and ETFs ... as of the latest trading day or at a "
            "specific time in history', any date after 2010-01-01 "
            "(alphavantage.co/documentation, verified 2026-09-10). "
            "get('alphavantage').universe('US', '2014-07-10', include_delisted=True) "
            "fixes the MEMBERSHIP half of survivorship. It does not price the dead names, "
            "which is the half a backtest needs."))
    if need.delisted and "edgar" in names:
        out.append((
            "edgar", "SEC filings are survivorship-free - Lehman, Bear Stearns and Enron "
            "are all still on file - so a fundamentals study can be made honest even when "
            "the price panel cannot."))
    if need.point_in_time and "fred" in names and need.asset_class == "macro":
        out.append(("fred", "ALFRED vintages make macro point-in-time; the same is not "
                            "available for prices at any price in this library."))
    if by_flag.get("history_years") and by_flag["history_years"].adapter == "tiingo":
        out.append(("tiingo", "60+ years of US history back to 1962, if the free tier's "
                              "500-symbols-a-month ceiling fits your universe."))
    return tuple(out)


def _paid_for(need: Need) -> tuple[PaidOption, ...]:
    wants = {flag for flag, on in (("delisted", need.delisted),
                                   ("point_in_time", need.point_in_time),
                                   ("redistribute", need.redistribute),
                                   ("history", need.history_years > 5)) if on}
    if not wants:
        return ()
    return tuple(p for p in PAID if wants & set(p.unlocks))


def _empty_reason(need: Need, refused: Sequence[Refusal]) -> str:
    """Name the flag that emptied the list, not just the fact that it is empty."""
    counts: dict[str, int] = {}
    for r in refused:
        counts[r.flag] = counts.get(r.flag, 0) + 1
    decisive = [f for f in ("delisted", "point_in_time", "redistribute", "store",
                            "history_years", "key_ok", "unreachable", "frequency",
                            "market", "asset_class", "method") if f in counts]
    if not decisive:
        return "no adapter is registered."
    flag = decisive[0]
    n = counts[flag]
    headline = {
        "delisted": (f"{n} adapter(s) were refused on includes_delisted=False. There is "
                     f"NO free source of delisted US equity price history - not in this "
                     f"library and not in the free tier of any vendor surveyed for it. "
                     f"The correct next step is to buy it or to state, in the result, "
                     f"that every number is an upper bound."),
        "point_in_time": (f"{n} adapter(s) were refused on point_in_time=False. Only "
                          f"vintage-carrying sources can answer 'what was known at t?', "
                          f"and in this library that is FRED for macro and EDGAR for "
                          f"filings - never a price source."),
        "redistribute": (f"{n} adapter(s) were refused on redistribution. The code "
                         f"licence of a client is not the licence of the data it "
                         f"fetches, and every free price source here declares "
                         f"redistribution=prohibited."),
        "store": (f"{n} adapter(s) were refused on cache_policy=no-persist - their terms "
                  f"forbid retaining the data at all."),
        "history_years": (f"{n} adapter(s) were refused on history depth: none of them "
                          f"reaches {float(need.history_years):g} years for this request."),
        "key_ok": (f"{n} adapter(s) need a key you said you would not register for."),
        "unreachable": (f"{n} adapter(s) that would otherwise serve this were checked and "
                        f"did not answer a script. Free and keyless is not a "
                        f"recommendation if the request comes back as an interstitial."),
        "frequency": f"{n} adapter(s) do not serve interval {need.frequency!r}.",
        "market": f"{n} adapter(s) have no {need.market} coverage.",
        "asset_class": f"{n} adapter(s) have no {need.asset_class} coverage.",
        "method": f"{n} adapter(s) do not serve {need.method}().",
    }[flag]
    return headline


# ------------------------------------------------------------------------------- the API
def recommend(need: Need | None = None, **kw: Any) -> Recommendation:
    """Rank the adapters that can serve `need`, or return none and say why.

    `recommend(delisted=True)` is shorthand for `recommend(Need(delisted=True))`.
    """
    if need is None:
        need = Need(**kw)
    elif kw:
        raise TypeError("pass a Need or keyword arguments, not both")

    options: list[Option] = []
    refusals: list[Refusal] = []
    for d in declare.declarations():
        cov = COVERAGE.get(d.name)
        if cov is None:
            raise KeyError(
                f"adapter {d.name!r} is registered but has no advise.COVERAGE row, so the "
                f"advisor cannot say what it reaches. Add one - with the page and date it "
                f"was verified at - in the same commit that registers the adapter.")
        bad = _refuse(need, d, cov)
        if bad is not None:
            refusals.append(bad)
            continue
        options.append(Option(adapter=d.name, decl=d, score=_score(need, d, cov),
                              cost=_cost_lines(d), cannot=_cannot_lines(d, cov),
                              licence=_licence_lines(d, need)))
    options.sort(key=lambda o: o.score)
    ranked = tuple(options)
    return Recommendation(
        need=need, ranked=ranked, refused=tuple(refusals),
        partial=() if ranked else _partial(need, refusals),
        empty_reason="" if ranked else _empty_reason(need, refusals),
        # paid vendors are named whenever nothing free serves the request, and also when
        # the caller has said money is available - in which case a free option that works
        # and a paid one that works better are both worth seeing
        paid=_paid_for(need) if (not ranked or need.paid_ok) else ())


def coverage_frame() -> pd.DataFrame:
    """One row per adapter: what it reaches, and where that was checked."""
    return pd.DataFrame([
        {"adapter": c.adapter, "asset_classes": ",".join(c.asset_classes),
         "markets": ",".join(c.markets), "serves": ",".join(c.serves),
         "max_history_years": ("unpublished" if c.max_history_years is None
                               else f"{c.max_history_years:g}"),
         "reachable": c.reachable, "source": c.source}
        for c in sorted(COVERAGE.values(), key=lambda c: c.adapter)])


__all__ = ["ASSET_CLASSES", "COVERAGE", "Coverage", "MARKETS", "METHODS", "Need",
           "Option", "PAID", "PaidOption", "Recommendation", "Refusal", "coverage_frame",
           "recommend"]
