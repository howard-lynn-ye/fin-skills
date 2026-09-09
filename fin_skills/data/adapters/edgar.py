"""SEC EDGAR through edgartools - the best-behaved client in the survey.

The SEC's own Internet Security Policy, verified 2026-09-09: "Current guidelines limit
users to a total of no more than 10 requests per second, regardless of the number of
machines used to submit requests", and "Once the rate of requests has dropped below the
threshold for 10 minutes, the user may resume accessing content on SEC.gov." Two
consequences the code has to respect rather than note: the ceiling is per USER, so
parallelism buys nothing and a cluster is exactly as limited as a laptop; and the penalty
is a ten-minute wall, which is far more expensive than the request it saved.

edgartools is the only client in the survey that is conservative by default -
`edgar/httprequests.py` sets `max_requests_per_second = 8`, verified in the source on
2026-09-09 - so this adapter keeps that number rather than raising it to the ceiling. The
one thing edgartools will not do for you is identify you: every request 403s until
`set_identity()` has been called, and the SEC checks the header for presence, not content,
so sending a fake browser User-Agent is a policy violation rather than a workaround. The
identity is caller-supplied and this adapter raises if it is missing.

An identity is not a credential. It is a public declaration of who is making the request,
which is why it is an ordinary argument here while an API key could never be.

Point-in-time is the reason EDGAR is in this layer at all: every filing is as-filed, so a
2018 10-K is what a 2018 investor actually saw. `available_at` comes from
`acceptanceDateTime`, not from `filed`, and anything accepted at or after the 16:00
exchange close is pushed to the next session - a filing accepted at 16:05 was not tradeable
that day. Note that the SEC's own two APIs disagree about the timezone of that stamp: the
Submissions API's `acceptanceDateTime` is UTC while the Financial Statement Data Sets'
`sub.txt.accepted` is Eastern, so the source zone is declared, never assumed.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import pandas as pd

from fin_skills.core.pit_fundamentals import available_at
from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import PerSecond
from fin_skills.data.schema import Fundamentals

TERMS = "https://www.sec.gov/about/privacy-information#security"

#: edgartools' own self-throttle, read from edgar/httprequests.py on 2026-09-09. It is
#: BELOW the SEC's 10/s ceiling, and this adapter keeps it there.
EDGARTOOLS_SELF_THROTTLE = 8

#: the environment variable edgartools itself reads for the required User-Agent
IDENTITY_ENV = "EDGAR_IDENTITY"

DECL = declare.Declaration(
    name="edgar",
    library="edgartools",
    library_license="MIT",
    licence_source="pypi",            # license_expression 'MIT' + classifier, 2026-09-09
    adjustment_default=None,          # filings have no price-adjustment convention
    adjustment_supported=(),
    calendar="XNYS",
    tz="America/New_York",
    bar_label="n/a",
    interval_support=(),
    includes_delisted=True,           # the filings of Lehman, Bear Stearns and Enron are
                                      # all intact: EDGAR itself is survivorship-free
    point_in_time=True,
    rate_limit=PerSecond(EDGARTOOLS_SELF_THROTTLE),
    free_tier=("free, no key; SEC fair access is 10 requests/second regardless of the "
               "number of machines, with a 10-minute cooldown after a breach "
               "(sec.gov Internet Security Policy, verified 2026-09-09)"),
    requires_key=False,               # an identity is a declaration, not a secret
    key_env_var="",
    key_sharing="unstated",
    cache_policy="persist-ok",
    non_display_use="permitted",
    terms_url=TERMS,
    redistribution="public-domain",   # US government works
    verified_on="2026-09-09",
    notes=(f"A User-Agent must be declared or every request 403s. Pass identity=..., or "
           f"set ${IDENTITY_ENV}; the SEC checks the header for presence, not content, "
           f"and a fake browser UA is a policy violation, not a workaround.\n"
           f"edgartools self-throttles at {EDGARTOOLS_SELF_THROTTLE}/s, below the SEC's "
           f"10/s ceiling; this adapter keeps that margin.\n"
           "available_at is derived from acceptanceDateTime, not from filed, and a "
           "filing accepted at or after the 16:00 close moves to the next session. The "
           "Submissions API reports that stamp in UTC while the Financial Statement Data "
           "Sets report it in Eastern - reading one as the other shifts every filing four "
           "or five hours early.\n"
           "The XBRL frames API is NOT point-in-time: it returns today's value after "
           "restatements, and cannot be made point-in-time. Iterate filings instead."),
)


def facts_to_frame(units: Sequence[dict], *, entity_id: str, tag: str,
                   entity_scheme: str = "cik", unit: str = "USD",
                   acceptance_tz: str = "UTC",
                   sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """A companyfacts `units` list -> the `Fundamentals` frame.

    Pure and offline. `available_at` is computed from `acceptanceDateTime` where the SEC
    supplies one and from `filed` otherwise - and where it falls back, the row's
    `acceptance_at` is NaT, which `validate_fundamentals` reports as a warning rather than
    hiding.
    """
    rows = []
    for f in units:
        acc_dt = f.get("acceptanceDateTime") or f.get("acceptance_at")
        filed = pd.Timestamp(f["filed"])
        if acc_dt:
            av = available_at(acc_dt, exchange_tz="America/New_York", sessions=sessions,
                              tz_in=acceptance_tz)
            acceptance = av.acceptance_utc
            avail = av.first_tradeable_session
        else:
            acceptance = pd.NaT
            avail = filed
        rows.append({
            "entity_id": str(entity_id), "entity_scheme": entity_scheme,
            "period_start": pd.to_datetime(f.get("start"), errors="coerce"),
            "period_end": pd.to_datetime(f["end"]),
            "filed_at": filed,
            "acceptance_at": acceptance,
            "available_at": pd.Timestamp(avail),
            "form": str(f.get("form", "")), "accn": str(f.get("accn", "")),
            "tag": str(tag), "value": pd.to_numeric(f.get("val"), errors="coerce"),
            "unit": unit,
            "is_amendment": str(f.get("form", "")).endswith("/A"),
            "fy": f.get("fy"), "fp": f.get("fp"),
        })
    return pd.DataFrame(rows)


class EdgarAdapter(Base):
    decl = DECL

    def __init__(self, identity: str | None = None, **client_kw: Any) -> None:
        super().__init__(**client_kw)
        self.identity = identity or os.environ.get(IDENTITY_ENV) or ""
        if not self.identity:
            raise ValueError(
                "SEC EDGAR requires a declared User-Agent and returns 403 without one. "
                f"Pass identity='Your Name your@email.com', or set ${IDENTITY_ENV}. The "
                "SEC checks the header for presence rather than content, so a fake "
                "browser User-Agent is a policy violation, not a workaround.")
        self._ready = False

    def _client(self):
        edgar = require("edgar", pip_name="edgartools")
        if not self._ready:
            edgar.set_identity(self.identity)
            self._ready = True
        return edgar

    def fundamentals(self, entities, tags=(), *, unit: str = "USD",
                     acceptance_tz: str = "UTC", sessions=None,
                     **kw) -> Fundamentals:
        edgar = self._client()
        ents = [entities] if isinstance(entities, str) else list(entities)
        wanted = tuple(tags) or ("Revenues",)
        idx = pd.DatetimeIndex(pd.to_datetime(list(sessions))) if sessions is not None \
            else None
        parts = []
        for ent in ents:
            self.limiter.acquire()
            company = edgar.Company(str(ent))
            facts = company.get_facts()
            raw = facts.to_dict() if hasattr(facts, "to_dict") else facts
            units = raw.get("facts", {}).get("us-gaap", {}) if isinstance(raw, dict) else {}
            for tag in wanted:
                unit_list = (units.get(tag, {}).get("units", {}).get(unit, [])
                             if isinstance(units, dict) else [])
                if not unit_list:
                    continue
                parts.append(facts_to_frame(unit_list, entity_id=str(ent), tag=str(tag),
                                            unit=unit, acceptance_tz=acceptance_tz,
                                            sessions=idx))
        if not parts:
            raise ValueError(f"no {list(wanted)} facts in {unit} for {ents}; EDGAR "
                             f"returned nothing rather than this adapter dropping rows")
        frame = pd.concat(parts, ignore_index=True)

        request = {"method": "fundamentals", "entities": [str(e) for e in ents],
                   "tags": list(wanted), "unit": unit,
                   "acceptance_tz": acceptance_tz,
                   "available_at_from": "acceptanceDateTime",
                   "post_close_rolls_to_next_session": True}
        prov = make_provenance("edgar", library_version=library_version(edgar),
                               request=request, content=frame,
                               vendor_vintage=str(frame["accn"].max()),
                               terms_url=TERMS, redistributable=True)
        return Fundamentals(frame=frame, provenance=prov, tz_of_record=acceptance_tz)

    def universe(self, market: str, as_of, *,
                 include_delisted: bool = False) -> pd.DataFrame:
        raise self._unsupported(
            "a ticker universe",
            "sec.gov/files/company_tickers.json is a CURRENT snapshot, so returning it as "
            "history would be survivorship bias with an SEC label on it. EDGAR's filings "
            "are bias-free - build the universe from filings in the window instead")


declare.register(EdgarAdapter, DECL)

__all__ = ["DECL", "EDGARTOOLS_SELF_THROTTLE", "EdgarAdapter", "IDENTITY_ENV",
           "facts_to_frame"]
