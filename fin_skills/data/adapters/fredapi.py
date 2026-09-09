"""FRED / ALFRED through fredapi - vintages, and a key that can never be shared.

Verified against the vendor's own pages on 2026-09-09:

  * **Every user brings their own key.** FRED's API-key page: "All users of an application
    shall use their own API key", and "Developers should request a distinct API key for
    each application they build." So this library ships none, embeds none and proxies
    none. The key is read from `$FRED_API_KEY` at the moment of the call, through
    `declare.credential()`, and there is no argument anywhere in this package through
    which a literal key could travel.
  * **Attribution is required, verbatim.** The Terms of Use require an application to
    display: "This product uses the FRED(R) API but is not endorsed or certified by the
    Federal Reserve Bank of St. Louis." It is carried on the Declaration and stamped into
    every Provenance this adapter produces.
  * **Third-party series may not be redistributed.** "Before using data series owned by
    third parties for anything other than your own personal use, you must contact the data
    owner to obtain permission." The flag is machine-detectable: those series carry the
    word "Copyright" in their notes, so `copyright_flagged()` checks and the adapter marks
    the Provenance non-redistributable when it fires.
  * **The published rate limit exists in two incompatible shapes.** The v1 errors page:
    "Up to 120 requests per minute are allowed before being served a 429 error code." The
    v2 errors page: "Up to 2 requests per second are allowed before being served a 429
    error code." Same magnitude, different granularity - 120/min permits a 120-request
    burst inside one second and 2/s does not - so a caller who wants a courtesy pace
    passes `courtesy_per_s=2.0`, which satisfies both. `rate_limit` is `Unpublished()` and
    this module hardcodes no number, because a rate constant invented by the library would
    be indistinguishable from the folklore it exists to replace.

And the bug this adapter routes around: `get_series_as_of_date()` is documented to return
"a Series where each index is the observation date" and actually returns
`get_series_all_releases(id)[realtime_start <= as_of]` - one row per REVISION, so treating
it as a series double-counts every revised observation. This adapter never calls it. It
uses `get_series_all_releases` and applies the deduplication itself, inside `Macro.as_of`.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from fin_skills.data import declare
from fin_skills.data.adapters import Base, library_version, require
from fin_skills.data.provenance import make as make_provenance
from fin_skills.data.ratelimit import Unpublished
from fin_skills.data.schema import Macro

TERMS = "https://fred.stlouisfed.org/docs/api/terms_of_use.html"

#: the exact string FRED's Terms of Use require an application to display
ATTRIBUTION = ("This product uses the FRED(R) API but is not endorsed or certified by "
               "the Federal Reserve Bank of St. Louis.")

DECL = declare.Declaration(
    name="fred",
    library="fredapi",
    library_license="Apache-2.0",
    licence_source="repo-LICENSE",     # PyPI declares NO license, no expression and no
                                       # classifier - metadata tooling sees nothing
    adjustment_default=None,           # macro series have no adjustment convention
    adjustment_supported=(),
    calendar="observed",
    tz="UTC",
    bar_label="n/a",
    interval_support=(),
    includes_delisted=False,           # discontinued series stay published, but this
                                       # adapter serves series, not a universe
    point_in_time=True,                # realtime_start is what makes it PIT
    rate_limit=Unpublished(),
    free_tier=("free, key required. Two published shapes of the same magnitude, verified "
               "2026-09-09: 'Up to 120 requests per minute' (v1 errors page) and 'Up to 2 "
               "requests per second' (v2 errors page). Pacing at 2/s satisfies both - pass "
               "Unpublished(courtesy_per_s=2.0) if you want the library to enforce it"),
    requires_key=True,
    key_env_var="FRED_API_KEY",
    key_sharing="byok-required",       # "All users of an application shall use their own
                                       # API key" - a shared key cannot ship
    cache_policy="persist-ok",
    non_display_use="unstated",
    terms_url=TERMS,
    redistribution="attribution",      # attribution required; third-party series excluded
    attribution=ATTRIBUTION,
    verified_on="2026-09-09",
    notes=("fredapi 0.5.2 was released 2024-05-05 and has had no code change since; PyPI "
           "declares no licence at all, so the Apache-2.0 above is resolved from the "
           "repository LICENSE.\n"
           "get_series_as_of_date() returns one row per revision, not one per date. This "
           "adapter never calls it: it uses get_series_all_releases and Macro.as_of() "
           "applies the groupby the docstring omits.\n"
           "Series whose notes contain the word 'Copyright' are third-party owned and may "
           "not be redistributed; copyright_flagged() detects them and the Provenance is "
           "marked non-redistributable."),
)


def copyright_flagged(notes: str | None) -> bool:
    """True when a series' notes mark it as third-party copyrighted.

    FRED does not expose a boolean for this; the word is in the prose. Pure and offline so
    it can be tested against a fixture rather than against the live API.
    """
    return bool(notes) and "copyright" in str(notes).lower()


def releases_to_frame(releases: pd.DataFrame, series_id: str, *,
                      sa_flag: str | None = None) -> pd.DataFrame:
    """`get_series_all_releases` output -> the `Macro` schema.

    Its columns are `date`, `realtime_start`, `value`; `realtime_end` is silently dropped
    upstream (the parse line is commented out in 0.5.2), so it is filled with NaT here
    rather than invented.
    """
    df = pd.DataFrame(releases).copy()
    for want in ("date", "realtime_start", "value"):
        if want not in df.columns:
            raise ValueError(f"expected a get_series_all_releases frame with a {want!r} "
                             f"column, got {list(df.columns)}")
    out = pd.DataFrame({
        "series_id": str(series_id),
        "obs_date": pd.to_datetime(df["date"]),
        "value": pd.to_numeric(df["value"], errors="coerce"),
        "realtime_start": pd.to_datetime(df["realtime_start"]),
        "realtime_end": pd.to_datetime(df["realtime_end"]) if "realtime_end" in df.columns
        else pd.NaT,
        "sa_flag": sa_flag,
    })
    return out.sort_values(["obs_date", "realtime_start"]).reset_index(drop=True)


class FredAdapter(Base):
    decl = DECL

    def _client(self) -> Any:
        fredapi = require("fredapi")
        # read at the moment of use; never stored on self, never logged, never serialised
        return fredapi.Fred(api_key=declare.credential(self.decl))

    def macro(self, series_ids, *, vintages: bool = True, **kw) -> Macro:
        ids = [series_ids] if isinstance(series_ids, str) else list(series_ids)
        if not vintages:
            raise ValueError(
                "vintages=False would return the fully-revised line, which is DISPLAY "
                "ONLY: FRED's own example is 2013Q4 GDP at 17102.5, then 17080.7, then "
                "17089.6. Fetch the vintages and call Macro.latest() if you want it.")
        fredapi = require("fredapi")
        client = self._client()
        parts, flagged = [], []
        for sid in ids:
            self.limiter.acquire()
            info = _series_info(client, str(sid))
            if copyright_flagged(info.get("notes")):
                flagged.append(str(sid))
            self.limiter.acquire()
            releases = client.get_series_all_releases(str(sid), **kw)
            parts.append(releases_to_frame(releases, str(sid),
                                           sa_flag=info.get("seasonal_adjustment_short")))
        frame = pd.concat(parts, ignore_index=True)

        request = {"method": "macro", "series_ids": [str(x) for x in ids],
                   "vintages": True, "endpoint": "get_series_all_releases",
                   "copyright_flagged": flagged}
        prov = make_provenance("fred", library_version=library_version(fredapi),
                               request=request, content=frame,
                               vendor_vintage=str(frame["realtime_start"].max().date()),
                               terms_url=TERMS, redistributable=not flagged,
                               attribution=ATTRIBUTION)
        return Macro(frame=frame, provenance=prov)


def _series_info(client: Any, series_id: str) -> dict:
    """The series metadata, tolerantly - `get_series_info` returns a Series of fields."""
    try:
        info = client.get_series_info(series_id)
    except Exception:                                       # noqa: BLE001 - vendor errors
        return {}
    return dict(info) if hasattr(info, "keys") else {}


declare.register(FredAdapter, DECL)

__all__ = ["ATTRIBUTION", "DECL", "FredAdapter", "copyright_flagged",
           "releases_to_frame"]
