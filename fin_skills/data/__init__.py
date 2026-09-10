"""fin_skills.data - one way in to market data, and a declaration attached to all of it.

    from fin_skills.data import adapters, get, to_bundle, Cache
    from fin_skills.api import check

    print(adapters())                       # the table to read BEFORE choosing a source
    bars = get("yfinance").bars("AAPL", "2020-01-01", "2024-01-01")
    print(check(to_bundle(bars)).summary())  # survivorship and adjustment, with no backtest

Three things hold everywhere in this package:

  * **No data ships in it.** Adapters ship code. There is no CSV in the wheel, no bundled
    parquet, no mirror this project controls, and nothing here ever reads from one. The
    cache writes to the user's own disk.
  * **No vendor library is imported until it is used.** `import fin_skills.data` works with
    none of yfinance, akshare, ccxt, fredapi, edgartools, tiingo, alpha-vantage or requests
    installed; numpy and pandas are the only hard dependencies. Every vendor import lives
    inside the method that needs it.
  * **No credential can be shared.** A key is read from the environment variable its
    Declaration names, at the moment it is used, and is never stored, logged, printed or
    written into a Provenance. FRED's own terms say why: "All users of an application shall
    use their own API key." The code path that would ship or proxy one does not exist.

The declarations carry what a normalised vendor schema hides - the adjustment convention,
whether delisted names are included, whether the source can answer "what was known at t?",
what the terms permit - because those are the four axes that decide whether a result is
real. `python -m fin_skills.data adapters` prints the table; `python -m fin_skills.data
manifest` prints what a cache holds.
"""
from __future__ import annotations

from fin_skills.data import adapters as _adapters_pkg
from fin_skills.data.cache import Cache, CachePolicyError, Divergence
from fin_skills.data.convert import (FILLS, fills, guard_convention, pit_used, to_bundle,
                                     to_long)
from fin_skills.data.declare import (Declaration, adapters, credential, declarations,
                                     describe, lookup, register)
from fin_skills.data.provenance import Provenance, content_hash
from fin_skills.data.ratelimit import (PerAccount, PerDay, PerHourDayMonth, PerIP,
                                       PerInstanceDelay, PerMinute, PerSecond, RateLimit,
                                       Unpublished, WeightedDaily)
from fin_skills.data.schema import (Adjustment, Bars, Fundamentals, Macro, stack_fields)
from fin_skills.data.validate import (validate_bars, validate_fundamentals,
                                      validate_macro)

# Importing the eight adapter modules registers their Declarations so `adapters()` can
# print the whole table. None of them imports a vendor library at module scope - that is
# the property `tests/test_data_adapters.py` blocks the vendor names on sys.meta_path to
# prove.
from fin_skills.data.adapters import akshare as _akshare              # noqa: F401,E402
from fin_skills.data.adapters import alphavantage as _alphavantage    # noqa: F401,E402
from fin_skills.data.adapters import ccxt as _ccxt                    # noqa: F401,E402
from fin_skills.data.adapters import edgar as _edgar                  # noqa: F401,E402
from fin_skills.data.adapters import fredapi as _fredapi              # noqa: F401,E402
from fin_skills.data.adapters import stooq as _stooq                  # noqa: F401,E402
from fin_skills.data.adapters import tiingo as _tiingo                # noqa: F401,E402
from fin_skills.data.adapters import yfinance as _yfinance            # noqa: F401,E402

# ...and these two come last. `advise` reads the registry the adapters just filled and
# answers "which of them can serve what I need, and what if none of them can"; `sample`
# is the seeded SYNTHETIC world every offline example runs on - no vendor data ships in
# this package, and none is downloaded to build it.
from fin_skills.data.advise import Need, Recommendation, recommend    # noqa: E402
from fin_skills.data.sample import Sample, sample                     # noqa: E402

get = _adapters_pkg.get

__version__ = "0.1.0"

__all__ = [
    "Adjustment", "Bars", "Cache", "CachePolicyError", "Declaration", "Divergence",
    "FILLS", "Fundamentals", "Macro", "Need", "PerAccount", "PerDay", "PerHourDayMonth",
    "PerIP", "PerInstanceDelay", "PerMinute", "PerSecond", "Provenance", "RateLimit",
    "Recommendation", "Sample", "Unpublished", "WeightedDaily", "adapters",
    "content_hash", "credential", "declarations", "describe", "fills", "get",
    "guard_convention", "lookup", "pit_used", "recommend", "register", "sample",
    "stack_fields", "to_bundle", "to_long", "validate_bars", "validate_fundamentals",
    "validate_macro",
]
