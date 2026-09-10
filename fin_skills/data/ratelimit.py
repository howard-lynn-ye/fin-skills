"""Rate limits differ in KIND, not just in magnitude, so one `requests_per_minute` knob
models none of them.

    PerInstanceDelay(ms)      ccxt - rateLimit ms x endpoint cost, leaky bucket, and the
                              limiter is PER EXCHANGE INSTANCE, so a fresh ccxt.binance()
                              per call resets it and is how people get banned
    PerIP(**buckets)          databento - 100 timeseries/s AND 20 batch-submit/MIN, per IP
    PerAccount(rpm)           alpaca - 200/min Basic, 10,000/min Algo Trader Plus
    WeightedDaily(budget, {}) eodhd - 20 CALLS/day where a whole-exchange bulk request
                              costs 100 calls; a naive request counter is 5-100x wrong
    PerHourDayMonth(...)      tiingo - 50/hr, 1,000/day, 500 symbols/mo, 1 GB/mo
    PerDay(n)                 alphavantage - 25 requests a DAY on the free key, and the
                              paid plans are quoted per minute with "No daily limits", so
                              the two tiers are not even the same shape
    PerSecond(n)              SEC EDGAR - "no more than 10 requests per second, regardless
                              of the number of machines used to submit requests"
    Unpublished()             yfinance, akshare, FRED - no usable published number. Paces
                              from X-RateLimit-* headers and backs off on 429; the library
                              never supplies a number of its own.

`Unpublished` is not a synonym for "unlimited". It is the honest shape for a source whose
number does not exist (yfinance: zero hits for 429/limiter/throttl in the whole doc index;
akshare: no published limit at all) or exists in two mutually inconsistent shapes. FRED is
the second case, verified 2026-09-09: the v1 errors page says "Up to 120 requests per
minute are allowed before being served a 429 error code" while the v2 errors page says "Up
to 2 requests per second are allowed before being served a 429 error code". Same magnitude,
two different shapes - 120/min permits a 120-request burst inside one second and 2/s does
not - so a caller who wants a courtesy pace passes `Unpublished(courtesy_per_s=2.0)`, which
satisfies both. The number is the caller's, not this module's: an adapter module that
hardcoded one would be indistinguishable from the folklore it exists to replace.

Every limiter takes an injectable clock, so the tests advance time instead of sleeping.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

__all__ = ["Clock", "PerAccount", "PerDay", "PerHourDayMonth", "PerIP",
           "PerInstanceDelay", "PerMinute", "PerSecond", "RateLimit", "SHAPES",
           "SystemClock", "Unpublished", "WeightedDaily"]


# ------------------------------------------------------------------------------- clock
class Clock(Protocol):
    """Monotonic time plus a sleep. Tests substitute a fake that advances on sleep()."""

    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real clock. `now()` is monotonic; wall time never enters a rate decision."""

    __slots__ = ()

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


# ------------------------------------------------------------------------- the protocol
@runtime_checkable
class RateLimit(Protocol):
    """What `declare.Declaration.rate_limit` must satisfy."""

    def acquire(self, cost: float = 1.0) -> None: ...
    def observe(self, headers: Mapping[str, str]) -> None: ...
    def describe(self) -> str: ...


class _Limiter:
    """Shared machinery: an injectable clock, header observation and 429 backoff.

    Header parsing is not decoration. Every published number in this module is a dated
    snapshot of a vendor's documentation; `X-RateLimit-*` is what the server is enforcing
    right now, so an observed limit always overrides a declared one.
    """

    #: set by subclasses that carry a documented number, so observe() knows what to update
    _observable = True

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()
        self._observed_limit: float | None = None      # requests per observed window
        self._observed_remaining: float | None = None
        self._observed_reset_in: float | None = None
        self._retry_after: float | None = None
        self._n_429 = 0

    # ------------------------------------------------------------------ observation
    def observe(self, headers: Mapping[str, str]) -> None:
        """Read `X-RateLimit-Limit/Remaining/Reset` and `Retry-After`. Header wins."""
        if not headers:
            return
        low = {str(k).lower(): v for k, v in headers.items()}
        self._observed_limit = _num(low.get("x-ratelimit-limit"), self._observed_limit)
        self._observed_remaining = _num(low.get("x-ratelimit-remaining"),
                                        self._observed_remaining)
        reset = _num(low.get("x-ratelimit-reset"))
        if reset is not None:
            # a small value is a delta in seconds; a large one is an epoch stamp
            self._observed_reset_in = reset if reset < 1e6 else max(reset - time.time(), 0.0)
        self._retry_after = _num(low.get("retry-after"), self._retry_after)

    @property
    def observed(self) -> dict[str, float | None]:
        return {"limit": self._observed_limit, "remaining": self._observed_remaining,
                "reset_in": self._observed_reset_in, "retry_after": self._retry_after}

    # ------------------------------------------------------------------- 429 backoff
    def backoff(self, attempt: int = 1, *, jitter: bool = True,
                base: float = 1.0, cap: float = 60.0) -> float:
        """Sleep after a 429 and return the seconds waited.

        A 429 that returns an empty frame instead of waiting is the failure mode this
        exists to prevent: the caller sees "no data" and writes it to disk.
        """
        self._n_429 += 1
        if self._retry_after is not None:                 # the server said how long
            wait = float(self._retry_after)
        else:
            wait = min(base * (2.0 ** max(attempt - 1, 0)), cap)
            if jitter:
                wait *= 0.5 + random.random() * 0.5
        self._clock.sleep(wait)
        self._retry_after = None
        return wait

    @property
    def n_429(self) -> int:
        return self._n_429

    def describe(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return self.describe()


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


class _Spaced(_Limiter):
    """A minimum interval between requests: the leaky bucket every per-second shape is."""

    def __init__(self, interval_s: float, *, clock: Clock | None = None) -> None:
        super().__init__(clock=clock)
        self._interval = float(interval_s)
        self._next_at: float | None = None

    def _effective_interval(self) -> float:
        return self._interval

    def acquire(self, cost: float = 1.0) -> None:
        interval = self._effective_interval() * max(float(cost), 0.0)
        now = self._clock.now()
        if self._next_at is not None and now < self._next_at:
            self._clock.sleep(self._next_at - now)
            now = self._clock.now()
        self._next_at = now + interval


class _Windowed(_Limiter):
    """A count inside a rolling window - the shape a per-minute or per-day quota is."""

    def __init__(self, buckets: Mapping[str, tuple[float, float]], *,
                 clock: Clock | None = None) -> None:
        super().__init__(clock=clock)
        #: {name: (allowance, window seconds)}
        self._buckets = {k: (float(n), float(w)) for k, (n, w) in buckets.items()}
        self._hits: dict[str, list[tuple[float, float]]] = {k: [] for k in self._buckets}

    def _prune(self, name: str, now: float) -> None:
        window = self._buckets[name][1]
        self._hits[name] = [(t, c) for t, c in self._hits[name] if t > now - window]

    def used(self, name: str) -> float:
        self._prune(name, self._clock.now())
        return sum(c for _, c in self._hits[name])

    def remaining(self, name: str) -> float:
        return self._buckets[name][0] - self.used(name)

    def _charge(self, names: list[str], cost: float) -> None:
        for name in names:
            allowance, _ = self._buckets[name]
            if cost > allowance:
                # refuse BEFORE waiting: a bulk call costing 100 against a 20-call budget
                # never succeeds, and sleeping out the day first only hides that
                raise ValueError(f"a single call costs {cost:g} against a {allowance:g} "
                                 f"{name} budget - it can never succeed")
        for name in names:
            allowance, window = self._buckets[name]
            now = self._clock.now()
            self._prune(name, now)
            used = sum(c for _, c in self._hits[name])
            while used + cost > allowance and self._hits[name]:
                oldest = min(t for t, _ in self._hits[name])
                self._clock.sleep(max(oldest + window - now, 0.0) + 1e-9)
                now = self._clock.now()
                self._prune(name, now)
                used = sum(c for _, c in self._hits[name])
            self._hits[name].append((self._clock.now(), cost))


# --------------------------------------------------------------------------- the shapes
class PerSecond(_Spaced):
    """A hard requests-per-second ceiling, spaced rather than bursted.

    SEC EDGAR, verified 2026-09-09 on sec.gov's Internet Security Policy: "Current
    guidelines limit users to a total of no more than 10 requests per second, regardless
    of the number of machines used to submit requests", and "Once the rate of requests has
    dropped below the threshold for 10 minutes, the user may resume accessing content on
    SEC.gov." The ceiling is per USER, so parallelism does not buy throughput, and the
    penalty is a ten-minute wall.
    """

    #: seconds a client stays blocked after tripping the SEC threshold (sec.gov, 2026-09-09)
    COOLDOWN_S = 600.0

    def __init__(self, n: float, *, clock: Clock | None = None) -> None:
        if n <= 0:
            raise ValueError("PerSecond needs a positive rate")
        super().__init__(1.0 / float(n), clock=clock)
        self.n = float(n)

    def _effective_interval(self) -> float:
        if self._observed_limit and self._observed_limit > 0:
            return max(self._interval, 1.0 / self._observed_limit)
        return self._interval

    def describe(self) -> str:
        return f"PerSecond({self.n:g}/s)"


class PerMinute(_Windowed):
    """A per-minute quota. Polygon/Massive's free tier is 5 calls/min (2026-09-09)."""

    def __init__(self, n: float, *, clock: Clock | None = None) -> None:
        super().__init__({"minute": (float(n), 60.0)}, clock=clock)
        self.n = float(n)

    def acquire(self, cost: float = 1.0) -> None:
        self._charge(["minute"], float(cost))

    def describe(self) -> str:
        return f"PerMinute({self.n:g}/min)"


class PerDay(_Windowed):
    """A flat daily allowance and nothing else - no per-second shape, no weighting.

    Alpha Vantage, verified 2026-09-10 on alphavantage.co/support: "We are pleased to
    provide free stock API service covering the majority of our datasets for 25 API
    requests per day". Its premium plans are quoted per MINUTE and say "No daily limits",
    so the free and paid tiers are not the same shape and a per-minute pace tells a free
    user nothing about when they will be cut off.

    Distinct from `WeightedDaily`, whose whole point is that one request can cost many
    calls; here every request costs exactly one, and saying so is the honest declaration.
    """

    def __init__(self, n: float, *, clock: Clock | None = None) -> None:
        if n <= 0:
            raise ValueError("PerDay needs a positive allowance")
        super().__init__({"day": (float(n), 86400.0)}, clock=clock)
        self.n = float(n)

    def acquire(self, cost: float = 1.0) -> None:
        self._charge(["day"], float(cost))

    def describe(self) -> str:
        return f"PerDay({self.n:g}/day)"


class PerInstanceDelay(_Spaced):
    """ccxt: `rateLimit` milliseconds times an endpoint's cost, in a leaky bucket that
    belongs to ONE exchange instance.

    The trap is structural, not numeric. `enableRateLimit` is True by default and the
    bucket lives on the instance, so `ccxt.binance().fetch_ohlcv(...)` in a loop
    constructs a fresh, empty bucket every iteration and sends at full speed. The adapter
    holds one instance per venue for its lifetime; this limiter mirrors that instance.
    """

    #: ccxt's own default when an exchange declares none (2026-09-09)
    DEFAULT_MS = 2000.0

    def __init__(self, ms: float = DEFAULT_MS, cost_fn: Callable[[str], float] | None = None,
                 *, clock: Clock | None = None) -> None:
        super().__init__(float(ms) / 1000.0, clock=clock)
        self.ms = float(ms)
        self.cost_fn = cost_fn

    def acquire(self, cost: float = 1.0, *, endpoint: str | None = None) -> None:
        if endpoint is not None and self.cost_fn is not None:
            cost = float(self.cost_fn(endpoint))
        super().acquire(cost)

    def describe(self) -> str:
        return f"PerInstanceDelay({self.ms:g}ms, per exchange instance)"


class PerAccount(_Windowed):
    """A per-account requests-per-minute quota. Alpaca: 200/min Basic, 10,000/min paid."""

    def __init__(self, rpm: float, *, clock: Clock | None = None) -> None:
        super().__init__({"minute": (float(rpm), 60.0)}, clock=clock)
        self.rpm = float(rpm)

    def acquire(self, cost: float = 1.0) -> None:
        self._charge(["minute"], float(cost))

    def describe(self) -> str:
        return f"PerAccount({self.rpm:g}/min per account)"


class PerIP(_Windowed):
    """Several simultaneous buckets, all scoped to the source IP rather than the account.

    Databento (2026-09-09): 100 timeseries requests/second, 20 metadata requests/second,
    20 batch-submit requests/MINUTE, 100 concurrent connections - per IP. Two colleagues
    behind one office NAT share every one of them.

    Buckets are named `<name>_per_s`, `_per_min`, `_per_hour` or `_per_day`:

        PerIP(timeseries_per_s=100, metadata_per_s=20, batch_submit_per_min=20)
    """

    _WINDOWS = {"per_s": 1.0, "per_sec": 1.0, "per_min": 60.0, "per_hour": 3600.0,
                "per_day": 86400.0}

    def __init__(self, *, clock: Clock | None = None, **buckets: float) -> None:
        parsed: dict[str, tuple[float, float]] = {}
        self.windows: dict[str, float] = {}
        for name, n in buckets.items():
            # longest suffix first, so "_per_sec" is not read as "_per_s" + "ec"
            for suffix in sorted(self._WINDOWS, key=len, reverse=True):
                if name.endswith("_" + suffix):
                    short = name[: -(len(suffix) + 1)]
                    parsed[short] = (float(n), self._WINDOWS[suffix])
                    self.windows[short] = self._WINDOWS[suffix]
                    break
            else:
                raise ValueError(f"bucket {name!r} must end in one of "
                                 f"{sorted('_' + s for s in self._WINDOWS)}")
        if not parsed:
            raise ValueError("PerIP needs at least one bucket")
        super().__init__(parsed, clock=clock)

    def acquire(self, cost: float = 1.0, *, bucket: str | None = None) -> None:
        names = [bucket] if bucket is not None else list(self._buckets)
        for n in names:
            if n not in self._buckets:
                raise KeyError(f"no bucket {n!r}; have {sorted(self._buckets)}")
        self._charge(names, float(cost))

    _UNIT = {1.0: "/s", 60.0: "/min", 3600.0: "/hr", 86400.0: "/day"}

    def describe(self) -> str:
        parts = ", ".join(f"{k} {n:g}{self._UNIT.get(w, '')}"
                          for k, (n, w) in sorted(self._buckets.items()))
        return f"PerIP({parts}) - shared by everyone behind the same address"


class WeightedDaily(_Windowed):
    """A daily budget spent in CALLS, where one request can cost many.

    EODHD, verified 2026-09-09: the free tier is 20 API calls a day, and the calls are
    weighted - technical/intraday/news cost 5, fundamental/options/bond cost 10, and a
    whole-exchange bulk request costs 100. A counter that counts requests mispredicts the
    quota by 5-100x, and the 20-call free tier cannot afford a single bulk call.
    """

    def __init__(self, day_budget: float, cost: Mapping[str, float] | Callable[[str], float]
                 | None = None, *, default_cost: float = 1.0,
                 clock: Clock | None = None) -> None:
        super().__init__({"day": (float(day_budget), 86400.0)}, clock=clock)
        self.day_budget = float(day_budget)
        self.cost = cost
        self.default_cost = float(default_cost)

    def cost_of(self, endpoint: str | None) -> float:
        if endpoint is None or self.cost is None:
            return self.default_cost
        if callable(self.cost):
            return float(self.cost(endpoint))
        return float(self.cost.get(endpoint, self.default_cost))

    def acquire(self, cost: float = 1.0, *, endpoint: str | None = None) -> None:
        charge = self.cost_of(endpoint) if endpoint is not None else float(cost)
        self._charge(["day"], charge)

    def describe(self) -> str:
        return f"WeightedDaily({self.day_budget:g} calls/day, weighted per endpoint)"


class PerHourDayMonth(_Windowed):
    """Nested quotas on different clocks, plus counters that are not requests at all.

    Tiingo's Starter plan (2026-09-09): 50 requests/hour, 1,000/day, 500 unique symbols a
    month and 1 GB of bandwidth a month. The monthly counters are the ones that bite,
    because nothing in a request tells you how close you are.
    """

    def __init__(self, per_hour: float, per_day: float, *,
                 symbols_per_month: float | None = None,
                 bytes_per_month: float | None = None, clock: Clock | None = None) -> None:
        buckets = {"hour": (float(per_hour), 3600.0), "day": (float(per_day), 86400.0)}
        super().__init__(buckets, clock=clock)
        self.per_hour, self.per_day = float(per_hour), float(per_day)
        self.symbols_per_month = symbols_per_month
        self.bytes_per_month = bytes_per_month
        self._symbols: set[str] = set()
        self._bytes = 0.0

    def acquire(self, cost: float = 1.0, *, symbols: tuple[str, ...] = ()) -> None:
        if self.symbols_per_month is not None and symbols:
            after = self._symbols | set(symbols)
            if len(after) > self.symbols_per_month:
                raise RuntimeError(
                    f"{len(after)} unique symbols this month exceeds the "
                    f"{self.symbols_per_month:g} the plan allows; the quota is on SYMBOLS, "
                    f"not on requests, so retrying does not help")
            self._symbols = after
        self._charge(["hour", "day"], float(cost))

    def record_bytes(self, n: float) -> None:
        self._bytes += float(n)

    @property
    def bytes_used(self) -> float:
        return self._bytes

    def describe(self) -> str:
        extra = []
        if self.symbols_per_month is not None:
            extra.append(f"{self.symbols_per_month:g} symbols/mo")
        if self.bytes_per_month is not None:
            extra.append(f"{self.bytes_per_month / 1e9:g} GB/mo")
        tail = (", " + ", ".join(extra)) if extra else ""
        return f"PerHourDayMonth({self.per_hour:g}/hr, {self.per_day:g}/day{tail})"


class Unpublished(_Limiter):
    """No usable published number exists, so this library invents none.

    Three sources in this layer are in that state: yfinance (no numeric limit anywhere -
    zero hits for 429, limiter or throttl in the whole documentation index), akshare (no
    published limit at all; its own docstrings only warn that heavy scraping gets your IP
    blocked), and FRED, whose two error pages give the same magnitude in two incompatible
    shapes - 120 requests per minute on the v1 page, 2 requests per second on v2, verified
    2026-09-09.

    Behaviour: no delay until something is learned, then pace from `X-RateLimit-*`, and on
    a 429 back off exponentially rather than returning an empty frame. A caller who wants
    a self-imposed floor passes it in - `Unpublished(courtesy_per_s=2.0)` satisfies both of
    FRED's published shapes - and that number belongs to the caller, not to this module.
    """

    def __init__(self, courtesy_per_s: float | None = None, *,
                 clock: Clock | None = None) -> None:
        super().__init__(clock=clock)
        if courtesy_per_s is not None and courtesy_per_s <= 0:
            raise ValueError("courtesy_per_s must be positive")
        self.courtesy_per_s = courtesy_per_s
        self._next_at: float | None = None

    def _effective_interval(self) -> float:
        """Observed headers beat a caller's courtesy pace; a courtesy pace beats nothing."""
        intervals = []
        if self.courtesy_per_s:
            intervals.append(1.0 / self.courtesy_per_s)
        if self._observed_limit and self._observed_limit > 0:
            window = self._observed_reset_in or 1.0
            intervals.append(window / self._observed_limit)
        return max(intervals) if intervals else 0.0

    def acquire(self, cost: float = 1.0) -> None:
        interval = self._effective_interval() * max(float(cost), 0.0)
        if interval <= 0:
            return
        now = self._clock.now()
        if self._next_at is not None and now < self._next_at:
            self._clock.sleep(self._next_at - now)
            now = self._clock.now()
        self._next_at = now + interval

    def describe(self) -> str:
        if self.courtesy_per_s:
            return (f"Unpublished(no vendor number; caller's courtesy pace "
                    f"{self.courtesy_per_s:g}/s, adaptive on 429)")
        return "Unpublished(no vendor number; adaptive backoff + X-RateLimit-* headers)"


#: every registered shape, for `python -m fin_skills.data adapters` and the tests
SHAPES: tuple[type, ...] = (PerInstanceDelay, PerIP, PerAccount, WeightedDaily,
                            PerHourDayMonth, PerSecond, PerMinute, PerDay, Unpublished)
