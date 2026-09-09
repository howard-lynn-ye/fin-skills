"""fin_skills.data.ratelimit - one test per SHAPE, with an injected clock.

Nothing here sleeps. `FakeClock` advances on sleep(), so a test can assert how long a
limiter would have waited without waiting.
"""
from __future__ import annotations

import pytest

from fin_skills.data.ratelimit import (PerAccount, PerHourDayMonth, PerIP,
                                       PerInstanceDelay, PerMinute, PerSecond, RateLimit,
                                       SHAPES, Unpublished)


class FakeClock:
    """Monotonic time that only moves when someone sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.waits: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.waits.append(seconds)
            self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


SHAPE_EXAMPLES = {
    PerSecond: lambda: PerSecond(10),                       # SEC EDGAR
    PerMinute: lambda: PerMinute(5),                        # Polygon/Massive free
    PerInstanceDelay: lambda: PerInstanceDelay(2000),       # ccxt
    PerAccount: lambda: PerAccount(200),                    # Alpaca Basic
    PerIP: lambda: PerIP(timeseries_per_s=100, batch_submit_per_min=20),   # Databento
    PerHourDayMonth: lambda: PerHourDayMonth(50, 1000),     # Tiingo Starter
    Unpublished: Unpublished,                               # yfinance, akshare, FRED
}


def test_every_registered_shape_satisfies_the_protocol():
    from fin_skills.data.ratelimit import WeightedDaily
    SHAPE_EXAMPLES[WeightedDaily] = lambda: WeightedDaily(20, {"bulk": 100.0})
    assert set(SHAPES) == set(SHAPE_EXAMPLES), "SHAPES and the examples must agree"
    for cls, build in SHAPE_EXAMPLES.items():
        obj = build()
        assert isinstance(obj, RateLimit), cls
        assert obj.describe() and repr(obj) == obj.describe()


# ------------------------------------------------------------------------- PerSecond
def test_per_second_spaces_requests(clock):
    lim = PerSecond(10, clock=clock)          # SEC's ceiling is 10/s, per USER
    for _ in range(5):
        lim.acquire()
    assert clock.waits == pytest.approx([0.1] * 4), "one gap of 1/n between calls"
    assert lim.describe() == "PerSecond(10/s)"
    assert PerSecond.COOLDOWN_S == 600.0, "sec.gov: 10 minutes below the threshold"


def test_a_header_can_only_make_per_second_slower(clock):
    lim = PerSecond(10, clock=clock)
    lim.observe({"X-RateLimit-Limit": "2"})   # the server is enforcing 2/s today
    lim.acquire()
    lim.acquire()
    assert clock.waits == pytest.approx([0.5])


# --------------------------------------------------------------- PerInstanceDelay (ccxt)
def test_per_instance_delay_uses_the_venues_ms_and_endpoint_cost(clock):
    lim = PerInstanceDelay(2000, cost_fn=lambda ep: 5.0 if ep == "ohlcv" else 1.0,
                           clock=clock)
    lim.acquire()
    lim.acquire(endpoint="ohlcv")
    lim.acquire(endpoint="ticker")
    assert clock.waits == pytest.approx([2.0, 10.0])
    assert PerInstanceDelay.DEFAULT_MS == 2000.0
    assert "per exchange instance" in lim.describe()


# --------------------------------------------------------------------------- PerAccount
def test_per_account_is_a_rolling_minute(clock):
    lim = PerAccount(200, clock=clock)         # Alpaca Basic
    for _ in range(200):
        lim.acquire()
    assert clock.waits == []
    lim.acquire()
    assert clock.waits and clock.waits[0] == pytest.approx(60.0, abs=1e-6)


# --------------------------------------------------------------------------------- PerIP
def test_per_ip_charges_every_bucket_at_once(clock):
    lim = PerIP(timeseries_per_s=100, batch_submit_per_min=20, clock=clock)
    lim.acquire(bucket="batch_submit")
    assert lim.remaining("batch_submit") == 19
    assert lim.remaining("timeseries") == 100, "a named bucket charges only itself"
    lim.acquire()
    assert lim.remaining("timeseries") == 99 and lim.remaining("batch_submit") == 18
    with pytest.raises(KeyError):
        lim.acquire(bucket="nope")
    with pytest.raises(ValueError, match="must end in"):
        PerIP(timeseries=100)


# --------------------------------------------------------------------------- WeightedDaily
def test_weighted_daily_charges_100_for_a_bulk_call_and_1_for_a_symbol(clock):
    """EODHD: 20 CALLS a day, and a whole-exchange bulk request costs 100 of them."""
    from fin_skills.data.ratelimit import WeightedDaily
    costs = {"eod": 1.0, "technical": 5.0, "fundamental": 10.0, "bulk": 100.0}
    lim = WeightedDaily(20, costs, clock=clock)

    assert lim.cost_of("eod") == 1.0 and lim.cost_of("bulk") == 100.0
    lim.acquire(endpoint="eod")
    assert lim.remaining("day") == 19
    lim.acquire(endpoint="fundamental")
    assert lim.remaining("day") == 9

    with pytest.raises(ValueError, match="can never succeed"):
        lim.acquire(endpoint="bulk")          # 100 against a 20-call budget
    assert clock.waits == [], "an impossible call must not wait a day first"


def test_weighted_daily_waits_out_the_day_when_the_budget_is_spent(clock):
    from fin_skills.data.ratelimit import WeightedDaily
    lim = WeightedDaily(20, {"eod": 1.0}, clock=clock)
    for _ in range(20):
        lim.acquire(endpoint="eod")
    assert clock.waits == []
    lim.acquire(endpoint="eod")
    assert clock.waits[0] == pytest.approx(86400.0, abs=1e-3)


# ------------------------------------------------------------------------ PerHourDayMonth
def test_per_hour_day_month_counts_symbols_not_only_requests(clock):
    lim = PerHourDayMonth(50, 1000, symbols_per_month=3, bytes_per_month=1e9, clock=clock)
    lim.acquire(symbols=("AAPL", "MSFT"))
    lim.acquire(symbols=("AAPL",))            # already counted: not a new symbol
    with pytest.raises(RuntimeError, match="quota is on SYMBOLS"):
        lim.acquire(symbols=("A", "B"))
    lim.record_bytes(2048)
    assert lim.bytes_used == 2048
    assert "GB/mo" in lim.describe()


# ------------------------------------------------------------------------- Unpublished
def test_unpublished_waits_for_nothing_until_it_learns_something(clock):
    lim = Unpublished(clock=clock)
    for _ in range(10):
        lim.acquire()
    assert clock.waits == [], "no vendor number exists, so none is invented"
    assert "no vendor number" in lim.describe()


def test_a_caller_may_impose_a_courtesy_pace(clock):
    """FRED publishes 120/min on one page and 2/s on another; 2/s satisfies both. The
    number belongs to the caller - passing it is opting in, not reading a constant."""
    lim = Unpublished(courtesy_per_s=2.0, clock=clock)
    lim.acquire()
    lim.acquire()
    assert clock.waits == pytest.approx([0.5])
    assert "courtesy pace 2/s" in lim.describe()
    with pytest.raises(ValueError):
        Unpublished(courtesy_per_s=0)


def test_headers_beat_a_courtesy_pace(clock):
    lim = Unpublished(courtesy_per_s=10.0, clock=clock)
    lim.observe({"x-ratelimit-limit": "2", "x-ratelimit-remaining": "1",
                 "x-ratelimit-reset": "60"})
    assert lim.observed["limit"] == 2 and lim.observed["reset_in"] == 60
    lim.acquire()
    lim.acquire()
    assert clock.waits == pytest.approx([30.0]), "60s window / 2 requests, not 1/10s"


# ---------------------------------------------------------------------------- backoff
def test_a_429_backs_off_instead_of_returning_an_empty_frame(clock):
    lim = Unpublished(clock=clock)
    waited = [lim.backoff(a, jitter=False) for a in (1, 2, 3, 4)]
    assert waited == [1.0, 2.0, 4.0, 8.0]
    assert lim.n_429 == 4
    assert clock.waits == waited


def test_retry_after_wins_over_the_exponential_guess(clock):
    lim = PerSecond(10, clock=clock)
    lim.observe({"Retry-After": "42"})
    assert lim.backoff(1) == 42.0
    assert lim.backoff(1, jitter=False) == 1.0, "the header applies once, then is spent"


def test_backoff_is_capped(clock):
    lim = Unpublished(clock=clock)
    assert lim.backoff(20, jitter=False) == 60.0


def test_garbage_headers_are_ignored(clock):
    lim = PerSecond(4, clock=clock)
    lim.observe({"X-RateLimit-Limit": "not a number"})
    lim.observe({})
    assert lim.observed["limit"] is None
