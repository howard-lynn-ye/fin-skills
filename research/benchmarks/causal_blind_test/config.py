"""Frozen configuration for the causal blind-test benchmark.

Every number here is fixed BEFORE any result is observed. Changing a value
invalidates the benchmark unless the change is recorded in the trial ledger.
"""
from __future__ import annotations

# ----------------------------------------------------------------- data window
# Universe is resolved as of UNIVERSE_ASOF -- the first evaluation day -- so the
# backtest never sees index additions that happened because a name did well.
UNIVERSE_ASOF = "2024-01-02"
EVAL_START = "2024-01-02"
EVAL_END = "2026-09-11"
# Warmup must cover the longest lookback (60d) plus slack for suspensions.
DATA_START = "2023-06-01"
WARMUP_DAYS = 60

# Number of names drawn from the point-in-time HS300 list, in code order.
# Code order is arbitrary with respect to future return, which is the point:
# it is a selection rule that cannot peek.
UNIVERSE_SIZE = 50

BENCHMARK_INDEX = "sh.000300"

# ------------------------------------------------------------------ execution
INITIAL_CAPITAL = 1_000_000.0
BOARD_LOT = 100

STAMP_DUTY_SELL = 0.0005      # 0.05%, seller only, post 2023-08-28
COMMISSION_RATE = 0.00025     # 0.025% each side
COMMISSION_MIN = 5.0          # RMB, per order
SLIPPAGE = 0.0005             # 0.05% each side

PRICE_LIMIT_PCT = 0.10        # +-10% main board; ST/GEM differences ignored
LIMIT_TOUCH_EPS = 0.01        # price within 1 fen of the limit counts as locked

# --------------------------------------------------------------- risk / sizing
RISK_FREE_ANNUAL = 0.02       # 2% -- what idle cash earns in reverse repo
TRADING_DAYS = 252
TARGET_ANNUAL_VOL = 0.12
MAX_SINGLE_WEIGHT = 0.20
REBALANCE_DEADBAND = 0.03     # skip trades below a 3pp weight change
N_SELECTED = 6                # names held by cross-sectional strategies

# ------------------------------------------------------------------ lookbacks
LOOKBACK_SHORT = 5
LOOKBACK_MED = 20
LOOKBACK_LONG = 60

REBALANCE_EVERY = 5           # trading days (weekly)
