"""Next-open trading around the installed FinRL StockTradingEnv implementation.

Native cash/share matching is reused. Its same-bar observation and reward are
replaced with previous-close observations and independently reconciled close NAV.
"""
from functools import lru_cache
import hashlib
from importlib import metadata, util
import sys

import gymnasium as gym
import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import integer, number


@lru_cache(maxsize=1)
def _native_class():
    # FinRL's package initializer imports unrelated broker/training dependencies.
    # Load this installed standalone module without copying or modifying its code.
    package = metadata.distribution("finrl")
    if package.version != "0.3.7":
        raise ImportError("this adapter is validated with finrl==0.3.7")
    path = package.locate_file("finrl/meta/env_stock_trading/env_stocktrading.py")
    name = "_fin_skills_finrl_stock_env"
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.StockTradingEnv, hashlib.sha256(path.read_bytes()).hexdigest()


class FinRLNextOpenEnv(gym.Wrapper):
    """Long-only integer-share simulation; no broker or external data access.

Rows need date/tic/open/close and optional close-time feature columns. The caller
is responsible for point-in-time features and corporate-action-consistent prices.
Dates are explicit session labels. All stocks must occur once per session.
"""
    def __init__(self, frame, *, features=(), initial_cash=100000., hmax=100,
                 buy_cost=.001, sell_cost=.001):
        if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
            raise TypeError("frame must have unique DataFrame columns")
        self.features = tuple(features)
        if (len(set(self.features)) != len(self.features) or
                set(self.features) & {"date", "tic", "open", "close", "_not_blocked"}):
            raise ValueError("features must be distinct nonreserved columns")
        required = {"date", "tic", "open", "close", *self.features}
        if not required <= set(frame):
            raise ValueError(f"missing columns: {sorted(required - set(frame))}")
        data = frame[list(required)].copy(deep=True)
        data["date"] = pd.to_datetime(data.date)
        if (data.empty or data.isna().any().any() or data.date.dt.tz is not None
                or not (data.date == data.date.dt.normalize()).all()
                or data.duplicated(["date", "tic"]).any()):
            raise ValueError("nonmissing unique timezone-naive date/tic rows required")
        if not all(isinstance(t, str) and t for t in data.tic):
            raise ValueError("tic must contain nonempty string identifiers")
        if not np.isfinite(data[["open", "close", *self.features]].to_numpy(dtype=float)).all():
            raise ValueError("prices and features must be finite")
        if (data[["open", "close"]] <= 0).any().any():
            raise ValueError("prices must be positive")
        self.dates = pd.DatetimeIndex(sorted(data.date.unique()))
        self.assets = tuple(sorted(data.tic.unique()))
        self.n = len(self.assets)
        if len(self.dates) < 2 or len(data) != len(self.dates) * self.n:
            raise ValueError("need >= 2 complete sessions with the same asset universe")
        data = data.sort_values(["date", "tic"]).reset_index(drop=True)
        self.opens = data.open.to_numpy(float).reshape(-1, self.n)
        self.closes = data.close.to_numpy(float).reshape(-1, self.n)
        self.feature_values = [data[c].to_numpy(float).reshape(-1, self.n) for c in self.features]
        self.initial_cash = number(initial_cash, "initial_cash", minimum=1e-8)
        self.buy_cost = number(buy_cost, "buy_cost", minimum=0, maximum=.99)
        self.sell_cost = number(sell_cost, "sell_cost", minimum=0, maximum=.99)
        hmax = integer(hmax, "hmax", maximum=1000000)
        # The native engine sees next-open execution prices; policy never sees them.
        native = data[["date", "tic"]].copy()
        native["date"] = native.date.dt.strftime("%Y-%m-%d")
        native["close"] = np.vstack([self.opens[1:], self.closes[-1:]]).ravel()
        native["_not_blocked"] = 0.0
        native.index = np.repeat(np.arange(len(self.dates)), self.n)
        cls, self.backend_sha256 = _native_class()
        backend = cls(df=native, stock_dim=self.n, hmax=hmax,
            initial_amount=self.initial_cash, num_stock_shares=[0] * self.n,
            buy_cost_pct=[self.buy_cost] * self.n, sell_cost_pct=[self.sell_cost] * self.n,
            reward_scaling=1., state_space=1 + 3 * self.n, action_space=self.n,
            tech_indicator_list=["_not_blocked"], make_plots=False, print_verbosity=10**9)
        super().__init__(backend)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            shape=(1 + (2 + len(self.features)) * self.n,), dtype=np.float32)
        self.ledger = []
        self.nav = self.initial_cash
        self.finished = False

    def _holdings(self):
        return np.asarray(self.env.state[1 + self.n:1 + 2 * self.n], dtype=float)

    def _observation(self):
        day = self.env.day
        return np.concatenate([[self.env.state[0]], self.closes[day], self._holdings(),
                               *(x[day] for x in self.feature_values)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("reset options are not supported")
        self.env.reset(seed=seed)
        self.nav, self.finished, self.ledger = self.initial_cash, False, []
        return self._observation(), {"date": self.dates[0].isoformat(), "nav": self.nav}

    def step(self, action):
        if self.finished:
            raise RuntimeError("episode finished; call reset")
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n,) or not np.isfinite(action).all() or (np.abs(action) > 1).any():
            raise ValueError("action must be finite with one value in [-1, 1] per asset")
        before_cash, before_units = float(self.env.state[0]), self._holdings().copy()
        previous_nav = self.nav
        self.env.step(action.copy())
        day = self.env.day
        units = self._holdings()
        traded = units - before_units
        fees = float(np.sum(np.abs(traded) * self.opens[day] *
                            np.where(traded >= 0, self.buy_cost, self.sell_cost)))
        expected_cash = before_cash - float(traded @ self.opens[day]) - fees
        cash = float(self.env.state[0])
        if not np.isclose(cash, expected_cash, rtol=1e-10, atol=1e-8):
            raise ArithmeticError("FinRL fills do not reconcile with independent cash ledger")
        self.nav = float(cash + units @ self.closes[day])
        if not np.isfinite(self.nav) or self.nav <= 0 or cash < -1e-8 or (units < 0).any():
            raise ArithmeticError("invalid long-only account state")
        reward = float(np.log(self.nav / previous_nav))
        receipt = dict(date=self.dates[day].isoformat(), decision_session=self.dates[day-1].isoformat(),
            cash=cash, units=units.tolist(), traded=traded.tolist(), fees=fees, nav=self.nav,
            log_return=reward, feedback="available after this session closes")
        self.ledger.append(receipt)
        self.finished = day == len(self.dates) - 1
        return self._observation(), reward, self.finished, False, dict(receipt)


def make_finrl_env(frame, **parameters):
    """Construct the checked, next-open adapter to installed FinRL 0.3.7."""
    return FinRLNextOpenEnv(frame, **parameters)
