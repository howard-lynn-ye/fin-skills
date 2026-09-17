"""Point-in-time universe resolution and disk-cached price loading."""
from __future__ import annotations

import hashlib
import json
import os

import baostock as bs
import pandas as pd

from config import (BENCHMARK_INDEX, DATA_START, EVAL_END, UNIVERSE_ASOF,
                    UNIVERSE_SIZE)

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")
_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE, exist_ok=True)
    return os.path.join(_CACHE, name)


def resolve_universe(asof: str = UNIVERSE_ASOF, size: int = UNIVERSE_SIZE) -> list[str]:
    """HS300 constituents as they stood on `asof`, not as they stand today.

    This is the difference between a point-in-time universe and a survivorship-
    biased one. Names added to the index after `asof` were added partly because
    they performed well; including them hands the backtest information it could
    not have had.
    """
    path = _cache_path(f"universe_{asof}_{size}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["codes"]

    bs.login()
    try:
        rs = bs.query_hs300_stocks(date=asof)
        rows = []
        while (rs.error_code == "0") & rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        raise RuntimeError(f"no HS300 constituents returned for {asof}")

    df = pd.DataFrame(rows, columns=rs.fields)
    # Sort by code so the selection rule is deterministic and return-blind.
    codes = sorted(df["code"].tolist())[:size]
    with open(path, "w") as fh:
        json.dump({"asof": asof, "size": size,
                   "update_date": df["updateDate"].iloc[0],
                   "codes": codes}, fh, indent=2)
    return codes


def _fetch_one(code: str, start: str, end: str, fields: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(code, fields, start_date=start,
                                      end_date=end, frequency="d", adjustflag="3")
    rows = []
    while (rs.error_code == "0") & rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def load_prices(codes: list[str], start: str = DATA_START,
                end: str = EVAL_END) -> dict[str, pd.DataFrame]:
    """Forward-adjusted daily bars, cached to disk keyed by the exact request."""
    key = hashlib.sha256(("|".join(sorted(codes)) + start + end).encode()).hexdigest()[:16]
    path = _cache_path(f"prices_{key}.csv")

    if os.path.exists(path):
        combined = pd.read_csv(path)
    else:
        bs.login()
        try:
            frames = []
            for code in codes:
                df = _fetch_one(code, start, end, _FIELDS)
                if len(df):
                    frames.append(df)
        finally:
            bs.logout()
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(path, index=False)

    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined["pctChg"] = combined["pctChg"] / 100.0

    def pivot(field: str, fill: bool = True) -> pd.DataFrame:
        out = combined.pivot(index="date", columns="code", values=field)
        return out.ffill().bfill() if fill else out

    return {
        "open": pivot("open"),
        "high": pivot("high"),
        "low": pivot("low"),
        "close": pivot("close"),
        "preclose": pivot("preclose"),
        # A suspended day has no return, not a forward-filled one.
        "pct": combined.pivot(index="date", columns="code", values="pctChg").fillna(0.0),
        "volume": pivot("volume"),
    }


def load_index(code: str = BENCHMARK_INDEX, start: str = DATA_START,
               end: str = EVAL_END) -> pd.Series:
    path = _cache_path(f"index_{code}_{start}_{end}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        bs.login()
        try:
            df = _fetch_one(code, start, end, "date,code,close")
        finally:
            bs.logout()
        df.to_csv(path, index=False)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.set_index("date")["close"].ffill().bfill()
