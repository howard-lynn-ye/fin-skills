"""Data loader for the 10-Year (2015-2026) Global Multi-Asset ETF Universe.

Features:
  - 8 core assets accessible to mainland retail investors with 0% stamp duty
  - Date alignment across domestic and QDII trading calendars
  - SHA256 / JSON disk caching
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "_cache_10y"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_ETF_UNIVERSE = {
    "510300": {"secid": "1.510300", "name": "沪深300ETF", "category": "A股核心大盘"},
    "510500": {"secid": "1.510500", "name": "中证500ETF", "category": "A股中盘成长"},
    "510880": {"secid": "1.510880", "name": "红利ETF", "category": "A股高股息价值"},
    "518880": {"secid": "1.518880", "name": "黄金ETF", "category": "大宗商品避险"},
    "511010": {"secid": "1.511010", "name": "国债ETF", "category": "固定收益避风港"},
    "513100": {"secid": "1.513100", "name": "纳指100ETF", "category": "美股科技成长(QDII)"},
    "513500": {"secid": "1.513500", "name": "标普500ETF", "category": "美股核心大盘(QDII)"},
    "510900": {"secid": "1.510900", "name": "恒生ETF", "category": "港股核心资产(QDII)"},
}

START_DATE = "20150101"
END_DATE = "20260911"


def fetch_10y_etf_data(start_date: str = START_DATE, end_date: str = END_DATE) -> dict[str, pd.DataFrame]:
    """Fetch 10-year daily bars for the 8 global ETFs and return aligned DataFrames."""
    cache_file = CACHE_DIR / f"raw_etf_{start_date}_{end_date}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        raw_data = {}
        for code, meta in GLOBAL_ETF_UNIVERSE.items():
            secid = meta["secid"]
            url = (
                f"http://push2his.eastmoney.com/api/qt/stock/kline/get?"
                f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57"
                f"&klt=101&fqt=1&beg={start_date}&end={end_date}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode())
                raw_data[code] = res["data"]["klines"]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False)

    records = []
    for code, klines in raw_data.items():
        for line in klines:
            parts = line.split(",")
            records.append({
                "date": parts[0],
                "code": code,
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
            })

    raw_df = pd.DataFrame(records)

    # Pivot into matrices aligned on A-share trading calendar (defined by 510300 dates)
    ashare_dates = sorted(raw_df[raw_df["code"] == "510300"]["date"].unique())
    date_index = pd.Index(ashare_dates, name="date")

    def pivot_field(field: str) -> pd.DataFrame:
        mat = raw_df.pivot(index="date", columns="code", values=field)
        mat = mat.reindex(date_index)
        # Forward fill across holidays (e.g. US market closed on Martin Luther King Jr. Day)
        return mat.ffill().bfill()

    open_mat = pivot_field("open")
    close_mat = pivot_field("close")
    high_mat = pivot_field("high")
    low_mat = pivot_field("low")
    vol_mat = pivot_field("volume")
    pct_mat = close_mat.pct_change().fillna(0.0)

    return {
        "open": open_mat,
        "close": close_mat,
        "high": high_mat,
        "low": low_mat,
        "volume": vol_mat,
        "pct": pct_mat,
        "dates": ashare_dates,
    }


if __name__ == "__main__":
    data = fetch_10y_etf_data()
    print("10-Year Global ETF Data Loaded:")
    print(f"Total Trading Days: {len(data['dates'])} ({data['dates'][0]} to {data['dates'][-1]})")
    print("Assets Close Matrix Head:")
    print(data["close"].head(2))
    print("Assets Close Matrix Tail:")
    print(data["close"].tail(2))
