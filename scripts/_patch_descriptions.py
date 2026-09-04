# -*- coding: utf-8 -*-
"""Targeted second pass on descriptions, driven by eval_triggers.py misses.

Each substitution below fixes one measured confusion. Kept in the repo so the
next person can see WHY a phrase is worded the way it is.
Run:  python scripts/_patch_descriptions.py
"""
from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# skill -> list of (old substring, new substring)
PATCHES = {
    # "panel" was stealing "run a Fama-MacBeth regression on this panel"
    # "minute bars"/"store" were missing entirely
    "market-data-engineering": [
        ("a panel too big for memory; partitioning;",
         "storing years of minute bars for thousands of tickers; a dataset too big for memory; partitioning;"),
    ],
    # lost "CIK maps to ticker" and "10-K ... parse" to market-data-sourcing
    "fundamental-and-macro-data": [
        ("accession number, CIK, ticker-to-CIK mapping, edgartools;",
         "accession number, CIK, \"which CIK is this ticker\", ticker-to-CIK mapping, edgartools; parsing an income statement or balance sheet out of a filing;"),
        ("revenue, EPS or balance-sheet history;",
         "revenue, EPS or balance-sheet history as it was known on a past date;"),
    ],
    # "delisted" and "ticker" were pulling filings and A-share queries here
    "market-data-sourcing": [
        ("need delisted tickers or a survivorship-free universe;",
         "need delisted US or global tickers, or a survivorship-free universe;"),
        ("SKIP for storing or joining data you already hold (market-data-engineering), filings and macro (fundamental-and-macro-data), or A-shares (china-ashare-data).",
         "Also covers 美股 and global 行情数据 requests. SKIP for storing, partitioning or as-of joining data you already hold (market-data-engineering); for EDGAR filings, XBRL, CIK and macro vintages (fundamental-and-macro-data); and for A-share, 沪深 or 退市 queries (china-ashare-data)."),
    ],
    # "moving average" was losing the whole query to signal-construction
    "backtesting-engines": [
        ("TRIGGER - \"backtest this\", simulate a strategy,",
         "TRIGGER - \"backtest this\", backtest a crossover or a moving-average strategy, simulate a strategy,"),
    ],
    # CTP/vnpy queries were landing here
    "broker-execution-apis": [
        ("SKIP for crypto exchanges (crypto-data-and-execution) and Chinese brokers (china-trading-stack).",
         "SKIP for crypto exchanges and ccxt (crypto-data-and-execution), and for vnpy, CTP, QMT or any Chinese broker gateway (china-trading-stack)."),
    ],
    # The router must LOSE to a specific skill. Naming the most-confused ones in the
    # SKIP clause makes those terms count against it. The package list is trimmed to
    # stay inside the 1024-char spec limit.
    "quant-stack-router": [
        ("Also read it whenever code will import yfinance, pandas-datareader, openbb, akshare, tushare, ccxt, TA-Lib, vectorbt, backtrader, zipline, nautilus_trader, freqtrade, qlib, ib_insync, ib_async, alpaca, alphalens, PyPortfolioOpt, riskfolio, skfolio, quantstats, mlfinlab, QuantLib or arch, because several of those are dead, relicensed, or have flipped a default since training.",
         "Also read it before importing any finance package whose status you are assuming from memory - several widely used ones are dead, relicensed, or have flipped a default since training."),
        ("SKIP when the task clearly belongs to one named domain skill; go straight there instead of routing.",
         "SKIP when the task already names its own domain - go straight to market-data-sourcing, backtesting-engines, broker-execution-apis, portfolio-and-risk, factor-and-timeseries-research, china-ashare-data or crypto-data-and-execution rather than routing through here."),
    ],
    # "delisted" needs to be winnable here
    "china-ashare-data": [
        ("ST, 退市, delisted A-shares;",
         "ST, 退市, delisted A-share tickers, 退市股票列表;"),
    ],
    # Korean delisted/short-ban query was going to fundamental-and-macro-data
    "asia-pacific-markets": [
        ("Korea, KRX, KOSPI, KOSDAQ, pykrx, FinanceDataReader, CSAT, short-selling bans;",
         "Korea, KRX, KOSPI, KOSDAQ, pykrx, FinanceDataReader, CSAT, Korean short-selling ban dates, Korean delisted-stock lists;"),
    ],
}


def main() -> int:
    total = 0
    for md in sorted(ROOT.glob("plugins/*/skills/*/SKILL.md")):
        name = md.parent.name
        if name not in PATCHES:
            continue
        text = md.read_text(encoding="utf-8")
        m = re.match(r"(?s)^---\n(.*?)\n---\n", text)
        fm = m.group(1)
        dm = re.search(r"(?m)^description: >-\n((?:  .*\n?)+)", fm)
        desc = " ".join(dm.group(1).split())

        for old, new in PATCHES[name]:
            if new in desc:
                continue                      # already applied — the script is re-runnable
            if old not in desc:
                print(f"  !! anchor missing in {name}: {old[:50]!r}")
                return 1
            desc = desc.replace(old, new)
        if len(desc) > 1024:
            print(f"  !! {name} now {len(desc)} chars (max 1024)")
            return 1

        block = "description: >-\n" + textwrap.fill(
            desc, width=98, initial_indent="  ", subsequent_indent="  ", break_on_hyphens=False)
        fm_new = fm[: dm.start()] + block + "\n" + fm[dm.end():]
        md.write_text(text[: m.start(1)] + fm_new.rstrip("\n") + text[m.end(1):], encoding="utf-8")
        print(f"  {len(desc):>5}  {name}")
        total += 1
    print(f"\npatched {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
