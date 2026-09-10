"""`advise`, `adapters` and `manifest` - what to use, what it declares, what you kept.

    python -m fin_skills.data advise --delisted             which source can serve this
    python -m fin_skills.data advise --asset-class crypto --market CRYPTO --no-key
    python -m fin_skills.data advise --coverage            what each adapter reaches
    python -m fin_skills.data adapters              the table, plus every warning
    python -m fin_skills.data adapters --name fred  one adapter in full
    python -m fin_skills.data adapters --table      just the grid
    python -m fin_skills.data manifest --root ./cache      what a cache holds
    python -m fin_skills.data manifest --root ./cache --verify   re-hash it all

No command imports a vendor library, and none touches the network. Everything printed is
ASCII, so it survives a stock Windows console. `advise` exits 0 when something can serve
the request and 3 when nothing can, so a pipeline can branch on the dead end rather than
grep for it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from fin_skills.api.base import ascii_only
from fin_skills.data import advise as advise_mod
from fin_skills.data import declare
from fin_skills.data.cache import Cache

#: `advise` exit code when NO adapter can serve the request - a real answer, not an error
NOTHING_SERVES = 3

#: where `manifest` looks when --root is not given
CACHE_ENV = "FIN_SKILLS_DATA_CACHE"


def _print(text: str) -> None:
    print(ascii_only(str(text)))


def cmd_advise(args: argparse.Namespace) -> int:
    if args.coverage:
        with pd.option_context("display.width", 240, "display.max_columns", 40,
                               "display.max_colwidth", 100):
            _print(advise_mod.coverage_frame().to_string(index=False))
        return 0
    need = advise_mod.Need(
        asset_class=args.asset_class, market=args.market, frequency=args.frequency,
        history_years=args.history_years, method=args.method, delisted=args.delisted,
        point_in_time=args.point_in_time, redistribute=args.redistribute,
        store=args.store, key_ok=not args.no_key, paid_ok=args.paid_ok)
    rec = advise_mod.recommend(need)
    if args.table:
        frame = rec.to_frame()
        _print(frame.to_string(index=False) if len(frame)
               else "no adapter can serve this request")
    else:
        _print(rec.report())
    return 0 if rec.ranked else NOTHING_SERVES


def cmd_adapters(args: argparse.Namespace) -> int:
    if args.name:
        _print(declare.describe(args.name))
        return 0
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        _print(declare.adapters().to_string(index=False))
    if not args.table:
        _print("")
        _print(declare.describe())
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    root = args.root or os.environ.get(CACHE_ENV)
    if not root:
        _print(f"no cache root: pass --root PATH or set ${CACHE_ENV}")
        return 2
    path = Path(root)
    if not path.is_dir():
        _print(f"no such cache directory: {path}")
        return 2
    cache = Cache(path)
    man = cache.manifest()
    if man.empty:
        _print(f"{path}: empty cache")
    else:
        with pd.option_context("display.width", 220, "display.max_columns", 40):
            _print(man.to_string(index=False))
    if args.verify:
        _print("")
        findings = cache.verify()
        for f in findings:
            _print(f"  {f}")
        n_err = sum(1 for f in findings if f.severity == "error")
        _print(f"{'FAIL' if n_err else 'OK'}  {len(findings)} artefact(s), "
               f"{n_err} hash mismatch(es)")
        return 1 if n_err else 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fin_skills.data",
                                description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser(
        "advise", help="which adapter can serve what you need - or why none can",
        description="Rank the adapters that can serve a stated need, or return nothing "
                    "and name the flag that emptied the list, what a partial answer "
                    "would be, and which paid vendor does serve it. Exits "
                    f"{NOTHING_SERVES} when no adapter can serve the request.")
    v.add_argument("--asset-class", default="equity",
                   choices=list(advise_mod.ASSET_CLASSES))
    v.add_argument("--market", default="US", choices=list(advise_mod.MARKETS))
    v.add_argument("--frequency", default="1d", help="an interval, e.g. 1d, 1h, 1wk")
    v.add_argument("--history-years", type=float, default=1.0,
                   help="how far back the study needs to reach")
    v.add_argument("--method", default="bars", choices=list(advise_mod.METHODS),
                   help="what you actually want to call")
    v.add_argument("--delisted", action="store_true",
                   help="delisted names must be present (the decisive flag)")
    v.add_argument("--point-in-time", action="store_true",
                   help="must answer 'what was known at t?'")
    v.add_argument("--redistribute", action="store_true",
                   help="the result will be passed on to someone else")
    v.add_argument("--store", action="store_true",
                   help="the result will be written to disk")
    v.add_argument("--no-key", action="store_true",
                   help="refuse any source that needs an API key")
    v.add_argument("--paid-ok", action="store_true",
                   help="paid vendors are acceptable")
    v.add_argument("--coverage", action="store_true",
                   help="print what each adapter reaches, and where that was verified")
    v.add_argument("--table", action="store_true", help="the ranked grid only")
    v.set_defaults(func=cmd_advise)

    a = sub.add_parser("adapters", help="what every registered adapter declares")
    a.add_argument("--name", help="one adapter (yfinance, akshare, ccxt, fred, edgar, "
                                  "tiingo, alphavantage, stooq)")
    a.add_argument("--table", action="store_true", help="the grid only, no warnings")
    a.set_defaults(func=cmd_adapters)

    m = sub.add_parser("manifest", help="one row per artefact in a cache")
    m.add_argument("--root", help=f"cache directory (default ${CACHE_ENV})")
    m.add_argument("--verify", action="store_true", help="re-hash every artefact")
    m.set_defaults(func=cmd_manifest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
