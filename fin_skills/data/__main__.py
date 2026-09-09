"""`python -m fin_skills.data adapters` and `python -m fin_skills.data manifest`.

    python -m fin_skills.data adapters              the table, plus every warning
    python -m fin_skills.data adapters --name fred  one adapter in full
    python -m fin_skills.data adapters --table      just the grid
    python -m fin_skills.data manifest --root ./cache      what a cache holds
    python -m fin_skills.data manifest --root ./cache --verify   re-hash it all

Neither command imports a vendor library, and neither touches the network.
Everything printed is ASCII, so it survives a stock Windows console.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from fin_skills.api.base import ascii_only
from fin_skills.data import declare
from fin_skills.data.cache import Cache

#: where `manifest` looks when --root is not given
CACHE_ENV = "FIN_SKILLS_DATA_CACHE"


def _print(text: str) -> None:
    print(ascii_only(str(text)))


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

    a = sub.add_parser("adapters", help="what every registered adapter declares")
    a.add_argument("--name", help="one adapter (yfinance, akshare, ccxt, fred, edgar)")
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
