"""Independently reconstruct every saved account from fills and the market snapshot.

Standard library only; does not import the policy, upstream broker or run-time ledger.
The archive is read in place and never extracted.
"""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import tarfile


def audit(path):
    counters = {"windows": 0, "fills": 0, "nav_observations": 0}
    largest_error = 0.0
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        def read(name):
            return json.load(archive.extractfile(name))
        market_bytes = archive.extractfile("data/kraken-daily.json").read()
        market = {int(row[0]): row for row in json.loads(market_bytes)["rows"]}
        result_name, = [n for n in names if n.endswith("/results.json")]
        result = read(result_name)
        assert result["complete"]
        assert hashlib.sha256(market_bytes).hexdigest() == result["protocol"]["data_sha256"]
        fee_rate = Decimal(str(result["protocol"]["commission_bps_per_fill"])) / Decimal(10000)
        half = Decimal(str(result["protocol"]["half_spread_bps"])) / Decimal(10000)
        def price(mid, side):
            factor = 1+half if side == "BUY" else 1-half
            return (Decimal(str(mid))*factor).quantize(Decimal(".01"))
        for name in sorted(n for n in names if n.endswith("/trace.json")):
            stem = name.removesuffix("trace.json")
            trace, fills, summary = read(name), read(stem+"fills.json"), read(stem+"summary.json")
            due = defaultdict(list)
            for fill in fills:
                assert fill["fill_time"] > fill["decision_time"]
                assert math.isclose(fill["fill_time"]-fill["decision_time"], .001, abs_tol=1e-6)
                due[int(fill["fill_time"])].append(fill)
            cash, units, previous = Decimal("10"), Decimal(0), Decimal("10")
            for row in trace:
                bar_time = int(round(row["time"]+.001))-86400
                bar = market[bar_time]
                for fill in due.pop(bar_time, []):
                    quantity, amount, fee = [Decimal(str(fill[k])) for k in ("quantity", "amount", "fee")]
                    assert math.isclose(float(amount), float(quantity*price(bar[3], fill["side"])), abs_tol=1e-9)
                    assert math.isclose(float(fee), float(amount*fee_rate), abs_tol=1e-10)
                    sign = 1 if fill["side"] == "BUY" else -1
                    cash -= sign*amount+fee
                    units += sign*quantity
                    assert cash >= Decimal("-1e-9") and units >= Decimal("-1e-9")
                    counters["fills"] += 1
                nav = cash+units*price(bar[4], "SELL")
                error = abs(float(nav)-row["nav"])
                largest_error = max(largest_error, error)
                assert error < 1e-8
                assert math.isclose(float(cash), row["cash"], abs_tol=1e-8)
                assert math.isclose(float(units), row["units"], abs_tol=1e-9)
                assert math.isclose(math.log(float(nav/previous)), row["reward"], abs_tol=1e-9)
                previous = nav
                counters["nav_observations"] += 1
            assert not due
            assert len(fills) == summary["fills"]
            assert math.isclose((float(previous)/10-1)*100, summary["net_return_pct"], abs_tol=1e-8)
            counters["windows"] += 1
    return {"status": "pass", **counters, "maximum_nav_absolute_error": largest_error,
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "checks": ["snapshot hash", "next-open fill ordering", "bid/ask fill arithmetic",
                       "commission", "nonnegative cash and inventory", "all NAVs and rewards",
                       "terminal return"],
            "scope": "Bar-level accounting under declared costs; not a proof of real-market execution or profitable learning."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
