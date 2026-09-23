"""Measure revision selection against a raw-event reference, including negative controls.

Public synthetic development cases; neither independent labels nor a human usability study.
Inputs and expected outputs are frozen before calling the library implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import time

from examples.collected_rag import observed_documents
from fin_skills.collect import Batch, Event, Store, Watch


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def fixture(seed, records=80):
    rng = random.Random(seed)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for number in range(records):
        first = rng.randrange(0, 48)
        for revision in range(rng.randint(1, 4)):
            observed = first + revision * 24
            published = observed + rng.choice((-36, -12, 0, 12))
            rows.append(dict(id=f"r{number}", revision=revision,
                observed_at=(origin + timedelta(hours=observed)).isoformat(),
                published_at=(origin + timedelta(hours=published)).isoformat()))
    rows.sort(key=lambda row: (row["observed_at"], row["id"], row["revision"]))
    cutoffs = [(origin + timedelta(hours=h)).isoformat() for h in (0, 12, 36, 60, 84, 120)]
    expected = []
    for cutoff in cutoffs:
        eligible = [row for row in rows if row["observed_at"] <= cutoff
                    and row["published_at"] <= cutoff]
        selected = {}
        for row in eligible:
            selected[row["id"]] = row["revision"]
        expected.append(dict(as_of=cutoff, selected=selected))
    return dict(seed=seed, rows=rows, expected=expected)


def event(row):
    return Event(row["id"], "synthetic", "revision", row["id"],
        f"synthetic:temporal/{row['id']}", row["observed_at"], row["published_at"],
        data={"text": str(row["revision"]), "revision": row["revision"]})


def populate(path, rows, *, restart=False):
    watch = Watch("development", "synthetic", "temporal", enabled=False)
    midpoint = len(rows) // 2
    store = Store(path)
    store.put_watch(watch)
    try:
        for index, row in enumerate(rows):
            if restart and index == midpoint:
                store.close()
                store = Store(path)
            batch = Batch(events=[event(row)], state={"cursor": index})
            store.save(watch, batch, next_due=0)
            store.save(watch, batch, next_due=0)  # Exact repeated observation, no new revision.
        before = store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        prior_state = store.state(watch.id)
        bad = dict(rows[-1], id="rollback-probe")
        try:
            store.save(watch, Batch(events=[event(bad)], state={"invalid": float("nan")}),
                       next_due=0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid batch unexpectedly committed")
        after = store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {"stored_revisions": before, "expected_revisions": len(rows),
                "duplicate_revisions": before - len(rows),
                "rollback_ok": before == after and store.state(watch.id) == prior_state}
    finally:
        store.close()


def evaluate(path, case):
    output = []
    with Store(path) as store:
        latest = store.events(latest=True, limit=1000)
        for query in case["expected"]:
            cutoff, expected = query["as_of"], query["selected"]
            for method in ("latest_unfiltered", "latest_then_filter", "versioned_library"):
                start = time.perf_counter()
                if method == "versioned_library":
                    docs = observed_documents(store, cutoff)
                    selected = {d.metadata["id"]: int(d.text.split("\n", 1)[1]) for d in docs}
                else:
                    candidates = latest if method == "latest_unfiltered" else [r for r in latest
                        if r["observed_at"] <= cutoff and r["published_at"] <= cutoff]
                    selected = {r["id"]: r["data"]["revision"] for r in candidates}
                elapsed = time.perf_counter() - start
                wrong = sum(expected.get(k) != v for k, v in selected.items())
                omitted = len(set(expected) - set(selected))
                output.append(dict(seed=case["seed"], method=method, as_of=cutoff,
                    selected=selected, expected=expected, wrong_versions=wrong,
                    omitted_records=omitted, exact=selected == expected,
                    expected_records=len(expected), elapsed_seconds=elapsed))
    return output


def run(output, seeds=(11, 23, 37), records=80):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = [fixture(seed, records) for seed in seeds]
    protocol = dict(scope="public_synthetic_development", seeds=list(seeds), records=records,
        cases=cases, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="Raw-event reference is the correct baseline; latest-only methods are "
        "intentional negative controls. Reopen and transaction rollback are not process-kill tests.")
    write(output / "protocol.json", protocol)
    rows, storage = [], []
    for case in cases:
        uninterrupted = output / f"s{case['seed']}-uninterrupted.sqlite3"
        reopened = output / f"s{case['seed']}-reopened.sqlite3"
        a = populate(uninterrupted, case["rows"])
        b = populate(reopened, case["rows"], restart=True)
        left, right = evaluate(uninterrupted, case), evaluate(reopened, case)
        matches = all(x["selected"] == y["selected"] for x, y in zip(left, right))
        storage.append(dict(seed=case["seed"], uninterrupted=a, reopened=b,
                            reopen_results_identical=matches))
        rows.extend(right)
    summary = {method: dict(queries=sum(r["method"] == method for r in rows),
        exact_queries=sum(r["exact"] for r in rows if r["method"] == method),
        wrong_versions=sum(r["wrong_versions"] for r in rows if r["method"] == method),
        omitted_records=sum(r["omitted_records"] for r in rows if r["method"] == method))
        for method in ("latest_unfiltered", "latest_then_filter", "versioned_library")}
    result = dict(scope=protocol["scope"], summary=summary, storage=storage, rows=rows,
        complete=True, protocol_sha256=hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest())
    write(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
