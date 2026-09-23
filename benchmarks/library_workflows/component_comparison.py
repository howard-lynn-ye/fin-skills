"""Correct native SQLite versus the library composition on identical stored revisions.

This isolates query behavior on the same database, not full-stack replacement or developer
effort. Both return the same id/revision map. The library path additionally constructs
provenance-bearing documents internally; timing is an endpoint cost, not equal work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sqlite3
import statistics
import time

from benchmarks.library_workflows.temporal import fixture, populate, write
from examples.collected_rag import observed_documents
from fin_skills.collect import Store

SQL = """WITH eligible AS (
 SELECT event_id,record,ROW_NUMBER() OVER (
  PARTITION BY watch_id,event_id ORDER BY julianday(observed_at) DESC,seq DESC
 ) AS position
 FROM events
 WHERE julianday(observed_at) <= julianday(?)
 AND julianday(COALESCE(json_extract(record,'$.published_at'), observed_at)) <= julianday(?)
)
SELECT event_id,json_extract(record,'$.data.revision') AS revision
FROM eligible WHERE position=1"""


def native_query(db, cutoff):
    return {row[0]: int(row[1]) for row in db.execute(SQL, (cutoff, cutoff))}


def library_query(store, cutoff):
    return {d.metadata["id"]: int(d.text.split("\n", 1)[1]) for d in observed_documents(store, cutoff)}


def run(output, *, sizes=(80, 800, 4000), seeds=(11, 23, 37), repetitions=7):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = [dict(records=n, **fixture(seed, n)) for n in sizes for seed in seeds]
    write(output / "protocol.json", dict(scope="public_development_native_component_comparison",
        cases=cases, repetitions=repetitions, sql=SQL, warmups=1,
        software=dict(python=platform.python_version(), sqlite=sqlite3.sqlite_version),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        design="same Store-populated SQLite file; alternating randomized query order; full results materialized",
        limits="Query endpoint comparison only; ingestion is shared, not compared. Library constructs "
        "provenance documents internally. Warm repeated timings are correlated and not human work time."))
    rows = []
    for case in cases:
        path = output / f"n{case['records']}-s{case['seed']}.sqlite3"
        storage = populate(path, case["rows"], restart=True)
        write(path.with_suffix(".storage.json"), storage)
        rng = random.Random(case["seed"] + case["records"])
        # Same database, separate real connections; neither path changes indices or cache settings.
        with Store(path) as store, sqlite3.connect(path) as db:
            for q in case["expected"]:
                functions = {"native_sql": lambda: native_query(db, q["as_of"]),
                             "library_composition": lambda: library_query(store, q["as_of"])}
                for fn in functions.values():
                    fn()
                values = {name: [] for name in functions}
                outputs = {name: [] for name in functions}
                for repetition in range(repetitions):
                    order = list(functions)
                    rng.shuffle(order)
                    for name in order:
                        start = time.perf_counter()
                        result = functions[name]()
                        values[name].append(time.perf_counter()-start)
                        outputs[name].append(result)
                for name in functions:
                    correct = [x == q["selected"] for x in outputs[name]]
                    row = dict(records=case["records"], stored_revisions=len(case["rows"]),
                        seed=case["seed"], as_of=q["as_of"], method=name,
                        exact=all(correct), repetitions_correct=sum(correct),
                        median_seconds=statistics.median(values[name]), timings_seconds=values[name],
                        selected=outputs[name][-1], expected=q["selected"])
                    rows.append(row)
        print(json.dumps(dict(records=case["records"], seed=case["seed"], completed=True)), flush=True)
    summary = []
    for n in sizes:
        for method in ("native_sql", "library_composition"):
            group = [r for r in rows if r["records"] == n and r["method"] == method]
            summary.append(dict(records=n, method=method, queries=len(group),
                exact_queries=sum(r["exact"] for r in group),
                median_query_seconds=statistics.median(r["median_seconds"] for r in group)))
    result = dict(complete=True, summary=summary, rows=rows)
    write(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("formal experiment requires Slurm")
    print(json.dumps(run(args.output)["summary"]))


if __name__ == "__main__":
    main()
