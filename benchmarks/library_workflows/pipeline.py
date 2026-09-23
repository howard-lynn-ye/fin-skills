"""Paired coding-agent development study: build, change, and persist a retrieval pipeline.

The independently implemented mathematical scorer does not import fin_skills. Cases are
author-generated public development data, not externally authored holdout labels. Coding
agent time is measured, never described as human developer time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

from benchmarks.agent_study.prepare_followup import MODEL, REVISION

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("build", "historical", "restore")
ARMS = ("components", "fin_skills")
TOKEN_PATTERN = r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+"
SPEC = r'''Write a Python module defining solve(request, state_dir) -> list of dictionaries
with exactly id (original document id) and score (finite float). Return the complete module
as one Python code block, with no prose or multiple blocks. No network, shell or processes.
The request has documents (list of {id,text,source,available_at}), query, top_k, as_of.
Texts are shorter than 500 characters; treat each document as exactly one passage.
Tokenize casefolded text using re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.casefold()).
BM25: k1=1.5, b=.75. For each UNIQUE query term, sum
log(1+(N-df+.5)/(df+.5)) * tf*2.5 / (tf+1.5*(.25+.75*doc_length/mean_length)).
N, df and mean_length use only eligible documents; use 1 if mean_length is zero.
Exclude scores <=0. Sort descending score, then ascending original document id. Return
at most top_k hits. Handle zero eligible documents or zero matching terms by returning [].
Use ordinary JSON values. Exact IDs/order and scores within 1e-9 are checked. Paths outside
state_dir and installed dependencies are unavailable. The evaluator has no expected answers
inside your filesystem. Public feedback and final evaluation use different input fixtures.
'''
STAGE_SPEC = {
    "build": "BUILD: documents is always a list. Ignore as_of in this stage. No persistence required.",
    "historical": "CHANGE: implement as_of (ISO datetime with timezone). Exclude a document if "
        "available_at is missing or after as_of. If as_of is null, all documents are eligible. "
        "Cutoff precedes BM25 statistics. Preserve the original BUILD behavior for a null cutoff.",
    "restore": "PERSIST AND RESTORE: preserve CHANGE behavior. When documents is a list, persist "
        "the complete supplied collection under state_dir. When documents is null, reload the "
        "last saved collection. Calls can use a newly started Python process and a different "
        "query/cutoff. Save all documents, not just the documents eligible for the first query.",
}
LIBRARY_DOCS = '''Available library API (you may also use standard Python):
from fin_skills.rag import RAGIndex
index = RAGIndex(documents, chunk_size=1200, overlap=0)
documents use id,text,source,available_at; no extra fields. Defaults implement the specified BM25.
hits = index.search(query, top_k=top_k, as_of=as_of)
Every hit has document_id (the original id), score, id (a chunk id), text, and provenance.
index.save(path) saves the complete collection as JSON; refuses to overwrite an existing file.
index = RAGIndex.load(path) restores the index. Path can be pathlib.Path or a string.
You own path creation/replacement and output conversion. Use original document_id in outputs.
'''
COMPONENT_DOCS = '''Available components: Python standard library, numpy, pandas, scipy.
collections.Counter counts tokens. math.log is natural logarithm. json.dump/load persist JSON.
pathlib.Path(state_dir) identifies the writable directory. datetime.fromisoformat accepts
timezone offsets; replace a final Z with +00:00 when needed. sorted supports tuple keys.
The fin_skills package is unavailable for this condition; implement the same specification.
'''


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def reference(request, documents, stage):
    eligible = [d for d in documents if stage == "build" or request["as_of"] is None or (
        d["available_at"] is not None and stamp(d["available_at"]) <= stamp(request["as_of"]))]
    if not eligible:
        return []
    counts = [Counter(re.findall(TOKEN_PATTERN, d["text"].casefold())) for d in eligible]
    terms = set(re.findall(TOKEN_PATTERN, request["query"].casefold()))
    avg = sum(sum(c.values()) for c in counts) / len(counts) or 1.0
    rows = []
    for doc, count in zip(eligible, counts):
        score = 0.0
        for term in sorted(terms):
            freq = count[term]
            if freq:
                df = sum(term in c for c in counts)
                score += math.log(1 + (len(counts)-df+.5)/(df+.5)) * freq*2.5 / (
                    freq + 1.5*(.25+.75*sum(count.values())/avg))
        if score > 0:
            rows.append({"id": doc["id"], "score": score})
    return sorted(rows, key=lambda r: (-r["score"], r["id"]))[:request["top_k"]]


def cases(seed, stage, *, public=False):
    rng = random.Random(seed + (0 if public else 810000))
    words = ("risk", "return", "cost", "cash", "trade", "filing", "revision", "收益")
    docs = [dict(id=f"d{i:02}", text=" ".join(rng.choices(words, k=rng.randint(2, 12))),
        source=f"synthetic:pipeline/{i}", available_at=(
            None if i == 0 else "2026-01-01T00:00:00Z" if i % 2 else "2026-06-01T00:00:00Z"))
        for i in range(8 if public else 17)]
    queries = [("risk risk cost", 3, None), ("收益 cash", 2, "2026-03-01T02:00:00+02:00")]
    if not public:
        queries += [("unknownterm", 5, None), ("trade", 30, "2025-01-01T00:00:00Z"),
                    ("risk cost", 8, "2026-08-01T00:00:00Z")]
    output = []
    for query, k, cutoff in queries:
        request = dict(documents=docs, query=query, top_k=k, as_of=cutoff)
        requests = [request]
        if stage == "restore":
            # The second process must reconstruct the full corpus, including future records.
            requests = [dict(request, query="filing", as_of="2026-02-01T00:00:00Z"),
                        dict(request, documents=None)]
        output.append(dict(requests=requests,
                           expected=[reference(r, docs, stage) for r in requests]))
    return output


def matches(actual, expected):
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if not isinstance(a, dict) or set(a) != {"id", "score"} or a["id"] != b["id"]:
            return False
        if (type(a["score"]) not in (int, float) or not math.isfinite(a["score"])
                or not math.isclose(a["score"], b["score"], rel_tol=1e-9, abs_tol=1e-9)):
            return False
    return True


def extract_code(content):
    stripped = content.strip()
    if stripped.startswith("```"):
        match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", stripped, re.DOTALL)
        if not match or "```" in match[1]:
            raise ValueError("return exactly one complete Python code block")
        stripped = match[1]
    compile(stripped, "submission.py", "exec")
    return stripped


def evaluate(source, fixtures, arm, *, timeout=45):
    rows = []
    for case in fixtures:
        outcomes = []
        with tempfile.TemporaryDirectory(prefix="pipeline-eval-") as td:
            box = Path(td)
            state = box / "state"
            for i, (request, expected) in enumerate(zip(case["requests"], case["expected"])):
                payload, result = box / f"input-{i}.json", box / f"output-{i}.json"
                write(payload, dict(source=source, request=request))
                cmd = [sys.executable, "-m", "benchmarks.library_workflows.pipeline_worker",
                       "--input", str(payload), "--output", str(result), "--state", str(state)]
                if arm == "fin_skills":
                    cmd.append("--library")
                try:
                    # Confinement denies clone/thread creation. Match the existing financial
                    # worker's single-thread BLAS configuration before importing numpy.
                    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
                    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                          timeout=timeout, env=env)
                    value = json.loads(result.read_text()) if result.exists() else {
                        "status": "worker_error", "error": proc.stderr[-1500:]}
                except subprocess.TimeoutExpired:
                    value = {"status": "timeout"}
                outcomes.append(dict(passed=value.get("status") == "executed"
                    and matches(value.get("value"), expected), output=value))
        rows.append(dict(passed=all(r["passed"] for r in outcomes), calls=outcomes))
    return dict(passed=all(r["passed"] for r in rows), cases=rows)


def run(output, seed, *, attempts=4, chat=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    order = list(ARMS)
    random.Random(seed).shuffle(order)
    fixture_set = {stage: {kind: cases(seed, stage, public=kind == "public")
                          for kind in ("public", "evaluation")} for stage in STAGES}
    write(output / "protocol.json", dict(scope="public_development_coding_agent", seed=seed,
        model=MODEL, revision=REVISION, max_tokens=4096, attempts_per_stage=attempts, arms=order,
        stages=list(STAGES), spec=SPEC, stage_spec=STAGE_SPEC,
        documentation={"components": COMPONENT_DOCS, "fin_skills": LIBRARY_DOCS},
        fixtures=fixture_set, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in Path(__file__).parent.glob("*.py")},
        limits="Model sees public-case feedback only. Evaluation after each final source freeze. "
        "No human usability claim; stages/seeds on the same specification are correlated. "
        "Landlock/seccomp isolate filesystem and network, not in-process evaluator tampering."))
    if chat is None:
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("model inference requires Slurm")
        from benchmarks.agent_study.transformers_chat import TransformersChat
        chat = TransformersChat(MODEL, REVISION, max_tokens=4096)
    rows = []
    for arm in order:
        previous = ""
        for stage_index, stage in enumerate(STAGES):
            cell = output / f"{arm}-{stage}"
            cell.mkdir()
            docs = LIBRARY_DOCS if arm == "fin_skills" else COMPONENT_DOCS
            messages = [{"role": "system", "content": SPEC + "\n" + docs},
                {"role": "user", "content": STAGE_SPEC[stage] + "\nPrevious source:\n" + previous
                 + "\nPublic inputs:\n" + json.dumps([x["requests"] for x in fixture_set[stage]["public"]])}]
            chat.seed, chat.calls = seed * 100 + stage_index, 0
            attempts_log, candidate = [], previous
            started = time.monotonic()
            for attempt in range(attempts):
                turn_start = time.monotonic()
                response = None
                try:
                    response = chat(messages)
                    content = response["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": content})
                    candidate = extract_code(content)
                    feedback = evaluate(candidate, fixture_set[stage]["public"], arm)
                except Exception as exc:
                    feedback = dict(passed=False, error=f"{type(exc).__name__}: {exc}")
                record = dict(attempt=attempt, response=response, feedback=feedback,
                              seconds=time.monotonic()-turn_start)
                write(cell / f"attempt-{attempt}.json", record)
                attempts_log.append(record)
                if feedback["passed"]:
                    break
                messages.append({"role": "user", "content": "Public test feedback: "
                                 + json.dumps(feedback) + "\nReturn the entire corrected module."})
            elapsed = time.monotonic() - started
            (cell / "submission.py").write_text(candidate, encoding="utf-8")
            write(cell / "submission-receipt.json", dict(
                sha256=hashlib.sha256(candidate.encode()).hexdigest(), elapsed_seconds=elapsed,
                attempts=len(attempts_log), slurm_job_id=os.environ.get("SLURM_JOB_ID")))
            # Freeze before evaluating separate cases; their feedback never enters later prompts.
            grade = evaluate(candidate, fixture_set[stage]["evaluation"], arm)
            write(cell / "grade.json", grade)
            row = dict(seed=seed, arm=arm, stage=stage, correct=grade["passed"],
                attempts=len(attempts_log), elapsed_seconds=elapsed,
                public_passed=attempts_log[-1]["feedback"]["passed"],
                tokens=sum((a["response"] or {}).get("usage", {}).get("total_tokens", 0)
                           for a in attempts_log), source_characters=len(candidate))
            rows.append(row)
            previous = candidate
            print(json.dumps(row), flush=True)
    result = dict(complete=True, scope="public_development_coding_agent", rows=rows)
    write(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.output, args.seed)


if __name__ == "__main__":
    main()
