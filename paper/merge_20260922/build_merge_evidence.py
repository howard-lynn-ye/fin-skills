"""Extract author-reported Overleaf values; do not relabel them as replicated results."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "overleaf_before"
OUT = ROOT / "paper" / "latex_naacl"


def number(cell):
    # Formatting in the observed table; color names must not become data.
    cell = re.sub(r"\\(?:cellcolor|color)\{[^}]+\}", "", cell)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
    if match is None:
        raise ValueError(cell)
    return float(match.group())


def main():
    appendix = (SOURCE / "sections/appendix.tex").read_text(encoding="utf-8")
    method = (SOURCE / "sections/method.tex").read_text(encoding="utf-8")
    seeds = []
    for line in appendix.splitlines():
        match = re.match(r"HiSTrim Seed (\d) .*?xid/(\d+)", line)
        if match:
            cells = line.split("&")
            seeds.append({"seed": int(match[1]), "source_run_id": match[2],
                          "kv_percent": number(cells[2]),
                          "pass32_percent": number(cells[6]),
                          "mrr32_times100": number(cells[8])})
    assert len(seeds) == 5 and len({s["seed"] for s in seeds}) == 5
    aggregates = {}
    for key in ("kv_percent", "pass32_percent", "mrr32_times100"):
        values = [row[key] for row in seeds]
        aggregates[key] = {"mean": statistics.mean(values),
                           "sample_sd": statistics.stdev(values)}

    # Retain every baseline and every task in the source's generative MRR table.
    matrix = []
    scale = None
    for line in method.splitlines():
        if "\\label{tab:cross_paradigm_qwen_dcn}" in line:
            break
        if "\\multirow{6}" in line:
            scale = "1.7B" if "1.7B" in line else "8B"
        if scale and line.lstrip().startswith("&"):
            cells = line.split("&")
            if len(cells) != 14:
                continue
            label = cells[1]
            names = ("Full Dense", "TokenDrop-50", "H2O", "SnapKV",
                     "Router-Tuning", "HiSTrim")
            name = next(n for n in names if n in label)
            matrix.append({"model": scale, "method": name,
                           "kv_percent": number(cells[2]),
                           "pass32": [number(cells[i]) for i in (3, 5, 7, 9, 11)],
                           "mrr32_times100": [number(cells[i]) for i in (4, 6, 8, 10, 12)]})
    assert len(matrix) == 12
    for scale in ("1.7B", "8B"):
        rows = [r for r in matrix if r["model"] == scale]
        baseline = next(r for r in rows if r["method"] == "Full Dense")
        for row in rows:
            row["mean_task_relative_mrr_percent"] = statistics.mean(
                100 * a / b for a, b in zip(row["mrr32_times100"], baseline["mrr32_times100"]))

    # Analytic identities and a counterexample to unconditional sink conservation.
    sink, removed = 0.0, [0.0, 0.0]
    exact_delta = math.log1p(sum(math.exp(s - sink) for s in removed))
    heuristic_delta = 0.15 * math.log1p(len(removed))
    assert math.isclose(math.exp(sink + exact_delta), 3.0)
    assert not math.isclose(math.exp(sink + heuristic_delta), 3.0)
    for gate in (-8.0, -1.0, 0.0, 2.0, 8.0):
        for utility in (-3.0, 0.0, 3.0):
            score = utility / (1 + math.exp(-gate))
            assert (score > 0) == (utility > 0) and (score < 0) == (utility < 0)

    sources = []
    for rel in ("main.tex", "sections/method.tex", "sections/experiment.tex",
                "sections/appendix.tex", "sections/abstract.tex",
                "PAPER_WRITING_IMPROVEMENTS_AND_GUIDELINES.md"):
        sources.append({"path": "overleaf_before/" + rel,
                        "sha256": hashlib.sha256((SOURCE / rel).read_bytes()).hexdigest(),
                        "status": "author_manuscript_read; underlying runs not accessed"})
    record = {"status": "WORKING_DRAFT_NOT_HUMAN_APPROVED",
              "overleaf_project": "https://www.overleaf.com/project/6aad9a03f27c3c07a182965d",
              "source_history": "Shwai He upload, September 22, 2026, 19:05 (UI display)",
              "sources": sources, "seed_rows": seeds, "seed_aggregates": aggregates,
              "generative_results": matrix,
              "arithmetic_scope": "Recomputed from rounded manuscript rows; no raw predictions or significance test",
              "sink_counterexample": {"target_mass": 3.0, "exact_delta": exact_delta,
                                      "heuristic_mass": math.exp(heuristic_delta)},
              "unresolved": ["HiSTrim training/evaluation code and original run receipts",
                             "User-level predictions, uncertainty and dataset permissions",
                             "Conflicting source caption configurations",
                             "Joint fin-skills/HiSTrim/online-memory experiment",
                             "Accountable human review of sources and scientific decisions"]}
    (HERE / "merge_evidence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    macro_map = {"HSeedPassMean": aggregates["pass32_percent"]["mean"],
                 "HSeedPassSD": aggregates["pass32_percent"]["sample_sd"],
                 "HSeedMrrMean": aggregates["mrr32_times100"]["mean"],
                 "HSeedMrrSD": aggregates["mrr32_times100"]["sample_sd"]}
    lines = ["% Generated from author-reported Overleaf rows, not independent replication."]
    lines += ["\\newcommand{\\" + k + "}{" + f"{v:.2f}" + "}" for k, v in macro_map.items()]
    (OUT / "merge_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = [r"\begin{table*}[t]", r"\centering\small",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & Method & Hist. KV (\%) & Ad & Product & Video & Label-cond. & Interactive \\",
             r"\midrule"]
    for row in matrix:
        values = " & ".join(f"{x:.2f}" for x in row["mrr32_times100"])
        table.append(f"{row['model']} & {row['method']} & {row['kv_percent']:.1f} & {values} " + r"\\")
        if row["method"] == "HiSTrim" and row["model"] == "1.7B":
            table.append(r"\midrule")
    table += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Author-reported generative recommendation results retained from the",
              r"Overleaf source table: MRR@32 multiplied by 100, with the source reporting",
              r"1,000 users per task and 32 beams. Historical KV retention excludes protected",
              r"tokens. These manuscript values have not been reproduced from predictions;",
              r"source significance markers are omitted pending the underlying paired records.}",
              r"\label{tab:histrim_source_mrr}", r"\end{table*}"]
    (OUT / "histrim_source_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps({"seed_aggregates": aggregates, "source_rows": len(matrix),
                      "sink_counterexample": record["sink_counterexample"]}, indent=2))


if __name__ == "__main__":
    main()
