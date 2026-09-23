"""One Slurm job; all writable runtime paths belong to its RADFM directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.beacon_campaign import validate_root, write_once


def main():
    root = Path(validate_root(sys.argv[1]))
    plan = json.loads((root / "campaign.json").read_text())
    job = next(j for j in plan["jobs"] if j["id"] == sys.argv[2])
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    work = root / "jobs" / job["id"]
    if root.resolve() != root or work.resolve() != work or REPO != root / "source":
        raise ValueError("runtime paths differ from frozen campaign")
    for name in ("tmp", "cache", "results"):
        target = work / name
        if target.resolve() != target:
            raise ValueError("output path redirected")
        target.mkdir(exist_ok=False)
    mapping = {"TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
        "XDG_CACHE_HOME": "cache/xdg", "XDG_CONFIG_HOME": "cache/config",
        "XDG_DATA_HOME": "cache/data", "XDG_STATE_HOME": "cache/state",
        "HF_HOME": "cache/hf", "HF_DATASETS_CACHE": "cache/datasets",
        "HF_MODULES_CACHE": "cache/modules", "TORCH_HOME": "cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TRITON_CACHE_DIR": "cache/triton",
        "CUDA_CACHE_PATH": "cache/cuda", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/mpl", "PIP_CACHE_DIR": "cache/pip",
        "PYTHONPYCACHEPREFIX": "cache/pycache"}
    os.environ.update({key: str(work / value) for key, value in mapping.items()})
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        PYTHONPATH=str(REPO), FIN_STUDY_ROOT=str(work), OPENBLAS_NUM_THREADS=str(job["cpus"]),
        FIN_STUDY_SHARED_RUNTIME="/beacon-projects/radfm/envs/rfj-gpu",
        OMP_NUM_THREADS=str(job["cpus"]), MKL_NUM_THREADS=str(job["cpus"]),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    os.chdir(REPO)
    started = time.monotonic()
    commands = []

    def run(arguments):
        command = [sys.executable, *arguments]
        result = subprocess.run(command, check=False)
        commands.append(dict(command=command, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f"subprocess failed with exit {result.returncode}")

    status = "failed"
    try:
        if job["study"] in ("agent", "pipeline", "feedback", "retrieval", "matched-repair", "decision-loop"):
            import torch
            if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3:
                raise RuntimeError("14B requires an allocated GPU with at least 40 GiB")
            cache = plan["runtime_root"] + "/cache/hf/hub"
            if job.get("model"):
                cache = str(root / "model-cache/hub")
            if job.get("model_cache"):
                cache = job["model_cache"]
                if not cache.startswith("/beacon-projects/radfm/wy891/"):
                    raise ValueError("model cache must remain in RADFM")
            os.environ.update(HF_HUB_CACHE=cache, HUGGINGFACE_HUB_CACHE=cache,
                FIN_STUDY_SHARED_RUNTIME="/beacon-projects/radfm/envs/rfj-gpu")
            run(["-m", "benchmarks.agent_study.sandbox_probe"])
            os.environ["FIN_STUDY_REQUIRE_SANDBOX"] = "1"
        if job["study"] == "decision-loop":
            run(["-m", "pytest", "-q", "tests/test_decision_loop.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}", "--basetemp", str(work / "tmp/pytest")])
            for current in job.get("qualification_models", [job]):
                os.environ.update(HF_HUB_CACHE=current["model_cache"],
                                  HUGGINGFACE_HUB_CACHE=current["model_cache"])
                suffix = current.get("family", "study")
                run(["-m", "benchmarks.agent_study.decision_loop",
                     "--output", str(work / f"results/decision-loop/{suffix}"), "--group", job["group"],
                     "--model", current["model"], "--revision", current["revision"], "--seed", str(job["seed"])])
        elif job["study"] == "model-staging":
            run(["-m", "pytest", "-q", "tests/test_matched_repair.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
            from benchmarks.agent_study.model_transfer import stage
            stage(work / "results/model-staging", root / "model-cache/hub")
        elif job["study"] == "boundary-recheck":
            run(["-m", "benchmarks.agent_study.sandbox_probe"])
            os.environ["FIN_STUDY_REQUIRE_SANDBOX"] = "1"
            run(["-m", "benchmarks.agent_study.boundary_recheck",
                 "--output", str(work / "results/boundary-recheck")])
        elif job["study"] == "audit-validity":
            run(["-m", "benchmarks.agent_study.sandbox_probe"])
            os.environ["FIN_STUDY_REQUIRE_SANDBOX"] = "1"
            run(["-m", "pytest", "-q", "tests/test_audit_validity.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            run(["-m", "benchmarks.agent_study.audit_validity", "--seed", str(job["seed"]),
                 "--output", str(work / "results/audit-validity")])
        elif job["study"] == "validation":
            from benchmarks.fly_reuse.verify_snapshot import verify_prior
            write_once(work / "results/upstream-verification.json", verify_prior(plan["fly_prior_root"]))
            run(["-m", "pytest", "-q", "tests/test_pipeline_benchmark.py", "tests/test_fly_snapshot.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}", "--basetemp", str(work / "tmp/pytest")])
        elif job["study"] == "components":
            run(["-m", "pytest", "-q", "tests/test_component_comparison.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}", "--basetemp", str(work / "tmp/pytest")])
            run(["-m", "benchmarks.library_workflows.component_comparison",
                 "--output", str(work / "results/components")])
        elif job["study"] == "pipeline":
            run(["-m", "pytest", "-q", "tests/test_pipeline_benchmark.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            run(["-m", "benchmarks.library_workflows.pipeline", "--seed", str(job["seed"]),
                 "--output", str(work / "results/pipeline")])
        elif job["study"] == "routing":
            os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
            run(["-m", "benchmarks.rag_jev.routing", "--output", str(work / "results/routing")])
        elif job["study"] == "retrieval":
            run(["-m", "pytest", "-q", "tests/test_retrieval_downstream.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
            for seed in (11, 23, 37):
                run(["-m", "benchmarks.rag_jev.downstream", "--seed", str(seed),
                     "--output", str(work / f"results/retrieval-s{seed}")])
        elif job["study"] == "matched-repair":
            run(["-m", "pytest", "-q", "tests/test_matched_repair.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            run(["-m", "benchmarks.agent_study.matched_repair", "--seed", str(job["seed"]),
                 "--output", str(work / "results/matched-repair"),
                 "--model", job.get("model", plan["model"]),
                 "--revision", job.get("revision", plan["revision"])])
        elif job["study"] == "feedback":
            run(["-m", "pytest", "-q", "tests/test_feedback_ablation.py",
                 "-o", f"cache_dir={work / 'cache/pytest'}",
                 "--basetemp", str(work / "tmp/pytest")])
            run(["-m", "benchmarks.agent_study.feedback_ablation", "--seed", str(job["seed"]),
                 "--output", str(work / "results/feedback")])
        elif job["study"] == "agent":
            output = work / "results/matrix"
            run(["-m", "benchmarks.agent_study.run_matrix", "--model", plan["model"],
                 "--revision", plan["revision"], "--output", str(output),
                 "--seeds", str(job["seed"]), "--repetitions", "1",
                 "--max-turns", "16", "--max-tokens", "2048"])
            run(["-m", "benchmarks.agent_study.summarize_matrix", str(output),
                 "--output", str(work / "results/summary.json")])
        elif job["study"] == "temporal":
            run(["-m", "benchmarks.library_workflows.temporal", "--output", str(work / "results/temporal")])
        elif job["study"] == "fly-gate":
            prior = plan["fly_prior_root"]
            os.environ["FLY_REUSE_UPSTREAM"] = prior + "/upstream"
            from benchmarks.fly_reuse.verify_snapshot import verify_prior
            write_once(work / "results/upstream-verification.json", verify_prior(prior))
            run(["-m", "benchmarks.fly_reuse.run", "--upstream", prior + "/upstream",
                 "--data", prior + "/data/kraken-daily.json", "--output", str(work / "results/paired"),
                 "--mode", "compact", "--paired-gate"])
        elif job["study"] == "rag-jev":
            key_file = root / "private/jev-key"
            if key_file.is_file():
                if key_file.resolve() != key_file:
                    raise ValueError("Jev credential path redirected")
                os.environ["TYPESAFE_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
                key_file.unlink()
            prepared, inferred = work / "results/prepared", work / "results/inference"
            run(["-m", "benchmarks.rag_jev.run", "prepare", "--dataset", "benchmarks/rag_jev/development.json",
                 "--output", str(prepared), "--top-k", "3", "--fetch-k", "8", "--max-context-chars", "4000"])
            try:
                run(["-m", "benchmarks.rag_jev.run", "infer", "--prepared", str(prepared),
                     "--output", str(inferred), "--allow-network", "--model", "jev-1.13.0", "--max-requests", "6"])
            finally:
                if (inferred / "receipt.json").exists():
                    run(["-m", "benchmarks.rag_jev.run", "score", "--prepared", str(prepared),
                         "--run", str(inferred), "--qrels", "benchmarks/rag_jev/development-qrels.json",
                         "--output", str(work / "results/score.json")])
        else:
            raise ValueError("unsupported frozen study")
        status = "completed"
    finally:
        write_once(work / "completion.json", dict(status=status,
            slurm_job_id=os.environ["SLURM_JOB_ID"], elapsed_seconds=time.monotonic() - started,
            commands=commands, interpretation="Completion is not evidence of an improvement; inspect scores."))


if __name__ == "__main__":
    main()
