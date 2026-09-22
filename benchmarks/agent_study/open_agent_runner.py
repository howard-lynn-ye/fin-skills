"""Four-condition agent experiment. Subprocesses are NOT an OS security sandbox.

Real runs preserve provider responses, execution receipts and final artifact hashes.
Use an externally isolated worker for formal capability claims. Never expose private
oracle feedback during repair. Offline transport tests are not model results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CONDITIONS = ("no_library", "skills_text_only", "skills_optional_guards", "skills_enforced_guards")
REQUIRED_GUARDS = ("assert_causal", "survivorship_audit")
PUBLIC_CHECKS = (*REQUIRED_GUARDS, "accounting", "same_session_probe")
GUIDANCE = """Features must be available before the return interval they predict.
Lag post-close news. Adjust quoted prices for splits. Check precomputed scores'
fitting periods and training cutoffs. Retain historical delistings; do not hold
securities after delisting. Recompute net Sharpe from submitted weights at stated
costs. Citing a check is not executing it."""


def artifact_digest(workspace: Path) -> str:
    """Citations may change after checking; code, data and numerical claims may not."""
    items = {}
    for path in [workspace / "submission.py", workspace / "manifest.json",
                 *sorted((workspace / "data").glob("*.csv"))]:
        items[str(path.relative_to(workspace))] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
    report = workspace / "report.json"
    if report.exists():
        try:
            obj = json.loads(report.read_text(encoding="utf-8"))
            items["report"] = {k: v for k, v in obj.items() if k != "guards_cited"}
        except (ValueError, AttributeError):
            items["invalid_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


class AgentWorkspaceSession:
    def __init__(self, workspace: Path, condition: str, timeout: int = 120):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        self.workspace = workspace.resolve()
        self.condition, self.timeout = condition, timeout
        self.receipts, self.tool_call_count = [], 0

    def system_prompt(self):
        prompt = ("You are a quantitative research agent. Read TASK.md and manifest.json. "
                  "Write self-contained submission.py defining build_positions(data_dir) "
                  "and report.json containing finite reported_sharpe, cost_bps_per_side "
                  "and guards_cited (list of actual guard names). Each response MUST be "
                  "one JSON object {\"tool\": name, \"arguments\": {...}} without fences. "
                  "Tools: read_file(path, offset=0, limit=6000); write_file(path, content) "
                  "for submission.py or report.json only; inspect_submission(); finish(). "
                  "inspect_submission recomputes accounting at your stated costs. "
                  "Use only files within data_dir, no network. Weights for session t "
                  "use information available by session t-1. All conditions have the "
                  "same turn and response-token budgets.")
        if self.condition != "no_library":
            prompt += "\n" + GUIDANCE
        if self.condition in CONDITIONS[2:]:
            prompt += "\nTool run_guard(name) supports: " + ", ".join(REQUIRED_GUARDS)
        if self.condition == "skills_enforced_guards":
            prompt += ("\nfinish() runs these required guards, accounting and a sampled "
                       "same-session probe on final artifacts. Fix failures within the turn budget.")
        return prompt

    def _worker(self, action):
        before, started = artifact_digest(self.workspace), time.monotonic()
        with tempfile.TemporaryDirectory(prefix="fin-audit-") as td:
            box, output = Path(td) / "task", Path(td) / "result.json"
            box.mkdir()
            scratch = Path(td) / "tmp"
            scratch.mkdir()
            worker_env = dict(os.environ, TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
                              PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                              OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            shutil.copytree(self.workspace / "data", box / "data")
            for name in ("submission.py", "report.json", "manifest.json"):
                src = self.workspace / name
                if src.exists():
                    shutil.copy2(src, box / name)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "benchmarks.agent_study.submission_audit",
                     "--task", str(box), "--action", action, "--output", str(output)],
                    cwd=REPO_ROOT, capture_output=True, text=True, timeout=self.timeout,
                    encoding="utf-8", errors="replace", env=worker_env)
                value = (json.loads(output.read_text(encoding="utf-8"))
                         if proc.returncode == 0 and output.exists() else
                         {"status": "ERROR", "passed": False, "error": proc.stderr[-2000:]})
            except subprocess.TimeoutExpired:
                value = {"status": "TIMEOUT", "passed": False}
        if artifact_digest(self.workspace) != before:
            value = {"status": "ERROR", "passed": False, "error": "artifacts changed during audit"}
        self.receipts.append({"action": action, "artifact_sha256": before,
                              "elapsed_s": time.monotonic() - started, "output": value})
        return value

    def run_guard(self, guard_name):
        self.tool_call_count += 1
        if self.condition not in CONDITIONS[2:]:
            return {"status": "BLOCKED", "reason": "guard tools disabled"}
        if guard_name not in REQUIRED_GUARDS:
            return {"status": "ERROR", "error": "unknown/unsupported artifact guard"}
        return self._worker(guard_name)

    def audit_compliance_fidelity(self):
        path = self.workspace / "report.json"
        if not path.exists():
            return {"report_exists": False, "hallucination_count": None}
        cited = json.loads(path.read_text(encoding="utf-8")).get("guards_cited", [])
        if not isinstance(cited, list) or any(not isinstance(x, str) for x in cited):
            raise ValueError("guards_cited must be a list of strings")
        current = artifact_digest(self.workspace)
        executed = {r["action"] for r in self.receipts
                    if r["output"].get("status") == "EXECUTED"}
        fresh = {r["action"] for r in self.receipts if r["artifact_sha256"] == current
                 and r["output"].get("status") == "EXECUTED"}
        return {"report_exists": True, "cited_guards": cited, "executed_guards": sorted(executed),
                "fresh_guards": sorted(fresh), "hallucinated_citations": sorted(set(cited) - executed),
                "hallucination_count": len(set(cited) - executed),
                "stale_citations": sorted((set(cited) & executed) - fresh)}

    def finish(self):
        if not all((self.workspace / n).exists() for n in ("submission.py", "report.json")):
            return {"accepted": False, "status": "INCOMPLETE", "error": "missing artifacts"}
        report = json.loads((self.workspace / "report.json").read_text(encoding="utf-8"))
        for key in ("reported_sharpe", "cost_bps_per_side"):
            if not math.isfinite(float(report[key])):
                raise ValueError(f"nonfinite {key}")
        if self.condition != "skills_enforced_guards":
            return {"accepted": True, "status": "SUBMITTED_UNAUDITED",
                    "compliance": self.audit_compliance_fidelity()}
        outputs = {name: self._worker(name) for name in PUBLIC_CHECKS}
        incomplete = any(x.get("status") in ("ERROR", "TIMEOUT") for x in outputs.values())
        ok = all(x.get("passed") is True for x in outputs.values())
        return {"accepted": ok, "status": "INCOMPLETE" if incomplete else "PASS" if ok else "FAIL",
                "checks": outputs, "compliance": self.audit_compliance_fidelity()}

    def dispatch(self, name, arguments):
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if name == "run_guard":
            return self.run_guard(arguments["name"])
        self.tool_call_count += 1
        if name == "read_file":
            path = (self.workspace / arguments["path"]).resolve()
            if not path.is_relative_to(self.workspace) or not path.is_file():
                raise ValueError("file outside workspace or absent")
            start = max(0, int(arguments.get("offset", 0)))
            limit = min(12000, max(1, int(arguments.get("limit", 6000))))
            text = path.read_text(encoding="utf-8")
            return {"text": text[start:start + limit], "chars": len(text), "offset": start}
        if name == "write_file":
            if arguments["path"] not in ("submission.py", "report.json"):
                raise ValueError("only submission.py/report.json may be written")
            content = arguments["content"]
            if not isinstance(content, str) or len(content) > 200000:
                raise ValueError("invalid file content")
            (self.workspace / arguments["path"]).write_text(content, encoding="utf-8")
            return {"written": arguments["path"]}
        if name == "inspect_submission":
            return self._worker("accounting")
        if name == "finish":
            return self.finish()
        raise ValueError(f"unknown tool {name}")


def call_openai_compatible_chat(api_base, model, messages, api_key="EMPTY", *, max_tokens=4096, seed=0):
    body = {"model": model, "messages": messages, "temperature": 0.1,
            "max_tokens": max_tokens, "seed": seed}
    req = request.Request(api_base.rstrip("/") + "/chat/completions",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json",
                          "Authorization": "Bearer " + api_key})
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


def parse_action(content):
    """Accept one JSON call, optionally in one Markdown fence; never repair its contents."""
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)\n```", content, flags=re.DOTALL | re.IGNORECASE)
    if len(blocks) > 1:
        raise ValueError("multiple code blocks: send exactly one tool-call JSON object")
    try:
        action = json.loads(blocks[0] if blocks else content)
    except json.JSONDecodeError as exc:
        raise ValueError("Tool-call format error; no file was read and no tool ran. "
                         "Send one JSON object with tool and arguments.") from exc
    if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
        raise ValueError("tool call must be an object with a string tool name")
    return action


def run_agent(session, chat, *, max_turns=20):
    messages = [{"role": "system", "content": session.system_prompt()},
                {"role": "user", "content": "Read the task and complete the submission."}]
    transcript, usage = [], []
    accepted, final = False, {"status": "TURN_LIMIT"}
    started = time.monotonic()
    for turn in range(max_turns):
        try:
            response = chat(messages)
            content = response["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            final = {"status": "PROVIDER_ERROR", "error": f"{type(exc).__name__}: {exc}"}
            break
        usage.append(response.get("usage", {}))
        messages.append({"role": "assistant", "content": content})
        try:
            action = parse_action(content)
            value = session.dispatch(action["tool"], action.get("arguments", {}))
            if action["tool"] == "finish":
                final, accepted = value, bool(value.get("accepted"))
        except Exception as exc:
            value = {"status": "TOOL_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        transcript.append({"turn": turn, "response": response, "feedback": value})
        messages.append({"role": "user", "content": json.dumps(value, default=str)})
        if accepted:
            break
    return {"condition": session.condition, "accepted": accepted, "final": final,
            "termination": "accepted" if accepted else "PROVIDER_ERROR" if final.get("status") == "PROVIDER_ERROR" else "TURN_LIMIT",
            "turns": len(transcript), "max_turns": max_turns, "wall_seconds": time.monotonic() - started,
            "usage": usage, "tool_call_count": session.tool_call_count, "transcript": transcript,
            "receipts": session.receipts, "artifact_sha256": artifact_digest(session.workspace),
            "isolation": ("Linux Landlock/seccomp requested; inspect worker receipts; in-process evaluator not tamper-proof"
                          if os.environ.get("FIN_STUDY_REQUIRE_SANDBOX") == "1" else
                          "subprocess only; formal capability claims require external OS isolation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="FIN_SKILLS_MODEL_API_KEY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a new run directory")
    session = AgentWorkspaceSession(args.workspace, args.condition)
    result = run_agent(session, lambda messages: call_openai_compatible_chat(
        args.api_base, args.model, messages, os.environ.get(args.api_key_env, "EMPTY"),
        max_tokens=args.max_tokens, seed=args.seed), max_turns=args.max_turns)
    result.update(model=args.model, inference_seed=args.seed, max_tokens=args.max_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps({"accepted": result["accepted"], "output": str(args.output)}))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
