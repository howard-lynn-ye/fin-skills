"""Execute a generated retrieval implementation in a fresh confined process.

This worker receives inputs, never expected answers. It is not an anti-tamper evaluator.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys


def execute(source, request, state, *, library=False, confined=True):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        if not library and any(n.split(".")[0] == "fin_skills" for n in names):
            raise ValueError("fin_skills is unavailable in the components condition")
    compiled = compile(tree, "submission.py", "exec")
    if confined:
        from benchmarks.agent_study.linux_sandbox import confine
        root = Path(__file__).resolve().parents[2]
        readonly = [Path(sys.prefix), Path(sys.base_prefix), Path("/usr"), Path("/lib"),
                    Path("/lib64"), Path("/etc/ld.so.cache")]
        shared = os.environ.get("FIN_STUDY_SHARED_RUNTIME")
        if shared:
            readonly.append(Path(shared))
        if library:
            readonly.append(root / "fin_skills")
        confine(state.parent, readonly)
    namespace = {"__name__": "submission", "__file__": "submission.py"}
    exec(compiled, namespace)
    return namespace["solve"](request, str(state))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--library", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    args.state.mkdir(exist_ok=True)
    try:
        value = execute(payload["source"], payload["request"], args.state, library=args.library)
        result = {"status": "executed", "value": value}
        encoded = json.dumps(result, allow_nan=False)
    except Exception as exc:
        encoded = json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
