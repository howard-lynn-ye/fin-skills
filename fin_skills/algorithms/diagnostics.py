"""Isolated optional-library import checks; never installs packages."""
import json
import subprocess
import sys

from .runtime import integer


def doctor(task=None, *, timeout=30, registry=None):
    if registry is None:
        from . import _DEFAULT
        registry = _DEFAULT
    timeout = integer(timeout, "timeout", maximum=120)
    backends = {a.library: a.module for a in registry.algorithms(task) if a.module}
    results = []
    code = "import importlib,sys; importlib.import_module(sys.argv[1]); print('IMPORT_OK')"
    for library, module in sorted(backends.items()):
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code, module],
                                  capture_output=True, text=True, errors="replace", timeout=timeout)
            results.append({"library": library, "module": module,
                "status": "importable" if proc.returncode == 0 else "import_failed",
                "returncode": proc.returncode, "detail": proc.stderr[-2000:]})
        except subprocess.TimeoutExpired:
            results.append({"library": library, "module": module, "status": "timeout"})
    return {"python": sys.executable, "backends": results,
            "scope": "isolated imports only; solver feasibility and algorithm performance are not checked"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    result = doctor(args.task, timeout=args.timeout)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if all(r["status"] == "importable" for r in result["backends"]) else 1
