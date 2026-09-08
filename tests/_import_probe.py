"""Import generated modules in a FRESH interpreter and record what importing did.

tests/test_imports.py runs this in a subprocess, because the only honest test of "importing
prints nothing" is a first import: inside the pytest process most modules are already
cached in sys.modules by the time any test looks.

Usage:  python _import_probe.py <report.json> <package_root> <module> [<module> ...]

Per module the report records captured stdout, captured stderr and any traceback. It also
records every file that appeared under the package root or the working directory while the
imports ran. Network access is refused at the socket level before the first import.
"""
from __future__ import annotations

import importlib
import io
import json
import socket
import sys
import traceback
from pathlib import Path


def _refuse(*_args, **_kwargs):
    raise RuntimeError("network access is disabled while importing fin_skills modules")


def _snapshot(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def main() -> int:
    report_path, pkg_root, *modules = sys.argv[1:]
    root = Path(pkg_root)
    cwd = Path.cwd()

    socket.socket.connect = _refuse            # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse         # type: ignore[method-assign]
    socket.create_connection = _refuse         # type: ignore[assignment]
    socket.getaddrinfo = _refuse               # type: ignore[assignment]

    before_pkg, before_cwd = _snapshot(root), _snapshot(cwd)
    results: dict[str, dict[str, str]] = {}
    for name in modules:
        out, err = io.StringIO(), io.StringIO()
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        error = ""
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001 - the traceback IS the report
            error = traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = real_out, real_err
        results[name] = {"stdout": out.getvalue(), "stderr": err.getvalue(), "error": error}

    new_files = sorted((_snapshot(root) - before_pkg)
                       | {"cwd:" + f for f in _snapshot(cwd) - before_cwd})
    Path(report_path).write_text(json.dumps({"modules": results, "new_files": new_files}),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
