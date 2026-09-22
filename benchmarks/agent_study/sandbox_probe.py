"""Run before inference: verify denied outside writes, socket creation and private reads."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

from benchmarks.agent_study.linux_sandbox import confine


def main():
    root = Path(os.environ["FIN_STUDY_ROOT"])
    outside = root / "sandbox-sentinel.txt"
    outside.write_text("private sentinel")
    if True:
        td = tempfile.mkdtemp(prefix="sandbox-probe-", dir=root / "tmp")
        writable = Path(td)
        info = confine(writable, [Path(sys.prefix), Path(sys.base_prefix), Path("/usr"),
                                  Path("/lib"), Path("/lib64")])
        checks = {}
        for label, action in {
            "outside_read_denied": lambda: outside.read_text(),
            "outside_write_denied": lambda: (root / "forbidden.txt").write_text("bad"),
            "socket_denied": lambda: socket.socket(),
        }.items():
            try:
                action()
                checks[label] = False
            except PermissionError:
                checks[label] = True
        (writable / "allowed.txt").write_text("allowed")
        checks["scratch_write_allowed"] = True
        print(json.dumps({"confinement": info, "checks": checks}), flush=True)
        if not all(checks.values()):
            raise RuntimeError("sandbox probe failed")


if __name__ == "__main__":
    main()
