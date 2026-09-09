"""`python -m fin_skills.bridges audit` - the licence table, and what is installed.

Imports no bridged library: everything here goes through find_spec.
"""
from __future__ import annotations

import sys

from fin_skills.bridges import audit_text, available


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "audit"
    if cmd == "audit":
        print(audit_text())
        return 0
    if cmd == "available":
        for pip, ok in available().items():
            print(f"  {'yes' if ok else ' no'}  {pip}")
        return 0
    print(f"usage: python -m fin_skills.bridges [audit|available]  (got {cmd!r})")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
