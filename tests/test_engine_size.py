"""fin_skills/engine/ stays small enough to read in an afternoon.

The engine's pitch against zipline-reloaded is "lines you can read", so the line budget is
a claim like any other and gets a test. Two numbers, because one of them can be gamed:

  * CODE   - non-blank, non-comment, non-docstring, the cloc convention for Python.
             Budget 900. This is what "how much logic is there" means.
  * TOTAL  - every physical line. Budget 1,400, so the code budget cannot be met by
             moving logic into a docstring.

Run `python -m pytest tests/test_engine_size.py -q -s` to print the per-module table.
"""
from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pytest

import fin_skills.engine as engine

ENGINE_DIR = Path(engine.__file__).resolve().parent
CODE_BUDGET = 900
TOTAL_BUDGET = 1_400


def counts(path: Path) -> tuple[int, int]:
    """(code lines, physical lines). A string that starts a logical line is a docstring."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    skip: set[int] = set()
    at_line_start = True
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            skip.update(range(tok.start[0], tok.end[0] + 1))
        elif tok.type == tokenize.STRING and at_line_start:
            skip.update(range(tok.start[0], tok.end[0] + 1))
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            at_line_start = True
        elif tok.type != tokenize.COMMENT:
            at_line_start = False
    return sum(1 for i, ln in enumerate(lines, 1) if ln.strip() and i not in skip), len(lines)


def table() -> dict[str, tuple[int, int]]:
    return {p.name: counts(p) for p in sorted(ENGINE_DIR.glob("*.py"))}


def test_the_engine_fits_in_its_line_budget(capsys):
    rows = table()
    code = sum(c for c, _ in rows.values())
    total = sum(t for _, t in rows.values())
    with capsys.disabled():
        print(f"\n{'module':<16}{'code':>7}{'total':>8}")
        for name, (c, t) in rows.items():
            print(f"{name:<16}{c:>7}{t:>8}")
        print(f"{'TOTAL':<16}{code:>7}{total:>8}   budget {CODE_BUDGET} / {TOTAL_BUDGET}")
    assert code < CODE_BUDGET, f"{code} code lines, budget {CODE_BUDGET}"
    assert total < TOTAL_BUDGET, f"{total} physical lines, budget {TOTAL_BUDGET}"


def test_every_module_the_design_names_is_present():
    assert set(table()) == {"__init__.py", "actions.py", "costs.py", "execute.py",
                            "invariants.py", "panel.py", "result.py", "sizing.py",
                            "spec.py", "universe.py"}


def test_the_engine_imports_numpy_and_pandas_and_nothing_new():
    """No new dependency: numpy, pandas and this repo's own api/ layer. Optional third-party
    imports are fin_skills/bridges/' job, not the engine's."""
    third_party = set()
    for path in sorted(ENGINE_DIR.glob("*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and " " in stripped:
                root = stripped.split()[1].split(".")[0]
                third_party.add(root)
    allowed = {"numpy", "pandas", "fin_skills", "__future__", "dataclasses", "typing",
               "hashlib"}
    assert third_party <= allowed, sorted(third_party - allowed)


@pytest.mark.parametrize("name", sorted(p.name for p in ENGINE_DIR.glob("*.py")))
def test_every_module_documents_itself_in_ascii(name):
    text = (ENGINE_DIR / name).read_text(encoding="utf-8")
    assert text.isascii(), f"{name} carries a non-ASCII character"
    assert text.lstrip().startswith('"""'), f"{name} has no module docstring"
