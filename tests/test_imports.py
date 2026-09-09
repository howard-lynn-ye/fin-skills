"""Every generated module imports cleanly, silently, offline, without writing, and is tested.

Coverage is enforced mechanically: the module list comes from the package itself (each
namespace's __all__), is cross-checked against plugins/*/skills/*/scripts/*.py when the
source tree is present, and every module must have its own tests/test_<ns>_<module>.py.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

import pytest

from _helpers import (GENERATED_HEADER, PACKAGE_ROOT, REPO_ROOT, TESTS_DIR,
                      expected_test_file, generated_modules, generated_namespaces,
                      module_source_path)

MODULES = generated_modules()
NAMESPACES = generated_namespaces()


def _namespace_of(plugin: str) -> str:
    # the rule in scripts/build_package.py: drop the fin- prefix, dashes become underscores
    n = plugin[4:] if plugin.startswith("fin-") else plugin
    return n.replace("-", "_")


def test_generated_modules_were_discovered():
    assert len(NAMESPACES) >= 1
    assert len(MODULES) >= 1
    assert "api" not in NAMESPACES, "the hand-written api/ package is not a generated namespace"


@pytest.mark.skipif(not (REPO_ROOT / "plugins").is_dir(),
                    reason="skill source tree (plugins/) not present in this checkout")
def test_module_list_matches_the_skill_scripts():
    expected = set()
    for script in (REPO_ROOT / "plugins").glob("*/skills/*/scripts/*.py"):
        plugin = script.parts[-5]
        expected.add(f"fin_skills.{_namespace_of(plugin)}.{script.stem}")
    assert set(MODULES) == expected


@pytest.mark.parametrize("qualname", MODULES)
def test_every_generated_module_has_a_test_file(qualname):
    path = expected_test_file(qualname)
    assert path.is_file(), f"missing {path.name}"


@pytest.mark.parametrize("ns", NAMESPACES)
def test_namespace_all_is_sorted_and_matches_the_directory(ns):
    pkg = importlib.import_module(f"fin_skills.{ns}")
    listed = list(pkg.__all__)
    assert listed == sorted(listed)
    on_disk = sorted(p.stem for p in (PACKAGE_ROOT / ns).glob("*.py") if p.stem != "__init__")
    assert on_disk == listed


@pytest.mark.parametrize("qualname", MODULES)
def test_module_is_a_generated_copy_with_a_docstring(qualname):
    mod = importlib.import_module(qualname)
    assert mod.__doc__ and mod.__doc__.strip()
    src = module_source_path(qualname)
    assert src.is_file()
    first_line = src.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith(GENERATED_HEADER)


# ------------------------------------------------------------ fresh-interpreter import probe
@pytest.fixture(scope="session")
def import_report(tmp_path_factory):
    """One subprocess imports every module with stdout/stderr captured and network refused."""
    base = tmp_path_factory.mktemp("import_probe")
    cwd = base / "cwd"
    cwd.mkdir()
    report = base / "report.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    # The probe runs from an empty cwd, so nothing puts THIS checkout on its sys.path. Without
    # the line below it imports whatever `fin_skills` the interpreter finds - and an editable
    # install (pip install -e) points at the tree it was installed from, which in a git
    # worktree is a DIFFERENT checkout. The probe then reports on modules that are not the
    # ones these tests enumerated: a module added in a worktree shows up as ModuleNotFoundError
    # and one deleted there still passes. Pin it to the package the rest of the file is about.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PACKAGE_ROOT.parent), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    cmd = [sys.executable, str(TESTS_DIR / "_import_probe.py"), str(report),
           str(PACKAGE_ROOT), *MODULES]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.parametrize("qualname", MODULES)
def test_import_is_silent_and_offline(import_report, qualname):
    rec = import_report["modules"][qualname]
    assert rec["error"] == "", rec["error"]
    assert rec["stdout"] == "", f"importing printed to stdout: {rec['stdout']!r}"
    assert "Traceback" not in rec["stderr"], rec["stderr"]


def test_importing_writes_no_files(import_report):
    assert import_report["new_files"] == []


def test_importing_in_process_is_offline_too():
    # the autouse fixture refuses sockets; a module reaching out at import time would raise
    for qualname in MODULES:
        importlib.import_module(qualname)
