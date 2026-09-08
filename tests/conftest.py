"""Shared fixtures for the fin_skills test suite.

Two guards run around EVERY test:

  * no network  - socket connections, name resolution and create_connection all raise, so a
                  module that reaches for the internet fails loudly instead of hanging;
  * cwd = tmp_path - every test runs inside its own temporary directory, so a relative
                  path written by a demo lands in tmp_path and never in the repository.

`run_main` executes a module's `if __name__ == "__main__":` demo under capsys and returns
what it printed, for the modules whose documented behaviour is the demo itself.
"""
from __future__ import annotations

import importlib.util
import runpy
import socket
import warnings

import pytest


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def requires(name: str):
    """Skip marker for tests that verify a script against an optional third-party library."""
    return pytest.mark.skipif(not has_module(name), reason=f"{name} is not installed")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise RuntimeError("network access is disabled in the fin_skills test suite")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def run_main(capsys):
    """Run `python -m <module>` in-process (the module's __main__ demo) and return stdout."""

    def _run(module_name: str) -> str:
        with warnings.catch_warnings():
            # runpy warns when the module was already imported by another test in this
            # process; running an imported module as __main__ is exactly what we intend.
            warnings.filterwarnings("ignore", message=".*found in sys.modules.*",
                                    category=RuntimeWarning)
            runpy.run_module(module_name, run_name="__main__")
        return capsys.readouterr().out

    return _run
