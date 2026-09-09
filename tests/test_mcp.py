"""fin_skills.mcp - the Model Context Protocol server, and its optional dependency.

The SDK is an EXTRA. The property that matters most here is therefore the negative one:
`import fin_skills.mcp` must work with `mcp` absent, and `python -m fin_skills.mcp` must
then say exactly what to install and exit non-zero - not traceback, not hang, not start a
half-server. Those tests run everywhere. The live-server tests need the real SDK and skip
without it.

Run:  python -m pytest tests/test_mcp.py -q
"""
from __future__ import annotations

import json
import socket
import sys

import pytest

from conftest import requires
import fin_skills.mcp as mcp_server
from fin_skills.tools import list_tools, mcp_tools
from fin_skills.tools.schema import excluded

# Captured at import, BEFORE conftest's autouse no-network fixture replaces it.
_REAL_CONNECT = socket.socket.connect


@pytest.fixture
def loopback_allowed(monkeypatch):
    """Let asyncio build its event loop without opening the suite to the internet.

    On Windows the proactor loop's self-pipe is a `socket.socketpair()`, which connects to
    127.0.0.1 - and conftest's no-network guard refuses every connect(). Narrow the guard
    to loopback for this test instead of removing it: a real outbound call still fails.
    """
    def connect(sock, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in ("127.0.0.1", "::1", "localhost"):
            return _REAL_CONNECT(sock, address, *args, **kwargs)
        raise RuntimeError("network access is disabled in the fin_skills test suite")

    monkeypatch.setattr(socket.socket, "connect", connect)


@pytest.fixture
def no_sdk(monkeypatch):
    """Make `import mcp` fail, whether or not the SDK is installed in this environment."""
    for name in [n for n in list(sys.modules) if n == "mcp" or n.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mcp", None)               # `import mcp` -> ImportError
    return mcp_server


# ============================================================ the module without the SDK
def test_the_module_imports_and_lists_tools_without_the_sdk(no_sdk):
    assert no_sdk.have_sdk() is False
    defs = no_sdk.tool_definitions()
    assert defs == mcp_tools() and len(defs) == len(list_tools())
    assert json.loads(json.dumps(defs, allow_nan=False)) == defs


def test_the_install_hint_names_the_extra_and_the_package():
    assert 'pip install "fin-skills[mcp]"' in mcp_server.INSTALL_HINT
    assert f'"{mcp_server.SDK_PACKAGE}>={mcp_server.SDK_MIN}"' in mcp_server.INSTALL_HINT
    assert mcp_server.INSTALL_HINT.isascii()
    assert "python -m fin_skills.tools" in mcp_server.INSTALL_HINT, \
        "a user without the SDK still needs a way to get the schemas"


def test_running_the_server_without_the_sdk_exits_non_zero_with_the_install_line(
        no_sdk, capsys):
    rc = no_sdk.main([])
    err = capsys.readouterr().err
    assert rc != 0
    assert 'pip install "fin-skills[mcp]"' in err
    assert "cannot start" in err


def test_check_reports_the_missing_sdk(no_sdk, capsys):
    assert no_sdk.main(["--check"]) == 1
    assert "is NOT installed" in capsys.readouterr().err


def test_build_server_raises_importerror_carrying_the_install_line(no_sdk):
    with pytest.raises(ImportError, match=r"fin-skills\[mcp\]"):
        no_sdk.build_server()
    with pytest.raises(ImportError, match=r"fin-skills\[mcp\]"):
        no_sdk.serve()


def test_help_and_json_work_without_the_sdk(no_sdk, capsys):
    with pytest.raises(SystemExit) as exc:
        no_sdk.main(["--help"])
    assert exc.value.code == 0
    assert "stdio" in capsys.readouterr().out

    assert no_sdk.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == mcp_tools()

    assert no_sdk.main(["--list-tools"]) == 0
    out = capsys.readouterr().out
    assert f"{len(list_tools())} MCP tools, {len(excluded())} guards excluded" in out
    for tool in list_tools():
        assert tool["name"] in out


# ======================================================== the payload the spec asks for
def test_tool_definitions_are_the_shape_the_spec_names():
    """MCP 2026-07-28 tools/list: name, description, inputSchema; object at the root."""
    for tool in mcp_server.tool_definitions():
        assert set(tool) == {"name", "description", "inputSchema"}
        assert 1 <= len(tool["name"]) <= 128
        assert tool["name"].replace("_", "").replace("-", "").replace(".", "").isalnum()
        assert tool["description"] and tool["description"].isascii()
        assert tool["inputSchema"]["type"] == "object"
        assert "$schema" not in tool["inputSchema"], "no $schema means 2020-12, the default"


def test_the_server_name_and_version_are_the_packages():
    import fin_skills
    assert mcp_server.SERVER_NAME == "fin-skills"
    assert mcp_server.sdk_version() is None or isinstance(mcp_server.sdk_version(), str)
    assert fin_skills.__version__


# =========================================================== the live server (needs mcp)
@requires("mcp")
def test_the_server_builds_against_the_real_sdk():
    server = mcp_server.build_server()
    assert type(server).__name__ == "Server"
    assert server.create_initialization_options() is not None


@requires("mcp")
def test_a_client_can_list_and_call_over_the_real_protocol(loopback_allowed):
    """Drive the server through the SDK's own in-process client - the real round trip."""
    import asyncio

    from mcp import Client

    server = mcp_server.build_server()

    async def exercise():
        async with Client(server) as client:
            listed = await client.list_tools()
            assert {t.name for t in listed.tools} == {t["name"] for t in list_tools()}

            ok = await client.call_tool("bundle_coverage",
                                        {"slots": ["returns", "turnover"]})
            assert not ok.is_error
            assert "cost_curve" in ok.structured_content["ready_over_json"]

            passed = await client.call_tool(
                "check_rf_convention",
                {"returns": {"values": [0.01, -0.005, 0.002, 0.004] * 50}, "rf": 0.05})
            assert not passed.is_error and passed.structured_content["passed"] is True

            failed = await client.call_tool(          # a real edge of 0.5 bp, 200% turnover
                "check_cost_curve",
                {"returns": {"values": [0.01, -0.0099] * 100},
                 "turnover": 2.0, "cost_bps": 10.0})
            # a FAILED CHECK is a normal result, never a protocol or tool error
            assert not failed.is_error
            assert failed.structured_content["passed"] is False
            assert failed.structured_content["findings"]

            # a BAD ARGUMENT is a tool-execution error the model can read and correct
            bad = await client.call_tool("check_cost_curve", {"returns": {"vals": [1.0]},
                                                              "turnover": 0.1})
            assert bad.is_error and "vals" in bad.content[0].text

            unknown = await client.call_tool("no_such_tool", {})
            assert unknown.is_error and "no tool named" in unknown.content[0].text

    asyncio.run(exercise())
