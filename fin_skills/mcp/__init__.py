"""A Model Context Protocol server over `fin_skills.tools` - stdio, exactly those tools.

    pip install "fin-skills[mcp]"
    python -m fin_skills.mcp          # or the console script: fin-skills-mcp

Then point any MCP client at that command. In Claude Code:

    claude mcp add fin-skills -- python -m fin_skills.mcp

Verified at the primary sources on 2026-09-09:

  * The current MCP protocol revision is **2026-07-28**
    (modelcontextprotocol.io/specification/versioning). A server declares
    `capabilities.tools`, answers `tools/list` with `{name, description, inputSchema}`
    entries and `tools/call` with `content` plus optional `structuredContent`.
    `inputSchema` is JSON Schema, defaulting to draft 2020-12 when no `$schema` key is
    present, with `type: "object"` required at the root - which is what
    `fin_skills.tools.export.mcp_tools()` emits.
  * The spec separates PROTOCOL errors (unknown tool, malformed request - JSON-RPC errors)
    from TOOL EXECUTION errors (bad arguments, business-logic failures - a normal result
    with `isError: true`), and says clients SHOULD hand the second kind to the model so it
    can self-correct. So every TypeError from `call_tool` - the ones that name the field -
    comes back as `isError: true`, never as a protocol error. A guard that FAILS its check
    is not an error at all: it is a result with `passed: false`.
  * The sanctioned Python implementation is the official SDK, `mcp` on PyPI
    (github.com/modelcontextprotocol/python-sdk), MIT licensed, latest 2.2.0, requiring
    Python >= 3.10. This module targets the v2 line's low-level `Server`, because the
    schemas here are derived at runtime and the high-level `MCPServer` derives its own
    from Python type hints instead.

The SDK is an OPTIONAL extra. `import fin_skills.mcp` works without it - the import is
deferred - and `python -m fin_skills.mcp` prints the install line and exits non-zero. The
base package still depends only on numpy, pandas and scipy.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fin_skills import __version__
from fin_skills.tools import call_tool, list_tools, mcp_tools
from fin_skills.tools.schema import excluded

SERVER_NAME = "fin-skills"
SDK_PACKAGE = "mcp"
SDK_MIN = "2.0"
INSTALL_HINT = (
    "the Model Context Protocol SDK is not installed.\n"
    '  pip install "fin-skills[mcp]"          # or: pip install "mcp>=2.0"\n'
    "The SDK is an optional extra - everything else in fin_skills works without it.\n"
    "Without a server you can still hand the same tools to an agent directly:\n"
    "  python -m fin_skills.tools --json --format anthropic"
)


def sdk_version() -> str | None:
    """The installed `mcp` version, or None when the SDK is absent."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:                                          # pragma: no cover
        return None
    try:
        return version(SDK_PACKAGE)
    except PackageNotFoundError:
        return None


def have_sdk() -> bool:
    """True when the official MCP Python SDK can be imported."""
    import importlib.util
    try:
        return importlib.util.find_spec("mcp.server") is not None
    except (ImportError, ValueError):                            # pragma: no cover
        return False


def _require_sdk():
    """Import the SDK pieces, or raise ImportError carrying the install line."""
    try:
        from mcp.server import Server                            # noqa: PLC0415
        from mcp.server.stdio import stdio_server                # noqa: PLC0415
        import mcp.types as types                                # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc
    return Server, stdio_server, types


def tool_definitions() -> list[dict[str, Any]]:
    """The `tools/list` payload as plain dicts - readable without the SDK installed."""
    return mcp_tools()


# ----------------------------------------------------------------------------- handlers
def _dumps(payload: Any) -> str:
    # allow_nan=False: NaN and Infinity are not JSON, and payloads.encode has already
    # turned them into null / "Infinity", so a failure here is a real bug, not a number.
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, default=str)


def build_server():
    """The configured low-level MCP `Server`. Raises ImportError if the SDK is absent."""
    Server, _stdio, types = _require_sdk()
    tools = [types.Tool.model_validate(t) for t in tool_definitions()]

    async def on_list_tools(_ctx, _params):
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(_ctx, params):
        # The spec's two error channels: a TypeError here is a TOOL EXECUTION error
        # (bad arguments the model can fix), so it comes back as isError, not as a
        # JSON-RPC error. A low-level handler that raises would produce the latter.
        try:
            result = call_tool(params.name, params.arguments or {})
        except TypeError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))], is_error=True)
        except Exception as exc:                                 # noqa: BLE001
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text", text=f"{type(exc).__name__}: {exc}")], is_error=True)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=_dumps(result))],
            structured_content=result, is_error=False)

    try:
        return Server(SERVER_NAME, version=__version__,
                      on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    except TypeError as exc:                                     # an SDK older than v2
        raise ImportError(
            f"the installed {SDK_PACKAGE} {sdk_version()} does not take constructor "
            f"handlers; this server targets the v2 low-level API. "
            f'pip install "{SDK_PACKAGE}>={SDK_MIN}"') from exc


def serve() -> None:
    """Run the server on stdio until the client disconnects."""
    import asyncio

    _Server, stdio_server, _types = _require_sdk()
    server = build_server()

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())


# ---------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m fin_skills.mcp",
        description=f"Serve the fin-skills guards as MCP tools over stdio "
                    f"(requires the optional extra: pip install \"fin-skills[mcp]\").")
    p.add_argument("--list-tools", action="store_true",
                   help="print the tool names and exit (no SDK needed)")
    p.add_argument("--json", action="store_true",
                   help="print the tools/list payload as JSON and exit (no SDK needed)")
    p.add_argument("--check", action="store_true",
                   help="report whether the SDK is installed and exit")
    p.add_argument("--version", action="version",
                   version=f"fin-skills {__version__}")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.json:
        print(_dumps(tool_definitions()))
        return 0
    if args.list_tools:
        tools = list_tools()
        print(f"{len(tools)} MCP tools, {len(excluded())} guards excluded "
              f"(python -m fin_skills.tools --excluded says why)")
        for t in tools:
            print("  " + t["name"])
        return 0
    if args.check:
        ver = sdk_version()
        if have_sdk():
            print(f"{SDK_PACKAGE} {ver or 'unknown version'} is installed; "
                  f"{len(tool_definitions())} tools ready")
            return 0
        print(f"{SDK_PACKAGE} is NOT installed", file=sys.stderr)
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    if not have_sdk():
        print(f"{SERVER_NAME}: cannot start - {INSTALL_HINT}", file=sys.stderr)
        return 1
    try:
        serve()
    except ImportError as exc:
        print(f"{SERVER_NAME}: cannot start - {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:                                    # pragma: no cover
        return 130
    return 0


__all__ = ["INSTALL_HINT", "SDK_MIN", "SDK_PACKAGE", "SERVER_NAME", "build_server",
           "have_sdk", "main", "sdk_version", "serve", "tool_definitions"]
