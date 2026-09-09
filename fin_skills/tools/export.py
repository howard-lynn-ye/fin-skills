"""The same tools in the shape each API wants, and a `--json` CLI that prints them.

One tool table (`fin_skills.tools.runner.list_tools`), three wire shapes. They differ only
in where the schema hangs and what the key is called - verified at the primary sources on
2026-09-09:

  MCP        {"name", "title"?, "description", "inputSchema"}
             - modelcontextprotocol.io/specification/2026-07-28/server/tools
             `inputSchema` MUST be a JSON Schema object, defaulting to draft 2020-12 when
             no `$schema` is present, and `type: "object"` is required at the root.
  Anthropic  {"name", "description", "input_schema"}
             - platform.claude.com/docs/en/agents-and-tools/tool-use/overview
  OpenAI     Responses API: {"type": "function", "name", "description", "parameters"}
             Chat Completions: {"type": "function", "function": {name, description,
             parameters}}
             - developers.openai.com/api/docs/guides/function-calling

Nothing is retyped: every shape is a rename of the neutral table, so a guard added to the
registry appears in all three.

    python -m fin_skills.tools --json                    # Anthropic shape (the default)
    python -m fin_skills.tools --json --format openai    # Responses API
    python -m fin_skills.tools --json --format mcp       # MCP tools/list entries
    python -m fin_skills.tools --list                   # names and one-liners
    python -m fin_skills.tools --excluded               # the guards JSON cannot carry
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fin_skills.tools.runner import list_tools
from fin_skills.tools.schema import excluded

FORMATS = ("anthropic", "openai", "openai-chat", "mcp", "neutral")


def anthropic_tools() -> list[dict[str, Any]]:
    """The Anthropic Messages API `tools` array."""
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["input_schema"]} for t in list_tools()]


def openai_tools(chat: bool = False) -> list[dict[str, Any]]:
    """OpenAI function tools - the Responses shape, or the Chat Completions nesting."""
    out = []
    for t in list_tools():
        body = {"name": t["name"], "description": t["description"],
                "parameters": t["input_schema"]}
        out.append({"type": "function", "function": body} if chat
                   else {"type": "function", **body})
    return out


def mcp_tools() -> list[dict[str, Any]]:
    """MCP `tools/list` entries (camelCase `inputSchema`)."""
    return [{"name": t["name"], "description": t["description"],
             "inputSchema": t["input_schema"]} for t in list_tools()]


def tools_for(fmt: str) -> list[dict[str, Any]]:
    if fmt == "anthropic":
        return anthropic_tools()
    if fmt == "openai":
        return openai_tools()
    if fmt == "openai-chat":
        return openai_tools(chat=True)
    if fmt == "mcp":
        return mcp_tools()
    if fmt == "neutral":
        return list_tools()
    raise ValueError(f"unknown format {fmt!r}; one of {list(FORMATS)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m fin_skills.tools",
        description="Print the fin-skills tool definitions for your agent framework.")
    p.add_argument("--json", action="store_true",
                   help="print the tool definitions as JSON (the default action)")
    p.add_argument("--format", choices=FORMATS, default="anthropic",
                   help="wire shape (default: anthropic)")
    p.add_argument("--list", action="store_true",
                   help="print names and one-line descriptions instead of JSON")
    p.add_argument("--excluded", action="store_true",
                   help="print the guards that cannot be driven over JSON, and why")
    p.add_argument("--indent", type=int, default=2, help="JSON indent (0 for compact)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):                      # tool text is ASCII; be safe
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.excluded:
        exc = excluded()
        print(f"{len(exc)} guard(s) cannot be driven over JSON:")
        for name in sorted(exc):
            print(f"  {name} [{exc[name].skill}]: {exc[name].reason}")
        return 0
    if args.list:
        tools = list_tools()
        print(f"{len(tools)} tools ({sum(1 for t in tools if 'guard' in t)} guards, "
              f"{sum(1 for t in tools if 'guard' not in t)} catalogue), "
              f"{len(excluded())} guards excluded")
        for t in tools:
            first = t["description"].split(". ")[0].rstrip(".")
            print(f"  {t['name']:<28} {first[:96]}")
        return 0
    print(json.dumps(tools_for(args.format), indent=args.indent or None,
                     sort_keys=False, ensure_ascii=True, allow_nan=False))
    return 0


__all__ = ["FORMATS", "anthropic_tools", "main", "mcp_tools", "openai_tools", "tools_for"]


if __name__ == "__main__":                                      # pragma: no cover - CLI
    sys.exit(main())
