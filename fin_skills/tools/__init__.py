"""fin_skills.tools - the guards as TOOLS an LLM agent can call, transport-independent.

`fin_skills.api` makes the guards importable. This layer makes them callable from outside
Python: a JSON Schema per guard, a JSON payload convention for the pandas objects they
want, and one `call_tool(name, arguments)` entry point that returns a JSON-serializable
verdict. `fin_skills.mcp` puts it on a Model Context Protocol stdio transport;
`fin_skills.tools.export` prints the same definitions in the Anthropic Messages API and
OpenAI function shapes for anything else.

    from fin_skills.tools import list_tools, call_tool

    list_tools()                                   # every tool: name, description, schema
    call_tool("bundle_coverage", {"slots": ["returns", "turnover"]})
    call_tool("check_cost_curve", {"returns": {"index": [...], "values": [...]},
                                   "turnover": 0.1, "cost_bps": 10})

Four rules hold here, and the tests assert all four:

  * **Nothing is written down twice.** Every schema is derived from the guard - its
    `required` / `optional` tuples, the annotations on `check()`, the Bundle slot each
    keyword resolves to, and the `Inputs` block of its docstring. A hand-maintained schema
    goes stale; this one cannot.
  * **A guard that cannot cross a JSON boundary is excluded, with a reason.** Four of them
    need a live Python object - the signal function, the indicator, the fold runner, a
    ResultCard - and `fin_skills.tools.schema.excluded()` says which and why. They are not
    silently broken and not silently dropped.
  * **A failed check is a result, not an error.** `call_tool` returns `passed: false` with
    the findings, the same way `GuardResult` does in Python.
  * **Bad input raises TypeError naming the field**, so a model can correct itself.

Payload shapes and their size caps: `fin_skills.tools.payloads`.
"""
from __future__ import annotations

from fin_skills.tools.export import (FORMATS, anthropic_tools, mcp_tools, openai_tools,
                                     tools_for)
from fin_skills.tools.payloads import (MAX_ARGUMENT_BYTES, MAX_POINTS, decode, encode,
                                       frame_to_payload, series_to_payload, to_frame,
                                       to_series)
from fin_skills.tools.runner import (bundle_coverage, call_guard, call_tool, check_backtest,
                                     describe_guard, list_guards, list_skills, list_tools,
                                     read_skill, search_skills, tool_names)
from fin_skills.tools.schema import (Excluded, Exported, excluded, exported, guard_schema,
                                     input_docs)

__all__ = [
    "Excluded", "Exported", "FORMATS", "MAX_ARGUMENT_BYTES", "MAX_POINTS",
    "anthropic_tools", "bundle_coverage", "call_guard", "call_tool", "check_backtest",
    "decode", "describe_guard", "encode", "excluded", "exported", "frame_to_payload",
    "guard_schema", "input_docs", "list_guards", "list_skills", "list_tools", "mcp_tools",
    "openai_tools", "read_skill", "search_skills", "series_to_payload", "to_frame",
    "to_series", "tool_names", "tools_for",
]
