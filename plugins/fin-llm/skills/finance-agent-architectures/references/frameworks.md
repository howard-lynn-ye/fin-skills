# Orchestration frameworks the finance agents are built on — verified 2026-09-08

Which framework a finance agent uses matters less than people think; what matters is which
*gate primitive* the framework gives you, because that is where a human or a piece of code
can stop the model. Every row was checked against the GitHub REST API, PyPI, and — for the
gate primitive — the framework's own source on 2026-09-08.

## 1. Statistics, exactly as fetched

| Repo | ★ | pushed_at | Archived | Licence (API) | PyPI (version, upload) |
|---|---:|---|---|---|---|
| `langchain-ai/langgraph` | 41,262 | 2026-09-06 | false | MIT | `langgraph` 1.2.11, 2026-08-11 |
| `langchain-ai/langchain` | 145,938 | 2026-09-08 | false | MIT | `langchain` 1.4.0, 2026-09-03 |
| `langchain-ai/deepagents` | 29,150 | 2026-09-08 | false | MIT | `deepagents` 0.7.13, 2026-09-02 |
| `microsoft/autogen` | 60,871 | **2026-04-15** | false | CC-BY-4.0 (docs; `LICENSE-CODE` = MIT ✅) | `autogen-agentchat` 0.7.5, **2025-09-30** |
| `microsoft/agent-framework` | 13,394 | 2026-09-08 | false | MIT | `agent-framework` 1.17.0, 2026-09-03 |
| `ag2ai/ag2` (community fork of AutoGen) | 4,913 | 2026-09-08 | false | Apache-2.0 | `ag2` 1.0.4, 2026-09-07 |
| `crewAIInc/crewAI` | 58,241 | 2026-09-08 | false | MIT | `crewai` 1.15.20, 2026-09-04 |
| `anthropics/claude-agent-sdk-python` | 8,056 | 2026-09-06 | false | MIT | `claude-agent-sdk` 0.2.152, 2026-09-02 |
| `openai/openai-agents-python` | 29,268 | 2026-09-08 | false | MIT | `openai-agents` 0.22.1, 2026-09-08 |
| `google/adk-python` | 21,454 | 2026-09-08 | false | Apache-2.0 | `google-adk` 2.8.0, 2026-08-26 |
| `pydantic/pydantic-ai` | 19,801 | 2026-09-08 | false | MIT | `pydantic-ai` 2.41.0, 2026-09-08 |
| `agno-agi/agno` | 42,101 | 2026-09-08 | false | Apache-2.0 | `agno` 3.0.7, 2026-09-08 (393 deps) |
| `huggingface/smolagents` | 29,236 | 2026-08-25 | false | Apache-2.0 | `smolagents` 1.26.0, 2026-05-29 |
| `letta-ai/letta` | 24,660 | 2026-08-23 | false | Apache-2.0 | `letta-client` 1.12.1, 2026-06-02 |
| `modelcontextprotocol/python-sdk` | 24,242 | 2026-09-07 | false | MIT | `mcp` 2.2.0, 2026-09-07 |
| `run-llama/llama_index` | 52,074 | 2026-09-05 | false | MIT | `llama-index` 0.14.24, 2026-08-19 |

Also: `pyautogen` 0.10.0 (2025-07-15) is now "a proxy package for autogen-agentchat";
`langchain-mcp-adapters` 0.3.2 (2026-08-06) is how LangGraph agents consume MCP servers.

## 2. What each one gives you at the gate

| Framework | Model of computation | Gate / human-in-the-loop primitive (✅ in source) | Used by |
|---|---|---|---|
| **LangGraph** | explicit `StateGraph` of nodes and conditional edges over a typed state; checkpointers (`libs/checkpoint-sqlite`, `checkpoint-postgres`) | `interrupt()` (`libs/langgraph/langgraph/types.py`, `def interrupt` at line 851) pauses a run at a node until a human resumes with `Command`; the checkpoint is the audit trail | TradingAgents (`StateGraph(AgentState)`, sqlite checkpointer), Vibe-Trading (`langgraph>=1.2.5`) |
| **AutoGen** | agents exchange messages in a `GroupChat` managed by a `GroupChatManager`; a `UserProxyAgent` is the human | the `UserProxyAgent` turn (human input mode) — the human is a participant, not a gate outside the loop | FinRobot's legacy package (`import autogen`) |
| 🔴 AutoGen status | README, verified: "AutoGen is now in **maintenance mode**. It will not receive new features ... New users should start with Microsoft Agent Framework" | — | — |
| **Microsoft Agent Framework** | the successor; multi-provider, A2A and MCP interop (README) | ⚠️ not read at source level here | none of the finance systems yet |
| **CrewAI** | role-playing agents grouped in a Crew executing Tasks; Flows for event-driven control | `Task.human_input: bool` (`lib/crewai/src/crewai/task.py`, line 233) — a human reviews that task's output before the crew continues; `crewai/core/providers/human_input.py` | none of the mainstream finance systems; common in tutorials |
| **Claude Agent SDK** | an agent loop with the Claude Code toolset, in-process MCP tools (`@tool` + `create_sdk_mcp_server`), subagents, hooks | `PreToolUse` hook returning `permissionDecision: "deny"` (`examples/hooks.py`) and a `can_use_tool` callback (`examples/tool_permission_callback.py`) — a Python function that runs **before** a tool call and can refuse it; `allowed_tools` is an allowlist, `disallowed_tools` removes | none of the finance systems; the natural home for "gates as hooks" |
| **OpenAI Agents SDK** | agents with handoffs, sessions, tracing | `src/agents/guardrail.py` (input/output guardrails) and `src/agents/tool_guardrails.py` — guardrail functions that run around the model and around each tool call and can trip | none of the finance systems |
| **PydanticAI** | typed agents whose outputs are Pydantic models | the output schema itself is the gate: a `Signal` that must validate | FinRobot Desktop (README) |
| **Google ADK**, **agno**, **smolagents**, **deepagents** | ⚠️ not read at source level here | — | none of the finance systems |
| **Letta** | memory-first agents (persistent, editable memory blocks) | — | none; relevant only to the memory axis |
| **MCP (`mcp` SDK)** | tools, resources and prompts served over stdio/HTTP to any client | none — MCP has no gate; a tool result is untrusted input by design (`../../finance-mcp-servers/SKILL.md` §3) | Vibe-Trading (`vibe-trading-mcp`), OpenBB, every vendor server |

## 3. What to take from this

1. **Pick the framework for its gate, not its demo.** A `PreToolUse` deny hook, an
   `interrupt()` before the execution node, an output guardrail on the `Signal` — those are the
   places code can say no. A framework whose only "human in the loop" is a human *participant*
   in the conversation (AutoGen's `UserProxyAgent`) gives the model the same standing as the
   human.
2. **The gate has to run outside the model's reach.** A prompt instruction ("do not trade
   without approval") is inside it. A hook, a guardrail, a typed schema, a `LiveTradingGate`
   in the order path (`../../../../fin-core/skills/broker-execution-apis/scripts/paper_account_guard.py`)
   is outside it.
3. **AutoGen is the one to migrate off**, and FinRobot's legacy package is built on it. `ag2`
   is the community continuation; `agent-framework` is Microsoft's.
4. **Every one of these is MIT or Apache-2.0**, so the licence problems in this domain come
   from the *finance* layers (TradingAgents-CN's proprietary `app/`, OpenBB's AGPL-3.0, the
   unlicensed research repos), not from the orchestration layer.
