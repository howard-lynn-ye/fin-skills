---
name: fin-skills-as-tools
description: >-
  How to hand this library to an agent as TOOLS rather than as reading - the MCP server, the
  exported Anthropic and OpenAI tool definitions, the JSON payload conventions, and the four
  guards that cannot cross a JSON boundary. TRIGGER - expose fin-skills to an LLM agent; run
  the fin-skills MCP server; "python -m fin_skills.mcp"; fin-skills-mcp; import the guards as
  function-calling tools; generate tool schemas for the Messages API or the OpenAI Responses
  API; send a pandas Series or DataFrame through a tool call; check_backtest, bundle_coverage,
  describe_guard, read_skill; why assert_causal or warmup_probe is missing from the tool list;
  payload size limits for a tool call; 把 guard 挂成 agent 工具. SKIP for how to design the agent
  pipeline itself (finance-agent-architectures), which third-party finance MCP servers exist
  and which can move money (finance-mcp-servers), and whether an LLM strategy makes money at
  all (llm-finance-agents).
license: MIT
metadata:
  version: "0.1.0"
  verified_on: "2026-09-09"
---

# fin-skills as tools an agent can call

An agent that can *read* these skills still cannot *run* the guards. Reading a skill about
look-ahead bias changes what the model says; running `check_cost_curve` changes what the
pipeline is allowed to report. This skill is the second thing: the transport-independent tool
layer (`fin_skills.tools`), the MCP server over it (`fin_skills.mcp`), and the exported schemas
for the Anthropic and OpenAI APIs.

The design rule of the whole layer: **nothing is written down twice.** Every schema is derived
at import time from the guard itself — its `required` / `optional` tuples, the annotations on
`check()`, the Bundle slot each keyword resolves to, and the `Inputs` block of its docstring. A
hand-maintained schema goes stale the first time an argument is added; this one cannot.

## 1. What exists, in numbers

✅ measured 2026-09-09 with `python -m fin_skills.tools --list` and
`python -c "from fin_skills.tools import list_tools, exported, excluded; ..."`.

| | Count | |
|---|---|---|
| Tools total | **31** | `python -m fin_skills.tools --list` |
| — catalogue tools (need no data) | 7 | `list_skills`, `read_skill`, `search_skills`, `list_guards`, `describe_guard`, `bundle_coverage`, `check_backtest` |
| — one per guard | 24 | `check_<guard>` |
| Guards in the registry | 28 | `fin_skills.api.registry()` |
| — exported as tools | 24 | can be driven entirely from JSON |
| — **excluded, with a reason** | **4** | §4 |
| Skills readable through `read_skill` | 71 | `fin_skills.names()` |
| Bundle slot names `check_backtest` accepts | 129 | 61 curated data slots + 68 per-guard tuning knobs |

The whole Anthropic-shaped tool array is **50,376 characters** of JSON
(`len(json.dumps(anthropic_tools()))`). The 7 catalogue tools alone are **3,809**. If your
context budget is tight, ship only those seven and let the agent reach the rest through
`describe_guard`; the descriptions are written so that is enough.

## 2. The MCP server

```bash
pip install "fin-skills[mcp]"        # the SDK is an OPTIONAL extra
python -m fin_skills.mcp             # stdio; or the console script: fin-skills-mcp
```

```bash
claude mcp add fin-skills -- python -m fin_skills.mcp
```

✅ verified at the primary sources on 2026-09-09:

- The **current MCP protocol revision is `2026-07-28`**
  (`modelcontextprotocol.io/specification/versioning`). Its `server/tools` page fixes the
  shapes this server emits: a server declares `capabilities.tools`, answers `tools/list` with
  `{name, title?, description, inputSchema}` entries, and `tools/call` with `content` plus an
  optional `structuredContent`. `inputSchema` **must** be a JSON Schema object, **defaults to
  draft 2020-12 when no `$schema` key is present**, and needs `"type": "object"` at the root.
  Tool names should be 1–128 characters of `A-Za-z0-9_-.`.
- The sanctioned Python implementation is the official SDK, **`mcp` on PyPI, MIT licensed,
  latest 2.2.0, `requires-python >= 3.10`** (`pypi.org/pypi/mcp/json`; source at
  `github.com/modelcontextprotocol/python-sdk`). `fin_skills.mcp` targets the **v2 low-level
  `Server`** — constructor handlers `on_list_tools=` / `on_call_tool=` and a hand-written
  `input_schema` dict. The high-level `MCPServer` derives its schema from Python type hints,
  which is exactly what a runtime-derived schema cannot use.

**The extra is optional and stays optional.** `pyproject.toml` adds
`[project.optional-dependencies] mcp = ["mcp>=2.0"]` and nothing to the base dependencies, which
remain numpy / pandas / scipy. `import fin_skills.mcp` works with the SDK absent — the import is
deferred — and `python -m fin_skills.mcp` then prints the install line and **exits 1**:

```
fin-skills: cannot start - the Model Context Protocol SDK is not installed.
  pip install "fin-skills[mcp]"          # or: pip install "mcp>=2.0"
```

`--help`, `--list-tools`, `--json` and `--check` all work without the SDK, so a CI job can
verify the tool table without installing a transport.

✅ run end to end against `mcp` 2.2.0 on 2026-09-09: a client launching
`python -m fin_skills.mcp` as a stdio subprocess got 31 tools from `tools/list`, a passing
verdict from `check_rf_convention`, a failing one from `check_cost_curve`, and `isError: true`
with the field named for a mistyped payload. `tests/test_mcp.py` is that round trip, behind
`requires("mcp")`; the tests that matter more — the module importing and exiting cleanly with
the SDK absent — run everywhere.

**The two error channels are respected, because the spec is explicit that they differ.** A
protocol error (JSON-RPC) is for things the model cannot fix; a tool-execution error
(`isError: true` in a normal result) is for things it can. So:

| What happened | What comes back |
|---|---|
| The guard ran and the check **failed** | a normal result, `isError` false, `"passed": false` plus the findings |
| Bad arguments — wrong payload shape, missing required input, unknown tool | a normal result with **`isError: true`** and a message naming the field |
| Nothing else | — the server never raises out of a handler, because a low-level handler that raises becomes an opaque `-32603` the model cannot learn from |

## 3. The same tools for the Messages API and for OpenAI

One table, three renames — `fin_skills.tools.export`. ✅ the Anthropic shape
(`name` / `description` / `input_schema`) verified 2026-09-09 at
`platform.claude.com/docs/en/agents-and-tools/tool-use/overview`, and the OpenAI **Responses**
shape (`type: "function"` beside `name` / `description` / `parameters`) at
`developers.openai.com/api/docs/guides/function-calling`. ⚠️ that page carries no Chat
Completions example, so the `{"type": "function", "function": {...}}` nesting that `--format
openai-chat` emits is secondhand — check it against the SDK you use.

```bash
python -m fin_skills.tools --json                     # Anthropic: name, description, input_schema
python -m fin_skills.tools --json --format openai     # Responses API: type=function, name, description, parameters
python -m fin_skills.tools --json --format openai-chat  # Chat Completions: {type, function:{...}}
python -m fin_skills.tools --json --format mcp        # tools/list entries (camelCase inputSchema)
python -m fin_skills.tools --list                     # names and one-liners
python -m fin_skills.tools --excluded                 # the four, and why
```

```python
import anthropic
from fin_skills.tools import anthropic_tools, call_tool

resp = anthropic.Anthropic().messages.create(
    model="claude-opus-5", max_tokens=2048, tools=anthropic_tools(),
    messages=[{"role": "user", "content": "Does this backtest survive 10 bps?"}])
for block in resp.content:                       # then, for each tool_use block:
    if block.type == "tool_use":
        result = call_tool(block.name, block.input)     # -> a JSON-serializable dict
```

`call_tool` is the whole contract. It raises `TypeError` naming the field for bad input, and
returns a dict for everything else — send `json.dumps(result)` back as the `tool_result`.

⚠️ The payload shapes use `$defs` and a local `$ref` (§5) — inlined instead, the Series
definition repeats about thirty times. MCP's tools page names `$ref`/`$defs` as available
2020-12 keywords; ⚠️ whether the Anthropic and OpenAI validators resolve a local `$ref` was
NOT verified at a primary source, so test one call before shipping. ✅ the OpenAI **strict**
mode requirement is on the page above: every property must be in `required` and
`additionalProperties` must be `false`. These schemas set the second but not the first, so pass
`strict: false` (the default) or post-process.

## 4. The four guards that cannot cross a JSON boundary

A guard is excluded when a **required** input has no JSON representation. Not silently dropped,
not silently broken: `python -m fin_skills.tools --excluded` prints the reason, `list_guards`
returns it in `excluded_because`, and `tests/test_tools.py` asserts that the set is exactly
these four.

| Guard | Skill | Required input that cannot travel |
|---|---|---|
| `assert_causal` | `signal-construction` | `fn` — the signal function. The check *re-runs it* on perturbed bars and asserts nothing before bar *k* moved. A JSON description of a function is not that function. |
| `warmup_probe` | `signal-construction` | `indicator` — same reason: the probe calls it at growing history lengths. |
| `fold_leak_test` | `market-data-engineering` | `run_fold` — the fold runner it executes in parallel to compare results. |
| `result_manifest` | `research-integrity-guards` | `card` — a live `fin_skills.core.result_manifest.ResultCard`; `check()` rejects anything else by type. |

🚨 **Three of the four are the leak detectors.** An agent driving this library over JSON alone
*cannot* run the causality check. That is the honest limit of the tool boundary, not an
oversight: fabricating a function so the tool "works" would test a function the caller never
wrote. The reason string says what to do instead —
`fin_skills.api.get("assert_causal").run(fn=..., df=..., k=250)` in Python, in the same process
that owns the function. An agent that writes code (Claude Code, a code-execution tool) should
call the Python API for these and the tools for the rest.

Two more guards keep an **optional** input out of their schema for the same reason; the tool
description says so under `Not available over JSON`:

| Guard | Dropped optional | Why |
|---|---|---|
| `purge_effect` | `score_fn` | a `Callable[..., np.ndarray]` |
| `contamination_probe` | `items`, `ask`, `match` | a sequence of `ProbeItem` objects, and two callables |

## 5. Payload conventions

pandas objects cross as JSON in one shape each, defined once in `fin_skills.tools.payloads` —
the schema fragments the tools publish *are* the shapes the decoder accepts, so they cannot
drift apart.

```jsonc
// Series                                    // DataFrame
{"index": ["2023-01-02T00:00:00", ...],      {"index": ["2023-01-02T00:00:00", ...],
 "values": [0.000605, 0.008558, ...],         "columns": ["open", "close"],
 "name": "strategy",                          "records": [{"open": 1.0, "close": 2.0}, ...],
 "dtype": "float64",                          "dtypes": {"open": "float64", ...}}
 "freq": "B"}
```

- `index`, `name`, `dtype` and `freq` are optional. A **bare list of numbers** is accepted
  wherever a Series is expected, and a bare list of records wherever a frame is.
- **Dates are ISO-8601 strings**, in the index and in cells.
- `freq` travels because a `pd.bdate_range` index has one and `pandas.testing.assert_series_equal`
  compares it. Drop it and the round trip is no longer exact — the test asserts it is.
- **JSON has no NaN.** A missing value is `null`; an infinity is the string `"Infinity"` or
  `"-Infinity"`, because a breakeven cost of infinity is a real and useful answer that must not
  become null.
- A mapping whose key is a near-miss for a payload key is rejected by name:
  `returns: key 'vals' looks like a mistyped 'values'.`

Size caps, as module constants so this section can quote the code
(`fin_skills.tools.payloads`):

| Constant | Value | What it bounds |
|---|---|---|
| `MAX_ARGUMENT_BYTES` | 8,000,000 | the whole `arguments` object, measured as compact JSON |
| `MAX_POINTS` | 1,000,000 | cells in one series or frame (rows × columns) |
| `MAX_EVIDENCE_ROWS` | 50 | rows of any one evidence table in the **result** |
| `MAX_EVIDENCE_ITEMS` | 500 | elements of any one evidence series or list |
| `MAX_EVIDENCE_CHARS` | 4,000 | characters of any one evidence string |

For scale: 504 daily returns with an index is a 23,137-character payload, and the whole
`check_backtest` result below is 6,977 characters. Evidence is *reduced* on the way out, never
returned whole — a frame is truncated to 50 rows, a series to 500 points, and the truncation is
declared in a `truncated` key rather than passed off as the whole thing.

## 6. A worked exchange

The agent has a strategy's per-period returns and its turnover, and wants to know what can be
checked. Two calls. ✅ real output, seed 11, 504 business days from 2023-01-02, produced
2026-09-09.

**Call 1 — `bundle_coverage`, which sends no data at all, only slot names.**

```jsonc
// -> {"slots": ["returns", "turnover", "rf"]}
{"have": ["returns", "turnover", "rf"],
 "ready_over_json": ["brinson_attribution", "cost_curve", "pit_universe", "rf_convention"],
 "not_ready": {"regime_coverage": ["dates", "underlying_returns", "position", "regime_labels"],
               "spa_test": ["benchmark_returns", "model_returns"],
               "survivorship_audit": ["prices"], ...},
 "one_slot_away": {"asset_returns": ["weight_traps"], "prices": ["survivorship_audit"],
                   "broker": ["paper_account_guard"], "pair": ["fx_conventions"],
                   "card": ["result_manifest"], "stitched": ["continuous_contract"]}}
```

`one_slot_away` is the useful half: it tells the agent that **one** more artifact — the price
panel — buys the survivorship audit. Call `bundle_coverage` with no arguments at all and it
returns the whole vocabulary instead, which is how an agent learns the slot names.

**Call 2 — `check_backtest`, the Bundle story in one call.** One slot feeds every guard that
means the same thing by it, so `returns` reaches `cost_curve` *and* `rf_convention`.

```jsonc
// -> {"slots": {"returns": {"index": [...], "values": [...], "freq": "B"},
//               "turnover": {...}, "rf": 0.05}}
{"passed": false, "ran": ["cost_curve", "rf_convention"], "failed": ["cost_curve"],
 "skipped": {"survivorship_audit": ["prices"], "spa_test": [...], ...},   // 20 of them
 "rejected": {"brinson_attribution": "...missing ['panel or panels']", "pit_universe": "..."},
 "not_callable_over_json": ["assert_causal", "fold_leak_test", "result_manifest", "warmup_probe"],
 "results": [{"guard": "cost_curve", "skill": "backtest-validation", "passed": false,
              "findings": [{"severity": "error", "where": "breakeven",
                            "message": "breakeven 6.0 bps round-trip is at or below the 10.0
                              bps you pay ... (gross Sharpe 1.51, net -0.94)"}],
              "n_errors": 1, "elapsed_s": 0.0155,
              "evidence": {"breakeven_bps": 5.971528589725494, "cost_bps": 10.0,
                           "sharpe_gross": 1.5134936665041614,
                           "sharpe_at_cost": -0.9433311685573142,
                           "avg_turnover": 0.9003518180377624,
                           "curve": {"index": [0.0, 5.0, 10.0, 20.0, 50.0], "records": [...]},
                           "table": "COST SENSITIVITY\n..."}},
             {"guard": "rf_convention", "passed": true, ...}]}
```

`evidence["table"]` is the guard's own rendering, ready to paste into a report:

```
   bps      total       ann   sharpe     maxDD  verdict
     0     31.1%    14.5%     1.51     -4.9%  tradeable
     5      4.5%     2.2%     0.28     -6.7%  marginal
    10    -16.7%    -8.7%    -0.94    -17.9%  DEAD
```

**A gross Sharpe of 1.51 that is −0.94 at the cost you say you pay is the point of the whole
exercise**, and the agent got it from one tool call rather than from a prior about
transaction costs. Note also what the response teaches: 20 skipped guards each name what they
still need, two `rejected` entries carry the either/or requirement their guard enforces
(`panel or panels`), and `not_callable_over_json` repeats the §4 limit in every response, so
the agent is never left to conclude that four checks simply passed.

## 7. Reading the tools before you call them

| Question | Tool |
|---|---|
| "Which skill covers this?" | `search_skills(["survivorship", "delisting"])` |
| "What does the skill actually say?" | `read_skill("backtest-validation")` |
| "What can I check at all?" | `list_guards()` |
| "What does this one need, exactly?" | `describe_guard("cost_curve")` — full schema, the wrapped functions, the Bundle slots its keywords resolve to |
| "What would run if I had X?" | `bundle_coverage(["returns", "turnover"])` |

`describe_guard` on an excluded guard returns `"tool": null` and the reason, so an agent that
asks about `assert_causal` is told what to do instead rather than getting a 404.

## 8. Where to go next

- Designing the pipeline these tools gate → `../finance-agent-architectures/SKILL.md` §3
- Third-party finance MCP servers, and which can move money → `../finance-mcp-servers/SKILL.md`
- Whether the LLM strategy is worth gating at all → `../llm-finance-agents/SKILL.md` §1
- The guards themselves, in Python → the `fin_skills.api` section of the repository README
- Cost curve, trial ledger, DSR and SPA → `../../../fin-core/skills/backtest-validation/SKILL.md`
- The five-gate audit and the result card → `../../../fin-core/skills/research-integrity-guards/SKILL.md`
