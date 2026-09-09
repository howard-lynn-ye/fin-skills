"""The tool layer itself: `list_tools()` and `call_tool(name, arguments)`, no transport.

Everything an agent can do with this library goes through these two functions. The MCP
server in `fin_skills.mcp` is a thin adapter over them; so is anything you wire into the
Anthropic or OpenAI APIs with `fin_skills.tools.export`. Nothing here imports a transport,
so the layer is testable without one.

Two kinds of tool:

  * **catalogue tools**, which need no data at all - `list_skills`, `read_skill`,
    `search_skills`, `list_guards`, `describe_guard`, `bundle_coverage`. These are what an
    agent reaches for first and most often, so their output is compact: names and one-line
    summaries, not whole documents, until you ask for one.
  * **guard tools**, one per guard that can be driven over JSON (`check_<guard>`), plus
    `check_backtest`, which takes a whole Bundle of slots and runs every guard whose
    inputs are present - the `fin_skills.api.check(bundle)` call as a single tool.

The result of a guard tool is the same verdict `fin_skills.api` gives in Python, reduced
to JSON: `passed`, the findings with severity and message, the evidence as scalars and
small tables, and the elapsed time. A FAILED CHECK IS NOT AN ERROR - it comes back with
`passed: false` and the findings that explain it, exactly as `GuardResult` does. Only bad
inputs raise, and they raise TypeError naming the field, so an MCP server can turn them
into a tool-execution error the model can correct.
"""
from __future__ import annotations

import difflib
import time
from typing import Any, Iterable, Mapping

import fin_skills
from fin_skills.api import Bundle, Coverage, GuardResult, check, get, registry
from fin_skills.api import slots as slot_vocabulary
from fin_skills.api.base import Guard, ascii_only
from fin_skills.api.bundle import DATA_SLOTS, _slot_of, vocabulary
from fin_skills.tools import payloads
from fin_skills.tools.schema import Exported, exported, excluded, input_docs, input_kinds

MAX_SKILL_CHARS = 200_000
"""Largest SKILL.md or reference `read_skill` will return whole."""

_SNIPPET = 160


# ================================================================= catalogue tool bodies
def list_skills(plugin: str | None = None) -> dict[str, Any]:
    """Every skill: name, plugin, one-line summary, what it ships. No SKILL.md text."""
    out = []
    for s in fin_skills.catalog():
        if plugin and s.get("plugin") != plugin:
            continue
        out.append({"name": s.get("name"), "plugin": s.get("plugin"),
                    "summary": ascii_only(s.get("summary") or "")[:240],
                    "scripts": s.get("scripts") or [],
                    "references": len(s.get("references") or []),
                    "verified_on": s.get("verified_on")})
    return {"skills": out, "count": len(out),
            "plugins": sorted({s["plugin"] for s in out if s.get("plugin")}),
            "next": "read_skill(name) returns the SKILL.md text for one of these"}


def read_skill(name: str, reference: str | None = None) -> dict[str, Any]:
    """The SKILL.md text for one skill, or one file from its references/."""
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty skill name; list_skills() lists them")
    try:
        refs = fin_skills.references(name)
    except KeyError as exc:
        close = difflib.get_close_matches(name, fin_skills.names(), n=3, cutoff=0.5)
        raise TypeError(f"name: no skill named {name!r}"
                        + (f" - did you mean {close}?" if close else "")) from exc
    if reference is not None:
        if reference not in refs:
            raise TypeError(f"reference: {name!r} has no reference {reference!r}; "
                            f"it has {sorted(refs)}")
        text = refs[reference]
        return {"skill": name, "reference": reference, "chars": len(text),
                "text": text[:MAX_SKILL_CHARS]}
    text = fin_skills.load(name)
    return {"skill": name, "chars": len(text), "references": sorted(refs),
            "guards": [c.name for c in registry() if c.skill == name],
            "text": text[:MAX_SKILL_CHARS]}


def search_skills(terms: Iterable[str] | str, limit: int = 20) -> dict[str, Any]:
    """Skills whose SKILL.md mentions every term, with the line each one matched on."""
    words = [terms] if isinstance(terms, str) else list(terms or [])
    words = [str(w).strip() for w in words if str(w).strip()]
    if not words:
        raise TypeError("terms must be a non-empty string or list of strings")
    hits = []
    for skill in fin_skills.find(*words):
        text = fin_skills.load(skill)
        line = next((ln.strip() for ln in text.splitlines()
                     if all(w.lower() in ln.lower() for w in words)), "")
        if not line:
            line = next((ln.strip() for ln in text.splitlines()
                         if words[0].lower() in ln.lower()), "")
        hits.append({"skill": skill, "match": ascii_only(line)[:_SNIPPET]})
        if len(hits) >= max(1, int(limit)):
            break
    return {"terms": words, "count": len(hits), "skills": hits}


def list_guards(skill: str | None = None) -> dict[str, Any]:
    """Every guard: what it refuses, what it needs, and whether it can be called as a tool."""
    exp, exc = exported(), excluded()
    out = []
    for cls in registry():
        if skill and cls.skill != skill:
            continue
        out.append({"guard": cls.name, "skill": cls.skill,
                    "summary": ascii_only(cls.summary),
                    "required": list(cls.required), "optional": list(cls.optional),
                    "tool": exp[cls.name].tool if cls.name in exp else None,
                    "callable_over_json": cls.name in exp,
                    "excluded_because": exc[cls.name].reason if cls.name in exc else None})
    return {"guards": out, "count": len(out),
            "exported": sum(1 for g in out if g["callable_over_json"]),
            "excluded": sum(1 for g in out if not g["callable_over_json"])}


def describe_guard(name: str) -> dict[str, Any]:
    """One guard in full: description, JSON Schema, what it wraps, what it refuses."""
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a guard name; list_guards() lists them")
    key = name[len("check_"):] if name.startswith("check_") else name
    by_name = {c.name: c for c in registry()}
    if key not in by_name:
        close = difflib.get_close_matches(key, sorted(by_name), n=3, cutoff=0.4)
        raise TypeError(f"name: no guard named {name!r}"
                        + (f" - did you mean {close}?" if close else ""))
    cls = by_name[key]
    exp, exc = exported(), excluded()
    out: dict[str, Any] = {
        "guard": cls.name, "skill": cls.skill, "summary": ascii_only(cls.summary),
        "wraps": list(cls.wraps),
        "required": list(cls.required), "optional": list(cls.optional),
        "requires": cls().missing({}),
        "inputs": input_docs(cls),
        "slots": {k: _slot_of(cls.name, k)
                  for k in tuple(cls.required) + tuple(cls.optional)
                  if _slot_of(cls.name, k) != k},
    }
    if key in exp:
        out.update(tool=exp[key].tool, description=exp[key].description,
                   input_schema=exp[key].schema,
                   omitted={k: why for k, why in exp[key].omitted})
    else:
        out.update(tool=None, callable_over_json=False,
                   excluded_because=exc[key].reason)
    return out


def bundle_coverage(slots: Iterable[str] | None = None) -> dict[str, Any]:
    """Which guards would run given a set of slot NAMES - no data, just the names.

    Called with nothing it returns the whole vocabulary instead: every slot, what it holds
    and which guards it reaches. That is the cheapest way for an agent to learn what
    `check_backtest` accepts.
    """
    names = [str(s).strip() for s in (slots or []) if str(s).strip()]
    vocab = vocabulary()
    if not names:
        curated = {s.name for s in DATA_SLOTS}
        listed = slot_vocabulary()
        return {"vocabulary": [{"slot": s.name, "kind": s.kind, "means": ascii_only(s.doc)}
                               for s in listed if s.name in curated],
                "parameters": sorted(s.name for s in listed if s.name not in curated),
                "count": len(listed),
                "next": "the `parameters` are per-guard tuning knobs - describe_guard(name) "
                        "documents each one. bundle_coverage(['returns','turnover']) says "
                        "which guards those unlock; check_backtest({'returns': ...}) runs them"}
    unknown = sorted(set(names) - vocab)
    known = [n for n in names if n in vocab]
    cov = _coverage_of(set(known))
    exp = exported()
    out: dict[str, Any] = {
        "have": known,
        "ready": cov.ready,
        "ready_over_json": [g for g in cov.ready if g in exp],
        "not_ready": {g: miss for g, miss in cov.missing.items()},
        "one_slot_away": cov.unlocks(),
        "summary": cov.summary(),
    }
    if unknown:
        out["unknown"] = {u: difflib.get_close_matches(u, sorted(vocab), n=3, cutoff=0.5)
                          for u in unknown}
    return out


def _missing_for(guard: Guard, have: set[str]) -> list[str]:
    """`Bundle.missing_for` without a Bundle - the caller has slot NAMES, not values."""
    return [s for s in (_slot_of(guard.name, k) for k in guard.required) if s not in have]


def _coverage_of(have: set[str]) -> Coverage:
    ready, missing = [], {}
    for cls in registry():
        g = cls()
        miss = _missing_for(g, have)
        (missing.__setitem__(g.name, miss) if miss else ready.append(g.name))
    return Coverage(ready=ready, missing=missing)


# ===================================================================== guard tool bodies
def _result(tool: str, r: GuardResult) -> dict[str, Any]:
    return {"tool": tool, "guard": r.guard, "skill": r.skill, "passed": bool(r.passed),
            "findings": [{"severity": f.severity, "message": ascii_only(f.message),
                          "where": f.where} for f in r.findings],
            "n_errors": len(r.errors), "n_warnings": len(r.warnings),
            "evidence": {k: payloads.encode(v, k) for k, v in r.evidence.items()},
            "elapsed_s": round(float(r.elapsed_s), 4),
            "summary": r.summary()}


def _decode_arguments(spec: Exported, arguments: Mapping[str, Any]) -> dict[str, Any]:
    props = spec.schema["properties"]
    unknown = sorted(set(arguments) - set(props))
    if unknown:
        omitted = {k for k, _ in spec.omitted}
        blocked = sorted(set(unknown) & omitted)
        extra = (f" ({', '.join(blocked)} cannot cross a JSON boundary - "
                 f"call the guard from Python)" if blocked else "")
        raise TypeError(f"{spec.tool} got unexpected argument(s) {unknown}{extra}; "
                        f"accepted: {sorted(props)}")
    kinds = input_kinds(spec.guard)
    return {k: payloads.decode(v, k, kinds.get(k, "any"))
            for k, v in arguments.items() if v is not None}


def call_guard(guard: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run one guard from a JSON `arguments` object. Same verdict as calling it in Python."""
    spec = exported().get(guard)
    if spec is None:
        why = excluded().get(guard)
        raise TypeError(f"{guard} is not callable as a tool: "
                        + (why.reason if why else f"no guard named {guard!r}"))
    args = dict(arguments or {})
    payloads.check_size(args)
    return _result(spec.tool, get(guard).run(**_decode_arguments(spec, args)))


def check_backtest(slots: Mapping[str, Any] | None = None,
                   guards: Iterable[str] | None = None,
                   strict: bool = False) -> dict[str, Any]:
    """Every guard whose slots are present, in one call - the Bundle story over JSON.

    `slots` is the shared vocabulary (`bundle_coverage()` with no arguments lists it):
    one slot feeds every guard that means the same thing by it, so `returns` reaches
    cost_curve, rf_convention and regime_coverage at once. Guards that are missing a slot
    are reported as skipped with what they still need; guards that cannot be driven over
    JSON at all are reported as excluded with the reason.
    """
    raw = dict(slots or {})
    payloads.check_size(raw)
    vocab = vocabulary()
    unknown = sorted(set(raw) - vocab)
    if unknown:
        hints = {u: difflib.get_close_matches(u, sorted(vocab), n=2, cutoff=0.5)
                 for u in unknown}
        raise TypeError(f"slots: unknown slot name(s) {unknown} - did you mean {hints}? "
                        f"bundle_coverage() lists the vocabulary")
    kinds = _slot_kinds()
    decoded = {k: payloads.decode(v, k, kinds.get(k, "any")) for k, v in raw.items()
               if v is not None}
    bundle = Bundle(**decoded)                      # raises TypeError naming the slot
    exp = excluded()
    wanted = list(guards) if guards else [c.name for c in registry() if c.name not in exp]
    bad = [g for g in wanted if g in exp]
    if bad:
        raise TypeError(f"guards: {bad} cannot be driven over JSON - "
                        + "; ".join(exp[g].reason for g in bad))
    t0 = time.perf_counter()
    report = check(bundle, guards=wanted, strict=strict)
    return {"tool": "check_backtest",
            "passed": bool(report.passed),
            "ran": report.ran,
            "failed": [r.guard for r in report.failed],
            "results": [_result("check_backtest", r) for r in report],
            "skipped": {g: miss for g, miss in report.skipped.items()},
            "rejected": dict(report.rejected),
            "not_callable_over_json": sorted(exp),
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "summary": report.summary()}


def _slot_kinds() -> dict[str, str]:
    """{slot: 'series' | 'frame' | 'other'} - the union of every guard's view of the slot."""
    out: dict[str, str] = {}
    for cls in registry():
        for keyword, kind in input_kinds(cls.name).items():
            slot = _slot_of(cls.name, keyword)
            if kind in ("series", "frame") or slot not in out:
                out[slot] = kind
    return out


# ============================================================================ the tool table
def _catalogue_tools() -> list[dict[str, Any]]:
    vocab_hint = ("Slot names come from bundle_coverage() called with no arguments.")
    return [
        {"name": "list_skills",
         "description": "List every skill in fin-skills: name, plugin, one-line summary, "
                        "which guard scripts it ships, and the date it was verified. No "
                        "SKILL.md text - call read_skill for that.",
         "input_schema": {"type": "object", "properties": {
             "plugin": {"type": "string",
                        "description": "Restrict to one plugin, e.g. 'fin-core'."}},
             "required": [], "additionalProperties": False}},
        {"name": "read_skill",
         "description": "The full SKILL.md text of one skill, or one file from its "
                        "references/. Use it when a guard has failed and you need the "
                        "reasoning behind the check.",
         "input_schema": {"type": "object", "properties": {
             "name": {"type": "string", "description": "Skill name from list_skills."},
             "reference": {"type": "string",
                           "description": "A file from that skill's references/ instead "
                                          "of SKILL.md."}},
             "required": ["name"], "additionalProperties": False}},
        {"name": "search_skills",
         "description": "Skills whose SKILL.md mentions EVERY term, with the line each one "
                        "matched on. Cheaper than reading skills to find the right one.",
         "input_schema": {"type": "object", "properties": {
             "terms": {"type": "array", "items": {"type": "string"},
                       "description": "All must appear, case-insensitive."},
             "limit": {"type": "integer", "description": "Maximum skills to return. "
                                                         "Default 20."}},
             "required": ["terms"], "additionalProperties": False}},
        {"name": "list_guards",
         "description": "Every executable guard: what it refuses, the inputs it needs, the "
                        "tool name that calls it, and - for the ones that cannot be driven "
                        "over JSON - the reason.",
         "input_schema": {"type": "object", "properties": {
             "skill": {"type": "string", "description": "Restrict to one owning skill."}},
             "required": [], "additionalProperties": False}},
        {"name": "describe_guard",
         "description": "One guard in full: its description, its JSON Schema, the "
                        "functions it wraps, the Bundle slots its keywords resolve to, and "
                        "any input that had to be omitted from the schema.",
         "input_schema": {"type": "object", "properties": {
             "name": {"type": "string",
                      "description": "Guard name, e.g. 'cost_curve' (or 'check_cost_curve')."}},
             "required": ["name"], "additionalProperties": False}},
        {"name": "bundle_coverage",
         "description": "Which guards would run if you had these slots - the cheap dry run "
                        "before check_backtest. Call it with NO arguments to list the whole "
                        "vocabulary: every slot name, what it holds and which guards it "
                        "reaches. Call it with slot names to get ready / not-ready / "
                        "one-slot-away.",
         "input_schema": {"type": "object", "properties": {
             "slots": {"type": "array", "items": {"type": "string"},
                       "description": "Slot NAMES you have data for, e.g. "
                                      "['returns','turnover','rf']. No data is sent."}},
             "required": [], "additionalProperties": False}},
        {"name": "check_backtest",
         "description": "Run every guard whose inputs are present, in one call. One slot "
                        "feeds every guard that means the same thing by it, so `returns` "
                        "reaches cost_curve, rf_convention and regime_coverage at once. "
                        "Returns each guard's verdict plus the guards that were skipped and "
                        "what they still needed. A failed check is a result, not an error. "
                        + vocab_hint,
         "input_schema": {"type": "object", "properties": {
             "slots": {"type": "object",
                       "description": "{slot name: value}. Series are "
                                      "{'index': [...], 'values': [...]}, frames are "
                                      "{'index': [...], 'records': [{col: value}, ...]}, "
                                      "dates are ISO strings. " + vocab_hint,
                       "additionalProperties": True},
             "guards": {"type": "array", "items": {"type": "string"},
                        "description": "Restrict to these guards. Default: all of them."},
             "strict": {"type": "boolean",
                        "description": "Raise instead of recording a guard that rejected "
                                       "its inputs. Default false."}},
             "required": ["slots"], "additionalProperties": False}},
    ]


_CATALOGUE = {
    "list_skills": list_skills,
    "read_skill": read_skill,
    "search_skills": search_skills,
    "list_guards": list_guards,
    "describe_guard": describe_guard,
    "bundle_coverage": bundle_coverage,
    "check_backtest": check_backtest,
}

_TOOLS: list[dict[str, Any]] | None = None


def list_tools() -> list[dict[str, Any]]:
    """Every tool, transport-neutral: {'name', 'description', 'input_schema'[, 'guard']}.

    Sorted with the catalogue tools first (they are what an agent needs to orient), then
    the per-guard tools alphabetically. `export.py` reshapes these for MCP, the Anthropic
    Messages API and the OpenAI APIs; nothing else in the codebase writes a tool definition.
    """
    global _TOOLS
    if _TOOLS is None:
        tools = _catalogue_tools()
        for name in sorted(exported()):
            spec = exported()[name]
            tools.append({"name": spec.tool, "description": spec.description,
                          "input_schema": spec.schema, "guard": spec.guard,
                          "skill": spec.skill})
        _TOOLS = tools
    return [dict(t) for t in _TOOLS]


def tool_names() -> list[str]:
    return [t["name"] for t in list_tools()]


def call_tool(name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Call one tool by name with a JSON `arguments` object; get a JSON-serializable result.

    Raises TypeError - naming the field - for an unknown tool, an unknown argument, a
    payload of the wrong shape or a missing required input. A guard that FAILS its check
    is not an error: the result carries `passed: false` and the findings.
    """
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a tool name; list_tools() lists them")
    args = dict(arguments or {})
    if name in _CATALOGUE:
        payloads.check_size(args)
        fn = _CATALOGUE[name]
        spec = next(t for t in list_tools() if t["name"] == name)
        allowed = set(spec["input_schema"]["properties"])
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise TypeError(f"{name} got unexpected argument(s) {unknown}; "
                            f"accepted: {sorted(allowed)}")
        t0 = time.perf_counter()
        out = fn(**args)
        out.setdefault("tool", name)
        out.setdefault("elapsed_s", round(time.perf_counter() - t0, 4))
        return out
    if name.startswith("check_") and name[len("check_"):] in exported():
        return call_guard(name[len("check_"):], args)
    close = difflib.get_close_matches(name, tool_names(), n=3, cutoff=0.4)
    blocked = excluded().get(name[len("check_"):] if name.startswith("check_") else name)
    if blocked is not None:
        raise TypeError(f"{name} is not a tool: {blocked.reason}")
    raise TypeError(f"no tool named {name!r}"
                    + (f" - did you mean {close}?" if close else "")
                    + f"; {len(tool_names())} tools are available")


__all__ = ["MAX_SKILL_CHARS", "bundle_coverage", "call_guard", "call_tool", "check_backtest",
           "describe_guard", "list_guards", "list_skills", "list_tools", "read_skill",
           "search_skills", "tool_names"]
