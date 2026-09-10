"""fin_skills.tools - the guards as JSON-callable tools.

Four properties are load-bearing and every one of them is asserted here:

  1. every exported schema is valid JSON Schema and survives json.dumps;
  2. every guard is either exported or excluded WITH A REASON - nothing is silently
     dropped and nothing is silently broken;
  3. a Series and a DataFrame survive the JSON round trip EXACTLY, index, dtype, name,
     freq and non-finite values included;
  4. a guard called through call_tool returns the same verdict as the same guard called
     directly through fin_skills.api, and a bad payload raises TypeError naming the field.

Run:  python -m pytest tests/test_tools.py -q
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import fin_skills.api as api
from conftest import has_module
from fin_skills.tools import (anthropic_tools, call_tool, excluded, exported,
                              frame_to_payload, list_tools, mcp_tools, openai_tools,
                              series_to_payload, to_frame, to_series, tool_names)
from fin_skills.tools import payloads, schema as tool_schema

TOOLS = list_tools()
EXPORTED = exported()
EXCLUDED = excluded()

# The guards whose required input is a live Python object. Named here so a change to the
# exclusion rule has to be a deliberate edit to this list, not a silent drift.
CANNOT_CROSS_JSON = {"assert_causal", "fold_leak_test", "result_manifest",
                     "synthesis_integrity", "warmup_probe"}


def _validator():
    """The strictest JSON Schema validator this environment has (2020-12 if available)."""
    import jsonschema
    return (getattr(jsonschema, "Draft202012Validator", None)
            or getattr(jsonschema, "Draft7Validator"))


# ======================================================================== the tool table
def test_every_guard_is_either_exported_or_excluded_with_a_reason():
    names = {cls.name for cls in api.registry()}
    assert set(EXPORTED) | set(EXCLUDED) == names
    assert not set(EXPORTED) & set(EXCLUDED)
    assert set(EXCLUDED) == CANNOT_CROSS_JSON
    for name, exc in EXCLUDED.items():
        assert exc.reason and name in str(exc)
        assert exc.inputs, f"{name} must name the input that blocks it"
        for keyword, why in exc.inputs:
            assert keyword in api.get(name).required
            assert why and why.isascii()
        assert "callable" in exc.reason or "live" in exc.reason
        assert f"get('{name}')" in exc.reason, "the reason must say what to do instead"


def test_the_excluded_guards_are_the_ones_that_need_a_python_object():
    # assert_causal must RE-RUN the signal function on perturbed bars; a JSON description
    # of a function is not that function. Same for the indicator and the fold runner.
    assert "fn" in dict(EXCLUDED["assert_causal"].inputs)
    assert "indicator" in dict(EXCLUDED["warmup_probe"].inputs)
    assert "run_fold" in dict(EXCLUDED["fold_leak_test"].inputs)
    assert "ResultCard" in EXCLUDED["result_manifest"].reason


def test_optional_inputs_that_cannot_cross_are_dropped_and_declared():
    purge = EXPORTED["purge_effect"]
    assert "score_fn" not in purge.schema["properties"]
    assert "score_fn" in dict(purge.omitted)
    assert "Not available over JSON" in purge.description and "score_fn" in purge.description
    probe = EXPORTED["contamination_probe"]
    assert {"items", "ask", "match"} <= set(dict(probe.omitted))
    assert not set(dict(probe.omitted)) & set(probe.schema["properties"])


def test_tool_count_and_naming():
    assert len(TOOLS) == 7 + len(EXPORTED)          # catalogue + check_backtest + one/guard
    assert len(tool_names()) == len(set(tool_names()))
    for name in tool_names():
        # MCP 2026-07-28: 1-128 chars, ASCII letters, digits, _ - . only.
        assert 1 <= len(name) <= 128 and name.replace("_", "").replace("-", "").isalnum()
    for guard, spec in EXPORTED.items():
        assert spec.tool == f"check_{guard}" and spec.tool in tool_names()
    for guard in EXCLUDED:
        assert f"check_{guard}" not in tool_names()


# ============================================================================== schemas
@pytest.mark.skipif(not has_module("jsonschema"), reason="jsonschema is not installed")
@pytest.mark.parametrize("tool", TOOLS, ids=[t["name"] for t in TOOLS])
def test_every_schema_is_valid_json_schema_and_round_trips(tool):
    s = tool["input_schema"]
    _validator().check_schema(s)
    assert s["type"] == "object"                    # MCP requires an object at the root
    assert isinstance(s.get("properties"), dict)
    assert isinstance(s.get("required", []), list)
    assert set(s.get("required", [])) <= set(s["properties"])
    text = json.dumps(tool, allow_nan=False)        # NaN/Infinity are not JSON
    assert json.loads(text) == tool
    assert text.isascii(), "tool definitions print on a stock Windows console"


def test_every_ref_resolves_inside_its_own_schema():
    """Each tool travels alone, so a $ref must point at a $defs entry the tool ships."""
    for tool in TOOLS:
        s = tool["input_schema"]
        refs: set[str] = set()
        tool_schema._refs_used(s.get("properties", {}), refs)
        assert refs <= set(s.get("$defs", {})), tool["name"]
        for key in s.get("$defs", {}):
            assert key in refs, f"{tool['name']} ships an unused $defs/{key}"


@pytest.mark.skipif(not has_module("jsonschema"), reason="jsonschema is not installed")
def test_a_real_payload_validates_against_its_schema():
    v = _validator()
    v(EXPORTED["rf_convention"].schema).validate({"returns": {"values": [0.1, -0.2]},
                                                  "rf": 0.05})
    v(EXPORTED["cost_curve"].schema).validate(
        {"returns": {"index": ["2024-01-02"], "values": [0.01]}, "turnover": 0.1,
         "cost_bps": 10.0})
    with pytest.raises(Exception):                  # jsonschema.ValidationError
        v(EXPORTED["rf_convention"].schema).validate({"returns": {"values": [0.1]},
                                                      "rf": 0.05, "nope": 1})


def test_schemas_are_derived_from_the_guard_not_written_down():
    """Every declared input reaches the schema, and every schema property is a real input."""
    for guard, spec in EXPORTED.items():
        cls = type(api.get(guard))
        inputs = set(cls.required) | set(cls.optional)
        blocked = set(dict(spec.omitted))
        assert set(spec.schema["properties"]) == inputs - blocked, guard
        # `required` follows the guard's own missing() - including its either/or forms
        for keyword in cls.required:
            if keyword not in blocked:
                assert keyword in spec.schema["required"], guard
        for keyword in cls.optional:
            assert keyword not in spec.schema["required"], guard


def test_descriptions_carry_the_docstring_and_the_owning_skill():
    for guard, spec in EXPORTED.items():
        assert spec.description.isascii() and len(spec.description) > 60
        assert f"read_skill('{spec.skill}')" in spec.description
        assert "Required:" in spec.description
    docs = tool_schema.input_docs("cost_curve")
    assert "GROSS" in docs["returns"] and "round-trip" in docs["bps"].lower()
    # the per-property description comes from that same block, never from a second copy
    assert docs["returns"] in EXPORTED["cost_curve"].schema["properties"]["returns"]["description"]


def test_either_or_requirements_survive_into_the_description():
    assert "universe or (members + rebalance_dates)" in EXPORTED["pit_universe"].description
    assert "panel or panels" in EXPORTED["brinson_attribution"].description
    assert "ledger or sharpes" in EXPORTED["trial_ledger"].description
    # ... and are NOT invented as JSON properties
    assert "panel or panels" not in EXPORTED["brinson_attribution"].schema["required"]


# =============================================================================== exports
def test_the_three_wire_shapes_are_renames_of_one_table():
    a, o, oc, m = anthropic_tools(), openai_tools(), openai_tools(chat=True), mcp_tools()
    assert len(a) == len(o) == len(oc) == len(m) == len(TOOLS)
    for t, at, ot, oct_, mt in zip(TOOLS, a, o, oc, m):
        assert set(at) == {"name", "description", "input_schema"}
        assert at["name"] == t["name"] and at["input_schema"] == t["input_schema"]
        assert ot["type"] == "function" and ot["parameters"] == t["input_schema"]
        assert oct_ == {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["input_schema"]}}
        assert set(mt) == {"name", "description", "inputSchema"}
        assert mt["inputSchema"] == t["input_schema"]
    for shape in (a, o, oc, m):
        assert json.loads(json.dumps(shape, allow_nan=False)) == shape


def test_export_cli_prints_json(capsys):
    from fin_skills.tools.export import main
    assert main(["--json", "--format", "mcp", "--indent", "0"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert len(parsed) == len(TOOLS) and parsed[0]["name"] == TOOLS[0]["name"]
    assert main(["--excluded"]) == 0
    out = capsys.readouterr().out
    assert f"{len(EXCLUDED)} guard(s) cannot be driven over JSON" in out
    for guard in EXCLUDED:
        assert guard in out


# ============================================================================== payloads
def test_a_series_survives_the_round_trip_exactly():
    idx = pd.bdate_range("2022-01-03", periods=60)          # a DatetimeIndex WITH a freq
    s = pd.Series(np.random.default_rng(3).normal(size=60), index=idx, name="returns")
    wire = json.loads(json.dumps(series_to_payload(s), allow_nan=False))
    assert wire["freq"] == "B", "freq must travel: assert_series_equal compares it"
    pd.testing.assert_series_equal(to_series(wire, "returns"), s)


def test_a_series_survives_nan_and_infinity():
    s = pd.Series([1.0, float("nan"), float("inf"), float("-inf"), -2.5], name="x")
    wire = json.loads(json.dumps(series_to_payload(s), allow_nan=False))
    assert wire["values"] == [1.0, None, "Infinity", "-Infinity", -2.5]
    pd.testing.assert_series_equal(to_series(wire, "x"), s)


def test_a_dataframe_survives_the_round_trip_exactly():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=40, name="date")
    df = pd.DataFrame({"close": rng.normal(100, 1, 40), "volume": np.arange(40),
                       "tag": list("ab") * 20}, index=idx)
    wire = json.loads(json.dumps(frame_to_payload(df), allow_nan=False))
    pd.testing.assert_frame_equal(to_frame(wire, "bars"), df)


def test_bare_lists_and_records_are_accepted():
    pd.testing.assert_series_equal(to_series([1.0, 2.0], "x"), pd.Series([1.0, 2.0]))
    pd.testing.assert_frame_equal(to_frame([{"a": 1.0}, {"a": 2.0}], "df"),
                                  pd.DataFrame({"a": [1.0, 2.0]}))


def test_the_size_cap_is_a_documented_constant():
    assert payloads.MAX_ARGUMENT_BYTES == 8_000_000 and payloads.MAX_POINTS == 1_000_000
    with pytest.raises(TypeError, match="MAX_POINTS"):
        to_series({"values": [0.0] * (payloads.MAX_POINTS + 1)}, "returns")
    huge = {"returns": {"values": ["x" * 1000] * 9000}}
    with pytest.raises(TypeError, match="MAX_ARGUMENT_BYTES"):
        payloads.check_size(huge)


def test_evidence_is_reduced_to_json_safe_scalars_and_small_tables():
    big = pd.DataFrame({"a": np.arange(500.0)})
    out = payloads.encode({"table": big, "series": pd.Series(np.arange(2000.0)),
                           "n": np.int64(3), "f": np.float64(float("nan")),
                           "when": pd.Timestamp("2024-01-02"), "obj": object()},
                          "evidence")
    assert len(out["table"]["records"]) == payloads.MAX_EVIDENCE_ROWS
    assert out["table"]["truncated"] == {"kept": payloads.MAX_EVIDENCE_ROWS, "of": 500}
    assert len(out["series"]["values"]) == payloads.MAX_EVIDENCE_ITEMS
    assert out["n"] == 3 and out["f"] is None and out["when"].startswith("2024-01-02")
    assert isinstance(out["obj"], str)
    json.dumps(out, allow_nan=False)


# ================================================== the verdict must match the Python one
def _same(result: dict, direct) -> None:
    assert result["passed"] is direct.passed
    assert result["guard"] == direct.guard and result["skill"] == direct.skill
    assert [(f["severity"], f["message"], f["where"]) for f in result["findings"]] == \
           [(f.severity, f.message, f.where) for f in direct.findings]
    json.dumps(result, allow_nan=False)


@pytest.fixture
def run():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=400)
    return {"idx": idx,
            "returns": pd.Series(rng.normal(0.0008, 0.01, 400), index=idx, name="strategy"),
            "turnover": pd.Series(np.abs(rng.normal(0.1, 0.02, 400)), index=idx)}


def test_cost_curve_through_the_tool_matches_the_api_and_fails(run):
    kwargs = dict(returns=run["returns"], turnover=run["turnover"], cost_bps=10.0)
    direct = api.get("cost_curve").run(**kwargs)
    assert direct.passed is False, "this fixture is the FAILING case"
    got = call_tool("check_cost_curve", {"returns": series_to_payload(run["returns"]),
                                         "turnover": series_to_payload(run["turnover"]),
                                         "cost_bps": 10.0})
    _same(got, direct)
    assert got["n_errors"] == 1 and "breakeven" in got["findings"][0]["message"]
    assert "curve" in got["evidence"] and got["elapsed_s"] >= 0.0


def test_rf_convention_through_the_tool_matches_the_api_and_passes(run):
    direct = api.get("rf_convention").run(returns=run["returns"], rf=0.05)
    assert direct.passed is True, "this fixture is the PASSING case"
    got = call_tool("check_rf_convention", {"returns": series_to_payload(run["returns"]),
                                            "rf": 0.05})
    _same(got, direct)


def test_safe_asof_through_the_tool_matches_the_api_both_ways():
    ts = pd.Timestamp
    left = pd.DataFrame({"time": [ts("2024-01-02 09:30:01")], "symbol": ["AAA"],
                         "signal": [1]})
    right = pd.DataFrame({"time": [ts("2024-01-02 09:29:59"), ts("2024-01-02 09:30:01"),
                                   ts("2024-01-02 09:30:05")],
                          "symbol": ["AAA"] * 3, "px": [100.0, 101.0, 102.0]})
    wire = {"left": frame_to_payload(left), "right": frame_to_payload(right),
            "on": "time", "by": "symbol", "tolerance": "5min"}
    _same(call_tool("check_safe_asof", dict(wire)),
          api.get("safe_asof").run(left=left, right=right, on="time", by="symbol",
                                   tolerance="5min"))
    _same(call_tool("check_safe_asof", dict(wire, allow_exact_matches=True)),
          api.get("safe_asof").run(left=left, right=right, on="time", by="symbol",
                                   tolerance="5min", allow_exact_matches=True))


def test_greeks_convention_through_the_tool_matches_the_api():
    """A guard whose input is a mapping, not a frame - the other side of the boundary."""
    greeks = {"price": 5.0, "delta": 0.55, "gamma": 0.02, "vega": 0.20, "theta": -0.01}
    _same(call_tool("check_greeks_convention", {"greeks": greeks, "flag": "c"}),
          api.get("greeks_convention").run(greeks=greeks, flag="c"))


def test_check_backtest_runs_every_ready_guard(run):
    got = call_tool("check_backtest", {"slots": {
        "returns": series_to_payload(run["returns"]),
        "turnover": series_to_payload(run["turnover"]), "rf": 0.05}})
    assert "cost_curve" in got["ran"] and "rf_convention" in got["ran"]
    assert got["failed"] == ["cost_curve"] and got["passed"] is False
    assert "survivorship_audit" in got["skipped"]
    assert set(got["not_callable_over_json"]) == CANNOT_CROSS_JSON
    json.dumps(got, allow_nan=False)
    # the same slots, run through the library, reach the same guards
    bundle = api.Bundle(returns=run["returns"], turnover=run["turnover"], rf=0.05)
    ran = {r.guard for r in api.check(bundle) }
    assert {"cost_curve", "rf_convention"} <= ran


# ============================================================== bad input names the field
@pytest.mark.parametrize("args,pattern", [
    ({"returns": {"vals": [1.0]}, "turnover": 0.1}, r"returns: key 'vals'"),
    ({"returns": {"values": [1.0], "nope": 1}, "turnover": 0.1}, r"returns"),
    ({"turnover": 0.1}, r"missing required input\(s\) \['returns'\]"),
    ({"returns": [0.1, 0.2], "turnover": 0.1, "nope": 1}, r"unexpected argument\(s\) \['nope'\]"),
])
def test_a_bad_payload_raises_typeerror_naming_the_field(args, pattern):
    with pytest.raises(TypeError, match=pattern):
        call_tool("check_cost_curve", args)


def test_an_index_of_the_wrong_length_names_the_field():
    with pytest.raises(TypeError, match=r"returns: index has 2 label\(s\) but there are 3"):
        to_series({"index": ["2024-01-02", "2024-01-03"], "values": [1.0, 2.0, 3.0]},
                  "returns")


def test_calling_an_excluded_guard_explains_itself():
    with pytest.raises(TypeError, match="is not a tool: required input fn is a Python callable"):
        call_tool("check_assert_causal", {"df": [], "fn": "lambda d: d"})
    with pytest.raises(TypeError, match="no tool named"):
        call_tool("check_nothing", {})


# ========================================================================= catalogue tools
def test_catalogue_tools_are_compact_and_useful():
    skills = call_tool("list_skills", {})
    assert skills["count"] == len(fin_skills_names())
    assert "fin-core" in skills["plugins"]
    assert all(len(s["summary"]) <= 240 for s in skills["skills"])

    guards = call_tool("list_guards", {})
    assert guards["count"] == len(api.registry())
    assert guards["exported"] == len(EXPORTED) and guards["excluded"] == len(EXCLUDED)

    one = call_tool("describe_guard", {"name": "check_cost_curve"})   # tool name accepted
    assert one["guard"] == "cost_curve" and one["input_schema"] == EXPORTED["cost_curve"].schema
    assert "fin_skills.core.cost_curve.cost_curve" in one["wraps"]

    blocked = call_tool("describe_guard", {"name": "assert_causal"})
    assert blocked["tool"] is None and "callable" in blocked["excluded_because"]
    assert blocked["slots"] == {"fn": "signal_fn", "df": "bars"}

    found = call_tool("search_skills", {"terms": ["survivorship", "delisting"]})
    assert found["count"] >= 1 and all(h["match"] for h in found["skills"])

    text = call_tool("read_skill", {"name": "backtest-validation"})
    assert text["chars"] > 1000 and "cost_curve" in text["guards"]


def test_bundle_coverage_with_no_slots_returns_the_vocabulary():
    vocab = call_tool("bundle_coverage", {})
    names = {s["slot"] for s in vocab["vocabulary"]}
    assert {"returns", "turnover", "bars", "signal_fn"} <= names
    assert vocab["count"] == len(api.slots())
    assert "tol" in vocab["parameters"] and "returns" not in vocab["parameters"]


def test_bundle_coverage_says_what_one_more_slot_would_unlock():
    cov = call_tool("bundle_coverage", {"slots": ["returns", "turnover", "retruns"]})
    assert "cost_curve" in cov["ready"] and "cost_curve" in cov["ready_over_json"]
    assert cov["not_ready"]["rf_convention"] == ["rf"]
    assert "rf_convention" in cov["one_slot_away"]["rf"]
    assert cov["unknown"]["retruns"][0] == "returns"


def test_read_skill_and_describe_guard_suggest_a_neighbour():
    with pytest.raises(TypeError, match="did you mean"):
        call_tool("read_skill", {"name": "backtest-validaton"})
    with pytest.raises(TypeError, match="did you mean"):
        call_tool("describe_guard", {"name": "cost_curv"})


def fin_skills_names():
    import fin_skills
    return fin_skills.names()
