"""Discovery must never inflate reference coverage into runnable capability."""
import json
from datetime import date

import pytest

from fin_skills.algorithms import get_method, method_coverage, search_methods
from fin_skills.algorithms.knowledge import validate_inventory
from fin_skills.model_zoo import model_catalog
from fin_skills.tools import call_tool


def test_inventory_integrity_and_live_adapter_join():
    coverage = method_coverage()
    assert validate_inventory() == {"valid": True, "total": coverage["total"]}
    assert sum(coverage["by_kind"].values()) == coverage["total"]
    assert sum(coverage["by_status"].values()) == coverage["total"]
    assert coverage["gaps"]
    for model in model_catalog():
        card = get_method("model:" + model["id"])
        assert card["availability"] == model["status"]
        assert card["operations"] == model["operations"]
        assert card["model_id"] == model["id"]
        assert card["performance_evidence"] is None


def test_references_are_not_executable_and_related_components_are_explicit():
    card = get_method("cointegration_pairs")
    assert card["status"] == "external"
    assert card["model_id"] is None and card["operations"] == []
    assert card["related_model_ids"] == ["engle_granger"]
    assert get_method("model:engle_granger")["model_id"] == "engle_granger"
    assert get_method("fx_carry")["status"] == "reference"
    for c in _all_results(status="external"):
        detail = get_method(c["id"])
        assert detail["sources"][0]["url"].startswith("https://")
        date.fromisoformat(detail["sources"][0]["verified_on"])


def _all_results(**kwargs):
    results, offset = [], 0
    while offset is not None:
        page = search_methods(limit=13, offset=offset, **kwargs)
        results.extend(page["methods"])
        offset = page["next_offset"]
    return results


def test_pagination_filters_and_chinese_english_search():
    results = _all_results()
    assert len(results) == method_coverage()["total"]
    assert len({c["id"] for c in results}) == len(results)
    assert search_methods("协整", kind="strategy")["methods"][0]["id"] == "cointegration_pairs"
    assert search_methods("CARRY", kind="strategy", asset_class="fx")["methods"][0]["id"] == "fx_carry"
    assert search_methods("funding arbitrage")["methods"][0]["id"] == "funding_rate_arbitrage"
    assert all(c["family"] == "options" for c in _all_results(family="options"))
    assert search_methods("no_such_financial_method")["total"] == 0
    assert search_methods(offset=100000)["next_offset"] is None


@pytest.mark.parametrize("arguments", [dict(limit=True), dict(limit=101), dict(limit=0),
    dict(offset=-1), dict(query=None), dict(status="profitable"), dict(kind=[]),
    dict(asset_class="unknown"), dict(family="unknown")])
def test_invalid_queries(arguments):
    with pytest.raises((TypeError, ValueError)):
        search_methods(**arguments)


def test_unknown_id_and_fresh_results():
    with pytest.raises(KeyError):
        get_method("not_registered")
    card = get_method("cointegration_pairs")
    card["sources"].clear()
    card["related_model_ids"].clear()
    card["required_data"].clear()
    assert get_method("cointegration_pairs")["sources"]
    assert get_method("cointegration_pairs")["related_model_ids"]
    assert get_method("cointegration_pairs")["required_data"]


def test_global_tools_and_default_agent_discovery():
    from fin_skills.tools.agent import DEFAULT_TOOLS
    names = {"search_quant_methods", "get_quant_method", "quant_method_coverage"}
    assert names <= set(DEFAULT_TOOLS)
    page = call_tool("search_quant_methods", {"query": "协整", "kind": "strategy"})
    card = call_tool("get_quant_method", {"method_id": page["methods"][0]["id"]})
    assert card["status"] == "external"
    assert call_tool("quant_method_coverage")["total"] == method_coverage()["total"]
    json.dumps(card, allow_nan=False)


def test_export_is_reproducible_and_detects_stale_snapshot(tmp_path):
    from scripts.export_quant_methods import export
    export(tmp_path)
    export(tmp_path, check=True)
    data = json.loads((tmp_path / "quant_methods.json").read_text(encoding="utf-8"))
    assert "availability" not in data["coverage"]
    assert all(c["availability"] == "check_model_catalog_at_runtime"
               for c in data["methods"] if c["status"] == "integrated")
    (tmp_path / "QUANT_METHODS.md").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="stale export"):
        export(tmp_path, check=True)
