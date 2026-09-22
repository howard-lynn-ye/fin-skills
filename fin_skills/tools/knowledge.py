"""Discovery tools for strategies, decisions and algorithms; no implicit execution."""
from fin_skills.algorithms.knowledge import get_method, method_coverage, search_methods

FUNCTIONS = {"search_quant_methods": search_methods, "get_quant_method": get_method,
             "quant_method_coverage": method_coverage}


def definitions():
    return [
        {"name": "search_quant_methods", "description": "Search strategies, decision rules and "
         "algorithms in English or Chinese. Distinguishes integrated adapters, external implementations "
         "and references. Results are discovery candidates, not profit rankings. Paginate with offset.",
         "input_schema": {"type": "object", "properties": {
             "query": {"type": "string", "maxLength": 1000},
             "kind": {"type": "string", "enum": ["strategy", "decision", "algorithm"]},
             "family": {"type": "string"}, "asset_class": {"type": "string"},
             "status": {"type": "string", "enum": ["integrated", "external", "reference"]},
             "limit": {"type": "integer", "minimum": 1, "maximum": 100},
             "offset": {"type": "integer", "minimum": 0}}, "additionalProperties": False}},
        {"name": "get_quant_method", "description": "Read method data requirements, cautions, "
         "primary sources and adapter links. Related model components are not full strategy implementations.",
         "input_schema": {"type": "object", "properties": {
             "method_id": {"type": "string", "minLength": 1}}, "required": ["method_id"],
             "additionalProperties": False}},
        {"name": "quant_method_coverage", "description": "Inspect catalog counts, filter vocabulary "
         "and collection gaps. The inventory is not exhaustive; counts include backend variants.",
         "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    ]
