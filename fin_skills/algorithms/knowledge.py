"""Searchable research inventory joined to the live adapter registry.

This is discovery, not a suitability score, trading recommendation or execution API.
No network access, package installation or model fitting occurs while searching.
"""
from collections import Counter
from copy import deepcopy
import re

from ._knowledge_data import ASSET_OVERRIDES, GAPS, GROUPS, RELATED_MODELS, SOURCES, VERIFIED_ON

SCOPE = ("Living inventory; not an exhaustive enumeration or evidence of profit. "
         "Counts are catalog entries, including backend variants, not distinct economic ideas. "
         "External means no complete model adapter; a native engine bridge may be available. "
         "Related models are components, not implementations of the full strategy.")
KINDS = ("strategy", "decision", "algorithm")
STATUSES = ("integrated", "external", "reference")


def _cards():
    from fin_skills.model_zoo import model_catalog
    models = {model["id"]: model for model in model_catalog()}
    linked = set()
    cards = []
    for group in GROUPS:
        source_id = group["source_id"]
        title, url = SOURCES[source_id]
        for row in group["rows"]:
            id_, name, name_zh, caution = row.strip().split("|")
            cards.append({k: deepcopy(v) for k, v in group.items() if k not in ("rows", "source_id")}
                         | dict(id=id_, name=name, name_zh=name_zh, caution=caution,
                                sources=[dict(id=source_id, title=title, url=url,
                                              verified_on=VERIFIED_ON,
                                              verification="primary documentation read; not executed")],
                                model_id=None, availability="not_integrated", operations=[],
                                related_model_ids=list(RELATED_MODELS.get(id_, [])),
                                evidence="method_existence", performance_evidence=None))
            if id_ in ASSET_OVERRIDES:
                cards[-1]["asset_classes"] = list(ASSET_OVERRIDES[id_])
            if source_id in ("qc", "options", "execution", "controls"):
                cards[-1]["engine_bridge"] = dict(
                    entrypoint="fin_skills.bridges.lean.LeanBacktest",
                    status="native_engine_sample_verified",
                    limitation="Caller supplies algorithm and data; this specific strategy is not verified.")
            if source_id in ("hummingbot", "avellaneda", "amm", "basis", "funding"):
                cards[-1]["engine_bridge"] = dict(
                    entrypoint="fin_skills.bridges.hummingbot." + (
                        "funding_strategy" if source_id == "funding" else "native_strategy"),
                    status="native_paper_quotes_verified" if id_ == "pure_market_making"
                           else "factory_registered_not_end_to_end_verified",
                    limitation="Requires caller-owned native runtime; no live connector acceptance test.")
            if id_ in models:
                model = models[id_]
                cards[-1].update(status="integrated", availability=model["status"],
                                 model_id=id_, operations=list(model["operations"]),
                                 json_run=model["json_run"], adapter_inputs=list(model["inputs"]),
                                 adapter_caution=model["caveat"], evidence="adapter_registered")
                linked.add(id_)
    for model in models.values():
        if model["id"] in linked:
            continue  # one canonical card; model: IDs remain accepted aliases
        # A numerical or signal adapter does not imply a complete trading strategy.
        cards.append(dict(id="model:" + model["id"], name=model.get("name", model["id"]),
                          name_zh="", kind="algorithm", family=model["task"],
                          asset_classes=["input_dependent"],
                          required_data=list(model.get("inputs", [])), status="integrated",
                          availability=model["status"], model_id=model["id"],
                          operations=list(model["operations"]),
                          json_run=model["json_run"], related_model_ids=[],
                          caution=model.get("caveat", ""),
                          sources=[dict(location=model.get("source", ""),
                                        verified_on=model.get("verified_on"),
                                        verification="inherited adapter metadata")],
                          evidence="adapter_registered", performance_evidence=None))
    return cards


def method_coverage():
    """Return exact inventory counts, filter vocabulary and explicit collection gaps."""
    cards = _cards()
    return dict(total=len(cards), by_kind=dict(Counter(c["kind"] for c in cards)),
                by_status=dict(Counter(c["status"] for c in cards)),
                by_family=dict(sorted(Counter(c["family"] for c in cards).items())),
                asset_classes=sorted({a for c in cards for a in c["asset_classes"]}),
                availability=dict(Counter(c["availability"] for c in cards)),
                curated_source_count=len(SOURCES), curated_verified_on=VERIFIED_ON,
                gaps=list(GAPS), scope=SCOPE)


def get_method(method_id):
    """Read a complete card; integrated model IDs have the 'model:' prefix."""
    if not isinstance(method_id, str) or not method_id:
        raise TypeError("method_id must be a nonempty string")
    for card in _cards():
        if card["id"] == method_id or method_id == "model:" + str(card["model_id"]):
            return card
    raise KeyError(f"unknown method: {method_id!r}; use search_methods")


def search_methods(query="", *, kind=None, family=None, asset_class=None, status=None,
                   limit=20, offset=0):
    """Case-insensitive AND substring search, then stable relevance/ID pagination.

    Chinese labels and English labels are indexed. No semantic search is implied.
    Asset filtering is exact; input_dependent and multi_asset do not assert that an
    implementation accepts arbitrary instruments. Search without that filter to see them.
    """
    if not isinstance(query, str) or len(query) > 1000:
        raise TypeError("query must be a string of at most 1000 characters")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    cards = _cards()
    allowed = dict(kind=KINDS, status=STATUSES, family={c["family"] for c in cards},
                   asset_class={a for c in cards for a in c["asset_classes"]})
    filters = dict(kind=kind, family=family, asset_class=asset_class, status=status)
    for key, value in filters.items():
        if value is not None and (not isinstance(value, str) or value not in allowed[key]):
            raise ValueError(f"{key} must be one of {sorted(allowed[key])}")
    terms = re.findall(r"\S+", query.casefold())
    hits = []
    for card in cards:
        if any(value is not None and (value not in card["asset_classes"]
               if key == "asset_class" else card[key] != value)
               for key, value in filters.items()):
            continue
        title = " ".join(str(card[k]) for k in ("id", "name", "name_zh")).casefold()
        body = " ".join([title, card["family"], card["caution"],
                         *card["required_data"], *card["asset_classes"],
                         *card["related_model_ids"]]).casefold()
        if all(term in body for term in terms):
            score = sum(2 if term in title else 1 for term in terms)
            hits.append((score, card))
    hits.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    selected = [card for _, card in hits[offset:offset + limit]]
    # Compact discovery: source details and cautions are available via get_method.
    fields = ("id", "name", "name_zh", "kind", "family", "status", "availability",
              "model_id", "required_data")
    return dict(methods=[{k: c[k] for k in fields} for c in selected],
                total=len(hits), offset=offset, count=len(selected),
                next_offset=offset + limit if offset + limit < len(hits) else None,
                ranking="lexical relevance, not suitability or expected return", scope=SCOPE)


def validate_inventory():
    """Offline consistency checks used by tests and the catalog export script."""
    from fin_skills.model_zoo import model_catalog
    cards = _cards()
    ids = [c["id"] for c in cards]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate method ID")
    models = {m["id"] for m in model_catalog()}
    for c in cards:
        if not re.fullmatch(r"(?:model:)?[a-z][a-z0-9_]*", c["id"]):
            raise ValueError(f"invalid method ID: {c['id']}")
        if c["kind"] not in KINDS or c["status"] not in STATUSES:
            raise ValueError(f"invalid taxonomy: {c['id']}")
        if not c["required_data"] or not c["caution"] or not c["sources"]:
            raise ValueError(f"incomplete card: {c['id']}")
        if not set(c["related_model_ids"]) <= models:
            raise ValueError(f"unknown related model: {c['id']}")
        if c["status"] != "integrated" and (c["model_id"] or c["operations"]):
            raise ValueError(f"reference masquerading as adapter: {c['id']}")
    return dict(valid=True, total=len(cards))
