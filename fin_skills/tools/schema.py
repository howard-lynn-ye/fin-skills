"""JSON Schema for every registered guard, derived - never written down twice.

A hand-maintained schema goes stale the first time somebody adds an optional argument to a
guard. So nothing here is hand-maintained: every property comes from the guard itself -
its `required` / `optional` tuples, the annotations on `check()`, the Bundle slot the
keyword resolves to, and the `Inputs` block of its class docstring. Add a guard to the
registry and its tool appears with a schema; add an argument and the schema grows a
property. The only hand-written things in this module are the rules for turning a Python
type into a JSON one, and those are about JSON, not about any particular guard.

Two facts have to be told honestly rather than papered over:

  * **Some guards cannot be driven over JSON at all.** `assert_causal` needs the signal
    FUNCTION so it can re-run it on perturbed bars; `warmup_probe` needs the indicator;
    `fold_leak_test` needs the fold runner; `result_manifest` needs a live ResultCard.
    A function does not survive `json.dumps`, and a fabricated one would not be the
    caller's function, so these are EXCLUDED from the exported tool set with a stated
    reason. `excluded()` returns that reason; the exclusion is asserted by the tests.
  * **Some optional arguments cannot cross either** (`purge_effect`'s `score_fn`,
    `contamination_probe`'s `ask`). Those are dropped from the schema and listed in
    `Exported.omitted`, and the tool description says so, so a caller is never told a
    knob exists that it cannot reach.

Everything else maps: pandas objects through the payload shapes in
`fin_skills.tools.payloads`, scalars and strings directly, sequences and mappings
structurally, dates as ISO strings.
"""
from __future__ import annotations

import inspect
import re
import sys
from dataclasses import dataclass
from typing import Any

from fin_skills.api.base import Guard, registry
from fin_skills.api.bundle import DATA_SLOTS, _slot_of
from fin_skills.tools import payloads

_SLOT_KIND: dict[str, str] = {s.name: s.kind for s in DATA_SLOTS}

# ---------------------------------------------------------------- Python type -> JSON type
# Leaf types. Anything not here and not a container below is treated as an opaque Python
# object, which is exactly the signal that the guard (or the argument) cannot cross JSON.
_LEAF: dict[str, dict[str, Any]] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "complex": {"type": "number"},
    "Path": {"type": "string", "description": "A filesystem path on the machine running "
                                              "the tool."},
    "date": {"type": "string", "description": "ISO-8601 date, e.g. '2024-01-02'."},
    "datetime": {"type": "string", "description": "ISO-8601 date-time."},
    "Timestamp": {"type": "string", "description": "ISO-8601 date or date-time."},
}
_SEQUENCES = frozenset({"Sequence", "Iterable", "Iterator", "Collection", "List", "list",
                        "Tuple", "tuple", "Set", "set", "frozenset", "MutableSequence"})
_MAPPINGS = frozenset({"Mapping", "MutableMapping", "Dict", "dict"})
_UNKNOWN = frozenset({"Any", "object", "", "None", "NoneType"})

# The payload shapes go in each tool schema's `$defs` and are referenced, not inlined:
# inlined, the Series definition alone repeats about thirty times and triples the size of
# the tool listing a model has to read. JSON Schema 2020-12 - the dialect MCP defaults to -
# allows `$ref` with sibling keywords, so a property keeps its own `description`.
REF_SERIES: dict[str, Any] = {"$ref": "#/$defs/series"}
REF_FRAME: dict[str, Any] = {"$ref": "#/$defs/frame"}
_DEFS: dict[str, dict[str, Any]] = {"series": payloads.SERIES_SCHEMA,
                                    "frame": payloads.FRAME_SCHEMA}

# The Bundle slot kinds, for the fallback when an annotation says only `Any` / `object`.
_KIND_SCHEMA: dict[str, dict[str, Any]] = {
    "series": REF_SERIES,
    "frame": REF_FRAME,
    "scalar": {"type": "number"},
    "text": {"type": "string"},
    "date": {"type": "string", "description": "ISO-8601 date, e.g. '2024-01-02'."},
    "sequence": {"type": "array"},
    "mapping": {"type": "object"},
    "any": {},
}


def _split_top(text: str, sep: str = "|") -> list[str]:
    """Split on `sep` at bracket depth zero - `Iterable[Mapping[str, Any] | Sequence[Any]]`
    is one member, not two."""
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse(text: str) -> tuple[str, list[str]]:
    """'Mapping[str, Any]' -> ('Mapping', ['str', 'Any']). Dotted prefixes are dropped."""
    text = text.strip().strip("'\"")
    i = text.find("[")
    base, args = (text, []) if i == -1 else (text[:i], _split_top(text[i + 1:-1], ","))
    return base.strip().split(".")[-1], args


def _qualname(name: str, module: str | None) -> str:
    """Best-effort dotted name for an opaque class, so the reason is specific."""
    mod = sys.modules.get(module or "")
    obj = getattr(mod, name, None) if mod is not None else None
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    return name


def _leaf_or_container(text: str, module: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """One union member -> (schema, None) if it crosses JSON, else (None, why it does not)."""
    base, args = _parse(text)
    if base in _UNKNOWN:
        return None, None                                     # unknown, not un-crossable
    if base == "Callable":
        return None, f"a Python callable ({text})"
    if base == "Series":
        return dict(REF_SERIES), None
    if base == "DataFrame":
        return dict(REF_FRAME), None
    if base == "ndarray":
        return {"type": "array", "items": {"type": ["number", "null"]},
                "description": "A 1-D array of numbers."}, None
    if base == "DatetimeIndex":
        return {"type": "array", "items": {"type": "string"},
                "description": "ISO-8601 timestamps."}, None
    if base in _LEAF:
        return dict(_LEAF[base]), None
    if base in _SEQUENCES:
        item_src = next((a for a in args if a.strip() != "..."), None)
        out: dict[str, Any] = {"type": "array"}
        if item_src:
            item, why = _fragment_of(item_src, module)
            if why:
                return None, f"a sequence of {why}"
            if item:
                out["items"] = item
        return out, None
    if base in _MAPPINGS:
        out = {"type": "object"}
        if len(args) == 2:
            val, why = _fragment_of(args[1], module)
            if why:
                return None, f"a mapping of {why}"
            if val:
                out["additionalProperties"] = val
        return out, None
    return None, f"a live {_qualname(base, module)} object"


def _fragment_of(text: str, module: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """A whole annotation (possibly a union) -> (schema, None) or (None, reason)."""
    members = [m for m in _split_top(text) if _parse(m)[0] not in ("None", "NoneType")]
    frags: list[dict[str, Any]] = []
    reasons: list[str] = []
    unknown = False
    for m in members:
        frag, why = _leaf_or_container(m, module)
        if why:
            reasons.append(why)
        elif frag is None:
            unknown = True
        elif frag not in frags:
            frags.append(frag)
    if frags:
        return (frags[0] if len(frags) == 1 else {"anyOf": frags}), None
    if unknown or not members:
        return None, None                                     # caller falls back to the slot
    return None, "; ".join(dict.fromkeys(reasons))


def json_type_of(guard: type[Guard] | Guard, keyword: str) -> tuple[dict[str, Any] | None, str | None]:
    """The JSON Schema for one guard input, or the reason it has none.

    Annotation first, Bundle slot kind second: `p_calm: object` is annotated too loosely
    to be useful, but the slot says `series`, so the schema says series.
    """
    cls = guard if isinstance(guard, type) else type(guard)
    ann = getattr(cls.check, "__annotations__", {}).get(keyword, "")
    frag, why = _fragment_of(str(ann), cls.__module__)
    if why:
        return None, why
    if frag is not None:
        return frag, None
    kind = _SLOT_KIND.get(_slot_of(cls.name, keyword), "any")
    if kind == "callable":
        return None, "a Python callable (the Bundle slot is a callable)"
    return dict(_KIND_SCHEMA.get(kind, {})), None


# ------------------------------------------------------------------ docstring -> per-input doc
_INPUT_LINE = re.compile(r"^(\s+)(\w+)\s*:\s*(.*)$")


def input_docs(guard: type[Guard] | Guard) -> dict[str, str]:
    """{input: one-line description} parsed from the `Inputs` block of the class docstring.

    Every guard writes that block; it is the only per-argument prose in the library, and
    copying it into a schema by hand is exactly the duplicate this module exists to avoid.
    """
    cls = guard if isinstance(guard, type) else type(guard)
    doc = cls.__doc__ or ""
    lines = doc.splitlines()
    out: dict[str, str] = {}
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "Inputs"), None)
    if start is None:
        return out
    base = len(lines[start]) - len(lines[start].lstrip())
    current: str | None = None
    for ln in lines[start + 1:]:
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent <= base:
            break
        m = _INPUT_LINE.match(ln)
        if m and m.group(2) in tuple(cls.required) + tuple(cls.optional):
            current = m.group(2)
            out[current] = m.group(3).strip()
        elif current:
            out[current] = (out[current] + " " + ln.strip()).strip()
    return {k: re.sub(r"\s+", " ", v).strip() for k, v in out.items()}


def _prose(guard: type[Guard]) -> tuple[str, str]:
    """(first line of the docstring, the paragraph after the Inputs block)."""
    lines = (guard.__doc__ or "").splitlines()
    first = lines[0].strip() if lines else ""
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "Inputs"), None)
    tail: list[str] = []
    if start is not None:
        base = len(lines[start]) - len(lines[start].lstrip())
        seen_end = False
        for ln in lines[start + 1:]:
            indent = len(ln) - len(ln.lstrip())
            if ln.strip() and indent <= base:
                seen_end = True
            if seen_end:
                tail.append(ln.strip())
    return first, re.sub(r"\s+", " ", " ".join(tail)).strip()


# ------------------------------------------------------------------------------ exported set
@dataclass(frozen=True)
class Exported:
    """One guard as a callable tool."""

    guard: str
    tool: str
    skill: str
    schema: dict[str, Any]
    description: str
    omitted: tuple[tuple[str, str], ...] = ()   # (optional input, why it cannot cross)


@dataclass(frozen=True)
class Excluded:
    """One guard that cannot be driven over JSON, and why."""

    guard: str
    skill: str
    reason: str
    inputs: tuple[tuple[str, str], ...] = ()

    def __str__(self) -> str:
        return f"{self.guard}: {self.reason}"


def _property_schema(cls: type[Guard], keyword: str, docs: dict[str, str],
                     default: Any) -> dict[str, Any] | None:
    frag, _why = json_type_of(cls, keyword)
    if frag is None:
        return None
    prop = dict(frag)
    parts = [docs.get(keyword, "")]
    if (default is not None and default != "" and keyword not in cls.required
            and "default" not in parts[0].lower()):     # the docstring often says it already
        parts.append(f"Default {default!r}.")
    text = " ".join(p for p in parts if p).strip()
    if text:
        prop["description"] = (text + " " + prop["description"]).strip() if "description" in prop \
            else text
    return prop


def _defaults(cls: type[Guard]) -> dict[str, Any]:
    try:
        sig = inspect.signature(cls.check)
    except (TypeError, ValueError):                            # pragma: no cover
        return {}
    return {n: (None if p.default is inspect.Parameter.empty else p.default)
            for n, p in sig.parameters.items() if n != "self"}


def _refs_used(node: Any, found: set[str]) -> None:
    """Collect the `$defs` keys a built schema actually points at, so each tool ships only
    the definitions it uses and is self-contained."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            found.add(ref.rsplit("/", 1)[-1])
        for v in node.values():
            _refs_used(v, found)
    elif isinstance(node, list):
        for v in node:
            _refs_used(v, found)


def _describe(cls: type[Guard], required: list[str], omitted: list[tuple[str, str]]) -> str:
    first, tail = _prose(cls)
    bits = [first, cls.summary]
    if tail:
        bits.append(tail)
    bits.append("Required: " + (", ".join(required) if required else "nothing"
                                ) + ".")
    if omitted:
        bits.append("Not available over JSON: "
                    + "; ".join(f"{k} ({why})" for k, why in omitted) + ".")
    bits.append(f"Owning skill: {cls.skill} - call read_skill('{cls.skill}') for the text.")
    return " ".join(b for b in bits if b)


def _build() -> tuple[dict[str, Exported], dict[str, Excluded]]:
    exported: dict[str, Exported] = {}
    excluded: dict[str, Excluded] = {}
    for cls in registry():
        docs = input_docs(cls)
        defaults = _defaults(cls)
        props: dict[str, Any] = {}
        blockers: list[tuple[str, str]] = []
        omitted: list[tuple[str, str]] = []
        for keyword in tuple(cls.required) + tuple(cls.optional):
            prop = _property_schema(cls, keyword, docs, defaults.get(keyword))
            if prop is None:
                _, why = json_type_of(cls, keyword)
                why = why or "no JSON representation"
                (blockers if keyword in cls.required else omitted).append((keyword, why))
                continue
            props[keyword] = prop
        if blockers:
            excluded[cls.name] = Excluded(
                guard=cls.name, skill=cls.skill, inputs=tuple(blockers),
                reason="required input " + " and ".join(f"{k} is {why}" for k, why in blockers)
                       + " - it cannot be sent as JSON, and a fabricated one would not be the "
                         "caller's. Use fin_skills.api.get("
                       + repr(cls.name) + ").run(...) in Python instead")
            continue
        required = cls().missing({})
        schema: dict[str, Any] = {"type": "object", "properties": props,
                                  "required": [k for k in required if k in props],
                                  "additionalProperties": False}
        used: set[str] = set()
        _refs_used(props, used)
        if used:
            schema["$defs"] = {k: _DEFS[k] for k in sorted(used)}
        exported[cls.name] = Exported(
            guard=cls.name, tool=f"check_{cls.name}", skill=cls.skill,
            description=_describe(cls, required, omitted),
            omitted=tuple(omitted), schema=schema)
    return exported, excluded


_CACHE: tuple[dict[str, Exported], dict[str, Excluded]] | None = None


def _cached() -> tuple[dict[str, Exported], dict[str, Excluded]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _build()
    return _CACHE


def exported() -> dict[str, Exported]:
    """{guard name: Exported} for every guard that can be driven over JSON."""
    return dict(_cached()[0])


def excluded() -> dict[str, Excluded]:
    """{guard name: Excluded} for every guard that cannot, with the reason."""
    return dict(_cached()[1])


def guard_schema(name: str) -> dict[str, Any]:
    """The JSON Schema of one guard's arguments. KeyError names the exclusion reason."""
    exp, exc = _cached()
    if name in exp:
        return exp[name].schema
    if name in exc:
        raise KeyError(f"{name} is not exported: {exc[name].reason}")
    raise KeyError(f"no guard named {name!r}")


def input_kinds(name: str) -> dict[str, str]:
    """{input: 'series' | 'frame' | 'other'} - what `payloads.decode` should do with each."""
    cls = {c.name: c for c in registry()}[name]
    out: dict[str, str] = {}
    for keyword in tuple(cls.required) + tuple(cls.optional):
        frag, why = json_type_of(cls, keyword)
        if frag is None:
            continue
        if frag == REF_SERIES:
            out[keyword] = "series"
        elif frag == REF_FRAME:
            out[keyword] = "frame"
        else:
            out[keyword] = "other"
    return out


__all__ = ["Excluded", "Exported", "excluded", "exported", "guard_schema", "input_docs",
           "input_kinds", "json_type_of"]
