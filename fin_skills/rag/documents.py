"""Text documents and deterministic chunks with source and availability metadata."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json


def json_copy(value, name="value"):
    def keys(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("object keys must be strings")
            for v in item.values():
                keys(v)
        elif isinstance(item, list):
            for v in item:
                keys(v)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("only JSON values are supported")
    try:
        keys(value)
        return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError(f"{name} must be finite JSON with string object keys") from None


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def timestamp(value, name):
    if not isinstance(value, (str, datetime)):
        raise ValueError(f"{name} must be an ISO timestamp with an explicit timezone")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError("timezone missing")
        return stamp.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an ISO timestamp with an explicit timezone") from None


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)
    available_at: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("document id must be a nonempty string")
        if not isinstance(self.text, str) or not isinstance(self.source, str):
            raise ValueError("document text and source must be strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("document metadata must be an object")
        object.__setattr__(self, "metadata", json_copy(self.metadata, "metadata"))
        object.__setattr__(self, "source", self.source or self.id)
        if self.available_at is not None:
            object.__setattr__(self, "available_at", timestamp(self.available_at, "available_at").isoformat())

    def to_dict(self):
        return json_copy(asdict(self), "document")


def coerce_documents(values):
    if isinstance(values, (str, bytes, dict)):
        raise TypeError("documents must be an iterable of Document objects or dictionaries")
    result, seen = [], set()
    for value in values:
        if isinstance(value, Document):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise TypeError("each document must be a Document or dictionary")
        doc = Document(**value)
        if doc.id in seen:
            raise ValueError(f"duplicate document id: {doc.id!r}")
        seen.add(doc.id)
        result.append(doc)
    return result


def chunk_documents(documents, *, chunk_size=1200, overlap=200):
    """Each chunk text is an exact source substring at Unicode character offsets."""
    positive(chunk_size, "chunk_size")
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be an integer between zero and chunk_size - 1")
    chunks = []
    for doc in coerce_documents(documents):
        start = 0
        while start < len(doc.text):
            end = min(len(doc.text), start + chunk_size)
            text = doc.text[start:end]
            if text.strip():
                identity = json.dumps([doc.id, start, end, text], ensure_ascii=True)
                digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
                chunks.append(dict(id=f"{doc.id}:{start}-{end}:{digest}", document_id=doc.id,
                                   text=text, source=doc.source, start=start, end=end,
                                   metadata=json_copy(doc.metadata), available_at=doc.available_at))
            if end == len(doc.text):
                break
            start = end - overlap
    return chunks


def documents_from_skills(names=None, *, include_references=True):
    """Load packaged skills. Verification dates are not evidence of historical availability."""
    import fin_skills
    if type(include_references) is not bool:
        raise TypeError("include_references must be a boolean")
    cards = {c["name"]: c for c in fin_skills.catalog()}
    if names is None:
        selected = sorted(cards)
    else:
        if isinstance(names, str):
            raise TypeError("names must be a sequence of skill names")
        selected = list(names)
        if any(not isinstance(n, str) or n not in cards for n in selected):
            raise ValueError("unknown skill name; fin_skills.names() lists the packaged skills")
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate skill names")
    documents = []
    for name in selected:
        metadata = {"skill": name, "plugin": cards[name].get("plugin"),
                    "verified_on": cards[name].get("verified_on")}
        documents.append(Document(f"skill/{name}/SKILL.md", fin_skills.load(name),
                                  source=f"fin-skills:{name}/SKILL.md", metadata=metadata))
        if include_references:
            for filename, text in sorted(fin_skills.references(name).items()):
                documents.append(Document(f"skill/{name}/references/{filename}", text,
                    source=f"fin-skills:{name}/references/{filename}",
                    metadata=dict(metadata, reference=filename)))
    return documents
