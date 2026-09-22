"""In-memory BM25/cosine retrieval, with explicit optional embedding callbacks."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from .documents import (chunk_documents, coerce_documents, documents_from_skills,
                        json_copy, positive, timestamp)

MAX_INDEX_BYTES = 64 * 1024 * 1024


def _tokens(text):
    # Unicode words and individual Han characters; no implicit tokenizer model.
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", text.casefold())


def _vectors(value, rows, dimension=None):
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("embeddings must be a finite numeric matrix") from None
    if (matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] < 1
            or dimension is not None and matrix.shape[1] != dimension
            or not np.isfinite(matrix).all()):
        raise ValueError("embedding shape/dimension or finite-value check failed")
    scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("embedding rows must have nonzero norm")
    scaled = matrix / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


class RAGIndex:
    """``embedder(list[str])`` returns one finite, nonzero numeric row per text.

    The caller owns the callback's model/network policy. ``embedding_id`` identifies
    its coordinate space. Default BM25 retrieval requires no model or credentials.
    """
    def __init__(self, documents, *, chunk_size=1200, overlap=200,
                 embedder=None, embedding_id=None):
        self._documents = coerce_documents(documents)
        self._chunks = chunk_documents(self._documents, chunk_size=chunk_size, overlap=overlap)
        self.chunk_size, self.overlap = chunk_size, overlap
        self._counts = [Counter(_tokens(c["text"])) for c in self._chunks]
        self._lengths = [sum(c.values()) for c in self._counts]
        if embedder is not None and not callable(embedder):
            raise TypeError("embedder must be callable")
        if embedding_id is not None and (not isinstance(embedding_id, str) or not embedding_id.strip()):
            raise ValueError("embedding_id must be a nonempty string")
        if embedder is not None and embedding_id is None:
            raise ValueError("embedding_id is required with an embedder")
        if embedder is None and embedding_id is not None:
            raise ValueError("embedding_id requires an embedder")
        self._embedder, self.embedding_id = embedder, embedding_id
        self._vectors = None
        if embedder is not None and self._chunks:
            self._vectors = _vectors(embedder([c["text"] for c in self._chunks]), len(self._chunks))

    @classmethod
    def from_documents(cls, documents, **options):
        return cls(documents, **options)

    @classmethod
    def from_skills(cls, names=None, *, include_references=True, **options):
        return cls(documents_from_skills(names, include_references=include_references), **options)

    @property
    def chunks(self):
        return json_copy(self._chunks)

    def search(self, query, *, top_k=5, method="lexical", as_of=None, min_score=0.):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        positive(top_k, "top_k")
        if method not in ("lexical", "vector"):
            raise ValueError("method must be lexical or vector")
        if type(min_score) not in (int, float) or not math.isfinite(min_score) or min_score < 0:
            raise ValueError("min_score must be a finite nonnegative number")
        cutoff = timestamp(as_of, "as_of") if as_of is not None else None
        eligible = [i for i, c in enumerate(self._chunks) if cutoff is None or (
            c["available_at"] is not None and timestamp(c["available_at"], "available_at") <= cutoff)]
        if method == "vector" and self._embedder is None:
            raise ValueError("vector search requires the matching embedder callback")
        if not eligible:
            return []
        if method == "vector":
            if self._vectors is None:
                raise ValueError("index has no stored vectors")
            q = _vectors(self._embedder([query]), 1, self._vectors.shape[1])[0]
            scores = self._vectors[eligible] @ q
        else:
            terms = sorted(set(_tokens(query)))
            if not terms:
                return []
            # Future documents must not affect past ranks through corpus statistics.
            n = len(eligible)
            average = sum(self._lengths[i] for i in eligible) / n or 1.
            df = {t: sum(t in self._counts[i] for i in eligible) for t in terms}
            scores = []
            for i in eligible:
                norm = 1.5 * (.25 + .75 * self._lengths[i] / average)
                scores.append(sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5))
                    * self._counts[i][t] * 2.5 / (self._counts[i][t] + norm)
                    for t in terms if self._counts[i][t]))
        hits = [dict(json_copy(self._chunks[i]), score=float(score),
                     score_kind="bm25" if method == "lexical" else "cosine")
                for i, score in zip(eligible, scores) if score > min_score]
        hits.sort(key=lambda h: (-h["score"], h["id"]))
        return hits[:top_k]

    def save(self, path):
        """Save bounded JSON, including optional vectors; never overwrite an existing file."""
        payload = dict(schema_version=1, documents=[d.to_dict() for d in self._documents],
                       chunk_size=self.chunk_size, overlap=self.overlap, embedding_id=self.embedding_id,
                       vectors=self._vectors.tolist() if self._vectors is not None else None)
        encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_INDEX_BYTES:
            raise ValueError("index exceeds the JSON persistence size limit")
        path = Path(path)
        with path.open("xb") as stream:
            stream.write(encoded)
        return path

    @classmethod
    def load(cls, path, *, embedder=None, embedding_id=None):
        """Load JSON, never executable pickle. Vector queries need a matching callback."""
        with Path(path).open("rb") as stream:
            raw = stream.read(MAX_INDEX_BYTES + 1)
        if len(raw) > MAX_INDEX_BYTES:
            raise ValueError("index exceeds the JSON persistence size limit")
        try:
            payload = json_copy(json.loads(raw), "index")
        except (UnicodeError, ValueError, RecursionError):
            raise ValueError("invalid RAG index JSON") from None
        required = {"schema_version", "documents", "chunk_size", "overlap", "embedding_id", "vectors"}
        if (not isinstance(payload, dict) or set(payload) != required
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
            raise ValueError("unsupported RAG index schema")
        index = cls(payload["documents"], chunk_size=payload["chunk_size"], overlap=payload["overlap"])
        saved_id = payload["embedding_id"]
        if saved_id is not None and (not isinstance(saved_id, str) or not saved_id.strip()):
            raise ValueError("invalid saved embedding_id")
        if payload["vectors"] is not None:
            if saved_id is None or not index._chunks:
                raise ValueError("saved vectors require chunks and an embedding_id")
            index._vectors = _vectors(payload["vectors"], len(index._chunks))
        elif saved_id is not None and index._chunks:
            raise ValueError("saved embedding index is missing vectors")
        if embedder is not None:
            if not callable(embedder) or saved_id is None or embedding_id != saved_id:
                raise ValueError("supply a callable embedder and matching saved embedding_id")
            index._embedder = embedder
        elif embedding_id is not None:
            raise ValueError("embedding_id requires an embedder")
        index.embedding_id = saved_id
        return index
