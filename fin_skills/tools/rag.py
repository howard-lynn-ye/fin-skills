"""Stateless, offline RAG context retrieval for JSON/MCP agents."""
from fin_skills.rag import RAGIndex, RAGPipeline, documents_from_skills


def retrieve_context(query, documents=None, skills=None, include_references=True,
                     top_k=5, chunk_size=1200, overlap=200, max_context_chars=12000, as_of=None):
    if documents is not None and not isinstance(documents, list):
        raise TypeError("documents must be a list of document objects")
    if skills is not None and not isinstance(skills, list):
        raise TypeError("skills must be a list of packaged skill names")
    if type(include_references) is not bool:
        raise TypeError("include_references must be a boolean")
    corpus = list(documents or [])
    if skills is not None or documents is None:
        corpus.extend(documents_from_skills(skills, include_references=include_references))
    index = RAGIndex.from_documents(corpus, chunk_size=chunk_size, overlap=overlap)
    return RAGPipeline(index).prepare(query, top_k=top_k, max_context_chars=max_context_chars,
                                      as_of=as_of)


FUNCTIONS = {"retrieve_context": retrieve_context}


def definitions():
    document = {"type": "object", "properties": {
        "id": {"type": "string", "minLength": 1}, "text": {"type": "string"},
        "source": {"type": "string"}, "metadata": {"type": "object"},
        "available_at": {"type": ["string", "null"]}},
        "required": ["id", "text"], "additionalProperties": False}
    return [{"name": "retrieve_context", "description":
             "Build cited RAG context from explicit document text and/or packaged skills using "
             "offline BM25. With neither source supplied, use all packaged skills. No file reads "
             "outside the packaged corpus, network or text generation. as_of excludes undated and "
             "future documents; timestamps require timezones. Python RAGPipeline adds optional "
             "embedding, reranking and generation callbacks.",
             "input_schema": {"type": "object", "properties": {
                 "query": {"type": "string", "minLength": 1},
                 "documents": {"type": "array", "items": document},
                 "skills": {"type": "array", "items": {"type": "string"}},
                 "include_references": {"type": "boolean"},
                 "top_k": {"type": "integer", "minimum": 1},
                 "chunk_size": {"type": "integer", "minimum": 1},
                 "overlap": {"type": "integer", "minimum": 0},
                 "max_context_chars": {"type": "integer", "minimum": 1},
                 "as_of": {"type": "string"}},
                 "required": ["query"], "additionalProperties": False}}]
