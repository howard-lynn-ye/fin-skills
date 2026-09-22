"""Built-in RAG: source-aware retrieval and caller-owned generation, offline by default."""
from .documents import Document, chunk_documents, documents_from_skills
from .index import RAGIndex
from .pipeline import RAGPipeline

__all__ = ["Document", "RAGIndex", "RAGPipeline", "chunk_documents", "documents_from_skills"]
