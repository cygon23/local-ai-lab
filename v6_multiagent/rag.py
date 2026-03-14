"""
rag.py
------
RAG pipeline using ChromaDB.

Storage strategy:
  - HF Spaces: uses /data/chroma (persistent volume, survives restarts)
  - Local:      uses ./data/chroma (relative to project folder)
  - Fallback:   in-memory ChromaDB if neither is writable

Embedding:
  - Uses nomic-embed-text via Ollama locally
  - Uses a lightweight sentence-transformers model on HF Spaces
    (no Ollama needed — runs directly in Python)
"""

import os
from pathlib import Path

# ── Storage path ──────────────────────────────────────────────────────

def get_chroma_path() -> str:
    # HF Spaces persistent storage
    if os.path.exists("/data"):
        p = Path("/data/chroma")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    # Local
    p = Path("data/chroma")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


# ── Embedding function ────────────────────────────────────────────────

def get_embeddings():
    """
    Returns an embedding function.
    Uses Ollama locally, sentence-transformers on HF Spaces.
    """
    from llm import is_ollama_available

    if is_ollama_available():
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(model="nomic-embed-text")
            # Quick test
            emb.embed_query("test")
            return emb
        except Exception:
            pass

    # HF Spaces / fallback — use sentence-transformers (no Ollama needed)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return None


def is_embedding_available() -> bool:
    return get_embeddings() is not None


def get_vectorstore():
    from langchain_chroma import Chroma
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError("No embedding model available.")

    try:
        # Try persistent storage first
        return Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
            persist_directory=get_chroma_path(),
        )
    except Exception:
        # Fall back to in-memory
        return Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
        )


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def get_chunk_count() -> int:
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0


def list_indexed_documents() -> list:
    try:
        vs = get_vectorstore()
        results = vs._collection.get(include=["metadatas"])
        sources = {}
        for meta in results.get("metadatas", []):
            src = meta.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return [{"source": k, "chunks": v} for k, v in sources.items()]
    except Exception:
        return []