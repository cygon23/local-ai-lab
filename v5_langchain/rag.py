"""
rag.py
------
LangChain retriever over our existing ChromaDB vector store.

WHAT CHANGED FROM V4:
  V4: we called vector_store.py and rag_pipeline.py manually,
      built a retrieval context string, injected it into the prompt ourselves.

  V5: LangChain wraps ChromaDB as a Retriever object.
      The retriever plugs directly into a RetrievalQA chain or
      can be used as a tool (search_knowledge_base).
      LangChain handles the retrieval → prompt injection automatically
      when used as a chain. As a tool it works identically to V4.

  We reuse the same ChromaDB collection from V4 — no re-indexing needed.
  The embeddings are still generated via Ollama nomic-embed-text.

EMBEDDING NOTE:
  LangChain needs an Embeddings object to query ChromaDB.
  We use OllamaEmbeddings which calls the same nomic-embed-text model.
"""

from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

CHROMA_DIR = "data/chroma"
COLLECTION = "local_ai_lab"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def get_vectorstore() -> Chroma:
    """Returns a LangChain Chroma vectorstore wrapping our existing collection."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(k: int = 4):
    """Returns a LangChain retriever that fetches the top-k most relevant chunks."""
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


def is_embedding_available() -> bool:
    """Check if nomic-embed-text is available in Ollama."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any("nomic-embed-text" in m for m in models)
    except Exception:
        return False


def get_chunk_count() -> int:
    """Returns the number of chunks in the vector store."""
    try:
        vs = get_vectorstore()
        return vs._collection.count()
    except Exception:
        return 0


def list_indexed_documents() -> list[dict]:
    """Returns unique document sources in the vector store."""
    try:
        vs = get_vectorstore()
        results = vs._collection.get(include=["metadatas"])
        sources = {}
        for meta in results.get("metadatas", []):
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = True
        return [{"source": s} for s in sources]
    except Exception:
        return []


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the indexed document knowledge base for information relevant to the query.
    Use when the user asks about documents they have uploaded or indexed.
    Returns the most relevant text passages found.
    """
    try:
        retriever = get_retriever(k=4)
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found for that query."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[{i}] Source: {source}\n{doc.page_content[:500]}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Knowledge base search error: {e}"