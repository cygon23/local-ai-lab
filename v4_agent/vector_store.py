"""
vector_store.py
---------------
A clean wrapper around ChromaDB for storing and searching document chunks.

WHAT IS CHROMADB?
  ChromaDB is a local vector database.
  It stores your chunks + their embeddings on disk.
  Given a query vector, it finds the most similar stored vectors instantly.
  No server required — it runs as a Python library, data saved to a folder.

WHAT LIVES IN CHROMADB:
  For each chunk we store THREE things:
    1. The embedding vector  → used for similarity search
    2. The document text     → returned so we can inject it into the prompt
    3. Metadata              → source file, chunk index, etc.

  ChromaDB ties these together under a unique ID.

HOW SIMILARITY SEARCH WORKS:
  Query:  "What are fish prices in Zanzibar?"
    → embed the query → get vector Q

  ChromaDB computes cosine similarity between Q and every stored vector.
  Returns the top-K chunks with highest similarity scores.

  Cosine similarity = 1.0 means identical meaning, 0.0 means unrelated.
  We typically retrieve top 3-5 chunks (configurable).

COLLECTIONS:
  ChromaDB organizes data into "collections" — like tables in SQL.
  We use one collection per "knowledge base".
  In V3 we have one collection: "documents"
  In V4+ you could have: "documents", "conversation_memory", "tools", etc.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path


CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "documents"


def get_client() -> chromadb.PersistentClient:
    """
    Get or create a persistent ChromaDB client.
    PersistentClient saves data to disk at CHROMA_DB_PATH.
    Data survives restarts — same as our JSON sessions.
    """
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection() -> chromadb.Collection:
    """
    Get or create the documents collection.
    If the collection already exists, we get it.
    If not, we create it fresh.

    WHY NO EMBEDDING FUNCTION HERE?
      We compute embeddings ourselves in embedder.py using Ollama.
      We pass pre-computed vectors to ChromaDB directly.
      This gives us full control over the embedding model.
      The alternative (letting ChromaDB embed) locks you into their models.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Add document chunks and their embeddings to ChromaDB.
    Returns the number of chunks added.

    chunks: list of dicts from document_processor.process_document()
    embeddings: list of vectors from embedder.get_embeddings_batch()

    They must be the same length — chunk[i] corresponds to embedding[i].
    """
    collection = get_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "file_type": chunk.get("file_type", "unknown"),
        }
        for chunk in chunks
    ]

    # ChromaDB's add() handles duplicates with upsert behavior
    # If a chunk_id already exists, it will be updated
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)


def search(
    query_embedding: list[float],
    top_k: int = 4,
    source_filter: str = None
) -> list[dict]:
    """
    Find the top-K most similar chunks to a query embedding.

    Returns a list of result dicts:
    [
        {
            "text": "chunk content",
            "source": "filename.pdf",
            "chunk_index": 3,
            "score": 0.87       ← cosine similarity (1.0 = perfect match)
        },
        ...
    ]

    source_filter: optionally restrict search to a specific document.
    """
    collection = get_collection()

    # Build optional where clause to filter by source file
    where = {"source": source_filter} if source_filter else None

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception:
        return []

    # Flatten and format results
    # ChromaDB returns nested lists (one list per query)
    # We sent one query so we take index [0]
    output = []
    if results["documents"] and results["documents"][0]:
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(1 - distance, 3)  # convert distance to similarity
            })

    return output


def list_indexed_documents() -> list[dict]:
    """
    Return a summary of all documents currently in the vector store.
    Used to show the user what's been indexed.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # Get all metadata — no embeddings needed for listing
    all_items = collection.get(include=["metadatas"])

    # Aggregate by source file
    sources = {}
    for meta in all_items["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in sources:
            sources[source] = {
                "source": source,
                "chunk_count": 0,
                "file_type": meta.get("file_type", ""),
            }
        sources[source]["chunk_count"] += 1

    return list(sources.values())


def delete_document(source_filename: str) -> int:
    """
    Delete all chunks belonging to a specific source file.
    Returns number of chunks deleted.
    """
    collection = get_collection()

    # Get IDs of chunks from this source
    results = collection.get(
        where={"source": source_filename},
        include=[]
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        return len(results["ids"])
    return 0


def get_chunk_count() -> int:
    """Total number of chunks currently indexed."""
    return get_collection().count()