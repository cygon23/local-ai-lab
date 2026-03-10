"""
embedder.py
-----------
Converts text into vectors (embeddings) using Ollama's embedding endpoint.

WHAT IS AN EMBEDDING?
  An embedding is a list of floating point numbers that represents the
  *meaning* of a piece of text. Example:

    "The cat sat on the mat"  →  [0.12, -0.84, 0.33, 0.71, ...]
    "A feline rested on a rug" → [0.11, -0.82, 0.35, 0.69, ...]
    "Tanzania exports coffee"  → [-0.45, 0.23, -0.67, 0.12, ...]

  The first two are semantically similar → their vectors are close.
  The third is unrelated → its vector is far away.

  Measuring "closeness" between vectors = cosine similarity.
  ChromaDB does this search for us automatically.

WHY USE OLLAMA FOR EMBEDDINGS?
  Ollama can run embedding models locally — no API key, no cost, no data
  leaving your machine. We use nomic-embed-text which is small (274MB)
  and specifically trained for semantic search tasks.

  The chat model (qwen3:1.7b) and the embedding model are DIFFERENT models
  with different jobs:
    - Chat model   → generates text (a "decoder" model)
    - Embed model  → converts text to vectors (an "encoder" model)

HOW TO GET THE EMBEDDING MODEL:
  ollama pull nomic-embed-text
"""

import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    """
    Convert a single text string into an embedding vector.

    HOW THIS WORKS:
      We POST to Ollama's /api/embeddings endpoint.
      Ollama runs the embedding model and returns a vector.
      The vector length (dimensions) depends on the model:
        nomic-embed-text → 768 dimensions
        mxbai-embed-large → 1024 dimensions

    WHEN TO CALL THIS:
      - When indexing: once per chunk when you upload a document
      - When querying: once per user question before searching ChromaDB
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to Ollama. Is `ollama serve` running?")
    except KeyError:
        raise ValueError(f"Unexpected response from Ollama embeddings API: {response.text}")


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts one by one.

    WHY NOT TRULY BATCH?
      Ollama's embedding endpoint processes one text at a time.
      For real production systems you'd use a batched endpoint
      (OpenAI, HuggingFace Inference API, etc.) for speed.
      For our learning purposes, sequential is fine and simpler to understand.
    """
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings


def is_embedding_model_available() -> bool:
    """Check if the embedding model is pulled and ready."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        models = [m["name"] for m in response.json().get("models", [])]
        return any(EMBEDDING_MODEL in m for m in models)
    except Exception:
        return False