"""
tools/knowledge.py
------------------
Wraps V3's RAG pipeline as a tool the agent can call.

THIS IS THE BRIDGE BETWEEN RAG AND AGENTS.

In V3, RAG was always ON — every question triggered retrieval.
In V4, RAG becomes a TOOL — the agent DECIDES when to search.

Why is this better?
  - For "what is 2+2?", the agent won't waste time searching documents
  - For "what did my FishHappy plan say about pricing?", it will
  - The agent applies judgment — retrieval only when relevant

This is a key architectural evolution:
  V3: retrieval is automatic (always happens)
  V4: retrieval is agentic (happens when the agent chooses)
"""

from embedder import get_embedding
from vector_store import search, get_chunk_count


SIMILARITY_THRESHOLD = 0.45
TOP_K = 4


def search_knowledge_base(query: str) -> str:
    """
    Search indexed documents for information relevant to the query.
    Returns formatted results as a string for the agent to read.
    """
    if get_chunk_count() == 0:
        return "Knowledge base is empty — no documents have been indexed yet."

    try:
        query_vector = get_embedding(query)
        results = search(query_embedding=query_vector, top_k=TOP_K)

        # Filter by threshold
        relevant = [r for r in results if r["score"] >= SIMILARITY_THRESHOLD]

        if not relevant:
            return f"No relevant information found in knowledge base for: '{query}'"

        lines = [f"Found {len(relevant)} relevant chunks:\n"]
        for i, chunk in enumerate(relevant, 1):
            lines.append(
                f"[Source {i}: {chunk['source']} | similarity: {chunk['score']}]\n"
                f"{chunk['text']}\n"
            )

        return "\n---\n".join(lines)

    except Exception as e:
        return f"Error searching knowledge base: {e}"