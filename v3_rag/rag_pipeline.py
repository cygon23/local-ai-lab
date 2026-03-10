"""
rag_pipeline.py
---------------
The core RAG loop: retrieve relevant chunks, build an augmented prompt,
generate an answer.

THIS IS THE HEART OF RAG.

The two modes this pipeline supports:

  MODE 1 — RAG MODE (documents are indexed):
    1. Embed the user's question
    2. Search ChromaDB for the most relevant chunks
    3. Build a prompt that includes those chunks as context
    4. Send to the LLM → get a grounded answer

  MODE 2 — PLAIN CHAT (no documents indexed, or RAG disabled):
    Skip retrieval. Send conversation history directly.
    Falls back to V2 behavior transparently.

WHY SEPARATE THIS FROM app.py?
  The RAG logic should be independent of the UI.
  In V4 the agent will call rag_pipeline.query() as a TOOL.
  If RAG is tangled into the UI, that becomes impossible.
  Always separate concerns.
"""

from embedder import get_embedding
from vector_store import search, get_chunk_count
from ollama_client import chat_stream


# Minimum similarity score to include a chunk
# Below this threshold, the chunk is probably not relevant
SIMILARITY_THRESHOLD = 0.45

# How many chunks to retrieve
TOP_K = 4


def build_rag_system_prompt(base_system_prompt: str, retrieved_chunks: list[dict]) -> str:
    """
    Inject retrieved chunks into the system prompt.

    PROMPT STRUCTURE:
      [Base instructions]
      [Retrieved context blocks]
      [Instructions for how to use the context]

    WHY INJECT INTO SYSTEM PROMPT AND NOT USER MESSAGE?
      The system prompt sets the frame for the entire conversation.
      Putting context there tells the model "this is your reference material".
      Putting it in the user message works too, but mixing it with the
      question can confuse the model about what to answer vs what is background.
      Both approaches are valid — this is a common RAG design choice.
    """
    if not retrieved_chunks:
        return base_system_prompt

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} | Relevance: {chunk['score']}]\n"
            f"{chunk['text']}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    augmented_prompt = f"""{base_system_prompt}

---
CONTEXT FROM YOUR KNOWLEDGE BASE:
The following excerpts were retrieved because they are relevant to the user's question.
Use this information to answer accurately. If the context doesn't contain the answer,
say so clearly — do not invent information.

{context_str}
---

When you use information from the context above, briefly mention which source it came from.
"""
    return augmented_prompt


def query(
    user_question: str,
    conversation_history: list[dict],
    model: str,
    base_system_prompt: str,
    use_rag: bool = True,
    source_filter: str = None,
) -> tuple:
    """
    The full RAG pipeline. Returns a generator for streaming + metadata.

    Returns: (stream_generator, retrieved_chunks, rag_was_used)

    PARAMETERS:
      user_question        → the current user input
      conversation_history → previous messages (for context)
      model                → Ollama model name
      base_system_prompt   → the user's custom system prompt
      use_rag              → toggle RAG on/off per query
      source_filter        → restrict retrieval to one document

    THE CALLER (app.py) receives:
      - A streaming generator to display tokens in real time
      - The list of retrieved chunks to show in the UI (transparency)
      - Whether RAG was actually used (for UI indicators)
    """
    retrieved_chunks = []
    rag_used = False

    if use_rag and get_chunk_count() > 0:
        # Step 1: Embed the question
        query_vector = get_embedding(user_question)

        # Step 2: Retrieve similar chunks
        raw_chunks = search(
            query_embedding=query_vector,
            top_k=TOP_K,
            source_filter=source_filter
        )

        # Step 3: Filter by similarity threshold
        retrieved_chunks = [c for c in raw_chunks if c["score"] >= SIMILARITY_THRESHOLD]

        if retrieved_chunks:
            rag_used = True

    # Step 4: Build (possibly augmented) system prompt
    final_system_prompt = build_rag_system_prompt(base_system_prompt, retrieved_chunks)

    # Step 5: Stream the response
    stream = chat_stream(
        messages=conversation_history,
        model=model,
        system_prompt=final_system_prompt,
    )

    return stream, retrieved_chunks, rag_used