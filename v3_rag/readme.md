# V3 — RAG (Retrieval-Augmented Generation)

## What's New vs V2

| Feature | V2 | V3 |
|---|---|---|
| Knowledge source | Model's training only | Model + your documents |
| Document upload | ❌ | ✅ PDF, TXT, MD, code files |
| Semantic search | ❌ | ✅ ChromaDB vector search |
| Source transparency | ❌ | ✅ Shows which chunks were retrieved |
| Grounded answers | ❌ | ✅ Cites document sources |

## Setup

```bash
# 1. Pull the embedding model (one time only)
ollama pull nomic-embed-text

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

## File Structure

```
v3_rag/
├── app.py                  ← UI + RAG orchestration
├── ollama_client.py        ← Ollama HTTP wrapper (from V1)
├── session_manager.py      ← Session persistence (from V2)
├── context_manager.py      ← Token counting (from V2)
├── embedder.py             ← NEW: text → vector via Ollama
├── document_processor.py   ← NEW: load + chunk documents
├── vector_store.py         ← NEW: ChromaDB wrapper
├── rag_pipeline.py         ← NEW: retrieve → inject → generate
├── data/
│   ├── sessions/           ← conversation JSON files
│   ├── documents/          ← uploaded source files
│   └── chroma_db/          ← ChromaDB vector database
└── requirements.txt
```

## The Two Models You're Now Running

| Model | Job | Command |
|---|---|---|
| `qwen3:1.7b` | Chat / generation | `ollama pull qwen3:1.7b` |
| `nomic-embed-text` | Embeddings | `ollama pull nomic-embed-text` |

These are DIFFERENT model types:
- Chat model = "decoder" = generates text token by token
- Embedding model = "encoder" = converts text to a fixed-size vector

## Key Concepts

### Chunking
A 10-page PDF becomes ~50 chunks of ~800 characters.
Each chunk gets its own embedding vector.
Search finds the right chunk, not the whole document.

### Cosine Similarity
How we measure "closeness" between vectors.
Score of 1.0 = identical meaning. 0.0 = completely unrelated.
We filter out chunks below 0.45 — they're not relevant enough.

### The RAG Prompt
We inject retrieved chunks into the SYSTEM prompt, not the user message.
The model sees: "Here is your reference material. Now answer this question."
This separation keeps the model focused.

### Toggle RAG On/Off
Use this to compare answers with and without your documents.
Ask the same question both ways — see the difference.
This builds intuition for when RAG helps vs. when it doesn't.

## Experiments to Run

1. Upload a PDF of any document. Ask questions about it.
   Notice how the model cites specific chunks.

2. Toggle RAG off. Ask the same question.
   The model answers from training only — watch it hallucinate or hedge.

3. Upload TWO documents on different topics.
   Ask about each. See how retrieval selects the right source.

4. Open `data/chroma_db/` — this is your vector database on disk.
   It's not human-readable (binary), but it's there. Real and local.

## What's Missing (Done in V4)

The model can only ANSWER questions. It can't:
- Decide to search for something
- Take actions based on the answer
- Chain multiple steps together
- Use tools (web search, code execution, file writing)

V4 gives the model hands. That's where agents begin.