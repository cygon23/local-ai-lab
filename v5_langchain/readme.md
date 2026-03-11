# V5 — LangChain Agent

Same agent as V4. Now powered by LangChain.

## What changed

The ReAct loop, tools, memory, and RAG are all the same.
What changed is who runs the loop:

- **V4**: hand-written `while True` loop in `agent.py`
- **V5**: `AgentExecutor` from LangChain

See the **V4 vs V5** tab in the UI for the full comparison.

## Stack

- Ollama + Qwen3:1.7b (local inference, no cloud)
- LangChain AgentExecutor (the loop)
- LangChain @tool decorator (tool registration)
- ConversationBufferMemory (session memory)
- ChromaDB + nomic-embed-text (RAG, reused from V4)
- SQLite (long-term memory across sessions)
- Streamlit (UI)

## Setup

```bash
# 1. Create and activate venv (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running
ollama serve

# 4. Pull models (if not already pulled)
ollama pull qwen3:1.7b
ollama pull nomic-embed-text

# 5. Run
streamlit run app.py
```

## Files

```
v5_langchain/
├── app.py                  Streamlit UI — 4 tabs
├── agent.py                LangChain AgentExecutor
├── tools.py                All tools via @tool decorator
├── memory.py               ConversationBufferMemory + SQLite
├── ollama_llm.py           Ollama wrapped as LangChain LLM
├── rag.py                  ChromaDB as LangChain retriever
├── session_manager.py      Session persistence (JSON)
├── document_processor.py   Document chunking
└── requirements.txt
```