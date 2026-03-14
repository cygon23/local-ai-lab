---
title: Local AI Lab V6 Multi-Agent
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Local AI Lab — V6 Multi-Agent

Part of the **Local AI Lab** series — building AI systems from scratch, one version at a time.

| Version | Focus |
|---|---|
| V1 | Basic Ollama chat |
| V2 | Sessions + context window |
| V3 | RAG + ChromaDB |
| V4 | ReAct agent — pure Python |
| V4 Extended | Web tools + observability + MCP + memory |
| V5 | LangChain + LangGraph + Phoenix tracing |
| **V6** | **Multi-agent — Orchestrator + Researcher + Executor** |

---

## Architecture

```
User goal
    ↓
🟣 Orchestrator  — analyzes goal, routes, synthesizes final answer
    ↓
🔵 Researcher    — web_search, fetch_webpage, knowledge base
🟠 Executor      — calculate, run_python, read/write files
    ↓
🟣 Orchestrator  — final answer
```

---

## Setup on HF Spaces

**Required secret:**
```
GROQ_API_KEY = your_key_here
```

Add it in: Space Settings → Variables and secrets → New secret

Get a free key at: https://console.groq.com (no billing required)

---

## Running locally

```bash
# Set your Groq key
export GROQ_API_KEY=your_key_here

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Or use Ollama without a Groq key:
```bash
ollama serve
streamlit run app.py
```

---

## Stack

```
LangGraph          multi-agent graph
LangChain          tool decorator, ChatGroq
Groq API           free hosted LLM (llama-3.1-8b-instant)
sentence-transformers  embeddings (no GPU needed)
ChromaDB           vector store — persistent on /data
SQLite             long-term memory — persistent on /data
Streamlit          UI
```

---

GitHub: [github.com/cygon23](https://github.com/cygon23)

Built in Arusha, Tanzania 