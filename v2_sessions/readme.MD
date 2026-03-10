# V2 — Sessions + Chat History + Context Window Management

## What's New vs V1

| Feature | V1 | V2 |
|---|---|---|
| Chat history | In-memory only | Saved to disk as JSON |
| Multiple conversations | ❌ | ✅ Create, switch, rename, delete |
| Survives page refresh | ❌ | ✅ |
| Context window awareness | ❌ (silent overflow) | ✅ Token meter + smart trim |

## Run It

```bash
ollama serve           # Terminal 1

cd v2_sessions
pip install -r requirements.txt
streamlit run app.py   # Terminal 2
```

## Project Structure

```
v2_sessions/
├── app.py                  ← UI + chat loop
├── ollama_client.py        ← Same as V1 (copy)
├── session_manager.py      ← Save/load/list/delete conversations
├── context_manager.py      ← Token counting + window trimming
├── data/
│   └── sessions/           ← JSON files, one per conversation
└── requirements.txt
```

## Key Concepts to Understand

### 1. Why JSON files and not a database?
At this scale, flat JSON files are perfect. Each session is one file.
You can open any conversation in a text editor and read it.
Later (V3+) we add ChromaDB for *vector* storage — a different need entirely.
Use the simplest storage that fits the problem.

### 2. The context window is not infinite
Every model has a token limit. Qwen3:1.7b = 4096 tokens.
In V1 we silently overflowed this. In V2 we:
- Count tokens (estimated) before every request
- Show usage as a % meter in the UI
- Trim oldest messages when near the limit
- Warn the user when trimming happens

### 3. What "trimming" actually means
We send a *copy* of the message list to the model, with old messages removed.
The FULL history is still saved to disk — the user can scroll back.
The model just can't "see" the oldest messages in its processing window.
This distinction is important: disk history ≠ model context.

### 4. Session state architecture
V1: `st.session_state.messages = [...]`
V2: `st.session_state.current_session = { id, title, messages, ... }`

Messages now live *inside* the session object.
This makes it trivial to switch between sessions — just swap the object.

## Questions to Ask Yourself Before V3

1. Open `data/sessions/` — read a JSON file. Does it make sense?
2. What happens if you open two browser tabs? (Two sessions at once?)
3. The token counting is approximate. When would an exact count matter?
4. How would you search through all your past conversations? (Hint: this is what V3's vector DB solves)

V3 introduces ChromaDB and RAG — giving the model access to documents
and making your entire conversation history searchable by meaning, not just text.