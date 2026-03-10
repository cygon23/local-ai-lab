# V1 — Basic Local AI Chat

The foundation. Everything else builds on this.

## What This Is
A chat interface connected to your local Ollama model (Qwen3:8b).
Responses stream token by token. Clean, minimal, fully understood.

## Run It

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py
```

Open your browser at http://localhost:8501

## What's Inside

```
v1_basic_chat/
├── app.py              ← Streamlit UI + session state management
├── ollama_client.py    ← Ollama HTTP API wrapper (reused in all versions)
└── requirements.txt
```

## Key Concepts You Should Understand After This

**1. How Ollama works**
Ollama runs a local HTTP server at `localhost:11434`.
You talk to it with standard POST requests — no special library needed.
Every model is just an endpoint.

**2. Streaming**
We don't wait for the full response. Ollama sends tokens one by one.
We yield each token and Streamlit renders it in real-time.
This is exactly how ChatGPT works — just streaming over a different API.

**3. Session State**
Streamlit reruns the entire script on every user interaction.
`st.session_state` is the only thing that persists between reruns.
Your chat history lives in `st.session_state.messages`.

**4. The messages format**
```python
[
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is..."},
    {"role": "user", "content": "Give me an example"}
]
```
This is the universal chat format (OpenAI-compatible).
Ollama, OpenAI, Groq, Anthropic — they all use this exact structure.
The model is stateless. "Memory" = sending the full history every time.

**5. System prompt**
The system message shapes the model's behavior.
Change it and the same model behaves completely differently.
This is your first control lever over a model.

## What's Missing (Done in V2)
- Chat history doesn't survive a page refresh
- No session management (can't have multiple conversations)
- No control over context window overflow (long chats will hit token limits)

## Questions to Ask Yourself
1. What happens if you send 100 messages? (hint: token limits)
2. What if you change the system prompt mid-conversation?
3. What if you want to save this conversation and load it tomorrow?

V2 answers all of these.