"""
app.py — V5 LangChain Agent
-----------------------------
Same UI structure as V4 Extended.
Key differences:
  - Memory is LangChain ConversationBufferMemory per session
    (stored in st.session_state, not a manual message list)
  - Agent is AgentExecutor, not our custom while loop
  - Tool registration is automatic via @tool decorator

Tabs:
  💬 Chat    — agent with real-time trace rendering
  🧠 Memory  — long-term SQLite memory
  📚 Docs    — knowledge base management
  🔍 Compare — V4 vs V5 side-by-side explanation
"""

import streamlit as st
from pathlib import Path

from ollama_llm import is_ollama_running, get_available_models
from session_manager import create_session, save_session, load_session, list_sessions, delete_session
from rag import is_embedding_available, get_chunk_count, list_indexed_documents, get_vectorstore
from document_processor import process_document

# Phoenix observability — auto-instruments all LangChain + LangGraph calls
def setup_phoenix():
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentation
        register(
            project_name="local-ai-lab-v5",
            endpoint="http://localhost:6006/v1/traces",
            auto_instrument=False,
            batch=False,
        )
        LangChainInstrumentation().instrument()
        return True
    except Exception:
        return False
from agent import run_agent, AGENT_TOOLS
from memory import (
    get_conversation_memory, get_all_memories, store_memory,
    delete_memory, get_memory_stats, build_memory_prompt
)

DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Local AI Lab — V5 LangChain",
    page_icon="🦜",
    layout="wide"
)


# ── State ────────────────────────────────────────────────────────────

def init_state():
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "session_list" not in st.session_state:
        st.session_state.session_list = list_sessions()
    if "model" not in st.session_state:
        st.session_state.model = "qwen3:1.7b"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a capable AI agent. Think step by step. "
            "Use tools when needed. Be concise in your final answers. "
            "When you learn something important about the user or their projects, "
            "use store_memory_tool to remember it."
        )
    # LangChain memory objects keyed by session_id
    if "lc_memories" not in st.session_state:
        st.session_state.lc_memories = {}

init_state()


def get_or_create_lc_memory(session_id: str):
    """Get or create a ConversationBufferMemory for this session."""
    if session_id not in st.session_state.lc_memories:
        st.session_state.lc_memories[session_id] = get_conversation_memory()
    return st.session_state.lc_memories[session_id]


def refresh_sessions():
    st.session_state.session_list = list_sessions()

def start_new_session():
    st.session_state.current_session = create_session(
        model=st.session_state.model,
        system_prompt=st.session_state.system_prompt,
    )

def switch_session(sid):
    s = load_session(sid)
    if s:
        st.session_state.current_session = s


# ── SIDEBAR ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🦜 Local AI Lab")
    st.caption("V5 — LangChain Agent")

    if not is_ollama_running():
        st.error(" Ollama not running — run `ollama serve`")
        st.stop()

    ca, cb = st.columns(2)
    ca.success(" Ollama")
    cb.success(" Embed") if is_embedding_available() else cb.warning("⚠️ No embed")

    if st.session_state.get("phoenix_active"):
        st.info("🔭 Phoenix → [localhost:6006](http://localhost:6006)")

    mem_stats = get_memory_stats()
    if mem_stats["total"] > 0:
        st.caption(f"🧠 {mem_stats['total']} memories")

    st.divider()

    available_models = get_available_models()
    if available_models:
        st.session_state.model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index("qwen3:1.7b") if "qwen3:1.7b" in available_models else 0
        )

    st.session_state.system_prompt = st.text_area(
        "System Prompt", value=st.session_state.system_prompt, height=80
    )

    st.divider()
    st.subheader("Conversations")
    if st.button("✏️ New conversation", use_container_width=True, type="primary"):
        start_new_session()
        st.rerun()

    current_id = st.session_state.current_session["id"] if st.session_state.current_session else None
    for s in st.session_state.session_list:
        c1, c2 = st.columns([5, 1])
        label = f"**{s['title'][:28]}**" if s["id"] == current_id else s["title"][:28]
        if c1.button(label, key=f"s_{s['id']}", use_container_width=True):
            switch_session(s["id"])
            st.rerun()
        if c2.button("🗑️", key=f"ds_{s['id']}"):
            delete_session(s["id"])
            if s["id"] == current_id:
                st.session_state.current_session = None
            refresh_sessions()
            st.rerun()


# ── TABS ─────────────────────────────────────────────────────────────

tab_chat, tab_memory, tab_docs, tab_compare = st.tabs([
    "💬 Chat", "🧠 Memory", "📚 Docs", "🔍 V4 vs V5"
])


# ══ TAB 1: CHAT ══════════════════════════════════════════════════════

with tab_chat:
    if st.session_state.current_session is None:
        st.title("Local AI Lab — V5 LangChain")
        st.markdown("""
        **Same agent as V4. Now powered by LangChain.**

        The ReAct loop, tools, memory, and RAG are all the same.
        What changed is who runs the loop — LangChain's `AgentExecutor`
        instead of our hand-written while loop.

        | Tool | What it does |
        |---|---|
        | `web_search` | DuckDuckGo web search |
        | `fetch_webpage` | Read any public URL |
        | `call_api` | Full REST API calls |
        | `calculate` | Exact math |
        | `run_python` | Execute Python code |
        | `read_file` / `write_file` | File operations |
        | `search_knowledge_base` | Search indexed docs |
        | `store_memory_tool` | Save facts long-term |
        | `recall_memory_tool` | Retrieve past facts |

        Click **✏️ New conversation** to start.
        """)
    else:
        session = st.session_state.current_session
        messages = session["messages"]
        lc_memory = get_or_create_lc_memory(session["id"])

        st.title(session.get("title", "Agent Session"))
        st.caption(
            f"Model: `{session['model']}` • "
            f"Framework: LangChain AgentExecutor • "
            f"{get_chunk_count()} docs • "
            f"{get_memory_stats()['total']} memories"
        )
        st.divider()

        # Render message history
        for message in messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("steps"):
                    steps = message["steps"]
                    tool_steps = [s for s in steps if s["type"] == "tool_call"]
                    if tool_steps:
                        with st.expander(f"🔍 Agent used {len(tool_steps)} tool(s)", expanded=False):
                            for step in steps:
                                if step["type"] == "tool_call":
                                    st.caption(f"🔧 `{step.get('tool_name', '')}`")
                                    st.code(step["content"], language="json")
                                elif step["type"] == "observation":
                                    st.caption(f"👁️ Result")
                                    preview = step["content"]
                                    st.text(preview[:400] + ("..." if len(preview) > 400 else ""))
                                    st.divider()
                st.markdown(message["content"])

        # New input
        if prompt := st.chat_input("Give me a goal..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                step_log = []
                final_answer = ""
                status_ph = st.empty()
                trace_ph = st.empty()
                answer_ph = st.empty()
                tool_call_count = 0

                # Show loading indicator immediately
                status_ph.status("🤔 Agent is thinking...", state="running")

                for step in run_agent(
                    user_message=prompt,
                    model=session["model"],
                    base_system_prompt=session.get("system_prompt", ""),
                    memory=lc_memory,
                ):
                    step_dict = {
                        "type": step.type,
                        "content": step.content,
                        "tool_name": step.tool_name,
                    }
                    step_log.append(step_dict)

                    if step.type == "tool_call":
                        tool_call_count += 1
                        status_ph.status(f"🔧 Calling `{step.tool_name}`...", state="running")
                        with trace_ph.container():
                            with st.expander(f"🔍 Step {tool_call_count}: `{step.tool_name}`", expanded=True):
                                for s in step_log:
                                    if s["type"] == "tool_call":
                                        st.caption(f"🔧 `{s['tool_name']}`")
                                        st.code(s["content"], language="json")
                                    elif s["type"] == "observation":
                                        st.caption("👁️ Result")
                                        st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

                    elif step.type == "observation":
                        status_ph.status("👁️ Processing result...", state="running")

                    elif step.type == "answer":
                        final_answer = step.content
                        status_ph.status("✅ Done", state="complete")
                        answer_ph.markdown(final_answer)

                    elif step.type == "error":
                        status_ph.status("❌ Error", state="error")
                        st.error(f"⚠️ {step.content}")

            # Save to LangChain memory (for next turn)
            lc_memory.save_context(prompt, final_answer or "Agent completed.")

            messages.append({
                "role": "assistant",
                "content": final_answer or "Agent completed.",
                "steps": step_log,
            })
            session["messages"] = messages
            save_session(session)
            refresh_sessions()
            st.rerun()


# ══ TAB 2: MEMORY ════════════════════════════════════════════════════

with tab_memory:
    st.header("🧠 Long-Term Memory")
    st.caption("Persists across sessions. Auto-injected into every agent run.")

    stats = get_memory_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total memories", stats["total"])
    c2.metric("Categories", len(stats["categories"]))

    with st.expander("➕ Add memory manually"):
        m_key = st.text_input("Key", placeholder="e.g. project_name")
        m_val = st.text_area("Value", placeholder="The fact to remember", height=70)
        m_cat = st.selectbox("Category", ["user_profile", "projects", "preferences", "general"])
        if st.button("💾 Store", use_container_width=True):
            if m_key and m_val:
                st.success(store_memory(m_key, m_val, m_cat))
                st.rerun()
            else:
                st.warning("Key and value required.")

    memory_prompt = build_memory_prompt()
    if memory_prompt:
        with st.expander("👁️ What the agent sees"):
            st.text(memory_prompt)

    st.divider()
    all_memories = get_all_memories()
    if not all_memories:
        st.info("No memories yet.")
    else:
        for mem in all_memories:
            ci, cd = st.columns([6, 1])
            with ci:
                st.markdown(f"**[{mem['category']}]** `{mem['key']}`")
                st.caption(mem["value"])
            with cd:
                if st.button("🗑️", key=f"dm_{mem['id']}"):
                    delete_memory(mem["key"])
                    st.rerun()
            st.divider()


# ══ TAB 3: DOCS ══════════════════════════════════════════════════════

with tab_docs:
    st.header("📚 Knowledge Base")
    st.caption(f"{get_chunk_count()} chunks indexed • Same ChromaDB as V4")

    if is_embedding_available():
        uploaded = st.file_uploader("Upload document", type=["txt", "md", "pdf", "py"])
        if uploaded and st.button("📥 Index", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    from langchain_ollama import OllamaEmbeddings
                    save_path = DOCUMENTS_DIR / uploaded.name
                    save_path.write_bytes(uploaded.read())
                    chunks = process_document(str(save_path), uploaded.name)
                    vs = get_vectorstore()
                    vs.add_texts(
                        texts=[c["text"] for c in chunks],
                        metadatas=[c["metadata"] for c in chunks],
                    )
                    st.success(f"✅ {len(chunks)} chunks indexed")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
    else:
        st.warning("nomic-embed-text not available. Run: `ollama pull nomic-embed-text`")

    for doc in list_indexed_documents():
        st.caption(f"📄 {doc['source']}")


# ══ TAB 4: V4 VS V5 COMPARE ══════════════════════════════════════════

with tab_compare:
    st.header("🔍 V4 vs V5 — What actually changed")
    st.caption("Same agent. Different expression.")

    st.markdown("""
    | What | V4 (pure Python) | V5 (LangChain 1.x / LangGraph) |
    |---|---|---|
    | **The loop** | Manual `while True` in `agent.py` | `langgraph.prebuilt.create_react_agent` |
    | **Tool parsing** | Regex on `<tool_call>` tags | LangGraph message parsing |
    | **Tool registration** | Two dicts: `TOOL_FUNCTIONS` + `TOOL_SCHEMAS` | `@tool` decorator |
    | **Tool schema** | Written manually as JSON | Auto-generated from docstring + type hints |
    | **Conversation memory** | Manual message list + token trimming | `SimpleMemory` — message list passed directly to LangGraph |
    | **Memory injection** | Manual string concat into system prompt | `SystemMessage` prepended to message list |
    | **RAG** | Direct ChromaDB calls | `Chroma.as_retriever()` |
    | **Prompt** | f-string built manually | `SystemMessage` + `HumanMessage` list |
    | **Error handling** | Our try/except in the loop | LangGraph handles retries internally |
    | **Max iterations** | Our counter | LangGraph `recursion_limit` (default 25) |

    ---

    ### Note on LangChain versions
    LangChain 1.x (released 2025) is a major restructure.
    `AgentExecutor` and `ConversationBufferMemory` were removed from the main package.
    The new standard is `langgraph.prebuilt.create_react_agent` — which is the same
    ReAct loop, now backed by a proper state machine in LangGraph.
    This is what V5 uses.

    ---

    ### What the framework gave us
    - Less boilerplate — tool registration dropped from ~200 lines to ~20 with `@tool`
    - Tool schemas auto-generated from docstrings and type hints — no manual JSON
    - LangGraph state machine handles the loop — no manual iteration counter
    - `Chroma.as_retriever()` — RAG wired in one line instead of manual calls

    ### What we gave up
    - Full visibility into the exact prompt being constructed
    - Control over exactly when and how context is trimmed
    - Easier debugging — LangGraph adds abstraction layers

    ### The verdict
    LangChain 1.x + LangGraph is the direction the ecosystem is moving.
    Having built V4 by hand means you understand what every layer is doing —
    the state machine, the tool dispatch, the message list management.
    When something breaks in production, that knowledge is what saves you.

    That is the point of this lab.
    """)