"""
app.py — V6 Multi-Agent + Hugging Face
----------------------------------------
Deploys to Hugging Face Spaces as a Streamlit app.
Supports HF Inference API (public) + Ollama (local fallback).

Tabs:
  🤖 Agent    — multi-agent chat with live agent trace
  🧠 Memory   — long-term SQLite memory
  📚 Docs     — knowledge base management
  🏗️ How it works — architecture explanation
"""

import streamlit as st
from pathlib import Path

from llm import get_llm, get_provider_label, is_groq_available, is_ollama_available
from session_manager import create_session, save_session, load_session, list_sessions, delete_session
from rag import is_embedding_available, get_chunk_count, list_indexed_documents, get_vectorstore
from document_processor import process_document
from graph import run_graph, stream_graph
from memory import get_all_memories, store_memory, delete_memory, get_memory_stats, build_memory_prompt

DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Local AI Lab — V6 Multi-Agent",
    page_icon="🤖",
    layout="wide"
)

# Phoenix observability
def setup_phoenix():
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentation
        register(
            project_name="local-ai-lab-v6",
            endpoint="http://localhost:6006/v1/traces",
            auto_instrument=False,
            batch=False,
        )
        LangChainInstrumentation().instrument()
        return True
    except Exception:
        return False


# ── State ─────────────────────────────────────────────────────────────

def init_state():
    if "startup_done" not in st.session_state:
        st.session_state.phoenix_active = setup_phoenix()
        st.session_state.startup_done = True
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "session_list" not in st.session_state:
        st.session_state.session_list = list_sessions()

init_state()


def refresh_sessions():
    st.session_state.session_list = list_sessions()

def start_new_session():
    st.session_state.current_session = create_session()

def switch_session(sid):
    s = load_session(sid)
    if s:
        st.session_state.current_session = s


# Agent color per role
AGENT_COLORS = {
    "orchestrator": "🟣",
    "researcher": "🔵",
    "executor": "🟠",
}


# ── SIDEBAR ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 Local AI Lab")
    st.caption("V6 — Multi-Agent System")

    # Provider status
    provider_label = get_provider_label()
    if is_groq_available():
        st.success(f"{provider_label}")
    elif is_ollama_available():
        st.info(f"🏠 {provider_label}")
    else:
        st.error(" No LLM available")
        st.caption("Run Ollama locally or set HF_TOKEN")
        st.stop()

    if st.session_state.get("phoenix_active"):
        st.info("🔭 Phoenix → [localhost:6006](http://localhost:6006)")

    mem_stats = get_memory_stats()
    if mem_stats["total"] > 0:
        st.caption(f"🧠 {mem_stats['total']} memories")

    st.divider()

    # Knowledge base
    st.subheader("📚 Knowledge Base")
    st.caption(f"{get_chunk_count()} chunks indexed")

    if is_embedding_available():
        uploaded = st.file_uploader("Upload document", type=["txt", "md", "pdf", "py"])
        if uploaded and st.button("📥 Index", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    save_path = DOCUMENTS_DIR / uploaded.name
                    save_path.write_bytes(uploaded.read())
                    chunks = process_document(str(save_path), uploaded.name)
                    vs = get_vectorstore()
                    vs.add_texts(
                        texts=[c["text"] for c in chunks],
                        metadatas=[c["metadata"] for c in chunks],
                    )
                    st.success(f" {len(chunks)} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()

    # Conversations
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


# ── TABS ──────────────────────────────────────────────────────────────

tab_agent, tab_memory, tab_docs, tab_arch = st.tabs([
    "🤖 Agent", "🧠 Memory", "📚 Docs", "🏗️ How it works"
])


# ══ TAB 1: AGENT ══════════════════════════════════════════════════════

with tab_agent:
    if st.session_state.current_session is None:
        st.title("Local AI Lab — V6 Multi-Agent")
        st.markdown("""
        **Three agents. One goal. LangGraph orchestrates everything.**

        | Agent | Role | Tools |
        |---|---|---|
        | 🟣 Orchestrator | Analyzes goal, routes, synthesizes | None — reasons only |
        | 🔵 Researcher | Gathers information | web_search, fetch_webpage, knowledge base |
        | 🟠 Executor | Computes and acts | calculate, run_python, read/write files |

        **Try these:**
        ```
        "Search for the latest LangGraph news and save a summary to langgraph.txt"
        "Calculate compound interest on $10,000 at 7% for 15 years and save to report.txt"
        "What is the capital of Tanzania?" (direct answer — no agents needed)
        "Search for FishHappy competitors in East Africa and write a comparison report"
        ```

        Click **✏️ New conversation** to start.
        """)
    else:
        session = st.session_state.current_session
        messages = session["messages"]

        st.title(session.get("title", "Multi-Agent Session"))
        st.caption(f"LLM: `{get_provider_label()}` • 3 agents • {get_chunk_count()} docs")
        st.divider()

        # Render history
        for message in messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("steps"):
                    steps = message["steps"]
                    tool_steps = [s for s in steps if s["type"] == "tool_call"]
                    agents_involved = list(dict.fromkeys(s["agent"] for s in steps))

                    agent_icons = " → ".join(
                        f"{AGENT_COLORS.get(a, '⚪')} {a.capitalize()}"
                        for a in agents_involved
                    )

                    with st.expander(
                        f"{agent_icons} • {len(tool_steps)} tool call(s)",
                        expanded=False
                    ):
                        for step in steps:
                            icon = AGENT_COLORS.get(step["agent"], "⚪")
                            if step["type"] == "routing":
                                st.caption(f"{icon} **Orchestrator** → routing decision")
                                st.text(step["content"][:300])
                            elif step["type"] == "tool_call":
                                st.caption(f"{icon} **{step['agent'].capitalize()}** → `{step.get('tool_name', '')}`")
                                st.code(step["content"], language="json")
                            elif step["type"] == "observation":
                                st.caption(f"👁️ Result from `{step.get('tool_name', '')}`")
                                st.text(step["content"][:300] + ("..." if len(step["content"]) > 300 else ""))
                            elif step["type"] == "answer":
                                st.caption(f"{icon} **{step['agent'].capitalize()}** → completed")
                                st.text(step["content"][:200])
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
                answer_ph = st.empty()

                with st.status("🟣 Orchestrator analyzing goal...", state="running") as status:
                    try:
                        llm = get_llm()

                        for event in stream_graph(user_goal=prompt, llm=llm):
                            if event["type"] == "step":
                                step = event["step"]
                                step_log.append(step)
                                icon = AGENT_COLORS.get(step["agent"], "⚪")

                                if step["type"] == "routing":
                                    route = event["state"].get("route", "")
                                    status.update(
                                        label=f"{icon} Orchestrator → routing to {route}",
                                        state="running"
                                    )
                                    st.caption(f"{icon} **Orchestrator** decided: route to ")

                                elif step["type"] == "tool_call":
                                    tool = step.get("tool_name", "")
                                    status.update(
                                        label=f"{icon} {step['agent'].capitalize()} → calling ...",
                                        state="running"
                                    )
                                    st.caption(f"{icon} **{step['agent'].capitalize()}** → ")
                                    if step.get("content"):
                                        st.code(step["content"][:300], language="json")

                                elif step["type"] == "observation":
                                    tool = step.get("tool_name", "")
                                    status.update(
                                        label=f"👁️ Got result from ",
                                        state="running"
                                    )
                                    st.caption(f"👁️ Result from ")
                                    st.text(step["content"][:400])

                                elif step["type"] == "answer" and step["agent"] != "orchestrator":
                                    status.update(
                                        label=f"{icon} {step['agent'].capitalize()} finished, returning to Orchestrator...",
                                        state="running"
                                    )
                                    st.caption(f"{icon} **{step['agent'].capitalize()}** completed ✓")

                            elif event["type"] == "done":
                                state = event["state"]
                                final_answer = state.get("final_answer", "")
                                route = state.get("route", "done")
                                agents_used = list(dict.fromkeys(s["agent"] for s in step_log))

                                if route == "done":
                                    status.update(label="✅ Done", state="complete")
                                else:
                                    agent_summary = " → ".join(
                                        f"{AGENT_COLORS.get(a, '')} {a.capitalize()}"
                                        for a in agents_used
                                    )
                                    status.update(
                                        label=f"✅ {agent_summary} → Done",
                                        state="complete"
                                    )

                        # Fallback — if stream_graph never emitted done
                        if not final_answer:
                            result = run_graph(user_goal=prompt, llm=llm)
                            final_answer = result.get("final_answer", "No answer returned.")
                            step_log = result.get("steps", step_log)
                            status.update(label="✅ Done", state="complete")

                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        final_answer = f"Error: {e}"

                # Always show the final answer outside the status box
                if final_answer:
                    answer_ph.markdown(final_answer)
                else:
                    answer_ph.warning("Agent finished but returned no answer. Check the trace above.")

            messages.append({
                "role": "assistant",
                "content": final_answer,
                "steps": step_log,
            })
            session["messages"] = messages
            save_session(session)
            refresh_sessions()
            st.rerun()


# ══ TAB 2: MEMORY ════════════════════════════════════════════════════

with tab_memory:
    st.header("🧠 Long-Term Memory")
    st.caption("Injected into the Orchestrator's system prompt on every run.")

    stats = get_memory_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total memories", stats["total"])
    c2.metric("Categories", len(stats["categories"]))

    with st.expander("➕ Add memory manually"):
        m_key = st.text_input("Key")
        m_val = st.text_area("Value", height=70)
        m_cat = st.selectbox("Category", ["user_profile", "projects", "preferences", "general"])
        if st.button("💾 Store", use_container_width=True):
            if m_key and m_val:
                st.success(store_memory(m_key, m_val, m_cat))
                st.rerun()

    memory_prompt = build_memory_prompt()
    if memory_prompt:
        with st.expander("👁️ What the Orchestrator sees"):
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
    st.caption(f"{get_chunk_count()} chunks • Used by the Researcher agent")

    if not is_embedding_available():
        st.warning("nomic-embed-text not available. Run: `ollama pull nomic-embed-text`")
    else:
        st.success("✅ Embedding model ready")

    for doc in list_indexed_documents():
        st.caption(f"📄 {doc['source']}")


# ══ TAB 4: HOW IT WORKS ══════════════════════════════════════════════

with tab_arch:
    st.header("🏗️ How it works")

    st.markdown("""
    ### The multi-agent architecture

    ```
    User goal
        ↓
    🟣 Orchestrator
        Analyzes the goal.
        Decides: researcher / executor / both / answer directly.
        ↓                    ↓                  ↓
    🔵 Researcher       🟠 Executor          Answer
    web_search          calculate
    fetch_webpage       run_python
    knowledge base      read/write files
        ↓                    ↓
    🟣 Orchestrator (synthesizes results → final answer)
    ```

    ### What LangGraph adds vs V5

    | V5 | V6 |
    |---|---|
    | One agent does everything | Three agents with defined roles |
    | Single ReAct loop | Graph with nodes and conditional edges |
    | All tools available to one agent | Tools scoped per agent |
    | State is the message list | State is a typed `AgentState` dict |
    | One Phoenix trace per run | One trace with sub-spans per agent |

    ### Why separate agents?

    Each agent is independently debuggable. If the Researcher fails,
    you know exactly where to look. If the Executor produces wrong output,
    you inspect only that node. The Orchestrator's routing decisions are
    visible as a separate step — you can see exactly why it chose which agent.

    In Phoenix, each agent run appears as a sub-span inside the parent trace.

    ### Running publicly on Hugging Face Spaces

    This app uses the HF Inference API when deployed to HF Spaces.
    No Ollama needed — the model runs on Hugging Face's servers.
    When running locally, it falls back to Ollama automatically.

    To deploy:
    1. Push this folder to a HF Space (Streamlit SDK)
    2. Add `HF_TOKEN` as a Space secret (optional, for higher rate limits)
    3. The app detects the environment and switches providers automatically
    """)