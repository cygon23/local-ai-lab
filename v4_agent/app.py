"""
app.py — V4 Extended: Real World Agent
----------------------------------------
Full UI with 5 tabs:
  💬 Chat     — agent with real-time thought process
  🧠 Memory   — view/add/delete long-term memories
  🔍 Traces   — every agent run logged with timing + tool stats
  🔌 MCP      — connect MCP servers, auto-register their tools
  🛠️ Tools    — inspect all registered tools + their schemas

RUN IT:
  streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from ollama_client import is_ollama_running, get_available_models
from session_manager import create_session, save_session, load_session, list_sessions, delete_session
from context_manager import get_context_stats, trim_messages_to_fit
from embedder import is_embedding_model_available, get_embeddings_batch
from document_processor import process_document
from vector_store import add_chunks, list_indexed_documents, delete_document, get_chunk_count
from agent import run_agent
from observability import load_recent_traces, setup_phoenix, is_phoenix_available
from memory import get_all_memories, store_memory, delete_memory, get_memory_stats, build_memory_prompt
from mcp_client import connect_mcp_server, get_connected_servers, call_mcp_tool
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Local AI Lab — V4 Extended", page_icon="🤖", layout="wide")

# One-time startup
if "startup_done" not in st.session_state:
    setup_phoenix()
    st.session_state.startup_done = True


def init_state():
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "session_list" not in st.session_state:
        st.session_state.session_list = list_sessions()
    if "model" not in st.session_state:
        st.session_state.model = "qwen3:1.7b"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a capable AI agent. Think step by step. Use tools when needed. "
            "Be concise in your final answers. When you learn something important about "
            "the user or their projects, use store_memory to remember it for future sessions."
        )
    if "mcp_name" not in st.session_state:
        st.session_state.mcp_name = ""
    if "mcp_cmd" not in st.session_state:
        st.session_state.mcp_cmd = ""

init_state()


def refresh_sessions():
    st.session_state.session_list = list_sessions()

def start_new_session():
    st.session_state.current_session = create_session(
        model=st.session_state.model,
        system_prompt=st.session_state.system_prompt
    )

def switch_session(sid):
    s = load_session(sid)
    if s:
        st.session_state.current_session = s


# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Local AI Lab")
    st.caption("V4 Extended — Real World Agent")

    if not is_ollama_running():
        st.error(" Ollama not running — run `ollama serve`")
        st.stop()

    ca, cb = st.columns(2)
    ca.success(" Ollama")
    cb.success(" Embed") if is_embedding_model_available() else cb.warning("⚠️ No embed")

    if is_phoenix_available():
        st.info("🔭 Phoenix active → [localhost:6006](http://localhost:6006)")

    mem_stats = get_memory_stats()
    if mem_stats["total"] > 0:
        st.caption(f"🧠 {mem_stats['total']} memories stored")

    mcp_servers = get_connected_servers()
    if mcp_servers:
        st.caption(f"🔌 MCP: {', '.join(mcp_servers)}")

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
    st.subheader("📚 Knowledge Base")
    st.caption(f"{get_chunk_count()} chunks indexed")

    if is_embedding_model_available():
        uploaded = st.file_uploader("Upload document", type=["txt", "md", "pdf", "py", "js"])
        if uploaded and st.button("📥 Index", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    save_path = DOCUMENTS_DIR / uploaded.name
                    save_path.write_bytes(uploaded.read())
                    chunks = process_document(str(save_path), uploaded.name)
                    embeddings = get_embeddings_batch([c["text"] for c in chunks])
                    add_chunks(chunks, embeddings)
                    st.success(f"✅ {len(chunks)} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    for doc in list_indexed_documents():
        c1, c2 = st.columns([4, 1])
        c1.caption(f"📄 {doc['source']}")
        if c2.button("🗑️", key=f"dd_{doc['source']}"):
            delete_document(doc["source"])
            st.rerun()

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


# ─── MAIN TABS ─────────────────────────────────────────────────────
tab_chat, tab_memory, tab_traces, tab_mcp, tab_tools = st.tabs([
    "💬 Chat", "🧠 Memory", "🔍 Traces", "🔌 MCP", "🛠️ Tools"
])


# ══ TAB 1: CHAT ════════════════════════════════════════════════════
with tab_chat:
    if st.session_state.current_session is None:
        st.title("Local AI Lab — V4 Extended")
        st.markdown("""
        **A real-world autonomous agent — 100% local.**

        | Tool | What it does |
        |---|---|
        | `web_search` | Search the internet (DuckDuckGo) |
        | `fetch_webpage` | Read any public webpage or JSON API |
        | `call_api` | Full REST API calls with headers + body |
        | `calculate` | Exact math — never guesses |
        | `run_python` | Write and execute Python code |
        | `read_file` / `write_file` | File operations in workspace |
        | `search_knowledge_base` | Search your indexed documents |
        | `store_memory` / `recall_memory` | Long-term memory across sessions |

        **Try these:**
        ```
        "Search for AI agent news today and save a summary to news.txt"
        "Remember that I'm building FishHappy for the Zanzibar fish market"
        "Calculate compound interest on $5000 at 8% for 10 years"
        "Write a Python script to generate random fish prices, run it"
        ```

        Click **✏️ New conversation** in the sidebar to start.
        """)
    else:
        session = st.session_state.current_session
        messages = session["messages"]
        stats = get_context_stats(messages, session.get("system_prompt", ""))

        st.title(session.get("title", "Agent Session"))
        st.caption(
            f"Model: `{session['model']}` • Context: {stats['usage_percent']}% • "
            f"{get_chunk_count()} docs • {get_memory_stats()['total']} memories"
        )
        st.progress(stats["usage_percent"] / 100)
        st.divider()

        # Render history
        for message in messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("steps"):
                    steps = message["steps"]
                    tool_steps = [s for s in steps if s["type"] == "tool_call"]
                    if tool_steps:
                        with st.expander(f"🔍 Agent used {len(tool_steps)} tool(s)", expanded=False):
                            for step in steps:
                                if step["type"] == "thinking":
                                    st.caption("💭 Thinking")
                                    st.text(step["content"][:300])
                                elif step["type"] == "tool_call":
                                    st.caption(f"🔧 Tool: `{step.get('tool_name', '')}`")
                                    st.code(step["content"], language="json")
                                elif step["type"] == "observation":
                                    st.caption(f"👁️ Result from `{step.get('tool_name', '')}`")
                                    preview = step["content"]
                                    st.text(preview[:400] + ("..." if len(preview) > 400 else ""))
                                    st.divider()
                st.markdown(message["content"])

        # New input
        if prompt := st.chat_input("Give me a goal..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})

            trimmed, _ = trim_messages_to_fit(list(messages[:-1]), session.get("system_prompt", ""))

            with st.chat_message("assistant"):
                step_log = []
                final_answer = ""
                status_ph = st.empty()
                trace_ph = st.empty()
                answer_ph = st.empty()
                tool_call_count = 0

                for step in run_agent(
                    user_message=prompt,
                    conversation_history=trimmed,
                    model=session["model"],
                    base_system_prompt=session.get("system_prompt", ""),
                    session_id=session["id"],
                ):
                    step_dict = {"type": step.type, "content": step.content, "tool_name": step.tool_name}
                    step_log.append(step_dict)

                    if step.type == "thinking":
                        status_ph.caption("💭 Thinking...")

                    elif step.type == "tool_call":
                        tool_call_count += 1
                        status_ph.caption(f"🔧 Calling `{step.tool_name}`...")
                        with trace_ph.container():
                            with st.expander(f"🔍 Step {tool_call_count}: `{step.tool_name}`", expanded=True):
                                for s in step_log:
                                    if s["type"] == "tool_call":
                                        st.caption(f"🔧 `{s['tool_name']}`")
                                        st.code(s["content"], language="json")
                                    elif s["type"] == "observation":
                                        st.caption(f"👁️ Result")
                                        st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

                    elif step.type == "observation":
                        status_ph.caption("👁️ Got result, deciding next step...")
                        with trace_ph.container():
                            with st.expander(f"🔍 Trace ({tool_call_count} tool calls)", expanded=True):
                                for s in step_log:
                                    if s["type"] == "tool_call":
                                        st.caption(f"🔧 `{s['tool_name']}`")
                                        st.code(s["content"], language="json")
                                    elif s["type"] == "observation":
                                        st.caption(f"👁️ Result")
                                        st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

                    elif step.type == "answer":
                        final_answer = step.content
                        status_ph.empty()
                        answer_ph.markdown(final_answer)

                    elif step.type == "error":
                        status_ph.error(f"⚠️ {step.content}")

            messages.append({
                "role": "assistant",
                "content": final_answer or "Agent completed.",
                "steps": step_log,
            })
            session["messages"] = messages
            save_session(session)
            refresh_sessions()
            st.rerun()


# ══ TAB 2: MEMORY ══════════════════════════════════════════════════
with tab_memory:
    st.header("🧠 Long-Term Memory")
    st.caption("Facts the agent remembers across sessions. Auto-injected into every run.")

    stats = get_memory_stats()
    c1, c2 = st.columns(2)
    c1.metric("Total memories", stats["total"])
    c2.metric("Categories", len(stats["categories"]))
    if stats["categories"]:
        st.caption("By category: " + " • ".join(f"{k} ({v})" for k, v in stats["categories"].items()))

    st.divider()

    with st.expander("➕ Add memory manually"):
        m_key = st.text_input("Key", placeholder="e.g. user_name, project_fishhappy")
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
        with st.expander("👁️ What the agent sees in its system prompt"):
            st.text(memory_prompt)

    st.divider()
    all_memories = get_all_memories()
    if not all_memories:
        st.info("No memories yet. Ask the agent to remember something in the Chat tab.")
    else:
        st.subheader(f"Stored memories ({len(all_memories)})")
        for mem in all_memories:
            col_info, col_del = st.columns([6, 1])
            with col_info:
                st.markdown(f"**[{mem['category']}]** `{mem['key']}`")
                st.caption(mem["value"])
                st.caption(f"Stored: {mem['created_at'][:10]}")
            with col_del:
                if st.button("🗑️", key=f"dm_{mem['id']}"):
                    delete_memory(mem["key"])
                    st.rerun()
            st.divider()


# ══ TAB 3: TRACES ══════════════════════════════════════════════════
with tab_traces:
    st.header("🔍 Agent Traces")
    st.caption("Every agent run logged automatically — see what the agent did and why.")

    if is_phoenix_available():
        st.info("🔭 Arize Phoenix running → [localhost:6006](http://localhost:6006) for visual traces")

    traces = load_recent_traces(20)

    if not traces:
        st.info("No traces yet. Run the agent in the Chat tab.")
    else:
        st.caption(f"Last {len(traces)} runs (newest first)")
        st.divider()
        for trace in traces:
            duration = f"{trace.get('duration_ms', 0)}ms" if trace.get("duration_ms") else "—"
            tool_calls = trace.get("tool_calls_made", [])
            ts = trace.get("timestamp", "")[:16].replace("T", " ")

            cq, cs = st.columns([3, 2])
            with cq:
                st.markdown(f"**{trace['user_message'][:80]}**")
                st.caption(f"{ts} • {trace.get('model', '—')}")
            with cs:
                st.caption(
                    f"⏱ {duration} • 🔁 {trace.get('llm_calls', 0)} LLM • "
                    f"🔧 {len(tool_calls)} tools • ~{trace.get('total_tokens_est', 0)} tokens"
                )
            if tool_calls:
                st.caption("Tools: " + " → ".join(f"`{t}`" for t in tool_calls))

            spans = trace.get("spans", [])
            if spans:
                with st.expander("Full trace"):
                    for span in spans:
                        if span["type"] == "llm":
                            st.caption(f"🤖 LLM call #{span.get('call_number')} — {span.get('duration_ms')}ms")
                            st.text(span.get("output_preview", "")[:200])
                        elif span["type"] == "tool":
                            st.caption(f"🔧 `{span['tool_name']}` — {span.get('duration_ms')}ms")
                            st.json(span.get("args", {}))
                            st.text("→ " + span.get("result_preview", "")[:200])
                        st.divider()

            if trace.get("error"):
                st.error(f"Error: {trace['error']}")
            st.divider()


# ══ TAB 4: MCP ═════════════════════════════════════════════════════
with tab_mcp:
    st.header("🔌 MCP Servers")
    st.caption("Connect MCP servers — their tools register into the agent automatically.")

    connected = get_connected_servers()
    if connected:
        st.success(f"Connected: {', '.join(connected)}")
    else:
        st.info("No MCP servers connected.")

    st.divider()
    st.subheader("Connect a server")
    st.caption("Quick presets (require Node.js):")

    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("📁 Filesystem", use_container_width=True):
        st.session_state.mcp_name = "filesystem"
        st.session_state.mcp_cmd = "npx -y @modelcontextprotocol/server-filesystem ./data/workspace"
        st.rerun()
    if pc2.button("🌐 Fetch", use_container_width=True):
        st.session_state.mcp_name = "fetch"
        st.session_state.mcp_cmd = "npx -y @modelcontextprotocol/server-fetch"
        st.rerun()
    if pc3.button("🗃️ SQLite", use_container_width=True):
        st.session_state.mcp_name = "sqlite"
        st.session_state.mcp_cmd = "npx -y @modelcontextprotocol/server-sqlite --db-path ./data/agent.db"
        st.rerun()

    server_name = st.text_input("Server name", value=st.session_state.mcp_name, placeholder="filesystem")
    server_cmd = st.text_input("Command", value=st.session_state.mcp_cmd, placeholder="npx -y @modelcontextprotocol/server-filesystem ./data/workspace")

    if st.button("🔌 Connect", use_container_width=True, type="primary"):
        if server_name and server_cmd:
            with st.spinner(f"Connecting to {server_name}..."):
                success, mcp_tools = connect_mcp_server(server_name, server_cmd.split())
                if success:
                    for tool in mcp_tools:
                        orig = tool.get("_mcp_original_name", "")
                        srv = tool.get("_mcp_server", "")
                        lname = tool["name"]
                        if not any(t["name"] == lname for t in TOOL_SCHEMAS):
                            TOOL_SCHEMAS.append(tool)
                        if lname not in TOOL_FUNCTIONS:
                            def make_caller(s, t):
                                return lambda **kwargs: call_mcp_tool(s, t, kwargs)
                            TOOL_FUNCTIONS[lname] = make_caller(srv, orig)
                    st.success(f"✅ {len(mcp_tools)} tools registered")
                    for t in mcp_tools:
                        st.caption(f"  • `{t['name']}`")
                    st.rerun()
                else:
                    st.error("Connection failed. Check Node.js is installed and command is correct.")
        else:
            st.warning("Enter both name and command.")

    st.divider()
    with st.expander("What is MCP?"):
        st.markdown("""
        **Model Context Protocol** — Anthropic's open standard for connecting agents to tools.

        Before MCP: every agent wrote custom integrations per service.
        After MCP: write one server → works with ALL compatible agents.
        Claude.ai uses MCP for Gmail, Drive, Slack, etc.
        Your agent uses the same protocol.

        **Free servers (need Node.js):**
        ```
        npx @modelcontextprotocol/server-filesystem /path
        npx @modelcontextprotocol/server-fetch
        npx @modelcontextprotocol/server-memory
        npx @modelcontextprotocol/server-sqlite --db-path ./data.db
        ```
        """)


# ══ TAB 5: TOOLS ═══════════════════════════════════════════════════
with tab_tools:
    st.header("🛠️ Tool Registry")
    st.caption(f"{len(TOOL_SCHEMAS)} tools available. The agent reads all schemas before deciding what to do.")

    local_tools = [t for t in TOOL_SCHEMAS if not t["name"].startswith("mcp_")]
    mcp_t = [t for t in TOOL_SCHEMAS if t["name"].startswith("mcp_")]

    st.subheader(f"Local tools ({len(local_tools)})")
    for tool in local_tools:
        with st.expander(f"`{tool['name']}` — {tool['description'][:65]}..."):
            st.markdown(f"**When to use:** {tool['description']}")
            if tool.get("parameters"):
                st.markdown("**Parameters:**")
                for p, d in tool["parameters"].items():
                    st.caption(f"  `{p}`: {d}")
            st.markdown(f"**Returns:** {tool.get('returns', '—')}")

    if mcp_t:
        st.divider()
        st.subheader(f"MCP tools ({len(mcp_t)})")
        for tool in mcp_t:
            with st.expander(f"`{tool['name']}` — {tool['description'][:65]}"):
                st.markdown(f"**Description:** {tool['description']}")
                if tool.get("parameters"):
                    for p, d in tool["parameters"].items():
                        st.caption(f"  `{p}`: {d}")

    st.divider()
    st.subheader("How the agent uses these tools")
    st.markdown("""
    All schemas above are injected into the agent's system prompt before every run.
    When the agent decides to use a tool it outputs:

    ```
    <tool_call>
    {"tool": "tool_name", "args": {"param": "value"}}
    </tool_call>
    ```

    The agent loop (`agent.py`) detects the tag → calls the Python function →
    feeds the result back as an observation → model decides what to do next.
    This repeats until the model outputs a response with no tool call tag.

    That loop is the **ReAct pattern** — every agent framework implements exactly this.
    """)