"""
app.py — V2: Sessions + Chat History + Context Window Management
----------------------------------------------------------------
What's new vs V1:
  - Conversations are saved to disk and survive page refreshes
  - Multiple sessions: start new, switch between, delete, rename
  - Token counter shows context window usage in real time
  - Smart trimming when approaching the context limit

RUN IT:
  streamlit run app.py

FOLDER STRUCTURE REQUIRED:
  data/sessions/   ← created automatically on first run
"""

import streamlit as st
from ollama_client import is_ollama_running, get_available_models, chat_stream
from session_manager import (
    create_session, save_session, load_session,
    list_sessions, delete_session, rename_session
)
from context_manager import (
    get_context_stats, trim_messages_to_fit, format_context_warning
)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI Lab — V2",
    page_icon="🧠",
    layout="wide",   # Wide layout so sidebar + chat both have space
)


# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
#
# V2 has more state than V1:
#   current_session  → the full session dict (messages, metadata, etc.)
#   session_list     → lightweight list for the sidebar
#
# Notice we no longer store messages directly in session_state.
# Messages live INSIDE current_session["messages"].
# This keeps everything together — metadata + messages in one object.
# ─────────────────────────────────────────────
def init_state():
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "session_list" not in st.session_state:
        st.session_state.session_list = list_sessions()
    if "model" not in st.session_state:
        st.session_state.model = "qwen3:1.7b"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a helpful, clear, and concise AI assistant. "
            "When answering technical questions, give practical examples."
        )
    if "renaming_id" not in st.session_state:
        st.session_state.renaming_id = None

init_state()


def refresh_session_list():
    """Reload session list from disk and update state."""
    st.session_state.session_list = list_sessions()


def start_new_session():
    """Create a new session and set it as current."""
    session = create_session(
        model=st.session_state.model,
        system_prompt=st.session_state.system_prompt
    )
    st.session_state.current_session = session
    # Don't save yet — save on first message


def switch_to_session(session_id: str):
    """Load a session from disk and set as current."""
    session = load_session(session_id)
    if session:
        st.session_state.current_session = session


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Local AI Lab")
    st.caption("V2 — Sessions & Context")

    # Ollama status check
    if not is_ollama_running():
        st.error(" Ollama is not running")
        st.code("ollama serve", language="bash")
        st.stop()
    else:
        st.success(" Ollama running")

    st.divider()

    # Model selector
    available_models = get_available_models()
    if available_models:
        st.session_state.model = st.selectbox(
            "Model",
            options=available_models,
            index=available_models.index("qwen3:1.7b")
                  if "qwen3:1.7b" in available_models else 0
        )

    # System prompt
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100,
    )

    st.divider()

    # New conversation button
    if st.button(" New conversation", use_container_width=True, type="primary"):
        start_new_session()
        st.rerun()

    st.divider()

    # ── Conversation History List ──
    st.subheader("Conversations")

    sessions = st.session_state.session_list

    if not sessions:
        st.caption("No saved conversations yet.")
    else:
        current_id = (
            st.session_state.current_session["id"]
            if st.session_state.current_session else None
        )

        for s in sessions:
            is_active = s["id"] == current_id

            # Rename mode
            if st.session_state.renaming_id == s["id"]:
                new_title = st.text_input(
                    "Rename",
                    value=s["title"],
                    key=f"rename_input_{s['id']}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save", key=f"save_rename_{s['id']}"):
                        rename_session(s["id"], new_title)
                        st.session_state.renaming_id = None
                        # If renaming the active session, reload it
                        if is_active:
                            switch_to_session(s["id"])
                        refresh_session_list()
                        st.rerun()
                with col2:
                    if st.button("Cancel", key=f"cancel_rename_{s['id']}"):
                        st.session_state.renaming_id = None
                        st.rerun()
                continue

            # Normal mode — show session row
            col_title, col_edit, col_del = st.columns([6, 1, 1])

            with col_title:
                label = f"**{s['title'][:35]}**" if is_active else s["title"][:35]
                if st.button(
                    label,
                    key=f"session_{s['id']}",
                    use_container_width=True,
                    help=f"{s['message_count']} messages • {s['model']}"
                ):
                    switch_to_session(s["id"])
                    st.rerun()

            with col_edit:
                if st.button("✏️", key=f"rename_{s['id']}", help="Rename"):
                    st.session_state.renaming_id = s["id"]
                    st.rerun()

            with col_del:
                if st.button("🗑️", key=f"del_{s['id']}", help="Delete"):
                    delete_session(s["id"])
                    if is_active:
                        st.session_state.current_session = None
                    refresh_session_list()
                    st.rerun()


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────

# No session selected
if st.session_state.current_session is None:
    st.title("Welcome to Local AI Lab V2")
    st.markdown("""
    **What's new in V2:**
    -  Conversations saved to disk — survive page refresh
    -  Multiple sessions — start new, switch, rename, delete
    -  Context window meter — see token usage in real time
    -  Smart trimming — model never silently overflows its memory
    
    **Start a conversation** by clicking **✏️ New conversation** in the sidebar.
    """)
    st.stop()

# Active session
session = st.session_state.current_session
messages = session["messages"]

# ── Header ──
col_title, col_stats = st.columns([3, 1])

with col_title:
    st.title(session.get("title", "New conversation"))
    st.caption(f"Model: `{session['model']}` • Session ID: `{session['id'][:8]}...`")

with col_stats:
    # Context window meter
    stats = get_context_stats(messages, session.get("system_prompt", ""))
    st.metric("Context used", f"{stats['usage_percent']}%")
    st.progress(stats["usage_percent"] / 100)
    st.caption(f"{stats['total_tokens']} / {stats['limit']} tokens (est.)")

st.divider()

# ── Render message history ──
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle new input ──
if prompt := st.chat_input("Ask anything..."):

    # Add user message
    messages.append({"role": "user", "content": prompt})

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Context window management ──
    # Before sending to the model, trim if we're over the limit.
    # We trim a COPY so the full history is still saved to disk.
    # The user can scroll back and see everything — the model just
    # can't "see" the very old messages in its processing window.
    trimmed_messages, dropped = trim_messages_to_fit(
        messages=list(messages),  # copy, not reference
        system_prompt=session.get("system_prompt", ""),
    )

    # Show warning if messages were dropped from the model's view
    warning = format_context_warning(
        get_context_stats(trimmed_messages, session.get("system_prompt", "")),
        dropped
    )
    if warning:
        st.warning(warning)

    # ── Stream response ──
    with st.chat_message("assistant"):
        full_response = st.write_stream(
            chat_stream(
                messages=trimmed_messages,
                model=session["model"],
                system_prompt=session.get("system_prompt", ""),
            )
        )

    # Add assistant response to FULL history (not trimmed)
    messages.append({"role": "assistant", "content": full_response})

    # Save to disk after every response
    session["messages"] = messages
    save_session(session)

    # Refresh sidebar session list (title may have updated)
    refresh_session_list()

    # Rerun to update the token meter
    st.rerun()