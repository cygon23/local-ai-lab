"""
app.py — V3: RAG (Retrieval-Augmented Generation)
--------------------------------------------------
What's new vs V2:
  - Upload documents (PDF, TXT, MD, PY)
  - Documents are chunked and indexed into ChromaDB
  - Every question searches the knowledge base first
  - Retrieved chunks are shown in the UI (transparency)
  - Toggle RAG on/off per conversation
  - Delete documents from the index

RUN IT:
  # First, pull the embedding model
  ollama pull nomic-embed-text

  # Install new dependencies
  pip install -r requirements.txt

  # Run
  streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import tempfile

from ollama_client import is_ollama_running, get_available_models
from session_manager import (
    create_session, save_session, load_session,
    list_sessions, delete_session
)
from context_manager import get_context_stats, trim_messages_to_fit, format_context_warning
from embedder import is_embedding_model_available, get_embeddings_batch
from document_processor import process_document
from vector_store import (
    add_chunks, list_indexed_documents, delete_document, get_chunk_count
)
from rag_pipeline import query as rag_query


DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI Lab — V3 RAG",
    page_icon="🧠",
    layout="wide",
)


# ─────────────────────────────────────────────
# SESSION STATE
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
            "You are a helpful, knowledgeable assistant. Answer clearly and concisely. "
            "When using information from provided documents, cite the source."
        )
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "indexing" not in st.session_state:
        st.session_state.indexing = False

init_state()


def refresh_session_list():
    st.session_state.session_list = list_sessions()

def start_new_session():
    st.session_state.current_session = create_session(
        model=st.session_state.model,
        system_prompt=st.session_state.system_prompt
    )

def switch_to_session(session_id):
    session = load_session(session_id)
    if session:
        st.session_state.current_session = session


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Local AI Lab")
    st.caption("V3 — RAG")

    # Ollama checks
    if not is_ollama_running():
        st.error("Ollama not running — run `ollama serve`")
        st.stop()

    embed_ready = is_embedding_model_available()
    if not embed_ready:
        st.warning(" Embedding model not found")
        st.code("ollama pull nomic-embed-text", language="bash")
    else:
        st.success(" Ollama + embeddings ready")

    st.divider()

    # Model
    available_models = get_available_models()
    if available_models:
        st.session_state.model = st.selectbox(
            "Chat model",
            options=available_models,
            index=available_models.index("qwen3:1.7b")
                  if "qwen3:1.7b" in available_models else 0
        )

    # System prompt
    st.session_state.system_prompt = st.text_area(
        "System Prompt", value=st.session_state.system_prompt, height=80
    )

    # RAG toggle
    st.session_state.use_rag = st.toggle(
        "🔍 RAG mode",
        value=st.session_state.use_rag,
        help="When ON, every question searches your knowledge base first."
    )

    st.divider()

    # ── Knowledge Base ──
    st.subheader("📚 Knowledge Base")
    chunk_count = get_chunk_count()
    st.caption(f"{chunk_count} chunks indexed")

    # Document upload
    if embed_ready:
        uploaded = st.file_uploader(
            "Upload document",
            type=["txt", "md", "pdf", "py", "js", "ts"],
            help="Document will be chunked and indexed into ChromaDB"
        )

        if uploaded is not None:
            if st.button(" Index document", use_container_width=True):
                with st.spinner(f"Indexing {uploaded.name}..."):
                    try:
                        # Save uploaded file to disk
                        save_path = DOCUMENTS_DIR / uploaded.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded.read())

                        # Process: load + chunk
                        chunks = process_document(str(save_path), uploaded.name)
                        st.info(f"Split into {len(chunks)} chunks. Embedding...")

                        # Embed all chunks
                        texts = [c["text"] for c in chunks]
                        embeddings = get_embeddings_batch(texts)

                        # Store in ChromaDB
                        added = add_chunks(chunks, embeddings)
                        st.success(f" Indexed {added} chunks from {uploaded.name}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

    # List indexed documents
    indexed_docs = list_indexed_documents()
    if indexed_docs:
        st.caption("Indexed documents:")
        for doc in indexed_docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(f" {doc['source']} ({doc['chunk_count']} chunks)")
            with col2:
                if st.button("🗑️", key=f"deldoc_{doc['source']}", help="Remove from index"):
                    deleted = delete_document(doc["source"])
                    st.success(f"Removed {deleted} chunks")
                    st.rerun()

    st.divider()

    # ── Conversations ──
    st.subheader("Conversations")

    if st.button("✏️ New conversation", use_container_width=True, type="primary"):
        start_new_session()
        st.rerun()

    sessions = st.session_state.session_list
    current_id = (
        st.session_state.current_session["id"]
        if st.session_state.current_session else None
    )

    for s in sessions:
        is_active = s["id"] == current_id
        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"**{s['title'][:30]}**" if is_active else s["title"][:30]
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                switch_to_session(s["id"])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delsess_{s['id']}"):
                delete_session(s["id"])
                if is_active:
                    st.session_state.current_session = None
                refresh_session_list()
                st.rerun()


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if st.session_state.current_session is None:
    st.title("Local AI Lab — V3 RAG")
    st.markdown("""
    **What's new in V3:**
    -  Upload documents — PDF, TXT, Markdown, code files
    -  Documents are split into chunks and embedded into ChromaDB
    -  Every question retrieves relevant chunks before answering
    -  Retrieved sources are shown so you can verify the model's reasoning
    -  Toggle RAG on/off to compare grounded vs. open-ended answers

    **Before you start:**
    ```bash
    ollama pull nomic-embed-text   # the embedding model
    ```

    Then upload a document in the sidebar and start chatting.
    """)

    # Explain what's happening under the hood
    with st.expander("🧠 How RAG works in this app"):
        st.markdown("""
        **INDEXING** (happens once when you upload):
        1. Document is loaded and split into ~800 char chunks
        2. Each chunk is sent to `nomic-embed-text` → returns a 768-dim vector
        3. Chunk text + vector + metadata saved to ChromaDB on disk

        **RETRIEVAL** (happens on every question):
        1. Your question is embedded into a vector
        2. ChromaDB finds the top-4 most similar chunk vectors (cosine similarity)
        3. Chunks scoring below 0.45 similarity are discarded

        **GENERATION**:
        1. Retrieved chunks are injected into the system prompt
        2. Full conversation history + augmented system prompt → sent to Ollama
        3. Model answers using both its training AND your documents
        """)
    st.stop()

# Active session
session = st.session_state.current_session
messages = session["messages"]

# ── Header ──
col_title, col_stats = st.columns([3, 1])
with col_title:
    st.title(session.get("title", "New conversation"))
    rag_status = "🔍 RAG ON" if st.session_state.use_rag else "💬 Plain chat"
    st.caption(f"Model: `{session['model']}` • {rag_status} • {get_chunk_count()} chunks indexed")

with col_stats:
    stats = get_context_stats(messages, session.get("system_prompt", ""))
    st.metric("Context used", f"{stats['usage_percent']}%")
    st.progress(stats["usage_percent"] / 100)

st.divider()

# ── Message history ──
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show retrieved sources if stored in message metadata
        if message.get("retrieved_chunks"):
            with st.expander(f"📎 {len(message['retrieved_chunks'])} sources retrieved"):
                for chunk in message["retrieved_chunks"]:
                    st.caption(
                        f"**{chunk['source']}** — chunk {chunk['chunk_index']} "
                        f"| similarity: {chunk['score']}"
                    )
                    st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                    st.divider()

# ── New input ──
if prompt := st.chat_input("Ask anything..."):

    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Trim messages for context window
    trimmed, dropped = trim_messages_to_fit(
        list(messages),
        session.get("system_prompt", "")
    )
    warning = format_context_warning(
        get_context_stats(trimmed, session.get("system_prompt", "")), dropped
    )
    if warning:
        st.warning(warning)

    # ── RAG query ──
    with st.chat_message("assistant"):
        # Show a "searching" indicator while retrieving
        if st.session_state.use_rag and get_chunk_count() > 0:
            with st.spinner("🔍 Searching knowledge base..."):
                stream, retrieved_chunks, rag_used = rag_query(
                    user_question=prompt,
                    conversation_history=trimmed,
                    model=session["model"],
                    base_system_prompt=session.get("system_prompt", ""),
                    use_rag=True,
                )
        else:
            stream, retrieved_chunks, rag_used = rag_query(
                user_question=prompt,
                conversation_history=trimmed,
                model=session["model"],
                base_system_prompt=session.get("system_prompt", ""),
                use_rag=False,
            )

        # Show retrieved sources BEFORE streaming response
        if retrieved_chunks:
            with st.expander(f"📎 {len(retrieved_chunks)} sources retrieved", expanded=False):
                for chunk in retrieved_chunks:
                    st.caption(
                        f"**{chunk['source']}** — chunk {chunk['chunk_index']} "
                        f"| similarity: {chunk['score']}"
                    )
                    st.text(chunk["text"][:300] + "...")
                    st.divider()

        # Stream the actual response
        full_response = st.write_stream(stream)

    # Save response — include retrieved chunks as metadata on the message
    assistant_message = {
        "role": "assistant",
        "content": full_response,
        "retrieved_chunks": retrieved_chunks,  # stored for display in history
        "rag_used": rag_used,
    }
    messages.append(assistant_message)

    session["messages"] = messages
    save_session(session)
    refresh_session_list()
    st.rerun()