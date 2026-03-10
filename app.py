"""
app.py — V1: Basic Local AI Chat
----------------------------------
This is the entire V1 application. Read every comment.
Each comment explains WHY, not just what.

RUN IT:
  streamlit run app.py

WHAT YOU'LL SEE:
  A chat interface connected to your local Qwen3:8b model.
  Responses stream token by token.
  Chat history stays visible during the session.

WHAT THIS VERSION DOES NOT DO YET (intentional):
  - No persistent history across sessions (V2)
  - No documents/RAG (V3)
  - No tools or autonomous behavior (V4)
  Keep it simple. Understand the base first.
"""

import streamlit as st
from ollama_client import is_ollama_running, get_available_models, chat_stream


# ─────────────────────────────────────────────
# PAGE CONFIG
# Must be the first Streamlit call in the script.
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Local AI Lab — V1",
    page_icon="🧠",
    layout="centered",
)


# ─────────────────────────────────────────────
# SESSION STATE — THE MOST IMPORTANT CONCEPT IN STREAMLIT
#
# Streamlit reruns the ENTIRE script top-to-bottom on every
# user interaction. That means all your variables reset.
#
# st.session_state is a dictionary that PERSISTS across reruns.
# This is how you keep chat history alive during a session.
#
# Think of it as RAM for your Streamlit app.
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # This list will hold our full conversation in the format:
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # This exact format is what we send to Ollama every time.

if "model" not in st.session_state:
    st.session_state.model = "qwen3:8b"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Check if Ollama is running
    if is_ollama_running():
        st.success(" Ollama is running")
        available_models = get_available_models()

        if available_models:
            st.session_state.model = st.selectbox(
                "Model",
                options=available_models,
                index=available_models.index("qwen3:8b") if "qwen3:8b" in available_models else 0
            )
    else:
        st.error(" Ollama is not running")
        st.code("ollama serve", language="bash")
        st.stop()  # Don't render the rest of the app if Ollama is down

    st.divider()

    # System prompt — this shapes the model's personality/behavior
    # This is the first lever you have to control the model.
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, clear, and concise AI assistant. "
              "When answering technical questions, give practical examples.",
        height=120,
        help="This is sent to the model before every conversation. "
             "It defines how the model behaves."
    )

    st.divider()

    # Clear conversation button
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**V1** — Basic Chat")
    st.caption("Next: V2 adds persistent sessions & history management")


# ─────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────
st.title("🧠 Local AI Lab")
st.caption(f"Model: `{st.session_state.model}` • Running locally via Ollama")

# Render all previous messages
# This loops through session_state.messages and draws each bubble.
# Streamlit's st.chat_message handles the user/assistant styling automatically.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─────────────────────────────────────────────
# HANDLE NEW USER INPUT
# ─────────────────────────────────────────────
if prompt := st.chat_input("Ask anything..."):

    # 1. Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # 2. Show the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Stream the assistant's response
    with st.chat_message("assistant"):

        # st.write_stream() takes a generator and renders tokens as they arrive.
        # Our chat_stream() function is that generator.
        # This is what creates the "typing" effect.
        full_response = st.write_stream(
            chat_stream(
                messages=st.session_state.messages,
                model=st.session_state.model,
                system_prompt=system_prompt,
            )
        )

    # 4. Save the complete assistant response to history
    # We save AFTER streaming so we have the full text.
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    # WHY WE SAVE THE FULL HISTORY EACH TIME:
    # Next time the user sends a message, we send ALL previous messages
    # back to the model. This is how the model "remembers" the conversation.
    # The model itself is stateless — it sees everything fresh each time.
    # The "memory" lives entirely in our messages list.