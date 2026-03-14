"""
llm.py
------
LLM provider for HF Spaces deployment.

Primary: Groq API (free, fast, no billing needed)
  - Sign up: console.groq.com
  - Add GROQ_API_KEY as a Space secret in HF Spaces settings
  - Free tier: generous limits, plenty for demos

Fallback: Ollama (local development only)
  - Automatically used when GROQ_API_KEY is not set and Ollama is running

Model: llama-3.1-8b-instant
  - Free on Groq
  - Fast (token generation ~750 tok/s)
  - Strong reasoning and tool use
"""

import os
import warnings
import requests

# Load .env file if present (for local development)
def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
_load_env()

warnings.filterwarnings("ignore", message=".*PyTorch was not found.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Config ────────────────────────────────────────────────────────────

GROQ_MODEL        = "llama-3.3-70b-versatile"   # orchestrator — strong reasoning
GROQ_TOOL_MODEL   = "llama-3.1-8b-instant"         # sub-agents — faster, better tool syntax
OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_BASE_URL = "http://localhost:11434"


# ── Availability checks ───────────────────────────────────────────────

def is_groq_available() -> bool:
    """True if GROQ_API_KEY is set."""
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def is_ollama_available() -> bool:
    """True if local Ollama is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_provider() -> str:
    """
    Returns 'groq' or 'ollama'.
    Groq takes priority when GROQ_API_KEY is set.
    Falls back to Ollama for local development.
    """
    if is_groq_available():
        return "groq"
    if is_ollama_available():
        return "ollama"
    return "none"


def get_llm(temperature: float = 0.3, tool_agent: bool = False):
    """
    Returns a LangChain-compatible LLM.
    Groq when deployed, Ollama when running locally.
    tool_agent=True uses the faster 8B model for sub-agents (researcher/executor).
    """
    provider = get_provider()

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = GROQ_TOOL_MODEL if tool_agent else GROQ_MODEL
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
            max_tokens=1024,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature,
            num_ctx=2048,
            keep_alive="10m",
        )

    raise RuntimeError(
        "No LLM available.\n"
        "• For HF Spaces: add GROQ_API_KEY as a Space secret\n"
        "• For local: run Ollama with `ollama serve`\n"
        "  Get a free Groq key at: https://console.groq.com"
    )


def get_provider_label() -> str:
    p = get_provider()
    if p == "groq":
        return f"Groq API ({GROQ_MODEL} / {GROQ_TOOL_MODEL})"
    if p == "ollama":
        return f"Ollama local ({OLLAMA_MODEL})"
    return "No LLM available"