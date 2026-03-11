"""
ollama_llm.py
-------------
Wraps Ollama as a LangChain-compatible LLM using langchain-ollama.

WHY THIS FILE EXISTS:
  LangChain needs an LLM object that implements its BaseChatModel interface.
  langchain-ollama provides ChatOllama which does exactly that.
  We wrap it here with our defaults so the rest of the code
  just imports get_llm() and never thinks about configuration again.

WHAT CHANGED FROM V4:
  V4: we called requests.post("http://localhost:11434/api/chat") manually
  V5: LangChain calls Ollama for us via ChatOllama
  Same model, same server, different caller.
"""

from langchain_ollama import ChatOllama
import requests


OLLAMA_BASE_URL = "http://localhost:11434"


def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_llm(model: str = "qwen3:1.7b", temperature: float = 0.3) -> ChatOllama:
    """
    Returns a LangChain ChatOllama instance.

    temperature=0.3 keeps tool call JSON deterministic.
    Lower = more consistent tool calls.
    Higher = more creative final answers.
    """
    return ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )