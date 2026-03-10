"""
ollama_client.py
----------------
A clean, minimal wrapper around Ollama's local HTTP API.

WHY THIS EXISTS AS A SEPARATE FILE:
  Every version (V1 → V7) will import this. When Ollama changes something,
  you fix it in one place. This is also how real production code is structured.

HOW OLLAMA WORKS UNDER THE HOOD:
  When you run `ollama serve`, it starts an HTTP server at localhost:11434.
  Every model you pull becomes available at POST /api/chat.
  We talk to it exactly like any REST API — no special SDK needed.
"""

import requests
import json
from typing import Generator


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


def is_ollama_running() -> bool:
    """
    Check if Ollama server is up before trying to chat.
    Always check this on startup — saves confusing error messages.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_available_models() -> list[str]:
    """
    Returns all models you have pulled locally.
    Useful for letting the user pick a model in the UI.
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []


def chat_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    system_prompt: str = None,
) -> Generator[str, None, None]:
    """
    Send a conversation to Ollama and stream the response token by token.

    WHAT 'messages' LOOKS LIKE:
      [
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is..."},
        {"role": "user", "content": "Give me more detail"}
      ]
      This is the standard OpenAI-compatible chat format.
      Ollama uses the exact same format — intentionally.

    WHAT STREAMING MEANS:
      Instead of waiting for the full response (could be 10 seconds),
      we get tokens one by one as they're generated. This is what makes
      ChatGPT feel "alive" — it's just streaming.

    HOW WE STREAM:
      We set stream=True in the request.
      Ollama sends back newline-delimited JSON chunks.
      Each chunk has {"message": {"content": "token"}, "done": false}
      When done=true, the response is complete.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.7,   # 0 = focused/deterministic, 1 = creative
            "num_ctx": 4096,      # context window size in tokens
        }
    }

    # Inject system prompt if provided
    if system_prompt:
        payload["messages"] = [
            {"role": "system", "content": system_prompt}
        ] + messages

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,        # <- this is what enables streaming
            timeout=120
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))

                    # Extract the token text from the chunk
                    token = chunk.get("message", {}).get("content", "")

                    if token:
                        yield token  # <- yield sends one token at a time

                    # When done=True, the model has finished
                    if chunk.get("done", False):
                        break

    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot connect to Ollama. Make sure `ollama serve` is running."
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"