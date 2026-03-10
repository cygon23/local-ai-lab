"""
context_manager.py
------------------
Manages the context window — one of the most important concepts in LLMs.

WHAT IS A CONTEXT WINDOW?
  Every LLM has a maximum number of tokens it can process at once.
  Our Qwen3:1.7b has a context window of 4096 tokens (we set this in ollama_client.py).
  
  A "token" is roughly 0.75 words in English.
  4096 tokens ≈ ~3000 words ≈ about 5-6 pages of text.

THE PROBLEM THIS SOLVES:
  In V1, we sent ALL messages to Ollama every time.
  After a long conversation, we'd silently overflow the context window.
  The model would either error out or start ignoring old messages.
  The user would never know why the model "forgot" things.

OUR STRATEGY — SLIDING WINDOW WITH SYSTEM PROMPT PROTECTION:
  
  [System Prompt]  ← ALWAYS included, never dropped
  [Message 1]      ← oldest, dropped first when we're near the limit
  [Message 2]      ← dropped second
  [Message 3]      ← ...
  [Message N-2]    ← recent messages always kept
  [Message N-1]    ← recent messages always kept
  [Message N]      ← most recent, always kept
  
  When we're near the limit, we drop from the MIDDLE —
  keeping the system prompt and the most recent messages.
  This is the "sliding window" approach.

  More sophisticated strategies (summarization, compression) come in V3/V4.

TOKEN COUNTING:
  Real tokenizers are model-specific and require loading the model's vocab.
  For our purposes, we use a good approximation:
  1 token ≈ 4 characters (works well for English, decent for other languages)
  This is the same heuristic OpenAI's tiktoken uses as a fallback.
"""


# How many tokens we allow for the conversation history
# We leave ~500 tokens headroom for the model's response
CONTEXT_LIMIT = 3500  # out of 4096 total
CHARS_PER_TOKEN = 4   # approximation


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from character count.
    Not perfectly accurate but good enough for windowing decisions.
    
    For production: use the actual tokenizer (tiktoken, transformers, etc.)
    For learning: this approximation teaches the concept without the complexity.
    """
    return len(text) // CHARS_PER_TOKEN


def estimate_message_tokens(message: dict) -> int:
    """
    Estimate tokens for a single message dict.
    We count the content + a small overhead for role/formatting.
    """
    content_tokens = estimate_tokens(message.get("content", ""))
    overhead = 10  # role field + JSON structure overhead
    return content_tokens + overhead


def get_context_stats(messages: list[dict], system_prompt: str = "") -> dict:
    """
    Return stats about current context usage.
    Used to show the token meter in the UI.
    """
    system_tokens = estimate_tokens(system_prompt) if system_prompt else 0
    message_tokens = sum(estimate_message_tokens(m) for m in messages)
    total = system_tokens + message_tokens

    return {
        "system_tokens": system_tokens,
        "message_tokens": message_tokens,
        "total_tokens": total,
        "limit": CONTEXT_LIMIT,
        "usage_percent": min(100, round((total / CONTEXT_LIMIT) * 100)),
        "is_near_limit": total > (CONTEXT_LIMIT * 0.85),  # warn at 85%
        "is_over_limit": total > CONTEXT_LIMIT,
    }


def trim_messages_to_fit(
    messages: list[dict],
    system_prompt: str = "",
    reserve_tokens: int = 500
) -> tuple[list[dict], int]:
    """
    Trim the message list so the total context fits within the limit.
    Returns (trimmed_messages, number_of_messages_dropped).

    STRATEGY:
      1. Always keep the last 4 messages (2 turns) regardless — preserves immediate context
      2. Drop from the oldest messages first (index 0, 1, 2...)
      3. Stop dropping once we're within the token limit

    WHY KEEP THE LAST 4?
      If we're mid-conversation and have to trim, the user's last question
      and the model's last answer must stay — otherwise the model loses
      the immediate thread entirely.
    """
    effective_limit = CONTEXT_LIMIT - reserve_tokens
    system_tokens = estimate_tokens(system_prompt) if system_prompt else 0
    dropped = 0

    # Always protect the last 4 messages
    PROTECTED_TAIL = 4

    while len(messages) > PROTECTED_TAIL:
        total = system_tokens + sum(estimate_message_tokens(m) for m in messages)

        if total <= effective_limit:
            break

        # Drop the oldest non-protected message
        messages = messages[1:]
        dropped += 1

    return messages, dropped


def format_context_warning(stats: dict, dropped: int = 0) -> str | None:
    """
    Return a human-readable warning string if context is getting full.
    Returns None if everything is fine.
    """
    if dropped > 0:
        return (
            f"⚠️ Context window trimmed: {dropped} older message(s) removed "
            f"to stay within the {CONTEXT_LIMIT} token limit. "
            f"The model no longer sees the earliest part of this conversation."
        )
    if stats["is_near_limit"]:
        remaining = CONTEXT_LIMIT - stats["total_tokens"]
        return (
            f"⚠️ Context window at {stats['usage_percent']}% "
            f"(~{remaining} tokens remaining). "
            f"Older messages will start being trimmed soon."
        )
    return None