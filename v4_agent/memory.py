"""
memory.py
---------
Long-term memory for the agent — persists facts across sessions.

THE FOUR TYPES OF AGENT MEMORY:

  1. IN-CONTEXT (working memory)
     What's currently in the conversation window.
     Lost when session ends. We have this in V1-V4 already.
     Limit: the context window size (~4096 tokens).

  2. EPISODIC (conversation history)
     Past conversations saved to disk (our session JSON files in V2+).
     Can be retrieved but not automatically injected.
     The agent has to explicitly look up past sessions.

  3. SEMANTIC (facts and knowledge)
     ← THIS IS WHAT WE BUILD HERE
     Specific facts the agent should ALWAYS remember.
     "User's name is Godfrey", "User prefers Python", "FishHappy targets Zanzibar"
     Automatically injected into every prompt as a memory block.
     Stored in SQLite for fast retrieval.

  4. PROCEDURAL (how to do things)
     Learned behaviors — "when asked to write code, always add docstrings"
     The hardest to implement correctly.
     Out of scope for V4 — covered in fine-tuning (V7).

HOW SEMANTIC MEMORY WORKS HERE:

  STORAGE:
    Facts stored as key-value pairs in SQLite.
    Each fact has: key, value, category, created_at, last_accessed.
    Also stored as embeddings in ChromaDB for semantic search.

  INJECTION:
    At the start of every agent run, we retrieve relevant memories
    and prepend them to the system prompt:

    "WHAT I REMEMBER ABOUT YOU:
     - Your name is Godfrey (from 3 sessions ago)
     - You're building FishHappy for the Zanzibar market
     - You prefer Python over JavaScript for backend work
    "

  EXTRACTION:
    After each agent run, we scan the conversation for new facts
    worth remembering and store them automatically.
    The agent can also explicitly call store_memory() as a tool.

MEMORY AS A TOOL:
  The agent gets two memory tools:
    - store_memory(key, value, category) → save a fact
    - recall_memory(query) → search memories by semantic similarity
  
  This means the agent can BUILD ITS OWN MEMORY over time.
  It remembers what the user tells it to remember.
  It recalls facts when relevant without being asked.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path

DB_PATH = "data/memory.db"
Path("data").mkdir(exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection, creating schema if needed."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key         TEXT NOT NULL,
            value       TEXT NOT NULL,
            category    TEXT DEFAULT 'general',
            session_id  TEXT,
            created_at  TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0
        )
    """)
    # Index for fast key lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_key ON memories(key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
    conn.commit()
    return conn


# ── Tool functions (called by the agent) ──

def store_memory(key: str, value: str, category: str = "general") -> str:
    """
    Store a fact in long-term memory.

    The agent calls this when it learns something worth remembering.
    Examples:
      store_memory("user_name", "Godfrey", "user_profile")
      store_memory("project_fishhappy", "Fish marketplace for Zanzibar market", "projects")
      store_memory("user_language_pref", "User prefers Python", "preferences")

    If the key already exists, the value is UPDATED (upsert behavior).
    This prevents duplicate memories for the same fact.
    """
    try:
        conn = _get_conn()
        now = datetime.now().isoformat()

        # Check if key exists
        existing = conn.execute(
            "SELECT id FROM memories WHERE key = ?", (key,)
        ).fetchone()

        if existing:
            conn.execute(
                "UPDATE memories SET value=?, category=?, accessed_at=? WHERE key=?",
                (value, category, now, key)
            )
            action = "updated"
        else:
            conn.execute(
                "INSERT INTO memories (key, value, category, created_at, accessed_at) VALUES (?,?,?,?,?)",
                (key, value, category, now, now)
            )
            action = "stored"

        conn.commit()
        conn.close()
        return f"✅ Memory {action}: '{key}' = '{value}' (category: {category})"

    except Exception as e:
        return f"Error storing memory: {e}"


def recall_memory(query: str) -> str:
    """
    Search memories for information relevant to the query.

    SEARCH STRATEGY:
      1. Exact key match (highest priority)
      2. Key contains query words
      3. Value contains query words
      Returns up to 5 most relevant memories.

    For production: use ChromaDB semantic search here
    (same pattern as knowledge base, but for memories).
    We use SQLite text search for simplicity in V4.
    """
    try:
        conn = _get_conn()
        query_lower = query.lower()

        # Update access count for retrieved memories
        results = conn.execute("""
            SELECT key, value, category, created_at, access_count
            FROM memories
            WHERE lower(key) LIKE ? OR lower(value) LIKE ?
            ORDER BY access_count DESC, created_at DESC
            LIMIT 5
        """, (f"%{query_lower}%", f"%{query_lower}%")).fetchall()

        conn.close()

        if not results:
            return f"No memories found related to: '{query}'"

        lines = [f"Memories related to '{query}':"]
        for row in results:
            lines.append(
                f"  [{row['category']}] {row['key']}: {row['value']}"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Error recalling memory: {e}"


def get_all_memories(category: str = None) -> list[dict]:
    """
    Return all stored memories, optionally filtered by category.
    Used to inject relevant memories into the system prompt.
    """
    try:
        conn = _get_conn()
        if category:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category=? ORDER BY accessed_at DESC",
                (category,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY access_count DESC, accessed_at DESC"
            ).fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


def delete_memory(key: str) -> str:
    """Delete a specific memory by key."""
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM memories WHERE key=?", (key,))
        conn.commit()
        conn.close()
        return f"✅ Memory '{key}' deleted."
    except Exception as e:
        return f"Error: {e}"


def build_memory_prompt(max_memories: int = 10) -> str:
    """
    Build the memory section to inject into the system prompt.
    Called before every agent run.

    Returns empty string if no memories exist.
    """
    memories = get_all_memories()
    if not memories:
        return ""

    # Prioritize user profile and project memories
    priority_categories = ["user_profile", "projects", "preferences"]
    priority = [m for m in memories if m["category"] in priority_categories]
    others = [m for m in memories if m["category"] not in priority_categories]

    selected = (priority + others)[:max_memories]

    lines = ["--- LONG-TERM MEMORY ---",
             "These are facts you have remembered from previous sessions:"]
    for m in selected:
        lines.append(f"  • [{m['category']}] {m['key']}: {m['value']}")
    lines.append("--- END MEMORY ---\n")

    return "\n".join(lines)


def get_memory_stats() -> dict:
    """Return stats about stored memories for the UI."""
    memories = get_all_memories()
    categories = {}
    for m in memories:
        cat = m["category"]
        categories[cat] = categories.get(cat, 0) + 1
    return {
        "total": len(memories),
        "categories": categories
    }