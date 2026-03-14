"""
memory.py
---------
Long-term SQLite memory.

Storage path:
  - HF Spaces: /data/memory.db (persistent volume)
  - Local:     ./data/memory.db
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime


def get_db_path() -> str:
    if os.path.exists("/data"):
        return "/data/memory.db"
    Path("data").mkdir(exist_ok=True)
    return "data/memory.db"


def _get_conn():
    conn = sqlite3.connect(get_db_path())
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def store_memory(key: str, value: str, category: str = "general") -> str:
    now = datetime.now().isoformat()
    try:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO memories (key, value, category, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value, category=excluded.category, updated_at=excluded.updated_at
        """, (key, value, category, now, now))
        conn.commit()
        conn.close()
        return f"Stored: [{category}] {key} = {value}"
    except Exception as e:
        return f"Error: {e}"


def recall_memory(query: str) -> str:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT key, value, category FROM memories WHERE key LIKE ? OR value LIKE ?",
            (f"%{query}%", f"%{query}%")
        ).fetchall()
        conn.close()
        if not rows:
            return f"No memories found for '{query}'."
        return "\n".join(f"[{cat}] {k}: {v}" for k, v, cat in rows)
    except Exception as e:
        return f"Error: {e}"


def get_all_memories() -> list:
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, key, value, category, created_at FROM memories ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [{"id": r[0], "key": r[1], "value": r[2], "category": r[3], "created_at": r[4]} for r in rows]
    except Exception:
        return []


def delete_memory(key: str) -> str:
    try:
        conn = _get_conn()
        conn.execute("DELETE FROM memories WHERE key = ?", (key,))
        conn.commit()
        conn.close()
        return f"Deleted: {key}"
    except Exception as e:
        return f"Error: {e}"


def get_memory_stats() -> dict:
    try:
        conn = _get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        cats = conn.execute("SELECT category, COUNT(*) FROM memories GROUP BY category").fetchall()
        conn.close()
        return {"total": total, "categories": {c: n for c, n in cats}}
    except Exception:
        return {"total": 0, "categories": {}}


def build_memory_prompt() -> str:
    memories = get_all_memories()
    if not memories:
        return ""
    lines = ["[Long-term memory]"]
    for m in memories:
        lines.append(f"  {m['key']}: {m['value']}")
    return "\n".join(lines)