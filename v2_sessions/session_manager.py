"""
session_manager.py
------------------
Handles everything related to saving and loading conversations.

WHY FILES AND NOT A DATABASE?
  At this stage, JSON files are perfect:
  - Zero setup (no database server needed)
  - Human readable (you can open any session in a text editor)
  - Easy to debug
  V3+ will introduce ChromaDB for vector storage, but chat history
  is just structured text — JSON is the right tool here.

HOW SESSIONS WORK:
  Each conversation = one JSON file in data/sessions/
  Filename = session ID (a UUID) + .json
  
  File structure:
  {
    "id": "abc123...",
    "title": "First message used as title",
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T10:45:00",
    "model": "qwen3:1.7b",
    "system_prompt": "You are a helpful assistant...",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
"""

import json
import uuid
from datetime import datetime
from pathlib import Path


SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def create_session(model: str, system_prompt: str) -> dict:
    """
    Create a brand new empty session.
    Returns the session dict — not saved to disk yet.
    We save on first message so we don't create empty ghost sessions.
    """
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "model": model,
        "system_prompt": system_prompt,
        "messages": []
    }


def save_session(session: dict) -> None:
    """
    Write session to disk as JSON.
    Called after every assistant response so nothing is ever lost.

    WHY AFTER EVERY MESSAGE AND NOT ON CLOSE:
      Browser tabs get closed, computers crash.
      Write on every turn = no data loss ever.
      JSON writes are fast — no performance concern at this scale.
    """
    session["updated_at"] = datetime.now().isoformat()

    # Auto-generate title from first user message
    if session["title"] == "New conversation" and session["messages"]:
        first_user_msg = next(
            (m["content"] for m in session["messages"] if m["role"] == "user"),
            None
        )
        if first_user_msg:
            # Truncate to 60 chars for a clean sidebar title
            session["title"] = first_user_msg[:60] + ("..." if len(first_user_msg) > 60 else "")

    filepath = SESSIONS_DIR / f"{session['id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def load_session(session_id: str) -> dict | None:
    """
    Load a session from disk by ID.
    Returns None if not found — always handle this case in the UI.
    """
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions(limit: int = 50) -> list[dict]:
    """
    Return all sessions sorted by most recently updated first.
    Returns lightweight summaries (no messages) for sidebar display.
    Loading full messages for every session would be slow with many sessions.
    """
    sessions = []

    for filepath in SESSIONS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Only load what the sidebar needs — not the full message history
                sessions.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                    "model": data.get("model", ""),
                    "message_count": len(data.get("messages", []))
                })
        except (json.JSONDecodeError, KeyError):
            continue  # Skip corrupted files silently

    # Sort by most recently updated
    sessions.sort(key=lambda x: x["updated_at"], reverse=True)
    return sessions[:limit]


def delete_session(session_id: str) -> bool:
    """
    Delete a session file. Returns True if deleted, False if not found.
    """
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def rename_session(session_id: str, new_title: str) -> bool:
    """
    Rename a session. Load, update title, save back.
    """
    session = load_session(session_id)
    if not session:
        return False
    session["title"] = new_title.strip()
    save_session(session)
    return True