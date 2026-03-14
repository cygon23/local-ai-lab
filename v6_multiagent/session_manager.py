"""
session_manager.py
------------------
Conversation session persistence.

Storage path:
  - HF Spaces: /data/sessions/
  - Local:     ./data/sessions/
"""

import os
import json
import uuid
from pathlib import Path
from datetime import datetime


def get_sessions_dir() -> Path:
    if os.path.exists("/data"):
        p = Path("/data/sessions")
    else:
        p = Path("data/sessions")
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_session() -> dict:
    return {
        "id": str(uuid.uuid4())[:8],
        "title": "New conversation",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }


def save_session(session: dict):
    try:
        path = get_sessions_dir() / f"{session['id']}.json"
        if session["messages"]:
            first_user = next(
                (m["content"] for m in session["messages"] if m["role"] == "user"), None
            )
            if first_user:
                session["title"] = first_user[:40]
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_session(session_id: str) -> dict | None:
    try:
        path = get_sessions_dir() / f"{session_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def list_sessions() -> list:
    try:
        files = sorted(
            get_sessions_dir().glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        sessions = []
        for f in files:
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({"id": s["id"], "title": s.get("title", "Untitled")})
            except Exception:
                pass
        return sessions
    except Exception:
        return []


def delete_session(session_id: str):
    try:
        path = get_sessions_dir() / f"{session_id}.json"
        if path.exists():
            path.unlink()
    except Exception:
        pass