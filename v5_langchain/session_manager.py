"""
session_manager.py
------------------
Session persistence for V5. Identical to V4 — JSON file storage.

The only difference: we no longer store raw message dicts in the session
for the conversation history (LangChain memory handles that).
We store session metadata and the display messages for the UI.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

SESSIONS_DIR = Path("data/sessions_v5")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def create_session(model: str = "qwen3:1.7b", system_prompt: str = "") -> dict:
    session_id = str(uuid.uuid4())[:8]
    session = {
        "id": session_id,
        "title": "New conversation",
        "model": model,
        "system_prompt": system_prompt,
        "messages": [],        # display messages for UI
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    save_session(session)
    return session


def save_session(session: dict):
    session["updated_at"] = datetime.now().isoformat()
    if session.get("messages"):
        first_user = next(
            (m["content"][:40] for m in session["messages"] if m["role"] == "user"), None
        )
        if first_user:
            session["title"] = first_user
    path = SESSIONS_DIR / f"{session['id']}.json"
    path.write_text(json.dumps(session, ensure_ascii=False, indent=2))


def load_session(session_id: str) -> dict | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_sessions() -> list[dict]:
    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({"id": data["id"], "title": data.get("title", "Untitled")})
        except Exception:
            continue
    return sessions


def delete_session(session_id: str):
    path = SESSIONS_DIR / f"{session_id}.json"
    if path.exists():
        path.unlink()