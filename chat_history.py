import json
import time
import uuid
from pathlib import Path

SESSIONS_DIR = "./chat_sessions"


class ChatSessionManager:
    """Persists chat sessions (message lists) to JSON files on disk."""

    def __init__(self, sessions_dir=SESSIONS_DIR):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)

    def _path(self, session_id):
        return self.sessions_dir / f"{session_id}.json"

    def create_session(self):
        session_id = uuid.uuid4().hex
        now = time.time()
        data = {"id": session_id, "title": "New Chat", "created_at": now, "updated_at": now, "messages": []}
        self._save(data)
        return session_id

    def _save(self, data):
        with open(self._path(data["id"]), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, session_id):
        path = self._path(session_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_sessions(self):
        """Return session summaries sorted by most recently updated first."""
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({"id": data["id"], "title": data["title"], "updated_at": data["updated_at"]})
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions

    def get_messages(self, session_id):
        data = self._load(session_id)
        return data["messages"] if data else []

    def append_message(self, session_id, role, content):
        data = self._load(session_id)
        if data is None:
            return
        data["messages"].append({"role": role, "content": content})
        if data["title"] == "New Chat" and role == "user":
            title = content.strip().splitlines()[0]
            data["title"] = (title[:40] + "...") if len(title) > 40 else title
        data["updated_at"] = time.time()
        self._save(data)

    def delete_session(self, session_id):
        path = self._path(session_id)
        if path.exists():
            path.unlink()
