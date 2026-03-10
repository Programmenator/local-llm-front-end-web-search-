"""
services/session_service.py

Purpose:
    Persistence and retrieval for saved conversation sessions.

What this file does:
    - Creates new session objects.
    - Saves sessions to JSON files.
    - Lists existing sessions for the UI.
    - Loads a selected session back into memory.
    - Deletes sessions when requested.

How this file fits into the system:
    The conversation session manager depends on a clean persistence layer. This
    service isolates filesystem and JSON concerns from the controller and UI.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from models.conversation_session import ConversationSession


class SessionService:
    """Manage saved conversation sessions on local disk."""

    def __init__(self, sessions_dir: Path | None = None) -> None:
        """Create the session storage directory reference."""
        self.sessions_dir = sessions_dir or Path(__file__).resolve().parent.parent / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, model_name: str, title: str | None = None) -> ConversationSession:
        """Create a new in-memory session object with a generated ID."""
        now = datetime.now()
        session_title = title or f"Session {now.strftime('%Y-%m-%d %I:%M %p')}"
        return ConversationSession(
            session_id=uuid4().hex,
            title=session_title,
            created_at=now,
            updated_at=now,
            model_name=model_name,
            messages=[],
        )

    def save_session(self, session: ConversationSession) -> ConversationSession:
        """Write a session to disk and return the same updated object.

        Stability note:
            Session files are now written through a temporary file plus atomic
            replace so an interrupted save is less likely to leave behind a
            corrupted conversation JSON file.
        """
        session.updated_at = datetime.now()
        file_path = self.sessions_dir / self._build_filename(session)
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(session.to_dict(), file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, file_path)
        self._remove_old_duplicates(session.session_id, keep_path=file_path)
        return session

    def list_sessions(self) -> List[ConversationSession]:
        """Load all saved sessions sorted by most recently updated first."""
        sessions: List[ConversationSession] = []
        for file_path in sorted(self.sessions_dir.glob("*.json")):
            try:
                with file_path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
                sessions.append(ConversationSession.from_dict(payload))
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        sessions.sort(key=lambda item: item.updated_at, reverse=True)
        return sessions

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load one saved session by its ID, returning None if not found."""
        for session in self.list_sessions():
            if session.session_id == session_id:
                return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete all files that match a saved session ID."""
        deleted_any = False
        for file_path in self.sessions_dir.glob(f"*__{session_id}.json"):
            try:
                file_path.unlink()
                deleted_any = True
            except OSError:
                continue
        return deleted_any

    def _build_filename(self, session: ConversationSession) -> str:
        """Generate a stable, readable filename for one session."""
        safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", session.title).strip("_") or "session"
        return f"{safe_title}__{session.session_id}.json"

    def _remove_old_duplicates(self, session_id: str, keep_path: Path) -> None:
        """Remove older files for the same session after a title change or resave."""
        for file_path in self.sessions_dir.glob(f"*__{session_id}.json"):
            if file_path != keep_path:
                try:
                    file_path.unlink()
                except OSError:
                    continue
