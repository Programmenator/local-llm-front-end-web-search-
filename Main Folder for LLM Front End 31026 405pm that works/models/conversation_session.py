"""
models/conversation_session.py

Purpose:
    Structured data model for saved conversation sessions.

What this file does:
    - Defines the session object persisted to disk.
    - Stores session metadata and message lists.
    - Provides JSON serialization helpers.

How this file fits into the system:
    The session manager needs a stable format for saving, loading, and listing
    conversations. This file keeps that format centralized and readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from models.chat_message import ChatMessage


@dataclass
class ConversationSession:
    """Represent one saved conversation session.

    Attributes:
        session_id: Stable identifier used for filenames and lookup.
        title: Human-readable label shown in the UI session list.
        created_at: When the session was first created.
        updated_at: When the session was last modified.
        model_name: Model selected when the session was last saved.
        messages: Ordered list of chat messages in the conversation.
    """

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model_name: str
    messages: List[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the session into a JSON-safe dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "model_name": self.model_name,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConversationSession":
        """Build a ConversationSession from previously saved JSON data."""
        return cls(
            session_id=str(payload.get("session_id", "")),
            title=str(payload.get("title", "Untitled Session")),
            created_at=datetime.fromisoformat(str(payload.get("created_at"))),
            updated_at=datetime.fromisoformat(str(payload.get("updated_at"))),
            model_name=str(payload.get("model_name", "")),
            messages=[ChatMessage.from_dict(item) for item in payload.get("messages", [])],
        )
