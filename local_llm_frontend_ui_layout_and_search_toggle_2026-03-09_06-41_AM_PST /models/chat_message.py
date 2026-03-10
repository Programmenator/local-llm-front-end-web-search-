"""
models/chat_message.py

Purpose:
    Structured data model for chat messages and stream-aware assistant state.

What this file does:
    - Defines a message object used across UI, controller, and persistence.
    - Stores visible assistant content and optional thinking trace text.
    - Provides helpers for JSON serialization and deserialization.

How this file fits into the system:
    The controller, UI, and session persistence layer all depend on one stable
    message shape. This file makes that shape explicit and extendable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class ChatMessage:
    """Represent one chat message in the conversation history.

    Attributes:
        role: Message role such as 'user', 'assistant', or 'system'.
        content: Visible message text that should appear in the main transcript.
        timestamp: Creation or completion time for the message.
        thinking: Optional reasoning trace text captured from thinking-capable
            Ollama models. This is stored separately so the UI can place it in a
            collapsible section instead of mixing it into the visible answer.
    """

    role: str
    content: str
    timestamp: datetime
    thinking: str = field(default="")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the message into a JSON-safe dictionary for persistence."""
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChatMessage":
        """Build a ChatMessage from previously saved JSON session data."""
        timestamp_value = payload.get("timestamp")
        if isinstance(timestamp_value, str):
            parsed_time = datetime.fromisoformat(timestamp_value)
        else:
            parsed_time = datetime.now()

        return cls(
            role=str(payload.get("role", "system")),
            content=str(payload.get("content", "")),
            timestamp=parsed_time,
            thinking=str(payload.get("thinking", "")),
        )
