"""Core data models and contracts for telegram-group-bot."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role in conversational turn."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MediaAttachment(BaseModel):
    """Represents a media attachment (photo, document, voice note) passed to providers."""

    file_path: Optional[Path] = None
    mime_type: str = "application/octet-stream"
    data: Optional[bytes] = None
    description: Optional[str] = None
    file_name: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    def get_bytes(self) -> bytes:
        """Return raw media bytes from memory buffer or file path."""
        if self.data is not None:
            return self.data
        if self.file_path is not None and self.file_path.is_file():
            return self.file_path.read_bytes()
        return b""


class UsageInfo(BaseModel):
    """Token usage metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0


class ProviderResponse(BaseModel):
    """Standardized response from an LLM/agent provider."""

    text: str
    thinking: Optional[str] = None
    conversation_id: Optional[str] = None
    finish_reason: str = "stop"
    usage: Optional[UsageInfo] = None
    model: Optional[str] = None
    raw_response: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}


class ChatMessage(BaseModel):
    """Structured representation of a Telegram chat message for memory and pipelines."""

    chat_id: int
    message_id: int
    user_id: int
    username: Optional[str] = None
    text: Optional[str] = None
    reply_to_msg_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    media_description: Optional[str] = None
    is_bot_reply: bool = False
