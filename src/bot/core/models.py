"""Core data models and contracts for telegram-mtproto-userbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


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
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    source: Optional[str] = None
    model: Optional[str] = None

    @model_validator(mode="after")
    def _sync_cache_fields(self) -> "UsageInfo":
        """Keep legacy cached_tokens and normalized cache_read_tokens aligned."""
        if self.cache_read_tokens == 0 and self.cached_tokens:
            self.cache_read_tokens = self.cached_tokens
        elif self.cached_tokens == 0 and self.cache_read_tokens:
            self.cached_tokens = self.cache_read_tokens
        return self


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


@dataclass(slots=True)
class GeneratedImageResult:
    """Validated image artifact produced by native Antigravity image generation."""

    image_bytes: bytes
    mime_type: str
    filename: str
    width: int = 0
    height: int = 0
    prompt_used: str = ""
    caption_suggestion: str = ""
    model_used: str = ""
    conversation_id: Optional[str] = None
    duration_seconds: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


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
