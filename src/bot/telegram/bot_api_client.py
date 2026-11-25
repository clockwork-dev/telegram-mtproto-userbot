"""Classic Telegram Bot API Client (HTTPS Long-Polling + Media Download).

Provides the exact same interface as TelegramUserbot (`start`, `stop`, `send_reply`,
`send_photo`, `send_document`, `send_voice`, `send_reaction`, `register_message_handler`,
`client.download_media`) using only a `@BotFather` token (`TELEGRAM_BOT_TOKEN`),
without requiring `api_id` or `api_hash`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from bot.telegram.client import (
    MAX_DOCUMENT_SIZE_BYTES,
    calculate_typing_delay,
    validate_document_path,
)
from bot.telegram.handlers import IncomingMessage, MessageDispatcher
from bot.telegram.heuristics import ReplyHeuristics

if TYPE_CHECKING:
    from bot.security.sanitizer import SecretSanitizer

logger = logging.getLogger(__name__)


# Official 73 emojis supported by Telegram Bot API ReactionTypeEmoji
_VALID_TELEGRAM_REACTIONS: frozenset[str] = frozenset({
    "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱",
    "🤬", "😢", "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊", "🤡",
    "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯", "🤣", "⚡",
    "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈",
    "😴", "😭", "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨",
    "🤝", "✍", "🤗", "🫡", "🎅", "🎄", "☃", "💅", "🤪", "🗿",
    "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾", "🤷‍♂",
    "🤷", "🤷‍♀", "😡",
})

# Common emoji aliases not directly accepted by Telegram Bot API setMessageReaction
_REACTION_EMOJI_ALIASES: Dict[str, str] = {
    "😂": "🤣",
    "❤️": "❤",
    "♥️": "❤",
    "♥": "❤",
    "☺️": "🥰",
    "😊": "🥰",
    "😆": "😁",
    "😅": "😁",
    "🙂": "👌",
    "💀": "👻",
    "☠️": "👻",
    "☠": "👻",
    "❤️‍🔥": "❤‍🔥",
    "🤦": "🤷",
    "🤦‍♂️": "🤷‍♂",
    "🤦‍♀️": "🤷‍♀",
    "🤷‍♂️": "🤷‍♂",
    "🤷‍♀️": "🤷‍♀",
    "☃️": "☃",
    "⚡️": "⚡",
    "✌️": "👌",
}


def normalize_reaction_emoji(emoji: str) -> str:
    """Normalize an emoji into a supported Telegram Bot API ReactionTypeEmoji."""
    raw = (emoji or "").strip()
    if not raw:
        return "👍"
    if raw in _REACTION_EMOJI_ALIASES:
        return _REACTION_EMOJI_ALIASES[raw]
    if raw in _VALID_TELEGRAM_REACTIONS:
        return raw
    no_vs = raw.replace("\ufe0f", "")
    if no_vs in _REACTION_EMOJI_ALIASES:
        return _REACTION_EMOJI_ALIASES[no_vs]
    if no_vs in _VALID_TELEGRAM_REACTIONS:
        return no_vs
    return "👍"


_TELEGRAM_MAX_CAPTION_LEN = 1024

_MIME_TO_IMAGE_EXT: Dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _truncate_telegram_caption(caption: Optional[str]) -> Optional[str]:
    """Truncate caption to Telegram Bot API's 1024-character limit."""
    if caption is None:
        return None
    if len(caption) > _TELEGRAM_MAX_CAPTION_LEN:
        return caption[: _TELEGRAM_MAX_CAPTION_LEN - 3] + "..."
    return caption


def _infer_image_mime_and_ext(
    data: bytes,
    *,
    mime_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> tuple[str, str]:
    """Infer `(mime_type, extension)` for an image payload from explicit MIME, magic bytes, or filename."""
    if mime_type and mime_type.strip():
        norm = mime_type.strip().lower()
        if norm == "image/jpg":
            norm = "image/jpeg"
        ext = _MIME_TO_IMAGE_EXT.get(norm)
        if ext is not None:
            return norm, ext
        guessed_ext = mimetypes.guess_extension(norm) or ".jpg"
        return norm, guessed_ext

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"

    if filename:
        guessed_mime = mimetypes.guess_type(filename)[0]
        if guessed_mime:
            norm = "image/jpeg" if guessed_mime == "image/jpg" else guessed_mime
            return norm, _MIME_TO_IMAGE_EXT.get(norm, Path(filename).suffix or ".jpg")

    return "image/jpeg", ".jpg"


def _encode_multipart_form_data(
    fields: Dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    mime_type: Optional[str] = None,
) -> tuple[bytes, str]:
    """Encode form fields and a single binary file into a multipart/form-data payload."""
    boundary = f"----TelegramBotBoundary{uuid.uuid4().hex}"
    safe_filename = Path(filename).name.replace("\r", "").replace("\n", "").replace('"', "_")
    if not safe_filename:
        safe_filename = "document"

    detected_mime = mime_type or mimetypes.guess_type(safe_filename)[0] or "application/octet-stream"

    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{safe_filename}"\r\n'
            f"Content-Type: {detected_mime}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


@dataclass
class _BotAPIFileInfo:
    """Adapter exposing `.name`, `.size`, `.mime_type`, `.emoji` for MediaProcessor."""

    file_id: str
    name: Optional[str] = None
    size: int = 0
    mime_type: Optional[str] = None
    emoji: Optional[str] = None


class BotAPIMessageEvent:
    """Adapter wrapping a Bot API `message` dict for ReplyHeuristics and MediaProcessor."""

    def __init__(self, msg_dict: Dict[str, Any], bot_id: int, *, _parse_reply: bool = True) -> None:
        self.raw_dict = msg_dict
        self.id: int = int(msg_dict.get("message_id", 0))
        chat = msg_dict.get("chat") or {}
        self.chat_id: int = int(chat.get("id", 0))
        self.chat_type: str = str(chat.get("type") or "").strip().lower()
        self.is_private: bool = self.chat_type == "private"
        self.date: Optional[int] = msg_dict.get("date")

        sender = msg_dict.get("from") or {}
        self.sender_id: int = int(sender.get("id", 0))
        self.sender_is_bot: bool = bool(sender.get("is_bot", False))
        first = sender.get("first_name") or ""
        last = sender.get("last_name") or ""
        full_name = f"{first} {last}".strip()
        raw_username = sender.get("username")
        username = raw_username.strip() if isinstance(raw_username, str) and raw_username.strip() else ""
        self.sender_username: Optional[str] = username or None
        self.sender_name: str = full_name or (f"@{username}" if username else f"User_{self.sender_id}")

        self.out: bool = self.sender_id == bot_id
        self.raw_text: str = msg_dict.get("text") or msg_dict.get("caption") or ""
        self.text: str = self.raw_text

        reply_msg = msg_dict.get("reply_to_message") if _parse_reply else None
        self.is_reply: bool = isinstance(reply_msg, dict)
        self.reply_to_msg_id: Optional[int] = (
            int(reply_msg["message_id"])
            if isinstance(reply_msg, dict) and "message_id" in reply_msg
            else None
        )
        reply_from = reply_msg.get("from") if isinstance(reply_msg, dict) else None
        self.reply_to_sender_id: Optional[int] = (
            int(reply_from["id"])
            if isinstance(reply_from, dict) and "id" in reply_from
            else None
        )
        self.is_reply_to_bot: bool = bool(
            self.reply_to_sender_id is not None and self.reply_to_sender_id == bot_id
        )
        self.reply_to_text: Optional[str] = None
        self.reply_to_author: Optional[str] = None
        self.reply_to_media_event: Optional["BotAPIMessageEvent"] = None
        if isinstance(reply_msg, dict):
            r_text = (reply_msg.get("text") or reply_msg.get("caption") or "").strip()
            self.reply_to_text = r_text or None
            if isinstance(reply_from, dict):
                r_first = reply_from.get("first_name") or ""
                r_last = reply_from.get("last_name") or ""
                r_full = f"{r_first} {r_last}".strip()
                r_user = reply_from.get("username") or ""
                r_id = int(reply_from.get("id", 0))
                self.reply_to_author = r_full or (f"@{r_user}" if r_user else f"User_{r_id}")
        self.entities: List[Any] = []

        # Media detection flags for MediaProcessor
        self.photo: bool = False
        self.sticker: bool = False
        self.voice: bool = False
        self.video_note: bool = False
        self.video: bool = False
        self.gif: bool = False
        self.animation: bool = False
        self.document: bool = False
        self.media: bool = False
        self.file: Optional[_BotAPIFileInfo] = None
        self.file_id: Optional[str] = None
        self._file_id: Optional[str] = None
        self.file_name: Optional[str] = None
        self.mime_type: Optional[str] = None
        self.file_size: int = 0
        self.message = self

        self._extract_media_info(msg_dict)
        if _parse_reply and not self.media and isinstance(reply_msg, dict):
            reply_ev = BotAPIMessageEvent(reply_msg, bot_id=bot_id, _parse_reply=False)
            if reply_ev.media:
                self.reply_to_media_event = reply_ev

    def _extract_media_info(self, msg_dict: Dict[str, Any]) -> None:
        if "photo" in msg_dict and isinstance(msg_dict["photo"], list) and msg_dict["photo"]:
            largest = msg_dict["photo"][-1]
            self.photo = True
            self.media = True
            self.file_id = largest.get("file_id")
            self._file_id = self.file_id
            self.file_name = "photo.jpg"
            self.file_size = int(largest.get("file_size") or 0)
            self.mime_type = "image/jpeg"
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )
        elif "sticker" in msg_dict and isinstance(msg_dict["sticker"], dict):
            stk = msg_dict["sticker"]
            # Only download static WebP stickers (skip animated .tgs / video .webm stickers)
            if not stk.get("is_animated") and not stk.get("is_video"):
                self.sticker = True
                self.media = True
                self.file_id = stk.get("file_id")
                self._file_id = self.file_id
                self.file_name = "sticker.webp"
                self.file_size = int(stk.get("file_size") or 0)
                self.mime_type = "image/webp"
                self.file = _BotAPIFileInfo(
                    file_id=self.file_id or "",
                    name=self.file_name,
                    size=self.file_size,
                    mime_type=self.mime_type,
                    emoji=stk.get("emoji"),
                )
        elif "voice" in msg_dict and isinstance(msg_dict["voice"], dict):
            vc = msg_dict["voice"]
            self.voice = True
            self.media = True
            self.file_id = vc.get("file_id")
            self._file_id = self.file_id
            self.file_name = "voice.ogg"
            self.file_size = int(vc.get("file_size") or 0)
            self.mime_type = vc.get("mime_type") or "audio/ogg"
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )
        elif "animation" in msg_dict and isinstance(msg_dict["animation"], dict):
            anim = msg_dict["animation"]
            self.gif = True
            self.animation = True
            self.video = True
            self.document = False
            self.media = True
            self.file_id = anim.get("file_id")
            self._file_id = self.file_id
            self.mime_type = anim.get("mime_type") or "video/mp4"
            default_ext = ".gif" if self.mime_type == "image/gif" else ".mp4"
            self.file_name = anim.get("file_name") or f"animation{default_ext}"
            self.file_size = int(anim.get("file_size") or 0)
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )
        elif "video" in msg_dict and isinstance(msg_dict["video"], dict):
            vid = msg_dict["video"]
            self.video = True
            self.document = False
            self.media = True
            self.file_id = vid.get("file_id")
            self._file_id = self.file_id
            self.mime_type = vid.get("mime_type") or "video/mp4"
            self.file_name = vid.get("file_name") or "video.mp4"
            self.file_size = int(vid.get("file_size") or 0)
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )
        elif "video_note" in msg_dict and isinstance(msg_dict["video_note"], dict):
            vn = msg_dict["video_note"]
            self.video_note = True
            self.video = True
            self.document = False
            self.media = True
            self.file_id = vn.get("file_id")
            self._file_id = self.file_id
            self.file_name = "video_note.mp4"
            self.file_size = int(vn.get("file_size") or 0)
            self.mime_type = "video/mp4"
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )
        elif "document" in msg_dict and isinstance(msg_dict["document"], dict):
            doc = msg_dict["document"]
            doc_mime = (doc.get("mime_type") or "").lower().strip()
            doc_name = doc.get("file_name") or ""
            doc_ext = Path(doc_name).suffix.lower() if doc_name else ""
            is_gif_doc = doc_mime == "image/gif" or doc_ext == ".gif"
            is_video_doc = (
                is_gif_doc
                or doc_mime.startswith("video/")
                or doc_ext in {".mp4", ".webm", ".mov", ".mkv", ".gif"}
            )
            self.file_id = doc.get("file_id")
            self._file_id = self.file_id
            self.file_size = int(doc.get("file_size") or 0)
            self.media = True
            if is_video_doc:
                self.video = True
                if is_gif_doc:
                    self.gif = True
                    self.animation = True
                self.document = False
                self.file_name = doc_name or ("animation.gif" if is_gif_doc else "video.mp4")
                self.mime_type = doc.get("mime_type") or (
                    "image/gif" if is_gif_doc else "video/mp4"
                )
            else:
                self.document = True
                self.file_name = doc_name or "document"
                self.mime_type = doc.get("mime_type")
            self.file = _BotAPIFileInfo(
                file_id=self.file_id or "",
                name=self.file_name,
                size=self.file_size,
                mime_type=self.mime_type,
            )


class _BotAPIDownloaderClient:
    """Provides `download_media(message, file=...)` for MediaProcessor compatibility."""

    def __init__(self, parent: "TelegramBotAPIClient") -> None:
        self._parent = parent

    async def download_media(self, message: Any, file: str) -> str:
        file_id = getattr(message, "file_id", None) or getattr(message, "_file_id", None)
        if not file_id and hasattr(message, "file") and message.file:
            file_id = getattr(message.file, "file_id", None)
        if not file_id:
            raise RuntimeError("No file_id present on BotAPIMessageEvent for download.")
        target = await self._parent.download_file_by_id(file_id, Path(file))
        return str(target)


class TelegramBotAPIClient:
    """Asynchronous Telegram Bot API client using long-polling (`getUpdates`)."""

    def __init__(
        self,
        bot_token: str,
        allowed_chat_ids: Optional[List[int]] = None,
        allowed_dm_user_ids: Optional[List[int]] = None,
        ignored_sender_usernames: Optional[List[str]] = None,
        bot_names: Optional[List[str]] = None,
        typing_cpm: int = 300,
        typing_min_seconds: float = 1.0,
        typing_max_seconds: float = 8.0,
        backlog_Callback: Optional[Callable[[IncomingMessage], Awaitable[None]]] = None,
        backlog_callback: Optional[Callable[[IncomingMessage], Awaitable[None]]] = None,
        sanitizer: Optional["SecretSanitizer"] = None,
    ) -> None:
        self.bot_token = bot_token.strip()
        if not self.bot_token:
            raise ValueError("Telegram bot_token cannot be empty.")

        self._sanitizer = sanitizer
        if self._sanitizer is not None:
            self._sanitizer.register_secret(self.bot_token)

        self.allowed_chat_ids: List[int] = (
            [int(cid) for cid in allowed_chat_ids] if allowed_chat_ids is not None else []
        )
        self._allowed_chat_ids_set = set(self.allowed_chat_ids)
        self.allowed_dm_user_ids: List[int] = (
            [int(uid) for uid in allowed_dm_user_ids] if allowed_dm_user_ids is not None else []
        )
        self._allowed_dm_user_ids_set = set(self.allowed_dm_user_ids)
        self.ignored_sender_usernames: List[str] = (
            list(ignored_sender_usernames)
            if ignored_sender_usernames is not None
            else ["hermes_kuk7yi6kfm3lcdfa_bot"]
        )
        self._ignored_sender_usernames_set = {
            u.lower().lstrip("@") for u in self.ignored_sender_usernames if u and u.strip()
        }
        self.bot_names: List[str] = list(bot_names) if bot_names is not None else []
        self.typing_cpm = typing_cpm
        self.typing_min_seconds = typing_min_seconds
        self.typing_max_seconds = typing_max_seconds
        self.backlog_callback = backlog_Callback or backlog_callback

        self.bot_id: int = 0
        self.bot_username: Optional[str] = None
        self.bot_first_name: Optional[str] = None
        self.dispatcher = MessageDispatcher()
        self.client = _BotAPIDownloaderClient(self)

        self._is_connected = False
        self._polling_task: Optional[asyncio.Task] = None
        self._offset: int = 0
        self._heuristics = ReplyHeuristics()

    def sync_access_lists(
        self,
        *,
        allowed_chat_ids: Optional[List[int]] = None,
        allowed_dm_user_ids: Optional[List[int]] = None,
        ignored_sender_usernames: Optional[List[str]] = None,
        bot_names: Optional[List[str]] = None,
    ) -> None:
        """Synchronize group/DM allowlists, ignored usernames, and bot aliases."""
        if allowed_chat_ids is not None:
            self.allowed_chat_ids = [int(cid) for cid in allowed_chat_ids]
            self._allowed_chat_ids_set = set(self.allowed_chat_ids)
        if allowed_dm_user_ids is not None:
            self.allowed_dm_user_ids = [int(uid) for uid in allowed_dm_user_ids]
            self._allowed_dm_user_ids_set = set(self.allowed_dm_user_ids)
        if ignored_sender_usernames is not None:
            self.ignored_sender_usernames = list(ignored_sender_usernames)
            self._ignored_sender_usernames_set = {
                u.lower().lstrip("@") for u in self.ignored_sender_usernames if u and u.strip()
            }
        if bot_names is not None:
            updated_names = list(bot_names)
            if self.bot_first_name and self.bot_first_name not in updated_names:
                updated_names.append(self.bot_first_name)
            self.bot_names = updated_names

    def _redact_secrets(self, value: Any) -> str:
        text = "" if value is None else str(value)
        if self._sanitizer is not None:
            text = self._sanitizer.sanitize(text)
        if self.bot_token and self.bot_token in text:
            text = text.replace(self.bot_token, "[REDACTED]")
        if self.bot_token:
            quoted_tok = urllib.parse.quote(self.bot_token, safe="")
            if quoted_tok and quoted_tok in text:
                text = text.replace(quoted_tok, "[REDACTED]")
        return text

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    def _file_url(self, file_path: str) -> str:
        return f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"

    def _sync_execute_request(
        self,
        method: str,
        data_bytes: Optional[bytes],
        headers: Dict[str, str],
        timeout: float = 35.0,
    ) -> Dict[str, Any]:
        url = self._api_url(method)
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            req = urllib.request.Request(url, data=data_bytes, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    parsed = json.loads(raw)
                    if not parsed.get("ok"):
                        desc = self._redact_secrets(parsed.get("description"))
                        raise RuntimeError(
                            f"Telegram Bot API {method} failed: {desc}"
                        )
                    return parsed
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                safe_err = self._redact_secrets(err_body)
                raise RuntimeError(
                    f"Telegram Bot API HTTP {exc.code} on {method}: {safe_err}"
                ) from None
            except (ConnectionResetError, urllib.error.URLError, TimeoutError) as exc:
                safe_exc = self._redact_secrets(exc)
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Telegram Bot API network error on {method}: {safe_exc}"
                    ) from None
                logger.warning(
                    "Transient network error on Bot API %s (attempt %d/%d): %s. Retrying in 0.2s...",
                    method,
                    attempt,
                    max_attempts,
                    safe_exc,
                )
                time.sleep(0.2)
        raise RuntimeError(f"Telegram Bot API {method} exhausted retries.")

    def _sync_call_api(
        self,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 35.0,
    ) -> Dict[str, Any]:
        data_bytes = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data_bytes = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return self._sync_execute_request(method, data_bytes, headers, timeout=timeout)

    def _sync_send_document(
        self,
        chat_id: int,
        file_path: Path | bytes,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        filename: Optional[str] = None,
        timeout: float = 120.0,
        *,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(file_path, (bytes, bytearray)):
            file_bytes = bytes(file_path)
            if filename:
                effective_filename = Path(filename).name
            else:
                _, ext = _infer_image_mime_and_ext(file_bytes, mime_type=mime_type)
                effective_filename = f"generated{ext}" if mime_type or ext else "document"
        else:
            effective_filename = Path(filename).name if filename else file_path.name
            file_bytes = file_path.read_bytes()

        fields: Dict[str, str] = {"chat_id": str(int(chat_id))}
        if caption:
            fields["caption"] = caption
        if reply_to_msg_id is not None:
            fields["reply_to_message_id"] = str(int(reply_to_msg_id))

        body, content_type = _encode_multipart_form_data(
            fields=fields,
            file_field="document",
            filename=effective_filename,
            file_bytes=file_bytes,
            mime_type=mime_type,
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        return self._sync_execute_request("sendDocument", body, headers, timeout=timeout)

    def _sync_send_photo(
        self,
        chat_id: int,
        photo_bytes: bytes,
        *,
        filename: str = "generated.jpg",
        mime_type: str = "image/jpeg",
        caption: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
        timeout: float = 90.0,
    ) -> Dict[str, Any]:
        fields: Dict[str, str] = {"chat_id": str(int(chat_id))}
        if caption:
            fields["caption"] = caption
        if reply_to_message_id is not None:
            fields["reply_parameters"] = json.dumps(
                {
                    "message_id": int(reply_to_message_id),
                    "allow_sending_without_reply": True,
                },
                ensure_ascii=False,
            )
        if parse_mode:
            fields["parse_mode"] = parse_mode
        if disable_notification:
            fields["disable_notification"] = "true"

        body, content_type = _encode_multipart_form_data(
            fields=fields,
            file_field="photo",
            filename=filename,
            file_bytes=photo_bytes,
            mime_type=mime_type,
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        return self._sync_execute_request("sendPhoto", body, headers, timeout=timeout)

    def _sync_send_voice(
        self,
        chat_id: int,
        voice_source: Path | bytes,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        duration: Optional[int | float] = None,
        timeout: float = 60.0,
        *,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(voice_source, (bytes, bytearray)):
            file_bytes = bytes(voice_source)
            effective_filename = Path(filename or "voice.ogg").name or "voice.ogg"
        else:
            file_bytes = voice_source.read_bytes()
            effective_filename = Path(filename).name if filename else (voice_source.name or "voice.ogg")

        fields: Dict[str, str] = {"chat_id": str(int(chat_id))}
        if caption:
            fields["caption"] = caption
        if reply_to_msg_id is not None:
            fields["reply_to_message_id"] = str(int(reply_to_msg_id))
        if duration is not None:
            fields["duration"] = str(max(0, int(round(float(duration)))))

        body, content_type = _encode_multipart_form_data(
            fields=fields,
            file_field="voice",
            filename=effective_filename,
            file_bytes=file_bytes,
            mime_type=mime_type or "audio/ogg",
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }
        return self._sync_execute_request("sendVoice", body, headers, timeout=timeout)

    async def call_api(
        self,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 35.0,
    ) -> Dict[str, Any]:
        """Execute a Bot API method without blocking the asyncio event loop."""
        return await asyncio.to_thread(self._sync_call_api, method, payload, timeout)

    def _sync_download_file(self, file_id: str, dest: Path) -> Path:
        info = self._sync_call_api("getFile", {"file_id": file_id}, timeout=20.0)
        remote_path = (info.get("result") or {}).get("file_path")
        if not remote_path:
            raise RuntimeError("Telegram getFile response did not contain file_path.")
        url = self._file_url(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60.0) as resp, open(dest, "wb") as out_f:
                out_f.write(resp.read())
        except Exception as exc:
            raise RuntimeError(
                f"Failed downloading Telegram file: {self._redact_secrets(exc)}"
            ) from None
        return dest.resolve()

    async def download_file_by_id(self, file_id: str, dest: Path) -> Path:
        """Download a file from Telegram Bot API by file_id into dest."""
        return await asyncio.to_thread(self._sync_download_file, file_id, dest)

    def is_connected(self) -> bool:
        return self._is_connected

    def register_message_handler(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        self.dispatcher.register(handler)

    def _convert_msg_dict(self, msg_dict: Dict[str, Any]) -> Optional[IncomingMessage]:
        ev = BotAPIMessageEvent(msg_dict, bot_id=self.bot_id)
        if ev.sender_is_bot:
            return None
        if ev.sender_username and ev.sender_username.lower().lstrip("@") in self._ignored_sender_usernames_set:
            return None
        if ev.out or ev.sender_id == self.bot_id:
            return None

        if ev.is_private:
            if not self._allowed_dm_user_ids_set:
                return None
            if (
                ev.sender_id not in self._allowed_dm_user_ids_set
                and ev.chat_id not in self._allowed_dm_user_ids_set
            ):
                return None
        else:
            if self._allowed_chat_ids_set:
                if ev.chat_id not in self._allowed_chat_ids_set:
                    return None
            elif self._allowed_dm_user_ids_set:
                return None

        is_mention = self._heuristics.is_mention(
            text=ev.raw_text,
            bot_names=self.bot_names,
            bot_username=self.bot_username,
            entities=None,
            bot_id=self.bot_id,
        )
        return IncomingMessage(
            msg_id=ev.id,
            chat_id=ev.chat_id,
            sender_id=ev.sender_id,
            sender_name=ev.sender_name,
            text=ev.raw_text,
            reply_to_msg_id=ev.reply_to_msg_id,
            reply_to_sender_id=ev.reply_to_sender_id,
            is_reply=ev.is_reply,
            is_mention=is_mention,
            raw_event=ev,
            media=ev.media if ev.media else None,
            reply_to_text=ev.reply_to_text,
            reply_to_author=ev.reply_to_author,
            reply_to_media_event=ev.reply_to_media_event,
            is_private=ev.is_private,
            sender_is_bot=ev.sender_is_bot,
            sender_username=ev.sender_username,
        )

    async def start(self) -> None:
        """Verify Bot API token via getMe, ingest initial backlog into memory, and start polling."""
        me_resp = await self.call_api("getMe", timeout=15.0)
        me = me_resp.get("result") or {}
        self.bot_id = int(me.get("id", 0))
        self.bot_username = me.get("username")
        self.bot_first_name = me.get("first_name")
        if self.bot_first_name and self.bot_first_name not in self.bot_names:
            self.bot_names.append(self.bot_first_name)

        self._is_connected = True
        logger.info(
            "Telegram Bot API connected as @%s ('%s', id: %s)",
            self.bot_username,
            self.bot_first_name,
            self.bot_id,
        )

        # Fetch pending backlog once so we don't spam replies to old messages on startup,
        # while optionally seeding short-term memory with recent chat messages.
        try:
            initial = await self.call_api("getUpdates", {"timeout": 0, "limit": 100}, timeout=15.0)
            updates = initial.get("result") or []
            for upd in updates:
                uid = int(upd.get("update_id", 0))
                if uid >= self._offset:
                    self._offset = uid + 1
                msg_dict = upd.get("message")
                if isinstance(msg_dict, dict) and self.backlog_callback is not None:
                    inc = self._convert_msg_dict(msg_dict)
                    if inc is not None:
                        try:
                            await self.backlog_callback(inc)
                        except Exception as exc:
                            logger.debug("Backlog seed error: %s", self._redact_secrets(exc))
            if updates:
                logger.info("Processed %d historical updates (offset=%d)", len(updates), self._offset)
        except Exception as exc:
            logger.warning(
                "Initial getUpdates check failed (will retry in loop): %s",
                self._redact_secrets(exc),
            )

        if self._polling_task is None or self._polling_task.done():
            self._polling_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        logger.info("Started Telegram Bot API long-polling loop")
        while self._is_connected:
            try:
                payload: Dict[str, Any] = {
                    "timeout": 20,
                    "allowed_updates": ["message"],
                }
                if self._offset > 0:
                    payload["offset"] = self._offset

                resp = await self.call_api("getUpdates", payload, timeout=30.0)
                updates = resp.get("result") or []
                for upd in updates:
                    uid = int(upd.get("update_id", 0))
                    if uid >= self._offset:
                        self._offset = uid + 1
                    msg_dict = upd.get("message")
                    if not isinstance(msg_dict, dict):
                        continue
                    incoming = self._convert_msg_dict(msg_dict)
                    if incoming is not None:
                        await self.dispatcher.dispatch(incoming)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "Telegram Bot API polling error: %s. Retrying in 3s...",
                    self._redact_secrets(exc),
                )
                await asyncio.sleep(3.0)

    async def stop(self) -> None:
        """Stop long-polling loop cleanly."""
        self._is_connected = False
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except (asyncio.CancelledError, Exception):
                pass
            self._polling_task = None
        logger.info("Telegram Bot API client stopped.")

    async def send_typing_action(self, chat_id: int, action: str = "typing") -> None:
        """Immediately trigger chat action (typing, record_voice, upload_photo, etc.) without sleeping."""
        act_map = {
            "record-voice": "record_voice",
            "record-audio": "record_voice",
            "voice": "record_voice",
            "upload-document": "upload_document",
            "document": "upload_document",
            "upload-photo": "upload_photo",
            "photo": "upload_photo",
            "upload-video": "upload_video",
            "video": "upload_video",
            "record-video": "record_video",
            "record-round": "record_video_note",
            "video_note": "record_video_note",
            "choose-sticker": "choose_sticker",
            "sticker": "choose_sticker",
        }
        api_action = act_map.get((action or "typing").lower(), action)
        try:
            await self.call_api(
                "sendChatAction",
                {"chat_id": int(chat_id), "action": api_action},
                timeout=5.0,
            )
        except Exception as exc:
            logger.debug(
                "sendChatAction %s failed (non-critical): %s",
                api_action,
                self._redact_secrets(exc),
            )

    async def send_read_acknowledge(self, chat_id: int, max_id: Optional[int] = None) -> None:
        """Read acknowledge stub for Telegram Bot API parity (Bot API does not support read receipts)."""
        pass

    async def _simulate_chat_action(self, chat_id: int, action: str, text: str) -> None:
        """Send a chat action and wait for a natural typing/upload delay."""
        jitter = random.uniform(0.1, 0.6)
        delay = calculate_typing_delay(
            text=text,
            cpm=self.typing_cpm,
            min_seconds=self.typing_min_seconds,
            max_seconds=self.typing_max_seconds,
            jitter=jitter,
        )
        try:
            await self.call_api(
                "sendChatAction",
                {"chat_id": int(chat_id), "action": action},
                timeout=10.0,
            )
            await asyncio.sleep(delay)
        except Exception as exc:
            logger.debug(
                "sendChatAction %s failed (non-critical): %s",
                action,
                self._redact_secrets(exc),
            )

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
    ) -> Any:
        """Send typing action, wait natural delay, and send message (optionally quoting)."""
        safe_text = self._redact_secrets(text)
        if simulate_typing:
            await self._simulate_chat_action(chat_id, "typing", safe_text)

        payload: Dict[str, Any] = {
            "chat_id": int(chat_id),
            "text": safe_text,
        }
        if reply_to_msg_id is not None:
            payload["reply_parameters"] = {
                "message_id": int(reply_to_msg_id),
                "allow_sending_without_reply": True,
            }

        return await self.call_api("sendMessage", payload, timeout=20.0)

    async def send_document(
        self,
        chat_id: int,
        file_path: str | Path | bytes,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
        filename: Optional[str] = None,
        timeout: float = 120.0,
        *,
        mime_type: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate file or raw bytes and upload a document via Bot API `POST /sendDocument`."""
        payload_source: Path | bytes
        resolved_filename = filename
        resolved_mime = mime_type
        if isinstance(file_path, (bytes, bytearray)):
            raw_bytes = bytes(file_path)
            if not raw_bytes:
                raise ValueError("Document bytes cannot be empty.")
            if len(raw_bytes) > MAX_DOCUMENT_SIZE_BYTES:
                raise ValueError(
                    f"Document size ({len(raw_bytes)} bytes) exceeds maximum allowed "
                    f"({MAX_DOCUMENT_SIZE_BYTES} bytes)."
                )
            payload_source = raw_bytes
            if not resolved_filename:
                inferred_mime, ext = _infer_image_mime_and_ext(raw_bytes, mime_type=mime_type)
                resolved_filename = f"generated{ext}"
                if resolved_mime is None:
                    resolved_mime = inferred_mime
        else:
            payload_source = validate_document_path(file_path)

        effective_reply_id = (
            reply_to_message_id if reply_to_message_id is not None else reply_to_msg_id
        )
        safe_caption = (
            _truncate_telegram_caption(self._redact_secrets(caption))
            if caption is not None
            else None
        )

        if simulate_typing:
            await self._simulate_chat_action(chat_id, "upload_document", safe_caption or "")

        return await asyncio.to_thread(
            self._sync_send_document,
            chat_id,
            payload_source,
            safe_caption,
            effective_reply_id,
            resolved_filename,
            timeout,
            mime_type=resolved_mime,
        )

    async def send_photo(
        self,
        chat_id: int,
        photo: str | Path | bytes,
        caption: str | None = None,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        reply_to_msg_id: int | None = None,
        reply_to_message_id: int | None = None,
        simulate_typing: bool = False,
        parse_mode: str | None = None,
        disable_notification: bool = False,
        timeout: float = 90.0,
    ) -> int:
        """Validate and upload a photo via Bot API `POST /sendPhoto`; returns sent message_id."""
        if isinstance(photo, (bytes, bytearray)):
            photo_bytes = bytes(photo)
            if not photo_bytes:
                raise ValueError("Photo bytes cannot be empty.")
            if len(photo_bytes) > MAX_DOCUMENT_SIZE_BYTES:
                raise ValueError(
                    f"Photo size ({len(photo_bytes)} bytes) exceeds maximum allowed "
                    f"({MAX_DOCUMENT_SIZE_BYTES} bytes)."
                )
            resolved_mime, ext = _infer_image_mime_and_ext(
                photo_bytes,
                mime_type=mime_type,
                filename=filename,
            )
            resolved_filename = Path(filename).name if filename else f"generated{ext}"
        else:
            validated_path = validate_document_path(photo)
            photo_bytes = validated_path.read_bytes()
            resolved_mime, ext = _infer_image_mime_and_ext(
                photo_bytes,
                mime_type=mime_type,
                filename=filename or validated_path.name,
            )
            resolved_filename = (
                Path(filename).name
                if filename
                else (validated_path.name or f"generated{ext}")
            )

        effective_reply_id = (
            reply_to_message_id if reply_to_message_id is not None else reply_to_msg_id
        )
        safe_caption = (
            _truncate_telegram_caption(self._redact_secrets(caption))
            if caption is not None
            else None
        )

        if simulate_typing:
            await self._simulate_chat_action(chat_id, "upload_photo", safe_caption or "")

        resp = await asyncio.to_thread(
            self._sync_send_photo,
            chat_id,
            photo_bytes,
            filename=resolved_filename,
            mime_type=resolved_mime,
            caption=safe_caption,
            reply_to_message_id=effective_reply_id,
            parse_mode=parse_mode,
            disable_notification=disable_notification,
            timeout=timeout,
        )
        result = resp.get("result") if isinstance(resp, dict) else None
        msg_id = result.get("message_id") if isinstance(result, dict) else None
        if isinstance(msg_id, int) and not isinstance(msg_id, bool):
            return msg_id
        return 0

    async def send_voice(
        self,
        chat_id: int,
        voice_path: str | Path | bytes,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        duration: Optional[int | float] = None,
        simulate_typing: bool = True,
        timeout: float = 60.0,
        *,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate file or raw bytes and upload a voice note via Bot API `POST /sendVoice`."""
        payload_source: Path | bytes
        if isinstance(voice_path, (bytes, bytearray)):
            raw_bytes = bytes(voice_path)
            if not raw_bytes:
                raise ValueError("Voice bytes cannot be empty.")
            if len(raw_bytes) > MAX_DOCUMENT_SIZE_BYTES:
                raise ValueError(
                    f"Voice size ({len(raw_bytes)} bytes) exceeds maximum allowed "
                    f"({MAX_DOCUMENT_SIZE_BYTES} bytes)."
                )
            payload_source = raw_bytes
        else:
            payload_source = validate_document_path(voice_path)

        effective_reply_id = (
            reply_to_message_id if reply_to_message_id is not None else reply_to_msg_id
        )
        safe_caption = self._redact_secrets(caption) if caption is not None else None

        if simulate_typing:
            await self._simulate_chat_action(chat_id, "record_voice", safe_caption or "")

        return await asyncio.to_thread(
            self._sync_send_voice,
            chat_id,
            payload_source,
            safe_caption,
            effective_reply_id,
            duration,
            timeout,
            filename=filename,
            mime_type=mime_type,
        )

    async def send_reaction(
        self,
        chat_id: int,
        msg_id: int,
        emoji: str,
    ) -> bool:
        """Set an emoji reaction on a message via Bot API setMessageReaction."""
        normalized_emoji = normalize_reaction_emoji(emoji)
        try:
            await self.call_api(
                "setMessageReaction",
                {
                    "chat_id": int(chat_id),
                    "message_id": int(msg_id),
                    "reaction": [{"type": "emoji", "emoji": normalized_emoji}],
                },
                timeout=10.0,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Failed to set Bot API reaction %s (normalized: %s) on msg %s: %s",
                emoji,
                normalized_emoji,
                msg_id,
                self._redact_secrets(exc),
            )
            return False


def resolve_bot_token_from_env_or_config(config_token: Optional[str] = None) -> Optional[str]:
    """Resolve TELEGRAM_BOT_TOKEN from config, environment, or local .env file."""
    if config_token and config_token.strip():
        return config_token.strip()
    env_tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if env_tok:
        return env_tok
    env_file = Path(".env")
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("TELEGRAM_BOT_TOKEN="):
                    val = stripped.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        return val
        except OSError:
            pass
    return None
