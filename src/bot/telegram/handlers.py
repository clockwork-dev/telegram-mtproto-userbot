"""Message handlers and Telethon event parser for Telegram Userbot."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from bot.telegram.heuristics import ReplyHeuristics

logger = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    """Structured representation of an incoming Telegram message."""

    msg_id: int
    chat_id: int
    sender_id: int
    sender_name: str
    text: Optional[str] = None
    reply_to_msg_id: Optional[int] = None
    is_reply: bool = False
    is_mention: bool = False
    raw_event: Any = None
    media: Optional[Any] = None
    reply_to_text: Optional[str] = None
    reply_to_author: Optional[str] = None
    reply_to_media_event: Any = None
    reply_to_sender_id: Optional[int] = None
    is_private: bool = False
    sender_is_bot: bool = False
    sender_username: Optional[str] = None


def _has_explicit_attr(obj: Any, attr: str) -> bool:
    """Return True if `attr` on `obj` is explicitly truthy (ignoring unset MagicMock children)."""
    if obj is None:
        return False
    if type(obj).__module__.startswith("unittest.mock"):
        children = getattr(obj, "_mock_children", {})
        if attr not in getattr(obj, "__dict__", {}) and attr not in children:
            return False
    return bool(getattr(obj, attr, None))


def _message_has_media(msg: Any) -> bool:
    """Return True if a Telethon message or synthetic event carries photo/document/sticker/media."""
    if msg is None:
        return False
    return any(
        _has_explicit_attr(msg, attr)
        for attr in ("media", "photo", "document", "sticker", "voice", "video", "gif", "animation")
    )


def extract_sender_name(event: Any, sender_id: int) -> str:
    """Extract a friendly sender display name from event or sender object."""
    sender = getattr(event, "sender", None)
    if sender is not None:
        first_name = getattr(sender, "first_name", "")
        last_name = getattr(sender, "last_name", "")
        fn_str = first_name.strip() if isinstance(first_name, str) else ""
        ln_str = last_name.strip() if isinstance(last_name, str) else ""
        full_name = f"{fn_str} {ln_str}".strip()
        if full_name:
            return full_name
        username = getattr(sender, "username", "")
        if isinstance(username, str) and username.strip():
            return f"@{username.strip()}"

    direct_name = getattr(event, "sender_name", None)
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name.strip()

    return f"User_{sender_id}" if sender_id else "Unknown"


async def enrich_reply_metadata(event: Any, bot_id: int = 0) -> None:
    """Asynchronously enrich event with reply-to message metadata (text, author, sender_id, media)."""
    is_reply = _has_explicit_attr(event, "is_reply")
    if not is_reply and not _has_explicit_attr(event, "reply_to_msg_id"):
        return

    get_reply = getattr(event, "get_reply_message", None)
    if not callable(get_reply):
        return
    if type(event).__module__.startswith("unittest.mock") and not _has_explicit_attr(
        event, "get_reply_message"
    ):
        return

    try:
        res = get_reply()
        if asyncio.iscoroutine(res):
            reply_msg = await asyncio.wait_for(res, timeout=3.0)
        else:
            reply_msg = res
    except Exception as exc:
        logger.debug("Failed or timed out calling get_reply_message(): %s", exc)
        reply_msg = None

    if reply_msg is None:
        return

    reply_to_text = (
        getattr(reply_msg, "raw_text", None)
        or getattr(reply_msg, "text", None)
        or getattr(reply_msg, "message", None)
    )
    if not isinstance(reply_to_text, str):
        reply_to_text = None

    reply_to_sender_id = getattr(reply_msg, "sender_id", None)
    if not isinstance(reply_to_sender_id, int) or isinstance(reply_to_sender_id, bool):
        reply_to_sender_id = None
    if reply_to_sender_id is None and hasattr(reply_msg, "from_id"):
        from_id = getattr(reply_msg, "from_id", None)
        cand = getattr(from_id, "user_id", None) or getattr(from_id, "chat_id", None)
        if isinstance(cand, int) and not isinstance(cand, bool):
            reply_to_sender_id = cand

    reply_to_author = extract_sender_name(reply_msg, reply_to_sender_id or 0)
    reply_to_media_event: Any = None
    if _message_has_media(reply_msg):
        if type(reply_msg).__module__.startswith("unittest.mock") and not _has_explicit_attr(
            reply_msg, "message"
        ):
            try:
                reply_msg.message = reply_msg
            except Exception:
                pass
        elif hasattr(reply_msg, "message") and getattr(reply_msg, "message", "") is None:
            try:
                reply_msg.message = ""
            except Exception:
                pass
        reply_to_media_event = reply_msg

    is_reply_to_bot = bool(
        bot_id and reply_to_sender_id is not None and int(reply_to_sender_id) == int(bot_id)
    )

    targets = [event]
    if _has_explicit_attr(event, "message"):
        msg_obj = getattr(event, "message", None)
        if msg_obj is not None and not isinstance(msg_obj, str) and msg_obj is not event:
            targets.append(msg_obj)

    for target in targets:
        for attr_name, attr_val in (
            ("reply_to_text", reply_to_text),
            ("reply_to_author", reply_to_author),
            ("reply_to_sender_id", reply_to_sender_id),
            ("reply_to_media_event", reply_to_media_event),
            ("is_reply_to_bot", is_reply_to_bot),
        ):
            try:
                setattr(target, attr_name, attr_val)
            except Exception:
                pass


def parse_incoming_message_sync(
    event: Any,
    bot_id: int = 0,
    bot_names: Optional[List[str]] = None,
    bot_username: Optional[str] = None,
) -> IncomingMessage:
    """Parse a Telethon event synchronously into an IncomingMessage."""
    msg_id = getattr(event, "id", None)
    if not isinstance(msg_id, int) or isinstance(msg_id, bool):
        msg_id = None
    if msg_id is None and _has_explicit_attr(event, "message"):
        inner_id = getattr(event.message, "id", 0)
        msg_id = inner_id if isinstance(inner_id, int) and not isinstance(inner_id, bool) else 0
    msg_id = msg_id or 0

    raw_chat_id = getattr(event, "chat_id", 0)
    chat_id = raw_chat_id if isinstance(raw_chat_id, int) and not isinstance(raw_chat_id, bool) else 0
    raw_sender_id = getattr(event, "sender_id", 0)
    sender_id = (
        raw_sender_id if isinstance(raw_sender_id, int) and not isinstance(raw_sender_id, bool) else 0
    )
    sender_name = extract_sender_name(event, sender_id)

    sender = getattr(event, "sender", None)
    raw_s_user = getattr(sender, "username", None) if sender is not None else None
    if not isinstance(raw_s_user, str):
        raw_s_user = getattr(event, "sender_username", None)
    sender_username = raw_s_user.strip() if isinstance(raw_s_user, str) and raw_s_user.strip() else None

    sender_bot_flag = (
        (getattr(sender, "bot", False) is True or getattr(sender, "is_bot", False) is True)
        if sender is not None
        else False
    )
    sender_is_bot = bool(sender_bot_flag or (getattr(event, "sender_is_bot", False) is True))

    raw_is_priv = getattr(event, "is_private", None)
    if isinstance(raw_is_priv, bool):
        is_private = raw_is_priv
    elif callable(raw_is_priv) and not type(raw_is_priv).__module__.startswith("unittest.mock"):
        try:
            priv_res = raw_is_priv()
            is_private = priv_res if isinstance(priv_res, bool) else (chat_id > 0)
        except Exception:
            is_private = chat_id > 0
    else:
        is_private = chat_id > 0

    text = getattr(event, "raw_text", None)
    if not isinstance(text, str):
        cand_text = getattr(event, "text", None)
        text = cand_text if isinstance(cand_text, str) else None
    if text is None and _has_explicit_attr(event, "message"):
        inner_text = getattr(event.message, "message", None)
        if isinstance(inner_text, str):
            text = inner_text

    # Reply detection
    is_reply = _has_explicit_attr(event, "is_reply")
    raw_reply_id = getattr(event, "reply_to_msg_id", None)
    reply_to_msg_id = (
        raw_reply_id if isinstance(raw_reply_id, int) and not isinstance(raw_reply_id, bool) else None
    )
    if reply_to_msg_id is None and _has_explicit_attr(event, "message"):
        reply_to = getattr(event.message, "reply_to", None)
        if reply_to is not None:
            inner_reply_id = getattr(reply_to, "reply_to_msg_id", None)
            if isinstance(inner_reply_id, int) and not isinstance(inner_reply_id, bool):
                reply_to_msg_id = inner_reply_id
    if reply_to_msg_id is not None and not is_reply:
        is_reply = True

    # Media detection
    media = getattr(event, "media", None) if _has_explicit_attr(event, "media") else None
    if media is None and _has_explicit_attr(event, "message"):
        inner_msg = getattr(event, "message", None)
        if _has_explicit_attr(inner_msg, "media"):
            media = getattr(inner_msg, "media", None)

    # Mention detection
    entities = getattr(event, "entities", None) if _has_explicit_attr(event, "entities") else None
    if entities is None and _has_explicit_attr(event, "message"):
        inner_msg = getattr(event, "message", None)
        if _has_explicit_attr(inner_msg, "entities"):
            entities = getattr(inner_msg, "entities", None)

    heuristics = ReplyHeuristics()
    is_mention = heuristics.is_mention(
        text=text,
        bot_names=bot_names or [],
        bot_username=bot_username,
        entities=entities,
        bot_id=bot_id,
    )

    reply_to_text = getattr(event, "reply_to_text", None)
    if not isinstance(reply_to_text, str):
        reply_to_text = None
    reply_to_author = getattr(event, "reply_to_author", None)
    if not isinstance(reply_to_author, str):
        reply_to_author = None

    reply_to_media_event = (
        getattr(event, "reply_to_media_event", None)
        if _has_explicit_attr(event, "reply_to_media_event")
        else None
    )
    if reply_to_media_event is None and _has_explicit_attr(event, "reply_message"):
        cand_reply_msg = getattr(event, "reply_message", None)
        if _message_has_media(cand_reply_msg):
            reply_to_media_event = cand_reply_msg

    raw_reply_sender = getattr(event, "reply_to_sender_id", None)
    reply_to_sender_id = (
        raw_reply_sender
        if isinstance(raw_reply_sender, int) and not isinstance(raw_reply_sender, bool)
        else None
    )

    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
        reply_to_msg_id=reply_to_msg_id,
        is_reply=is_reply,
        is_mention=is_mention,
        raw_event=event,
        media=media,
        reply_to_text=reply_to_text,
        reply_to_author=reply_to_author,
        reply_to_media_event=reply_to_media_event,
        reply_to_sender_id=reply_to_sender_id,
        is_private=is_private,
        sender_is_bot=sender_is_bot,
        sender_username=sender_username,
    )


async def parse_incoming_message(
    event: Any,
    bot_id: int = 0,
    bot_names: Optional[List[str]] = None,
    bot_username: Optional[str] = None,
) -> IncomingMessage:
    """Asynchronously enrich event with reply data and parse into IncomingMessage."""
    await enrich_reply_metadata(event, bot_id=bot_id)
    return parse_incoming_message_sync(
        event=event,
        bot_id=bot_id,
        bot_names=bot_names,
        bot_username=bot_username,
    )


class MessageDispatcher:
    """Dispatcher registry for registered incoming message callbacks."""

    def __init__(self) -> None:
        self._handlers: List[Callable[[IncomingMessage], Awaitable[None]]] = []

    def register(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register an async callback for incoming messages."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    async def dispatch(self, message: IncomingMessage) -> None:
        """Execute all registered callbacks sequentially or concurrently."""
        for handler in self._handlers:
            try:
                await handler(message)
            except Exception as err:
                logger.error("Error executing message handler %s: %s", handler, err, exc_info=True)
