"""Message handlers and Telethon event parser for Telegram Userbot."""

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
    text: Optional[str]
    reply_to_msg_id: Optional[int]
    is_reply: bool
    is_mention: bool
    raw_event: Any
    media: Optional[Any] = None
    reply_to_text: Optional[str] = None
    reply_to_author: Optional[str] = None
    reply_to_media_event: Optional[Any] = None
    reply_to_sender_id: Optional[int] = None
    is_private: bool = False
    sender_is_bot: bool = False
    sender_username: Optional[str] = None


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

    # Check for direct attribute on event
    direct_name = getattr(event, "sender_name", None)
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name.strip()

    return f"User_{sender_id}" if sender_id else "Unknown"


async def enrich_reply_metadata(event: Any, bot_id: int = 0) -> None:
    """Asynchronously enrich event with reply-to message metadata (text, author, sender_id, media)."""
    if not bool(getattr(event, "is_reply", False)):
        return

    get_reply = getattr(event, "get_reply_message", None)
    if not callable(get_reply):
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
    if reply_to_sender_id is None and hasattr(reply_msg, "from_id"):
        from_id = getattr(reply_msg, "from_id", None)
        reply_to_sender_id = getattr(from_id, "user_id", None) or getattr(from_id, "chat_id", None)

    reply_to_author = extract_sender_name(reply_msg, reply_to_sender_id or 0)
    reply_media = getattr(reply_msg, "media", None)
    reply_to_media_event = reply_msg if reply_media else None

    is_reply_to_bot = bool(
        bot_id and reply_to_sender_id is not None and int(reply_to_sender_id) == int(bot_id)
    )

    for target in (event, getattr(event, "message", None)):
        if target is None:
            continue
        try:
            target.reply_to_text = reply_to_text
        except Exception:
            pass
        try:
            target.reply_to_author = reply_to_author
        except Exception:
            pass
        try:
            target.reply_to_sender_id = reply_to_sender_id
        except Exception:
            pass
        try:
            target.reply_to_media_event = reply_to_media_event
        except Exception:
            pass
        try:
            target.is_reply_to_bot = is_reply_to_bot
        except Exception:
            pass


def parse_incoming_message_sync(
    event: Any,
    bot_id: int = 0,
    bot_names: Optional[List[str]] = None,
    bot_username: Optional[str] = None,
) -> IncomingMessage:
    """Parse a Telethon event synchronously into an IncomingMessage."""
    # Support both Telethon event objects and synthetic event mocks
    msg_id = getattr(event, "id", None)
    if msg_id is None and hasattr(event, "message"):
        msg_id = getattr(event.message, "id", 0)
    msg_id = msg_id or 0

    chat_id = getattr(event, "chat_id", 0)
    sender_id = getattr(event, "sender_id", 0)
    sender_name = extract_sender_name(event, sender_id)

    sender = getattr(event, "sender", None)
    sender_username = getattr(sender, "username", None) or getattr(event, "sender_username", None)
    sender_is_bot = bool(getattr(sender, "bot", False) or getattr(event, "sender_is_bot", False))
    is_private = bool(getattr(event, "is_private", False) or (chat_id > 0))

    text = getattr(event, "raw_text", None)
    if text is None:
        text = getattr(event, "text", None)
    if text is None and hasattr(event, "message"):
        text = getattr(event.message, "message", "")

    # Reply detection
    is_reply = bool(getattr(event, "is_reply", False))
    reply_to_msg_id = getattr(event, "reply_to_msg_id", None)
    if reply_to_msg_id is None and hasattr(event, "message"):
        reply_to = getattr(event.message, "reply_to", None)
        if reply_to is not None:
            reply_to_msg_id = getattr(reply_to, "reply_to_msg_id", None)

    # Media detection
    media = getattr(event, "media", None)
    if media is None and hasattr(event, "message"):
        media = getattr(event.message, "media", None)

    # Mention detection
    entities = getattr(event, "entities", None)
    if entities is None and hasattr(event, "message"):
        entities = getattr(event.message, "entities", None)

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
    reply_to_media_event = getattr(event, "reply_to_media_event", None)
    reply_to_sender_id = getattr(event, "reply_to_sender_id", None)

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
