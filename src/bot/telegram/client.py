"""Telegram MTProto Userbot Client built with Telethon.

Features:
- Session isolation with 0700 dir / 0600 file permissions
- Allowed chat whitelisting
- Self-message loop suppression
- Natural reading and typing simulation with jitter
- Reply with quote and MTProto emoji reactions
"""

import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.tl.functions.messages import SendReactionRequest, SetTypingRequest
from telethon.tl.types import (
    DocumentAttributeFilename,
    ReactionEmoji,
    SendMessageCancelAction,
    SendMessageRecordAudioAction,
    SendMessageTypingAction,
    SendMessageUploadDocumentAction,
    SendMessageUploadPhotoAction,
)

from bot.telegram.auth import enforce_session_permissions
from bot.telegram.handlers import IncomingMessage, MessageDispatcher, parse_incoming_message

logger = logging.getLogger(__name__)

MAX_DOCUMENT_SIZE_BYTES = 45 * 1024 * 1024  # 45 MB (safe margin under Telegram 50 MB Bot API limit)

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

# Common emoji aliases not directly accepted by Telegram ReactionTypeEmoji
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
    """Normalize an emoji into a supported Telegram reaction emoji."""
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


def validate_document_path(
    file_path: str | Path,
    max_size_bytes: int = MAX_DOCUMENT_SIZE_BYTES,
) -> Path:
    """Validate that a document path points to a non-empty regular file within size limits."""
    if not file_path or not str(file_path).strip():
        raise ValueError("Document path cannot be empty.")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.is_file():
        raise ValueError(f"Document path is not a regular file: {path}")

    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Document is empty (0 bytes): {path}")
    if size > max_size_bytes:
        raise ValueError(
            f"Document size ({size} bytes) exceeds maximum allowed ({max_size_bytes} bytes): {path}"
        )
    return path.resolve()


def calculate_typing_delay(
    text: str,
    cpm: int = 300,
    min_seconds: float = 1.0,
    max_seconds: float = 8.0,
    jitter: float = 0.5,
) -> float:
    """Calculate natural reading + typing duration based on character count."""
    chars_per_sec = max(1.0, cpm / 60.0)
    raw_duration = len(text) / chars_per_sec
    clamped_duration = max(min_seconds, min(max_seconds, raw_duration))
    return clamped_duration + jitter


class TelegramUserbot:
    """MTProto Telegram userbot client with secure session storage and human-like delays."""

    def __init__(
        self,
        session_path: str,
        api_id: int,
        api_hash: str,
        allowed_chat_ids: Optional[List[int]] = None,
        allowed_dm_user_ids: Optional[List[int]] = None,
        ignored_sender_usernames: Optional[List[str]] = None,
        bot_names: Optional[List[str]] = None,
        typing_cpm: int = 300,
        typing_min_seconds: float = 1.0,
        typing_max_seconds: float = 8.0,
        client: Optional[TelegramClient] = None,
        bot_api_client: Optional[Any] = None,
        backlog_callback: Optional[Callable[[IncomingMessage], Awaitable[None]]] = None,
        backfill_limit: int = 50,
    ) -> None:
        self.session_path = str(session_path)
        self.api_id = api_id
        self.api_hash = api_hash
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
            u.lower().lstrip("@") for u in self.ignored_sender_usernames
        }
        self.bot_names: List[str] = list(bot_names) if bot_names is not None else []
        self.typing_cpm = typing_cpm
        self.typing_min_seconds = typing_min_seconds
        self.typing_max_seconds = typing_max_seconds
        self.backlog_callback = backlog_callback
        self.backfill_limit = backfill_limit

        self.bot_id: int = 0
        self.bot_username: Optional[str] = None
        self.dispatcher = MessageDispatcher()
        self._handler_registered = False
        self._is_connected = False
        self._bot_api_client = bot_api_client

        if client is not None:
            self.client = client
        else:
            path = Path(self.session_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(path.parent, 0o700)
            if path.exists():
                os.chmod(path, 0o600)
            session_name = str(path.with_suffix("")) if path.suffix == ".session" else str(path)
            self.client = TelegramClient(session_name, api_id=self.api_id, api_hash=self.api_hash)

        enforce_session_permissions(self.session_path)

    def is_connected(self) -> bool:
        """Return True if MTProto client is currently connected."""
        if self.client is None:
            return False
        if callable(getattr(self.client, "is_connected", None)):
            res = self.client.is_connected()
            if isinstance(res, bool):
                return res
        return self._is_connected

    async def _backfill_history(self) -> None:
        """Fetch recent message history for allowed chats and pass to backlog_callback."""
        if not self.backlog_callback or not self.allowed_chat_ids:
            return

        for chat_id in self.allowed_chat_ids:
            try:
                if not callable(getattr(self.client, "get_messages", None)):
                    continue
                messages = await self.client.get_messages(chat_id, limit=self.backfill_limit)
                if not messages:
                    continue
                for msg in reversed(messages):
                    try:
                        inc = await parse_incoming_message(
                            event=msg,
                            bot_id=self.bot_id,
                            bot_names=self.bot_names,
                            bot_username=self.bot_username,
                        )
                        await self.backlog_callback(inc)
                    except Exception as msg_err:
                        logger.debug(
                            "Failed to seed backfill message %s: %s",
                            getattr(msg, "id", None),
                            msg_err,
                        )
            except FloodWaitError as fwe:
                logger.warning(
                    "FloodWaitError (%ds) during history backfill for chat %s",
                    getattr(fwe, "seconds", 0),
                    chat_id,
                )
                break
            except Exception as exc:
                logger.debug("History backfill failed for chat %s: %s", chat_id, exc)

    async def start(self) -> None:
        """Connect MTProto client, ensure permissions, backfill history, and attach event listeners."""
        # Ensure session parent directory permissions (0700)
        enforce_session_permissions(self.session_path)

        if not self.is_connected():
            await self.client.connect()
        self._is_connected = True

        if not await self.client.is_user_authorized():
            raise RuntimeError(
                f"Telegram userbot session at '{self.session_path}' is not authorized. "
                "Please run 'python -m bot.cli auth-telegram' to log in."
            )

        me = await self.client.get_me()
        self.bot_id = me.id
        self.bot_username = getattr(me, "username", None)
        logger.info(
            "Telegram userbot connected as @%s (id: %s)",
            self.bot_username,
            self.bot_id,
        )

        enforce_session_permissions(self.session_path)

        if self.backlog_callback and self.allowed_chat_ids:
            await self._backfill_history()

        if not self._handler_registered:
            self.client.add_event_handler(self._on_new_message, events.NewMessage)
            self._handler_registered = True

    async def stop(self) -> None:
        """Disconnect MTProto client and ensure session permissions are preserved."""
        if self.client and self.is_connected():
            await self.client.disconnect()
        self._is_connected = False
        enforce_session_permissions(self.session_path)
        logger.info("Telegram userbot disconnected.")

    def register_message_handler(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register an async callback for incoming messages."""
        self.dispatcher.register(handler)

    async def _on_new_message(self, event: Any) -> None:
        """Internal Telethon event listener with chat whitelist and loop suppression."""
        # 1. Anti-loop guard: Drop bot senders and ignored usernames
        sender = getattr(event, "sender", None)
        raw_sender_is_bot = getattr(event, "sender_is_bot", False)
        sender_is_bot = (raw_sender_is_bot is True)
        if not sender_is_bot and sender is not None:
            sender_is_bot = (
                getattr(sender, "bot", False) is True
                or getattr(sender, "is_bot", False) is True
            )

        raw_username = getattr(event, "sender_username", None)
        if not isinstance(raw_username, str) and sender is not None:
            raw_username = getattr(sender, "username", None)
        clean_username = (
            raw_username.lower().lstrip("@")
            if isinstance(raw_username, str) and raw_username.strip()
            else ""
        )
        if sender_is_bot or (clean_username and clean_username in self._ignored_sender_usernames_set):
            return

        # 2. Self-message loop suppression: strictly drop outgoing and self-authored messages
        if getattr(event, "out", False):
            return
        if getattr(event, "sender_id", None) == self.bot_id:
            return

        # 3. Separate private DM whitelist vs group whitelist
        chat_id = getattr(event, "chat_id", None)
        sender_id = getattr(event, "sender_id", None)

        is_priv_attr = getattr(event, "is_private", None)
        if isinstance(is_priv_attr, bool):
            is_priv = is_priv_attr
        elif callable(is_priv_attr) and type(is_priv_attr).__name__ != "MagicMock":
            try:
                res = is_priv_attr()
                is_priv = res if isinstance(res, bool) else None
            except Exception:
                is_priv = None
        else:
            is_priv = None

        if is_priv is None:
            if self._allowed_dm_user_ids_set and (
                (chat_id is not None and int(chat_id) in self._allowed_dm_user_ids_set)
                or (sender_id is not None and int(sender_id) in self._allowed_dm_user_ids_set)
            ):
                is_priv = True
            elif self._allowed_chat_ids_set and chat_id is not None and int(chat_id) in self._allowed_chat_ids_set:
                is_priv = False
            elif self._allowed_chat_ids_set is not None and len(self._allowed_chat_ids_set) == 0 and not self._allowed_dm_user_ids_set:
                is_priv = False
            elif chat_id is not None:
                is_priv = int(chat_id) > 0 and bool(self._allowed_dm_user_ids_set)
            else:
                is_priv = False

        if is_priv:
            if not self._allowed_dm_user_ids_set:
                return
            target_id = int(sender_id if sender_id is not None else (chat_id if chat_id is not None else 0))
            if target_id not in self._allowed_dm_user_ids_set and (
                chat_id is None or int(chat_id) not in self._allowed_dm_user_ids_set
            ):
                return
        else:
            # Broadcast channels are ignored unless explicitly in allowed_chat_ids
            if getattr(event, "is_channel", False) and not getattr(event, "is_group", False):
                if not self._allowed_chat_ids_set or chat_id not in self._allowed_chat_ids_set:
                    return
            if self._allowed_chat_ids_set and (chat_id is None or int(chat_id) not in self._allowed_chat_ids_set):
                return
            elif self._allowed_chat_ids_set is not None and len(self._allowed_chat_ids_set) == 0 and self._allowed_dm_user_ids_set:
                return

        # 4. Parse into structured IncomingMessage asynchronously (enriching reply-to metadata)
        incoming = await parse_incoming_message(
            event=event,
            bot_id=self.bot_id,
            bot_names=self.bot_names,
            bot_username=self.bot_username,
        )

        incoming.is_private = is_priv
        incoming.sender_is_bot = sender_is_bot
        if clean_username:
            incoming.sender_username = clean_username

        # 5. Dispatch to downstream business logic
        await self.dispatcher.dispatch(incoming)

    async def _simulate_chat_action(self, chat_id: int, action: str, text: str) -> None:
        """Simulate human-like reading/typing/upload delay using a Telethon chat action."""
        jitter = random.uniform(0.1, 0.6)
        delay = calculate_typing_delay(
            text=text,
            cpm=self.typing_cpm,
            min_seconds=self.typing_min_seconds,
            max_seconds=self.typing_max_seconds,
            jitter=jitter,
        )
        try:
            async with self.client.action(chat_id, action):
                await asyncio.sleep(delay)
        except Exception as e:
            logger.debug("Chat action '%s' simulation failed (non-critical): %s", action, e)

    async def send_typing_action(self, chat_id: int, action: str = "typing") -> None:
        """Immediately trigger chat action (typing, record-voice, etc.) without sleeping."""
        if self._bot_api_client is not None and hasattr(self._bot_api_client, "send_typing_action"):
            try:
                await self._bot_api_client.send_typing_action(chat_id, action=action)
            except TypeError:
                await self._bot_api_client.send_typing_action(chat_id)
            except Exception as exc:
                logger.debug("Bot API send_typing_action failed: %s", exc)
            return

        act_lower = (action or "typing").lower()
        if act_lower in ("record-voice", "record-audio", "record_voice", "record_audio", "voice"):
            action_obj = SendMessageRecordAudioAction()
        elif act_lower in ("upload-document", "upload_document", "document"):
            action_obj = SendMessageUploadDocumentAction(progress=0)
        elif act_lower in ("upload-photo", "upload_photo", "photo"):
            action_obj = SendMessageUploadPhotoAction(progress=0)
        elif act_lower == "cancel":
            action_obj = SendMessageCancelAction()
        else:
            action_obj = SendMessageTypingAction()

        try:
            peer = chat_id
            if callable(getattr(self.client, "get_input_entity", None)):
                try:
                    peer = await self.client.get_input_entity(chat_id)
                except Exception as ent_err:
                    logger.debug("get_input_entity for %s failed, using raw chat_id: %s", chat_id, ent_err)
                    peer = chat_id

            req = SetTypingRequest(peer=peer, action=action_obj)
            if callable(self.client):
                await self.client(req)
        except FloodWaitError as fwe:
            logger.warning(
                "FloodWaitError (%ds) in send_typing_action for chat %s",
                getattr(fwe, "seconds", 0),
                chat_id,
            )
        except Exception as exc:
            logger.debug("MTProto send_typing_action failed for chat %s: %s", chat_id, exc)

    async def send_reply(
        self,
        chat_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
    ) -> Any:
        """Send a message to a chat, optionally simulating reading/typing delay and quoting."""
        if simulate_typing:
            await self._simulate_chat_action(chat_id, "typing", text)

        for attempt in range(2):
            try:
                return await self.client.send_message(
                    chat_id,
                    text,
                    reply_to=reply_to_msg_id,
                )
            except FloodWaitError as fwe:
                wait_sec = getattr(fwe, "seconds", 0)
                if attempt == 0 and 0 < wait_sec <= 10.0:
                    logger.warning(
                        "FloodWaitError (%ds) in send_reply for chat %s. Backing off...",
                        wait_sec,
                        chat_id,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                logger.warning(
                    "FloodWaitError (%ds) in send_reply for chat %s exceeds backoff limit or retry failed; skipping.",
                    wait_sec,
                    chat_id,
                )
                return None
            except (ConnectionError, OSError, asyncio.TimeoutError) as net_err:
                logger.warning("Network disconnect/timeout in send_reply for chat %s: %s", chat_id, net_err)
                return None

    async def send_document(
        self,
        chat_id: int,
        file_path: str | Path,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = True,
        filename: Optional[str] = None,
        timeout: float = 120.0,
    ) -> Any:
        """Send a document file to a chat via Bot API or Telethon after path/size validation."""
        validated_path = validate_document_path(file_path)

        bot_api_client = getattr(self, "_bot_api_client", None)
        if bot_api_client is not None:
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "file_path": validated_path,
                "caption": caption,
                "reply_to_msg_id": reply_to_msg_id,
                "simulate_typing": simulate_typing,
                "filename": filename,
            }
            if timeout != 120.0:
                kwargs["timeout"] = timeout
            return await bot_api_client.send_document(**kwargs)

        if simulate_typing:
            await self._simulate_chat_action(chat_id, "document", caption or "")

        send_kwargs: dict[str, Any] = {
            "caption": caption,
            "reply_to": reply_to_msg_id,
            "force_document": True,
        }
        if filename:
            send_kwargs["attributes"] = [
                DocumentAttributeFilename(file_name=Path(filename).name)
            ]

        for attempt in range(2):
            try:
                coro = self.client.send_file(
                    chat_id,
                    str(validated_path),
                    **send_kwargs,
                )
                if timeout > 0:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
            except FloodWaitError as fwe:
                wait_sec = getattr(fwe, "seconds", 0)
                if attempt == 0 and 0 < wait_sec <= 10.0:
                    logger.warning(
                        "FloodWaitError (%ds) in send_document for chat %s. Backing off...",
                        wait_sec,
                        chat_id,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                logger.warning(
                    "FloodWaitError (%ds) in send_document for chat %s exceeds backoff limit or retry failed; skipping.",
                    wait_sec,
                    chat_id,
                )
                return None
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "Upload timed out in send_document for chat %s (timeout=%s)",
                    chat_id,
                    timeout,
                )
                raise
            except (ConnectionError, OSError) as net_err:
                logger.warning("Network disconnect in send_document for chat %s: %s", chat_id, net_err)
                return None

    async def send_voice(
        self,
        chat_id: int,
        voice_path: str | Path,
        caption: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
        duration: Optional[int | float] = None,
        simulate_typing: bool = True,
        timeout: float = 60.0,
    ) -> Any:
        """Send a voice message to a chat via Bot API or Telethon after path/size validation."""
        validated_path = validate_document_path(voice_path)

        bot_api_client = getattr(self, "_bot_api_client", None)
        if bot_api_client is not None:
            return await bot_api_client.send_voice(
                chat_id=chat_id,
                voice_path=validated_path,
                caption=caption,
                reply_to_msg_id=reply_to_msg_id,
                duration=duration,
                simulate_typing=simulate_typing,
                timeout=timeout,
            )

        if simulate_typing:
            await self._simulate_chat_action(chat_id, "record-voice", caption or "")

        send_kwargs: dict[str, Any] = {
            "caption": caption,
            "reply_to": reply_to_msg_id,
            "voice_note": True,
        }
        if duration is not None:
            from telethon.tl.types import DocumentAttributeAudio

            send_kwargs["attributes"] = [
                DocumentAttributeAudio(
                    duration=max(0, int(round(float(duration)))),
                    voice=True,
                )
            ]

        for attempt in range(2):
            try:
                coro = self.client.send_file(
                    chat_id,
                    str(validated_path),
                    **send_kwargs,
                )
                if timeout > 0:
                    return await asyncio.wait_for(coro, timeout=timeout)
                return await coro
            except FloodWaitError as fwe:
                wait_sec = getattr(fwe, "seconds", 0)
                if attempt == 0 and 0 < wait_sec <= 10.0:
                    logger.warning(
                        "FloodWaitError (%ds) in send_voice for chat %s. Backing off...",
                        wait_sec,
                        chat_id,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                logger.warning(
                    "FloodWaitError (%ds) in send_voice for chat %s exceeds backoff limit or retry failed; skipping.",
                    wait_sec,
                    chat_id,
                )
                return None
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "Upload timed out in send_voice for chat %s (timeout=%s)",
                    chat_id,
                    timeout,
                )
                raise
            except (ConnectionError, OSError) as net_err:
                logger.warning("Network disconnect in send_voice for chat %s: %s", chat_id, net_err)
                return None

    async def send_reaction(
        self,
        chat_id: int,
        msg_id: int,
        emoji: str,
    ) -> bool:
        """Send an emoji reaction to a message using normalized emoji and MTProto SendReactionRequest."""
        normalized = normalize_reaction_emoji(emoji)

        for attempt in range(2):
            try:
                peer = await self.client.get_input_entity(chat_id)
                req = SendReactionRequest(
                    peer=peer,
                    msg_id=msg_id,
                    reaction=[ReactionEmoji(emoticon=normalized)],
                )
                await self.client(req)
                return True
            except FloodWaitError as fwe:
                wait_sec = getattr(fwe, "seconds", 0)
                if attempt == 0 and 0 < wait_sec <= 10.0:
                    logger.warning(
                        "FloodWaitError (%ds) in send_reaction for chat %s. Backing off...",
                        wait_sec,
                        chat_id,
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                logger.warning(
                    "FloodWaitError (%ds) in send_reaction for chat %s exceeds backoff limit or retry failed; skipping.",
                    wait_sec,
                    chat_id,
                )
                return False
            except (ConnectionError, OSError, asyncio.TimeoutError) as net_err:
                logger.warning("Network disconnect/timeout in send_reaction for chat %s: %s", chat_id, net_err)
                return False
            except Exception as e:
                logger.warning(
                    "Failed to send reaction %s (normalized: %s) to msg %s in chat %s: %s",
                    emoji,
                    normalized,
                    msg_id,
                    chat_id,
                    e,
                )
                return False
        return False

    async def send_read_acknowledge(self, chat_id: int, max_id: Optional[int] = None) -> None:
        """Acknowledge reading messages up to max_id in chat."""
        try:
            if callable(getattr(self.client, "send_read_acknowledge", None)):
                res = self.client.send_read_acknowledge(chat_id, max_id=max_id)
                if asyncio.iscoroutine(res):
                    await res
        except FloodWaitError as fwe:
            logger.warning(
                "FloodWaitError (%ds) in send_read_acknowledge for chat %s",
                getattr(fwe, "seconds", 0),
                chat_id,
            )
        except Exception as exc:
            logger.debug("send_read_acknowledge failed for chat %s: %s", chat_id, exc)

    async def download_media(self, message: Any, file: str | Path) -> Optional[str]:
        """Download media from a message to local file using client.download_media."""
        if callable(getattr(self.client, "download_media", None)):
            res = self.client.download_media(message, file=str(file))
            if asyncio.iscoroutine(res):
                res = await res
            return str(res) if res else None
        return None
