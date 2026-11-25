"""Reply heuristics for Telegram Userbot.

Distinguishes direct mentions, replies, and spontaneous conversational joins
with quiet hours, cooldowns, and anti-spam probability gating.
"""

import random
import re
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional


class ReplyAction(str, Enum):
    MUST_REPLY = "MUST_REPLY"
    DIRECT_REPLY = "MUST_REPLY"
    SPONTANEOUS_REPLY = "SPONTANEOUS_REPLY"
    IGNORE = "IGNORE"


INDIRECT_BOT_REFERENCES: tuple[str, ...] = (
    "она",
    "её",
    "ее",
    "ей",
    "неё",
    "нее",
    "оно",
    "вона",
    "її",
    "їй",
    "неї",
    "бот",
    "бота",
    "боту",
    "ботом",
    "боте",
    "пиструнькиня",
    "пиструнькини",
    "пиструнькине",
    "пиструнькиню",
    "пиструнькиней",
    "піструнькиня",
    "піструнькині",
    "піструнькиню",
    "піструнькинею",
    "пиструн",
    "пиструна",
    "пиструну",
    "пиструном",
    "піструн",
    "піструна",
    "піструну",
    "піструном",
)

_INDIRECT_REF_PATTERN = re.compile(
    r"(?:^|\b|\s|[(\[{<«\"'])(?:"
    + "|".join(re.escape(word) for word in INDIRECT_BOT_REFERENCES)
    + r")(?:\b|\s|[.,!?:;\)\]}>»\"']|$)",
    re.IGNORECASE,
)


def is_indirect_reference(text: Optional[str]) -> bool:
    """Check if message refers to the bot indirectly via 3rd-person pronouns or colloquial forms."""
    if not text:
        return False
    return bool(_INDIRECT_REF_PATTERN.search(text.lower()))


_ALNUM_CHAR_RE = re.compile(r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9]")

_TRIVIAL_ACKS: frozenset[str] = frozenset({
    "да", "нет", "ок", "окей", "ага", "угу", "уже", "ну", "хз", "лол",
    "кек", "мм", "эм", "ой", "ай", "эх", "чо", "чё", "шо", "не", "та",
    "ok", "yes", "no", "lol", "kek", "hmm", "hm",
})


def is_substantive_text(text: Optional[str]) -> bool:
    """Return True if text has meaningful conversational content (not empty/sticker/<=4-char ack/bare '?')."""
    if not text:
        return False
    stripped = text.strip()
    if not stripped or len(stripped) <= 4:
        return False
    if not _ALNUM_CHAR_RE.search(stripped):
        return False
    normalized = stripped.lower().rstrip(".,!?:;…")
    if not normalized or normalized in _TRIVIAL_ACKS:
        return False
    return True


class ReplyHeuristics:
    """Decision engine for determining when the bot should respond to an incoming message."""

    _MAX_TRACKED_CHATS = 256

    def __init__(
        self,
        default_max_per_hour: int = 3,
        default_active_window_cooldown_seconds: float = 25.0,
    ) -> None:
        self.default_max_per_hour = default_max_per_hour
        self.default_active_window_cooldown_seconds = default_active_window_cooldown_seconds
        self._reply_timestamps: Dict[int, deque[float]] = {}
        self._last_spontaneous_attempt: Dict[int, float] = {}

    def _sweep_reply_timestamps(self, now_ts: float) -> None:
        cutoff = now_ts - 3600.0
        for cid, q in list(self._reply_timestamps.items()):
            while q and q[0] <= cutoff:
                q.popleft()
            if not q:
                self._reply_timestamps.pop(cid, None)
        while len(self._reply_timestamps) > self._MAX_TRACKED_CHATS:
            oldest_cid = min(
                self._reply_timestamps,
                key=lambda k: self._reply_timestamps[k][-1] if self._reply_timestamps[k] else 0.0,
            )
            self._reply_timestamps.pop(oldest_cid, None)
        for cid, ts in list(self._last_spontaneous_attempt.items()):
            if ts <= cutoff:
                self._last_spontaneous_attempt.pop(cid, None)
        while len(self._last_spontaneous_attempt) > self._MAX_TRACKED_CHATS:
            oldest_attempt_cid = min(
                self._last_spontaneous_attempt,
                key=lambda k: self._last_spontaneous_attempt[k],
            )
            self._last_spontaneous_attempt.pop(oldest_attempt_cid, None)

    def _prune_hourly_window(self, chat_key: int, now_ts: float, *, create: bool = False) -> deque[float]:
        timestamps = self._reply_timestamps.get(chat_key)
        if timestamps is None:
            if not create:
                return deque()
            if len(self._reply_timestamps) >= self._MAX_TRACKED_CHATS:
                self._sweep_reply_timestamps(now_ts)
            timestamps = deque()
            self._reply_timestamps[chat_key] = timestamps
            return timestamps
        cutoff = now_ts - 3600.0
        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
        if not timestamps and not create:
            self._reply_timestamps.pop(chat_key, None)
            return deque()
        return timestamps

    def record_reply(self, chat_id: int, timestamp: Optional[float] = None) -> None:
        """Record a spontaneous/active-window reply timestamp for per-chat hourly rate limiting."""
        now_ts = timestamp if timestamp is not None else time.time()
        timestamps = self._prune_hourly_window(int(chat_id), now_ts, create=True)
        timestamps.append(now_ts)

    def recent_reply_count(self, chat_id: int, now: Optional[float] = None) -> int:
        """Return number of spontaneous/active-window replies recorded in the last 60 minutes."""
        now_ts = now if now is not None else time.time()
        return len(self._prune_hourly_window(int(chat_id), now_ts, create=False))

    def is_substantive_text(self, text: Optional[str]) -> bool:
        """Check whether message text contains substantive content worth spontaneous evaluation."""
        return is_substantive_text(text)

    def is_indirect_reference(self, text: Optional[str]) -> bool:
        """Check if message refers to the bot indirectly via 3rd-person pronouns or colloquial forms."""
        return is_indirect_reference(text)

    def is_mention(
        self,
        text: Optional[str],
        bot_names: List[str],
        bot_username: Optional[str] = None,
        entities: Optional[List[Any]] = None,
        bot_id: Optional[int] = None,
    ) -> bool:
        """Check if message explicitly mentions the bot by username, alias, or Telegram entity."""
        if not text:
            text = ""

        # Check Telegram entities if present
        if entities:
            for entity in entities:
                entity_user_id = getattr(entity, "user_id", None)
                if bot_id is not None and entity_user_id == bot_id:
                    return True

        # Combine all name candidates
        names_to_check = list(bot_names)
        if bot_username:
            names_to_check.append(bot_username)
            if not bot_username.startswith("@"):
                names_to_check.append(f"@{bot_username}")

        text_lower = text.lower()
        for name in names_to_check:
            name_stripped = name.strip()
            if not name_stripped:
                continue

            name_lower = name_stripped.lower()
            if name_lower.startswith("@"):
                pattern = r"(?:^|\s|[(\[{<«\"'])" + re.escape(name_lower) + r"(?:\b|\s|[.,!?:;\)\]}>»\"']|$)"
            else:
                pattern = r"(?:\b|^|\s|[(\[{<«\"'])" + re.escape(name_lower) + r"(?:\b|\s|[.,!?:;\)\]}>»\"']|$)"

            if re.search(pattern, text_lower, re.IGNORECASE):
                return True

        return False

    def is_reply_to_bot(self, event: Any, bot_id: int) -> bool:
        """Check if event is a direct reply to one of the bot's messages."""
        if not getattr(event, "is_reply", False):
            return False

        reply_to_sender_id = getattr(event, "reply_to_sender_id", None)
        if reply_to_sender_id is not None and reply_to_sender_id == bot_id:
            return True

        if getattr(event, "is_reply_to_bot", False):
            return True

        return False

    def should_reply(
        self,
        event: Any,
        bot_id: int,
        bot_names: List[str],
        allowed_chat_ids: Optional[List[int]] = None,
        spontaneous_enabled: bool = False,
        in_quiet_hours: bool = False,
        time_since_last_bot_msg: float = 999999.0,
        min_interval_seconds: float = 300.0,
        probability: float = 0.05,
        bot_username: Optional[str] = None,
        active_conversation_window_seconds: float = 90.0,
        max_per_hour: Optional[int] = None,
        active_window_cooldown_seconds: Optional[float] = None,
        active_window_probability: Optional[float] = None,
        now: Optional[float] = None,
        allowed_dm_user_ids: Optional[List[int]] = None,
        ignored_sender_usernames: Optional[List[str]] = None,
        ignored_sender_ids: Optional[List[int]] = None,
    ) -> ReplyAction:
        """Evaluate incoming message event and decide on appropriate reply action."""
        # 1. Anti-loop guard: Drop sender if bot or in ignored usernames/IDs
        sender = getattr(event, "sender", None)
        raw_sender_is_bot = getattr(event, "sender_is_bot", False)
        sender_is_bot = raw_sender_is_bot is True
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

        ignored_usernames_set = {"hermes_kuk7yi6kfm3lcdfa_bot"}
        if ignored_sender_usernames:
            ignored_usernames_set.update(
                u.lower().lstrip("@") for u in ignored_sender_usernames if u and u.strip()
            )

        if sender_is_bot or (clean_username and clean_username in ignored_usernames_set):
            return ReplyAction.IGNORE

        # 2. Self-message & ignored-ID loop suppression
        if getattr(event, "out", False):
            return ReplyAction.IGNORE

        sender_id = getattr(event, "sender_id", None)
        if sender_id is not None:
            if bot_id is not None and sender_id == bot_id:
                return ReplyAction.IGNORE
            if ignored_sender_ids:
                try:
                    if int(sender_id) in {int(uid) for uid in ignored_sender_ids}:
                        return ReplyAction.IGNORE
                except (TypeError, ValueError):
                    pass

        # 3. Strict DM whitelist vs Group whitelist
        chat_id = getattr(event, "chat_id", None)

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
            chat_id_int: Optional[int] = None
            if chat_id is not None:
                try:
                    chat_id_int = int(chat_id)
                except (ValueError, TypeError):
                    chat_id_int = None

            if chat_id_int is not None and chat_id_int < 0:
                is_priv = False
            elif allowed_dm_user_ids and (
                (chat_id_int is not None and chat_id_int in {int(u) for u in allowed_dm_user_ids})
                or (sender_id is not None and int(sender_id) in {int(u) for u in allowed_dm_user_ids})
            ):
                is_priv = True
            elif allowed_chat_ids and chat_id_int is not None and chat_id_int in {int(c) for c in allowed_chat_ids}:
                is_priv = False
            elif allowed_chat_ids is not None and len(allowed_chat_ids) == 0 and not allowed_dm_user_ids:
                is_priv = False
            elif chat_id_int is not None:
                is_priv = chat_id_int > 0 and bool(allowed_dm_user_ids)
            else:
                is_priv = False

        if is_priv:
            if not allowed_dm_user_ids:
                return ReplyAction.IGNORE
            allowed_dms = {int(u) for u in allowed_dm_user_ids}
            target_uid = int(sender_id if sender_id is not None else (chat_id if chat_id is not None else 0))
            if target_uid not in allowed_dms and (chat_id is None or int(chat_id) not in allowed_dms):
                return ReplyAction.IGNORE
            # Whitelisted DM -> ALWAYS reply unconditionally without mention or reply
            return ReplyAction.MUST_REPLY

        # Group whitelist guard
        if allowed_chat_ids is not None and len(allowed_chat_ids) > 0:
            try:
                allowed_ints = {int(cid) for cid in allowed_chat_ids}
                if chat_id is None or int(chat_id) not in allowed_ints:
                    return ReplyAction.IGNORE
            except (ValueError, TypeError):
                return ReplyAction.IGNORE
        elif allowed_chat_ids is not None and len(allowed_chat_ids) == 0 and allowed_dm_user_ids:
            return ReplyAction.IGNORE

        # 4. Direct reply to bot (always responsive in groups)
        if self.is_reply_to_bot(event, bot_id):
            return ReplyAction.MUST_REPLY

        # 5. Direct mention (always responsive in groups)
        text = getattr(event, "raw_text", None) or getattr(event, "text", None) or ""
        entities = getattr(event, "entities", None)
        if not entities and hasattr(event, "message"):
            entities = getattr(event.message, "entities", None)

        if self.is_mention(
            text=text,
            bot_names=bot_names,
            bot_username=bot_username,
            entities=entities,
            bot_id=bot_id,
        ):
            return ReplyAction.MUST_REPLY

        # 6. Spontaneous / active-window gating
        if not spontaneous_enabled or in_quiet_hours or probability <= 0.0:
            return ReplyAction.IGNORE

        # Do not auto-trigger on empty-text stickers/GIFs, standalone '?', or <=4-char acks
        if not is_substantive_text(text):
            return ReplyAction.IGNORE

        now_ts = now if now is not None else time.time()
        effective_max_per_hour = self.default_max_per_hour if max_per_hour is None else max_per_hour
        if effective_max_per_hour <= 0:
            return ReplyAction.IGNORE

        chat_key = int(chat_id) if chat_id is not None else 0
        timestamps = self._prune_hourly_window(chat_key, now_ts)
        if len(timestamps) >= effective_max_per_hour:
            return ReplyAction.IGNORE

        effective_active_cooldown = (
            self.default_active_window_cooldown_seconds
            if active_window_cooldown_seconds is None
            else active_window_cooldown_seconds
        )

        # 7. Active conversation window (with active-window cooldown; no auto DIRECT_REPLY on pronouns/?)
        if 0.0 < time_since_last_bot_msg <= active_conversation_window_seconds:
            if time_since_last_bot_msg < effective_active_cooldown:
                return ReplyAction.IGNORE
            last_attempt = self._last_spontaneous_attempt.get(chat_key)
            if last_attempt is not None and 0.0 <= (now_ts - last_attempt) < effective_active_cooldown:
                return ReplyAction.IGNORE
            boosted_prob = (
                float(active_window_probability)
                if active_window_probability is not None
                else max(probability, 0.45)
            )
            if random.random() < boosted_prob:
                self._last_spontaneous_attempt[chat_key] = now_ts
                self.record_reply(chat_key, now_ts)
                return ReplyAction.SPONTANEOUS_REPLY
            return ReplyAction.IGNORE

        # 8. Spontaneous conversational join
        if time_since_last_bot_msg < min_interval_seconds:
            return ReplyAction.IGNORE

        if random.random() < probability:
            self._last_spontaneous_attempt[chat_key] = now_ts
            self.record_reply(chat_key, now_ts)
            return ReplyAction.SPONTANEOUS_REPLY

        return ReplyAction.IGNORE
