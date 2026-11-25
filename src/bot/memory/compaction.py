"""Background Memory Compaction, Subject Canonicalization, and Upsert Fact Extraction Engine."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FactRecord

logger = logging.getLogger(__name__)

_AGREEMENT_KEYWORDS = (
    "договор",
    "соглас",
    "утверд",
    "план",
    "давайте",
    "решили",
    "соберем",
    "собираем",
    "встреч",
    "будемо",
    "підемо",
    "катку",
    "катки",
    "играть",
    "ґрати",
)

_EVENT_KEYWORDS = (
    "случил",
    "произош",
    "новост",
    "сегодня",
    "вчера",
    "запустил",
    "вышел",
    "сломал",
    "приехал",
    "подорожал",
    "теперь по",
    "маршрутк",
)

_BIO_KEYWORDS = (
    "люблю",
    "любит",
    "обожа",
    "нравит",
    "предпочита",
    "ненавиж",
    "работа",
    "живу",
    "учусь",
    "купил",
    "смотрю",
    "слушаю",
    "записал",
    "эксперимент",
    "шаурм",
    "пельмен",
    "клавиатур",
)

_PARTICIPANT_LINE_RE = re.compile(
    r"^[-*•\d.)\s]*\[?(?P<author>[^]:\n(]+?)(?:\s*\((?:ID|user_id)\s*:\s*(?P<uid>-?\d+)\))?\]?\s*:\s*(?P<fact>.+)$",
    re.IGNORECASE,
)

_EMBEDDED_ID_RE = re.compile(
    r"\((?:ID|user_id)\s*:\s*(?P<uid>-?\d+)\)",
    re.IGNORECASE,
)

_TRAILING_ID_TAG_RE = re.compile(
    r"\s*\((?:ID|user_id)\s*:\s*(?P<uid>-?\d+)\)\s*$",
    re.IGNORECASE,
)

_USER_FALLBACK_SUBJ_RE = re.compile(
    r"^User_(?P<uid>-?\d+)$",
    re.IGNORECASE,
)

_LAUGHTER_TOKEN_RE = re.compile(
    r"^(?:(?:а+х+|х+а+|х+е+|п+х+а+|в+х+п*х*а+|з+а+х+){2,}[ахпвзео]*"
    r"|(?:пхах|вхах|вхпх|хпх)[ахпвз]*"
    r"|(?:ахпх|хапх)[ахпв]*"
    r"|(?:ha){2,}h?|(?:he){2,}h?"
    r"|(?:lol|kek|лол|кек|ржу|ору)+)$",
    re.IGNORECASE,
)

_GROUP_SUBJECT_ALIASES = frozenset({"group", "группа", "чат", "general"})

_STANDARD_FACT_CATEGORIES = frozenset(
    {"lore", "event", "agreement", "friend_bio", "observation"}
)

_OP_ALIASES: Dict[str, str] = {
    "add": "add",
    "insert": "add",
    "create": "add",
    "new": "add",
    "update": "update",
    "replace": "update",
    "edit": "update",
    "modify": "update",
    "supersede": "supersede",
    "delete": "supersede",
    "obsolete": "supersede",
    "archive": "supersede",
    "remove": "supersede",
}

_PISTRUNKINYA_VARIANTS = frozenset({"піструнькиня", "пиструнькиня", "піструн", "пиструн"})
_DEFAULT_IGNORED_BOT_NAMES = frozenset(
    {
        "hermes_kuk7yi6kfm3lcdfa_bot",
        "піструнькиня",
        "пиструнькиня",
        "піструн",
        "пиструн",
        "джефф",
        "jeff",
        "гермес",
        "hermes",
    }
)

_MIN_SUBSTANTIVE_CHARS = 15
_MIN_SUBSTANTIVE_LETTERS = 5


def _coerce_optional_int(value: Any) -> Optional[int]:
    """Safely convert int/numeric string to int, rejecting booleans and invalid values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and re.fullmatch(r"-?\d+", cleaned):
            try:
                return int(cleaned)
            except ValueError:
                return None
    return None


def _coerce_positive_ttl_days(value: Any) -> Optional[float]:
    """Return positive float ttl_days when value is a positive int/float (or numeric string)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if val > 0.0 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            try:
                val = float(cleaned)
                return val if val > 0.0 else None
            except ValueError:
                return None
    return None


def _extract_profile_name_and_aliases(profile: Any) -> Tuple[str, List[str]]:
    """Extract canonical name and alias list from a FriendProfile object, dict, or string."""
    if profile is None:
        return "", []
    if isinstance(profile, str):
        return profile.strip(), []
    if isinstance(profile, Mapping):
        name = str(profile.get("name") or "").strip()
        raw_aliases = profile.get("aliases") or []
        aliases = [str(a).strip() for a in raw_aliases if a and str(a).strip()]
        return name, aliases
    name = str(getattr(profile, "name", "") or "").strip()
    raw_aliases = getattr(profile, "aliases", None) or []
    aliases = [str(a).strip() for a in raw_aliases if a and str(a).strip()]
    return name, aliases


def _normalize_friends_map(
    friends_map: Optional[Mapping[int, Any]],
) -> Dict[int, Tuple[str, List[str]]]:
    """Normalize friends_map keys to int and extract (name, aliases) pairs."""
    if not friends_map:
        return {}
    normalized: Dict[int, Tuple[str, List[str]]] = {}
    for raw_uid, profile in friends_map.items():
        uid = _coerce_optional_int(raw_uid)
        if uid is None:
            continue
        name, aliases = _extract_profile_name_and_aliases(profile)
        if name:
            normalized[uid] = (name, aliases)
    return normalized


def _build_bot_name_set(bot_names: Optional[Sequence[str]]) -> Set[str]:
    """Normalize configured bot names and always include ignored sibling bot handles."""
    names: Set[str] = set(_DEFAULT_IGNORED_BOT_NAMES)
    if not bot_names:
        return names
    for raw in bot_names:
        if not raw:
            continue
        norm = str(raw).strip().lstrip("@").lower()
        if norm:
            names.add(norm)
    if names & _PISTRUNKINYA_VARIANTS:
        names.update(_PISTRUNKINYA_VARIANTS)
    return names


def is_explicit_empty_fact_response(raw_result: Any) -> bool:
    """Return True when LLM explicitly answered with an empty JSON array/object (0 durable facts)."""
    if isinstance(raw_result, list):
        return len(raw_result) == 0
    if isinstance(raw_result, dict):
        for key in ("facts", "operations"):
            if isinstance(raw_result.get(key), list):
                return len(raw_result[key]) == 0
        return False
    if isinstance(raw_result, str):
        cleaned = raw_result.strip()
        if not cleaned:
            return False
        fence_match = re.search(
            r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        candidate = fence_match.group(1).strip() if fence_match else cleaned
        try:
            parsed = json.loads(candidate)
        except Exception:
            return False
        if isinstance(parsed, list):
            return len(parsed) == 0
        if isinstance(parsed, dict):
            for key in ("facts", "operations"):
                if isinstance(parsed.get(key), list):
                    return len(parsed[key]) == 0
    return False


def strip_fact_id_tag(fact_text: str) -> Tuple[str, Optional[int]]:
    """Extract and strip trailing `(ID:12345)` / `(user_id: 12345)` suffix from fact text if present."""
    if not fact_text:
        return "", None
    text = str(fact_text).strip()
    match = _TRAILING_ID_TAG_RE.search(text)
    if not match:
        return text, None
    uid = _coerce_optional_int(match.group("uid"))
    cleaned = text[: match.start()].strip()
    return cleaned, uid


def canonicalize_subject(
    raw_subject: str,
    *,
    user_id: Optional[int] = None,
    friends_map: Optional[Mapping[int, Any]] = None,
    author_id_by_name: Optional[Mapping[str, int]] = None,
    bot_id: Optional[int] = None,
    bot_names: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """Canonicalize fact subject name and user_id, rejecting self-bot facts as `(None, None)`."""
    uid = _coerce_optional_int(user_id)
    subj_raw = str(raw_subject or "").strip()

    # Extract embedded (ID:12345) / (user_id: 12345) if present in raw_subject
    id_match = _EMBEDDED_ID_RE.search(subj_raw)
    if id_match:
        embedded_uid = _coerce_optional_int(id_match.group("uid"))
        if uid is None and embedded_uid is not None:
            uid = embedded_uid
        subj_raw = _EMBEDDED_ID_RE.sub("", subj_raw)

    clean_subj = subj_raw.strip()
    if clean_subj.startswith("[") and clean_subj.endswith("]") and len(clean_subj) > 2:
        clean_subj = clean_subj[1:-1].strip()

    if uid is None and clean_subj:
        user_fb_match = _USER_FALLBACK_SUBJ_RE.fullmatch(clean_subj)
        if user_fb_match:
            uid = _coerce_optional_int(user_fb_match.group("uid"))

    norm_bot_id = _coerce_optional_int(bot_id)
    if uid is not None and norm_bot_id is not None and uid == norm_bot_id:
        return None, None

    bot_name_set = _build_bot_name_set(bot_names)
    clean_lower = clean_subj.lstrip("@").lower()
    if clean_lower and clean_lower in bot_name_set:
        return None, None

    if clean_subj.lower() in _GROUP_SUBJECT_ALIASES:
        return "Group", None

    norm_friends = _normalize_friends_map(friends_map)
    if uid is not None and uid in norm_friends:
        return norm_friends[uid][0], int(uid)

    if uid is None and clean_lower and norm_friends:
        matched_friends: List[Tuple[int, str]] = []
        for friend_uid, (profile_name, profile_aliases) in norm_friends.items():
            alias_lowers = {a.lstrip("@").lower() for a in profile_aliases if a}
            if clean_lower == profile_name.lower() or clean_lower in alias_lowers:
                matched_friends.append((friend_uid, profile_name))
        if len(matched_friends) == 1:
            matched_uid, matched_name = matched_friends[0]
            if norm_bot_id is not None and matched_uid == norm_bot_id:
                return None, None
            return matched_name, int(matched_uid)

    if uid is None and clean_lower and author_id_by_name:
        norm_author_map: Dict[str, int] = {}
        for raw_name, mapped_val in author_id_by_name.items():
            mapped_int = _coerce_optional_int(mapped_val)
            if raw_name and mapped_int is not None:
                norm_author_map[str(raw_name).strip().lower()] = mapped_int
        resolved_uid = norm_author_map.get(clean_subj.lower()) or norm_author_map.get(clean_lower)
        if resolved_uid is not None:
            if norm_bot_id is not None and resolved_uid == norm_bot_id:
                return None, None
            if resolved_uid in norm_friends:
                return norm_friends[resolved_uid][0], int(resolved_uid)
            return clean_subj, int(resolved_uid)

    return (clean_subj or "Group"), (int(uid) if uid is not None else None)


def _is_substantive_message(
    msg: ChatMessage,
    excluded_users: Optional[Set[int]] = None,
) -> bool:
    """Return True if a message is a substantive human message (not bot or trivial noise)."""
    if excluded_users and int(msg.user_id) in excluded_users:
        return False
    text = (msg.text or "").strip()
    if len(text) < _MIN_SUBSTANTIVE_CHARS:
        return False
    letters = re.findall(r"[^\W\d_]", text)
    if len(letters) < _MIN_SUBSTANTIVE_LETTERS:
        return False
    tokens = re.findall(r"[^\W\d_]+", text.lower())
    if tokens and all(_LAUGHTER_TOKEN_RE.fullmatch(tok) for tok in tokens):
        return False
    return True


def _infer_fact_category(text: str) -> str:
    """Infer a semantic fact category from message or summary text."""
    lower = text.lower()
    if any(k in lower for k in _AGREEMENT_KEYWORDS):
        return "agreement"
    if any(k in lower for k in _EVENT_KEYWORDS):
        return "event"
    if any(k in lower for k in _BIO_KEYWORDS):
        return "friend_bio"
    return "friend_bio"


def format_transcript_for_compaction(
    messages: List[ChatMessage],
    friends_map: Optional[Mapping[int, Any]] = None,
) -> str:
    """Format messages into a clean transcript for LLM fact extraction using canonical friend names."""
    norm_friends = _normalize_friends_map(friends_map)
    lines = []
    for m in messages:
        text = m.text or ""
        if m.media_summary:
            text = f"{text} [Media: {m.media_summary}]".strip()
        uid = _coerce_optional_int(m.user_id)
        if uid is not None and uid in norm_friends:
            display_name = norm_friends[uid][0]
        else:
            display_name = m.author_name
        lines.append(f"[{display_name} (ID:{m.user_id})]: {text}")
    return "\n".join(lines)


def build_compaction_prompt(
    transcript: str,
    existing_facts: Optional[Sequence[Any]] = None,
) -> str:
    """Build Russian prompt for upsert-aware fact extraction from a conversation batch."""
    existing_block = ""
    if existing_facts:
        lines = []
        for f in existing_facts:
            fid = getattr(f, "id", None)
            subj = getattr(f, "subject", "Group")
            cat = getattr(f, "category", "lore")
            fact_txt = getattr(f, "fact", "")
            if fid is not None and fact_txt:
                lines.append(f"- [ID:{fid}] ({cat}) {subj}: {fact_txt}")
        if lines:
            existing_block = (
                "Уже известные активные факты по участникам этого чата "
                "(НЕ дублируй их; если новый факт уточняет или отменяет старый — используй "
                'op="update" или op="supersede" с точным target_id из списка):\n'
                + "\n".join(lines)
                + "\n\n"
            )

    return (
        "Проанализируй фрагмент переписки участников чата и выдели ТОЛЬКО 1-3 по-настоящему "
        "долгосрочных факта (важные договорённости, реальные события, устойчивые интересы, "
        "характерные наблюдения или изменения в жизни участников). "
        "Игнорируй односложную болтовню, смех, приветствия, мелкий бытовой шум и реплики самого бота. "
        "Если ничего по-настоящему долгосрочного нет — верни пустой массив [].\n"
        "Используй каноническое имя участника БЕЗ суффикса `(ID:...)` в поле `subject`, "
        "а числовой ID передавай отдельно в `subject_user_id`.\n"
        "Формат строго JSON-массив операций:\n"
        '[{"op": "add|update|supersede", "target_id": null, "subject": "Имя или Group", '
        '"subject_user_id": 123456, "fact": "...", '
        '"category": "lore|event|agreement|friend_bio|observation", "ttl_days": null}]\n'
        "(Для временных договорённостей/событий можно указать `ttl_days`, например 7 или 14; "
        "для постоянных фактов `friend_bio`/`lore`/`observation` оставляй `ttl_days`: null).\n\n"
        f"{existing_block}"
        f"Переписка:\n{transcript}"
    )


class CompactionEngine:
    """Engine responsible for summarizing older chat logs and upserting canonical durable facts."""

    def __init__(
        self,
        memory_store: MemoryStore,
        summarize_fn: Optional[Callable[..., Any]] = None,
        threshold: Optional[int] = None,
        batch_size: int = 30,
        excluded_user_ids: Optional[Sequence[int]] = None,
        min_substantive_messages: int = 6,
        friends_provider: Optional[Callable[[], Optional[Mapping[int, Any]]]] = None,
        bot_names: Optional[Sequence[str]] = None,
        embedder: Optional[Any] = None,
        agreement_ttl_days: Optional[int] = None,
        event_ttl_days: Optional[int] = None,
        bot_id: Optional[int] = None,
    ) -> None:
        self.memory_store = memory_store
        self.summarize_fn = summarize_fn
        self.batch_size = batch_size
        self.threshold = threshold if threshold is not None else batch_size
        self.excluded_user_ids: Set[int] = (
            {int(uid) for uid in excluded_user_ids} if excluded_user_ids else set()
        )
        self.min_substantive_messages = min_substantive_messages
        self.friends_provider = friends_provider
        self.bot_names: List[str] = [str(n) for n in bot_names if n] if bot_names else []
        self.embedder = embedder
        self.agreement_ttl_days = agreement_ttl_days
        self.event_ttl_days = event_ttl_days
        self.bot_id = _coerce_optional_int(bot_id)

    def _resolve_excluded_users(
        self,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> Set[int]:
        if not excluded_user_ids:
            return set(self.excluded_user_ids)
        return self.excluded_user_ids | {int(uid) for uid in excluded_user_ids}

    def _resolve_friends_map(self) -> Optional[Mapping[int, Any]]:
        if self.friends_provider is None:
            return None
        try:
            return self.friends_provider()
        except Exception as exc:
            logger.debug("Failed to resolve friends_map in CompactionEngine: %s", type(exc).__name__)
            return None

    @staticmethod
    def _build_author_context(
        substantive: Sequence[ChatMessage],
        friends_map: Optional[Mapping[int, Any]] = None,
    ) -> Tuple[Dict[str, int], List[int], List[str]]:
        """Extract unambiguous author_id_by_name, participant_uids, and participant_names from messages."""
        norm_friends = _normalize_friends_map(friends_map)
        uids_by_name: Dict[str, Set[int]] = {}
        participant_uids_set: Set[int] = set()
        participant_names: List[str] = []
        seen_names_lower: Set[str] = set()

        for msg in substantive:
            uid = _coerce_optional_int(msg.user_id)
            if uid is not None:
                participant_uids_set.add(uid)
            raw_author = (msg.author_name or "").strip()
            if raw_author and uid is not None:
                uids_by_name.setdefault(raw_author.lower(), set()).add(uid)
            if raw_author and raw_author.lower() not in seen_names_lower:
                seen_names_lower.add(raw_author.lower())
                participant_names.append(raw_author)
            if uid is not None and uid in norm_friends:
                canon_name = norm_friends[uid][0]
                if canon_name:
                    uids_by_name.setdefault(canon_name.lower(), set()).add(uid)
                    if canon_name.lower() not in seen_names_lower:
                        seen_names_lower.add(canon_name.lower())
                        participant_names.append(canon_name)

        author_id_by_name = {
            name_lower: next(iter(uids))
            for name_lower, uids in uids_by_name.items()
            if name_lower and len(uids) == 1
        }
        return author_id_by_name, sorted(participant_uids_set), participant_names

    async def _invoke_summarize_fn(
        self,
        transcript: str,
        existing_facts: List[FactRecord],
        chat_id: int,
    ) -> Any:
        """Call summarize_fn with backward-compatible signature inspection."""
        assert self.summarize_fn is not None
        call_kwargs: Dict[str, Any] = {}
        try:
            sig = inspect.signature(self.summarize_fn)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_var_kw:
                call_kwargs["existing_facts"] = existing_facts
                call_kwargs["chat_id"] = chat_id
            else:
                if "existing_facts" in sig.parameters:
                    p = sig.parameters["existing_facts"]
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        call_kwargs["existing_facts"] = existing_facts
                if "chat_id" in sig.parameters:
                    p = sig.parameters["chat_id"]
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        call_kwargs["chat_id"] = chat_id
        except (TypeError, ValueError):
            call_kwargs = {}

        if asyncio.iscoroutinefunction(self.summarize_fn):
            return await self.summarize_fn(transcript, **call_kwargs)
        res = self.summarize_fn(transcript, **call_kwargs)
        return await res if asyncio.iscoroutine(res) else res

    def _compute_expires_at(
        self,
        ttl_days_raw: Any,
        category: str,
        now_ts: int,
    ) -> Optional[int]:
        """Compute Unix expiration timestamp without forcing short TTL onto timeless categories."""
        ttl_days = _coerce_positive_ttl_days(ttl_days_raw)
        if ttl_days is not None:
            return now_ts + int(ttl_days * 86400)
        cat_lower = category.lower()
        if cat_lower in ("friend_bio", "lore", "observation"):
            return None
        if cat_lower == "agreement" and self.agreement_ttl_days and self.agreement_ttl_days > 0:
            return now_ts + int(self.agreement_ttl_days * 86400)
        if cat_lower == "event" and self.event_ttl_days and self.event_ttl_days > 0:
            return now_ts + int(self.event_ttl_days * 86400)
        return None

    async def _add_fact_compat(
        self,
        *,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int],
        subject_user_id: Optional[int] = None,
        expires_at: Optional[int] = None,
    ) -> int:
        """Insert fact using extended lifecycle kwargs with TypeError fallback for 4-arg mocks."""
        try:
            res = await self.memory_store.add_fact(
                subject=subject,
                fact=fact,
                category=category,
                source_chat_id=source_chat_id,
                subject_user_id=subject_user_id,
                status="active",
                expires_at=expires_at,
            )
        except TypeError:
            res = await self.memory_store.add_fact(
                subject=subject,
                fact=fact,
                category=category,
                source_chat_id=source_chat_id,
            )
        return int(res or 0)

    async def _index_fact_safe(
        self,
        fact_id: Optional[int],
        subject: str,
        fact_text: str,
        category: str,
    ) -> None:
        """Index newly created or updated fact in embedder without failing compaction on errors."""
        if self.embedder is None or not fact_id or int(fact_id) <= 0:
            return
        try:
            await self.embedder.index_fact(
                self.memory_store,
                int(fact_id),
                subject,
                fact_text,
                category,
            )
        except Exception as exc:
            logger.debug(
                "Embedder index_fact failed during compaction for fact_id=%s: %s",
                fact_id,
                type(exc).__name__,
            )

    async def compact_if_needed(
        self,
        chat_id: int,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """Alias for run_compaction_for_chat to compact messages when threshold is met."""
        return await self.run_compaction_for_chat(
            chat_id,
            excluded_user_ids=excluded_user_ids,
        )

    async def run_compaction_for_chat(
        self,
        chat_id: int,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> int:
        """Run a single compaction cycle for a chat if threshold is met."""
        required_count = max(self.threshold, self.batch_size)
        all_candidates = await self.memory_store.get_uncompacted_messages(
            chat_id, limit=required_count
        )
        if len(all_candidates) < self.threshold or len(all_candidates) < self.batch_size:
            return 0
        uncompacted = all_candidates[: self.batch_size]
        effective_excluded = self._resolve_excluded_users(excluded_user_ids)
        substantive = [
            m for m in uncompacted if _is_substantive_message(m, effective_excluded)
        ]

        msg_ids = [m.id for m in uncompacted if m.id is not None]
        if len(substantive) < self.min_substantive_messages:
            await self.memory_store.mark_compacted(msg_ids)
            logger.info(
                "Marked %d low-signal messages compacted for chat %d (substantive=%d < %d)",
                len(msg_ids),
                chat_id,
                len(substantive),
                self.min_substantive_messages,
            )
            return len(msg_ids)

        friends_map = self._resolve_friends_map()
        author_id_by_name, participant_uids, participant_names = self._build_author_context(
            substantive, friends_map=friends_map
        )

        existing_facts: List[FactRecord] = []
        if hasattr(self.memory_store, "get_active_facts"):
            try:
                existing_facts = await self.memory_store.get_active_facts(
                    chat_id=chat_id,
                    subject_user_ids=participant_uids or None,
                    subjects=["Group", *participant_names],
                    limit=12,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to load existing active facts for chat %s: %s",
                    chat_id,
                    type(exc).__name__,
                )
                existing_facts = []

        facts_to_store: List[Dict[str, Any]] = []
        llm_explicit_empty = False

        # 1. If summarize_fn provided, extract summary/upsert ops from substantive human transcript
        if self.summarize_fn:
            transcript = format_transcript_for_compaction(substantive, friends_map=friends_map)
            try:
                raw_result = await self._invoke_summarize_fn(
                    transcript,
                    existing_facts=existing_facts,
                    chat_id=chat_id,
                )
                if is_explicit_empty_fact_response(raw_result):
                    llm_explicit_empty = True
                else:
                    facts_to_store.extend(self._parse_extracted_facts(raw_result))
            except Exception as exc:
                logger.error(
                    "Error during summarization for chat %s: %s",
                    chat_id,
                    type(exc).__name__,
                )

        # 2. Deterministic fallback ONLY when summarize_fn is None or failed/returned unstructured empty string
        if not facts_to_store and not llm_explicit_empty:
            summary_fact = self._build_batch_summary_fact(substantive, friends_map=friends_map)
            if summary_fact:
                facts_to_store.append(summary_fact)
            participant_facts = self.extract_participant_facts(
                substantive,
                excluded_user_ids=effective_excluded,
            )
            facts_to_store.extend(participant_facts)

        # 3. Execute add/update/supersede operations safely with strict chat target isolation
        now_ts = int(time.time())
        existing_by_id: Dict[int, FactRecord] = {
            int(ef.id): ef
            for ef in existing_facts
            if getattr(ef, "id", None) is not None and ef.source_chat_id == chat_id
        }
        allowed_target_ids: Set[int] = set(existing_by_id.keys())

        seen_adds: Set[Tuple[str, str]] = set()
        for ef in existing_facts:
            ef_clean_fact, _ = strip_fact_id_tag(ef.fact or "")
            ef_norm_fact = (ef_clean_fact or (ef.fact or "").strip()).lower()
            ef_subj = (ef.subject or "Group").strip().lower()
            if ef_norm_fact:
                seen_adds.add((ef_subj, ef_norm_fact))

        eff_bot_id = (
            self.bot_id
            if self.bot_id is not None
            else (next(iter(effective_excluded)) if len(effective_excluded) == 1 else None)
        )

        for f in facts_to_store:
            op = str(f.get("op") or "add").strip().lower()
            if op not in ("add", "update", "supersede"):
                continue

            target_id = _coerce_optional_int(f.get("target_id"))
            if op in ("update", "supersede"):
                if target_id is None or target_id not in allowed_target_ids:
                    logger.debug(
                        "Dropping compaction op=%s with unauthorized target_id=%s in chat %s",
                        op,
                        target_id,
                        chat_id,
                    )
                    continue

            raw_fact_text = str(f.get("fact") or f.get("fact_text") or "").strip()
            clean_fact_body, embedded_fact_uid = strip_fact_id_tag(raw_fact_text)

            # Pure supersede without replacement fact text
            if op == "supersede" and not raw_fact_text:
                assert target_id is not None
                if hasattr(self.memory_store, "supersede_fact"):
                    await self.memory_store.supersede_fact(target_id)
                allowed_target_ids.discard(target_id)
                continue

            if not raw_fact_text:
                continue

            target_record = existing_by_id.get(target_id) if target_id is not None else None
            raw_uid = _coerce_optional_int(
                f.get("subject_user_id")
                if f.get("subject_user_id") is not None
                else f.get("user_id")
            )
            if raw_uid is None and embedded_fact_uid is not None:
                raw_uid = embedded_fact_uid
            if raw_uid is None and target_record is not None and not f.get("_explicit_subject", True):
                raw_uid = _coerce_optional_int(getattr(target_record, "subject_user_id", None))

            raw_subject = str(f.get("subject") or "").strip()
            if not raw_subject and target_record is not None:
                raw_subject = target_record.subject

            canon_subj, canon_uid = canonicalize_subject(
                raw_subject or "Group",
                user_id=raw_uid,
                friends_map=friends_map,
                author_id_by_name=author_id_by_name,
                bot_id=eff_bot_id,
                bot_names=self.bot_names,
            )
            if canon_subj is None:
                continue
            if canon_uid is not None and canon_uid in effective_excluded:
                continue

            raw_category = str(f.get("category") or "").strip()
            if not raw_category and target_record is not None:
                raw_category = target_record.category
            if not raw_category:
                raw_category = "lore"
            category = (
                raw_category.lower()
                if raw_category.lower() in _STANDARD_FACT_CATEGORIES
                else raw_category
            )

            fact_text = (
                raw_fact_text
                if f.get("_keep_id_tag")
                else (clean_fact_body or raw_fact_text)
            )
            norm_fact_key = (clean_fact_body or fact_text).strip().lower()
            if not norm_fact_key:
                continue

            expires_at = self._compute_expires_at(f.get("ttl_days"), category, now_ts)

            if op == "update":
                assert target_id is not None
                if not hasattr(self.memory_store, "update_fact"):
                    continue
                if (
                    expires_at is None
                    and f.get("ttl_days") is None
                    and category.lower() not in ("friend_bio", "lore", "observation")
                    and target_record is not None
                ):
                    expires_at = getattr(target_record, "expires_at", None)
                await self.memory_store.update_fact(
                    target_id,
                    fact=fact_text,
                    subject=canon_subj,
                    subject_user_id=canon_uid,
                    category=category,
                    expires_at=expires_at,
                )
                seen_adds.add((canon_subj.lower(), norm_fact_key))
                await self._index_fact_safe(target_id, canon_subj, fact_text, category)
            elif op == "supersede":
                assert target_id is not None
                if not hasattr(self.memory_store, "supersede_fact"):
                    continue
                new_id = await self._add_fact_compat(
                    subject=canon_subj,
                    fact=fact_text,
                    category=category,
                    source_chat_id=chat_id,
                    subject_user_id=canon_uid,
                    expires_at=expires_at,
                )
                try:
                    await self.memory_store.supersede_fact(target_id, new_fact_id=new_id)
                except TypeError:
                    await self.memory_store.supersede_fact(target_id)
                allowed_target_ids.discard(target_id)
                seen_adds.add((canon_subj.lower(), norm_fact_key))
                await self._index_fact_safe(new_id, canon_subj, fact_text, category)
            else:
                dedup_key = (canon_subj.lower(), norm_fact_key)
                if dedup_key in seen_adds:
                    continue
                seen_adds.add(dedup_key)
                new_id = await self._add_fact_compat(
                    subject=canon_subj,
                    fact=fact_text,
                    category=category,
                    source_chat_id=chat_id,
                    subject_user_id=canon_uid,
                    expires_at=expires_at,
                )
                await self._index_fact_safe(new_id, canon_subj, fact_text, category)

        # 4. Mark messages compacted
        await self.memory_store.mark_compacted(msg_ids)
        logger.info("Compacted %d messages for chat %d", len(msg_ids), chat_id)
        return len(msg_ids)

    def _build_batch_summary_fact(
        self,
        messages: List[ChatMessage],
        friends_map: Optional[Mapping[int, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a concise deterministic summary fact for a compacted message batch."""
        if not messages:
            return None
        norm_friends = _normalize_friends_map(friends_map)
        participants: List[str] = []
        snippets: List[str] = []
        for m in messages:
            uid = _coerce_optional_int(m.user_id)
            if uid is not None and uid in norm_friends:
                author = norm_friends[uid][0]
            else:
                author = (m.author_name or f"User_{m.user_id}").strip()
            if author and author not in participants:
                participants.append(author)
            content = (m.text or "").strip()
            if m.media_summary:
                content = f"{content} [Media: {m.media_summary}]".strip()
            if content and len(snippets) < 5:
                snippets.append(f"{author}: {content[:120]}")
        if not snippets:
            return None
        joined_authors = ", ".join(participants[:6])
        summary_text = f"Участники ({joined_authors}): " + " | ".join(snippets)
        return {
            "op": "add",
            "subject": "Group",
            "fact": summary_text,
            "category": "lore",
        }

    def extract_participant_facts(
        self,
        messages: List[ChatMessage],
        max_per_participant: int = 3,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract individual participant facts (user_id, category, fact_text) from substantive messages."""
        effective_excluded = self._resolve_excluded_users(excluded_user_ids)
        by_participant: Dict[Tuple[int, str], List[str]] = {}
        for m in messages:
            if not _is_substantive_message(m, effective_excluded):
                continue
            content = (m.text or "").strip()
            if m.media_summary:
                media_note = f"[Media: {m.media_summary.strip()}]"
                content = f"{content} {media_note}".strip()
            author = (m.author_name or f"User_{m.user_id}").strip()
            key = (int(m.user_id), author)
            bucket = by_participant.setdefault(key, [])
            if content not in bucket:
                bucket.append(content)

        extracted: List[Dict[str, Any]] = []
        for (user_id, author), contents in by_participant.items():
            selected = contents[:max_per_participant]
            for text in selected:
                category = _infer_fact_category(text)
                id_tag = f"(ID:{user_id})"
                fact_text = text if id_tag in text else f"{text} {id_tag}"
                extracted.append(
                    {
                        "op": "add",
                        "user_id": str(user_id),
                        "subject_user_id": int(user_id),
                        "subject": author,
                        "category": category,
                        "fact": fact_text,
                        "fact_text": fact_text,
                        "_keep_id_tag": True,
                    }
                )
        return extracted

    def _parse_extracted_facts(self, raw_result: Any) -> List[Dict[str, Any]]:
        """Parse LLM or callback output (list, dict, fenced JSON, or summary text) into op/fact dicts."""
        if raw_result is None:
            return []

        if isinstance(raw_result, list):
            parsed: List[Dict[str, Any]] = []
            for item in raw_result:
                if isinstance(item, dict):
                    raw_op = str(item.get("op") or item.get("action") or "add").strip().lower()
                    op = _OP_ALIASES.get(raw_op)
                    if op is None:
                        continue

                    target_id: Optional[int] = None
                    if op != "add":
                        target_id = _coerce_optional_int(
                            item.get("target_id")
                            if item.get("target_id") is not None
                            else (
                                item.get("id")
                                if item.get("id") is not None
                                else item.get("fact_id")
                            )
                        )

                    fact_val = (
                        item.get("fact")
                        or item.get("fact_text")
                        or item.get("text")
                        or item.get("summary")
                    )
                    fact_str = str(fact_val).strip() if fact_val is not None else ""

                    if op in ("add", "update") and not fact_str:
                        continue
                    if op == "supersede" and target_id is None and not fact_str:
                        continue

                    subj_uid = _coerce_optional_int(
                        item.get("subject_user_id")
                        if item.get("subject_user_id") is not None
                        else item.get("user_id")
                    )
                    explicit_subj = any(
                        item.get(k) is not None and str(item.get(k)).strip() != ""
                        for k in ("subject", "author", "user")
                    )
                    if explicit_subj:
                        subj = str(
                            item.get("subject") or item.get("author") or item.get("user")
                        ).strip()
                    elif subj_uid is not None:
                        subj = f"User_{subj_uid}"
                    elif op == "add":
                        subj = "Group"
                    else:
                        subj = ""

                    raw_cat = item.get("category")
                    if raw_cat is not None and str(raw_cat).strip():
                        cat_clean = str(raw_cat).strip()
                        category = (
                            cat_clean.lower()
                            if cat_clean.lower() in _STANDARD_FACT_CATEGORIES
                            else cat_clean
                        )
                    else:
                        category = "lore" if op == "add" else ""

                    entry: Dict[str, Any] = {
                        "op": op,
                        "subject": subj,
                        "fact": fact_str,
                        "category": category,
                        "_explicit_subject": explicit_subj or (subj_uid is not None),
                    }
                    if target_id is not None:
                        entry["target_id"] = target_id
                    if subj_uid is not None:
                        entry["subject_user_id"] = subj_uid
                        entry["user_id"] = str(subj_uid)
                    ttl_days = _coerce_positive_ttl_days(item.get("ttl_days"))
                    if ttl_days is not None:
                        entry["ttl_days"] = ttl_days
                    if item.get("_keep_id_tag"):
                        entry["_keep_id_tag"] = True
                    parsed.append(entry)
                elif isinstance(item, FactRecord):
                    rec_entry: Dict[str, Any] = {
                        "op": "add",
                        "subject": item.subject,
                        "fact": item.fact,
                        "category": item.category,
                    }
                    rec_uid = _coerce_optional_int(getattr(item, "subject_user_id", None))
                    if rec_uid is not None:
                        rec_entry["subject_user_id"] = rec_uid
                        rec_entry["user_id"] = str(rec_uid)
                    parsed.append(rec_entry)
                elif isinstance(item, str) and item.strip():
                    parsed.extend(self._parse_text_summary_lines(item))
            return parsed

        if isinstance(raw_result, dict):
            if isinstance(raw_result.get("facts"), list):
                return self._parse_extracted_facts(raw_result["facts"])
            if isinstance(raw_result.get("operations"), list):
                return self._parse_extracted_facts(raw_result["operations"])
            return self._parse_extracted_facts([raw_result])

        if isinstance(raw_result, str):
            cleaned = raw_result.strip()
            if not cleaned:
                return []

            # 1. Try direct JSON load
            try:
                data = json.loads(cleaned)
                return self._parse_extracted_facts(data)
            except Exception:
                pass

            # 2. Try stripping markdown code fences or extracting embedded JSON array/object
            fence_match = re.search(
                r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if fence_match:
                try:
                    data = json.loads(fence_match.group(1))
                    return self._parse_extracted_facts(data)
                except Exception:
                    pass

            bracket_match = re.search(r"(\[\s*\{.*?\}\s*\])", cleaned, flags=re.DOTALL)
            if bracket_match:
                try:
                    data = json.loads(bracket_match.group(1))
                    return self._parse_extracted_facts(data)
                except Exception:
                    pass

            # 3. Fallback: parse plain-text summary and participant lines
            return self._parse_text_summary_lines(cleaned)

        return []

    def _parse_text_summary_lines(self, text: str) -> List[Dict[str, Any]]:
        """Extract summary and participant facts from plain-text summary lines."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        results: List[Dict[str, Any]] = [
            {
                "op": "add",
                "subject": "Group",
                "fact": " ".join(lines),
                "category": "lore",
            }
        ]
        for line in lines:
            m = _PARTICIPANT_LINE_RE.match(line)
            if m:
                author = m.group("author").strip()
                uid = m.group("uid")
                fact_body = m.group("fact").strip()
                if author and fact_body:
                    if uid and f"ID:{uid}" not in fact_body:
                        fact_body = f"{fact_body} (ID:{uid})"
                    item: Dict[str, Any] = {
                        "op": "add",
                        "subject": author,
                        "fact": fact_body,
                        "category": _infer_fact_category(fact_body),
                        "_keep_id_tag": True,
                    }
                    uid_int = _coerce_optional_int(uid)
                    if uid_int is not None:
                        item["subject_user_id"] = uid_int
                        item["user_id"] = str(uid_int)
                    results.append(item)
        return results
