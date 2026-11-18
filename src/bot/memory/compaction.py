"""Background Memory Compaction and Context Summarization Engine."""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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

_LAUGHTER_TOKEN_RE = re.compile(
    r"^(?:(?:а+х+|х+а+|х+е+|п+х+а+|в+х+п*х*а+|з+а+х+){2,}[ахпвзео]*"
    r"|(?:пхах|вхах|вхпх|хпх)[ахпвз]*"
    r"|(?:ахпх|хапх)[ахпв]*"
    r"|(?:ha){2,}h?|(?:he){2,}h?"
    r"|(?:lol|kek|лол|кек|ржу|ору)+)$",
    re.IGNORECASE,
)

_MIN_SUBSTANTIVE_CHARS = 15
_MIN_SUBSTANTIVE_LETTERS = 5


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


def format_transcript_for_compaction(messages: List[ChatMessage]) -> str:
    """Format messages into a clean transcript for LLM fact extraction."""
    lines = []
    for m in messages:
        text = m.text or ""
        if m.media_summary:
            text = f"{text} [Media: {m.media_summary}]".strip()
        lines.append(f"[{m.author_name} (ID:{m.user_id})]: {text}")
    return "\n".join(lines)


class CompactionEngine:
    """Engine responsible for summarizing older chat logs and extracting durable facts."""

    def __init__(
        self,
        memory_store: MemoryStore,
        summarize_fn: Optional[Callable[[str], Any]] = None,
        threshold: Optional[int] = None,
        batch_size: int = 30,
        excluded_user_ids: Optional[Sequence[int]] = None,
        min_substantive_messages: int = 6,
    ) -> None:
        self.memory_store = memory_store
        self.summarize_fn = summarize_fn
        self.batch_size = batch_size
        self.threshold = threshold if threshold is not None else batch_size
        self.excluded_user_ids: Set[int] = (
            {int(uid) for uid in excluded_user_ids} if excluded_user_ids else set()
        )
        self.min_substantive_messages = min_substantive_messages

    def _resolve_excluded_users(
        self,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> Set[int]:
        if not excluded_user_ids:
            return set(self.excluded_user_ids)
        return self.excluded_user_ids | {int(uid) for uid in excluded_user_ids}

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

        facts_to_store: List[Dict[str, str]] = []

        # 1. If summarize_fn provided, extract summary facts from substantive human transcript
        if self.summarize_fn:
            transcript = format_transcript_for_compaction(substantive)
            try:
                if asyncio.iscoroutinefunction(self.summarize_fn):
                    raw_result = await self.summarize_fn(transcript)
                else:
                    res = self.summarize_fn(transcript)
                    raw_result = await res if asyncio.iscoroutine(res) else res

                facts_to_store.extend(self._parse_extracted_facts(raw_result))
            except Exception as e:
                logger.error("Error during summarization for chat %s: %s", chat_id, e)

        # 2. Deterministic fallback only when summarize_fn is None or failed/returned empty
        if not facts_to_store:
            summary_fact = self._build_batch_summary_fact(substantive)
            if summary_fact:
                facts_to_store.append(summary_fact)
            participant_facts = self.extract_participant_facts(
                substantive,
                excluded_user_ids=effective_excluded,
            )
            facts_to_store.extend(participant_facts)

        # 3. Persist deduplicated facts into memory_facts (and FTS5 index via triggers)
        seen: set[Tuple[str, str]] = set()
        for f in facts_to_store:
            subject = str(f.get("subject") or "Group").strip()
            fact_text = str(f.get("fact") or f.get("fact_text") or "").strip()
            category = str(f.get("category") or "lore").strip()
            if not fact_text:
                continue
            dedup_key = (subject.lower(), fact_text.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            await self.memory_store.add_fact(
                subject=subject,
                fact=fact_text,
                category=category,
                source_chat_id=chat_id,
            )

        # 4. Mark messages compacted
        await self.memory_store.mark_compacted(msg_ids)
        logger.info("Compacted %d messages for chat %d", len(msg_ids), chat_id)
        return len(msg_ids)

    def _build_batch_summary_fact(
        self, messages: List[ChatMessage]
    ) -> Optional[Dict[str, str]]:
        """Build a concise deterministic summary fact for a compacted message batch."""
        if not messages:
            return None
        participants: List[str] = []
        snippets: List[str] = []
        for m in messages:
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
            "subject": "Group",
            "fact": summary_text,
            "category": "lore",
        }

    def extract_participant_facts(
        self,
        messages: List[ChatMessage],
        max_per_participant: int = 3,
        excluded_user_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, str]]:
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

        extracted: List[Dict[str, str]] = []
        for (user_id, author), contents in by_participant.items():
            # Select up to max_per_participant notable messages for this participant
            selected = contents[:max_per_participant]
            for text in selected:
                category = _infer_fact_category(text)
                id_tag = f"(ID:{user_id})"
                fact_text = text if id_tag in text else f"{text} {id_tag}"
                extracted.append(
                    {
                        "user_id": str(user_id),
                        "subject": author,
                        "category": category,
                        "fact": fact_text,
                        "fact_text": fact_text,
                    }
                )
        return extracted

    def _parse_extracted_facts(self, raw_result: Any) -> List[Dict[str, str]]:
        """Parse LLM or callback output (list, dict, fenced JSON, or summary text) into fact dicts."""
        if raw_result is None:
            return []

        if isinstance(raw_result, list):
            parsed: List[Dict[str, str]] = []
            for item in raw_result:
                if isinstance(item, dict):
                    fact_val = (
                        item.get("fact")
                        or item.get("fact_text")
                        or item.get("text")
                        or item.get("summary")
                    )
                    if fact_val:
                        subj = (
                            item.get("subject")
                            or item.get("author")
                            or item.get("user")
                            or (f"User_{item['user_id']}" if "user_id" in item else "Group")
                        )
                        parsed.append(
                            {
                                "subject": str(subj).strip(),
                                "fact": str(fact_val).strip(),
                                "category": str(item.get("category") or "lore").strip(),
                            }
                        )
                elif isinstance(item, FactRecord):
                    parsed.append(
                        {
                            "subject": item.subject,
                            "fact": item.fact,
                            "category": item.category,
                        }
                    )
                elif isinstance(item, str) and item.strip():
                    parsed.extend(self._parse_text_summary_lines(item))
            return parsed

        if isinstance(raw_result, dict):
            if isinstance(raw_result.get("facts"), list):
                return self._parse_extracted_facts(raw_result["facts"])
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
                r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE
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

    def _parse_text_summary_lines(self, text: str) -> List[Dict[str, str]]:
        """Extract summary and participant facts from plain-text summary lines."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        results: List[Dict[str, str]] = [
            {
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
                    results.append(
                        {
                            "subject": author,
                            "fact": fact_body,
                            "category": _infer_fact_category(fact_body),
                        }
                    )
        return results
