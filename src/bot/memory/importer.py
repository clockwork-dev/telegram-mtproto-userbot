"""Zero-token Telegram Desktop JSON chat history importer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bot.memory.db import ChatMessage, MemoryStore

# Historical noisy bots in exports that should not pollute archive search results
DEFAULT_SKIP_USER_IDS: frozenset[int] = frozenset({7872766132, 7938711591, 8077207013})


@dataclass
class ImportStats:
    """Summary counters for a single imported Telegram JSON export."""

    file_path: str
    chat_id: int
    chat_name: str
    total_raw: int
    parsed: int
    inserted: int
    skipped_empty: int
    skipped_service_or_bot: int


def flatten_export_text(raw_text: Any) -> str:
    """Convert Telegram Desktop string or entity-list text into plain text."""
    if isinstance(raw_text, str):
        return raw_text.strip()
    if isinstance(raw_text, list):
        parts: List[str] = []
        for item in raw_text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def resolve_export_chat_id(data: Dict[str, Any], override_chat_id: Optional[int] = None) -> int:
    """Resolve Bot API compatible chat_id from Telegram Desktop export metadata."""
    if override_chat_id is not None:
        return int(override_chat_id)
    raw_id = int(data.get("id", 0))
    chat_type = str(data.get("type", "")).lower()
    if raw_id > 0 and ("supergroup" in chat_type or "channel" in chat_type):
        return int(f"-100{raw_id}")
    return raw_id


def parse_export_actor_id(raw_from_id: Any) -> int:
    """Extract numeric Telegram user_id/channel_id from Telegram Desktop from_id."""
    if raw_from_id is None:
        return 0
    s = str(raw_from_id).strip()
    if s.startswith("user"):
        s_num = s[4:]
        return int(s_num) if s_num.lstrip("-").isdigit() else 0
    if s.startswith("channel"):
        s_num = s[7:]
        return int(f"-100{s_num}") if s_num.isdigit() else 0
    match = re.search(r"-?\d+", s)
    return int(match.group(0)) if match else 0


def extract_export_media(msg: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Build lightweight media_type and media_summary from export item."""
    media_type = msg.get("media_type")
    dur = msg.get("duration_seconds")
    if media_type == "sticker":
        emoji = msg.get("sticker_emoji") or ""
        return "sticker", f"стикер {emoji}".strip()
    if media_type == "voice_message":
        label = f"голосовое ({dur}с)" if dur else "голосовое"
        return "voice_message", label
    if media_type == "video_message":
        label = f"кружочек ({dur}с)" if dur else "кружочек"
        return "video_message", label
    if media_type == "animation":
        return "animation", "GIF"
    if media_type == "video_file":
        return "video_file", "видео"
    if media_type == "audio_file":
        return "audio_file", "аудио"
    if msg.get("photo"):
        return "photo", "фото"
    if msg.get("file"):
        fname = msg.get("file_name") or "файл"
        return "document", str(fname)
    return None, None


def _parse_timestamp(msg: Dict[str, Any]) -> int:
    unix_raw = msg.get("date_unixtime")
    if unix_raw is not None:
        try:
            return int(unix_raw)
        except (TypeError, ValueError):
            pass
    date_str = msg.get("date")
    if isinstance(date_str, str) and date_str:
        try:
            return int(datetime.fromisoformat(date_str).timestamp())
        except ValueError:
            pass
    return 0


def parse_export_file(
    json_path: Path | str,
    author_map: Optional[Dict[int, str]] = None,
    override_chat_id: Optional[int] = None,
    skip_user_ids: Optional[Set[int]] = None,
) -> Tuple[ImportStats, List[ChatMessage]]:
    """Parse a Telegram Desktop result.json export into deduplicated ChatMessage rows."""
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    chat_id = resolve_export_chat_id(data, override_chat_id=override_chat_id)
    chat_name = str(data.get("name") or f"Chat_{chat_id}")
    raw_items = data.get("messages", [])
    authors = author_map or {}
    skipped_bots = set(DEFAULT_SKIP_USER_IDS if skip_user_ids is None else skip_user_ids)

    parsed_msgs: List[ChatMessage] = []
    skipped_empty = 0
    skipped_service = 0

    for item in raw_items:
        if item.get("type") != "message":
            skipped_service += 1
            continue

        msg_id = item.get("id")
        if msg_id is None:
            skipped_empty += 1
            continue

        user_id = parse_export_actor_id(item.get("from_id"))
        if user_id in skipped_bots:
            skipped_service += 1
            continue

        text = flatten_export_text(item.get("text"))
        media_type, media_summary = extract_export_media(item)

        # Skip pure empty rows or trivial 1-character non-media noise
        if not media_summary and len(text) < 2:
            skipped_empty += 1
            continue

        author_name = authors.get(user_id) or str(item.get("from") or f"User_{user_id}").strip()
        ts = _parse_timestamp(item)
        reply_to = item.get("reply_to_message_id")
        reply_id = int(reply_to) if isinstance(reply_to, int) else None

        parsed_msgs.append(
            ChatMessage(
                telegram_msg_id=int(msg_id),
                chat_id=chat_id,
                user_id=user_id,
                timestamp=ts,
                author_name=author_name,
                text=text or None,
                reply_to_msg_id=reply_id,
                media_type=media_type,
                media_summary=media_summary,
                compacted=1,
            )
        )

    stats = ImportStats(
        file_path=str(p),
        chat_id=chat_id,
        chat_name=chat_name,
        total_raw=len(raw_items),
        parsed=len(parsed_msgs),
        inserted=0,
        skipped_empty=skipped_empty,
        skipped_service_or_bot=skipped_service,
    )
    return stats, parsed_msgs


async def import_telegram_exports(
    memory_store: MemoryStore,
    json_paths: Iterable[Path | str],
    author_map: Optional[Dict[int, str]] = None,
    override_chat_id: Optional[int] = None,
    skip_user_ids: Optional[Set[int]] = None,
) -> List[ImportStats]:
    """Parse and bulk-insert one or more Telegram JSON exports into MemoryStore."""
    await memory_store.init_db()
    results: List[ImportStats] = []
    for path in json_paths:
        stats, msgs = parse_export_file(
            path,
            author_map=author_map,
            override_chat_id=override_chat_id,
            skip_user_ids=skip_user_ids,
        )
        inserted = await memory_store.bulk_save_archive_messages(msgs)
        stats.inserted = inserted
        results.append(stats)
    return results
