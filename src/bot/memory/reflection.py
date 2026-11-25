"""Periodic Memory Reflection, Consolidation, and Existing Database Normalization."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from bot.memory.compaction import (
    _DEFAULT_IGNORED_BOT_NAMES,
    _EMBEDDED_ID_RE,
    _PISTRUNKINYA_VARIANTS,
    _coerce_optional_int,
    canonicalize_subject,
    strip_fact_id_tag,
)
from bot.memory.fts import FactRecord

logger = logging.getLogger(__name__)

_ALLOWED_REFLECTION_CATEGORIES = frozenset({"observation", "friend_bio", "lore"})


_DEFAULT_PRESERVE_RECENT_FACTS = 10
_DEFAULT_MAX_BATCH_FACTS = 20
_DEFAULT_MIN_OLDER_CANDIDATES = 8
_DEFAULT_MAX_ARCHIVED_PER_BATCH = 15
_DEFAULT_MIN_ARCHIVED_PER_BATCH = 2


def build_reflection_prompt(
    subject: str,
    facts: Sequence[Any],
    target_summary_facts: int = 4,
) -> str:
    """Build a concise Russian prompt instructing non-destructive windowed consolidation of micro-facts."""
    target_max = max(2, int(target_summary_facts))
    fact_lines: List[str] = []
    for f in facts:
        fid = getattr(f, "id", None)
        cat = getattr(f, "category", "lore") or "lore"
        txt = str(getattr(f, "fact", "") or "").strip()
        ts = int(getattr(f, "updated_at", 0) or getattr(f, "created_at", 0) or 0)
        if fid is None or not txt:
            continue
        if ts > 0:
            date_label = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            meta = f"{cat}, {date_label}"
        else:
            meta = str(cat)
        fact_lines.append(f"- [ID:{fid}] ({meta}) {txt}")

    joined_facts = "\n".join(fact_lines)
    return (
        f"Проведи бережную консолидацию пачки старых микро-фактов об участнике / теме «{subject}».\n"
        f"Ниже дано окно старых записей (не более 20 штук). Объединяй ТОЛЬКО дублирующиеся, "
        f"пересекающиеся по смыслу или устаревшие промежуточные микро-факты из этого списка "
        f"в 2..{target_max} содержательных конкретных фактов категории `observation`.\n"
        "Строгие правила:\n"
        "1. СОХРАНЯЙ КОНКРЕТИКУ: имена, названия игр/технологий/сервисов, точные детали биографии, "
        "характерные цитаты и локальные мемы не размывай в общие фразы.\n"
        "2. НЕ ТРОГАЙ УНИКАЛЬНЫЕ ФАКТЫ: самостоятельные непохожие друг на друга факты и яркие мемы "
        "оставляй нетронутыми — НЕ включай их ID в `archived_ids` (они останутся активными сами по себе).\n"
        "3. В `archived_ids` перечисли ТОЛЬКО целочисленные ID из списка ниже (от 2 до максимум 15 штук), "
        "которые действительно объединены в новые `synthesized` записи. Если объединять нечего — верни "
        '`{"synthesized": [], "archived_ids": []}`.\n'
        "4. Не выдумывай фактов от себя и не вызывай никаких инструментов.\n"
        "Верни СТРОГО валидный JSON-объект без лишнего текста:\n"
        '{"synthesized": [{"fact": "...", "category": "observation"}], '
        '"archived_ids": [1, 2]}\n\n'
        f"Старые микро-факты по «{subject}»:\n{joined_facts}"
    )


def _parse_reflection_response(
    raw_output: Any,
    max_synthesized: int,
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Parse LLM reflection output into `(valid_synthesized, raw_archived_ids)`."""
    if raw_output is None:
        return [], []

    if hasattr(raw_output, "text") and not isinstance(raw_output, (dict, str)):
        raw_output = getattr(raw_output, "text")

    data: Any = None
    if isinstance(raw_output, dict):
        data = raw_output
    elif isinstance(raw_output, str):
        cleaned = raw_output.strip()
        if not cleaned:
            return [], []
        try:
            data = json.loads(cleaned)
        except Exception:
            fence_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                cleaned,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if fence_match:
                try:
                    data = json.loads(fence_match.group(1))
                except Exception:
                    data = None
            if data is None:
                obj_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
                if obj_match:
                    try:
                        data = json.loads(obj_match.group(1))
                    except Exception:
                        data = None

    if not isinstance(data, dict):
        return [], []

    raw_synth = data.get("synthesized")
    raw_archived = data.get("archived_ids")
    if not isinstance(raw_synth, list):
        return [], []

    valid_synthesized: List[Tuple[str, str]] = []
    seen_texts: Set[str] = set()
    cap = max(int(max_synthesized) * 2, 8)
    for item in raw_synth:
        if isinstance(item, dict):
            fact_text = str(
                item.get("fact") or item.get("fact_text") or item.get("text") or ""
            ).strip()
            raw_cat = str(item.get("category") or "observation").strip().lower()
        elif isinstance(item, str):
            fact_text = item.strip()
            raw_cat = "observation"
        else:
            continue

        clean_fact, _ = strip_fact_id_tag(fact_text)
        final_fact = (clean_fact or fact_text).strip()
        if not final_fact:
            continue
        norm_key = final_fact.lower()
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)
        category = raw_cat if raw_cat in _ALLOWED_REFLECTION_CATEGORIES else "observation"
        valid_synthesized.append((final_fact, category))
        if len(valid_synthesized) >= cap:
            break

    parsed_archived: List[int] = []
    if isinstance(raw_archived, list):
        seen_ids: Set[int] = set()
        for raw_id in raw_archived:
            aid = _coerce_optional_int(raw_id)
            if aid is not None and aid not in seen_ids:
                seen_ids.add(aid)
                parsed_archived.append(aid)

    return valid_synthesized, parsed_archived


class ReflectionEngine:
    """Stage 3 background engine for TTL archiving and high-signal subject consolidation."""

    def __init__(
        self,
        memory_store: Any,
        reflect_fn: Optional[Callable[..., Any]] = None,
        *,
        min_active_facts_per_subject: int = 15,
        max_subjects_per_cycle: int = 3,
        target_summary_facts: int = 4,
        subject_cooldown_hours: float = 72.0,
        agreement_default_ttl_days: int = 14,
        event_default_ttl_days: int = 30,
        embedder: Optional[Any] = None,
        friends_provider: Optional[Callable[[], Optional[Mapping[int, Any]]]] = None,
        bot_id: Optional[int] = None,
        bot_names: Optional[Sequence[str]] = None,
        preserve_recent_count: int = _DEFAULT_PRESERVE_RECENT_FACTS,
        max_batch_facts: int = _DEFAULT_MAX_BATCH_FACTS,
        min_older_candidates: int = _DEFAULT_MIN_OLDER_CANDIDATES,
        max_archived_per_batch: int = _DEFAULT_MAX_ARCHIVED_PER_BATCH,
        min_archived_per_batch: int = _DEFAULT_MIN_ARCHIVED_PER_BATCH,
    ) -> None:
        self.memory_store = memory_store
        self.reflect_fn = reflect_fn
        self.min_active_facts_per_subject = int(min_active_facts_per_subject)
        self.max_subjects_per_cycle = int(max_subjects_per_cycle)
        self.target_summary_facts = int(target_summary_facts)
        self.subject_cooldown_hours = float(subject_cooldown_hours)
        self.agreement_default_ttl_days = int(agreement_default_ttl_days)
        self.event_default_ttl_days = int(event_default_ttl_days)
        self.embedder = embedder
        self.friends_provider = friends_provider
        self.bot_id = _coerce_optional_int(bot_id)
        self.bot_names: List[str] = [str(n) for n in bot_names if n] if bot_names else []
        self.preserve_recent_count = max(0, int(preserve_recent_count))
        self.max_batch_facts = max(2, int(max_batch_facts))
        self.min_older_candidates = max(2, int(min_older_candidates))
        self.max_archived_per_batch = max(2, int(max_archived_per_batch))
        self.min_archived_per_batch = max(1, int(min_archived_per_batch))

    def update_config(self, config: Any) -> None:
        """Hot-reload reflection thresholds and TTL defaults from MemoryReflectionConfig."""
        if config is None:
            return
        for attr in (
            "min_active_facts_per_subject",
            "max_subjects_per_cycle",
            "target_summary_facts",
            "subject_cooldown_hours",
            "agreement_default_ttl_days",
            "event_default_ttl_days",
        ):
            if hasattr(config, attr):
                setattr(self, attr, getattr(config, attr))

    @staticmethod
    def build_reflection_prompt(
        subject: str,
        facts: Sequence[Any],
        target_summary_facts: int = 4,
    ) -> str:
        """Delegate to module-level build_reflection_prompt."""
        return build_reflection_prompt(
            subject=subject,
            facts=facts,
            target_summary_facts=target_summary_facts,
        )

    def _resolve_friends_map(self) -> Optional[Mapping[int, Any]]:
        if self.friends_provider is None:
            return None
        try:
            return self.friends_provider()
        except Exception as exc:
            logger.debug("Failed to resolve friends_map in ReflectionEngine: %s", type(exc).__name__)
            return None

    def _sync_load_active_facts_from_conn(self, now: int) -> List[FactRecord]:
        conn: sqlite3.Connection = self.memory_store._get_connection()
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(memory_facts)").fetchall()
        }
        has_lifecycle = {"subject_user_id", "status", "updated_at", "expires_at", "superseded_by"} <= cols
        if has_lifecycle:
            rows = conn.execute(
                """
                SELECT id, subject, fact, category, source_chat_id, created_at,
                       subject_user_id, status, updated_at, expires_at, superseded_by
                FROM memory_facts
                WHERE status = 'active'
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY updated_at DESC, id DESC
                """,
                (now,),
            ).fetchall()
            result: List[FactRecord] = []
            for r in rows:
                try:
                    rec = FactRecord(
                        id=int(r[0]),
                        subject=str(r[1]),
                        fact=str(r[2]),
                        category=str(r[3]),
                        source_chat_id=r[4],
                        created_at=int(r[5] or 0),
                        subject_user_id=r[6],
                        status=str(r[7] or "active"),
                        updated_at=int(r[8] or 0),
                        expires_at=r[9],
                        superseded_by=r[10],
                    )
                except TypeError:
                    rec = FactRecord(
                        id=int(r[0]),
                        subject=str(r[1]),
                        fact=str(r[2]),
                        category=str(r[3]),
                        source_chat_id=r[4],
                        created_at=int(r[5] or 0),
                    )
                result.append(rec)
            return result

        rows = conn.execute(
            """
            SELECT id, subject, fact, category, source_chat_id, created_at
            FROM memory_facts
            ORDER BY id DESC
            """
        ).fetchall()
        return [
            FactRecord(
                id=int(r[0]),
                subject=str(r[1]),
                fact=str(r[2]),
                category=str(r[3]),
                source_chat_id=r[4],
                created_at=int(r[5] or 0),
            )
            for r in rows
        ]

    async def _load_all_active_facts(self, now: int) -> List[FactRecord]:
        """Load all active non-expired facts via SQLite connection or get_active_facts fallback."""
        if hasattr(self.memory_store, "_get_connection") and callable(
            self.memory_store._get_connection
        ):
            lock = getattr(self.memory_store, "_lock", None)
            if isinstance(lock, asyncio.Lock):
                async with lock:
                    return await asyncio.to_thread(self._sync_load_active_facts_from_conn, now)
            return await asyncio.to_thread(self._sync_load_active_facts_from_conn, now)

        if hasattr(self.memory_store, "get_active_facts"):
            try:
                return list(await self.memory_store.get_active_facts(limit=5000, now_ts=now))
            except TypeError:
                return list(await self.memory_store.get_active_facts(limit=5000))
        return []

    async def _invoke_reflect_fn(
        self,
        *,
        subject: str,
        facts: List[FactRecord],
        chat_id: Optional[int],
    ) -> Any:
        assert self.reflect_fn is not None
        try:
            sig = inspect.signature(self.reflect_fn)
            params = list(sig.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            candidate_kwargs = {
                "subject": subject,
                "facts": facts,
                "chat_id": chat_id,
                "target_summary_facts": self.target_summary_facts,
            }
            if has_var_kw:
                res = self.reflect_fn(**candidate_kwargs)
            elif len(params) == 1 and params[0].name not in candidate_kwargs:
                prompt = build_reflection_prompt(
                    subject=subject,
                    facts=facts,
                    target_summary_facts=self.target_summary_facts,
                )
                res = self.reflect_fn(prompt)
            else:
                filtered_kwargs = {
                    k: v
                    for k, v in candidate_kwargs.items()
                    if k in sig.parameters
                    and sig.parameters[k].kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                res = self.reflect_fn(**filtered_kwargs)
        except (TypeError, ValueError):
            res = self.reflect_fn(
                subject=subject,
                facts=facts,
                chat_id=chat_id,
                target_summary_facts=self.target_summary_facts,
            )

        return await res if inspect.isawaitable(res) else res

    async def _add_fact_compat(
        self,
        *,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int],
        subject_user_id: Optional[int],
        created_at: int,
    ) -> int:
        try:
            res = await self.memory_store.add_fact(
                subject=subject,
                fact=fact,
                category=category,
                source_chat_id=source_chat_id,
                created_at=created_at,
                subject_user_id=subject_user_id,
                status="active",
                expires_at=None,
            )
        except TypeError:
            res = await self.memory_store.add_fact(
                subject=subject,
                fact=fact,
                category=category,
                source_chat_id=source_chat_id,
            )
        return int(res or 0)

    async def run_cycle(
        self,
        *,
        is_rate_limited_fn: Optional[Callable[[], bool]] = None,
        now_ts: Optional[int] = None,
    ) -> Dict[str, int]:
        """Run TTL expiration sweep and consolidate up to max_subjects_per_cycle dense fact buckets."""
        now = int(now_ts) if now_ts is not None else int(time.time())
        expired = 0
        if hasattr(self.memory_store, "archive_expired_facts"):
            expired = int(
                await self.memory_store.archive_expired_facts(
                    now_ts=now,
                    agreement_ttl_days=self.agreement_default_ttl_days,
                    event_ttl_days=self.event_default_ttl_days,
                )
                or 0
            )

        if self.reflect_fn is None or (is_rate_limited_fn is not None and bool(is_rate_limited_fn())):
            return {
                "expired_archived": expired,
                "subjects_reflected": 0,
                "facts_synthesized": 0,
                "facts_archived": 0,
            }

        friends_map = self._resolve_friends_map()
        active_facts = await self._load_all_active_facts(now)

        # Group active facts by (source_chat_id, canonical_uid, canonical_subject_lower)
        buckets: Dict[Tuple[Optional[int], Optional[int], str], List[FactRecord]] = {}
        bucket_display_subj: Dict[Tuple[Optional[int], Optional[int], str], str] = {}

        for fact in active_facts:
            raw_uid = _coerce_optional_int(getattr(fact, "subject_user_id", None))
            canon_subj, canon_uid = canonicalize_subject(
                fact.subject,
                user_id=raw_uid,
                friends_map=friends_map,
                bot_id=self.bot_id,
                bot_names=self.bot_names,
            )
            if canon_subj is None:
                continue
            subj_key = "" if canon_uid is not None else canon_subj.lower()
            key = (fact.source_chat_id, canon_uid, subj_key)
            buckets.setdefault(key, []).append(fact)
            bucket_display_subj.setdefault(key, canon_subj)

        # Build windowed oldest-first non-observation candidate batches while always protecting
        # the `preserve_recent_count` (default 10) newest active facts in every subject bucket.
        candidate_windows: Dict[Tuple[Optional[int], Optional[int], str], List[FactRecord]] = {}
        for key, items in buckets.items():
            if len(items) < self.min_active_facts_per_subject:
                continue
            newest_first = sorted(
                items,
                key=lambda f: (
                    max(int(getattr(f, "updated_at", 0) or 0), int(getattr(f, "created_at", 0) or 0)),
                    int(getattr(f, "id", 0) or 0),
                ),
                reverse=True,
            )
            older_pool = newest_first[self.preserve_recent_count :]
            older_raw_desc = [
                f
                for f in older_pool
                if str(getattr(f, "category", "") or "").strip().lower() != "observation"
            ]
            if len(older_raw_desc) < self.min_older_candidates:
                continue
            oldest_first = list(reversed(older_raw_desc))
            candidate_windows[key] = oldest_first[: self.max_batch_facts]

        eligible_keys = sorted(
            candidate_windows.keys(),
            key=lambda k: (len(buckets[k]), len(candidate_windows[k])),
            reverse=True,
        )

        cooldown_seconds = max(0.0, float(self.subject_cooldown_hours) * 3600.0)
        subjects_reflected = 0
        facts_synthesized = 0
        facts_archived = 0

        for key in eligible_keys:
            if subjects_reflected >= self.max_subjects_per_cycle:
                break
            if is_rate_limited_fn is not None and bool(is_rate_limited_fn()):
                break

            chat_id, canon_uid, subj_lower = key
            canon_subj = bucket_display_subj[key]
            batch_facts = candidate_windows[key]

            chat_label = str(chat_id) if chat_id is not None else "global"
            uid_or_subj = str(canon_uid) if canon_uid is not None else subj_lower
            cooldown_key = f"reflect_last:{chat_label}:{uid_or_subj}"

            if cooldown_seconds > 0.0 and hasattr(self.memory_store, "get_setting"):
                last_raw = await self.memory_store.get_setting(cooldown_key)
                if last_raw is not None:
                    try:
                        if (now - float(last_raw)) < cooldown_seconds:
                            continue
                    except (TypeError, ValueError):
                        pass

            try:
                raw_response = await self._invoke_reflect_fn(
                    subject=canon_subj,
                    facts=batch_facts,
                    chat_id=chat_id,
                )
            except Exception as exc:
                logger.warning(
                    "Reflection callback failed for chat=%s subject_uid=%s: %s",
                    chat_id,
                    canon_uid,
                    type(exc).__name__,
                )
                continue

            valid_synthesized, raw_archived_ids = _parse_reflection_response(
                raw_response,
                max_synthesized=self.target_summary_facts,
            )
            if len(valid_synthesized) < 1:
                logger.debug(
                    "Reflection produced no valid synthesized facts for chat=%s uid=%s; skipping archive.",
                    chat_id,
                    canon_uid,
                )
                continue

            batch_id_set: Set[int] = {
                int(f.id) for f in batch_facts if getattr(f, "id", None) is not None
            }
            safe_archived_ids = [aid for aid in raw_archived_ids if aid in batch_id_set]
            if len(safe_archived_ids) < self.min_archived_per_batch:
                logger.debug(
                    "Reflection archived_ids count (%d < %d) below anti-churn minimum for chat=%s uid=%s; skipping.",
                    len(safe_archived_ids),
                    self.min_archived_per_batch,
                    chat_id,
                    canon_uid,
                )
                continue
            if len(safe_archived_ids) > self.max_archived_per_batch:
                safe_archived_ids = safe_archived_ids[: self.max_archived_per_batch]

            inserted_facts: List[Tuple[int, str, str]] = []
            for synth_fact, synth_cat in valid_synthesized:
                new_id = await self._add_fact_compat(
                    subject=canon_subj,
                    fact=synth_fact,
                    category=synth_cat,
                    source_chat_id=chat_id,
                    subject_user_id=canon_uid,
                    created_at=now,
                )
                if new_id > 0:
                    inserted_facts.append((new_id, synth_fact, synth_cat))

            if not inserted_facts:
                continue

            first_new_id = inserted_facts[0][0]
            archived_in_bucket = 0
            if safe_archived_ids and hasattr(self.memory_store, "supersede_fact"):
                for old_id in safe_archived_ids:
                    try:
                        ok = await self.memory_store.supersede_fact(
                            old_id,
                            new_fact_id=first_new_id,
                            status="archived",
                        )
                    except TypeError:
                        ok = await self.memory_store.supersede_fact(
                            old_id,
                            new_fact_id=first_new_id,
                        )
                    if ok is not False:
                        archived_in_bucket += 1

            if self.embedder is not None:
                for new_id, synth_fact, synth_cat in inserted_facts:
                    try:
                        await self.embedder.index_fact(
                            self.memory_store,
                            new_id,
                            canon_subj,
                            synth_fact,
                            synth_cat,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Embedder index_fact failed during reflection for fact_id=%s: %s",
                            new_id,
                            type(exc).__name__,
                        )

            if hasattr(self.memory_store, "set_setting"):
                await self.memory_store.set_setting(cooldown_key, str(now))

            subjects_reflected += 1
            facts_synthesized += len(inserted_facts)
            facts_archived += archived_in_bucket

        return {
            "expired_archived": expired,
            "subjects_reflected": subjects_reflected,
            "facts_synthesized": facts_synthesized,
            "facts_archived": facts_archived,
        }


@dataclass
class _NormFactCandidate:
    id: int
    raw_subject: str
    raw_fact: str
    clean_fact: str
    trailing_fact_uid: Optional[int]
    source_chat_id: Optional[int]
    created_at: int
    updated_at: int
    orig_uid: Optional[int]
    canon_subject: str
    canon_uid: Optional[int]


def _sync_load_normalization_snapshot(
    conn: sqlite3.Connection,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, int]], Dict[str, int]]:
    """Read active memory_facts rows and unambiguous message author->user_id maps from SQLite."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "memory_facts" not in tables:
        return [], {}, {}

    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(memory_facts)").fetchall()
    }
    has_uid = "subject_user_id" in cols
    has_status = "status" in cols
    has_updated = "updated_at" in cols

    select_uid = "subject_user_id" if has_uid else "NULL AS subject_user_id"
    select_updated = "updated_at" if has_updated else "created_at AS updated_at"
    where_clause = "WHERE status = 'active'" if has_status else ""

    rows = conn.execute(
        f"""
        SELECT id, subject, fact, category, source_chat_id, created_at,
               {select_uid}, {select_updated}
        FROM memory_facts
        {where_clause}
        ORDER BY id ASC
        """
    ).fetchall()
    fact_rows = [
        {
            "id": int(r[0]),
            "subject": str(r[1] or ""),
            "fact": str(r[2] or ""),
            "category": str(r[3] or "lore"),
            "source_chat_id": r[4],
            "created_at": int(r[5] or 0),
            "subject_user_id": _coerce_optional_int(r[6]),
            "updated_at": int(r[7] or 0),
        }
        for r in rows
    ]

    per_chat_uids: Dict[Tuple[int, str], Set[int]] = {}
    global_uids: Dict[str, Set[int]] = {}
    if "messages" in tables:
        msg_rows = conn.execute(
            """
            SELECT DISTINCT chat_id, user_id, author_name
            FROM messages
            WHERE author_name IS NOT NULL AND TRIM(author_name) != ''
            """
        ).fetchall()
        for chat_id_raw, user_id_raw, author_raw in msg_rows:
            cid = _coerce_optional_int(chat_id_raw)
            uid = _coerce_optional_int(user_id_raw)
            name_lower = str(author_raw or "").strip().lower()
            if uid is None or not name_lower:
                continue
            global_uids.setdefault(name_lower, set()).add(uid)
            if cid is not None:
                per_chat_uids.setdefault((cid, name_lower), set()).add(uid)

    per_chat_map: Dict[int, Dict[str, int]] = {}
    for (cid, name_lower), uids in per_chat_uids.items():
        if len(uids) == 1:
            per_chat_map.setdefault(cid, {})[name_lower] = next(iter(uids))

    global_map: Dict[str, int] = {
        name_lower: next(iter(uids))
        for name_lower, uids in global_uids.items()
        if len(uids) == 1
    }
    return fact_rows, per_chat_map, global_map


def _sync_fallback_update_row(
    conn: sqlite3.Connection,
    fact_id: int,
    subject: str,
    subject_user_id: Optional[int],
    now_ts: int,
) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)").fetchall()}
    assignments = ["subject = ?"]
    params: List[Any] = [subject]
    if "subject_user_id" in cols:
        assignments.append("subject_user_id = ?")
        params.append(subject_user_id)
    if "updated_at" in cols:
        assignments.append("updated_at = ?")
        params.append(now_ts)
    params.append(fact_id)
    conn.execute(
        f"UPDATE memory_facts SET {', '.join(assignments)} WHERE id = ?",
        params,
    )
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "memory_fact_embeddings" in tables:
        conn.execute("DELETE FROM memory_fact_embeddings WHERE fact_id = ?", (fact_id,))
    conn.commit()


def _sync_fallback_supersede_row(
    conn: sqlite3.Connection,
    fact_id: int,
    new_fact_id: Optional[int],
    status: str,
    now_ts: int,
) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)").fetchall()}
    if "status" not in cols:
        return
    assignments = ["status = ?"]
    params: List[Any] = [status]
    if "superseded_by" in cols:
        assignments.append("superseded_by = ?")
        params.append(new_fact_id)
    if "updated_at" in cols:
        assignments.append("updated_at = ?")
        params.append(now_ts)
    params.append(fact_id)
    conn.execute(
        f"UPDATE memory_facts SET {', '.join(assignments)} WHERE id = ?",
        params,
    )
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "memory_fact_embeddings" in tables:
        conn.execute("DELETE FROM memory_fact_embeddings WHERE fact_id = ?", (fact_id,))
    conn.commit()


async def normalize_existing_facts_in_db(
    memory_store: Any,
    *,
    friends_map: Optional[Mapping[int, Any]] = None,
    bot_id: Optional[int] = None,
    bot_names: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Canonicalize existing active facts, archive self-bot facts, and supersede exact duplicates."""
    effective_bot_names = (
        list(bot_names or [])
        + sorted(_DEFAULT_IGNORED_BOT_NAMES)
        + sorted(_PISTRUNKINYA_VARIANTS)
    )
    norm_bot_id = _coerce_optional_int(bot_id)

    lock = getattr(memory_store, "_lock", None)
    if isinstance(lock, asyncio.Lock):
        async with lock:
            conn = memory_store._get_connection()
            fact_rows, per_chat_map, global_map = await asyncio.to_thread(
                _sync_load_normalization_snapshot,
                conn,
            )
    else:
        conn = memory_store._get_connection()
        fact_rows, per_chat_map, global_map = await asyncio.to_thread(
            _sync_load_normalization_snapshot,
            conn,
        )

    now_ts = int(time.time())
    scanned = 0
    updated = 0
    bot_archived = 0
    duplicates_superseded = 0

    bot_row_ids: List[int] = []
    dedup_groups: Dict[Tuple[Optional[int], str, str], List[_NormFactCandidate]] = {}

    for row in fact_rows:
        scanned += 1
        row_id = row["id"]
        raw_subject = row["subject"]
        raw_fact = row["fact"]
        source_chat_id = _coerce_optional_int(row["source_chat_id"])
        orig_uid = row["subject_user_id"]

        clean_fact, trailing_uid = strip_fact_id_tag(raw_fact)
        inferred_uid = orig_uid
        if inferred_uid is None:
            if trailing_uid is not None:
                inferred_uid = trailing_uid
            else:
                emb_match = _EMBEDDED_ID_RE.search(raw_fact)
                if emb_match:
                    inferred_uid = _coerce_optional_int(emb_match.group("uid"))

        author_map = dict(global_map)
        if source_chat_id is not None and source_chat_id in per_chat_map:
            author_map.update(per_chat_map[source_chat_id])
            # Prevent cross-chat name ambiguity if a name was ambiguous in global_map
            # while absent in this chat; global_map already contains only globally single-UID names.

        canon_subj, canon_uid = canonicalize_subject(
            raw_subject,
            user_id=inferred_uid,
            friends_map=friends_map,
            author_id_by_name=author_map,
            bot_id=norm_bot_id,
            bot_names=effective_bot_names,
        )

        if canon_subj is None:
            bot_row_ids.append(row_id)
            continue

        norm_fact = (clean_fact or raw_fact.strip()).lower()
        dedup_key = (source_chat_id, canon_subj.lower(), norm_fact)
        dedup_groups.setdefault(dedup_key, []).append(
            _NormFactCandidate(
                id=row_id,
                raw_subject=raw_subject,
                raw_fact=raw_fact,
                clean_fact=clean_fact or raw_fact.strip(),
                trailing_fact_uid=trailing_uid,
                source_chat_id=source_chat_id,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                orig_uid=orig_uid,
                canon_subject=canon_subj,
                canon_uid=canon_uid,
            )
        )

    has_supersede = hasattr(memory_store, "supersede_fact")
    has_update = hasattr(memory_store, "update_fact")

    # 1. Archive self-bot facts
    for row_id in bot_row_ids:
        if has_supersede:
            await memory_store.supersede_fact(row_id, new_fact_id=None, status="archived")
        elif isinstance(lock, asyncio.Lock):
            async with lock:
                await asyncio.to_thread(
                    _sync_fallback_supersede_row,
                    memory_store._get_connection(),
                    row_id,
                    None,
                    "archived",
                    now_ts,
                )
        else:
            await asyncio.to_thread(
                _sync_fallback_supersede_row,
                memory_store._get_connection(),
                row_id,
                None,
                "archived",
                now_ts,
            )
        bot_archived += 1

    # 2. Deduplicate exact active facts and canonicalize kept rows
    for group in dedup_groups.values():
        group_sorted = sorted(
            group,
            key=lambda c: (
                1 if c.trailing_fact_uid is None else 0,
                max(c.updated_at, c.created_at),
                1 if (c.raw_subject == c.canon_subject and c.orig_uid == c.canon_uid) else 0,
                c.id,
            ),
            reverse=True,
        )
        kept = group_sorted[0]
        duplicates = group_sorted[1:]

        if kept.raw_subject != kept.canon_subject or kept.orig_uid != kept.canon_uid:
            if has_update:
                await memory_store.update_fact(
                    kept.id,
                    subject=kept.canon_subject,
                    subject_user_id=kept.canon_uid,
                )
            elif isinstance(lock, asyncio.Lock):
                async with lock:
                    await asyncio.to_thread(
                        _sync_fallback_update_row,
                        memory_store._get_connection(),
                        kept.id,
                        kept.canon_subject,
                        kept.canon_uid,
                        now_ts,
                    )
            else:
                await asyncio.to_thread(
                    _sync_fallback_update_row,
                    memory_store._get_connection(),
                    kept.id,
                    kept.canon_subject,
                    kept.canon_uid,
                    now_ts,
                )
            updated += 1

        for dup in duplicates:
            if has_supersede:
                await memory_store.supersede_fact(
                    dup.id,
                    new_fact_id=kept.id,
                    status="superseded",
                )
            elif isinstance(lock, asyncio.Lock):
                async with lock:
                    await asyncio.to_thread(
                        _sync_fallback_supersede_row,
                        memory_store._get_connection(),
                        dup.id,
                        kept.id,
                        "superseded",
                        now_ts,
                    )
            else:
                await asyncio.to_thread(
                    _sync_fallback_supersede_row,
                    memory_store._get_connection(),
                    dup.id,
                    kept.id,
                    "superseded",
                    now_ts,
                )
            duplicates_superseded += 1

    return {
        "scanned": scanned,
        "updated": updated,
        "bot_archived": bot_archived,
        "duplicates_superseded": duplicates_superseded,
    }
