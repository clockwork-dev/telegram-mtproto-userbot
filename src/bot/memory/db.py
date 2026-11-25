"""SQLite Message Store and Integrated MemoryStore."""

import asyncio
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from bot.memory.fts import _UNSET, FactRecord, FTSStore, build_archive_fts_query


@dataclass
class ChatMessage:
    telegram_msg_id: int
    chat_id: int
    user_id: int
    timestamp: int
    author_name: str
    text: Optional[str] = None
    reply_to_msg_id: Optional[int] = None
    media_type: Optional[str] = None
    media_summary: Optional[str] = None
    compacted: int = 0
    id: Optional[int] = None


@dataclass
class SelfEditRecord:
    """Persisted record of an autonomous or admin-requested self-edit attempt."""

    trigger_type: str
    signature: str
    reason: str
    affected_files: List[str] = field(default_factory=list)
    verdict: str = "NO_CHANGE"
    model: str = ""
    branch: str = ""
    commit_sha: str = ""
    diff_lines: int = 0
    test_result: str = "skipped"
    deploy_result: str = "skipped"
    rollback_commit: str = ""
    summary: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: Optional[int] = None


class MemoryStore:
    """Asynchronous Memory Store managing short-term message history and long-term FTS facts."""

    _USAGE_RETENTION_DAYS = 90
    _USAGE_PRUNE_EVERY_N_WRITES = 200

    def __init__(self, db_path: str = "data/memory.sqlite") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._usage_writes_since_prune: int = 0

    def _get_connection(self) -> sqlite3.Connection:
        with self._sync_lock:
            if self._conn is None:
                if self.db_path != ":memory:":
                    p = Path(self.db_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=5.0,
                )
                self._conn.row_factory = sqlite3.Row
                try:
                    self._conn.execute("PRAGMA busy_timeout = 5000")
                    if self.db_path != ":memory:":
                        self._conn.execute("PRAGMA journal_mode = WAL")
                        self._conn.execute("PRAGMA synchronous = NORMAL")
                except sqlite3.OperationalError:
                    pass
            return self._conn

    def _sync_init_db(self) -> None:
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_msg_id INTEGER NOT NULL,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                author_name TEXT NOT NULL,
                text TEXT,
                reply_to_msg_id INTEGER,
                media_type TEXT,
                media_summary TEXT,
                compacted INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_time ON messages(chat_id, timestamp DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_chat_msg ON messages(chat_id, telegram_msg_id);

            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                source TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER NOT NULL,
                cache_write_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_history_ts ON usage_history(timestamp DESC);

            CREATE TABLE IF NOT EXISTS self_edit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                signature TEXT NOT NULL,
                reason TEXT NOT NULL,
                affected_files TEXT NOT NULL,
                verdict TEXT NOT NULL,
                model TEXT NOT NULL,
                branch TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                diff_lines INTEGER NOT NULL,
                test_result TEXT NOT NULL,
                deploy_result TEXT NOT NULL,
                rollback_commit TEXT NOT NULL,
                summary TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_self_edit_created ON self_edit_history(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_self_edit_sig ON self_edit_history(signature, created_at DESC);

            CREATE TABLE IF NOT EXISTS bot_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.commit()
        FTSStore.init_schema(conn)
        FTSStore.init_messages_fts(conn)

    async def init_db(self) -> None:
        """Initialize database schema asynchronously."""
        async with self._lock:
            await asyncio.to_thread(self._sync_init_db)

    def _sync_save_message(self, msg: ChatMessage) -> None:
        with self._sync_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO messages (
                    telegram_msg_id, chat_id, user_id, timestamp, author_name,
                    text, reply_to_msg_id, media_type, media_summary, compacted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, telegram_msg_id) DO UPDATE SET
                    author_name = excluded.author_name,
                    text = excluded.text,
                    reply_to_msg_id = excluded.reply_to_msg_id,
                    media_type = excluded.media_type,
                    media_summary = excluded.media_summary,
                    compacted = MAX(messages.compacted, excluded.compacted)
                """,
                (
                    msg.telegram_msg_id,
                    msg.chat_id,
                    msg.user_id,
                    msg.timestamp,
                    msg.author_name,
                    msg.text,
                    msg.reply_to_msg_id,
                    msg.media_type,
                    msg.media_summary,
                    msg.compacted,
                ),
            )
            conn.commit()

    async def save_message(self, msg: ChatMessage) -> None:
        """Store or update a message in rolling short-term history."""
        async with self._lock:
            await asyncio.to_thread(self._sync_save_message, msg)

    def _sync_bulk_save_archive_messages(self, messages: List[ChatMessage]) -> int:
        if not messages:
            return 0
        conn = self._get_connection()
        before = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        rows = [
            (
                msg.telegram_msg_id,
                msg.chat_id,
                msg.user_id,
                msg.timestamp,
                msg.author_name,
                msg.text,
                msg.reply_to_msg_id,
                msg.media_type,
                msg.media_summary,
                1,
            )
            for msg in messages
        ]
        with conn:
            conn.executemany(
                """
                INSERT INTO messages (
                    telegram_msg_id, chat_id, user_id, timestamp, author_name,
                    text, reply_to_msg_id, media_type, media_summary, compacted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, telegram_msg_id) DO NOTHING
                """,
                rows,
            )
        after = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return after - before

    async def bulk_save_archive_messages(self, messages: List[ChatMessage]) -> int:
        """Bulk-insert historical archive messages without overwriting live messages."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_bulk_save_archive_messages, messages)

    def _run_messages_fts(
        self,
        conn: sqlite3.Connection,
        fts_query: str,
        min_length: int,
        fetch_limit: int,
        excluded_ids: Set[int],
        excluded_users: Set[int],
        seen_ids: Set[int],
        results: List[ChatMessage],
        limit: int,
    ) -> None:
        if not fts_query or len(results) >= limit:
            return
        try:
            cursor = conn.execute(
                """
                SELECT msg.id, msg.telegram_msg_id, msg.chat_id, msg.user_id, msg.timestamp,
                       msg.author_name, msg.text, msg.reply_to_msg_id, msg.media_type,
                       msg.media_summary, msg.compacted
                FROM messages_fts mf
                JOIN messages msg ON msg.id = mf.rowid
                WHERE messages_fts MATCH ?
                  AND LENGTH(COALESCE(msg.text, '')) >= ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, min_length, fetch_limit),
            )
        except sqlite3.OperationalError:
            return
        for row in cursor.fetchall():
            rid = row["id"]
            if rid in excluded_ids or rid in seen_ids or row["user_id"] in excluded_users:
                continue
            seen_ids.add(rid)
            results.append(
                ChatMessage(
                    id=rid,
                    telegram_msg_id=row["telegram_msg_id"],
                    chat_id=row["chat_id"],
                    user_id=row["user_id"],
                    timestamp=row["timestamp"],
                    author_name=row["author_name"],
                    text=row["text"],
                    reply_to_msg_id=row["reply_to_msg_id"],
                    media_type=row["media_type"],
                    media_summary=row["media_summary"],
                    compacted=row["compacted"],
                )
            )
            if len(results) >= limit:
                break

    def _sync_search_messages(
        self,
        query: str,
        limit: int = 4,
        min_length: int = 18,
        exclude_db_ids: Optional[Set[int]] = None,
        exclude_user_ids: Optional[Set[int]] = None,
    ) -> List[ChatMessage]:
        and_query = build_archive_fts_query(query, max_terms=3, operator="AND")
        or_query = build_archive_fts_query(query, max_terms=5, operator="OR")
        if not or_query:
            return []
        conn = self._get_connection()
        fetch_limit = max(limit * 4, 16)
        excluded_ids = exclude_db_ids or set()
        excluded_users = exclude_user_ids or set()
        seen_ids: Set[int] = set()
        results: List[ChatMessage] = []

        if and_query and and_query != or_query:
            self._run_messages_fts(
                conn, and_query, min_length, fetch_limit,
                excluded_ids, excluded_users, seen_ids, results, limit,
            )
        if len(results) < limit:
            self._run_messages_fts(
                conn, or_query, min_length, fetch_limit,
                excluded_ids, excluded_users, seen_ids, results, limit,
            )
        return results

    async def search_messages(
        self,
        query: str,
        limit: int = 4,
        min_length: int = 18,
        exclude_db_ids: Optional[Set[int]] = None,
        exclude_user_ids: Optional[Set[int]] = None,
    ) -> List[ChatMessage]:
        """Search historical chat messages using selective Russian/Ukrainian FTS5."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_search_messages,
                query,
                limit,
                min_length,
                exclude_db_ids,
                exclude_user_ids,
            )

    def _sync_get_recent_history(self, chat_id: int, limit: int = 30) -> List[ChatMessage]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT * FROM (
                SELECT id, telegram_msg_id, chat_id, user_id, timestamp, author_name,
                       text, reply_to_msg_id, media_type, media_summary, compacted
                FROM messages
                WHERE chat_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            ) ORDER BY timestamp ASC, id ASC
            """,
            (chat_id, limit),
        )
        rows = cursor.fetchall()
        return [
            ChatMessage(
                id=row["id"],
                telegram_msg_id=row["telegram_msg_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                author_name=row["author_name"],
                text=row["text"],
                reply_to_msg_id=row["reply_to_msg_id"],
                media_type=row["media_type"],
                media_summary=row["media_summary"],
                compacted=row["compacted"],
            )
            for row in rows
        ]

    async def get_recent_history(self, chat_id: int, limit: int = 30) -> List[ChatMessage]:
        """Retrieve recent chat history in chronological order."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_recent_history, chat_id, limit)

    def _sync_get_uncompacted_messages(self, chat_id: int, limit: int = 30) -> List[ChatMessage]:
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT id, telegram_msg_id, chat_id, user_id, timestamp, author_name,
                   text, reply_to_msg_id, media_type, media_summary, compacted
            FROM messages
            WHERE chat_id = ? AND compacted = 0
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = cursor.fetchall()
        return [
            ChatMessage(
                id=row["id"],
                telegram_msg_id=row["telegram_msg_id"],
                chat_id=row["chat_id"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
                author_name=row["author_name"],
                text=row["text"],
                reply_to_msg_id=row["reply_to_msg_id"],
                media_type=row["media_type"],
                media_summary=row["media_summary"],
                compacted=row["compacted"],
            )
            for row in rows
        ]

    async def get_uncompacted_messages(self, chat_id: int, limit: int = 30) -> List[ChatMessage]:
        """Fetch oldest uncompacted messages for summarization."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_uncompacted_messages, chat_id, limit)

    def _sync_mark_compacted(self, message_ids: List[int]) -> None:
        if not message_ids:
            return
        conn = self._get_connection()
        placeholders = ",".join("?" for _ in message_ids)
        conn.execute(
            f"UPDATE messages SET compacted = 1 WHERE id IN ({placeholders})",
            message_ids,
        )
        conn.commit()

    async def mark_compacted(self, message_ids: List[int]) -> None:
        """Mark specific messages by DB id as compacted."""
        async with self._lock:
            await asyncio.to_thread(self._sync_mark_compacted, message_ids)

    def _sync_search_facts(
        self,
        query: str,
        limit: int = 5,
        exclude_fact_ids: Optional[Sequence[int]] = None,
        *,
        chat_id: Optional[int] = None,
        author_user_id: Optional[int] = None,
        target_user_ids: Optional[Sequence[int]] = None,
        query_vector: Optional[Any] = None,
        embedding_model: Optional[str] = None,
        min_vector_similarity: float = 0.52,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.search_facts(
                conn,
                query,
                limit,
                exclude_fact_ids,
                chat_id=chat_id,
                author_user_id=author_user_id,
                target_user_ids=target_user_ids,
                query_vector=query_vector,
                embedding_model=embedding_model,
                min_vector_similarity=min_vector_similarity,
                now_ts=now_ts,
            )

    async def search_facts(
        self,
        query: str,
        limit: int = 5,
        exclude_fact_ids: Optional[Sequence[int]] = None,
        *,
        chat_id: Optional[int] = None,
        author_user_id: Optional[int] = None,
        target_user_ids: Optional[Sequence[int]] = None,
        query_vector: Optional[Any] = None,
        embedding_model: Optional[str] = None,
        min_vector_similarity: float = 0.52,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        """Search active long-term facts via chat-scoped hybrid FTS5 + vector RRF."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_search_facts,
                query,
                limit,
                exclude_fact_ids,
                chat_id=chat_id,
                author_user_id=author_user_id,
                target_user_ids=target_user_ids,
                query_vector=query_vector,
                embedding_model=embedding_model,
                min_vector_similarity=min_vector_similarity,
                now_ts=now_ts,
            )

    def _sync_add_fact(
        self,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int] = None,
        created_at: Optional[int] = None,
        subject_user_id: Optional[int] = None,
        status: str = "active",
        expires_at: Optional[int] = None,
    ) -> int:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.add_fact(
                conn,
                subject,
                fact,
                category,
                source_chat_id,
                created_at,
                subject_user_id,
                status,
                expires_at,
            )

    async def add_fact(
        self,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int] = None,
        created_at: Optional[int] = None,
        subject_user_id: Optional[int] = None,
        status: str = "active",
        expires_at: Optional[int] = None,
    ) -> int:
        """Add a long-term fact to SQLite and FTS index."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_add_fact,
                subject,
                fact,
                category,
                source_chat_id,
                created_at,
                subject_user_id,
                status,
                expires_at,
            )

    def _sync_update_fact(
        self,
        fact_id: int,
        *,
        fact: Optional[str] = None,
        subject: Optional[str] = None,
        subject_user_id: Any = _UNSET,
        category: Optional[str] = None,
        expires_at: Any = _UNSET,
        status: Optional[str] = None,
        updated_at: Optional[int] = None,
    ) -> bool:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.update_fact(
                conn,
                fact_id,
                fact=fact,
                subject=subject,
                subject_user_id=subject_user_id,
                category=category,
                expires_at=expires_at,
                status=status,
                updated_at=updated_at,
            )

    async def update_fact(
        self,
        fact_id: int,
        *,
        fact: Optional[str] = None,
        subject: Optional[str] = None,
        subject_user_id: Any = _UNSET,
        category: Optional[str] = None,
        expires_at: Any = _UNSET,
        status: Optional[str] = None,
        updated_at: Optional[int] = None,
    ) -> bool:
        """Update an existing fact in place and invalidate stale embeddings if text changed."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_update_fact,
                fact_id,
                fact=fact,
                subject=subject,
                subject_user_id=subject_user_id,
                category=category,
                expires_at=expires_at,
                status=status,
                updated_at=updated_at,
            )

    def _sync_supersede_fact(
        self,
        fact_id: int,
        *,
        new_fact_id: Optional[int] = None,
        status: str = "superseded",
        updated_at: Optional[int] = None,
    ) -> bool:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.supersede_fact(
                conn,
                fact_id,
                new_fact_id=new_fact_id,
                status=status,
                updated_at=updated_at,
            )

    async def supersede_fact(
        self,
        fact_id: int,
        *,
        new_fact_id: Optional[int] = None,
        status: str = "superseded",
        updated_at: Optional[int] = None,
    ) -> bool:
        """Mark a fact as superseded or archived and remove its embedding."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_supersede_fact,
                fact_id,
                new_fact_id=new_fact_id,
                status=status,
                updated_at=updated_at,
            )

    def _sync_archive_expired_facts(
        self,
        *,
        now_ts: Optional[int] = None,
        agreement_ttl_days: Optional[int] = None,
        event_ttl_days: Optional[int] = None,
        chat_id: Optional[int] = None,
    ) -> int:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.archive_expired_facts(
                conn,
                now_ts=now_ts,
                agreement_ttl_days=agreement_ttl_days,
                event_ttl_days=event_ttl_days,
                chat_id=chat_id,
            )

    async def archive_expired_facts(
        self,
        *,
        now_ts: Optional[int] = None,
        agreement_ttl_days: Optional[int] = None,
        event_ttl_days: Optional[int] = None,
        chat_id: Optional[int] = None,
    ) -> int:
        """Archive active facts past explicit expires_at or category TTL."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_archive_expired_facts,
                now_ts=now_ts,
                agreement_ttl_days=agreement_ttl_days,
                event_ttl_days=event_ttl_days,
                chat_id=chat_id,
            )

    def _sync_get_active_facts(
        self,
        *,
        chat_id: Optional[int] = None,
        subject_user_ids: Optional[Sequence[int]] = None,
        subjects: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        include_global: bool = False,
        limit: int = 50,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.get_active_facts(
                conn,
                chat_id=chat_id,
                subject_user_ids=subject_user_ids,
                subjects=subjects,
                categories=categories,
                include_global=include_global,
                limit=limit,
                now_ts=now_ts,
            )

    async def get_active_facts(
        self,
        *,
        chat_id: Optional[int] = None,
        subject_user_ids: Optional[Sequence[int]] = None,
        subjects: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        include_global: bool = False,
        limit: int = 50,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        """Fetch active, non-expired facts filtered by chat scope, subjects, or categories."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_active_facts,
                chat_id=chat_id,
                subject_user_ids=subject_user_ids,
                subjects=subjects,
                categories=categories,
                include_global=include_global,
                limit=limit,
                now_ts=now_ts,
            )

    def _sync_get_facts_by_ids(self, fact_ids: Sequence[int]) -> Dict[int, FactRecord]:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.get_facts_by_ids(conn, fact_ids)

    async def get_facts_by_ids(self, fact_ids: Sequence[int]) -> Dict[int, FactRecord]:
        """Fetch facts keyed by ID regardless of lifecycle status."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_facts_by_ids, fact_ids)

    def _sync_upsert_fact_embedding(
        self,
        fact_id: int,
        model: str,
        dim: int,
        vector_bytes: bytes,
        updated_at: Optional[int] = None,
    ) -> None:
        with self._sync_lock:
            conn = self._get_connection()
            FTSStore.upsert_fact_embedding(conn, fact_id, model, dim, vector_bytes, updated_at)

    async def upsert_fact_embedding(
        self,
        fact_id: int,
        model: str,
        dim: int,
        vector_bytes: bytes,
        updated_at: Optional[int] = None,
    ) -> None:
        """Persist or update a float32 embedding vector for an active fact."""
        async with self._lock:
            await asyncio.to_thread(
                self._sync_upsert_fact_embedding,
                fact_id,
                model,
                dim,
                vector_bytes,
                updated_at,
            )

    def _sync_delete_fact_embedding(self, fact_id: int) -> None:
        with self._sync_lock:
            conn = self._get_connection()
            FTSStore.delete_fact_embedding(conn, fact_id)

    async def delete_fact_embedding(self, fact_id: int) -> None:
        """Delete stored embedding row for fact_id."""
        async with self._lock:
            await asyncio.to_thread(self._sync_delete_fact_embedding, fact_id)

    def _sync_get_unembedded_active_facts(
        self,
        model: str,
        limit: int = 32,
        chat_id: Optional[int] = None,
        *,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        with self._sync_lock:
            conn = self._get_connection()
            return FTSStore.get_unembedded_active_facts(
                conn,
                model,
                limit,
                chat_id,
                now_ts=now_ts,
            )

    async def get_unembedded_active_facts(
        self,
        model: str,
        limit: int = 32,
        chat_id: Optional[int] = None,
        *,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        """Fetch active non-expired facts lacking an embedding row for `model`."""
        async with self._lock:
            return await asyncio.to_thread(
                self._sync_get_unembedded_active_facts,
                model,
                limit,
                chat_id,
                now_ts=now_ts,
            )

    def _sync_delete_fact(self, fact_id: int) -> None:
        with self._sync_lock:
            conn = self._get_connection()
            FTSStore.delete_fact(conn, fact_id)

    async def delete_fact(self, fact_id: int) -> None:
        """Delete fact from long-term memory."""
        async with self._lock:
            await asyncio.to_thread(self._sync_delete_fact, fact_id)

    def _sync_count_uncompacted(self, chat_id: int) -> int:
        with self._sync_lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND compacted = 0",
                (chat_id,),
            ).fetchone()
            return int(row[0]) if row else 0

    async def compact_old_messages(
        self,
        chat_id: int,
        summarize_fn: Callable[[List[ChatMessage]], Any],
        threshold: int = 60,
        batch_size: int = 30,
    ) -> int:
        """Compact messages when uncompacted count reaches threshold."""
        # Check count of uncompacted messages off the event loop
        async with self._lock:
            count = await asyncio.to_thread(self._sync_count_uncompacted, chat_id)

        if count < threshold:
            return 0

        uncompacted = await self.get_uncompacted_messages(chat_id, limit=batch_size)
        if not uncompacted:
            return 0

        # Invoke summarize_fn (support both async and sync callbacks)
        if asyncio.iscoroutinefunction(summarize_fn):
            facts = await summarize_fn(uncompacted)
        else:
            facts = summarize_fn(uncompacted)

        if facts:
            for item in facts:
                if isinstance(item, dict):
                    await self.add_fact(
                        subject=item.get("subject", "General"),
                        fact=item.get("fact", ""),
                        category=item.get("category", "lore"),
                        source_chat_id=chat_id,
                        subject_user_id=item.get("subject_user_id"),
                        expires_at=item.get("expires_at"),
                    )
                elif isinstance(item, FactRecord):
                    await self.add_fact(
                        subject=item.subject,
                        fact=item.fact,
                        category=item.category,
                        source_chat_id=chat_id,
                        subject_user_id=item.subject_user_id,
                        expires_at=item.expires_at,
                    )

        msg_ids = [m.id for m in uncompacted if m.id is not None]
        await self.mark_compacted(msg_ids)
        return len(msg_ids)

    def _sync_prune_usage_history(self, retention_days: int = _USAGE_RETENTION_DAYS) -> int:
        if retention_days <= 0:
            return 0
        cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=int(retention_days))).isoformat()
        with self._sync_lock:
            conn = self._get_connection()
            cur = conn.execute(
                "DELETE FROM usage_history WHERE timestamp < ?",
                (cutoff_iso,),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    async def prune_usage_history(self, retention_days: int = _USAGE_RETENTION_DAYS) -> int:
        """Delete usage_history rows older than `retention_days`."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_prune_usage_history, retention_days)

    def _sync_record_usage(
        self,
        *,
        timestamp: str,
        model: str,
        source: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        total_tokens: int,
    ) -> None:
        with self._sync_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO usage_history (
                    timestamp, model, source, input_tokens, output_tokens,
                    reasoning_tokens, cache_read_tokens, cache_write_tokens, total_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    model,
                    source,
                    int(input_tokens),
                    int(output_tokens),
                    int(reasoning_tokens),
                    int(cache_read_tokens),
                    int(cache_write_tokens),
                    int(total_tokens),
                ),
            )
            self._usage_writes_since_prune += 1
            if self._usage_writes_since_prune >= self._USAGE_PRUNE_EVERY_N_WRITES:
                self._usage_writes_since_prune = 0
                cutoff_iso = (
                    datetime.now(timezone.utc) - timedelta(days=self._USAGE_RETENTION_DAYS)
                ).isoformat()
                conn.execute("DELETE FROM usage_history WHERE timestamp < ?", (cutoff_iso,))
            conn.commit()

    async def record_usage(
        self,
        *,
        timestamp: str,
        model: str,
        source: str,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        total_tokens: int,
    ) -> None:
        """Persist a token usage row into usage_history."""
        async with self._lock:
            await asyncio.to_thread(
                self._sync_record_usage,
                timestamp=timestamp,
                model=model,
                source=source,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                total_tokens=total_tokens,
            )

    def _sync_get_usage_rows(self, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._sync_lock:
            conn = self._get_connection()
            if since_iso:
                rows = conn.execute(
                    "SELECT id, timestamp, model, source, input_tokens, output_tokens, "
                    "reasoning_tokens, cache_read_tokens, cache_write_tokens, total_tokens "
                    "FROM usage_history WHERE timestamp >= ? ORDER BY id ASC",
                    (since_iso,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, timestamp, model, source, input_tokens, output_tokens, "
                    "reasoning_tokens, cache_read_tokens, cache_write_tokens, total_tokens "
                    "FROM usage_history ORDER BY id ASC"
                ).fetchall()
            return [dict(r) for r in rows]

    async def get_usage_rows(self, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch usage_history rows, optionally filtered by ISO timestamp >= since_iso."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_usage_rows, since_iso)

    def _sync_record_self_edit(self, record: SelfEditRecord) -> int:
        conn = self._get_connection()
        created_at = record.created_at or datetime.now(timezone.utc).isoformat()
        files_json = json.dumps(list(record.affected_files or []), ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO self_edit_history (
                created_at, trigger_type, signature, reason, affected_files,
                verdict, model, branch, commit_sha, diff_lines,
                test_result, deploy_result, rollback_commit, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                record.trigger_type,
                record.signature,
                record.reason,
                files_json,
                record.verdict,
                record.model,
                record.branch,
                record.commit_sha,
                int(record.diff_lines),
                record.test_result,
                record.deploy_result,
                record.rollback_commit,
                record.summary,
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)

    async def record_self_edit(self, record: SelfEditRecord) -> int:
        """Persist a SelfEditRecord into self_edit_history."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_record_self_edit, record)

    def _sync_get_recent_self_edits(self, limit: int = 20) -> List[SelfEditRecord]:
        with self._sync_lock:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM self_edit_history ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out: List[SelfEditRecord] = []
            for r in rows:
                try:
                    files = json.loads(r["affected_files"] or "[]")
                    if not isinstance(files, list):
                        files = []
                except Exception:
                    files = []
                out.append(
                    SelfEditRecord(
                        id=r["id"],
                        created_at=r["created_at"],
                        trigger_type=r["trigger_type"],
                        signature=r["signature"],
                        reason=r["reason"],
                        affected_files=[str(x) for x in files],
                        verdict=r["verdict"],
                        model=r["model"],
                        branch=r["branch"],
                        commit_sha=r["commit_sha"],
                        diff_lines=int(r["diff_lines"]),
                        test_result=r["test_result"],
                        deploy_result=r["deploy_result"],
                        rollback_commit=r["rollback_commit"],
                        summary=r["summary"],
                    )
                )
            return out

    async def get_recent_self_edits(self, limit: int = 20) -> List[SelfEditRecord]:
        """Return recent SelfEditRecord entries ordered newest-first."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_recent_self_edits, limit)

    def _sync_ensure_settings_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _sync_get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT value FROM bot_settings WHERE key = ?",
                (str(key),),
            ).fetchone()
        except sqlite3.OperationalError:
            self._sync_ensure_settings_table(conn)
            conn.commit()
            return default
        if row is None:
            return default
        return str(row["value"])

    async def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Read a persisted key-value runtime setting from SQLite."""
        async with self._lock:
            return await asyncio.to_thread(self._sync_get_setting, key, default)

    def _sync_set_setting(self, key: str, value: str) -> None:
        conn = self._get_connection()
        self._sync_ensure_settings_table(conn)
        conn.execute(
            """
            INSERT INTO bot_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (str(key), str(value), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    async def set_setting(self, key: str, value: str) -> None:
        """Persist a key-value runtime setting into SQLite."""
        async with self._lock:
            await asyncio.to_thread(self._sync_set_setting, key, value)

    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            with self._sync_lock:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
