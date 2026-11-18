"""SQLite Message Store and Integrated MemoryStore."""

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Set

from bot.memory.fts import FactRecord, FTSStore, build_archive_fts_query


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


class MemoryStore:
    """Asynchronous Memory Store managing short-term message history and long-term FTS facts."""

    def __init__(self, db_path: str = "data/memory.sqlite") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                p = Path(self.db_path)
                p.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
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
        """)
        conn.commit()
        FTSStore.init_schema(conn)
        FTSStore.init_messages_fts(conn)

    async def init_db(self) -> None:
        """Initialize database schema asynchronously."""
        async with self._lock:
            await asyncio.to_thread(self._sync_init_db)

    def _sync_save_message(self, msg: ChatMessage) -> None:
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
                compacted = excluded.compacted
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

    async def search_facts(
        self,
        query: str,
        limit: int = 5,
        exclude_fact_ids: Optional[Sequence[int]] = None,
    ) -> List[FactRecord]:
        """Search long-term facts using Russian prefix full-text search."""
        async with self._lock:
            conn = self._get_connection()
            return await asyncio.to_thread(
                FTSStore.search_facts,
                conn,
                query,
                limit,
                exclude_fact_ids,
            )

    async def add_fact(
        self,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int] = None,
    ) -> int:
        """Add a long-term fact to SQLite and FTS index."""
        async with self._lock:
            conn = self._get_connection()
            return await asyncio.to_thread(
                FTSStore.add_fact, conn, subject, fact, category, source_chat_id
            )

    async def delete_fact(self, fact_id: int) -> None:
        """Delete fact from long-term memory."""
        async with self._lock:
            conn = self._get_connection()
            await asyncio.to_thread(FTSStore.delete_fact, conn, fact_id)

    async def compact_old_messages(
        self,
        chat_id: int,
        summarize_fn: Callable[[List[ChatMessage]], Any],
        threshold: int = 60,
        batch_size: int = 30,
    ) -> int:
        """Compact messages when uncompacted count reaches threshold."""
        # Check count of uncompacted messages
        async with self._lock:
            conn = self._get_connection()
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND compacted = 0",
                (chat_id,),
            ).fetchone()[0]

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
                    )
                elif isinstance(item, FactRecord):
                    await self.add_fact(
                        subject=item.subject,
                        fact=item.fact,
                        category=item.category,
                        source_chat_id=chat_id,
                    )

        msg_ids = [m.id for m in uncompacted if m.id is not None]
        await self.mark_compacted(msg_ids)
        return len(msg_ids)

    async def close(self) -> None:
        """Close database connection."""
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
