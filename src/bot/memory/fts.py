"""SQLite FTS5 Long-Term Memory, Lifecycle CRUD, and Hybrid RRF Retrieval."""

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

_UNSET: Any = object()

THREE_CHAR_ENDINGS = (
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
    "ать", "ять", "ить", "ешь", "ете", "ишь", "ите",
)
TWO_CHAR_ENDINGS = (
    "ом", "ем", "ым", "им", "ой", "ей", "ая", "яя", "ое", "ее",
    "ые", "ие", "ую", "юю", "ам", "ям", "ах", "ях", "ов", "ев",
    "их", "ых", "ся", "сь", "ий", "ый",
)
ONE_CHAR_ENDINGS = ("а", "я", "у", "ю", "о", "е", "ы", "и", "ь", "й")

_ARCHIVE_STOPWORDS = frozenset(
    {
        # Russian & Ukrainian pronouns / particles / greetings / filler
        "привет", "привіт", "пока", "давай", "давайте", "может", "можно", "нужно", "надо",
        "хочу", "будет", "буду", "было", "были", "быть", "есть", "нету", "если", "когда",
        "чтобы", "чтоб", "потом", "потому", "просто", "только", "очень", "вообще", "кста",
        "кстати", "короче", "типа", "вроде", "сегодня", "вчера", "завтра", "сейчас", "щас",
        "меня", "тебя", "тебе", "себя", "себе", "него", "нему", "ними", "мене", "мені",
        "тобі", "його", "нами", "вами", "вами", "вона", "воно", "вони", "таке", "такой",
        "такая", "такое", "такие", "какой", "какая", "какие", "кото", "который", "этого",
        "этом", "этот", "этой", "этих", "треба", "може", "пішли", "гоу", "норм", "лагает",
        "ахах", "ахаха", "ахахах", "хаха", "хахаха", "вхпхах", "бля", "блять", "сука",
        "дела", "дело", "справи", "жизнь", "доброе", "добрый", "утро", "вечер", "ночи",
        "спокойной", "нормально", "понятно", "ладно", "хорошо", "ясно", "конечно",
        "пиздец", "нахуй", "ебать", "похуй", "пиструнькиня", "піструнькиня",
        # English common filler
        "that", "this", "with", "from", "have", "what", "when", "where", "they", "your",
    }
)

_FACT_STOPWORDS = _ARCHIVE_STOPWORDS | frozenset(
    {
        # Conversational Russian & Ukrainian fillers, pronouns, particles, prepositions
        "да", "нет", "уже", "ну", "ага", "угу", "ок", "окей", "шо", "що", "чё", "че",
        "как", "что", "это", "вот", "там", "тут", "здесь", "сюда", "туда", "где", "куда",
        "кто", "кого", "кому", "кем", "чем", "чей", "чья", "чьё", "чье", "чьи",
        "она", "оно", "они", "его", "её", "ее", "ему", "им", "ими", "них", "ней", "нём", "нем",
        "її", "йому", "їм", "них", "нам", "вам", "мне", "мной", "тобой", "собой",
        "мой", "моя", "моё", "мое", "мои", "твой", "твоя", "твоё", "твое", "твои",
        "наш", "наша", "наше", "наши", "ваш", "ваша", "ваше", "ваши",
        "свой", "своя", "своё", "свое", "свои", "весь", "вся", "всё", "все", "всех", "всем",
        "так", "також", "тоже", "даже", "ведь", "лишь", "хотя", "пусть", "будто", "вроде",
        "или", "либо", "для", "при", "про", "без", "над", "под", "между", "через", "после", "перед",
        "ещё", "еще", "уже", "очень", "прям", "прямо", "совсем", "почти", "чуть", "много", "мало",
        "был", "была", "были", "будут", "стал", "стала", "стали", "могу", "можешь", "можем",
        "хочешь", "хочет", "хотим", "знаю", "знаешь", "знает", "думаю", "кажется", "значит",
        "почему", "зачем", "откуда", "когда", "тогда", "всегда", "никогда", "иногда",
        "ничего", "ничто", "никто", "нигде", "никак", "каждый", "любой", "другой",
        "того", "той", "том", "тот", "тех", "тем", "эту", "этой", "эти", "этим",
        "привет", "привіт", "здарова", "здорово", "спасибо", "дякую", "пожалуйста", "будь",
        "ласка", "сорри", "сори", "плиз", "блин", "капец", "пипец", "вау", "ого", "хмм", "ммм", "эмм",
        "лол", "кек", "кста", "вже", "ще", "трохи", "дуже", "майже", "тільки", "мабуть", "звісно",
        "добре", "гаразд", "бувай", "чому", "навіщо", "колись", "завжди", "ніколи", "щось", "хтось",
        # English conversational fillers
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was",
        "one", "our", "out", "yes", "no", "lol", "wtf", "omg", "idk", "btw", "imo", "tbh", "fyi",
        "okay", "yeah", "yep", "nope", "hey", "hi", "hello", "bye", "thanks", "thank", "please",
        "sorry", "well", "just", "like", "very", "much", "more", "some", "only", "also", "then",
        "here", "there", "who", "why", "how", "she", "him", "his", "its", "them", "their",
        "been", "will", "would", "could", "should",
    }
)

_FACT_SELECT_COLUMNS = (
    "f.id, f.subject, f.fact, f.category, f.source_chat_id, f.created_at, "
    "f.subject_user_id, f.status, f.updated_at, f.expires_at, f.superseded_by"
)

_SQL_CHUNK_SIZE = 500


@dataclass
class FactRecord:
    id: int
    subject: str
    fact: str
    category: str
    source_chat_id: Optional[int] = None
    created_at: int = 0
    subject_user_id: Optional[int] = None
    status: str = "active"
    updated_at: int = 0
    expires_at: Optional[int] = None
    superseded_by: Optional[int] = None


def _unicode_lower(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).lower()


def _register_unicode_lower(conn: sqlite3.Connection) -> None:
    try:
        conn.create_function("py_lower", 1, _unicode_lower, deterministic=True)
    except TypeError:
        conn.create_function("py_lower", 1, _unicode_lower)


def _row_to_fact_record(row: Any) -> FactRecord:
    return FactRecord(
        id=int(row[0]),
        subject=str(row[1]),
        fact=str(row[2]),
        category=str(row[3]),
        source_chat_id=int(row[4]) if row[4] is not None else None,
        created_at=int(row[5] or 0),
        subject_user_id=int(row[6]) if row[6] is not None else None,
        status=str(row[7] or "active"),
        updated_at=int(row[8] or 0),
        expires_at=int(row[9]) if row[9] is not None else None,
        superseded_by=int(row[10]) if row[10] is not None else None,
    )


def _coerce_query_vector(query_vector: Any) -> Optional[np.ndarray]:
    if query_vector is None:
        return None
    if isinstance(query_vector, (bytes, bytearray, memoryview)):
        raw = bytes(query_vector)
        if not raw or len(raw) % 4 != 0:
            return None
        arr = np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=True)
    else:
        try:
            arr = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
    if arr.ndim != 1 or arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return arr / np.float32(norm)


def stem_russian_token(token: str) -> str:
    """Strip common Russian inflectional endings down to stem (min length 3)."""
    t = token.lower()
    for end in THREE_CHAR_ENDINGS:
        if t.endswith(end) and len(t) - len(end) >= 3:
            return t[:-len(end)]
    for end in TWO_CHAR_ENDINGS:
        if t.endswith(end) and len(t) - len(end) >= 3:
            return t[:-len(end)]
    for end in ONE_CHAR_ENDINGS:
        if t.endswith(end) and len(t) - len(end) >= 3:
            return t[:-len(end)]
    return t


def build_fts_query(query_text: str) -> str:
    """Build sanitized FTS5 prefix query for Russian inflection matching."""
    if not query_text:
        return ""
    tokens = re.findall(r"[\w]+", query_text)
    terms = []
    for t in tokens:
        if len(t) >= 3:
            stem = stem_russian_token(t)
            terms.append(f'"{stem}"*')
        else:
            terms.append(f'"{t}"')
    return " OR ".join(terms) if terms else ""


def _extract_selective_terms(
    query_text: str,
    stopwords: frozenset[str],
    min_token_len: int,
    max_terms: int,
    allow_alnum: bool = False,
) -> List[str]:
    """Extract sorted selective stems while stripping stopwords and short/numeric noise."""
    if not query_text or max_terms <= 0:
        return []
    pattern = r"[^\W_]+" if allow_alnum else r"[^\W\d_]+"
    tokens = re.findall(pattern, query_text.lower())
    seen: set[str] = set()
    candidates: List[str] = []
    for tok in tokens:
        if len(tok) < min_token_len or tok.isdigit() or tok in stopwords:
            continue
        stem = stem_russian_token(tok)
        if len(stem) < 3 or stem in stopwords:
            continue
        if stem not in seen:
            seen.add(stem)
            candidates.append(stem)
    candidates.sort(key=len, reverse=True)
    return candidates[:max_terms]


def extract_archive_terms(query_text: str, max_terms: int = 5) -> List[str]:
    """Extract sorted selective Russian/Ukrainian/English stems for archive lookup."""
    return _extract_selective_terms(
        query_text,
        stopwords=_ARCHIVE_STOPWORDS,
        min_token_len=4,
        max_terms=max_terms,
        allow_alnum=False,
    )


def build_archive_fts_query(query_text: str, max_terms: int = 5, operator: str = "OR") -> str:
    """Build selective FTS5 prefix query for historical chat archive lookup.

    Strips stopwords, pure numbers, and short (<4 char) filler tokens so small talk
    returns an empty query (0 prompt overhead) while specific topics hit archive lore.
    """
    selected = extract_archive_terms(query_text, max_terms=max_terms)
    if not selected:
        return ""
    op = " AND " if operator.upper() == "AND" else " OR "
    return op.join(f'"{stem}"*' for stem in selected)


def extract_fact_terms(query_text: str, max_terms: int = 6) -> List[str]:
    """Extract sorted meaningful (>=3 char) non-stopword stems for memory fact lookup."""
    return _extract_selective_terms(
        query_text,
        stopwords=_FACT_STOPWORDS,
        min_token_len=3,
        max_terms=max_terms,
        allow_alnum=True,
    )


def build_fact_fts_query(query_text: str, max_terms: int = 6, operator: str = "OR") -> str:
    """Build selective FTS5 prefix query for memory_facts lookup.

    Filters out conversational fillers, pronouns, pure digits, and <3-char tokens so
    low-signal replies like 'Да', 'Уже', or 'Ок' produce an empty query.
    """
    selected = extract_fact_terms(query_text, max_terms=max_terms)
    if not selected:
        return ""
    op = " AND " if operator.upper() == "AND" else " OR "
    return op.join(f'"{stem}"*' for stem in selected)


class FTSStore:
    """Synchronous SQLite helper for FTS5 long-term memory facts and hybrid retrieval."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Create/migrate memory_facts, memory_fts, embeddings table, indexes, and triggers."""
        _register_unicode_lower(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                source_chat_id INTEGER,
                created_at INTEGER NOT NULL,
                subject_user_id INTEGER,
                status TEXT NOT NULL DEFAULT 'active',
                updated_at INTEGER NOT NULL DEFAULT 0,
                expires_at INTEGER,
                superseded_by INTEGER
            )
        """)

        existing_cols = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(memory_facts)").fetchall()
        }
        migrations = (
            ("subject_user_id", "INTEGER"),
            ("status", "TEXT NOT NULL DEFAULT 'active'"),
            ("updated_at", "INTEGER NOT NULL DEFAULT 0"),
            ("expires_at", "INTEGER"),
            ("superseded_by", "INTEGER"),
        )
        for col_name, col_ddl in migrations:
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE memory_facts ADD COLUMN {col_name} {col_ddl}")

        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                subject,
                fact,
                category,
                content='memory_facts',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS trg_memory_ai AFTER INSERT ON memory_facts BEGIN
                INSERT INTO memory_fts(rowid, subject, fact, category)
                VALUES (new.id, new.subject, new.fact, new.category);
            END;

            CREATE TRIGGER IF NOT EXISTS trg_memory_ad AFTER DELETE ON memory_facts BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, subject, fact, category)
                VALUES ('delete', old.id, old.subject, old.fact, old.category);
            END;

            CREATE TRIGGER IF NOT EXISTS trg_memory_au AFTER UPDATE ON memory_facts BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, subject, fact, category)
                VALUES ('delete', old.id, old.subject, old.fact, old.category);
                INSERT INTO memory_fts(rowid, subject, fact, category)
                VALUES (new.id, new.subject, new.fact, new.category);
            END;

            CREATE INDEX IF NOT EXISTS idx_memory_facts_status_chat_subj
                ON memory_facts(status, source_chat_id, subject_user_id);
            CREATE INDEX IF NOT EXISTS idx_memory_facts_updated
                ON memory_facts(updated_at DESC);

            CREATE TABLE IF NOT EXISTS memory_fact_embeddings (
                fact_id INTEGER PRIMARY KEY REFERENCES memory_facts(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TRIGGER IF NOT EXISTS trg_memory_emb_ad AFTER DELETE ON memory_facts BEGIN
                DELETE FROM memory_fact_embeddings WHERE fact_id = old.id;
            END;
        """)

        fact_count = int(conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0])
        if fact_count > 0:
            docsize_count = int(
                conn.execute("SELECT COUNT(*) FROM memory_fts_docsize").fetchone()[0]
            )
            if docsize_count == 0:
                conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")

        conn.execute(
            "UPDATE memory_facts SET updated_at = created_at "
            "WHERE updated_at = 0 AND created_at > 0"
        )
        conn.execute(
            "UPDATE memory_facts SET status = 'active' "
            "WHERE status IS NULL OR status = ''"
        )
        conn.commit()

    @staticmethod
    def init_messages_fts(conn: sqlite3.Connection) -> None:
        """Create messages_fts virtual table, sync triggers, and backfill existing rows."""
        conn.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                author_name,
                text,
                media_summary,
                content='messages',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS trg_messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, author_name, text, media_summary)
                VALUES (new.id, new.author_name, COALESCE(new.text, ''), COALESCE(new.media_summary, ''));
            END;

            CREATE TRIGGER IF NOT EXISTS trg_messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, author_name, text, media_summary)
                VALUES ('delete', old.id, old.author_name, COALESCE(old.text, ''), COALESCE(old.media_summary, ''));
            END;

            CREATE TRIGGER IF NOT EXISTS trg_messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts, rowid, author_name, text, media_summary)
                VALUES ('delete', old.id, old.author_name, COALESCE(old.text, ''), COALESCE(old.media_summary, ''));
                INSERT INTO messages_fts(rowid, author_name, text, media_summary)
                VALUES (new.id, new.author_name, COALESCE(new.text, ''), COALESCE(new.media_summary, ''));
            END;
        """)
        msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        if msg_count > 0:
            docsize_count = conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]
            if docsize_count == 0:
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()

    @staticmethod
    def add_fact(
        conn: sqlite3.Connection,
        subject: str,
        fact: str,
        category: str,
        source_chat_id: Optional[int] = None,
        created_at: Optional[int] = None,
        subject_user_id: Optional[int] = None,
        status: str = "active",
        expires_at: Optional[int] = None,
    ) -> int:
        """Insert a fact into memory_facts (trigger automatically populates memory_fts)."""
        ts = int(created_at) if created_at is not None else int(time.time())
        eff_status = status or "active"
        subj_uid = int(subject_user_id) if subject_user_id is not None else None
        exp_ts = int(expires_at) if expires_at is not None else None
        cursor = conn.execute(
            """
            INSERT INTO memory_facts (
                subject, fact, category, source_chat_id, created_at,
                subject_user_id, status, updated_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (subject, fact, category, source_chat_id, ts, subj_uid, eff_status, ts, exp_ts),
        )
        conn.commit()
        return int(cursor.lastrowid)  # type: ignore[arg-type]

    @staticmethod
    def update_fact(
        conn: sqlite3.Connection,
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
        """Update an existing fact and invalidate embeddings if subject or fact changed."""
        fid = int(fact_id)
        existing = conn.execute(
            "SELECT subject, fact FROM memory_facts WHERE id = ?",
            (fid,),
        ).fetchone()
        if existing is None:
            return False

        old_subject = str(existing[0])
        old_fact = str(existing[1])
        ts = int(updated_at) if updated_at is not None else int(time.time())

        assignments: List[str] = ["updated_at = ?"]
        params: List[Any] = [ts]

        text_changed = False
        if subject is not None:
            assignments.append("subject = ?")
            params.append(subject)
            if subject != old_subject:
                text_changed = True
        if fact is not None:
            assignments.append("fact = ?")
            params.append(fact)
            if fact != old_fact:
                text_changed = True
        if subject_user_id is not _UNSET:
            assignments.append("subject_user_id = ?")
            params.append(int(subject_user_id) if subject_user_id is not None else None)
        if category is not None:
            assignments.append("category = ?")
            params.append(category)
        if expires_at is not _UNSET:
            assignments.append("expires_at = ?")
            params.append(int(expires_at) if expires_at is not None else None)
        if status is not None:
            assignments.append("status = ?")
            params.append(status)

        params.append(fid)
        conn.execute(
            f"UPDATE memory_facts SET {', '.join(assignments)} WHERE id = ?",
            params,
        )
        if text_changed or status in ("superseded", "archived"):
            conn.execute("DELETE FROM memory_fact_embeddings WHERE fact_id = ?", (fid,))
        conn.commit()
        return True

    @staticmethod
    def supersede_fact(
        conn: sqlite3.Connection,
        fact_id: int,
        *,
        new_fact_id: Optional[int] = None,
        status: str = "superseded",
        updated_at: Optional[int] = None,
    ) -> bool:
        """Mark fact as superseded/archived, record superseded_by, and drop stale embedding."""
        fid = int(fact_id)
        eff_status = status if status in ("superseded", "archived") else "superseded"
        ts = int(updated_at) if updated_at is not None else int(time.time())
        replacement_id = int(new_fact_id) if new_fact_id is not None else None

        cur = conn.execute(
            "UPDATE memory_facts SET status = ?, superseded_by = ?, updated_at = ? WHERE id = ?",
            (eff_status, replacement_id, ts, fid),
        )
        if (cur.rowcount or 0) == 0:
            conn.commit()
            return False
        conn.execute("DELETE FROM memory_fact_embeddings WHERE fact_id = ?", (fid,))
        conn.commit()
        return True

    @staticmethod
    def archive_expired_facts(
        conn: sqlite3.Connection,
        *,
        now_ts: Optional[int] = None,
        agreement_ttl_days: Optional[int] = None,
        event_ttl_days: Optional[int] = None,
        chat_id: Optional[int] = None,
    ) -> int:
        """Archive active facts past explicit expires_at or category TTL and prune embeddings."""
        now = int(now_ts) if now_ts is not None else int(time.time())
        expiry_clauses: List[str] = ["(expires_at IS NOT NULL AND expires_at <= ?)"]
        params: List[Any] = [now]

        if agreement_ttl_days is not None and agreement_ttl_days > 0:
            cutoff = now - int(agreement_ttl_days * 86400)
            expiry_clauses.append(
                "(expires_at IS NULL AND category = 'agreement' AND updated_at <= ?)"
            )
            params.append(cutoff)

        if event_ttl_days is not None and event_ttl_days > 0:
            cutoff = now - int(event_ttl_days * 86400)
            expiry_clauses.append(
                "(expires_at IS NULL AND category = 'event' AND updated_at <= ?)"
            )
            params.append(cutoff)

        sql = (
            "SELECT id FROM memory_facts "
            f"WHERE status = 'active' AND ({' OR '.join(expiry_clauses)})"
        )
        if chat_id is not None:
            sql += " AND source_chat_id = ?"
            params.append(int(chat_id))

        rows = conn.execute(sql, params).fetchall()
        expired_ids = [int(r[0]) for r in rows]
        if not expired_ids:
            return 0

        for start in range(0, len(expired_ids), _SQL_CHUNK_SIZE):
            chunk = expired_ids[start : start + _SQL_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(
                f"UPDATE memory_facts SET status = 'archived', updated_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *chunk],
            )
            conn.execute(
                f"DELETE FROM memory_fact_embeddings WHERE fact_id IN ({placeholders})",
                chunk,
            )
        conn.commit()
        return len(expired_ids)

    @staticmethod
    def get_active_facts(
        conn: sqlite3.Connection,
        *,
        chat_id: Optional[int] = None,
        subject_user_ids: Optional[Sequence[int]] = None,
        subjects: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        include_global: bool = False,
        limit: int = 50,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        """Return active, non-expired facts filtered by chat scope, subjects, and categories."""
        if limit <= 0:
            return []

        now = int(now_ts) if now_ts is not None else int(time.time())
        sql = (
            f"SELECT {_FACT_SELECT_COLUMNS} FROM memory_facts f "
            "WHERE f.status = 'active' AND (f.expires_at IS NULL OR f.expires_at > ?)"
        )
        params: List[Any] = [now]

        if chat_id is not None:
            if include_global:
                sql += " AND (f.source_chat_id = ? OR f.source_chat_id IS NULL)"
            else:
                sql += " AND f.source_chat_id = ?"
            params.append(int(chat_id))

        if subject_user_ids is not None or subjects is not None:
            uid_list = [int(u) for u in subject_user_ids] if subject_user_ids else []
            subj_list = (
                [str(s).strip().lower() for s in subjects if str(s).strip()]
                if subjects
                else []
            )
            if not uid_list and not subj_list:
                return []
            subj_clauses: List[str] = []
            if uid_list:
                uid_ph = ",".join("?" for _ in uid_list)
                subj_clauses.append(f"f.subject_user_id IN ({uid_ph})")
                params.extend(uid_list)
            if subj_list:
                subj_ph = ",".join("?" for _ in subj_list)
                subj_clauses.append(f"py_lower(f.subject) IN ({subj_ph})")
                params.extend(subj_list)
            sql += f" AND ({' OR '.join(subj_clauses)})"

        if categories is not None:
            cat_list = [str(c).strip().lower() for c in categories if str(c).strip()]
            if not cat_list:
                return []
            cat_ph = ",".join("?" for _ in cat_list)
            sql += f" AND py_lower(f.category) IN ({cat_ph})"
            params.extend(cat_list)

        sql += " ORDER BY f.updated_at DESC, f.id DESC LIMIT ?"
        params.append(int(limit))

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            _register_unicode_lower(conn)
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_fact_record(r) for r in rows]

    @staticmethod
    def get_facts_by_ids(
        conn: sqlite3.Connection,
        fact_ids: Sequence[int],
    ) -> Dict[int, FactRecord]:
        """Fetch facts keyed by ID regardless of lifecycle status."""
        unique_ids = sorted({int(fid) for fid in fact_ids}) if fact_ids else []
        if not unique_ids:
            return {}
        out: Dict[int, FactRecord] = {}
        for start in range(0, len(unique_ids), _SQL_CHUNK_SIZE):
            chunk = unique_ids[start : start + _SQL_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT {_FACT_SELECT_COLUMNS} FROM memory_facts f "
                f"WHERE f.id IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                rec = _row_to_fact_record(r)
                out[rec.id] = rec
        return out

    @staticmethod
    def upsert_fact_embedding(
        conn: sqlite3.Connection,
        fact_id: int,
        model: str,
        dim: int,
        vector_bytes: bytes,
        updated_at: Optional[int] = None,
    ) -> None:
        """Insert or update a float32 embedding blob for an active fact."""
        fid = int(fact_id)
        row = conn.execute("SELECT status FROM memory_facts WHERE id = ?", (fid,)).fetchone()
        if row is None or str(row[0] or "active") != "active":
            return
        ts = int(updated_at) if updated_at is not None else int(time.time())
        conn.execute(
            """
            INSERT INTO memory_fact_embeddings (fact_id, model, dim, vector, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_id) DO UPDATE SET
                model = excluded.model,
                dim = excluded.dim,
                vector = excluded.vector,
                updated_at = excluded.updated_at
            """,
            (fid, str(model), int(dim), sqlite3.Binary(bytes(vector_bytes)), ts),
        )
        conn.commit()

    @staticmethod
    def delete_fact_embedding(conn: sqlite3.Connection, fact_id: int) -> None:
        """Delete embedding row for fact_id."""
        conn.execute("DELETE FROM memory_fact_embeddings WHERE fact_id = ?", (int(fact_id),))
        conn.commit()

    @staticmethod
    def get_unembedded_active_facts(
        conn: sqlite3.Connection,
        model: str,
        limit: int = 32,
        chat_id: Optional[int] = None,
        *,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        """Return active non-expired facts missing an embedding row for `model`."""
        if limit <= 0:
            return []
        now = int(now_ts) if now_ts is not None else int(time.time())
        sql = f"""
            SELECT {_FACT_SELECT_COLUMNS}
            FROM memory_facts f
            LEFT JOIN memory_fact_embeddings e
              ON e.fact_id = f.id AND e.model = ?
            WHERE f.status = 'active'
              AND (f.expires_at IS NULL OR f.expires_at > ?)
              AND e.fact_id IS NULL
        """
        params: List[Any] = [str(model), now]
        if chat_id is not None:
            sql += " AND (f.source_chat_id = ? OR f.source_chat_id IS NULL)"
            params.append(int(chat_id))
        sql += " ORDER BY f.updated_at DESC, f.id DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_fact_record(r) for r in rows]

    @staticmethod
    def _execute_fact_query(
        conn: sqlite3.Connection,
        fts_query: str,
        limit: int,
        excluded_ids: Sequence[int],
        *,
        chat_id: Optional[int] = None,
        now_ts: Optional[int] = None,
    ) -> List[FactRecord]:
        if not fts_query or limit <= 0:
            return []
        now = int(now_ts) if now_ts is not None else int(time.time())
        excluded_set = set(excluded_ids)
        inline_excluded = excluded_ids if len(excluded_ids) <= _SQL_CHUNK_SIZE else ()
        sql_limit = int(limit) + (len(excluded_ids) if not inline_excluded else 0)

        sql = f"""
            SELECT {_FACT_SELECT_COLUMNS}
            FROM memory_fts m
            JOIN memory_facts f ON f.id = m.rowid
            WHERE memory_fts MATCH ?
              AND f.status = 'active'
              AND (f.expires_at IS NULL OR f.expires_at > ?)
        """
        params: List[Any] = [fts_query, now]
        if chat_id is not None:
            sql += " AND (f.source_chat_id = ? OR f.source_chat_id IS NULL)"
            params.append(int(chat_id))
        if inline_excluded:
            placeholders = ",".join("?" for _ in inline_excluded)
            sql += f" AND f.id NOT IN ({placeholders})"
            params.extend(inline_excluded)
        sql += " ORDER BY rank LIMIT ?"
        params.append(sql_limit)
        try:
            rows = conn.execute(sql, params).fetchall()
            out: List[FactRecord] = []
            for row in rows:
                rec = _row_to_fact_record(row)
                if rec.id in excluded_set:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    break
            return out
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _execute_vector_query(
        conn: sqlite3.Connection,
        qvec: np.ndarray,
        excluded_ids: Sequence[int],
        *,
        chat_id: Optional[int] = None,
        embedding_model: Optional[str] = None,
        min_vector_similarity: float = 0.52,
        now_ts: int,
    ) -> List[FactRecord]:
        dim = int(qvec.shape[0])
        expected_bytes = dim * 4
        excluded_set = set(excluded_ids)
        inline_excluded = excluded_ids if len(excluded_ids) <= _SQL_CHUNK_SIZE else ()

        sql = f"""
            SELECT {_FACT_SELECT_COLUMNS}, e.vector
            FROM memory_fact_embeddings e
            JOIN memory_facts f ON f.id = e.fact_id
            WHERE f.status = 'active'
              AND (f.expires_at IS NULL OR f.expires_at > ?)
              AND e.dim = ?
        """
        params: List[Any] = [now_ts, dim]
        if embedding_model:
            sql += " AND e.model = ?"
            params.append(str(embedding_model))
        if chat_id is not None:
            sql += " AND (f.source_chat_id = ? OR f.source_chat_id IS NULL)"
            params.append(int(chat_id))
        if inline_excluded:
            placeholders = ",".join("?" for _ in inline_excluded)
            sql += f" AND f.id NOT IN ({placeholders})"
            params.extend(inline_excluded)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []

        records: List[FactRecord] = []
        blobs: List[bytes] = []
        for row in rows:
            fid = int(row[0])
            if fid in excluded_set:
                continue
            blob = row[11]
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                continue
            raw_blob = bytes(blob)
            if len(raw_blob) != expected_bytes:
                continue
            records.append(_row_to_fact_record(row))
            blobs.append(raw_blob)

        if not blobs:
            return []

        mat = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), dim)
        norms = np.linalg.norm(mat, axis=1)
        safe_norms = np.where(norms > 1e-12, norms, np.float32(1.0))
        sims = (mat @ qvec) / safe_norms
        sims = np.where(norms > 1e-12, sims, np.float32(-1.0))

        mask = np.isfinite(sims) & (sims >= float(min_vector_similarity))
        matching_idxs = np.flatnonzero(mask).tolist()
        if not matching_idxs:
            return []

        matching_idxs.sort(
            key=lambda idx: (
                float(sims[idx]),
                records[idx].updated_at,
                -records[idx].id,
            ),
            reverse=True,
        )
        return [records[idx] for idx in matching_idxs]

    @staticmethod
    def search_facts(
        conn: sqlite3.Connection,
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
        """Search active facts via chat-scoped Hybrid FTS5 + Vector RRF + entity/recency boost."""
        if limit <= 0:
            return []

        now = int(now_ts) if now_ts is not None else int(time.time())
        terms = extract_fact_terms(query, max_terms=6)
        qvec = _coerce_query_vector(query_vector)
        if qvec is None and not terms:
            return []

        excluded_ids = sorted({int(fid) for fid in exclude_fact_ids}) if exclude_fact_ids else []
        pool_size = max(int(limit) * 6, 24)

        candidates_by_id: Dict[int, FactRecord] = {}
        fts_rank_by_id: Dict[int, int] = {}
        vec_rank_by_id: Dict[int, int] = {}
        and_match_ids: Set[int] = set()

        if terms:
            if len(terms) >= 2:
                and_query = " AND ".join(f'"{stem}"*' for stem in terms)
                and_hits = FTSStore._execute_fact_query(
                    conn,
                    and_query,
                    pool_size,
                    excluded_ids,
                    chat_id=chat_id,
                    now_ts=now,
                )
                for rec in and_hits:
                    if rec.id not in fts_rank_by_id:
                        fts_rank_by_id[rec.id] = len(fts_rank_by_id)
                        and_match_ids.add(rec.id)
                        candidates_by_id[rec.id] = rec

            # In pure FTS mode (qvec is None), preserve exact AND-first filtering with OR fallback
            # when AND returns 0 results; in hybrid mode, gather remaining OR candidates up to pool_size.
            if (not fts_rank_by_id or qvec is not None) and len(fts_rank_by_id) < pool_size:
                or_query = " OR ".join(f'"{stem}"*' for stem in terms)
                or_hits = FTSStore._execute_fact_query(
                    conn,
                    or_query,
                    pool_size,
                    excluded_ids,
                    chat_id=chat_id,
                    now_ts=now,
                )
                for rec in or_hits:
                    if rec.id not in fts_rank_by_id:
                        fts_rank_by_id[rec.id] = len(fts_rank_by_id)
                        candidates_by_id[rec.id] = rec
                        if len(fts_rank_by_id) >= pool_size:
                            break

        if qvec is not None:
            vec_hits = FTSStore._execute_vector_query(
                conn,
                qvec,
                excluded_ids,
                chat_id=chat_id,
                embedding_model=embedding_model,
                min_vector_similarity=min_vector_similarity,
                now_ts=now,
            )
            for rank_vec, rec in enumerate(vec_hits):
                vec_rank_by_id[rec.id] = rank_vec
                if rec.id not in candidates_by_id:
                    candidates_by_id[rec.id] = rec

        if not candidates_by_id:
            return []

        boosted_uids: Set[int] = set()
        if author_user_id is not None:
            boosted_uids.add(int(author_user_id))
        if target_user_ids:
            for uid in target_user_ids:
                if uid is not None:
                    boosted_uids.add(int(uid))

        rrf_k = 60.0
        scored: List[tuple[float, int, int, FactRecord]] = []
        for fid, rec in candidates_by_id.items():
            r_fts = fts_rank_by_id.get(fid)
            r_vec = vec_rank_by_id.get(fid)
            score = (1.0 / (rrf_k + r_fts) if r_fts is not None else 0.0) + (
                1.0 / (rrf_k + r_vec) if r_vec is not None else 0.0
            )
            if fid in and_match_ids:
                score += 0.004
            if rec.subject_user_id is not None and rec.subject_user_id in boosted_uids:
                score += 0.0025
            ref_ts = max(int(rec.updated_at or 0), int(rec.created_at or 0))
            age_days = max(0.0, (now - ref_ts) / 86400.0)
            score += 0.0015 / (1.0 + age_days / 30.0)
            scored.append((score, int(rec.updated_at or 0), -int(fid), rec))

        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[3] for item in scored[: int(limit)]]

    @staticmethod
    def delete_fact(conn: sqlite3.Connection, fact_id: int) -> None:
        """Delete fact by id (triggers automatically update memory_fts and embeddings)."""
        conn.execute("DELETE FROM memory_facts WHERE id = ?", (int(fact_id),))
        conn.commit()
