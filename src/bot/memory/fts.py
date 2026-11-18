"""SQLite FTS5 Long-Term Memory and Russian Query Tokenizer."""

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

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


@dataclass
class FactRecord:
    id: int
    subject: str
    fact: str
    category: str
    source_chat_id: Optional[int] = None
    created_at: int = 0


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
    """Synchronous SQLite helper for FTS5 long-term memory facts."""

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        """Create memory_facts table, memory_fts virtual table, and triggers."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL,
                source_chat_id INTEGER,
                created_at INTEGER NOT NULL
            );

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
        """)
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
    ) -> int:
        """Insert a fact into memory_facts (trigger automatically populates memory_fts)."""
        ts = created_at if created_at is not None else int(time.time())
        cursor = conn.execute(
            """
            INSERT INTO memory_facts (subject, fact, category, source_chat_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (subject, fact, category, source_chat_id, ts),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    @staticmethod
    def _execute_fact_query(
        conn: sqlite3.Connection,
        fts_query: str,
        limit: int,
        excluded_ids: Sequence[int],
    ) -> List[FactRecord]:
        if not fts_query or limit <= 0:
            return []
        sql = """
            SELECT f.id, f.subject, f.fact, f.category, f.source_chat_id, f.created_at
            FROM memory_fts m
            JOIN memory_facts f ON f.id = m.rowid
            WHERE memory_fts MATCH ?
        """
        params: List[Any] = [fts_query]
        if excluded_ids:
            placeholders = ",".join("?" for _ in excluded_ids)
            sql += f" AND f.id NOT IN ({placeholders})"
            params.extend(excluded_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = conn.execute(sql, params).fetchall()
            return [
                FactRecord(
                    id=row[0],
                    subject=row[1],
                    fact=row[2],
                    category=row[3],
                    source_chat_id=row[4],
                    created_at=row[5],
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def search_facts(
        conn: sqlite3.Connection,
        query: str,
        limit: int = 5,
        exclude_fact_ids: Optional[Sequence[int]] = None,
    ) -> List[FactRecord]:
        """Search facts using selective Russian prefix search with AND-first fallback to OR."""
        if limit <= 0:
            return []
        terms = extract_fact_terms(query, max_terms=6)
        if not terms:
            return []

        excluded_ids = sorted({int(fid) for fid in exclude_fact_ids}) if exclude_fact_ids else []
        if len(terms) >= 2:
            and_query = " AND ".join(f'"{stem}"*' for stem in terms)
            results = FTSStore._execute_fact_query(conn, and_query, limit, excluded_ids)
            if results:
                return results

        or_query = " OR ".join(f'"{stem}"*' for stem in terms)
        return FTSStore._execute_fact_query(conn, or_query, limit, excluded_ids)

    @staticmethod
    def delete_fact(conn: sqlite3.Connection, fact_id: int) -> None:
        """Delete fact by id (trigger automatically updates memory_fts)."""
        conn.execute("DELETE FROM memory_facts WHERE id = ?", (fact_id,))
        conn.commit()
