"""Unit tests for SQLite Short-Term Buffer, FTS5 Russian Memory, Triggers & Compaction."""

import time

import pytest

from bot.memory.compaction import CompactionEngine
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FTSStore, build_fact_fts_query, build_fts_query, stem_russian_token


@pytest.mark.asyncio
async def test_russian_query_builder():
    """Verify Russian inflection stripping and prefix wildcard query generation."""
    # Test stemmer on various Russian inflections
    assert stem_russian_token("шаурма") == "шаурм"
    assert stem_russian_token("шаурму") == "шаурм"
    assert stem_russian_token("вокзале") == "вокзал"
    assert stem_russian_token("вокзал") == "вокзал"
    assert stem_russian_token("московском") == "московск"
    assert stem_russian_token("московский") == "московск"

    # Test build_fts_query
    q = build_fts_query("Любит шаурму на Московском вокзале")
    assert '"любит"*' in q
    assert '"шаурм"*' in q
    assert '"московск"*' in q
    assert '"вокзал"*' in q
    assert " OR " in q

    # Empty / punctuation-only queries
    assert build_fts_query("") == ""
    assert build_fts_query("!@#$%^&*()") == ""


@pytest.mark.asyncio
async def test_messages_crud_and_ordering():
    """Verify storing, updating, and chronological retrieval of chat messages."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -10012345
    t0 = int(time.time())

    # Insert 5 sequential messages
    for i in range(5):
        msg = ChatMessage(
            telegram_msg_id=100 + i,
            chat_id=chat_id,
            user_id=1001,
            timestamp=t0 + i * 10,
            author_name="Алексей",
            text=f"Сообщение {i}",
        )
        await store.save_message(msg)

    # Fetch recent history
    history = await store.get_recent_history(chat_id, limit=3)
    assert len(history) == 3
    # Verify chronological order (oldest first among the last 3)
    assert history[0].text == "Сообщение 2"
    assert history[1].text == "Сообщение 3"
    assert history[2].text == "Сообщение 4"

    # Test update via ON CONFLICT (message edited)
    edited = ChatMessage(
        telegram_msg_id=104,
        chat_id=chat_id,
        user_id=1001,
        timestamp=t0 + 40,
        author_name="Алексей",
        text="Отредактированное сообщение 4",
    )
    await store.save_message(edited)
    history_after = await store.get_recent_history(chat_id, limit=1)
    assert history_after[0].text == "Отредактированное сообщение 4"

    await store.close()


@pytest.mark.asyncio
async def test_fts5_triggers_and_russian_search():
    """Verify FTS5 automatic trigger synchronization and Russian inflection search."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    # Insert initial fact: Russian text with inflections
    fact_id = await store.add_fact(
        subject="Алексей",
        fact="Любит шаурму на Московском вокзале",
        category="friend_bio",
        source_chat_id=123,
    )
    assert fact_id > 0

    # Search with inflected terms
    # "шаурма" (nominative) should match "шаурму" (accusative) in the fact
    res1 = await store.search_facts("шаурма")
    assert len(res1) == 1
    assert res1[0].subject == "Алексей"
    assert "шаурму" in res1[0].fact

    # "вокзал" (nominative) should match "вокзале" (prepositional)
    res2 = await store.search_facts("вокзал")
    assert len(res2) == 1
    assert res2[0].id == fact_id

    # Test trigger trg_memory_au (update trigger)
    conn = store._get_connection()
    conn.execute(
        "UPDATE memory_facts SET fact = ? WHERE id = ?",
        ("Предпочитает пиццу с сыром", fact_id),
    )
    conn.commit()

    # Old term should no longer match
    assert len(await store.search_facts("шаурма")) == 0
    # New term should match
    res_pizza = await store.search_facts("пицца")
    assert len(res_pizza) == 1
    assert "пиццу" in res_pizza[0].fact

    # Test trigger trg_memory_ad (delete trigger)
    await store.delete_fact(fact_id)
    assert len(await store.search_facts("пицца")) == 0

    await store.close()


@pytest.mark.asyncio
async def test_compaction_flow():
    """Verify uncompacted message detection, fact extraction callback, and compaction marking."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -10099
    t0 = int(time.time())

    # Add 10 uncompacted messages
    for i in range(10):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=i + 1,
                chat_id=chat_id,
                user_id=2000 + i,
                timestamp=t0 + i,
                author_name=f"User_{i}",
                text=f"Обсуждение фичи номер {i}",
                compacted=0,
            )
        )

    uncompacted = await store.get_uncompacted_messages(chat_id, limit=10)
    assert len(uncompacted) == 10

    # Compaction callback that extracts durable facts
    async def mock_summarizer(msgs):
        return [
            {
                "subject": "Фича",
                "fact": f"Команда утвердила фичи 0-{len(msgs)-1}",
                "category": "agreement",
            }
        ]

    # Threshold set to 5, batch size 5
    compacted_count = await store.compact_old_messages(
        chat_id=chat_id,
        summarize_fn=mock_summarizer,
        threshold=5,
        batch_size=5,
    )
    assert compacted_count == 5

    # Remaining uncompacted should now be 5
    remaining = await store.get_uncompacted_messages(chat_id, limit=10)
    assert len(remaining) == 5

    # Check that extracted fact is now in memory
    facts = await store.search_facts("утвердила фичи")
    assert len(facts) >= 1
    assert facts[0].category == "agreement"

    # Test CompactionEngine helper
    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=lambda transcript: [{"subject": "Transcript", "fact": "Итог чата", "category": "lore"}],
        batch_size=5,
    )
    engine_count = await engine.run_compaction_for_chat(chat_id)
    assert engine_count == 5

    # Now all 10 are compacted
    final_remaining = await store.get_uncompacted_messages(chat_id, limit=10)
    assert len(final_remaining) == 0

    await store.close()


@pytest.mark.asyncio
async def test_compaction_llm_summary_does_not_duplicate_participant_facts():
    """Verify successful LLM summary stores only summary facts without duplicating raw participant messages."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -1003845062562
    t0 = int(time.time())

    batch_messages = [
        ChatMessage(
            telegram_msg_id=201,
            chat_id=chat_id,
            user_id=891777515,
            timestamp=t0 + 1,
            author_name="Аліна",
            text="Сегодня маршрутки подорожали, теперь по 20 грн!",
        ),
        ChatMessage(
            telegram_msg_id=202,
            chat_id=chat_id,
            user_id=380227601,
            timestamp=t0 + 2,
            author_name="Женя",
            text="Любит проводить кулинарные эксперименты с пельменями",
        ),
        ChatMessage(
            telegram_msg_id=203,
            chat_id=chat_id,
            user_id=447164588,
            timestamp=t0 + 3,
            author_name="Анюта",
            text="Договорились вечером собраться играть в настолки",
        ),
        ChatMessage(
            telegram_msg_id=204,
            chat_id=chat_id,
            user_id=382962362,
            timestamp=t0 + 4,
            author_name="Родион",
            text="Навел немецкий порядок в таблицах и разобрал правила",
        ),
        ChatMessage(
            telegram_msg_id=205,
            chat_id=chat_id,
            user_id=676566529,
            timestamp=t0 + 5,
            author_name="Владос",
            text="Купил новый термостат и проверяет давление в котле",
        ),
        ChatMessage(
            telegram_msg_id=206,
            chat_id=chat_id,
            user_id=891777515,
            timestamp=t0 + 6,
            author_name="Аліна",
            text="Записалась на утренний забег вдоль набережной",
        ),
    ]
    for msg in batch_messages:
        await store.save_message(msg)

    seen_transcripts = []

    def fenced_summarizer(transcript: str) -> str:
        seen_transcripts.append(transcript)
        return (
            "```json\n"
            '[{"subject": "летальний ісход", "fact": "Обсудили настолки и планы на выходные", "category": "lore"}]\n'
            "```"
        )

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=fenced_summarizer,
        threshold=6,
        batch_size=6,
    )
    compacted = await engine.run_compaction_for_chat(chat_id)
    assert compacted == 6
    assert len(seen_transcripts) == 1

    # Only the LLM-extracted fact should be stored (no raw participant message duplication)
    conn = store._get_connection()
    total_facts = conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
    assert total_facts == 1

    summary_hits = await store.search_facts("летальний ісход")
    assert len(summary_hits) == 1
    assert await store.search_facts("маршрутка") == []
    assert await store.search_facts("пельмени") == []

    await store.close()


@pytest.mark.asyncio
async def test_compaction_fallback_populates_searchable_participant_facts_and_excludes_bot():
    """Verify deterministic fallback extracts substantive human facts only when LLM summary is missing/empty."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -1003845062563
    bot_id = 8710978944
    t0 = int(time.time())

    batch_messages = [
        ChatMessage(
            telegram_msg_id=301,
            chat_id=chat_id,
            user_id=bot_id,
            timestamp=t0 + 1,
            author_name="піструнькиня",
            text="Я бы сама проехалась в этой маршрутке, но без кофе не выхожу!",
        ),
        ChatMessage(
            telegram_msg_id=302,
            chat_id=chat_id,
            user_id=891777515,
            timestamp=t0 + 2,
            author_name="Аліна",
            text="Сегодня маршрутки подорожали, теперь по 20 грн!",
        ),
        ChatMessage(
            telegram_msg_id=303,
            chat_id=chat_id,
            user_id=380227601,
            timestamp=t0 + 3,
            author_name="Женя",
            text="Любит проводить кулинарные эксперименты с пельменями",
        ),
        ChatMessage(
            telegram_msg_id=304,
            chat_id=chat_id,
            user_id=447164588,
            timestamp=t0 + 4,
            author_name="Анюта",
            text="Договорились вечером собраться играть в настолки",
        ),
        ChatMessage(
            telegram_msg_id=305,
            chat_id=chat_id,
            user_id=382962362,
            timestamp=t0 + 5,
            author_name="Родион",
            text="Навел немецкий порядок в таблицах и разобрал правила",
        ),
        ChatMessage(
            telegram_msg_id=306,
            chat_id=chat_id,
            user_id=676566529,
            timestamp=t0 + 6,
            author_name="Владос",
            text="Собрал новый компьютер и настроил вентиляторы",
        ),
        ChatMessage(
            telegram_msg_id=307,
            chat_id=chat_id,
            user_id=891777515,
            timestamp=t0 + 7,
            author_name="Аліна",
            text="Уехала проверять новый велосипедный маршрут",
        ),
        ChatMessage(
            telegram_msg_id=308,
            chat_id=chat_id,
            user_id=380227601,
            timestamp=t0 + 8,
            author_name="Женя",
            text="ахахахахахахахаха",
        ),
    ]
    for msg in batch_messages:
        await store.save_message(msg)

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=lambda _transcript: "",  # empty LLM output -> deterministic fallback
        threshold=8,
        batch_size=8,
        excluded_user_ids=[bot_id],
    )
    compacted = await engine.run_compaction_for_chat(chat_id)
    assert compacted == 8

    # Bot replies and laughter must never appear in stored facts
    conn = store._get_connection()
    stored = conn.execute("SELECT subject, fact FROM memory_facts").fetchall()
    assert all("піструнькиня" not in row[0] and str(bot_id) not in row[1] for row in stored)
    assert all("ахаха" not in row[1] for row in stored)

    alina_hits = await store.search_facts("маршрутка")
    assert len(alina_hits) >= 1
    assert any(f.subject == "Аліна" and "891777515" in f.fact for f in alina_hits)

    zhenya_hits = await store.search_facts("пельмени")
    assert len(zhenya_hits) >= 1
    assert any(f.subject == "Женя" and f.category == "friend_bio" for f in zhenya_hits)

    anyuta_hits = await store.search_facts("настолки")
    assert len(anyuta_hits) >= 1
    assert any(f.subject == "Анюта" and f.category == "agreement" for f in anyuta_hits)

    rodion_hits = await store.search_facts("немецкий порядок")
    assert len(rodion_hits) >= 1
    assert any(f.subject == "Родион" for f in rodion_hits)

    await store.close()


@pytest.mark.asyncio
async def test_compaction_skips_llm_when_batch_lacks_substantive_messages():
    """Verify bot/noise batches with <6 substantive human messages skip summarize_fn and mark compacted."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -1003845062564
    bot_id = 8710978944
    t0 = int(time.time())

    low_signal_batch = [
        ChatMessage(telegram_msg_id=1, chat_id=chat_id, user_id=bot_id, timestamp=t0 + 1, author_name="Bot", text="Очень длинный ответ самого бота который нужно пропустить"),
        ChatMessage(telegram_msg_id=2, chat_id=chat_id, user_id=bot_id, timestamp=t0 + 2, author_name="Bot", text="Ещё один длинный ответ самого бота в том же чате"),
        ChatMessage(telegram_msg_id=3, chat_id=chat_id, user_id=101, timestamp=t0 + 3, author_name="Аліна", text="Да"),
        ChatMessage(telegram_msg_id=4, chat_id=chat_id, user_id=102, timestamp=t0 + 4, author_name="Женя", text="Уже ок"),
        ChatMessage(telegram_msg_id=5, chat_id=chat_id, user_id=103, timestamp=t0 + 5, author_name="Родион", text="ахахахахахахахахаха"),
        ChatMessage(telegram_msg_id=6, chat_id=chat_id, user_id=104, timestamp=t0 + 6, author_name="Владос", text="😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂"),
        ChatMessage(telegram_msg_id=7, chat_id=chat_id, user_id=101, timestamp=t0 + 7, author_name="Аліна", text=None, media_summary="Стикер с котом"),
        ChatMessage(telegram_msg_id=8, chat_id=chat_id, user_id=102, timestamp=t0 + 8, author_name="Женя", text="Это единственное нормальное сообщение в пачке"),
    ]
    for m in low_signal_batch:
        await store.save_message(m)

    calls = 0

    async def fail_if_called(_transcript: str) -> str:
        nonlocal calls
        calls += 1
        return '[{"subject": "X", "fact": "Y", "category": "lore"}]'

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=fail_if_called,
        threshold=8,
        batch_size=8,
        excluded_user_ids=[bot_id],
    )
    compacted = await engine.run_compaction_for_chat(chat_id)
    assert compacted == 8
    assert calls == 0
    assert await store.get_uncompacted_messages(chat_id, limit=10) == []

    conn = store._get_connection()
    assert conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0] == 0

    await store.close()


@pytest.mark.asyncio
async def test_fact_fts_stopword_and_filler_suppression():
    """Verify low-signal conversational fillers return empty queries and zero fact hits."""
    assert build_fact_fts_query("Да") == ""
    assert build_fact_fts_query("Уже") == ""
    assert build_fact_fts_query("Ок") == ""
    assert build_fact_fts_query("Ну ага, вот это да! Шо чё как там она его её вона") == ""
    assert build_fact_fts_query("12345 67") == ""
    # Backward compatibility of build_fts_query
    assert '"я"' in build_fts_query("я и ты")

    store = MemoryStore(db_path=":memory:")
    await store.init_db()
    await store.add_fact(
        subject="Аліна",
        fact="Она уже давно там живёт, вот это да",
        category="friend_bio",
    )

    for filler in ("Да", "Уже", "Ок", "она", "его", "её", "вона", "там", "вот это"):
        assert await store.search_facts(filler) == []

    # Meaningful token still matches
    hits = await store.search_facts("живёт")
    assert len(hits) == 1
    assert hits[0].subject == "Аліна"

    await store.close()


@pytest.mark.asyncio
async def test_fact_fts_and_priority_with_or_fallback_and_exclude_ids():
    """Verify AND-first matching, OR fallback, and exclude_fact_ids cooldown filtering."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    FTSStore.init_schema(conn)

    id1 = FTSStore.add_fact(conn, subject="Аліна", fact="Ездит на маршрутке утром", category="friend_bio")
    id2 = FTSStore.add_fact(conn, subject="Женя", fact="Любит домашние пельмени с маслом", category="friend_bio")
    id3 = FTSStore.add_fact(conn, subject="Аліна", fact="Пила вкусный латте в кофейне", category="friend_bio")

    # 2+ meaningful tokens: AND matches only id1 (both Аліна and маршрутке), not id3
    hits_and = FTSStore.search_facts(conn, "Аліна маршрутка", limit=5)
    assert [h.id for h in hits_and] == [id1]

    # When AND has 0 results (e.g. terms across different facts), falls back to OR
    hits_or = FTSStore.search_facts(conn, "маршрутка пельмени", limit=5)
    assert {h.id for h in hits_or} == {id1, id2}

    # exclude_fact_ids skips cooled-down facts while still honoring limit
    hits_excluded = FTSStore.search_facts(conn, "Аліна", limit=5, exclude_fact_ids=[id1])
    assert [h.id for h in hits_excluded] == [id3]

    # When AND hit is excluded, OR fallback can return remaining matching facts
    hits_and_excluded = FTSStore.search_facts(
        conn,
        "Аліна маршрутка",
        limit=5,
        exclude_fact_ids=[id1],
    )
    assert [h.id for h in hits_and_excluded] == [id3]

    conn.close()


@pytest.mark.asyncio
async def test_memory_store_search_facts_forwards_exclude_fact_ids():
    """Verify async MemoryStore.search_facts forwards exclude_fact_ids to FTSStore."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    id1 = await store.add_fact(subject="Аліна", fact="Любит горный велосипед", category="friend_bio")
    id2 = await store.add_fact(subject="Женя", fact="Купил шоссейный велосипед", category="friend_bio")

    all_hits = await store.search_facts("велосипед", limit=5)
    assert {h.id for h in all_hits} == {id1, id2}

    filtered_hits = await store.search_facts("велосипед", limit=5, exclude_fact_ids=[id1])
    assert [h.id for h in filtered_hits] == [id2]

    await store.close()


@pytest.mark.asyncio
async def test_archive_query_and_messages_fts_search():
    """Verify stopword stripping and selective FTS5 archive search."""
    from bot.memory.fts import build_archive_fts_query

    assert build_archive_fts_query("привет как дела что тут") == ""
    q = build_archive_fts_query("помнишь стиралку на 500 оборотов?")
    assert '"стиралк"*' in q
    assert '"оборот"*' in q

    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    await store.bulk_save_archive_messages(
        [
            ChatMessage(
                telegram_msg_id=10,
                chat_id=-1003323178808,
                user_id=676566529,
                timestamp=1700000000,
                author_name="Vlados",
                text="Запустил легендарную стиралку на 500 оборотов в общаге",
                compacted=1,
            ),
            ChatMessage(
                telegram_msg_id=11,
                chat_id=-1003845062562,
                user_id=380227601,
                timestamp=1794590000,
                author_name="Женя",
                text="Написал гайд по рыбалке с опарышем в курином яйце",
                compacted=1,
            ),
            ChatMessage(
                telegram_msg_id=12,
                chat_id=-1003845062562,
                user_id=8710978944,
                timestamp=1794591000,
                author_name="піструнькиня",
                text="Стиралка лечит всё, кроме вайбкода",
                compacted=1,
            ),
        ]
    )

    # Should match historical messages and exclude bot user_id
    hits = await store.search_messages("что там со стиралкой?", exclude_user_ids={8710978944})
    assert len(hits) == 1
    assert hits[0].author_name == "Vlados"
    assert hits[0].chat_id == -1003323178808

    # Excludes messages already in short-term history by DB id
    all_recent = await store.get_recent_history(-1003845062562, limit=5)
    recent_ids = {m.id for m in all_recent if m.id is not None}
    assert await store.search_messages("про опарыша в яйце", exclude_db_ids=recent_ids) == []

    # Should produce zero hits for filler/small-talk
    assert await store.search_messages("привет как дела") == []

    await store.close()


@pytest.mark.asyncio
async def test_telegram_desktop_json_importer(tmp_path):
    """Verify Telegram Desktop result.json parsing, author normalization, and idempotency."""
    import json

    from bot.memory.importer import import_telegram_exports

    export_file = tmp_path / "result.json"
    payload = {
        "name": "Анальные козырьки",
        "type": "private_supergroup",
        "id": 3323178808,
        "messages": [
            {"id": -999, "type": "service", "date_unixtime": "1655300000", "text": ""},
            {
                "id": -998,
                "type": "message",
                "date_unixtime": "1655300010",
                "from": "Женьок",
                "from_id": "user676566529",
                "text": ["Смотри ", {"type": "bold", "text": "таро"}, " без свечки"],
            },
            {
                "id": -997,
                "type": "message",
                "date_unixtime": "1655300020",
                "from": "ОсуждаюАнтонаБот",
                "from_id": "user7872766132",
                "text": "Осуждаем такое в чате!",
            },
            {
                "id": -996,
                "type": "message",
                "date_unixtime": "1655300030",
                "from": "Rodion",
                "from_id": "user382962362",
                "text": "+",
            },
        ],
    }
    export_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    db_path = tmp_path / "archive.sqlite"
    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    # Pre-seed a live message with same telegram_msg_id to check ON CONFLICT DO NOTHING
    await store.save_message(
        ChatMessage(
            telegram_msg_id=500,
            chat_id=-1003323178808,
            user_id=382962362,
            timestamp=1700000000,
            author_name="Rodion",
            text="Живое сообщение не должно затереться",
            compacted=0,
        )
    )

    stats_list = await import_telegram_exports(
        store,
        [export_file],
        author_map={676566529: "Vlados"},
    )
    assert len(stats_list) == 1
    st = stats_list[0]
    assert st.chat_id == -1003323178808
    assert st.parsed == 1
    assert st.inserted == 1

    # Re-run must be idempotent
    stats_again = await import_telegram_exports(store, [export_file], author_map={676566529: "Vlados"})
    assert stats_again[0].inserted == 0

    # No uncompacted inflation for imported history
    uncompacted = await store.get_uncompacted_messages(-1003323178808, limit=10)
    assert len(uncompacted) == 1
    assert uncompacted[0].telegram_msg_id == 500

    # FTS5 archive search finds the normalized Vlados message
    hits = await store.search_messages("расклад таро без свечки")
    assert len(hits) == 1
    assert hits[0].author_name == "Vlados"

    await store.close()


@pytest.mark.asyncio
async def test_legacy_6_column_schema_migration_and_lifecycle_crud(tmp_path):
    """Verify legacy 6-column memory_facts migration, FTS5 sync, and embedding invalidation."""
    import sqlite3

    import numpy as np

    from bot.memory.embeddings import vector_to_bytes

    db_file = tmp_path / "legacy_memory.sqlite"
    raw_conn = sqlite3.connect(str(db_file))
    raw_conn.execute(
        """
        CREATE TABLE memory_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            fact TEXT NOT NULL,
            category TEXT NOT NULL,
            source_chat_id INTEGER,
            created_at INTEGER NOT NULL
        )
        """
    )
    raw_conn.execute(
        "INSERT INTO memory_facts (subject, fact, category, source_chat_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("Женя", "Купил красную электрогитару Fender", "friend_bio", -100111, 1700000100),
    )
    raw_conn.execute(
        "INSERT INTO memory_facts (subject, fact, category, source_chat_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("Аліна", "Планирует поездку во Львов на выходные", "event", -100111, 1700000200),
    )
    raw_conn.commit()
    raw_conn.close()

    store = MemoryStore(db_path=str(db_file))
    await store.init_db()
    # Second init_db must be completely idempotent
    await store.init_db()

    conn = store._get_connection()
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memory_facts)").fetchall()}
    assert {
        "id",
        "subject",
        "fact",
        "category",
        "source_chat_id",
        "created_at",
        "subject_user_id",
        "status",
        "updated_at",
        "expires_at",
        "superseded_by",
    } <= cols

    by_id = await store.get_facts_by_ids([1, 2])
    assert by_id[1].status == "active"
    assert by_id[1].updated_at == 1700000100
    assert by_id[2].status == "active"
    assert by_id[2].updated_at == 1700000200

    # FTS5 search works on pre-existing legacy rows rebuilt during init_schema
    legacy_hits = await store.search_facts("электрогитара", chat_id=-100111)
    assert [h.id for h in legacy_hits] == [1]

    vec_blob = vector_to_bytes(np.ones(64, dtype=np.float32))
    await store.upsert_fact_embedding(1, "gemini-embedding-001", 64, vec_blob, updated_at=1700000300)
    await store.upsert_fact_embedding(2, "gemini-embedding-001", 64, vec_blob, updated_at=1700000300)
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_fact_embeddings WHERE fact_id IN (1, 2)").fetchone()[0]
        == 2
    )

    # update_fact updates FTS5 and invalidates stale embedding when text changes
    ok_upd = await store.update_fact(
        1,
        fact="Продал старую гитару и купил синтезатор Yamaha",
        subject_user_id=380227601,
        updated_at=1700000500,
    )
    assert ok_upd is True
    assert await store.search_facts("электрогитара", chat_id=-100111) == []
    synth_hits = await store.search_facts("синтезатор", chat_id=-100111)
    assert [h.id for h in synth_hits] == [1]
    assert synth_hits[0].subject_user_id == 380227601
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_fact_embeddings WHERE fact_id = 1").fetchone()[0]
        == 0
    )

    # supersede_fact hides superseded fact from search_facts and deletes its embedding
    ok_sup = await store.supersede_fact(2, new_fact_id=1, status="superseded", updated_at=1700000600)
    assert ok_sup is True
    assert await store.search_facts("Львов", chat_id=-100111) == []
    sup_rec = (await store.get_facts_by_ids([2]))[2]
    assert sup_rec.status == "superseded"
    assert sup_rec.superseded_by == 1
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_fact_embeddings WHERE fact_id = 2").fetchone()[0]
        == 0
    )

    # DELETE trigger cleans memory_fact_embeddings even when foreign_keys = OFF
    conn.execute("PRAGMA foreign_keys = OFF")
    fid3 = await store.add_fact(
        subject="Владос",
        fact="Собрал кастомную механическую клавиатуру",
        category="friend_bio",
        source_chat_id=-100111,
    )
    await store.upsert_fact_embedding(fid3, "gemini-embedding-001", 64, vec_blob)
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_fact_embeddings WHERE fact_id = ?", (fid3,)).fetchone()[0]
        == 1
    )
    await store.delete_fact(fid3)
    assert (
        conn.execute("SELECT COUNT(*) FROM memory_fact_embeddings WHERE fact_id = ?", (fid3,)).fetchone()[0]
        == 0
    )

    await store.close()


@pytest.mark.asyncio
async def test_strict_chat_id_isolation_and_global_facts():
    """Verify DM and group facts never leak across chat_ids while global facts remain shared."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    dm_id = await store.add_fact(
        subject="Родион",
        fact="Личный секретный сервер стоит дома на балконе",
        category="friend_bio",
        source_chat_id=111,
    )
    group_a_id = await store.add_fact(
        subject="Женя",
        fact="Игровой сервер майнкрафт поднят в группе А",
        category="lore",
        source_chat_id=-100222,
    )
    group_b_id = await store.add_fact(
        subject="Саша",
        fact="Тестовый сервер аналитики крутится в группе Б",
        category="lore",
        source_chat_id=-100333,
    )
    global_id = await store.add_fact(
        subject="Group",
        fact="Общий резервный сервер мониторинга доступен всем",
        category="lore",
        source_chat_id=None,
    )

    hits_group_a = await store.search_facts("сервер", limit=10, chat_id=-100222)
    assert {h.id for h in hits_group_a} == {group_a_id, global_id}

    hits_dm = await store.search_facts("сервер", limit=10, chat_id=111)
    assert {h.id for h in hits_dm} == {dm_id, global_id}

    hits_group_b = await store.search_facts("сервер", limit=10, chat_id=-100333)
    assert {h.id for h in hits_group_b} == {group_b_id, global_id}

    # Backward-compatible unscoped search when chat_id=None
    hits_unscoped = await store.search_facts("сервер", limit=10, chat_id=None)
    assert {h.id for h in hits_unscoped} == {dm_id, group_a_id, group_b_id, global_id}

    await store.close()


