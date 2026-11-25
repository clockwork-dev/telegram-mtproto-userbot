"""Tests for Stage 1 canonical upsert compaction and Stage 3 reflection + DB normalizer."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

from bot.config.models import FriendProfile
from bot.memory.compaction import CompactionEngine, canonicalize_subject, strip_fact_id_tag
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.embeddings import vector_to_bytes
from bot.memory.reflection import (
    ReflectionEngine,
    build_reflection_prompt,
    normalize_existing_facts_in_db,
)


def test_canonicalize_subject_aliases_group_and_bot_rejection():
    friends_map = {
        380227601: FriendProfile(name="Женя", aliases=["Женьок", "Zhenya"]),
        891777515: FriendProfile(name="Аліна", aliases=["Алина", "alina_k"]),
    }
    author_map = {"владос": 676566529, "женьок": 380227601}
    bot_id = 8710978944
    bot_names = ["піструнькиня", "гермес", "Джефф"]

    # Strips embedded (ID:...) and extracts user_id
    assert canonicalize_subject("Женя (ID:380227601)") == ("Женя", 380227601)
    assert strip_fact_id_tag("Любит пельмени с уксусом (ID:380227601)") == (
        "Любит пельмени с уксусом",
        380227601,
    )

    # Resolves alias via friends_map
    assert canonicalize_subject(
        "Женьок",
        friends_map=friends_map,
        bot_id=bot_id,
        bot_names=bot_names,
    ) == ("Женя", 380227601)
    assert canonicalize_subject(
        "Алина",
        friends_map=friends_map,
        bot_id=bot_id,
        bot_names=bot_names,
    ) == ("Аліна", 891777515)

    # Resolves via author_id_by_name when not in friends_map
    assert canonicalize_subject(
        "Владос",
        friends_map=friends_map,
        author_id_by_name=author_map,
        bot_id=bot_id,
        bot_names=bot_names,
    ) == ("Владос", 676566529)

    # Group aliases normalize to ("Group", None)
    for grp in ("Group", "группа", "Чат", "general"):
        assert canonicalize_subject(grp, friends_map=friends_map) == ("Group", None)

    # Rejects self-bot facts by bot_id and Russian/Ukrainian pistrunkinya variants
    assert canonicalize_subject("піструнькиня", bot_id=bot_id, bot_names=bot_names) == (None, None)
    assert canonicalize_subject("пиструнькиня", bot_id=bot_id, bot_names=bot_names) == (None, None)
    assert canonicalize_subject("@гермес", bot_id=bot_id, bot_names=bot_names) == (None, None)
    assert canonicalize_subject("Женя", user_id=bot_id, bot_id=bot_id, bot_names=bot_names) == (
        None,
        None,
    )

    # Even when configured with Potuzhnych's MTProto persona names only, self and all sibling/legacy bots are rejected
    potuzhnych_names = ["Потужнич", "Потужный", "potuzhnych"]
    assert canonicalize_subject("Потужнич", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("@potuzhnych", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("Джефф", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("@jeff", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("hermes_kuk7yi6kfm3lcdfa_bot", bot_id=777000, bot_names=potuzhnych_names) == (
        None,
        None,
    )
    assert canonicalize_subject("піструнькиня", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("пиструнькиня", bot_id=777000, bot_names=potuzhnych_names) == (None, None)
    assert canonicalize_subject("піструн", bot_id=777000, bot_names=potuzhnych_names) == (None, None)


@pytest.mark.asyncio
async def test_compaction_engine_upsert_ttl_dedup_and_cross_chat_tamper_protection():
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -100500
    other_chat_id = -100999
    bot_id = 8710978944
    t0 = int(time.time())

    friends_map = {
        380227601: FriendProfile(name="Женя", aliases=["Женьок"]),
        891777515: FriendProfile(name="Аліна", aliases=["Алина"]),
    }

    # Existing active facts in chat_id
    id_to_update = await store.add_fact(
        subject="Женя",
        fact="Ищет новую квартиру на Подоле",
        category="event",
        source_chat_id=chat_id,
        subject_user_id=380227601,
    )
    id_to_supersede_plain = await store.add_fact(
        subject="Group",
        fact="Собираемся в пятницу в 19:00 в баре",
        category="agreement",
        source_chat_id=chat_id,
    )
    id_to_supersede_replace = await store.add_fact(
        subject="Аліна",
        fact="Работает стажёром в стартапе",
        category="friend_bio",
        source_chat_id=chat_id,
        subject_user_id=891777515,
    )
    id_existing_dup = await store.add_fact(
        subject="Женя",
        fact="Обожает острую шаурму",
        category="friend_bio",
        source_chat_id=chat_id,
        subject_user_id=380227601,
    )

    # Foreign fact in a DIFFERENT chat — LLM must NEVER be able to update or supersede this!
    foreign_fact_id = await store.add_fact(
        subject="Секрет",
        fact="Приватная заметка другого чата, неприкосновенна",
        category="friend_bio",
        source_chat_id=other_chat_id,
        subject_user_id=111222,
    )

    # Seed 6 substantive human messages so CompactionEngine runs LLM summarization
    for idx in range(6):
        uid = 380227601 if idx % 2 == 0 else 891777515
        author = "Женьок" if idx % 2 == 0 else "Алина"
        await store.save_message(
            ChatMessage(
                telegram_msg_id=10 + idx,
                chat_id=chat_id,
                user_id=uid,
                timestamp=t0 + idx,
                author_name=author,
                text=f"Содержательное обсуждение переезда и повышения в компании #{idx}",
            )
        )

    indexed_calls: List[tuple[int, str, str, str]] = []

    class FakeEmbedder:
        async def index_fact(
            self,
            _memory_store: Any,
            fact_id: int,
            subject: str,
            fact: str,
            category: str = "",
        ) -> bool:
            indexed_calls.append((fact_id, subject, fact, category))
            return True

    seen_existing_ids: List[int] = []

    async def upsert_summarizer(
        transcript: str,
        existing_facts: Sequence[Any] = (),
        chat_id: int | None = None,
    ) -> str:
        # Transcript uses canonical friend names
        assert "[Женя (ID:380227601)]" in transcript
        assert "[Аліна (ID:891777515)]" in transcript
        seen_existing_ids.extend(f.id for f in existing_facts)
        ops = [
            # 1. Update in-chat fact
            {
                "op": "update",
                "target_id": id_to_update,
                "subject": "Женьок (ID:380227601)",
                "fact": "Переехал в новую квартиру на Подоле",
                "category": "friend_bio",
            },
            # 2. Pure supersede without replacement text
            {
                "op": "supersede",
                "target_id": id_to_supersede_plain,
            },
            # 3. Supersede WITH replacement fact
            {
                "op": "supersede",
                "target_id": id_to_supersede_replace,
                "subject": "Алина",
                "fact": "Получила повышение до Middle Product Designer",
                "category": "friend_bio",
            },
            # 4. Add temporary agreement with ttl_days
            {
                "op": "add",
                "subject": "группа",
                "fact": "Договорились пойти в поход через неделю",
                "category": "agreement",
                "ttl_days": 7,
            },
            # 5. Duplicate of existing fact -> must be deduplicated
            {
                "op": "add",
                "subject": "Женьок",
                "fact": "Обожает острую шаурму",
                "category": "friend_bio",
            },
            # 6. Self-bot fact -> must be rejected
            {
                "op": "add",
                "subject": "піструнькиня",
                "fact": "Я самый ехидный бот чата",
                "category": "observation",
            },
            # 7. Malicious/hallucinated cross-chat update & supersede -> MUST BE BLOCKED!
            {
                "op": "update",
                "target_id": foreign_fact_id,
                "subject": "Взлом",
                "fact": "Попытка перезаписать чужой чат",
                "category": "lore",
            },
            {
                "op": "supersede",
                "target_id": foreign_fact_id,
            },
        ]
        return json.dumps(ops, ensure_ascii=False)

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=upsert_summarizer,
        threshold=6,
        batch_size=6,
        excluded_user_ids=[bot_id],
        friends_provider=lambda: friends_map,
        bot_names=["піструнькиня"],
        embedder=FakeEmbedder(),
        agreement_ttl_days=14,
        event_ttl_days=30,
        bot_id=bot_id,
    )

    compacted = await engine.run_compaction_for_chat(chat_id)
    assert compacted == 6
    assert id_to_update in seen_existing_ids
    assert foreign_fact_id not in seen_existing_ids

    # Foreign fact in other_chat_id is 100% untouched and still active!
    foreign_rec = (await store.get_facts_by_ids([foreign_fact_id]))[foreign_fact_id]
    assert foreign_rec.status == "active"
    assert foreign_rec.subject == "Секрет"
    assert foreign_rec.fact == "Приватная заметка другого чата, неприкосновенна"

    # Updated fact updated in place with canonical subject + uid
    upd_rec = (await store.get_facts_by_ids([id_to_update]))[id_to_update]
    assert upd_rec.status == "active"
    assert upd_rec.subject == "Женя"
    assert upd_rec.subject_user_id == 380227601
    assert upd_rec.fact == "Переехал в новую квартиру на Подоле"

    # Plain superseded fact marked superseded
    sup_plain_rec = (await store.get_facts_by_ids([id_to_supersede_plain]))[id_to_supersede_plain]
    assert sup_plain_rec.status == "superseded"

    # Replaced fact marked superseded with superseded_by pointing to new active fact
    sup_rep_rec = (await store.get_facts_by_ids([id_to_supersede_replace]))[id_to_supersede_replace]
    assert sup_rep_rec.status == "superseded"
    assert sup_rep_rec.superseded_by is not None
    new_alina_rec = (await store.get_facts_by_ids([sup_rep_rec.superseded_by]))[sup_rep_rec.superseded_by]
    assert new_alina_rec.status == "active"
    assert new_alina_rec.subject == "Аліна"
    assert new_alina_rec.subject_user_id == 891777515
    assert "Middle Product Designer" in new_alina_rec.fact

    # Active facts in chat_id contain no bot facts and no duplicate шаурма fact
    active_in_chat = await store.get_active_facts(chat_id=chat_id, limit=50)
    assert all("піструнькиня" not in f.subject.lower() for f in active_in_chat)
    shaurma_facts = [f for f in active_in_chat if "шаурму" in f.fact]
    assert [f.id for f in shaurma_facts] == [id_existing_dup]

    hike_facts = [f for f in active_in_chat if "поход" in f.fact]
    assert len(hike_facts) == 1
    assert hike_facts[0].subject == "Group"
    assert hike_facts[0].expires_at is not None
    assert hike_facts[0].expires_at >= t0 + 6 * 86400

    # Embedder indexed updated fact, replacement fact, and added hike fact
    assert len(indexed_calls) == 3

    await store.close()


@pytest.mark.asyncio
async def test_reflection_engine_ttl_archival_consolidation_cooldown_and_failsafe():
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    now = 1_800_100_000
    chat_id = -100600

    # 1. Seed expired / aged TTL facts + timeless bio fact
    explicit_exp_id = await store.add_fact(
        subject="Group",
        fact="Временный созвон в среду",
        category="agreement",
        source_chat_id=chat_id,
        created_at=now - 3600,
        expires_at=now - 10,
    )
    aged_agreement_id = await store.add_fact(
        subject="Group",
        fact="Старая договорённость двухнедельной давности",
        category="agreement",
        source_chat_id=chat_id,
        created_at=now - 15 * 86400,
    )
    aged_event_id = await store.add_fact(
        subject="Владос",
        fact="Месяц назад отключили свет на час",
        category="event",
        source_chat_id=chat_id,
        created_at=now - 31 * 86400,
        subject_user_id=676566529,
    )
    timeless_bio_id = await store.add_fact(
        subject="Владос",
        fact="Работает инженером-теплотехником",
        category="friend_bio",
        source_chat_id=chat_id,
        created_at=now - 90 * 86400,
        subject_user_id=676566529,
    )
    vec_blob = vector_to_bytes(np.ones(64, dtype=np.float32))
    await store.upsert_fact_embedding(aged_agreement_id, "gemini-embedding-001", 64, vec_blob)

    # Even when provider is rate-limited, TTL archival still runs!
    reflect_calls = 0

    async def dummy_reflect(**_kwargs: Any) -> Dict[str, Any]:
        nonlocal reflect_calls
        reflect_calls += 1
        return {}

    rate_limited_engine = ReflectionEngine(
        memory_store=store,
        reflect_fn=dummy_reflect,
        min_active_facts_per_subject=5,
        agreement_default_ttl_days=14,
        event_default_ttl_days=30,
    )
    rl_stats = await rate_limited_engine.run_cycle(
        is_rate_limited_fn=lambda: True,
        now_ts=now,
    )
    assert rl_stats == {
        "expired_archived": 3,
        "subjects_reflected": 0,
        "facts_synthesized": 0,
        "facts_archived": 0,
    }
    assert reflect_calls == 0

    fetched_ttl = await store.get_facts_by_ids(
        [explicit_exp_id, aged_agreement_id, aged_event_id, timeless_bio_id]
    )
    assert fetched_ttl[explicit_exp_id].status == "archived"
    assert fetched_ttl[aged_agreement_id].status == "archived"
    assert fetched_ttl[aged_event_id].status == "archived"
    assert fetched_ttl[timeless_bio_id].status == "active"

    # 2. Seed dense subject bucket (8 older coffee micro-facts + 1 older meme + 10 recent facts = 19 >= 15)
    zhenya_ids: List[int] = []
    for i in range(8):
        fid = await store.add_fact(
            subject="Женя",
            fact=f"Мелкий повторяющийся факт про кофе и обжарку #{i}",
            category="friend_bio",
            source_chat_id=chat_id,
            created_at=now - (300 - i),
            subject_user_id=380227601,
        )
        zhenya_ids.append(fid)
    unique_meme_id = await store.add_fact(
        subject="Женя",
        fact="Легендарный автор гайда про опарыша в курином яйце",
        category="lore",
        source_chat_id=chat_id,
        created_at=now - 200,
        subject_user_id=380227601,
    )
    recent_zhenya_ids: List[int] = []
    for r_idx in range(10):
        rfid = await store.add_fact(
            subject="Женя",
            fact=f"Свежий недавний факт про проекты Жени #{r_idx}",
            category="friend_bio",
            source_chat_id=chat_id,
            created_at=now - (50 - r_idx),
            subject_user_id=380227601,
        )
        recent_zhenya_ids.append(rfid)

    prompt_preview = build_reflection_prompt(
        "Женя",
        await store.get_active_facts(chat_id=chat_id, subject_user_ids=[380227601]),
        target_summary_facts=3,
    )
    assert "Женя" in prompt_preview
    assert f"[ID:{zhenya_ids[0]}]" in prompt_preview

    # First test fail-safe on malformed / empty synthesized output -> zero facts archived!
    bad_engine = ReflectionEngine(
        memory_store=store,
        reflect_fn=lambda **_kw: json.dumps(
            {"synthesized": [], "archived_ids": zhenya_ids}
        ),
        min_active_facts_per_subject=15,
    )
    bad_stats = await bad_engine.run_cycle(now_ts=now)
    assert bad_stats["subjects_reflected"] == 0
    assert bad_stats["facts_archived"] == 0
    for fid in zhenya_ids:
        assert (await store.get_facts_by_ids([fid]))[fid].status == "active"

    # Now run valid consolidation that also attempts out-of-bucket archival of timeless_bio_id & recent fact
    embedded_new_ids: List[int] = []

    class FakeEmbedder:
        async def index_fact(
            self,
            _store: Any,
            fact_id: int,
            subject: str,
            fact: str,
            category: str = "",
        ) -> bool:
            embedded_new_ids.append(fact_id)
            return True

    async def good_reflect(subject: str, facts: Sequence[Any], **_kw: Any) -> str:
        assert subject == "Женя"
        # Only the 9 older facts (8 coffee + 1 meme) are passed; all 10 newest facts are protected!
        assert len(facts) == 9
        passed_ids = {f.id for f in facts}
        assert passed_ids.isdisjoint(recent_zhenya_ids)
        return json.dumps(
            {
                "synthesized": [
                    {
                        "fact": "Серьёзно увлекается спешелти-кофе и альтернативной заваркой",
                        "category": "observation",
                    },
                    {
                        "fact": "Постоянно экспериментирует с зерном светлой обжарки",
                        "category": "observation",
                    },
                ],
                # Includes timeless_bio_id & recent_zhenya_ids[0] (out of batch!) — must be ignored!
                # Leaves unique_meme_id out of archived_ids so unique meme stays active!
                "archived_ids": [*zhenya_ids, recent_zhenya_ids[0], timeless_bio_id, 999999],
            },
            ensure_ascii=False,
        )

    good_engine = ReflectionEngine(
        memory_store=store,
        reflect_fn=good_reflect,
        min_active_facts_per_subject=15,
        max_subjects_per_cycle=2,
        target_summary_facts=3,
        subject_cooldown_hours=72.0,
        embedder=FakeEmbedder(),
    )
    stats1 = await good_engine.run_cycle(now_ts=now)
    assert stats1 == {
        "expired_archived": 0,
        "subjects_reflected": 1,
        "facts_synthesized": 2,
        "facts_archived": 8,
    }
    assert len(embedded_new_ids) == 2

    # Absorbed micro-facts are archived; unique meme, 10 newest facts & out-of-bucket Vlados bio stay active!
    after_map = await store.get_facts_by_ids(
        [*zhenya_ids, unique_meme_id, *recent_zhenya_ids, timeless_bio_id]
    )
    for fid in zhenya_ids:
        assert after_map[fid].status == "archived"
        assert after_map[fid].superseded_by == embedded_new_ids[0]
    assert after_map[unique_meme_id].status == "active"
    assert after_map[timeless_bio_id].status == "active"
    for rfid in recent_zhenya_ids:
        assert after_map[rfid].status == "active"

    active_zhenya = await store.get_active_facts(chat_id=chat_id, subject_user_ids=[380227601])
    assert len(active_zhenya) == 13  # 10 recent + 2 synthesized + 1 preserved meme

    # Add 8 more facts so older pool is eligible again, then verify 72h subject cooldown skips immediate cycle
    for j in range(8):
        await store.add_fact(
            subject="Женя",
            fact=f"Дополнительный факт после рефлексии #{j}",
            category="event",
            source_chat_id=chat_id,
            created_at=now + 10 + j,
            subject_user_id=380227601,
        )
    stats_cooldown = await good_engine.run_cycle(now_ts=now + 600)
    assert stats_cooldown["subjects_reflected"] == 0

    await store.close()


@pytest.mark.asyncio
async def test_reflection_engine_windowed_40_facts_preserves_newest_skips_observations_and_caps_archive():
    """A 40-fact bucket protects 10 newest + existing observations, slices <=20 oldest, and caps archive at 15."""
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    now = 1_800_500_000
    chat_id = -100601
    uid = 382962362

    # 2 oldest already-synthesized observations (must NOT be re-fed to routine reflection)
    obs_ids: List[int] = []
    for i in range(2):
        oid = await store.add_fact(
            subject="Rodion",
            fact=f"Уже синтезированное наблюдение #{i}",
            category="observation",
            source_chat_id=chat_id,
            created_at=now - 1000 + i,
            subject_user_id=uid,
        )
        obs_ids.append(oid)

    # 28 older raw micro-facts (created_at ascending from now-900 to now-873)
    older_raw_ids: List[int] = []
    for i in range(28):
        fid = await store.add_fact(
            subject="Rodion",
            fact=f"Старый сырой микро-факт про сервер и игры #{i}",
            category="friend_bio" if i % 2 == 0 else "lore",
            source_chat_id=chat_id,
            created_at=now - 900 + i,
            subject_user_id=uid,
        )
        older_raw_ids.append(fid)

    # 10 newest active facts (must ALWAYS remain untouched)
    newest_10_ids: List[int] = []
    for i in range(10):
        nfid = await store.add_fact(
            subject="Rodion",
            fact=f"Самый свежий факт текущей недели #{i}",
            category="friend_bio",
            source_chat_id=chat_id,
            created_at=now - 20 + i,
            subject_user_id=uid,
        )
        newest_10_ids.append(nfid)

    all_active_before = await store.get_active_facts(chat_id=chat_id, subject_user_ids=[uid], limit=100)
    assert len(all_active_before) == 40

    # 1. Anti-churn guard: if LLM only proposes archiving 1 fact (<2), skip without churning
    churn_engine = ReflectionEngine(
        memory_store=store,
        reflect_fn=lambda facts, **_kw: {
            "synthesized": [{"fact": "Одиночный пересказ", "category": "observation"}],
            "archived_ids": [facts[0].id],
        },
    )
    churn_stats = await churn_engine.run_cycle(now_ts=now)
    assert churn_stats["subjects_reflected"] == 0
    assert churn_stats["facts_archived"] == 0

    # 2. Windowed pass: receives strictly the 20 oldest non-observation facts and caps archived_ids at <= 15
    captured_batch_ids: List[int] = []

    async def windowed_reflect(subject: str, facts: Sequence[Any], **_kw: Any) -> Dict[str, Any]:
        assert subject == "Rodion"
        assert len(facts) == 20
        captured_batch_ids.extend(f.id for f in facts)
        return {
            "synthesized": [
                {"fact": "Поддерживает домашний сервер и инфраструктуру бота", "category": "observation"},
                {"fact": "Регулярно обсуждает кооперативные игры с друзьями", "category": "observation"},
                {"fact": "Следит за порядком в конфигах и автоматизации", "category": "observation"},
            ],
            # Try to archive all 20 batch facts + all 10 newest + both observations -> must cap at 15 from batch!
            "archived_ids": [*(f.id for f in facts), *newest_10_ids, *obs_ids],
        }

    engine = ReflectionEngine(
        memory_store=store,
        reflect_fn=windowed_reflect,
    )
    stats = await engine.run_cycle(now_ts=now)
    assert captured_batch_ids == older_raw_ids[:20]
    assert set(captured_batch_ids).isdisjoint(newest_10_ids)
    assert set(captured_batch_ids).isdisjoint(obs_ids)
    assert stats == {
        "expired_archived": 0,
        "subjects_reflected": 1,
        "facts_synthesized": 3,
        "facts_archived": 15,
    }

    after = await store.get_facts_by_ids([*obs_ids, *older_raw_ids, *newest_10_ids])
    for nid in newest_10_ids:
        assert after[nid].status == "active"
    for oid in obs_ids:
        assert after[oid].status == "active"
    for aid in older_raw_ids[:15]:
        assert after[aid].status == "archived"
    for uid_kept in older_raw_ids[15:]:
        assert after[uid_kept].status == "active"

    await store.close()


@pytest.mark.asyncio
async def test_normalize_existing_facts_in_db_merges_aliases_archives_bot_and_preserves_messages():
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    chat_id = -1003845062562
    bot_id = 8710978944
    friends_map = {
        380227601: FriendProfile(name="Женя", aliases=["Женьок"]),
    }

    # Historical messages used to infer user_id for authors not in friends.yaml
    await store.save_message(
        ChatMessage(
            telegram_msg_id=1,
            chat_id=chat_id,
            user_id=676566529,
            timestamp=1700000010,
            author_name="Владос",
            text="Настроил котёл отопления",
        )
    )
    await store.save_message(
        ChatMessage(
            telegram_msg_id=2,
            chat_id=chat_id,
            user_id=380227601,
            timestamp=1700000020,
            author_name="Женя",
            text="Варит пельмени",
        )
    )

    # Dirty facts simulating legacy production DB rows
    dup_tagged_id = await store.add_fact(
        subject="Женя (ID:380227601)",
        fact="Любит домашние пельмени (ID:380227601)",
        category="friend_bio",
        source_chat_id=chat_id,
        created_at=1700000100,
    )
    dup_clean_id = await store.add_fact(
        subject="Женьок",
        fact="Любит домашние пельмени",
        category="friend_bio",
        source_chat_id=chat_id,
        created_at=1700000200,
    )
    vlados_id = await store.add_fact(
        subject="Владос",
        fact="Разбирается в газовых котлах",
        category="friend_bio",
        source_chat_id=chat_id,
        created_at=1700000300,
    )
    bot_subj_id = await store.add_fact(
        subject="піструнькиня",
        fact="Я отвечаю голосовыми сообщениями",
        category="lore",
        source_chat_id=chat_id,
        created_at=1700000400,
    )
    bot_uid_id = await store.add_fact(
        subject="Гермес (ID:8710978944)",
        fact="Служебная реплика бота",
        category="lore",
        source_chat_id=chat_id,
        created_at=1700000500,
    )

    stats = await normalize_existing_facts_in_db(
        store,
        friends_map=friends_map,
        bot_id=bot_id,
        bot_names=["піструнькиня", "гермес"],
    )
    assert stats["scanned"] == 5
    assert stats["bot_archived"] == 2
    assert stats["duplicates_superseded"] == 1
    assert stats["updated"] == 2  # dup_clean_id ('Женьок' -> 'Женя'+uid) and vlados_id (+uid)

    all_rows = await store.get_facts_by_ids(
        [dup_tagged_id, dup_clean_id, vlados_id, bot_subj_id, bot_uid_id]
    )
    assert all_rows[bot_subj_id].status == "archived"
    assert all_rows[bot_uid_id].status == "archived"

    # Cleanest/newest row kept active and canonicalized; duplicate superseded pointing to kept
    assert all_rows[dup_clean_id].status == "active"
    assert all_rows[dup_clean_id].subject == "Женя"
    assert all_rows[dup_clean_id].subject_user_id == 380227601

    assert all_rows[dup_tagged_id].status == "superseded"
    assert all_rows[dup_tagged_id].superseded_by == dup_clean_id

    # Vlados subject_user_id inferred from messages table
    assert all_rows[vlados_id].status == "active"
    assert all_rows[vlados_id].subject == "Владос"
    assert all_rows[vlados_id].subject_user_id == 676566529

    # Messages table is untouched!
    conn = store._get_connection()
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert msg_count == 2

    await store.close()
