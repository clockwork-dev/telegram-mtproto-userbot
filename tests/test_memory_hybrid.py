"""Unit and integration tests for Gemini MemoryEmbedder and Hybrid FTS5 + Vector RRF recall."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from bot.config.models import MemoryEmbeddingsConfig
from bot.memory.db import MemoryStore
from bot.memory.embeddings import (
    MemoryEmbedder,
    bytes_to_vector,
    normalize_vector,
    should_run_semantic_lookup,
    vector_to_bytes,
)


def _basis_vec(dim: int, index: int, secondary_index: int | None = None, secondary_weight: float = 0.0) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    if secondary_index is not None:
        vec[secondary_index] = float(secondary_weight)
    return normalize_vector(vec)


def test_vector_helpers_normalize_and_roundtrip():
    raw = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
    normed = normalize_vector(raw)
    assert normed.dtype == np.float32
    assert np.allclose(normed, np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(float(np.linalg.norm(normed)), 1.0)

    # Zero and non-finite vectors return safe zero float32 vectors
    zeros = normalize_vector(np.zeros(8, dtype=np.float32))
    assert zeros.dtype == np.float32
    assert np.all(zeros == 0.0)

    nan_vec = normalize_vector(np.array([1.0, np.nan, 2.0], dtype=np.float32))
    assert np.all(nan_vec == 0.0)

    blob = vector_to_bytes(raw)
    assert len(blob) == 16
    restored = bytes_to_vector(blob, dim=4)
    assert restored.dtype == np.float32
    assert np.allclose(restored, normed)

    with pytest.raises(ValueError):
        bytes_to_vector(b"\x00\x01\x02")
    with pytest.raises(ValueError):
        bytes_to_vector(blob, dim=8)


def test_should_run_semantic_lookup_accepts_paraphrased_queries_and_rejects_noise():
    accepted = [
        "на чем он сейчас пишет?",
        "шо там по пятнице?",
        "куда переехал Женя?",
        "хто купив нову кавоварку?",
        "чем вообще занимается Родион?",
        "у него сильная аллергия",
        "Аліна записалась на марафон",
    ]
    for query in accepted:
        assert should_run_semantic_lookup(query) is True, f"Expected True for {query!r}"

    rejected = [
        "",
        "   ",
        "Привет",
        "Ок",
        "Да",
        "Ахахах",
        "Ахахахахахах",
        "вхахахахах лол",
        "Спокойной ночи",
        "Доброй ночи всем!",
        "/help",
        "/imagine красивый киберпанк город",
        "https://example.com/some/really/long/url?query=1",
        "😂😂😂🔥🔥🔥💯💯",
    ]
    for noise in rejected:
        assert should_run_semantic_lookup(noise) is False, f"Expected False for {noise!r}"


@pytest.mark.asyncio
async def test_memory_embedder_lru_cache_hot_reload_index_and_backfill(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Without injected client and without API key, embedder reports unavailable cleanly
    no_key_embedder = MemoryEmbedder(MemoryEmbeddingsConfig(gemini_api_key=None))
    assert no_key_embedder.is_available() is False
    assert await no_key_embedder.embed_query("на чем он сейчас пишет?") is None

    dim = 64
    calls: List[tuple[str, Any, Any, Any]] = []

    class FakeModels:
        def __init__(self) -> None:
            self.reject_config_kwarg = False

        def embed_content(self, *, model: str, contents: Any, **kwargs: Any) -> Any:
            if self.reject_config_kwarg and "config" in kwargs:
                raise TypeError("embed_content() got an unexpected keyword argument 'config'")
            cfg = kwargs.get("config")
            task_type = getattr(cfg, "task_type", None)
            out_dim = getattr(cfg, "output_dimensionality", None)
            calls.append((model, contents, task_type, out_dim))

            items = [contents] if isinstance(contents, str) else list(contents)
            embeddings = []
            for idx, text in enumerate(items):
                vec = _basis_vec(dim, (len(str(text)) + idx) % dim)
                embeddings.append(SimpleNamespace(values=vec.tolist()))
            return SimpleNamespace(embeddings=embeddings)

    fake_models = FakeModels()
    fake_client = SimpleNamespace(models=fake_models)

    cfg = MemoryEmbeddingsConfig(
        output_dimensionality=dim,
        query_cache_size=2,
        backfill_batch_size=10,
        cooldown_seconds=60.0,
    )
    embedder = MemoryEmbedder(cfg, client=fake_client)
    assert embedder.is_available() is True

    # 1. LRU cache deduplicates normalized repeated queries
    v1 = await embedder.embed_query("  На чем он   сейчас пишет? ")
    v2 = await embedder.embed_query("на чем он сейчас пишет?")
    assert v1 is not None and v2 is not None
    assert v1.shape == (dim,)
    assert np.allclose(v1, v2)
    assert len(calls) == 1
    assert calls[0][2] == "RETRIEVAL_QUERY"
    assert calls[0][3] == dim

    # Fill cache (size=2) and verify oldest eviction
    await embedder.embed_query("второй уникальный запрос?")
    assert len(calls) == 2
    # Touch first query so second becomes LRU
    await embedder.embed_query("на чем он сейчас пишет?")
    assert len(calls) == 2
    await embedder.embed_query("третий уникальный запрос?")
    assert len(calls) == 3
    # First query still cached; second was evicted
    await embedder.embed_query("на чем он сейчас пишет?")
    assert len(calls) == 3
    await embedder.embed_query("второй уникальный запрос?")
    assert len(calls) == 4

    # 2. Fallback when older SDK/mock rejects `config` kwarg
    fake_models.reject_config_kwarg = True
    v_legacy = await embedder.embed_query("четвертый запрос без конфига?")
    assert v_legacy is not None
    assert len(calls) == 5
    fake_models.reject_config_kwarg = False

    # 3. index_fact + backfill_missing against real in-memory MemoryStore
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    f1 = await store.add_fact("Женя", "Пишет сервисы на Rust и Go", "friend_bio", source_chat_id=-1001)
    f2 = await store.add_fact("Аліна", "Бегает полумарафоны по утрам", "observation", source_chat_id=-1001)
    f3 = await store.add_fact("Владос", "Устаревший факт", "event", source_chat_id=-1001)
    await store.supersede_fact(f3)

    indexed_one = await embedder.index_fact(store, f1, "Женя", "Пишет сервисы на Rust и Go", "friend_bio")
    assert indexed_one is True

    unembedded_before = await store.get_unembedded_active_facts(embedder.model, limit=10)
    assert [f.id for f in unembedded_before] == [f2]

    backfilled = await embedder.backfill_missing(store)
    assert backfilled == 1
    assert await store.get_unembedded_active_facts(embedder.model, limit=10) == []

    # Hot-reload shrinking cache size works cleanly
    embedder.update_config(
        MemoryEmbeddingsConfig(
            gemini_model="gemini-embedding-002",
            output_dimensionality=dim,
            query_cache_size=1,
        )
    )
    assert embedder.model == "gemini-embedding-002"
    assert len(embedder._query_cache) <= 1

    await store.close()


@pytest.mark.asyncio
async def test_memory_embedder_429_and_timeout_cooldown_with_secret_redaction(caplog):
    dummy_secret = "AIzaSyDummySecretKey12345678901234567"
    sensitive_query = "секретный приватный вопрос пользователя про ключи"
    sdk_calls = 0

    class FailingModels:
        def embed_content(self, *, model: str, contents: Any, **kwargs: Any) -> Any:
            nonlocal sdk_calls
            sdk_calls += 1
            raise RuntimeError(
                f"429 RESOURCE_EXHAUSTED: Quota exceeded for key={dummy_secret}"
            )

    embedder = MemoryEmbedder(
        MemoryEmbeddingsConfig(
            gemini_api_key=dummy_secret,
            output_dimensionality=64,
            cooldown_seconds=120.0,
        ),
        client=SimpleNamespace(models=FailingModels()),
    )

    with caplog.at_level(logging.WARNING, logger="bot.memory.embeddings"):
        res = await embedder.embed_query(sensitive_query)

    assert res is None
    assert sdk_calls == 1
    assert embedder.is_available() is False

    # While in cooldown, no additional SDK calls are made
    assert await embedder.embed_query("ещё один запрос во время кулдауна") is None
    assert sdk_calls == 1

    joined_logs = "\n".join(r.getMessage() for r in caplog.records)
    assert "[REDACTED]" in joined_logs
    assert dummy_secret not in joined_logs
    assert sensitive_query not in joined_logs

    # Timeout also triggers cooldown and returns None cleanly
    class SlowModels:
        def embed_content(self, *, model: str, contents: Any, **kwargs: Any) -> Any:
            import time as _time

            _time.sleep(0.6)
            return SimpleNamespace(embeddings=[])

    slow_embedder = MemoryEmbedder(
        MemoryEmbeddingsConfig(
            output_dimensionality=64,
            timeout_seconds=0.25,
            cooldown_seconds=90.0,
        ),
        client=SimpleNamespace(models=SlowModels()),
    )
    assert await slow_embedder.embed_query("медленный запрос к сети?") is None
    assert slow_embedder.is_available() is False


@pytest.mark.asyncio
async def test_embed_query_enforces_tight_timeout_while_embed_documents_allows_batch_scaling():
    """embed_query enforces strict timeout_seconds, whereas embed_documents scales timeout for batches."""
    import time as _time

    dim = 64

    class BatchLatencyModels:
        def embed_content(self, *, model: str, contents: Any, **kwargs: Any) -> Any:
            # 0.35s latency exceeds tight 0.22s query timeout, yet fits well inside batch timeout (>=15s)
            _time.sleep(0.35)
            items = [contents] if isinstance(contents, str) else list(contents)
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=_basis_vec(dim, idx % dim).tolist())
                    for idx in range(len(items))
                ]
            )

    cfg = MemoryEmbeddingsConfig(
        output_dimensionality=dim,
        timeout_seconds=0.22,
        cooldown_seconds=60.0,
    )

    # 1. Live query times out strictly at 0.22s so chat replies never stall
    query_embedder = MemoryEmbedder(cfg, client=SimpleNamespace(models=BatchLatencyModels()))
    assert await query_embedder.embed_query("быстрый вопрос в чате?") is None
    assert query_embedder.is_available() is False

    # 2. Background 25-document batch completes cleanly beyond timeout_seconds without cooldown
    batch_embedder = MemoryEmbedder(cfg, client=SimpleNamespace(models=BatchLatencyModels()))
    docs = [f"[friend_bio] Участник: факт номер {i}" for i in range(25)]
    assert batch_embedder._document_batch_timeout(len(docs)) >= 15.0 + 0.25 * 25
    vectors = await batch_embedder.embed_documents(docs)
    assert vectors is not None
    assert len(vectors) == 25
    assert batch_embedder.is_available() is True


@pytest.mark.asyncio
async def test_hybrid_rrf_semantic_recall_threshold_entity_recency_and_and_bonus():
    store = MemoryStore(db_path=":memory:")
    await store.init_db()

    dim = 64
    chat_id = -100777
    other_chat_id = -100888
    now = 1_800_000_000
    model = "gemini-embedding-001"

    # 1. Semantic-only recall (0 FTS stem overlap with query "на чем он сейчас пишет?")
    rust_id = await store.add_fact(
        subject="Женя",
        fact="Разрабатывает высоконагруженные микросервисы на Rust и Tokio",
        category="friend_bio",
        source_chat_id=chat_id,
        created_at=now - 3600,
        subject_user_id=380227601,
    )
    noise_id = await store.add_fact(
        subject="Владос",
        fact="Купил зимнюю резину для автомобиля",
        category="event",
        source_chat_id=chat_id,
        created_at=now - 3600,
        subject_user_id=676566529,
    )
    foreign_id = await store.add_fact(
        subject="Чужой",
        fact="Тоже программирует на Rust в другом чате",
        category="friend_bio",
        source_chat_id=other_chat_id,
        created_at=now - 100,
        subject_user_id=999999,
    )

    q_rust = _basis_vec(dim, 0)
    vec_high_sim = _basis_vec(dim, 0, secondary_index=1, secondary_weight=0.2)  # cos ~ 0.98
    vec_low_sim = _basis_vec(dim, 1, secondary_index=0, secondary_weight=0.2)   # cos ~ 0.196 < 0.52

    await store.upsert_fact_embedding(rust_id, model, dim, vector_to_bytes(vec_high_sim), updated_at=now)
    await store.upsert_fact_embedding(noise_id, model, dim, vector_to_bytes(vec_low_sim), updated_at=now)
    await store.upsert_fact_embedding(foreign_id, model, dim, vector_to_bytes(q_rust), updated_at=now)

    # Pure FTS finds nothing for "на чем он сейчас пишет?" because stem "пишет" is absent
    fts_only = await store.search_facts("на чем он сейчас пишет?", limit=5, chat_id=chat_id, now_ts=now)
    assert fts_only == []

    # Hybrid vector RRF recalls rust_id, filters out low-sim noise_id (< 0.52), and isolates foreign_id!
    hybrid_hits = await store.search_facts(
        "на чем он сейчас пишет?",
        limit=5,
        chat_id=chat_id,
        query_vector=q_rust,
        embedding_model=model,
        min_vector_similarity=0.52,
        now_ts=now,
    )
    assert [h.id for h in hybrid_hits] == [rust_id]

    # When query vector only matches low-similarity noise (< min_vector_similarity), returns []
    q_unmatched = _basis_vec(dim, 5)
    assert (
        await store.search_facts(
            "на чем он сейчас пишет?",
            limit=5,
            chat_id=chat_id,
            query_vector=q_unmatched,
            embedding_model=model,
            min_vector_similarity=0.52,
            now_ts=now,
        )
        == []
    )

    # 2. Entity boost (author_user_id / target_user_ids) and recency tie-breaking
    tie_chat = -100778
    f_old_user_a = await store.add_fact(
        subject="Саша",
        fact="Предпочитает фильтр-кофе из Эфиопии",
        category="friend_bio",
        source_chat_id=tie_chat,
        created_at=now - 40 * 86400,
        subject_user_id=1001,
    )
    f_new_user_b = await store.add_fact(
        subject="Дима",
        fact="Предпочитает фильтр-кофе из Колумбии",
        category="friend_bio",
        source_chat_id=tie_chat,
        created_at=now - 60,
        subject_user_id=1002,
    )

    # Without entity boost, fresher f_new_user_b wins on recency tie-breaker
    recency_hits = await store.search_facts("фильтр кофе", limit=2, chat_id=tie_chat, now_ts=now)
    assert [h.id for h in recency_hits] == [f_new_user_b, f_old_user_a]

    # With author_user_id=1001 (+0.0025 entity boost > recency delta), Саша's fact ranks first
    entity_hits = await store.search_facts(
        "фильтр кофе",
        limit=2,
        chat_id=tie_chat,
        author_user_id=1001,
        now_ts=now,
    )
    assert [h.id for h in entity_hits] == [f_old_user_a, f_new_user_b]

    # Same boost via target_user_ids (e.g. reply target)
    target_hits = await store.search_facts(
        "фильтр кофе",
        limit=2,
        chat_id=tie_chat,
        target_user_ids=[1001],
        now_ts=now,
    )
    assert [h.id for h in target_hits] == [f_old_user_a, f_new_user_b]

    # 3. Multi-term FTS AND bonus (+0.004) beats partial single-term OR hit even in hybrid mode
    and_chat = -100779
    exact_and_id = await store.add_fact(
        subject="Родион",
        fact="Настроил домашний кластер Kubernetes",
        category="friend_bio",
        source_chat_id=and_chat,
        created_at=now - 10 * 86400,
    )
    partial_or_id = await store.add_fact(
        subject="Владос",
        fact="Настроил старый роутер",
        category="event",
        source_chat_id=and_chat,
        created_at=now - 10,
    )
    same_vec = _basis_vec(dim, 12)
    await store.upsert_fact_embedding(exact_and_id, model, dim, vector_to_bytes(same_vec), updated_at=now)
    await store.upsert_fact_embedding(partial_or_id, model, dim, vector_to_bytes(same_vec), updated_at=now)

    hybrid_and_hits = await store.search_facts(
        "настроил кластер",
        limit=2,
        chat_id=and_chat,
        query_vector=same_vec,
        embedding_model=model,
        now_ts=now,
    )
    assert [h.id for h in hybrid_and_hits] == [exact_and_id, partial_or_id]

    await store.close()
