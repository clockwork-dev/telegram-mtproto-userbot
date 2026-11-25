"""Memory Subsystem Package."""

from typing import Any

from bot.memory.compaction import CompactionEngine, format_transcript_for_compaction
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import (
    FactRecord,
    FTSStore,
    build_archive_fts_query,
    build_fact_fts_query,
    build_fts_query,
    extract_archive_terms,
    extract_fact_terms,
    stem_russian_token,
)

__all__ = [
    "ChatMessage",
    "FactRecord",
    "MemoryStore",
    "FTSStore",
    "CompactionEngine",
    "build_fts_query",
    "build_fact_fts_query",
    "build_archive_fts_query",
    "extract_fact_terms",
    "extract_archive_terms",
    "stem_russian_token",
    "format_transcript_for_compaction",
    "MemoryEmbedder",
    "ReflectionEngine",
    "canonicalize_subject",
    "normalize_existing_facts_in_db",
    "should_run_semantic_lookup",
    "normalize_vector",
    "vector_to_bytes",
    "bytes_to_vector",
]

_EMBEDDINGS_EXPORTS = frozenset(
    {
        "MemoryEmbedder",
        "should_run_semantic_lookup",
        "normalize_vector",
        "vector_to_bytes",
        "bytes_to_vector",
    }
)

_REFLECTION_EXPORTS = frozenset(
    {
        "ReflectionEngine",
        "normalize_existing_facts_in_db",
    }
)


def __getattr__(name: str) -> Any:
    if name in _EMBEDDINGS_EXPORTS:
        from bot.memory import embeddings as _emb

        return getattr(_emb, name)
    if name in _REFLECTION_EXPORTS:
        from bot.memory import reflection as _refl

        return getattr(_refl, name)
    if name == "canonicalize_subject":
        from bot.memory import compaction as _comp

        return getattr(_comp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
