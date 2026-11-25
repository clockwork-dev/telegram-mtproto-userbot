"""Gemini semantic embeddings engine for hybrid long-term memory recall.

Provides float32 vector normalization/serialization helpers, selective semantic
lookup gating for conversational Russian/Ukrainian/English messages, and an
async `MemoryEmbedder` with LRU query caching, timeout protection, rate-limit
cooldowns, and secret-sanitized diagnostics.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

from bot.config.models import MemoryEmbeddingsConfig
from bot.memory.fts import _FACT_STOPWORDS
from bot.security.sanitizer import SecretSanitizer

logger = logging.getLogger(__name__)

_URL_RE = re.compile(
    r"(?:https?://|www\.|t\.me/)[^\s<>()\[\]]+(?:\([^\s<>()]*\))?",
    re.IGNORECASE,
)
_WORD_TOKEN_RE = re.compile(r"[^\W_]+")
_LAUGHTER_TOKEN_RE = re.compile(
    r"^[авпхеиоaehio]*(?:ах|ха|хе|хи|пх|вх|ha|ah|he){2,}[авпхеиоaehio]*$",
    re.IGNORECASE,
)

_SEMANTIC_STOPWORDS: frozenset[str] = _FACT_STOPWORDS | frozenset(
    {
        # Additional RU/UK/EN greetings, farewells, pronouns, and conversational fillers
        "він",
        "здравствуй",
        "здравствуйте",
        "приветик",
        "приветики",
        "привітик",
        "здаров",
        "салют",
        "хай",
        "хелло",
        "хеллоу",
        "доброй",
        "доброго",
        "добрий",
        "добрая",
        "добрые",
        "утра",
        "ранку",
        "ранок",
        "вечера",
        "вечір",
        "вечора",
        "день",
        "дня",
        "ночки",
        "ночь",
        "ніч",
        "ночі",
        "спокійної",
        "добраніч",
        "сладких",
        "снов",
        "сни",
        "приятного",
        "аппетита",
        "смачного",
        "хорошего",
        "хорошей",
        "спасибки",
        "пасиб",
        "сяб",
        "спс",
        "дякс",
        "плиз",
        "плз",
        "соррян",
        "сорян",
        "извини",
        "извините",
        "вибач",
        "вибачте",
        "перепрошую",
        "ладненько",
        "понятненько",
        "ясненько",
        "окейси",
        "чатик",
        "согласен",
        "согласна",
        "точно",
        "верно",
        "реально",
        "серьезно",
        "серьёзно",
        "круто",
        "классно",
        "класс",
        "супер",
        "кайф",
        "имба",
        "жиза",
        "база",
        "кринж",
        "жесть",
        "треш",
        "трэш",
        "ору",
        "орнул",
        "орнула",
        "ржу",
        "смешно",
        "прикол",
        "прикольно",
    }
)

_COOLDOWN_ERROR_MARKERS: tuple[str, ...] = (
    "429",
    "quota",
    "resource_exhausted",
    "rate",
    "too many requests",
    "timeout",
    "timed out",
    "deadline",
    "network",
    "connect",
    "connection",
    "unavailable",
    "500",
    "502",
    "503",
    "504",
    "401",
    "403",
    "api_key",
    "unauthenticated",
    "permission_denied",
)


def _redact_secret_text(
    text: Any,
    explicit_secrets: Sequence[Optional[str]] = (),
) -> str:
    """Redact API keys and credential patterns before logging any error text."""
    secrets = [s.strip() for s in explicit_secrets if isinstance(s, str) and s.strip()]
    sanitizer = SecretSanitizer(extra_secrets=secrets, load_environ=True)
    sanitized = sanitizer.sanitize(text)
    collapsed = re.sub(r"\s+", " ", sanitized).strip()
    if len(collapsed) > 240:
        return f"{collapsed[:240]}..."
    return collapsed


def _create_genai_client(api_key: str, timeout_sec: float) -> Any:
    """Instantiate google.genai.Client, passing timeout HttpOptions when supported."""
    if genai is None:
        raise RuntimeError(
            "google-genai package is not installed; cannot create Gemini embeddings client."
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if genai_types is not None and hasattr(genai_types, "HttpOptions"):
        try:
            sig = inspect.signature(genai.Client)
            accepts_http_options = "http_options" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_http_options = True
        if accepts_http_options:
            http_timeout_sec = max(timeout_sec * 10.0, 45.0)
            kwargs["http_options"] = genai_types.HttpOptions(
                api_version="v1beta",
                timeout=max(1000, int(http_timeout_sec * 1000)),
            )
    return genai.Client(**kwargs)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a 1D L2-normalized float32 vector (or safe zero vector if norm == 0)."""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return np.zeros_like(arr, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / np.float32(norm)).astype(np.float32, copy=False)


def vector_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize an L2-normalized float32 vector to raw little-endian bytes."""
    return normalize_vector(vec).astype(np.float32, copy=False).tobytes()


def bytes_to_vector(blob: bytes, dim: Optional[int] = None) -> np.ndarray:
    """Deserialize raw float32 bytes into a 1D numpy float32 vector."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected bytes-like object for vector blob, got {type(blob).__name__}")
    raw = memoryview(blob)
    if raw.nbytes % 4 != 0:
        raise ValueError(f"Invalid float32 vector blob byte length: {raw.nbytes}")
    vec = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and vec.shape[0] != dim:
        raise ValueError(f"Vector dimension mismatch: expected {dim}, got {vec.shape[0]}")
    return vec


def _is_laughter_token(token: str) -> bool:
    """Return True if token is conversational laughter filler (e.g. 'ахахах', 'хаха', 'hahaha')."""
    if not token:
        return False
    lower = token.lower()
    if lower in {"лол", "кек", "ахах", "хаха", "хехе", "хихи", "lol", "lmao", "rofl", "haha"}:
        return True
    if _LAUGHTER_TOKEN_RE.fullmatch(lower):
        return True
    if len(lower) >= 4 and set(lower) <= {"х", "а"}:
        return True
    return False


def should_run_semantic_lookup(text: str) -> bool:
    """Return True when message text is substantive enough to warrant semantic vector lookup.

    Requires:
    - Stripped text length >= 8
    - Not a slash command, pure URL, pure emoji/punctuation, or pure laughter/greeting
    - Contains:
      * >= 2 non-stopword tokens of >= 3 letters, OR
      * '?' in text and >= 1 non-stopword token of >= 3 letters, OR
      * >= 1 non-stopword token of >= 5 letters alongside at least one other word.
    """
    if not text or not isinstance(text, str):
        return False

    stripped = text.strip()
    if len(stripped) < 8:
        return False

    if stripped.startswith("/"):
        return False

    cleaned = _URL_RE.sub(" ", stripped).strip()
    if not cleaned:
        return False

    raw_tokens = _WORD_TOKEN_RE.findall(cleaned.lower())
    words: list[tuple[str, str]] = []
    for tok in raw_tokens:
        letters = "".join(ch for ch in tok if ch.isalpha())
        if not letters:
            continue
        if _is_laughter_token(letters):
            continue
        words.append((tok, letters))

    if not words:
        return False

    content_3: list[str] = []
    content_5: list[str] = []
    for tok, letters in words:
        if tok in _SEMANTIC_STOPWORDS or letters in _SEMANTIC_STOPWORDS:
            continue
        if len(letters) >= 3:
            content_3.append(letters)
        if len(letters) >= 5:
            content_5.append(letters)

    if len(content_3) >= 2:
        return True
    if "?" in cleaned and len(content_3) >= 1:
        return True
    if len(content_5) >= 1 and len(words) >= 2:
        return True
    return False


def _format_fact_document(subject: str, fact: str, category: str = "") -> str:
    """Format a memory fact into a clean text string for document embedding."""
    subj_clean = (subject or "").strip()
    fact_clean = (fact or "").strip()
    cat_clean = (category or "").strip()
    if cat_clean and subj_clean:
        return f"[{cat_clean}] {subj_clean}: {fact_clean}"
    if subj_clean:
        return f"{subj_clean}: {fact_clean}"
    if cat_clean:
        return f"[{cat_clean}] {fact_clean}"
    return fact_clean


def _extract_vectors_from_response(response: Any) -> Optional[List[np.ndarray]]:
    """Extract normalized float32 vectors from a google-genai EmbedContentResponse or mock."""
    if response is None:
        return None

    raw_embeddings = getattr(response, "embeddings", None)
    if raw_embeddings is None and isinstance(response, dict):
        raw_embeddings = response.get("embeddings")

    if raw_embeddings is None:
        single = getattr(response, "embedding", None)
        if single is None and isinstance(response, dict):
            single = response.get("embedding")
        if single is not None:
            raw_embeddings = [single]

    if not raw_embeddings:
        return None

    vectors: List[np.ndarray] = []
    for item in raw_embeddings:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if values is None and isinstance(item, (list, tuple, np.ndarray)):
            values = item
        if values is None:
            return None
        vec = normalize_vector(np.asarray(values, dtype=np.float32))
        if vec.size == 0 or not np.any(vec):
            return None
        vectors.append(vec)
    return vectors


def _should_open_cooldown(exc: BaseException) -> bool:
    """Return True when an embedding error represents quota/rate-limit/network unavailability."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True

    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500 or status_code in {401, 403, 408}):
        return True

    mod_name = type(exc).__module__ or ""
    if mod_name.startswith(("httpx", "httpcore")):
        return True

    probe = f"{type(exc).__name__} {exc}".lower()
    return any(marker in probe for marker in _COOLDOWN_ERROR_MARKERS)


class MemoryEmbedder:
    """Async Gemini embedding wrapper with bounded LRU query cache and cooldown protection."""

    def __init__(
        self,
        config: Optional[MemoryEmbeddingsConfig] = None,
        *,
        client: Optional[Any] = None,
    ) -> None:
        cfg = config if config is not None else MemoryEmbeddingsConfig()
        self._injected_client: Optional[Any] = client
        self._client: Optional[Any] = client
        self._client_signature: Optional[tuple[str, str, float]] = None
        self._cooldown_until: float = 0.0
        self._query_cache: OrderedDict[tuple[str, int, str], np.ndarray] = OrderedDict()
        self._apply_config_fields(cfg)

    def _apply_config_fields(self, config: MemoryEmbeddingsConfig) -> None:
        self.config: MemoryEmbeddingsConfig = config
        self.enabled: bool = bool(config.enabled)
        self.provider: str = str(config.provider)
        self.model: str = str(config.gemini_model)
        self.gemini_model: str = self.model
        self.gemini_api_key_env: str = str(config.gemini_api_key_env)
        self.gemini_api_key: Optional[str] = config.gemini_api_key
        self.output_dimensionality: int = int(config.output_dimensionality)
        self.timeout_seconds: float = float(config.timeout_seconds)
        self.min_similarity: float = float(config.min_similarity)
        self.query_cache_size: int = int(config.query_cache_size)
        self.backfill_batch_size: int = int(config.backfill_batch_size)
        self.cooldown_seconds: float = float(config.cooldown_seconds)

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve Gemini API key from explicit config or configured environment variable."""
        if isinstance(self.gemini_api_key, str) and self.gemini_api_key.strip():
            return self.gemini_api_key.strip()
        if self.gemini_api_key_env:
            env_val = os.environ.get(self.gemini_api_key_env)
            if isinstance(env_val, str) and env_val.strip():
                return env_val.strip()
        return None

    def _close_owned_client(self) -> None:
        if self._client is not None and self._client is not self._injected_client:
            close_fn = getattr(self._client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        self._client = self._injected_client
        self._client_signature = None

    def update_config(self, config: MemoryEmbeddingsConfig) -> None:
        """Hot-reload configuration and reset cached client if model, key, or timeout changed."""
        old_model = self.model
        old_key = self._resolve_api_key() or ""
        old_timeout = self.timeout_seconds

        self._apply_config_fields(config)
        new_key = self._resolve_api_key() or ""

        if (self.model, new_key, self.timeout_seconds) != (old_model, old_key, old_timeout):
            self._close_owned_client()
            if (self.model, new_key) != (old_model, old_key):
                self._cooldown_until = 0.0

        if self.query_cache_size <= 0:
            self._query_cache.clear()
        else:
            while len(self._query_cache) > self.query_cache_size:
                self._query_cache.popitem(last=False)

    def is_available(self) -> bool:
        """Return True when embeddings are enabled, backed by SDK/client + key, and not in cooldown."""
        if not self.enabled:
            return False
        if time.monotonic() < self._cooldown_until:
            return False
        if self._injected_client is not None:
            return True
        if genai is None:
            return False
        return self._resolve_api_key() is not None

    def _get_or_create_client(self) -> Optional[Any]:
        if self._injected_client is not None:
            return self._injected_client
        api_key = self._resolve_api_key()
        if not api_key or genai is None:
            return None
        sig = (self.model, api_key, self.timeout_seconds)
        if self._client is None or self._client_signature != sig:
            self._close_owned_client()
            self._client = _create_genai_client(api_key=api_key, timeout_sec=self.timeout_seconds)
            self._client_signature = sig
        return self._client

    def _call_embed_content_sync(
        self,
        client: Any,
        contents: Any,
        task_type: str,
    ) -> Any:
        embed_cfg: Any
        if genai_types is not None and hasattr(genai_types, "EmbedContentConfig"):
            embed_cfg = genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dimensionality,
            )
        else:
            embed_cfg = {
                "task_type": task_type,
                "output_dimensionality": self.output_dimensionality,
            }

        try:
            return client.models.embed_content(
                model=self.model,
                contents=contents,
                config=embed_cfg,
            )
        except TypeError:
            return client.models.embed_content(
                model=self.model,
                contents=contents,
            )

    def _document_batch_timeout(self, batch_size: int) -> float:
        """Compute scaled timeout for background document embedding batches."""
        return max(self.timeout_seconds * 5.0, 15.0 + 0.25 * max(1, int(batch_size)))

    def _handle_embed_failure(
        self,
        exc: BaseException,
        operation: str,
        *,
        timeout_used: Optional[float] = None,
    ) -> None:
        resolved_key = self._resolve_api_key()
        secrets = [resolved_key] if resolved_key else ()
        eff_timeout = timeout_used if timeout_used is not None else self.timeout_seconds
        if isinstance(exc, asyncio.TimeoutError):
            safe_reason = f"TimeoutError after {eff_timeout:.1f}s"
        else:
            safe_reason = f"{type(exc).__name__}: {_redact_secret_text(exc, explicit_secrets=secrets)}"

        if _should_open_cooldown(exc) and self.cooldown_seconds > 0:
            self._cooldown_until = time.monotonic() + self.cooldown_seconds
            logger.warning(
                "Gemini embeddings %s failed (%s); entering %.0fs cooldown (model=%s).",
                operation,
                safe_reason,
                self.cooldown_seconds,
                self.model,
            )
        else:
            logger.debug(
                "Gemini embeddings %s failed (%s, model=%s).",
                operation,
                safe_reason,
                self.model,
            )

    async def embed_query(self, text: str) -> Optional[np.ndarray]:
        """Embed a search query using RETRIEVAL_QUERY task type and bounded LRU cache."""
        if not text or not isinstance(text, str):
            return None
        cleaned = text.strip()
        if not cleaned or not self.is_available():
            return None

        normalized_key = re.sub(r"\s+", " ", cleaned).lower()
        cache_key = (self.model, self.output_dimensionality, normalized_key)
        if self.query_cache_size > 0 and cache_key in self._query_cache:
            self._query_cache.move_to_end(cache_key)
            return self._query_cache[cache_key].copy()

        client = self._get_or_create_client()
        if client is None:
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_embed_content_sync,
                    client,
                    cleaned,
                    "RETRIEVAL_QUERY",
                ),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            self._handle_embed_failure(exc, "query")
            return None

        vectors = _extract_vectors_from_response(response)
        if not vectors:
            logger.debug("Gemini embeddings query returned no valid vector (model=%s).", self.model)
            return None

        vec = vectors[0]
        if self.query_cache_size > 0:
            self._query_cache[cache_key] = vec
            self._query_cache.move_to_end(cache_key)
            while len(self._query_cache) > self.query_cache_size:
                self._query_cache.popitem(last=False)

        return vec.copy()

    async def embed_documents(self, texts: Sequence[str]) -> Optional[List[np.ndarray]]:
        """Batch embed document strings using RETRIEVAL_DOCUMENT task type."""
        if not texts:
            return []
        if not self.is_available():
            return None

        docs = [str(t).strip() for t in texts]
        if any(not d for d in docs):
            return None

        client = self._get_or_create_client()
        if client is None:
            return None

        batch_timeout = self._document_batch_timeout(len(docs))
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._call_embed_content_sync,
                    client,
                    docs,
                    "RETRIEVAL_DOCUMENT",
                ),
                timeout=batch_timeout,
            )
        except Exception as exc:
            self._handle_embed_failure(exc, "document_batch", timeout_used=batch_timeout)
            return None

        vectors = _extract_vectors_from_response(response)
        if not vectors or len(vectors) != len(docs):
            logger.debug(
                "Gemini embeddings document_batch returned %d vectors for %d inputs (model=%s).",
                len(vectors) if vectors else 0,
                len(docs),
                self.model,
            )
            return None

        return vectors

    async def index_fact(
        self,
        memory_store: Any,
        fact_id: int,
        subject: str,
        fact: str,
        category: str = "",
    ) -> bool:
        """Embed a single memory fact and persist its vector in `memory_store`."""
        try:
            fid = int(fact_id)
        except (TypeError, ValueError):
            return False
        if fid <= 0 or not (fact or "").strip():
            return False
        if not self.is_available() or not hasattr(memory_store, "upsert_fact_embedding"):
            return False

        doc = _format_fact_document(subject, fact, category)
        vectors = await self.embed_documents([doc])
        if not vectors or len(vectors) != 1:
            return False

        vec = normalize_vector(vectors[0])
        if vec.size == 0 or not np.any(vec):
            return False

        try:
            await memory_store.upsert_fact_embedding(
                fid,
                self.model,
                int(vec.shape[0]),
                vector_to_bytes(vec),
            )
            return True
        except Exception as exc:
            resolved_key = self._resolve_api_key()
            secrets = [resolved_key] if resolved_key else ()
            logger.debug(
                "Failed to persist embedding for fact_id=%d (%s: %s).",
                fid,
                type(exc).__name__,
                _redact_secret_text(exc, explicit_secrets=secrets),
            )
            return False

    async def backfill_missing(
        self,
        memory_store: Any,
        *,
        limit: Optional[int] = None,
        chat_id: Optional[int] = None,
    ) -> int:
        """Fetch unembedded active facts and batch-index their embeddings."""
        if not self.is_available():
            return 0
        if not hasattr(memory_store, "get_unembedded_active_facts") or not hasattr(
            memory_store, "upsert_fact_embedding"
        ):
            return 0

        batch_limit = (
            int(limit) if (limit is not None and int(limit) > 0) else self.backfill_batch_size
        )
        try:
            unembedded = await memory_store.get_unembedded_active_facts(
                self.model,
                limit=batch_limit,
                chat_id=chat_id,
            )
        except Exception as exc:
            resolved_key = self._resolve_api_key()
            secrets = [resolved_key] if resolved_key else ()
            logger.debug(
                "Failed to query unembedded active facts (%s: %s).",
                type(exc).__name__,
                _redact_secret_text(exc, explicit_secrets=secrets),
            )
            return 0

        if not unembedded:
            return 0

        valid_pairs: list[tuple[int, str]] = []
        for rec in unembedded:
            rec_id = getattr(rec, "id", None)
            rec_fact = getattr(rec, "fact", "")
            if rec_id is None or int(rec_id) <= 0 or not str(rec_fact).strip():
                continue
            doc = _format_fact_document(
                str(getattr(rec, "subject", "") or ""),
                str(rec_fact),
                str(getattr(rec, "category", "") or ""),
            )
            valid_pairs.append((int(rec_id), doc))

        if not valid_pairs:
            return 0

        vectors = await self.embed_documents([doc for _, doc in valid_pairs])
        if not vectors or len(vectors) != len(valid_pairs):
            return 0

        indexed = 0
        for (fid, _), raw_vec in zip(valid_pairs, vectors):
            vec = normalize_vector(raw_vec)
            if vec.size == 0 or not np.any(vec):
                continue
            try:
                await memory_store.upsert_fact_embedding(
                    fid,
                    self.model,
                    int(vec.shape[0]),
                    vector_to_bytes(vec),
                )
                indexed += 1
            except Exception as exc:
                resolved_key = self._resolve_api_key()
                secrets = [resolved_key] if resolved_key else ()
                logger.debug(
                    "Failed to backfill embedding for fact_id=%d (%s: %s).",
                    fid,
                    type(exc).__name__,
                    _redact_secret_text(exc, explicit_secrets=secrets),
                )
        return indexed
