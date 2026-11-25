"""Core Orchestrator and Daemon Runner for the Autonomous Telegram Userbot Agent."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import signal
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from bot.antigravity import (
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    ErrorCategory,
    ImageGenerationError,
    ImagePolicyRefusalError,
    ImageTimeoutError,
    ImageValidationError,
    ToolProfile,
    classify_error,
    extract_retry_after,
)
from bot.config.manager import ConfigManager
from bot.config.models import BotConfig, FriendProfile
from bot.core.models import GeneratedImageResult, MediaAttachment, ProviderResponse, UsageInfo
from bot.documents.builder import DocumentBuilder, GeneratedDocument, extract_document_specs
from bot.evolution.manager import SelfEvolutionManager
from bot.media.image_gen import (
    IMAGE_INTENT_RE,
    ImageCommandParseResult,
    ImageGenerationLimiter,
    ImageRateLimitError,
    extract_image_specs,
    parse_imagine_command,
    should_send_as_document,
)
from bot.media.processor import MediaProcessor, ProcessedMedia, _has_media_flag
from bot.media.stt import STTTranscriber
from bot.media.tts import TTSSkippedError, TTSSynthesizer, strip_tts_emotion_tags
from bot.memory.compaction import (
    CompactionEngine,
    build_compaction_prompt,
    canonicalize_subject,
)
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.embeddings import MemoryEmbedder, should_run_semantic_lookup
from bot.memory.fts import FactRecord, extract_archive_terms, extract_fact_terms
from bot.memory.reflection import ReflectionEngine, build_reflection_prompt
from bot.proactive.engine import ProactiveEngine
from bot.quota.manager import QuotaBudgetManager
from bot.security.sanitizer import SecretSanitizer, install_log_sanitizer
from bot.telegram.bot_api_client import (
    TelegramBotAPIClient,
    normalize_reaction_emoji,
    resolve_bot_token_from_env_or_config,
)
from bot.telegram.client import TelegramUserbot
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

logger = logging.getLogger(__name__)


def build_antigravity_config(bot_cfg: BotConfig) -> AntigravityConfig:
    """Build runtime bot.antigravity.AntigravityConfig from BotConfig.antigravity."""
    agy_cfg = bot_cfg.antigravity
    return AntigravityConfig(
        data_dir=Path(agy_cfg.data_dir),
        model=agy_cfg.model,
        effort=agy_cfg.effort,
        enable_sdk=agy_cfg.enable_sdk,
        enable_cli_fallback=agy_cfg.enable_cli_fallback,
        max_retries=agy_cfg.max_retries,
        rate_limit_max_retries=agy_cfg.rate_limit_max_retries,
        rate_limit_cooldown_seconds=agy_cfg.rate_limit_cooldown_seconds,
        backoff_base=agy_cfg.backoff_base,
        backoff_max=agy_cfg.backoff_max,
    )


class _EventTextProxy:
    """Read-through wrapper overriding text/raw_text on an incoming event after STT."""

    def __init__(self, base_event: Any, text: str) -> None:
        self._base_event = base_event
        self.text = text
        self.raw_text = text

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_event, name)


class GroupBotOrchestrator:
    """Central orchestrator connecting Telegram MTProto, Media, Memory, Config, Proactive & Antigravity."""

    _FACT_COOLDOWN_REPLIES = 5
    _FACT_COOLDOWN_SECONDS = 1800.0
    _MAX_TRACKED_CHATS = 256
    _MAX_DEBOUNCE_BUFFER = 50
    _TOOL_INTENT_RE = re.compile(
        r"(?:https?://|\b(?:pdf|docx|xlsx|pptx|word|excel|google|search|news|weather|btc)\b"
        r"|пдф|ворд|эксел|ексел|документ|файл|отч[её]т|звіт|таблиц|слайд|презентац"
        r"|найд[иі]|знайд|поищ|пошук|погугл|гугл|новост|новин|погод|курс|цен[аыу]|цін[аиу]"
        r"|сколько\s+стоит|скільки\s+кошту|кто\s+тако|что\s+тако|що\s+таке|хто\s+так"
        r"|ссылк|посилан|источник|джерел|проверь|перевір|расписан|розклад)",
        re.IGNORECASE,
    )
    _OPS_INTENT_RE = re.compile(
        r"(?:\b(?:run|exec|execute|restart|reboot|deploy|install|uninstall|delete|remove|edit|"
        r"create|file|files|dir|folder|ls|cat|grep|git|docker|systemctl|journalctl|ssh|sudo|bash|shell|"
        r"python|pip|npm|logs|cron|schedule|status|disk|cpu|ram|memory|uptime|inspect|processes|service|"
        r"server|backup|evolve|quota|subagent|workflow|script|bug|fix|patch)\b"
        r"|(?:^|\s)(?:/(?:etc|var|opt|home|tmp|usr|proc)/|~/|\./)"
        r"|запуст|выполни|виконай|перезапуст|перезагруз|рестарт|задеплой|установ|встанови|удали|видали|"
        r"измени|зміни|отредактир|відредагу|создай|створи|напиши\s+(?:код|скрипт)|прочита|"
        r"покажи\s+(?:лог|файл|статус|процесс|диск|памят)|логи|команд|терминал|консол|сервер|"
        r"скрипт|код\b|баг(?:а|и|у|ов|е)?\b|исправ|виправ|почини|пофикс|статус|"
        r"диск(?:а|е|у|ов)?\b|процесс(?:ы|ов)?\b|служб|сервис|"
        r"крон|расписани|бэкап|бекап|эволюц|квот|субагент|воркфлоу)",
        re.IGNORECASE,
    )
    _ARCHIVE_CUE_RE = re.compile(
        r"(?:помн|пам['’]?ята|раньше|раніш|в\s+прошл|минулог|тогда|тоді|колись|архив|архів|"
        r"говорил|писал|скидывал|скидав|обсуждал|обговорювал)",
        re.IGNORECASE,
    )
    _SKIP_TAG_RE = re.compile(r"\[SKIP\]", re.IGNORECASE)
    _REACTION_TAG_RE = re.compile(r"\[REACT:\s*([^\[\]\s]+)\s*\]", re.IGNORECASE)
    _OPS_MODE_SETTING_KEY = "ops_tools_mode"
    _VALID_OPS_TOOLS_MODES = frozenset({"on", "admin", "off"})

    def __init__(
        self,
        config_mgr: ConfigManager,
        memory_store: Optional[MemoryStore] = None,
        media_processor: Optional[MediaProcessor] = None,
        provider: Optional[Any] = None,
        userbot: Optional[TelegramUserbot] = None,
        proactive_engine: Optional[ProactiveEngine] = None,
        document_builder: Optional[DocumentBuilder] = None,
        debounce_seconds: Optional[float] = None,
        stt_transcriber: Optional[Any] = None,
        tts_synthesizer: Optional[Any] = None,
        sanitizer: Optional[SecretSanitizer] = None,
        quota_manager: Optional[QuotaBudgetManager] = None,
        evolution_manager: Optional[Any] = None,
        image_limiter: Optional[ImageGenerationLimiter] = None,
        memory_embedder: Optional[MemoryEmbedder] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
    ) -> None:
        self.config_mgr = config_mgr
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            raise ValueError("BotConfig must be loaded in ConfigManager before initializing orchestrator.")

        env_candidate = Path(".env")
        self.sanitizer = sanitizer or SecretSanitizer(
            env_path=env_candidate if env_candidate.is_file() else None,
        )
        self._register_config_secrets(bot_cfg)
        install_log_sanitizer(self.sanitizer)

        self._owns_stt_transcriber = stt_transcriber is None
        self._owns_tts_synthesizer = tts_synthesizer is None
        self._owns_media_processor = media_processor is None
        self._owns_memory_embedder = memory_embedder is None

        self.memory_store = memory_store or MemoryStore(db_path=bot_cfg.memory.db_path)
        self.memory_embedder = (
            memory_embedder
            if memory_embedder is not None
            else MemoryEmbedder(bot_cfg.memory.embeddings)
        )
        self._default_memory_embedder: Optional[MemoryEmbedder] = (
            self.memory_embedder if memory_embedder is None else None
        )
        self.stt_transcriber = (
            stt_transcriber
            if stt_transcriber is not None
            else STTTranscriber(
                config=bot_cfg.voice.stt,
                ffmpeg_path=bot_cfg.media.ffmpeg_path,
            )
        )
        self.tts_synthesizer = (
            tts_synthesizer
            if tts_synthesizer is not None
            else TTSSynthesizer(
                config=bot_cfg.voice.tts,
                ffmpeg_path=bot_cfg.media.ffmpeg_path,
                temp_dir_base=bot_cfg.media.temp_dir,
            )
        )
        self._voice_stt_model_key = self._stt_model_key(bot_cfg)

        if media_processor is not None:
            self.media_processor = media_processor
            if (
                stt_transcriber is not None
                and getattr(self.media_processor, "stt_transcriber", None) is None
            ):
                self.media_processor.stt_transcriber = stt_transcriber
        else:
            self.media_processor = MediaProcessor(
                max_doc_size_mb=bot_cfg.media.max_doc_size_mb,
                ffmpeg_path=bot_cfg.media.ffmpeg_path,
                temp_dir_base=bot_cfg.media.temp_dir,
                stt_transcriber=self.stt_transcriber,
            )
        self.document_builder = document_builder or DocumentBuilder(
            temp_dir_base=bot_cfg.media.temp_dir if bot_cfg else "data/tmp",
        )
        self._image_limiter = image_limiter or ImageGenerationLimiter(bot_cfg.image_generation)
        self.provider = provider or AntigravityProvider(build_antigravity_config(bot_cfg))
        self.quota_manager = quota_manager or QuotaBudgetManager(
            memory_store=self.memory_store,
            config=bot_cfg,
        )
        if hasattr(self.provider, "set_quota_warning_callback"):
            self.provider.set_quota_warning_callback(self.quota_manager.record_quota_warning)
        self.evolution_manager = evolution_manager or SelfEvolutionManager(
            config_mgr=self.config_mgr,
            memory_store=self.memory_store,
            quota_manager=self.quota_manager,
            provider=self.provider,
            sanitizer=self.sanitizer,
        )
        self.proactive_engine = proactive_engine or ProactiveEngine(
            config_mgr=self.config_mgr,
            memory_store=self.memory_store,
            antigravity_provider=self.provider,
            quota_manager=self.quota_manager,
        )
        if getattr(self.proactive_engine, "quota_manager", None) is None:
            self.proactive_engine.quota_manager = self.quota_manager
        self.heuristics = ReplyHeuristics(
            default_max_per_hour=bot_cfg.behavior.spontaneous_replies.max_per_hour,
        )
        if debounce_seconds is not None:
            self.debounce_seconds = max(0.0, float(debounce_seconds))
        else:
            self.debounce_seconds = 2.5 if userbot is None else 0.0

        effective_dm_ids = self._effective_dm_user_ids(bot_cfg)

        if userbot is not None:
            self.userbot: Any = userbot
        else:
            resolved_token = resolve_bot_token_from_env_or_config(bot_cfg.telegram.bot_token)
            if resolved_token:
                self.sanitizer.register_secret(resolved_token)
            use_bot_api = bot_cfg.telegram.mode == "bot_api" or (
                bot_cfg.telegram.mode == "auto" and bool(resolved_token)
            )
            if use_bot_api and resolved_token:
                self.userbot = TelegramBotAPIClient(
                    bot_token=resolved_token,
                    allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                    allowed_dm_user_ids=effective_dm_ids,
                    ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                    bot_names=bot_cfg.behavior.bot_names,
                    typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                    typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                    typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                    backlog_Callback=self._seed_backlog_message,
                    sanitizer=self.sanitizer,
                )
            else:
                env_api_id_str = os.environ.get("TELEGRAM_API_ID", "").strip()
                eff_api_id = (
                    bot_cfg.telegram.api_id
                    if bot_cfg.telegram.api_id is not None
                    else (int(env_api_id_str) if env_api_id_str.isdigit() else 0)
                )
                eff_api_hash = (
                    bot_cfg.telegram.api_hash
                    if bot_cfg.telegram.api_hash is not None
                    else os.environ.get("TELEGRAM_API_HASH", "")
                )
                self.userbot = TelegramUserbot(
                    session_path=bot_cfg.telegram.session_path or "data/session/userbot.session",
                    api_id=eff_api_id,
                    api_hash=eff_api_hash,
                    allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                    allowed_dm_user_ids=effective_dm_ids,
                    ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                    bot_names=bot_cfg.behavior.bot_names,
                    typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                    typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                    typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                    backlog_callback=self._seed_backlog_message,
                )

        self.config_mgr.register_reload_callback(self._on_config_reloaded)

        bot_uid = getattr(self.userbot, "bot_id", None)
        excluded_uids = [int(bot_uid)] if bot_uid else None
        refl_cfg = bot_cfg.memory.reflection
        eff_bot_names = self._effective_bot_subject_names(bot_cfg)
        engine_embedder = (
            self.memory_embedder
            if (
                memory_embedder is not None
                or isinstance(self.provider, (AntigravityProvider, AntigravityCLIClient))
            )
            else None
        )
        self.compaction_engine = CompactionEngine(
            memory_store=self.memory_store,
            summarize_fn=self._summarize_transcript_via_provider,
            threshold=bot_cfg.memory.compaction_threshold,
            batch_size=bot_cfg.memory.compaction_batch_size,
            excluded_user_ids=excluded_uids,
            friends_provider=self._get_friends_map,
            bot_names=eff_bot_names,
            embedder=engine_embedder,
            agreement_ttl_days=refl_cfg.agreement_default_ttl_days,
            event_ttl_days=refl_cfg.event_default_ttl_days,
            bot_id=int(bot_uid) if bot_uid else None,
        )
        self.reflection_engine = (
            reflection_engine
            if reflection_engine is not None
            else ReflectionEngine(
                memory_store=self.memory_store,
                reflect_fn=self._reflect_subject_via_provider,
                min_active_facts_per_subject=refl_cfg.min_active_facts_per_subject,
                max_subjects_per_cycle=refl_cfg.max_subjects_per_cycle,
                target_summary_facts=refl_cfg.target_summary_facts,
                subject_cooldown_hours=refl_cfg.subject_cooldown_hours,
                agreement_default_ttl_days=refl_cfg.agreement_default_ttl_days,
                event_default_ttl_days=refl_cfg.event_default_ttl_days,
                embedder=engine_embedder,
                friends_provider=self._get_friends_map,
                bot_id=int(bot_uid) if bot_uid else None,
                bot_names=eff_bot_names,
            )
        )
        self._reflection_task: Optional[asyncio.Task[Any]] = None
        self.userbot.register_message_handler(self._on_incoming_message)

        self._debounce_buffers: Dict[int, List[IncomingMessage]] = {}
        self._debounce_tasks: Dict[int, asyncio.Task[Any]] = {}
        self._debounce_started_at: Dict[int, float] = {}
        self._chat_locks: Dict[int, asyncio.Lock] = {}
        self._compaction_locks: Dict[int, asyncio.Lock] = {}
        self._recent_fact_usage: Dict[int, Dict[int, Tuple[int, float]]] = {}
        self._bot_reply_seq: Dict[int, int] = {}

        default_ops_mode = str(
            getattr(bot_cfg.antigravity, "ops_tools_default_mode", "on") or "on"
        ).strip().lower()
        self._ops_tools_mode: str = (
            default_ops_mode if default_ops_mode in self._VALID_OPS_TOOLS_MODES else "on"
        )
        self._ops_tools_mode_loaded: bool = False

        self.chat_states: Dict[int, Dict[str, float]] = {}
        now = time.time()
        for cid in bot_cfg.telegram.allowed_chat_ids:
            self.chat_states[int(cid)] = {
                "last_chat_msg_time": now,
                "last_bot_msg_time": 0.0,
            }

    @staticmethod
    def _effective_dm_user_ids(bot_cfg: BotConfig) -> List[int]:
        """Merge allowed_dm_user_ids and admin_user_ids preserving order without duplicates."""
        merged: List[int] = []
        seen: Set[int] = set()
        for uid in list(getattr(bot_cfg.telegram, "allowed_dm_user_ids", []) or []) + list(
            getattr(bot_cfg.telegram, "admin_user_ids", []) or []
        ):
            val = int(uid)
            if val not in seen:
                seen.add(val)
                merged.append(val)
        return merged

    @staticmethod
    def _effective_bot_subject_names(bot_cfg: BotConfig) -> List[str]:
        """Merge persona bot_names and ignored_sender_usernames for self/sibling-bot fact rejection."""
        merged: List[str] = []
        seen: Set[str] = set()
        for raw in list(getattr(bot_cfg.behavior, "bot_names", []) or []) + list(
            getattr(bot_cfg.telegram, "ignored_sender_usernames", []) or []
        ):
            val = str(raw or "").strip()
            if not val:
                continue
            key = val.lstrip("@").lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(val)
        return merged

    @classmethod
    def _is_private_dm_message(cls, msg: IncomingMessage, bot_cfg: BotConfig) -> bool:
        """Return True when IncomingMessage represents a 1-on-1 private DM rather than a group chat."""
        chat_id_int = int(msg.chat_id)
        if chat_id_int <= 0:
            return False

        raw = getattr(msg, "raw_event", None)
        raw_dict = getattr(raw, "raw_dict", None) if raw is not None else None
        if isinstance(raw_dict, dict):
            chat_type = str((raw_dict.get("chat") or {}).get("type") or "").strip().lower()
            if chat_type:
                return chat_type == "private"

        allowed_groups = {
            int(cid) for cid in (getattr(bot_cfg.telegram, "allowed_chat_ids", []) or [])
        }
        if chat_id_int in allowed_groups:
            return False

        if getattr(msg, "is_private", False) is True:
            return True

        if raw is not None:
            raw_priv = getattr(raw, "is_private", None)
            if isinstance(raw_priv, bool):
                return raw_priv
            if callable(raw_priv) and type(raw_priv).__name__ != "MagicMock":
                try:
                    res = raw_priv()
                    if isinstance(res, bool):
                        return res
                except Exception:
                    pass

        effective_dms = set(cls._effective_dm_user_ids(bot_cfg))
        sender_id = getattr(raw, "sender_id", None) if raw is not None else None
        if sender_id is None:
            sender_id = msg.sender_id
        if effective_dms and chat_id_int in effective_dms:
            if sender_id is None or int(sender_id) == chat_id_int:
                return True
        return False

    def _register_config_secrets(self, bot_cfg: BotConfig) -> None:
        """Register configured non-dummy Telegram/TTS tokens/hashes in SecretSanitizer."""
        if bot_cfg.telegram.bot_token:
            self.sanitizer.register_secret(bot_cfg.telegram.bot_token)
        api_hash = (bot_cfg.telegram.api_hash or "").strip()
        if api_hash and "0123456789abcdef" not in api_hash.lower():
            self.sanitizer.register_secret(api_hash)
        phone = str(getattr(bot_cfg.telegram, "phone", None) or "").strip()
        if phone and len(phone) >= 7:
            self.sanitizer.register_secret(phone)
        env_hash = os.environ.get("TELEGRAM_API_HASH", "").strip()
        if env_hash and "0123456789abcdef" not in env_hash.lower():
            self.sanitizer.register_secret(env_hash)
        env_phone = os.environ.get("TELEGRAM_PHONE", "").strip()
        if env_phone and len(env_phone) >= 7:
            self.sanitizer.register_secret(env_phone)
        tts_cfg = bot_cfg.voice.tts
        if tts_cfg.gemini_api_key:
            self.sanitizer.register_secret(tts_cfg.gemini_api_key)
        emb_cfg = getattr(bot_cfg.memory, "embeddings", None)
        if emb_cfg is not None and emb_cfg.gemini_api_key:
            self.sanitizer.register_secret(emb_cfg.gemini_api_key)
        for env_name in (
            tts_cfg.gemini_api_key_env,
            getattr(emb_cfg, "gemini_api_key_env", None),
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ):
            if env_name:
                env_val = os.environ.get(env_name)
                if env_val:
                    self.sanitizer.register_secret(env_val)

    def _get_friends_map(self) -> Dict[int, FriendProfile]:
        """Return active friends map from ConfigManager safely."""
        friends_cfg = getattr(self.config_mgr, "friends", None)
        return dict(getattr(friends_cfg, "friends", None) or {})

    def _is_embedder_active(self) -> bool:
        """Return True when semantic embeddings are available and not running against an unmocked offline test provider."""
        embedder = getattr(self, "memory_embedder", None)
        if embedder is None or not embedder.is_available():
            return False
        if not getattr(self, "_owns_memory_embedder", True):
            return True
        if embedder is not getattr(self, "_default_memory_embedder", None):
            return True
        if getattr(embedder, "_injected_client", None) is not None:
            return True
        return isinstance(self.provider, (AntigravityProvider, AntigravityCLIClient))

    def _sync_bot_identity_to_memory_engines(self) -> None:
        """Propagate resolved Telegram bot_id into compaction and reflection engines."""
        bot_uid = getattr(self.userbot, "bot_id", None)
        if bot_uid:
            uid_int = int(bot_uid)
            self.compaction_engine.bot_id = uid_int
            self.compaction_engine.excluded_user_ids.add(uid_int)
            self.reflection_engine.bot_id = uid_int

    def _on_config_reloaded(self) -> None:
        """Synchronize Telegram access lists, voice/memory settings, image limiter, and secret sanitizer on reload."""
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            return
        self._register_config_secrets(bot_cfg)
        self._sync_voice_configs(bot_cfg)
        self._image_limiter.update_config(bot_cfg.image_generation)
        if hasattr(self, "memory_embedder") and self.memory_embedder is not None:
            self.memory_embedder.update_config(bot_cfg.memory.embeddings)
        eff_bot_names = self._effective_bot_subject_names(bot_cfg)
        if hasattr(self, "compaction_engine") and self.compaction_engine is not None:
            self.compaction_engine.threshold = bot_cfg.memory.compaction_threshold
            self.compaction_engine.batch_size = bot_cfg.memory.compaction_batch_size
            self.compaction_engine.bot_names = eff_bot_names
            self.compaction_engine.agreement_ttl_days = (
                bot_cfg.memory.reflection.agreement_default_ttl_days
            )
            self.compaction_engine.event_ttl_days = (
                bot_cfg.memory.reflection.event_default_ttl_days
            )
        if hasattr(self, "reflection_engine") and self.reflection_engine is not None:
            self.reflection_engine.update_config(bot_cfg.memory.reflection)
            self.reflection_engine.bot_names = eff_bot_names
        self._sync_bot_identity_to_memory_engines()
        resolved_token = resolve_bot_token_from_env_or_config(bot_cfg.telegram.bot_token)
        if resolved_token:
            self.sanitizer.register_secret(resolved_token)
        if hasattr(self.userbot, "sync_access_lists"):
            self.userbot.sync_access_lists(
                allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                allowed_dm_user_ids=self._effective_dm_user_ids(bot_cfg),
                ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                bot_names=bot_cfg.behavior.bot_names,
            )

    async def send_sanitized_message(
        self,
        chat_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        simulate_typing: bool = False,
    ) -> Any:
        """Sanitize and send a message (used for admin DM / error notifications)."""
        safe_text = self.sanitizer.sanitize(text)
        return await self.userbot.send_reply(
            chat_id=chat_id,
            text=safe_text,
            reply_to_msg_id=reply_to_msg_id,
            simulate_typing=simulate_typing,
        )

    async def _seed_backlog_message(self, incoming: IncomingMessage) -> None:
        """Save historical message from startup getUpdates or MTProto backfill into SQLite without replying."""
        friend = self.config_mgr.get_friend_profile(incoming.sender_id)
        author_display = friend.name if friend else incoming.sender_name
        msg_date = getattr(incoming.raw_event, "date", None)
        if hasattr(msg_date, "timestamp"):
            ts = int(msg_date.timestamp())
        else:
            ts = int(time.time())

        await self.memory_store.save_message(
            ChatMessage(
                telegram_msg_id=incoming.msg_id or int(time.time() * 1000),
                chat_id=incoming.chat_id,
                user_id=incoming.sender_id,
                timestamp=ts,
                author_name=author_display,
                text=incoming.text,
                reply_to_msg_id=incoming.reply_to_msg_id,
            )
        )

    def _is_provider_rate_limited(self, role: str = "group") -> bool:
        """Return True only when all candidate models for ``role`` are rate-limited on the provider."""
        checker = getattr(self.provider, "is_rate_limited", None)
        if not callable(checker):
            return False

        candidates: List[str] = []
        qm = getattr(self, "quota_manager", None)
        if qm is not None and hasattr(qm, "select_model"):
            try:
                primary_model, _ = qm.select_model(role)
                if primary_model:
                    candidates.append(primary_model)
            except Exception:
                pass

        bot_cfg = self.config_mgr.bot if getattr(self, "config_mgr", None) is not None else None
        models_cfg = getattr(bot_cfg, "models", None) if bot_cfg is not None else None
        for fb in getattr(models_cfg, "fallback_models", []) or []:
            fb_model = getattr(fb, "model", None)
            if fb_model and fb_model not in candidates:
                candidates.append(fb_model)

        if not candidates:
            return checker() is True

        for candidate in candidates:
            try:
                limited = checker(model=candidate) is True
            except TypeError:
                return checker() is True
            if not limited:
                return False
        return True

    def _provider_lacks_binary_media(
        self,
        attachments: Optional[Sequence[MediaAttachment]] = None,
        media_text: Optional[str] = None,
    ) -> bool:
        """Return True when the active CLI provider cannot inspect the given binary media."""
        if media_text and media_text.strip():
            return False

        if attachments and any(
            (att.mime_type or "").lower().startswith("image/")
            and ((att.file_path is not None and att.file_path.exists()) or bool(att.data))
            for att in attachments
        ):
            if isinstance(self.provider, AntigravityCLIClient):
                return not getattr(self.provider, "supports_image_vision", False)
            cli_client = getattr(self.provider, "_cli_client", None)
            sdk_client = getattr(self.provider, "_sdk_client", None)
            if sdk_client is not None:
                return False
            if isinstance(cli_client, AntigravityCLIClient):
                return not getattr(cli_client, "supports_image_vision", False)
            return False

        if isinstance(self.provider, AntigravityCLIClient):
            return True
        cli_client = getattr(self.provider, "_cli_client", None)
        sdk_client = getattr(self.provider, "_sdk_client", None)
        return isinstance(cli_client, AntigravityCLIClient) and sdk_client is None

    @staticmethod
    def _stt_model_key(bot_cfg: BotConfig) -> Tuple[str, str, int, Optional[str]]:
        stt = bot_cfg.voice.stt
        return (stt.model_size, stt.compute_type, int(stt.cpu_threads), stt.model_dir)

    def _sync_voice_configs(self, bot_cfg: BotConfig) -> None:
        """Synchronize TTS/STT/MediaProcessor settings after config hot-reload."""
        if self._owns_media_processor:
            self.media_processor.max_doc_size_mb = bot_cfg.media.max_doc_size_mb
            self.media_processor.ffmpeg_path = bot_cfg.media.ffmpeg_path
            self.media_processor.temp_dir_base = bot_cfg.media.temp_dir

        if self._owns_tts_synthesizer:
            tts_cfg = bot_cfg.voice.tts
            self.tts_synthesizer.enabled = bool(tts_cfg.enabled)
            self.tts_synthesizer.provider = tts_cfg.provider
            self.tts_synthesizer.fallback_to_edge = bool(tts_cfg.fallback_to_edge)
            self.tts_synthesizer.ru_voice = tts_cfg.ru_voice
            self.tts_synthesizer.uk_voice = tts_cfg.uk_voice
            self.tts_synthesizer.rate = tts_cfg.rate
            self.tts_synthesizer.pitch = tts_cfg.pitch
            self.tts_synthesizer.max_chars = int(tts_cfg.max_chars)
            self.tts_synthesizer.timeout_sec = float(tts_cfg.timeout_sec)
            self.tts_synthesizer.gemini_model = tts_cfg.gemini_model
            self.tts_synthesizer.gemini_ru_voice = tts_cfg.gemini_ru_voice
            self.tts_synthesizer.gemini_uk_voice = tts_cfg.gemini_uk_voice
            self.tts_synthesizer.gemini_style_prompt = tts_cfg.gemini_style_prompt
            self.tts_synthesizer.gemini_api_key_env = tts_cfg.gemini_api_key_env
            self.tts_synthesizer.gemini_api_key = tts_cfg.gemini_api_key
            self.tts_synthesizer.ffmpeg_path = bot_cfg.media.ffmpeg_path
            self.tts_synthesizer.temp_dir_base = bot_cfg.media.temp_dir

        if self._owns_stt_transcriber:
            stt_cfg = bot_cfg.voice.stt
            new_key = self._stt_model_key(bot_cfg)
            if new_key != self._voice_stt_model_key:
                old_stt = self.stt_transcriber
                self.stt_transcriber = STTTranscriber(
                    config=stt_cfg,
                    ffmpeg_path=bot_cfg.media.ffmpeg_path,
                )
                self._voice_stt_model_key = new_key
                if self._owns_media_processor:
                    self.media_processor.stt_transcriber = self.stt_transcriber
                if hasattr(old_stt, "close"):
                    try:
                        old_stt.close()
                    except Exception:
                        pass
            else:
                self.stt_transcriber.enabled = bool(stt_cfg.enabled)
                self.stt_transcriber.max_duration_sec = int(stt_cfg.max_duration_sec)
                self.stt_transcriber.timeout_sec = float(stt_cfg.timeout_sec)
                self.stt_transcriber.allowed_languages = tuple(
                    str(lang).strip().lower()
                    for lang in stt_cfg.allowed_languages
                    if str(lang).strip()
                )
                self.stt_transcriber.ffmpeg_path = bot_cfg.media.ffmpeg_path

    async def _summarize_transcript_via_provider(
        self,
        transcript: str,
        existing_facts: Optional[Sequence[FactRecord]] = None,
        chat_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Extract genuinely durable upsert fact operations from human conversation transcript using Antigravity."""
        if self._is_provider_rate_limited(role="compaction"):
            logger.info("Skipping compaction summarization while provider is rate-limited.")
            return []
        prompt = build_compaction_prompt(transcript, existing_facts=existing_facts)
        chosen_model: Optional[str] = None
        chosen_effort: Optional[str] = None
        qm = getattr(self, "quota_manager", None)
        if qm is not None and hasattr(qm, "select_model"):
            try:
                chosen_model, chosen_effort = qm.select_model("compaction")
            except Exception:
                chosen_model, chosen_effort = None, None
        try:
            try:
                if chosen_model is not None:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        model=chosen_model,
                        effort=chosen_effort,
                        tool_profile="chat",
                    )
                else:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        tool_profile="chat",
                    )
            except TypeError:
                try:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        tool_profile="chat",
                    )
                except TypeError:
                    resp = await self.provider.generate_response(prompt=prompt)
            usage = getattr(resp, "usage", None)
            if usage is not None and qm is not None and hasattr(qm, "record_usage"):
                try:
                    model_name = getattr(resp, "model", None) or chosen_model or "unknown"
                    rec = qm.record_usage(
                        usage,
                        model=model_name,
                        source="compaction",
                    )
                    if inspect.isawaitable(rec):
                        await rec
                except Exception as rec_exc:
                    logger.debug("Failed to record compaction usage: %s", rec_exc)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return self.compaction_engine._parse_extracted_facts(text)
        except Exception as exc:
            logger.debug(
                "Compaction summarization fallback triggered (chat=%s): %s",
                chat_id,
                exc,
            )
            return []

    async def _reflect_subject_via_provider(
        self,
        *,
        subject: str,
        facts: Sequence[FactRecord],
        chat_id: Optional[int] = None,
        target_summary_facts: int = 4,
    ) -> str:
        """Consolidate accumulated subject micro-facts via Antigravity under strict chat profile."""
        if self._is_provider_rate_limited(role="compaction"):
            logger.info("Skipping memory reflection while provider is rate-limited.")
            return ""
        prompt = build_reflection_prompt(
            subject=subject,
            facts=facts,
            target_summary_facts=target_summary_facts,
        )
        chosen_model: Optional[str] = None
        chosen_effort: Optional[str] = None
        qm = getattr(self, "quota_manager", None)
        if qm is not None and hasattr(qm, "select_model"):
            try:
                chosen_model, chosen_effort = qm.select_model("compaction")
            except Exception:
                chosen_model, chosen_effort = None, None
        try:
            try:
                if chosen_model is not None:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        model=chosen_model,
                        effort=chosen_effort,
                        tool_profile="chat",
                    )
                else:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        tool_profile="chat",
                    )
            except TypeError:
                try:
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        tool_profile="chat",
                    )
                except TypeError:
                    resp = await self.provider.generate_response(prompt=prompt)
            usage = getattr(resp, "usage", None)
            if usage is not None and qm is not None and hasattr(qm, "record_usage"):
                try:
                    model_name = getattr(resp, "model", None) or chosen_model or "unknown"
                    rec = qm.record_usage(
                        usage,
                        model=model_name,
                        source="compaction",
                    )
                    if inspect.isawaitable(rec):
                        await rec
                except Exception as rec_exc:
                    logger.debug("Failed to record reflection usage: %s", rec_exc)
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as exc:
            logger.debug(
                "Memory reflection generation failed (chat=%s): %s",
                chat_id,
                type(exc).__name__,
            )
            return ""

    @staticmethod
    def summarize_processed_media(processed: Optional[ProcessedMedia]) -> Optional[str]:
        """Build human-readable summary of ProcessedMedia for memory and prompt context."""
        if processed is None:
            return None
        if processed.media_type == "document":
            fname = processed.metadata.get("file_name", "document")
            snippet = (processed.text_content or "")[:300].strip()
            return f"Документ '{fname}': {snippet}" if snippet else f"Документ '{fname}'"
        if processed.media_type == "sticker":
            emoji = processed.metadata.get("emoji") or ""
            return f"Стикер {emoji}".strip()
        if processed.media_type == "photo":
            return "Фотография / изображение"
        if processed.media_type == "voice":
            transcript = (
                processed.text_content or processed.metadata.get("transcript") or ""
            ).strip()
            lang = processed.metadata.get("language")
            if transcript:
                label = f"Голосовое сообщение ({lang})" if lang else "Голосовое сообщение"
                return f"{label}: {transcript[:300]}"
            return "Голосовое сообщение (аудио WAV 16kHz)"
        if processed.media_type == "video_note":
            transcript = (
                processed.text_content or processed.metadata.get("transcript") or ""
            ).strip()
            return f"Видео-кружочек: {transcript[:300]}" if transcript else "Видео-кружочек (аудио + кадр)"
        if processed.media_type == "animation":
            fname = processed.metadata.get("file_name", "animation.mp4")
            return f"GIF-анимация / демотиватор '{fname}' (извлеченный кадр)"
        if processed.media_type == "video":
            if processed.metadata.get("is_video_note"):
                transcript = (
                    processed.text_content or processed.metadata.get("transcript") or ""
                ).strip()
                return (
                    f"Видео-кружочек: {transcript[:300]}"
                    if transcript
                    else "Видео-кружочек (аудио + кадр)"
                )
            fname = processed.metadata.get("file_name", "video.mp4")
            return f"Видео '{fname}' (извлеченный кадр)"
        return processed.media_type

    @staticmethod
    def build_media_attachments(processed: Optional[ProcessedMedia]) -> List[MediaAttachment]:
        """Convert ProcessedMedia into Antigravity MediaAttachment objects."""
        if processed is None:
            return []
        attachments: List[MediaAttachment] = []
        summary = GroupBotOrchestrator.summarize_processed_media(processed) or processed.media_type
        file_name = processed.metadata.get("file_name")

        visual_path = processed.image_path
        visual_mime = processed.mime_type
        if visual_path is None and processed.keyframe_path and processed.keyframe_path.exists():
            visual_path = processed.keyframe_path
            visual_mime = "image/jpeg"
        elif (
            visual_path is None
            and processed.file_path is not None
            and processed.file_path.exists()
            and processed.media_type in ("photo", "sticker", "document")
        ):
            visual_path = processed.file_path

        if visual_path and visual_path.exists():
            attachments.append(
                MediaAttachment(
                    file_path=visual_path,
                    mime_type=visual_mime or "application/octet-stream",
                    description=summary,
                    file_name=file_name or visual_path.name,
                )
            )
        elif processed.audio_path and processed.audio_path.exists():
            attachments.append(
                MediaAttachment(
                    file_path=processed.audio_path,
                    mime_type=processed.mime_type or "audio/wav",
                    description=summary,
                    file_name=file_name or processed.audio_path.name,
                )
            )
        elif processed.text_content:
            attachments.append(
                MediaAttachment(
                    mime_type=processed.mime_type or "text/plain",
                    description=summary,
                    file_name=file_name,
                )
            )
        return attachments

    _ARCHIVE_CHAT_LABELS: Dict[int, str] = {
        -1003845062562: "летальний ісход",
        -1003323178808: "Анальные козырьки",
    }

    @classmethod
    def _needs_search_or_doc_tools(cls, text: str) -> bool:
        """Return True if text plausibly requests web search, live facts, links, or document creation."""
        if not text:
            return False
        return bool(cls._TOOL_INTENT_RE.search(text))

    @classmethod
    def _should_search_archive(cls, text: str) -> bool:
        """Return True only when message is substantive or asks about past group context."""
        cleaned = (text or "").strip()
        if len(cleaned) < 20:
            return False
        terms = extract_archive_terms(cleaned)
        if len(terms) >= 2:
            return True
        return bool(terms and cls._ARCHIVE_CUE_RE.search(cleaned))

    @staticmethod
    def _has_incoming_media(incoming: IncomingMessage) -> bool:
        """Check cheaply whether an IncomingMessage carries media without downloading it."""
        if incoming.media:
            return True
        raw_msg = getattr(incoming.raw_event, "message", incoming.raw_event)
        if raw_msg is None:
            return False
        for attr in ("photo", "sticker", "voice", "video_note", "animation", "video", "document", "media"):
            if _has_media_flag(raw_msg, attr):
                return True
        return False

    @staticmethod
    def _infer_cheap_media_type(incoming: IncomingMessage) -> Optional[str]:
        """Infer media_type from raw Telegram event metadata without downloading files."""
        raw_msg = getattr(incoming.raw_event, "message", incoming.raw_event)
        if raw_msg is None:
            return None
        for attr in ("photo", "sticker", "voice", "video_note", "animation", "video", "document"):
            if _has_media_flag(raw_msg, attr):
                return attr
        return "media" if incoming.media else None

    def _prune_chat_caches(self) -> None:
        """Bound per-chat in-memory caches so long uptimes do not accumulate stale entries."""
        cap = self._MAX_TRACKED_CHATS
        bot_cfg = self.config_mgr.bot
        protected_chats: Set[int] = set()
        if bot_cfg is not None:
            protected_chats.update(int(c) for c in (bot_cfg.telegram.allowed_chat_ids or []))
            protected_chats.update(self._effective_dm_user_ids(bot_cfg))

        for lock_map in (self._chat_locks, self._compaction_locks):
            if len(lock_map) > cap:
                for cid in list(lock_map.keys()):
                    if len(lock_map) <= cap:
                        break
                    if cid not in protected_chats and not lock_map[cid].locked():
                        lock_map.pop(cid, None)

        if len(self.chat_states) > cap:
            removable = [cid for cid in self.chat_states if cid not in protected_chats]
            removable.sort(
                key=lambda cid: float(self.chat_states[cid].get("last_chat_msg_time", 0.0))
            )
            for cid in removable:
                if len(self.chat_states) <= cap:
                    break
                self.chat_states.pop(cid, None)

        if len(self._recent_fact_usage) > cap:
            for cid in list(self._recent_fact_usage.keys()):
                if len(self._recent_fact_usage) <= cap:
                    break
                if cid not in protected_chats:
                    self._recent_fact_usage.pop(cid, None)

        if len(self._bot_reply_seq) > cap:
            for cid in list(self._bot_reply_seq.keys()):
                if len(self._bot_reply_seq) <= cap:
                    break
                if cid not in protected_chats:
                    self._bot_reply_seq.pop(cid, None)

    def _get_recent_fact_ids(self, chat_id: int, now: float) -> List[int]:
        """Return active per-chat cooled-down fact IDs (within last N bot replies or 30m TTL)."""
        bucket = self._recent_fact_usage.get(chat_id)
        if not bucket:
            if bucket is not None:
                self._recent_fact_usage.pop(chat_id, None)
            return []
        cur_seq = self._bot_reply_seq.get(chat_id, 0)
        expired = [
            fid
            for fid, (seq, ts) in bucket.items()
            if (cur_seq - seq) >= self._FACT_COOLDOWN_REPLIES
            or (now - ts) >= self._FACT_COOLDOWN_SECONDS
        ]
        for fid in expired:
            bucket.pop(fid, None)
        if not bucket:
            self._recent_fact_usage.pop(chat_id, None)
            return []
        return list(bucket.keys())

    def _record_used_facts(self, chat_id: int, facts: Sequence[FactRecord], now: float) -> None:
        """Advance per-chat reply sequence and record any injected fact IDs for cooldown."""
        seq = self._bot_reply_seq.get(chat_id, 0) + 1
        self._bot_reply_seq[chat_id] = seq
        existing = self._recent_fact_usage.get(chat_id)
        if existing:
            for fid, (old_seq, old_ts) in list(existing.items()):
                if (seq - old_seq) >= self._FACT_COOLDOWN_REPLIES or (
                    now - old_ts
                ) >= self._FACT_COOLDOWN_SECONDS:
                    existing.pop(fid, None)
            if not existing:
                self._recent_fact_usage.pop(chat_id, None)
        if facts:
            bucket = self._recent_fact_usage.setdefault(chat_id, {})
            for fact in facts:
                if getattr(fact, "id", None) is not None:
                    bucket[int(fact.id)] = (seq, now)
        if (
            len(self._recent_fact_usage) > self._MAX_TRACKED_CHATS
            or len(self._bot_reply_seq) > self._MAX_TRACKED_CHATS
        ):
            self._prune_chat_caches()

    @staticmethod
    def _filter_overused_facts(
        facts: Sequence[FactRecord],
        recent_history: Sequence[ChatMessage],
        bot_id: Optional[int],
        bot_names: Sequence[str],
    ) -> List[FactRecord]:
        """Suppress facts whose key tokens already appeared in the bot's last ~5 replies."""
        if not facts:
            return []
        bot_names_lower = {n.strip().lower() for n in bot_names if n and n.strip()}
        bot_texts: List[str] = []
        for msg in recent_history:
            is_bot = (bot_id is not None and msg.user_id == int(bot_id)) or (
                bool(msg.author_name) and msg.author_name.strip().lower() in bot_names_lower
            )
            if is_bot and msg.text:
                bot_texts.append(msg.text)
        if not bot_texts:
            return list(facts[:3])

        recent_bot_terms: Set[str] = set(extract_fact_terms(" ".join(bot_texts[-5:]), max_terms=60))
        if not recent_bot_terms:
            return list(facts[:3])

        filtered: List[FactRecord] = []
        for fact in facts:
            subject_terms = set(extract_fact_terms(fact.subject or "", max_terms=5))
            fact_terms = set(extract_fact_terms(fact.fact or "", max_terms=10)) - subject_terms
            if fact_terms and (fact_terms & recent_bot_terms):
                continue
            filtered.append(fact)
            if len(filtered) >= 3:
                break
        return filtered

    def build_context_prompt(
        self,
        incoming: IncomingMessage,
        recent_history: List[ChatMessage],
        facts: List[FactRecord],
        friend_profile: Optional[FriendProfile],
        media_summary: Optional[str] = None,
        media_text: Optional[str] = None,
        archive_hits: Optional[List[ChatMessage]] = None,
        incoming_messages: Optional[List[IncomingMessage]] = None,
        include_tool_instructions: Optional[bool] = None,
        media_from_reply: bool = False,
        voice_reply_mode: bool = False,
        ops_tools_enabled: bool = False,
    ) -> str:
        """Assemble concise contextual prompt with passive memory, history, and gated tool specs."""
        sections: List[str] = []

        # 1. Friend profile injection (role/style + exclusions; omit banter_topics checklist)
        if friend_profile is not None:
            forbidden_global = (
                self.config_mgr.bot.proactive.forbidden_topics if self.config_mgr.bot else []
            )
            exclusions = sorted(
                set(list(forbidden_global) + list(friend_profile.sensitive_topics or []))
            )
            aliases = ", ".join(friend_profile.aliases) if friend_profile.aliases else "нет"
            excl_str = ", ".join(exclusions) if exclusions else "нет"
            sections.append(
                "ПРОФИЛЬ АВТОРА СООБЩЕНИЯ (из friends.yaml):\n"
                f"- Telegram ID: {incoming.sender_id}\n"
                f"- Имя: {friend_profile.name} (ники: {aliases})\n"
                f"- Роль/Стиль: {friend_profile.bio or friend_profile.role}; "
                f"{friend_profile.communication_style or 'дружеский'}\n"
                f"- СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ: {excl_str}"
            )

        # 2. Long-term FTS5 memory facts (strictly passive context, max 3)
        if facts:
            fact_lines = [f"- [{f.category}] {f.subject}: {f.fact}" for f in facts[:3]]
            sections.append(
                "ПАССИВНЫЙ КОНТЕКСТ (не навязывай и не впихивай специально; "
                "используй только если реально уместно по текущей теме):\n"
                + "\n".join(fact_lines)
            )

        # 2b. Relevant historical archive excerpts (strictly passive context, max 3)
        if archive_hits:
            archive_lines: List[str] = []
            for m in archive_hits[:3]:
                date_str = (
                    datetime.fromtimestamp(m.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
                    if m.timestamp > 0
                    else "архив"
                )
                chat_label = self._ARCHIVE_CHAT_LABELS.get(m.chat_id, str(m.chat_id))
                snippet = (m.text or m.media_summary or "").strip().replace("\n", " ")
                if len(snippet) > 180:
                    snippet = snippet[:177] + "..."
                archive_lines.append(f"- [{date_str} | {chat_label}] {m.author_name}: {snippet}")
            if archive_lines:
                sections.append(
                    "АРХИВ ПЕРЕПИСОК (ПАССИВНЫЙ КОНТЕКСТ — не навязывай и не впихивай специально; "
                    "используй только если реально уместно по текущей теме):\n"
                    + "\n".join(archive_lines)
                )

        # 3. Short-term chat history (compact, last 10)
        if recent_history:
            hist_lines = []
            for m in recent_history[-10:]:
                txt = m.text or ""
                if m.media_summary:
                    txt = f"{txt} [Медиа: {m.media_summary}]".strip()
                hist_lines.append(f"[{m.author_name}]: {txt}")
            sections.append("ПОСЛЕДНИЕ СООБЩЕНИЯ В ЧАТЕ:\n" + "\n".join(hist_lines))

        # 4. Current incoming message(s) + reply & multimodal details
        burst = incoming_messages if incoming_messages and len(incoming_messages) > 1 else None
        if burst:
            burst_lines = []
            for m in burst:
                m_text = (m.text or "").strip()
                if m_text:
                    burst_lines.append(f"- {m.sender_name}: {m_text}")
                elif self._has_incoming_media(m):
                    burst_lines.append(f"- {m.sender_name}: [отправил вложение]")
            combined_msg_text = "\n".join(burst_lines)
            msg_block = (
                f"НОВЫЕ СООБЩЕНИЯ (серия подряд, основное от {incoming.sender_name}, ID: {incoming.sender_id}):\n"
                + combined_msg_text
            ).strip()
        else:
            combined_msg_text = incoming.text or ""
            msg_block = (
                f"НОВОЕ СООБЩЕНИЕ ОТ {incoming.sender_name} (ID: {incoming.sender_id}):\n"
                f"{combined_msg_text}"
            ).strip()

        reply_author = getattr(incoming, "reply_to_author", None) or (
            getattr(incoming.raw_event, "reply_to_author", None)
        )
        reply_text = getattr(incoming, "reply_to_text", None) or (
            getattr(incoming.raw_event, "reply_to_text", None)
        )
        if reply_text:
            author_label = f" ({reply_author})" if reply_author else ""
            msg_block += f"\n[Ответ на сообщение{author_label}: {reply_text[:400]}]"

        if media_summary:
            if media_from_reply:
                author_label = f" от {reply_author}" if reply_author else ""
                msg_block += f"\n[Вложение из сообщения{author_label}, на которое отвечают: {media_summary}]"
            else:
                msg_block += f"\n[Вложение: {media_summary}]"
        if media_text:
            msg_block += f"\n[Содержимое вложения]:\n{media_text[:2000]}"
        sections.append(msg_block)

        raw_for_tools = (
            "\n".join(m.text or "" for m in burst) if burst else (incoming.text or "")
        )
        should_include_tools = (
            include_tool_instructions
            if include_tool_instructions is not None
            else self._needs_search_or_doc_tools(raw_for_tools)
        )

        guidance = (
            "Ответь естественно, кратко (1-2 предложения, до 3 по делу) и в соответствии с системным промптом. "
            "Если реплика сухая/не требует комментария, ответь ровно [SKIP]. "
            "Если уместна эмоциональная реакция, добавь в конец тег [REACT:🔥] "
            "(разрешённые: 👍, 👎, ❤, 🔥, 🤣, 🥰, 😱, 🎉, 💩, 🤡, 🗿, 🤔, 👀, 🫡 — никогда 😂), "
            "или ответь ТОЛЬКО тегом [REACT:🤣]."
        )
        if should_include_tools:
            guidance += (
                "\nДОКУМЕНТЫ И ИССЛЕДОВАНИЯ: ты УМЕЕШЬ создавать и отправлять реальные файлы "
                "(pdf, docx, xlsx, pptx) прямо в чат — НИКОГДА не говори, что у тебя нет инструментов. "
                "Если просят найти инфу, проверить свежие данные или сделать документ/PDF: "
                "используй `search_web` и `read_url_content`, а для отправки файла добавь в конец тег "
                '[DOC:{"format":"pdf|docx|xlsx|pptx","title":"...","subtitle":"...","filename":"name.pdf",'
                '"caption":"...","sections":[{"heading":"...","body":"...","bullets":["..."],'
                '"table":[["A","B"],["1","2"]]}],"sources":["https://..."]}] '
                "(можно с короткой живой репликой перед тегом)."
            )
        bot_cfg = self.config_mgr.bot
        img_cfg = bot_cfg.image_generation if bot_cfg is not None else None
        if (
            img_cfg is not None
            and img_cfg.enabled
            and img_cfg.allow_natural_language_trigger
            and IMAGE_INTENT_RE.search(raw_for_tools)
        ):
            guidance += (
                "\nГЕНЕРАЦИЯ КАРТИНОК: ты УМЕЕШЬ рисовать и отправлять картинки прямо в чат "
                "(используй ТОЛЬКО если пользователь действительно просит нарисовать/сгенерировать изображение "
                "и запрос не нарушает запреты). Для генерации добавь в конец ровно один тег "
                '[IMG:{"prompt":"detailed English or Russian image description","caption":"optional short Russian caption","aspect_ratio":"1:1"}] '
                "(допустимые aspect_ratio: 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 21:9; "
                "можно добавить короткую живую реплику перед тегом)."
            )
        if voice_reply_mode:
            guidance += (
                "\nГОЛОСОВОЙ ОТВЕТ: собеседник отправил голосовое сообщение, твой ответ будет озвучен голосом (TTS). "
                "Начни ответ с короткой директивы настроения под момент в формате "
                "`[MOOD: <2-6 English words describing emotion/pace>]` "
                "(например: `[MOOD: dry stoic deadpan, confident]`, `[MOOD: grounded masculine banter]`, "
                "`[MOOD: casual cheeky banter]`, `[MOOD: amused eye-roll, lively]`, "
                "`[MOOD: candid, slightly exasperated]`, `[MOOD: teasing laugh, natural pace]`, "
                "`[MOOD: friendly sarcastic roast]`). "
                "Далее пиши 1–2 живые устные фразы, как реальный человек наговаривает голосовое в чате: "
                "разговорная речь (русский/украинский/суржик) без книжных метафор и канцелярита, "
                "допустимы естественные связки (`ну`, `слушай`, `короче`, `по факту`, `типа`, `блин`, `та ладно`, `ага`), "
                "живой темп, вздохи и вдохи (`[inhale]`, `[exhale]`, `[sigh]`, `[chuckles]`, `[laughs]`, "
                "`[scoffs]`, `[short pause]`, `...` — только как короткая живая заминка в середине мысли, не томный шёпот). "
                "ЗАПРЕЩЕНО: `[whispers]`, томный/флиртующий/соблазнительный тон, "
                "книжные или монотонно-литературные формулировки, "
                "Markdown-разметка, таблицы, списки, блоки кода и URL-ссылки."
            )
        if ops_tools_enabled:
            guidance += (
                "\nРАСШИРЕННЫЙ РЕЖИМ ИНСТРУМЕНТОВ (ops, без root): доступны команды, файлы, поиск, "
                "субагенты и /evolve /quota /backup — используй ТОЛЬКО если просьба этого требует; "
                "секреты (.env, токены, пароли) не раскрывай."
            )
        sections.append(guidance)
        return "\n\n".join(sections)

    @classmethod
    def is_skip_response(cls, text: str) -> bool:
        """Return True if the model intentionally declined to reply via [SKIP]."""
        if not text:
            return False
        stripped = cls._SKIP_TAG_RE.sub("", text).strip(" \t\r\n.!?,;:-")
        return bool(cls._SKIP_TAG_RE.search(text) and not stripped)

    @classmethod
    def extract_reaction_tag(cls, text: str) -> Tuple[Optional[str], str]:
        """Extract first [REACT:<emoji>] tag, normalize for Bot API, and return (emoji, cleaned_text)."""
        if not text:
            return None, ""
        match = cls._REACTION_TAG_RE.search(text)
        emoji = normalize_reaction_emoji(match.group(1).strip()) if match else None
        cleaned = cls._REACTION_TAG_RE.sub("", text).strip()
        return emoji, cleaned

    @staticmethod
    def _extract_sent_message_id(sent_res: Any) -> int:
        """Extract integer Telegram message_id from int, Bot API dict, or Telethon Message object."""
        if isinstance(sent_res, int) and not isinstance(sent_res, bool):
            return sent_res if sent_res > 0 else (int(time.time() * 1000) + 1)

        candidate: Any = None
        if isinstance(sent_res, dict):
            res_obj = sent_res.get("result")
            if isinstance(res_obj, dict) and "message_id" in res_obj:
                candidate = res_obj.get("message_id")
            elif "message_id" in sent_res:
                candidate = sent_res.get("message_id")
        elif sent_res is not None:
            candidate = getattr(sent_res, "id", None)
            if candidate is None:
                candidate = getattr(sent_res, "message_id", None)

        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
        return int(time.time() * 1000) + 1

    async def _on_incoming_message(self, incoming: IncomingMessage) -> None:
        """Incoming handler registered with Telegram client; coalesces rapid non-command bursts."""
        text_stripped = (incoming.text or "").strip()
        if self.debounce_seconds <= 0.0 or text_stripped.startswith("/"):
            await self.handle_incoming_message(incoming)
            return

        chat_id = incoming.chat_id
        if (
            chat_id not in self._debounce_buffers
            and len(self._debounce_buffers) >= self._MAX_TRACKED_CHATS
        ):
            for stale_cid in list(self._debounce_tasks.keys()):
                stale_task = self._debounce_tasks.get(stale_cid)
                if stale_task is None or stale_task.done():
                    self._debounce_tasks.pop(stale_cid, None)
                    self._debounce_buffers.pop(stale_cid, None)
                    self._debounce_started_at.pop(stale_cid, None)
        buf = self._debounce_buffers.setdefault(chat_id, [])
        if len(buf) >= self._MAX_DEBOUNCE_BUFFER:
            del buf[: len(buf) - self._MAX_DEBOUNCE_BUFFER + 1]
        buf.append(incoming)
        first_ts = self._debounce_started_at.setdefault(chat_id, time.monotonic())

        existing_task = self._debounce_tasks.get(chat_id)
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()

        max_wait = max(self.debounce_seconds * 2.5, 6.0)
        elapsed = time.monotonic() - first_ts
        delay = max(0.0, min(self.debounce_seconds, max_wait - elapsed))
        self._debounce_tasks[chat_id] = asyncio.create_task(
            self._debounced_flush(chat_id, delay)
        )

    async def _debounced_flush(self, chat_id: int, delay: float) -> None:
        """Wait for the debounce window to expire, then process buffered messages."""
        try:
            if delay > 0.0:
                await asyncio.sleep(delay)
            await self._process_buffered_messages(chat_id)
        except asyncio.CancelledError:
            return
        except AntigravityNativeToolAbort as exc:
            if not getattr(exc, "_native_tool_abort_logged", False):
                logger.warning(
                    "Native tool execution aborted while flushing chat %s: %s",
                    chat_id,
                    self.sanitizer.sanitize(str(exc)),
                )
            else:
                logger.debug(
                    "Native tool execution abort handled during flush for chat %s.",
                    chat_id,
                )
            bot_cfg = self.config_mgr.bot
            admin_ids = (
                {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
                if bot_cfg is not None
                else set()
            )
            if int(chat_id) > 0 and int(chat_id) in admin_ids:
                await self._send_native_tool_abort_notice(
                    chat_id=chat_id,
                    tool_profile=self._extract_abort_profile(exc),
                )
        except Exception as exc:
            logger.exception("Error flushing debounced messages for chat %s: %s", chat_id, exc)

    async def _process_buffered_messages(self, chat_id: int) -> Optional[str]:
        """Pop and process all buffered incoming messages for a chat as one coalesced turn."""
        messages = self._debounce_buffers.pop(chat_id, [])
        self._debounce_started_at.pop(chat_id, None)
        current_task = asyncio.current_task()
        if self._debounce_tasks.get(chat_id) is current_task or current_task is None:
            self._debounce_tasks.pop(chat_id, None)
        elif (
            chat_id in self._debounce_tasks
            and self._debounce_tasks[chat_id].done()
        ):
            self._debounce_tasks.pop(chat_id, None)
        if not messages:
            return None
        return await self._process_coalesced_messages(messages)

    async def handle_incoming_message(self, incoming: IncomingMessage) -> Optional[str]:
        """Process a single incoming Telegram message immediately through heuristics, media, memory, and LLM."""
        return await self._process_coalesced_messages([incoming])

    def _evaluate_batch_heuristics(
        self,
        messages: List[IncomingMessage],
        bot_cfg: Any,
        time_since_bot: float,
        in_quiet: bool,
        now: float,
    ) -> Tuple[ReplyAction, IncomingMessage]:
        """Evaluate ReplyHeuristics across a single message or coalesced burst before media download."""
        spont_cfg = bot_cfg.behavior.spontaneous_replies
        active_cooldown = getattr(spont_cfg, "active_window_cooldown_seconds", None)

        # 1. Check direct mention / reply-to-bot across all messages in the batch first
        effective_dm_ids = self._effective_dm_user_ids(bot_cfg)
        must_reply_msg: Optional[IncomingMessage] = None
        for msg in messages:
            ev_obj: Any = msg.raw_event if msg.raw_event is not None else msg
            if msg.raw_event is None and not msg.is_private and self._is_private_dm_message(msg, bot_cfg):
                ev_obj = replace(msg, is_private=True)
            res = self.heuristics.should_reply(
                event=ev_obj,
                bot_id=self.userbot.bot_id,
                bot_names=bot_cfg.behavior.bot_names,
                allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                allowed_dm_user_ids=effective_dm_ids,
                ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                spontaneous_enabled=False,
                in_quiet_hours=in_quiet,
                time_since_last_bot_msg=time_since_bot,
                min_interval_seconds=float(spont_cfg.min_interval_seconds),
                probability=spont_cfg.probability,
                bot_username=self.userbot.bot_username,
                active_conversation_window_seconds=float(spont_cfg.active_conversation_window_seconds),
                max_per_hour=spont_cfg.max_per_hour,
                active_window_cooldown_seconds=active_cooldown,
                now=now,
            )
            if res == ReplyAction.MUST_REPLY:
                must_reply_msg = msg

        if must_reply_msg is not None:
            anchor = (
                messages[-1]
                if messages[-1].sender_id == must_reply_msg.sender_id
                else must_reply_msg
            )
            return ReplyAction.MUST_REPLY, anchor

        # 2. Evaluate spontaneous/active-window reply once for the entire batch
        primary = messages[-1]
        primary_is_priv = self._is_private_dm_message(primary, bot_cfg)
        if len(messages) == 1:
            eval_event = primary.raw_event if primary.raw_event is not None else primary
            if primary.raw_event is None and primary_is_priv and not primary.is_private:
                eval_event = replace(primary, is_private=True)
        else:
            combined_text = "\n".join((m.text or "").strip() for m in messages if (m.text or "").strip())
            raw = primary.raw_event

            class _CoalescedEvent:
                chat_id = primary.chat_id
                sender_id = primary.sender_id
                out = bool(getattr(raw, "out", False))
                is_reply = bool(getattr(raw, "is_reply", False) or primary.is_reply)
                reply_to_msg_id = getattr(raw, "reply_to_msg_id", primary.reply_to_msg_id)
                reply_to_sender_id = getattr(raw, "reply_to_sender_id", None)
                is_reply_to_bot = getattr(raw, "is_reply_to_bot", False)
                raw_text = combined_text
                text = combined_text
                entities = getattr(raw, "entities", None)
                message = getattr(raw, "message", None)
                is_private = primary_is_priv
                sender_is_bot = getattr(primary, "sender_is_bot", getattr(raw, "sender_is_bot", False))
                sender_username = getattr(primary, "sender_username", getattr(raw, "sender_username", None))
                sender = getattr(raw, "sender", None)

            eval_event = _CoalescedEvent()

        decision = self.heuristics.should_reply(
            event=eval_event,
            bot_id=self.userbot.bot_id,
            bot_names=bot_cfg.behavior.bot_names,
            allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
            allowed_dm_user_ids=effective_dm_ids,
            ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
            spontaneous_enabled=spont_cfg.enabled,
            in_quiet_hours=in_quiet,
            time_since_last_bot_msg=time_since_bot,
            min_interval_seconds=float(spont_cfg.min_interval_seconds),
            probability=spont_cfg.probability,
            bot_username=self.userbot.bot_username,
            active_conversation_window_seconds=float(spont_cfg.active_conversation_window_seconds),
            max_per_hour=spont_cfg.max_per_hour,
            active_window_cooldown_seconds=active_cooldown,
            now=now,
        )
        return decision, primary

    async def _save_incoming_batch(
        self,
        messages: List[IncomingMessage],
        now: float,
        media_target_id: Optional[int] = None,
        media_type: Optional[str] = None,
        media_summary: Optional[str] = None,
    ) -> None:
        """Persist one or more incoming messages to SQLite history."""
        for idx, msg in enumerate(messages):
            friend = self.config_mgr.get_friend_profile(msg.sender_id)
            author_display = friend.name if friend else msg.sender_name
            is_media_target = media_target_id is not None and id(msg) == media_target_id
            m_type = media_type if is_media_target else self._infer_cheap_media_type(msg)
            m_summary = media_summary if is_media_target else None
            await self.memory_store.save_message(
                ChatMessage(
                    telegram_msg_id=msg.msg_id or int(now * 1000) + idx,
                    chat_id=msg.chat_id,
                    user_id=msg.sender_id,
                    timestamp=int(now),
                    author_name=author_display,
                    text=msg.text,
                    reply_to_msg_id=msg.reply_to_msg_id,
                    media_type=m_type,
                    media_summary=m_summary,
                )
            )

    def _is_voice_message(self, incoming: IncomingMessage) -> bool:
        """Return True if an IncomingMessage carries a Telegram voice note."""
        raw_msg = getattr(incoming.raw_event, "message", incoming.raw_event)
        if raw_msg is None:
            return False
        if _has_media_flag(raw_msg, "voice"):
            return True
        detector = getattr(self.media_processor, "_detect_media_kind", None)
        if callable(detector):
            return detector(raw_msg) == "voice"
        return False

    @staticmethod
    def _enrich_message_with_transcript(msg: IncomingMessage, transcript: str) -> IncomingMessage:
        """Return a copy of IncomingMessage with voice transcript merged into text and raw_event."""
        clean_transcript = transcript.strip()
        if not clean_transcript:
            return msg
        existing_caption = (msg.text or "").strip()
        enriched_text = (
            f"{existing_caption}\n{clean_transcript}" if existing_caption else clean_transcript
        )
        return replace(
            msg,
            text=enriched_text,
            raw_event=_EventTextProxy(msg.raw_event, enriched_text),
        )

    def _is_batch_permitted(self, messages: List[IncomingMessage], bot_cfg: BotConfig) -> bool:
        """Return True if chat is whitelisted and batch contains at least one non-bot message."""
        allowed_chats = bot_cfg.telegram.allowed_chat_ids
        allowed_dms = self._effective_dm_user_ids(bot_cfg)
        allowed_group_ints = {int(cid) for cid in allowed_chats} if allowed_chats else set()
        allowed_dm_ints = {int(uid) for uid in allowed_dms} if allowed_dms else set()

        bot_id = getattr(self.userbot, "bot_id", None)
        ignored_usernames = {"hermes_kuk7yi6kfm3lcdfa_bot"}
        if bot_cfg.telegram.ignored_sender_usernames:
            ignored_usernames.update(
                u.lower().lstrip("@") for u in bot_cfg.telegram.ignored_sender_usernames
            )

        for m in messages:
            raw = m.raw_event
            if getattr(raw, "out", False):
                continue
            sender_id = getattr(raw, "sender_id", m.sender_id)
            if bot_id is not None and sender_id == bot_id:
                continue

            # Anti-loop check on sender
            sender = getattr(raw, "sender", None)
            sender_is_bot = bool(
                getattr(m, "sender_is_bot", False)
                or getattr(raw, "sender_is_bot", False)
                or (
                    sender is not None
                    and (
                        getattr(sender, "bot", False) is True
                        or getattr(sender, "is_bot", False) is True
                    )
                )
            )
            raw_username = (
                getattr(m, "sender_username", None)
                or getattr(raw, "sender_username", None)
                or (getattr(sender, "username", None) if sender is not None else None)
                or ""
            )
            clean_username = str(raw_username).lower().lstrip("@") if raw_username else ""
            if sender_is_bot or (clean_username and clean_username in ignored_usernames):
                continue

            # Whitelist verification (strictly separate group chats vs private DMs):
            is_priv = self._is_private_dm_message(m, bot_cfg)

            if is_priv:
                if allowed_dm_ints and (
                    int(m.chat_id) in allowed_dm_ints
                    or (sender_id is not None and int(sender_id) in allowed_dm_ints)
                ):
                    return True
            else:
                if int(m.chat_id) in allowed_group_ints:
                    return True
                if not allowed_group_ints and not allowed_dm_ints:
                    return True

        return False

    async def _process_coalesced_messages(self, messages: List[IncomingMessage]) -> Optional[str]:
        """Process a single message or debounced burst through heuristics, media, memory, and LLM."""
        if not messages:
            return None
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            return None

        self._sync_voice_configs(bot_cfg)
        await self._ensure_ops_tools_mode_loaded()

        working_messages = list(messages)
        if not self._is_batch_permitted(working_messages, bot_cfg):
            await self._save_incoming_batch(working_messages, now=time.time())
            return None

        if (
            len(self._chat_locks) > self._MAX_TRACKED_CHATS
            or len(self.chat_states) > self._MAX_TRACKED_CHATS
        ):
            self._prune_chat_caches()

        chat_id = messages[-1].chat_id
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        reply_result: Optional[str] = None
        async with lock:
            now = time.time()
            state = self.chat_states.setdefault(
                chat_id,
                {"last_chat_msg_time": now, "last_bot_msg_time": 0.0},
            )
            state["last_chat_msg_time"] = now

            in_quiet = self.proactive_engine.is_in_quiet_hours()
            time_since_bot = now - state.get("last_bot_msg_time", 0.0)

            media_candidates = [m for m in working_messages if self._has_incoming_media(m)]
            media_target = media_candidates[-1] if media_candidates else None
            voice_candidates = [m for m in working_messages if self._is_voice_message(m)]

            if voice_candidates and bot_cfg.voice.stt.enabled:
                voice_target = (
                    media_target
                    if media_target is not None and self._is_voice_message(media_target)
                    else voice_candidates[-1]
                )
                raw_voice_msg = getattr(voice_target.raw_event, "message", voice_target.raw_event)
                async with self.media_processor.managed_media(
                    raw_voice_msg, client=self.userbot.client
                ) as processed:
                    media_summary = self.summarize_processed_media(processed)
                    media_type = processed.media_type if processed else None
                    media_text = processed.text_content if processed else None
                    attachments = self.build_media_attachments(processed)

                    if media_text and media_text.strip():
                        enriched_voice = self._enrich_message_with_transcript(
                            voice_target, media_text
                        )
                        target_idx = next(
                            (i for i, m in enumerate(working_messages) if m is voice_target),
                            -1,
                        )
                        if target_idx >= 0:
                            working_messages[target_idx] = enriched_voice
                        if media_target is voice_target:
                            media_target = enriched_voice
                        voice_target = enriched_voice

                    decision, primary = self._evaluate_batch_heuristics(
                        messages=working_messages,
                        bot_cfg=bot_cfg,
                        time_since_bot=time_since_bot,
                        in_quiet=in_quiet,
                        now=now,
                    )

                    await self._save_incoming_batch(
                        working_messages,
                        now=now,
                        media_target_id=id(voice_target),
                        media_type=media_type or "voice",
                        media_summary=media_summary,
                    )

                    if decision == ReplyAction.IGNORE:
                        return None

                    combined_text = "\n".join(
                        (m.text or "").strip()
                        for m in working_messages
                        if (m.text or "").strip()
                    ).strip()
                    is_voice_target = self._is_voice_message(primary) or (
                        voice_target.sender_id == primary.sender_id
                    )
                    prompt_media_text = (
                        None
                        if (
                            media_type == "voice"
                            and media_text
                            and media_text.strip() in combined_text
                        )
                        else media_text
                    )
                    reply_result = await self._complete_reply_turn(
                        chat_id=chat_id,
                        bot_cfg=bot_cfg,
                        state=state,
                        now=now,
                        decision=decision,
                        primary=primary,
                        messages=working_messages,
                        combined_text=combined_text,
                        media_target=voice_target,
                        is_reply_media=False,
                        media_summary=media_summary,
                        media_text=prompt_media_text,
                        raw_media_text=media_text,
                        attachments=attachments,
                        is_voice_target=is_voice_target,
                    )
            else:
                # Explicit /imagine and /img commands: intercept AFTER permission check and BEFORE heuristics
                # so bare `/imagine <prompt>` works in groups without hitting quiet-hours / spontaneous probability.
                imagine_match = self._find_imagine_command(working_messages)
                if imagine_match is not None:
                    cmd_msg, cmd_res = imagine_match
                    if cmd_res.addressed_to_other_bot:
                        await self._save_incoming_batch(working_messages, now=now)
                        return None
                    if cmd_res.is_command:
                        return await self._handle_imagine_command(
                            chat_id=chat_id,
                            bot_cfg=bot_cfg,
                            state=state,
                            now=now,
                            cmd_msg=cmd_msg,
                            cmd_res=cmd_res,
                            messages=working_messages,
                        )

                # Explicit /ops, /tools, /safemode toggle commands: intercept BEFORE heuristics
                ops_mode_match = self._find_ops_mode_command(working_messages)
                if ops_mode_match is not None:
                    cmd_msg, action, addressed_to_other_bot = ops_mode_match
                    if addressed_to_other_bot:
                        await self._save_incoming_batch(working_messages, now=now)
                        return None
                    return await self._handle_ops_mode_command(
                        chat_id=chat_id,
                        bot_cfg=bot_cfg,
                        now=now,
                        cmd_msg=cmd_msg,
                        action=action,
                        messages=working_messages,
                    )

                # Explicit /quota, /backup, /evolve, /self_edit commands: intercept BEFORE heuristics
                ops_action_match = self._find_ops_action_command(working_messages)
                if ops_action_match is not None:
                    cmd_msg, cmd_name, cmd_arg, addressed_to_other_bot = ops_action_match
                    if addressed_to_other_bot:
                        await self._save_incoming_batch(working_messages, now=now)
                        return None
                    handled, action_reply = await self._handle_ops_action_command(
                        chat_id=chat_id,
                        bot_cfg=bot_cfg,
                        now=now,
                        cmd_msg=cmd_msg,
                        cmd_name=cmd_name,
                        cmd_arg=cmd_arg,
                        messages=working_messages,
                    )
                    if handled:
                        return action_reply

                # Standard non-voice path: evaluate heuristics BEFORE downloading/transcoding any media
                decision, primary = self._evaluate_batch_heuristics(
                    messages=working_messages,
                    bot_cfg=bot_cfg,
                    time_since_bot=time_since_bot,
                    in_quiet=in_quiet,
                    now=now,
                )

                if (
                    decision == ReplyAction.SPONTANEOUS_REPLY
                    and not self.quota_manager.should_allow_spontaneous()
                ):
                    decision = ReplyAction.IGNORE

                if decision == ReplyAction.IGNORE:
                    await self._save_incoming_batch(working_messages, now=now)
                    return None

                combined_text = "\n".join(
                    (m.text or "").strip() for m in working_messages if (m.text or "").strip()
                ).strip()

                reply_media_raw: Any = None
                is_reply_media = False
                if media_target is None:
                    for m in reversed(working_messages):
                        resolved_reply_media = await self._resolve_reply_media_raw(m)
                        if resolved_reply_media is not None:
                            reply_media_raw = resolved_reply_media
                            is_reply_media = True
                            break

                # Skip blind media-only non-direct messages when provider cannot see images
                if (
                    media_target is not None
                    and not combined_text
                    and decision != ReplyAction.MUST_REPLY
                ):
                    await self._save_incoming_batch(working_messages, now=now)
                    return None

                if media_target is not None:
                    raw_msg = getattr(media_target.raw_event, "message", media_target.raw_event)
                    if isinstance(raw_msg, str):
                        raw_msg = media_target.raw_event
                elif reply_media_raw is not None:
                    raw_msg = reply_media_raw
                else:
                    raw_msg = getattr(primary.raw_event, "message", primary.raw_event)
                    if isinstance(raw_msg, str):
                        raw_msg = primary.raw_event
                async with self.media_processor.managed_media(raw_msg, client=self.userbot.client) as processed:
                    media_summary = self.summarize_processed_media(processed)
                    media_type = processed.media_type if processed else None
                    media_text = processed.text_content if processed else None
                    attachments = self.build_media_attachments(processed)

                    await self._save_incoming_batch(
                        working_messages,
                        now=now,
                        media_target_id=(
                            id(media_target)
                            if (media_target is not None and not is_reply_media)
                            else None
                        ),
                        media_type=media_type if not is_reply_media else None,
                        media_summary=media_summary if not is_reply_media else None,
                    )

                    is_voice_target = self._is_voice_message(primary) or (
                        media_target is not None
                        and not is_reply_media
                        and media_target.sender_id == primary.sender_id
                        and self._is_voice_message(media_target)
                    )
                    reply_result = await self._complete_reply_turn(
                        chat_id=chat_id,
                        bot_cfg=bot_cfg,
                        state=state,
                        now=now,
                        decision=decision,
                        primary=primary,
                        messages=working_messages,
                        combined_text=combined_text,
                        media_target=media_target,
                        is_reply_media=is_reply_media,
                        media_summary=media_summary,
                        media_text=media_text,
                        raw_media_text=media_text,
                        attachments=attachments,
                        is_voice_target=is_voice_target,
                    )

        if reply_result:
            await self._maybe_run_compaction(chat_id)
        return reply_result

    async def _maybe_run_compaction(self, chat_id: int) -> None:
        """Run chat history compaction OUTSIDE `_chat_locks` so LLM summarization never blocks replies."""
        if self._is_provider_rate_limited(role="compaction"):
            return
        comp_lock = self._compaction_locks.setdefault(chat_id, asyncio.Lock())
        if comp_lock.locked():
            return
        async with comp_lock:
            if self._is_provider_rate_limited(role="compaction"):
                return
            self._sync_bot_identity_to_memory_engines()
            bot_uid = getattr(self.userbot, "bot_id", None)
            excluded_uids = [int(bot_uid)] if bot_uid else None
            await self.compaction_engine.run_compaction_for_chat(
                chat_id,
                excluded_user_ids=excluded_uids,
            )

    def _find_imagine_command(
        self,
        messages: Sequence[IncomingMessage],
    ) -> Optional[Tuple[IncomingMessage, ImageCommandParseResult]]:
        """Return `(msg, parse_result)` if any message in the batch is a `/imagine` or `/img` command."""
        bot_username = getattr(self.userbot, "bot_username", None)
        for msg in reversed(messages):
            candidate_text = msg.text or getattr(msg.raw_event, "caption", None) or ""
            if not candidate_text.strip().startswith("/"):
                continue
            parsed = parse_imagine_command(candidate_text, bot_username=bot_username)
            if parsed.is_command or parsed.addressed_to_other_bot:
                return msg, parsed
        return None

    @staticmethod
    async def _resolve_reply_media_raw(msg: IncomingMessage) -> Any:
        """Resolve replied-to media message across Bot API and Telethon MTProto events."""
        r_media = getattr(msg, "reply_to_media_event", None) or getattr(
            getattr(msg, "raw_event", None), "reply_to_media_event", None
        )
        if r_media is not None:
            raw_reply_media = getattr(r_media, "message", r_media)
            if isinstance(raw_reply_media, str):
                return r_media
            return raw_reply_media

        raw = getattr(msg, "raw_event", None)
        if raw is None or not (msg.is_reply or getattr(raw, "is_reply", False)):
            return None

        get_reply_fn = getattr(raw, "get_reply_message", None)
        if callable(get_reply_fn) and type(get_reply_fn).__name__ != "MagicMock":
            try:
                reply_msg = get_reply_fn()
                if inspect.isawaitable(reply_msg):
                    reply_msg = await reply_msg
                if reply_msg is not None and any(
                    _has_media_flag(reply_msg, attr)
                    for attr in (
                        "photo",
                        "sticker",
                        "voice",
                        "video_note",
                        "animation",
                        "video",
                        "document",
                        "media",
                    )
                ):
                    return reply_msg
            except Exception as exc:
                logger.debug("Failed fetching replied media message: %s", exc)
        return None

    async def _extract_image_attachments_from_raw(
        self,
        raw_msg: Any,
    ) -> Tuple[List[MediaAttachment], Optional[str], Optional[str]]:
        """Download and buffer image attachments from a raw Telegram message for image-to-image."""
        if raw_msg is None:
            return [], None, None
        try:
            async with self.media_processor.managed_media(
                raw_msg, client=self.userbot.client
            ) as processed:
                summary = self.summarize_processed_media(processed)
                m_type = processed.media_type if processed else None
                raw_atts = self.build_media_attachments(processed)
                buffered: List[MediaAttachment] = []
                for att in raw_atts:
                    if not (att.mime_type or "").lower().startswith("image/"):
                        continue
                    raw_bytes = att.get_bytes()
                    if raw_bytes:
                        buffered.append(
                            MediaAttachment(
                                data=raw_bytes,
                                mime_type=att.mime_type,
                                description=att.description,
                                file_name=att.file_name,
                            )
                        )
                return buffered, m_type, summary
        except Exception as exc:
            logger.warning("Failed to prepare reference image attachment: %s", exc)
            return [], None, None

    async def _keep_chat_action_alive(
        self,
        chat_id: int,
        action: str = "upload-photo",
        interval_seconds: float = 4.5,
    ) -> None:
        """Periodically refresh a Telegram chat action while image generation is in flight."""
        if not hasattr(self.userbot, "send_typing_action"):
            return
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                try:
                    await self.userbot.send_typing_action(chat_id, action=action)
                except TypeError:
                    await self.userbot.send_typing_action(chat_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Chat action keepalive (%s) failed: %s", action, exc)

    @staticmethod
    def _build_image_usage_info(usage: Any) -> Optional[UsageInfo]:
        """Convert GeneratedImageResult.usage dict or UsageInfo into UsageInfo for QuotaBudgetManager."""
        if usage is None:
            return None
        if isinstance(usage, UsageInfo):
            return usage
        if isinstance(usage, dict) and usage:
            try:
                return UsageInfo.model_validate(usage)
            except Exception:
                numeric_fields = {
                    k: int(v)
                    for k, v in usage.items()
                    if k in UsageInfo.model_fields
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                }
                if numeric_fields:
                    return UsageInfo(**numeric_fields)
        return None

    async def _handle_imagine_command(
        self,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        state: Dict[str, float],
        now: float,
        cmd_msg: IncomingMessage,
        cmd_res: ImageCommandParseResult,
        messages: List[IncomingMessage],
    ) -> Optional[str]:
        """Handle explicit `/imagine` and `/img` commands in allowed groups and DMs."""
        reply_to_id = cmd_msg.msg_id if cmd_msg.msg_id else None
        img_cfg = bot_cfg.image_generation

        if not img_cfg.enabled:
            await self._save_incoming_batch(messages, now=now)
            disabled_msg = "Генерация картинок сейчас отключена."
            await self.send_sanitized_message(
                chat_id=chat_id,
                text=disabled_msg,
                reply_to_msg_id=reply_to_id,
            )
            return disabled_msg

        prompt_text = (cmd_res.prompt or "").strip()
        reply_text = (
            getattr(cmd_msg, "reply_to_text", None)
            or getattr(cmd_msg.raw_event, "reply_to_text", None)
            or ""
        ).strip()
        if not prompt_text and reply_text:
            prompt_text = reply_text
        elif prompt_text and reply_text:
            prompt_text = f"{prompt_text}\nКонтекст сообщения: {reply_text[:400]}"

        if not prompt_text:
            await self._save_incoming_batch(messages, now=now)
            usage_hint = (
                "Напиши, что нарисовать: /imagine <описание> [--ar 16:9] "
                "(или ответь командой на сообщение/фото)."
            )
            await self.send_sanitized_message(
                chat_id=chat_id,
                text=usage_hint,
                reply_to_msg_id=reply_to_id,
            )
            return usage_hint

        # Optional reference image from command photo caption or replied-to photo/static sticker
        reference_images: List[MediaAttachment] = []
        saved_media_target_id: Optional[int] = None
        saved_media_type: Optional[str] = None
        saved_media_summary: Optional[str] = None

        if self._has_incoming_media(cmd_msg):
            raw_cmd_msg = getattr(cmd_msg.raw_event, "message", cmd_msg.raw_event)
            if isinstance(raw_cmd_msg, str):
                raw_cmd_msg = cmd_msg.raw_event
            reference_images, saved_media_type, saved_media_summary = (
                await self._extract_image_attachments_from_raw(raw_cmd_msg)
            )
            saved_media_target_id = id(cmd_msg)
        else:
            raw_reply_media = await self._resolve_reply_media_raw(cmd_msg)
            if raw_reply_media is not None:
                reference_images, _, _ = await self._extract_image_attachments_from_raw(
                    raw_reply_media
                )

        await self._save_incoming_batch(
            messages,
            now=now,
            media_target_id=saved_media_target_id,
            media_type=saved_media_type,
            media_summary=saved_media_summary,
        )

        return await self._execute_image_generation(
            chat_id=chat_id,
            user_id=cmd_msg.sender_id,
            bot_cfg=bot_cfg,
            state=state,
            prompt=prompt_text,
            caption=None,
            aspect_ratio=cmd_res.aspect_ratio,
            reference_images=reference_images or None,
            reply_to_msg_id=reply_to_id,
            decision=ReplyAction.MUST_REPLY,
        )

    async def _execute_image_generation(
        self,
        *,
        chat_id: int,
        user_id: int,
        bot_cfg: BotConfig,
        state: Dict[str, float],
        prompt: str,
        caption: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        reference_images: Optional[List[MediaAttachment]] = None,
        reply_to_msg_id: Optional[int] = None,
        decision: ReplyAction = ReplyAction.MUST_REPLY,
        send_error_reply: bool = True,
    ) -> Optional[str]:
        """Unified image generation pipeline: rate limit, chat action, provider call, delivery & memory."""
        img_cfg = bot_cfg.image_generation
        clean_prompt = (prompt or "").strip()[: int(img_cfg.max_prompt_chars)].strip()
        if not clean_prompt:
            return None

        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
        is_admin = int(user_id) in admin_ids

        try:
            reservation_token = self._image_limiter.check_and_reserve(
                user_id=user_id,
                chat_id=chat_id,
                is_admin=is_admin,
            )
        except ImageRateLimitError as exc:
            safe_notice = self.sanitizer.sanitize(exc.user_message)
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=safe_notice,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return safe_notice

        generate_fn = getattr(self.provider, "generate_image", None)
        if not callable(generate_fn):
            self._image_limiter.refund(user_id, chat_id, reservation_token)
            unavail_msg = "Генерация картинок сейчас недоступна для текущего провайдера."
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=unavail_msg,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return unavail_msg

        # Signal upload_photo immediately and keep it refreshed during generation
        if hasattr(self.userbot, "send_typing_action"):
            try:
                try:
                    await self.userbot.send_typing_action(chat_id, action="upload-photo")
                except TypeError:
                    await self.userbot.send_typing_action(chat_id)
            except Exception as exc:
                logger.debug("Initial upload-photo chat action failed: %s", exc)

        keepalive_task = asyncio.create_task(
            self._keep_chat_action_alive(chat_id, action="upload-photo", interval_seconds=4.5)
        )
        try:
            result = await generate_fn(
                clean_prompt,
                aspect_ratio=aspect_ratio,
                reference_images=reference_images,
                model=img_cfg.model,
                timeout_seconds=img_cfg.timeout_seconds,
                output_dir=img_cfg.output_dir,
                max_image_size_mb=img_cfg.max_image_size_mb,
                cleanup_artifacts=img_cfg.cleanup_after_send,
            )
            if not isinstance(result, GeneratedImageResult) or not result.image_bytes:
                raise ImageValidationError("Image provider returned an empty or invalid image result.")
        except ImagePolicyRefusalError as exc:
            # Do not refund on policy refusals to prevent repeated policy-probing bypasses.
            logger.info("Image generation refused by policy in chat %s: %s", chat_id, exc)
            refusal_msg = "Не могу это нарисовать — сработало ограничение контента."
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=refusal_msg,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return refusal_msg
        except ImageTimeoutError as exc:
            self._image_limiter.refund(user_id, chat_id, reservation_token)
            logger.warning("Image generation timed out in chat %s: %s", chat_id, exc)
            timeout_msg = "Не успел дорисовать картинку — время ожидания вышло. Попробуй ещё раз чуть позже."
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=timeout_msg,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return timeout_msg
        except (ImageValidationError, ImageGenerationError, Exception) as exc:
            self._image_limiter.refund(user_id, chat_id, reservation_token)
            verdict = getattr(exc, "verdict", None) or classify_error(exc)
            logger.warning("Image generation failed in chat %s: %s", chat_id, exc)
            if verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.BILLING):
                err_msg = "Лимит генерации картинок временно исчерпан, попробуй чуть позже."
            else:
                err_msg = "Не получилось сгенерировать картинку, попробуй ещё раз чуть позже."
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=err_msg,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return err_msg
        finally:
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

        # Record LLM/image token usage immediately after generation succeeds, BEFORE Telegram
        # upload so spent tokens are tracked even if network delivery fails.
        usage_info = self._build_image_usage_info(result.usage)
        if usage_info is not None and self.quota_manager is not None:
            try:
                await self.quota_manager.record_usage(
                    usage_info,
                    model=result.model_used or img_cfg.model or bot_cfg.antigravity.model,
                    source="image_gen",
                )
            except Exception as exc:
                logger.debug("Failed recording image generation usage: %s", exc)

        raw_caption = (caption or "").strip() or (result.caption_suggestion or "").strip()
        safe_caption: Optional[str] = None
        if raw_caption:
            sanitized_cap = self.sanitizer.sanitize(raw_caption).strip()
            if len(sanitized_cap) > 1024:
                sanitized_cap = sanitized_cap[:1021].rstrip() + "..."
            safe_caption = sanitized_cap or None

        send_as_doc = should_send_as_document(
            len(result.image_bytes),
            result.width,
            result.height,
            threshold_mb=img_cfg.send_as_document_threshold_mb,
        ) or not hasattr(self.userbot, "send_photo")

        try:
            if send_as_doc:
                sent_res = await self.userbot.send_document(
                    chat_id,
                    result.image_bytes,
                    filename=result.filename,
                    caption=safe_caption,
                    mime_type=result.mime_type,
                    reply_to_message_id=reply_to_msg_id,
                    simulate_typing=False,
                )
                saved_media_type = "document"
            else:
                sent_res = await self.userbot.send_photo(
                    chat_id,
                    result.image_bytes,
                    caption=safe_caption,
                    filename=result.filename,
                    mime_type=result.mime_type,
                    reply_to_message_id=reply_to_msg_id,
                )
                saved_media_type = "photo"
        except Exception as exc:
            self._image_limiter.refund(user_id, chat_id, reservation_token)
            logger.warning("Failed delivering generated image to chat %s: %s", chat_id, exc)
            delivery_err = "Картинка сгенерировалась, но не удалось отправить её в Telegram. Попробуй чуть позже."
            if send_error_reply:
                await self.send_sanitized_message(
                    chat_id=chat_id,
                    text=delivery_err,
                    reply_to_msg_id=reply_to_msg_id,
                )
            return delivery_err

        sent_ts = time.time()
        state["last_bot_msg_time"] = sent_ts
        if decision != ReplyAction.SPONTANEOUS_REPLY:
            self.heuristics.record_reply(chat_id, timestamp=sent_ts)

        safe_prompt = self.sanitizer.sanitize(result.prompt_used or clean_prompt).strip()
        media_summary = f"Сгенерированное изображение: {safe_prompt[:220]}"
        history_text = safe_caption or f"[Изображение: {safe_prompt[:180]}]"
        sent_msg_id = self._extract_sent_message_id(sent_res)
        bot_name = bot_cfg.behavior.bot_names[0] if bot_cfg.behavior.bot_names else "Потужнич"

        await self.memory_store.save_message(
            ChatMessage(
                telegram_msg_id=sent_msg_id,
                chat_id=chat_id,
                user_id=self.userbot.bot_id or 1,
                timestamp=int(sent_ts),
                author_name=bot_name,
                text=history_text,
                reply_to_msg_id=reply_to_msg_id,
                media_type=saved_media_type,
                media_summary=media_summary,
            )
        )
        return history_text

    async def _ensure_ops_tools_mode_loaded(self) -> str:
        """Load persisted ops_tools_mode from SQLite once on startup or first message."""
        if self._ops_tools_mode_loaded:
            return self._ops_tools_mode
        self._ops_tools_mode_loaded = True
        getter = getattr(self.memory_store, "get_setting", None)
        if callable(getter):
            try:
                saved = await getter(self._OPS_MODE_SETTING_KEY)
                if isinstance(saved, str):
                    norm = saved.strip().lower()
                    if norm in self._VALID_OPS_TOOLS_MODES:
                        self._ops_tools_mode = norm
            except Exception as exc:
                logger.warning("Failed to load persisted %s: %s", self._OPS_MODE_SETTING_KEY, exc)
        return self._ops_tools_mode

    @staticmethod
    def _is_admin_sender(msg: IncomingMessage, bot_cfg: BotConfig) -> bool:
        """Return True if msg was sent by a non-bot configured administrator."""
        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
        if not admin_ids:
            return False
        raw = getattr(msg, "raw_event", None)
        if (
            getattr(msg, "sender_is_bot", False)
            or getattr(raw, "sender_is_bot", False)
            or getattr(raw, "out", False)
        ):
            return False
        return int(msg.sender_id) in admin_ids

    @classmethod
    def _is_admin_private_dm(
        cls,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        primary: IncomingMessage,
        messages: List[IncomingMessage],
    ) -> bool:
        """Return True ONLY for non-bot private DMs from admin_user_ids."""
        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
        if not admin_ids or int(chat_id) <= 0:
            return False

        if not cls._is_private_dm_message(primary, bot_cfg):
            return False

        raw = getattr(primary, "raw_event", None)
        if int(primary.sender_id) != int(chat_id) or int(primary.sender_id) not in admin_ids:
            return False

        if (
            getattr(primary, "sender_is_bot", False)
            or getattr(raw, "sender_is_bot", False)
            or getattr(raw, "out", False)
        ):
            return False

        for msg in messages:
            m_raw = getattr(msg, "raw_event", None)
            if (
                getattr(msg, "sender_is_bot", False)
                or getattr(m_raw, "sender_is_bot", False)
                or getattr(m_raw, "out", False)
                or int(msg.sender_id) not in admin_ids
            ):
                return False

        return True

    @staticmethod
    def _split_slash_command(
        text: str,
        bot_username: Optional[str] = None,
    ) -> Optional[Tuple[str, str, bool]]:
        """Parse leading `/cmd[@bot] [arg]` returning `(cmd_lower, arg, addressed_to_other_bot)`."""
        stripped = (text or "").strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped.split(None, 1)
        head = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if "@" in head:
            cmd_part, target_bot = head.split("@", 1)
            cmd_name = cmd_part.lower()
            norm_target = target_bot.strip().lstrip("@").lower()
            norm_self = (bot_username or "").strip().lstrip("@").lower()
            if norm_target and norm_self and norm_target != norm_self:
                return (cmd_name, arg, True)
        else:
            cmd_name = head.lower()
        return (cmd_name, arg, False)

    @classmethod
    def _parse_ops_mode_command(
        cls,
        text: str,
        bot_username: Optional[str] = None,
    ) -> Optional[Tuple[str, bool]]:
        """Parse `/ops`, `/tools`, or `/safemode` toggle commands into `(action, addressed_to_other_bot)`."""
        parsed = cls._split_slash_command(text, bot_username=bot_username)
        if parsed is None:
            return None
        cmd_name, arg, other_bot = parsed
        if cmd_name not in ("/ops", "/tools", "/safemode"):
            return None
        if other_bot:
            return ("status", True)

        token = arg.split()[0].strip().lower() if arg else ""
        if cmd_name == "/safemode":
            if not token or token in ("status", "state"):
                return ("status", False)
            if token in ("on", "enable", "enabled", "1", "true", "вкл"):
                return ("off", False)
            if token in ("off", "disable", "disabled", "0", "false", "выкл"):
                return ("on", False)
            return ("help", False)

        if not token or token in ("status", "state"):
            return ("status", False)
        if token in ("on", "enable", "enabled", "1", "true", "all", "вкл"):
            return ("on", False)
        if token in ("off", "disable", "disabled", "0", "false", "safe", "chat", "выкл"):
            return ("off", False)
        if token in ("admin", "dm", "admin_only", "admin-only"):
            return ("admin", False)
        return ("help", False)

    def _find_ops_mode_command(
        self,
        messages: Sequence[IncomingMessage],
    ) -> Optional[Tuple[IncomingMessage, str, bool]]:
        """Find a `/ops`, `/tools`, or `/safemode` command in the message batch."""
        bot_username = getattr(self.userbot, "bot_username", None)
        for msg in reversed(messages):
            candidate = (msg.text or "").strip()
            if not candidate.startswith("/"):
                continue
            parsed = self._parse_ops_mode_command(candidate, bot_username=bot_username)
            if parsed is not None:
                return msg, parsed[0], parsed[1]
        return None

    async def _handle_ops_mode_command(
        self,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        now: float,
        cmd_msg: IncomingMessage,
        action: str,
        messages: List[IncomingMessage],
    ) -> str:
        """Handle `/ops on|off|admin|status` mode switching with SQLite persistence."""
        await self._save_incoming_batch(messages, now=now)
        mode_descriptions = {
            "on": "🔓 ВКЛ (`on`) — полный набор инструментов Antigravity (команды, файлы, редактирование, субагенты, задачи, /evolve, /quota, /backup; без root) открыт во всех разрешённых группах и ЛС.",
            "admin": "🛡 ТОЛЬКО АДМИН (`admin`) — полный набор инструментов работает только в ЛС администратора; группы и остальные ЛС в безопасном режиме `chat`.",
            "off": "🔒 ВЫКЛ (`off`) — расширенный режим отключён везде; бот работает только в безопасном режиме `chat` (веб-поиск и чтение ссылок).",
        }
        if action in ("status", "help"):
            desc = mode_descriptions.get(self._ops_tools_mode, self._ops_tools_mode)
            reply_text = (
                f"Режим расширенных инструментов: {desc}\n"
                "Команды (для админа): `/ops off` — отключить, `/ops on` — включить везде, "
                "`/ops admin` — только ЛС админа, `/ops status` — статус."
            )
        elif not self._is_admin_sender(cmd_msg, bot_cfg):
            reply_text = "⛔ Переключать режим расширенных инструментов может только администратор."
        else:
            self._ops_tools_mode = action
            setter = getattr(self.memory_store, "set_setting", None)
            if callable(setter):
                try:
                    await setter(self._OPS_MODE_SETTING_KEY, action)
                except Exception as exc:
                    logger.warning("Failed persisting %s=%s: %s", self._OPS_MODE_SETTING_KEY, action, exc)
            desc = mode_descriptions.get(action, action)
            reply_text = f"Готово. {desc}"

        safe_reply = self.sanitizer.sanitize(reply_text)
        await self.userbot.send_reply(
            chat_id=chat_id,
            text=safe_reply,
            reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
        )
        return safe_reply

    def _find_ops_action_command(
        self,
        messages: Sequence[IncomingMessage],
    ) -> Optional[Tuple[IncomingMessage, str, str, bool]]:
        """Find `/quota`, `/backup`, `/evolve`, or `/self_edit` in the message batch."""
        bot_username = getattr(self.userbot, "bot_username", None)
        for msg in reversed(messages):
            candidate = (msg.text or "").strip()
            if not candidate.startswith("/"):
                continue
            parsed = self._split_slash_command(candidate, bot_username=bot_username)
            if parsed is None:
                continue
            cmd_name, arg, other_bot = parsed
            if cmd_name in ("/quota", "/backup", "/evolve", "/self_edit"):
                return msg, cmd_name, arg, other_bot
        return None

    async def _handle_ops_action_command(
        self,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        now: float,
        cmd_msg: IncomingMessage,
        cmd_name: str,
        cmd_arg: str,
        messages: List[IncomingMessage],
    ) -> Tuple[bool, Optional[str]]:
        """Execute `/quota`, `/backup`, `/evolve`, `/self_edit` respecting active ops_tools_mode."""
        turn_profile, _, _ = self._resolve_turn_tool_profile(
            chat_id=chat_id,
            bot_cfg=bot_cfg,
            primary=cmd_msg,
            messages=messages,
        )
        is_admin = self._is_admin_sender(cmd_msg, bot_cfg)
        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}

        if cmd_name == "/quota":
            if turn_profile != "ops" and not is_admin:
                return False, None
            await self._save_incoming_batch(messages, now=now)
            await asyncio.to_thread(self.quota_manager.sync_live_quota)
            report = self.sanitizer.sanitize(self.quota_manager.format_quota_report())
            await self.userbot.send_reply(
                chat_id=chat_id,
                text=report,
                reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
            )
            return True, report

        if cmd_name == "/backup":
            if turn_profile != "ops" and not is_admin:
                return False, None
            await self._save_incoming_batch(messages, now=now)
            res = await self._run_memory_backup_once()
            if res is None:
                msg = "Бэкап не выполнен: файл базы памяти ещё не создан."
            else:
                msg = (
                    f"Бэкап памяти создан: `{res.backup_path.name}` "
                    f"({res.size_bytes / 1024:.1f} KiB, удалено старых: {res.pruned_count})."
                )
            msg = self.sanitizer.sanitize(msg)
            await self.userbot.send_reply(
                chat_id=chat_id,
                text=msg,
                reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
            )
            return True, msg

        if cmd_name in ("/evolve", "/self_edit"):
            if not admin_ids:
                return False, None
            if turn_profile != "ops":
                await self._save_incoming_batch(messages, now=now)
                denied = self.sanitizer.sanitize(
                    f"🔒 Самоэволюция недоступна: расширенный режим инструментов отключён "
                    f"(текущий режим: `{self._ops_tools_mode}`). Администратор может включить его командой `/ops on`."
                )
                await self.userbot.send_reply(
                    chat_id=chat_id,
                    text=denied,
                    reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
                )
                return True, denied

            if not cmd_arg:
                await self._save_incoming_batch(messages, now=now)
                usage_msg = "Использование: `/evolve <описание задачи для самоэволюции>`"
                await self.userbot.send_reply(
                    chat_id=chat_id,
                    text=usage_msg,
                    reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
                )
                return True, usage_msg

            if self.evolution_manager is None:
                return False, None

            await self._save_incoming_batch(messages, now=now)
            allow_non_admin = self._ops_tools_mode == "on"
            try:
                evo_res = await self.evolution_manager.run_admin_task(
                    user_id=cmd_msg.sender_id,
                    task=cmd_arg,
                    allow_non_admin=allow_non_admin,
                )
            except TypeError:
                evo_res = await self.evolution_manager.run_admin_task(
                    user_id=cmd_msg.sender_id,
                    task=cmd_arg,
                )
            if evo_res is None:
                reply_msg = "⚠️ Самоэволюция не выполнена (отключена в конфиге, отклонена или достигнут лимит)."
            else:
                verdict = getattr(evo_res, "verdict", "")
                deploy_res = getattr(evo_res, "deploy_result", "")
                summary = getattr(evo_res, "summary", "") or str(evo_res)
                reply_msg = f"🛠 Self-Edit ({verdict or 'done'} / {deploy_res or 'n/a'}): {summary}"
            reply_msg = self.sanitizer.sanitize(reply_msg)
            await self.userbot.send_reply(
                chat_id=chat_id,
                text=reply_msg,
                reply_to_msg_id=cmd_msg.msg_id if cmd_msg.msg_id else None,
            )
            return True, str(evo_res) if evo_res is not None else reply_msg

        return False, None

    @staticmethod
    def _resolve_ops_workspace(agy_cfg: Any) -> Optional[str]:
        """Resolve ops working directory: explicit `ops_cwd` override, else a lightweight workspace.

        The default workspace (`antigravity.ops_workspace_dir`) is created automatically and
        deliberately contains no AGENTS.md, so ops turns do not pull repo instructions into context.
        """
        if agy_cfg is None:
            return None
        explicit = getattr(agy_cfg, "ops_cwd", None)
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
        workspace = getattr(agy_cfg, "ops_workspace_dir", None)
        if not isinstance(workspace, str) or not workspace.strip():
            return None
        try:
            path = Path(workspace.strip()).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            path.mkdir(parents=True, exist_ok=True)
            return str(path.resolve())
        except OSError as exc:
            logger.warning("Cannot create ops workspace %s: %s", workspace, exc)
            return None

    _ABORT_PROFILE_RE = re.compile(r"\bprofile=([A-Za-z0-9_.:-]+)")

    @classmethod
    def _extract_abort_profile(cls, exc: BaseException) -> str:
        """Extract sanitized profile label from an AntigravityNativeToolAbort message."""
        match = cls._ABORT_PROFILE_RE.search(str(exc))
        return match.group(1) if match else "chat"

    async def _send_native_tool_abort_notice(
        self,
        *,
        chat_id: int,
        tool_profile: Optional[str] = None,
        reply_to_msg_id: Optional[int] = None,
    ) -> None:
        """Notify direct callers/admin DMs with a sanitized notice when a native tool call aborts."""
        profile_label = (tool_profile or "chat").strip() or "chat"
        notice = self.sanitizer.sanitize(
            f"⚠️ Сработала защита инструментов (profile={profile_label}, запрос отклонён политикой безопасности)."
        )
        try:
            await self.userbot.send_reply(
                chat_id=chat_id,
                text=notice,
                reply_to_msg_id=reply_to_msg_id,
                simulate_typing=False,
            )
        except Exception as notify_exc:
            logger.debug(
                "Failed to send native tool abort notice to chat %s: %s",
                chat_id,
                self.sanitizer.sanitize(str(notify_exc)),
            )

    @classmethod
    def _needs_ops_tools(cls, text: str, media_text: Optional[str] = None) -> bool:
        """Return True when the message or attachment text plausibly asks for an action that needs the ops toolset."""
        for candidate in (text, media_text):
            if candidate and (
                cls._OPS_INTENT_RE.search(candidate) or cls._TOOL_INTENT_RE.search(candidate)
            ):
                return True
        return False

    def _resolve_turn_tool_profile(
        self,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        primary: IncomingMessage,
        messages: List[IncomingMessage],
        ops_tools_mode: Optional[str] = None,
    ) -> Tuple[ToolProfile, Optional[str], Optional[str]]:
        """Resolve active tool_profile ('ops' vs 'chat') according to ops_tools_mode and access rules."""
        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
        # Fail-closed if no administrators are configured to control/disable ops mode.
        if not admin_ids:
            return ("chat", None, None)

        mode = (ops_tools_mode if ops_tools_mode is not None else self._ops_tools_mode).strip().lower()
        if mode == "off":
            return ("chat", None, None)

        raw = getattr(primary, "raw_event", None)
        if (
            getattr(primary, "sender_is_bot", False)
            or getattr(raw, "sender_is_bot", False)
            or getattr(raw, "out", False)
        ):
            return ("chat", None, None)

        for msg in messages:
            m_raw = getattr(msg, "raw_event", None)
            if (
                getattr(msg, "sender_is_bot", False)
                or getattr(m_raw, "sender_is_bot", False)
                or getattr(m_raw, "out", False)
            ):
                return ("chat", None, None)

        agy_cfg = getattr(bot_cfg, "antigravity", None)
        ops_mode = getattr(agy_cfg, "ops_mode", "auto") if agy_cfg is not None else "auto"
        ops_cwd = self._resolve_ops_workspace(agy_cfg)

        if mode == "admin":
            if self._is_admin_private_dm(
                chat_id=chat_id,
                bot_cfg=bot_cfg,
                primary=primary,
                messages=messages,
            ):
                return ("ops", ops_mode, ops_cwd)
            return ("chat", None, None)

        if mode == "on":
            if int(chat_id) <= 0:
                allowed_chats = {
                    int(cid) for cid in (getattr(bot_cfg.telegram, "allowed_chat_ids", []) or [])
                }
                if int(chat_id) not in allowed_chats:
                    return ("chat", None, None)
                return ("ops", ops_mode, ops_cwd)

            allowed_dms = set(self._effective_dm_user_ids(bot_cfg))
            if int(chat_id) not in allowed_dms or int(primary.sender_id) not in allowed_dms:
                return ("chat", None, None)
            return ("ops", ops_mode, ops_cwd)

        return ("chat", None, None)

    def _resolve_reply_target_user_ids(
        self,
        primary: IncomingMessage,
        messages: Sequence[IncomingMessage],
        bot_cfg: BotConfig,
    ) -> Optional[List[int]]:
        """Resolve unambiguous reply target user IDs from IncomingMessage metadata or friends directory."""
        bot_uid = getattr(self.userbot, "bot_id", None)
        bot_uid_int = int(bot_uid) if bot_uid else None
        resolved: List[int] = []
        seen: Set[int] = set()

        for msg in (primary, *messages):
            raw = getattr(msg, "raw_event", None)
            for attr in ("reply_to_user_id", "reply_to_sender_id"):
                candidate = getattr(msg, attr, None)
                if candidate is None and raw is not None:
                    candidate = getattr(raw, attr, None)
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    if candidate > 0 and candidate != bot_uid_int and candidate not in seen:
                        seen.add(candidate)
                        resolved.append(candidate)

        if not resolved:
            raw_primary = getattr(primary, "raw_event", None)
            reply_author = getattr(primary, "reply_to_author", None) or (
                getattr(raw_primary, "reply_to_author", None) if raw_primary is not None else None
            )
            if isinstance(reply_author, str) and reply_author.strip():
                _, matched_uid = canonicalize_subject(
                    reply_author,
                    friends_map=self._get_friends_map(),
                    bot_id=bot_uid_int,
                    bot_names=bot_cfg.behavior.bot_names,
                )
                if matched_uid is not None and matched_uid != bot_uid_int:
                    resolved.append(int(matched_uid))

        return resolved or None

    async def _complete_reply_turn(
        self,
        *,
        chat_id: int,
        bot_cfg: BotConfig,
        state: Dict[str, float],
        now: float,
        decision: ReplyAction,
        primary: IncomingMessage,
        messages: List[IncomingMessage],
        combined_text: str,
        media_target: Optional[IncomingMessage],
        is_reply_media: bool,
        media_summary: Optional[str],
        media_text: Optional[str],
        raw_media_text: Optional[str],
        attachments: List[MediaAttachment],
        is_voice_target: bool,
    ) -> Optional[str]:
        """Run FTS5 context lookup, LLM generation, voice/text/doc delivery, and memory persistence."""
        # Avoid blind LLM calls on audio/non-image media-only messages when CLI lacks transcription/vision
        if (
            media_target is not None
            and not is_reply_media
            and not combined_text
            and not (raw_media_text or "").strip()
            and self._provider_lacks_binary_media(attachments, media_text=raw_media_text)
        ):
            logger.info(
                "Skipping blind media-only LLM call in chat %s (provider cannot inspect binary attachment).",
                chat_id,
            )
            return None

        # Fail fast if provider circuit breaker is open for all candidate models of this role
        is_admin_dm_early = self._is_admin_private_dm(
            chat_id=chat_id,
            bot_cfg=bot_cfg,
            primary=primary,
            messages=messages,
        )
        early_role = "dm_admin" if is_admin_dm_early else "group"
        if self._is_provider_rate_limited(role=early_role):
            logger.info("Skipping LLM reply for chat %s while provider is rate-limited.", chat_id)
            return None

        friend = self.config_mgr.get_friend_profile(primary.sender_id)

        # 2. Selective chat-scoped hybrid FTS5 + semantic vector memory fact lookup
        facts: List[FactRecord] = []
        meaningful_terms = extract_fact_terms(combined_text)
        embedder = getattr(self, "memory_embedder", None)
        wants_semantic = bool(
            self._is_embedder_active() and should_run_semantic_lookup(combined_text)
        )
        if (len(combined_text) >= 12 and len(meaningful_terms) >= 2) or wants_semantic:
            excluded_fact_ids = self._get_recent_fact_ids(chat_id, now=now)
            query_vec = await embedder.embed_query(combined_text) if wants_semantic and embedder is not None else None
            target_uids = self._resolve_reply_target_user_ids(primary, messages, bot_cfg)
            emb_cfg = bot_cfg.memory.embeddings
            try:
                facts = await self.memory_store.search_facts(
                    combined_text,
                    limit=3,
                    exclude_fact_ids=excluded_fact_ids or None,
                    chat_id=chat_id,
                    author_user_id=primary.sender_id,
                    target_user_ids=target_uids,
                    query_vector=query_vec,
                    embedding_model=emb_cfg.gemini_model,
                    min_vector_similarity=emb_cfg.min_similarity,
                )
            except TypeError:
                if excluded_fact_ids:
                    try:
                        facts = await self.memory_store.search_facts(
                            combined_text,
                            limit=3,
                            exclude_fact_ids=excluded_fact_ids,
                        )
                    except TypeError:
                        facts = await self.memory_store.search_facts(combined_text, limit=3)
                else:
                    facts = await self.memory_store.search_facts(combined_text, limit=3)

        history_limit = min(int(bot_cfg.memory.short_term_limit), 10)
        recent_history = await self.memory_store.get_recent_history(
            chat_id,
            limit=history_limit,
        )
        facts = self._filter_overused_facts(
            facts=facts,
            recent_history=recent_history,
            bot_id=getattr(self.userbot, "bot_id", None),
            bot_names=bot_cfg.behavior.bot_names,
        )

        exclude_ids = {m.id for m in recent_history if m.id is not None}
        exclude_users = (
            {int(self.userbot.bot_id)} if self.userbot and self.userbot.bot_id else None
        )
        archive_hits: List[ChatMessage] = []
        if hasattr(self.memory_store, "search_messages") and self._should_search_archive(
            combined_text
        ):
            archive_hits = await self.memory_store.search_messages(
                combined_text,
                limit=3,
                min_length=18,
                exclude_db_ids=exclude_ids,
                exclude_user_ids=exclude_users,
            )

        wants_voice_reply = bool(
            is_voice_target
            and bot_cfg.voice.tts.enabled
            and bot_cfg.voice.tts.reply_voice_to_voice
        )

        is_admin_dm = is_admin_dm_early
        tool_profile, ops_mode, ops_cwd = self._resolve_turn_tool_profile(
            chat_id=chat_id,
            bot_cfg=bot_cfg,
            primary=primary,
            messages=messages,
        )
        # Ops is *permitted* here; for non-admin-DM turns only escalate to the expensive tool
        # profile when the message or attachment/STT text looks like an action/tool request.
        # Admin private DMs retain ops whenever ops_tools_mode permits it (not "off").
        attachment_text = (
            "\n".join(
                dict.fromkeys(
                    s.strip()
                    for s in (media_text, raw_media_text)
                    if isinstance(s, str) and s.strip()
                )
            )
            or None
        )
        if (
            tool_profile != "chat"
            and not is_admin_dm
            and not self._needs_ops_tools(combined_text, media_text=attachment_text)
        ):
            tool_profile, ops_mode, ops_cwd = "chat", None, None

        prompt = self.build_context_prompt(
            incoming=primary,
            recent_history=recent_history,
            facts=facts,
            friend_profile=friend,
            media_summary=media_summary,
            media_text=media_text,
            archive_hits=archive_hits,
            incoming_messages=messages,
            media_from_reply=is_reply_media,
            voice_reply_mode=wants_voice_reply,
            ops_tools_enabled=(tool_profile != "chat"),
        )

        # Immediately signal typing status while LLM generates the response
        pre_typed = False
        if bot_cfg.behavior.typing_simulation and hasattr(self.userbot, "send_typing_action"):
            typing_action = "record-voice" if wants_voice_reply else "typing"
            try:
                await self.userbot.send_typing_action(chat_id, action=typing_action)
            except TypeError:
                await self.userbot.send_typing_action(chat_id)
            pre_typed = True

        usage_role = "dm_admin" if is_admin_dm else "group"
        # Keep the system prompt a byte-stable prefix (Prompt Caching). Dynamic quota state is
        # appended to the END of the per-turn prompt, and only when actionable.
        effective_system_prompt = self.config_mgr.system_prompt or ""
        quota_hint = self.quota_manager.format_context_hint()
        if quota_hint:
            prompt = f"{prompt}\n\n{quota_hint}"

        resp: Optional[ProviderResponse] = None
        chosen_model: str = bot_cfg.models.group.model
        max_fallback_tries = max(2, 1 + len(bot_cfg.models.fallback_models))
        tried_models: set[str] = set()

        for _attempt in range(max_fallback_tries):
            chosen_model, chosen_effort = self.quota_manager.select_model(usage_role)
            if chosen_model in tried_models and len(tried_models) >= 1:
                break
            tried_models.add(chosen_model)
            try:
                if tool_profile != "chat":
                    resp = await self.provider.generate_response(
                        prompt=prompt,
                        system_instructions=effective_system_prompt,
                        media_attachments=attachments,
                        conversation_id=None,
                        model=chosen_model,
                        effort=chosen_effort,
                        tool_profile=tool_profile,
                        mode=ops_mode,
                        cwd=ops_cwd,
                    )
                else:
                    try:
                        resp = await self.provider.generate_response(
                            prompt=prompt,
                            system_instructions=effective_system_prompt,
                            media_attachments=attachments,
                            conversation_id=None,
                            model=chosen_model,
                            effort=chosen_effort,
                            tool_profile="chat",
                        )
                    except TypeError:
                        resp = await self.provider.generate_response(
                            prompt=prompt,
                            system_instructions=effective_system_prompt,
                            media_attachments=attachments,
                            conversation_id=None,
                        )
                break
            except Exception as exc:
                verdict = getattr(exc, "verdict", None) or classify_error(exc)
                if verdict.category == ErrorCategory.NATIVE_TOOL_ABORT or isinstance(
                    exc, AntigravityNativeToolAbort
                ):
                    if not getattr(exc, "_native_tool_abort_logged", False):
                        logger.warning(
                            "Native tool execution aborted in chat %s (model=%s, role=%s): %s",
                            chat_id,
                            chosen_model,
                            usage_role,
                            self.sanitizer.sanitize(str(exc)),
                        )
                        try:
                            setattr(exc, "_native_tool_abort_logged", True)
                        except Exception:
                            pass
                    if is_admin_dm or decision == ReplyAction.MUST_REPLY:
                        await self._send_native_tool_abort_notice(
                            chat_id=chat_id,
                            tool_profile=tool_profile,
                            reply_to_msg_id=primary.msg_id if primary.msg_id else None,
                        )
                    return None
                if verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.MODEL_NOT_FOUND):
                    retry_after = extract_retry_after(exc)
                    self.quota_manager.record_rate_limit(
                        chosen_model,
                        retry_after=retry_after,
                        bucket_wide=False,
                        reason=str(exc),
                    )
                    next_model, _ = self.quota_manager.select_model(usage_role)
                    if next_model not in tried_models:
                        logger.warning(
                            "Model %s failed (%s); falling back to %s",
                            chosen_model,
                            verdict.category.value,
                            next_model,
                        )
                        continue
                raise

        if resp is None:
            raise RuntimeError("No provider response produced after fallback attempts.")

        if resp.usage is not None:
            await self.quota_manager.record_usage(
                resp.usage,
                model=resp.model or chosen_model,
                source=usage_role,
            )

        if self.quota_manager.consume_economy_notification_flag():
            for admin_id in getattr(bot_cfg.telegram, "admin_user_ids", []) or []:
                try:
                    await self.send_sanitized_message(
                        int(admin_id),
                        "⚠️ Достигнут порог бюджета токенов — включён режим экономии квоты "
                        "(проактивные дайджесты и спонтанные ответы в группе временно приглушены).",
                    )
                except Exception as exc:
                    logger.warning("Failed to send economy-mode warning to admin %s: %s", admin_id, exc)

        raw_reply_text = self.sanitizer.sanitize((resp.text or "").strip())
        if not raw_reply_text or self.is_skip_response(raw_reply_text):
            return None

        doc_specs, text_without_docs = extract_document_specs(raw_reply_text)
        text_without_imgs, image_specs = extract_image_specs(text_without_docs, max_specs=1)
        text_without_skip = self._SKIP_TAG_RE.sub("", text_without_imgs).strip()
        reaction_emoji, reply_text = self.extract_reaction_tag(text_without_skip)

        img_cfg = bot_cfg.image_generation
        should_generate_image = bool(
            image_specs and img_cfg.enabled and img_cfg.allow_natural_language_trigger
        )

        reaction_sent = False
        if reaction_emoji and bot_cfg.behavior.reactions_enabled and primary.msg_id:
            await self.userbot.send_reaction(chat_id, primary.msg_id, reaction_emoji)
            reaction_sent = True

        if not reply_text and not doc_specs and not should_generate_image:
            if reaction_sent:
                react_ts = time.time()
                state["last_bot_msg_time"] = react_ts
                if decision != ReplyAction.SPONTANEOUS_REPLY:
                    self.heuristics.record_reply(chat_id, timestamp=react_ts)
                self._record_used_facts(chat_id, facts, now=react_ts)
            return None

        delivered_docs: List[GeneratedDocument] = []
        last_doc_res: Any = None
        doc_send_failed = False
        reply_sent_as_caption = False
        if doc_specs:
            try:
                async with self.document_builder.managed_generated_documents(
                    doc_specs
                ) as generated_docs:
                    for idx, gen_doc in enumerate(generated_docs):
                        stripped_reply_caption = (
                            strip_tts_emotion_tags(reply_text) if idx == 0 and reply_text else ""
                        )
                        use_reply_caption = (
                            bool(stripped_reply_caption) and len(stripped_reply_caption) <= 1024
                        )
                        doc_caption = (
                            stripped_reply_caption if use_reply_caption else gen_doc.caption
                        )
                        try:
                            doc_res = await self.userbot.send_document(
                                chat_id=chat_id,
                                file_path=gen_doc.path,
                                caption=doc_caption,
                                reply_to_msg_id=primary.msg_id if primary.msg_id else None,
                                simulate_typing=(
                                    bot_cfg.behavior.typing_simulation
                                    and not pre_typed
                                    and idx == 0
                                ),
                                filename=gen_doc.filename,
                            )
                            if doc_res:
                                delivered_docs.append(gen_doc)
                                last_doc_res = doc_res
                                if use_reply_caption:
                                    reply_sent_as_caption = True
                            else:
                                doc_send_failed = True
                                logger.warning(
                                    "send_document returned None/falsy for '%s' in chat %s; not recording delivery",
                                    gen_doc.filename,
                                    chat_id,
                                )
                        except Exception as exc:
                            doc_send_failed = True
                            logger.exception(
                                "Failed to send generated document '%s' to chat %s: %s",
                                gen_doc.filename,
                                chat_id,
                                exc,
                            )
            except Exception as exc:
                doc_send_failed = True
                logger.exception(
                    "Failed to build generated documents for chat %s: %s",
                    chat_id,
                    exc,
                )

        sent_res: Any = None
        sent_as_voice = False
        voice_lang: Optional[str] = None
        if reply_text and not reply_sent_as_caption:
            if (
                wants_voice_reply
                and not doc_specs
                and not delivered_docs
                and hasattr(self.userbot, "send_voice")
            ):
                try:
                    async with self.tts_synthesizer.managed_speech(reply_text) as tts_res:
                        sent_res = await self.userbot.send_voice(
                            chat_id=chat_id,
                            voice_path=tts_res.audio_path,
                            reply_to_msg_id=primary.msg_id if primary.msg_id else None,
                            duration=tts_res.duration,
                            simulate_typing=(
                                bot_cfg.behavior.typing_simulation and not pre_typed
                            ),
                        )
                        if sent_res:
                            sent_as_voice = True
                            voice_lang = tts_res.language
                        else:
                            logger.warning(
                                "send_voice returned None/falsy for chat %s; falling back to text",
                                chat_id,
                            )
                except TTSSkippedError as exc:
                    logger.info(
                        "Skipping voice reply in chat %s (%s); falling back to text.",
                        chat_id,
                        exc.reason,
                    )
                except Exception as exc:
                    logger.warning(
                        "Voice reply failed in chat %s; falling back to text: %s",
                        chat_id,
                        self.sanitizer.sanitize(exc),
                    )

            if not sent_as_voice:
                reply_text = strip_tts_emotion_tags(reply_text)
                if reply_text:
                    sent_res = await self.userbot.send_reply(
                        chat_id=chat_id,
                        text=reply_text,
                        reply_to_msg_id=primary.msg_id if primary.msg_id else None,
                        simulate_typing=(
                            bot_cfg.behavior.typing_simulation
                            and not pre_typed
                            and not delivered_docs
                        ),
                    )

        clean_history_reply = strip_tts_emotion_tags(reply_text) if reply_text else ""
        history_text = ""
        if clean_history_reply or delivered_docs or sent_as_voice:
            sent_ts = time.time()
            state["last_bot_msg_time"] = sent_ts
            if decision != ReplyAction.SPONTANEOUS_REPLY:
                self.heuristics.record_reply(chat_id, timestamp=sent_ts)
            self._record_used_facts(chat_id, facts, now=sent_ts)

            sent_msg_id = self._extract_sent_message_id(
                sent_res if sent_res is not None else last_doc_res
            )

            doc_summary = ""
            if delivered_docs:
                doc_names = ", ".join(doc.filename for doc in delivered_docs)
                doc_summary = f"[Документы: {doc_names}]"

            history_text = (
                f"{clean_history_reply} {doc_summary}".strip()
                if doc_summary
                else clean_history_reply
            )

            saved_media_type: Optional[str] = None
            saved_media_summary: Optional[str] = None
            if delivered_docs:
                saved_media_type = "document"
            elif sent_as_voice:
                saved_media_type = "voice"
                saved_media_summary = (
                    f"Голосовой ответ ({voice_lang})" if voice_lang else "Голосовой ответ"
                )

            # Save bot's own response into short-term history
            bot_name = bot_cfg.behavior.bot_names[0] if bot_cfg.behavior.bot_names else "Потужнич"
            await self.memory_store.save_message(
                ChatMessage(
                    telegram_msg_id=sent_msg_id,
                    chat_id=chat_id,
                    user_id=self.userbot.bot_id or 1,
                    timestamp=int(sent_ts),
                    author_name=bot_name,
                    text=history_text,
                    reply_to_msg_id=primary.msg_id,
                    media_type=saved_media_type,
                    media_summary=saved_media_summary,
                )
            )
        elif not should_generate_image:
            if doc_send_failed:
                logger.warning(
                    "Document generation/send failed with no text reply for chat %s",
                    chat_id,
                )
            if reaction_sent:
                react_ts = time.time()
                state["last_bot_msg_time"] = react_ts
                if decision != ReplyAction.SPONTANEOUS_REPLY:
                    self.heuristics.record_reply(chat_id, timestamp=react_ts)
                self._record_used_facts(chat_id, facts, now=react_ts)
            return None

        if should_generate_image:
            img_spec = image_specs[0]
            ref_images = [
                att
                for att in attachments
                if (att.mime_type or "").lower().startswith("image/")
                and ((att.file_path is not None and att.file_path.exists()) or bool(att.data))
            ]
            img_result_text = await self._execute_image_generation(
                chat_id=chat_id,
                user_id=primary.sender_id,
                bot_cfg=bot_cfg,
                state=state,
                prompt=img_spec.prompt,
                caption=img_spec.caption or None,
                aspect_ratio=img_spec.aspect_ratio,
                reference_images=ref_images or None,
                reply_to_msg_id=primary.msg_id if primary.msg_id else None,
                decision=decision,
                send_error_reply=True,
            )
            if not history_text and img_result_text:
                history_text = img_result_text
                self._record_used_facts(chat_id, facts, now=time.time())

        if not history_text:
            return None

        return history_text

    def _get_proactive_target_chat_ids(self) -> List[int]:
        """Return configured allowed chat IDs for proactive cron delivery fallback."""
        bot_cfg = self.config_mgr.bot
        if not bot_cfg or not bot_cfg.telegram.allowed_chat_ids:
            return []
        return [int(cid) for cid in bot_cfg.telegram.allowed_chat_ids if int(cid) < 0]

    async def _send_proactive(self, chat_id: int, text: str) -> None:
        """Send a proactive message/digest, update chat_states, and persist it to MemoryStore."""
        bot_cfg = self.config_mgr.bot
        safe_text = self.sanitizer.sanitize(text)
        sent_res = await self.userbot.send_reply(
            chat_id=chat_id,
            text=safe_text,
            reply_to_msg_id=None,
            simulate_typing=bool(bot_cfg and bot_cfg.behavior.typing_simulation),
        )
        now = time.time()
        state = self.chat_states.setdefault(
            chat_id,
            {"last_chat_msg_time": now, "last_bot_msg_time": 0.0},
        )
        state["last_bot_msg_time"] = now

        bot_name = (
            bot_cfg.behavior.bot_names[0]
            if bot_cfg and bot_cfg.behavior.bot_names
            else "Потужнич"
        )
        sent_msg_id = self._extract_sent_message_id(sent_res)
        await self.memory_store.save_message(
            ChatMessage(
                telegram_msg_id=sent_msg_id,
                chat_id=chat_id,
                user_id=self.userbot.bot_id or 1,
                timestamp=int(now),
                author_name=bot_name,
                text=safe_text,
                reply_to_msg_id=None,
            )
        )

    async def _seed_chat_states_from_memory(self) -> None:
        """Seed chat_states timestamps from MemoryStore so restarts don't trigger stale banter."""
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            return
        now = time.time()
        bot_id = getattr(self.userbot, "bot_id", None)
        bot_names_lower = {
            n.strip().lower() for n in bot_cfg.behavior.bot_names if n and n.strip()
        }

        for raw_cid in bot_cfg.telegram.allowed_chat_ids:
            cid = int(raw_cid)
            try:
                history = await self.memory_store.get_recent_history(cid, limit=50)
            except Exception as exc:
                logger.warning("Failed seeding chat_states for chat %s: %s", cid, exc)
                continue

            if not history:
                self.chat_states[cid] = {
                    "last_chat_msg_time": now,
                    "last_bot_msg_time": now,
                }
                continue

            human_timestamps: List[float] = []
            bot_timestamps: List[float] = []
            for msg in history:
                is_bot = (bot_id is not None and int(bot_id) > 0 and msg.user_id == int(bot_id)) or (
                    bool(msg.author_name) and msg.author_name.strip().lower() in bot_names_lower
                )
                ts = float(msg.timestamp)
                if is_bot:
                    bot_timestamps.append(ts)
                else:
                    human_timestamps.append(ts)

            last_bot_ts = max(bot_timestamps) if bot_timestamps else now
            last_chat_ts = max(human_timestamps) if human_timestamps else last_bot_ts
            self.chat_states[cid] = {
                "last_chat_msg_time": last_chat_ts,
                "last_bot_msg_time": last_bot_ts,
            }

    async def start(self) -> None:
        """Start all subsystems (SQLite DB, config watcher, Telegram MTProto client, proactive loop)."""
        await self.memory_store.init_db()
        await self._ensure_ops_tools_mode_loaded()
        await self.quota_manager.sync_from_db()
        await self.config_mgr.start_watcher()
        await self.userbot.start()
        self._sync_bot_identity_to_memory_engines()
        self.evolution_manager.userbot = self.userbot
        await self.evolution_manager.reconcile_pending_deploy()
        await self._seed_chat_states_from_memory()

        self.proactive_engine.start_loop(
            get_chat_states=lambda: self.chat_states,
            send_message_fn=self._send_proactive,
            poll_interval=60.0,
            get_target_chat_ids=self._get_proactive_target_chat_ids,
        )
        self._backup_task = asyncio.create_task(
            self._memory_backup_loop(),
            name="memory-backup-loop",
        )
        self._reflection_task = asyncio.create_task(
            self._run_reflection_loop(),
            name="memory-reflection",
        )

    async def _run_memory_backup_once(self) -> Optional[Any]:
        """Run one hot SQLite backup of the configured memory database."""
        from bot.memory.backup import async_backup_sqlite_database

        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            return None
        db_path = Path(self.memory_store.db_path)
        if not db_path.exists():
            return None
        bcfg = bot_cfg.memory.backup
        return await async_backup_sqlite_database(
            db_path=db_path,
            backup_dir=bcfg.backup_dir,
            compress=bcfg.compress,
            retain=bcfg.retain_count,
        )

    async def _memory_backup_loop(self) -> None:
        """Periodically snapshot memory.sqlite according to memory.backup config."""
        from bot.memory.backup import list_backups

        while True:
            try:
                bot_cfg = self.config_mgr.bot
                bcfg = bot_cfg.memory.backup if bot_cfg is not None else None
                if bcfg is not None and bcfg.enabled:
                    interval_sec = max(60.0, float(bcfg.interval_hours) * 3600.0)
                    existing = list_backups(bcfg.backup_dir)
                    should_run = not existing
                    if existing:
                        age = time.time() - existing[0].stat().st_mtime
                        should_run = age >= interval_sec
                    if should_run:
                        res = await self._run_memory_backup_once()
                        if res is not None:
                            logger.info(
                                "Scheduled memory backup saved: %s (%d bytes, pruned=%d)",
                                res.backup_path,
                                res.size_bytes,
                                res.pruned_count,
                            )
                await asyncio.sleep(3600.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Scheduled memory backup failed: %s", exc)
                await asyncio.sleep(3600.0)

    async def _run_reflection_loop(self) -> None:
        """Periodically backfill missing fact embeddings, archive expired facts, and reflect dense subjects."""
        try:
            await asyncio.sleep(45.0)
        except asyncio.CancelledError:
            return

        while True:
            try:
                bot_cfg = self.config_mgr.bot
                self._sync_bot_identity_to_memory_engines()

                if (
                    bot_cfg is not None
                    and bot_cfg.memory.embeddings.enabled
                    and self.memory_embedder is not None
                    and self._is_embedder_active()
                ):
                    indexed = await self.memory_embedder.backfill_missing(self.memory_store)
                    if indexed > 0:
                        logger.info("Backfilled %d missing memory fact embeddings.", indexed)

                refl_cfg = bot_cfg.memory.reflection if bot_cfg is not None else None
                if refl_cfg is not None and refl_cfg.enabled and self.reflection_engine is not None:
                    stats = await self.reflection_engine.run_cycle(
                        is_rate_limited_fn=lambda: self._is_provider_rate_limited(
                            role="compaction"
                        ),
                    )
                    if any(stats.values()):
                        logger.info("Memory reflection cycle completed: %s", stats)

                sleep_sec = (
                    max(300.0, float(refl_cfg.interval_hours) * 3600.0)
                    if refl_cfg is not None
                    else 86400.0
                )
                await asyncio.sleep(sleep_sec)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "Memory reflection loop failed: %s",
                    self.sanitizer.sanitize(exc),
                )
                try:
                    await asyncio.sleep(3600.0)
                except asyncio.CancelledError:
                    break

    async def stop(self) -> None:
        """Gracefully stop all background tasks and close connections."""
        reflection_task = getattr(self, "_reflection_task", None)
        if reflection_task is not None and not reflection_task.done():
            reflection_task.cancel()
            try:
                await reflection_task
            except asyncio.CancelledError:
                pass
        self._reflection_task = None

        backup_task = getattr(self, "_backup_task", None)
        if backup_task is not None and not backup_task.done():
            backup_task.cancel()
            try:
                await backup_task
            except asyncio.CancelledError:
                pass
        self._backup_task = None

        cancelled_debounce: List[asyncio.Task[Any]] = []
        for task in list(self._debounce_tasks.values()):
            if not task.done():
                task.cancel()
                cancelled_debounce.append(task)
        self._debounce_tasks.clear()
        self._debounce_buffers.clear()
        self._debounce_started_at.clear()
        if cancelled_debounce:
            await asyncio.gather(*cancelled_debounce, return_exceptions=True)

        proactive_task = getattr(self.proactive_engine, "_loop_task", None)
        self.proactive_engine.stop_loop()
        if isinstance(proactive_task, asyncio.Task) and not proactive_task.done():
            await asyncio.gather(proactive_task, return_exceptions=True)

        watcher_task = getattr(self.config_mgr, "_watcher_task", None)
        self.config_mgr.stop_watcher()
        if isinstance(watcher_task, asyncio.Task) and not watcher_task.done():
            await asyncio.gather(watcher_task, return_exceptions=True)

        await self.userbot.stop()
        if self._owns_stt_transcriber and hasattr(self.stt_transcriber, "close"):
            self.stt_transcriber.close()
        if getattr(self, "_owns_memory_embedder", False) and hasattr(
            getattr(self, "memory_embedder", None), "_close_owned_client"
        ):
            self.memory_embedder._close_owned_client()
        if hasattr(self.provider, "close"):
            self.provider.close()
        await self.memory_store.close()


class _DryRunProvider:
    """Deterministic provider used during offline/smoke dry-run verification."""

    async def generate_response(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        media_attachments: Optional[List[MediaAttachment]] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        tool_profile: ToolProfile = "chat",
        mode: Optional[str] = None,
        cwd: Optional[Path | str] = None,
    ) -> ProviderResponse:
        media_note = f" (вложений: {len(media_attachments)})" if media_attachments else ""
        return ProviderResponse(
            text=f"Привет! Вижу сообщение и контекст{media_note} — всё работает отлично.",
            model="dry-run-smoke-model",
        )

    def close(self) -> None:
        pass


class _DryRunTelegramClient:
    """Mock Telethon client for dry-run smoke testing without live network calls."""

    def __init__(self) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.reactions: List[Dict[str, Any]] = []

    def is_connected(self) -> bool:
        return True

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def is_user_authorized(self) -> bool:
        return True

    async def get_me(self) -> Any:
        class _Me:
            id = 777000
            username = "potuzhnych_userbot"
        return _Me()

    def add_event_handler(self, callback: Any, event_filter: Any = None) -> None:
        pass

    async def send_message(self, chat_id: int, text: str, reply_to: Optional[int] = None) -> Any:
        self.sent_messages.append({"chat_id": chat_id, "text": text, "reply_to": reply_to})
        return {"id": len(self.sent_messages), "text": text}


async def run_dry_run_smoke(config_path: str | Path = "config/bot.yaml") -> Dict[str, Any]:
    """Execute a full-pipeline dry-run smoke test (incoming msg + media + FTS5 + friend profile + proactive)."""
    cfg_file = Path(config_path).resolve()
    config_dir = cfg_file.parent if cfg_file.is_file() else Path("config").resolve()
    config_mgr = (
        ConfigManager(config_path=cfg_file)
        if cfg_file.is_file()
        else ConfigManager(config_dir=config_dir)
    )

    with tempfile.TemporaryDirectory(prefix="tg_bot_smoke_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        memory_store = MemoryStore(db_path=str(tmp_path / "smoke_memory.sqlite"))
        await memory_store.init_db()

        media_proc = MediaProcessor(
            max_doc_size_mb=config_mgr.bot.media.max_doc_size_mb if config_mgr.bot else 10,
            ffmpeg_path=config_mgr.bot.media.ffmpeg_path if config_mgr.bot else None,
            temp_dir_base=str(tmp_path / "media_tmp"),
        )
        dry_provider = _DryRunProvider()
        dry_tg_client = _DryRunTelegramClient()

        allowed_chats = (
            config_mgr.bot.telegram.allowed_chat_ids
            if config_mgr.bot and config_mgr.bot.telegram.allowed_chat_ids
            else None
        )
        effective_dms = (
            GroupBotOrchestrator._effective_dm_user_ids(config_mgr.bot)
            if config_mgr.bot
            else []
        )
        is_dm_testing = False
        if not allowed_chats and effective_dms:
            allowed_chats = effective_dms
            is_dm_testing = True
        if not allowed_chats:
            allowed_chats = [-1001234567890]
        chat_id = int(allowed_chats[0])

        allowed_group_chats = [chat_id] if not is_dm_testing else []
        allowed_dm_chats = [chat_id] if is_dm_testing else effective_dms

        userbot = TelegramUserbot(
            session_path=str(tmp_path / "session" / "smoke.session"),
            api_id=config_mgr.bot.telegram.api_id if config_mgr.bot else 12345,
            api_hash=config_mgr.bot.telegram.api_hash if config_mgr.bot else "0123456789abcdef",
            allowed_chat_ids=allowed_group_chats,
            allowed_dm_user_ids=allowed_dm_chats,
            ignored_sender_usernames=(
                config_mgr.bot.telegram.ignored_sender_usernames if config_mgr.bot else []
            ),
            bot_names=config_mgr.bot.behavior.bot_names if config_mgr.bot else ["Потужнич"],
            typing_min_seconds=0.01,
            typing_max_seconds=0.02,
            client=dry_tg_client,
        )
        userbot.bot_id = 777000
        userbot.bot_username = "potuzhnych_userbot"

        orchestrator = GroupBotOrchestrator(
            config_mgr=config_mgr,
            memory_store=memory_store,
            media_processor=media_proc,
            provider=dry_provider,
            userbot=userbot,
        )

        # 1. Seed a Russian long-term fact in FTS5
        await memory_store.add_fact(
            subject="Алексей",
            fact="Любит собирать кастомные механические клавиатуры и писать сервисы на Go",
            category="friend_bio",
            source_chat_id=chat_id,
        )

        # 2. Create a synthetic photo attachment to verify multimodal pipeline + temp cleanup
        sample_img_path = tmp_path / "sample_photo.jpg"
        Image.new("RGB", (64, 64), color=(40, 120, 200)).save(sample_img_path, format="JPEG")

        friend_ids = list(config_mgr.friends.friends.keys())
        sender_id = int(friend_ids[0]) if friend_ids else 123456789

        class _SyntheticEvent:
            id = 101
            out = False
            is_reply = False
            reply_to_msg_id = None
            raw_text = "Потужнич, посмотри на мою новую сборку клавиатуры!"
            text = raw_text
            photo = True
            media = True
            file_path = str(sample_img_path)

        synth_event = _SyntheticEvent()
        synth_event.chat_id = chat_id
        synth_event.sender_id = chat_id if is_dm_testing else sender_id
        synth_event.is_private = is_dm_testing

        incoming = IncomingMessage(
            msg_id=101,
            chat_id=chat_id,
            sender_id=synth_event.sender_id,
            sender_name="Родион" if is_dm_testing else "Алексей",
            text=synth_event.raw_text,
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=True,
            raw_event=synth_event,
            media=True,
            is_private=is_dm_testing,
        )

        reply_text = await orchestrator.handle_incoming_message(incoming)

        # 3. Verify proactive humor/fake-news generation with friend profile injection
        proactive_text = await orchestrator.proactive_engine.humor_gen.generate_banter(
            recent_messages=await memory_store.get_recent_history(chat_id, limit=5),
            friend=config_mgr.get_friend_profile(sender_id),
            facts=await memory_store.search_facts("клавиатура", limit=3),
            trigger_type="banter",
            humor_style=(
                config_mgr.bot.proactive.humor_style if config_mgr.bot else "friendly_sarcasm"
            ),
            forbidden_topics=(
                config_mgr.bot.proactive.forbidden_topics if config_mgr.bot else []
            ),
        )

        # Verify temp media directory has zero leftover bot_media_* folders
        leftover_dirs = list((tmp_path / "media_tmp").glob("bot_media_*"))

        await orchestrator.stop()

        return {
            "status": "ok",
            "chat_id": chat_id,
            "friend_id": sender_id,
            "reply_text": reply_text,
            "proactive_text": proactive_text,
            "sent_messages_count": len(dry_tg_client.sent_messages),
            "leftover_temp_dirs": len(leftover_dirs),
        }


async def run_daemon(config_path: str | Path = "config/bot.yaml") -> None:
    """Run the production Telegram userbot daemon until SIGINT/SIGTERM."""
    cfg_file = Path(config_path).resolve()
    config_dir = cfg_file.parent if cfg_file.is_file() else Path("config").resolve()
    config_mgr = (
        ConfigManager(config_path=cfg_file)
        if cfg_file.is_file()
        else ConfigManager(config_dir=config_dir)
    )
    orchestrator = GroupBotOrchestrator(config_mgr=config_mgr, debounce_seconds=2.5)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    await orchestrator.start()
    logger.info("Telegram group userbot is running. Press Ctrl+C to stop.")
    try:
        await stop_event.wait()
    finally:
        logger.info("Shutting down Telegram group userbot...")
        await orchestrator.stop()
