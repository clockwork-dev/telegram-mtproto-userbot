"""Core Orchestrator and Daemon Runner for the Autonomous Telegram Userbot Agent."""

from __future__ import annotations

import asyncio
import logging
import re
import signal
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from bot.antigravity import AntigravityCLIClient, AntigravityConfig, AntigravityProvider
from bot.config.manager import ConfigManager
from bot.config.models import BotConfig, FriendProfile
from bot.core.models import MediaAttachment, ProviderResponse
from bot.documents.builder import DocumentBuilder, GeneratedDocument, extract_document_specs
from bot.media.processor import MediaProcessor, ProcessedMedia, _has_media_flag
from bot.media.stt import STTTranscriber
from bot.media.tts import TTSSkippedError, TTSSynthesizer
from bot.memory.compaction import CompactionEngine
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FactRecord, extract_archive_terms, extract_fact_terms
from bot.proactive.engine import ProactiveEngine
from bot.telegram.bot_api_client import (
    TelegramBotAPIClient,
    normalize_reaction_emoji,
    resolve_bot_token_from_env_or_config,
)
from bot.telegram.client import TelegramUserbot
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

logger = logging.getLogger(__name__)


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
    _TOOL_INTENT_RE = re.compile(
        r"(?:https?://|\b(?:pdf|docx|xlsx|pptx|word|excel|google|search|news|weather|btc)\b"
        r"|пдф|ворд|эксел|ексел|документ|файл|отч[её]т|звіт|таблиц|слайд|презентац"
        r"|найд[иі]|знайд|поищ|пошук|погугл|гугл|новост|новин|погод|курс|цен[аыу]|цін[аиу]"
        r"|сколько\s+стоит|скільки\s+кошту|кто\s+тако|что\s+тако|що\s+таке|хто\s+так"
        r"|ссылк|посилан|источник|джерел|проверь|перевір|расписан|розклад)",
        re.IGNORECASE,
    )
    _ARCHIVE_CUE_RE = re.compile(
        r"(?:помн|пам['’]?ята|раньше|раніш|в\s+прошл|минулог|тогда|тоді|колись|архив|архів|"
        r"говорил|писал|скидывал|скидав|обсуждал|обговорювал)",
        re.IGNORECASE,
    )
    _SKIP_TAG_RE = re.compile(r"\[SKIP\]", re.IGNORECASE)
    _REACTION_TAG_RE = re.compile(r"\[REACT:\s*([^\[\]\s]+)\s*\]", re.IGNORECASE)

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
    ) -> None:
        self.config_mgr = config_mgr
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None:
            raise ValueError("BotConfig must be loaded in ConfigManager before initializing orchestrator.")

        self._owns_stt_transcriber = stt_transcriber is None
        self._owns_tts_synthesizer = tts_synthesizer is None
        self._owns_media_processor = media_processor is None

        self.memory_store = memory_store or MemoryStore(db_path=bot_cfg.memory.db_path)
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
        self.provider = provider or AntigravityProvider(AntigravityConfig())
        self.proactive_engine = proactive_engine or ProactiveEngine(
            config_mgr=self.config_mgr,
            memory_store=self.memory_store,
            antigravity_provider=self.provider,
        )
        self.heuristics = ReplyHeuristics(
            default_max_per_hour=bot_cfg.behavior.spontaneous_replies.max_per_hour,
        )
        if debounce_seconds is not None:
            self.debounce_seconds = max(0.0, float(debounce_seconds))
        else:
            self.debounce_seconds = 2.5 if userbot is None else 0.0

        if userbot is not None:
            self.userbot: Any = userbot
        else:
            resolved_token = resolve_bot_token_from_env_or_config(bot_cfg.telegram.bot_token)
            use_bot_api = bot_cfg.telegram.mode == "bot_api" or (
                bot_cfg.telegram.mode == "auto" and bool(resolved_token)
            )
            if use_bot_api and resolved_token:
                self.userbot = TelegramBotAPIClient(
                    bot_token=resolved_token,
                    allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                    bot_names=bot_cfg.behavior.bot_names,
                    typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                    typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                    typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                    backlog_Callback=self._seed_backlog_message,
                )
            else:
                self.userbot = TelegramUserbot(
                    session_path=bot_cfg.telegram.session_path,
                    api_id=bot_cfg.telegram.api_id,
                    api_hash=bot_cfg.telegram.api_hash,
                    allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                    allowed_dm_user_ids=bot_cfg.telegram.allowed_dm_user_ids,
                    ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                    bot_names=bot_cfg.behavior.bot_names,
                    typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                    typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                    typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                    backlog_callback=self._seed_backlog_message,
                )

        bot_uid = getattr(self.userbot, "bot_id", None)
        excluded_uids = [int(bot_uid)] if bot_uid else None
        self.compaction_engine = CompactionEngine(
            memory_store=self.memory_store,
            summarize_fn=self._summarize_transcript_via_provider,
            threshold=bot_cfg.memory.compaction_threshold,
            batch_size=bot_cfg.memory.compaction_batch_size,
            excluded_user_ids=excluded_uids,
        )
        self.userbot.register_message_handler(self._on_incoming_message)

        self._debounce_buffers: Dict[int, List[IncomingMessage]] = {}
        self._debounce_tasks: Dict[int, asyncio.Task[Any]] = {}
        self._debounce_started_at: Dict[int, float] = {}
        self._chat_locks: Dict[int, asyncio.Lock] = {}
        self._recent_fact_usage: Dict[int, Dict[int, Tuple[int, float]]] = {}
        self._bot_reply_seq: Dict[int, int] = {}

        self.chat_states: Dict[int, Dict[str, float]] = {}
        now = time.time()
        for cid in bot_cfg.telegram.allowed_chat_ids:
            self.chat_states[int(cid)] = {
                "last_chat_msg_time": now,
                "last_bot_msg_time": 0.0,
            }

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

    def _is_provider_rate_limited(self) -> bool:
        """Return True if the LLM provider has an active rate-limit/billing circuit breaker."""
        checker = getattr(self.provider, "is_rate_limited", None)
        return callable(checker) and checker() is True

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
            self.tts_synthesizer.ru_voice = tts_cfg.ru_voice
            self.tts_synthesizer.uk_voice = tts_cfg.uk_voice
            self.tts_synthesizer.rate = tts_cfg.rate
            self.tts_synthesizer.pitch = tts_cfg.pitch
            self.tts_synthesizer.max_chars = int(tts_cfg.max_chars)
            self.tts_synthesizer.timeout_sec = float(tts_cfg.timeout_sec)
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

    async def _summarize_transcript_via_provider(self, transcript: str) -> List[Dict[str, str]]:
        """Extract genuinely durable facts from human conversation transcript using Antigravity."""
        if self._is_provider_rate_limited():
            logger.info("Skipping compaction summarization while provider is rate-limited.")
            return []
        prompt = (
            "Проанализируй фрагмент переписки участников чата и выдели ТОЛЬКО 1-2 по-настоящему "
            "долгосрочных новых факта (важные договорённости, реальные события, устойчивые интересы "
            "или изменения в жизни участников). Игнорируй односложную болтовню, шутки, смех, приветствия "
            "и старые мемы. Если ничего по-настоящему долгосрочного нет — верни пустой массив [].\n"
            "Формат строго JSON-массив: "
            '[{"subject": "...", "fact": "...", "category": "lore|event|agreement|friend_bio"}].\n\n'
            f"Переписка:\n{transcript}"
        )
        try:
            resp = await self.provider.generate_response(prompt=prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return self.compaction_engine._parse_extracted_facts(text)
        except Exception as exc:
            logger.debug("Compaction summarization fallback triggered: %s", exc)
            return []

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

    def _get_recent_fact_ids(self, chat_id: int, now: float) -> List[int]:
        """Return active per-chat cooled-down fact IDs (within last N bot replies or 30m TTL)."""
        bucket = self._recent_fact_usage.get(chat_id)
        if not bucket:
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
        return list(bucket.keys())

    def _record_used_facts(self, chat_id: int, facts: Sequence[FactRecord], now: float) -> None:
        """Advance per-chat reply sequence and record any injected fact IDs for cooldown."""
        seq = self._bot_reply_seq.get(chat_id, 0) + 1
        self._bot_reply_seq[chat_id] = seq
        if not facts:
            return
        bucket = self._recent_fact_usage.setdefault(chat_id, {})
        for fact in facts:
            if getattr(fact, "id", None) is not None:
                bucket[int(fact.id)] = (seq, now)

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
        if voice_reply_mode:
            guidance += (
                "\nГОЛОСОВОЙ ОТВЕТ: собеседник отправил голосовое сообщение, твой ответ будет озвучен голосом (TTS). "
                "Отвечай максимально естественно, живо и разговорно (1-2 коротких предложения), "
                "БЕЗ Markdown-разметки, БЕЗ таблиц, списков, блоков кода и URL-ссылок."
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
        """Extract integer Telegram message_id from Bot API dict or Telethon Message object."""
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
        buf = self._debounce_buffers.setdefault(chat_id, [])
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
        except Exception as exc:
            logger.exception("Error flushing debounced messages for chat %s: %s", chat_id, exc)

    async def _process_buffered_messages(self, chat_id: int) -> Optional[str]:
        """Pop and process all buffered incoming messages for a chat as one coalesced turn."""
        messages = self._debounce_buffers.pop(chat_id, [])
        self._debounce_started_at.pop(chat_id, None)
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
        must_reply_msg: Optional[IncomingMessage] = None
        for msg in messages:
            res = self.heuristics.should_reply(
                event=msg.raw_event,
                bot_id=self.userbot.bot_id,
                bot_names=bot_cfg.behavior.bot_names,
                allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
                allowed_dm_user_ids=bot_cfg.telegram.allowed_dm_user_ids,
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
        if len(messages) == 1:
            eval_event = primary.raw_event
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
                is_private = getattr(primary, "is_private", getattr(raw, "is_private", False))
                sender_is_bot = getattr(primary, "sender_is_bot", getattr(raw, "sender_is_bot", False))
                sender_username = getattr(primary, "sender_username", getattr(raw, "sender_username", None))
                sender = getattr(raw, "sender", None)

            eval_event = _CoalescedEvent()

        decision = self.heuristics.should_reply(
            event=eval_event,
            bot_id=self.userbot.bot_id,
            bot_names=bot_cfg.behavior.bot_names,
            allowed_chat_ids=bot_cfg.telegram.allowed_chat_ids,
            allowed_dm_user_ids=bot_cfg.telegram.allowed_dm_user_ids,
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
        allowed_dms = bot_cfg.telegram.allowed_dm_user_ids
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

            # Whitelist verification:
            if m.chat_id in allowed_group_ints:
                return True
            if m.chat_id in allowed_dm_ints or (
                sender_id is not None and sender_id in allowed_dm_ints
            ):
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

        chat_id = messages[-1].chat_id
        lock = self._chat_locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            now = time.time()
            state = self.chat_states.setdefault(
                chat_id,
                {"last_chat_msg_time": now, "last_bot_msg_time": 0.0},
            )
            state["last_chat_msg_time"] = now

            in_quiet = self.proactive_engine.is_in_quiet_hours()
            time_since_bot = now - state.get("last_bot_msg_time", 0.0)

            working_messages = list(messages)
            media_candidates = [m for m in working_messages if self._has_incoming_media(m)]
            media_target = media_candidates[-1] if media_candidates else None
            voice_candidates = [m for m in working_messages if self._is_voice_message(m)]

            if voice_candidates and bot_cfg.voice.stt.enabled:
                if not self._is_batch_permitted(working_messages, bot_cfg):
                    await self._save_incoming_batch(working_messages, now=now)
                    return None

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
                    return await self._complete_reply_turn(
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

            # Standard non-voice path: evaluate heuristics BEFORE downloading/transcoding any media
            decision, primary = self._evaluate_batch_heuristics(
                messages=working_messages,
                bot_cfg=bot_cfg,
                time_since_bot=time_since_bot,
                in_quiet=in_quiet,
                now=now,
            )

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
                    r_media = getattr(m, "reply_to_media_event", None) or getattr(
                        getattr(m, "raw_event", None), "reply_to_media_event", None
                    )
                    if r_media is not None:
                        reply_media_raw = getattr(r_media, "message", r_media)
                        if isinstance(reply_media_raw, str):
                            reply_media_raw = r_media
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
                return await self._complete_reply_turn(
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

        # Fail fast if provider circuit breaker is open
        if self._is_provider_rate_limited():
            logger.info("Skipping LLM reply for chat %s while provider is rate-limited.", chat_id)
            return None

        friend = self.config_mgr.get_friend_profile(primary.sender_id)

        # 2. Selective FTS5 memory fact lookup (text ONLY, never author_display)
        facts: List[FactRecord] = []
        meaningful_terms = extract_fact_terms(combined_text)
        if len(combined_text) >= 12 and len(meaningful_terms) >= 2:
            excluded_fact_ids = self._get_recent_fact_ids(chat_id, now=now)
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

        resp: ProviderResponse = await self.provider.generate_response(
            prompt=prompt,
            system_instructions=self.config_mgr.system_prompt,
            media_attachments=attachments,
            conversation_id=None,
        )
        raw_reply_text = (resp.text or "").strip()
        if not raw_reply_text or self.is_skip_response(raw_reply_text):
            return None

        doc_specs, text_without_docs = extract_document_specs(raw_reply_text)
        text_without_skip = self._SKIP_TAG_RE.sub("", text_without_docs).strip()
        reaction_emoji, reply_text = self.extract_reaction_tag(text_without_skip)

        reaction_sent = False
        if reaction_emoji and bot_cfg.behavior.reactions_enabled and primary.msg_id:
            await self.userbot.send_reaction(chat_id, primary.msg_id, reaction_emoji)
            reaction_sent = True

        if not reply_text and not doc_specs:
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
                        use_reply_caption = (
                            idx == 0 and bool(reply_text) and len(reply_text) <= 1024
                        )
                        doc_caption = reply_text if use_reply_caption else gen_doc.caption
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
                        exc,
                    )

            if not sent_as_voice:
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

        if not reply_text and not delivered_docs:
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

        history_text = f"{reply_text} {doc_summary}".strip() if doc_summary else reply_text

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
        bot_name = bot_cfg.behavior.bot_names[0] if bot_cfg.behavior.bot_names else "Джефф"
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

        # Trigger background compaction if threshold reached and provider is not rate-limited
        if not self._is_provider_rate_limited():
            bot_uid = getattr(self.userbot, "bot_id", None)
            excluded_uids = [int(bot_uid)] if bot_uid else None
            await self.compaction_engine.run_compaction_for_chat(
                chat_id,
                excluded_user_ids=excluded_uids,
            )
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
        sent_res = await self.userbot.send_reply(
            chat_id=chat_id,
            text=text,
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
            else "Джефф"
        )
        sent_msg_id = self._extract_sent_message_id(sent_res)
        await self.memory_store.save_message(
            ChatMessage(
                telegram_msg_id=sent_msg_id,
                chat_id=chat_id,
                user_id=self.userbot.bot_id or 1,
                timestamp=int(now),
                author_name=bot_name,
                text=text,
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
        await self.config_mgr.start_watcher()
        await self.userbot.start()
        await self._seed_chat_states_from_memory()

        self.proactive_engine.start_loop(
            get_chat_states=lambda: self.chat_states,
            send_message_fn=self._send_proactive,
            poll_interval=60.0,
            get_target_chat_ids=self._get_proactive_target_chat_ids,
        )

    async def stop(self) -> None:
        """Gracefully stop all background tasks and close connections."""
        for task in list(self._debounce_tasks.values()):
            if not task.done():
                task.cancel()
        self._debounce_tasks.clear()
        self._debounce_buffers.clear()
        self._debounce_started_at.clear()

        self.proactive_engine.stop_loop()
        self.config_mgr.stop_watcher()
        await self.userbot.stop()
        if self._owns_stt_transcriber and hasattr(self.stt_transcriber, "close"):
            self.stt_transcriber.close()
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
            username = "jeff_userbot"
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
        is_dm_testing = False
        if not allowed_chats and config_mgr.bot and config_mgr.bot.telegram.allowed_dm_user_ids:
            allowed_chats = config_mgr.bot.telegram.allowed_dm_user_ids
            is_dm_testing = True
        if not allowed_chats:
            allowed_chats = [-1001234567890]
        chat_id = int(allowed_chats[0])

        allowed_group_chats = [chat_id] if not is_dm_testing else []
        allowed_dm_chats = (
            [chat_id]
            if is_dm_testing
            else (config_mgr.bot.telegram.allowed_dm_user_ids if config_mgr.bot else [])
        )

        userbot = TelegramUserbot(
            session_path=str(tmp_path / "session" / "smoke.session"),
            api_id=config_mgr.bot.telegram.api_id if config_mgr.bot else 12345,
            api_hash=config_mgr.bot.telegram.api_hash if config_mgr.bot else "0123456789abcdef",
            allowed_chat_ids=allowed_group_chats,
            allowed_dm_user_ids=allowed_dm_chats,
            ignored_sender_usernames=(
                config_mgr.bot.telegram.ignored_sender_usernames if config_mgr.bot else []
            ),
            bot_names=config_mgr.bot.behavior.bot_names if config_mgr.bot else ["Джефф"],
            typing_min_seconds=0.01,
            typing_max_seconds=0.02,
            client=dry_tg_client,
        )
        userbot.bot_id = 777000
        userbot.bot_username = "jeff_userbot"

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
            raw_text = "Джефф, посмотри на мою новую сборку клавиатуры!"
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
