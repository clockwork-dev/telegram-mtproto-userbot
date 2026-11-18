"""Pydantic v2 Configuration Models for Bot, Telegram, Memory, Friends & Proactive."""

from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator


class TelegramConfig(BaseModel):
    """Configuration for Telegram client (supports both Classic Bot API and MTProto Userbot)."""

    mode: str = "auto"  # "auto", "bot_api", or "userbot"
    bot_token: Optional[str] = None
    api_id: int
    api_hash: str
    phone: Optional[str] = None
    session_path: str = "data/session/userbot.session"
    allowed_chat_ids: List[int] = Field(default_factory=list)
    allowed_dm_user_ids: List[int] = Field(default_factory=list)
    ignored_sender_usernames: List[str] = Field(
        default_factory=lambda: ["hermes_kuk7yi6kfm3lcdfa_bot"]
    )

    @field_validator("allowed_chat_ids", "allowed_dm_user_ids", "ignored_sender_usernames", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        if v is None:
            return []
        return v


class SpontaneousRepliesConfig(BaseModel):
    """Heuristic settings for spontaneous in-chat conversation joining."""

    enabled: bool = True
    min_interval_seconds: int = 300
    probability: float = 0.20
    max_per_hour: int = 3
    active_conversation_window_seconds: int = Field(default=90, ge=0)


class BehaviorConfig(BaseModel):
    """In-chat conversational persona and behavioral timings."""

    bot_names: List[str] = Field(default_factory=lambda: ["Джефф", "бот"])
    typing_simulation: bool = True
    typing_speed_cpm: int = 300
    typing_min_seconds: float = 1.5
    typing_max_seconds: float = 8.0
    reactions_enabled: bool = True
    spontaneous_replies: SpontaneousRepliesConfig = Field(default_factory=SpontaneousRepliesConfig)


class QuietHoursConfig(BaseModel):
    """Quiet hours window during which proactive actions are silenced."""

    enabled: bool = True
    start: str = "23:00"
    end: str = "08:00"
    timezone: str = "Europe/Moscow"


class CronDigestJobConfig(BaseModel):
    """Scheduled cron job configuration for proactive news digests."""

    id: str
    enabled: bool = True
    cron: str
    timezone: str = "Europe/Kyiv"
    slot_label: str = "morning"
    prompt_hint: str = ""
    topics: List[str] = Field(default_factory=list)
    respect_quiet_hours: bool = True
    chat_ids: List[int] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("CronDigestJobConfig.id must not be empty")
        return cleaned

    @field_validator("cron")
    @classmethod
    def _validate_cron(cls, value: str) -> str:
        # Lazy import avoids circular dependency via bot.proactive.__init__ -> engine -> manager.
        from bot.proactive.cron import CronExpression

        parsed = CronExpression.parse(value)
        return parsed.expression

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Timezone must not be empty")
        try:
            ZoneInfo(cleaned)
        except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid timezone: {value!r}") from exc
        return cleaned


class ProactiveConfig(BaseModel):
    """Proactive message scheduling, triggers, and content limits."""

    enabled: bool = True
    quiet_hours: QuietHoursConfig = Field(default_factory=QuietHoursConfig)
    silence_trigger_hours: float = 6.0
    scheduled_interval_minutes: int = 180
    jitter_minutes: int = 45
    cooldown_minutes: int = 90
    humor_style: str = "friendly_sarcasm"
    forbidden_topics: List[str] = Field(default_factory=list)
    cron_digests: List[CronDigestJobConfig] = Field(default_factory=list)


class MediaConfig(BaseModel):
    """Media processing thresholds and tool binaries."""

    max_doc_size_mb: int = 10
    temp_dir: str = "data/tmp"
    ffmpeg_path: Optional[str] = None


class TTSConfig(BaseModel):
    """Text-to-Speech (Edge TTS) synthesis and voice-reply configuration."""

    enabled: bool = True
    ru_voice: str = "ru-RU-SvetlanaNeural"
    uk_voice: str = "uk-UA-PolinaNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"
    max_chars: int = Field(default=600, ge=50, le=4000)
    timeout_sec: float = Field(default=25.0, gt=0.0)
    reply_voice_to_voice: bool = True


class STTConfig(BaseModel):
    """Speech-to-Text (local transcription) configuration for incoming voice notes."""

    enabled: bool = True
    model_size: str = "base"
    compute_type: str = "int8"
    cpu_threads: int = Field(default=2, ge=1, le=16)
    max_duration_sec: int = Field(default=120, ge=5, le=600)
    timeout_sec: float = Field(default=45.0, gt=0.0)
    allowed_languages: List[str] = Field(default_factory=lambda: ["ru", "uk"])
    model_dir: Optional[str] = "data/models/whisper"


class VoiceConfig(BaseModel):
    """Combined TTS and STT configuration."""

    tts: TTSConfig = Field(default_factory=TTSConfig)
    stt: STTConfig = Field(default_factory=STTConfig)


class MemoryConfig(BaseModel):
    """Memory database settings and compaction thresholds."""

    db_path: str = "data/memory.sqlite"
    short_term_limit: int = 30
    compaction_threshold: int = 60
    compaction_batch_size: int = 30


class BotConfig(BaseModel):
    """Root configuration model combining all subsystem configs."""

    telegram: TelegramConfig
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)


class FriendProfile(BaseModel):
    """Rich conversational profile of a known friend or group participant."""

    name: str
    aliases: List[str] = Field(default_factory=list)
    bio: str = ""
    role: str = ""
    communication_style: str = ""
    banter_topics: List[str] = Field(default_factory=list)
    sensitive_topics: List[str] = Field(default_factory=list)


class FriendsConfig(BaseModel):
    """Friends directory mapping telegram_user_id to FriendProfile."""

    friends: Dict[int, FriendProfile] = Field(default_factory=dict)
