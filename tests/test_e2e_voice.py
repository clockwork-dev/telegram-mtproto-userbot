"""End-to-End Integration Tests for the Voice Pipeline (STT -> Heuristics/LLM -> TTS -> send_voice)."""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from bot.antigravity import AntigravityCLIClient
from bot.config.manager import ConfigManager
from bot.core.models import ProviderResponse
from bot.main import GroupBotOrchestrator
from bot.media import MediaProcessor, STTResult, STTTranscriber, TTSError, TTSSynthesizer
from bot.media.audio import resolve_ffmpeg_path
from bot.memory.db import MemoryStore
from bot.telegram.bot_api_client import BotAPIMessageEvent
from bot.telegram.handlers import IncomingMessage

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_tts_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep offline E2E voice tests isolated from ambient GEMINI_API_KEY/GOOGLE_API_KEY in .env."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _write_test_wav(path: Path, duration_sec: float = 0.6, sample_rate: int = 16000) -> Path:
    """Create a valid 16kHz mono 16-bit PCM WAV file for offline tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x01" * nframes)
    return path


def _write_test_ogg(path: Path, duration_sec: float = 0.6) -> Path:
    """Create a valid OGG/Opus file using ffmpeg from a synthetic PCM WAV."""
    wav_src = path.with_suffix(".src.wav")
    _write_test_wav(wav_src, duration_sec=duration_sec)
    ffmpeg_bin = resolve_ffmpeg_path()
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(wav_src),
            "-c:a",
            "libopus",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wav_src.unlink(missing_ok=True)
    return path


def _copy_test_config(tmp_path: Path) -> Path:
    """Create an isolated test config directory using Potuzhnych's MTProto defaults and public templates."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    friends_example = REPO_ROOT / "config" / "friends.example.yaml"
    if friends_example.exists():
        shutil.copy2(friends_example, cfg_dir / "friends.yaml")

    userbot_prompt = REPO_ROOT / "config" / "userbot_system_prompt.md"
    if not userbot_prompt.exists():
        userbot_prompt = REPO_ROOT / "config" / "userbot_system_prompt.example.md"
    prompt_text = userbot_prompt.read_text(encoding="utf-8") if userbot_prompt.exists() else "Ты — Потужнич."
    (cfg_dir / "userbot_system_prompt.md").write_text(prompt_text, encoding="utf-8")
    (cfg_dir / "system_prompt.md").write_text(prompt_text, encoding="utf-8")

    data: dict[str, Any] = {
        "telegram": {
            "mode": "userbot",
            "api_id": 12345678,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "phone": None,
            "session_path": str(tmp_path / "session" / "test.session"),
            "allowed_chat_ids": [-1003845062562],
            "allowed_dm_user_ids": [382962362],
            "admin_user_ids": [382962362],
            "ignored_sender_usernames": ["hermes_kuk7yi6kfm3lcdfa_bot"],
        },
        "behavior": {
            "bot_names": ["Потужнич", "Потужный", "Потужний", "potuzhnych"],
            "typing_simulation": True,
            "typing_speed_cpm": 300,
            "typing_min_seconds": 1.5,
            "typing_max_seconds": 8.0,
            "reactions_enabled": True,
            "spontaneous_replies": {
                "enabled": True,
                "min_interval_seconds": 300,
                "probability": 0.15,
                "max_per_hour": 3,
                "active_conversation_window_seconds": 60,
            },
        },
        "quiet_hours": {
            "enabled": True,
            "start": "22:00",
            "end": "08:00",
            "timezone": "Europe/Kyiv",
        },
        "proactive": {
            "enabled": False,
            "forbidden_topics": ["тяжелые болезни", "личные финансы", "тема Польши и поляков"],
        },
        "media": {
            "max_doc_size_mb": 10,
            "temp_dir": str(tmp_path / "media_tmp"),
            "ffmpeg_path": None,
        },
        "memory": {
            "db_path": str(tmp_path / "memory.sqlite"),
            "short_term_limit": 30,
            "compaction_threshold": 100,
            "compaction_batch_size": 50,
        },
        "voice": {
            "tts": {
                "enabled": True,
                "reply_voice_to_voice": True,
            },
            "stt": {
                "enabled": True,
                "model_size": "base",
                "compute_type": "int8",
                "cpu_threads": 2,
                "max_duration_sec": 120,
                "timeout_sec": 45.0,
                "allowed_languages": ["ru", "uk"],
                "model_dir": str(tmp_path / "whisper_models"),
            },
        },
        "image_generation": {
            "enabled": True,
            "output_dir": str(tmp_path / "images_tmp"),
        },
        "antigravity": {
            "enable_sdk": False,
            "ops_workspace_dir": str(tmp_path / "ops_ws"),
        },
        "self_evolution": {
            "enabled": False,
        },
    }
    yaml_content = yaml.safe_dump(data, allow_unicode=True)
    (cfg_dir / "bot.yaml").write_text(yaml_content, encoding="utf-8")
    (cfg_dir / "userbot.yaml").write_text(yaml_content, encoding="utf-8")
    return cfg_dir


def _build_mock_userbot(bot_id: int = 777000111, bot_username: str = "potuzhnych_test_userbot") -> MagicMock:
    userbot = MagicMock()
    userbot.bot_id = bot_id
    userbot.bot_username = bot_username
    userbot.client = MagicMock()

    async def _download_media(message: Any, file: str) -> str:
        src = Path(getattr(message, "file_path", ""))
        shutil.copy2(src, file)
        return file

    userbot.client.download_media = AsyncMock(side_effect=_download_media)
    userbot.send_typing_action = AsyncMock()
    userbot.send_reaction = AsyncMock()
    userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9001}})
    userbot.send_voice = AsyncMock(return_value={"ok": True, "result": {"message_id": 9002}})
    userbot.send_document = AsyncMock(return_value={"ok": True, "result": {"message_id": 9003}})
    userbot.stop = AsyncMock()
    return userbot


def _make_fake_edge_tts_module(captured_calls: List[dict[str, str]], sample_wav: Path):
    """Return a fake `edge_tts` module producing valid MP3 audio via ffmpeg."""
    ffmpeg_bin = resolve_ffmpeg_path()

    class _FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
            self.text = text
            self.voice = voice
            self.rate = rate
            self.pitch = pitch
            captured_calls.append({"text": text, "voice": voice, "rate": rate, "pitch": pitch})

        async def save(self, out_path: str) -> None:
            subprocess.run(
                [ffmpeg_bin, "-y", "-i", str(sample_wav), "-c:a", "libmp3lame", out_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    mod = MagicMock()
    mod.Communicate = _FakeCommunicate
    return mod


def _make_voice_incoming(
    *,
    chat_id: int,
    msg_id: int,
    sender_id: int,
    sender_name: str,
    ogg_path: Path,
    bot_id: int = 777000111,
    caption: str = "",
    reply_to_msg_id: Optional[int] = None,
    reply_to_is_bot: bool = False,
) -> IncomingMessage:
    msg_dict: dict[str, Any] = {
        "message_id": msg_id,
        "chat": {"id": chat_id},
        "from": {"id": sender_id, "first_name": sender_name},
        "voice": {"file_id": f"voice_{msg_id}", "file_size": ogg_path.stat().st_size, "mime_type": "audio/ogg"},
    }
    if caption:
        msg_dict["caption"] = caption
    if reply_to_msg_id is not None:
        msg_dict["reply_to_message"] = {
            "message_id": reply_to_msg_id,
            "from": {"id": bot_id if reply_to_is_bot else 999999, "first_name": "Потужнич" if reply_to_is_bot else "Friend"},
            "text": "предыдущее сообщение",
        }

    ev = BotAPIMessageEvent(msg_dict, bot_id=bot_id)
    ev.file_path = str(ogg_path)  # type: ignore[attr-defined]
    return IncomingMessage(
        msg_id=msg_id,
        chat_id=chat_id,
        sender_id=sender_id,
        sender_name=sender_name,
        text=caption,
        reply_to_msg_id=reply_to_msg_id,
        is_reply=reply_to_msg_id is not None,
        is_mention=False,
        raw_event=ev,
        media=True,
    )


@pytest.mark.asyncio
async def test_e2e_voice_ru_mention_in_audio_to_voice_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Incoming RU voice without caption mentioning Jeff -> STT -> CLI LLM -> Edge TTS (DmitryNeural) -> send_voice + FTS5."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    memory_store = MemoryStore(db_path=str(tmp_path / "voice_e2e.sqlite"))
    await memory_store.init_db()

    sample_wav = _write_test_wav(tmp_path / "tts_src.wav", duration_sec=0.8)
    edge_calls: List[dict[str, str]] = []
    monkeypatch.setattr("bot.media.tts.edge_tts", _make_fake_edge_tts_module(edge_calls, sample_wav))

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Потужнич, привет! Посмотри, пожалуйста, новый серверный билд.",
            language="ru",
            duration=1.2,
            language_probability=0.99,
        )
    )

    # CLI-only provider: verifies _provider_lacks_binary_media does NOT drop transcribed voice
    mock_provider = MagicMock()
    mock_provider._sdk_client = None
    mock_provider._cli_client = MagicMock(spec=AntigravityCLIClient)
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text="Привет! Билд выглядит бодро, но **память** всё равно проверь.",
            model="test-model",
        )
    )

    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.tts_synthesizer.temp_dir_base = str(tmp_path / "tts_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    ogg_in = _write_test_ogg(tmp_path / "in_ru.ogg")
    incoming = _make_voice_incoming(
        chat_id=chat_id,
        msg_id=501,
        sender_id=447164588,
        sender_name="Анюта",
        ogg_path=ogg_in,
    )

    result = await orchestrator.handle_incoming_message(incoming)
    assert result == "Привет! Билд выглядит бодро, но **память** всё равно проверь."

    fake_stt.transcribe.assert_awaited_once()
    mock_provider.generate_response.assert_awaited_once()
    prompt_used = mock_provider.generate_response.await_args.kwargs["prompt"]
    assert "серверный билд" in prompt_used
    assert "ГОЛОСОВОЙ ОТВЕТ" in prompt_used
    assert "[MOOD:" in prompt_used
    for mood_example in (
        "[MOOD: casual cheeky banter]",
        "[MOOD: amused eye-roll, lively]",
        "[MOOD: candid, slightly exasperated]",
        "[MOOD: teasing laugh, natural pace]",
        "[MOOD: friendly sarcastic roast]",
    ):
        assert mood_example in prompt_used
    for vocal_tag in ("[inhale]", "[exhale]", "[sigh]", "[chuckles]", "[laughs]", "[scoffs]", "[short pause]"):
        assert vocal_tag in prompt_used
    assert "1–2 живые устные фразы" in prompt_used
    assert "ЗАПРЕЩЕНО: `[whispers]`" in prompt_used

    userbot.send_voice.assert_awaited_once()
    userbot.send_reply.assert_not_awaited()
    voice_kwargs = userbot.send_voice.await_args.kwargs
    assert voice_kwargs["chat_id"] == chat_id
    assert voice_kwargs["reply_to_msg_id"] == 501
    assert voice_kwargs["duration"] > 0

    assert len(edge_calls) == 1
    assert config_mgr.bot.voice.tts.ru_voice == "ru-RU-DmitryNeural"
    assert edge_calls[0]["voice"] == "ru-RU-DmitryNeural"
    assert "**" not in edge_calls[0]["text"]

    history = await memory_store.get_recent_history(chat_id, limit=10)
    assert len(history) == 2
    assert history[0].media_type == "voice"
    assert "серверный билд" in (history[0].text or "")
    assert "Голосовое сообщение (ru)" in (history[0].media_summary or "")
    assert history[1].media_type == "voice"
    assert history[1].media_summary == "Голосовой ответ (ru)"
    assert "Билд выглядит бодро" in (history[1].text or "")

    fts_hits = await memory_store.search_messages("серверный билд", limit=5)
    assert any(h.telegram_msg_id == 501 for h in fts_hits)

    # Confirm managed temp dirs were cleaned up
    assert list((tmp_path / "tts_tmp").glob("bot_tts_*")) == []
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_uk_reply_to_bot_selects_uk_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Incoming UK voice replying to Jeff -> STT -> LLM (UK) -> Edge TTS (OstapNeural) -> send_voice."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    memory_store = MemoryStore(db_path=str(tmp_path / "voice_uk.sqlite"))
    await memory_store.init_db()

    sample_wav = _write_test_wav(tmp_path / "tts_uk_src.wav", duration_sec=0.7)
    edge_calls: List[dict[str, str]] = []
    monkeypatch.setattr("bot.media.tts.edge_tts", _make_fake_edge_tts_module(edge_calls, sample_wav))

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Як справи, чи йдеш сьогодні на каву?",
            language="uk",
            duration=1.1,
            language_probability=0.98,
        )
    )

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text="Усе чудово, але каву я вже випив зранку!",
            model="test-model",
        )
    )

    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.tts_synthesizer.temp_dir_base = str(tmp_path / "tts_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    ogg_in = _write_test_ogg(tmp_path / "in_uk.ogg")
    incoming = _make_voice_incoming(
        chat_id=chat_id,
        msg_id=502,
        sender_id=447164588,
        sender_name="Анюта",
        ogg_path=ogg_in,
        reply_to_msg_id=100,
        reply_to_is_bot=True,
    )

    result = await orchestrator.handle_incoming_message(incoming)
    assert result == "Усе чудово, але каву я вже випив зранку!"
    userbot.send_voice.assert_awaited_once()
    userbot.send_reply.assert_not_awaited()
    assert config_mgr.bot.voice.tts.uk_voice == "uk-UA-OstapNeural"
    assert edge_calls[0]["voice"] == "uk-UA-OstapNeural"

    history = await memory_store.get_recent_history(chat_id, limit=10)
    assert history[-1].media_type == "voice"
    assert history[-1].media_summary == "Голосовой ответ (uk)"
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_irrelevant_chatter_saved_without_llm_call(tmp_path: Path) -> None:
    """Voice message without bot mention and with spontaneous disabled -> transcribed, saved to FTS5, NO LLM call."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    assert config_mgr.bot is not None
    config_mgr.bot.behavior.spontaneous_replies.enabled = False
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])

    memory_store = MemoryStore(db_path=str(tmp_path / "voice_ignore.sqlite"))
    await memory_store.init_db()

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Ребята, я пошёл за кофе и вернусь через десять минут.",
            language="ru",
            duration=2.1,
            language_probability=0.97,
        )
    )

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock()

    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    ogg_in = _write_test_ogg(tmp_path / "in_noise.ogg")
    incoming = _make_voice_incoming(
        chat_id=chat_id,
        msg_id=503,
        sender_id=447164588,
        sender_name="Анюта",
        ogg_path=ogg_in,
    )

    assert await orchestrator.handle_incoming_message(incoming) is None
    fake_stt.transcribe.assert_awaited_once()
    mock_provider.generate_response.assert_not_awaited()
    userbot.send_voice.assert_not_awaited()
    userbot.send_reply.assert_not_awaited()

    fts_hits = await memory_store.search_messages("пошёл кофе десять", limit=5)
    assert len(fts_hits) == 1
    assert fts_hits[0].media_type == "voice"
    assert "Голосовое сообщение (ru)" in (fts_hits[0].media_summary or "")
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_fallbacks_to_text_on_code_long_or_tts_error(tmp_path: Path) -> None:
    """Code blocks, >max_chars, TTS failure, and send_voice error all fall back safely to send_reply."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    assert config_mgr.bot is not None
    config_mgr.bot.voice.tts.max_chars = 60
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])

    memory_store = MemoryStore(db_path=str(tmp_path / "voice_fallback.sqlite"))
    await memory_store.init_db()

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(text="Потужнич, покажи пример кода", language="ru", duration=1.0)
    )

    mock_provider = MagicMock()
    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    ogg_in = _write_test_ogg(tmp_path / "in_fb.ogg")

    # 1. Code block -> TTSSkippedError -> send_reply
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Держи:\n```python\nprint('ok')\n```", model="m")
    )
    inc1 = _make_voice_incoming(chat_id=chat_id, msg_id=601, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
    await orchestrator.handle_incoming_message(inc1)
    userbot.send_voice.assert_not_awaited()
    userbot.send_reply.assert_awaited_once()

    # 2. Exceeds max_chars (60) -> TTSSkippedError -> send_reply
    userbot.send_reply.reset_mock()
    long_reply = "Это очень длинный ответ который точно не поместится в лимит шестидесяти символов."
    mock_provider.generate_response = AsyncMock(return_value=ProviderResponse(text=long_reply, model="m"))
    inc2 = _make_voice_incoming(chat_id=chat_id, msg_id=602, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
    await orchestrator.handle_incoming_message(inc2)
    userbot.send_voice.assert_not_awaited()
    userbot.send_reply.assert_awaited_once()

    # 3. TTSError during synthesis -> send_reply
    userbot.send_reply.reset_mock()
    mock_provider.generate_response = AsyncMock(return_value=ProviderResponse(text="Короткий ответ.", model="m"))

    @asynccontextmanager
    async def _failing_speech(*args, **kwargs):
        raise TTSError("edge-tts network timeout")
        yield  # pragma: no cover

    orchestrator.tts_synthesizer.managed_speech = _failing_speech  # type: ignore[method-assign]
    inc3 = _make_voice_incoming(chat_id=chat_id, msg_id=603, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
    await orchestrator.handle_incoming_message(inc3)
    userbot.send_voice.assert_not_awaited()
    userbot.send_reply.assert_awaited_once()

    # 4. send_voice raises -> send_reply
    userbot.send_reply.reset_mock()
    fake_ogg_out = _write_test_ogg(tmp_path / "out_fake.ogg")

    @asynccontextmanager
    async def _ok_speech(*args, **kwargs):
        res = MagicMock()
        res.audio_path = fake_ogg_out
        res.duration = 0.6
        res.language = "ru"
        yield res

    orchestrator.tts_synthesizer.managed_speech = _ok_speech  # type: ignore[method-assign]
    userbot.send_voice = AsyncMock(side_effect=RuntimeError("Telegram 500"))
    inc4 = _make_voice_incoming(chat_id=chat_id, msg_id=604, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
    await orchestrator.handle_incoming_message(inc4)
    userbot.send_voice.assert_awaited_once()
    userbot.send_reply.assert_awaited_once()

    # 5. Document spec in reply -> sends document instead of voice
    userbot.send_voice.reset_mock()
    userbot.send_reply.reset_mock()
    userbot.send_document.reset_mock()
    doc_payload = (
        'Вот файл! [DOC:{"format":"pdf","title":"Отчет","filename":"report.pdf",'
        '"sections":[{"heading":"Итог","body":"Все готово."}]}]'
    )
    mock_provider.generate_response = AsyncMock(return_value=ProviderResponse(text=doc_payload, model="m"))
    inc5 = _make_voice_incoming(chat_id=chat_id, msg_id=605, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
    await orchestrator.handle_incoming_message(inc5)
    userbot.send_document.assert_awaited_once()
    userbot.send_voice.assert_not_awaited()
    assert isinstance(orchestrator.media_processor, MediaProcessor)
    assert isinstance(orchestrator.tts_synthesizer, TTSSynthesizer)
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_reply_to_voice_message_transcribes_context_and_respects_hot_reload(tmp_path: Path) -> None:
    """Text reply to a voice message transcribes replied voice context; hot-reload disables voice reply."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    memory_store = MemoryStore(db_path=str(tmp_path / "voice_reply_ctx.sqlite"))
    await memory_store.init_db()

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Зустрічаємося о сьомій біля метро.",
            language="uk",
            duration=1.5,
        )
    )

    mock_provider = MagicMock()
    mock_provider._sdk_client = None
    mock_provider._cli_client = MagicMock(spec=AntigravityCLIClient)
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Він каже, що зустріч о сьомій біля метро.", model="m")
    )

    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    replied_ogg = _write_test_ogg(tmp_path / "replied_voice.ogg")
    msg_dict = {
        "message_id": 701,
        "chat": {"id": chat_id},
        "from": {"id": 447164588, "first_name": "Анюта"},
        "text": "@potuzhnych_test_userbot що він сказав?",
        "reply_to_message": {
            "message_id": 700,
            "from": {"id": 555111, "first_name": "Макс"},
            "voice": {"file_id": "voice_700", "file_size": replied_ogg.stat().st_size, "mime_type": "audio/ogg"},
        },
    }
    ev = BotAPIMessageEvent(msg_dict, bot_id=userbot.bot_id)
    ev.reply_to_media_event.file_path = str(replied_ogg)  # type: ignore[attr-defined]
    incoming = IncomingMessage(
        msg_id=701,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev.text,
        reply_to_msg_id=700,
        is_reply=True,
        is_mention=True,
        raw_event=ev,
        media=False,
        reply_to_author=ev.reply_to_author,
        reply_to_media_event=ev.reply_to_media_event,
    )

    res = await orchestrator.handle_incoming_message(incoming)
    assert res is not None
    prompt_used = mock_provider.generate_response.await_args.kwargs["prompt"]
    assert "Зустрічаємося о сьомій біля метро" in prompt_used
    userbot.send_reply.assert_awaited_once()
    userbot.send_voice.assert_not_awaited()

    # Verify hot-reload propagates voice.tts.reply_voice_to_voice = false
    bot_yaml_path = cfg_dir / "bot.yaml"
    yaml_text = bot_yaml_path.read_text(encoding="utf-8")
    bot_yaml_path.write_text(
        yaml_text.replace("reply_voice_to_voice: true", "reply_voice_to_voice: false"),
        encoding="utf-8",
    )
    os.utime(bot_yaml_path, (os.stat(bot_yaml_path).st_mtime + 2.0, os.stat(bot_yaml_path).st_mtime + 2.0))
    assert config_mgr.reload_changed() is True

    userbot.send_reply.reset_mock()
    userbot.send_voice.reset_mock()
    ogg_in2 = _write_test_ogg(tmp_path / "in_after_reload.ogg")
    inc2 = _make_voice_incoming(
        chat_id=chat_id,
        msg_id=702,
        sender_id=447164588,
        sender_name="Анюта",
        ogg_path=ogg_in2,
        reply_to_msg_id=701,
        reply_to_is_bot=True,
    )
    await orchestrator.handle_incoming_message(inc2)
    userbot.send_voice.assert_not_awaited()
    userbot.send_reply.assert_awaited_once()
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_tts_hot_reload_propagates_gemini_fields_and_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot-reloading voice.tts propagates all Gemini/Edge fields and registers configured API key env secrets."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    memory_store = MemoryStore(db_path=str(tmp_path / "voice_hot_reload.sqlite"))
    await memory_store.init_db()

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock()
    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
    )

    try:
        assert orchestrator.tts_synthesizer.provider == "auto"
        assert orchestrator.tts_synthesizer.fallback_to_edge is True
        assert orchestrator.tts_synthesizer.gemini_model == "gemini-2.5-flash-preview-tts"
        assert orchestrator.tts_synthesizer.gemini_ru_voice == "Fenrir"
        assert orchestrator.tts_synthesizer.gemini_uk_voice == "Fenrir"
        assert orchestrator.tts_synthesizer.ru_voice == "ru-RU-DmitryNeural"
        assert orchestrator.tts_synthesizer.uk_voice == "uk-UA-OstapNeural"
        assert orchestrator.tts_synthesizer.gemini_api_key_env == "GEMINI_API_KEY"

        custom_key = "AIzaSyHotReloadCustomKey1234567890abcdef"
        monkeypatch.setenv("CUSTOM_GEMINI_TTS_TOKEN_VAR", custom_key)

        bot_yaml_path = cfg_dir / "bot.yaml"
        data = yaml.safe_load(bot_yaml_path.read_text(encoding="utf-8"))
        data["voice"]["tts"].update(
            {
                "provider": "gemini",
                "fallback_to_edge": False,
                "max_chars": 420,
                "timeout_sec": 12.5,
                "gemini_model": "gemini-2.5-pro-preview-tts",
                "gemini_ru_voice": "Charon",
                "gemini_uk_voice": "Orus",
                "gemini_style_prompt": "Lively conversational tone with clear diction.",
                "gemini_api_key_env": "CUSTOM_GEMINI_TTS_TOKEN_VAR",
            }
        )
        bot_yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        stat = os.stat(bot_yaml_path)
        os.utime(bot_yaml_path, (stat.st_mtime + 2.0, stat.st_mtime + 2.0))

        assert config_mgr.reload_changed() is True

        tts = orchestrator.tts_synthesizer
        assert tts.provider == "gemini"
        assert tts.fallback_to_edge is False
        assert tts.max_chars == 420
        assert tts.timeout_sec == 12.5
        assert tts.gemini_model == "gemini-2.5-pro-preview-tts"
        assert tts.gemini_ru_voice == "Charon"
        assert tts.gemini_uk_voice == "Orus"
        assert tts.gemini_style_prompt == "Lively conversational tone with clear diction."
        assert tts.gemini_api_key_env == "CUSTOM_GEMINI_TTS_TOKEN_VAR"

        # Custom env secret and AIza... Google API key pattern must be masked
        assert custom_key not in orchestrator.sanitizer.sanitize(f"leak={custom_key}")
        short_google_key = "AIzaSyShortTestKey1234567890"
        assert short_google_key not in orchestrator.sanitizer.sanitize(f"k={short_google_key}")
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_text_fallback_strips_tts_emotion_tags(tmp_path: Path) -> None:
    """When TTS skips/fails or send_voice fails, outgoing text and saved history strip TTS emotion tags."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    assert config_mgr.bot is not None
    config_mgr.bot.voice.tts.max_chars = 80
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])

    memory_store = MemoryStore(db_path=str(tmp_path / "voice_strip_tags.sqlite"))
    await memory_store.init_db()

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(text="Потужнич, расскажи что там с логами", language="ru", duration=1.0)
    )

    mock_provider = MagicMock()
    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    ogg_in = _write_test_ogg(tmp_path / "in_strip.ogg")

    try:
        # 1. TTSSkippedError (too_long) -> plain text fallback strips [MOOD:...] and [laughs] / [sigh]
        userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9101}})
        long_tagged = (
            "[MOOD: playful sarcasm]\n"
            "[laughs] Ну ты и даёшь, [sigh] но этот ответ слишком длинный для голосовухи, "
            "так что читай руками."
        )
        mock_provider.generate_response = AsyncMock(return_value=ProviderResponse(text=long_tagged, model="m"))
        inc1 = _make_voice_incoming(chat_id=chat_id, msg_id=801, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
        await orchestrator.handle_incoming_message(inc1)
        userbot.send_voice.assert_not_awaited()
        userbot.send_reply.assert_awaited_once()
        sent1 = userbot.send_reply.await_args.kwargs["text"]
        assert "[MOOD:" not in sent1
        assert "[laughs]" not in sent1
        assert "[sigh]" not in sent1
        assert "Ну ты и даёшь" in sent1

        # 2. TTSError during synthesis -> plain text fallback strips [STYLE:...] / [whispers] / [excitement]
        userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9102}})
        mock_provider.generate_response = AsyncMock(
            return_value=ProviderResponse(
                text="[STYLE: conspiratorial whisper] [whispers] Тише, [excitement] билд уже зелёный!",
                model="m",
            )
        )

        @asynccontextmanager
        async def _failing_speech(*args, **kwargs):
            raise TTSError("Gemini TTS unavailable")
            yield  # pragma: no cover

        orchestrator.tts_synthesizer.managed_speech = _failing_speech  # type: ignore[method-assign]
        inc2 = _make_voice_incoming(chat_id=chat_id, msg_id=802, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
        await orchestrator.handle_incoming_message(inc2)
        userbot.send_reply.assert_awaited_once()
        sent2 = userbot.send_reply.await_args.kwargs["text"]
        assert "[STYLE:" not in sent2
        assert "[whispers]" not in sent2
        assert "[excitement]" not in sent2
        assert "Тише, билд уже зелёный!" == sent2

        # 3. send_voice returns falsy -> fallback strips [TONE:...] and emotion tags
        userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9103}})
        fake_ogg_out = _write_test_ogg(tmp_path / "out_falsy.ogg")

        @asynccontextmanager
        async def _ok_speech(*args, **kwargs):
            res = MagicMock()
            res.audio_path = fake_ogg_out
            res.duration = 0.6
            res.language = "ru"
            yield res

        orchestrator.tts_synthesizer.managed_speech = _ok_speech  # type: ignore[method-assign]
        userbot.send_voice = AsyncMock(return_value=None)
        mock_provider.generate_response = AsyncMock(
            return_value=ProviderResponse(
                text="[TONE: cheerful] [chuckles] Готово, проверяй! [short pause]",
                model="m",
            )
        )
        inc3 = _make_voice_incoming(chat_id=chat_id, msg_id=803, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
        await orchestrator.handle_incoming_message(inc3)
        userbot.send_voice.assert_awaited_once()
        userbot.send_reply.assert_awaited_once()
        sent3 = userbot.send_reply.await_args.kwargs["text"]
        assert "[TONE:" not in sent3
        assert "[chuckles]" not in sent3
        assert "[short pause]" not in sent3
        assert sent3 == "Готово, проверяй!"

        # 4. Tags-only response -> TTSSkippedError("empty") -> no empty send_reply and no MemoryStore pollution
        userbot.send_reply.reset_mock()
        userbot.send_voice.reset_mock()
        orchestrator.tts_synthesizer = TTSSynthesizer(
            config=config_mgr.bot.voice.tts,
            ffmpeg_path=config_mgr.bot.media.ffmpeg_path,
            temp_dir_base=str(tmp_path / "tts_tmp"),
        )
        mock_provider.generate_response = AsyncMock(
            return_value=ProviderResponse(text="[MOOD: amused] [laughs] [sigh]", model="m")
        )
        inc4 = _make_voice_incoming(chat_id=chat_id, msg_id=804, sender_id=447164588, sender_name="Анюта", ogg_path=ogg_in)
        res4 = await orchestrator.handle_incoming_message(inc4)
        assert res4 is None
        userbot.send_voice.assert_not_awaited()
        userbot.send_reply.assert_not_awaited()

        history = await memory_store.get_recent_history(chat_id, limit=20)
        bot_msgs = [m.text for m in history if m.user_id == userbot.bot_id]
        assert len(bot_msgs) == 3
        assert all(
            tag not in (msg or "")
            for msg in bot_msgs
            for tag in (
                "[MOOD:",
                "[STYLE:",
                "[TONE:",
                "[laughs]",
                "[sigh]",
                "[whispers]",
                "[excitement]",
                "[chuckles]",
                "[short pause]",
            )
        )
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_e2e_voice_success_keeps_tags_for_tts_but_strips_mood_and_tags_from_memory_store(
    tmp_path: Path,
) -> None:
    """When send_voice succeeds, TTS receives [MOOD:...] + inline tags, while MemoryStore saves clean text."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir)
    assert config_mgr.bot is not None
    assert config_mgr.bot.antigravity.enable_sdk is False
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])

    memory_store = MemoryStore(db_path=str(tmp_path / "voice_mem_clean.sqlite"))
    await memory_store.init_db()

    fake_stt = MagicMock(spec=STTTranscriber)
    fake_stt.enabled = True
    fake_stt.transcribe = AsyncMock(
        return_value=STTResult(text="Потужнич, ну как там релиз?", language="ru", duration=1.0)
    )

    tagged_reply = "[MOOD: amused, playful sarcasm]\n[sigh] Ну наконец-то, [laughs softly] релиз уже катится!"
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text=tagged_reply, model="m")
    )
    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
        stt_transcriber=fake_stt,
    )
    orchestrator.media_processor.temp_dir_base = str(tmp_path / "media_tmp")
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    captured_tts_inputs: List[str] = []
    fake_ogg_out = _write_test_ogg(tmp_path / "out_expressive.ogg")

    @asynccontextmanager
    async def _capturing_speech(text: str, *args, **kwargs):
        captured_tts_inputs.append(text)
        res = MagicMock()
        res.audio_path = fake_ogg_out
        res.duration = 0.8
        res.language = "ru"
        yield res

    orchestrator.tts_synthesizer.managed_speech = _capturing_speech  # type: ignore[method-assign]
    ogg_in = _write_test_ogg(tmp_path / "in_expressive.ogg")

    try:
        inc = _make_voice_incoming(
            chat_id=chat_id,
            msg_id=850,
            sender_id=447164588,
            sender_name="Анюта",
            ogg_path=ogg_in,
        )
        result = await orchestrator.handle_incoming_message(inc)
        assert captured_tts_inputs == [tagged_reply]
        userbot.send_voice.assert_awaited_once()
        userbot.send_reply.assert_not_awaited()

        assert result == "Ну наконец-то, релиз уже катится!"
        history = await memory_store.get_recent_history(chat_id, limit=10)
        bot_msgs = [m for m in history if m.user_id == userbot.bot_id]
        assert len(bot_msgs) == 1
        assert bot_msgs[0].media_type == "voice"
        assert bot_msgs[0].text == "Ну наконец-то, релиз уже катится!"
        assert "[MOOD:" not in (bot_msgs[0].text or "")
        assert "[sigh]" not in (bot_msgs[0].text or "")
        assert "[laughs softly]" not in (bot_msgs[0].text or "")
    finally:
        await orchestrator.stop()


@pytest.mark.asyncio
async def test_build_context_prompt_voice_reply_mode_and_userbot_system_prompt_conversational_guidance(
    tmp_path: Path,
) -> None:
    """build_context_prompt(voice_reply_mode=True) and config/userbot_system_prompt.md align on Jeff's spoken guidance."""
    cfg_dir = _copy_test_config(tmp_path)
    config_mgr = ConfigManager(config_dir=cfg_dir, config_path=cfg_dir / "userbot.yaml")
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    memory_store = MemoryStore(db_path=str(tmp_path / "voice_prompt_guidance.sqlite"))
    await memory_store.init_db()

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock()
    userbot = _build_mock_userbot()
    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        provider=mock_provider,
        userbot=userbot,
    )

    try:
        ogg_in = _write_test_ogg(tmp_path / "in_prompt_check.ogg")
        incoming = _make_voice_incoming(
            chat_id=chat_id,
            msg_id=901,
            sender_id=447164588,
            sender_name="Анюта",
            ogg_path=ogg_in,
            caption="Ну что там с тестами?",
        )

        prompt_text = orchestrator.build_context_prompt(
            incoming=incoming,
            recent_history=[],
            facts=[],
            friend_profile=None,
            voice_reply_mode=False,
        )
        assert "ГОЛОСОВОЙ ОТВЕТ" not in prompt_text

        prompt_voice = orchestrator.build_context_prompt(
            incoming=incoming,
            recent_history=[],
            facts=[],
            friend_profile=None,
            voice_reply_mode=True,
        )
        prompt_file = REPO_ROOT / "config" / "userbot_system_prompt.md"
        if not prompt_file.exists():
            prompt_file = REPO_ROOT / "config" / "userbot_system_prompt.example.md"
        sys_prompt = prompt_file.read_text(encoding="utf-8")

        # Anti-loop and persona invariants in userbot_system_prompt(.example).md
        assert "Потужнич" in sys_prompt and "Potuzhnych" in sys_prompt
        if prompt_file.name == "userbot_system_prompt.md":
            assert "@hermes_kuk7yi6kfm3lcdfa_bot" in sys_prompt
            assert "382962362" in sys_prompt
        assert "піструнькиня" in sys_prompt and "пиструнькиня" in sys_prompt
        assert "Джефф" in sys_prompt  # explicitly listed as ignored sibling/legacy alias
        assert '[IMG:{"prompt":' in sys_prompt
        assert '[DOC:{"format":' in sys_prompt

        expected_moods = (
            "[MOOD: dry stoic deadpan, confident]",
            "[MOOD: grounded masculine banter]",
            "[MOOD: casual cheeky banter]",
            "[MOOD: amused eye-roll, lively]",
            "[MOOD: candid, slightly exasperated]",
            "[MOOD: teasing laugh, natural pace]",
            "[MOOD: friendly sarcastic roast]",
        )
        expected_tags = (
            "[inhale]",
            "[exhale]",
            "[sigh]",
            "[chuckles]",
            "[laughs]",
            "[scoffs]",
            "[short pause]",
        )

        for target_text in (prompt_voice, sys_prompt):
            assert "ГОЛОСОВОЙ ОТВЕТ" in target_text
            for mood in expected_moods:
                assert mood in target_text
            for tag in expected_tags:
                assert tag in target_text
            assert "1–2 живые устные фразы" in target_text
            assert "ЗАПРЕЩЕНО: `[whispers]`" in target_text
            assert "Markdown-разметка, таблицы, списки, блоки кода и URL-ссылки" in target_text
    finally:
        await orchestrator.stop()
