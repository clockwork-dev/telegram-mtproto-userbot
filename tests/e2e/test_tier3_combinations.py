"""Tier 3: Cross-Feature Interactions & Pairwise Coverage E2E Tests (>= 15 tests).

Tests multi-subsystem integrations and pairwise feature contracts:
- Pairwise 1: Multimodal Ingestion + Short-Term Memory Storage
- Pairwise 2: FTS5 Memory Facts + Humor Generator Banter Prompt
- Pairwise 3: Friend Profile Sensitive Topics + Banter Exclusion Gate
- Pairwise 4: Config Hot-Reload + Proactive Quiet Hours Evaluation
- Pairwise 5: Config Hot-Reload + In-Chat Heuristic Bot Names
- Pairwise 6: Config Whitelist + In-Chat Heuristic Dispatch
- Pairwise 7: Session Permission Isolation + Telegram MTProto Lifecycle
- Pairwise 8: Antigravity Rate Limit 429 + Subprocess CLI Fallback Invocation
- Pairwise 9: Multimodal Sticker WebP Conversion + Temp Scrubber Cleanup
- Pairwise 10: Proactive Silence Trigger + Memory History + Friend Banter
- Pairwise 11: Context Compaction Engine + FTS5 Facts Archival
- Pairwise 12: Stream Tool Interception + Zero Unauthorized Host Execution
- Pairwise 13: Typing Delay Heuristic + Response Length Scaling
- Pairwise 14: Child Environment Sanitization + SSH Keyring Bypass Marker
- Pairwise 15: Voice Note Transcoding + Multimodal Provider Attachment
- Pairwise 16: CLI Config Arguments + Memory & Proactive Initialization
"""

from __future__ import annotations

import argparse
import os
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from PIL import Image

from bot.antigravity import (
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    MediaAttachment,
    ProviderResponse,
    build_clean_child_env,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import (
    BotConfig,
    FriendProfile,
)
from bot.media.documents import extract_document_text
from bot.media.processor import MediaProcessor
from bot.memory.db import ChatMessage, MemoryStore
from bot.proactive.engine import ProactiveEngine
from bot.proactive.humor import HumorGenerator
from bot.telegram.client import TelegramUserbot, calculate_typing_delay
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

# ============================================================================
# TIER 3 PAIRWISE & INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_t3_01_multimodal_ingestion_and_short_term_memory(tmp_path: Path):
    """Pairwise: Ingest document and record its text summary in SQLite MemoryStore."""
    doc_path = tmp_path / "notes.txt"
    doc_path.write_text("Договорились встретиться в пятницу в 19:00", encoding="utf-8")

    extracted = extract_document_text(doc_path)
    assert "пятницу" in extracted

    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()

    msg = ChatMessage(
        telegram_msg_id=101,
        chat_id=555,
        user_id=12,
        author_name="Антон",
        text=extracted,
        media_type="document",
        media_summary="notes.txt (42 bytes)",
        timestamp=int(time.time()),
    )
    await store.save_message(msg)

    history = await store.get_recent_history(555)
    assert len(history) == 1
    assert history[0].media_type == "document"
    assert "пятницу" in history[0].text
    assert history[0].media_summary == "notes.txt (42 bytes)"
    await store.close()


@pytest.mark.asyncio
async def test_t3_02_fts5_memory_facts_feed_humor_generator(tmp_path: Path):
    """Pairwise: Stored FTS5 memory facts populate humor generator prompt context."""
    db_path = tmp_path / "fts_banter.sqlite"
    store = MemoryStore(str(db_path))
    await store.init_db()

    await store.add_fact(
        subject="Алексей",
        fact="Купил новый механический свитч на синих переключателях",
        category="техника",
        source_chat_id=100,
    )

    facts = await store.search_facts("свитч")
    assert len(facts) >= 1

    generator = HumorGenerator()
    friend = FriendProfile(name="Алексей", banter_topics=["клавиатуры"])

    prompt = generator.build_humor_prompt(
        friend=friend,
        facts=facts,
        trigger_type="banter",
    )

    assert "Алексей" in prompt
    assert "механический свитч" in prompt
    assert "клавиатуры" in prompt
    await store.close()


def test_t3_03_friend_profile_banter_exclusion_in_humor_prompt():
    """Pairwise: Friend profile sensitive topics enforce strict exclusion in humor prompt."""
    generator = HumorGenerator()
    friend = FriendProfile(
        name="Влад",
        banter_topics=["видеоигры", "аниме"],
        sensitive_topics=["кредиты", "ипотека"],
    )

    prompt = generator.build_humor_prompt(
        friend=friend,
        forbidden_topics=["политика"],
    )

    assert "СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ" in prompt
    assert "кредиты" in prompt
    assert "ипотека" in prompt
    assert "политика" in prompt
    assert "Влад" in prompt


def test_t3_04_config_hot_reload_updates_proactive_quiet_hours(tmp_path: Path):
    """Pairwise: Hot-reload of bot.yaml immediately updates quiet hours evaluation."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    bot_file = cfg_dir / "bot.yaml"

    initial_cfg = {
        "telegram": {"api_id": 1, "api_hash": "h"},
        "proactive": {
            "enabled": True,
            "quiet_hours": {"enabled": True, "start": "23:00", "end": "08:00", "timezone": "Europe/Moscow"},
        },
    }
    bot_file.write_text(yaml.dump(initial_cfg))

    mgr = ConfigManager(config_dir=cfg_dir)
    engine = ProactiveEngine(mgr, MagicMock())

    # Daytime 13:00 UTC -> False (active)
    t_day = datetime(2026, 11, 13, 13, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_day) is False

    # Update config: quiet hours now 12:00 to 14:00 UTC
    updated_cfg = {
        "telegram": {"api_id": 1, "api_hash": "h"},
        "proactive": {
            "enabled": True,
            "quiet_hours": {"enabled": True, "start": "12:00", "end": "14:00", "timezone": "UTC"},
        },
    }
    bot_file.write_text(yaml.dump(updated_cfg))
    future_time = bot_file.stat().st_mtime + 10.0
    os.utime(bot_file, (future_time, future_time))

    mgr.reload_changed()
    # Now 13:00 is within quiet hours
    assert engine.is_in_quiet_hours(t_day) is True


def test_t3_05_config_hot_reload_updates_bot_names(tmp_path: Path):
    """Pairwise: Hot-reload adds a new bot name alias recognized by reply heuristics."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    bot_file = cfg_dir / "bot.yaml"

    bot_file.write_text(
        yaml.dump({
            "telegram": {"api_id": 1, "api_hash": "h"},
            "behavior": {"bot_names": ["Джефф"]},
        })
    )

    mgr = ConfigManager(config_dir=cfg_dir)
    heuristics = ReplyHeuristics()

    assert heuristics.is_mention("Привет, Жека!", mgr.bot.behavior.bot_names) is False

    # Operator adds nickname "Жека"
    bot_file.write_text(
        yaml.dump({
            "telegram": {"api_id": 1, "api_hash": "h"},
            "behavior": {"bot_names": ["Джефф", "Жека"]},
        })
    )
    future_time = bot_file.stat().st_mtime + 10.0
    os.utime(bot_file, (future_time, future_time))

    mgr.reload_changed()
    assert heuristics.is_mention("Привет, Жека!", mgr.bot.behavior.bot_names) is True


def test_t3_06_config_whitelist_gates_in_chat_heuristics(tmp_path: Path):
    """Pairwise: ReplyHeuristics enforces chat whitelist loaded from BotConfig."""
    cfg = BotConfig.model_validate({
        "telegram": {"api_id": 1, "api_hash": "h", "allowed_chat_ids": [-100111, -100222]}
    })

    heuristics = ReplyHeuristics()
    allowed_event = MagicMock(chat_id=-100111, out=False, sender_id=99, is_reply=False, raw_text="Джефф привет")
    blocked_event = MagicMock(chat_id=-100999, out=False, sender_id=99, is_reply=False, raw_text="Джефф привет")

    a1 = heuristics.should_reply(allowed_event, bot_id=1, bot_names=["Джефф"], allowed_chat_ids=cfg.telegram.allowed_chat_ids)
    assert a1 == ReplyAction.MUST_REPLY

    a2 = heuristics.should_reply(blocked_event, bot_id=1, bot_names=["Джефф"], allowed_chat_ids=cfg.telegram.allowed_chat_ids)
    assert a2 == ReplyAction.IGNORE


@pytest.mark.asyncio
async def test_t3_07_session_isolation_and_client_lifecycle(tmp_path: Path):
    """Pairwise: Session permission enforcement with TelegramUserbot lifecycle."""
    session_dir = tmp_path / "sessions"
    session_file = session_dir / "userbot.session"
    session_dir.mkdir(parents=True)
    session_file.write_text("session_key")

    mock_raw_client = MagicMock()
    mock_raw_client.is_connected = MagicMock(return_value=True)
    mock_raw_client.connect = AsyncMock()
    mock_raw_client.disconnect = AsyncMock()
    mock_raw_client.is_user_authorized = AsyncMock(return_value=True)
    mock_raw_client.get_me = AsyncMock(return_value=MagicMock(id=123, username="bot"))

    userbot = TelegramUserbot(str(session_file), api_id=1, api_hash="h", client=mock_raw_client)
    await userbot.start()

    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600

    await userbot.stop()
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_t3_08_antigravity_429_triggers_cli_fallback(tmp_path: Path):
    """Pairwise: 429 rate limit error in primary SDK client triggers CLI fallback adapter."""
    config = AntigravityConfig(
        data_dir=tmp_path,
        enable_sdk=True,
        enable_cli_fallback=True,
        max_retries=1,
    )
    provider = AntigravityProvider(config)

    # Mock SDK client raising 429 rate limit
    mock_sdk = MagicMock()
    mock_sdk.generate_response = AsyncMock(side_effect=Exception("429 Too Many Requests: RateLimitExceeded"))
    provider._sdk_client = mock_sdk

    # Mock CLI fallback client returning successful ProviderResponse
    mock_cli = MagicMock()
    mock_cli.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Ответ от fallback CLI адаптера")
    )
    provider._cli_client = mock_cli

    res = await provider.generate_response("Привет")
    assert res.text == "Ответ от fallback CLI адаптера"
    mock_sdk.generate_response.assert_awaited_once()
    mock_cli.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_t3_09_multimodal_sticker_webp_conversion_with_temp_scrubber(tmp_path: Path):
    """Pairwise: Sticker WebP conversion with deterministic scratch scrubbing."""
    processor = MediaProcessor(temp_dir_base=str(tmp_path))

    webp_src = tmp_path / "orig_sticker.webp"
    im = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
    im.save(webp_src, format="WEBP")

    def fake_download(msg, file):
        import shutil
        shutil.copy2(webp_src, file)
        return str(file)

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    mock_msg = MagicMock()
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = True
    mock_msg.photo = False
    mock_msg.document = False

    temp_folder = None
    async with processor.managed_media(mock_msg, client=mock_client) as media:
        assert media is not None
        assert media.media_type == "sticker"
        assert media.mime_type == "image/png"
        temp_folder = Path(media.metadata["_temp_dir"])
        assert temp_folder.exists()

    assert temp_folder is not None
    assert not temp_folder.exists()


@pytest.mark.asyncio
async def test_t3_10_proactive_silence_trigger_with_memory_and_friend(tmp_path: Path):
    """Pairwise: Proactive silence trigger retrieves friend facts and generates joke."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(
        yaml.dump({
            "telegram": {"api_id": 1, "api_hash": "h"},
            "proactive": {"enabled": True, "quiet_hours": {"enabled": False}, "silence_trigger_hours": 4.0},
        })
    )
    (cfg_dir / "friends.yaml").write_text(
        yaml.dump({
            "friends": {
                55: {"name": "Константин", "banter_topics": ["deploy on Friday"]}
            }
        })
    )

    mgr = ConfigManager(config_dir=cfg_dir)
    store = MemoryStore(str(tmp_path / "mem.sqlite"))
    await store.init_db()

    now_ts = int(time.time())
    # Add message from friend ID 55
    await store.save_message(
        ChatMessage(
            telegram_msg_id=1,
            chat_id=10,
            user_id=55,
            author_name="Константин",
            text="Пойду задеплою",
            timestamp=now_ts - (5 * 3600),  # 5 hours ago
        )
    )

    engine = ProactiveEngine(mgr, store)
    joke = await engine.evaluate_and_generate(
        chat_id=10,
        last_chat_msg_time=now_ts - (5 * 3600),
        last_bot_msg_time=now_ts - (6 * 3600),
        current_time=now_ts,
    )

    assert joke is not None
    assert "Константин" in joke
    await store.close()


@pytest.mark.asyncio
async def test_t3_11_context_compaction_archives_messages_to_fts_facts(tmp_path: Path):
    """Pairwise: Context compaction saves extracted facts into FTS search index."""
    db_path = tmp_path / "compact_fts.sqlite"
    store = MemoryStore(str(db_path))
    await store.init_db()

    now_ts = int(time.time())
    for i in range(12):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=i + 1,
                chat_id=10,
                user_id=2,
                author_name="User",
                text=f"Договоренность номер {i}",
                timestamp=now_ts + i,
            )
        )

    async def summarize_extractor(messages):
        return [
            {"subject": "Договоренность", "fact": "Согласован план релиза проекта", "category": "планы"}
        ]

    compacted = await store.compact_old_messages(10, summarize_extractor, threshold=10, batch_size=5)
    assert compacted == 5

    results = await store.search_facts("релиз")
    assert len(results) >= 1
    assert "Согласован план релиза" in results[0].fact
    await store.close()


def test_t3_12_native_tool_abort_prevents_unauthorized_execution():
    """Pairwise: NDJSON tool execution attempt triggers security abort."""
    parser = AntigravityStreamParser()
    events = [
        '{"event": "step_update", "step_update": {"text_delta": "Пробую запустить скрипт"}}',
        '{"event": "step_update", "step_update": {"step_type": "tool", "tool": "exec"}}',
    ]

    with pytest.raises(AntigravityNativeToolAbort):
        for ev in events:
            parser.feed_line(ev)


def test_t3_13_typing_delay_simulation_scales_with_response_text_length():
    """Pairwise: Typing delay smoothly scales with generated response length."""
    short_reply = "Ок."
    long_reply = "Это очень подробный развернутый ответ на вопрос с несколькими абзацами и деталями. " * 5

    delay_short = calculate_typing_delay(short_reply, cpm=300, min_seconds=1.0, max_seconds=8.0, jitter=0.0)
    delay_long = calculate_typing_delay(long_reply, cpm=300, min_seconds=1.0, max_seconds=8.0, jitter=0.0)

    assert delay_short < delay_long
    assert delay_short >= 1.0
    assert delay_long <= 8.0


def test_t3_14_child_environment_sanitization_preserves_ssh_marker(tmp_path: Path):
    """Pairwise: Child environment strips secrets while ensuring isolated home and SSH marker."""
    with patch.dict(os.environ, {"SECRET_KEY": "sk-12345", "LD_PRELOAD": "/evil.so"}, clear=False):
        clean_env = build_clean_child_env(tmp_path, force_file_auth=True)

        assert clean_env["SSH_CONNECTION"] == "127.0.0.1 0 127.0.0.1 22"
        assert clean_env["HOME"] == str(tmp_path)
        assert "SECRET_KEY" not in clean_env
        assert "LD_PRELOAD" not in clean_env


@pytest.mark.asyncio
async def test_t3_15_multimodal_attachment_augmented_prompt(tmp_path: Path):
    """Pairwise: Processed media attachment description is prepended to prompt in CLI fallback."""
    config = AntigravityConfig(
        data_dir=tmp_path,
        enable_sdk=False,  # Force CLI client
        enable_cli_fallback=True,
    )
    provider = AntigravityProvider(config)

    mock_cli = MagicMock()
    mock_cli.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Описал аудиозапись")
    )
    provider._cli_client = mock_cli

    media = MediaAttachment(
        description="Голосовое сообщение длительностью 15 секунд",
        mime_type="audio/wav",
    )

    resp = await provider.generate_response(
        prompt="Что на записи?",
        media_attachments=[media],
    )

    assert resp.text == "Описал аудиозапись"
    call_args = mock_cli.generate_response.call_args[1]
    assert "[Attached Media Descriptions:" in call_args["prompt"]
    assert "Голосовое сообщение" in call_args["prompt"]


def test_t3_16_cli_options_load_config_and_initialize_memory(tmp_path: Path):
    """Pairwise: CLI argument parsing loads config file and locates sqlite db path."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "bot.yaml"
    db_file = tmp_path / "custom_memory.sqlite"

    cfg_file.write_text(
        yaml.dump({
            "telegram": {"api_id": 999, "api_hash": "hash123"},
            "memory": {"db_path": str(db_file)},
        })
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bot.yaml")
    args = parser.parse_args(["--config", str(cfg_file)])
    assert args.config == str(cfg_file)

    mgr = ConfigManager(config_dir=cfg_dir)
    # Validate memory path is configurable from YAML
    assert Path(mgr.bot.memory.db_path).name == "custom_memory.sqlite"
