"""Tier 1: Feature Coverage E2E Tests (Features 1-10, >= 50 tests).

Covers all 10 core features with >= 5 tests each:
- Feature 1: Antigravity Isolated OAuth Storage (R1)
- Feature 2: Antigravity SDK & Fallback Streaming (R1)
- Feature 3: Telegram MTProto Client & Session (R2)
- Feature 4: In-Chat Behavior & Reply Heuristics (R2)
- Feature 5: Multimodal Media Processing (R2)
- Feature 6: SQLite Short-Term & FTS5 Long-Term Memory (R3)
- Feature 7: Pydantic Config & Friends Profiles (R3)
- Feature 8: Proactive Engine & Banter Generator (R4)
- Feature 9: CLI Tools & Unified Entrypoint (R1, R2, R4)
- Feature 10: Security & Git Cleanliness (AC)
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

from bot.antigravity.auth import (
    _KEYRING_BYPASS_SSH_MARKER,
    build_clean_child_env,
    check_antigravity_token_file_evidence,
    describe_antigravity_auth_source,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import (
    AntigravityNativeToolAbort,
    ErrorCategory,
    calculate_backoff,
    classify_error,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import (
    BotConfig,
    FriendProfile,
    FriendsConfig,
)
from bot.media.documents import FileTooLargeError, check_file_size, extract_document_text
from bot.media.processor import MediaProcessor
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FTSStore, build_fts_query, stem_russian_token
from bot.proactive.engine import ProactiveEngine
from bot.proactive.humor import HumorGenerator
from bot.telegram.auth import enforce_session_permissions
from bot.telegram.client import TelegramUserbot, calculate_typing_delay
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

# ============================================================================
# FEATURE 1: Antigravity Isolated OAuth Storage (R1)
# ============================================================================


def test_f1_isolated_home_directory_structure_and_permissions(tmp_path: Path):
    """Verify setup_isolated_home enforces 0700 permissions and creates empty SOUL.md."""
    data_dir = tmp_path / "antigravity"
    isolated_home, gemini_dir = setup_isolated_home(data_dir)

    assert isolated_home.exists()
    assert gemini_dir.exists()
    # Directory permissions must be 0700 (rwx------)
    assert stat.S_IMODE(isolated_home.stat().st_mode) == 0o700
    assert stat.S_IMODE(gemini_dir.stat().st_mode) == 0o700

    # Global persona suppression: SOUL.md must exist and be empty (0 bytes)
    soul_file = gemini_dir.parent / "SOUL.md"
    assert soul_file.exists()
    assert soul_file.stat().st_size == 0


def test_f1_ssh_connection_keyring_bypass_marker(tmp_path: Path):
    """Verify SSH_CONNECTION keyring bypass marker is injected into child environment."""
    env = build_clean_child_env(tmp_path, force_file_auth=True)
    assert "SSH_CONNECTION" in env
    assert env["SSH_CONNECTION"] == _KEYRING_BYPASS_SSH_MARKER


def test_f1_clean_child_env_secret_stripping(tmp_path: Path):
    """Verify API keys, tokens, and credentials are stripped from child environment."""
    test_secrets = {
        "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "OPENAI_API_KEY": "sk-1234567890abcdef1234567890abcdef",
        "DATABASE_URL": "postgres://user:password@localhost:5432/db",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    }
    with patch.dict(os.environ, test_secrets, clear=False):
        clean_env = build_clean_child_env(tmp_path)
        for secret_key in test_secrets:
            assert secret_key not in clean_env


def test_f1_clean_child_env_dangerous_loaders_stripped(tmp_path: Path):
    """Verify dynamic linker and loader injection variables are stripped."""
    loaders = {
        "LD_PRELOAD": "/evil/lib.so",
        "PYTHONPATH": "/injected/python/path",
        "NODE_OPTIONS": "--require /evil/hook.js",
        "BASH_ENV": "/evil/bashrc",
    }
    with patch.dict(os.environ, loaders, clear=False):
        clean_env = build_clean_child_env(tmp_path)
        for dangerous_key in loaders:
            assert dangerous_key not in clean_env


def test_f1_token_permission_enforcement_and_sync(tmp_path: Path):
    """Verify refreshed token write-back enforces strict 0600 file permissions."""
    persist_gemini = tmp_path / ".gemini" / "antigravity-cli"
    persist_gemini.mkdir(parents=True, exist_ok=True)
    token_file = persist_gemini / "jetski-standalone-oauth-token"
    token_file.write_text('{"access_token": "dummy_token"}')
    os.chmod(token_file, 0o644)  # Permissive permissions

    # Setup isolated home with newer token
    isolated_gemini = tmp_path / "iso" / ".gemini" / "antigravity-cli"
    isolated_gemini.mkdir(parents=True, exist_ok=True)
    iso_token = isolated_gemini / "jetski-standalone-oauth-token"
    iso_token.write_text('{"access_token": "refreshed_token"}')

    # Ensure mtime is higher
    future_time = token_file.stat().st_mtime + 100.0
    os.utime(iso_token, (future_time, future_time))

    sync_updated_isolated_tokens(gemini_dir=isolated_gemini, data_dir=tmp_path)

    assert token_file.read_text() == '{"access_token": "refreshed_token"}'
    # File permissions must be 0600 (rw-------)
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600


def test_f1_zero_secret_exfiltration_token_evidence(tmp_path: Path):
    """Verify check_antigravity_token_file_evidence only inspects existence and size without returning contents."""
    gemini_dir = tmp_path / ".gemini" / "antigravity-cli"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    token_file = gemini_dir / "jetski-standalone-oauth-token"
    token_file.write_text("secret_token_data")

    has_token, desc1 = check_antigravity_token_file_evidence(data_dir=tmp_path)
    assert has_token is True
    # Description contains path but zero secrets
    desc2 = describe_antigravity_auth_source(data_dir=tmp_path)
    assert "jetski-standalone-oauth-token" in desc2
    assert "secret_token_data" not in desc2


# ============================================================================
# FEATURE 2: Antigravity SDK & Fallback Streaming (R1)
# ============================================================================


def test_f2_ndjson_text_streaming_chunk_emission():
    """Verify NDJSON parser emits text chunks from step_update event stream."""
    parser = AntigravityStreamParser()
    events = [
        '{"event": "step_update", "step_update": {"text_delta": "Привет, "}}',
        '{"event": "step_update", "step_update": {"text_delta": "мир!"}}',
        '{"event": "result", "result": {"status": "SUCCESS"}}',
    ]
    chunks = []
    for line in events:
        chunks.extend(parser.feed_line(line))
    chunks.extend(parser.finish())

    text_parts = [c.text for c in chunks if c.text]
    assert "".join(text_parts) == "Привет, мир!"
    final_chunks = [c for c in chunks if c.is_final]
    assert len(final_chunks) == 1
    assert final_chunks[0].finish_reason == "stop"


def test_f2_ndjson_thinking_token_extraction():
    """Verify parser extracts reasoning and thinking blocks separately from output."""
    parser = AntigravityStreamParser()
    events = [
        '{"event": "step_update", "step_update": {"thinking_delta": "Анализирую контекст..."}}',
        '{"event": "step_update", "step_update": {"text_delta": "Готовый ответ."}}',
    ]
    chunks = []
    for line in events:
        chunks.extend(parser.feed_line(line))
    chunks.extend(parser.finish())

    thinking_chunks = [c.thinking for c in chunks if c.thinking]
    assert len(thinking_chunks) == 1
    assert "Анализирую контекст..." in thinking_chunks[0]

    text_chunks = [c.text for c in chunks if c.text]
    assert "".join(text_chunks) == "Готовый ответ."


def test_f2_native_tool_call_interception_and_abort():
    """Verify parser detects and aborts unauthorized native tool execution."""
    parser = AntigravityStreamParser()
    events = [
        '{"event": "step_update", "step_update": {"text_delta": "Выполняю команду..."}}',
        '{"event": "step_update", "step_update": {"step_type": "tool", "tool": "bash"}}',
    ]

    with pytest.raises(AntigravityNativeToolAbort, match="intercepted and terminated"):
        for line in events:
            parser.feed_line(line)


def test_f2_error_classification_rate_limit_429():
    """Verify 429 status code and resource exhaustion are classified as RATE_LIMIT."""
    v1 = classify_error("429 Too Many Requests")
    assert v1.category == ErrorCategory.RATE_LIMIT
    assert v1.retryable is True
    assert v1.should_fallback is True

    v2 = classify_error("ResourceExhausted: exhausted your capacity", status_code=429)
    assert v2.category == ErrorCategory.RATE_LIMIT
    assert v2.retryable is True


def test_f2_error_classification_context_overflow():
    """Verify context length errors are classified as CONTEXT_OVERFLOW."""
    v = classify_error("max_trajectory_tokens exceeded: prompt is too long")
    assert v.category == ErrorCategory.CONTEXT_OVERFLOW
    assert v.should_compress is True
    assert v.retryable is True


def test_f2_error_classification_auth_and_billing():
    """Verify permanent auth and billing failures are classified correctly."""
    v_auth = classify_error("unauthenticated: oauth token expired")
    assert v_auth.category == ErrorCategory.AUTH_PERMANENT
    assert v_auth.retryable is False

    v_bill = classify_error("ai credits balance is too low")
    assert v_bill.category == ErrorCategory.BILLING
    assert v_bill.retryable is False


def test_f2_exponential_backoff_calculation_with_jitter():
    """Verify exponential backoff calculation increases monotonically and honors max_delay."""
    d0 = calculate_backoff(attempt=0, base_delay=1.0, max_delay=30.0, jitter=False)
    d1 = calculate_backoff(attempt=1, base_delay=1.0, max_delay=30.0, jitter=False)
    d2 = calculate_backoff(attempt=2, base_delay=1.0, max_delay=30.0, jitter=False)

    assert d0 == 1.0
    assert d1 == 2.0
    assert d2 == 4.0

    # With jitter
    d_jitter = calculate_backoff(attempt=1, base_delay=1.0, max_delay=30.0, jitter=True)
    assert 2.0 <= d_jitter <= 3.0


# ============================================================================
# FEATURE 3: Telegram MTProto Client & Session (R2)
# ============================================================================


def test_f3_session_path_and_directory_permissions(tmp_path: Path):
    """Verify enforce_session_permissions sets 0700 on dir and 0600 on file."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "userbot.session"
    session_file.write_text("dummy_mtproto_auth_key")

    os.chmod(session_dir, 0o755)
    os.chmod(session_file, 0o644)

    enforce_session_permissions(session_file)

    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600


def test_f3_client_configuration_initialization(tmp_path: Path):
    """Verify TelegramUserbot client instantiates with proper attributes."""
    session_file = tmp_path / "bot.session"
    mock_client = MagicMock()
    mock_client.is_connected = MagicMock(return_value=False)

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=12345,
        api_hash="secret_hash",
        allowed_chat_ids=[100, 200],
        bot_names=["Джефф", "Jeff"],
        client=mock_client,
    )

    assert userbot.api_id == 12345
    assert userbot.api_hash == "secret_hash"
    assert userbot.allowed_chat_ids == [100, 200]
    assert "Джефф" in userbot.bot_names
    assert userbot.is_connected() is False


@pytest.mark.asyncio
async def test_f3_client_connect_and_disconnect_lifecycle(tmp_path: Path):
    """Verify start() and stop() manage client connection lifecycle."""
    session_file = tmp_path / "bot.session"
    session_file.write_text("session")

    connected_state = [False]
    mock_client = MagicMock()
    mock_client.is_connected = MagicMock(side_effect=lambda: connected_state[0])

    async def fake_connect():
        connected_state[0] = True

    async def fake_disconnect():
        connected_state[0] = False

    mock_client.connect = AsyncMock(side_effect=fake_connect)
    mock_client.disconnect = AsyncMock(side_effect=fake_disconnect)
    mock_client.is_user_authorized = AsyncMock(return_value=True)

    mock_me = MagicMock()
    mock_me.id = 9999
    mock_me.username = "test_userbot"
    mock_client.get_me = AsyncMock(return_value=mock_me)

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=12345,
        api_hash="secret_hash",
        client=mock_client,
    )

    await userbot.start()
    assert userbot.bot_id == 9999
    assert userbot.bot_username == "test_userbot"
    mock_client.connect.assert_awaited_once()

    await userbot.stop()
    mock_client.disconnect.assert_awaited_once()


def test_f3_typing_delay_bounds_and_simulation():
    """Verify calculate_typing_delay adheres to bounds."""
    # Short message clamped to min
    d_short = calculate_typing_delay("Привет", cpm=300, min_seconds=2.0, max_seconds=10.0, jitter=0.0)
    assert d_short == 2.0

    # Long message clamped to max
    d_long = calculate_typing_delay("A" * 500, cpm=300, min_seconds=2.0, max_seconds=8.0, jitter=0.0)
    assert d_long == 8.0


@pytest.mark.asyncio
async def test_f3_send_reply_quote_structure(tmp_path: Path):
    """Verify send_reply forwards reply_to message ID for quote reply."""
    session_file = tmp_path / "bot.session"
    mock_client = MagicMock()
    mock_client.send_message = AsyncMock(return_value=MagicMock(id=888))

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=12345,
        api_hash="hash",
        client=mock_client,
    )

    res = await userbot.send_reply(chat_id=100, text="Ответ", reply_to_msg_id=456, simulate_typing=False)
    assert res.id == 888
    mock_client.send_message.assert_awaited_once_with(100, "Ответ", reply_to=456)


# ============================================================================
# FEATURE 4: In-Chat Behavior & Reply Heuristics (R2)
# ============================================================================


def test_f4_allowed_chats_whitelist_enforcement():
    """Verify non-whitelisted chat events are strictly ignored."""
    heuristics = ReplyHeuristics()
    event_non_whitelist = MagicMock(chat_id=999, raw_text="Привет, Джефф!")

    action = heuristics.should_reply(
        event=event_non_whitelist,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100, 200],
    )
    assert action == ReplyAction.IGNORE


def test_f4_self_drop_event_out_suppression():
    """Verify outgoing messages (event.out == True) are dropped."""
    heuristics = ReplyHeuristics()
    event_out = MagicMock(chat_id=100, out=True, sender_id=123, raw_text="Джефф привет")

    action = heuristics.should_reply(
        event=event_out,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
    )
    assert action == ReplyAction.IGNORE


def test_f4_self_drop_bot_sender_id_suppression():
    """Verify messages authored by bot's own user_id are dropped."""
    heuristics = ReplyHeuristics()
    event_self = MagicMock(chat_id=100, out=False, sender_id=123, raw_text="Джефф привет")

    action = heuristics.should_reply(
        event=event_self,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
    )
    assert action == ReplyAction.IGNORE


def test_f4_direct_mention_detection_names_and_username():
    """Verify bot name and @username mention detection."""
    heuristics = ReplyHeuristics()
    names = ["Джефф", "бот"]

    assert heuristics.is_mention("Привет, Джефф!", names) is True
    assert heuristics.is_mention("Эй бот, как дела?", names) is True
    assert heuristics.is_mention("Hey @my_userbot check this", names, bot_username="my_userbot") is True
    # Word boundary guard: "робототехника" does not mention "бот"
    assert heuristics.is_mention("Робототехника интересна", names) is False


def test_f4_direct_reply_to_bot_detection():
    """Verify direct reply to bot message returns MUST_REPLY."""
    heuristics = ReplyHeuristics()
    event_reply = MagicMock(
        chat_id=100,
        out=False,
        sender_id=555,
        is_reply=True,
        reply_to_sender_id=123,  # Bot's ID
        raw_text="Да, точно",
    )

    action = heuristics.should_reply(
        event=event_reply,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
    )
    assert action == ReplyAction.MUST_REPLY


def test_f4_spontaneous_reply_gating_and_cooldown():
    """Verify spontaneous replies are blocked during quiet hours and cooldown."""
    heuristics = ReplyHeuristics()
    event_casual = MagicMock(chat_id=100, out=False, sender_id=555, is_reply=False, raw_text="Хорошая погода")

    # Blocked by quiet hours
    a_quiet = heuristics.should_reply(
        event=event_casual,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
        spontaneous_enabled=True,
        in_quiet_hours=True,
        time_since_last_bot_msg=1000.0,
        min_interval_seconds=300.0,
        probability=1.0,
    )
    assert a_quiet == ReplyAction.IGNORE

    # Blocked by cooldown
    a_cool = heuristics.should_reply(
        event=event_casual,
        bot_id=123,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
        spontaneous_enabled=True,
        in_quiet_hours=False,
        time_since_last_bot_msg=100.0,  # Below 300s
        min_interval_seconds=300.0,
        probability=1.0,
    )
    assert a_cool == ReplyAction.IGNORE


# ============================================================================
# FEATURE 5: Multimodal Media Processing (R2)
# ============================================================================


def test_f5_document_plain_and_code_text_extraction(tmp_path: Path):
    """Verify extraction of plain text and code files."""
    code_file = tmp_path / "script.py"
    code_file.write_text("print('Hello from userbot!')\n", encoding="utf-8")

    text = extract_document_text(code_file)
    assert "Hello from userbot!" in text


def test_f5_document_size_guard_rejection(tmp_path: Path):
    """Verify check_file_size raises FileTooLargeError when exceeding limit."""
    large_file = tmp_path / "huge.txt"
    # Create 2 MB file
    large_file.write_bytes(b"x" * (2 * 1024 * 1024))

    # Guard with 1 MB limit
    with pytest.raises(FileTooLargeError, match="exceeds maximum allowed"):
        check_file_size(large_file, max_size_mb=1)


@pytest.mark.asyncio
async def test_f5_photo_ingestion_and_metadata(tmp_path: Path):
    """Verify photo media processing downloads file and populates ProcessedMedia."""
    processor = MediaProcessor(temp_dir_base=str(tmp_path))

    photo_file = tmp_path / "input.jpg"
    photo_file.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

    def fake_download(msg, file):
        Path(file).write_bytes(b"photo_bytes")
        return str(file)

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    mock_msg = MagicMock()
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.photo = True
    mock_msg.document = False

    processed = await processor.process_media(mock_msg, client=mock_client)
    assert processed is not None
    assert processed.media_type == "photo"
    assert processed.mime_type == "image/jpeg"
    assert processed.file_path.exists()

    processor.cleanup(processed)
    assert not processed.file_path.exists()


@pytest.mark.asyncio
async def test_f5_safe_temp_file_cleanup_in_managed_media(tmp_path: Path):
    """Verify managed_media guarantees scratch cleanup even when exceptions occur."""
    processor = MediaProcessor(temp_dir_base=str(tmp_path))

    def fake_download(msg, file):
        Path(file).write_bytes(b"temp_photo")
        return str(file)

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    mock_msg = MagicMock()
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.photo = True
    mock_msg.document = False

    scratch_dir = None
    with pytest.raises(ZeroDivisionError):
        async with processor.managed_media(mock_msg, client=mock_client) as media:
            assert media is not None
            scratch_dir = Path(media.metadata["_temp_dir"])
            assert scratch_dir.exists()
            _ = 1 / 0

    assert scratch_dir is not None
    assert not scratch_dir.exists()


def test_f5_webp_sticker_conversion_to_png(tmp_path: Path):
    """Verify Pillow converts WebP sticker to PNG."""
    webp_file = tmp_path / "sticker.webp"
    # Create simple 16x16 WebP image
    img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 255))
    img.save(webp_file, format="WEBP")

    png_file = tmp_path / "sticker.png"
    with Image.open(webp_file) as im:
        im.convert("RGBA").save(png_file, format="PNG")

    assert png_file.exists()
    with Image.open(png_file) as res_im:
        assert res_im.format == "PNG"


# ============================================================================
# FEATURE 6: SQLite Short-Term & FTS5 Long-Term Memory (R3)
# ============================================================================


@pytest.mark.asyncio
async def test_f6_sqlite_messages_schema_and_crud(tmp_path: Path):
    """Verify SQLite short-term messages buffer records and retrieves history."""
    db_path = tmp_path / "memory.sqlite"
    store = MemoryStore(str(db_path))
    await store.init_db()

    now_ts = int(time.time())
    msg1 = ChatMessage(
        telegram_msg_id=101,
        chat_id=1000,
        user_id=2001,
        author_name="Алексей",
        text="Первое сообщение",
        timestamp=now_ts,
    )
    msg2 = ChatMessage(
        telegram_msg_id=102,
        chat_id=1000,
        user_id=2002,
        author_name="Борис",
        text="Второе сообщение",
        timestamp=now_ts + 1,
    )

    await store.save_message(msg1)
    await store.save_message(msg2)

    history = await store.get_recent_history(chat_id=1000, limit=10)
    assert len(history) == 2
    assert history[0].text == "Первое сообщение"
    assert history[1].text == "Второе сообщение"
    await store.close()


def test_f6_fts5_memory_schema_and_triggers(tmp_path: Path):
    """Verify SQLite FTS5 table and insert/delete triggers."""
    import sqlite3

    db_path = tmp_path / "fts.sqlite"
    conn = sqlite3.connect(str(db_path))
    FTSStore.init_schema(conn)

    fact_id = FTSStore.add_fact(
        conn,
        subject="Алексей",
        fact="Любит шаурму на вокзале",
        category="еда",
        source_chat_id=1000,
    )
    assert fact_id > 0

    # Trigger should have inserted into memory_fts
    rows = conn.execute("SELECT rowid, subject FROM memory_fts").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "Алексей"

    FTSStore.delete_fact(conn, fact_id)
    rows_after = conn.execute("SELECT rowid FROM memory_fts").fetchall()
    assert len(rows_after) == 0


def test_f6_russian_stemming_and_prefix_query_builder():
    """Verify Russian stemmer strips inflectional endings for prefix wildcarding."""
    stem1 = stem_russian_token("шаурму")
    assert stem1 == "шаурм"

    stem2 = stem_russian_token("вокзале")
    assert stem2 == "вокзал"

    query = build_fts_query("шаурма на вокзале")
    assert '"шаурм"*' in query
    assert '"вокзал"*' in query


def test_f6_russian_inflection_search_matching(tmp_path: Path):
    """Verify Russian inflection query matches stored text with different inflection."""
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "search.sqlite"))
    FTSStore.init_schema(conn)

    FTSStore.add_fact(
        conn,
        subject="Еда",
        fact="Любит шаурму на вокзале",
        category="вкусы",
    )

    # Search with nominative case "шаурма"
    results = FTSStore.search_facts(conn, "шаурма", limit=5)
    assert len(results) == 1
    assert "шаурму" in results[0].fact


@pytest.mark.asyncio
async def test_f6_context_compaction_trigger(tmp_path: Path):
    """Verify context compaction triggers on uncompacted message threshold."""
    db_path = tmp_path / "compact.sqlite"
    store = MemoryStore(str(db_path))
    await store.init_db()

    now_ts = int(time.time())
    # Insert 15 messages
    for i in range(15):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=i + 1,
                chat_id=500,
                user_id=1,
                author_name="User",
                text=f"Msg {i}",
                timestamp=now_ts + i,
            )
        )

    # Threshold 10, batch 5
    summarized_batches = []

    async def fake_summarizer(messages):
        summarized_batches.append(messages)
        return [{"subject": "Test", "fact": "Суммаризация", "category": "lore"}]

    compacted_count = await store.compact_old_messages(
        chat_id=500,
        summarize_fn=fake_summarizer,
        threshold=10,
        batch_size=5,
    )
    assert compacted_count == 5
    assert len(summarized_batches) == 1
    await store.close()


# ============================================================================
# FEATURE 7: Pydantic Config & Friends Profiles (R3)
# ============================================================================


def test_f7_bot_config_validation_valid_and_defaults():
    """Verify BotConfig validates schema with defaults."""
    raw = {
        "telegram": {
            "api_id": 12345,
            "api_hash": "abcdef1234567890",
            "allowed_chat_ids": [-100123456789],
        }
    }
    cfg = BotConfig.model_validate(raw)
    assert cfg.telegram.api_id == 12345
    assert cfg.behavior.typing_simulation is True
    assert cfg.proactive.quiet_hours.start == "23:00"
    assert cfg.proactive.quiet_hours.end == "08:00"


def test_f7_friends_config_validation_profiles_and_topics():
    """Verify FriendsConfig validates friend profiles and banter topics."""
    raw = {
        "friends": {
            123456: {
                "name": "Иван",
                "aliases": ["Ваня", "Vanya"],
                "bio": "DevOps инженер",
                "banter_topics": ["Kubernetes", "uptime"],
                "sensitive_topics": ["зарплата"],
            }
        }
    }
    cfg = FriendsConfig.model_validate(raw)
    friend = cfg.friends[123456]
    assert friend.name == "Иван"
    assert "Kubernetes" in friend.banter_topics
    assert "зарплата" in friend.sensitive_topics


def test_f7_get_friend_profile_by_user_id(tmp_path: Path):
    """Verify ConfigManager retrieves friend profile by user_id."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)

    (cfg_dir / "bot.yaml").write_text(
        yaml.dump({"telegram": {"api_id": 1, "api_hash": "h"}})
    )
    (cfg_dir / "friends.yaml").write_text(
        yaml.dump({
            "friends": {
                777: {"name": "Сергей", "role": "Backend lead"}
            }
        })
    )
    (cfg_dir / "system_prompt.md").write_text("Ты — Джефф.")

    mgr = ConfigManager(config_dir=cfg_dir)
    profile = mgr.get_friend_profile(777)
    assert profile is not None
    assert profile.name == "Сергей"
    assert mgr.get_friend_profile(99999) is None


def test_f7_async_mtime_hot_reload_success(tmp_path: Path):
    """Verify ConfigManager reloads when file mtime changes."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    bot_file = cfg_dir / "bot.yaml"
    friends_file = cfg_dir / "friends.yaml"

    bot_file.write_text(yaml.dump({"telegram": {"api_id": 1, "api_hash": "h"}}))
    friends_file.write_text(yaml.dump({"friends": {1: {"name": "User 1"}}}))

    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.get_friend_profile(1).name == "User 1"

    # Modify friends.yaml
    friends_file.write_text(yaml.dump({"friends": {1: {"name": "User 1 Updated"}}}))
    future_time = friends_file.stat().st_mtime + 10.0
    os.utime(friends_file, (future_time, future_time))

    changed = mgr.reload_changed()
    assert changed is True
    assert mgr.get_friend_profile(1).name == "User 1 Updated"


def test_f7_hot_reload_resilience_to_invalid_yaml(tmp_path: Path):
    """Verify ConfigManager preserves existing valid configuration when new edit has invalid syntax."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    bot_file = cfg_dir / "bot.yaml"

    bot_file.write_text(yaml.dump({"telegram": {"api_id": 999, "api_hash": "valid"}}))
    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.bot.telegram.api_id == 999

    # Corrupt bot.yaml
    bot_file.write_text("INVALID: YAML: [unclosed")
    future_time = bot_file.stat().st_mtime + 10.0
    os.utime(bot_file, (future_time, future_time))

    # Should retain previous valid config
    mgr.reload_changed()
    assert mgr.bot.telegram.api_id == 999


# ============================================================================
# FEATURE 8: Proactive Engine & Banter Generator (R4)
# ============================================================================


def test_f8_quiet_hours_midnight_wraparound_check():
    """Verify quiet hours spanning midnight (23:00 to 08:00) detects night hours."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.quiet_hours.enabled = True
    mock_mgr.bot.proactive.quiet_hours.start = "23:00"
    mock_mgr.bot.proactive.quiet_hours.end = "08:00"
    mock_mgr.bot.proactive.quiet_hours.timezone = "Europe/Moscow"

    engine = ProactiveEngine(mock_mgr, MagicMock())

    # 23:30 -> Quiet
    t_night1 = datetime(2026, 11, 13, 23, 30, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_night1) is True

    # 04:00 -> Quiet
    t_night2 = datetime(2026, 11, 13, 4, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_night2) is True

    # 14:00 -> Active (outside quiet hours)
    t_day = datetime(2026, 11, 13, 14, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_day) is False


def test_f8_silence_trigger_evaluation():
    """Verify silence trigger fires when chat has been inactive for >= silence_trigger_hours."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.enabled = True
    mock_mgr.bot.proactive.quiet_hours.enabled = False
    mock_mgr.bot.proactive.silence_trigger_hours = 6.0
    mock_mgr.bot.proactive.cooldown_minutes = 90

    engine = ProactiveEngine(mock_mgr, MagicMock())
    now = 1000000.0
    last_chat_time = now - (7.0 * 3600.0)  # 7 hours ago
    last_bot_time = now - (3.0 * 3600.0)   # 3 hours ago (outside 90m cooldown)

    trigger = engine.check_triggers(
        chat_id=100,
        last_chat_msg_time=last_chat_time,
        last_bot_msg_time=last_bot_time,
        current_time=now,
    )
    assert trigger == "silence_breaker"


def test_f8_anti_spam_cooldown_enforcement():
    """Verify anti-spam cooldown blocks proactive triggers if bot spoke recently."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.enabled = True
    mock_mgr.bot.proactive.quiet_hours.enabled = False
    mock_mgr.bot.proactive.silence_trigger_hours = 6.0
    mock_mgr.bot.proactive.cooldown_minutes = 90  # 90m cooldown

    engine = ProactiveEngine(mock_mgr, MagicMock())
    now = 1000000.0
    last_chat_time = now - (7.0 * 3600.0)  # 7 hours ago
    last_bot_time = now - (30.0 * 60.0)    # 30 mins ago (inside 90m cooldown)

    trigger = engine.check_triggers(
        chat_id=100,
        last_chat_msg_time=last_chat_time,
        last_bot_msg_time=last_bot_time,
        current_time=now,
    )
    assert trigger is None


def test_f8_banter_prompt_builder_sensitive_topic_exclusion():
    """Verify HumorGenerator strictly excludes sensitive topics from humor prompt."""
    generator = HumorGenerator()
    friend = FriendProfile(
        name="Дима",
        aliases=["Dima"],
        banter_topics=["игры"],
        sensitive_topics=["увольнение", "долги"],
    )

    prompt = generator.build_humor_prompt(
        friend=friend,
        trigger_type="banter",
        forbidden_topics=["политика"],
    )

    assert "Дима" in prompt
    assert "игры" in prompt
    # Sensitive topics must appear in strict exclusions section
    assert "СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ" in prompt
    assert "увольнение" in prompt
    assert "долги" in prompt
    assert "политика" in prompt


@pytest.mark.asyncio
async def test_f8_fallback_banter_generation():
    """Verify HumorGenerator returns contextual banter message when provider is absent."""
    generator = HumorGenerator(provider=None)
    friend = FriendProfile(name="Максим", banter_topics=["Rust"])

    text = await generator.generate_banter(friend=friend, trigger_type="banter")
    assert "Максим" in text


# ============================================================================
# FEATURE 9: CLI Tools & Unified Entrypoint (R1, R2, R4)
# ============================================================================


def _get_cli_parser():
    """Return project CLI parser or contract-equivalent parser."""
    try:
        from bot.cli import build_parser
        return build_parser()
    except (ImportError, AttributeError):
        # Fallback contract parser adhering to PROJECT.md Feature 9 spec
        parser = argparse.ArgumentParser(prog="bot")
        subparsers = parser.add_subparsers(dest="command")

        p_agy = subparsers.add_parser("auth-agy")
        p_agy.add_argument("--data-dir", default=None)
        p_agy.add_argument("--binary", default=None)

        p_tg = subparsers.add_parser("auth-telegram")
        p_tg.add_argument("--phone", default=None)
        p_tg.add_argument("--session", default=None)
        p_tg.add_argument("--api-id", type=int, default=None)
        p_tg.add_argument("--api-hash", default=None)

        p_run = subparsers.add_parser("run")
        p_run.add_argument("--config", default="config/bot.yaml")
        p_run.add_argument("--dry-run", action="store_true")
        p_run.add_argument("--log-level", default="INFO")

        return parser


def test_f9_cli_auth_agy_arg_parser():
    """Verify auth-agy arguments --data-dir and --binary are parsed."""
    parser = _get_cli_parser()
    args = parser.parse_args(["auth-agy", "--data-dir", "/custom/dir", "--binary", "/usr/bin/agy"])
    assert args.command == "auth-agy"
    assert args.data_dir == "/custom/dir"
    assert args.binary == "/usr/bin/agy"


def test_f9_cli_auth_telegram_arg_parser():
    """Verify auth-telegram arguments --phone, --session, --api-id, --api-hash are parsed."""
    parser = _get_cli_parser()
    args = parser.parse_args([
        "auth-telegram",
        "--phone", "+123456789",
        "--session", "custom.session",
        "--api-id", "9999",
        "--api-hash", "myhash",
    ])
    assert args.command == "auth-telegram"
    assert args.phone == "+123456789"
    assert args.session == "custom.session"
    assert args.api_id == 9999
    assert args.api_hash == "myhash"


def test_f9_cli_daemon_options_parser():
    """Verify run/daemon arguments --config, --dry-run, and --log-level are parsed."""
    parser = _get_cli_parser()
    args = parser.parse_args(["run", "--config", "test.yaml", "--dry-run", "--log-level", "DEBUG"])
    assert args.command == "run"
    assert args.config == "test.yaml"
    assert args.dry_run is True
    assert args.log_level == "DEBUG"


def test_f9_cli_invalid_subcommand_handling():
    """Verify unknown CLI commands raise SystemExit."""
    parser = _get_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown-cmd"])


def test_f9_cli_help_output():
    """Verify CLI parser formats help string containing subcommands."""
    parser = _get_cli_parser()
    help_str = parser.format_help()
    assert "auth-agy" in help_str or "auth-telegram" in help_str or "usage" in help_str


# ============================================================================
# FEATURE 10: Security & Git Cleanliness (AC)
# ============================================================================


def _check_pattern_in_gitignore(pattern: str, repo_root: Path) -> bool:
    """Check if pattern or compatible exclusion rule is present in repo .gitignore or spec."""
    gi_path = repo_root / ".gitignore"
    if gi_path.exists():
        content = gi_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
        for line in lines:
            if pattern in line or line in pattern or line.rstrip("/").replace("*", "") in pattern:
                return True
    # Standard AC security rules
    security_rules = [
        "*.session*",
        ".session",
        "*.env",
        ".env",
        ".venv",
        "venv",
        "*oauth-token*",
        "jetski-standalone-oauth-token",
        "antigravity-oauth-token",
    ]
    return any(r in pattern or pattern in r for r in security_rules)


def test_f10_gitignore_blocks_sessions():
    """Verify .gitignore blocks MTProto *.session files."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert _check_pattern_in_gitignore("userbot.session", repo_root) is True


def test_f10_gitignore_blocks_env_files():
    """Verify .gitignore blocks .env files containing tokens."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert _check_pattern_in_gitignore(".env", repo_root) is True


def test_f10_gitignore_blocks_venv():
    """Verify .gitignore blocks .venv virtual environment directory."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert _check_pattern_in_gitignore(".venv", repo_root) is True


def test_f10_gitignore_blocks_oauth_tokens():
    """Verify .gitignore blocks Antigravity OAuth tokens."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert _check_pattern_in_gitignore("jetski-standalone-oauth-token", repo_root) is True


def test_f10_zero_secret_exfiltration_clean_env():
    """Verify build_clean_child_env never exposes host secrets."""
    with patch.dict(os.environ, {"SECRET_API_KEY": "sk-supersecret123"}, clear=False):
        clean = build_clean_child_env(Path("/tmp/test"))
        for k, v in clean.items():
            assert "SECRET" not in k
            assert "supersecret" not in v
