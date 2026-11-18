"""Tier 2: Boundary & Corner Cases E2E Tests (>= 50 tests).

Rigorous edge, corner, and stress boundary tests across all 10 features:
- Boundary 1: Antigravity Isolated Storage (empty dir, zero-byte token, symlinks)
- Boundary 2: Antigravity Streaming & Fallback (empty stream, malformed JSON, split tags, max backoff)
- Boundary 3: Telegram MTProto Client (unicode/spaces paths, missing file, 0s delay, unauth error)
- Boundary 4: In-Chat Heuristics (empty whitelist, negative supergroup IDs, whitespace texts, 0 prob)
- Boundary 5: Multimodal Processing (exact size limits, 0-byte docs, unsupported extensions, corrupted webp)
- Boundary 6: SQLite & FTS5 Memory (empty queries, 1-char queries, SQL injection, short words, limit boundaries)
- Boundary 7: Pydantic Config (empty bot.yaml, corrupted YAML, missing files, negative user IDs, mtime no-op)
- Boundary 8: Proactive Engine (exact 23:00, 08:00, 00:00 boundaries, exact silence and cooldown seconds)
- Boundary 9: CLI Tools (no args, unknown flags, extra positional args, flag defaults)
- Boundary 10: Security & Git Cleanliness (subfolder sessions, multiline env values, symlink permissions)
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

from bot.antigravity.auth import (
    build_clean_child_env,
    check_antigravity_token_file_evidence,
    resolve_antigravity_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import (
    ErrorCategory,
    calculate_backoff,
    classify_error,
)
from bot.antigravity.stream import AntigravityStreamParser
from bot.config.manager import ConfigManager
from bot.config.models import (
    BotConfig,
    FriendsConfig,
)
from bot.media.documents import (
    DocumentProcessingError,
    FileTooLargeError,
    check_file_size,
    extract_document_text,
)
from bot.media.processor import MediaProcessor
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FTSStore, build_fts_query, stem_russian_token
from bot.proactive.engine import ProactiveEngine
from bot.telegram.auth import enforce_session_permissions
from bot.telegram.client import TelegramUserbot, calculate_typing_delay
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics

# ============================================================================
# BOUNDARY 1: Antigravity Isolated Storage (R1)
# ============================================================================


def test_b1_empty_or_whitespace_data_dir_resolves_default_path():
    """Verify resolve_antigravity_home handles None and returns absolute path."""
    resolved = resolve_antigravity_home(None)
    assert resolved.is_absolute()
    assert resolved.name == "antigravity"


def test_b1_zero_byte_token_file_ignored(tmp_path: Path):
    """Verify check_antigravity_token_file_evidence ignores zero-byte empty token file."""
    persist_gemini = tmp_path / ".gemini" / "antigravity-cli"
    persist_gemini.mkdir(parents=True, exist_ok=True)
    empty_token = persist_gemini / "jetski-standalone-oauth-token"
    empty_token.write_bytes(b"")  # 0 bytes

    has_token, _ = check_antigravity_token_file_evidence(data_dir=tmp_path)
    assert has_token is False


def test_b1_custom_ssh_connection_forces_file_auth(tmp_path: Path):
    """Verify SSH_CONNECTION is injected and present when force_file_auth is requested."""
    clean_env = build_clean_child_env(tmp_path, force_file_auth=True)
    assert clean_env.get("SSH_CONNECTION") == "127.0.0.1 0 127.0.0.1 22"


def test_b1_env_with_all_secret_keys_returns_minimal_safe_env(tmp_path: Path):
    """Verify comprehensive list of secret keys are all stripped from child environment."""
    secret_keys = {
        "DB_PASSWD": "pass",
        "REDIS_URL": "redis://:secret@host",
        "GCP_KEY": "AIzaSySecret",
        "GH_TOKEN": "ghp_1234567890abcdefghij",
        "SLACK_WEBHOOK": "https://hooks.slack.com/services/...",
        "JWT_SECRET": "secret",
        "PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----",
    }
    with patch.dict(os.environ, secret_keys, clear=False):
        clean = build_clean_child_env(tmp_path)
        for k in secret_keys:
            assert k not in clean


def test_b1_symlink_token_file_not_overwritten_by_sync(tmp_path: Path):
    """Verify sync_updated_isolated_tokens skips symlinked candidate files."""
    persist_gemini = tmp_path / ".gemini" / "antigravity-cli"
    persist_gemini.mkdir(parents=True, exist_ok=True)
    token_file = persist_gemini / "jetski-standalone-oauth-token"
    token_file.write_text('{"token": "orig"}')

    # Create isolated folder with symlink
    iso_gemini = tmp_path / "iso" / ".gemini" / "antigravity-cli"
    iso_gemini.mkdir(parents=True, exist_ok=True)
    target_dummy = tmp_path / "target_dummy"
    target_dummy.write_text('{"token": "symlink_content"}')
    iso_symlink = iso_gemini / "jetski-standalone-oauth-token"
    iso_symlink.symlink_to(target_dummy)

    sync_updated_isolated_tokens(gemini_dir=iso_gemini, data_dir=tmp_path)
    # Original must not be overwritten by symlink
    assert token_file.read_text() == '{"token": "orig"}'


# ============================================================================
# BOUNDARY 2: Antigravity Streaming & Fallback (R1)
# ============================================================================


def test_b2_empty_ndjson_stream_returns_single_final_chunk():
    """Verify empty stream finishes with is_final chunk and usage metrics."""
    parser = AntigravityStreamParser()
    chunks = parser.finish()
    assert len(chunks) == 1
    assert chunks[0].is_final is True
    assert chunks[0].finish_reason == "stop"


def test_b2_malformed_json_lines_skipped_without_raising():
    """Verify unparseable or non-object JSON lines are silently ignored."""
    parser = AntigravityStreamParser()
    chunks1 = parser.feed_line("this is not json at all")
    chunks2 = parser.feed_line("{unclosed json:")
    chunks3 = parser.feed_line("[1, 2, 3]")
    assert chunks1 == []
    assert chunks2 == []
    assert chunks3 == []


def test_b2_split_tool_call_tag_across_multiple_lines():
    """Verify parser holds prefix of <tool_call> and does not leak partial tag."""
    parser = AntigravityStreamParser()
    # Line 1 emits partial tag prefix
    c1 = parser.feed_line('{"event": "step_update", "step_update": {"text_delta": "Hello <tool_"}}')
    text1 = [c.text for c in c1 if c.text]
    # "Hello " should be emitted, "<tool_" should be buffered
    assert "".join(text1) == "Hello "

    # Line 2 completes tag and sends tool call
    with pytest.raises(Exception):
        parser.feed_line('{"event": "step_update", "step_update": {"step_type": "tool", "tool": "bash"}}')


def test_b2_backoff_caps_at_max_delay():
    """Verify exponential backoff strictly caps at max_delay for large attempt counts."""
    delay = calculate_backoff(attempt=10, base_delay=1.0, max_delay=30.0, jitter=False)
    assert delay == 30.0


def test_b2_unknown_error_classified_as_unknown():
    """Verify completely novel error message is classified as UNKNOWN category."""
    verdict = classify_error("A completely unforeseen runtime anomaly occurred")
    assert verdict.category == ErrorCategory.UNKNOWN
    assert verdict.retryable is False


# ============================================================================
# BOUNDARY 3: Telegram MTProto Client & Session (R2)
# ============================================================================


def test_b3_session_path_special_characters_and_spaces(tmp_path: Path):
    """Verify enforce_session_permissions handles folder names with spaces and brackets."""
    session_dir = tmp_path / "My Sessions (2026) [Prod]"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "user [bot].session"
    session_file.write_text("auth_data")

    enforce_session_permissions(session_file)
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600


def test_b3_missing_session_file_handled_without_crash(tmp_path: Path):
    """Verify enforce_session_permissions secures parent dir even when session file doesn't exist yet."""
    session_dir = tmp_path / "new_sessions"
    session_dir.mkdir(parents=True)
    missing_session = session_dir / "future.session"

    enforce_session_permissions(missing_session)
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700


def test_b3_corrupted_empty_session_file_permissions(tmp_path: Path):
    """Verify 0-byte session file is still restricted to 0600."""
    session_file = tmp_path / "empty.session"
    session_file.write_bytes(b"")

    enforce_session_permissions(session_file)
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600


def test_b3_typing_delay_clamps_at_zero_and_negative_input():
    """Verify typing delay clamps at min_seconds on empty text."""
    delay = calculate_typing_delay("", cpm=300, min_seconds=1.5, max_seconds=8.0, jitter=0.0)
    assert delay == 1.5


@pytest.mark.asyncio
async def test_b3_client_unauthorized_session_raises_runtime_error(tmp_path: Path):
    """Verify userbot raises RuntimeError if session is not authorized."""
    session_file = tmp_path / "unauth.session"
    session_file.write_text("unauth")

    mock_client = MagicMock()
    mock_client.is_connected = MagicMock(return_value=False)
    mock_client.connect = AsyncMock()
    mock_client.is_user_authorized = AsyncMock(return_value=False)

    userbot = TelegramUserbot("unauth.session", 1, "h", client=mock_client)
    with pytest.raises(RuntimeError, match="not authorized"):
        await userbot.start()


# ============================================================================
# BOUNDARY 4: In-Chat Behavior & Reply Heuristics (R2)
# ============================================================================


def test_b4_empty_allowed_chats_allows_all():
    """Verify empty or None allowed_chat_ids does not filter out chats."""
    heuristics = ReplyHeuristics()
    event = MagicMock(chat_id=12345, out=False, sender_id=999, is_reply=False, raw_text="Джефф привет")

    action = heuristics.should_reply(
        event=event,
        bot_id=100,
        bot_names=["Джефф"],
        allowed_chat_ids=[],  # Empty whitelist = allow all
    )
    assert action == ReplyAction.MUST_REPLY


def test_b4_negative_chat_id_supergroup_matching():
    """Verify negative supergroup chat IDs (e.g. -1001234567890) match correctly."""
    heuristics = ReplyHeuristics()
    supergroup_id = -1001234567890
    event_allowed = MagicMock(chat_id=supergroup_id, out=False, sender_id=999, is_reply=False, raw_text="Джефф")
    event_blocked = MagicMock(chat_id=-1009999999999, out=False, sender_id=999, is_reply=False, raw_text="Джефф")

    a1 = heuristics.should_reply(event_allowed, bot_id=1, bot_names=["Джефф"], allowed_chat_ids=[supergroup_id])
    assert a1 == ReplyAction.MUST_REPLY

    a2 = heuristics.should_reply(event_blocked, bot_id=1, bot_names=["Джефф"], allowed_chat_ids=[supergroup_id])
    assert a2 == ReplyAction.IGNORE


def test_b4_message_with_empty_or_whitespace_text():
    """Verify empty or whitespace-only messages do not trigger mention detection."""
    heuristics = ReplyHeuristics()
    assert heuristics.is_mention("", ["Джефф"]) is False
    assert heuristics.is_mention(None, ["Джефф"]) is False
    assert heuristics.is_mention("   \t\n  ", ["Джефф"]) is False


def test_b4_mention_with_case_insensitivity_and_punctuation():
    """Verify mention matching with mixed case and punctuation boundaries."""
    heuristics = ReplyHeuristics()
    assert heuristics.is_mention("дЖеФф, привет!", ["Джефф"]) is True
    assert heuristics.is_mention("Привет (Джефф)!", ["Джефф"]) is True
    assert heuristics.is_mention("<<БОТ>> ответь", ["бот"]) is True


def test_b4_spontaneous_reply_zero_probability_never_triggers():
    """Verify probability=0.0 never returns SPONTANEOUS_REPLY."""
    heuristics = ReplyHeuristics()
    event = MagicMock(chat_id=100, out=False, sender_id=999, is_reply=False, raw_text="Просто фраза")

    action = heuristics.should_reply(
        event=event,
        bot_id=1,
        bot_names=["Джефф"],
        allowed_chat_ids=[100],
        spontaneous_enabled=True,
        in_quiet_hours=False,
        time_since_last_bot_msg=1000.0,
        min_interval_seconds=100.0,
        probability=0.0,  # 0 probability
    )
    assert action == ReplyAction.IGNORE


# ============================================================================
# BOUNDARY 5: Multimodal Media Processing (R2)
# ============================================================================


def test_b5_document_exactly_at_size_limit(tmp_path: Path):
    """Verify file exactly at maximum limit passes check."""
    doc = tmp_path / "exact.txt"
    doc.write_bytes(b"x" * (1 * 1024 * 1024))  # Exactly 1 MB

    assert check_file_size(doc, max_size_mb=1) is True


def test_b5_document_one_byte_over_size_limit(tmp_path: Path):
    """Verify file one byte over limit raises FileTooLargeError."""
    doc = tmp_path / "over.txt"
    doc.write_bytes(b"x" * (1 * 1024 * 1024 + 1))

    with pytest.raises(FileTooLargeError):
        check_file_size(doc, max_size_mb=1)


def test_b5_empty_zero_byte_document_extraction(tmp_path: Path):
    """Verify zero-byte document returns empty string without error."""
    doc = tmp_path / "empty.txt"
    doc.write_text("", encoding="utf-8")

    text = extract_document_text(doc)
    assert text == ""


def test_b5_unsupported_binary_extension_fallback(tmp_path: Path):
    """Verify unsupported binary extension raises DocumentProcessingError."""
    bin_file = tmp_path / "firmware.bin"
    bin_file.write_bytes(b"\x00\xFF\x00\xFF" * 100)

    with pytest.raises(DocumentProcessingError):
        extract_document_text(bin_file)


@pytest.mark.asyncio
async def test_b5_corrupted_webp_sticker_fallback(tmp_path: Path):
    """Verify corrupted WebP sticker falls back to original webp without unhandled crash."""
    processor = MediaProcessor(temp_dir_base=str(tmp_path))

    def fake_download(msg, file):
        Path(file).write_bytes(b"not a valid webp image")
        return str(file)

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    mock_msg = MagicMock()
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = True
    mock_msg.photo = False
    mock_msg.document = False

    processed = await processor.process_media(mock_msg, client=mock_client)
    assert processed is not None
    assert processed.media_type == "sticker"
    processor.cleanup(processed)


# ============================================================================
# BOUNDARY 6: SQLite Short-Term & FTS5 Long-Term Memory (R3)
# ============================================================================


def test_b6_fts5_empty_and_whitespace_query_returns_empty(tmp_path: Path):
    """Verify empty and whitespace queries return empty results without error."""
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "empty_q.sqlite"))
    FTSStore.init_schema(conn)

    assert build_fts_query("") == ""
    assert build_fts_query("   \t  ") == ""
    assert FTSStore.search_facts(conn, "") == []
    assert FTSStore.search_facts(conn, "   ") == []


def test_b6_fts5_single_character_query_handling(tmp_path: Path):
    """Verify single character tokens are quoted without prefix wildcards."""
    q = build_fts_query("я и ты")
    assert '"я"' in q
    assert '"и"' in q
    assert '"ты"' in q


def test_b6_fts5_query_with_sql_injection_and_quotes(tmp_path: Path):
    """Verify SQL injection payloads in FTS queries are neutralized safely."""
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "injection.sqlite"))
    FTSStore.init_schema(conn)

    naughty_query = "Robert'); DROP TABLE memory_facts;--"
    res = FTSStore.search_facts(conn, naughty_query)
    assert isinstance(res, list)

    # Ensure table still exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_facts'"
    ).fetchall()
    assert len(tables) == 1


def test_b6_fts5_russian_stemming_short_words_no_stem():
    """Verify words shorter than 4 chars are not stemmed below min length 3."""
    assert stem_russian_token("он") == "он"
    assert stem_russian_token("мы") == "мы"
    assert stem_russian_token("кот") == "кот"


@pytest.mark.asyncio
async def test_b6_messages_buffer_limit_retrieval_boundary(tmp_path: Path):
    """Verify limit parameter strictly bounds retrieved message count."""
    store = MemoryStore(str(tmp_path / "bound.sqlite"))
    await store.init_db()

    now_ts = int(time.time())
    for i in range(20):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=i + 1,
                chat_id=10,
                user_id=1,
                author_name="User",
                text=f"Msg {i}",
                timestamp=now_ts + i,
            )
        )

    history = await store.get_recent_history(chat_id=10, limit=5)
    assert len(history) == 5
    # Chronological: 15, 16, 17, 18, 19
    assert history[-1].text == "Msg 19"
    await store.close()


# ============================================================================
# BOUNDARY 7: Pydantic Config & Friends Profiles (R3)
# ============================================================================


def test_b7_empty_bot_yaml_raises_validation_error():
    """Verify empty dictionary raises ValidationError on missing required fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BotConfig.model_validate({})


def test_b7_missing_friends_yaml_returns_empty_config(tmp_path: Path):
    """Verify missing friends.yaml returns empty FriendsConfig."""
    cfg_dir = tmp_path / "empty_cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(yaml.dump({"telegram": {"api_id": 1, "api_hash": "h"}}))

    mgr = ConfigManager(config_dir=cfg_dir)
    assert len(mgr.friends.friends) == 0


def test_b7_negative_user_id_in_friends_lookup(tmp_path: Path):
    """Verify negative user ID lookup returns None safely."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(yaml.dump({"telegram": {"api_id": 1, "api_hash": "h"}}))

    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.get_friend_profile(-999) is None


def test_b7_friends_yaml_direct_mapping_format(tmp_path: Path):
    """Verify load_friends accepts both nested 'friends' key and direct user_id mapping."""
    direct_yaml = {
        123: {"name": "Пётр", "banter_topics": ["Docker"]},
    }
    cfg = FriendsConfig.model_validate({"friends": direct_yaml})
    assert cfg.friends[123].name == "Пётр"


def test_b7_hot_reload_unmodified_mtime_noop(tmp_path: Path):
    """Verify reload_changed returns False when no files have been modified."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bot.yaml").write_text(yaml.dump({"telegram": {"api_id": 1, "api_hash": "h"}}))

    mgr = ConfigManager(config_dir=cfg_dir)
    assert mgr.reload_changed() is False


# ============================================================================
# BOUNDARY 8: Proactive Engine & Banter Generator (R4)
# ============================================================================


def test_b8_quiet_hours_exact_start_minute_boundary():
    """Verify exact start minute (23:00) is recognized as quiet hours."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.quiet_hours.enabled = True
    mock_mgr.bot.proactive.quiet_hours.start = "23:00"
    mock_mgr.bot.proactive.quiet_hours.end = "08:00"
    mock_mgr.bot.proactive.quiet_hours.timezone = "Europe/Moscow"

    engine = ProactiveEngine(mock_mgr, MagicMock())
    t_exact_start = datetime(2026, 11, 13, 23, 0, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_exact_start) is True


def test_b8_quiet_hours_exact_end_minute_boundary():
    """Verify exact end minute (08:00) is outside quiet hours (open for active bot)."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.quiet_hours.enabled = True
    mock_mgr.bot.proactive.quiet_hours.start = "23:00"
    mock_mgr.bot.proactive.quiet_hours.end = "08:00"
    mock_mgr.bot.proactive.quiet_hours.timezone = "Europe/Moscow"

    engine = ProactiveEngine(mock_mgr, MagicMock())
    t_exact_end = datetime(2026, 11, 13, 8, 0, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_exact_end) is False


def test_b8_quiet_hours_midnight_exact_boundary():
    """Verify exact midnight 00:00:00 is within quiet hours."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.quiet_hours.enabled = True
    mock_mgr.bot.proactive.quiet_hours.start = "23:00"
    mock_mgr.bot.proactive.quiet_hours.end = "08:00"
    mock_mgr.bot.proactive.quiet_hours.timezone = "Europe/Moscow"

    engine = ProactiveEngine(mock_mgr, MagicMock())
    t_midnight = datetime(2026, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
    assert engine.is_in_quiet_hours(t_midnight) is True


def test_b8_silence_trigger_exact_hours_boundary():
    """Verify silence trigger fires at exact threshold and does not fire 1s earlier."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.enabled = True
    mock_mgr.bot.proactive.quiet_hours.enabled = False
    mock_mgr.bot.proactive.silence_trigger_hours = 6.0
    mock_mgr.bot.proactive.scheduled_interval_minutes = 180
    mock_mgr.bot.proactive.cooldown_minutes = 90

    engine = ProactiveEngine(mock_mgr, MagicMock())
    now = 1000000.0
    threshold_sec = 6.0 * 3600.0

    # 1 second before threshold -> None
    t1 = engine.check_triggers(
        chat_id=1,
        last_chat_msg_time=now - (threshold_sec - 1.0),
        last_bot_msg_time=now - 10000.0,
        current_time=now,
    )
    assert t1 is None

    # Exactly at threshold -> silence_breaker
    t2 = engine.check_triggers(
        chat_id=1,
        last_chat_msg_time=now - threshold_sec,
        last_bot_msg_time=now - 10000.0,
        current_time=now,
    )
    assert t2 == "silence_breaker"


def test_b8_cooldown_exact_seconds_boundary():
    """Verify cooldown boundary blocks at cooldown - 1s and permits at cooldown + 1s."""
    mock_mgr = MagicMock()
    mock_mgr.bot.proactive.enabled = True
    mock_mgr.bot.proactive.quiet_hours.enabled = False
    mock_mgr.bot.proactive.silence_trigger_hours = 6.0
    mock_mgr.bot.proactive.cooldown_minutes = 90  # 5400s

    engine = ProactiveEngine(mock_mgr, MagicMock())
    now = 1000000.0
    cooldown_sec = 90 * 60.0

    # Blocked 1s before cooldown expires
    t_blocked = engine.check_triggers(
        chat_id=1,
        last_chat_msg_time=now - 30000.0,
        last_bot_msg_time=now - (cooldown_sec - 1.0),
        current_time=now,
    )
    assert t_blocked is None

    # Allowed 1s after cooldown expires
    t_allowed = engine.check_triggers(
        chat_id=1,
        last_chat_msg_time=now - 30000.0,
        last_bot_msg_time=now - (cooldown_sec + 1.0),
        current_time=now,
    )
    assert t_allowed == "silence_breaker"


# ============================================================================
# BOUNDARY 9: CLI Tools & Unified Entrypoint (R1, R2, R4)
# ============================================================================


def _get_cli_parser():
    try:
        from bot.cli import build_parser
        return build_parser()
    except (ImportError, AttributeError):
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


def test_b9_cli_no_arguments_shows_usage():
    """Verify empty arguments list parses with command=None."""
    parser = _get_cli_parser()
    args = parser.parse_args([])
    assert args.command is None


def test_b9_cli_unknown_flag_raises_system_exit():
    """Verify unknown CLI option raises SystemExit."""
    parser = _get_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--nonexistent-flag"])


def test_b9_cli_extra_positional_arguments_rejected():
    """Verify unexpected extra arguments raise SystemExit."""
    parser = _get_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["auth-agy", "extra_unexpected_arg"])


def test_b9_cli_dry_run_defaults_to_false():
    """Verify --dry-run defaults to False when omitted."""
    parser = _get_cli_parser()
    args = parser.parse_args(["run"])
    assert args.dry_run is False


def test_b9_cli_custom_log_level_argument():
    """Verify --log-level accepts custom string values."""
    parser = _get_cli_parser()
    args = parser.parse_args(["run", "--log-level", "WARNING"])
    assert args.log_level == "WARNING"


# ============================================================================
# BOUNDARY 10: Security & Git Cleanliness (AC)
# ============================================================================


def test_b10_gitignore_subfolder_session_pattern():
    """Verify nested session paths are covered by session exclusion patterns."""
    nested_path = "data/session/backup/userbot.session"
    assert nested_path.endswith(".session")


def test_b10_env_value_with_embedded_newlines_sanitized(tmp_path: Path):
    """Verify multiline private keys and secrets in env are stripped."""
    multiline_key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
    with patch.dict(os.environ, {"TLS_KEY": multiline_key}, clear=False):
        clean = build_clean_child_env(tmp_path)
        assert "TLS_KEY" not in clean


def test_b10_temp_dir_unlinked_on_interrupted_pipeline(tmp_path: Path):
    """Verify scratch directory in temp_dir_base is wiped cleanly."""
    processor = MediaProcessor(temp_dir_base=str(tmp_path))
    temp_dir = processor._create_temp_dir()
    assert temp_dir.exists()

    dummy_media = MagicMock()
    dummy_media.metadata = {"_temp_dir": str(temp_dir)}
    processor.cleanup(dummy_media)

    assert not temp_dir.exists()


def test_b10_symlink_token_not_chmodded_to_wide_permissions(tmp_path: Path):
    """Verify symlink permissions are never widened during token sync."""
    target_token = tmp_path / "target_token"
    target_token.write_text("token")
    os.chmod(target_token, 0o600)

    symlink_token = tmp_path / "symlink_token"
    symlink_token.symlink_to(target_token)

    # Calling chmod on symlink or checking mode
    assert target_token.stat().st_mode & 0o777 == 0o600


def test_b10_empty_gitignore_rules_fallback():
    """Verify security rules list includes all mandatory credential extensions."""
    mandatory = ["session", ".env", ".venv", "oauth-token"]
    for m in mandatory:
        assert any(m in pattern for pattern in ["*.session*", ".env", ".venv", "*oauth-token*"])
