"""Tests for token-leak optimizations, sibling-bot ID isolation, targeted group slash commands, compaction empty-JSON handling, and STT idle unload."""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.config.manager import ConfigManager
from bot.config.models import BotConfig
from bot.main import GroupBotOrchestrator
from bot.media.image_gen import parse_imagine_command
from bot.media.stt import STTTranscriber
from bot.memory.compaction import (
    CompactionEngine,
    canonicalize_subject,
    is_explicit_empty_fact_response,
)
from bot.memory.db import ChatMessage, MemoryStore
from bot.telegram.handlers import IncomingMessage
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics


@dataclass
class _DummyEv:
    chat_id: int = -1003845062562
    sender_id: int = 8710978944
    out: bool = False
    is_reply: bool = False
    reply_to_sender_id: Optional[int] = None
    raw_text: Optional[str] = "Я сама в шоке"
    entities: Optional[List[Any]] = None
    is_private: bool = False
    sender_is_bot: bool = False
    sender_username: Optional[str] = None


def test_ignored_sender_ids_and_other_bot_slash_commands() -> None:
    heuristics = ReplyHeuristics()
    ev = _DummyEv(sender_id=8710978944, raw_text="Потужнич, привет")
    assert (
        heuristics.should_reply(
            event=ev,
            bot_id=9218923960,
            bot_names=["Потужнич"],
            allowed_chat_ids=[-1003845062562],
            ignored_sender_ids=[8710978944],
        )
        == ReplyAction.IGNORE
    )

    res = parse_imagine_command("/imagine@hermes_kuk7yi6kfm3lcdfa_bot кибер-кот", bot_username=None)
    assert res.is_command is False
    assert res.addressed_to_other_bot is True

    split_res = GroupBotOrchestrator._split_slash_command(
        "/ops@hermes_kuk7yi6kfm3lcdfa_bot status",
        bot_username=None,
    )
    assert split_res == ("/ops", "status", True)


@pytest.mark.asyncio
async def test_group_slash_commands_require_target_prevents_double_trigger(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "userbot_system_prompt.md").write_text("Ты — Потужнич.", encoding="utf-8")
    (cfg_dir / "friends.yaml").write_text("friends: {}\n", encoding="utf-8")
    (cfg_dir / "userbot.yaml").write_text(
        "\n".join(
            [
                "telegram:",
                '  mode: "userbot"',
                "  api_id: 123456",
                '  api_hash: "0123456789abcdef0123456789abcdef"',
                "  allowed_chat_ids: [-1003845062562]",
                "  allowed_dm_user_ids: [382962362]",
                "  admin_user_ids: [382962362]",
                "  ignored_sender_ids: [8710978944]",
                "  group_slash_commands_require_target: true",
            ]
        ),
        encoding="utf-8",
    )
    cfg_mgr = ConfigManager(config_dir=cfg_dir, config_path=cfg_dir / "userbot.yaml")
    store = MemoryStore(db_path=str(tmp_path / "mem.sqlite"))
    await store.init_db()

    mock_ub = MagicMock()
    mock_ub.bot_id = 9218923960
    mock_ub.bot_username = None
    mock_ub.send_reply = AsyncMock(return_value=101)
    mock_ub.send_photo = AsyncMock(return_value=102)
    mock_ub.stop = AsyncMock()

    mock_provider = MagicMock()
    mock_provider.generate_image = AsyncMock()
    mock_provider.generate_response = AsyncMock()

    orch = GroupBotOrchestrator(
        config_mgr=cfg_mgr,
        memory_store=store,
        provider=mock_provider,
        userbot=mock_ub,
        debounce_seconds=0.0,
    )
    try:
        # 1. Bare /imagine in group chat is ignored by Потужнич (handled only by sibling Bot API bot)
        bare_group_img = IncomingMessage(
            msg_id=1,
            chat_id=-1003845062562,
            sender_id=382962362,
            sender_name="Rodion",
            text="/imagine гигачад в качалке",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=False,
            raw_event=SimpleNamespace(
                chat_id=-1003845062562,
                sender_id=382962362,
                is_private=False,
                is_reply=False,
                text="/imagine гигачад в качалке",
            ),
            is_private=False,
        )
        assert await orch.handle_incoming_message(bare_group_img) is None
        mock_provider.generate_image.assert_not_awaited()

        # 2. Bare /ops status in private DM DOES execute immediately
        dm_ops = IncomingMessage(
            msg_id=2,
            chat_id=382962362,
            sender_id=382962362,
            sender_name="Rodion",
            text="/ops status",
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=False,
            raw_event=SimpleNamespace(
                chat_id=382962362,
                sender_id=382962362,
                is_private=True,
                is_reply=False,
                text="/ops status",
            ),
            is_private=True,
        )
        res_dm = await orch.handle_incoming_message(dm_ops)
        assert res_dm is not None
        assert "Режим расширенных инструментов" in res_dm

        # 3. Group /ops status replying directly to Потужнич DOES execute
        reply_group_ops = IncomingMessage(
            msg_id=3,
            chat_id=-1003845062562,
            sender_id=382962362,
            sender_name="Rodion",
            text="/ops status",
            reply_to_msg_id=55,
            reply_to_sender_id=9218923960,
            is_reply=True,
            is_mention=False,
            raw_event=SimpleNamespace(
                chat_id=-1003845062562,
                sender_id=382962362,
                is_private=False,
                is_reply=True,
                reply_to_sender_id=9218923960,
                text="/ops status",
            ),
            is_private=False,
        )
        res_group = await orch.handle_incoming_message(reply_group_ops)
        assert res_group is not None
        assert "Режим расширенных инструментов" in res_group
    finally:
        await orch.stop()


def test_active_window_probability_and_skip_cooldown() -> None:
    heuristics = ReplyHeuristics(default_max_per_hour=5, default_active_window_cooldown_seconds=25.0)
    ev = _DummyEv(sender_id=382962362, raw_text="Содержательная реплика в активном окне")

    with patch("random.random", return_value=0.30):
        assert (
            heuristics.should_reply(
                event=ev,
                bot_id=9218923960,
                bot_names=["Потужнич"],
                allowed_chat_ids=[-1003845062562],
                spontaneous_enabled=True,
                time_since_last_bot_msg=35.0,
                probability=0.08,
                active_conversation_window_seconds=60.0,
                active_window_probability=0.20,
                now=1000.0,
            )
            == ReplyAction.IGNORE
        )


@pytest.mark.asyncio
async def test_compaction_explicit_empty_json_and_stt_idle_unload(tmp_path: Path) -> None:
    assert is_explicit_empty_fact_response("[]") is True
    assert is_explicit_empty_fact_response("```json\n[]\n```") is True
    assert is_explicit_empty_fact_response("") is False
    assert canonicalize_subject("піструнькиня")[0] is None

    store = MemoryStore(db_path=":memory:")
    await store.init_db()
    chat_id = -1003845062562
    for i in range(8):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=200 + i,
                chat_id=chat_id,
                user_id=382962362,
                timestamp=1700000000 + i,
                author_name="Rodion",
                text=f"Короткое обсуждение без фактов номер {i}",
            )
        )

    engine = CompactionEngine(
        memory_store=store,
        summarize_fn=lambda _t: "[]",
        threshold=8,
        batch_size=8,
        min_substantive_messages=6,
    )
    assert await engine.run_compaction_for_chat(chat_id) == 8
    assert store._get_connection().execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0] == 0
    await store.close()

    wav_file = tmp_path / "sample.wav"
    with wave.open(str(wav_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)

    stt = STTTranscriber(
        enabled=True,
        idle_unload_seconds=0.08,
        model_factory=lambda *a, **k: SimpleNamespace(
            transcribe=lambda audio, **kw: (
                [SimpleNamespace(text="база на связи")],
                SimpleNamespace(language="ru", language_probability=0.99),
            )
        ),
    )
    try:
        res = await stt.transcribe(wav_file)
        assert res.text == "база на связи"
        deadline = time.monotonic() + 1.0
        while stt._model is not None and time.monotonic() < deadline:
            time.sleep(0.03)
        assert stt._model is None
    finally:
        stt.close()


def test_bot_config_accepts_new_fields() -> None:
    cfg = BotConfig.model_validate(
        {
            "telegram": {
                "api_id": 123,
                "api_hash": "abc",
                "ignored_sender_ids": [8710978944],
                "group_slash_commands_require_target": True,
            },
            "behavior": {
                "spontaneous_replies": {
                    "probability": 0.08,
                    "active_window_probability": 0.22,
                    "max_per_hour": 2,
                }
            },
            "memory": {
                "compaction_threshold": 220,
                "compaction_batch_size": 140,
                "compaction_min_substantive": 12,
            },
            "voice": {"stt": {"idle_unload_seconds": 300.0}},
        }
    )
    assert cfg.telegram.group_slash_commands_require_target is True
    assert cfg.telegram.ignored_sender_ids == [8710978944]
