"""Unit tests for Proactive Engine, Midnight Quiet Hours, Silence Trigger, Cooldowns & Humor."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from bot.config.models import (
    BotConfig,
    FriendProfile,
    FriendsConfig,
    ProactiveConfig,
    QuietHoursConfig,
    TelegramConfig,
)
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import FactRecord
from bot.proactive.engine import ProactiveEngine
from bot.proactive.humor import HumorGenerator


class DummyConfigManager:
    """Mock ConfigManager for testing proactive rules in isolation."""

    def __init__(self, bot: BotConfig, friends: FriendsConfig):
        self.bot = bot
        self.friends = friends
        self.system_prompt = "Ты — Джефф."

    def get_friend_profile(self, user_id: int):
        return self.friends.friends.get(int(user_id))


@pytest.fixture
def proactive_setup():
    bot = BotConfig(
        telegram=TelegramConfig(api_id=1, api_hash="abc", allowed_chat_ids=[-100]),
        proactive=ProactiveConfig(
            enabled=True,
            quiet_hours=QuietHoursConfig(
                enabled=True,
                start="23:00",
                end="08:00",
                timezone="Europe/Moscow",
            ),
            silence_trigger_hours=6.0,
            scheduled_interval_minutes=180,
            jitter_minutes=0,
            cooldown_minutes=90,
            forbidden_topics=["политика", "религия"],
        ),
    )
    friends = FriendsConfig(
        friends={
            1001: FriendProfile(
                name="Алексей",
                aliases=["Леша"],
                bio="Go разработчик",
                banter_topics=["Go vs Rust", "механические клавиатуры"],
                sensitive_topics=["увольнение 2023"],
            )
        }
    )
    store = MemoryStore(db_path=":memory:")
    cfg_mgr = DummyConfigManager(bot=bot, friends=friends)
    engine = ProactiveEngine(config_mgr=cfg_mgr, memory_store=store)  # type: ignore[arg-type]
    return engine, cfg_mgr, store


def test_quiet_hours_midnight_wraparound(proactive_setup):
    """Verify quiet hours logic across midnight (23:30 -> quiet, 07:15 -> quiet, 14:00 -> active)."""
    engine, cfg_mgr, _ = proactive_setup
    tz = ZoneInfo("Europe/Moscow")

    # 23:30 Moscow time -> Quiet
    dt_2330 = datetime(2026, 11, 13, 23, 30, tzinfo=tz)
    assert engine.is_in_quiet_hours(dt_2330) is True

    # 07:15 Moscow time -> Quiet
    dt_0715 = datetime(2026, 11, 14, 7, 15, tzinfo=tz)
    assert engine.is_in_quiet_hours(dt_0715) is True

    # 14:00 Moscow time -> Active (Not quiet)
    dt_1400 = datetime(2026, 11, 14, 14, 0, tzinfo=tz)
    assert engine.is_in_quiet_hours(dt_1400) is False

    # Disabled quiet hours
    cfg_mgr.bot.proactive.quiet_hours.enabled = False
    assert engine.is_in_quiet_hours(dt_2330) is False
    assert engine.is_in_quiet_hours(dt_0715) is False


def test_silence_trigger_and_anti_spam_cooldown(proactive_setup):
    """Verify silence trigger firing after inactivity and suppression by anti-spam cooldown."""
    engine, _, _ = proactive_setup
    chat_id = -100

    # Fixed base timestamp: 14:00 (outside quiet hours)
    tz = ZoneInfo("Europe/Moscow")
    base_dt = datetime(2026, 11, 14, 14, 0, tzinfo=tz)
    now = base_dt.timestamp()

    # 1. Normal active chat: last msg 10 mins ago, last bot msg 100 mins ago
    last_chat_msg = now - 600
    last_bot_msg = now - 6000
    trigger = engine.check_triggers(chat_id, last_chat_msg, last_bot_msg, current_time=now)
    assert trigger is None

    # 2. Silence trigger: chat silent for 6.5 hours (silence_trigger_hours = 6.0), bot last msg 3 hours ago (>90m cooldown)
    last_chat_msg_silent = now - (6.5 * 3600)
    last_bot_msg_old = now - (3.0 * 3600)
    trigger_silent = engine.check_triggers(
        chat_id, last_chat_msg_silent, last_bot_msg_old, current_time=now
    )
    assert trigger_silent == "silence_breaker"

    # 3. Anti-spam cooldown: chat silent for 6.5 hours, but bot sent message 30 mins ago (<90m cooldown)
    last_bot_msg_recent = now - (30 * 60)
    trigger_cooldown = engine.check_triggers(
        chat_id, last_chat_msg_silent, last_bot_msg_recent, current_time=now
    )
    assert trigger_cooldown is None

    # 4. Scheduled banter: chat not silent (active 1 hour ago), but bot hasn't spoken for 3.5 hours (>180m interval)
    last_chat_msg_recent = now - 3600
    last_bot_msg_interval = now - (3.5 * 3600)
    trigger_banter = engine.check_triggers(
        chat_id, last_chat_msg_recent, last_bot_msg_interval, current_time=now
    )
    assert trigger_banter == "scheduled_banter"


def test_humor_prompt_sensitive_topics_exclusion():
    """Verify humor prompt strictly excludes friend sensitive topics and forbidden topics."""
    generator = HumorGenerator()

    friend = FriendProfile(
        name="Алексей",
        aliases=["Леша"],
        bio="Senior Go Developer",
        banter_topics=["Go vs Rust", "эргономичные клавиатуры"],
        sensitive_topics=["увольнение 2023", "стартап с криптой"],
    )

    facts = [
        FactRecord(
            id=1,
            subject="Алексей",
            fact="Собрал клавиатуру с синими свитчами",
            category="lore",
        )
    ]

    messages = [
        ChatMessage(
            telegram_msg_id=1,
            chat_id=-100,
            user_id=1001,
            timestamp=1000,
            author_name="Алексей",
            text="Свитчи слишком громкие",
        )
    ]

    prompt = generator.build_humor_prompt(
        recent_messages=messages,
        friend=friend,
        facts=facts,
        trigger_type="banter",
        humor_style="friendly_sarcasm",
        forbidden_topics=["политика", "религия"],
    )

    # Verify friend banter topics are included
    assert "Go vs Rust" in prompt
    assert "эргономичные клавиатуры" in prompt

    # Verify sensitive and forbidden topics are in the STRICT EXCLUSION section
    assert "СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ" in prompt
    assert "увольнение 2023" in prompt
    assert "стартап с криптой" in prompt
    assert "политика" in prompt
    assert "религия" in prompt


@pytest.mark.asyncio
async def test_evaluate_and_generate_end_to_end(proactive_setup):
    """Verify evaluate_and_generate produces proactive banter with memory facts."""
    engine, _, store = proactive_setup
    await store.init_db()

    chat_id = -100
    tz = ZoneInfo("Europe/Moscow")
    base_dt = datetime(2026, 11, 14, 14, 0, tzinfo=tz)
    now = base_dt.timestamp()

    # Save a chat message from Алексей
    await store.save_message(
        ChatMessage(
            telegram_msg_id=50,
            chat_id=chat_id,
            user_id=1001,
            timestamp=int(now - 7 * 3600),
            author_name="Алексей",
            text="Привет всем",
        )
    )

    # Save a fact about Алексей
    await store.add_fact(
        subject="Алексей",
        fact="Любит клавиатуры",
        category="lore",
        source_chat_id=chat_id,
    )

    # Inactive chat for 7 hours -> triggers silence breaker
    last_chat = now - (7 * 3600)
    last_bot = now - (4 * 3600)

    generated_msg = await engine.evaluate_and_generate(
        chat_id=chat_id,
        last_chat_msg_time=last_chat,
        last_bot_msg_time=last_bot,
        current_time=now,
    )

    assert generated_msg is not None
    assert isinstance(generated_msg, str)
    assert len(generated_msg) > 5

    await store.close()


@pytest.mark.asyncio
async def test_run_proactive_tick_skips_private_chats(proactive_setup):
    """Verify run_proactive_tick strictly skips private chats (chat_id >= 0) from digests and banter."""
    from unittest.mock import AsyncMock, MagicMock

    from bot.config.models import CronDigestJobConfig
    from bot.core.models import ProviderResponse

    engine, cfg_mgr, store = proactive_setup
    await store.init_db()

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Утренний дайджест готов!", model="test-model")
    )
    engine.provider = mock_provider
    engine.humor_gen.provider = mock_provider
    engine.digest_gen.provider = mock_provider

    cfg_mgr.bot.proactive.cron_digests = [
        CronDigestJobConfig(
            id="morning_digest",
            cron="0 9 * * *",
            timezone="Europe/Kyiv",
            slot_label="morning",
        )
    ]

    kyiv = ZoneInfo("Europe/Kyiv")
    t_cursor = datetime(2026, 11, 14, 8, 55, 0, tzinfo=kyiv)
    t_morning = datetime(2026, 11, 14, 9, 0, 10, tzinfo=kyiv)
    engine.init_cron_cursor(now_dt=t_cursor)

    chat_states = {
        382962362: {
            "last_chat_msg_time": t_morning.timestamp() - 10 * 3600,
            "last_bot_msg_time": t_morning.timestamp() - 10 * 3600,
        },
        -100: {
            "last_chat_msg_time": t_morning.timestamp() - 10 * 3600,
            "last_bot_msg_time": t_morning.timestamp() - 10 * 3600,
        },
    }
    sent: list[tuple[int, str]] = []

    async def _send(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    await engine.run_proactive_tick(
        get_chat_states=lambda: chat_states,
        send_message_fn=_send,
        get_target_chat_ids=lambda: [382962362, -100],
        now_dt=t_morning,
    )

    # Only group -100 received cron digest; DM 382962362 was strictly skipped
    assert len(sent) == 1
    assert sent[0][0] == -100
    assert "Утренний дайджест готов!" in sent[0][1]

    # Verify private chat state was NOT updated by cron digest
    assert chat_states[382962362]["last_bot_msg_time"] == t_morning.timestamp() - 10 * 3600
    assert chat_states[-100]["last_bot_msg_time"] == pytest.approx(t_morning.timestamp())

    await store.close()
