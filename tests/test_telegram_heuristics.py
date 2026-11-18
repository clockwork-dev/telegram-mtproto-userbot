"""Unit tests for Telegram userbot reply heuristics."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bot.telegram.heuristics import ReplyAction, ReplyHeuristics


@dataclass
class DummyEvent:
    chat_id: int = 100
    sender_id: int = 200
    out: bool = False
    is_reply: bool = False
    reply_to_sender_id: Optional[int] = None
    raw_text: Optional[str] = "Hello world"
    entities: Optional[List[Any]] = None


def test_mention_detection_direct_names():
    heuristics = ReplyHeuristics()
    bot_names = ["Джефф", "бот", "Jeff"]

    assert heuristics.is_mention("Привет, Джефф! Как дела?", bot_names) is True
    assert heuristics.is_mention("эй, БОТ, ответь мне", bot_names) is True
    assert heuristics.is_mention("Hey Jeff, what is up?", bot_names) is True
    assert heuristics.is_mention("Это обычное сообщение", bot_names) is False


def test_mention_detection_word_boundaries_no_false_positives():
    heuristics = ReplyHeuristics()
    bot_names = ["бот", "Jeff"]

    # "робототехника" contains "бот" but is not "бот"
    assert heuristics.is_mention("Робототехника шагает вперед", bot_names) is False
    # "Jefferson" contains "Jeff" but is not "Jeff"
    assert heuristics.is_mention("President Jefferson was historical", bot_names) is False


def test_mention_detection_username():
    heuristics = ReplyHeuristics()
    assert heuristics.is_mention("Hey @my_bot check this", ["Jeff"], bot_username="my_bot") is True
    assert heuristics.is_mention("Hey @my_bot check this", ["Jeff"], bot_username="@my_bot") is True
    assert heuristics.is_mention("Hey @other_bot check this", ["Jeff"], bot_username="my_bot") is False


def test_mention_detection_via_entity_user_id():
    heuristics = ReplyHeuristics()
    mock_entity = MagicMock()
    mock_entity.user_id = 99999

    assert heuristics.is_mention("Hey you", [], entities=[mock_entity], bot_id=99999) is True
    assert heuristics.is_mention("Hey you", [], entities=[mock_entity], bot_id=11111) is False


def test_should_reply_whitelist_blocks_unauthorized_chat():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=999, raw_text="Hey @my_bot")

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        bot_username="my_bot",
        allowed_chat_ids=[100, 200],
    )
    assert action == ReplyAction.IGNORE


def test_should_reply_whitelist_allows_authorized_chat():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=100, raw_text="Hey @my_bot")

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        bot_username="my_bot",
        allowed_chat_ids=[100, 200],
    )
    assert action == ReplyAction.MUST_REPLY


def test_should_reply_self_message_suppression():
    heuristics = ReplyHeuristics()

    # Outgoing event (event.out == True)
    event_out = DummyEvent(chat_id=100, out=True, sender_id=123, raw_text="Self message")
    assert heuristics.should_reply(event_out, bot_id=123, bot_names=["Jeff"]) == ReplyAction.IGNORE

    # Message authored by bot itself
    event_self = DummyEvent(chat_id=100, out=False, sender_id=123, raw_text="Self message")
    assert heuristics.should_reply(event_self, bot_id=123, bot_names=["Jeff"]) == ReplyAction.IGNORE


def test_should_reply_direct_reply_to_bot():
    heuristics = ReplyHeuristics()
    event = DummyEvent(
        chat_id=100,
        sender_id=555,
        is_reply=True,
        reply_to_sender_id=123,
        raw_text="Yes exactly",
    )

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        allowed_chat_ids=[100],
    )
    assert action == ReplyAction.MUST_REPLY


def test_should_reply_reply_to_other_user_does_not_force_reply():
    heuristics = ReplyHeuristics()
    event = DummyEvent(
        chat_id=100,
        sender_id=555,
        is_reply=True,
        reply_to_sender_id=888,  # Different user
        raw_text="Talking to someone else",
    )

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        allowed_chat_ids=[100],
        spontaneous_enabled=False,
    )
    assert action == ReplyAction.IGNORE


def test_spontaneous_reply_gating_disabled():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=100, raw_text="Random conversation message")

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        allowed_chat_ids=[100],
        spontaneous_enabled=False,
    )
    assert action == ReplyAction.IGNORE


def test_spontaneous_reply_gating_quiet_hours():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=100, raw_text="Late night chat")

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        allowed_chat_ids=[100],
        spontaneous_enabled=True,
        in_quiet_hours=True,
        time_since_last_bot_msg=1000.0,
        min_interval_seconds=300.0,
        probability=1.0,
    )
    assert action == ReplyAction.IGNORE


def test_spontaneous_reply_gating_cooldown():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=100, raw_text="Rapid chat")

    action = heuristics.should_reply(
        event=event,
        bot_id=123,
        bot_names=["Jeff"],
        allowed_chat_ids=[100],
        spontaneous_enabled=True,
        in_quiet_hours=False,
        time_since_last_bot_msg=100.0,  # Below min 300s
        min_interval_seconds=300.0,
        probability=1.0,
    )
    assert action == ReplyAction.IGNORE


def test_spontaneous_reply_probability_success_and_failure():
    heuristics = ReplyHeuristics()
    event = DummyEvent(chat_id=100, raw_text="Casual chat statement")

    # When random rolls 0.04 (below probability 0.10) -> SPONTANEOUS_REPLY
    with patch("random.random", return_value=0.04):
        action = heuristics.should_reply(
            event=event,
            bot_id=123,
            bot_names=["Jeff"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            in_quiet_hours=False,
            time_since_last_bot_msg=500.0,
            min_interval_seconds=300.0,
            probability=0.10,
        )
        assert action == ReplyAction.SPONTANEOUS_REPLY

    # When random rolls 0.25 (above probability 0.10) -> IGNORE
    with patch("random.random", return_value=0.25):
        action = heuristics.should_reply(
            event=event,
            bot_id=123,
            bot_names=["Jeff"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            in_quiet_hours=False,
            time_since_last_bot_msg=500.0,
            min_interval_seconds=300.0,
            probability=0.10,
        )
        assert action == ReplyAction.IGNORE


def test_indirect_reference_pronouns_and_colloquial_forms():
    heuristics = ReplyHeuristics()

    assert heuristics.is_indirect_reference("Че она не отвечает") is True
    assert heuristics.is_indirect_reference("Её 500 оборотов") is True
    assert heuristics.is_indirect_reference("А шо ей надо вообще") is True
    assert heuristics.is_indirect_reference("У неё опять свои приколы") is True
    assert heuristics.is_indirect_reference("Спроси у бота") is True
    assert heuristics.is_indirect_reference("Это піструнькиня написала") is True

    # Word boundary guard against false positives
    assert heuristics.is_indirect_reference("Корона Российской Империи") is False
    assert heuristics.is_indirect_reference("Робототехника развивается") is False


def test_active_conversation_window_no_auto_direct_reply_on_pronouns_questions_or_acks():
    heuristics = ReplyHeuristics()

    # Within 90s active window: generic pronouns and questions must NOT auto-trigger DIRECT_REPLY
    with patch("random.random", return_value=0.99):
        for sample_text in (
            "Че она не отвечает",
            "Её 500 оборотов",
            "А вона що сказала",
            "Де її відповідь",
            "А почему так быстро?",
        ):
            ev = DummyEvent(chat_id=100, raw_text=sample_text)
            action = heuristics.should_reply(
                event=ev,
                bot_id=123,
                bot_names=["Джефф"],
                allowed_chat_ids=[100],
                spontaneous_enabled=True,
                in_quiet_hours=False,
                time_since_last_bot_msg=35.0,
                min_interval_seconds=300.0,
                probability=0.20,
                active_conversation_window_seconds=90.0,
                now=1000.0,
            )
            assert action == ReplyAction.IGNORE

    # Standalone '?', trivial <=4-char acks, and empty-text stickers/GIFs must NEVER trigger even at prob=1.0
    for trivial_text in ("?", "???", "да", "уже", "ок", "ага", "окей", "", "   "):
        ev_trivial = DummyEvent(chat_id=100, raw_text=trivial_text)
        action = heuristics.should_reply(
            event=ev_trivial,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            in_quiet_hours=False,
            time_since_last_bot_msg=40.0,
            min_interval_seconds=300.0,
            probability=1.0,
            active_conversation_window_seconds=90.0,
            now=1000.0,
        )
        assert action == ReplyAction.IGNORE


def test_active_conversation_window_boosted_spontaneous_followup():
    heuristics = ReplyHeuristics()
    ev_followup = DummyEvent(chat_id=100, raw_text="Ну да, логично в принципе")

    # Within 90s window and past 25s active-window cooldown, base probability 0.20 is boosted to 0.45
    with patch("random.random", return_value=0.35):
        action = heuristics.should_reply(
            event=ev_followup,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            in_quiet_hours=False,
            time_since_last_bot_msg=45.0,
            min_interval_seconds=300.0,
            probability=0.20,
            active_conversation_window_seconds=90.0,
        )
        assert action == ReplyAction.SPONTANEOUS_REPLY


def test_heuristics_active_window_cooldown_blocks_rapid_burst():
    heuristics = ReplyHeuristics(default_active_window_cooldown_seconds=25.0)
    ev = DummyEvent(chat_id=100, raw_text="Уже унесло совсем далеко")

    with patch("random.random", return_value=0.01):
        # 12s after bot message (< 25s cooldown) -> IGNORE
        a_too_soon = heuristics.should_reply(
            event=ev,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            time_since_last_bot_msg=12.0,
            probability=1.0,
            active_conversation_window_seconds=90.0,
            now=1000.0,
        )
        assert a_too_soon == ReplyAction.IGNORE

        # 30s after bot message (>= 25s cooldown) -> SPONTANEOUS_REPLY
        a_ready = heuristics.should_reply(
            event=ev,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100],
            spontaneous_enabled=True,
            time_since_last_bot_msg=30.0,
            probability=1.0,
            active_conversation_window_seconds=90.0,
            now=1018.0,
        )
        assert a_ready == ReplyAction.SPONTANEOUS_REPLY


def test_heuristics_max_per_hour_sliding_window_and_explicit_bypass():
    heuristics = ReplyHeuristics(default_max_per_hour=3)
    ev_chat1 = DummyEvent(chat_id=100, raw_text="Интересная мысль сегодня")
    ev_chat2 = DummyEvent(chat_id=200, raw_text="Другой чат тоже общается")

    with patch("random.random", return_value=0.0):
        for offset in (0.0, 400.0, 800.0):
            assert heuristics.should_reply(
                event=ev_chat1,
                bot_id=123,
                bot_names=["Джефф"],
                allowed_chat_ids=[100, 200],
                spontaneous_enabled=True,
                time_since_last_bot_msg=500.0,
                min_interval_seconds=300.0,
                probability=0.5,
                now=1000.0 + offset,
            ) == ReplyAction.SPONTANEOUS_REPLY

        # 4th spontaneous attempt within 60 minutes for chat 100 is blocked
        assert heuristics.should_reply(
            event=ev_chat1,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100, 200],
            spontaneous_enabled=True,
            time_since_last_bot_msg=500.0,
            min_interval_seconds=300.0,
            probability=0.5,
            now=2000.0,
        ) == ReplyAction.IGNORE

        # Separate chat 200 still has its own hourly budget
        assert heuristics.should_reply(
            event=ev_chat2,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100, 200],
            spontaneous_enabled=True,
            time_since_last_bot_msg=500.0,
            min_interval_seconds=300.0,
            probability=0.5,
            now=2000.0,
        ) == ReplyAction.SPONTANEOUS_REPLY

        # Explicit mention and direct reply to bot still respond MUST_REPLY even when hourly cap is reached
        ev_mention = DummyEvent(chat_id=100, raw_text="Джефф, да")
        assert heuristics.should_reply(
            event=ev_mention,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100, 200],
            spontaneous_enabled=True,
            time_since_last_bot_msg=5.0,
            now=2001.0,
        ) == ReplyAction.MUST_REPLY

        ev_direct_reply = DummyEvent(
            chat_id=100,
            is_reply=True,
            reply_to_sender_id=123,
            raw_text="да",
        )
        assert heuristics.should_reply(
            event=ev_direct_reply,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100, 200],
            spontaneous_enabled=True,
            time_since_last_bot_msg=5.0,
            now=2002.0,
        ) == ReplyAction.MUST_REPLY

        # Once the oldest reply expires (> 3600s), chat 100 can reply again
        assert heuristics.should_reply(
            event=ev_chat1,
            bot_id=123,
            bot_names=["Джефф"],
            allowed_chat_ids=[100, 200],
            spontaneous_enabled=True,
            time_since_last_bot_msg=500.0,
            min_interval_seconds=300.0,
            probability=0.5,
            now=4601.0,
        ) == ReplyAction.SPONTANEOUS_REPLY


def test_whitelisted_dm_user_must_reply_unconditionally():
    """Verify whitelisted DM user (Rodion ID 382962362) receives MUST_REPLY without mention or reply."""
    heuristics = ReplyHeuristics()
    # Plain message in DM without mention or reply
    event_plain = DummyEvent(
        chat_id=382962362,
        sender_id=382962362,
        raw_text="Привет, как дела?",
    )
    setattr(event_plain, "is_private", True)

    action = heuristics.should_reply(
        event=event_plain,
        bot_id=123,
        bot_names=["Джефф", "jeff"],
        allowed_chat_ids=[],
        allowed_dm_user_ids=[382962362],
        spontaneous_enabled=False,
    )
    assert action == ReplyAction.MUST_REPLY


def test_non_whitelisted_dm_user_strictly_ignored():
    """Verify non-whitelisted DM user receives IGNORE even if mentioning bot or directly replying."""
    heuristics = ReplyHeuristics()
    # Unknown user DM
    event_unauthorized = DummyEvent(
        chat_id=999999999,
        sender_id=999999999,
        raw_text="Джефф, ответь мне срочно!",
    )
    setattr(event_unauthorized, "is_private", True)

    action = heuristics.should_reply(
        event=event_unauthorized,
        bot_id=123,
        bot_names=["Джефф", "jeff"],
        allowed_chat_ids=[],
        allowed_dm_user_ids=[382962362],
        spontaneous_enabled=True,
        probability=1.0,
    )
    assert action == ReplyAction.IGNORE


def test_anti_loop_drop_hermes_bot_in_group():
    """Verify bot sender @hermes_kuk7yi6kfm3lcdfa_bot is strictly ignored in groups."""
    heuristics = ReplyHeuristics()
    event_hermes = DummyEvent(
        chat_id=-1003845062562,
        sender_id=8710978944,
        raw_text="Джефф, вот утренний дайджест",
    )
    setattr(event_hermes, "is_private", False)
    setattr(event_hermes, "sender_username", "hermes_kuk7yi6kfm3lcdfa_bot")
    setattr(event_hermes, "sender_is_bot", True)

    action = heuristics.should_reply(
        event=event_hermes,
        bot_id=123,
        bot_names=["Джефф", "jeff"],
        allowed_chat_ids=[-1003845062562],
        allowed_dm_user_ids=[382962362],
        ignored_sender_usernames=["hermes_kuk7yi6kfm3lcdfa_bot"],
    )
    assert action == ReplyAction.IGNORE


def test_anti_loop_drop_generic_bot_sender():
    """Verify any sender with bot=True is strictly ignored."""
    heuristics = ReplyHeuristics()
    mock_sender = MagicMock()
    mock_sender.bot = True
    mock_sender.username = "some_random_bot"

    event_bot = DummyEvent(
        chat_id=-1003845062562,
        sender_id=555555,
        raw_text="Джефф привет",
    )
    setattr(event_bot, "is_private", False)
    setattr(event_bot, "sender", mock_sender)

    action = heuristics.should_reply(
        event=event_bot,
        bot_id=123,
        bot_names=["Джефф", "jeff"],
        allowed_chat_ids=[-1003845062562],
        allowed_dm_user_ids=[382962362],
    )
    assert action == ReplyAction.IGNORE


def test_unapproved_group_ignored_in_dm_mode():
    """Verify groups are ignored when allowed_chat_ids is empty and userbot is in DM mode."""
    heuristics = ReplyHeuristics()
    event_group = DummyEvent(
        chat_id=-1003845062562,
        sender_id=382962362,
        raw_text="Джефф, ты тут?",
    )
    setattr(event_group, "is_private", False)

    action = heuristics.should_reply(
        event=event_group,
        bot_id=123,
        bot_names=["Джефф", "jeff"],
        allowed_chat_ids=[],  # Groups disabled
        allowed_dm_user_ids=[382962362],
    )
    assert action == ReplyAction.IGNORE

