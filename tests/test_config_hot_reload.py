"""Unit tests for Pydantic Config Models, Friends Profile Lookup & Async Hot-Reload."""

import asyncio
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bot.config.manager import ConfigManager
from bot.config.models import (
    FriendProfile,
    FriendsConfig,
    TelegramConfig,
)


def create_sample_configs(tmp_path: Path):
    bot_file = tmp_path / "bot.yaml"
    friends_file = tmp_path / "friends.yaml"
    prompt_file = tmp_path / "system_prompt.md"

    bot_data = {
        "telegram": {
            "api_id": 999888,
            "api_hash": "hash_sample_12345",
            "session_path": "data/session/userbot.session",
            "allowed_chat_ids": [-100111222],
        },
        "behavior": {
            "bot_names": ["ТестБот"],
            "typing_simulation": True,
        },
        "proactive": {
            "enabled": True,
            "quiet_hours": {
                "enabled": True,
                "start": "23:00",
                "end": "08:00",
                "timezone": "Europe/Moscow",
            },
            "silence_trigger_hours": 4.0,
            "cooldown_minutes": 60,
        },
    }
    bot_file.write_text(yaml.dump(bot_data), encoding="utf-8")

    friends_data = {
        "friends": {
            123456789: {
                "name": "Алексей",
                "aliases": ["Леша", "alex"],
                "bio": "Senior Go-разработчик",
                "role": "техлид",
                "communication_style": "ироничный",
                "banter_topics": ["Go против Rust"],
                "sensitive_topics": ["увольнение в 2023"],
            }
        }
    }
    friends_file.write_text(yaml.dump(friends_data), encoding="utf-8")
    prompt_file.write_text("Ты — тестовый бот.", encoding="utf-8")

    return bot_file, friends_file, prompt_file


def test_pydantic_models_validation():
    """Verify validation passes on valid data and raises ValidationError on invalid data."""
    tg_cfg = TelegramConfig(api_id=123, api_hash="abc", allowed_chat_ids=[-100])
    assert tg_cfg.api_id == 123
    assert tg_cfg.session_path == "data/session/userbot.session"

    # Missing api_hash raises ValidationError
    with pytest.raises(ValidationError):
        TelegramConfig(api_id=123)

    friend = FriendProfile(
        name="Мария",
        aliases=["Маша"],
        banter_topics=["матча латте"],
        sensitive_topics=["авария"],
    )
    assert friend.name == "Мария"
    assert "матча латте" in friend.banter_topics

    cfg = FriendsConfig.model_validate({"friends": {"98765": {"name": "Test"}}})
    assert 98765 in cfg.friends
    assert cfg.friends[98765].name == "Test"


def test_config_manager_loading_lookup_and_userbot_fallback(tmp_path: Path):
    """Verify ConfigManager loads configs, falls back to userbot.yaml, and retrieves friend profiles."""
    create_sample_configs(tmp_path)
    mgr = ConfigManager(config_dir=tmp_path)

    assert mgr.bot is not None
    assert mgr.bot.telegram.api_id == 999888
    assert mgr.bot.behavior.bot_names == ["ТестБот"]
    assert mgr.system_prompt == "Ты — тестовый бот."

    profile = mgr.get_friend_profile(123456789)
    assert profile is not None and profile.name == "Алексей"
    profile_str = mgr.get_friend_profile("123456789")  # type: ignore[arg-type]
    assert profile_str is not None and profile_str.name == "Алексей"
    assert mgr.get_friend_profile(999999999) is None

    # Verify automatic resolution of userbot.yaml + userbot_system_prompt.md when bot.yaml is absent
    ub_dir = tmp_path / "ub_cfg"
    ub_dir.mkdir()
    (ub_dir / "userbot.yaml").write_text(
        yaml.dump({"telegram": {"mode": "userbot", "api_id": 4242, "api_hash": "hash_ub"}}),
        encoding="utf-8",
    )
    (ub_dir / "userbot_system_prompt.md").write_text("Ты — Джефф.", encoding="utf-8")
    ub_mgr = ConfigManager(config_dir=ub_dir)
    assert ub_mgr.bot is not None
    assert ub_mgr.bot.telegram.api_id == 4242
    assert ub_mgr.system_prompt == "Ты — Джефф."


@pytest.mark.asyncio
async def test_async_hot_reload_on_mtime_and_callbacks(tmp_path: Path):
    """Verify modifying config files triggers atomic hot-reload and registered callbacks."""
    bot_file, friends_file, prompt_file = create_sample_configs(tmp_path)
    mgr = ConfigManager(config_dir=tmp_path)
    callback_hits: list[int] = []
    mgr.register_reload_callback(lambda: callback_hits.append(1))

    task = await mgr.start_watcher(interval=0.1)
    try:
        await asyncio.sleep(0.15)

        with open(bot_file, "r", encoding="utf-8") as f:
            bot_data = yaml.safe_load(f)
        bot_data["telegram"]["api_id"] = 777777
        bot_data["behavior"]["bot_names"] = ["ОбновленныйБот"]

        time.sleep(0.05)
        bot_file.write_text(yaml.dump(bot_data), encoding="utf-8")

        await asyncio.sleep(0.3)
        assert mgr.bot is not None
        assert mgr.bot.telegram.api_id == 777777
        assert mgr.bot.behavior.bot_names == ["ОбновленныйБот"]
        assert len(callback_hits) >= 1

        new_friends = {
            "friends": {
                555: {
                    "name": "НовыйДруг",
                    "role": "дизайнер",
                    "banter_topics": ["Figma"],
                }
            }
        }
        time.sleep(0.05)
        friends_file.write_text(yaml.dump(new_friends), encoding="utf-8")

        await asyncio.sleep(0.3)
        assert mgr.get_friend_profile(555) is not None
        assert mgr.get_friend_profile(555).name == "НовыйДруг"

        time.sleep(0.05)
        prompt_file.write_text("Новый системный промпт 2.0", encoding="utf-8")

        await asyncio.sleep(0.3)
        assert mgr.system_prompt == "Новый системный промпт 2.0"

    finally:
        mgr.stop_watcher()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_hot_reload_failure_recovery(tmp_path: Path):
    """Verify corrupt/invalid configuration retains existing valid config and logs error."""
    bot_file, _, _ = create_sample_configs(tmp_path)
    mgr = ConfigManager(config_dir=tmp_path)
    assert mgr.bot is not None
    initial_api_id = mgr.bot.telegram.api_id

    time.sleep(0.05)
    bot_file.write_text("invalid_yaml: [this is broken syntax", encoding="utf-8")
    assert not mgr.reload_changed()
    assert mgr.bot.telegram.api_id == initial_api_id

    time.sleep(0.05)
    bot_file.write_text("telegram: { api_id: 'not_an_int' }", encoding="utf-8")
    assert not mgr.reload_changed()
    assert mgr.bot.telegram.api_id == initial_api_id

    time.sleep(0.05)
    valid_data = {"telegram": {"api_id": 112233, "api_hash": "restored_hash"}}
    bot_file.write_text(yaml.dump(valid_data), encoding="utf-8")
    assert mgr.reload_changed()
    assert mgr.bot.telegram.api_id == 112233
