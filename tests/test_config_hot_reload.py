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
    # Valid TelegramConfig
    tg_cfg = TelegramConfig(api_id=123, api_hash="abc", allowed_chat_ids=[-100])
    assert tg_cfg.api_id == 123
    assert tg_cfg.session_path == "data/session/userbot.session"

    # Missing api_hash raises ValidationError
    with pytest.raises(ValidationError):
        TelegramConfig(api_id=123)  # type: ignore[call-arg]

    # Valid FriendProfile
    friend = FriendProfile(
        name="Мария",
        aliases=["Маша"],
        banter_topics=["матча латте"],
        sensitive_topics=["авария"],
    )
    assert friend.name == "Мария"
    assert "матча латте" in friend.banter_topics

    # FriendsConfig coercing string digit keys to int
    cfg = FriendsConfig.model_validate({"friends": {"98765": {"name": "Test"}}})
    assert 98765 in cfg.friends
    assert cfg.friends[98765].name == "Test"


def test_config_manager_loading_and_lookup(tmp_path: Path):
    """Verify ConfigManager loads configs and retrieves friend profiles."""
    create_sample_configs(tmp_path)
    mgr = ConfigManager(config_dir=tmp_path)

    assert mgr.bot is not None
    assert mgr.bot.telegram.api_id == 999888
    assert mgr.bot.behavior.bot_names == ["ТестБот"]
    assert mgr.system_prompt == "Ты — тестовый бот."

    # Look up existing friend
    profile = mgr.get_friend_profile(123456789)
    assert profile is not None
    assert profile.name == "Алексей"
    assert profile.role == "техлид"

    # String user_id lookup
    profile_str = mgr.get_friend_profile("123456789")  # type: ignore[arg-type]
    assert profile_str is not None
    assert profile_str.name == "Алексей"

    # Unknown user_id
    assert mgr.get_friend_profile(999999999) is None


@pytest.mark.asyncio
async def test_async_hot_reload_on_mtime(tmp_path: Path):
    """Verify modifying config files triggers atomic hot-reload in watcher loop."""
    bot_file, friends_file, prompt_file = create_sample_configs(tmp_path)
    mgr = ConfigManager(config_dir=tmp_path)

    # Start watcher with fast 0.1s polling for tests
    task = await mgr.start_watcher(interval=0.1)

    try:
        # Sleep to ensure initial mtime baseline
        await asyncio.sleep(0.15)

        # 1. Update bot.yaml
        with open(bot_file, "r", encoding="utf-8") as f:
            bot_data = yaml.safe_load(f)
        bot_data["telegram"]["api_id"] = 777777
        bot_data["behavior"]["bot_names"] = ["ОбновленныйБот"]

        # Ensure mtime timestamp changes
        time.sleep(0.05)
        bot_file.write_text(yaml.dump(bot_data), encoding="utf-8")

        # Wait for watcher to detect
        await asyncio.sleep(0.3)
        assert mgr.bot.telegram.api_id == 777777
        assert mgr.bot.behavior.bot_names == ["ОбновленныйБот"]

        # 2. Update friends.yaml
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

        # 3. Update system_prompt.md
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
    initial_api_id = mgr.bot.telegram.api_id

    # Corrupt bot.yaml with invalid syntax
    time.sleep(0.05)
    bot_file.write_text("invalid_yaml: [this is broken syntax", encoding="utf-8")

    # Call reload_changed directly
    changed = mgr.reload_changed()
    assert not changed
    # Existing valid config must be preserved
    assert mgr.bot.telegram.api_id == initial_api_id

    # Corrupt bot.yaml with invalid pydantic schema (missing required fields)
    time.sleep(0.05)
    bot_file.write_text("telegram: { api_id: 'not_an_int' }", encoding="utf-8")
    changed2 = mgr.reload_changed()
    assert not changed2
    assert mgr.bot.telegram.api_id == initial_api_id

    # Restore valid bot.yaml
    time.sleep(0.05)
    valid_data = {
        "telegram": {
            "api_id": 112233,
            "api_hash": "restored_hash",
        }
    }
    bot_file.write_text(yaml.dump(valid_data), encoding="utf-8")

    changed3 = mgr.reload_changed()
    assert changed3
    assert mgr.bot.telegram.api_id == 112233
