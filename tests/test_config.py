"""Unit tests for MTProto & Bot API configuration models, Potuzhnych's male TTS defaults, and ConfigManager."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import bot.antigravity as agy_pkg
import bot.config as cfg_pkg
from bot.config.manager import ConfigManager
from bot.config.models import (
    DEFAULT_MALE_GEMINI_STYLE_PROMPT,
    AntigravityProviderConfig,
    BehaviorConfig,
    BotConfig,
    ImageGenerationConfig,
    MemoryBackupConfig,
    MemoryConfig,
    MemoryEmbeddingsConfig,
    MemoryReflectionConfig,
    ModelRouterConfig,
    QuotaConfig,
    SelfEvolutionConfig,
    TelegramConfig,
    TTSConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_telegram_config_userbot_and_bot_api_modes() -> None:
    # Default mode is "userbot", validates without bot_token
    tg_userbot = TelegramConfig(
        api_id=123456,
        api_hash="0123456789abcdef0123456789abcdef",
        allowed_dm_user_ids=[382962362],
        admin_user_ids=[382962362],
    )
    assert tg_userbot.mode == "userbot"
    assert tg_userbot.bot_token is None
    assert tg_userbot.admin_user_ids == [382962362]
    assert tg_userbot.ignored_sender_usernames == ["hermes_kuk7yi6kfm3lcdfa_bot"]
    # repr=False hides raw api_hash
    assert "0123456789abcdef0123456789abcdef" not in repr(tg_userbot)

    # Missing api_hash in userbot mode raises ValidationError
    with pytest.raises(ValidationError):
        TelegramConfig(api_id=123456)

    # None coercion for list fields
    tg_nones = TelegramConfig(
        api_id=123,
        api_hash="abc",
        allowed_chat_ids=None,  # type: ignore[arg-type]
        allowed_dm_user_ids=None,  # type: ignore[arg-type]
        admin_user_ids=None,  # type: ignore[arg-type]
    )
    assert tg_nones.allowed_chat_ids == []
    assert tg_nones.allowed_dm_user_ids == []
    assert tg_nones.admin_user_ids == []

    # bot_api mode validates cleanly both with bot_token alone and with api_id/api_hash
    tg_bot_only = TelegramConfig(mode="bot_api", bot_token="123456:AAFakeTokenValue1234567890")
    assert tg_bot_only.mode == "bot_api"
    assert "AAFakeTokenValue" not in repr(tg_bot_only)

    tg_bot_full = TelegramConfig(
        mode="bot_api",
        api_id=12345678,
        api_hash="0123456789abcdef0123456789abcdef",
    )
    assert tg_bot_full.mode == "bot_api"


def test_tts_config_defaults_match_potuzhnych_male_persona() -> None:
    cfg = TTSConfig()
    assert cfg.provider == "auto"
    assert cfg.fallback_to_edge is True
    assert cfg.gemini_model == "gemini-2.5-flash-preview-tts"
    assert cfg.gemini_ru_voice == "Fenrir"
    assert cfg.gemini_uk_voice == "Fenrir"
    assert cfg.ru_voice == "ru-RU-DmitryNeural"
    assert cfg.uk_voice == "uk-UA-OstapNeural"
    assert cfg.gemini_style_prompt == DEFAULT_MALE_GEMINI_STYLE_PROMPT
    assert "potuzhnych" in cfg.gemini_style_prompt.lower()
    assert "never robotic, monotone, asmr-whispering, flirtatious, or theatrical" in cfg.gemini_style_prompt.lower()
    assert "Потужнич" in BehaviorConfig().bot_names


def test_self_evolution_disabled_by_default_and_subsystems_present() -> None:
    bot_cfg = BotConfig.model_validate({"telegram": {"api_id": 1, "api_hash": "h"}})
    assert isinstance(bot_cfg.self_evolution, SelfEvolutionConfig)
    assert bot_cfg.self_evolution.enabled is False
    assert "telegram-group-bot" not in bot_cfg.self_evolution.staging_dir
    assert "config/userbot.yaml" in bot_cfg.self_evolution.protected_paths

    assert isinstance(bot_cfg.antigravity, AntigravityProviderConfig)
    assert bot_cfg.antigravity.ops_tools_default_mode == "on"
    assert isinstance(bot_cfg.image_generation, ImageGenerationConfig)
    assert bot_cfg.image_generation.send_as_document_threshold_mb == 9.5
    assert isinstance(bot_cfg.quota, QuotaConfig)
    assert isinstance(bot_cfg.models, ModelRouterConfig)
    assert isinstance(bot_cfg.memory.backup, MemoryBackupConfig)
    assert isinstance(bot_cfg.memory.embeddings, MemoryEmbeddingsConfig)
    assert isinstance(bot_cfg.memory.reflection, MemoryReflectionConfig)

    # QuotaConfig percentage aliases
    q_alias = QuotaConfig.model_validate({"economy_threshold_pct": 0.8, "admin_reserve_pct": 0.15})
    assert q_alias.economy_threshold_ratio == 0.8
    assert q_alias.admin_reserve_ratio == 0.15

    # bot.config must not shadow bot.antigravity.AntigravityConfig
    assert hasattr(cfg_pkg, "AntigravityProviderConfig")
    assert not hasattr(cfg_pkg, "AntigravityConfig")
    assert agy_pkg.AntigravityConfig is not AntigravityProviderConfig


def test_memory_embeddings_and_reflection_config_defaults() -> None:
    emb = MemoryEmbeddingsConfig()
    assert emb.enabled is True
    assert emb.provider == "gemini"
    assert emb.gemini_model == "gemini-embedding-001"
    assert emb.output_dimensionality == 768
    assert emb.timeout_seconds == 3.0
    assert emb.min_similarity == 0.52

    with_key = MemoryEmbeddingsConfig(gemini_api_key="AIzaSyDummySecret1234567890", provider="GEMINI")  # type: ignore[arg-type]
    assert with_key.provider == "gemini"
    assert "AIzaSyDummySecret1234567890" not in repr(with_key)

    with pytest.raises(ValidationError):
        MemoryEmbeddingsConfig(output_dimensionality=32)

    refl = MemoryReflectionConfig()
    assert refl.enabled is True
    assert refl.min_active_facts_per_subject == 15
    assert refl.subject_cooldown_hours == 72.0

    mem = MemoryConfig()
    assert mem.backup.enabled is True
    assert mem.embeddings.gemini_model == "gemini-embedding-001"
    assert mem.reflection.event_default_ttl_days == 30


def test_config_manager_loads_userbot_yaml_and_example_cleanly() -> None:
    example_path = REPO_ROOT / "config" / "userbot.example.yaml"
    raw_example = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    cfg_example = BotConfig.model_validate(raw_example)
    assert cfg_example.telegram.mode == "userbot"

    live_userbot_path = REPO_ROOT / "config" / "userbot.yaml"
    if live_userbot_path.exists():
        mgr = ConfigManager(config_path=live_userbot_path)
        assert mgr.bot is not None
        assert mgr.bot.telegram.mode == "userbot"
        assert mgr.bot.telegram.api_id is not None
        assert bool(mgr.bot.telegram.api_hash)
        assert mgr.bot.self_evolution.enabled is False
