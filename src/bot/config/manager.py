"""Configuration Manager with Async Mtime Hot-Reload and Profile Lookup."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from bot.config.models import BotConfig, FriendProfile, FriendsConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages bot configuration, friends directory, system prompt, and async hot-reload."""

    def __init__(
        self,
        config_dir: Union[str, Path] = "config",
        config_path: Optional[Union[str, Path]] = None,
        system_prompt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if config_path is not None:
            self.bot_path = Path(config_path).resolve()
            self.config_dir = self.bot_path.parent
        elif Path(config_dir).is_file():
            self.bot_path = Path(config_dir).resolve()
            self.config_dir = self.bot_path.parent
        else:
            self.config_dir = Path(config_dir).resolve()
            for candidate_name in ("bot.yaml", "userbot.yaml", "userbot.example.yaml"):
                cand = self.config_dir / candidate_name
                if cand.exists():
                    self.bot_path = cand
                    break
            else:
                self.bot_path = self.config_dir / "userbot.yaml"

        self.friends_path = self.config_dir / "friends.yaml"
        if system_prompt_path is not None:
            self.system_prompt_path = Path(system_prompt_path).resolve()
        else:
            for sp_name in (
                "userbot_system_prompt.md",
                "userbot_system_prompt.example.md",
                "system_prompt.md",
            ):
                sp_cand = self.config_dir / sp_name
                if sp_cand.exists():
                    self.system_prompt_path = sp_cand
                    break
            else:
                self.system_prompt_path = self.config_dir / "userbot_system_prompt.md"

        self.bot: Optional[BotConfig] = None
        self.friends: FriendsConfig = FriendsConfig()
        self.system_prompt: str = ""

        self._mtimes: Dict[Path, float] = {}
        self._watcher_task: Optional[asyncio.Task] = None
        self._running: bool = False

        self.load_all()

    def _get_mtime(self, path: Path) -> float:
        try:
            return os.stat(path).st_mtime
        except OSError:
            return 0.0

    def load_bot(self) -> BotConfig:
        """Load and validate bot.yaml."""
        if not self.bot_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.bot_path}")
        with open(self.bot_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML content in {self.bot_path}")
        return BotConfig.model_validate(raw)

    def load_friends(self) -> FriendsConfig:
        """Load and validate friends.yaml."""
        if not self.friends_path.exists():
            return FriendsConfig()
        with open(self.friends_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            return FriendsConfig()
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid YAML content in {self.friends_path}")
        # Support either top-level friends key or direct user_id mapping
        if "friends" in raw:
            return FriendsConfig.model_validate(raw)
        return FriendsConfig.model_validate({"friends": raw})

    def load_system_prompt(self) -> str:
        """Load system prompt markdown text."""
        if not self.system_prompt_path.exists():
            return ""
        return self.system_prompt_path.read_text(encoding="utf-8").strip()

    def load_all(self) -> None:
        """Load all configuration files synchronously on startup."""
        self.bot = self.load_bot()
        self.friends = self.load_friends()
        self.system_prompt = self.load_system_prompt()

        self._mtimes[self.bot_path] = self._get_mtime(self.bot_path)
        self._mtimes[self.friends_path] = self._get_mtime(self.friends_path)
        self._mtimes[self.system_prompt_path] = self._get_mtime(self.system_prompt_path)

    def get_friend_profile(self, user_id: int) -> Optional[FriendProfile]:
        """Look up friend profile by Telegram user ID."""
        try:
            uid = int(user_id)
        except (ValueError, TypeError):
            return None
        return self.friends.friends.get(uid)

    def reload_changed(self) -> bool:
        """Check file mtimes and reload any updated configuration file atomically."""
        updated = False

        # 1. bot.yaml
        cur_bot_mtime = self._get_mtime(self.bot_path)
        if cur_bot_mtime != self._mtimes.get(self.bot_path):
            try:
                new_bot = self.load_bot()
                self.bot = new_bot
                self._mtimes[self.bot_path] = cur_bot_mtime
                logger.info("Hot-reloaded %s successfully", self.bot_path)
                updated = True
            except Exception as e:
                logger.warning(
                    "Failed to reload %s, retaining existing valid config: %s",
                    self.bot_path,
                    e,
                )

        # 2. friends.yaml
        cur_friends_mtime = self._get_mtime(self.friends_path)
        if cur_friends_mtime != self._mtimes.get(self.friends_path):
            try:
                new_friends = self.load_friends()
                self.friends = new_friends
                self._mtimes[self.friends_path] = cur_friends_mtime
                logger.info("Hot-reloaded %s successfully", self.friends_path)
                updated = True
            except Exception as e:
                logger.warning(
                    "Failed to reload %s, retaining existing valid config: %s",
                    self.friends_path,
                    e,
                )

        # 3. system_prompt.md
        cur_prompt_mtime = self._get_mtime(self.system_prompt_path)
        if cur_prompt_mtime != self._mtimes.get(self.system_prompt_path):
            try:
                new_prompt = self.load_system_prompt()
                self.system_prompt = new_prompt
                self._mtimes[self.system_prompt_path] = cur_prompt_mtime
                logger.info("Hot-reloaded %s successfully", self.system_prompt_path)
                updated = True
            except Exception as e:
                logger.warning(
                    "Failed to reload %s, retaining existing system prompt: %s",
                    self.system_prompt_path,
                    e,
                )

        return updated

    async def watch_loop(self, interval: float = 2.0) -> None:
        """Asynchronous polling loop checking mtime every `interval` seconds."""
        self._running = True
        logger.info("Starting config watcher loop (interval: %.1fs)", interval)
        while self._running:
            try:
                await asyncio.sleep(interval)
                self.reload_changed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Unexpected error in config watcher: %s", e)

    async def start_watcher(self, interval: float = 2.0) -> asyncio.Task:
        """Start async background watcher task."""
        if self._watcher_task is not None and not self._watcher_task.done():
            return self._watcher_task
        self._watcher_task = asyncio.create_task(self.watch_loop(interval))
        return self._watcher_task

    def stop_watcher(self) -> None:
        """Stop watcher task cleanly."""
        self._running = False
        if self._watcher_task is not None:
            self._watcher_task.cancel()
            self._watcher_task = None
