# Telegram MTProto Userbot (`telegram-mtproto-userbot`)

Autonomous Telegram user account agent powered by **Telethon (MTProto)** and the **Antigravity SDK/CLI**, featuring isolated OAuth storage, Russian/Ukrainian Whisper STT + Edge TTS voice replies, SQLite FTS5 memory, strict DM/group whitelisting, and anti-ban `FloodWaitError` protection.

> Note: This repository is standalone for the **MTProto Userbot** (`mode: "userbot"`). The classic Bot API service lives in the separate `telegram-group-bot` repository.

## Features
- **MTProto / Telethon Transport**: Runs as a dedicated Telegram user account (`auth-telegram`) with `0700`/`0600` `.session` permissions, `FloodWaitError` handling, typing/recording indicators, reactions, voice notes, and documents.
- **DM Whitelist & Group Anti-Loop**: Strict `allowed_dm_user_ids` (`MUST_REPLY`), separate `allowed_chat_ids`, and `ignored_sender_usernames` to prevent bot-to-bot loops.
- **Voice & Media**: Lazy-loaded `faster-whisper` (`base`, `int8`) STT + Edge TTS voice replies, plus image/sticker/PDF/DOCX/XLSX analysis.
- **Isolated Antigravity OAuth**: Dedicated `data/antigravity` home per service.

## Quick Start
1. Copy templates and fill in your credentials/IDs locally (never commit `.env` or runtime configs):
   ```bash
   cp .env.example .env
   cp config/userbot.example.yaml config/userbot.yaml
   cp config/userbot_system_prompt.example.md config/userbot_system_prompt.md
   ```
2. Install dependencies (`python >= 3.11`, `ffmpeg`):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   ```
3. Authenticate Antigravity OAuth & Telegram MTProto session:
   ```bash
   .venv/bin/telegram-group-bot auth-agy --data-dir data/antigravity
   .venv/bin/telegram-group-bot auth-telegram --config config/userbot.yaml --env-file .env
   ```
4. Verify offline and start systemd unit:
   ```bash
   .venv/bin/ruff check .
   .venv/bin/python -m pytest -q
   .venv/bin/telegram-group-bot run --config config/userbot.yaml --dry-run
   ```

## AI Assistant Documentation
See [AGENTS.md](AGENTS.md) for architecture contracts, invariants, and verification commands.
