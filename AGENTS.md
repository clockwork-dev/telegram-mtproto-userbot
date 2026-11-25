# AGENTS.md — AI Assistant Guide (`telegram-mtproto-userbot`, MTProto / Telethon mode)

## 1. Repository Scope
This repository contains the **Telegram MTProto Userbot agent** (`mode: "userbot"` via Telethon, persona **«Потужнич»** / `Potuzhnych`), deployed on production servers under `/opt/telegram-mtproto-userbot` (`telegram-mtproto-userbot.service`, `User=botops`, `Group=botops`).

**Strictly out of scope for this repo:**
- The sibling **classic Bot API agent** (`telegram-group-bot`) lives in a separate repository and runs isolated under `/opt/telegram-group-bot` (`telegram-userbot.service`). Never modify, overwrite, or couple code/scripts to `/opt/telegram-group-bot`.
- Never commit `.env`, `config/userbot.yaml`, `config/userbot_system_prompt.md`, `config/friends.yaml`, `data/`, `*.session*`, `*.sqlite*`, or OAuth/API tokens. Use `config/userbot.example.yaml`, `config/userbot_system_prompt.example.md`, `config/friends.example.yaml`, and `.env.example`.

## 2. Architecture & Key Modules
- **CLI & Auth**: `src/bot/cli.py` (`auth-telegram` including `--qr`, `auth-agy`, `run`, `run-digest`, `backup-memory`, `restore-memory`), `src/bot/telegram/auth.py` (interactive terminal QR-code login & phone/code/2FA login, strict `0700`/`0600` session permissions), `deploy/reauth_userbot.sh` (interactive re-authentication & secret rotation wizard)
- **Orchestrator**: `src/bot/main.py` (`GroupBotOrchestrator`)
- **MTProto Transport**:
  - `src/bot/telegram/client.py` (`TelegramUserbot` — Telethon client, `FloodWaitError` handling, `send_read_acknowledge`, bounded startup backfill, media download, `send_typing_action`, `send_photo`, `send_voice`, `send_document`, `send_reaction`, `sync_access_lists`)
  - `src/bot/telegram/handlers.py` (`IncomingMessage`, async `event.get_reply_message()` enrichment including `reply_to_media_event` for img2img & replied voice context)
  - `src/bot/telegram/heuristics.py` (`ReplyHeuristics` — strict DM whitelist `allowed_dm_user_ids` / `admin_user_ids` with `MUST_REPLY`, group mention/reply/spontaneous rules, anti-loop `ignored_sender_usernames`)
- **Antigravity Provider & Tool Isolation**:
  - `src/bot/antigravity/` (isolated `data/antigravity` OAuth home, `ToolProfile`: `"chat"`, `"ops"`, `"self_dev"`, `"image"`; `ALWAYS_FORBIDDEN_TOOLS`; per-model circuit breaker; isolated `0700` image-generation sandbox with fail-closed `.agents/hooks.json` `PreToolUse` firewall)
- **Security & Quota**:
  - `src/bot/security/sanitizer.py` (`SecretSanitizer`, `SanitizingLogFilter`)
  - `src/bot/quota/manager.py` (`QuotaBudgetManager`, `usage_history` SQLite tracking, per-model/bucket cooldowns, economy mode, `/quota` admin command)
- **Media, Voice & Image Generation**:
  - `src/bot/media/` (`faster-whisper` `base`/`int8` lazy STT for `ru`/`uk`, expressive Gemini TTS `gemini-2.5-flash-preview-tts` with male voice `Fenrir` + `[MOOD: ...]` / inline vocal tags & Edge TTS `ru-RU-DmitryNeural` / `uk-UA-OstapNeural` fallback, `image_gen.py` for `/imagine`, `/img`, and `[IMG:{...}]`)
- **Hindsight-lite Memory, Backups & Proactive**:
  - `src/bot/memory/` (hybrid `FTS5` + `gemini-embedding-001` `dim=768` vector search with RRF `k=60`, `add | update | supersede` compaction, `ReflectionEngine` consolidation & TTL archival, hot SQLite WAL gzip backups in `data/backups/` & `/backup` admin command)
  - `src/bot/proactive/` (cron digests & banter)
- **Deployment**: `systemd/telegram-mtproto-userbot.service` (guarded by `ConditionPathExists=/opt/telegram-mtproto-userbot/data/session/userbot.session`), `deploy/deploy_userbot.sh`, `deploy/deploy_remote.sh`, `deploy/reauth_userbot.sh`

## 3. Critical Invariants (Do Not Break)
1. **Strict DM, Group & Tool Profile Boundaries**:
   - Private messages outside `allowed_dm_user_ids` / `admin_user_ids` and groups outside `allowed_chat_ids` MUST be ignored without calling the LLM.
   - Bots in `ignored_sender_usernames` (e.g. sibling Bot API bot) and outgoing messages MUST be ignored to prevent bot-to-bot loops.
   - Proactive loops, compaction, and non-ops turns MUST strictly use `tool_profile="chat"` (`{"search_web", "read_url_content"}`); image generation runs strictly under `tool_profile="image"` with `.agents/hooks.json`.
   - Keep `self_evolution.enabled: false` so the userbot never writes to another service's self-deploy trigger.
2. **Session, Secret & Account Safety**:
   - Always maintain `0700` on `data/session` and `0600` on `*.session*` and `.env` files.
   - All outbound messages, captions, admin alerts, and logs pass through `SecretSanitizer`.
   - Preserve `FloodWaitError` backoff/handling so Telegram rate limits never crash the daemon.
3. **Antigravity Isolation**:
   - Keep `ANTIGRAVITY_DATA_DIR=data/antigravity` dedicated to this service so OAuth token refreshes never race with other processes.

## 4. Verification & Development Commands
```bash
# Lint
ruff check .

# Shell syntax check
bash -n deploy/*.sh

# Full test suite
PYTHONPATH=src pytest -q

# Offline dry-run smoke test
PYTHONPATH=src python -m bot.cli run --config config/userbot.example.yaml --dry-run

# Interactive Telegram MTProto QR or code login (on target host)
telegram-group-bot auth-telegram --config config/userbot.yaml --env-file .env --qr
```

## 5. Deployment Workflow
- Deploy to `/opt/telegram-mtproto-userbot` via `REMOTE_HOST=user@host ./deploy/deploy_userbot.sh` (uses SSH keys or `SSH_PASSWORD` env var; never hardcode credentials).
- Wait for `userbot.session` authorization (or run `./deploy/reauth_userbot.sh`) before enabling `telegram-mtproto-userbot.service`.
