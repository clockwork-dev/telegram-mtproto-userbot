"""Unified Command-Line Interface for the Autonomous Telegram Userbot Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from bot.antigravity.auth import run_isolated_antigravity_login
from bot.config.manager import ConfigManager
from bot.main import build_antigravity_config, run_daemon, run_dry_run_smoke
from bot.security.sanitizer import SecretSanitizer, install_log_sanitizer
from bot.telegram.auth import interactive_auth


def load_dotenv_if_present(env_path: Path = Path(".env")) -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
    except (OSError, UnicodeDecodeError):
        pass


def _parse_int_env(key: str) -> Optional[int]:
    """Safely parse integer from environment variable, ignoring empty or non-numeric strings."""
    val = os.environ.get(key, "").strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the bot CLI."""
    default_config = os.environ.get("BOT_CONFIG_PATH", "config/bot.yaml")
    parser = argparse.ArgumentParser(
        prog="telegram-group-bot",
        description="Autonomous Telegram MTProto userbot agent powered by Google Antigravity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # 1. auth-agy
    p_agy = subparsers.add_parser(
        "auth-agy",
        help="Authenticate isolated Antigravity OAuth account in dedicated data directory.",
    )
    p_agy.add_argument(
        "--data-dir",
        default=None,
        help="Path to isolated Antigravity data directory (default: data/antigravity).",
    )
    p_agy.add_argument(
        "--binary",
        default=None,
        help="Path to the 'agy' executable (default: auto-detected).",
    )

    # 2. auth-telegram
    p_tg = subparsers.add_parser(
        "auth-telegram",
        help="Authenticate dedicated Telegram MTProto userbot account and create 0600 .session.",
    )
    p_tg.add_argument("--phone", default=None, help="Telegram phone number (+XXXXXXXXXXX).")
    p_tg.add_argument(
        "--session",
        default=None,
        help="Path to Telethon .session file (overrides config/bot.yaml).",
    )
    p_tg.add_argument(
        "--api-id",
        type=int,
        default=None,
        help="Telegram API ID from my.telegram.org (overrides config/bot.yaml).",
    )
    p_tg.add_argument(
        "--api-hash",
        default=None,
        help="Telegram API Hash from my.telegram.org (overrides config/bot.yaml).",
    )
    p_tg.add_argument(
        "--config",
        default=default_config,
        help=f"Path to bot.yaml config file (default: {default_config}).",
    )
    p_tg.add_argument(
        "--env-file",
        default=None,
        help="Optional path to .env file to load credentials from.",
    )
    p_tg.add_argument(
        "--qr",
        action="store_true",
        help="Authenticate by scanning a QR code in the terminal (no SMS/login code needed).",
    )

    # 3. run
    p_run = subparsers.add_parser(
        "run",
        help="Start the autonomous Telegram group userbot daemon (or run --dry-run smoke test).",
    )
    p_run.add_argument(
        "--config",
        default=default_config,
        help=f"Path to bot.yaml configuration file (default: {default_config}).",
    )
    p_run.add_argument(
        "--env-file",
        default=None,
        help="Optional path to .env file to load credentials from.",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute full-pipeline dry-run smoke test and exit.",
    )
    p_run.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )

    # 4. import-history
    p_imp = subparsers.add_parser(
        "import-history",
        help="Import Telegram Desktop JSON chat history exports into SQLite memory without LLM tokens.",
    )
    p_imp.add_argument(
        "files",
        nargs="+",
        help="One or more paths to Telegram Desktop result.json export files.",
    )
    p_imp.add_argument(
        "--db",
        default=None,
        help="Path to SQLite memory database (defaults to config memory.db_path or data/memory.sqlite).",
    )
    p_imp.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml configuration file (used to load friends.yaml and default db path).",
    )
    p_imp.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Optional chat_id override applied to imported files.",
    )

    # 5. run-digest
    p_digest = subparsers.add_parser(
        "run-digest",
        help="Generate (and optionally send) scheduled news digests on demand or for due cron slots.",
    )
    p_digest.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml configuration file (default: config/bot.yaml).",
    )
    p_digest.add_argument(
        "--slot",
        default=None,
        help="Digest job ID or slot label (e.g. morning_digest, evening_digest) to run immediately.",
    )
    p_digest.add_argument(
        "--all-due",
        action="store_true",
        help="Evaluate all enabled cron_digests due within the lookback window.",
    )
    p_digest.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Optional target chat_id override (defaults to job.chat_ids or allowed_chat_ids).",
    )
    p_digest.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate offline fallback digest preview without LLM calls or Telegram sending.",
    )
    p_digest.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="Lookback window in minutes when evaluating --all-due (default: 60).",
    )
    p_digest.add_argument(
        "--ignore-quiet-hours",
        action="store_true",
        help="Bypass quiet hours check when running --all-due.",
    )
    p_digest.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level (default: INFO).",
    )

    # 6. backup-memory
    p_backup = subparsers.add_parser(
        "backup-memory",
        help="Create a hot SQLite backup of the bot memory database with retention pruning.",
    )
    p_backup.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml configuration file (default: config/bot.yaml).",
    )
    p_backup.add_argument(
        "--db",
        default=None,
        help="Override path to SQLite database (defaults to memory.db_path in config).",
    )
    p_backup.add_argument(
        "--output-dir",
        default=None,
        help="Override backup directory (defaults to memory.backup.backup_dir in config).",
    )
    p_backup.add_argument(
        "--retain",
        type=int,
        default=None,
        help="Number of recent backups to retain (defaults to memory.backup.retain_count).",
    )
    p_backup.add_argument(
        "--no-compress",
        action="store_true",
        help="Store uncompressed .sqlite snapshot instead of .sqlite.gz.",
    )

    # 7. restore-memory
    p_restore = subparsers.add_parser(
        "restore-memory",
        help="Verify and restore a SQLite memory backup (creates a .pre-restore safety copy).",
    )
    p_restore.add_argument(
        "--backup",
        required=True,
        help="Path to .sqlite or .sqlite.gz backup file to restore.",
    )
    p_restore.add_argument(
        "--config",
        default="config/bot.yaml",
        help="Path to bot.yaml configuration file (default: config/bot.yaml).",
    )
    p_restore.add_argument(
        "--db",
        default=None,
        help="Override target SQLite database path (defaults to memory.db_path in config).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run CLI command and return exit status code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "auth-agy":
        return run_isolated_antigravity_login(data_dir=args.data_dir, binary=args.binary)

    if args.command == "auth-telegram":
        env_file_arg = getattr(args, "env_file", None)
        if env_file_arg:
            load_dotenv_if_present(Path(env_file_arg))
        else:
            load_dotenv_if_present()
        cfg_file = Path(args.config).resolve()
        for candidate_env in (cfg_file.parent.parent / ".env", cfg_file.parent / ".env"):
            if candidate_env.is_file():
                load_dotenv_if_present(candidate_env)

        cfg_mgr: Optional[ConfigManager] = None
        try:
            if cfg_file.is_file():
                cfg_mgr = ConfigManager(config_path=cfg_file)
            else:
                cfg_mgr = ConfigManager()
        except Exception:
            pass

        tg_cfg = cfg_mgr.bot.telegram if cfg_mgr and cfg_mgr.bot else None

        session_path = (
            args.session
            or (tg_cfg.session_path if tg_cfg and tg_cfg.session_path else None)
            or os.environ.get("TELEGRAM_SESSION_PATH", "").strip()
            or "data/session/userbot.session"
        )
        env_api_id = _parse_int_env("TELEGRAM_API_ID")
        env_api_hash = os.environ.get("TELEGRAM_API_HASH", "").strip() or None
        cfg_api_id = tg_cfg.api_id if tg_cfg and tg_cfg.api_id else None
        cfg_api_hash = tg_cfg.api_hash if tg_cfg and tg_cfg.api_hash else None

        if env_api_id and (not cfg_api_id or cfg_api_id in (12345, 12345678)):
            api_id = args.api_id or env_api_id
        else:
            api_id = args.api_id or cfg_api_id or env_api_id

        if env_api_hash and (not cfg_api_hash or "0123456789abcdef" in cfg_api_hash):
            api_hash = args.api_hash or env_api_hash
        else:
            api_hash = args.api_hash or cfg_api_hash or env_api_hash

        phone = args.phone or os.environ.get("TELEGRAM_PHONE", "").strip() or None

        dummy_api_ids = {12345, 12345678}
        dummy_api_hash_substrings = ("0123456789abcdef",)

        is_dummy_id = False
        try:
            if api_id is not None and int(api_id) in dummy_api_ids:
                is_dummy_id = True
        except (ValueError, TypeError):
            pass

        is_dummy_hash = False
        if api_hash is not None:
            hash_str = str(api_hash).strip().lower()
            if any(dummy in hash_str for dummy in dummy_api_hash_substrings):
                is_dummy_hash = True

        if not api_id or not api_hash or is_dummy_id or is_dummy_hash:
            print(
                "[ERROR] Missing Telegram API credentials: api_id and api_hash must be set in config or environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)",
                file=sys.stderr,
            )
            return 1

        auth_call_kwargs: dict[str, object] = {
            "session_path": str(session_path),
            "api_id": int(api_id),
            "api_hash": str(api_hash),
            "phone": phone,
        }
        if getattr(args, "qr", False):
            auth_call_kwargs["qr"] = True
        ok = interactive_auth(**auth_call_kwargs)  # type: ignore[arg-type]
        return 0 if ok is not False else 1

    if args.command == "run":
        loaded_env_paths: list[Path] = []
        env_file_arg = getattr(args, "env_file", None)
        if env_file_arg:
            p = Path(env_file_arg)
            load_dotenv_if_present(p)
            if p.is_file():
                loaded_env_paths.append(p)
        else:
            default_env = Path(".env")
            load_dotenv_if_present(default_env)
            if default_env.is_file():
                loaded_env_paths.append(default_env)
            cfg_file = Path(args.config).resolve()
            for candidate_env in (cfg_file.parent.parent / ".env", cfg_file.parent / ".env"):
                if candidate_env.is_file():
                    load_dotenv_if_present(candidate_env)
                    loaded_env_paths.append(candidate_env)

        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        sanitizer = SecretSanitizer()
        for env_p in loaded_env_paths:
            sanitizer.load_env_file(env_p)
        install_log_sanitizer(sanitizer)

        if args.dry_run:
            result = asyncio.run(run_dry_run_smoke(config_path=args.config))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status") == "ok" else 1

        asyncio.run(run_daemon(config_path=args.config))
        return 0

    if args.command == "import-history":
        from dataclasses import asdict

        from bot.memory.db import MemoryStore
        from bot.memory.importer import import_telegram_exports

        cfg_file = Path(args.config).resolve()
        config_dir = cfg_file.parent if cfg_file.is_file() else Path("config").resolve()
        cfg_mgr = (
            ConfigManager(config_path=cfg_file)
            if cfg_file.is_file()
            else ConfigManager(config_dir=config_dir)
        )
        db_path = args.db or (cfg_mgr.bot.memory.db_path if cfg_mgr.bot else "data/memory.sqlite")
        friends_cfg = cfg_mgr.friends
        author_map = {
            int(uid): prof.name for uid, prof in friends_cfg.friends.items() if prof.name
        } if friends_cfg else {}

        async def _run_import() -> list:
            store = MemoryStore(db_path=str(db_path))
            try:
                return await import_telegram_exports(
                    store,
                    args.files,
                    author_map=author_map,
                    override_chat_id=args.chat_id,
                )
            finally:
                await store.close()

        stats = asyncio.run(_run_import())
        print(json.dumps([asdict(s) for s in stats], ensure_ascii=False, indent=2))
        return 0

    if args.command == "backup-memory":
        from bot.memory.backup import backup_sqlite_database

        cfg_file = Path(args.config).resolve()
        cfg_mgr = (
            ConfigManager(config_path=cfg_file)
            if cfg_file.is_file()
            else ConfigManager(config_dir=cfg_file.parent if cfg_file.parent.exists() else Path("config").resolve())
        )
        bot_cfg = cfg_mgr.bot
        db_path = args.db or (bot_cfg.memory.db_path if bot_cfg else "data/memory.sqlite")
        backup_dir = args.output_dir or (bot_cfg.memory.backup.backup_dir if bot_cfg else "data/backups")
        retain = args.retain if args.retain is not None else (bot_cfg.memory.backup.retain_count if bot_cfg else 14)
        compress = False if args.no_compress else (bot_cfg.memory.backup.compress if bot_cfg else True)

        res = backup_sqlite_database(db_path, backup_dir, compress=compress, retain=retain)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "backup_path": str(res.backup_path),
                    "source_db_path": str(res.source_db_path),
                    "size_bytes": res.size_bytes,
                    "compressed": res.compressed,
                    "pruned_count": res.pruned_count,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "restore-memory":
        from bot.memory.backup import restore_sqlite_database

        cfg_file = Path(args.config).resolve()
        cfg_mgr = (
            ConfigManager(config_path=cfg_file)
            if cfg_file.is_file()
            else ConfigManager(config_dir=cfg_file.parent if cfg_file.parent.exists() else Path("config").resolve())
        )
        bot_cfg = cfg_mgr.bot
        target_db = args.db or (bot_cfg.memory.db_path if bot_cfg else "data/memory.sqlite")
        restored = restore_sqlite_database(args.backup, target_db)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "restored_db_path": str(restored),
                    "backup_path": str(Path(args.backup).resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "run-digest":
        import time as time_mod
        from datetime import datetime, timedelta
        from datetime import timezone as dt_timezone

        from bot.antigravity import AntigravityProvider
        from bot.main import GroupBotOrchestrator
        from bot.memory.db import ChatMessage, MemoryStore
        from bot.proactive.cron import CronExpression, parse_timezone
        from bot.proactive.digest import NewsDigestGenerator
        from bot.proactive.engine import ProactiveEngine
        from bot.telegram.bot_api_client import (
            TelegramBotAPIClient,
            resolve_bot_token_from_env_or_config,
        )
        from bot.telegram.client import TelegramUserbot

        cfg_file = Path(args.config).resolve()
        loaded_env_paths = []
        default_env = Path(".env")
        load_dotenv_if_present(default_env)
        if default_env.is_file():
            loaded_env_paths.append(default_env)
        for candidate_env in (cfg_file.parent.parent / ".env", cfg_file.parent / ".env"):
            if candidate_env.is_file():
                load_dotenv_if_present(candidate_env)
                loaded_env_paths.append(candidate_env)

        logging.basicConfig(
            level=getattr(logging, str(args.log_level).upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        sanitizer = SecretSanitizer()
        for env_p in loaded_env_paths:
            sanitizer.load_env_file(env_p)
        install_log_sanitizer(sanitizer)

        config_dir = cfg_file.parent if cfg_file.is_file() else Path("config").resolve()
        cfg_mgr = (
            ConfigManager(config_path=cfg_file)
            if cfg_file.is_file()
            else ConfigManager(config_dir=config_dir)
        )
        bot_cfg = cfg_mgr.bot
        if bot_cfg is None:
            print("[ERROR] Failed to load bot configuration.", file=sys.stderr)
            return 1
        if bot_cfg.telegram.api_hash and "0123456789abcdef" not in bot_cfg.telegram.api_hash.lower():
            sanitizer.register_secret(bot_cfg.telegram.api_hash)
        if bot_cfg.telegram.phone and len(str(bot_cfg.telegram.phone).strip()) >= 7:
            sanitizer.register_secret(str(bot_cfg.telegram.phone).strip())

        proactive_cfg = bot_cfg.proactive
        jobs = list(proactive_cfg.cron_digests or [])
        now_utc = datetime.now(dt_timezone.utc)
        selected_jobs = []

        if args.slot:
            matched_job = next((j for j in jobs if j.id == args.slot), None) or next(
                (j for j in jobs if j.slot_label == args.slot), None
            )
            if matched_job is None:
                available = ", ".join(j.id for j in jobs) or "none"
                print(
                    f"[ERROR] Unknown digest slot '{args.slot}'. Available jobs: {available}",
                    file=sys.stderr,
                )
                return 1
            tz = parse_timezone(matched_job.timezone)
            selected_jobs.append((matched_job, now_utc.astimezone(tz)))
        else:
            window_mins = max(1, int(args.window_minutes))
            since_utc = now_utc - timedelta(minutes=window_mins)
            helper_store = MemoryStore(db_path=":memory:")
            helper_engine = ProactiveEngine(config_mgr=cfg_mgr, memory_store=helper_store)
            for job in jobs:
                if not job.enabled:
                    continue
                cron_expr = CronExpression.parse(job.cron, timezone=job.timezone)
                slots = cron_expr.due_slots(
                    since_utc,
                    now_utc,
                    max_lookback_minutes=window_mins,
                )
                if not args.ignore_quiet_hours and job.respect_quiet_hours:
                    slots = [s for s in slots if not helper_engine.is_in_quiet_hours(s)]
                if slots:
                    selected_jobs.append((job, slots[-1]))

        forbidden_topics = list(proactive_cfg.forbidden_topics or [])
        sensitive_topics = list(
            dict.fromkeys(
                t
                for friend in cfg_mgr.friends.friends.values()
                for t in (friend.sensitive_topics or [])
                if t and t.strip()
            )
        )

        async def _exec_digest() -> tuple[int, dict]:
            db_path = bot_cfg.memory.db_path
            if args.dry_run and not Path(db_path).exists():
                db_path = ":memory:"
            store = MemoryStore(db_path=str(db_path))
            await store.init_db()

            provider = (
                None if args.dry_run else AntigravityProvider(build_antigravity_config(bot_cfg))
            )
            digest_gen = NewsDigestGenerator(provider=provider)

            tg_client = None
            if not args.dry_run and selected_jobs:
                target_all = []
                for job, _ in selected_jobs:
                    if args.chat_id is not None:
                        target_all.append(int(args.chat_id))
                    elif job.chat_ids:
                        target_all.extend(int(c) for c in job.chat_ids)
                    else:
                        target_all.extend(int(c) for c in bot_cfg.telegram.allowed_chat_ids)
                target_all = list(dict.fromkeys(target_all))

                resolved_token = resolve_bot_token_from_env_or_config(bot_cfg.telegram.bot_token)
                if resolved_token:
                    sanitizer.register_secret(resolved_token)
                use_bot_api = bot_cfg.telegram.mode == "bot_api" or (
                    bot_cfg.telegram.mode == "auto" and bool(resolved_token)
                )
                if use_bot_api and resolved_token:
                    tg_client = TelegramBotAPIClient(
                        bot_token=resolved_token,
                        allowed_chat_ids=target_all,
                        bot_names=bot_cfg.behavior.bot_names,
                        typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                        typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                        typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                        sanitizer=sanitizer,
                    )
                else:
                    tg_client = TelegramUserbot(
                        session_path=bot_cfg.telegram.session_path or "data/session/userbot.session",
                        api_id=bot_cfg.telegram.api_id or _parse_int_env("TELEGRAM_API_ID") or 0,
                        api_hash=bot_cfg.telegram.api_hash or os.environ.get("TELEGRAM_API_HASH", ""),
                        allowed_chat_ids=target_all,
                        allowed_dm_user_ids=GroupBotOrchestrator._effective_dm_user_ids(bot_cfg),
                        ignored_sender_usernames=bot_cfg.telegram.ignored_sender_usernames,
                        bot_names=bot_cfg.behavior.bot_names,
                        typing_cpm=bot_cfg.behavior.typing_speed_cpm,
                        typing_min_seconds=bot_cfg.behavior.typing_min_seconds,
                        typing_max_seconds=bot_cfg.behavior.typing_max_seconds,
                    )
                await tg_client.start()

            results = []
            had_error = False
            try:
                bot_name = (
                    bot_cfg.behavior.bot_names[0] if bot_cfg.behavior.bot_names else "Потужнич"
                )
                for job, slot_dt in selected_jobs:
                    if args.chat_id is not None:
                        target_chats = [int(args.chat_id)]
                    elif job.chat_ids:
                        target_chats = list(dict.fromkeys(int(c) for c in job.chat_ids))
                    else:
                        target_chats = list(
                            dict.fromkeys(int(c) for c in bot_cfg.telegram.allowed_chat_ids)
                        )

                    for cid in target_chats:
                        recent = await store.get_recent_history(cid, limit=10)
                        text = await digest_gen.generate_digest(
                            slot_name=job.id,
                            slot_label=job.slot_label,
                            timezone=job.timezone,
                            now_dt=slot_dt,
                            topics=job.topics or None,
                            prompt_hint=job.prompt_hint,
                            recent_messages=recent,
                            forbidden_topics=forbidden_topics,
                            sensitive_topics=sensitive_topics,
                            humor_style=proactive_cfg.humor_style,
                            fallback_when_no_provider=bool(args.dry_run),
                        )
                        sent = False
                        if not text:
                            had_error = True
                        elif not args.dry_run and tg_client is not None:
                            safe_text = sanitizer.sanitize(text)
                            sent_res = await tg_client.send_reply(
                                chat_id=cid,
                                text=safe_text,
                                reply_to_msg_id=None,
                                simulate_typing=bool(bot_cfg.behavior.typing_simulation),
                            )
                            sent = True
                            text = safe_text
                            sent_msg_id = GroupBotOrchestrator._extract_sent_message_id(sent_res)
                            await store.save_message(
                                ChatMessage(
                                    telegram_msg_id=sent_msg_id,
                                    chat_id=cid,
                                    user_id=tg_client.bot_id or 1,
                                    timestamp=int(time_mod.time()),
                                    author_name=bot_name,
                                    text=text,
                                    reply_to_msg_id=None,
                                )
                            )

                        results.append(
                            {
                                "job_id": job.id,
                                "slot_label": job.slot_label,
                                "slot_time": slot_dt.isoformat(),
                                "chat_id": cid,
                                "sent": sent,
                                "text": text,
                            }
                        )
            finally:
                if tg_client is not None:
                    await tg_client.stop()
                if provider is not None and hasattr(provider, "close"):
                    provider.close()
                await store.close()

            status = "error" if had_error else "ok"
            return (1 if had_error else 0), {
                "status": status,
                "dry_run": bool(args.dry_run),
                "sent": any(bool(r.get("sent")) for r in results),
                "results": results,
            }

        rc, payload = asyncio.run(_exec_digest())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return rc

    parser.print_help()
    return 1


def cli_entrypoint() -> None:
    """Console script entrypoint for setuptools."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli_entrypoint()
