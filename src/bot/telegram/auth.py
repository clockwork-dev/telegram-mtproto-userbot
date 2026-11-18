"""Interactive authentication helper for Telegram MTProto Userbot."""

import asyncio
import getpass
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    SessionPasswordNeededError,
)

logger = logging.getLogger(__name__)


def enforce_session_permissions(session_path: str | Path) -> None:
    """Enforce 0700 on the session directory and 0600 on the session file."""
    path = Path(session_path).resolve()
    session_dir = path.parent
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(session_dir, 0o700)
    except OSError as err:
        logger.debug("Failed to chmod 0700 session directory %s: %s", session_dir, err)

    base_stem = path.stem if path.suffix == ".session" else path.name

    # Telethon creates files with .session extension and SQLite journal/wal/shm sidecars
    candidate_files = [
        path,
        path.with_suffix(".session"),
        Path(f"{path}.session"),
        session_dir / f"{base_stem}.session-journal",
        session_dir / f"{base_stem}.session-wal",
        session_dir / f"{base_stem}.session-shm",
        session_dir / f"{path.name}-journal",
        session_dir / f"{path.name}-wal",
        session_dir / f"{path.name}-shm",
    ]

    try:
        for f in session_dir.glob(f"{base_stem}.session*"):
            if f.is_file() and f not in candidate_files:
                candidate_files.append(f)
    except OSError:
        pass

    for candidate in candidate_files:
        if candidate.exists() and candidate.is_file():
            try:
                os.chmod(candidate, 0o600)
            except OSError as err:
                logger.debug("Failed to chmod 0600 on session file %s: %s", candidate, err)


async def authenticate_session(
    session_path: str | Path,
    api_id: int,
    api_hash: str,
    phone: Optional[str] = None,
    code_callback: Optional[Callable[[], str]] = None,
    password_callback: Optional[Callable[[], str]] = None,
) -> Dict[str, Any]:
    """Authenticate a Telegram userbot session asynchronously.

    Handles phone verification, login code, and optional 2FA password.
    Enforces strict 0700/0600 permissions.
    """
    path = Path(session_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)

    # Telethon expects session name without .session suffix
    session_name = str(path.with_suffix("")) if path.suffix == ".session" else str(path)

    client = TelegramClient(session_name, api_id=api_id, api_hash=api_hash)
    await client.connect()

    try:
        if await client.is_user_authorized():
            me = await client.get_me()
            enforce_session_permissions(session_path)
            return {
                "authorized": True,
                "user_id": me.id,
                "username": getattr(me, "username", None),
                "phone": getattr(me, "phone", None),
                "first_name": getattr(me, "first_name", ""),
            }

        if not phone:
            phone = input("Enter Telegram phone number (+XXXXXXXXXXX): ").strip()

        logger.info("Requesting login code for %s...", phone)
        await client.send_code_request(phone)

        if code_callback:
            code = code_callback()
        else:
            code = input("Enter the login code received in Telegram: ").strip()

        try:
            await client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            logger.info("Two-step verification (2FA) is enabled for this account.")
            if password_callback:
                pwd = password_callback()
            else:
                pwd = getpass.getpass("Enter your Telegram 2FA Cloud Password: ")
            await client.sign_in(password=pwd)

        me = await client.get_me()
        enforce_session_permissions(session_path)

        return {
            "authorized": True,
            "user_id": me.id,
            "username": getattr(me, "username", None),
            "phone": getattr(me, "phone", None),
            "first_name": getattr(me, "first_name", ""),
        }
    finally:
        await client.disconnect()
        enforce_session_permissions(session_path)


def interactive_auth(
    session_path: str,
    api_id: int,
    api_hash: str,
    phone: Optional[str] = None,
) -> None:
    """Terminal command runner for auth-telegram."""
    print("=" * 60)
    print(" Telegram MTProto Userbot Interactive Authentication")
    print("=" * 60)

    try:
        user_info = asyncio.run(
            authenticate_session(
                session_path=session_path,
                api_id=api_id,
                api_hash=api_hash,
                phone=phone,
            )
        )
        print("\n[SUCCESS] Telegram userbot authenticated successfully!")
        print(f" User ID:    {user_info.get('user_id')}")
        print(f" Username:   @{user_info.get('username')}")
        print(f" First Name: {user_info.get('first_name')}")
        print(f" Session:    {session_path} (secured with 0600 permissions)")
    except FloodWaitError as e:
        print(f"\n[ERROR] Telegram rate limit: FloodWait for {e.seconds} seconds.")
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as e:
        print(f"\n[ERROR] Invalid or expired code: {e}")
    except PasswordHashInvalidError as e:
        print(f"\n[ERROR] Incorrect 2FA password: {e}")
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}")
