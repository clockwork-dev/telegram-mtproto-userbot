"""Interactive authentication helper for Telegram MTProto Userbot."""

import asyncio
import getpass
import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import qrcode
from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PasswordHashInvalidError,
    PhoneCodeExpiredError,
    PhoneCodeInvalidError,
    SessionPasswordNeededError,
)

logger = logging.getLogger(__name__)


def render_qr_ascii(url: str) -> str:
    """Render a compact Unicode terminal QR code for a login URL."""
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    buf = io.StringIO()
    qr.print_ascii(out=buf, invert=True)
    return buf.getvalue()


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
    qr: bool = False,
    qr_timeout: float = 30.0,
    max_qr_attempts: int = 6,
    qr_display_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """Authenticate a Telegram userbot session asynchronously.

    Supports both QR-code login (`qr=True`) and phone/login-code flow (`qr=False`),
    plus optional 2FA Cloud Password. Enforces strict 0700/0600 permissions.
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

        if qr:
            qr_login = await client.qr_login()
            for attempt in range(1, max(1, max_qr_attempts) + 1):
                if attempt > 1:
                    await qr_login.recreate()
                if qr_display_callback is not None:
                    qr_display_callback(qr_login.url, attempt, max_qr_attempts)
                else:
                    if attempt > 1:
                        print(f"\n[QR-код обновлён ({attempt}/{max_qr_attempts}) — отсканируй свежий код]")
                    else:
                        print("\nОткрой Telegram на телефоне: Настройки -> Устройства -> Подключить устройство")
                    print(render_qr_ascii(qr_login.url))
                    print(f"Прямая ссылка (для Desktop): {qr_login.url}\n")

                try:
                    await qr_login.wait(timeout=qr_timeout)
                    break
                except SessionPasswordNeededError:
                    logger.info("Two-step verification (2FA) is enabled for this account.")
                    if password_callback:
                        pwd = password_callback()
                    else:
                        pwd = getpass.getpass("Enter your Telegram 2FA Cloud Password: ")
                    await client.sign_in(password=pwd)
                    break
                except asyncio.TimeoutError as exc:
                    if attempt >= max(1, max_qr_attempts):
                        raise TimeoutError("QR login timed out before code was scanned.") from exc
                    continue
        else:
            if not phone:
                phone = input("Enter Telegram phone number (+XXXXXXXXXXX): ").strip()

            logger.info("Requesting login code for %s...", phone)
            sent = await client.send_code_request(phone)
            sent_type = getattr(sent, "type", None)
            if sent_type is not None and type(sent_type).__module__.startswith("telethon"):
                print(f"[i] Code delivery method: {type(sent_type).__name__}")

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
    qr: bool = False,
) -> bool:
    """Terminal command runner for auth-telegram."""
    print("=" * 60)
    if qr:
        print(" Telegram MTProto Userbot QR-Code Authentication")
    else:
        print(" Telegram MTProto Userbot Interactive Authentication")
    print("=" * 60)

    try:
        auth_kwargs: Dict[str, Any] = {
            "session_path": session_path,
            "api_id": api_id,
            "api_hash": api_hash,
            "phone": phone,
        }
        if qr:
            auth_kwargs["qr"] = True
        user_info = asyncio.run(authenticate_session(**auth_kwargs))
        print("\n[SUCCESS] Telegram userbot authenticated successfully!")
        print(f" User ID:    {user_info.get('user_id')}")
        print(f" Username:   @{user_info.get('username')}")
        print(f" First Name: {user_info.get('first_name')}")
        print(f" Session:    {session_path} (secured with 0600 permissions)")
        return True
    except FloodWaitError as e:
        print(f"\n[ERROR] Telegram rate limit: FloodWait for {e.seconds} seconds.")
    except (PhoneCodeInvalidError, PhoneCodeExpiredError) as e:
        print(f"\n[ERROR] Invalid or expired code: {e}")
    except PasswordHashInvalidError as e:
        print(f"\n[ERROR] Incorrect 2FA password: {e}")
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}")
    return False
