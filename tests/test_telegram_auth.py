"""Unit tests for Telegram userbot authentication flows and permission enforcement."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from telethon.errors import (
    FloodWaitError,
    SessionPasswordNeededError,
)

from bot.telegram.auth import authenticate_session, interactive_auth


@pytest.mark.asyncio
async def test_authenticate_session_already_authorized(tmp_path):
    session_file = tmp_path / "already_auth.session"
    session_file.write_text("session")

    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.is_user_authorized = AsyncMock(return_value=True)

    mock_me = MagicMock()
    mock_me.id = 777
    mock_me.username = "bot_hero"
    mock_me.first_name = "Bot"
    mock_me.phone = "+1234567890"
    mock_client.get_me = AsyncMock(return_value=mock_me)

    with patch("bot.telegram.auth.TelegramClient", return_value=mock_client):
        res = await authenticate_session(
            session_path=session_file,
            api_id=123,
            api_hash="abc",
        )

    assert res["authorized"] is True
    assert res["user_id"] == 777
    assert res["username"] == "bot_hero"
    mock_client.connect.assert_awaited_once()
    mock_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_authenticate_session_standard_login(tmp_path):
    session_file = tmp_path / "standard.session"
    session_file.write_text("session")

    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.is_user_authorized = AsyncMock(return_value=False)
    mock_client.send_code_request = AsyncMock()
    mock_client.sign_in = AsyncMock()

    mock_me = MagicMock()
    mock_me.id = 888
    mock_me.username = "new_bot"
    mock_me.first_name = "New"
    mock_me.phone = "+9876543210"
    mock_client.get_me = AsyncMock(return_value=mock_me)

    with patch("bot.telegram.auth.TelegramClient", return_value=mock_client):
        res = await authenticate_session(
            session_path=session_file,
            api_id=123,
            api_hash="abc",
            phone="+9876543210",
            code_callback=lambda: "12345",
        )

    assert res["authorized"] is True
    assert res["user_id"] == 888
    mock_client.send_code_request.assert_awaited_once_with("+9876543210")
    mock_client.sign_in.assert_awaited_once_with(phone="+9876543210", code="12345")


@pytest.mark.asyncio
async def test_authenticate_session_with_2fa(tmp_path):
    session_file = tmp_path / "twofa.session"
    session_file.write_text("session")

    mock_client = MagicMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.is_user_authorized = AsyncMock(return_value=False)
    mock_client.send_code_request = AsyncMock()

    # First sign_in call raises SessionPasswordNeededError, second succeeds
    mock_client.sign_in = AsyncMock(
        side_effect=[
            SessionPasswordNeededError(request=MagicMock()),
            MagicMock(),
        ]
    )

    mock_me = MagicMock()
    mock_me.id = 999
    mock_me.username = "secure_bot"
    mock_me.first_name = "Secure"
    mock_me.phone = "+1122334455"
    mock_client.get_me = AsyncMock(return_value=mock_me)

    with patch("bot.telegram.auth.TelegramClient", return_value=mock_client):
        res = await authenticate_session(
            session_path=session_file,
            api_id=123,
            api_hash="abc",
            phone="+1122334455",
            code_callback=lambda: "54321",
            password_callback=lambda: "my_secret_2fa",
        )

    assert res["authorized"] is True
    assert res["user_id"] == 999
    assert mock_client.sign_in.await_count == 2
    mock_client.sign_in.assert_awaited_with(password="my_secret_2fa")


def test_interactive_auth_error_handling(capsys):
    err = FloodWaitError(request=None)
    err.seconds = 42
    with patch(
        "bot.telegram.auth.authenticate_session",
        side_effect=err,
    ):
        interactive_auth(
            session_path="dummy.session",
            api_id=123,
            api_hash="abc",
        )

    captured = capsys.readouterr()
    assert "FloodWait for 42 seconds" in captured.out


def test_interactive_auth_passes_phone_to_authenticate_session():
    """Verify interactive_auth forwards phone argument to authenticate_session."""
    with patch("bot.telegram.auth.authenticate_session", return_value={"user_id": 123, "username": "u"}) as mock_auth:
        interactive_auth(
            session_path="data/test.session",
            api_id=123,
            api_hash="abc",
            phone="+1234567890",
        )
    mock_auth.assert_called_once_with(
        session_path="data/test.session",
        api_id=123,
        api_hash="abc",
        phone="+1234567890",
    )


def test_enforce_session_permissions_all_sidecars(tmp_path):
    """Verify enforce_session_permissions sets 0700 dir and 0600 on session and sidecars."""
    import stat

    from bot.telegram.auth import enforce_session_permissions

    session_dir = tmp_path / "sessions_test"
    session_file = session_dir / "userbot.session"
    session_journal = session_dir / "userbot.session-journal"
    session_wal = session_dir / "userbot.session-wal"
    session_shm = session_dir / "userbot.session-shm"

    # Directory doesn't exist yet; enforce should create it with 0700
    enforce_session_permissions(session_file)
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700

    # Write files with open permissions
    for f in (session_file, session_journal, session_wal, session_shm):
        f.write_text("data")
        f.chmod(0o666)

    enforce_session_permissions(session_file)
    for f in (session_file, session_journal, session_wal, session_shm):
        assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_cli_auth_telegram_custom_config_and_empty_env(tmp_path, monkeypatch):
    """Verify CLI auth-telegram loads userbot.yaml and tolerates empty string env vars."""
    import yaml

    from bot.cli import main

    cfg_file = tmp_path / "userbot.yaml"
    cfg_data = {
        "telegram": {
            "api_id": 98765,
            "api_hash": "custom_userbot_hash",
            "session_path": str(tmp_path / "userbot.session"),
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    # Set empty string env vars which previously crashed int(os.environ["TELEGRAM_API_ID"])
    monkeypatch.setenv("TELEGRAM_API_ID", "")
    monkeypatch.setenv("TELEGRAM_API_HASH", "")

    with patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(["auth-telegram", "--config", str(cfg_file), "--phone", "+380501234567"])
        assert ret == 0
        mock_interactive.assert_called_once_with(
            session_path=str(tmp_path / "userbot.session"),
            api_id=98765,
            api_hash="custom_userbot_hash",
            phone="+380501234567",
        )


def test_cli_auth_telegram_env_file_override(tmp_path, monkeypatch):
    """Verify CLI auth-telegram with --env-file overrides placeholder config values."""
    import yaml

    from bot.cli import main

    cfg_file = tmp_path / "userbot.yaml"
    cfg_data = {
        "telegram": {
            "api_id": 12345678,  # placeholder
            "api_hash": "0123456789abcdef0123456789abcdef",  # placeholder
            "session_path": str(tmp_path / "userbot.session"),
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    env_file = tmp_path / ".env.userbot"
    env_file.write_text(
        "TELEGRAM_API_ID=445566\n"
        "TELEGRAM_API_HASH=real_secret_hash_value\n"
        "TELEGRAM_PHONE=+380507778899\n",
        encoding="utf-8",
    )

    # Clear env vars
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("TELEGRAM_PHONE", raising=False)

    with patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(["auth-telegram", "--config", str(cfg_file), "--env-file", str(env_file)])
        assert ret == 0
        mock_interactive.assert_called_once_with(
            session_path=str(tmp_path / "userbot.session"),
            api_id=445566,
            api_hash="real_secret_hash_value",
            phone="+380507778899",
        )


def test_cli_auth_telegram_bot_config_path_env(tmp_path, monkeypatch):
    """Verify BOT_CONFIG_PATH env variable is respected as default config."""
    import yaml

    from bot.cli import main

    cfg_file = tmp_path / "custom_bot.yaml"
    cfg_data = {
        "telegram": {
            "api_id": 771122,
            "api_hash": "bot_config_hash",
            "session_path": str(tmp_path / "custom.session"),
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.setenv("BOT_CONFIG_PATH", str(cfg_file))
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("TELEGRAM_PHONE", raising=False)

    with patch("bot.cli.load_dotenv_if_present"), patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(["auth-telegram"])
        assert ret == 0
        mock_interactive.assert_called_once_with(
            session_path=str(tmp_path / "custom.session"),
            api_id=771122,
            api_hash="bot_config_hash",
            phone=None,
        )


def test_cli_auth_telegram_missing_credentials_fails(capsys, monkeypatch, tmp_path):
    """Verify CLI auth-telegram fails with error code 1 when credentials are missing."""
    from bot.cli import main

    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("TELEGRAM_PHONE", raising=False)

    empty_cfg = tmp_path / "empty_cfg.yaml"
    empty_cfg.write_text("telegram: {}\n", encoding="utf-8")

    with patch("bot.cli.load_dotenv_if_present"), patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(["auth-telegram", "--config", str(empty_cfg)])
        assert ret == 1
        mock_interactive.assert_not_called()

    captured = capsys.readouterr()
    assert (
        "[ERROR] Missing Telegram API credentials: api_id and api_hash must be set in config or environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)"
        in captured.err
    )


def test_cli_auth_telegram_dummy_placeholder_credentials_fails(capsys, monkeypatch, tmp_path):
    """Verify CLI auth-telegram detects dummy placeholders and aborts with code 1."""
    import yaml

    from bot.cli import main

    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    monkeypatch.delenv("TELEGRAM_PHONE", raising=False)

    cfg_file = tmp_path / "placeholder_userbot.yaml"
    cfg_data = {
        "telegram": {
            "api_id": 12345678,
            "api_hash": "0123456789abcdef0123456789abcdef",
            "session_path": str(tmp_path / "userbot.session"),
        }
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    with patch("bot.cli.load_dotenv_if_present"), patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(["auth-telegram", "--config", str(cfg_file)])
        assert ret == 1
        mock_interactive.assert_not_called()

    captured = capsys.readouterr()
    assert (
        "[ERROR] Missing Telegram API credentials: api_id and api_hash must be set in config or environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)"
        in captured.err
    )


def test_cli_auth_telegram_dummy_placeholder_in_cli_args_fails(capsys, monkeypatch):
    """Verify passing dummy credentials directly on CLI args aborts with code 1."""
    from bot.cli import main

    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)

    with patch("bot.cli.load_dotenv_if_present"), patch("bot.cli.interactive_auth") as mock_interactive:
        ret = main(
            [
                "auth-telegram",
                "--api-id",
                "12345",
                "--api-hash",
                "0123456789abcdef0123456789abcdef",
            ]
        )
        assert ret == 1
        mock_interactive.assert_not_called()

    captured = capsys.readouterr()
    assert (
        "[ERROR] Missing Telegram API credentials: api_id and api_hash must be set in config or environment variables (TELEGRAM_API_ID, TELEGRAM_API_HASH)"
        in captured.err
    )


def test_load_dotenv_if_present_unicode_decode_error(tmp_path):
    """Verify load_dotenv_if_present handles invalid UTF-8 bytes without raising UnicodeDecodeError."""
    from bot.cli import load_dotenv_if_present

    corrupt_env = tmp_path / ".env.corrupt"
    corrupt_env.write_bytes(b"\xff\xfe\x00\x00corrupt_invalid_utf8_payload")

    # Should not raise UnicodeDecodeError or OSError
    load_dotenv_if_present(corrupt_env)


def test_deploy_remote_shell_guards():
    """Verify deploy_remote.sh rejects unknown profiles and catches trailing slash collisions."""
    import subprocess

    repo_root = Path(__file__).resolve().parent.parent
    deploy_script = repo_root / "deploy" / "deploy_remote.sh"
    assert deploy_script.is_file()

    # 1. Invalid profile rejection (exit 1)
    res_invalid = subprocess.run(
        [str(deploy_script), "invalid_profile"],
        capture_output=True,
        text=True,
    )
    assert res_invalid.returncode == 1
    assert "[ERROR] Unknown deployment profile 'invalid_profile'" in res_invalid.stderr

    # 2. Collision with trailing slash (exit 2)
    res_col1 = subprocess.run(
        [str(deploy_script), "userbot", "root@192.168.1.164", "/opt/telegram-group-bot/"],
        capture_output=True,
        text=True,
    )
    assert res_col1.returncode == 2
    assert "[FATAL] Profile 'userbot' must NEVER target '/opt/telegram-group-bot'!" in res_col1.stderr

    # 3. Collision with duplicate slash (exit 2)
    res_col2 = subprocess.run(
        [str(deploy_script), "userbot", "root@192.168.1.164", "/opt//telegram-group-bot"],
        capture_output=True,
        text=True,
    )
    assert res_col2.returncode == 2
    assert "[FATAL] Profile 'userbot' must NEVER target '/opt/telegram-group-bot'!" in res_col2.stderr

    # 4. Collision bot_api targeting userbot with trailing slash (exit 2)
    res_col3 = subprocess.run(
        [str(deploy_script), "bot_api", "root@192.168.1.164", "/opt/telegram-mtproto-userbot/"],
        capture_output=True,
        text=True,
    )
    assert res_col3.returncode == 2
    assert "[FATAL] Profile 'bot_api' must NEVER target '/opt/telegram-mtproto-userbot'!" in res_col3.stderr


