"""Tests for Antigravity OAuth isolation, keyring bypass, environment sanitization, and session cleanup."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bot.antigravity.auth import (
    _KEYRING_BYPASS_SSH_MARKER,
    build_clean_child_env,
    check_antigravity_auth,
    check_antigravity_token_file_evidence,
    check_linux_secret_service_auth,
    check_macos_keychain_auth,
    check_windows_credential_manager_auth,
    cleanup_old_antigravity_sessions,
    describe_antigravity_auth_source,
    resolve_antigravity_binary,
    resolve_antigravity_home,
    run_isolated_antigravity_login,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime), follow_symlinks=False)


def test_setup_isolated_home_permissions_and_soul(temp_data_dir: Path):
    """Verify 0700 directory permissions and empty SOUL.md persona suppression."""
    isolated_home, gemini_dir = setup_isolated_home(temp_data_dir)

    assert isolated_home.is_dir()
    assert gemini_dir.is_dir()

    mode_home = stat.S_IMODE(isolated_home.stat().st_mode)
    mode_gemini = stat.S_IMODE(gemini_dir.stat().st_mode)
    assert mode_home == 0o700
    assert mode_gemini == 0o700

    soul_file = gemini_dir.parent / "SOUL.md"
    assert soul_file.is_file()
    assert soul_file.stat().st_size == 0


def test_setup_isolated_home_token_linking_with_0600(temp_data_dir: Path):
    """Verify OAuth token files are populated with strict 0600 permissions."""
    source_gemini = temp_data_dir / ".gemini" / "antigravity-cli"
    source_gemini.mkdir(parents=True, exist_ok=True)
    source_token = source_gemini / "jetski-standalone-oauth-token"
    source_token.write_text("mock_refresh_token_payload", encoding="utf-8")
    os.chmod(source_token, 0o644)

    sub_data = temp_data_dir / "isolated_workspace"
    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}):
        _, gemini_dir = setup_isolated_home(sub_data)

    token_target = gemini_dir / "jetski-standalone-oauth-token"
    assert token_target.is_file()
    assert token_target.stat().st_size > 0
    assert stat.S_IMODE(token_target.stat().st_mode) == 0o600


def test_build_clean_child_env_ssh_connection_injection(temp_data_dir: Path):
    """Verify SSH_CONNECTION marker injection for keyring bypass."""
    env = build_clean_child_env(temp_data_dir, force_file_auth=True)
    assert env.get("SSH_CONNECTION") == _KEYRING_BYPASS_SSH_MARKER

    g_dir = temp_data_dir / ".gemini" / "antigravity-cli"
    g_dir.mkdir(parents=True, exist_ok=True)
    tok = g_dir / "jetski-standalone-oauth-token"
    tok.write_text("token_data", encoding="utf-8")

    env_with_token = build_clean_child_env(temp_data_dir, force_file_auth=False)
    assert env_with_token.get("SSH_CONNECTION") == _KEYRING_BYPASS_SSH_MARKER


def test_build_clean_child_env_pins_isolated_home(temp_data_dir: Path):
    """Verify HOME, USERPROFILE, and XDG_CONFIG_HOME point strictly to isolated directory."""
    env = build_clean_child_env(temp_data_dir)
    assert env["HOME"] == str(temp_data_dir)
    assert env["USERPROFILE"] == str(temp_data_dir)
    assert env["XDG_CONFIG_HOME"] == str(temp_data_dir / ".config")


def test_build_clean_child_env_strips_secret_names_values_and_loaders(temp_data_dir: Path):
    """Verify secret variable names, secret values, and dynamic loader injection vars are purged."""
    dirty_env = {
        "OPENAI_API_KEY": "sk-12345678901234567890",
        "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
        "MY_CUSTOM_VAR": "sk-abcdefghijklmnopqrstuvwxyz123456",
        "LD_PRELOAD": "/usr/lib/evil.so",
        "PYTHONPATH": "/tmp/injected_modules",
        "NODE_OPTIONS": "--require /tmp/evil.js",
        "PATH": "/bin:/usr/bin",
    }
    with patch.dict(os.environ, dirty_env, clear=True):
        clean_strict = build_clean_child_env(temp_data_dir, strict=True)
        clean_heuristic = build_clean_child_env(temp_data_dir, strict=False)

        for key in dirty_env:
            if key == "PATH":
                continue
            assert key not in clean_strict
            assert key not in clean_heuristic


def test_zero_secret_exfiltration_keyring_probes():
    """Verify OS keyring probes never read secret payloads into stdout."""
    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[/org/freedesktop/secrets/collection/login/12]\nlabel = gemini:antigravity\n",
        )
        assert check_linux_secret_service_auth() is True
        assert mock_run.call_args[1].get("stdout") == subprocess.DEVNULL

    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert check_macos_keychain_auth() is True
        cmd = mock_run.call_args[0][0]
        assert "-w" not in cmd and "-g" not in cmd
        assert mock_run.call_args[1].get("stdout") == subprocess.DEVNULL

    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Target: gemini:antigravity\nType: Generic\nUser: antigravity\n",
        )
        assert check_windows_credential_manager_auth() is True


def test_sync_updated_isolated_tokens_and_login_and_status(temp_data_dir: Path):
    source_dir = temp_data_dir / "source" / ".gemini" / "antigravity-cli"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_token = source_dir / "jetski-standalone-oauth-token"
    source_token.write_text("old_token_version", encoding="utf-8")
    os.chmod(source_token, 0o600)
    old_mtime = time.time() - 100.0
    os.utime(source_token, (old_mtime, old_mtime))

    isolated_dir = temp_data_dir / "worker" / ".gemini" / "antigravity-cli"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    isolated_token = isolated_dir / "jetski-standalone-oauth-token"
    isolated_token.write_text("new_refreshed_token_version", encoding="utf-8")
    new_mtime = time.time()
    os.utime(isolated_token, (new_mtime, new_mtime))

    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir / "source")}):
        sync_updated_isolated_tokens(isolated_dir, data_dir=temp_data_dir / "source")
        assert source_token.read_text(encoding="utf-8") == "new_refreshed_token_version"
        assert stat.S_IMODE(source_token.stat().st_mode) == 0o600
        has_ev, desc = check_antigravity_token_file_evidence(data_dir=temp_data_dir / "source")
        assert has_ev is True and "jetski-standalone-oauth-token" in desc
        assert check_antigravity_auth(data_dir=temp_data_dir / "source") is True
        assert "OAuth token file" in describe_antigravity_auth_source(data_dir=temp_data_dir / "source")

    with (
        patch("bot.antigravity.auth.resolve_antigravity_binary", return_value="/mock/bin/agy"),
        patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run,
    ):
        assert run_isolated_antigravity_login(data_dir=temp_data_dir) == 0
        assert mock_run.call_args[0][0] == ["/mock/bin/agy", "acp", "--login"]

    with (
        patch.dict(os.environ, {"ANTIGRAVITY_BINARY": "/custom/bin/agy", "ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}),
        patch("bot.antigravity.auth._is_usable_binary", return_value=True),
    ):
        assert resolve_antigravity_binary() == "/custom/bin/agy"
        assert resolve_antigravity_home() == temp_data_dir.resolve()


def test_cleanup_preserves_oauth_tokens_fresh_active_and_rejects_symlinks(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_secret = outside_dir / "secret.txt"
    outside_secret.write_text("DO_NOT_DELETE", encoding="utf-8")

    data_dir = tmp_path / "agy_data"
    _, gemini_dir = setup_isolated_home(data_dir)
    now = time.time()
    ancient = now - 10 * 86400.0

    tok = gemini_dir / "jetski-standalone-oauth-token"
    tok.write_text("sensitive-token", encoding="utf-8")
    _set_mtime(tok, ancient)

    conv_dir = gemini_dir / "conversations"
    conv_dir.mkdir(exist_ok=True)
    old_conv = conv_dir / "old-session.db"
    old_conv.write_text("old", encoding="utf-8")
    _set_mtime(old_conv, ancient)

    active_conv = conv_dir / "active-cid.db"
    active_conv.write_text("active", encoding="utf-8")
    _set_mtime(active_conv, ancient)

    fresh_conv = conv_dir / "fresh-cid.db"
    fresh_conv.write_text("fresh", encoding="utf-8")
    _set_mtime(fresh_conv, now - 60.0)

    os.symlink(outside_secret, conv_dir / "evil-link.db")

    stats = cleanup_old_antigravity_sessions(
        data_dir,
        active_conversation_ids={"active-cid"},
        now=now,
    )
    assert stats["removed"] == 1
    assert not old_conv.exists()
    assert tok.is_file()
    assert active_conv.is_file()
    assert fresh_conv.is_file()
    assert outside_secret.read_text(encoding="utf-8") == "DO_NOT_DELETE"
