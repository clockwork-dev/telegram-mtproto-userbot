"""Tests for Antigravity OAuth isolation, keyring bypass, and environment sanitization."""

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


def test_setup_isolated_home_permissions_and_soul(temp_data_dir: Path):
    """Verify 0700 directory permissions and empty SOUL.md persona suppression."""
    isolated_home, gemini_dir = setup_isolated_home(temp_data_dir)

    assert isolated_home.is_dir()
    assert gemini_dir.is_dir()

    # Verify 0700 permissions
    mode_home = stat.S_IMODE(isolated_home.stat().st_mode)
    mode_gemini = stat.S_IMODE(gemini_dir.stat().st_mode)
    assert mode_home == 0o700
    assert mode_gemini == 0o700

    # Verify SOUL.md exists and is empty (persona leak suppression)
    soul_file = gemini_dir.parent / "SOUL.md"
    assert soul_file.is_file()
    assert soul_file.stat().st_size == 0


def test_setup_isolated_home_token_linking_with_0600(temp_data_dir: Path):
    """Verify OAuth token files are populated with strict 0600 permissions."""
    # Pre-create a mock source token in the data dir
    source_gemini = temp_data_dir / ".gemini" / "antigravity-cli"
    source_gemini.mkdir(parents=True, exist_ok=True)
    source_token = source_gemini / "jetski-standalone-oauth-token"
    source_token.write_text("mock_refresh_token_payload", encoding="utf-8")
    os.chmod(source_token, 0o644)

    # Now run setup_isolated_home in another subfolder
    sub_data = temp_data_dir / "isolated_workspace"
    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}):
        isolated_home, gemini_dir = setup_isolated_home(sub_data)

    token_target = gemini_dir / "jetski-standalone-oauth-token"
    assert token_target.is_file()
    assert token_target.stat().st_size > 0
    mode = stat.S_IMODE(token_target.stat().st_mode)
    assert mode == 0o600


def test_build_clean_child_env_ssh_connection_injection(temp_data_dir: Path):
    """Verify SSH_CONNECTION marker injection for keyring bypass."""
    # When force_file_auth is True
    env = build_clean_child_env(temp_data_dir, force_file_auth=True)
    assert env.get("SSH_CONNECTION") == _KEYRING_BYPASS_SSH_MARKER

    # When an isolated token file is present in temp_data_dir/.gemini/antigravity-cli
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


def test_build_clean_child_env_strips_secret_names(temp_data_dir: Path):
    """Verify secret-bearing environment variable names are purged."""
    dirty_env = {
        "OPENAI_API_KEY": "sk-12345678901234567890",
        "TELEGRAM_BOT_TOKEN": "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
        "HERMES_AUTH_TOKEN": "secret_token_val",
        "USER_PASSWORD": "SuperSecretPassword123",
        "PATH": "/bin:/usr/bin",
    }
    with patch.dict(os.environ, dirty_env, clear=True):
        clean_strict = build_clean_child_env(temp_data_dir, strict=True)
        assert "OPENAI_API_KEY" not in clean_strict
        assert "TELEGRAM_BOT_TOKEN" not in clean_strict
        assert "AWS_SECRET_ACCESS_KEY" not in clean_strict
        assert "DATABASE_URL" not in clean_strict
        assert "HERMES_AUTH_TOKEN" not in clean_strict
        assert "USER_PASSWORD" not in clean_strict

        clean_heuristic = build_clean_child_env(temp_data_dir, strict=False)
        assert "OPENAI_API_KEY" not in clean_heuristic
        assert "TELEGRAM_BOT_TOKEN" not in clean_heuristic
        assert "AWS_SECRET_ACCESS_KEY" not in clean_heuristic
        assert "DATABASE_URL" not in clean_heuristic
        assert "USER_PASSWORD" not in clean_heuristic


def test_build_clean_child_env_strips_secret_values(temp_data_dir: Path):
    """Verify secret patterns in arbitrary variable names are detected and stripped in heuristic mode."""
    dirty_env = {
        "MY_CUSTOM_VAR": "sk-abcdefghijklmnopqrstuvwxyz123456",
        "GH_INLINE": "ghp_123456789012345678901234567890",
        "GOOGLE_KEY": "AIzaSyD-1234567890abcdefghijklmnopq",
        "OAUTH_YA29": "ya29.a0AfH6SMB1234567890abcdefghijklmnopq",
        "PATH": "/bin:/usr/bin",
    }
    with patch.dict(os.environ, dirty_env, clear=True):
        clean = build_clean_child_env(temp_data_dir, strict=False)
        assert "MY_CUSTOM_VAR" not in clean
        assert "GH_INLINE" not in clean
        assert "GOOGLE_KEY" not in clean
        assert "OAUTH_YA29" not in clean


def test_build_clean_child_env_strips_dynamic_loaders(temp_data_dir: Path):
    """Verify dynamic loader injection variables are never forwarded."""
    dangerous = {
        "LD_PRELOAD": "/usr/lib/evil.so",
        "LD_LIBRARY_PATH": "/tmp/evil/lib",
        "PYTHONPATH": "/tmp/injected_modules",
        "PYTHONHOME": "/tmp/fake_python",
        "NODE_OPTIONS": "--require /tmp/evil.js",
        "BASH_ENV": "/tmp/evil.sh",
        "PATH": "/bin:/usr/bin",
    }
    with patch.dict(os.environ, dangerous, clear=True):
        clean_strict = build_clean_child_env(temp_data_dir, strict=True)
        clean_heuristic = build_clean_child_env(temp_data_dir, strict=False)

        for var in dangerous:
            if var == "PATH":
                continue
            assert var not in clean_strict
            assert var not in clean_heuristic


def test_zero_secret_exfiltration_linux_keyring_probe():
    """Verify Linux secret-tool probe runs with stdout=DEVNULL (no secret in memory)."""
    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stderr="[/org/freedesktop/secrets/collection/login/12]\nlabel = gemini:antigravity\n",
        )

        result = check_linux_secret_service_auth()
        assert result is True

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("stdout") == subprocess.DEVNULL
        assert call_kwargs.get("stdin") == subprocess.DEVNULL


def test_zero_secret_exfiltration_macos_keychain_probe():
    """Verify macOS security tool probe runs with stdout=DEVNULL."""
    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        result = check_macos_keychain_auth()
        assert result is True

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "find-generic-password" in cmd
        assert "-w" not in cmd  # No password printing flag
        assert "-g" not in cmd
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("stdout") == subprocess.DEVNULL


def test_zero_secret_exfiltration_windows_probe():
    """Verify Windows cmdkey probe queries target metadata only."""
    with patch("os.path.isfile", return_value=True), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Target: gemini:antigravity\nType: Generic\nUser: antigravity\n",
        )

        result = check_windows_credential_manager_auth()
        assert result is True

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert any("cmdkey" in str(x) for x in cmd)
        assert "/list:gemini:antigravity" in cmd


def test_sync_updated_isolated_tokens(temp_data_dir: Path):
    """Verify refreshed token file is copied back to persistent store with 0600 permissions."""
    # Create source token
    source_dir = temp_data_dir / "source" / ".gemini" / "antigravity-cli"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_token = source_dir / "jetski-standalone-oauth-token"
    source_token.write_text("old_token_version", encoding="utf-8")
    os.chmod(source_token, 0o600)
    old_mtime = time.time() - 100.0
    os.utime(source_token, (old_mtime, old_mtime))

    # Create isolated refreshed token
    isolated_dir = temp_data_dir / "worker" / ".gemini" / "antigravity-cli"
    isolated_dir.mkdir(parents=True, exist_ok=True)
    isolated_token = isolated_dir / "jetski-standalone-oauth-token"
    isolated_token.write_text("new_refreshed_token_version", encoding="utf-8")
    new_mtime = time.time()
    os.utime(isolated_token, (new_mtime, new_mtime))

    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir / "source")}):
        sync_updated_isolated_tokens(isolated_dir, data_dir=temp_data_dir / "source")

    assert source_token.read_text(encoding="utf-8") == "new_refreshed_token_version"
    mode = stat.S_IMODE(source_token.stat().st_mode)
    assert mode == 0o600


def test_run_isolated_antigravity_login(temp_data_dir: Path):
    """Verify interactive login helper runs agy acp --login in isolated environment."""
    with patch("bot.antigravity.auth.resolve_antigravity_binary", return_value="/mock/bin/agy"), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        rc = run_isolated_antigravity_login(data_dir=temp_data_dir)
        assert rc == 0

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["/mock/bin/agy", "acp", "--login"]
        call_env = mock_run.call_args[1]["env"]
        assert call_env["SSH_CONNECTION"] == _KEYRING_BYPASS_SSH_MARKER
        assert call_env["HOME"] == str(temp_data_dir.resolve())


def test_auth_status_and_evidence(temp_data_dir: Path):
    """Verify check_antigravity_token_file_evidence, check_antigravity_auth, and describe_antigravity_auth_source."""
    # 1. No credentials
    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}), \
         patch("bot.antigravity.auth.check_os_keyring_auth", return_value=False):
        has_ev, desc = check_antigravity_token_file_evidence(data_dir=temp_data_dir)
        assert has_ev is False
        assert desc == ""

        has_auth = check_antigravity_auth(data_dir=temp_data_dir)
        assert has_auth is False
        assert describe_antigravity_auth_source(data_dir=temp_data_dir) == ""

    # 2. Token file present
    g_dir = temp_data_dir / ".gemini" / "antigravity-cli"
    g_dir.mkdir(parents=True, exist_ok=True)
    tok = g_dir / "jetski-standalone-oauth-token"
    tok.write_text("valid_token", encoding="utf-8")

    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}):
        has_ev, desc = check_antigravity_token_file_evidence(data_dir=temp_data_dir)
        assert has_ev is True
        assert "jetski-standalone-oauth-token" in desc

        has_auth = check_antigravity_auth(data_dir=temp_data_dir)
        assert has_auth is True

        source_desc = describe_antigravity_auth_source(data_dir=temp_data_dir)
        assert "OAuth token file" in source_desc


def test_resolve_antigravity_binary_and_home(temp_data_dir: Path):
    """Verify binary resolution and data directory resolution."""
    # Custom binary override
    with patch.dict(os.environ, {"ANTIGRAVITY_BINARY": "/custom/bin/agy"}), \
         patch("bot.antigravity.auth._is_usable_binary", return_value=True):
        bin_path = resolve_antigravity_binary()
        assert bin_path == "/custom/bin/agy"

    # Home resolution with env
    with patch.dict(os.environ, {"ANTIGRAVITY_DATA_DIR": str(temp_data_dir)}):
        home_path = resolve_antigravity_home()
        assert home_path == temp_data_dir.resolve()

