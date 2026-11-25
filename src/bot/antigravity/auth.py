"""Antigravity OAuth token management, account isolation, and environment sanitization."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File names used by agy across versions:
# - agy >= 1.2: jetski-standalone-oauth-token
# - agy 1.1:    antigravity-oauth-token
_OAUTH_TOKEN_FILENAMES: tuple[str, ...] = (
    "jetski-standalone-oauth-token",
    "antigravity-oauth-token",
)

# Subdirectories inside <data_dir>/.gemini/antigravity-cli eligible for old-session cleanup
_CLEANUP_TARGET_SUBDIRS: tuple[str, ...] = (
    "conversations",
    "brain",
    "implicit",
    "presence",
    "log",
)

# Explicit deny-list of protected entries inside gemini_dir that must NEVER be deleted
_PROTECTED_GEMINI_ENTRIES: frozenset[str] = frozenset({
    *_OAUTH_TOKEN_FILENAMES,
    "installation_id",
    "settings.json",
    "builtin",
    "bin",
    "annotations",
    "jetski_state.pbtxt",
    "SOUL.md",
    "conversation_summaries.db",
    "conversation_summaries.db-wal",
    "conversation_summaries.db-shm",
    "jetbox_summaries_proto.pb",
})

# Keyring bypass marker: when SSH_CONNECTION is set, agy's Go CompositeTokenStorage
# forces file-based token storage rather than reading/writing host OS keyrings.
_KEYRING_BYPASS_SSH_MARKER = "127.0.0.1 0 127.0.0.1 22"

_KEYRING_SERVICE = "gemini"
_KEYRING_ACCOUNT = "antigravity"
_WINDOWS_CREDENTIAL_TARGET = f"{_KEYRING_SERVICE}:{_KEYRING_ACCOUNT}"
_KEYRING_PROBE_TIMEOUT_SECONDS = 3.0

# Safe environment variables allowed in child execution
_SAFE_ENV_KEYS = frozenset({
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "TZ",
    # Windows
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    # Linux D-Bus / XDG
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    # TLS CA bundles
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    # Antigravity override
    "ANTIGRAVITY_CONFIG_DIR",
})

# Patterns matching secret-bearing variable names
_SECRET_KEY_PATTERNS = re.compile(
    r"(?:^|_)(?:"
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|AUTH|BEARER|"
    r"COOKIE|SESSION|PRIVATE|JWT|SIGNATURE|CERT|PEM|WEBHOOK|CONN|CONNECTION|"
    r"DSN|DATABASE_URL|REDIS_URL|MONGO_URL|AMQP_URL|SMTP_URL|"
    r"AWS_ACCESS|AWS_SECRET|AWS_SESSION|AZURE_CLIENT|GCP_|GOOGLE_APPLICATION|"
    r"OPENAI|ANTHROPIC|GEMINI|CLAUDE|COHERE|MISTRAL|GROQ|DEEPSEEK|OPENROUTER|"
    r"HUGGINGFACE|HF_TOKEN|REPLICATE|TOGETHER|FIREWORKS|PERPLEXITY|XAI|"
    r"GITHUB|GH_TOKEN|GITLAB|BITBUCKET|NPM_TOKEN|PYPI|DOCKER|KUBECONFIG|"
    r"SLACK|DISCORD|TELEGRAM|TWILIO|SENDGRID|STRIPE|PAYPAL|CLOUDFLARE|"
    r"SUPABASE|NEON|PLANETSCALE|TURSO|UPSTASH|ALGOLIA|DATADOG|SENTRY|"
    r"VAULT|CONSUL|NOMAD|TERRAFORM|PULUMI|ANSIBLE"
    r")(?:$|_)",
    re.IGNORECASE,
)

# Values that look like API keys, tokens, JWTs, or private keys
_SECRET_VALUE_PATTERNS = re.compile(
    r"^(?:"
    r"sk-[a-zA-Z0-9_\-]{16,}|"
    r"rk-[a-zA-Z0-9_\-]{16,}|"
    r"pk-[a-zA-Z0-9_\-]{16,}|"
    r"gh[pousr]_[a-zA-Z0-9_]{20,}|"
    r"github_pat_[a-zA-Z0-9_]{20,}|"
    r"glpat-[a-zA-Z0-9_\-]{16,}|"
    r"xox[baprs]-[a-zA-Z0-9\-]{16,}|"
    r"AIza[0-9A-Za-z_\-]{30,}|"
    r"ya29\.[0-9A-Za-z_\-]{20,}|"
    r"AKIA[0-9A-Z]{16}|"
    r"ASIA[0-9A-Z]{16}|"
    r"eyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}|"
    r"hf_[a-zA-Z0-9]{20,}|"
    r"r8_[a-zA-Z0-9]{20,}|"
    r"gsk_[a-zA-Z0-9]{20,}|"
    r"pplx-[a-zA-Z0-9]{20,}|"
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|"
    r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://[^/\s]+:[^@\s]+@"
    r")"
)

# Proxy variables stripped unless explicitly passed
_PROXY_ENV_KEYS = frozenset({
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
})

# Dynamic loaders and injection variables that must never leak to child processes
_DANGEROUS_LOADER_ENV_KEYS = frozenset({
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "PYTHONSTARTUP",
    "PYTHONPATH",
    "PYTHONHOME",
    "NODE_OPTIONS",
    "NODE_PATH",
    "PERL5OPT",
    "RUBYOPT",
    "JAVA_TOOL_OPTIONS",
    "BASH_ENV",
    "ENV",
    "PROMPT_COMMAND",
    "GIT_SSH_COMMAND",
    "GIT_ASKPASS",
    "SSH_ASKPASS",
})

_SSH_AGENT_ENV_KEYS = frozenset({
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
})


def _is_usable_binary(path: str | Path) -> bool:
    """Return True only when path is an executable regular file."""
    try:
        p = Path(path)
        return p.is_file() and os.access(p, os.X_OK)
    except OSError:
        return False


def resolve_antigravity_binary() -> str:
    """Locate the agy CLI executable."""
    for env_key in ("ANTIGRAVITY_BINARY", "AGY_CLI_PATH"):
        env_override = os.environ.get(env_key, "").strip()
        if env_override and _is_usable_binary(env_override):
            return env_override

    which_agy = shutil.which("agy")
    if which_agy and _is_usable_binary(which_agy):
        return which_agy

    candidates = [
        Path.home() / ".local" / "bin" / "agy",
        Path.home() / ".antigravity" / "bin" / "agy",
        Path("/usr/local/bin/agy"),
        Path("/usr/bin/agy"),
        Path("/opt/antigravity/bin/agy"),
        Path("/opt/homebrew/bin/agy"),
    ]

    for candidate in candidates:
        if _is_usable_binary(candidate):
            return str(candidate)

    return ""


def resolve_antigravity_home(data_dir: Path | str | None = None) -> Path:
    """Return the dedicated directory for Antigravity OAuth tokens and sandboxed home.

    Resolution:
    1. data_dir if explicitly passed
    2. ANTIGRAVITY_DATA_DIR / ANTIGRAVITY_BOT_HOME environment variable
    3. Project local directory `data/antigravity`
    """
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    env_dir = os.environ.get("ANTIGRAVITY_DATA_DIR") or os.environ.get("ANTIGRAVITY_BOT_HOME")
    if env_dir and env_dir.strip():
        return Path(env_dir.strip()).expanduser().resolve()

    return (Path.cwd() / "data" / "antigravity").resolve()


def _resolve_oauth_token_candidates(
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> list[Path]:
    """Return candidate OAuth token paths in priority order."""
    config_override = os.environ.get("ANTIGRAVITY_CONFIG_DIR", "").strip()
    if config_override:
        base = Path(config_override).expanduser()
        return [base / name for name in _OAUTH_TOKEN_FILENAMES]

    candidates: list[Path] = []

    # 1. Dedicated data directory isolated profile
    bot_home = resolve_antigravity_home(data_dir)
    bot_gemini_dir = bot_home / ".gemini" / "antigravity-cli"
    bot_gemini_root = bot_home / ".gemini"

    for name in _OAUTH_TOKEN_FILENAMES:
        candidates.append(bot_gemini_dir / name)
    candidates.append(bot_gemini_root / "jetski-standalone-oauth-token")

    # 2. User's global directory (only when home_dir is explicitly passed or data_dir is not pinned)
    if home_dir is not None or data_dir is None:
        user_root = home_dir if home_dir is not None else Path.home()
        user_gemini_dir = user_root / ".gemini" / "antigravity-cli"
        user_gemini_root = user_root / ".gemini"

        for name in _OAUTH_TOKEN_FILENAMES:
            candidates.append(user_gemini_dir / name)
        candidates.append(user_gemini_root / "jetski-standalone-oauth-token")

    return candidates


def find_existing_oauth_token(
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> Path | None:
    """Return the first existing, non-empty OAuth token file without reading its contents."""
    for candidate in _resolve_oauth_token_candidates(home_dir, data_dir):
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except OSError:
            continue
    return None


def check_antigravity_token_file_evidence(
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> tuple[bool, str]:
    """Subprocess-free check for an OAuth token file on disk."""
    token_path = find_existing_oauth_token(home_dir, data_dir)
    if token_path is not None:
        return True, f"OAuth token file ({token_path})"
    return False, ""


def check_linux_secret_service_auth() -> bool:
    """Check Linux FreeDesktop Secret Service without reading secrets into Python memory."""
    secret_tool_bin = "/usr/bin/secret-tool"
    if not os.path.isfile(secret_tool_bin):
        secret_tool_bin = shutil.which("secret-tool") or ""

    if secret_tool_bin:
        try:
            res = subprocess.run(
                [
                    secret_tool_bin,
                    "search",
                    "--all",
                    "service",
                    _KEYRING_SERVICE,
                    "username",
                    _KEYRING_ACCOUNT,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_KEYRING_PROBE_TIMEOUT_SECONDS,
                check=False,
            )
            if res.returncode == 0 and "[/org/freedesktop/secrets/" in (res.stderr or ""):
                return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Linux secret-tool probe failed: %s", exc)

    try:
        import contextlib

        import secretstorage  # type: ignore[import-not-found]

        with contextlib.closing(secretstorage.dbus_init()) as connection:
            collection = secretstorage.get_default_collection(connection)
            items = collection.search_items({
                "service": _KEYRING_SERVICE,
                "username": _KEYRING_ACCOUNT,
            })
            for item in items:
                attrs = item.get_attributes()
                if (
                    attrs.get("service") == _KEYRING_SERVICE
                    and attrs.get("username") == _KEYRING_ACCOUNT
                ):
                    return True
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Linux secretstorage D-Bus probe failed: %s", exc)

    return False


def check_macos_keychain_auth() -> bool:
    """Check macOS Keychain for gemini/antigravity without printing the secret."""
    security_bin = "/usr/bin/security"
    if not os.path.isfile(security_bin):
        security_bin = shutil.which("security") or ""
    if not security_bin:
        return False
    try:
        res = subprocess.run(
            [
                security_bin,
                "find-generic-password",
                "-s",
                _KEYRING_SERVICE,
                "-a",
                _KEYRING_ACCOUNT,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_KEYRING_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("macOS keychain probe failed: %s", exc)
        return False


def check_windows_credential_manager_auth() -> bool:
    """Check Windows Credential Manager without printing secrets."""
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"
    cmdkey_bin = os.path.join(system_root, "System32", "cmdkey.exe")
    if not os.path.isfile(cmdkey_bin):
        cmdkey_bin = shutil.which("cmdkey") or ("cmdkey" if sys.platform == "win32" else "")
    if not cmdkey_bin:
        return False

    popen_kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

    try:
        res = subprocess.run(
            [cmdkey_bin, f"/list:{_WINDOWS_CREDENTIAL_TARGET}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_KEYRING_PROBE_TIMEOUT_SECONDS,
            check=False,
            **popen_kwargs,
        )
        if res.returncode != 0:
            return False
        stdout_lower = (res.stdout or "").lower()
        return _WINDOWS_CREDENTIAL_TARGET.lower() in stdout_lower
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Windows Credential Manager probe failed: %s", exc)
        return False


def check_os_keyring_auth() -> bool:
    """Platform-dispatched check for credentials in the OS keyring without secret exfiltration."""
    if sys.platform == "darwin":
        return check_macos_keychain_auth()
    if sys.platform == "win32":
        return check_windows_credential_manager_auth()
    if sys.platform.startswith("linux"):
        return check_linux_secret_service_auth()
    return False


def check_antigravity_auth(
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> bool:
    """Verify whether Antigravity OAuth credentials exist in file or OS keyring."""
    if find_existing_oauth_token(home_dir, data_dir) is not None:
        return True
    return check_os_keyring_auth()


def describe_antigravity_auth_source(
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> str:
    """Return a human-readable description of credential source without secret data."""
    token_path = find_existing_oauth_token(home_dir, data_dir)
    if token_path is not None:
        bot_home = resolve_antigravity_home(data_dir)
        try:
            if token_path.is_relative_to(bot_home):
                return f"Bot isolated OAuth token file ({token_path})"
        except (ValueError, AttributeError):
            pass
        return f"OAuth token file ({token_path})"
    if sys.platform == "darwin" and check_macos_keychain_auth():
        return f"macOS Keychain (service={_KEYRING_SERVICE}, account={_KEYRING_ACCOUNT})"
    if sys.platform == "win32" and check_windows_credential_manager_auth():
        return f"Windows Credential Manager (target={_WINDOWS_CREDENTIAL_TARGET})"
    if sys.platform.startswith("linux") and check_linux_secret_service_auth():
        return f"Linux Secret Service (service={_KEYRING_SERVICE}, username={_KEYRING_ACCOUNT})"
    return ""


def _same_underlying_file(path_a: Path, path_b: Path) -> bool:
    """Return True if two paths resolve to the same file."""
    try:
        return os.path.samefile(path_a, path_b)
    except OSError:
        try:
            return path_a.resolve() == path_b.resolve()
        except OSError:
            return False


def _link_or_copy_token(source: Path, dest: Path) -> None:
    """Link or copy source token file to dest with strict 0600 permissions."""
    try:
        if os.path.abspath(source) == os.path.abspath(dest):
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
            return
    except OSError:
        return

    if os.path.islink(dest):
        try:
            dest.unlink()
        except OSError:
            pass

    if dest.exists():
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
        return

    for link_fn in (os.symlink, os.link):
        try:
            link_fn(source, dest)
            try:
                os.chmod(dest, 0o600)
            except OSError:
                pass
            return
        except OSError:
            continue

    try:
        shutil.copy2(source, dest)
        os.chmod(dest, 0o600)
    except OSError as exc:
        logger.warning("Failed copying OAuth token to %s: %s", dest, exc)


def setup_isolated_home(
    data_dir: Path | str,
    source_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Set up an isolated home directory for Antigravity with 0700 dir and 0600 token permissions.

    1. Creates `<data_dir>` and `<data_dir>/.gemini/antigravity-cli` with 0700 permissions.
    2. Links or copies active token files with 0600 permissions.
    3. Creates empty `<data_dir>/.gemini/SOUL.md` to prevent global persona leak.
    4. On macOS, symlinks Library/Keychains if available.

    Returns `(isolated_home, gemini_dir)`.
    """
    isolated_home = Path(data_dir).expanduser().resolve()
    gemini_root = isolated_home / ".gemini"
    gemini_dir = gemini_root / "antigravity-cli"

    gemini_dir.mkdir(parents=True, exist_ok=True)
    for directory in (isolated_home, gemini_root, gemini_dir):
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass

    # Populate token files if existing
    source_token = find_existing_oauth_token(data_dir=source_dir)
    if source_token is not None:
        for basename in _OAUTH_TOKEN_FILENAMES:
            target = gemini_dir / basename
            _link_or_copy_token(source_token, target)
        _link_or_copy_token(source_token, gemini_root / "jetski-standalone-oauth-token")

    # Global persona suppression: empty SOUL.md
    soul_file = gemini_root / "SOUL.md"
    if not soul_file.exists():
        try:
            soul_file.write_text("", encoding="utf-8")
        except OSError:
            pass

    # macOS keychains link
    if sys.platform == "darwin":
        real_keychains = Path.home() / "Library" / "Keychains"
        isolated_keychains = isolated_home / "Library" / "Keychains"
        if real_keychains.is_dir() and not isolated_keychains.exists():
            try:
                isolated_keychains.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(real_keychains, isolated_keychains, target_is_directory=True)
            except OSError:
                pass

    return isolated_home, gemini_dir


def sync_updated_isolated_tokens(
    gemini_dir: Path | None,
    home_dir: Path | None = None,
    data_dir: Path | str | None = None,
) -> None:
    """Sync refreshed OAuth access tokens written by agy back to persistent store with 0600 permissions."""
    if gemini_dir is None:
        return
    source_token = find_existing_oauth_token(home_dir, data_dir)
    if source_token is None:
        return

    try:
        source_mtime = source_token.stat().st_mtime
    except OSError:
        return

    gemini_root = gemini_dir.parent
    isolated_candidates = (
        *(gemini_dir / name for name in _OAUTH_TOKEN_FILENAMES),
        gemini_root / "jetski-standalone-oauth-token",
    )

    newest_candidate: Path | None = None
    newest_mtime = source_mtime

    for candidate in isolated_candidates:
        try:
            if os.path.islink(candidate):
                continue
            if not candidate.is_file():
                continue
            if _same_underlying_file(candidate, source_token):
                continue
            st = candidate.stat()
            if st.st_size > 0 and st.st_mtime > newest_mtime:
                newest_mtime = st.st_mtime
                newest_candidate = candidate
        except OSError:
            continue

    if newest_candidate is not None:
        tmp_target = source_token.parent / f".{source_token.name}.tmp.{os.getpid()}"
        try:
            shutil.copy2(newest_candidate, tmp_target)
            os.chmod(tmp_target, 0o600)
            os.replace(tmp_target, source_token)
            os.chmod(source_token, 0o600)
            logger.info("Synchronized refreshed Antigravity OAuth token to %s", source_token)
        except OSError as exc:
            logger.debug("Failed syncing refreshed token to %s: %s", source_token, exc)
            try:
                if tmp_target.exists():
                    tmp_target.unlink()
            except OSError:
                pass


def _isolated_home_has_token_file(isolated_home: Path) -> bool:
    """Return True if isolated_home contains an OAuth token file."""
    gemini_root = isolated_home / ".gemini"
    gemini_dir = gemini_root / "antigravity-cli"
    for candidate in (
        *(gemini_dir / name for name in _OAUTH_TOKEN_FILENAMES),
        gemini_root / "jetski-standalone-oauth-token",
    ):
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def build_clean_child_env(
    isolated_home: Path,
    *,
    force_file_auth: bool = True,
    strict: bool = True,
) -> dict[str, str]:
    """Build sanitized environment dict stripping secret variables and dynamic loaders."""
    clean_env: dict[str, str] = {}

    if strict:
        for key, val in os.environ.items():
            if not isinstance(val, str):
                continue
            key_upper = key.upper()
            if key in _SAFE_ENV_KEYS or key_upper in _SAFE_ENV_KEYS:
                if key_upper in _DANGEROUS_LOADER_ENV_KEYS:
                    continue
                clean_env[key] = val
            elif key.startswith("LC_") and not _SECRET_VALUE_PATTERNS.search(val.strip()):
                clean_env[key] = val
    else:
        has_token_file = _isolated_home_has_token_file(isolated_home)
        for key, val in os.environ.items():
            if not isinstance(val, str):
                continue
            key_upper = key.upper()
            if key_upper in _DANGEROUS_LOADER_ENV_KEYS:
                continue
            if key in _PROXY_ENV_KEYS or key_upper in _PROXY_ENV_KEYS:
                continue
            if not has_token_file and key_upper in _SSH_AGENT_ENV_KEYS:
                continue
            if key_upper.startswith("ANTIGRAVITY_") and key_upper != "ANTIGRAVITY_CONFIG_DIR":
                continue
            if key_upper not in _SAFE_ENV_KEYS:
                if _SECRET_KEY_PATTERNS.search(key):
                    continue
                if _SECRET_VALUE_PATTERNS.search(val.strip()):
                    continue
            clean_env[key] = val

    # Proxy opt-in
    explicit_proxy = os.environ.get("ANTIGRAVITY_PROXY", "").strip()
    if explicit_proxy:
        clean_env["HTTPS_PROXY"] = explicit_proxy
        clean_env["HTTP_PROXY"] = explicit_proxy
        clean_env["ALL_PROXY"] = explicit_proxy
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if no_proxy:
            clean_env["NO_PROXY"] = no_proxy

    # Pin isolated home
    clean_env["HOME"] = str(isolated_home)
    clean_env["USERPROFILE"] = str(isolated_home)
    clean_env["XDG_CONFIG_HOME"] = str(isolated_home / ".config")

    # Ensure POSIX standard binary dirs in PATH
    if sys.platform != "win32":
        raw_path = clean_env.get("PATH", "")
        parts = [p for p in raw_path.split(os.pathsep) if p] if raw_path else []
        for fallback in (
            str(Path.home() / ".local" / "bin"),
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ):
            if fallback not in parts:
                parts.append(fallback)
        clean_env["PATH"] = os.pathsep.join(parts)

    # Keyring bypass marker: forces local file-based token storage
    if force_file_auth or _isolated_home_has_token_file(isolated_home):
        clean_env["SSH_CONNECTION"] = _KEYRING_BYPASS_SSH_MARKER

    return clean_env


def run_isolated_antigravity_login(
    data_dir: Path | str | None = None,
    binary: str | None = None,
) -> int:
    """Run `agy acp --login` interactively in isolated data directory with SSH_CONNECTION marker."""
    bot_home = resolve_antigravity_home(data_dir)
    isolated_home, gemini_dir = setup_isolated_home(bot_home)

    agy_bin = binary or resolve_antigravity_binary()
    if not agy_bin:
        raise FileNotFoundError(
            "The 'agy' CLI executable was not found. Please install agy or set ANTIGRAVITY_BINARY."
        )

    login_env = build_clean_child_env(isolated_home, force_file_auth=True)
    res = subprocess.run([agy_bin, "acp", "--login"], cwd=str(isolated_home), env=login_env, check=False)

    # Synchronize tokens and enforce 0600 permissions
    gemini_root = gemini_dir.parent
    candidate_paths = [
        *(gemini_dir / name for name in _OAUTH_TOKEN_FILENAMES),
        gemini_root / "jetski-standalone-oauth-token",
    ]

    newest_token: Path | None = None
    newest_mtime: float = -1.0
    for path in candidate_paths:
        try:
            if path.is_file() and path.stat().st_size > 0:
                mtime = path.stat().st_mtime
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_token = path
        except OSError:
            continue

    if newest_token is not None:
        for target in candidate_paths:
            try:
                if target != newest_token:
                    if os.path.islink(target):
                        target.unlink()
                    shutil.copy2(newest_token, target)
                os.chmod(target, 0o600)
            except OSError:
                pass

    return res.returncode


def _effective_entry_mtime(entry_path: Path, base_mtime: float, *, is_dir: bool) -> float:
    """Return latest mtime across a directory entry and its immediate non-symlink children."""
    latest = base_mtime
    if not is_dir:
        return latest
    try:
        with os.scandir(entry_path) as it:
            for child in it:
                try:
                    if child.is_symlink():
                        continue
                    child_mtime = child.stat(follow_symlinks=False).st_mtime
                    if child_mtime > latest:
                        latest = child_mtime
                except OSError:
                    continue
    except OSError:
        pass
    return latest


def _matches_active_conversation(entry_name: str, active_ids: set[str]) -> bool:
    """Return True if entry_name belongs to any active conversation ID."""
    if not active_ids:
        return False
    for cid in active_ids:
        if entry_name == cid or entry_name.startswith(f"{cid}."):
            return True
    return False


def _safe_delete_confined_entry(entry_path: Path, allowed_subdir: Path) -> bool:
    """Safely remove a file or directory strictly inside allowed_subdir without following symlinks."""
    if entry_path.is_symlink():
        return False
    try:
        resolved = entry_path.resolve(strict=True)
    except OSError:
        return False
    try:
        if not resolved.is_relative_to(allowed_subdir) or resolved == allowed_subdir:
            return False
    except ValueError:
        return False

    if resolved.is_file():
        resolved.unlink()
        return True

    if resolved.is_dir():
        for root_str, dirnames, filenames in os.walk(resolved, topdown=False, followlinks=False):
            root_path = Path(root_str)
            for fname in filenames:
                fp = root_path / fname
                if fp.is_symlink():
                    continue
                try:
                    fp_resolved = fp.resolve(strict=True)
                    if not fp_resolved.is_relative_to(allowed_subdir):
                        continue
                    fp_resolved.unlink()
                except (OSError, ValueError):
                    continue
            for dname in dirnames:
                dp = root_path / dname
                if dp.is_symlink():
                    continue
                try:
                    dp_resolved = dp.resolve(strict=True)
                    if not dp_resolved.is_relative_to(allowed_subdir) or dp_resolved == allowed_subdir:
                        continue
                    dp_resolved.rmdir()
                except (OSError, ValueError):
                    continue
        try:
            resolved.rmdir()
            return True
        except OSError:
            return not resolved.exists()

    return False


def cleanup_old_antigravity_sessions(
    data_dir: Path | str,
    *,
    max_age_seconds: float = 24.0 * 3600.0,
    max_entries_per_dir: int = 30,
    min_fresh_age_seconds: float = 3600.0,
    active_conversation_ids: Collection[str] | None = None,
    now: float | None = None,
) -> dict[str, int]:
    """Safely prune stale oneshot agy session artifacts inside `<data_dir>/.gemini/antigravity-cli`.

    Deletes ONLY entries strictly inside the allowlisted session subdirectories
    (`conversations/`, `brain/`, `implicit/`, `presence/`, `log/`) that are older than
    `max_age_seconds` (default 24h) or beyond `max_entries_per_dir` (default 30 newest).

    Hard safety invariants:
    - Never touches top-level files/dirs in `gemini_dir` (OAuth tokens, `installation_id`,
      `settings.json`, `builtin/`, `bin/`, `annotations/`, `jetski_state.pbtxt`, `cli.log`, etc.).
    - Never deletes fresh entries (`age < min_fresh_age_seconds`, default 1h), even when over limit.
    - Never deletes entries matching `active_conversation_ids`.
    - Never follows symlinks and never operates outside `<data_dir>/.gemini/antigravity-cli`.
    """
    stats = {"removed": 0, "skipped_protected": 0, "errors": 0}

    try:
        isolated_home = Path(data_dir).expanduser().resolve()
    except OSError:
        return stats

    gemini_root = isolated_home / ".gemini"
    gemini_dir = gemini_root / "antigravity-cli"

    if gemini_root.is_symlink() or gemini_dir.is_symlink() or not gemini_dir.is_dir():
        return stats

    try:
        resolved_gemini_dir = gemini_dir.resolve(strict=True)
        if not resolved_gemini_dir.is_relative_to(isolated_home):
            return stats
        if resolved_gemini_dir.name != "antigravity-cli" or resolved_gemini_dir.parent.name != ".gemini":
            return stats
    except (OSError, ValueError):
        return stats

    now_ts = time.time() if now is None else float(now)
    min_fresh = max(0.0, float(min_fresh_age_seconds))
    max_age = max(min_fresh, float(max_age_seconds))
    keep_limit = max(0, int(max_entries_per_dir))
    active_ids = {
        str(cid).strip()
        for cid in (active_conversation_ids or ())
        if isinstance(cid, str) and cid.strip()
    }

    for subdir_name in _CLEANUP_TARGET_SUBDIRS:
        if subdir_name in _PROTECTED_GEMINI_ENTRIES:
            continue
        subdir = resolved_gemini_dir / subdir_name
        if subdir.is_symlink() or not subdir.is_dir():
            continue
        try:
            resolved_subdir = subdir.resolve(strict=True)
            if not resolved_subdir.is_relative_to(resolved_gemini_dir) or resolved_subdir == resolved_gemini_dir:
                continue
        except (OSError, ValueError):
            continue

        candidates: list[tuple[float, float, Path, bool]] = []
        try:
            with os.scandir(resolved_subdir) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            stats["skipped_protected"] += 1
                            continue
                        if entry.name in _PROTECTED_GEMINI_ENTRIES:
                            stats["skipped_protected"] += 1
                            continue
                        entry_path = Path(entry.path)
                        resolved_entry = entry_path.resolve(strict=True)
                        if (
                            not resolved_entry.is_relative_to(resolved_subdir)
                            or resolved_entry == resolved_subdir
                        ):
                            stats["skipped_protected"] += 1
                            continue
                        st = entry.stat(follow_symlinks=False)
                        eff_mtime = _effective_entry_mtime(
                            entry_path,
                            st.st_mtime,
                            is_dir=entry.is_dir(follow_symlinks=False),
                        )
                        age = now_ts - eff_mtime
                        is_protected = (
                            age < min_fresh
                            or _matches_active_conversation(entry.name, active_ids)
                        )
                        candidates.append((eff_mtime, age, entry_path, is_protected))
                    except (OSError, ValueError):
                        stats["errors"] += 1
                        continue
        except OSError:
            stats["errors"] += 1
            continue

        # Sort newest first so the most recent `keep_limit` entries are retained
        candidates.sort(key=lambda item: item[0], reverse=True)

        for rank, (_eff_mtime, age, entry_path, is_protected) in enumerate(candidates):
            if is_protected:
                stats["skipped_protected"] += 1
                continue
            if age > max_age or rank >= keep_limit:
                try:
                    if _safe_delete_confined_entry(entry_path, resolved_subdir):
                        stats["removed"] += 1
                    else:
                        stats["skipped_protected"] += 1
                except OSError:
                    stats["errors"] += 1

    if stats["removed"] > 0:
        logger.debug(
            "Antigravity session cleanup in %s removed=%d protected=%d errors=%d",
            resolved_gemini_dir,
            stats["removed"],
            stats["skipped_protected"],
            stats["errors"],
        )
    return stats
