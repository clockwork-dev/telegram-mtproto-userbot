"""Secret sanitization for Python logs, exception traces, and outbound Telegram messages."""

from __future__ import annotations

import logging
import os
import re
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Optional, Pattern, Sequence

REDACTED = "[REDACTED]"
_MIN_SECRET_LEN = 6

_NON_SECRET_ENV_KEYS: frozenset[str] = frozenset(
    {
        "TELEGRAM_API_ID",
        "TELEGRAM_SESSION_PATH",
        "ANTIGRAVITY_DATA_DIR",
        "ANTIGRAVITY_BOT_HOME",
        "ANTIGRAVITY_BINARY",
        "AGY_CLI_PATH",
        "BOT_CONFIG_PATH",
        "PATH",
        "HOME",
        "USERPROFILE",
        "PWD",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "LOG_LEVEL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "TMPDIR",
        "TEMP",
        "TMP",
    }
)

_DUMMY_VALUES: frozenset[str] = frozenset(
    {
        "0123456789abcdef0123456789abcdef",
        "0123456789abcdef",
        "12345678",
        "12345",
        REDACTED,
    }
)

_PROXY_ENV_KEYS: frozenset[str] = frozenset(
    {
        "ANTIGRAVITY_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    }
)

_SECRET_ENV_KEY_RE = re.compile(
    r"(?:^|_)(?:"
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|AUTH|BEARER|"
    r"COOKIE|PRIVATE|JWT|SIGNATURE|WEBHOOK|CONN|CONNECTION|"
    r"DSN|DATABASE_URL|REDIS_URL|MONGO_URL|AMQP_URL|SMTP_URL|"
    r"AWS_ACCESS|AWS_SECRET|AWS_SESSION|AZURE_CLIENT|GCP_|GOOGLE_APPLICATION|"
    r"OPENAI|ANTHROPIC|GEMINI|CLAUDE|COHERE|MISTRAL|GROQ|DEEPSEEK|OPENROUTER|"
    r"HUGGINGFACE|HF_TOKEN|REPLICATE|TOGETHER|FIREWORKS|PERPLEXITY|XAI|"
    r"GITHUB|GH_TOKEN|GITLAB|BITBUCKET|NPM_TOKEN|PYPI|SLACK|DISCORD|"
    r"TELEGRAM_BOT_TOKEN|TELEGRAM_API_HASH|TELEGRAM_PHONE|API_HASH|PHONE|PROXY"
    r")(?:$|_)",
    re.IGNORECASE,
)

_EXPLICIT_SECRET_ENV_KEYS: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_API_HASH",
    "TELEGRAM_PHONE",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTIGRAVITY_API_KEY",
)

_URL_USERINFO_EXTRACT_RE = re.compile(
    r"://([^/\s:@]+):([^@\s]+)@"
)

# Pattern-based secret redactors (applied in order)
_TELEGRAM_BOT_TOKEN_RE = re.compile(r"(?<!\d)\d{6,}:[A-Za-z0-9_-]{25,}")
_BEARER_TOKEN_RE = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-~+/]+=*")
_API_TOKEN_RE = re.compile(
    r"(?:"
    r"ya29\.[0-9A-Za-z_\-]{16,}"
    r"|1//0[0-9A-Za-z_\-]{20,}"
    r"|AIza[0-9A-Za-z_\-]{20,}"
    r"|AQ\.[0-9A-Za-z_\-]{20,}"
    r"|sk-(?:proj-|ant-[A-Za-z0-9_\-]*|svcacct-)?[A-Za-z0-9_\-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9_\-]{16,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{16,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|eyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}"
    r")"
)
_PEM_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
)
_PEM_PRIVATE_KEY_HEADER_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_WIREGUARD_KEY_RE = re.compile(
    r"(?i)((?:PrivateKey|PresharedKey|private[_\s]+key|preshared[_\s]+key)\s*[:=]\s*)"
    r"([A-Za-z0-9+/]{42,44}={0,2})"
)
_LABELED_HEX_HASH_RE = re.compile(
    r"(?i)((?:api_hash|TELEGRAM_API_HASH|app_hash)\s*[:=]\s*[\"']?)"
    r"([0-9a-f]{32,64})"
    r"([\"']?)"
)
_URL_USERINFO_REDACT_RE = re.compile(
    r"((?:https?|socks5h?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|rediss?|amqps?)://[^/\s:@]+:)[^@\s]+(@)"
)


def _looks_like_non_secret_path_or_id(value: str) -> bool:
    """Return True if value is a pure numeric ID or plain filesystem path without credentials."""
    if value.isdigit():
        return True
    if (value.startswith("/") or value.startswith("data/") or value.startswith("./")) and (
        "://" not in value and "PRIVATE KEY" not in value
    ):
        return True
    return False


class SecretSanitizer:
    """Thread-safe sanitizer redacting registered secrets and credential patterns."""

    def __init__(
        self,
        extra_secrets: Sequence[str] = (),
        env_path: Optional[Path | str] = None,
        load_environ: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._secrets: set[str] = set()
        self._literal_pattern: Optional[Pattern[str]] = None
        self._pattern_dirty = False

        if load_environ:
            self._load_from_environ()
        if env_path is not None:
            self.load_env_file(Path(env_path))
        for secret in extra_secrets:
            self.register_secret(secret)

    def _add_secret_variants_locked(self, cleaned: str) -> None:
        if len(cleaned) < _MIN_SECRET_LEN:
            return
        if cleaned in _DUMMY_VALUES or cleaned.lower() in _DUMMY_VALUES:
            return
        self._secrets.add(cleaned)
        quoted = urllib.parse.quote(cleaned, safe="")
        if quoted != cleaned and len(quoted) >= _MIN_SECRET_LEN:
            self._secrets.add(quoted)
        quoted_plus = urllib.parse.quote_plus(cleaned)
        if quoted_plus != cleaned and len(quoted_plus) >= _MIN_SECRET_LEN:
            self._secrets.add(quoted_plus)
        self._pattern_dirty = True

    def register_secret(self, value: Optional[str]) -> None:
        """Register a literal secret string (and URL-encoded variants) for redaction."""
        if not value or not isinstance(value, str):
            return
        cleaned = value.strip()
        if len(cleaned) < _MIN_SECRET_LEN:
            return
        if cleaned in _DUMMY_VALUES or cleaned.lower() in _DUMMY_VALUES:
            return
        with self._lock:
            self._add_secret_variants_locked(cleaned)

    def _register_env_entry(self, key: str, value: str) -> None:
        key_clean = key.strip()
        key_upper = key_clean.upper()
        if not key_clean or key_upper in _NON_SECRET_ENV_KEYS:
            return
        val_clean = value.strip()
        if len(val_clean) < _MIN_SECRET_LEN:
            return
        if val_clean in _DUMMY_VALUES or val_clean.lower() in _DUMMY_VALUES:
            return

        if key_upper in _PROXY_ENV_KEYS or "PROXY" in key_upper:
            m = _URL_USERINFO_EXTRACT_RE.search(val_clean)
            if m:
                user, password = m.group(1), m.group(2)
                self.register_secret(f"{user}:{password}")
                self.register_secret(password)
            return

        if _looks_like_non_secret_path_or_id(val_clean):
            return

        if key_upper in _EXPLICIT_SECRET_ENV_KEYS or _SECRET_ENV_KEY_RE.search(key_clean):
            self.register_secret(val_clean)

    def _load_from_environ(self) -> None:
        for key in _EXPLICIT_SECRET_ENV_KEYS:
            val = os.environ.get(key)
            if val:
                self._register_env_entry(key, val)
        for key, val in os.environ.items():
            if val and (_SECRET_ENV_KEY_RE.search(key) or key.upper() in _PROXY_ENV_KEYS):
                self._register_env_entry(key, val)

    def load_env_file(self, path: Path | str) -> int:
        """Parse a .env file without mutating os.environ or logging secrets.

        Returns the number of entries processed for registration.
        """
        env_file = Path(path)
        if not env_file.is_file():
            return 0
        try:
            content = env_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0

        count = 0
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if not k or not v:
                continue
            if k.upper() in _NON_SECRET_ENV_KEYS:
                continue
            if k.upper() in _PROXY_ENV_KEYS or "PROXY" in k.upper():
                m = _URL_USERINFO_EXTRACT_RE.search(v)
                if m:
                    self.register_secret(f"{m.group(1)}:{m.group(2)}")
                    self.register_secret(m.group(2))
                    count += 1
                continue
            if _looks_like_non_secret_path_or_id(v):
                continue
            if len(v) >= _MIN_SECRET_LEN and v not in _DUMMY_VALUES and v.lower() not in _DUMMY_VALUES:
                self.register_secret(v)
                count += 1
        return count

    def _get_literal_pattern(self) -> Optional[Pattern[str]]:
        with self._lock:
            if self._pattern_dirty:
                if self._secrets:
                    ordered = sorted(self._secrets, key=len, reverse=True)
                    self._literal_pattern = re.compile(
                        "|".join(re.escape(s) for s in ordered)
                    )
                else:
                    self._literal_pattern = None
                self._pattern_dirty = False
            return self._literal_pattern

    def sanitize(self, text: Any) -> str:
        """Redact registered secrets and credential patterns from text."""
        if text is None:
            return ""
        result = text if isinstance(text, str) else str(text)
        if not result:
            return result

        literal_pat = self._get_literal_pattern()
        if literal_pat is not None:
            result = literal_pat.sub(REDACTED, result)

        result = _PEM_PRIVATE_KEY_BLOCK_RE.sub(REDACTED, result)
        result = _PEM_PRIVATE_KEY_HEADER_RE.sub(REDACTED, result)
        result = _TELEGRAM_BOT_TOKEN_RE.sub(REDACTED, result)
        result = _BEARER_TOKEN_RE.sub(rf"\1{REDACTED}", result)
        result = _API_TOKEN_RE.sub(REDACTED, result)
        result = _WIREGUARD_KEY_RE.sub(rf"\1{REDACTED}", result)
        result = _LABELED_HEX_HASH_RE.sub(rf"\1{REDACTED}\3", result)
        result = _URL_USERINFO_REDACT_RE.sub(rf"\1{REDACTED}\2", result)
        return result


class SanitizingLogFilter(logging.Filter):
    """Logging filter that formats and redacts secrets from message, args, exc_info, and stack_info."""

    def __init__(self, sanitizer: SecretSanitizer) -> None:
        super().__init__()
        self.sanitizer = sanitizer

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            formatted = record.getMessage()
        except Exception:
            raw_msg = str(record.msg)
            if record.args:
                if isinstance(record.args, tuple):
                    safe_args = " ".join(self.sanitizer.sanitize(a) for a in record.args)
                elif isinstance(record.args, dict):
                    safe_args = " ".join(
                        f"{k}={self.sanitizer.sanitize(v)}" for k, v in record.args.items()
                    )
                else:
                    safe_args = self.sanitizer.sanitize(record.args)
                formatted = f"{raw_msg} {safe_args}"
            else:
                formatted = raw_msg

        record.msg = self.sanitizer.sanitize(formatted)
        record.args = None

        if record.exc_info and not record.exc_text:
            try:
                formatter = logging.Formatter()
                record.exc_text = formatter.formatException(record.exc_info)
            except Exception:
                record.exc_text = repr(record.exc_info[1]) if len(record.exc_info) > 1 else ""
        if record.exc_text:
            record.exc_text = self.sanitizer.sanitize(record.exc_text)

        if record.stack_info:
            record.stack_info = self.sanitizer.sanitize(record.stack_info)

        return True


def _attach_filter_idempotent(target: logging.Filterer, log_filter: SanitizingLogFilter) -> None:
    for existing in list(target.filters):
        if isinstance(existing, SanitizingLogFilter):
            target.removeFilter(existing)
    target.addFilter(log_filter)


def install_log_sanitizer(sanitizer: SecretSanitizer) -> SanitizingLogFilter:
    """Install SanitizingLogFilter idempotently on root handlers and key loggers."""
    log_filter = SanitizingLogFilter(sanitizer)
    root_logger = logging.getLogger()
    _attach_filter_idempotent(root_logger, log_filter)
    for handler in root_logger.handlers:
        _attach_filter_idempotent(handler, log_filter)

    for logger_name in (
        "bot",
        "telethon",
        "httpx",
        "httpx2",
        "google_genai",
        "google",
        "aiohttp",
        "edge_tts",
        "telegram",
    ):
        sub_logger = logging.getLogger(logger_name)
        _attach_filter_idempotent(sub_logger, log_filter)
        for handler in sub_logger.handlers:
            _attach_filter_idempotent(handler, log_filter)

    return log_filter
