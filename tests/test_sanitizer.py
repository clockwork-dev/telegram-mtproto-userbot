"""Unit tests for SecretSanitizer, SanitizingLogFilter, and log/credential redaction."""

from __future__ import annotations

import logging
import os
import urllib.parse
from pathlib import Path

from bot.security.sanitizer import (
    REDACTED,
    SanitizingLogFilter,
    SecretSanitizer,
    install_log_sanitizer,
)


def test_sanitizer_env_loading_and_non_secret_exclusions(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "TELEGRAM_API_ID=987654321",
                "TELEGRAM_SESSION_PATH=/opt/data/session/userbot.session",
                "ANTIGRAVITY_DATA_DIR=/opt/data/antigravity",
                "ANTIGRAVITY_BINARY=/usr/local/bin/agy",
                "TELEGRAM_BOT_TOKEN=123456789:AAFakeBotTokenForTest1234567890abc",
                "TELEGRAM_API_HASH=deadbeefcafebabe1234567890abcdef",
                "TELEGRAM_PHONE=+79991112233",
                "ANTIGRAVITY_PROXY=http://proxyuser:SuperProxyPass99@10.0.0.2:8080",
                "SHORT_VAL=abcde",
                "DUMMY_HASH=0123456789abcdef0123456789abcdef",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    sanitizer = SecretSanitizer(env_path=env_file, load_environ=False)
    # load_env_file must NOT mutate os.environ
    assert "TELEGRAM_BOT_TOKEN" not in os.environ

    # Secrets and proxy credentials redacted
    sample = (
        "tok=123456789:AAFakeBotTokenForTest1234567890abc "
        "hash=deadbeefcafebabe1234567890abcdef "
        "phone=+79991112233 "
        "proxy=http://proxyuser:SuperProxyPass99@10.0.0.2:8080"
    )
    cleaned = sanitizer.sanitize(sample)
    assert "AAFakeBotTokenForTest" not in cleaned
    assert "deadbeefcafebabe1234567890abcdef" not in cleaned
    assert "+79991112233" not in cleaned
    assert "SuperProxyPass99" not in cleaned
    assert "10.0.0.2:8080" in cleaned  # host/port preserved

    # Non-secret env values, short values (<6 chars), and dummy placeholders must NOT be redacted
    safe_text = (
        "id=987654321 sess=/opt/data/session/userbot.session "
        "agy=/usr/local/bin/agy short=abcde dummy=0123456789abcdef0123456789abcdef"
    )
    assert sanitizer.sanitize(safe_text) == safe_text


def test_sanitizer_patterns_and_git_sha_preservation() -> None:
    sanitizer = SecretSanitizer(extra_secrets=["My$ecret/Value+1"], load_environ=False)

    # URL-encoded registered secret
    encoded = urllib.parse.quote("My$ecret/Value+1", safe="")
    assert sanitizer.sanitize(f"raw=My$ecret/Value+1 enc={encoded}") == f"raw={REDACTED} enc={REDACTED}"

    # Telegram bot token inside Bot API URL (preceded directly by `bot`)
    url_msg = "POST https://api.telegram.org/bot8877665544:AAHTestSecretTokenValue123456789/getUpdates failed"
    redacted_url = sanitizer.sanitize(url_msg)
    assert "8877665544:AAHTestSecretTokenValue123456789" not in redacted_url
    assert f"https://api.telegram.org/bot{REDACTED}/getUpdates" in redacted_url

    # Bearer, Google OAuth access token (ya29.), Google OAuth refresh token (1//0...), API keys, AWS, JWT
    tokens = (
        "Authorization: Bearer abc.def.ghi12345 "
        "oauth=ya29.a0AfH6SMC1234567890abcdef "
        "refresh=1//0gTestRefreshTokenValue1234567890ABCDEF "
        "gkey=AIzaSyDummyKeyValue1234567890abcdef "
        "aqkey=AQ.Ab8RN6DummyGoogleKeyValue1234567890abcdef "
        "openai=sk-proj-TestKey1234567890abcdef "
        "gh=ghp_1234567890abcdefghij1234567890 "
        "aws=AKIAIOSFODNN7EXAMPLE "
        "jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
    )
    cleaned_tokens = sanitizer.sanitize(tokens)
    for leaked in (
        "abc.def.ghi12345",
        "ya29.a0AfH6SMC1234567890abcdef",
        "1//0gTestRefreshTokenValue1234567890ABCDEF",
        "AIzaSyDummyKeyValue1234567890abcdef",
        "AQ.Ab8RN6DummyGoogleKeyValue1234567890abcdef",
        "sk-proj-TestKey1234567890abcdef",
        "ghp_1234567890abcdefghij1234567890",
        "AKIAIOSFODNN7EXAMPLE",
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ):
        assert leaked not in cleaned_tokens

    # WireGuard PrivateKey & PresharedKey (context-scoped)
    wg_cfg = (
        "[Interface]\n"
        "PrivateKey = cGFzc3dvcmRfMzJfYnl0ZXNfd2lyZWd1YXJkX2tleQ==\n"
        "[Peer]\n"
        "PresharedKey = YW5vdGhlcl9wcmVzaGFyZWRfa2V5XzMyX2J5dGVzX2s=\n"
    )
    cleaned_wg = sanitizer.sanitize(wg_cfg)
    assert "cGFzc3dvcmRfMzJfYnl0ZXNfd2lyZWd1YXJkX2tleQ==" not in cleaned_wg
    assert "YW5vdGhlcl9wcmVzaGFyZWRfa2V5XzMyX2J5dGVzX2s=" not in cleaned_wg
    assert f"PrivateKey = {REDACTED}" in cleaned_wg
    assert f"PresharedKey = {REDACTED}" in cleaned_wg

    # PEM Private Key
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAtest\n-----END RSA PRIVATE KEY-----"
    assert sanitizer.sanitize(pem) == REDACTED

    # Labeled api_hash redacted, bare 40-char git commit SHA preserved!
    commit_sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    labeled_and_sha = (
        f"TELEGRAM_API_HASH=0f1e2d3c4b5a69788796a5b4c3d2e1f0 "
        f"Deployed commit {commit_sha} cleanly"
    )
    cleaned_hash = sanitizer.sanitize(labeled_and_sha)
    assert "0f1e2d3c4b5a69788796a5b4c3d2e1f0" not in cleaned_hash
    assert f"TELEGRAM_API_HASH={REDACTED}" in cleaned_hash
    assert commit_sha in cleaned_hash


def test_logging_filter_redacts_args_exc_info_and_telethon_tts_http_loggers() -> None:
    secret = "super_secret_runtime_token_xyz"
    gemini_secret = "AIzaSyLogSanitizerTestKey1234567890abcdef"
    sanitizer = SecretSanitizer(extra_secrets=[secret, gemini_secret], load_environ=False)

    root_logger = logging.getLogger()
    records_out: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records_out.append(self.format(record))

    handler = _ListHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger.addHandler(handler)
    old_level = root_logger.level
    root_logger.setLevel(logging.DEBUG)

    target_loggers = ("telethon", "httpx", "httpx2", "google_genai", "google", "aiohttp", "edge_tts")
    saved_states = {}
    try:
        # Install idempotently twice
        install_log_sanitizer(sanitizer)
        install_log_sanitizer(sanitizer)
        assert sum(isinstance(f, SanitizingLogFilter) for f in handler.filters) == 1

        child = logging.getLogger("bot.telegram.client")
        child.warning(
            "Request failed for %s",
            f"https://api.telegram.org/bot9988776655:AAFtestBotToken1234567890abcdef/getMe?x={secret}",
        )

        try:
            raise RuntimeError(f"Connection failed using 1//0testRefreshToken123456789012345 and {secret}")
        except RuntimeError:
            logging.getLogger("httpx").exception("HTTP call crashed")

        for name in target_loggers:
            sub = logging.getLogger(name)
            saved_states[name] = (sub.level, sub.propagate)
            sub.setLevel(logging.DEBUG)
            sub.propagate = True
            assert any(isinstance(f, SanitizingLogFilter) for f in sub.filters)
            sub.info("Request failed key=%s aq=AQ.Ab8RN6SecretToken1234567890", gemini_secret)

        joined = "\n".join(records_out)
        assert "AAFtestBotToken1234567890abcdef" not in joined
        assert "1//0testRefreshToken123456789012345" not in joined
        assert secret not in joined
        assert gemini_secret not in joined
        assert "AQ.Ab8RN6SecretToken1234567890" not in joined
        assert REDACTED in joined
    finally:
        for name, (lvl, prop) in saved_states.items():
            sub = logging.getLogger(name)
            sub.setLevel(lvl)
            sub.propagate = prop
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
