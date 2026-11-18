"""Antigravity provider package: Dual interface (Direct SDK + Subprocess CLI adapter)."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from bot.antigravity.auth import (
    build_clean_child_env,
    check_antigravity_auth,
    check_antigravity_token_file_evidence,
    check_os_keyring_auth,
    describe_antigravity_auth_source,
    find_existing_oauth_token,
    resolve_antigravity_binary,
    resolve_antigravity_home,
    run_isolated_antigravity_login,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.cli_client import AntigravityCLIClient
from bot.antigravity.errors import (
    AntigravityAuthError,
    AntigravityBillingError,
    AntigravityContextOverflowError,
    AntigravityError,
    AntigravityNativeToolAbort,
    AntigravityRateLimitError,
    AntigravityServerError,
    ErrorCategory,
    ErrorVerdict,
    calculate_backoff,
    classify_error,
    extract_retry_after,
    retry_with_backoff,
)
from bot.antigravity.sdk_client import AntigravitySDKClient, has_gemini_api_key, is_sdk_available
from bot.antigravity.stream import AntigravityStreamParser, StreamChunk, parse_ndjson_stream
from bot.core.models import MediaAttachment, ProviderResponse, UsageInfo

logger = logging.getLogger(__name__)


class AntigravityConfig(BaseModel):
    """Configuration for Antigravity provider."""

    data_dir: Path = Field(default_factory=lambda: Path("data/antigravity"))
    model: str = Field(
        default_factory=lambda: os.environ.get("ANTIGRAVITY_MODEL", "gemini-4-hydrogen").strip()
        or "gemini-4-hydrogen"
    )
    effort: Optional[str] = Field(
        default_factory=lambda: os.environ.get("ANTIGRAVITY_EFFORT", "low").strip() or "low"
    )
    enable_sdk: bool = True
    enable_cli_fallback: bool = True
    max_retries: int = 3
    rate_limit_max_retries: int = 0
    rate_limit_cooldown_seconds: float = 300.0
    backoff_base: float = 1.0
    backoff_max: float = 30.0


class AntigravityProvider:
    """Unified Antigravity Provider adhering to PROJECT.md interface contract.

    Primary transport: Direct Python SDK (`google-antigravity`).
    Fallback transport: Subprocess CLI adapter (`agy --output-format stream-json`).
    """

    def __init__(self, config: Optional[AntigravityConfig] = None) -> None:
        self.config = config or AntigravityConfig()
        self.data_dir = resolve_antigravity_home(self.config.data_dir)

        # Shared rate-limit / billing circuit breaker state
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str = ""
        self._cooldown_category: Optional[ErrorCategory] = None

        # Skip dead SDK client when GEMINI_API_KEY is not configured
        self._sdk_client: Optional[AntigravitySDKClient] = None
        if self.config.enable_sdk and is_sdk_available() and has_gemini_api_key():
            try:
                self._sdk_client = AntigravitySDKClient(
                    data_dir=self.data_dir,
                    default_model=self.config.model,
                )
            except Exception as exc:
                logger.warning("Could not initialize Antigravity SDK client: %s", exc)
        elif self.config.enable_sdk and is_sdk_available() and not has_gemini_api_key():
            logger.debug("Skipping direct SDK client initialization (GEMINI_API_KEY is not set).")

        self._cli_client: Optional[AntigravityCLIClient] = None
        if self.config.enable_cli_fallback:
            self._cli_client = AntigravityCLIClient(
                data_dir=self.data_dir,
                default_model=self.config.model,
                default_effort=self.config.effort,
            )

    def cooldown_remaining(self) -> float:
        """Return remaining seconds on the shared rate-limit/billing cooldown, or 0.0."""
        if self._cooldown_until <= 0.0:
            return 0.0
        remaining = self._cooldown_until - time.monotonic()
        if remaining <= 0.0:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            self._cooldown_category = None
            return 0.0
        return remaining

    def is_rate_limited(self) -> bool:
        """Return True while the shared rate-limit/billing circuit breaker is open."""
        return self.cooldown_remaining() > 0.0

    def trigger_cooldown(
        self,
        duration_seconds: float,
        reason: str = "",
        category: ErrorCategory = ErrorCategory.RATE_LIMIT,
    ) -> None:
        """Open or extend the shared circuit breaker cooldown for `duration_seconds`."""
        if duration_seconds <= 0.0:
            return
        new_until = time.monotonic() + float(duration_seconds)
        if new_until > self._cooldown_until:
            self._cooldown_until = new_until
            self._cooldown_reason = reason
            self._cooldown_category = category

    def reset_cooldown(self) -> None:
        """Clear any active rate-limit/billing circuit breaker cooldown."""
        self._cooldown_until = 0.0
        self._cooldown_reason = ""
        self._cooldown_category = None

    def check_auth(self) -> bool:
        """Verify credential presence."""
        return check_antigravity_auth(data_dir=self.data_dir)

    def describe_auth(self) -> str:
        """Return credential location description."""
        return describe_antigravity_auth_source(data_dir=self.data_dir)

    def login(self) -> int:
        """Interactive terminal OAuth login runner."""
        return run_isolated_antigravity_login(data_dir=self.data_dir)

    def close(self) -> None:
        """Clean up background workers."""
        if self._cli_client is not None:
            self._cli_client.close()

    async def generate_response(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        media_attachments: Optional[Sequence[MediaAttachment]] = None,
        conversation_id: Optional[str] = None,
    ) -> ProviderResponse:
        """Generate response conforming to PROJECT.md § Interface Contracts.

        Tries direct Python SDK first; automatically falls back to subprocess CLI adapter.
        Fails fast when the shared rate-limit/billing circuit breaker is active.
        """
        remaining = self.cooldown_remaining()
        if remaining > 0.0:
            if self._cooldown_category == ErrorCategory.BILLING:
                msg = (
                    f"Antigravity billing quota has been exhausted; circuit breaker active "
                    f"({remaining:.1f}s remaining): {self._cooldown_reason or 'cooldown'}"
                )
                raise AntigravityBillingError(
                    msg,
                    verdict=ErrorVerdict(
                        category=ErrorCategory.BILLING,
                        retryable=False,
                        should_compress=False,
                        should_fallback=False,
                        message=msg,
                    ),
                    retry_after=remaining,
                )
            msg = (
                f"Antigravity rate limit circuit breaker active "
                f"({remaining:.1f}s remaining): {self._cooldown_reason or 'cooldown'}"
            )
            raise AntigravityRateLimitError(
                msg,
                verdict=ErrorVerdict(
                    category=ErrorCategory.RATE_LIMIT,
                    retryable=True,
                    should_compress=False,
                    should_fallback=False,
                    message=msg,
                ),
                retry_after=remaining,
            )

        async def _attempt_call() -> ProviderResponse:
            sdk_error: Optional[Exception] = None

            # 1. Attempt Direct Python SDK
            if self._sdk_client is not None:
                try:
                    return await self._sdk_client.generate_response(
                        prompt=prompt,
                        system_instructions=system_instructions,
                        media_attachments=media_attachments,
                        conversation_id=conversation_id,
                        model=self.config.model,
                    )
                except AntigravityNativeToolAbort:
                    raise
                except Exception as exc:
                    verdict = classify_error(exc)
                    if not verdict.should_fallback:
                        raise
                    if "Direct SDK requires GEMINI_API_KEY" in str(exc):
                        logger.debug("Skipping direct SDK (no GEMINI_API_KEY set); using OAuth CLI adapter.")
                    else:
                        logger.info(
                            "Antigravity SDK invocation failed (%s). Triggering CLI fallback adapter.",
                            exc,
                        )
                    sdk_error = exc

            # 2. Subprocess CLI Fallback (supports native image vision via ACP when binary image bytes exist)
            if self._cli_client is not None:
                # Prepare media prompt augmentation if attachments present
                augmented_prompt = prompt
                if media_attachments:
                    descriptions = [
                        att.description for att in media_attachments if att.description
                    ]
                    if descriptions:
                        augmented_prompt = (
                            f"[Attached Media Descriptions: {'; '.join(descriptions)}]\n\n"
                            f"{prompt}"
                        )

                try:
                    return await self._cli_client.generate_response(
                        prompt=augmented_prompt,
                        system_instructions=system_instructions,
                        media_attachments=media_attachments,
                        conversation_id=conversation_id,
                        model=self.config.model,
                        effort=self.config.effort,
                    )
                except TypeError as exc:
                    if "media_attachments" not in str(exc):
                        raise
                    return await self._cli_client.generate_response(
                        prompt=augmented_prompt,
                        system_instructions=system_instructions,
                        conversation_id=conversation_id,
                        model=self.config.model,
                        effort=self.config.effort,
                    )

            if sdk_error is not None:
                raise sdk_error
            raise RuntimeError("No Antigravity provider client (SDK or CLI) is available.")

        try:
            return await retry_with_backoff(
                _attempt_call,
                max_attempts=self.config.max_retries,
                base_delay=self.config.backoff_base,
                max_delay=self.config.backoff_max,
                rate_limit_max_attempts=max(1, 1 + self.config.rate_limit_max_retries),
            )
        except Exception as exc:
            verdict = getattr(exc, "verdict", None) or classify_error(exc)
            if verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.BILLING):
                retry_after = extract_retry_after(exc)
                cooldown = (
                    retry_after
                    if retry_after is not None and retry_after > 0
                    else self.config.rate_limit_cooldown_seconds
                )
                self.trigger_cooldown(cooldown, reason=str(exc), category=verdict.category)
                logger.warning(
                    "Antigravity %s circuit breaker tripped for %.1fs: %s",
                    verdict.category.value,
                    cooldown,
                    exc,
                )
            raise


__all__ = [
    "AntigravityAuthError",
    "AntigravityBillingError",
    "AntigravityCLIClient",
    "AntigravityConfig",
    "AntigravityContextOverflowError",
    "AntigravityError",
    "AntigravityNativeToolAbort",
    "AntigravityProvider",
    "AntigravityRateLimitError",
    "AntigravitySDKClient",
    "AntigravityServerError",
    "AntigravityStreamParser",
    "ErrorCategory",
    "ErrorVerdict",
    "MediaAttachment",
    "ProviderResponse",
    "StreamChunk",
    "UsageInfo",
    "build_clean_child_env",
    "calculate_backoff",
    "check_antigravity_auth",
    "check_antigravity_token_file_evidence",
    "check_os_keyring_auth",
    "describe_antigravity_auth_source",
    "extract_retry_after",
    "find_existing_oauth_token",
    "has_gemini_api_key",
    "is_sdk_available",
    "parse_ndjson_stream",
    "resolve_antigravity_binary",
    "resolve_antigravity_home",
    "retry_with_backoff",
    "run_isolated_antigravity_login",
    "setup_isolated_home",
    "sync_updated_isolated_tokens",
]
