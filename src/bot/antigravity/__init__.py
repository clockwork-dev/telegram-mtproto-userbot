"""Antigravity provider package: Dual interface (Direct SDK + Subprocess CLI adapter)."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from bot.antigravity.auth import (
    build_clean_child_env,
    check_antigravity_auth,
    check_antigravity_token_file_evidence,
    check_os_keyring_auth,
    cleanup_old_antigravity_sessions,
    describe_antigravity_auth_source,
    find_existing_oauth_token,
    resolve_antigravity_binary,
    resolve_antigravity_home,
    run_isolated_antigravity_login,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.cli_client import (
    AntigravityCLIClient,
    write_chat_pre_tool_hook,
    write_image_pre_tool_hook,
)
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
    ImageGenerationError,
    ImagePolicyRefusalError,
    ImageTimeoutError,
    ImageValidationError,
    calculate_backoff,
    classify_error,
    extract_retry_after,
    retry_with_backoff,
)
from bot.antigravity.sdk_client import AntigravitySDKClient, has_gemini_api_key, is_sdk_available
from bot.antigravity.stream import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    IMAGE_ALLOWED_SUBAGENT_TYPES,
    OPS_ALLOWED_TOOLS,
    SELF_DEV_ALLOWED_TOOLS,
    AntigravityStreamParser,
    StreamChunk,
    ToolProfile,
    is_allowed_image_subagent_step,
    parse_ndjson_stream,
    resolve_allowed_tools,
)
from bot.core.models import GeneratedImageResult, MediaAttachment, ProviderResponse, UsageInfo

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
    turn_timeout_seconds: float = 300.0
    cleanup_interval_seconds: float = 3600.0


class AntigravityProvider:
    """Unified Antigravity Provider adhering to PROJECT.md interface contract.

    Primary transport: Direct Python SDK (`google-antigravity`).
    Fallback transport: Subprocess CLI adapter (`agy --output-format stream-json`).
    """

    def __init__(
        self,
        config: Optional[AntigravityConfig] = None,
        *,
        on_quota_warning: Optional[Callable[[Optional[str], Optional[float], Optional[str]], None]] = None,
    ) -> None:
        self.config = config or AntigravityConfig()
        self.data_dir = resolve_antigravity_home(self.config.data_dir)
        self.on_quota_warning = on_quota_warning

        # Shared & per-model rate-limit / billing circuit breaker state
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str = ""
        self._cooldown_category: Optional[ErrorCategory] = None
        self._model_cooldowns: dict[str, tuple[float, str, ErrorCategory]] = {}

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
                turn_timeout_seconds=self.config.turn_timeout_seconds,
                cleanup_interval_seconds=self.config.cleanup_interval_seconds,
                on_quota_warning=self.on_quota_warning,
            )

    def set_quota_warning_callback(
        self,
        callback: Optional[Callable[[Optional[str], Optional[float], Optional[str]], None]],
    ) -> None:
        """Register or replace soft `[quota]` warning callback on provider and CLI adapter."""
        self.on_quota_warning = callback
        if self._cli_client is not None:
            self._cli_client.set_quota_warning_callback(callback)

    def _model_cooldown_info(self, model: str) -> tuple[float, str, Optional[ErrorCategory]]:
        entry = self._model_cooldowns.get(model)
        if not entry:
            return 0.0, "", None
        until, reason, cat = entry
        remaining = until - time.monotonic()
        if remaining <= 0.0:
            self._model_cooldowns.pop(model, None)
            return 0.0, "", None
        return remaining, reason, cat

    def cooldown_remaining(self, model: Optional[str] = None) -> float:
        """Return remaining seconds on cooldown for ``model`` (or default model / global breaker), or 0.0."""
        target = model if model is not None else self.config.model
        if target is not None:
            rem_m, _, _ = self._model_cooldown_info(target)
            if rem_m > 0.0:
                return rem_m
        if self._cooldown_until <= 0.0:
            return 0.0
        remaining = self._cooldown_until - time.monotonic()
        if remaining <= 0.0:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            self._cooldown_category = None
            return 0.0
        return remaining

    def is_rate_limited(self, model: Optional[str] = None) -> bool:
        """Return True while the circuit breaker (for ``model`` or default model) is open."""
        target = model if model is not None else self.config.model
        return self.cooldown_remaining(model=target) > 0.0

    def trigger_cooldown(
        self,
        duration_seconds: float,
        reason: str = "",
        category: ErrorCategory = ErrorCategory.RATE_LIMIT,
        model: Optional[str] = None,
    ) -> None:
        """Open or extend circuit breaker cooldown for ``model`` (or global if ``model`` is None)."""
        if duration_seconds <= 0.0:
            return
        new_until = time.monotonic() + float(duration_seconds)
        if model is not None:
            prev = self._model_cooldowns.get(model)
            if prev is None or new_until > prev[0]:
                self._model_cooldowns[model] = (new_until, reason, category)
            return
        if new_until > self._cooldown_until:
            self._cooldown_until = new_until
            self._cooldown_reason = reason
            self._cooldown_category = category

    def reset_cooldown(self, model: Optional[str] = None) -> None:
        """Clear active cooldown for ``model``, or all cooldowns when ``model`` is None."""
        if model is not None:
            self._model_cooldowns.pop(model, None)
            return
        self._cooldown_until = 0.0
        self._cooldown_reason = ""
        self._cooldown_category = None
        self._model_cooldowns.clear()

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

    def _raise_if_cooldown_active(self, effective_model: str) -> None:
        """Fail fast when per-model or global rate-limit/billing cooldown is open."""
        rem_model, reason_model, cat_model = self._model_cooldown_info(effective_model)
        if rem_model > 0.0:
            remaining = rem_model
            cd_reason = reason_model
            cd_category = cat_model
        elif self._cooldown_until > 0.0:
            remaining = self._cooldown_until - time.monotonic()
            if remaining <= 0.0:
                self._cooldown_until = 0.0
                self._cooldown_reason = ""
                self._cooldown_category = None
                remaining = 0.0
            cd_reason = self._cooldown_reason
            cd_category = self._cooldown_category
        else:
            remaining = 0.0
            cd_reason = ""
            cd_category = None

        if remaining > 0.0:
            if cd_category == ErrorCategory.BILLING:
                msg = (
                    f"Antigravity billing quota has been exhausted; circuit breaker active "
                    f"({remaining:.1f}s remaining): {cd_reason or 'cooldown'}"
                )
                raise AntigravityBillingError(
                    msg,
                    verdict=ErrorVerdict(
                        category=ErrorCategory.BILLING,
                        retryable=False,
                        should_compress=False,
                        should_fallback=True,
                        message=msg,
                    ),
                    retry_after=remaining,
                )
            msg = (
                f"Antigravity rate limit circuit breaker active for {effective_model} "
                f"({remaining:.1f}s remaining): {cd_reason or 'cooldown'}"
            )
            raise AntigravityRateLimitError(
                msg,
                verdict=ErrorVerdict(
                    category=ErrorCategory.RATE_LIMIT,
                    retryable=True,
                    should_compress=False,
                    should_fallback=True,
                    message=msg,
                ),
                retry_after=remaining,
            )

    def _trip_cooldown_if_applicable(self, exc: Exception, effective_model: str) -> None:
        """Trip per-model circuit breaker when exception indicates rate limit or billing exhaustion."""
        verdict = getattr(exc, "verdict", None) or classify_error(exc)
        if verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.BILLING):
            retry_after = extract_retry_after(exc)
            cooldown = (
                retry_after
                if retry_after is not None and retry_after > 0
                else self.config.rate_limit_cooldown_seconds
            )
            self.trigger_cooldown(
                cooldown,
                reason=str(exc),
                category=verdict.category,
                model=effective_model,
            )
            logger.warning(
                "Antigravity %s circuit breaker tripped for %s (%.1fs): %s",
                verdict.category.value,
                effective_model,
                cooldown,
                exc,
            )

    async def generate_image(
        self,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        reference_images: list[MediaAttachment] | None = None,
        model: Optional[str] = None,
        timeout_seconds: float = 120.0,
        output_dir: str | Path = "data/tmp/images",
        max_image_size_mb: float = 20.0,
        cleanup_artifacts: bool = True,
    ) -> GeneratedImageResult:
        """Generate an image natively via isolated `agy` CLI `stream-json` oneshot."""
        effective_model = (model or "").strip() or self.config.model
        self._raise_if_cooldown_active(effective_model)

        if self._cli_client is None:
            raise ImageGenerationError(
                "Antigravity CLI client is disabled or unavailable for native image generation."
            )

        try:
            return await self._cli_client.generate_image(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                reference_images=reference_images,
                model=effective_model,
                timeout_seconds=timeout_seconds,
                output_dir=output_dir,
                max_image_size_mb=max_image_size_mb,
                cleanup_artifacts=cleanup_artifacts,
            )
        except Exception as exc:
            self._trip_cooldown_if_applicable(exc, effective_model)
            raise

    async def generate_response(
        self,
        prompt: str,
        system_instructions: Optional[str] = None,
        media_attachments: Optional[Sequence[MediaAttachment]] = None,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        tool_profile: ToolProfile = "chat",
        mode: Optional[str] = None,
        cwd: Optional[Path | str] = None,
        allowed_tools: Optional[frozenset[str] | Collection[str]] = None,
    ) -> ProviderResponse:
        """Generate response conforming to PROJECT.md § Interface Contracts.

        Tries direct Python SDK first (when no custom `allowed_tools` restriction is requested);
        automatically falls back to subprocess CLI adapter.
        Fails fast when the shared rate-limit/billing circuit breaker is active.
        """
        passed_allowed = (
            frozenset(str(name).strip().lower() for name in allowed_tools if str(name).strip())
            if allowed_tools is not None
            else None
        )
        resolve_allowed_tools(tool_profile, allowed_tools=passed_allowed)
        effective_model = model or self.config.model
        effective_effort = effort if effort is not None else self.config.effort

        self._raise_if_cooldown_active(effective_model)

        async def _attempt_call() -> ProviderResponse:
            sdk_error: Optional[Exception] = None

            # 1. Attempt Direct Python SDK (skipped when explicit allowed_tools isolation is requested)
            if self._sdk_client is not None and passed_allowed is None:
                try:
                    try:
                        return await self._sdk_client.generate_response(
                            prompt=prompt,
                            system_instructions=system_instructions,
                            media_attachments=media_attachments,
                            conversation_id=conversation_id,
                            model=effective_model,
                            tool_profile=tool_profile,
                            mode=mode,
                            cwd=cwd,
                        )
                    except TypeError:
                        if tool_profile != "chat" or mode is not None or cwd is not None:
                            raise
                        return await self._sdk_client.generate_response(
                            prompt=prompt,
                            system_instructions=system_instructions,
                            media_attachments=media_attachments,
                            conversation_id=conversation_id,
                            model=effective_model,
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

                cli_kwargs: dict[str, object] = {
                    "prompt": augmented_prompt,
                    "system_instructions": system_instructions,
                    "media_attachments": media_attachments,
                    "conversation_id": conversation_id,
                    "model": effective_model,
                    "effort": effective_effort,
                    "tool_profile": tool_profile,
                    "mode": mode,
                    "cwd": cwd,
                }
                if passed_allowed is not None:
                    cli_kwargs["allowed_tools"] = passed_allowed

                try:
                    return await self._cli_client.generate_response(**cli_kwargs)  # type: ignore[arg-type]
                except TypeError as exc:
                    if (
                        tool_profile != "chat"
                        or mode is not None
                        or cwd is not None
                        or passed_allowed is not None
                    ):
                        raise
                    err_s = str(exc)
                    if "media_attachments" in err_s:
                        return await self._cli_client.generate_response(
                            prompt=augmented_prompt,
                            system_instructions=system_instructions,
                            conversation_id=conversation_id,
                            model=effective_model,
                            effort=effective_effort,
                        )
                    if "tool_profile" in err_s or "unexpected keyword" in err_s:
                        return await self._cli_client.generate_response(
                            prompt=augmented_prompt,
                            system_instructions=system_instructions,
                            media_attachments=media_attachments,
                            conversation_id=conversation_id,
                            model=effective_model,
                            effort=effective_effort,
                        )
                    raise

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
            self._trip_cooldown_if_applicable(exc, effective_model)
            raise


__all__ = [
    "ALWAYS_FORBIDDEN_TOOLS",
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
    "CHAT_ALLOWED_TOOLS",
    "ErrorCategory",
    "ErrorVerdict",
    "GeneratedImageResult",
    "IMAGE_ALLOWED_SUBAGENT_TYPES",
    "ImageGenerationError",
    "ImagePolicyRefusalError",
    "ImageTimeoutError",
    "ImageValidationError",
    "MediaAttachment",
    "OPS_ALLOWED_TOOLS",
    "ProviderResponse",
    "SELF_DEV_ALLOWED_TOOLS",
    "StreamChunk",
    "ToolProfile",
    "UsageInfo",
    "build_clean_child_env",
    "calculate_backoff",
    "check_antigravity_auth",
    "check_antigravity_token_file_evidence",
    "check_os_keyring_auth",
    "cleanup_old_antigravity_sessions",
    "describe_antigravity_auth_source",
    "extract_retry_after",
    "find_existing_oauth_token",
    "has_gemini_api_key",
    "is_allowed_image_subagent_step",
    "is_sdk_available",
    "parse_ndjson_stream",
    "resolve_allowed_tools",
    "resolve_antigravity_binary",
    "resolve_antigravity_home",
    "retry_with_backoff",
    "run_isolated_antigravity_login",
    "setup_isolated_home",
    "sync_updated_isolated_tokens",
    "write_chat_pre_tool_hook",
    "write_image_pre_tool_hook",
]
