"""Error classification, exceptions, and backoff policies for Antigravity providers."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_AFTER_RE = re.compile(
    r"(?:retry[_\s-]*after|retry\s+in|reset[s]?\s+in)\s*[:=]?\s*"
    r"(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes)?\b",
    re.IGNORECASE,
)


def extract_retry_after(error: Exception | str) -> float | None:
    """Extract a retry-after duration in seconds from an exception or message, if present."""
    if isinstance(error, BaseException):
        attr_val = getattr(error, "retry_after", None)
        if isinstance(attr_val, (int, float)) and attr_val > 0:
            return float(attr_val)
        parts: list[str] = []
        curr: BaseException | None = error
        seen: set[int] = set()
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            parts.append(str(curr))
            curr = curr.__cause__ or curr.__context__
        text = " | ".join(p for p in parts if p)
    else:
        text = str(error)

    match = _RETRY_AFTER_RE.search(text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    unit = (match.group(2) or "s").lower()
    if unit.startswith("m"):
        value *= 60.0
    return value


class ErrorCategory(str, Enum):
    """Normalized categories of Antigravity errors."""

    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow"
    SERVER_ERROR = "server_error"
    BILLING = "billing"
    AUTH_PERMANENT = "auth_permanent"
    MODEL_NOT_FOUND = "model_not_found"
    NATIVE_TOOL_ABORT = "native_tool_abort"
    UNKNOWN = "unknown"


class ErrorVerdict(BaseModel):
    """Structured recovery guidance for classified errors."""

    category: ErrorCategory
    retryable: bool
    should_compress: bool
    should_fallback: bool
    message: str


class AntigravityError(Exception):
    """Base exception for Antigravity provider operations."""

    def __init__(
        self,
        message: str,
        verdict: ErrorVerdict | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.verdict = verdict or classify_error(message)
        self.retry_after = retry_after if retry_after is not None else extract_retry_after(message)


class AntigravityRateLimitError(AntigravityError):
    """429 / RESOURCE_EXHAUSTED capacity error."""


class AntigravityContextOverflowError(AntigravityError):
    """Context window or trajectory token limit exceeded."""


class AntigravityAuthError(AntigravityError):
    """Authentication or OAuth expiration error."""


class AntigravityBillingError(AntigravityError):
    """Insufficient AI credits or billing quota exhaustion."""


class AntigravityNativeToolAbort(AntigravityError):
    """Raised when agy attempts unauthorized host tool execution."""


class AntigravityServerError(AntigravityError):
    """Upstream server or output token truncation error."""


def classify_error(error: Exception | str, status_code: int | None = None) -> ErrorVerdict:
    """Classify Antigravity error string or exception into actionable recovery verdict."""
    if isinstance(error, BaseException):
        parts: list[str] = []
        curr: BaseException | None = error
        seen: set[int] = set()
        while curr is not None and id(curr) not in seen:
            seen.add(id(curr))
            parts.append(str(curr))
            curr = curr.__cause__ or curr.__context__
        full_error_str = " | ".join(p for p in parts if p) or str(error)
    else:
        full_error_str = str(error)
    msg = full_error_str.lower()

    if status_code == 429:
        return ErrorVerdict(
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 1. Output token truncation / server error
    if any(
        p in msg
        for p in (
            "exceeded the output token limit",
            "previous response was cut off",
            "output token limit",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.SERVER_ERROR,
            retryable=True,
            should_compress=False,
            should_fallback=False,
            message=str(error),
        )

    # 2. Context window or trajectory token overflow
    if any(
        p in msg
        for p in (
            "max_trajectory_tokens",
            "trajectory token",
            "subscriber fell behind",
            "prompt is too long",
            "context length exceeded",
            "input exceeds the maximum",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.CONTEXT_OVERFLOW,
            retryable=True,
            should_compress=True,
            should_fallback=False,
            message=str(error),
        )

    # 3. AI credits / billing quota
    if any(
        p in msg
        for p in (
            "ai credits balance is too low",
            "credits balance",
            "insufficient credits",
            "quota has been exhausted",
            "upgrade your plan",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.BILLING,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 4. OAuth and account verification
    if any(
        p in msg
        for p in (
            "verify your account",
            "terms of service",
            "oauth token expired",
            "not authenticated",
            "no google antigravity oauth credentials",
            "unauthenticated",
            "invalid_grant",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.AUTH_PERMANENT,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 5. Model not found
    if any(
        p in msg
        for p in (
            "model not found",
            "unknown model",
            "unsupported model",
            "model_not_found",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.MODEL_NOT_FOUND,
            retryable=False,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 6. Rate limits & quota
    if any(
        p in msg
        for p in (
            "resource_exhausted",
            "exhausted your capacity",
            "429 too many requests",
            "ratelimitexceeded",
            "rate limit",
        )
    ):
        return ErrorVerdict(
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
            should_compress=False,
            should_fallback=True,
            message=str(error),
        )

    # 7. Native tool interception
    if "native tool" in msg or "tool execution aborted" in msg:
        return ErrorVerdict(
            category=ErrorCategory.NATIVE_TOOL_ABORT,
            retryable=False,
            should_compress=False,
            should_fallback=False,
            message=str(error),
        )

    return ErrorVerdict(
        category=ErrorCategory.UNKNOWN,
        retryable=False,
        should_compress=False,
        should_fallback=True,
        message=str(error),
    )


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    calculated = min(max_delay, base_delay * (2 ** attempt))
    if jitter:
        jitter_val = random.uniform(0.1, 1.0)
        return min(max_delay, calculated + jitter_val)
    return calculated


async def retry_with_backoff(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    rate_limit_max_attempts: int = 1,
    retryable_categories: tuple[ErrorCategory, ...] = (
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER_ERROR,
    ),
    on_retry: Callable[[Exception, int, float], Any] | None = None,
) -> T:
    """Execute async operation with exponential backoff on retryable errors."""
    for attempt in range(max_attempts):
        try:
            return await operation()
        except Exception as exc:
            verdict = classify_error(exc)
            effective_max = (
                min(max_attempts, max(1, rate_limit_max_attempts))
                if verdict.category == ErrorCategory.RATE_LIMIT
                else max_attempts
            )
            if (
                attempt >= effective_max - 1
                or not verdict.retryable
                or verdict.category not in retryable_categories
            ):
                raise
            retry_after = extract_retry_after(exc) if verdict.category == ErrorCategory.RATE_LIMIT else None
            delay = (
                min(max_delay, retry_after)
                if retry_after is not None and retry_after > 0
                else calculate_backoff(attempt, base_delay=base_delay, max_delay=max_delay)
            )
            logger.warning(
                "Antigravity retryable error (attempt %d/%d): %s. Backing off for %.2fs",
                attempt + 1,
                effective_max,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(exc, attempt + 1, delay)
            await asyncio.sleep(delay)

    raise RuntimeError("Unreachable retry loop exit")
