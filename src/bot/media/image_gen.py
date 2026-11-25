"""Image generation helpers: command/tag parsing, rate limiting, and image validation."""

from __future__ import annotations

import io
import json
import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from PIL import Image

try:
    from bot.config.models import ImageGenerationConfig
except ImportError:  # pragma: no cover - transitional fallback before config reload
    from pydantic import BaseModel, Field

    class ImageGenerationConfig(BaseModel):  # type: ignore[no-redef]
        enabled: bool = True
        allow_natural_language_trigger: bool = True
        timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
        max_prompt_chars: int = Field(default=1000, ge=32, le=4000)
        max_per_user_per_hour: int = Field(default=5, ge=1, le=200)
        user_cooldown_seconds: float = Field(default=30.0, ge=0.0, le=3600.0)
        max_per_chat_per_hour: int = Field(default=15, ge=1, le=500)
        admin_bypass_limits: bool = True
        output_dir: str = "data/tmp/images"
        max_image_size_mb: float = Field(default=20.0, gt=0.0, le=45.0)
        send_as_document_threshold_mb: float = Field(default=9.5, gt=0.0, le=10.0)
        model: Optional[str] = None
        cleanup_after_send: bool = True

logger = logging.getLogger(__name__)

ALLOWED_ASPECT_RATIOS: frozenset[str] = frozenset(
    {"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"}
)

_IMAGINE_CMD_RE = re.compile(
    r"^/(?:imagine|img)(?:@([A-Za-z0-9_]+))?(?:\s+([\s\S]*)|$)",
    re.IGNORECASE,
)
_ASPECT_FLAG_RE = re.compile(
    r"(?:^|(?<=\s))--(?:ar|ratio)(?:=|\s+)(\S+)",
    re.IGNORECASE,
)
_IMG_TAG_PREFIX_RE = re.compile(r"\[(?:IMG|IMAGE)\s*:", re.IGNORECASE)

# Gates [IMG:{...}] prompt instructions in regular chat turns (RU / UK / EN)
IMAGE_INTENT_RE = re.compile(
    r"(?:"
    r"нарису[йюе]|нарисовать|порису[йю]|перерису[йю]|дорису[йю]"
    r"|намалю[йює]|намалювати|перемалю[йю]|домалю[йю]"
    r"|(?:с?генерир\w*|созда\w*|сдела\w*|покаж\w*)\s+(?:мне\s+|нам\s+)?"
    r"(?:картинк|изображен|арт\b|фото|рисун|иллюстрац|мем\b|логотип|аватар|постер|обложк|обои)"
    r"|(?:згенер\w*|створ\w*|зроб\w*)\s+(?:мені\s+|нам\s+)?"
    r"(?:картинк|зображен|арт\b|фото|малюн|ілюстрац|мем\b|логотип|аватар|постер|обкладин)"
    r"|\b(?:draw|sketch|paint|illustrate)\b"
    r"|\b(?:generate|create|make|render)\s+(?:me\s+|us\s+)?(?:an?\s+)?"
    r"(?:image|picture|pic|photo|art|artwork|illustration|meme|logo|avatar|poster|wallpaper)\b"
    r")",
    re.IGNORECASE,
)

_TELEGRAM_PHOTO_MAX_DIMENSION_SUM = 10_000
_TELEGRAM_PHOTO_MAX_ASPECT_RATIO = 20.0


def normalize_aspect_ratio(raw: Optional[str]) -> Optional[str]:
    """Validate and normalize an aspect ratio string against ALLOWED_ASPECT_RATIOS."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned in ALLOWED_ASPECT_RATIOS:
        return cleaned
    return None


@dataclass(slots=True)
class ImageRequestSpec:
    """Structured image generation directive parsed from an LLM [IMG:{...}] tag."""

    prompt: str
    caption: str = ""
    aspect_ratio: Optional[str] = None


@dataclass(slots=True)
class ImageCommandParseResult:
    """Parsed result of an explicit /imagine or /img Telegram command."""

    is_command: bool
    prompt: str = ""
    aspect_ratio: Optional[str] = None
    addressed_to_other_bot: bool = False


def parse_imagine_command(
    text: Optional[str],
    bot_username: Optional[str] = None,
) -> ImageCommandParseResult:
    """Parse `/imagine` or `/img` (with optional `@bot_username` and `--ar`/`--ratio` flags)."""
    if not text or not isinstance(text, str):
        return ImageCommandParseResult(is_command=False)

    stripped = text.strip()
    match = _IMAGINE_CMD_RE.match(stripped)
    if match is None:
        return ImageCommandParseResult(is_command=False)

    target_bot = match.group(1)
    if target_bot:
        expected = (bot_username or "").strip().lstrip("@").lower()
        if not expected or target_bot.strip().lower() != expected:
            return ImageCommandParseResult(
                is_command=False,
                addressed_to_other_bot=True,
            )

    raw_body = (match.group(2) or "").strip()
    if not raw_body:
        return ImageCommandParseResult(is_command=True, prompt="", aspect_ratio=None)

    aspect_ratio: Optional[str] = None

    def _capture_ratio(flag_match: re.Match[str]) -> str:
        nonlocal aspect_ratio
        if aspect_ratio is None:
            candidate = normalize_aspect_ratio(flag_match.group(1))
            if candidate is not None:
                aspect_ratio = candidate
        return ""

    cleaned_body = _ASPECT_FLAG_RE.sub(_capture_ratio, raw_body)
    cleaned_body = re.sub(r"[ \t]{2,}", " ", cleaned_body)
    cleaned_body = re.sub(r" *\n *", "\n", cleaned_body).strip()

    return ImageCommandParseResult(
        is_command=True,
        prompt=cleaned_body,
        aspect_ratio=aspect_ratio,
        addressed_to_other_bot=False,
    )


def _find_balanced_json_end(text: str, start_idx: int) -> Optional[int]:
    """Return index of matching `}` for `{` at `start_idx`, respecting JSON strings."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def _find_closing_bracket(text: str, start_idx: int) -> Optional[int]:
    """Find the next `]` after `start_idx` outside double-quoted strings."""
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "]":
            return idx
    return None


def extract_image_specs(
    text: str,
    *,
    max_specs: int = 1,
) -> Tuple[str, List[ImageRequestSpec]]:
    """Extract `[IMG:{...}]` tags from LLM output and strip ALL tags (including malformed ones).

    Returns `(clean_text, specs)` where `len(specs) <= max(0, max_specs)`.
    Raw `[IMG:...]` tags are always removed so they never leak to Telegram users.
    """
    if not text:
        return "", []

    cap = max(0, int(max_specs))
    specs: List[ImageRequestSpec] = []
    spans_to_strip: List[Tuple[int, int]] = []
    cursor = 0
    text_len = len(text)

    while cursor < text_len:
        match = _IMG_TAG_PREFIX_RE.search(text, cursor)
        if match is None:
            break

        pos = match.end()
        while pos < text_len and text[pos].isspace():
            pos += 1

        if pos < text_len and text[pos] == "{":
            json_end = _find_balanced_json_end(text, pos)
            if json_end is not None:
                close_pos = json_end + 1
                while close_pos < text_len and text[close_pos].isspace():
                    close_pos += 1

                if close_pos < text_len and text[close_pos] == "]":
                    span_end = close_pos + 1
                    json_payload = text[pos : json_end + 1]
                    try:
                        parsed = json.loads(json_payload)
                        if isinstance(parsed, dict):
                            raw_prompt = parsed.get("prompt") or parsed.get("description") or ""
                            prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""
                            if prompt and len(specs) < cap:
                                raw_caption = parsed.get("caption")
                                caption = (
                                    raw_caption.strip() if isinstance(raw_caption, str) else ""
                                )
                                raw_ar = (
                                    parsed.get("aspect_ratio")
                                    or parsed.get("ar")
                                    or parsed.get("ratio")
                                )
                                specs.append(
                                    ImageRequestSpec(
                                        prompt=prompt,
                                        caption=caption,
                                        aspect_ratio=normalize_aspect_ratio(
                                            raw_ar if isinstance(raw_ar, str) else None
                                        ),
                                    )
                                )
                            elif not prompt:
                                logger.warning(
                                    "Stripping [IMG:...] tag with empty prompt at offset %d",
                                    match.start(),
                                )
                        else:
                            logger.warning(
                                "Stripping non-object [IMG:...] tag at offset %d",
                                match.start(),
                            )
                    except Exception as exc:
                        logger.warning(
                            "Stripping malformed JSON in [IMG:...] tag at offset %d: %s",
                            match.start(),
                            exc,
                        )
                    spans_to_strip.append((match.start(), span_end))
                    cursor = span_end
                    continue

                # Balanced `{...}` without closing `]`: strip up to `}` to prevent tag leak
                logger.warning(
                    "Stripping unclosed [IMG:{...} tag missing ']' at offset %d",
                    match.start(),
                )
                spans_to_strip.append((match.start(), json_end + 1))
                cursor = json_end + 1
                continue

            # Unclosed `{` inside [IMG:
            bracket_pos = _find_closing_bracket(text, pos + 1)
            span_end = (bracket_pos + 1) if bracket_pos is not None else text_len
            logger.warning(
                "Stripping unclosed JSON in [IMG:...] tag at offset %d",
                match.start(),
            )
            spans_to_strip.append((match.start(), span_end))
            cursor = span_end
            continue

        # Non-JSON payload after `[IMG:`
        bracket_pos = _find_closing_bracket(text, pos)
        span_end = (bracket_pos + 1) if bracket_pos is not None else text_len
        logger.warning("Stripping invalid non-JSON [IMG:...] tag at offset %d", match.start())
        spans_to_strip.append((match.start(), span_end))
        cursor = span_end

    if not spans_to_strip:
        return text.strip(), []

    parts: List[str] = []
    last_idx = 0
    for start, end in spans_to_strip:
        parts.append(text[last_idx:start])
        last_idx = end
    parts.append(text[last_idx:])

    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip(), specs


class ImageRateLimitError(RuntimeError):
    """Raised when an image generation request exceeds cooldown or hourly quotas."""

    def __init__(
        self,
        *,
        reason: str,
        retry_after_seconds: float,
        user_message: str,
    ) -> None:
        super().__init__(user_message)
        self.reason = reason
        self.retry_after_seconds = max(0.0, float(retry_after_seconds))
        self.user_message = user_message


class ImageGenerationLimiter:
    """Sliding-window per-user cooldown and per-user/per-chat hourly limiter for image gen."""

    _WINDOW_SECONDS = 3600.0

    def __init__(self, config: Optional[ImageGenerationConfig] = None) -> None:
        self._config: ImageGenerationConfig = config or ImageGenerationConfig()
        self._lock = threading.Lock()
        self._user_timestamps: Dict[int, Deque[float]] = {}
        self._chat_timestamps: Dict[int, Deque[float]] = {}

    def update_config(self, config: ImageGenerationConfig) -> None:
        """Hot-reload limiter configuration without discarding active usage windows."""
        with self._lock:
            self._config = config

    _MAX_TRACKED_KEYS = 256

    @staticmethod
    def _prune(bucket: Deque[float], cutoff: float) -> None:
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    @classmethod
    def _sweep_buckets(cls, buckets: Dict[int, Deque[float]], cutoff: float) -> None:
        """Evict expired/empty buckets to prevent unbounded growth."""
        for key, q in list(buckets.items()):
            cls._prune(q, cutoff)
            if not q:
                buckets.pop(key, None)
        while len(buckets) > cls._MAX_TRACKED_KEYS:
            oldest_key = min(buckets, key=lambda k: buckets[k][-1] if buckets[k] else 0.0)
            buckets.pop(oldest_key, None)

    def check_and_reserve(
        self,
        user_id: int,
        chat_id: int,
        *,
        is_admin: bool = False,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """Validate limits and reserve one image slot.

        Returns a reservation timestamp token (for `refund()` on internal failure),
        or `None` if limits were bypassed for an admin.
        """
        with self._lock:
            cfg = self._config
            if is_admin and cfg.admin_bypass_limits:
                return None

            ts_now = float(now) if now is not None else time.monotonic()
            cutoff = ts_now - self._WINDOW_SECONDS

            if len(self._user_timestamps) > self._MAX_TRACKED_KEYS:
                self._sweep_buckets(self._user_timestamps, cutoff)
            if len(self._chat_timestamps) > self._MAX_TRACKED_KEYS:
                self._sweep_buckets(self._chat_timestamps, cutoff)

            uid = int(user_id)
            cid = int(chat_id)
            user_q = self._user_timestamps.get(uid)
            if user_q is not None:
                self._prune(user_q, cutoff)
                if not user_q:
                    self._user_timestamps.pop(uid, None)
                    user_q = None
            chat_q = self._chat_timestamps.get(cid)
            if chat_q is not None:
                self._prune(chat_q, cutoff)
                if not chat_q:
                    self._chat_timestamps.pop(cid, None)
                    chat_q = None

            # 1. Per-user cooldown
            cooldown = float(cfg.user_cooldown_seconds)
            if cooldown > 0.0 and user_q:
                elapsed = ts_now - user_q[-1]
                if elapsed < cooldown:
                    retry_after = max(0.0, cooldown - elapsed)
                    wait_sec = max(1, math.ceil(retry_after))
                    raise ImageRateLimitError(
                        reason="cooldown",
                        retry_after_seconds=retry_after,
                        user_message=(
                            f"Подожди немного (~{wait_sec} сек.) перед следующей картинкой."
                        ),
                    )

            # 2. Per-user hourly cap
            max_user = int(cfg.max_per_user_per_hour)
            if user_q is not None and len(user_q) >= max_user:
                retry_after = max(0.0, (user_q[0] + self._WINDOW_SECONDS) - ts_now)
                wait_min = max(1, math.ceil(retry_after / 60.0))
                raise ImageRateLimitError(
                    reason="user_hourly",
                    retry_after_seconds=retry_after,
                    user_message=(
                        f"Достигнут твой лимит картинок ({max_user}/час). "
                        f"Попробуй через ~{wait_min} мин."
                    ),
                )

            # 3. Per-chat hourly cap
            max_chat = int(cfg.max_per_chat_per_hour)
            if chat_q is not None and len(chat_q) >= max_chat:
                retry_after = max(0.0, (chat_q[0] + self._WINDOW_SECONDS) - ts_now)
                wait_min = max(1, math.ceil(retry_after / 60.0))
                raise ImageRateLimitError(
                    reason="chat_hourly",
                    retry_after_seconds=retry_after,
                    user_message=(
                        f"В этом чате достигнут лимит картинок ({max_chat}/час). "
                        f"Попробуй через ~{wait_min} мин."
                    ),
                )

            if user_q is None:
                user_q = deque()
                self._user_timestamps[uid] = user_q
            if chat_q is None:
                chat_q = deque()
                self._chat_timestamps[cid] = chat_q
            user_q.append(ts_now)
            chat_q.append(ts_now)
            return ts_now

    @staticmethod
    def _remove_latest_token(bucket: Optional[Deque[float]], token: float) -> None:
        if not bucket:
            return
        items = list(bucket)
        for idx in range(len(items) - 1, -1, -1):
            if items[idx] == token:
                del items[idx]
                bucket.clear()
                bucket.extend(items)
                return

    def refund(
        self,
        user_id: int,
        chat_id: int,
        token: Optional[float],
    ) -> None:
        """Refund a previously reserved slot (e.g., on upstream timeout or provider error)."""
        if token is None:
            return
        with self._lock:
            uid = int(user_id)
            cid = int(chat_id)
            user_q = self._user_timestamps.get(uid)
            self._remove_latest_token(user_q, token)
            if user_q is not None and not user_q:
                self._user_timestamps.pop(uid, None)
            chat_q = self._chat_timestamps.get(cid)
            self._remove_latest_token(chat_q, token)
            if chat_q is not None and not chat_q:
                self._chat_timestamps.pop(cid, None)

    def release(
        self,
        user_id: int,
        chat_id: int,
        *,
        token: Optional[float] = None,
    ) -> None:
        """Alias for `refund()` supporting keyword `token`."""
        self.refund(user_id, chat_id, token)


@dataclass(slots=True)
class ValidatedImageInfo:
    """Validated binary image metadata."""

    mime_type: str
    extension: str
    width: int
    height: int
    size_bytes: int


def _detect_magic_image_format(data: bytes) -> Optional[Tuple[str, str, str]]:
    """Return `(mime_type, extension, pil_format)` from magic bytes, or `None` if unsupported."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ("image/png", ".png", "PNG")
    if data.startswith(b"\xff\xd8\xff"):
        return ("image/jpeg", ".jpg", "JPEG")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ("image/webp", ".webp", "WEBP")
    return None


def validate_image_bytes(
    data: bytes,
    *,
    max_size_mb: float = 20.0,
) -> ValidatedImageInfo:
    """Validate raw image bytes (size, magic header, Pillow structural verify, dimensions)."""
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise ValueError("Image data is empty.")

    raw = bytes(data)
    size_bytes = len(raw)
    max_bytes = int(float(max_size_mb) * 1024 * 1024)
    if max_bytes <= 0 or size_bytes > max_bytes:
        raise ValueError(
            f"Image size ({size_bytes} bytes) exceeds maximum allowed ({max_bytes} bytes)."
        )

    detected = _detect_magic_image_format(raw)
    if detected is None:
        raise ValueError("Unsupported image format; expected JPEG, PNG, or WEBP.")
    mime_type, extension, expected_format = detected

    try:
        with Image.open(io.BytesIO(raw)) as img:
            pil_format = (img.format or "").upper()
            width, height = img.size
            img.verify()
    except Exception as exc:
        raise ValueError(f"Invalid or corrupted image data: {exc}") from exc

    if pil_format != expected_format:
        raise ValueError(
            f"Image header format mismatch: magic={expected_format}, decoded={pil_format or 'unknown'}."
        )
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}.")

    return ValidatedImageInfo(
        mime_type=mime_type,
        extension=extension,
        width=int(width),
        height=int(height),
        size_bytes=size_bytes,
    )


def should_send_as_document(
    size_bytes: int,
    width: int,
    height: int,
    *,
    threshold_mb: float = 9.5,
) -> bool:
    """Return True if an image should be sent as a Telegram document instead of a photo.

    Telegram `sendPhoto` rejects files >10 MB, dimension sum >10000, or aspect ratio >20.
    """
    if size_bytes > float(threshold_mb) * 1024 * 1024:
        return True
    if (int(width) + int(height)) > _TELEGRAM_PHOTO_MAX_DIMENSION_SUM:
        return True
    if width > 0 and height > 0:
        ratio = max(width, height) / min(width, height)
        if ratio > _TELEGRAM_PHOTO_MAX_ASPECT_RATIO:
            return True
    return False
