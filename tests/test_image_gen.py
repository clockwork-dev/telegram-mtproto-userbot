"""Unit tests for ImageGenerationConfig, image_gen helpers/limiter/validator, and Bot API send_photo."""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PIL import Image
from pydantic import ValidationError

from bot.media.image_gen import (
    ALLOWED_ASPECT_RATIOS,
    IMAGE_INTENT_RE,
    ImageGenerationConfig,
    ImageGenerationLimiter,
    ImageRateLimitError,
    extract_image_specs,
    normalize_aspect_ratio,
    parse_imagine_command,
    should_send_as_document,
    validate_image_bytes,
)
from bot.security.sanitizer import SecretSanitizer
from bot.telegram.bot_api_client import TelegramBotAPIClient


def _make_pil_image_bytes(fmt: str, size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    mode = "RGBA" if fmt == "PNG" else "RGB"
    color = (20, 120, 220, 255) if mode == "RGBA" else (20, 120, 220)
    Image.new(mode, size, color=color).save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. ImageGenerationConfig defaults & validation bounds
# ---------------------------------------------------------------------------


def test_image_generation_config_defaults_and_backward_compat():
    cfg = ImageGenerationConfig()
    assert cfg.enabled is True
    assert cfg.allow_natural_language_trigger is True
    assert cfg.timeout_seconds == 120.0
    assert cfg.max_prompt_chars == 1000
    assert cfg.max_per_user_per_hour == 5
    assert cfg.user_cooldown_seconds == 30.0
    assert cfg.max_per_chat_per_hour == 15
    assert cfg.admin_bypass_limits is True
    assert cfg.output_dir == "data/tmp/images"
    assert cfg.max_image_size_mb == 20.0
    assert cfg.send_as_document_threshold_mb == 9.5
    assert cfg.model is None
    assert cfg.cleanup_after_send is True

    from bot.config.models import BotConfig

    if "image_generation" in BotConfig.model_fields:
        legacy = BotConfig.model_validate({"telegram": {"api_id": 123, "api_hash": "abc"}})
        assert legacy.image_generation.enabled is True


def test_image_generation_config_validation_bounds():
    with pytest.raises(ValidationError):
        ImageGenerationConfig(timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(timeout_seconds=601.0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(max_prompt_chars=10)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(max_per_user_per_hour=0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(user_cooldown_seconds=-1.0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(max_per_chat_per_hour=0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(max_image_size_mb=50.0)
    with pytest.raises(ValidationError):
        ImageGenerationConfig(send_as_document_threshold_mb=10.5)


# ---------------------------------------------------------------------------
# 2. Aspect ratio, /imagine command parser, [IMG:{...}] extractor, intent RE
# ---------------------------------------------------------------------------


def test_normalize_aspect_ratio():
    for ratio in ALLOWED_ASPECT_RATIOS:
        assert normalize_aspect_ratio(f"  {ratio} ") == ratio
    assert normalize_aspect_ratio("5:4") is None
    assert normalize_aspect_ratio("") is None
    assert normalize_aspect_ratio(None) is None


def test_parse_imagine_command_variants():
    # Basic /imagine and case-insensitive /IMG
    r1 = parse_imagine_command("/imagine кот в космосе")
    assert r1.is_command is True
    assert r1.prompt == "кот в космосе"
    assert r1.aspect_ratio is None
    assert r1.addressed_to_other_bot is False

    r2 = parse_imagine_command("/IMG   sunset over Alps")
    assert r2.is_command is True
    assert r2.prompt == "sunset over Alps"

    # Bot username matching & mismatching
    r3 = parse_imagine_command("/imagine@My_Bot cyber cat", bot_username="my_bot")
    assert r3.is_command is True
    assert r3.prompt == "cyber cat"
    assert r3.addressed_to_other_bot is False

    r4 = parse_imagine_command("/imagine@other_bot cyber cat", bot_username="@my_bot")
    assert r4.is_command is False
    assert r4.addressed_to_other_bot is True

    # Aspect ratio flags (--ar, --ratio, = syntax, any position)
    r5 = parse_imagine_command("/imagine --ar 16:9 astronaut cat")
    assert r5.is_command is True
    assert r5.aspect_ratio == "16:9"
    assert r5.prompt == "astronaut cat"

    r6 = parse_imagine_command("/img portrait of wizard --ratio 9:16")
    assert r6.is_command is True
    assert r6.aspect_ratio == "9:16"
    assert r6.prompt == "portrait of wizard"

    r7 = parse_imagine_command("/imagine futuristic car --ar=21:9 in neon rain")
    assert r7.is_command is True
    assert r7.aspect_ratio == "21:9"
    assert r7.prompt == "futuristic car in neon rain"

    r8 = parse_imagine_command("/imagine --ar 99:1 dragon")
    assert r8.is_command is True
    assert r8.aspect_ratio is None
    assert r8.prompt == "dragon"

    # Empty prompt
    r9 = parse_imagine_command("/imagine")
    assert r9.is_command is True
    assert r9.prompt == ""

    # Non-image commands and regular text
    assert parse_imagine_command("/quota").is_command is False
    assert parse_imagine_command("нарисуй кота").is_command is False
    assert parse_imagine_command("").is_command is False


def test_extract_image_specs_valid_nested_and_capped():
    raw = (
        'Вот тебе неон! [IMG:{"prompt":"neon sign saying \\"hello {world}\\" [v2]",'
        '"caption":"Неоновая вывеска","aspect_ratio":"16:9"}]'
    )
    clean, specs = extract_image_specs(raw)
    assert clean == "Вот тебе неон!"
    assert len(specs) == 1
    assert specs[0].prompt == 'neon sign saying "hello {world}" [v2]'
    assert specs[0].caption == "Неоновая вывеска"
    assert specs[0].aspect_ratio == "16:9"

    # Multiple tags capped to 1, ALL stripped cleanly
    multi = (
        'Держи [IMG:{"prompt":"first cat"}] и ещё '
        '[IMG:{"prompt":"second dog","aspect_ratio":"4:3"}] готово'
    )
    clean_m, specs_m = extract_image_specs(multi, max_specs=1)
    assert "[IMG:" not in clean_m
    assert clean_m == "Держи и ещё готово"
    assert len(specs_m) == 1
    assert specs_m[0].prompt == "first cat"


def test_extract_image_specs_strips_all_malformed_tags():
    cases = [
        "Привет [IMG:{broken: json}] мир",
        'Привет [IMG:{"caption":"missing prompt"}] мир',
        'Привет [IMG:{"prompt":"   "}] мир',
        "Привет [IMG:not json at all] мир",
        'Привет [IMG:{"prompt":"unclosed json',
        'Привет [IMG:{"prompt":"missing bracket"} мир',
    ]
    for sample in cases:
        clean, specs = extract_image_specs(sample)
        assert specs == []
        assert "[IMG:" not in clean
        assert "Привет" in clean


def test_image_intent_regex():
    positives = [
        "Джефф, нарисуй кота в скафандре",
        "Намалюй мені дракона на даху",
        "Сгенерируй картинку киберпанк-города",
        "Згенеруй зображення космосу",
        "Сделай мем про понедельник",
        "Please draw a futuristic train",
        "Generate an image of a red fox in snow",
    ]
    for text in positives:
        assert IMAGE_INTENT_RE.search(text), f"Expected match for: {text!r}"

    negatives = [
        "Привет, как дела?",
        "Пошли вечером в кино",
        "Какая завтра погода в Киеве?",
    ]
    for text in negatives:
        assert not IMAGE_INTENT_RE.search(text), f"Unexpected match for: {text!r}"


# ---------------------------------------------------------------------------
# 3. ImageGenerationLimiter
# ---------------------------------------------------------------------------


def test_image_generation_limiter_cooldown_hourly_admin_refund_and_hot_reload():
    cfg = ImageGenerationConfig(
        user_cooldown_seconds=30.0,
        max_per_user_per_hour=2,
        max_per_chat_per_hour=3,
        admin_bypass_limits=True,
    )
    limiter = ImageGenerationLimiter(cfg)

    # First request passes
    tok1 = limiter.check_and_reserve(user_id=10, chat_id=-100, now=1000.0)
    assert tok1 == 1000.0

    # Immediate second request hits cooldown
    with pytest.raises(ImageRateLimitError) as exc_cd:
        limiter.check_and_reserve(user_id=10, chat_id=-100, now=1010.0)
    assert exc_cd.value.reason == "cooldown"
    assert exc_cd.value.retry_after_seconds == pytest.approx(20.0)
    assert "сек." in exc_cd.value.user_message

    # After cooldown passes, second request succeeds
    tok2 = limiter.check_and_reserve(user_id=10, chat_id=-100, now=1035.0)
    assert tok2 == 1035.0

    # Third request from same user within 1 hour hits user_hourly
    with pytest.raises(ImageRateLimitError) as exc_uh:
        limiter.check_and_reserve(user_id=10, chat_id=-100, now=1100.0)
    assert exc_uh.value.reason == "user_hourly"
    assert "2/час" in exc_uh.value.user_message

    # Refund tok2 -> user can reserve again!
    limiter.refund(user_id=10, chat_id=-100, token=tok2)
    tok2_retry = limiter.check_and_reserve(user_id=10, chat_id=-100, now=1100.0)
    assert tok2_retry == 1100.0

    # release() alias also works
    limiter.release(user_id=10, chat_id=-100, token=tok2_retry)
    tok2_again = limiter.check_and_reserve(user_id=10, chat_id=-100, now=1101.0)
    assert tok2_again == 1101.0

    # Another user takes 3rd chat slot, 4th slot in chat hits chat_hourly
    limiter.check_and_reserve(user_id=20, chat_id=-100, now=1110.0)
    with pytest.raises(ImageRateLimitError) as exc_ch:
        limiter.check_and_reserve(user_id=30, chat_id=-100, now=1120.0)
    assert exc_ch.value.reason == "chat_hourly"
    assert "3/час" in exc_ch.value.user_message

    # Admin bypass skips all limits and returns None
    assert limiter.check_and_reserve(user_id=999, chat_id=-100, is_admin=True, now=1121.0) is None
    assert limiter.check_and_reserve(user_id=999, chat_id=-100, is_admin=True, now=1122.0) is None

    # Hot-reload higher chat cap + disable admin bypass
    limiter.update_config(
        ImageGenerationConfig(
            user_cooldown_seconds=10.0,
            max_per_user_per_hour=5,
            max_per_chat_per_hour=10,
            admin_bypass_limits=False,
        )
    )
    assert limiter.check_and_reserve(user_id=30, chat_id=-100, now=1125.0) == 1125.0
    assert limiter.check_and_reserve(user_id=999, chat_id=-100, is_admin=True, now=1126.0) == 1126.0
    with pytest.raises(ImageRateLimitError) as exc_admin_cd:
        limiter.check_and_reserve(user_id=999, chat_id=-100, is_admin=True, now=1130.0)
    assert exc_admin_cd.value.reason == "cooldown"

    # After 1 hour window expires, old slots are pruned
    assert limiter.check_and_reserve(user_id=10, chat_id=-100, now=5000.0) == 5000.0


# ---------------------------------------------------------------------------
# 4. validate_image_bytes & should_send_as_document
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pil_fmt", "expected_mime", "expected_ext"),
    [
        ("PNG", "image/png", ".png"),
        ("JPEG", "image/jpeg", ".jpg"),
        ("WEBP", "image/webp", ".webp"),
    ],
)
def test_validate_image_bytes_valid_formats(pil_fmt: str, expected_mime: str, expected_ext: str):
    data = _make_pil_image_bytes(pil_fmt, size=(120, 80))
    info = validate_image_bytes(data)
    assert info.mime_type == expected_mime
    assert info.extension == expected_ext
    assert info.width == 120
    assert info.height == 80
    assert info.size_bytes == len(data)


def test_validate_image_bytes_rejects_empty_corrupt_and_oversized():
    with pytest.raises(ValueError, match="empty"):
        validate_image_bytes(b"")

    with pytest.raises(ValueError, match="Unsupported"):
        validate_image_bytes(b"<html>not an image</html>")

    # Fake PNG header with corrupt payload
    with pytest.raises(ValueError, match="corrupted|Invalid"):
        validate_image_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    # Oversized relative to tiny max_size_mb
    png_bytes = _make_pil_image_bytes("PNG", size=(200, 200))
    with pytest.raises(ValueError, match="exceeds"):
        validate_image_bytes(png_bytes, max_size_mb=0.00001)


def test_should_send_as_document_thresholds():
    # Normal 1024x1024 under 9.5 MB
    assert should_send_as_document(500_000, 1024, 1024) is False
    # Exact boundary sum=10000 and ratio=20.0 -> still photo
    assert should_send_as_document(1_000_000, 5000, 5000) is False
    assert should_send_as_document(1_000_000, 2000, 100) is False

    # Oversized (> 9.5 MB)
    assert should_send_as_document(int(9.6 * 1024 * 1024), 1024, 1024) is True
    # Sum of dimensions > 10000
    assert should_send_as_document(500_000, 6000, 4001) is True
    # Extreme aspect ratio > 20:1 (both wide and tall)
    assert should_send_as_document(500_000, 2100, 100) is True
    assert should_send_as_document(500_000, 100, 2100) is True


# ---------------------------------------------------------------------------
# 5. TelegramBotAPIClient.send_photo & send_document bytes fallback
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_bot_api_send_photo_bytes_caption_cap_redaction_and_reply_params(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=90.0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["data"] = req.data
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            json.dumps({"ok": True, "result": {"message_id": 4242}}).encode("utf-8")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    sanitizer = SecretSanitizer(extra_secrets=["SUPER_SECRET_API_KEY_999"], load_environ=False)
    bot_client = TelegramBotAPIClient(
        bot_token="123456:TEST_TOKEN_SECRET_XYZ1234567890",
        sanitizer=sanitizer,
    )

    png_bytes = _make_pil_image_bytes("PNG", size=(32, 32))
    long_caption = "Leak SUPER_SECRET_API_KEY_999 and 123456:TEST_TOKEN_SECRET_XYZ1234567890 " + (
        "А" * 1200
    )

    msg_id = await bot_client.send_photo(
        chat_id=-100999,
        photo=png_bytes,
        caption=long_caption,
        reply_to_message_id=777,
        parse_mode="HTML",
        disable_notification=True,
    )

    assert msg_id == 4242
    assert captured["url"].endswith("/sendPhoto")
    assert captured["method"] == "POST"
    assert captured["headers"]["content-type"].startswith("multipart/form-data; boundary=")

    body: bytes = captured["data"]
    assert b'name="chat_id"\r\n\r\n-100999\r\n' in body
    assert b'name="photo"; filename="generated.png"\r\n' in body
    assert b"Content-Type: image/png\r\n\r\n" in body
    assert png_bytes in body
    assert b'name="parse_mode"\r\n\r\nHTML\r\n' in body
    assert b'name="disable_notification"\r\n\r\ntrue\r\n' in body

    # Secrets must be redacted and caption truncated to <= 1024 chars
    assert b"SUPER_SECRET_API_KEY_999" not in body
    assert b"TEST_TOKEN_SECRET_XYZ1234567890" not in body
    assert b"[REDACTED]" in body

    caption_marker = b'name="caption"\r\n\r\n'
    cap_start = body.index(caption_marker) + len(caption_marker)
    cap_end = body.index(b"\r\n--", cap_start)
    sent_caption = body[cap_start:cap_end].decode("utf-8")
    assert len(sent_caption) == 1024
    assert sent_caption.endswith("...")

    # reply_parameters with allow_sending_without_reply=True
    rp_marker = b'name="reply_parameters"\r\n\r\n'
    rp_start = body.index(rp_marker) + len(rp_marker)
    rp_end = body.index(b"\r\n--", rp_start)
    reply_params = json.loads(body[rp_start:rp_end].decode("utf-8"))
    assert reply_params == {"message_id": 777, "allow_sending_without_reply": True}


@pytest.mark.asyncio
async def test_bot_api_send_photo_from_path_and_validation_errors(tmp_path: Path, monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=90.0):
        captured["data"] = req.data
        return _FakeHTTPResponse(
            json.dumps({"ok": True, "result": {"message_id": 5150}}).encode("utf-8")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    jpg_bytes = _make_pil_image_bytes("JPEG", size=(64, 64))
    jpg_file = tmp_path / "cat.jpg"
    jpg_file.write_bytes(jpg_bytes)

    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")
    msg_id = await bot_client.send_photo(chat_id=101, photo=jpg_file, caption="Котик")
    assert msg_id == 5150
    assert b'name="photo"; filename="cat.jpg"\r\n' in captured["data"]
    assert b"Content-Type: image/jpeg\r\n\r\n" in captured["data"]
    assert jpg_bytes in captured["data"]

    with pytest.raises(ValueError, match="empty"):
        await bot_client.send_photo(chat_id=101, photo=b"")

    with pytest.raises(FileNotFoundError):
        await bot_client.send_photo(chat_id=101, photo=tmp_path / "missing.jpg")


@pytest.mark.asyncio
async def test_bot_api_send_document_accepts_raw_bytes_fallback(monkeypatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=120.0):
        captured["url"] = req.full_url
        captured["data"] = req.data
        return _FakeHTTPResponse(
            json.dumps({"ok": True, "result": {"message_id": 6060}}).encode("utf-8")
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")
    bot_client.call_api = AsyncMock(return_value={"ok": True, "result": True})

    webp_bytes = _make_pil_image_bytes("WEBP", size=(64, 64))
    res = await bot_client.send_document(
        chat_id=-100555,
        file_path=webp_bytes,
        caption="Большая картинка",
        reply_to_message_id=88,
        simulate_typing=False,
        filename="wide_panorama.webp",
        mime_type="image/webp",
    )

    assert res["ok"] is True
    assert res["result"]["message_id"] == 6060
    assert captured["url"].endswith("/sendDocument")
    assert b'name="document"; filename="wide_panorama.webp"\r\n' in captured["data"]
    assert b"Content-Type: image/webp\r\n\r\n" in captured["data"]
    assert b'name="reply_to_message_id"\r\n\r\n88\r\n' in captured["data"]
    assert webp_bytes in captured["data"]

    with pytest.raises(ValueError, match="empty"):
        await bot_client.send_document(chat_id=-100555, file_path=b"", simulate_typing=False)
