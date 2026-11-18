"""Unit tests for Scheduled Sarcastic Real-News Digest Generator."""

import logging
from datetime import datetime
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo

import pytest

from bot.core.models import ProviderResponse
from bot.memory.db import ChatMessage
from bot.proactive import NewsDigestGenerator
from bot.proactive.digest import MAX_TELEGRAM_DIGEST_CHARS


class MockAsyncProvider:
    """Async mock provider matching AntigravityProvider.generate_response signature."""

    def __init__(self, response=None, exc: Exception | None = None) -> None:
        self.response = response
        self.exc = exc
        self.calls: list[dict] = []

    async def generate_response(
        self,
        prompt: str,
        system_instructions: str | None = None,
        media_attachments=None,
        conversation_id: str | None = None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "system_instructions": system_instructions,
                "media_attachments": media_attachments,
                "conversation_id": conversation_id,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.response


def test_build_digest_prompt_date_injection_and_timezone_conversion():
    """Verify exact local date/time injection in Europe/Kyiv from UTC or naive datetime."""
    gen = NewsDigestGenerator()
    # 2026-11-13 22:30 UTC == 2026-11-14 00:30 Europe/Kyiv (UTC+2 in November)
    utc_dt = datetime(2026, 11, 13, 22, 30, tzinfo=dt_timezone.utc)

    prompt = gen.build_digest_prompt(
        slot_name="morning_digest",
        slot_label="morning",
        timezone="Europe/Kyiv",
        now_dt=utc_dt,
    )

    assert "2026-11-14" in prompt
    assert "00:30 (Europe/Kyiv)" in prompt
    assert "14 ноября 2026" in prompt
    assert "search_web" in prompt
    assert "піструнькиня" in prompt
    assert "4–6" in prompt
    assert str(MAX_TELEGRAM_DIGEST_CHARS) in prompt


def test_build_digest_prompt_morning_vs_evening_vs_custom_tone():
    """Verify distinct framing for morning, evening, and custom digest slots."""
    gen = NewsDigestGenerator()
    kyiv_tz = ZoneInfo("Europe/Kyiv")

    morning_prompt = gen.build_digest_prompt(
        slot_name="morning_digest",
        slot_label="morning",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 9, 0, tzinfo=kyiv_tz),
    )
    evening_prompt = gen.build_digest_prompt(
        slot_name="evening_digest",
        slot_label="evening",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 20, 0, tzinfo=kyiv_tz),
    )
    custom_prompt = gen.build_digest_prompt(
        slot_name="tech_flash",
        slot_label="tech_special",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 15, 0, tzinfo=kyiv_tz),
    )

    assert "утренний дайджест" in morning_prompt
    assert "утреннего брифинга к кофе" in morning_prompt
    assert "вечерний дайджест" in evening_prompt
    assert "итогов дня" in evening_prompt
    assert "тематическая сводка" in custom_prompt
    assert "(id: tech_flash)" in custom_prompt


def test_build_digest_prompt_topics_hint_recent_and_forbidden_enforcement():
    """Verify topics, prompt_hint, chat context, and combined forbidden/sensitive exclusions."""
    gen = NewsDigestGenerator()
    kyiv_tz = ZoneInfo("Europe/Kyiv")

    messages = [
        ChatMessage(
            telegram_msg_id=1,
            chat_id=-100,
            user_id=10,
            timestamp=1700000000,
            author_name="Аня",
            text="Опять релиз упал в пятницу",
            media_summary="Стикер 🗿",
        )
    ]

    prompt = gen.build_digest_prompt(
        slot_label="evening",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 20, 0, tzinfo=kyiv_tz),
        topics=["IT и AI", "кринж стартапов", "личные финансы", "тяжелые болезни"],
        prompt_hint="Сделай акцент на провалах корпораций и странных техно-трендах.",
        recent_messages=messages,
        forbidden_topics=["политика", "личные финансы"],
        sensitive_topics=["тяжелые болезни", "Политика"],  # case-insensitive duplicate
        humor_style="edgy_sarcasm",
    )

    assert "IT и AI" in prompt
    assert "кринж стартапов" in prompt
    assert "Сделай акцент на провалах корпораций" in prompt
    assert "Аня: Опять релиз упал в пятницу [Стикер 🗿]" in prompt
    assert "СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ" in prompt

    # Banned topics must appear in the strict exclusion section, NOT in priority topics
    priority_line = next(
        line for line in prompt.splitlines() if line.startswith("3. Приоритетные темы:")
    )
    assert "личные финансы" not in priority_line
    assert "тяжелые болезни" not in priority_line

    exclusion_section = prompt.split("СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ", 1)[1]
    assert "- личные финансы" in exclusion_section
    assert "- политика" in exclusion_section
    assert "- тяжелые болезни" in exclusion_section


def test_build_digest_prompt_invalid_timezone_falls_back_to_utc():
    """Verify invalid timezone string safely falls back to UTC without crashing."""
    gen = NewsDigestGenerator()
    prompt = gen.build_digest_prompt(
        slot_label="morning",
        timezone="Mars/Olympus_Mons",
        now_dt=datetime(2026, 11, 14, 9, 0),
    )
    assert "09:00 (UTC)" in prompt


@pytest.mark.asyncio
async def test_generate_digest_invokes_provider_statelessly_and_strips_text():
    """Verify generate_digest passes search_web prompt, system_instructions, conversation_id=None."""
    mock_resp = ProviderResponse(
        text="  ☕ Утренняя сводка:\n• Новость 1\n• Новость 2\n\nПошли пить кофе.  ",
        model="gemini-4-hydrogen",
    )
    provider = MockAsyncProvider(response=mock_resp)
    gen = NewsDigestGenerator(provider=provider)

    result = await gen.generate_digest(
        slot_name="morning_digest",
        slot_label="morning",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 9, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
        topics=["технологии", "курьёзы"],
        forbidden_topics=["личные финансы"],
    )

    assert result == "☕ Утренняя сводка:\n• Новость 1\n• Новость 2\n\nПошли пить кофе."
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["conversation_id"] is None
    assert "search_web" in call["prompt"]
    assert "2026-11-14" in call["prompt"]
    assert "search_web" in (call["system_instructions"] or "")
    assert "личные финансы" in call["prompt"]


@pytest.mark.asyncio
async def test_generate_digest_supports_string_response_and_truncates_oversized_text():
    """Verify plain str provider responses and Telegram max-length truncation."""
    oversized = "Начало сводки.\n" + ("Очень длинный текст новости. " * 200)
    provider = MockAsyncProvider(response=oversized)
    gen = NewsDigestGenerator(provider=provider)

    result = await gen.generate_digest(
        slot_label="evening",
        system_instructions="Custom system prompt",
    )

    assert result is not None
    assert len(result) <= MAX_TELEGRAM_DIGEST_CHARS
    assert result.endswith("...")
    assert provider.calls[0]["system_instructions"] == "Custom system prompt"


@pytest.mark.asyncio
async def test_generate_digest_runtime_error_logs_warning_and_returns_none(caplog):
    """Verify transient provider error logs warning and returns None by default."""
    provider = MockAsyncProvider(exc=RuntimeError("429 quota exhausted"))
    gen = NewsDigestGenerator(provider=provider)

    with caplog.at_level(logging.WARNING):
        result = await gen.generate_digest(slot_label="morning")

    assert result is None
    assert any("Provider error generating news digest" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_generate_digest_empty_response_returns_none(caplog):
    """Verify whitespace-only provider output logs warning and returns None."""
    provider = MockAsyncProvider(response=ProviderResponse(text="   \n  "))
    gen = NewsDigestGenerator(provider=provider)

    with caplog.at_level(logging.WARNING):
        result = await gen.generate_digest(slot_label="evening")

    assert result is None
    assert any("Empty response received" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_generate_digest_fallback_when_provider_is_none():
    """Verify deterministic sarcastic fallback when provider=None, and opt-out option."""
    gen = NewsDigestGenerator(provider=None)
    kyiv_dt = datetime(2026, 11, 14, 20, 0, tzinfo=ZoneInfo("Europe/Kyiv"))

    evening_fallback = await gen.generate_digest(
        slot_label="evening",
        timezone="Europe/Kyiv",
        now_dt=kyiv_dt,
        topics=["личные финансы", "гаджеты"],
        forbidden_topics=["личные финансы"],
    )
    assert evening_fallback is not None
    assert "2026-11-14" in evening_fallback
    assert "20:00 Europe/Kyiv" in evening_fallback
    assert "гаджеты" in evening_fallback
    assert "личные финансы" not in evening_fallback

    disabled_fallback = await gen.generate_digest(
        slot_label="morning",
        fallback_when_no_provider=False,
    )
    assert disabled_fallback is None


@pytest.mark.asyncio
async def test_generate_digest_optional_fallback_on_error():
    """Verify explicit fallback_on_error=True returns fallback on provider failure."""
    provider = MockAsyncProvider(exc=TimeoutError("upstream timeout"))
    gen = NewsDigestGenerator(provider=provider)

    result = await gen.generate_digest(
        slot_label="morning",
        timezone="Europe/Kyiv",
        now_dt=datetime(2026, 11, 14, 9, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
        fallback_on_error=True,
    )
    assert result is not None
    assert "Утренняя сводка" in result
    assert "2026-11-14" in result


@pytest.mark.asyncio
async def test_generate_digest_skips_when_provider_rate_limited():
    """Verify NewsDigestGenerator respects provider.is_rate_limited() circuit breaker."""
    provider = MockAsyncProvider(response=ProviderResponse(text="Should skip"))
    provider.is_rate_limited = lambda: True  # type: ignore[attr-defined]
    gen = NewsDigestGenerator(provider=provider)

    prompt = gen.build_digest_prompt(slot_label="morning")
    assert "НЕ впихивай мемы" in prompt

    result = await gen.generate_digest(slot_label="morning")
    assert result is None
    assert provider.calls == []
