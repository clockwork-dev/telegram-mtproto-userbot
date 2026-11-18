"""Scheduled Sarcastic Real-News Digest Generator."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from bot.memory.db import ChatMessage

logger = logging.getLogger(__name__)

MAX_TELEGRAM_DIGEST_CHARS = 3800

_RUSSIAN_WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

_RUSSIAN_MONTHS = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

DEFAULT_DIGEST_TOPICS: Tuple[str, ...] = (
    "мировые новости",
    "IT и технологии",
    "абсурд и курьёзы",
    "культура и интернет",
)

DEFAULT_DIGEST_SYSTEM_INSTRUCTIONS = (
    "Ты — піструнькиня (Джефф), независимая и острая на язык участница дружеского Telegram-чата "
    "со своим характером, вкусами и настроением. Всегда пиши о себе в женском роде на живом русском "
    "(или с лёгким суржиком/украинским, когда к месту), без шаблонов AI-ассистента. "
    "Для новостной сводки обязательно используй `search_web` (и при необходимости `read_url_content`), "
    "чтобы находить реальные свежие события за сегодня и последние 24 часа. "
    "Никогда не выдумывай новости, не скатывайся во внутренние мемы чата и строго соблюдай запрещённые темы."
)


def _resolve_local_datetime(
    timezone: str,
    now_dt: Optional[datetime],
) -> Tuple[datetime, str]:
    """Convert or initialize current datetime in the target timezone."""
    tz_name = (timezone or "Europe/Kyiv").strip() or "Europe/Kyiv"
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid digest timezone '%s'; falling back to UTC", tz_name)
        tz_name = "UTC"
        tz = ZoneInfo("UTC")

    if now_dt is None:
        local_dt = datetime.now(tz)
    elif now_dt.tzinfo is None:
        local_dt = now_dt.replace(tzinfo=tz)
    else:
        local_dt = now_dt.astimezone(tz)
    return local_dt, tz_name


def _merge_forbidden_topics(
    forbidden_topics: Optional[Sequence[str]],
    sensitive_topics: Optional[Sequence[str]],
) -> List[str]:
    """Combine global forbidden_topics and friend sensitive_topics without duplicates."""
    seen_lower: set[str] = set()
    merged: List[str] = []
    for raw in list(forbidden_topics or []) + list(sensitive_topics or []):
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            merged.append(cleaned)
    return sorted(merged, key=str.lower)


def _filter_safe_topics(
    topics: Optional[Sequence[str]],
    exclusions: Sequence[str],
) -> List[str]:
    """Drop requested topics that match or contain any forbidden/sensitive keyword."""
    if not topics:
        return []
    excl_lower = [e.strip().lower() for e in exclusions if e.strip()]
    safe: List[str] = []
    for topic in topics:
        cleaned = topic.strip()
        if not cleaned:
            continue
        t_low = cleaned.lower()
        if any(excl in t_low or t_low in excl for excl in excl_lower):
            continue
        safe.append(cleaned)
    return safe


def _classify_slot_kind(slot_label: str) -> str:
    """Normalize slot label into 'morning', 'evening', or 'custom'."""
    low = (slot_label or "").strip().lower()
    if "morning" in low or "утр" in low:
        return "morning"
    if "evening" in low or "вечер" in low or "ноч" in low:
        return "evening"
    return "custom"


def _describe_time_of_day(hour: int) -> str:
    """Return Russian time-of-day label for local hour."""
    if 5 <= hour < 12:
        return "утро"
    if 12 <= hour < 17:
        return "день"
    if 17 <= hour < 23:
        return "вечер"
    return "ночь"


def _format_recent_context(
    recent_messages: Optional[Sequence[ChatMessage]],
    limit: int = 8,
) -> str:
    """Format recent chat messages as brief vibe context."""
    if not recent_messages:
        return "В чате пока тихо."
    lines: List[str] = []
    for msg in list(recent_messages)[-limit:]:
        text = (msg.text or "").strip()
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        if msg.media_summary:
            text = f"{text} [{msg.media_summary}]".strip()
        if text:
            lines.append(f"- {msg.author_name}: {text}")
    return "\n".join(lines) if lines else "В чате пока тихо."


def _truncate_telegram_text(text: str, max_chars: int = MAX_TELEGRAM_DIGEST_CHARS) -> str:
    """Ensure digest stays within Telegram message length bounds cleanly."""
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cutoff = max_chars - 3
    split_at = cleaned.rfind("\n", 0, cutoff)
    if split_at < cutoff // 2:
        split_at = cleaned.rfind(" ", 0, cutoff)
    if split_at <= 0:
        split_at = cutoff
    return cleaned[:split_at].rstrip() + "..."


class NewsDigestGenerator:
    """Generates sarcastic Russian news digests grounded in live search_web results."""

    def __init__(
        self,
        provider: Optional[Any] = None,
        default_system_instructions: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.default_system_instructions = (
            default_system_instructions
            if default_system_instructions is not None
            else DEFAULT_DIGEST_SYSTEM_INSTRUCTIONS
        )

    def build_digest_prompt(
        self,
        slot_name: str = "morning",
        slot_label: Optional[str] = None,
        timezone: str = "Europe/Kyiv",
        now_dt: Optional[datetime] = None,
        topics: Optional[Sequence[str]] = None,
        prompt_hint: str = "",
        recent_messages: Optional[Sequence[ChatMessage]] = None,
        forbidden_topics: Optional[Sequence[str]] = None,
        sensitive_topics: Optional[Sequence[str]] = None,
        humor_style: str = "edgy_sarcasm",
    ) -> str:
        """Construct structured news digest prompt with exact date, search_web rules, and taboos."""
        effective_label = (slot_label or slot_name or "morning").strip()
        slot_kind = _classify_slot_kind(effective_label)
        local_dt, resolved_tz = _resolve_local_datetime(timezone=timezone, now_dt=now_dt)

        iso_date = local_dt.strftime("%Y-%m-%d")
        time_str = local_dt.strftime("%H:%M")
        weekday_ru = _RUSSIAN_WEEKDAYS[local_dt.weekday()]
        month_ru = _RUSSIAN_MONTHS[local_dt.month]
        human_date = f"{local_dt.day} {month_ru} {local_dt.year} ({iso_date}), {weekday_ru}"
        day_part = _describe_time_of_day(local_dt.hour)

        all_exclusions = _merge_forbidden_topics(forbidden_topics, sensitive_topics)
        safe_topics = _filter_safe_topics(topics, all_exclusions)
        if not safe_topics:
            safe_topics = _filter_safe_topics(DEFAULT_DIGEST_TOPICS, all_exclusions)
        topics_str = ", ".join(safe_topics) if safe_topics else "нейтральные курьёзы и технологии"

        if slot_kind == "morning":
            slot_brief = (
                f"Слот: {effective_label} (утренний дайджест). Задай тон бодрого, язвительного "
                "утреннего брифинга к кофе: что мир уже успел натворить за ночь и к этому часу, "
                "и дай едкое утреннее напутствие в финале."
            )
        elif slot_kind == "evening":
            slot_brief = (
                f"Слот: {effective_label} (вечерний дайджест). Задай тон циничных итогов дня: "
                "что человечество успело сломать, выкинуть или провалить за сегодняшний день, "
                "и заверши едкой вечерней репликой перед отдыхом или посиделками."
            )
        else:
            slot_brief = (
                f"Слот: {effective_label} (тематическая сводка). Собери свежий срез главных и самых "
                "абсурдных событий за последние 24 часа и заверши язвительной концовкой под текущее время суток."
            )

        slot_id_note = ""
        if slot_name and slot_name.strip() and slot_name.strip() != effective_label:
            slot_id_note = f" (id: {slot_name.strip()})"

        hint_block = ""
        if prompt_hint and prompt_hint.strip():
            hint_block = f"\nДОПОЛНИТЕЛЬНЫЙ ФОКУС / ПОЖЕЛАНИЕ:\n{prompt_hint.strip()}\n"

        context_str = _format_recent_context(recent_messages)
        exclusions_str = (
            "\n".join(f"- {topic}" for topic in all_exclusions) if all_exclusions else "- Нет"
        )

        prompt = f"""Ты — піструнькиня (Джефф), независимая, острая на язык и немного циничная подруга в дружеском Telegram-чате.
У тебя свой взгляд на мир, своё настроение и свой вкус — ты приносишь живую подборку реальных новостей с едкими комментариями, а не сухую сводку и не заезженные внутренние шутки чата.

ТЕКУЩАЯ ДАТА И ВРЕМЯ:
- Дата: {human_date}
- Время: {time_str} ({resolved_tz}), время суток: {day_part}
- Режим выпуска{slot_id_note}: {slot_brief}

ОБЯЗАТЕЛЬНЫЙ ПОИСК РЕАЛЬНЫХ НОВОСТЕЙ (`search_web`):
1. ПЕРЕД тем как писать текст, ОБЯЗАТЕЛЬНО используй инструмент `search_web` (и при необходимости `read_url_content`), чтобы найти РЕАЛЬНЫЕ свежие новости за {iso_date} и последние 24 часа.
2. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать новости, придумывать события, даты, компании или цитаты из головы. Если по теме не нашлось свежих новостей — сделай другой запрос через `search_web` по разрешённым направлениям.
3. Приоритетные темы: {topics_str}.
{hint_block}
ВАЙБ ЧАТА (ПОСЛЕДНИЕ СООБЩЕНИЯ — только пассивный фон; НЕ пересказывай их и НЕ впихивай мемы/тропы чата в новости):
{context_str}

СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ (НИ В КОЕМ СЛУЧАЕ НЕ ИСКАТЬ, НЕ УПОМИНАТЬ И НЕ НАМЕКАТЬ):
{exclusions_str}
Ни при каких обстоятельствах не включай новости, имена, локации или шутки, связанные с этим списком.

ФОРМАТ И СТИЛЬ ВЫВОДА:
1. Пиши на живом, естественном русском языке от первого лица в женском роде («я глянула», «сама в шоке», «принесла вам»).
2. Стиль: {humor_style} — острый сарказм, живая ирония и собственный вкус без унылого официоза, морализаторства и зацикленности на старых приколах чата.
3. Структура:
   - Короткий цепляющий заголовок-заход с отсылкой к дате/времени суток.
   - 4–6 ёмких буллитов (например, с `•` или подходящим эмодзи в начале строки): в каждом — реальный свежий инфоповод за последние 24 часа + едкий ироничный комментарий.
   - Короткая язвительная финальная фраза / утренний или вечерний панч.
4. Длина: ориентир ~900–2800 символов, строго не длиннее {MAX_TELEGRAM_DIGEST_CHARS} символов (формат одного сообщения в Telegram).
5. Без AI-штампов («Привет! Вот дайджест...», «По результатам поиска...», «Надеюсь, было полезно!»), без тегов `[DOC:...]` и `[REACT:...]`. Выведи только готовый текст дайджеста."""
        return prompt.strip()

    def build_fallback_digest(
        self,
        slot_name: str = "morning",
        slot_label: Optional[str] = None,
        timezone: str = "Europe/Kyiv",
        now_dt: Optional[datetime] = None,
        topics: Optional[Sequence[str]] = None,
        forbidden_topics: Optional[Sequence[str]] = None,
        sensitive_topics: Optional[Sequence[str]] = None,
    ) -> str:
        """Build deterministic offline fallback digest when no LLM provider is configured."""
        effective_label = (slot_label or slot_name or "morning").strip()
        slot_kind = _classify_slot_kind(effective_label)
        local_dt, resolved_tz = _resolve_local_datetime(timezone=timezone, now_dt=now_dt)
        date_str = local_dt.strftime("%Y-%m-%d")
        time_str = local_dt.strftime("%H:%M")

        all_exclusions = _merge_forbidden_topics(forbidden_topics, sensitive_topics)
        safe_topics = _filter_safe_topics(topics, all_exclusions)
        topic_note = safe_topics[0] if safe_topics else "технологии и абсурд"

        if slot_kind == "morning":
            header = f"☕ Утренняя сводка на {date_str} ({time_str} {resolved_tz}):"
            closing = "Заваривайте кофе, мир сам себя не доломает."
        elif slot_kind == "evening":
            header = f"🌙 Вечерние итоги за {date_str} ({time_str} {resolved_tz}):"
            closing = "День пережит, можно выдыхать и расходиться по своим делам."
        else:
            header = f"🗞 Внезапная сводка на {date_str} ({time_str} {resolved_tz}):"
            closing = "Держу в курсе, чтобы вы не скучали в тишине."

        bullets = [
            f"• По теме «{topic_note}» внешняя лента пока молчит — я честно проверила, чудес не завезли.",
            "• Алгоритмы снова обещают всё упростить, а в итоге проще не стало никому.",
            "• В интернетах стабильность: половина людей спорит ни о чём, вторая просто смотрит.",
            "• Календарь уверенно идёт дальше, а здравый смысл опять взял отгул.",
        ]
        return _truncate_telegram_text("\n".join([header, "", *bullets, "", closing]))

    async def generate_digest(
        self,
        slot_name: str = "morning",
        slot_label: Optional[str] = None,
        timezone: str = "Europe/Kyiv",
        now_dt: Optional[datetime] = None,
        topics: Optional[Sequence[str]] = None,
        prompt_hint: str = "",
        recent_messages: Optional[Sequence[ChatMessage]] = None,
        forbidden_topics: Optional[Sequence[str]] = None,
        sensitive_topics: Optional[Sequence[str]] = None,
        humor_style: str = "edgy_sarcasm",
        system_instructions: Optional[str] = None,
        fallback_when_no_provider: bool = True,
        fallback_on_error: bool = False,
    ) -> Optional[str]:
        """Generate a real-news sarcastic digest via provider or explicit fallback."""
        effective_label = (slot_label or slot_name or "morning").strip()
        prompt = self.build_digest_prompt(
            slot_name=slot_name,
            slot_label=slot_label,
            timezone=timezone,
            now_dt=now_dt,
            topics=topics,
            prompt_hint=prompt_hint,
            recent_messages=recent_messages,
            forbidden_topics=forbidden_topics,
            sensitive_topics=sensitive_topics,
            humor_style=humor_style,
        )

        if self.provider is None:
            if fallback_when_no_provider:
                return self.build_fallback_digest(
                    slot_name=slot_name,
                    slot_label=slot_label,
                    timezone=timezone,
                    now_dt=now_dt,
                    topics=topics,
                    forbidden_topics=forbidden_topics,
                    sensitive_topics=sensitive_topics,
                )
            return None

        rate_limiter = getattr(self.provider, "is_rate_limited", None)
        if callable(rate_limiter) and rate_limiter() is True:
            logger.info("Skipping news digest '%s' while provider is rate-limited.", effective_label)
            if fallback_on_error:
                return self.build_fallback_digest(
                    slot_name=slot_name,
                    slot_label=slot_label,
                    timezone=timezone,
                    now_dt=now_dt,
                    topics=topics,
                    forbidden_topics=forbidden_topics,
                    sensitive_topics=sensitive_topics,
                )
            return None

        if not hasattr(self.provider, "generate_response"):
            logger.warning(
                "Configured provider %r lacks generate_response; skipping digest for slot '%s'",
                type(self.provider).__name__,
                effective_label,
            )
            if fallback_on_error:
                return self.build_fallback_digest(
                    slot_name=slot_name,
                    slot_label=slot_label,
                    timezone=timezone,
                    now_dt=now_dt,
                    topics=topics,
                    forbidden_topics=forbidden_topics,
                    sensitive_topics=sensitive_topics,
                )
            return None

        resolved_sys = (
            system_instructions
            if system_instructions is not None
            else self.default_system_instructions
        )

        try:
            resp = await self.provider.generate_response(
                prompt=prompt,
                system_instructions=resolved_sys,
                conversation_id=None,
            )
        except Exception as exc:
            logger.warning(
                "Provider error generating news digest for slot '%s': %s",
                effective_label,
                exc,
            )
            if fallback_on_error:
                return self.build_fallback_digest(
                    slot_name=slot_name,
                    slot_label=slot_label,
                    timezone=timezone,
                    now_dt=now_dt,
                    topics=topics,
                    forbidden_topics=forbidden_topics,
                    sensitive_topics=sensitive_topics,
                )
            return None

        if hasattr(resp, "text"):
            raw_text = resp.text or ""
        elif isinstance(resp, str):
            raw_text = resp
        else:
            raw_text = ""

        cleaned = raw_text.strip()
        if not cleaned:
            logger.warning(
                "Empty response received while generating news digest for slot '%s'",
                effective_label,
            )
            if fallback_on_error:
                return self.build_fallback_digest(
                    slot_name=slot_name,
                    slot_label=slot_label,
                    timezone=timezone,
                    now_dt=now_dt,
                    topics=topics,
                    forbidden_topics=forbidden_topics,
                    sensitive_topics=sensitive_topics,
                )
            return None

        return _truncate_telegram_text(cleaned)
