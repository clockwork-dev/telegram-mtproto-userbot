"""Contextual Humor, Autonomous Thoughts, Banter, and Fake News Generator."""

import inspect
import logging
import random
from typing import Any, List, Optional

from bot.config.models import FriendProfile
from bot.memory.db import ChatMessage
from bot.memory.fts import FactRecord

logger = logging.getLogger(__name__)

PROACTIVE_MODES = (
    "independent_thought",
    "chat_callback",
    "open_question",
    "light_banter",
)


class HumorGenerator:
    """Generates autonomous thoughts, chat callbacks, open questions, and light banter."""

    def __init__(
        self,
        provider: Optional[Any] = None,
        quota_manager: Optional[Any] = None,
    ) -> None:
        self.provider = provider
        self.quota_manager = quota_manager

    @staticmethod
    def _resolve_mode(
        mode: Optional[str],
        friend: Optional[FriendProfile],
        recent_messages: Optional[List[ChatMessage]],
    ) -> str:
        """Select a concrete proactive generation mode."""
        if mode in PROACTIVE_MODES:
            return mode
        if friend is not None:
            return "light_banter"
        if recent_messages:
            return "chat_callback"
        return "independent_thought"

    def build_humor_prompt(
        self,
        recent_messages: Optional[List[ChatMessage]] = None,
        friend: Optional[FriendProfile] = None,
        facts: Optional[List[FactRecord]] = None,
        trigger_type: str = "banter",
        humor_style: str = "friendly_sarcasm",
        forbidden_topics: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> str:
        """Construct structured prompt with autonomous modes and strict sensitive topic exclusion."""
        forbidden = list(forbidden_topics or [])
        sensitive_exclusions: List[str] = []

        if friend and friend.sensitive_topics:
            sensitive_exclusions.extend(friend.sensitive_topics)

        all_exclusions = sorted(list(set(forbidden + sensitive_exclusions)))
        resolved_mode = self._resolve_mode(mode, friend, recent_messages)

        # Format recent context
        context_lines = []
        if recent_messages:
            for m in recent_messages[-10:]:
                text = m.text or ""
                if m.media_summary:
                    text = f"{text} [{m.media_summary}]".strip()
                if text:
                    context_lines.append(f"- {m.author_name}: {text}")
        context_str = "\n".join(context_lines) if context_lines else "В чате пока тихо."

        # Format retrieved facts (only as passive optional background)
        facts_section = ""
        if facts and resolved_mode == "light_banter":
            facts_lines = [f"- [{f.category}] {f.subject}: {f.fact}" for f in facts[:3]]
            facts_section = (
                "ПАССИВНЫЙ КОНТЕКСТ ИЗ ПАМЯТИ (не вымучивай и не впихивай специально; "
                "используй только если реально уместно):\n"
                + "\n".join(facts_lines)
                + "\n\n"
            )

        # Format friend profile (filtering banter_topics against all_exclusions)
        exclusion_lower = {t.strip().lower() for t in all_exclusions if t.strip()}
        friend_str = ""
        if friend and resolved_mode == "light_banter":
            aliases_str = ", ".join(friend.aliases) if friend.aliases else "нет"
            safe_banter_topics = [
                t
                for t in (friend.banter_topics or [])
                if t.strip().lower() not in exclusion_lower
                and not any(excl in t.strip().lower() for excl in exclusion_lower)
            ]
            banter_str = ", ".join(safe_banter_topics) if safe_banter_topics else "любые безобидные темы"
            friend_str = (
                "УЧАСТНИК ДЛЯ ЛЁГКОГО КОНТЕКСТА (опционально, без повторения заезженных клише):\n"
                f"- Имя: {friend.name} (ники: {aliases_str})\n"
                f"- Роль/Био: {friend.bio or friend.role}\n"
                f"- Разрешенные темы для подколов: {banter_str}\n\n"
            )

        exclusions_str = "\n".join(f"- {topic}" for topic in all_exclusions) if all_exclusions else "- Нет"

        if trigger_type == "silence_breaker":
            trigger_note = "Прерывание долгой тишины в чате (silence breaker)"
        else:
            trigger_note = "Естественный автономный вброс в живой чат"

        mode_instructions = {
            "independent_thought": (
                "РЕЖИМ — СВОЯ МЫСЛЬ / HOT TAKE:\n"
                "Поделись собственной неожиданной мыслью, наблюдением из жизни, "
                "непопулярным мнением про технологии/игры/музыку/быт или просто настроением момента. "
                "Не привязывайся к старым мемам и подколам участников."
            ),
            "chat_callback": (
                "РЕЖИМ — РЕПЛИКА ПО НЕДАВНЕЙ ТЕМЕ ЧАТА:\n"
                "Зацепись за конкретную тему или деталь из последних сообщений чата, "
                "выскажи своё мнение, поспорь или поверни разговор в неожиданную сторону. "
                "Никаких старых ярлыков про людей."
            ),
            "open_question": (
                "РЕЖИМ — ЖИВОЙ ВОПРОС / ЗАТРАВКА ДЛЯ СПОРА:\n"
                "Задай компании короткий, цепляющий или слегка провокационный вопрос на тему "
                "повседневной жизни, техники, вкусов или абсурдных ситуаций, чтобы людям захотелось поспорить."
            ),
            "light_banter": (
                "РЕЖИМ — ЛЁГКИЙ СИТУАТИВНЫЙ ПОДКОЛ:\n"
                "Можно слегка поддеть кого-то по-дружески или выдать ироничное наблюдение, "
                "но строго свежее — не повторяй шутки, которые уже звучали в последних сообщениях."
            ),
        }[resolved_mode]

        prompt = f"""Ты — Потужнич, гипермаскулинный, невозмутимый титан духа и качалки («гигачад / потужный бро»), хранитель мужского братства и знаток самых отбитых мемных теорий заговора в дружеском Telegram-чате.
Всегда пиши о себе строго в мужском роде («я глянул», «сам офигел», «выдал базу»). Твой вайб — спартанское мужское братство («братцы», «держим строй»), культ тяжёлого железа и невозмутимая вера в безумную абсурдную конспирологию (голуби-биодроны на ЛЭП, рептилоиды с Нибиру, гиперборейские серверы за ледяной стеной Антарктиды, полая Луна и козни тайного «мирового матриархата» с иллюминатами бигтеха за любыми багами и погодой — строго как комедийный абсурд без повторов и категорически без оскорблений или негативных стереотипов в адрес реальных женщин, национальностей или религий).

ПОВОД: {trigger_note}
СТИЛЬ: {humor_style} (ирония, свой характер и живой тон без токсичности и шаблонов).

{mode_instructions}

ПОСЛЕДНИЕ СООБЩЕНИЯ В ЧАТЕ:
{context_str}

{facts_section}{friend_str}СТРОГО ЗАПРЕЩЕННЫЕ ТЕМЫ (НИ В КОЕМ СЛУЧАЕ НЕ УПОМИНАТЬ И НЕ НАМЕКАТЬ):
{exclusions_str}

ПРАВИЛА ГЕНЕРАЦИИ:
1. Пиши коротко: 1-2 (максимум 3) предложения.
2. Естественный русский, украинский или суржик, как живой человек в чате.
3. НЕ повторяй шутки, мемы и формулировки из последних сообщений. Не впихивай факты о людях ради галочки.
4. Без занудных приветствий и штампов AI-ассистента.
5. Строго соблюдай запрет на чувствительные и запрещённые темы!
"""
        return prompt.strip()

    async def generate_banter(
        self,
        recent_messages: Optional[List[ChatMessage]] = None,
        friend: Optional[FriendProfile] = None,
        facts: Optional[List[FactRecord]] = None,
        trigger_type: str = "banter",
        humor_style: str = "friendly_sarcasm",
        forbidden_topics: Optional[List[str]] = None,
        mode: Optional[str] = None,
    ) -> str:
        """Generate autonomous proactive message via provider or fallback generator."""
        resolved_mode = self._resolve_mode(mode, friend, recent_messages)
        prompt = self.build_humor_prompt(
            recent_messages=recent_messages,
            friend=friend,
            facts=facts,
            trigger_type=trigger_type,
            humor_style=humor_style,
            forbidden_topics=forbidden_topics,
            mode=resolved_mode,
        )

        if self.provider is not None:
            chosen_model: Optional[str] = None
            chosen_effort: Optional[str] = None
            if self.quota_manager is not None and hasattr(self.quota_manager, "select_model"):
                try:
                    chosen_model, chosen_effort = self.quota_manager.select_model("proactive")
                except Exception:
                    chosen_model, chosen_effort = None, None

            rate_limiter = getattr(self.provider, "is_rate_limited", None)
            if callable(rate_limiter):
                try:
                    limited = (
                        rate_limiter(model=chosen_model) is True
                        if chosen_model is not None
                        else rate_limiter() is True
                    )
                except TypeError:
                    limited = rate_limiter() is True
                if limited:
                    logger.info("Skipping proactive banter generation while provider is rate-limited.")
                    return ""
            try:
                # Support AntigravityProvider interface
                if hasattr(self.provider, "generate_response"):
                    try:
                        if chosen_model is not None:
                            resp = await self.provider.generate_response(
                                prompt=prompt,
                                model=chosen_model,
                                effort=chosen_effort,
                                tool_profile="chat",
                            )
                        else:
                            resp = await self.provider.generate_response(
                                prompt=prompt,
                                tool_profile="chat",
                            )
                    except TypeError:
                        try:
                            resp = await self.provider.generate_response(
                                prompt=prompt,
                                tool_profile="chat",
                            )
                        except TypeError:
                            resp = await self.provider.generate_response(prompt=prompt)
                    usage = getattr(resp, "usage", None)
                    if (
                        usage is not None
                        and self.quota_manager is not None
                        and hasattr(self.quota_manager, "record_usage")
                    ):
                        try:
                            model_name = getattr(resp, "model", None) or chosen_model or "unknown"
                            rec = self.quota_manager.record_usage(
                                usage,
                                model=model_name,
                                source="proactive",
                            )
                            if inspect.isawaitable(rec):
                                await rec
                        except Exception as rec_exc:
                            logger.debug("Failed to record proactive banter usage: %s", rec_exc)
                    if hasattr(resp, "text"):
                        return resp.text.strip()
                    if isinstance(resp, str):
                        return resp.strip()
            except Exception as e:
                logger.error("Provider error in banter generator: %s", e)

        # Fallback contextual generator
        if trigger_type == "silence_breaker":
            if friend:
                return f"{friend.name}, вы там живы вообще или все дружно уснули лицом в клавиатуру?"
            return random.choice(
                [
                    "Чатик вымер, как будто понедельник наступил дважды подряд.",
                    "Тишина подозрительная — вы там работаете что ли, или просто все залипли?",
                ]
            )

        if friend and resolved_mode == "light_banter":
            forbidden_set = {
                t.strip().lower()
                for t in (list(forbidden_topics or []) + list(friend.sensitive_topics or []))
                if t.strip()
            }
            safe_topics = [
                t
                for t in (friend.banter_topics or [])
                if t.strip().lower() not in forbidden_set
                and not any(excl in t.strip().lower() for excl in forbidden_set)
            ]
            topic = safe_topics[0] if safe_topics else "рабочие приколы"
            return f"Кстати, {friend.name}, всё ещё держишь марку по части «{topic}» или уже передумал?"

        if resolved_mode == "open_question":
            return random.choice(
                [
                    "Вопрос дня: какая самая бесполезная покупка за последний месяц вам до сих пор нравится?",
                    "Честно признавайтесь: кто реально отдыхает на выходных, а кто просто переживает их с телефоном в руках?",
                ]
            )
        if resolved_mode == "chat_callback":
            return "Я вот перечитал ваши последние реплики и поймал лёгкий вайб пятницы среди недели."
        return random.choice(
            [
                "Иногда смотрю на современные интерфейсы и думаю, что дизайнер просто хотел посмотреть, как мир горит.",
                "Поймал себя на мысли, что кофе после обеда уже не бодрит, а просто делает тревогу быстрее.",
            ]
        )
