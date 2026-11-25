"""Proactive Engine with Quiet Hours, Silence Detection, Cooldowns & Cron News Digests."""

import asyncio
import logging
import random
import time as time_mod
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from bot.config.manager import ConfigManager
from bot.memory.db import MemoryStore
from bot.memory.fts import FactRecord
from bot.proactive.cron import CronExpression
from bot.proactive.digest import NewsDigestGenerator
from bot.proactive.humor import HumorGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CronDigestDelivery:
    """Generated scheduled cron digest ready for delivery to a target chat."""

    job_id: str
    chat_id: int
    slot_dt: datetime
    slot_label: str
    text: str


class ProactiveEngine:
    """Manages proactive conversation triggers, quiet hours gating, cooldowns, and cron digests."""

    _MAX_SENT_CRON_SLOTS = 512
    _MAX_TRACKED_CHATS = 256

    def __init__(
        self,
        config_mgr: ConfigManager,
        memory_store: MemoryStore,
        antigravity_provider: Optional[Any] = None,
        quota_manager: Optional[Any] = None,
    ) -> None:
        self.config_mgr = config_mgr
        self.memory_store = memory_store
        self.provider = antigravity_provider
        self._quota_manager = quota_manager
        self.humor_gen = HumorGenerator(provider=antigravity_provider, quota_manager=quota_manager)
        self.digest_gen = NewsDigestGenerator(
            provider=antigravity_provider,
            quota_manager=quota_manager,
        )
        self._loop_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._cron_cursor: Optional[datetime] = None
        self._sent_cron_slots: OrderedDict[Tuple[str, int, str], float] = OrderedDict()
        self._banter_jitter_offsets: Dict[Tuple[int, float], float] = {}
        self._recent_modes: Dict[int, deque[str]] = {}
        self._recent_fact_ids: Dict[int, deque[int]] = {}
        self._recent_friend_ids: Dict[int, deque[int]] = {}

    @property
    def quota_manager(self) -> Optional[Any]:
        return self._quota_manager

    @quota_manager.setter
    def quota_manager(self, value: Optional[Any]) -> None:
        self._quota_manager = value
        if getattr(self, "humor_gen", None) is not None:
            self.humor_gen.quota_manager = value
        if getattr(self, "digest_gen", None) is not None:
            self.digest_gen.quota_manager = value

    def _is_provider_rate_limited(self) -> bool:
        """Return True only when all candidate proactive models are rate-limited on the provider."""
        checker = getattr(self.provider, "is_rate_limited", None)
        if not callable(checker):
            return False

        candidates: List[str] = []
        qm = getattr(self, "quota_manager", None)
        if qm is not None and hasattr(qm, "select_model"):
            try:
                primary_model, _ = qm.select_model("proactive")
                if primary_model:
                    candidates.append(primary_model)
            except Exception:
                pass

        bot_cfg = self.config_mgr.bot if getattr(self, "config_mgr", None) is not None else None
        models_cfg = getattr(bot_cfg, "models", None) if bot_cfg is not None else None
        for fb in getattr(models_cfg, "fallback_models", []) or []:
            fb_model = getattr(fb, "model", None)
            if fb_model and fb_model not in candidates:
                candidates.append(fb_model)

        if not candidates:
            return checker() is True

        for candidate in candidates:
            try:
                limited = checker(model=candidate) is True
            except TypeError:
                return checker() is True
            if not limited:
                return False
        return True

    @classmethod
    def _cap_dict(cls, mapping: Dict[Any, Any]) -> None:
        while len(mapping) > cls._MAX_TRACKED_CHATS:
            oldest_key = next(iter(mapping))
            mapping.pop(oldest_key, None)

    def _get_scheduled_interval_seconds(
        self,
        chat_id: int,
        last_bot_msg_time: float,
        proactive_cfg: Any,
        cooldown_sec: float,
    ) -> float:
        """Compute per-cycle jittered scheduled_banter interval in seconds."""
        base_sec = float(proactive_cfg.scheduled_interval_minutes) * 60.0
        raw_jitter = getattr(proactive_cfg, "jitter_minutes", 0)
        jitter_minutes = (
            float(raw_jitter)
            if isinstance(raw_jitter, (int, float)) and not isinstance(raw_jitter, bool) and raw_jitter > 0
            else 0.0
        )
        if jitter_minutes <= 0.0:
            return max(cooldown_sec, base_sec)

        key = (int(chat_id), float(last_bot_msg_time))
        if key not in self._banter_jitter_offsets:
            for existing_key in [k for k in self._banter_jitter_offsets if k[0] == int(chat_id)]:
                self._banter_jitter_offsets.pop(existing_key, None)
            jitter_sec = jitter_minutes * 60.0
            self._banter_jitter_offsets[key] = random.uniform(-jitter_sec, jitter_sec)
            self._cap_dict(self._banter_jitter_offsets)
        return max(cooldown_sec, base_sec + self._banter_jitter_offsets[key])

    def is_in_quiet_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if given datetime (or now) is within configured quiet hours window."""
        if not self.config_mgr.bot:
            return False

        proactive_cfg = self.config_mgr.bot.proactive
        quiet_cfg = proactive_cfg.quiet_hours

        if not quiet_cfg.enabled:
            return False

        try:
            tz = ZoneInfo(quiet_cfg.timezone)
        except Exception:
            tz = ZoneInfo("UTC")

        if dt is None:
            now_dt = datetime.now(tz)
        elif dt.tzinfo is None:
            now_dt = dt.replace(tzinfo=tz)
        else:
            now_dt = dt.astimezone(tz)

        t = now_dt.time()

        try:
            sh, sm = map(int, quiet_cfg.start.split(":"))
            eh, em = map(int, quiet_cfg.end.split(":"))
            start_t = time(sh, sm)
            end_t = time(eh, em)
        except (ValueError, AttributeError):
            return False

        # Support midnight wrap-around (e.g. 23:00 to 08:00)
        if start_t > end_t:
            return t >= start_t or t < end_t
        return start_t <= t < end_t

    def check_triggers(
        self,
        chat_id: int,
        last_chat_msg_time: float,
        last_bot_msg_time: float,
        current_time: Optional[float] = None,
    ) -> Optional[str]:
        """Evaluate if chat qualifies for a proactive message trigger."""
        if not self.config_mgr.bot or not self.config_mgr.bot.proactive.enabled:
            return None

        now = current_time if current_time is not None else time_mod.time()
        dt_now = datetime.fromtimestamp(now, tz=ZoneInfo("UTC"))

        if self.is_in_quiet_hours(dt_now):
            return None

        proactive_cfg = self.config_mgr.bot.proactive
        cooldown_sec = proactive_cfg.cooldown_minutes * 60.0

        # Anti-spam cooldown check
        if (now - last_bot_msg_time) < cooldown_sec:
            return None

        # 1. Silence trigger check
        silence_threshold_sec = proactive_cfg.silence_trigger_hours * 3600.0
        if (now - last_chat_msg_time) >= silence_threshold_sec:
            return "silence_breaker"

        # 2. Scheduled banter check (only if human messages arrived after the bot's last message)
        if last_chat_msg_time > last_bot_msg_time:
            interval_sec = self._get_scheduled_interval_seconds(
                chat_id=chat_id,
                last_bot_msg_time=last_bot_msg_time,
                proactive_cfg=proactive_cfg,
                cooldown_sec=cooldown_sec,
            )
            if (now - last_bot_msg_time) >= interval_sec:
                return "scheduled_banter"

        return None

    def _select_proactive_mode(
        self,
        chat_id: int,
        trigger: str,
        has_recent_messages: bool,
    ) -> str:
        """Choose an autonomous proactive mode while preventing repetitive friend-roast cycles."""
        history = self._recent_modes.setdefault(chat_id, deque(maxlen=4))
        self._cap_dict(self._recent_modes)
        has_friends = bool(getattr(getattr(self.config_mgr, "friends", None), "friends", None))

        if self.provider is None and trigger == "silence_breaker" and has_friends:
            mode = "light_banter"
            history.append(mode)
            return mode

        recent_list = list(history)
        can_banter = has_friends and ("light_banter" not in recent_list[-2:])
        if can_banter and random.random() < 0.28:
            mode = "light_banter"
            history.append(mode)
            return mode

        pool = (
            ["independent_thought", "chat_callback", "open_question"]
            if has_recent_messages
            else ["independent_thought", "open_question"]
        )
        if recent_list:
            filtered = [m for m in pool if m != recent_list[-1]]
            if filtered:
                pool = filtered
        mode = random.choice(pool)
        history.append(mode)
        return mode

    async def evaluate_and_generate(
        self,
        chat_id: int,
        last_chat_msg_time: float,
        last_bot_msg_time: float,
        current_time: Optional[float] = None,
    ) -> Optional[str]:
        """Check triggers and generate appropriate proactive message if triggered."""
        trigger = self.check_triggers(
            chat_id, last_chat_msg_time, last_bot_msg_time, current_time
        )
        if not trigger:
            return None

        if self._is_provider_rate_limited():
            logger.info("Skipping proactive generation for chat %s while provider is rate-limited.", chat_id)
            return None

        recent = await self.memory_store.get_recent_history(chat_id, limit=10)
        mode = self._select_proactive_mode(
            chat_id=chat_id,
            trigger=trigger,
            has_recent_messages=bool(recent),
        )

        # Only inject friend profile & FTS facts during occasional light_banter mode
        target_friend = None
        facts: List[FactRecord] = []
        if mode == "light_banter" and self.config_mgr.friends.friends:
            friend_ids = list(self.config_mgr.friends.friends.keys())
            recent_friends = self._recent_friend_ids.setdefault(chat_id, deque(maxlen=3))
            self._cap_dict(self._recent_friend_ids)
            recent_user_ids = {m.user_id for m in recent}
            matching = [fid for fid in friend_ids if fid in recent_user_ids]
            candidates = [fid for fid in (matching or friend_ids) if fid not in recent_friends]
            if not candidates:
                candidates = matching or friend_ids
            chosen_id = random.choice(candidates)
            recent_friends.append(chosen_id)
            target_friend = self.config_mgr.get_friend_profile(chosen_id)

            query_terms = target_friend.name if target_friend else "мем"
            used_fact_ids = list(self._recent_fact_ids.setdefault(chat_id, deque(maxlen=10)))
            self._cap_dict(self._recent_fact_ids)
            if used_fact_ids:
                try:
                    facts = await self.memory_store.search_facts(
                        query_terms,
                        limit=2,
                        exclude_fact_ids=used_fact_ids,
                    )
                except TypeError:
                    facts = await self.memory_store.search_facts(query_terms, limit=2)
            else:
                facts = await self.memory_store.search_facts(query_terms, limit=2)
            for fact in facts:
                if getattr(fact, "id", None) is not None:
                    self._recent_fact_ids[chat_id].append(int(fact.id))

        proactive_cfg = self.config_mgr.bot.proactive
        forbidden_topics, sensitive_topics = self._collect_exclusion_topics()
        combined_forbidden = list(dict.fromkeys(list(forbidden_topics) + list(sensitive_topics)))
        return await self.humor_gen.generate_banter(
            recent_messages=recent,
            friend=target_friend,
            facts=facts,
            trigger_type=trigger,
            humor_style=proactive_cfg.humor_style,
            forbidden_topics=combined_forbidden,
            mode=mode,
        )

    @staticmethod
    def _resolve_utc_dt(
        now_dt: Optional[datetime] = None,
        current_time: Optional[float] = None,
    ) -> datetime:
        """Resolve an optional datetime or epoch timestamp into a UTC-aware datetime."""
        if now_dt is not None:
            if now_dt.tzinfo is None:
                return now_dt.replace(tzinfo=timezone.utc)
            return now_dt.astimezone(timezone.utc)
        if current_time is not None:
            return datetime.fromtimestamp(current_time, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def init_cron_cursor(
        self,
        now_dt: Optional[datetime] = None,
        current_time: Optional[float] = None,
    ) -> datetime:
        """Initialize the cron evaluation cursor if unset so startup skips stale past slots."""
        resolved = self._resolve_utc_dt(now_dt=now_dt, current_time=current_time)
        if self._cron_cursor is None:
            self._cron_cursor = resolved
        return self._cron_cursor

    def _collect_exclusion_topics(self) -> Tuple[List[str], List[str]]:
        """Gather global forbidden_topics and all friend sensitive_topics."""
        proactive_cfg = self.config_mgr.bot.proactive if self.config_mgr.bot else None
        forbidden_topics = list(proactive_cfg.forbidden_topics) if proactive_cfg else []

        sensitive_topics: List[str] = []
        friends_cfg = getattr(self.config_mgr, "friends", None)
        friends_map = getattr(friends_cfg, "friends", None) if friends_cfg else None
        if friends_map:
            for friend in friends_map.values():
                if friend and friend.sensitive_topics:
                    sensitive_topics.extend(friend.sensitive_topics)

        dedup_sensitive = list(dict.fromkeys(t for t in sensitive_topics if t and t.strip()))
        return forbidden_topics, dedup_sensitive

    def _record_sent_cron_slot(self, key: Tuple[str, int, str], timestamp: float) -> None:
        """Record a sent (job_id, chat_id, slot_iso) key in a bounded LRU map."""
        if key in self._sent_cron_slots:
            self._sent_cron_slots.move_to_end(key)
        self._sent_cron_slots[key] = timestamp
        while len(self._sent_cron_slots) > self._MAX_SENT_CRON_SLOTS:
            self._sent_cron_slots.popitem(last=False)

    def check_due_cron_jobs(
        self,
        chat_ids: Optional[Sequence[int]] = None,
        now_dt: Optional[datetime] = None,
        since_dt: Optional[datetime] = None,
        current_time: Optional[float] = None,
    ) -> List[Tuple[Any, int, datetime]]:
        """Return due unsent `(job, chat_id, slot_dt)` tuples within `(since_dt, now_dt]`."""
        now_utc = self._resolve_utc_dt(now_dt=now_dt, current_time=current_time)
        if not self.config_mgr.bot or not self.config_mgr.bot.proactive.enabled:
            return []

        proactive_cfg = self.config_mgr.bot.proactive
        jobs = getattr(proactive_cfg, "cron_digests", None) or []
        if not jobs:
            return []

        if since_dt is not None:
            since_utc = self._resolve_utc_dt(now_dt=since_dt)
        elif self._cron_cursor is not None:
            since_utc = self._cron_cursor
        else:
            self._cron_cursor = now_utc
            return []

        if now_utc <= since_utc:
            return []

        fallback_chats = list(dict.fromkeys(int(cid) for cid in (chat_ids or [])))
        due_items: List[Tuple[Any, int, datetime]] = []

        for job in jobs:
            if not getattr(job, "enabled", True):
                continue

            try:
                cron_expr = CronExpression.parse(job.cron, timezone=job.timezone)
            except ValueError as exc:
                logger.warning("Invalid cron expression for job '%s': %s", job.id, exc)
                continue

            slots = cron_expr.due_slots(since_utc, now_utc)
            if not slots:
                continue

            job_chat_ids = getattr(job, "chat_ids", None) or []
            target_chats = (
                list(dict.fromkeys(int(cid) for cid in job_chat_ids))
                if job_chat_ids
                else fallback_chats
            )
            if not target_chats:
                continue

            respect_qh = bool(getattr(job, "respect_quiet_hours", True))

            for cid in target_chats:
                eligible_slots: List[datetime] = []
                for slot_dt in slots:
                    if respect_qh and self.is_in_quiet_hours(slot_dt):
                        continue
                    key = (job.id, cid, slot_dt.isoformat())
                    if key in self._sent_cron_slots:
                        continue
                    eligible_slots.append(slot_dt)

                if eligible_slots:
                    # Pick latest unsent due slot per job+chat to avoid bursting missed slots.
                    due_items.append((job, cid, eligible_slots[-1]))

        return due_items

    async def evaluate_cron_digests(
        self,
        chat_ids: Optional[Sequence[int]] = None,
        now_dt: Optional[datetime] = None,
        since_dt: Optional[datetime] = None,
        current_time: Optional[float] = None,
        fallback_when_no_provider: bool = False,
    ) -> List[CronDigestDelivery]:
        """Evaluate due cron digest jobs, generate digest messages, and deduplicate sent slots."""
        now_utc = self._resolve_utc_dt(now_dt=now_dt, current_time=current_time)

        if not self.config_mgr.bot or not self.config_mgr.bot.proactive.enabled:
            if since_dt is None:
                self._cron_cursor = (
                    now_utc if self._cron_cursor is None else max(self._cron_cursor, now_utc)
                )
            return []

        if since_dt is None and self._cron_cursor is None:
            self._cron_cursor = now_utc
            return []

        if self._is_provider_rate_limited():
            logger.info("Skipping due cron digest evaluation while provider is rate-limited.")
            if since_dt is None:
                self._cron_cursor = (
                    now_utc if self._cron_cursor is None else max(self._cron_cursor, now_utc)
                )
            return []

        due_items = self.check_due_cron_jobs(
            chat_ids=chat_ids,
            now_dt=now_utc,
            since_dt=since_dt,
        )

        if since_dt is None:
            self._cron_cursor = (
                now_utc if self._cron_cursor is None else max(self._cron_cursor, now_utc)
            )

        if not due_items:
            return []

        proactive_cfg = self.config_mgr.bot.proactive
        forbidden_topics, sensitive_topics = self._collect_exclusion_topics()

        recent_cache: Dict[int, List[Any]] = {}
        deliveries: List[CronDigestDelivery] = []

        for job, cid, slot_dt in due_items:
            if cid not in recent_cache:
                recent_cache[cid] = await self.memory_store.get_recent_history(cid, limit=10)
            recent_messages = recent_cache[cid]

            text = await self.digest_gen.generate_digest(
                slot_name=job.id,
                slot_label=job.slot_label,
                timezone=job.timezone,
                now_dt=slot_dt,
                topics=job.topics or None,
                prompt_hint=job.prompt_hint,
                recent_messages=recent_messages,
                forbidden_topics=forbidden_topics,
                sensitive_topics=sensitive_topics,
                humor_style=proactive_cfg.humor_style,
                fallback_when_no_provider=fallback_when_no_provider,
            )
            if not text:
                continue

            self._record_sent_cron_slot(
                (job.id, cid, slot_dt.isoformat()),
                now_utc.timestamp(),
            )
            deliveries.append(
                CronDigestDelivery(
                    job_id=job.id,
                    chat_id=cid,
                    slot_dt=slot_dt,
                    slot_label=job.slot_label,
                    text=text,
                )
            )

        return deliveries

    async def run_proactive_tick(
        self,
        get_chat_states: Optional[Callable[[], Dict[int, Dict[str, float]]]] = None,
        send_message_fn: Optional[Callable[[int, str], Any]] = None,
        get_target_chat_ids: Optional[Callable[[], Sequence[int]]] = None,
        now_dt: Optional[datetime] = None,
        current_time: Optional[float] = None,
        fallback_when_no_provider: bool = False,
        *,
        chat_states: Optional[Dict[int, Dict[str, float]]] = None,
        send_callback: Optional[Callable[[int, str], Any]] = None,
        target_chat_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Run one proactive evaluation tick: due cron digests first, then banter."""
        if self.quota_manager is not None and hasattr(self.quota_manager, "should_allow_proactive"):
            if not self.quota_manager.should_allow_proactive():
                logger.debug("Proactive tick skipped: economy mode active")
                return
        now_utc = self._resolve_utc_dt(now_dt=now_dt, current_time=current_time)
        now_ts = now_utc.timestamp()

        resolved_states = (
            get_chat_states()
            if get_chat_states is not None
            else (chat_states if chat_states is not None else {})
        )
        sender = send_message_fn or send_callback
        if sender is None:
            raise ValueError("Either send_message_fn or send_callback must be provided")
        chat_states = resolved_states
        send_message_fn = sender
        extra_chat_ids = (
            list(get_target_chat_ids())
            if get_target_chat_ids is not None
            else (list(target_chat_ids) if target_chat_ids is not None else [])
        )
        resolved_target_ids = list(
            dict.fromkeys(
                [int(cid) for cid in extra_chat_ids if int(cid) < 0]
                + [int(cid) for cid in chat_states.keys() if int(cid) < 0]
            )
        )

        cron_sent_chats: set[int] = set()
        deliveries = await self.evaluate_cron_digests(
            chat_ids=resolved_target_ids,
            now_dt=now_utc,
            fallback_when_no_provider=fallback_when_no_provider,
        )
        for delivery in deliveries:
            try:
                logger.info(
                    "Proactive cron digest '%s' generated for chat %s (slot %s)",
                    delivery.job_id,
                    delivery.chat_id,
                    delivery.slot_dt.isoformat(),
                )
                res = send_message_fn(delivery.chat_id, delivery.text)
                if asyncio.iscoroutine(res):
                    await res
                state = chat_states.setdefault(
                    delivery.chat_id,
                    {"last_chat_msg_time": now_ts, "last_bot_msg_time": 0.0},
                )
                state["last_bot_msg_time"] = now_ts
                cron_sent_chats.add(delivery.chat_id)
            except Exception as exc:
                logger.error(
                    "Failed sending cron digest '%s' to chat %s: %s",
                    delivery.job_id,
                    delivery.chat_id,
                    exc,
                )

        for chat_id, state in list(chat_states.items()):
            if int(chat_id) >= 0:
                continue
            if chat_id in cron_sent_chats:
                continue
            last_chat = state.get("last_chat_msg_time", 0.0)
            last_bot = state.get("last_bot_msg_time", 0.0)

            msg = await self.evaluate_and_generate(
                chat_id,
                last_chat,
                last_bot,
                current_time=now_ts,
            )
            if msg:
                logger.info("Proactive message generated for chat %s: %s", chat_id, msg)
                res = send_message_fn(chat_id, msg)
                if asyncio.iscoroutine(res):
                    await res
                state["last_bot_msg_time"] = now_ts

    async def proactive_loop(
        self,
        get_chat_states: Callable[[], Dict[int, Dict[str, float]]],
        send_message_fn: Callable[[int, str], Any],
        poll_interval: float = 60.0,
        get_target_chat_ids: Optional[Callable[[], Sequence[int]]] = None,
    ) -> None:
        """Background loop evaluating proactive triggers with periodic jitter."""
        self._running = True
        self.init_cron_cursor()
        logger.info("Starting proactive evaluation loop")
        while self._running:
            try:
                # Add jitter (e.g. +/- 10% of poll_interval)
                jitter = random.uniform(-poll_interval * 0.1, poll_interval * 0.1)
                await asyncio.sleep(max(5.0, poll_interval + jitter))

                await self.run_proactive_tick(
                    get_chat_states=get_chat_states,
                    send_message_fn=send_message_fn,
                    get_target_chat_ids=get_target_chat_ids,
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in proactive loop: %s", e)

    def start_loop(
        self,
        get_chat_states: Callable[[], Dict[int, Dict[str, float]]],
        send_message_fn: Callable[[int, str], Any],
        poll_interval: float = 60.0,
        get_target_chat_ids: Optional[Callable[[], Sequence[int]]] = None,
    ) -> asyncio.Task:
        """Start proactive loop as asyncio Task."""
        if self._loop_task is not None and not self._loop_task.done():
            return self._loop_task
        self._loop_task = asyncio.create_task(
            self.proactive_loop(
                get_chat_states,
                send_message_fn,
                poll_interval,
                get_target_chat_ids=get_target_chat_ids,
            )
        )
        return self._loop_task

    def stop_loop(self) -> None:
        """Stop background proactive loop cleanly."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            self._loop_task = None
