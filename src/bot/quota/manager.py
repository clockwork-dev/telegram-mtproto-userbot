"""Quota & Budget Manager, ModelRouter, and live `agy -p /quota` integration."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from bot.antigravity.auth import (
    build_clean_child_env,
    resolve_antigravity_binary,
    resolve_antigravity_home,
)
from bot.config.models import BotConfig, ModelRoleConfig, ModelRouterConfig, QuotaConfig
from bot.core.models import UsageInfo

if TYPE_CHECKING:
    from bot.memory.db import MemoryStore

logger = logging.getLogger(__name__)

_EFFORT_SUFFIXES = ("-xhigh", "-high", "-medium", "-low", "-minimal")
_EXCLUDED_AUTO_PREFIXES = ("gemini-4-argon", "gemini-pro-orchestrator")


def _strip_effort_suffix(model: str) -> str:
    m = model.strip()
    for suf in _EFFORT_SUFFIXES:
        if m.endswith(suf):
            return m[: -len(suf)]
    return m


def _parse_iso_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    raw = ts.strip().replace(" ", "T")
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class _UsageSample:
    ts_epoch: float
    model: str
    source: str
    total_tokens: int


class QuotaBudgetManager:
    """Tracks token budgets in SQLite, per-model/bucket cooldowns, and role-based routing."""

    _DEFAULT_WARNING_TTL_SECONDS = 3600.0
    _MAX_QUOTA_WARNINGS = 64

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        config: Optional[BotConfig] = None,
        quota_config: Optional[QuotaConfig] = None,
        model_router: Optional[ModelRouterConfig] = None,
        data_dir: Optional[Path | str] = None,
    ) -> None:
        self.memory_store: Any = memory_store
        if config is not None:
            self.quota_config = quota_config or config.quota
            self.model_router = model_router or config.models
            self.data_dir = resolve_antigravity_home(data_dir or config.antigravity.data_dir)
        else:
            self.quota_config = quota_config or QuotaConfig()
            self.model_router = model_router or ModelRouterConfig()
            self.data_dir = resolve_antigravity_home(data_dir or "data/antigravity")

        self._model_cooldowns: Dict[str, Tuple[float, str]] = {}
        self._bucket_cooldowns: Dict[str, Tuple[float, str]] = {}
        self._warnings_lock = threading.RLock()
        self._quota_warnings: Dict[str, Dict[str, Any]] = {}
        self._live_quota_rows: List[Dict[str, Any]] = []
        self._usage_samples: Deque[_UsageSample] = deque()
        self._economy_notified: bool = False

        # Hydrate recent 24h usage synchronously when store is already available
        self._try_initial_sync()

    def _try_initial_sync(self) -> None:
        if self.memory_store is None:
            return
        try:
            since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            rows = self.memory_store._sync_get_usage_rows(since_iso=since)
            self._load_rows_into_window(rows)
        except Exception:
            logger.debug("Initial usage_history sync skipped", exc_info=True)

    def _load_rows_into_window(self, rows: List[Dict[str, Any]]) -> None:
        self._usage_samples.clear()
        cutoff = time.time() - 86400.0
        for r in rows:
            dt = _parse_iso_dt(str(r.get("timestamp") or ""))
            ts_epoch = dt.timestamp() if dt is not None else time.time()
            if ts_epoch < cutoff:
                continue
            self._usage_samples.append(
                _UsageSample(
                    ts_epoch=ts_epoch,
                    model=str(r.get("model") or "unknown"),
                    source=str(r.get("source") or "group"),
                    total_tokens=int(r.get("total_tokens") or 0),
                )
            )

    async def sync_from_db(self) -> None:
        """Reload the last 24h of usage_history from MemoryStore."""
        if self.memory_store is None:
            return
        since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = await self.memory_store.get_usage_rows(since_iso=since)
        self._load_rows_into_window(rows)

    async def refresh_from_store(self) -> None:
        """Alias for sync_from_db()."""
        await self.sync_from_db()

    def _prune_samples(self, now_epoch: Optional[float] = None) -> None:
        now = now_epoch if now_epoch is not None else time.time()
        cutoff = now - 86400.0
        while self._usage_samples and self._usage_samples[0].ts_epoch < cutoff:
            self._usage_samples.popleft()

    def get_usage_totals(self) -> Tuple[int, int]:
        """Return (hourly_tokens, daily_tokens) over the rolling 1h and 24h windows."""
        now = time.time()
        self._prune_samples(now)
        hour_cutoff = now - 3600.0
        hourly = 0
        daily = 0
        for s in self._usage_samples:
            daily += s.total_tokens
            if s.ts_epoch >= hour_cutoff:
                hourly += s.total_tokens
        return hourly, daily

    async def record_usage(
        self,
        usage: Optional[UsageInfo],
        model: str,
        source: str = "group",
    ) -> None:
        """Persist token usage to SQLite usage_history and update rolling budget windows."""
        if usage is None:
            return

        def _safe_int(val: Any) -> int:
            return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0

        in_tok = _safe_int(getattr(usage, "input_tokens", 0)) or _safe_int(getattr(usage, "prompt_tokens", 0))
        out_tok = _safe_int(getattr(usage, "output_tokens", 0)) or _safe_int(getattr(usage, "completion_tokens", 0))
        reas_tok = _safe_int(getattr(usage, "reasoning_tokens", 0))
        cache_r = _safe_int(getattr(usage, "cache_read_tokens", 0)) or _safe_int(getattr(usage, "cached_tokens", 0))
        cache_w = _safe_int(getattr(usage, "cache_write_tokens", 0))
        total = _safe_int(getattr(usage, "total_tokens", 0)) or (in_tok + out_tok + reas_tok + cache_r + cache_w)
        safe_model = model if isinstance(model, str) else str(model or "unknown")
        safe_source = source if isinstance(source, str) else "group"

        now_dt = datetime.now(timezone.utc)
        self._usage_samples.append(
            _UsageSample(
                ts_epoch=now_dt.timestamp(),
                model=safe_model,
                source=safe_source,
                total_tokens=total,
            )
        )
        self._prune_samples(now_dt.timestamp())

        if self.memory_store is not None:
            await self.memory_store.record_usage(
                timestamp=now_dt.isoformat(),
                model=safe_model,
                source=safe_source,
                input_tokens=in_tok,
                output_tokens=out_tok,
                reasoning_tokens=reas_tok,
                cache_read_tokens=cache_r,
                cache_write_tokens=cache_w,
                total_tokens=total,
            )

    def get_bucket_for_model(self, model: str) -> str:
        """Map a model ID (with or without effort suffix) to its quota bucket."""
        low = model.strip().lower()
        base = _strip_effort_suffix(low)
        for bucket_name, prefixes in self.quota_config.buckets.items():
            for pref in prefixes:
                p_low = pref.strip().lower()
                if low.startswith(p_low) or base.startswith(p_low):
                    return bucket_name
        if "argon" in low or "orchestrator" in low:
            return "gemini_argon"
        if low.startswith(("claude-", "gpt-")):
            return "claude_gpt"
        if low.startswith("gemini-"):
            return "gemini_models"
        return "other"

    def record_rate_limit(
        self,
        model: str,
        retry_after: Optional[float] = None,
        bucket_wide: bool = False,
        reason: str = "",
    ) -> None:
        """Put ``model`` (and optionally its entire quota bucket) on cooldown."""
        if retry_after is not None and retry_after <= 0.0:
            return
        duration = (
            float(retry_after)
            if (retry_after is not None and retry_after > 0.0)
            else float(self.quota_config.default_cooldown_seconds)
        )
        until = time.monotonic() + duration
        norm = model.strip()
        base = _strip_effort_suffix(norm)
        for key in {norm, base}:
            prev = self._model_cooldowns.get(key)
            if prev is None or until > prev[0]:
                self._model_cooldowns[key] = (until, reason)

        if bucket_wide:
            bucket = self.get_bucket_for_model(norm)
            prev_b = self._bucket_cooldowns.get(bucket)
            if prev_b is None or until > prev_b[0]:
                self._bucket_cooldowns[bucket] = (until, reason)

    def _prune_quota_warnings(self, now_dt: Optional[datetime] = None) -> None:
        """Remove expired quota warnings once `expires_at` / `resets_at` has passed or TTL expires."""
        with self._warnings_lock:
            if not self._quota_warnings:
                return
            ref = now_dt if now_dt is not None else datetime.now(timezone.utc)
            for key, warn in list(self._quota_warnings.items()):
                exp_raw = warn.get("expires_at")
                exp_dt = _parse_iso_dt(str(exp_raw)) if exp_raw else None
                if exp_dt is not None:
                    if ref >= exp_dt:
                        self._quota_warnings.pop(key, None)
                    continue
                rec_raw = warn.get("recorded_at")
                rec_dt = _parse_iso_dt(str(rec_raw)) if rec_raw else None
                if rec_dt is None or (ref - rec_dt).total_seconds() >= self._DEFAULT_WARNING_TTL_SECONDS:
                    self._quota_warnings.pop(key, None)
            while len(self._quota_warnings) > self._MAX_QUOTA_WARNINGS:
                oldest_key = next(iter(self._quota_warnings))
                self._quota_warnings.pop(oldest_key, None)

    def record_quota_warning(
        self,
        model: Optional[str],
        remaining_pct: Optional[float],
        resets_at: Optional[str] = None,
    ) -> None:
        """Record a soft `[quota]` warning with automatic TTL/reset expiration."""
        now_dt = datetime.now(timezone.utc)
        key = (model or "unknown").strip()
        reset_dt = _parse_iso_dt(str(resets_at)) if resets_at else None
        default_exp = now_dt + timedelta(seconds=self._DEFAULT_WARNING_TTL_SECONDS)
        if reset_dt is not None and reset_dt > now_dt:
            expires_dt = min(reset_dt, now_dt + timedelta(hours=24))
        else:
            expires_dt = default_exp
        with self._warnings_lock:
            self._prune_quota_warnings(now_dt)
            self._quota_warnings[key] = {
                "model": key,
                "remaining_pct": remaining_pct,
                "resets_at": resets_at,
                "recorded_at": now_dt.isoformat(),
                "expires_at": expires_dt.isoformat(),
            }

    def is_model_cooled_down(self, model: str) -> bool:
        """Return True if ``model`` or its quota bucket is currently in cooldown."""
        now = time.monotonic()
        norm = model.strip()
        base = _strip_effort_suffix(norm)
        for key in (norm, base):
            entry = self._model_cooldowns.get(key)
            if entry is not None:
                if entry[0] > now:
                    return True
                self._model_cooldowns.pop(key, None)

        bucket = self.get_bucket_for_model(norm)
        b_entry = self._bucket_cooldowns.get(bucket)
        if b_entry is not None:
            if b_entry[0] > now:
                return True
            self._bucket_cooldowns.pop(bucket, None)
        return False

    def is_economy_mode(self) -> bool:
        """Return True when token usage exceeds budget threshold or low-quota warning is active."""
        if not self.quota_config.enabled:
            return False
        with self._warnings_lock:
            self._prune_quota_warnings()
            warnings_snapshot = list(self._quota_warnings.values())
        for warn in warnings_snapshot:
            pct = warn.get("remaining_pct")
            if isinstance(pct, (int, float)) and float(pct) <= 20.0:
                return True
        h_cap = int(self.quota_config.hourly_token_budget)
        d_cap = int(self.quota_config.daily_token_budget)
        if h_cap <= 0 and d_cap <= 0:
            return False
        hourly, daily = self.get_usage_totals()
        ratio = float(self.quota_config.economy_threshold_ratio)
        if h_cap > 0 and hourly >= h_cap * ratio:
            return True
        if d_cap > 0 and daily >= d_cap * ratio:
            return True
        return False

    def consume_economy_notification_flag(self) -> bool:
        """Return True once upon entering economy mode, resetting if usage drops."""
        in_econ = self.is_economy_mode()
        if not in_econ:
            self._economy_notified = False
            return False
        if not self._economy_notified:
            self._economy_notified = True
            return True
        return False

    def should_allow_proactive(self) -> bool:
        """Suppress proactive digests/messages while in economy mode."""
        return not self.is_economy_mode()

    def should_allow_spontaneous(self) -> bool:
        """Suppress spontaneous unmentioned group replies while in economy mode."""
        return not self.is_economy_mode()

    def should_allow_self_edit(self, priority: str = "auto") -> Tuple[bool, str]:
        """Allow admin-priority self-edits within reserve, defer auto edits in economy mode."""
        if priority != "admin" and self.is_economy_mode():
            return False, "economy mode active: autonomous self-edit deferred"
        return True, "ok"

    def _is_auto_excluded(self, model: str) -> bool:
        low = model.strip().lower()
        return any(low.startswith(pref) for pref in _EXCLUDED_AUTO_PREFIXES)

    def _split_model_effort(
        self,
        model_id: str,
        default_effort: Optional[str],
    ) -> Tuple[str, str]:
        raw = model_id.strip()
        if raw == "gpt-oss-120b-medium":
            return raw, (default_effort or "medium")
        for suf in _EFFORT_SUFFIXES:
            if raw.endswith(suf):
                return raw[: -len(suf)], (default_effort or suf.lstrip("-"))
        return raw, (default_effort or "low")

    def select_model(self, role: str = "group") -> Tuple[str, str]:
        """Select (model, effort) for ``role`` respecting economy mode & cooldowns."""
        role_map: Dict[str, ModelRoleConfig] = {
            "group": self.model_router.group,
            "proactive": self.model_router.proactive,
            "compaction": self.model_router.compaction,
            "dm_admin": self.model_router.dm_admin,
            "self_edit_judge": self.model_router.self_edit_judge,
            "self_edit": self.model_router.self_edit,
        }
        primary_cfg = role_map.get(role, self.model_router.group)
        preserve_high = role in ("self_edit", "self_edit_judge")

        candidates: List[Tuple[str, str]] = []
        if self.is_economy_mode() and role in ("group", "proactive", "compaction"):
            candidates.append(("gemini-3.8-flash", "low"))

        p_model, p_effort = self._split_model_effort(primary_cfg.model, primary_cfg.effort)
        if preserve_high:
            p_effort = "high"
        candidates.append((p_model, p_effort))

        for fb in self.model_router.fallback_models:
            fb_model, fb_effort = self._split_model_effort(fb.model, fb.effort)
            if preserve_high:
                fb_effort = "high"
            candidates.append((fb_model, fb_effort))

        # Deduplicate preserving order and excluding argon/orchestrator from auto-rotation
        seen: set[str] = set()
        ordered: List[Tuple[str, str]] = []
        for m, e in candidates:
            if self._is_auto_excluded(m):
                continue
            if m not in seen:
                seen.add(m)
                ordered.append((m, e))

        for m, e in ordered:
            if not self.is_model_cooled_down(m):
                return m, e

        # All candidates cooled down: return primary (or safe default) cleanly
        if ordered:
            return ordered[0]
        return "gemini-3.8-flash", ("high" if preserve_high else "low")

    def parse_quota_output(self, tsv_text: str) -> List[Dict[str, Any]]:
        """Parse TSV lines produced by `agy -p /quota`."""
        parsed: List[Dict[str, Any]] = []
        for raw_line in (tsv_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 3:
                continue
            bucket_label = parts[0]
            window_label = parts[1]
            pct_str = parts[2].rstrip("%").strip()
            resets_at = parts[3] if len(parts) > 3 else ""
            try:
                pct = float(pct_str)
            except ValueError:
                pct = None
            parsed.append(
                {
                    "bucket": bucket_label,
                    "window": window_label,
                    "remaining_pct": pct,
                    "raw_pct": parts[2],
                    "resets_at": resets_at,
                }
            )
        if parsed:
            self._live_quota_rows = parsed
        return parsed

    def sync_live_quota(self, timeout: float = 12.0) -> List[Dict[str, Any]]:
        """Execute `agy -p /quota` safely and parse TSV output without raising."""
        try:
            binary = resolve_antigravity_binary()
            env = build_clean_child_env(isolated_home=self.data_dir, strict=True)
            proc = subprocess.run(
                [binary, "-p", "/quota"],
                env=env,
                cwd=str(self.data_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                logger.debug("agy -p /quota exited with rc=%s: %s", proc.returncode, proc.stderr)
                return []
            return self.parse_quota_output(proc.stdout)
        except Exception as exc:
            logger.debug("sync_live_quota failed: %s", exc)
            return []

    def _active_cooldowns_snapshot(self) -> Dict[str, float]:
        now = time.monotonic()
        active: Dict[str, float] = {}
        for k, (until, _) in list(self._model_cooldowns.items()):
            rem = until - now
            if rem > 0:
                active[k] = rem
            else:
                self._model_cooldowns.pop(k, None)
        for b, (until, _) in list(self._bucket_cooldowns.items()):
            if until <= now:
                self._bucket_cooldowns.pop(b, None)
        return active

    def format_quota_report(self) -> str:
        """Render a human-readable Telegram report for `/quota`."""
        with self._warnings_lock:
            self._prune_quota_warnings()
            warnings_snapshot = list(self._quota_warnings.values())
        hourly, daily = self.get_usage_totals()
        mode_str = "ECONOMY" if self.is_economy_mode() else "NORMAL"
        lines = [
            f"📊 *Quota & Budget Status* (`{mode_str}`)",
            f"• Токены (1ч / 24ч): `{hourly:,}` / `{daily:,}`",
            f"• Лимиты (1ч / 24ч): `{self.quota_config.hourly_token_budget:,}` / `{self.quota_config.daily_token_budget:,}`",
        ]

        if self._live_quota_rows:
            lines.append("\n*Live Antigravity Quota:*")
            for row in self._live_quota_rows:
                pct_disp = row["raw_pct"]
                resets = row["resets_at"] or "-"
                lines.append(f"• {row['bucket']} — {row['window']}: `{pct_disp}` (reset: `{resets}`)")

        active_cd = self._active_cooldowns_snapshot()
        if active_cd:
            lines.append("\n*Active Cooldowns:*")
            for m, rem in sorted(active_cd.items()):
                lines.append(f"• `{m}`: `{rem:.0f}s`")

        if warnings_snapshot:
            lines.append("\n*Warnings:*")
            for w in warnings_snapshot:
                lines.append(
                    f"• `{w['model']}`: `{w.get('remaining_pct')}%` (resets `{w.get('resets_at') or '-'}`)"
                )

        grp_m, grp_e = self.select_model("group")
        dm_m, dm_e = self.select_model("dm_admin")
        lines.append(f"\n• Активные модели: group=`{grp_m}` (`{grp_e}`), admin=`{dm_m}` (`{dm_e}`)")
        return "\n".join(lines)

    def format_context_summary(self) -> str:
        """Render compact quota/model state block injected into LLM system instructions."""
        hourly, daily = self.get_usage_totals()
        mode_str = "economy" if self.is_economy_mode() else "normal"
        active_cd = self._active_cooldowns_snapshot()
        cd_text = ", ".join(f"{m}({rem:.0f}s)" for m, rem in sorted(active_cd.items())) or "none"
        grp_m, _ = self.select_model("group")
        return (
            f"[Quota & Model Context: mode={mode_str}, active_group_model={grp_m}, "
            f"tokens_1h={hourly}, tokens_24h={daily}, cooled_down={cd_text}]"
        )

    def format_context_hint(self) -> str:
        """Return a compact quota hint ONLY when it is actionable, otherwise an empty string.

        Unlike `format_context_summary()` this never includes rolling token counters (they change
        on every call and would defeat provider-side Prompt Caching). It is meant to be appended
        to the *end* of the per-turn prompt (never to the stable system-prompt prefix), and is
        empty in the normal steady state (mode=normal, no cooldowns).
        """
        economy = self.is_economy_mode()
        active_cd = self._active_cooldowns_snapshot()
        if not economy and not active_cd:
            return ""
        mode_str = "economy" if economy else "normal"
        cd_text = ", ".join(sorted(active_cd)) or "none"
        grp_m, _ = self.select_model("group")
        return (
            f"[Quota & Model Context: mode={mode_str}, active_group_model={grp_m}, "
            f"cooled_down={cd_text}]"
        )
