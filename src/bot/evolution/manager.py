"""SelfEvolutionManager: runtime error triage, Adversarial Judge, escalation & staging deploy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Tuple

from bot.antigravity import ErrorCategory, classify_error, extract_retry_after
from bot.config.manager import ConfigManager
from bot.quota.manager import QuotaBudgetManager
from bot.security.sanitizer import SecretSanitizer

if TYPE_CHECKING:
    from bot.memory.db import MemoryStore

try:
    from bot.memory.db import SelfEditRecord  # type: ignore[attr-defined]
except ImportError:

    @dataclass
    class SelfEditRecord:  # type: ignore[no-redef]
        """Fallback record definition while memory.db is being upgraded."""

        trigger_type: str
        signature: str
        reason: str
        affected_files: List[str] = field(default_factory=list)
        verdict: str = "NO_CHANGE"
        model: str = ""
        branch: str = ""
        commit_sha: str = ""
        diff_lines: int = 0
        test_result: str = "skipped"
        deploy_result: str = "rejected"
        summary: str = ""
        created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        id: Optional[int] = None


logger = logging.getLogger(__name__)

_TRANSIENT_KEYWORDS = (
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "network is unreachable",
    "temporary failure in name resolution",
    "getupdates timed out",
    "429",
    "resource_exhausted",
    "exhausted your capacity",
    "too many requests",
    "rate limit",
    "ratelimitexceeded",
    "quota exceeded",
    "low on quota",
)

_VERDICT_RE = re.compile(r"\b(CODE_FIX_REQUIRED|CONFIG_ONLY|NO_CHANGE)\b")


def is_transient_external_error(exc: Exception | str) -> bool:
    """Return True for transient network, timeout, or 429/quota errors that should never trigger self-edit."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    verdict = getattr(exc, "verdict", None) or classify_error(exc)
    if verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.BILLING, ErrorCategory.AUTH_PERMANENT):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


class SelfEvolutionManager:
    """Manages safe autonomous & admin-driven self-modification in staging worktree."""

    _MAX_ERROR_WINDOWS = 256
    _GIT_SUBPROCESS_TIMEOUT = 30.0

    def __init__(
        self,
        config_mgr: ConfigManager,
        memory_store: MemoryStore,
        quota_manager: QuotaBudgetManager,
        provider: Any,
        notify_admin: Optional[Callable[[str], Any]] = None,
        userbot: Optional[Any] = None,
        sanitizer: Optional[SecretSanitizer] = None,
    ) -> None:
        self.config_mgr = config_mgr
        self.memory_store: Any = memory_store
        self.quota_manager = quota_manager
        self.provider = provider
        self.notify_admin = notify_admin
        self.userbot = userbot
        self.sanitizer = sanitizer or SecretSanitizer()
        self._error_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._evo_lock = asyncio.Lock()

    async def _send_admin_report(self, text: str) -> None:
        safe_text = self.sanitizer.sanitize(text)
        if self.notify_admin is not None:
            try:
                res = self.notify_admin(safe_text)
                if hasattr(res, "__await__"):
                    await res
                return
            except Exception as exc:
                logger.warning("notify_admin callback failed: %s", exc)
        if self.userbot is not None and self.config_mgr.bot is not None:
            for admin_id in getattr(self.config_mgr.bot.telegram, "admin_user_ids", []) or []:
                try:
                    await self.userbot.send_reply(chat_id=int(admin_id), text=safe_text)
                except Exception as exc:
                    logger.warning("Failed sending evolution report to admin %s: %s", admin_id, exc)

    def _sync_check_daily_limit(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        max_edits = int(cfg.max_daily_edits) if cfg else 2
        if max_edits <= 0:
            return False, "max_daily_edits=0 disables autonomous self-edits"

        ref_dt = now or datetime.now(timezone.utc)
        today_str = ref_dt.astimezone(timezone.utc).date().isoformat()
        try:
            recent = self.memory_store._sync_get_recent_self_edits(limit=100)
        except Exception:
            recent = []
        count_today = 0
        for rec in recent:
            if rec.verdict != "CODE_FIX_REQUIRED" and rec.deploy_result not in ("applied", "pending"):
                continue
            created = (rec.created_at or "")[:10]
            if created == today_str:
                count_today += 1
        if count_today >= max_edits:
            return False, f"daily limit reached ({count_today}/{max_edits} on {today_str})"
        return True, "ok"

    def check_daily_limit(self, now: Optional[datetime] = None) -> Any:
        """Check whether today's UTC autonomous self-edit cap allows another edit.

        Supports both sync `check_daily_limit(now=...) -> bool` and async `await check_daily_limit() -> (bool, str)`.
        """
        if now is not None:
            return self._sync_check_daily_limit(now=now)[0]

        async def _coro() -> Tuple[bool, str]:
            return self._sync_check_daily_limit(now=None)

        return _coro()

    async def check_signature_allowed(self, signature: str) -> Tuple[bool, str]:
        """Return (False, reason) if ``signature`` was handled within signature_cooldown_seconds."""
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        cooldown = int(cfg.signature_cooldown_seconds) if cfg else 21600
        if cooldown <= 0:
            return True, "ok"
        now_dt = datetime.now(timezone.utc)
        recent = await self.memory_store.get_recent_self_edits(limit=50)
        for rec in recent:
            if rec.signature != signature:
                continue
            try:
                rec_dt = datetime.fromisoformat(rec.created_at.replace("Z", "+00:00"))
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                elapsed = (now_dt - rec_dt).total_seconds()
                if elapsed < cooldown:
                    return False, f"signature '{signature}' on cooldown ({cooldown - elapsed:.0f}s left)"
            except Exception:
                continue
        return True, "ok"

    async def check_file_allowed(self, rel_path: str) -> Tuple[bool, str]:
        """Return (False, reason) if ``rel_path`` is protected or within file_cooldown_seconds."""
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        norm = rel_path.strip().lstrip("./")
        protected = (
            list(cfg.protected_paths)
            if cfg
            else [
                ".env",
                "data/",
                "config/userbot.yaml",
                "config/bot.yaml",
                "config/friends.yaml",
                "deploy/bot-self-deploy.sh",
                "systemd/",
            ]
        )
        for prot in protected:
            p_norm = prot.strip().lstrip("./")
            if norm == p_norm or (p_norm.endswith("/") and norm.startswith(p_norm)):
                return False, f"protected path: {norm}"

        cooldown = int(cfg.file_cooldown_seconds) if cfg else 14400
        if cooldown <= 0:
            return True, "ok"
        now_dt = datetime.now(timezone.utc)
        recent = await self.memory_store.get_recent_self_edits(limit=50)
        for rec in recent:
            if norm not in rec.affected_files:
                continue
            try:
                rec_dt = datetime.fromisoformat(rec.created_at.replace("Z", "+00:00"))
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                elapsed = (now_dt - rec_dt).total_seconds()
                if elapsed < cooldown:
                    return False, f"file '{norm}' on cooldown ({cooldown - elapsed:.0f}s left)"
            except Exception:
                continue
        return True, "ok"

    def _parse_judge_verdict(self, raw_text: str) -> Tuple[str, str]:
        text = (raw_text or "").strip()
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                data = json.loads(text[start : end + 1])
                v = str(data.get("verdict") or "").strip().upper()
                r = str(data.get("reason") or "").strip()
                if v in ("CODE_FIX_REQUIRED", "CONFIG_ONLY", "NO_CHANGE"):
                    return v, r or v
        except Exception:
            pass
        upper = text.upper()
        if re.search(r"\b(?:NOT|NO|NEVER|WITHOUT|REJECT|DENY|REFUSE|НЕ)\s+(?:A\s+)?CODE_FIX_REQUIRED\b", upper):
            if "CONFIG_ONLY" in upper:
                return "CONFIG_ONLY", text[:300]
            return "NO_CHANGE", text[:300] or "NO_CHANGE"
        if "NO_CHANGE" in upper:
            return "NO_CHANGE", text[:300]
        if "CONFIG_ONLY" in upper:
            return "CONFIG_ONLY", text[:300]
        m = _VERDICT_RE.search(upper)
        verdict = m.group(1) if m else "NO_CHANGE"
        return verdict, text[:300] or verdict

    async def _run_adversarial_judge(self, trigger_type: str, signature: str, reason: str) -> Tuple[str, str, str]:
        """Critically evaluate whether a code patch is truly required using a non-editing profile."""
        judge_model, judge_effort = self.quota_manager.select_model("self_edit_judge")
        judge_prompt = (
            "Ты строгий Adversarial Judge (критик-оценщик). Попытайся фальсифицировать гипотезу о том, "
            "что нужна правка кода бота. Откажись от правки, если сбой одноразовый, внешний (сеть/429) "
            "или решается изменением конфига.\n"
            f"Источник: {trigger_type}\n"
            f"Сигнатура: {signature}\n"
            f"Описание: {reason}\n\n"
            'Верни строго JSON: {"verdict": "NO_CHANGE" | "CONFIG_ONLY" | "CODE_FIX_REQUIRED", "reason": "..."}'
        )
        judge_system_instructions = (
            "You are a strict adversarial code-change gatekeeper. "
            "Do NOT invoke any tools, web searches, commands, or file reads; "
            "judge strictly from the provided description and return JSON only."
        )
        try:
            resp = await self.provider.generate_response(
                prompt=judge_prompt,
                system_instructions=judge_system_instructions,
                model=judge_model,
                effort=judge_effort,
                tool_profile="chat",
                allowed_tools=frozenset(),
            )
        except TypeError as exc:
            err_s = str(exc)
            if "allowed_tools" not in err_s and "unexpected keyword" not in err_s:
                raise
            resp = await self.provider.generate_response(
                prompt=judge_prompt,
                system_instructions=judge_system_instructions,
                model=judge_model,
                effort=judge_effort,
                tool_profile="chat",
            )
        if getattr(resp, "usage", None) is not None:
            await self.quota_manager.record_usage(
                resp.usage,
                model=resp.model or judge_model,
                source="self_edit",
            )
        verdict, judge_reason = self._parse_judge_verdict(resp.text or "")
        return verdict, judge_reason, (resp.model or judge_model)

    def _inspect_and_commit_staging(
        self,
        *,
        staging_dir: Path,
        signature: str,
        reason: str,
    ) -> Tuple[bool, str, str, str, List[str], int]:
        """Validate changed files in staging worktree, commit them, and return metadata."""
        if not staging_dir.is_dir():
            return False, "staging_dir missing", "", "", [], 0

        status_proc = subprocess.run(
            ["git", "-C", str(staging_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        )
        changed_files: List[str] = []
        for line in status_proc.stdout.splitlines():
            if len(line) >= 4:
                path_part = line[3:].strip().split(" -> ")[-1].strip()
                if path_part:
                    changed_files.append(path_part)

        if not changed_files:
            return False, "no changes produced in staging", "", "", [], 0

        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        protected = (
            list(cfg.protected_paths)
            if cfg
            else [
                ".env",
                "data/",
                "config/userbot.yaml",
                "config/bot.yaml",
                "config/friends.yaml",
                "deploy/bot-self-deploy.sh",
                "systemd/",
            ]
        )
        for rel in changed_files:
            norm = rel.lstrip("./")
            for prot in protected:
                p_norm = prot.strip().lstrip("./")
                if norm == p_norm or (p_norm.endswith("/") and norm.startswith(p_norm)):
                    subprocess.run(
                        ["git", "-C", str(staging_dir), "checkout", "--", rel],
                        capture_output=True,
                        timeout=self._GIT_SUBPROCESS_TIMEOUT,
                    )
                    return False, f"protected path modified: {norm}", "", "", changed_files, 0

        diff_proc = subprocess.run(
            ["git", "-C", str(staging_dir), "diff", "--numstat"],
            capture_output=True,
            text=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        )
        diff_lines = 0
        for line in diff_proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    diff_lines += int(parts[0]) + int(parts[1])
                except ValueError:
                    pass
        # Also count untracked file lines
        for rel in changed_files:
            p = staging_dir / rel
            if p.is_file():
                diff_lines = max(diff_lines, 1)

        max_diff = int(cfg.max_diff_lines) if cfg else 400
        if diff_lines > max_diff:
            return False, f"diff exceeds max_diff_lines ({diff_lines} > {max_diff})", "", "", changed_files, diff_lines

        branch = subprocess.check_output(
            ["git", "-C", str(staging_dir), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        ).strip()
        if branch in ("HEAD", "main", "master"):
            safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", signature)[:32].strip("-") or "fix"
            branch = f"auto-self-edit/{safe_slug}-{int(time.time())}"
            subprocess.run(
                ["git", "-C", str(staging_dir), "checkout", "-b", branch],
                check=True,
                capture_output=True,
                timeout=self._GIT_SUBPROCESS_TIMEOUT,
            )

        subprocess.run(
            ["git", "-C", str(staging_dir), "add", "--", *changed_files],
            check=True,
            capture_output=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging_dir),
                "-c",
                "user.name=BotOps",
                "-c",
                "user.email=botops@localhost",
                "commit",
                "-m",
                f"self-edit: {signature}",
            ],
            check=True,
            capture_output=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        )
        commit_sha = subprocess.check_output(
            ["git", "-C", str(staging_dir), "rev-parse", "HEAD"],
            text=True,
            timeout=self._GIT_SUBPROCESS_TIMEOUT,
        ).strip()
        return True, "committed", branch, commit_sha, changed_files, diff_lines

    def _write_deploy_request_atomic(
        self,
        *,
        trigger_dir: Path,
        branch: str,
        commit_sha: str,
        reason: str,
    ) -> Path:
        trigger_dir.mkdir(parents=True, exist_ok=True)
        req_path = trigger_dir / "request.json"
        tmp_path = trigger_dir / f".request.{os.getpid()}.tmp"
        payload = {
            "request_id": f"req-{int(time.time() * 1000)}",
            "branch": branch,
            "commit_sha": commit_sha,
            "reason": reason[:500],
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, req_path)
        return req_path

    def _new_self_edit_record(self, **kwargs: Any) -> Any:
        """Instantiate SelfEditRecord from memory.db when available, or local fallback."""
        try:
            from bot.memory.db import SelfEditRecord as DbSelfEditRecord

            return DbSelfEditRecord(**kwargs)
        except ImportError:
            return SelfEditRecord(**kwargs)

    async def _execute_evolution_pipeline(
        self,
        *,
        trigger_type: str,
        signature: str,
        reason: str,
    ) -> Optional[SelfEditRecord]:
        async with self._evo_lock:
            return await self._execute_evolution_pipeline_locked(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
            )

    async def _execute_evolution_pipeline_locked(
        self,
        *,
        trigger_type: str,
        signature: str,
        reason: str,
    ) -> Optional[SelfEditRecord]:
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        if cfg is None or not cfg.enabled:
            return None

        try:
            verdict, judge_reason, judge_model = await self._run_adversarial_judge(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
            )
        except Exception as exc:
            logger.warning("Adversarial Judge failed (failing closed): %s", exc)
            rec = self._new_self_edit_record(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
                affected_files=[],
                verdict="NO_CHANGE",
                model="",
                test_result="skipped",
                deploy_result="rejected",
                summary=f"Judge failed closed: {exc}",
            )
            await self.memory_store.record_self_edit(rec)
            await self._send_admin_report(f"🛡 Self-Edit отменён (ошибка критика): {exc}")
            return rec

        if verdict in ("NO_CHANGE", "CONFIG_ONLY"):
            rec = self._new_self_edit_record(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
                affected_files=[],
                verdict=verdict,
                model=judge_model,
                test_result="skipped",
                deploy_result="rejected",
                summary=judge_reason,
            )
            await self.memory_store.record_self_edit(rec)
            await self._send_admin_report(
                f"🛡 Self-Edit отклонён критиком ({verdict}): {judge_reason}"
            )
            return rec

        # CODE_FIX_REQUIRED -> Escalate across configured models in staging worktree
        staging_dir = Path(cfg.staging_dir)
        trigger_dir = Path(cfg.request_file).parent if cfg.request_file else Path(cfg.trigger_dir)
        mode = self.config_mgr.bot.antigravity.self_dev_mode if self.config_mgr.bot else "auto"
        effort = cfg.self_edit_effort or "high"
        escalation = list(cfg.escalation_models or ["gemini-4-hydrogen", "claude-sonnet-5-5", "claude-opus-5-5"])

        dev_prompt = (
            f"Исправь проблему в staging-дереве `{staging_dir}`.\n"
            f"Причина ({trigger_type}): {reason}\n"
            f"Заключение критика: {judge_reason}\n"
            "Не трогай `.env`, `data/`, `config/userbot.yaml`, `config/bot.yaml`, `config/friends.yaml`, "
            "`deploy/bot-self-deploy.sh`, `systemd/`."
        )

        last_err: Optional[Exception] = None
        used_model = ""
        succeeded = False

        for candidate_model in escalation:
            if self.quota_manager.is_model_cooled_down(candidate_model):
                continue
            used_model = candidate_model
            try:
                resp = await self.provider.generate_response(
                    prompt=dev_prompt,
                    system_instructions="You are performing a targeted, minimal fix in an isolated staging git worktree.",
                    model=candidate_model,
                    effort=effort,
                    tool_profile="self_dev",
                    mode=mode,
                    cwd=staging_dir,
                )
                if getattr(resp, "usage", None) is not None:
                    await self.quota_manager.record_usage(
                        resp.usage,
                        model=resp.model or candidate_model,
                        source="self_edit",
                    )
                succeeded = True
                used_model = resp.model or candidate_model
                break
            except Exception as exc:
                last_err = exc
                err_verdict = getattr(exc, "verdict", None) or classify_error(exc)
                if err_verdict.category in (ErrorCategory.RATE_LIMIT, ErrorCategory.MODEL_NOT_FOUND):
                    self.quota_manager.record_rate_limit(
                        candidate_model,
                        retry_after=extract_retry_after(exc),
                        bucket_wide=False,
                        reason=str(exc),
                    )
                logger.warning("self_dev attempt on %s failed: %s", candidate_model, exc)

        if not succeeded:
            summary = f"All escalation models failed or cooled down ({last_err or 'cooldown'})"
            rec = self._new_self_edit_record(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
                verdict=verdict,
                model=used_model or escalation[-1],
                test_result="skipped",
                deploy_result="failed",
                summary=summary,
            )
            await self.memory_store.record_self_edit(rec)
            await self._send_admin_report(f"⚠️ Self-Edit прерван: {summary}")
            return rec

        ok, msg, branch, commit_sha, changed_files, diff_lines = await asyncio.to_thread(
            self._inspect_and_commit_staging,
            staging_dir=staging_dir,
            signature=signature,
            reason=reason,
        )
        if not ok:
            rec = self._new_self_edit_record(
                trigger_type=trigger_type,
                signature=signature,
                reason=reason,
                affected_files=changed_files,
                verdict=verdict,
                model=used_model,
                diff_lines=diff_lines,
                test_result="skipped",
                deploy_result="rejected",
                summary=msg,
            )
            await self.memory_store.record_self_edit(rec)
            await self._send_admin_report(f"⚠️ Self-Edit отклонён перед деплоем: {msg}")
            return rec

        self._write_deploy_request_atomic(
            trigger_dir=trigger_dir,
            branch=branch,
            commit_sha=commit_sha,
            reason=reason,
        )
        rec = self._new_self_edit_record(
            trigger_type=trigger_type,
            signature=signature,
            reason=reason,
            affected_files=changed_files,
            verdict=verdict,
            model=used_model,
            branch=branch,
            commit_sha=commit_sha,
            diff_lines=diff_lines,
            test_result="pending",
            deploy_result="pending",
            summary=f"Committed {commit_sha[:8]} ({len(changed_files)} files, {diff_lines} lines) -> gatekeeper triggered",
        )
        await self.memory_store.record_self_edit(rec)
        await self._send_admin_report(
            f"🛠 Self-Edit отправлен в gatekeeper (`{commit_sha[:8]}`, модель `{used_model}`, файлов: {len(changed_files)})."
        )
        return rec

    async def handle_runtime_error(
        self,
        exc: Exception,
        signature: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[SelfEditRecord]:
        """Evaluate a runtime exception for autonomous self-evolution."""
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        if cfg is None or not cfg.enabled:
            return None
        if is_transient_external_error(exc):
            return None

        sig = signature or f"{type(exc).__name__}:{str(exc)[:80]}"
        now_ts = time.time()
        cutoff = now_ts - float(cfg.error_window_seconds)
        if len(self._error_windows) >= self._MAX_ERROR_WINDOWS:
            for old_sig, old_win in list(self._error_windows.items()):
                while old_win and old_win[0] < cutoff:
                    old_win.popleft()
                if not old_win:
                    self._error_windows.pop(old_sig, None)
            while len(self._error_windows) >= self._MAX_ERROR_WINDOWS:
                oldest_sig = min(
                    self._error_windows,
                    key=lambda k: self._error_windows[k][-1] if self._error_windows[k] else 0.0,
                )
                self._error_windows.pop(oldest_sig, None)

        win = self._error_windows[sig]
        win.append(now_ts)
        while win and win[0] < cutoff:
            win.popleft()
        if len(win) < int(cfg.error_threshold_count):
            if not win:
                self._error_windows.pop(sig, None)
            return None

        allowed_daily, _ = self._sync_check_daily_limit()
        if not allowed_daily:
            return None

        allowed_sig, _ = await self.check_signature_allowed(sig)
        if not allowed_sig:
            return None

        allow_quota, _ = self.quota_manager.should_allow_self_edit(priority="auto")
        if not allow_quota:
            return None

        self._error_windows.pop(sig, None)
        return await self._execute_evolution_pipeline(
            trigger_type="runtime_error",
            signature=sig,
            reason=f"{type(exc).__name__}: {exc}",
        )

    async def run_admin_task(
        self,
        user_id: int,
        task: str,
        *,
        allow_non_admin: bool = False,
    ) -> Optional[SelfEditRecord]:
        """Execute an explicit self-edit task requested by an authorized user."""
        bot_cfg = self.config_mgr.bot
        if bot_cfg is None or not bot_cfg.self_evolution.enabled:
            return None
        admin_ids = {int(uid) for uid in (getattr(bot_cfg.telegram, "admin_user_ids", []) or [])}
        is_admin = int(user_id) in admin_ids
        if not is_admin and not allow_non_admin:
            logger.warning("Rejected non-admin self_edit task from user_id=%s", user_id)
            return None

        allow_quota, reason_q = self.quota_manager.should_allow_self_edit(
            priority="admin" if is_admin else "auto"
        )
        if not allow_quota:
            await self._send_admin_report(f"⚠️ Self-Edit отложен по квоте: {reason_q}")
            return None

        sig_prefix = "dm" if is_admin else "ops"
        sig = f"{sig_prefix}:{hashlib.sha256(task.strip().encode('utf-8')).hexdigest()[:12]}"
        return await self._execute_evolution_pipeline(
            trigger_type="admin_dm" if is_admin else "ops_chat",
            signature=sig,
            reason=task.strip(),
        )

    async def handle_admin_task(
        self,
        task: str,
        requester_id: int,
    ) -> Optional[SelfEditRecord]:
        """Alias for run_admin_task(user_id=requester_id, task=task)."""
        return await self.run_admin_task(user_id=requester_id, task=task)

    async def run_admin_self_edit(
        self,
        task: str,
        requester_id: int,
    ) -> Optional[SelfEditRecord]:
        """Alias for run_admin_task(user_id=requester_id, task=task)."""
        return await self.run_admin_task(user_id=requester_id, task=task)

    async def reconcile_pending_deploy(self) -> Optional[Dict[str, Any]]:
        """Check trigger_dir/result.json after restart/deploy and notify admin."""
        cfg = self.config_mgr.bot.self_evolution if self.config_mgr.bot else None
        if cfg is None:
            return None
        result_path = Path(cfg.result_file) if cfg.result_file else (Path(cfg.trigger_dir) / "result.json")
        if not result_path.is_file():
            return None
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed reading deploy result.json: %s", exc)
            return None

        status = str(payload.get("status") or "unknown")
        req_id = str(payload.get("request_id") or "")
        new_c = str(payload.get("new_commit") or "")[:8]
        prev_c = str(payload.get("prev_commit") or "")[:8]
        err = str(payload.get("error") or "")
        msg = f"📦 Gatekeeper result (`{req_id}`): status=`{status}`, prev=`{prev_c}`, new=`{new_c}`"
        if err:
            msg += f"\nПричина: {err[:300]}"
        await self._send_admin_report(msg)
        try:
            result_path.unlink(missing_ok=True)
        except OSError:
            pass
        return payload

    async def reconcile_deploy_result(self) -> Optional[Dict[str, Any]]:
        """Alias for reconcile_pending_deploy()."""
        return await self.reconcile_pending_deploy()

    async def reconcile_pending_result(self) -> Optional[Dict[str, Any]]:
        """Alias for reconcile_pending_deploy()."""
        return await self.reconcile_pending_deploy()
