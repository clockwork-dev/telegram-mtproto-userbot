"""Subprocess CLI adapter for agy fallback with persistent worker pool and watchdog."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

from PIL import Image

from bot.antigravity.auth import (
    build_clean_child_env,
    resolve_antigravity_binary,
    resolve_antigravity_home,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import (
    AntigravityError,
    AntigravityNativeToolAbort,
    AntigravityRateLimitError,
    classify_error,
)
from bot.antigravity.stream import (
    _ALLOWED_NATIVE_TOOLS,
    _FORBIDDEN_NATIVE_TOOLS,
    AntigravityStreamParser,
)
from bot.core.models import MediaAttachment, ProviderResponse, UsageInfo

logger = logging.getLogger(__name__)

_ERROR_LOG_PATTERNS = re.compile(
    r"(RESOURCE_EXHAUSTED|exhausted your capacity|quota.*reset|429\s+Too\s+Many\s+Requests|model_Validation|RateLimitExceeded)",
    re.IGNORECASE,
)


def popen_isolated_session(
    cmd: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    """Launch cmd in a new isolated process group with UTF-8 line-buffered pipes."""
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "env": env,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        popen_kwargs["creationflags"] = flags
    else:
        popen_kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **popen_kwargs)


def terminate_process_group(proc: subprocess.Popen[Any] | None, timeout: float = 3.0) -> None:
    """Terminate a child process and its process group gracefully, escalating to SIGKILL."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return

    pid = getattr(proc, "pid", None)
    if sys.platform == "win32":
        try:
            proc.terminate()
        except OSError:
            pass
    else:
        if isinstance(pid, int) and pid > 0:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGTERM)
            except OSError:
                try:
                    proc.terminate()
                except OSError:
                    pass
        else:
            try:
                proc.terminate()
            except OSError:
                pass

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    if sys.platform == "win32":
        try:
            proc.kill()
        except OSError:
            pass
    else:
        if isinstance(pid, int) and pid > 0:
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
        else:
            try:
                proc.kill()
            except OSError:
                pass

    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass


class StderrDrainer:
    """Background daemon thread draining child stderr pipe into a bounded ring buffer."""

    def __init__(self, stream: IO[str] | None, *, max_lines: int = 100) -> None:
        self._stream = stream
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if stream is not None:
            self._thread = threading.Thread(
                target=self._drain_loop,
                name="agy-stderr-drainer",
                daemon=True,
            )
            self._thread.start()

    def _drain_loop(self) -> None:
        if self._stream is None:
            return
        try:
            for line in iter(self._stream.readline, ""):
                if not line:
                    break
                stripped = line.rstrip("\r\n")
                if stripped:
                    with self._lock:
                        self._lines.append(stripped)
        except (OSError, ValueError):
            pass

    def get_output(self) -> str:
        """Return buffered stderr lines."""
        with self._lock:
            return "\n".join(self._lines)


def watch_log_for_errors(
    gemini_dir: Path,
    stop_event: threading.Event,
    error_holder: list[str],
    on_error_kill: Callable[[], None] | None = None,
) -> None:
    """Watch gemini_dir logs for fatal rate limits or capacity errors."""
    top_cli_log = gemini_dir / "cli.log"
    log_dirs = (gemini_dir / "log", gemini_dir / "logs")
    start_time = time.time() - 2.0
    file_offsets: dict[Path, int] = {}

    while not stop_event.is_set():
        try:
            candidate_logs: list[Path] = []
            if top_cli_log.is_file():
                candidate_logs.append(top_cli_log)
            for ldir in log_dirs:
                if ldir.is_dir():
                    candidate_logs.extend(ldir.glob("*.log"))

            for log_file in candidate_logs:
                try:
                    st = log_file.stat()
                    if st.st_mtime < start_time:
                        continue
                    offset = file_offsets.get(log_file, 0)
                    if st.st_size <= offset:
                        continue
                    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        file_offsets[log_file] = fh.tell()
                    for line in chunk.splitlines():
                        if _ERROR_LOG_PATTERNS.search(line):
                            error_holder.append(line.strip())
                            if on_error_kill is not None:
                                try:
                                    on_error_kill()
                                except Exception:
                                    pass
                            return
                except OSError:
                    continue
        except OSError:
            pass
        stop_event.wait(0.25)


_IMAGE_EXT_MIME: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _normalize_image_attachment(att: MediaAttachment) -> tuple[str, str] | None:
    """Return (base64_data, mime_type) for an image MediaAttachment, or None if non-image/empty."""
    mime = (att.mime_type or "").lower().strip()
    if not mime.startswith("image/") and att.file_path is not None:
        mime = _IMAGE_EXT_MIME.get(att.file_path.suffix.lower(), "")
    if not mime.startswith("image/"):
        return None

    raw_bytes = att.get_bytes()
    if not raw_bytes:
        return None

    target_mime = mime
    if target_mime not in ("image/jpeg", "image/png"):
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                out = io.BytesIO()
                mode = "RGBA" if img.mode in ("RGBA", "LA") or "transparency" in img.info else "RGB"
                img.convert(mode).save(out, format="PNG")
                raw_bytes = out.getvalue()
                target_mime = "image/png"
        except Exception as exc:
            logger.debug("Could not transcode image %s to PNG: %s", mime, exc)

    return base64.b64encode(raw_bytes).decode("ascii"), target_mime


def _select_acp_model(
    available_models: list[dict[str, Any]],
    model: str,
    effort: str | None,
) -> str | None:
    """Resolve the best matching ACP modelId from session/new availableModels."""
    ids = [str(m.get("modelId") or "").strip() for m in available_models if isinstance(m, dict)]
    ids = [mid for mid in ids if mid]
    if not ids:
        return None
    if effort:
        candidate = f"{model}-{effort}"
        if candidate in ids:
            return candidate
    if model in ids:
        return model
    for mid in ids:
        if mid.startswith(f"{model}-"):
            return mid
    return None


class AntigravityCLIClient:
    """Subprocess adapter executing agy CLI with fresh stateless cascades and idle-reaped workers."""

    supports_image_vision: bool = True

    def __init__(
        self,
        data_dir: Path | str | None = None,
        binary: str | None = None,
        default_model: str = "gemini-4-hydrogen",
        default_effort: str | None = "low",
        worker_idle_timeout_seconds: float = 300.0,
    ) -> None:
        self.data_dir = resolve_antigravity_home(data_dir)
        self.binary = binary or resolve_antigravity_binary()
        self.default_model = default_model
        self.default_effort = default_effort
        self.worker_idle_timeout_seconds = max(0.0, float(worker_idle_timeout_seconds))

        self._worker_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._worker_proc: subprocess.Popen[str] | None = None
        self._worker_stderr: StderrDrainer | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_gemini_dir: Path | None = None
        self._worker_last_used: float = 0.0
        self._worker_busy: bool = False
        self._idle_timer: threading.Timer | None = None

    def close(self) -> None:
        """Kill any persistent worker, cancel idle reaper, and sync updated tokens."""
        self._kill_worker()

    def _cancel_idle_timer_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _schedule_idle_cleanup(self) -> None:
        if self.worker_idle_timeout_seconds <= 0.0:
            return
        with self._state_lock:
            self._cancel_idle_timer_locked()
            if self._worker_proc is None:
                return
            self._worker_last_used = time.monotonic()
            timer = threading.Timer(self.worker_idle_timeout_seconds, self._on_idle_timeout)
            timer.daemon = True
            timer.name = "agy-worker-idle-reaper"
            self._idle_timer = timer
            timer.start()

    def _on_idle_timeout(self) -> None:
        with self._state_lock:
            if self._worker_proc is None or self._worker_busy:
                return
            idle_elapsed = time.monotonic() - self._worker_last_used
            if idle_elapsed < self.worker_idle_timeout_seconds:
                remaining = self.worker_idle_timeout_seconds - idle_elapsed
                timer = threading.Timer(remaining, self._on_idle_timeout)
                timer.daemon = True
                timer.name = "agy-worker-idle-reaper"
                self._idle_timer = timer
                timer.start()
                return
        self._kill_worker()

    def _kill_worker(self) -> None:
        with self._state_lock:
            self._cancel_idle_timer_locked()
            proc = self._worker_proc
            g_dir = self._worker_gemini_dir
            self._worker_proc = None
            self._worker_stderr = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_gemini_dir = None
            self._worker_busy = False

        if proc is not None:
            terminate_process_group(proc)
        if g_dir is not None:
            sync_updated_isolated_tokens(g_dir, data_dir=self.data_dir)

    def _ensure_worker(
        self,
        model: str,
        effort: str | None,
    ) -> tuple[subprocess.Popen[str], StderrDrainer, Path]:
        with self._state_lock:
            self._cancel_idle_timer_locked()
            proc = self._worker_proc
            drainer = self._worker_stderr
            g_dir = self._worker_gemini_dir
            same_cfg = (self._worker_model == model and self._worker_effort == effort)

        if proc is not None and drainer is not None and g_dir is not None and same_cfg and proc.poll() is None:
            with self._state_lock:
                self._worker_last_used = time.monotonic()
            return proc, drainer, g_dir

        self._kill_worker()

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        cmd = [
            self.binary,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--model",
            model,
        ]
        if effort:
            cmd.extend(["--effort", effort])

        new_proc = popen_isolated_session(cmd, cwd=isolated_home, env=clean_env)
        new_drainer = StderrDrainer(new_proc.stderr)

        with self._state_lock:
            self._worker_proc = new_proc
            self._worker_stderr = new_drainer
            self._worker_model = model
            self._worker_effort = effort
            self._worker_gemini_dir = gemini_dir
            self._worker_last_used = time.monotonic()

        self._schedule_idle_cleanup()
        return new_proc, new_drainer, gemini_dir

    def _spawn_oneshot(
        self,
        prompt: str,
        model: str,
        effort: str | None,
        conversation_id: str | None = None,
    ) -> tuple[subprocess.Popen[str], StderrDrainer, Path]:
        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        cmd = [
            self.binary,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--model",
            model,
        ]
        if effort:
            cmd.extend(["--effort", effort])
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])

        proc = popen_isolated_session(cmd, cwd=isolated_home, env=clean_env)
        drainer = StderrDrainer(proc.stderr)

        payload = json.dumps({"event": "user", "message": {"content": prompt}})
        if proc.stdin is not None:
            try:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
            except OSError as exc:
                terminate_process_group(proc)
                raise RuntimeError(f"Failed writing prompt to oneshot agy stdin: {exc}") from exc
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        return proc, drainer, gemini_dir

    def _sync_execute_acp_turn(
        self,
        prompt: str,
        image_payloads: list[tuple[str, str]],
        system_instructions: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> ProviderResponse:
        """Execute a multimodal turn via `agy acp --stdio` with native image blocks."""
        resolved_model = model or self.default_model
        resolved_effort = effort if effort is not None else self.default_effort

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        cmd = [self.binary, "acp", "--stdio"]
        proc = popen_isolated_session(cmd, cwd=isolated_home, env=clean_env)
        drainer = StderrDrainer(proc.stderr)

        watchdog_errors: list[str] = []
        stop_watchdog = threading.Event()
        watchdog_thread = threading.Thread(
            target=watch_log_for_errors,
            args=(gemini_dir, stop_watchdog, watchdog_errors, lambda: terminate_process_group(proc)),
            name="agy-acp-watchdog",
            daemon=True,
        )
        watchdog_thread.start()

        def _send_rpc(msg_id: int | str | None, method: str | None, params_or_result: dict[str, Any], *, is_response: bool = False) -> None:
            if proc.stdin is None:
                raise RuntimeError("ACP stdin pipe is unavailable.")
            if is_response:
                frame: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "result": params_or_result}
            else:
                frame = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params_or_result}
            proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            proc.stdin.flush()

        def _check_acp_tool_allowed(update: dict[str, Any]) -> None:
            raw_name = str(
                update.get("title")
                or update.get("toolName")
                or update.get("name")
                or (update.get("toolCall") or {}).get("name")
                or ""
            ).strip().lower()
            if not raw_name:
                return
            if raw_name in _FORBIDDEN_NATIVE_TOOLS or raw_name not in _ALLOWED_NATIVE_TOOLS:
                terminate_process_group(proc)
                raise AntigravityNativeToolAbort(
                    f"Antigravity ACP native tool execution intercepted: {raw_name}"
                )

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        session_id: str | None = conversation_id
        finish_reason = "stop"
        usage: UsageInfo | None = None

        def _wait_rpc_response(expected_id: int) -> dict[str, Any]:
            if proc.stdout is None:
                raise RuntimeError("ACP stdout pipe is unavailable.")
            for line in iter(proc.stdout.readline, ""):
                if watchdog_errors:
                    raise AntigravityRateLimitError(
                        f"Rate limit detected in logs: {watchdog_errors[0]}"
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    msg = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue

                # Handle server->client permission requests safely (never auto-approve forbidden tools)
                req_method = msg.get("method")
                if req_method == "session/request_permission" and "id" in msg:
                    params = msg.get("params") or {}
                    tool_call = params.get("toolCall") or {}
                    tool_name = str(tool_call.get("name") or tool_call.get("title") or "").strip().lower()
                    options = params.get("options") or []
                    chosen_option = None
                    if tool_name in _ALLOWED_NATIVE_TOOLS:
                        for opt in options:
                            if str(opt.get("kind") or "").startswith("allow"):
                                chosen_option = opt.get("optionId")
                                break
                    if chosen_option:
                        _send_rpc(
                            msg["id"],
                            None,
                            {"outcome": {"outcome": "selected", "optionId": chosen_option}},
                            is_response=True,
                        )
                    else:
                        _send_rpc(
                            msg["id"],
                            None,
                            {"outcome": {"outcome": "cancelled"}},
                            is_response=True,
                        )
                    continue

                if req_method == "session/update":
                    update = (msg.get("params") or {}).get("update") or {}
                    upd_type = str(update.get("sessionUpdate") or "")
                    if upd_type == "agent_message_chunk":
                        content = update.get("content") or {}
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            text_parts.append(content["text"])
                    elif upd_type == "agent_thought_chunk":
                        content = update.get("content") or {}
                        if isinstance(content, dict) and isinstance(content.get("text"), str):
                            thinking_parts.append(content["text"])
                    elif upd_type == "tool_call":
                        _check_acp_tool_allowed(update)
                    continue

                if msg.get("id") == expected_id:
                    if "error" in msg and msg["error"]:
                        err = msg["error"]
                        err_msg = err.get("message") if isinstance(err, dict) else str(err)
                        raise RuntimeError(f"Antigravity ACP error: {err_msg}")
                    return msg.get("result") or {}

            rc = proc.poll()
            stderr_out = drainer.get_output().strip()
            raise RuntimeError(f"agy acp exited unexpectedly (rc={rc}): {stderr_out}")

        try:
            _send_rpc(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
            _wait_rpc_response(1)

            new_params: dict[str, Any] = {
                "cwd": str(isolated_home),
                "mcpServers": [],
            }
            if system_instructions:
                new_params["_meta"] = {
                    "antigravity": {"systemPrompt": {"append": system_instructions}}
                }
            _send_rpc(2, "session/new", new_params)
            new_res = _wait_rpc_response(2)
            session_id = str(new_res.get("sessionId") or "") or session_id
            if not session_id:
                raise RuntimeError("agy acp session/new did not return a sessionId.")

            models_info = new_res.get("models") or {}
            available_models = models_info.get("availableModels") or []
            target_model_id = _select_acp_model(available_models, resolved_model, resolved_effort)
            current_model_id = models_info.get("currentModelId")
            if target_model_id and target_model_id != current_model_id:
                _send_rpc(
                    3,
                    "session/set_model",
                    {"sessionId": session_id, "modelId": target_model_id},
                )
                _wait_rpc_response(3)

            prompt_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64_data, mime_type in image_payloads:
                prompt_blocks.append(
                    {"type": "image", "data": b64_data, "mimeType": mime_type}
                )

            _send_rpc(
                4,
                "session/prompt",
                {"sessionId": session_id, "prompt": prompt_blocks},
            )
            prompt_res = _wait_rpc_response(4)
            finish_reason = str(prompt_res.get("stopReason") or "stop")
            raw_usage = prompt_res.get("usage")
            if isinstance(raw_usage, dict):
                usage = UsageInfo(
                    prompt_tokens=int(raw_usage.get("inputTokens") or 0),
                    completion_tokens=int(raw_usage.get("outputTokens") or 0),
                    total_tokens=int(raw_usage.get("totalTokens") or 0),
                    reasoning_tokens=int(raw_usage.get("thoughtTokens") or 0),
                    cached_tokens=int(raw_usage.get("cachedReadTokens") or 0),
                )

            if watchdog_errors:
                raise AntigravityRateLimitError(
                    f"Rate limit detected in logs: {watchdog_errors[0]}"
                )
        except AntigravityNativeToolAbort:
            terminate_process_group(proc)
            raise
        except Exception as exc:
            verdict = classify_error(exc)
            if verdict.category == "rate_limit":
                raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
            raise AntigravityError(str(exc), verdict=verdict) from exc
        finally:
            stop_watchdog.set()
            terminate_process_group(proc)

        sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)

        return ProviderResponse(
            text="".join(text_parts),
            thinking="".join(thinking_parts) or None,
            conversation_id=session_id,
            finish_reason=finish_reason,
            usage=usage,
            model=resolved_model,
        )

    def _sync_execute_turn(
        self,
        prompt: str,
        system_instructions: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        media_attachments: Sequence[MediaAttachment] | None = None,
    ) -> ProviderResponse:
        resolved_model = model or self.default_model
        resolved_effort = effort if effort is not None else self.default_effort

        image_payloads: list[tuple[str, str]] = []
        if media_attachments:
            for att in media_attachments:
                normalized = _normalize_image_attachment(att)
                if normalized is not None:
                    image_payloads.append(normalized)

        if image_payloads:
            return self._sync_execute_acp_turn(
                prompt=prompt,
                image_payloads=image_payloads,
                system_instructions=system_instructions,
                conversation_id=conversation_id,
                model=resolved_model,
                effort=resolved_effort,
            )

        full_prompt = prompt
        if system_instructions:
            full_prompt = f"{system_instructions}\n\n{prompt}"

        # Stateless full-context turns (conversation_id is None) always run in a fresh
        # isolated cascade so trajectory tokens never accumulate O(N^2) across turns
        # and idle workers do not remain alive polling fetchAvailableModels/loadCodeAssist.
        proc, drainer, gemini_dir = self._spawn_oneshot(
            prompt=full_prompt,
            model=resolved_model,
            effort=resolved_effort,
            conversation_id=conversation_id,
        )

        # Error watchdog
        watchdog_errors: list[str] = []
        stop_watchdog = threading.Event()
        watchdog_thread = threading.Thread(
            target=watch_log_for_errors,
            args=(gemini_dir, stop_watchdog, watchdog_errors, lambda: terminate_process_group(proc)),
            name="agy-cli-watchdog",
            daemon=True,
        )
        watchdog_thread.start()

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        extracted_conv_id = conversation_id
        finish_reason = "stop"
        usage: UsageInfo | None = None

        def on_conv(cid: str) -> None:
            nonlocal extracted_conv_id
            extracted_conv_id = cid

        parser = AntigravityStreamParser(
            on_conversation_id=on_conv,
            on_native_tool_intercept=lambda: terminate_process_group(proc),
        )

        try:
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    if watchdog_errors:
                        raise AntigravityRateLimitError(
                            f"Rate limit detected in logs: {watchdog_errors[0]}"
                        )
                    for chunk in parser.feed_line(line):
                        if chunk.text:
                            text_parts.append(chunk.text)
                        if chunk.thinking:
                            thinking_parts.append(chunk.thinking)
                    if parser.saw_result_event:
                        break

            for chunk in parser.finish():
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.thinking:
                    thinking_parts.append(chunk.thinking)
                if chunk.is_final:
                    finish_reason = chunk.finish_reason or "stop"
                    usage = chunk.usage

            if watchdog_errors:
                raise AntigravityRateLimitError(
                    f"Rate limit detected in logs: {watchdog_errors[0]}"
                )

            if not parser.saw_result_event and not text_parts:
                rc = proc.poll()
                stderr_out = drainer.get_output().strip()
                if rc not in (None, 0) and stderr_out:
                    raise RuntimeError(f"agy exited with code {rc}: {stderr_out}")

        except AntigravityNativeToolAbort:
            terminate_process_group(proc)
            raise
        except Exception as exc:
            verdict = classify_error(exc)
            if verdict.category == "rate_limit":
                raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
            raise AntigravityError(str(exc), verdict=verdict) from exc
        finally:
            stop_watchdog.set()
            terminate_process_group(proc)

        sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)

        return ProviderResponse(
            text="".join(text_parts),
            thinking="".join(thinking_parts) or None,
            conversation_id=extracted_conv_id,
            finish_reason=finish_reason,
            usage=usage,
            model=resolved_model,
        )

    async def generate_response(
        self,
        prompt: str,
        system_instructions: str | None = None,
        media_attachments: Sequence[MediaAttachment] | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
    ) -> ProviderResponse:
        """Asynchronously generate a response without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._sync_execute_turn,
            prompt,
            system_instructions,
            conversation_id,
            model,
            effort,
            media_attachments,
        )
