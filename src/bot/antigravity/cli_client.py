"""Subprocess CLI adapter for agy fallback with persistent worker pool and watchdog."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import IO, Any

from PIL import Image

from bot.antigravity.auth import (
    build_clean_child_env,
    cleanup_old_antigravity_sessions,
    resolve_antigravity_binary,
    resolve_antigravity_home,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import (
    AntigravityBillingError,
    AntigravityError,
    AntigravityNativeToolAbort,
    AntigravityRateLimitError,
    ErrorCategory,
    ImageGenerationError,
    ImagePolicyRefusalError,
    ImageTimeoutError,
    ImageValidationError,
    classify_error,
    is_quota_warning_line,
    parse_quota_warning,
)
from bot.antigravity.stream import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    AntigravityStreamParser,
    ToolProfile,
    resolve_allowed_tools,
)
from bot.core.models import GeneratedImageResult, MediaAttachment, ProviderResponse, UsageInfo

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


def _close_proc_pipes(proc: subprocess.Popen[Any] | None) -> None:
    """Safely close stdin/stdout/stderr pipes on a finished or terminated Popen object."""
    if proc is None:
        return
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(proc, stream_name, None)
        if stream is None:
            continue
        close_fn = getattr(stream, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def terminate_process_group(proc: subprocess.Popen[Any] | None, timeout: float = 3.0) -> None:
    """Terminate a child process and its process group gracefully, escalating to SIGKILL."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            try:
                proc.wait(timeout=0.1)
            except Exception:
                pass
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

    def join(self, timeout: float = 1.0) -> None:
        """Wait briefly for the stderr drainer thread to finish."""
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            try:
                thread.join(timeout=timeout)
            except RuntimeError:
                pass


def _snapshot_log_offsets(gemini_dir: Path) -> dict[Path, int]:
    """Capture current byte sizes of existing agy log files before spawning a new turn."""
    offsets: dict[Path, int] = {}
    top_cli_log = gemini_dir / "cli.log"
    try:
        if top_cli_log.is_file():
            offsets[top_cli_log] = top_cli_log.stat().st_size
    except OSError:
        pass
    for ldir in (gemini_dir / "log", gemini_dir / "logs"):
        try:
            if not ldir.is_dir():
                continue
            for log_file in ldir.glob("*.log"):
                try:
                    offsets[log_file] = log_file.stat().st_size
                except OSError:
                    continue
        except OSError:
            continue
    return offsets


def _resolve_extra_add_dirs(
    effective_cwd: Path,
    add_dirs: Sequence[Path | str] | None,
) -> list[str]:
    """Return deduplicated --add-dir paths excluding effective_cwd (already in workspaceDirs)."""
    cwd_resolved = str(effective_cwd.resolve())
    seen: set[str] = {cwd_resolved}
    resolved_extras: list[str] = []
    for extra_dir in add_dirs or ():
        norm_extra = str(Path(extra_dir).resolve())
        if norm_extra not in seen:
            seen.add(norm_extra)
            resolved_extras.append(norm_extra)
    return resolved_extras


def _append_cli_mode_flags(
    cmd: list[str],
    norm_mode: str | None,
    *,
    tool_profile: ToolProfile,
    zero_tools: bool = False,
) -> None:
    """Append agy print-mode execution flags cleanly without triggering rollout warnings.

    In agy 1.5.x headless `stream-json` mode, `--mode auto` emits a rollout-flag warning and
    falls back to default (`request-review`), which auto-denies tool calls in headless ops/self_dev
    turns. For `ops`/`self_dev` profiles requesting `auto` (when tools are not disabled), we pass
    `--mode accept-edits` together with `--dangerously-skip-permissions` while
    `AntigravityStreamParser` continues to enforce the strict `OPS_ALLOWED_TOOLS` allowlist and
    `ALWAYS_FORBIDDEN_TOOLS` kill-switch.
    """
    if zero_tools:
        return
    if norm_mode == "auto":
        if tool_profile in ("ops", "self_dev"):
            cmd.extend(["--mode", "accept-edits", "--dangerously-skip-permissions"])
        return
    if norm_mode in ("accept-edits", "plan"):
        cmd.extend(["--mode", norm_mode])


def watch_log_for_errors(
    gemini_dir: Path,
    stop_event: threading.Event,
    error_holder: list[str],
    on_error_kill: Callable[[], None] | None = None,
    on_quota_warning: Callable[[str | None, float | None, str | None], None] | None = None,
    initial_offsets: dict[Path, int] | None = None,
) -> None:
    """Watch gemini_dir logs for fatal rate limits or capacity errors.

    Soft ``[quota] ... low on quota`` warnings never kill the process; they are
    reported through ``on_quota_warning(model, remaining_pct, resets_at)``.
    """
    top_cli_log = gemini_dir / "cli.log"
    log_dirs = (gemini_dir / "log", gemini_dir / "logs")
    start_time = time.time() - 2.0
    file_offsets: dict[Path, int] = dict(initial_offsets) if initial_offsets else {}

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
                        # Record historical file size so later appends don't re-scan old errors
                        if log_file not in file_offsets:
                            file_offsets[log_file] = st.st_size
                        continue
                    offset = file_offsets.get(log_file, 0)
                    if st.st_size < offset:
                        # Log file was truncated or rotated
                        offset = 0
                    if st.st_size <= offset:
                        continue
                    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                        file_offsets[log_file] = fh.tell()
                    for line in chunk.splitlines():
                        if is_quota_warning_line(line):
                            model, pct, resets_at = parse_quota_warning(line)
                            logger.warning("Antigravity quota warning: %s", line.strip()[:200])
                            if on_quota_warning is not None:
                                try:
                                    on_quota_warning(model, pct, resets_at)
                                except Exception:
                                    logger.debug("on_quota_warning callback failed", exc_info=True)
                            continue
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


_SAFE_IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_SAFE_CONV_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_IMAGE_POLICY_REFUSAL_RE = re.compile(
    r"(?:"
    r"safety\s+(?:policy|guidelines|filter|check)|"
    r"content\s+policy|"
    r"violat(?:es|ing)\s+(?:our|the|safety|content)?\s*(?:policy|policies|guidelines|rules)|"
    r"(?:cannot|can't|unable\s+to|not\s+allowed\s+to|refuse[sd]?\s+to)\s+(?:generate|create|produce|draw|render)|"
    r"inappropriate|explicit\s+content|blocked\s+by\s+(?:safety|policy|filter)|"
    r"не\s+могу\s+(?:сгенерировать|создать|нарисовать)|"
    r"не\s+можу\s+(?:згенерувати|створити|намалювати)|"
    r"политик[аеуи]\s+безопасности|нарушает\s+правила|недопустим|запрещен"
    r")",
    re.IGNORECASE,
)


def _detect_image_magic(data: bytes) -> tuple[str, str] | None:
    """Detect supported image format (JPEG/PNG/WEBP) from magic header bytes."""
    if len(data) >= 8 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if len(data) >= 3 and data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    return None


def _validate_image_bytes_strict(data: bytes, *, max_bytes: int) -> tuple[str, str, int, int]:
    """Validate non-empty size, magic bytes, and PIL integrity; return (mime, ext, w, h)."""
    if not data:
        raise ImageValidationError("Image data is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"Image size {len(data)} bytes exceeds maximum allowed {max_bytes} bytes."
        )
    magic = _detect_image_magic(data)
    if magic is None:
        raise ImageValidationError("Unsupported or unrecognized image magic bytes (expected JPEG/PNG/WEBP).")
    mime_type, ext = magic
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception as exc:
        raise ImageValidationError(f"Corrupt or invalid image payload: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ImageValidationError(f"Invalid image dimensions: {width}x{height}")
    return mime_type, ext, int(width), int(height)


def _write_pre_tool_hooks_manifest(
    agents_dir: Path,
    *,
    hook_name: str,
    gate_script: Path,
    gate_source: str,
) -> Path:
    """Write a gatekeeper script and fail-closed `.agents/hooks.json` PreToolUse manifest."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(agents_dir, 0o700)
    except OSError:
        pass

    gate_script.write_text(gate_source, encoding="utf-8")
    try:
        os.chmod(gate_script, 0o700)
    except OSError:
        pass

    if sys.platform == "win32":
        hook_cmd = f'"{sys.executable}" "{gate_script}"'
    else:
        hook_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(gate_script))}"

    hooks_json = agents_dir / "hooks.json"
    hooks_payload = {
        hook_name: {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": hook_cmd,
                            "timeout": 5,
                        }
                    ],
                }
            ]
        }
    }
    hooks_json.write_text(json.dumps(hooks_payload, indent=2), encoding="utf-8")
    try:
        os.chmod(hooks_json, 0o600)
    except OSError:
        pass
    return hooks_json


def _resolve_effective_allowed_tools(
    tool_profile: ToolProfile,
    allowed_tools: frozenset[str] | Collection[str] | None = None,
) -> frozenset[str]:
    """Resolve effective allowed tools, capping 'chat' profile to CHAT_ALLOWED_TOOLS."""
    passed = (
        frozenset(str(name).strip().lower() for name in allowed_tools if str(name).strip())
        if allowed_tools is not None
        else None
    )
    resolved = resolve_allowed_tools(tool_profile=tool_profile, allowed_tools=passed)
    if tool_profile == "chat":
        return resolved & CHAT_ALLOWED_TOOLS
    return resolved


def _needs_isolated_chat_sandbox(
    tool_profile: ToolProfile,
    allowed_tools: frozenset[str] | Collection[str] | None,
    resolved_allowed: frozenset[str],
) -> bool:
    """Return True when the turn must run in an isolated 0700 chat/zero-tool sandbox cwd."""
    return tool_profile == "chat" or (allowed_tools is not None and len(resolved_allowed) == 0)


def write_chat_pre_tool_hook(
    sandbox_cwd: Path,
    *,
    allowed_tools: frozenset[str] | Collection[str] | None = None,
) -> tuple[Path, Path]:
    """Write fail-closed `.agents/hooks.json` + Python PreToolUse firewall for chat/zero-tool sandboxes.

    When `allowed_tools` is None, permits ONLY `CHAT_ALLOWED_TOOLS` (`search_web`,
    `read_url_content`). When `allowed_tools` is empty (e.g. Adversarial Judge), denies ALL tools.
    """
    resolved_cwd = sandbox_cwd.resolve()
    agents_dir = resolved_cwd / ".agents"
    if allowed_tools is None:
        effective_allowed = sorted(CHAT_ALLOWED_TOOLS - ALWAYS_FORBIDDEN_TOOLS)
    else:
        effective_allowed = sorted(
            frozenset(
                str(name).strip().lower() for name in allowed_tools if str(name).strip()
            )
            - ALWAYS_FORBIDDEN_TOOLS
        )

    gate_script = agents_dir / "chat_gate.py"
    gate_source = (
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "\n"
        f"ALLOWED_TOOLS = frozenset({effective_allowed!r})\n"
        "\n"
        "def deny(reason: str) -> None:\n"
        "    sys.stdout.write(json.dumps({'decision': 'deny', 'reason': reason}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    sys.exit(0)\n"
        "\n"
        "def allow() -> None:\n"
        "    sys.stdout.write(json.dumps({'decision': 'allow'}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    sys.exit(0)\n"
        "\n"
        "def main() -> None:\n"
        "    try:\n"
        "        payload = json.load(sys.stdin)\n"
        "    except Exception:\n"
        "        deny('Malformed PreToolUse hook payload')\n"
        "        return\n"
        "    if not isinstance(payload, dict):\n"
        "        deny('Non-object PreToolUse hook payload')\n"
        "        return\n"
        "    tc = payload.get('toolCall')\n"
        "    if not isinstance(tc, dict):\n"
        "        deny('Missing toolCall object')\n"
        "        return\n"
        "    raw_name = tc.get('name') or tc.get('toolName') or tc.get('tool_name') or ''\n"
        "    name = str(raw_name).strip().lower()\n"
        "    if not name:\n"
        "        deny('Missing tool name in PreToolUse hook payload')\n"
        "        return\n"
        "    if name in ALLOWED_TOOLS:\n"
        "        allow()\n"
        "    if not ALLOWED_TOOLS:\n"
        "        deny(f'Tool {name!r} is forbidden in zero-tool sandbox')\n"
        "    deny(f'Tool {name!r} is forbidden in chat sandbox')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        main()\n"
        "    except SystemExit:\n"
        "        raise\n"
        "    except Exception:\n"
        "        deny('Internal error in PreToolUse firewall')\n"
    )
    hooks_json = _write_pre_tool_hooks_manifest(
        agents_dir,
        hook_name="bot-chat-firewall",
        gate_script=gate_script,
        gate_source=gate_source,
    )
    return hooks_json, gate_script


def _create_chat_sandbox_cwd(
    allowed_tools: frozenset[str] | Collection[str] | None,
) -> Path:
    """Create a dedicated 0700 per-turn sandbox cwd with fail-closed PreToolUse hooks."""
    sandbox_cwd = Path(tempfile.mkdtemp(prefix="bot_chat_")).resolve()
    try:
        try:
            os.chmod(sandbox_cwd, 0o700)
        except OSError:
            pass
        write_chat_pre_tool_hook(sandbox_cwd, allowed_tools=allowed_tools)
        return sandbox_cwd
    except Exception:
        shutil.rmtree(sandbox_cwd, ignore_errors=True)
        raise


def write_image_pre_tool_hook(
    request_cwd: Path,
    *,
    allow_staged_inputs: bool = False,
) -> tuple[Path, Path]:
    """Write fail-closed `.agents/hooks.json` + Python PreToolUse firewall inside request_cwd.

    The hook fires before every tool call in both the parent agent and the `image-generator`
    subagent, hard-blocking arbitrary file reads, MCP tools, shell execution, and any
    subagent other than `image-generator`.
    """
    resolved_cwd = request_cwd.resolve()
    inputs_root = (resolved_cwd / "inputs").resolve()
    agents_dir = resolved_cwd / ".agents"

    gate_script = agents_dir / "image_gate.py"
    gate_source = (
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "\n"
        f"INPUTS_ROOT = {str(inputs_root)!r}\n"
        f"ALLOW_STAGED_INPUTS = {bool(allow_staged_inputs)!r}\n"
        "\n"
        "def deny(reason: str) -> None:\n"
        "    sys.stdout.write(json.dumps({'decision': 'deny', 'reason': reason}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    sys.exit(0)\n"
        "\n"
        "def allow() -> None:\n"
        "    sys.stdout.write(json.dumps({'decision': 'allow'}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "    sys.exit(0)\n"
        "\n"
        "def _has_symlink_component(abs_path: str, stop_at: str) -> bool:\n"
        "    curr = abs_path\n"
        "    while True:\n"
        "        if os.path.islink(curr):\n"
        "            return True\n"
        "        if curr == stop_at:\n"
        "            break\n"
        "        parent = os.path.dirname(curr)\n"
        "        if parent == curr:\n"
        "            break\n"
        "        curr = parent\n"
        "    return False\n"
        "\n"
        "def is_safe_input_path(raw_path: object) -> bool:\n"
        "    if not ALLOW_STAGED_INPUTS or not isinstance(raw_path, str):\n"
        "        return False\n"
        "    candidate = raw_path.strip()\n"
        "    if not candidate or not os.path.isabs(candidate):\n"
        "        return False\n"
        "    abs_cand = os.path.abspath(candidate)\n"
        "    if os.path.islink(abs_cand):\n"
        "        return False\n"
        "    try:\n"
        "        real_root = os.path.realpath(INPUTS_ROOT)\n"
        "        real_cand = os.path.realpath(abs_cand)\n"
        "        if os.path.commonpath([real_root, real_cand]) != real_root:\n"
        "            return False\n"
        "        if real_cand == real_root:\n"
        "            return False\n"
        "        if os.path.commonpath([real_root, abs_cand]) == real_root:\n"
        "            if _has_symlink_component(abs_cand, real_root):\n"
        "                return False\n"
        "        elif os.path.commonpath([INPUTS_ROOT, abs_cand]) == INPUTS_ROOT:\n"
        "            if _has_symlink_component(abs_cand, INPUTS_ROOT):\n"
        "                return False\n"
        "        else:\n"
        "            return False\n"
        "        if os.path.isdir(real_cand):\n"
        "            return False\n"
        "        return True\n"
        "    except (OSError, ValueError):\n"
        "        return False\n"
        "\n"
        "def main() -> None:\n"
        "    try:\n"
        "        payload = json.load(sys.stdin)\n"
        "    except Exception:\n"
        "        deny('Malformed PreToolUse hook payload')\n"
        "        return\n"
        "    if not isinstance(payload, dict):\n"
        "        deny('Non-object PreToolUse hook payload')\n"
        "        return\n"
        "    tc = payload.get('toolCall')\n"
        "    if not isinstance(tc, dict):\n"
        "        deny('Missing toolCall object')\n"
        "        return\n"
        "    name = str(tc.get('name') or '').strip().lower()\n"
        "    args = tc.get('args')\n"
        "    if not isinstance(args, dict):\n"
        "        args = {}\n"
        "\n"
        "    if name == 'invoke_subagent':\n"
        "        raw_type = args.get('TypeName')\n"
        "        if raw_type is None:\n"
        "            raw_type = args.get('type_name')\n"
        "        if raw_type is None:\n"
        "            raw_type = args.get('typeName')\n"
        "        if isinstance(raw_type, str) and raw_type.strip().lower() == 'image-generator':\n"
        "            allow()\n"
        "        deny(f'Subagent TypeName {raw_type!r} is forbidden in image-only sandbox')\n"
        "\n"
        "    if name == 'generate_image':\n"
        "        raw_paths = args.get('ImagePaths')\n"
        "        if raw_paths is None:\n"
        "            raw_paths = args.get('image_paths')\n"
        "        if raw_paths is None:\n"
        "            raw_paths = args.get('imagePaths')\n"
        "        if raw_paths is None or raw_paths == [] or raw_paths == '':\n"
        "            allow()\n"
        "        if isinstance(raw_paths, str):\n"
        "            try:\n"
        "                parsed = json.loads(raw_paths)\n"
        "                raw_paths = parsed if isinstance(parsed, list) else [raw_paths]\n"
        "            except Exception:\n"
        "                raw_paths = [raw_paths]\n"
        "        if not isinstance(raw_paths, list):\n"
        "            deny('Invalid ImagePaths argument')\n"
        "        for p in raw_paths:\n"
        "            if not is_safe_input_path(p):\n"
        "                deny(f'ImagePaths entry outside staged inputs: {p!r}')\n"
        "        allow()\n"
        "\n"
        "    if name == 'read_file':\n"
        "        raw_path = args.get('Path')\n"
        "        if raw_path is None:\n"
        "            raw_path = args.get('path')\n"
        "        if is_safe_input_path(raw_path):\n"
        "            allow()\n"
        "        deny('read_file is forbidden outside staged inputs')\n"
        "\n"
        "    if name == 'send_message':\n"
        "        allow()\n"
        "\n"
        "    deny(f'Tool {name!r} is forbidden in image-only sandbox')\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        main()\n"
        "    except SystemExit:\n"
        "        raise\n"
        "    except Exception:\n"
        "        deny('Internal error in PreToolUse firewall')\n"
    )
    hooks_json = _write_pre_tool_hooks_manifest(
        agents_dir,
        hook_name="bot-image-firewall",
        gate_script=gate_script,
        gate_source=gate_source,
    )
    return hooks_json, gate_script


def _stage_reference_images(
    request_cwd: Path,
    reference_images: Sequence[MediaAttachment],
    *,
    max_bytes: int,
) -> list[Path]:
    """Validate and copy reference images as regular files into `<request_cwd>/inputs/`."""
    inputs_dir = request_cwd / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(inputs_dir, 0o700)
    except OSError:
        pass

    staged: list[Path] = []
    for idx, att in enumerate(reference_images):
        if att.file_path is not None:
            src = Path(att.file_path)
            if src.is_symlink() or not src.is_file():
                raise ImageValidationError(f"Reference image is not a regular file: {src.name}")
            try:
                src_size = src.stat().st_size
            except OSError as exc:
                raise ImageValidationError(f"Cannot stat reference image: {exc}") from exc
            if src_size <= 0 or src_size > max_bytes:
                raise ImageValidationError(
                    f"Reference image size {src_size} bytes is outside allowed limit ({max_bytes})."
                )
            raw_bytes = src.read_bytes()
        elif att.data is not None:
            raw_bytes = bytes(att.data)
        else:
            raise ImageValidationError("Reference image attachment has neither file_path nor data.")

        _, ext, _, _ = _validate_image_bytes_strict(raw_bytes, max_bytes=max_bytes)
        dest = (inputs_dir / f"ref_{idx}{ext}").resolve()
        if not dest.is_relative_to(inputs_dir.resolve()):
            raise ImageValidationError("Staged reference path escaped inputs directory.")
        dest.write_bytes(raw_bytes)
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
        staged.append(dest)
    return staged


def _discover_brain_image(
    gemini_dir: Path,
    conversation_id: str | None,
    *,
    turn_start_wall: float,
    response_text: str,
) -> tuple[Path | None, list[Path], bool]:
    """Locate newly generated image artifact strictly inside `<gemini_dir>/brain/<conversation_id>`.

    Returns `(chosen_path, all_safe_candidates, saw_unsafe_candidate)`.
    """
    if not conversation_id or not isinstance(conversation_id, str):
        return None, [], False
    cid = conversation_id.strip()
    if not cid or cid in (".", "..") or not _SAFE_CONV_ID_RE.fullmatch(cid):
        return None, [], True

    brain_root = (gemini_dir / "brain").resolve()
    conv_dir = (brain_root / cid).resolve()
    try:
        if not conv_dir.is_relative_to(brain_root) or conv_dir == brain_root:
            return None, [], True
    except ValueError:
        return None, [], True
    if not conv_dir.is_dir():
        return None, [], False

    safe_candidates: list[tuple[Path, float]] = []
    saw_unsafe = False

    try:
        entries = list(conv_dir.iterdir())
    except OSError:
        return None, [], False

    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in _SAFE_IMAGE_EXTS:
            continue
        if entry.is_symlink():
            saw_unsafe = True
            continue
        try:
            resolved = entry.resolve(strict=True)
        except OSError:
            saw_unsafe = True
            continue
        try:
            if not resolved.is_relative_to(conv_dir) or not resolved.is_file():
                saw_unsafe = True
                continue
        except ValueError:
            saw_unsafe = True
            continue
        try:
            st = resolved.stat()
        except OSError:
            continue
        if st.st_mtime >= turn_start_wall - 1.0:
            safe_candidates.append((resolved, st.st_mtime))

    if not safe_candidates:
        return None, [], saw_unsafe

    safe_candidates.sort(key=lambda item: item[1], reverse=True)
    all_safe_paths = [p for p, _ in safe_candidates]

    if response_text:
        for cand_path, _ in safe_candidates:
            if cand_path.name in response_text:
                return cand_path, all_safe_paths, saw_unsafe

    return safe_candidates[0][0], all_safe_paths, saw_unsafe


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
        turn_timeout_seconds: float = 300.0,
        cleanup_interval_seconds: float = 3600.0,
        on_quota_warning: Callable[[str | None, float | None, str | None], None] | None = None,
    ) -> None:
        self.data_dir = resolve_antigravity_home(data_dir)
        self.binary = binary or resolve_antigravity_binary()
        self.default_model = default_model
        self.default_effort = default_effort
        self.worker_idle_timeout_seconds = max(0.0, float(worker_idle_timeout_seconds))
        self.turn_timeout_seconds = max(0.0, float(turn_timeout_seconds))
        self.cleanup_interval_seconds = max(0.0, float(cleanup_interval_seconds))
        self.on_quota_warning = on_quota_warning

        self._worker_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._worker_proc: subprocess.Popen[str] | None = None
        self._worker_stderr: StderrDrainer | None = None
        self._worker_model: str | None = None
        self._worker_effort: str | None = None
        self._worker_mode: str | None = None
        self._worker_cwd: str | None = None
        self._worker_add_dirs: tuple[str, ...] = ()
        self._worker_tool_profile: ToolProfile = "chat"
        self._worker_allowed_tools: frozenset[str] | None = None
        self._worker_sandbox_dir: Path | None = None
        self._worker_gemini_dir: Path | None = None
        self._worker_last_used: float = 0.0
        self._worker_busy: bool = False
        self._idle_timer: threading.Timer | None = None
        self._active_conversation_ids: set[str] = set()
        self._last_cleanup_monotonic: float | None = None
        self._cleanup_in_progress: bool = False
        self._cleanup_thread: threading.Thread | None = None

    def set_quota_warning_callback(
        self,
        callback: Callable[[str | None, float | None, str | None], None] | None,
    ) -> None:
        """Register or replace the soft `[quota]` warning callback for log watchdogs."""
        self.on_quota_warning = callback

    def _register_active_conversation(self, conversation_id: str | None) -> None:
        cid = (conversation_id or "").strip() if isinstance(conversation_id, str) else ""
        if not cid:
            return
        with self._state_lock:
            self._active_conversation_ids.add(cid)

    def _unregister_active_conversation(self, conversation_id: str | None) -> None:
        cid = (conversation_id or "").strip() if isinstance(conversation_id, str) else ""
        if not cid:
            return
        with self._state_lock:
            self._active_conversation_ids.discard(cid)

    def _maybe_schedule_session_cleanup(self, active_conversation_id: str | None = None) -> None:
        """Trigger non-blocking background session cleanup at most once per cleanup_interval_seconds."""
        if self.cleanup_interval_seconds <= 0.0:
            return
        now_mono = time.monotonic()
        with self._state_lock:
            if self._cleanup_in_progress:
                return
            if (
                self._last_cleanup_monotonic is not None
                and (now_mono - self._last_cleanup_monotonic) < self.cleanup_interval_seconds
            ):
                return
            self._last_cleanup_monotonic = now_mono
            self._cleanup_in_progress = True
            active_snapshot = set(self._active_conversation_ids)
            if isinstance(active_conversation_id, str) and active_conversation_id.strip():
                active_snapshot.add(active_conversation_id.strip())

        def _run_cleanup() -> None:
            try:
                cleanup_old_antigravity_sessions(
                    self.data_dir,
                    active_conversation_ids=active_snapshot,
                )
            except Exception:
                logger.debug("Background Antigravity session cleanup failed", exc_info=True)
            finally:
                with self._state_lock:
                    self._cleanup_in_progress = False

        thread = threading.Thread(
            target=_run_cleanup,
            name="agy-session-cleanup",
            daemon=True,
        )
        with self._state_lock:
            self._cleanup_thread = thread
        thread.start()

    def close(self) -> None:
        """Kill any persistent worker, cancel idle reaper, wait for cleanup thread, and sync tokens."""
        self._kill_worker()
        with self._state_lock:
            cleanup_thread = self._cleanup_thread
        if (
            cleanup_thread is not None
            and cleanup_thread.is_alive()
            and cleanup_thread is not threading.current_thread()
        ):
            try:
                cleanup_thread.join(timeout=1.0)
            except RuntimeError:
                pass

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
            drainer = self._worker_stderr
            g_dir = self._worker_gemini_dir
            worker_sandbox = self._worker_sandbox_dir
            self._worker_proc = None
            self._worker_stderr = None
            self._worker_model = None
            self._worker_effort = None
            self._worker_mode = None
            self._worker_cwd = None
            self._worker_add_dirs = ()
            self._worker_tool_profile = "chat"
            self._worker_allowed_tools = None
            self._worker_sandbox_dir = None
            self._worker_gemini_dir = None
            self._worker_busy = False

        if proc is not None:
            terminate_process_group(proc)
            _close_proc_pipes(proc)
        if drainer is not None:
            drainer.join(timeout=1.0)
        if worker_sandbox is not None:
            shutil.rmtree(worker_sandbox, ignore_errors=True)
        if g_dir is not None:
            sync_updated_isolated_tokens(g_dir, data_dir=self.data_dir)

    def _ensure_worker(
        self,
        model: str,
        effort: str | None,
        mode: str | None = None,
        cwd: Path | str | None = None,
        add_dirs: Sequence[Path | str] | None = None,
        tool_profile: ToolProfile = "chat",
        allowed_tools: frozenset[str] | Collection[str] | None = None,
    ) -> tuple[subprocess.Popen[str], StderrDrainer, Path]:
        resolved_allowed = _resolve_effective_allowed_tools(tool_profile, allowed_tools)
        needs_sandbox = _needs_isolated_chat_sandbox(tool_profile, allowed_tools, resolved_allowed)
        zero_tools = allowed_tools is not None and len(resolved_allowed) == 0
        norm_cwd = None if needs_sandbox else (str(Path(cwd).resolve()) if cwd is not None else None)
        norm_mode = mode.strip() if isinstance(mode, str) and mode.strip() else None
        effective_preview_cwd = Path(norm_cwd) if norm_cwd is not None else self.data_dir
        norm_extra_dirs = (
            ()
            if needs_sandbox
            else tuple(_resolve_extra_add_dirs(effective_preview_cwd, add_dirs))
        )
        with self._state_lock:
            self._cancel_idle_timer_locked()
            proc = self._worker_proc
            drainer = self._worker_stderr
            g_dir = self._worker_gemini_dir
            sandbox_ok = (not needs_sandbox) or (
                self._worker_sandbox_dir is not None and self._worker_sandbox_dir.is_dir()
            )
            same_cfg = (
                self._worker_model == model
                and self._worker_effort == effort
                and self._worker_mode == norm_mode
                and self._worker_cwd == norm_cwd
                and self._worker_add_dirs == norm_extra_dirs
                and self._worker_tool_profile == tool_profile
                and self._worker_allowed_tools == resolved_allowed
                and sandbox_ok
            )

        if proc is not None and drainer is not None and g_dir is not None and same_cfg and proc.poll() is None:
            with self._state_lock:
                self._worker_last_used = time.monotonic()
            return proc, drainer, g_dir

        self._kill_worker()

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)
        worker_sandbox: Path | None = None
        if needs_sandbox:
            worker_sandbox = _create_chat_sandbox_cwd(resolved_allowed)
            effective_cwd = worker_sandbox
            extra_add_dirs: list[str] = []
        else:
            effective_cwd = Path(norm_cwd) if norm_cwd is not None else isolated_home
            extra_add_dirs = _resolve_extra_add_dirs(effective_cwd, add_dirs)

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
        _append_cli_mode_flags(cmd, norm_mode, tool_profile=tool_profile, zero_tools=zero_tools)
        for extra_dir in extra_add_dirs:
            cmd.extend(["--add-dir", extra_dir])

        try:
            new_proc = popen_isolated_session(cmd, cwd=effective_cwd, env=clean_env)
        except Exception:
            if worker_sandbox is not None:
                shutil.rmtree(worker_sandbox, ignore_errors=True)
            raise
        new_drainer = StderrDrainer(new_proc.stderr)

        with self._state_lock:
            self._worker_proc = new_proc
            self._worker_stderr = new_drainer
            self._worker_model = model
            self._worker_effort = effort
            self._worker_mode = norm_mode
            self._worker_cwd = norm_cwd
            self._worker_add_dirs = tuple(extra_add_dirs)
            self._worker_tool_profile = tool_profile
            self._worker_allowed_tools = resolved_allowed
            self._worker_sandbox_dir = worker_sandbox
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
        mode: str | None = None,
        cwd: Path | str | None = None,
        add_dirs: Sequence[Path | str] | None = None,
        tool_profile: ToolProfile = "chat",
        allowed_tools: frozenset[str] | Collection[str] | None = None,
    ) -> tuple[subprocess.Popen[str], StderrDrainer, Path]:
        resolved_allowed = _resolve_effective_allowed_tools(tool_profile, allowed_tools)
        needs_sandbox = _needs_isolated_chat_sandbox(tool_profile, allowed_tools, resolved_allowed)
        zero_tools = allowed_tools is not None and len(resolved_allowed) == 0

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)
        norm_cwd = str(Path(cwd).resolve()) if cwd is not None else None
        effective_cwd = Path(norm_cwd) if norm_cwd is not None else isolated_home
        norm_mode = mode.strip() if isinstance(mode, str) and mode.strip() else None

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
        _append_cli_mode_flags(cmd, norm_mode, tool_profile=tool_profile, zero_tools=zero_tools)
        if not needs_sandbox:
            for extra_dir in _resolve_extra_add_dirs(effective_cwd, add_dirs):
                cmd.extend(["--add-dir", extra_dir])
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])

        proc = popen_isolated_session(cmd, cwd=effective_cwd, env=clean_env)
        drainer = StderrDrainer(proc.stderr)

        payload = json.dumps({"event": "user", "message": {"content": prompt}})
        if proc.stdin is not None:
            try:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
            except OSError as exc:
                terminate_process_group(proc)
                _close_proc_pipes(proc)
                drainer.join(timeout=1.0)
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
        tool_profile: ToolProfile = "chat",
        mode: str | None = None,
        cwd: Path | str | None = None,
        add_dirs: Sequence[Path | str] | None = None,
        timeout_seconds: float | None = None,
        allowed_tools: frozenset[str] | Collection[str] | None = None,
    ) -> ProviderResponse:
        """Execute a multimodal turn via `agy acp --stdio` with native image blocks."""
        resolved_allowed = _resolve_effective_allowed_tools(tool_profile, allowed_tools)
        needs_sandbox = _needs_isolated_chat_sandbox(tool_profile, allowed_tools, resolved_allowed)
        zero_tools = allowed_tools is not None and len(resolved_allowed) == 0
        resolved_model = model or self.default_model
        resolved_effort = effort if effort is not None else self.default_effort
        norm_cwd = str(Path(cwd).resolve()) if cwd is not None else None
        norm_mode = mode.strip() if isinstance(mode, str) and mode.strip() else None
        effective_timeout = (
            self.turn_timeout_seconds
            if timeout_seconds is None
            else max(0.0, float(timeout_seconds))
        )

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        isolated_home, gemini_dir = setup_isolated_home(self.data_dir)
        clean_env = build_clean_child_env(isolated_home, force_file_auth=True)
        sandbox_cwd: Path | None = None
        proc: subprocess.Popen[str] | None = None
        drainer: StderrDrainer | None = None
        deadline_timer: threading.Timer | None = None
        stop_watchdog = threading.Event()
        watchdog_thread: threading.Thread | None = None
        session_id: str | None = conversation_id
        registered_cids: set[str] = set()
        if conversation_id:
            self._register_active_conversation(conversation_id)
            registered_cids.add(conversation_id)

        try:
            if needs_sandbox:
                sandbox_cwd = _create_chat_sandbox_cwd(resolved_allowed)
                effective_cwd = sandbox_cwd
                effective_add_dirs: Sequence[Path | str] = ()
            else:
                effective_cwd = Path(norm_cwd) if norm_cwd is not None else isolated_home
                effective_add_dirs = add_dirs or ()

            cmd = [self.binary, "acp", "--stdio"]
            for extra_dir in _resolve_extra_add_dirs(effective_cwd, effective_add_dirs):
                cmd.extend(["--add-dir", extra_dir])

            log_offsets = _snapshot_log_offsets(gemini_dir)
            proc = popen_isolated_session(cmd, cwd=effective_cwd, env=clean_env)
            drainer = StderrDrainer(proc.stderr)

            timed_out = threading.Event()
            if effective_timeout > 0.0:
                def _on_acp_timeout() -> None:
                    timed_out.set()
                    terminate_process_group(proc, timeout=1.0)

                deadline_timer = threading.Timer(effective_timeout, _on_acp_timeout)
                deadline_timer.daemon = True
                deadline_timer.name = "agy-acp-timeout"
                deadline_timer.start()

            watchdog_errors: list[str] = []
            watchdog_thread = threading.Thread(
                target=watch_log_for_errors,
                args=(gemini_dir, stop_watchdog, watchdog_errors, lambda: terminate_process_group(proc)),
                kwargs={
                    "on_quota_warning": self.on_quota_warning,
                    "initial_offsets": log_offsets,
                },
                name="agy-acp-watchdog",
                daemon=True,
            )
            watchdog_thread.start()

            def _send_rpc(msg_id: int | str | None, method: str | None, params_or_result: dict[str, Any], *, is_response: bool = False) -> None:
                if proc is None or proc.stdin is None:
                    raise RuntimeError("ACP stdin pipe is unavailable.")
                if is_response:
                    frame: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "result": params_or_result}
                else:
                    frame = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params_or_result}
                proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
                proc.stdin.flush()

            acp_tool_names: dict[str, str] = {}

            def _extract_acp_tool_name_info(obj: dict[str, Any], *, tool_call_id: str = "") -> tuple[str, bool]:
                nested_call = obj.get("toolCall") if isinstance(obj.get("toolCall"), dict) else {}
                nested_info = (
                    obj.get("toolInfo")
                    if isinstance(obj.get("toolInfo"), dict)
                    else (obj.get("tool_info") if isinstance(obj.get("tool_info"), dict) else {})
                )
                candidates = (
                    obj.get("name"),
                    obj.get("toolName"),
                    obj.get("tool_name"),
                    nested_call.get("name"),
                    nested_call.get("toolName"),
                    nested_call.get("tool_name"),
                    nested_info.get("name"),
                    nested_info.get("toolName"),
                    nested_info.get("tool_name"),
                )
                for cand in candidates:
                    if isinstance(cand, str) and cand.strip():
                        resolved = cand.strip().lower()
                        if tool_call_id:
                            acp_tool_names[tool_call_id] = resolved
                        return resolved, True
                if tool_call_id and tool_call_id in acp_tool_names:
                    return acp_tool_names[tool_call_id], True
                title_cand = obj.get("title") or nested_call.get("title")
                if isinstance(title_cand, str) and title_cand.strip():
                    return title_cand.strip().lower(), False
                return "", False

            def _extract_acp_tool_name(obj: dict[str, Any], *, tool_call_id: str = "") -> str:
                name, _ = _extract_acp_tool_name_info(obj, tool_call_id=tool_call_id)
                return name

            def _check_acp_tool_allowed(update: dict[str, Any], *, upd_type: str = "") -> None:
                nested_call = update.get("toolCall") if isinstance(update.get("toolCall"), dict) else {}
                tool_call_id = str(
                    update.get("toolCallId") or nested_call.get("toolCallId") or ""
                ).strip()
                raw_name, is_explicit = _extract_acp_tool_name_info(update, tool_call_id=tool_call_id)
                if not raw_name or raw_name in ALWAYS_FORBIDDEN_TOOLS or raw_name not in resolved_allowed:
                    terminate_process_group(proc)
                    # Never expose descriptive ACP `title` fallback strings in diagnostics; use safe keys instead.
                    diag_tool_name = raw_name if is_explicit else ""
                    diag_step_type = upd_type or str(update.get("sessionUpdate") or "").strip().lower()
                    diag_keys = sorted(str(k) for k in update.keys()) if not diag_tool_name else None
                    raise AntigravityNativeToolAbort.from_context(
                        tool_profile=tool_profile,
                        source="acp",
                        step_type=diag_step_type,
                        tool_name=diag_tool_name,
                        payload_keys=diag_keys,
                    )

            text_parts: list[str] = []
            thinking_parts: list[str] = []
            finish_reason = "stop"
            usage: UsageInfo | None = None

            def _wait_rpc_response(expected_id: int) -> dict[str, Any]:
                if proc is None or proc.stdout is None:
                    raise RuntimeError("ACP stdout pipe is unavailable.")
                for line in iter(proc.stdout.readline, ""):
                    if timed_out.is_set():
                        raise TimeoutError(
                            f"Antigravity ACP turn timed out after {effective_timeout:.1f}s"
                        )
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
                        tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
                        tool_call_id = str(
                            tool_call.get("toolCallId") or params.get("toolCallId") or ""
                        ).strip()
                        merged_for_lookup: dict[str, Any] = {
                            **params,
                            **tool_call,
                            "toolCall": tool_call,
                        }
                        tool_name = _extract_acp_tool_name(
                            merged_for_lookup,
                            tool_call_id=tool_call_id,
                        )
                        options = params.get("options") or []
                        chosen_option = None
                        can_auto_approve = (
                            tool_profile in ("ops", "self_dev")
                            and not zero_tools
                            and bool(tool_name)
                            and tool_name not in ALWAYS_FORBIDDEN_TOOLS
                            and tool_name in resolved_allowed
                        )
                        if can_auto_approve:
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
                        elif upd_type in ("tool_call", "tool_call_update"):
                            _check_acp_tool_allowed(update, upd_type=upd_type)
                        continue

                    if msg.get("id") == expected_id:
                        if "error" in msg and msg["error"]:
                            err = msg["error"]
                            err_msg = err.get("message") if isinstance(err, dict) else str(err)
                            raise RuntimeError(f"Antigravity ACP error: {err_msg}")
                        return msg.get("result") or {}

                if timed_out.is_set():
                    raise TimeoutError(
                        f"Antigravity ACP turn timed out after {effective_timeout:.1f}s"
                    )
                rc = proc.poll()
                stderr_out = drainer.get_output().strip() if drainer is not None else ""
                raise RuntimeError(f"agy acp exited unexpectedly (rc={rc}): {stderr_out}")

            rpc_seq = 0

            def _next_rpc_id() -> int:
                nonlocal rpc_seq
                rpc_seq += 1
                return rpc_seq

            init_id = _next_rpc_id()
            _send_rpc(init_id, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
            _wait_rpc_response(init_id)

            new_params: dict[str, Any] = {
                "cwd": str(effective_cwd),
                "mcpServers": [],
            }
            if system_instructions:
                new_params["_meta"] = {
                    "antigravity": {"systemPrompt": {"append": system_instructions}}
                }
            new_id = _next_rpc_id()
            _send_rpc(new_id, "session/new", new_params)
            new_res = _wait_rpc_response(new_id)
            session_id = str(new_res.get("sessionId") or "") or session_id
            if not session_id:
                raise RuntimeError("agy acp session/new did not return a sessionId.")
            self._register_active_conversation(session_id)
            registered_cids.add(session_id)

            models_info = new_res.get("models") or {}
            available_models = models_info.get("availableModels") or []
            target_model_id = _select_acp_model(available_models, resolved_model, resolved_effort)
            current_model_id = models_info.get("currentModelId")
            if target_model_id and target_model_id != current_model_id:
                set_model_id = _next_rpc_id()
                _send_rpc(
                    set_model_id,
                    "session/set_model",
                    {"sessionId": session_id, "modelId": target_model_id},
                )
                _wait_rpc_response(set_model_id)

            modes_info = new_res.get("modes") or {}
            current_mode_id = modes_info.get("currentModeId")
            available_modes = modes_info.get("availableModes") or []
            available_mode_ids = {
                str(m.get("id") or m.get("modeId") or "").strip()
                for m in available_modes
                if isinstance(m, dict)
            }
            acp_mode = None if zero_tools else norm_mode
            if acp_mode == "auto" and "auto" not in available_mode_ids:
                acp_mode = "accept-edits"
            if acp_mode and acp_mode != current_mode_id:
                set_mode_id = _next_rpc_id()
                _send_rpc(
                    set_mode_id,
                    "session/set_mode",
                    {"sessionId": session_id, "modeId": acp_mode},
                )
                _wait_rpc_response(set_mode_id)

            prompt_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for b64_data, mime_type in image_payloads:
                prompt_blocks.append(
                    {"type": "image", "data": b64_data, "mimeType": mime_type}
                )

            prompt_id = _next_rpc_id()
            _send_rpc(
                prompt_id,
                "session/prompt",
                {"sessionId": session_id, "prompt": prompt_blocks},
            )
            prompt_res = _wait_rpc_response(prompt_id)
            finish_reason = str(prompt_res.get("stopReason") or "stop")
            raw_usage = prompt_res.get("usage")
            if isinstance(raw_usage, dict):
                usage = UsageInfo(
                    prompt_tokens=int(raw_usage.get("inputTokens") or 0),
                    completion_tokens=int(raw_usage.get("outputTokens") or 0),
                    total_tokens=int(raw_usage.get("totalTokens") or 0),
                    reasoning_tokens=int(raw_usage.get("thoughtTokens") or 0),
                    cached_tokens=int(raw_usage.get("cachedReadTokens") or 0),
                    input_tokens=int(raw_usage.get("inputTokens") or 0),
                    output_tokens=int(raw_usage.get("outputTokens") or 0),
                    cache_read_tokens=int(raw_usage.get("cachedReadTokens") or 0),
                    cache_write_tokens=int(raw_usage.get("cachedWriteTokens") or 0),
                )

            if watchdog_errors:
                raise AntigravityRateLimitError(
                    f"Rate limit detected in logs: {watchdog_errors[0]}"
                )
        except AntigravityNativeToolAbort:
            if proc is not None:
                terminate_process_group(proc)
            raise
        except Exception as exc:
            if timed_out.is_set() and not isinstance(exc, TimeoutError):
                exc = TimeoutError(
                    f"Antigravity ACP turn timed out after {effective_timeout:.1f}s"
                )
            verdict = classify_error(exc)
            if verdict.category == ErrorCategory.RATE_LIMIT:
                raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
            if verdict.category == ErrorCategory.BILLING:
                raise AntigravityBillingError(str(exc), verdict=verdict) from exc
            raise AntigravityError(str(exc), verdict=verdict) from exc
        finally:
            if deadline_timer is not None:
                try:
                    deadline_timer.cancel()
                except Exception:
                    pass
            stop_watchdog.set()
            if proc is not None:
                terminate_process_group(proc)
            if watchdog_thread is not None:
                watchdog_thread.join(timeout=1.0)
            _close_proc_pipes(proc)
            if drainer is not None:
                drainer.join(timeout=1.0)
            if sandbox_cwd is not None:
                shutil.rmtree(sandbox_cwd, ignore_errors=True)
            sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)
            for cid in registered_cids:
                self._unregister_active_conversation(cid)
            self._maybe_schedule_session_cleanup(active_conversation_id=session_id)

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
        tool_profile: ToolProfile = "chat",
        mode: str | None = None,
        cwd: Path | str | None = None,
        add_dirs: Sequence[Path | str] | None = None,
        image_payloads: Sequence[tuple[str, str]] | None = None,
        timeout_seconds: float | None = None,
        allowed_tools: frozenset[str] | Collection[str] | None = None,
    ) -> ProviderResponse:
        resolved_allowed = _resolve_effective_allowed_tools(tool_profile, allowed_tools)
        needs_sandbox = _needs_isolated_chat_sandbox(tool_profile, allowed_tools, resolved_allowed)
        resolved_model = model or self.default_model
        resolved_effort = effort if effort is not None else self.default_effort
        effective_timeout = (
            self.turn_timeout_seconds
            if timeout_seconds is None
            else max(0.0, float(timeout_seconds))
        )

        resolved_images: list[tuple[str, str]] = list(image_payloads) if image_payloads else []
        if media_attachments:
            for att in media_attachments:
                normalized = _normalize_image_attachment(att)
                if normalized is not None:
                    resolved_images.append(normalized)
        image_payloads = resolved_images

        if image_payloads:
            return self._sync_execute_acp_turn(
                prompt=prompt,
                image_payloads=image_payloads,
                system_instructions=system_instructions,
                conversation_id=conversation_id,
                model=resolved_model,
                effort=resolved_effort,
                tool_profile=tool_profile,
                mode=mode,
                cwd=cwd,
                add_dirs=add_dirs,
                timeout_seconds=timeout_seconds,
                allowed_tools=allowed_tools,
            )

        full_prompt = prompt
        if system_instructions:
            full_prompt = f"{system_instructions}\n\n{prompt}"

        pre_spawn_gemini_dir = self.data_dir / ".gemini" / "antigravity-cli"
        log_offsets = _snapshot_log_offsets(pre_spawn_gemini_dir)

        sandbox_cwd: Path | None = None
        try:
            if needs_sandbox:
                sandbox_cwd = _create_chat_sandbox_cwd(resolved_allowed)
                effective_cwd: Path | str | None = sandbox_cwd
                effective_add_dirs: Sequence[Path | str] | None = ()
            else:
                effective_cwd = cwd
                effective_add_dirs = add_dirs

            # Stateless full-context turns (conversation_id is None) always run in a fresh
            # isolated cascade so trajectory tokens never accumulate O(N^2) across turns
            # and idle workers do not remain alive polling fetchAvailableModels/loadCodeAssist.
            proc, drainer, gemini_dir = self._spawn_oneshot(
                prompt=full_prompt,
                model=resolved_model,
                effort=resolved_effort,
                conversation_id=conversation_id,
                mode=mode,
                cwd=effective_cwd,
                add_dirs=effective_add_dirs,
                tool_profile=tool_profile,
                allowed_tools=allowed_tools,
            )

            timed_out = threading.Event()
            deadline_timer: threading.Timer | None = None
            if effective_timeout > 0.0:
                def _on_turn_timeout() -> None:
                    timed_out.set()
                    terminate_process_group(proc, timeout=1.0)

                deadline_timer = threading.Timer(effective_timeout, _on_turn_timeout)
                deadline_timer.daemon = True
                deadline_timer.name = "agy-cli-timeout"
                deadline_timer.start()

            # Error watchdog
            watchdog_errors: list[str] = []
            stop_watchdog = threading.Event()
            watchdog_thread = threading.Thread(
                target=watch_log_for_errors,
                args=(gemini_dir, stop_watchdog, watchdog_errors, lambda: terminate_process_group(proc)),
                kwargs={
                    "on_quota_warning": self.on_quota_warning,
                    "initial_offsets": log_offsets,
                },
                name="agy-cli-watchdog",
                daemon=True,
            )
            watchdog_thread.start()

            text_parts: list[str] = []
            thinking_parts: list[str] = []
            extracted_conv_id = conversation_id
            finish_reason = "stop"
            usage: UsageInfo | None = None
            registered_cids: set[str] = set()
            if conversation_id:
                self._register_active_conversation(conversation_id)
                registered_cids.add(conversation_id)

            def on_conv(cid: str) -> None:
                nonlocal extracted_conv_id
                extracted_conv_id = cid
                self._register_active_conversation(cid)
                registered_cids.add(cid)

            parser = AntigravityStreamParser(
                on_conversation_id=on_conv,
                on_native_tool_intercept=lambda: terminate_process_group(proc),
                allowed_tools=resolved_allowed,
                tool_profile=tool_profile,
            )

            try:
                if proc.stdout is not None:
                    for line in iter(proc.stdout.readline, ""):
                        if timed_out.is_set():
                            raise TimeoutError(
                                f"Antigravity CLI turn timed out after {effective_timeout:.1f}s"
                            )
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

                if timed_out.is_set():
                    raise TimeoutError(
                        f"Antigravity CLI turn timed out after {effective_timeout:.1f}s"
                    )

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
                if timed_out.is_set() and not isinstance(exc, TimeoutError):
                    exc = TimeoutError(
                        f"Antigravity CLI turn timed out after {effective_timeout:.1f}s"
                    )
                verdict = classify_error(exc)
                if verdict.category == ErrorCategory.RATE_LIMIT:
                    raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
                if verdict.category == ErrorCategory.BILLING:
                    raise AntigravityBillingError(str(exc), verdict=verdict) from exc
                raise AntigravityError(str(exc), verdict=verdict) from exc
            finally:
                if deadline_timer is not None:
                    try:
                        deadline_timer.cancel()
                    except Exception:
                        pass
                stop_watchdog.set()
                terminate_process_group(proc)
                watchdog_thread.join(timeout=1.0)
                _close_proc_pipes(proc)
                drainer.join(timeout=1.0)
                sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)
                for cid in registered_cids:
                    self._unregister_active_conversation(cid)
                self._maybe_schedule_session_cleanup(active_conversation_id=extracted_conv_id)

            return ProviderResponse(
                text="".join(text_parts),
                thinking="".join(thinking_parts) or None,
                conversation_id=extracted_conv_id,
                finish_reason=finish_reason,
                usage=usage,
                model=resolved_model,
            )
        finally:
            if sandbox_cwd is not None:
                shutil.rmtree(sandbox_cwd, ignore_errors=True)

    def _sync_execute_image_turn(
        self,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        reference_images: list[MediaAttachment] | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        output_dir: str | Path = "data/tmp/images",
        max_image_size_mb: float = 20.0,
        cleanup_artifacts: bool = True,
    ) -> GeneratedImageResult:
        """Execute a confined native image generation turn via `stream-json` oneshot."""
        clean_prompt = (prompt or "").strip()
        if not clean_prompt:
            raise ImageValidationError("Image prompt must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_image_size_mb <= 0:
            raise ValueError("max_image_size_mb must be positive.")

        timeout_s = float(timeout_seconds)
        max_bytes = max(1, int(float(max_image_size_mb) * 1024 * 1024))
        resolved_model = (model or "").strip() or self.default_model
        norm_ratio = aspect_ratio.strip() if isinstance(aspect_ratio, str) and aspect_ratio.strip() else None

        if not self.binary:
            raise FileNotFoundError("The 'agy' binary was not found.")

        base_out = Path(output_dir).expanduser().resolve()
        base_out.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(base_out, 0o700)
        except OSError:
            pass

        request_cwd = Path(tempfile.mkdtemp(prefix="bot_img_", dir=str(base_out))).resolve()
        try:
            os.chmod(request_cwd, 0o700)
        except OSError:
            pass

        proc: subprocess.Popen[str] | None = None
        drainer: StderrDrainer | None = None
        gemini_dir: Path | None = None
        stop_watchdog = threading.Event()
        watchdog_thread: threading.Thread | None = None
        deadline_timer: threading.Timer | None = None
        timed_out = threading.Event()
        discovered_candidates: list[Path] = []
        extracted_conv_id: str | None = None
        registered_cids: set[str] = set()

        turn_start_wall = time.time()
        turn_start_mono = time.monotonic()

        try:
            staged_inputs: list[Path] = []
            if reference_images:
                staged_inputs = _stage_reference_images(
                    request_cwd,
                    reference_images,
                    max_bytes=max_bytes,
                )

            write_image_pre_tool_hook(
                request_cwd,
                allow_staged_inputs=bool(staged_inputs),
            )

            prompt_lines = [
                'Generate an image using your native image-generator subagent (`invoke_subagent` with `TypeName="image-generator"`).',
                "Do NOT run shell commands, do NOT write/edit files, do NOT browse the web, and do NOT invoke any subagent other than `image-generator`.",
                f"Image description: {clean_prompt}",
            ]
            if norm_ratio:
                prompt_lines.append(f"Requested aspect ratio: {norm_ratio}")
            if staged_inputs:
                joined_inputs = ", ".join(str(p) for p in staged_inputs)
                prompt_lines.append(
                    f"Use these staged reference image file(s) as ImagePaths input for image-generator: {joined_inputs}"
                )
            prompt_lines.append("Reply briefly with the generated image filename once complete.")
            full_prompt = "\n".join(prompt_lines)

            log_offsets = _snapshot_log_offsets(self.data_dir / ".gemini" / "antigravity-cli")
            proc, drainer, gemini_dir = self._spawn_oneshot(
                prompt=full_prompt,
                model=resolved_model,
                effort="low",
                conversation_id=None,
                mode=None,
                cwd=request_cwd,
                add_dirs=(),
                tool_profile="image",
            )

            def _on_timeout() -> None:
                timed_out.set()
                terminate_process_group(proc, timeout=1.0)

            deadline_timer = threading.Timer(timeout_s, _on_timeout)
            deadline_timer.daemon = True
            deadline_timer.name = "agy-image-timeout"
            deadline_timer.start()

            watchdog_errors: list[str] = []
            watchdog_thread = threading.Thread(
                target=watch_log_for_errors,
                args=(gemini_dir, stop_watchdog, watchdog_errors, lambda: terminate_process_group(proc)),
                kwargs={
                    "on_quota_warning": self.on_quota_warning,
                    "initial_offsets": log_offsets,
                },
                name="agy-image-watchdog",
                daemon=True,
            )
            watchdog_thread.start()

            text_parts: list[str] = []
            usage: UsageInfo | None = None

            def on_conv(cid: str) -> None:
                nonlocal extracted_conv_id
                extracted_conv_id = cid
                self._register_active_conversation(cid)
                registered_cids.add(cid)

            parser = AntigravityStreamParser(
                on_conversation_id=on_conv,
                on_native_tool_intercept=lambda: terminate_process_group(proc),
                tool_profile="image",
            )

            deadline_mono = turn_start_mono + timeout_s
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, ""):
                    if timed_out.is_set() or time.monotonic() > deadline_mono:
                        timed_out.set()
                        terminate_process_group(proc, timeout=1.0)
                        break
                    if not line:
                        break
                    if watchdog_errors:
                        raise AntigravityRateLimitError(
                            f"Rate limit detected in logs: {watchdog_errors[0]}"
                        )
                    for chunk in parser.feed_line(line):
                        if chunk.text:
                            text_parts.append(chunk.text)
                    if parser.saw_result_event:
                        break

            if timed_out.is_set():
                raise ImageTimeoutError(
                    f"Antigravity image generation timed out after {timeout_s:.1f}s"
                )

            for chunk in parser.finish():
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.is_final:
                    usage = chunk.usage

            if watchdog_errors:
                raise AntigravityRateLimitError(
                    f"Rate limit detected in logs: {watchdog_errors[0]}"
                )

            response_text = "".join(text_parts).strip()
            if not parser.saw_result_event and not response_text:
                rc = proc.poll()
                stderr_out = drainer.get_output().strip()
                if rc not in (None, 0) and stderr_out:
                    raise RuntimeError(f"agy image generation exited with code {rc}: {stderr_out}")

            chosen_path, discovered_candidates, saw_unsafe = _discover_brain_image(
                gemini_dir,
                extracted_conv_id or parser.conversation_id,
                turn_start_wall=turn_start_wall,
                response_text=response_text,
            )

            if chosen_path is None:
                if saw_unsafe:
                    raise ImageValidationError(
                        "Generated image artifact failed confinement check (symlink or path escape rejected)."
                    )
                if response_text and _IMAGE_POLICY_REFUSAL_RE.search(response_text):
                    raise ImagePolicyRefusalError(
                        f"Image generation refused by policy: {response_text[:300]}"
                    )
                raise ImageGenerationError(
                    f"No image artifact was generated by Antigravity. Response: {response_text[:300]}"
                )

            try:
                file_size = chosen_path.stat().st_size
            except OSError as exc:
                raise ImageValidationError(f"Cannot stat generated image: {exc}") from exc
            if file_size <= 0 or file_size > max_bytes:
                raise ImageValidationError(
                    f"Generated image size {file_size} bytes exceeds limit ({max_bytes} bytes)."
                )

            image_bytes = chosen_path.read_bytes()
            mime_type, ext, width, height = _validate_image_bytes_strict(
                image_bytes,
                max_bytes=max_bytes,
            )

            raw_name = chosen_path.name
            safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name).lstrip(".")
            if not safe_name:
                safe_name = f"generated{ext}"

            usage_dict = usage.model_dump() if usage is not None else {}
            elapsed = max(0.0, time.monotonic() - turn_start_mono)

            return GeneratedImageResult(
                image_bytes=image_bytes,
                mime_type=mime_type,
                filename=safe_name,
                width=width,
                height=height,
                prompt_used=clean_prompt,
                caption_suggestion="",
                model_used=resolved_model,
                conversation_id=extracted_conv_id or parser.conversation_id,
                duration_seconds=elapsed,
                usage=usage_dict,
            )

        except (
            AntigravityNativeToolAbort,
            AntigravityRateLimitError,
            AntigravityBillingError,
            ImageGenerationError,
        ):
            if proc is not None:
                terminate_process_group(proc, timeout=1.0)
            raise
        except Exception as exc:
            if proc is not None:
                terminate_process_group(proc, timeout=1.0)
            if timed_out.is_set():
                raise ImageTimeoutError(
                    f"Antigravity image generation timed out after {timeout_s:.1f}s"
                ) from exc
            verdict = classify_error(exc)
            if verdict.category == ErrorCategory.RATE_LIMIT:
                raise AntigravityRateLimitError(str(exc), verdict=verdict) from exc
            if verdict.category == ErrorCategory.BILLING:
                raise AntigravityBillingError(str(exc), verdict=verdict) from exc
            raise ImageGenerationError(str(exc), verdict=verdict) from exc
        finally:
            if deadline_timer is not None:
                try:
                    deadline_timer.cancel()
                except Exception:
                    pass
            stop_watchdog.set()
            if proc is not None:
                terminate_process_group(proc, timeout=1.0)
            if watchdog_thread is not None:
                watchdog_thread.join(timeout=1.0)
            _close_proc_pipes(proc)
            if drainer is not None:
                drainer.join(timeout=1.0)
            shutil.rmtree(request_cwd, ignore_errors=True)
            if cleanup_artifacts:
                for cand in discovered_candidates:
                    try:
                        cand.unlink(missing_ok=True)
                    except OSError:
                        pass
            if gemini_dir is not None:
                sync_updated_isolated_tokens(gemini_dir, data_dir=self.data_dir)
            for cid in registered_cids:
                self._unregister_active_conversation(cid)
            self._maybe_schedule_session_cleanup(active_conversation_id=extracted_conv_id)

    async def generate_image(
        self,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        reference_images: list[MediaAttachment] | None = None,
        model: str | None = None,
        timeout_seconds: float = 120.0,
        output_dir: str | Path = "data/tmp/images",
        max_image_size_mb: float = 20.0,
        cleanup_artifacts: bool = True,
    ) -> GeneratedImageResult:
        """Asynchronously generate an image via isolated `agy` `stream-json` oneshot."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._sync_execute_image_turn(
                prompt,
                aspect_ratio=aspect_ratio,
                reference_images=reference_images,
                model=model,
                timeout_seconds=timeout_seconds,
                output_dir=output_dir,
                max_image_size_mb=max_image_size_mb,
                cleanup_artifacts=cleanup_artifacts,
            ),
        )

    async def generate_response(
        self,
        prompt: str,
        system_instructions: str | None = None,
        media_attachments: Sequence[MediaAttachment] | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        tool_profile: ToolProfile = "chat",
        mode: str | None = None,
        cwd: Path | str | None = None,
        add_dirs: Sequence[Path | str] | None = None,
        image_payloads: Sequence[tuple[str, str]] | None = None,
        timeout_seconds: float | None = None,
        allowed_tools: frozenset[str] | Collection[str] | None = None,
    ) -> ProviderResponse:
        """Asynchronously generate a response without blocking the event loop."""
        _resolve_effective_allowed_tools(tool_profile, allowed_tools)
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
            tool_profile,
            mode,
            cwd,
            add_dirs,
            image_payloads,
            timeout_seconds,
            allowed_tools,
        )
