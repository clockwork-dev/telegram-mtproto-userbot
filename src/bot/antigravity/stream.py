"""NDJSON streaming parser, thinking token extraction, and native tool interception."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, Literal, NoReturn

from pydantic import BaseModel

from bot.antigravity.errors import AntigravityNativeToolAbort
from bot.core.models import UsageInfo

logger = logging.getLogger(__name__)

ToolProfile = Literal["chat", "ops", "self_dev", "image"]

IMAGE_ALLOWED_SUBAGENT_TYPES: frozenset[str] = frozenset({"image-generator"})
_SAFE_NON_TOOL_STEP_TYPES: frozenset[str] = frozenset(
    {
        "user_input",
        "agent_response",
        "system_message",
        "thinking",
        "thought",
        "reasoning",
        "planner_response",
        "status",
        "metadata",
    }
)

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call(?:\s[^>]*)?>")
_UNCLOSED_TOOL_CALL_TAIL_RE = re.compile(r"<tool_call(?:\s[^>]*)?$")
_TOOL_CALL_CLOSE = "</tool_call>"

CHAT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "search_web",
        "read_url_content",
    }
)

OPS_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "search_web",
        "read_url_content",
        "read_file",
        "edit_file",
        "write_file",
        "run_command",
        "view_file",
        "list_dir",
        "grep_search",
        "find_by_name",
        "replace_file_content",
        "multi_replace_file_content",
        "write_to_file",
        "sed_file",
        "command_status",
        "send_command_input",
        "invoke_subagent",
        "define_subagent",
        "manage_subagents",
        "manage_task",
        "send_message",
        "run_workflow",
        "schedule",
    }
)

SELF_DEV_ALLOWED_TOOLS: frozenset[str] = OPS_ALLOWED_TOOLS

ALWAYS_FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {
        "browser_subagent",
        "open_browser_url",
        "read_browser_page",
        "execute_browser_javascript",
        "browser_click_element",
        "browser_drag_pixel_to_pixel",
        "browser_get_dom",
        "browser_get_network_request",
        "browser_input",
        "browser_list_network_requests",
        "browser_mouse_down",
        "browser_mouse_up",
        "browser_move_mouse",
        "browser_press_key",
        "browser_refresh_page",
        "browser_resize_window",
        "browser_scroll",
        "browser_scroll_dom",
        "browser_select_option",
        "capture_browser_console_logs",
        "capture_browser_screenshot",
        "click_browser_pixel",
        "list_browser_pages",
        "call_mcp_tool",
        "list_resources",
        "read_resource",
        "manage_inbox",
        "notebook_edit",
        "notebook_execution",
        "swarm",
        "delete_knowledge",
        "ask_question",
        "ask_permission",
        "ask_custom_permission",
        "list_permissions",
        "list_plugin_accounts",
        "search_marketplace",
        "generate_image",
        "bash",
        "exec",
        "shell",
        "finish",
        "wait",
        "wait_5_seconds",
    }
)

# Backward-compatible aliases reflecting default "chat" profile
_ALLOWED_NATIVE_TOOLS: frozenset[str] = CHAT_ALLOWED_TOOLS
_FORBIDDEN_NATIVE_TOOLS: frozenset[str] = ALWAYS_FORBIDDEN_TOOLS | (
    OPS_ALLOWED_TOOLS - CHAT_ALLOWED_TOOLS
)


def resolve_allowed_tools(
    tool_profile: ToolProfile = "chat",
    allowed_tools: frozenset[str] | None = None,
) -> frozenset[str]:
    """Resolve the active tool allowlist for a profile, never permitting ALWAYS_FORBIDDEN_TOOLS."""
    if tool_profile == "chat":
        profile_tools = CHAT_ALLOWED_TOOLS
    elif tool_profile == "ops":
        profile_tools = OPS_ALLOWED_TOOLS
    elif tool_profile == "self_dev":
        profile_tools = SELF_DEV_ALLOWED_TOOLS
    elif tool_profile == "image":
        # Image profile permits NO generic direct tools; only strictly arg-gated
        # invoke_subagent(type_name="image-generator") is allowed via is_allowed_image_subagent_step.
        profile_tools = frozenset()
    else:
        raise ValueError(f"Unsupported Antigravity tool_profile: {tool_profile!r}")

    if allowed_tools is not None:
        normalized = frozenset(
            str(name).strip().lower() for name in allowed_tools if str(name).strip()
        )
        return normalized - ALWAYS_FORBIDDEN_TOOLS
    return profile_tools - ALWAYS_FORBIDDEN_TOOLS


def _extract_native_tool_name(obj: dict[str, Any]) -> str:
    """Extract normalized tool name from an agy step_update or assistant block dict."""
    for key in ("tool_name", "toolName", "toolname", "tool", "name"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    for info_key in ("tool_info", "toolInfo"):
        tool_info = obj.get(info_key)
        if isinstance(tool_info, dict):
            info_name = (
                tool_info.get("name")
                or tool_info.get("tool_name")
                or tool_info.get("toolName")
            )
            if isinstance(info_name, str) and info_name.strip():
                return info_name.strip().lower()
    fn_obj = obj.get("function")
    if isinstance(fn_obj, dict):
        fn_name = fn_obj.get("name")
        if isinstance(fn_name, str) and fn_name.strip():
            return fn_name.strip().lower()
    return ""


def _has_tool_or_subagent_payload(obj: dict[str, Any]) -> bool:
    """Return True if a step_update carries explicit tool or subagent metadata."""
    return bool(
        obj.get("tool_name")
        or obj.get("toolName")
        or obj.get("toolname")
        or obj.get("tool")
        or obj.get("tool_info")
        or obj.get("toolInfo")
        or obj.get("function")
        or obj.get("subagent_info")
        or obj.get("subagentInfo")
    )


def _extract_subagent_types(obj: dict[str, Any]) -> list[str] | None:
    """Extract normalized subagent type_name entries if subagent metadata is present."""
    subagent_info = obj.get("subagent_info")
    if subagent_info is None:
        subagent_info = obj.get("subagentInfo")
    step_type = str(obj.get("step_type") or "").strip().lower()
    if not isinstance(subagent_info, dict):
        return [] if step_type == "subagent" else None
    subagents = subagent_info.get("subagents")
    if not isinstance(subagents, list):
        return []
    types: list[str] = []
    for entry in subagents:
        if isinstance(entry, dict):
            raw_type = (
                entry.get("type_name")
                if entry.get("type_name") is not None
                else (
                    entry.get("typeName")
                    if entry.get("typeName") is not None
                    else entry.get("TypeName")
                )
            )
            if isinstance(raw_type, str) and raw_type.strip():
                types.append(raw_type.strip().lower())
    return types


def is_allowed_image_subagent_step(obj: dict[str, Any]) -> bool:
    """Fail-closed check: True ONLY when obj invokes exclusively the 'image-generator' subagent."""
    tool_name = _extract_native_tool_name(obj)
    if tool_name != "invoke_subagent":
        return False
    subagent_info = obj.get("subagent_info")
    if subagent_info is None:
        subagent_info = obj.get("subagentInfo")
    if not isinstance(subagent_info, dict):
        return False
    subagents = subagent_info.get("subagents")
    if not isinstance(subagents, list) or not subagents:
        return False
    for entry in subagents:
        if not isinstance(entry, dict):
            return False
        raw_type = (
            entry.get("type_name")
            if entry.get("type_name") is not None
            else (
                entry.get("typeName")
                if entry.get("typeName") is not None
                else entry.get("TypeName")
            )
        )
        if (
            not isinstance(raw_type, str)
            or raw_type.strip().lower() not in IMAGE_ALLOWED_SUBAGENT_TYPES
        ):
            return False
    return True


def is_allowed_native_tool(
    obj: dict[str, Any],
    *,
    allowed_tools: frozenset[str] | None = None,
    tool_profile: ToolProfile = "chat",
) -> bool:
    """Return True only if the tool event explicitly names an allowed tool for the profile."""
    resolved_allowed = resolve_allowed_tools(
        tool_profile=tool_profile,
        allowed_tools=allowed_tools,
    )
    step_type = str(obj.get("step_type") or "").strip().lower()
    has_subagent_info = bool(obj.get("subagent_info") or obj.get("subagentInfo"))
    if tool_profile == "image":
        if step_type == "subagent" or has_subagent_info:
            return is_allowed_image_subagent_step(obj)
        return False
    if step_type == "subagent" or has_subagent_info:
        if tool_profile not in ("ops", "self_dev"):
            return False
        tool_name = _extract_native_tool_name(obj)
        if tool_name:
            if tool_name in ALWAYS_FORBIDDEN_TOOLS:
                return False
            return tool_name in resolved_allowed
        return "invoke_subagent" in resolved_allowed
    tool_name = _extract_native_tool_name(obj)
    if not tool_name:
        return False
    if tool_name in ALWAYS_FORBIDDEN_TOOLS:
        return False
    return tool_name in resolved_allowed


def _is_allowed_native_tool(obj: dict[str, Any]) -> bool:
    """Backward-compatible wrapper checking default 'chat' profile."""
    return is_allowed_native_tool(obj, tool_profile="chat")


class StreamChunk(BaseModel):
    """Normalized chunk from streaming response."""

    text: str | None = None
    thinking: str | None = None
    is_final: bool = False
    finish_reason: str | None = None
    usage: UsageInfo | None = None
    conversation_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


def _longest_tag_prefix_tail(text: str, tag: str) -> int:
    """Return the length of the longest suffix of text that is a strict prefix of tag
    or an unclosed '<tool_call ...' opening tag with attributes.

    Used to avoid emitting incomplete '<tool_call>' tags before the tag is fully received.
    """
    m = _UNCLOSED_TOOL_CALL_TAIL_RE.search(text)
    if m is not None:
        return len(text) - m.start()
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class AntigravityStreamParser:
    """Stateful line-by-line NDJSON parser with safe tag buffering and tool interception."""

    def __init__(
        self,
        *,
        on_conversation_id: Callable[[str], None] | None = None,
        on_native_tool_intercept: Callable[[], None] | None = None,
        allowed_tools: frozenset[str] | None = None,
        tool_profile: ToolProfile = "chat",
    ) -> None:
        self._on_conversation_id = on_conversation_id
        self._on_native_tool_intercept = on_native_tool_intercept
        self.tool_profile: ToolProfile = tool_profile
        self.allowed_tools: frozenset[str] = resolve_allowed_tools(
            tool_profile=tool_profile,
            allowed_tools=allowed_tools,
        )

        self.text_buffer = ""
        self.in_tool_call = False
        self.tool_call_buffer = ""
        self.emitted_tool_calls = 0

        self.full_content_parts: list[str] = []
        self.full_reasoning_parts: list[str] = []
        self.usage_stats: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self.conversation_id: str | None = None
        self.saw_result_event = False
        self.result_stop_reason = "end_turn"
        self.native_tool_intercepted = False

    def _is_allowed_tool(self, obj: dict[str, Any]) -> bool:
        return is_allowed_native_tool(
            obj,
            allowed_tools=self.allowed_tools,
            tool_profile=self.tool_profile,
        )

    def _is_allowed_step(self, obj: dict[str, Any], *, step_type: str = "") -> bool:
        norm_type = step_type.strip().lower() if step_type else str(obj.get("step_type") or "").strip().lower()
        has_subagent_info = bool(obj.get("subagent_info") or obj.get("subagentInfo"))
        if self.tool_profile == "image":
            if norm_type == "subagent" or has_subagent_info:
                return is_allowed_image_subagent_step(obj)
            return False
        if norm_type == "subagent" or has_subagent_info:
            if self.tool_profile not in ("ops", "self_dev"):
                return False
            tool_name = _extract_native_tool_name(obj)
            if tool_name:
                if tool_name in ALWAYS_FORBIDDEN_TOOLS:
                    return False
                return tool_name in self.allowed_tools
            return "invoke_subagent" in self.allowed_tools
        return self._is_allowed_tool(obj)

    def _raise_native_tool_abort(
        self,
        obj: dict[str, Any],
        *,
        source: str,
        step_type: str = "",
    ) -> NoReturn:
        self.native_tool_intercepted = True
        if self._on_native_tool_intercept:
            self._on_native_tool_intercept()
        tool_name = _extract_native_tool_name(obj)
        subagents = _extract_subagent_types(obj)
        payload_keys = sorted(str(k) for k in obj.keys())
        abort_exc = AntigravityNativeToolAbort.from_context(
            tool_profile=self.tool_profile,
            source=source,
            step_type=step_type or str(obj.get("step_type") or "").strip().lower(),
            tool_name=tool_name,
            subagents=subagents,
            payload_keys=payload_keys,
        )
        safe_keys = abort_exc.payload_keys[:12]
        keys_suffix = f", keys={','.join(safe_keys)}" if safe_keys else ""
        logger.warning(
            "Native tool execution intercepted: tool_profile=%s, source=%s, step_type=%s, tool_name=%s, subagents=%s%s",
            abort_exc.tool_profile or "",
            abort_exc.source or "",
            abort_exc.step_type or "",
            abort_exc.tool_name or "",
            ",".join(abort_exc.subagents) if abort_exc.subagents else "none",
            keys_suffix,
        )
        setattr(abort_exc, "_native_tool_abort_logged", True)
        raise abort_exc

    def _parse_tool_call_json(self, raw: str, index: int) -> dict[str, Any] | None:
        """Parse raw XML tool call payload into dict if valid JSON."""
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        name = str(data.get("name") or data.get("tool") or "").strip()
        args = data.get("arguments") or data.get("args") or {}
        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
        return {
            "index": index,
            "id": f"call_{index + 1}",
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        }

    def _flush_safe_text(self, pending: deque[StreamChunk], *, final: bool = False) -> None:
        while True:
            if not self.in_tool_call:
                if not self.text_buffer:
                    return
                open_match = _TOOL_CALL_OPEN_RE.search(self.text_buffer)
                if open_match is not None:
                    before = self.text_buffer[: open_match.start()]
                    if before:
                        self.full_content_parts.append(before)
                        pending.append(StreamChunk(text=before))
                    self.text_buffer = self.text_buffer[open_match.end() :]
                    self.in_tool_call = True
                    self.tool_call_buffer = ""
                    continue

                if final:
                    self.full_content_parts.append(self.text_buffer)
                    pending.append(StreamChunk(text=self.text_buffer))
                    self.text_buffer = ""
                    return

                hold_len = _longest_tag_prefix_tail(self.text_buffer, _TOOL_CALL_OPEN)
                emit_len = len(self.text_buffer) - hold_len
                if emit_len > 0:
                    to_emit = self.text_buffer[:emit_len]
                    self.full_content_parts.append(to_emit)
                    pending.append(StreamChunk(text=to_emit))
                    self.text_buffer = self.text_buffer[emit_len:]
                return
            else:
                self.tool_call_buffer += self.text_buffer
                self.text_buffer = ""
                close_idx = self.tool_call_buffer.find(_TOOL_CALL_CLOSE)
                if close_idx != -1:
                    raw_payload = self.tool_call_buffer[:close_idx]
                    remainder = self.tool_call_buffer[close_idx + len(_TOOL_CALL_CLOSE) :]
                    self.in_tool_call = False
                    self.tool_call_buffer = ""
                    self.text_buffer = remainder

                    tc = self._parse_tool_call_json(raw_payload, self.emitted_tool_calls)
                    if tc is not None:
                        self.emitted_tool_calls += 1
                        pending.append(StreamChunk(tool_calls=[tc]))
                    else:
                        literal = f"{_TOOL_CALL_OPEN}{raw_payload}{_TOOL_CALL_CLOSE}"
                        self.full_content_parts.append(literal)
                        pending.append(StreamChunk(text=literal))
                    continue

                if final:
                    tc = self._parse_tool_call_json(self.tool_call_buffer, self.emitted_tool_calls)
                    if tc is not None:
                        self.emitted_tool_calls += 1
                        pending.append(StreamChunk(tool_calls=[tc]))
                    else:
                        literal = f"{_TOOL_CALL_OPEN}{self.tool_call_buffer}"
                        self.full_content_parts.append(literal)
                        pending.append(StreamChunk(text=literal))
                    self.in_tool_call = False
                    self.tool_call_buffer = ""
                return

    def feed_line(self, line: str) -> list[StreamChunk]:
        """Process one line of NDJSON output from agy and return newly emitted chunks."""
        pending: deque[StreamChunk] = deque()
        stripped = line.strip()
        if not stripped:
            return []

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return []

        if not isinstance(event, dict):
            return []

        conv_id = event.get("conversation_id") or event.get("session_id")
        if isinstance(conv_id, str) and conv_id.strip():
            self.conversation_id = conv_id.strip()
            if self._on_conversation_id:
                self._on_conversation_id(self.conversation_id)

        ev_type = str(event.get("event") or event.get("type") or "").strip().lower()

        if ev_type == "init":
            init_obj = event.get("init")
            if isinstance(init_obj, dict):
                init_conv = init_obj.get("conversation_id") or init_obj.get("session_id")
                if isinstance(init_conv, str) and init_conv.strip():
                    self.conversation_id = init_conv.strip()
                    if self._on_conversation_id:
                        self._on_conversation_id(self.conversation_id)
            pending.append(StreamChunk(conversation_id=self.conversation_id))
            return list(pending)

        if ev_type == "step_update":
            step_obj = event.get("step_update")
            if isinstance(step_obj, dict):
                step_conv = step_obj.get("conversation_id")
                if isinstance(step_conv, str) and step_conv.strip():
                    self.conversation_id = step_conv.strip()
                    if self._on_conversation_id:
                        self._on_conversation_id(self.conversation_id)

                step_type = str(step_obj.get("step_type") or "").strip().lower()
                is_tool_or_subagent = step_type in ("tool", "subagent") or (
                    step_type not in _SAFE_NON_TOOL_STEP_TYPES
                    and _has_tool_or_subagent_payload(step_obj)
                )
                if is_tool_or_subagent:
                    if not self._is_allowed_step(step_obj, step_type=step_type):
                        # Native tool or subagent interception security trigger
                        self._raise_native_tool_abort(
                            step_obj,
                            source="step_update",
                            step_type=step_type,
                        )
                    return list(pending)

                raw_usage = step_obj.get("usage")
                if isinstance(raw_usage, dict):
                    for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens"):
                        if k in raw_usage:
                            self.usage_stats[k] = int(raw_usage.get(k) or 0)
                    if "reasoning_tokens" not in raw_usage and "thinking_tokens" in raw_usage:
                        self.usage_stats["reasoning_tokens"] = int(raw_usage.get("thinking_tokens") or 0)

                thinking_delta = step_obj.get("thinking_delta") or step_obj.get("thinking")
                if isinstance(thinking_delta, str) and thinking_delta:
                    self.full_reasoning_parts.append(thinking_delta)
                    pending.append(StreamChunk(thinking=thinking_delta))

                text_delta = step_obj.get("text_delta")
                if isinstance(text_delta, str) and text_delta:
                    self.text_buffer += text_delta
                    self._flush_safe_text(pending, final=False)

            return list(pending)

        if ev_type == "error":
            err_msg = str(event.get("error") or event.get("message") or "Unknown Antigravity error")
            raise RuntimeError(f"Antigravity model error: {err_msg}")

        if ev_type == "assistant":
            msg_obj = event.get("message")
            if isinstance(msg_obj, dict):
                step_type = str(msg_obj.get("step_type") or "").strip().lower()
                if step_type in ("tool", "subagent") and not self._is_allowed_step(
                    msg_obj, step_type=step_type
                ):
                    self._raise_native_tool_abort(
                        msg_obj,
                        source="assistant",
                        step_type=step_type,
                    )

                blocks = msg_obj.get("content")
                if isinstance(blocks, list):
                    for block in blocks:
                        if not isinstance(block, dict):
                            continue
                        b_step_type = str(block.get("step_type") or "").strip().lower()
                        b_type = str(block.get("type") or "").strip().lower()
                        if b_step_type in ("tool", "subagent") or b_type in (
                            "tool",
                            "tool_use",
                            "tool_call",
                            "subagent",
                        ):
                            if not self._is_allowed_step(
                                block, step_type=b_step_type or b_type
                            ):
                                self._raise_native_tool_abort(
                                    block,
                                    source="assistant.content",
                                    step_type=b_step_type or b_type,
                                )
                            continue
                        if b_type == "thinking":
                            thinking_text = str(block.get("thinking") or "")
                            if thinking_text:
                                self.full_reasoning_parts.append(thinking_text)
                                pending.append(StreamChunk(thinking=thinking_text))
                        elif b_type == "text":
                            chunk_text = str(block.get("text") or "")
                            if chunk_text:
                                self.text_buffer += chunk_text
                                self._flush_safe_text(pending, final=False)
            return list(pending)

        if ev_type == "result":
            self.saw_result_event = True
            res_obj = event.get("result")
            if isinstance(res_obj, dict):
                res_conv = res_obj.get("conversation_id")
                if isinstance(res_conv, str) and res_conv.strip():
                    self.conversation_id = res_conv.strip()
                    if self._on_conversation_id:
                        self._on_conversation_id(self.conversation_id)
                res_status = str(res_obj.get("status") or "").strip().upper()
                res_err = res_obj.get("error")
                if res_err or res_status == "ERROR":
                    raise RuntimeError(f"Antigravity model error: {res_err or res_status}")
                stop_r = res_obj.get("stop_reason") or res_obj.get("status")
                if isinstance(stop_r, str) and stop_r.strip():
                    self.result_stop_reason = stop_r.strip()
                result_text = res_obj.get("response") or res_obj.get("result")
                raw_usage = res_obj.get("usage") or event.get("usage")
            else:
                if event.get("is_error"):
                    err_detail = str(event.get("error") or event.get("result") or "Unknown error")
                    raise RuntimeError(f"Antigravity execution failed: {err_detail}")
                stop_r = event.get("stop_reason")
                if isinstance(stop_r, str) and stop_r.strip():
                    self.result_stop_reason = stop_r.strip()
                result_text = res_obj if isinstance(res_obj, str) else None
                raw_usage = event.get("usage")

            if (
                isinstance(result_text, str)
                and result_text.strip()
                and not self.full_content_parts
                and not self.text_buffer
                and not self.in_tool_call
                and self.emitted_tool_calls == 0
            ):
                self.text_buffer += result_text
                self._flush_safe_text(pending, final=False)

            if isinstance(raw_usage, dict):
                for k in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens"):
                    if k in raw_usage:
                        self.usage_stats[k] = int(raw_usage.get(k) or 0)
                if "reasoning_tokens" not in raw_usage and "thinking_tokens" in raw_usage:
                    self.usage_stats["reasoning_tokens"] = int(raw_usage.get("thinking_tokens") or 0)

            return list(pending)

        return []

    def finish(self) -> list[StreamChunk]:
        """Flush final text buffers and emit final chunk with usage metrics."""
        pending: deque[StreamChunk] = deque()
        self._flush_safe_text(pending, final=True)

        finish_reason = "tool_calls" if self.emitted_tool_calls > 0 else "stop"
        if self.result_stop_reason in ("max_tokens", "length") and finish_reason == "stop":
            finish_reason = "length"

        usage = UsageInfo(
            prompt_tokens=self.usage_stats["input_tokens"] + self.usage_stats["cache_read_tokens"],
            completion_tokens=self.usage_stats["output_tokens"] + self.usage_stats["reasoning_tokens"],
            total_tokens=sum(self.usage_stats.values()),
            reasoning_tokens=self.usage_stats["reasoning_tokens"],
            cached_tokens=self.usage_stats["cache_read_tokens"],
            input_tokens=self.usage_stats["input_tokens"],
            output_tokens=self.usage_stats["output_tokens"],
            cache_read_tokens=self.usage_stats["cache_read_tokens"],
            cache_write_tokens=self.usage_stats["cache_write_tokens"],
        )

        final_chunk = StreamChunk(
            is_final=True,
            finish_reason=finish_reason,
            usage=usage,
            conversation_id=self.conversation_id,
        )
        pending.append(final_chunk)
        return list(pending)


def parse_ndjson_stream(
    lines: Iterator[str],
    *,
    on_conversation_id: Callable[[str], None] | None = None,
    on_native_tool_intercept: Callable[[], None] | None = None,
    allowed_tools: frozenset[str] | None = None,
    tool_profile: ToolProfile = "chat",
) -> Iterator[StreamChunk]:
    """Generator parsing an iterator of NDJSON lines into StreamChunks."""
    parser = AntigravityStreamParser(
        on_conversation_id=on_conversation_id,
        on_native_tool_intercept=on_native_tool_intercept,
        allowed_tools=allowed_tools,
        tool_profile=tool_profile,
    )
    for line in lines:
        for chunk in parser.feed_line(line):
            yield chunk
    for chunk in parser.finish():
        yield chunk
