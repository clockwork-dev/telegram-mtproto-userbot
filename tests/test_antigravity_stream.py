"""Tests for Antigravity NDJSON stream parsing, ToolProfile gating, and secret-free abort diagnostics."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from bot.antigravity.errors import AntigravityNativeToolAbort, ErrorCategory, classify_error
from bot.antigravity.stream import (
    ALWAYS_FORBIDDEN_TOOLS,
    CHAT_ALLOWED_TOOLS,
    IMAGE_ALLOWED_SUBAGENT_TYPES,
    OPS_ALLOWED_TOOLS,
    SELF_DEV_ALLOWED_TOOLS,
    AntigravityStreamParser,
    _longest_tag_prefix_tail,
    is_allowed_image_subagent_step,
    parse_ndjson_stream,
    resolve_allowed_tools,
)


def test_tool_profile_invariants_and_always_forbidden():
    assert CHAT_ALLOWED_TOOLS == frozenset({"search_web", "read_url_content"})
    assert len(OPS_ALLOWED_TOOLS) == 23
    assert SELF_DEV_ALLOWED_TOOLS == OPS_ALLOWED_TOOLS
    assert IMAGE_ALLOWED_SUBAGENT_TYPES == frozenset({"image-generator"})
    for forbidden in (
        "browser_subagent",
        "open_browser_url",
        "call_mcp_tool",
        "swarm",
        "generate_image",
        "ask_question",
        "ask_permission",
        "ask_custom_permission",
    ):
        assert forbidden in ALWAYS_FORBIDDEN_TOOLS
    assert resolve_allowed_tools("image") == frozenset()
    assert "swarm" not in resolve_allowed_tools("ops", allowed_tools=frozenset({"swarm", "read_file"}))
    with pytest.raises(ValueError):
        resolve_allowed_tools("invalid_profile")  # type: ignore[arg-type]


def test_longest_tag_prefix_tail_and_split_chunks():
    tag = "<tool_call>"
    assert _longest_tag_prefix_tail("hello <tool_c", tag) == len("<tool_c")
    assert _longest_tag_prefix_tail("hello <tool_call>", tag) == 0

    parser = AntigravityStreamParser()
    chunks1 = parser.feed_line(json.dumps({
        "event": "step_update",
        "step_update": {"text_delta": "Prefix text: <tool_c"},
    }))
    assert len(chunks1) == 1 and chunks1[0].text == "Prefix text: "

    chunks2 = parser.feed_line(json.dumps({
        "event": "step_update",
        "step_update": {
            "text_delta": 'all>{"name": "check_status", "arguments": {"id": 42}}</tool_call>',
        },
    }))
    assert len(chunks2) == 1
    assert chunks2[0].tool_calls is not None
    assert chunks2[0].tool_calls[0]["function"]["name"] == "check_status"


def test_result_event_and_chat_allowed_vs_forbidden_tools():
    events = [
        json.dumps({"event": "init", "conversation_id": "conv_search_1"}),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_type": "tool",
                "tool_name": "search_web",
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_type": "tool",
                "tool_name": "read_url_content",
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_type": "agent_response",
                "text_delta": "В Сумах +8 °C.",
            },
        }),
        json.dumps({
            "event": "result",
            "result": {
                "conversation_id": "conv_search_1",
                "status": "SUCCESS",
                "usage": {"input_tokens": 2000, "output_tokens": 90, "thinking_tokens": 45, "cache_read_tokens": 800},
            },
        }),
    ]
    all_chunks = list(parse_ndjson_stream(iter(events)))
    assert "".join(c.text for c in all_chunks if c.text) == "В Сумах +8 °C."
    final = [c for c in all_chunks if c.is_final][0]
    assert final.usage is not None
    assert final.usage.prompt_tokens == 2800
    assert final.usage.completion_tokens == 135
    assert final.usage.reasoning_tokens == 45
    assert final.usage.cache_read_tokens == 800

    for forbidden_tool in ("run_command", "write_to_file", "view_file", "list_dir"):
        p_forbid = AntigravityStreamParser(tool_profile="chat")
        with pytest.raises(AntigravityNativeToolAbort):
            p_forbid.feed_line(
                json.dumps({
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": forbidden_tool},
                })
            )


@pytest.mark.parametrize("subagent_type", ["image-generator", "self", "research"])
def test_stream_parser_blocks_subagent_in_chat_and_allows_ops(subagent_type: str):
    event = {
        "event": "step_update",
        "step_update": {
            "conversation_id": "conv-123",
            "step_type": "subagent",
            "tool_name": "invoke_subagent",
            "subagent_info": {"subagents": [{"type_name": subagent_type}]},
        },
    }
    p_chat = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort):
        p_chat.feed_line(json.dumps(event))

    p_ops = AntigravityStreamParser(tool_profile="ops")
    p_ops.feed_line(json.dumps(event))
    assert p_ops.native_tool_intercepted is False


@pytest.mark.parametrize(
    "bad_step",
    [
        {"step_type": "subagent", "tool_name": "invoke_subagent", "subagent_info": {"subagents": [{"type_name": "self"}]}},
        {"step_type": "subagent", "tool_name": "invoke_subagent", "subagent_info": {"subagents": [{"type_name": "image-generator"}, {"type_name": "self"}]}},
        {"step_type": "subagent", "tool_name": "invoke_subagent", "subagent_info": {"subagents": []}},
        {"step_type": "subagent", "tool_name": "browser_subagent", "subagent_info": {"subagents": [{"type_name": "image-generator"}]}},
        {"step_type": "tool", "tool_name": "run_command"},
        {"step_type": "tool", "tool_name": "read_file"},
        {"step_type": "tool", "tool_name": "search_web"},
        {"step_type": "tool", "tool_name": "generate_image"},
        {"step_type": "tool"},
    ],
)
def test_image_profile_allows_only_image_generator_and_aborts_unsafe(bad_step: dict[str, Any]):
    ok_step = {
        "step_type": "subagent",
        "tool_name": "invoke_subagent",
        "subagent_info": {"subagents": [{"type_name": "Image-Generator"}]},
    }
    assert is_allowed_image_subagent_step(ok_step) is True
    p_ok = AntigravityStreamParser(tool_profile="image")
    p_ok.feed_line(json.dumps({"event": "step_update", "step_update": ok_step}))
    assert p_ok.native_tool_intercepted is False

    p_bad = AntigravityStreamParser(tool_profile="image")
    with pytest.raises(AntigravityNativeToolAbort):
        p_bad.feed_line(json.dumps({"event": "step_update", "step_update": bad_step}))


def test_abort_diagnostics_secret_free_and_safe_non_tool_steps(caplog: pytest.LogCaptureFixture) -> None:
    parser = AntigravityStreamParser(tool_profile="chat")
    with caplog.at_level(logging.WARNING, logger="bot.antigravity.stream"):
        parser.feed_line(json.dumps({"event": "step_update", "step_update": {"name": "step_one", "text_delta": "Hi"}}))
    assert parser.native_tool_intercepted is False
    assert caplog.records == []

    secret_tok = "sk-proj-SUPERSECRET1234567890"
    secret_cmd = "cat /opt/telegram-mtproto-userbot/.env"
    p_abort = AntigravityStreamParser(tool_profile="chat")
    with pytest.raises(AntigravityNativeToolAbort) as exc_info:
        p_abort.feed_line(
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "tool_name": f"bash {secret_tok}",
                    "tool_info": {"parameters": {"command": secret_cmd}},
                },
            })
        )
    msg = str(exc_info.value)
    assert "tool=<redacted>" in msg
    assert secret_tok not in msg
    assert secret_cmd not in msg
    assert classify_error(msg).category == ErrorCategory.NATIVE_TOOL_ABORT
