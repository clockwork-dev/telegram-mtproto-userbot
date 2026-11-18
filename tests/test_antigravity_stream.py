"""Tests for Antigravity NDJSON stream parsing, thinking extraction, and tool interception."""

from __future__ import annotations

import json

import pytest

from bot.antigravity.errors import AntigravityNativeToolAbort
from bot.antigravity.stream import (
    AntigravityStreamParser,
    _longest_tag_prefix_tail,
    parse_ndjson_stream,
)


def test_longest_tag_prefix_tail():
    tag = "<tool_call>"
    assert _longest_tag_prefix_tail("hello <tool_c", tag) == len("<tool_c")
    assert _longest_tag_prefix_tail("hello <tool_", tag) == len("<tool_")
    assert _longest_tag_prefix_tail("hello <", tag) == len("<")
    assert _longest_tag_prefix_tail("hello <tool_call>", tag) == 0  # full tag, not strict prefix
    assert _longest_tag_prefix_tail("hello world", tag) == 0


def test_parse_init_event():
    parser = AntigravityStreamParser()
    line = json.dumps({"event": "init", "conversation_id": "conv_12345"})
    chunks = parser.feed_line(line)

    assert len(chunks) == 1
    assert chunks[0].conversation_id == "conv_12345"
    assert parser.conversation_id == "conv_12345"


def test_parse_step_update_text_and_thinking():
    parser = AntigravityStreamParser()

    # Step update with thinking
    line1 = json.dumps({
        "event": "step_update",
        "step_update": {"thinking_delta": "Thinking about the question..."},
    })
    chunks1 = parser.feed_line(line1)
    assert len(chunks1) == 1
    assert chunks1[0].thinking == "Thinking about the question..."

    # Step update with text
    line2 = json.dumps({
        "event": "step_update",
        "step_update": {"text_delta": "Hello world!"},
    })
    chunks2 = parser.feed_line(line2)
    assert len(chunks2) == 1
    assert chunks2[0].text == "Hello world!"


def test_native_tool_interception_step_update():
    interception_called = False

    def on_intercept():
        nonlocal interception_called
        interception_called = True

    parser = AntigravityStreamParser(on_native_tool_intercept=on_intercept)
    line = json.dumps({
        "event": "step_update",
        "step_update": {"step_type": "tool", "tool_name": "bash"},
    })

    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(line)

    assert interception_called is True
    assert parser.native_tool_intercepted is True


def test_native_tool_interception_assistant():
    interception_called = False

    def on_intercept():
        nonlocal interception_called
        interception_called = True

    parser = AntigravityStreamParser(on_native_tool_intercept=on_intercept)
    line = json.dumps({
        "event": "assistant",
        "message": {"step_type": "tool"},
    })

    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(line)

    assert interception_called is True


def test_safe_tag_prefix_buffering_split_across_chunks():
    parser = AntigravityStreamParser()

    # Chunk 1 ends with partial tag: "<tool_c"
    chunks1 = parser.feed_line(json.dumps({
        "event": "step_update",
        "step_update": {"text_delta": "Prefix text: <tool_c"},
    }))
    # "Prefix text: " should be emitted, but "<tool_c" held in buffer!
    assert len(chunks1) == 1
    assert chunks1[0].text == "Prefix text: "

    # Chunk 2 completes the tag and includes payload
    chunks2 = parser.feed_line(json.dumps({
        "event": "step_update",
        "step_update": {
            "text_delta": 'all>{"name": "check_status", "arguments": {"id": 42}}</tool_call>',
        },
    }))
    assert len(chunks2) == 1
    assert chunks2[0].tool_calls is not None
    assert chunks2[0].tool_calls[0]["function"]["name"] == "check_status"


def test_parse_error_event():
    parser = AntigravityStreamParser()
    line = json.dumps({"event": "error", "error": "Model capacity exhausted"})
    with pytest.raises(RuntimeError) as exc_info:
        parser.feed_line(line)
    assert "Model capacity exhausted" in str(exc_info.value)


def test_result_event_and_finish_usage():
    lines = [
        json.dumps({"event": "init", "conversation_id": "conv_99"}),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "text_delta": "Generated response content.",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "reasoning_tokens": 10,
                    "cache_read_tokens": 5,
                },
            },
        }),
        json.dumps({
            "event": "result",
            "result": {"status": "SUCCESS", "stop_reason": "end_turn"},
        }),
    ]

    all_chunks = list(parse_ndjson_stream(iter(lines)))
    assert any(c.text == "Generated response content." for c in all_chunks)

    final_chunks = [c for c in all_chunks if c.is_final]
    assert len(final_chunks) == 1
    final = final_chunks[0]
    assert final.finish_reason == "stop"
    assert final.conversation_id == "conv_99"
    assert final.usage is not None
    assert final.usage.prompt_tokens == 105
    assert final.usage.completion_tokens == 35


def test_search_web_and_read_url_content_allowed_while_run_command_aborted():
    """Verify search_web and read_url_content tool calls are allowed while run_command is aborted."""
    intercept_count = 0

    def on_intercept():
        nonlocal intercept_count
        intercept_count += 1

    parser = AntigravityStreamParser(on_native_tool_intercept=on_intercept)

    # Multi-step turn with search_web and read_url_content (matching live agy stream-json output)
    events = [
        json.dumps({"event": "init", "conversation_id": "conv_search_1"}),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 1,
                "state": "DONE",
                "step_type": "agent_response",
                "usage": {"input_tokens": 1000, "output_tokens": 50, "thinking_tokens": 30},
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 2,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "search_web",
                "tool_info": {"name": "search_web", "parameters": {"query": "погода в Сумах"}},
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "search_web",
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 3,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "read_url_content",
                "tool_info": {"name": "read_url_content", "parameters": {"url": "https://example.com"}},
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 4,
                "state": "ACTIVE",
                "step_type": "agent_response",
                "text_delta": "В Сумах сегодня ",
            },
        }),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv_search_1",
                "step_index": 4,
                "state": "DONE",
                "step_type": "agent_response",
                "text_delta": "+8 °C и облачно.",
            },
        }),
        json.dumps({
            "event": "result",
            "result": {
                "conversation_id": "conv_search_1",
                "status": "SUCCESS",
                "response": "В Сумах сегодня +8 °C и облачно.",
                "usage": {"input_tokens": 2000, "output_tokens": 90, "thinking_tokens": 45, "cache_read_tokens": 800},
            },
        }),
    ]

    emitted_chunks = []
    for line in events:
        emitted_chunks.extend(parser.feed_line(line))
    emitted_chunks.extend(parser.finish())

    assert intercept_count == 0
    assert parser.native_tool_intercepted is False
    combined_text = "".join(c.text for c in emitted_chunks if c.text)
    assert combined_text == "В Сумах сегодня +8 °C и облачно."

    final = [c for c in emitted_chunks if c.is_final][0]
    assert final.usage is not None
    assert final.usage.prompt_tokens == 2800
    assert final.usage.completion_tokens == 135
    assert final.usage.reasoning_tokens == 45

    # Dangerous local tools (run_command, write_to_file, view_file) must be strictly aborted
    for forbidden_tool in ("run_command", "write_to_file", "view_file", "list_dir"):
        p_forbid = AntigravityStreamParser(on_native_tool_intercept=on_intercept)
        with pytest.raises(AntigravityNativeToolAbort):
            p_forbid.feed_line(
                json.dumps({
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "tool_name": forbidden_tool},
                })
            )
        assert p_forbid.native_tool_intercepted is True
    assert intercept_count == 4
