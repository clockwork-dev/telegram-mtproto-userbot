"""Tests for error classification, per-model circuit breaker, PreToolUse hooks, CLI/SDK, and image_gen."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from bot.antigravity import (
    AntigravityBillingError,
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityNativeToolAbort,
    AntigravityProvider,
    AntigravityRateLimitError,
    AntigravitySDKClient,
    ErrorCategory,
    GeneratedImageResult,
    ImageGenerationError,
    ImagePolicyRefusalError,
    ImageTimeoutError,
    ImageValidationError,
    calculate_backoff,
    classify_error,
    extract_retry_after,
    write_chat_pre_tool_hook,
    write_image_pre_tool_hook,
)
from bot.antigravity.cli_client import watch_log_for_errors
from bot.core.models import MediaAttachment, ProviderResponse


def _make_image_bytes(fmt: str = "JPEG", size: tuple[int, int] = (32, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (220, 40, 40)).save(buf, format=fmt)
    return buf.getvalue()


def _run_hook(gate_script: Path, payload: str) -> dict[str, Any]:
    res = subprocess.run(
        [sys.executable, str(gate_script)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return json.loads(res.stdout.strip())


class _MockPopen:
    """Mock subprocess.Popen for CLI client testing."""

    def __init__(self, stdout_lines: list[str], *, block_until_killed: bool = False, returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self._lines = list(stdout_lines)
        self._idx = 0
        self._block_until_killed = block_until_killed
        self._killed = threading.Event()
        self.stderr = io.StringIO("")
        self.pid = 4242
        self.returncode = returncode
        self._terminated = False
        outer = self

        class _Stdout:
            def readline(self) -> str:
                if outer._block_until_killed:
                    outer._killed.wait(timeout=5.0)
                    return ""
                if outer._idx < len(outer._lines):
                    line = outer._lines[outer._idx]
                    outer._idx += 1
                    return line if line.endswith("\n") else line + "\n"
                return ""

            def close(self) -> None:
                pass

        self.stdout = _Stdout()

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True
        self._killed.set()

    def kill(self) -> None:
        self._terminated = True
        self._killed.set()

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        self._killed.set()
        return self.returncode


class _InteractiveAcpPopen:
    """Mock subprocess.Popen simulating an `agy acp --stdio` JSON-RPC 2.0 server."""

    def __init__(self) -> None:
        self.sent_requests: list[dict[str, Any]] = []
        self._responses: list[str] = []
        self._resp_idx = 0
        self.stderr = io.StringIO("")
        self.pid = 5252
        self.returncode = 0
        self._terminated = False
        outer = self

        class _StdinProxy:
            def write(self, data: str) -> int:
                for raw in data.splitlines():
                    if raw.strip():
                        outer._handle_client_line(raw)
                return len(data)

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        class _StdoutProxy:
            def readline(self) -> str:
                if outer._resp_idx < len(outer._responses):
                    line = outer._responses[outer._resp_idx]
                    outer._resp_idx += 1
                    return line
                return ""

            def close(self) -> None:
                pass

        self.stdin = _StdinProxy()
        self.stdout = _StdoutProxy()

    def _handle_client_line(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent_requests.append(msg)
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            self._responses.append(json.dumps({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}) + "\n")
        elif method == "session/new":
            self._responses.append(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "sessionId": "acp_sess_1",
                        "models": {
                            "currentModelId": "gemini-4-argon-medium",
                            "availableModels": [{"modelId": "gemini-3.8-flash-low", "name": "Gemini 3.8 Flash"}],
                        },
                    },
                })
                + "\n"
            )
        elif method == "session/set_model":
            self._responses.append(json.dumps({"jsonrpc": "2.0", "id": mid, "result": {}}) + "\n")
        elif method == "session/prompt":
            self._responses.append(
                json.dumps({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "acp_sess_1",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "На фото рыжий кот."},
                        },
                    },
                })
                + "\n"
            )
            self._responses.append(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 250, "outputTokens": 12, "totalTokens": 262},
                    },
                })
                + "\n"
            )

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        return self.returncode


def test_error_classification_retry_after_and_backoff():
    assert classify_error("429 Too Many Requests: Resource has been exhausted").category == ErrorCategory.RATE_LIMIT
    assert classify_error("Previous response was cut off because it exceeded the output token limit").category == ErrorCategory.SERVER_ERROR
    assert classify_error("max_trajectory_tokens exceeded: prompt is too long").category == ErrorCategory.CONTEXT_OVERFLOW
    assert classify_error("AI credits balance is too low").category == ErrorCategory.BILLING
    assert classify_error("OAuth token expired, please re-authenticate").category == ErrorCategory.AUTH_PERMANENT
    assert classify_error("Unknown model foo-99").category == ErrorCategory.MODEL_NOT_FOUND
    assert extract_retry_after("429 RESOURCE_EXHAUSTED, retry in 1h 2m 3s") == 3723.0
    assert calculate_backoff(2, base_delay=1.0, max_delay=30.0, jitter=False) == 4.0


@pytest.mark.asyncio
async def test_cli_and_sdk_execution_and_acp_vision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cli = AntigravityCLIClient(data_dir=tmp_path / "agy", binary="/mock/bin/agy", default_model="gemini-3.8-flash", default_effort="low")
    stdout_lines = [
        json.dumps({"event": "init", "conversation_id": "conv_cli_01"}),
        json.dumps({
            "event": "step_update",
            "step_update": {
                "text_delta": "Response from CLI.",
                "thinking_delta": "Thinking...",
                "usage": {"input_tokens": 50, "output_tokens": 15},
            },
        }),
        json.dumps({"event": "result", "result": {"status": "SUCCESS"}}),
    ]
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=_MockPopen(stdout_lines)):
        resp = await cli.generate_response(prompt="Hello CLI")
    assert resp.text == "Response from CLI."
    assert resp.conversation_id == "conv_cli_01"

    # ACP image vision
    jpeg_path = tmp_path / "cat.jpg"
    jpeg_path.write_bytes(_make_image_bytes("JPEG"))
    acp_proc = _InteractiveAcpPopen()
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=acp_proc):
        resp_img = await cli.generate_response(
            prompt="Что на картинке?",
            media_attachments=[MediaAttachment(file_path=jpeg_path, mime_type="image/jpeg")],
        )
    assert resp_img.text == "На фото рыжий кот."
    cli.close()

    # SDK primary -> CLI fallback
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-api-key")
    provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "p_agy", enable_sdk=True, enable_cli_fallback=True, max_retries=1))
    if provider._sdk_client is not None:
        provider._sdk_client.generate_response = AsyncMock(side_effect=RuntimeError("SDK unavailable 503"))
    assert provider._cli_client is not None
    provider._cli_client.generate_response = AsyncMock(return_value=ProviderResponse(text="Fallback OK", model="gemini-3.8-flash"))
    resp_fb = await provider.generate_response(prompt="Fallback")
    assert resp_fb.text == "Fallback OK"
    provider.close()

    # Direct SDK unit check if installed
    if provider._sdk_client is not None:
        sdk_client = AntigravitySDKClient(data_dir=tmp_path / "sdk_agy", default_model="gemini-3.8-flash")
        mock_response = MagicMock(text=AsyncMock(return_value="SDK OK"), stop_reason="stop", usage_metadata=None)
        mock_agent = MagicMock(chat=AsyncMock(return_value=mock_response), conversation_id="sdk_1")
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)
        with patch("bot.antigravity.sdk_client.Agent", return_value=mock_agent):
            sdk_resp = await sdk_client.generate_response(prompt="Hi SDK")
        assert sdk_resp.text == "SDK OK"


@pytest.mark.asyncio
async def test_per_model_circuit_breaker_and_watch_log_quota_warning(tmp_path: Path):
    provider = AntigravityProvider(
        AntigravityConfig(
            data_dir=tmp_path / "agy",
            enable_sdk=False,
            enable_cli_fallback=True,
            rate_limit_max_retries=0,
            rate_limit_cooldown_seconds=300.0,
        )
    )
    assert provider._cli_client is not None
    provider._cli_client.generate_response = AsyncMock(
        side_effect=AntigravityRateLimitError("429 Too Many Requests: retry_after=120s")
    )

    with pytest.raises(AntigravityRateLimitError):
        await provider.generate_response(prompt="Hi", model="gemini-4-hydrogen")
    assert provider.is_rate_limited("gemini-4-hydrogen") is True
    # Other models remain unblocked!
    assert provider.is_rate_limited("gemini-3.8-flash") is False

    provider._cli_client.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Flash still works", model="gemini-3.8-flash")
    )
    ok_resp = await provider.generate_response(prompt="Hi flash", model="gemini-3.8-flash")
    assert ok_resp.text == "Flash still works"

    # Historical log skip + soft quota warning forwarding
    gemini_dir = tmp_path / ".gemini" / "antigravity-cli"
    gemini_dir.mkdir(parents=True)
    cli_log = gemini_dir / "cli.log"
    cli_log.write_text("RESOURCE_EXHAUSTED: historical error\n", encoding="utf-8")
    old_mtime = time.time() - 60.0
    os.utime(cli_log, (old_mtime, old_mtime))

    stop_ev = threading.Event()
    errors: list[str] = []
    warnings: list[tuple[str | None, float | None, str | None]] = []
    t = threading.Thread(
        target=watch_log_for_errors,
        args=(gemini_dir, stop_ev, errors, None),
        kwargs={"on_quota_warning": lambda m, p, r: warnings.append((m, p, r))},
        daemon=True,
    )
    t.start()
    time.sleep(0.3)
    assert errors == []
    with cli_log.open("a", encoding="utf-8") as fh:
        fh.write("[quota] model gemini-4-hydrogen is low on quota (15% remaining, resets at 2026-11-25T00:00:00Z)\n")
        fh.flush()
    deadline = time.monotonic() + 2.0
    while not warnings and time.monotonic() < deadline:
        time.sleep(0.05)
    stop_ev.set()
    t.join(timeout=1.0)
    assert len(warnings) == 1 and warnings[0][0] == "gemini-4-hydrogen" and warnings[0][1] == 15.0
    provider.close()


def test_chat_and_image_pre_tool_hook_firewalls(tmp_path: Path):
    # 1. Chat & zero-tool PreToolUse firewall
    _, chat_gate = write_chat_pre_tool_hook(tmp_path / "chat_sb")
    assert _run_hook(chat_gate, json.dumps({"toolCall": {"name": "search_web", "args": {}}}))["decision"] == "allow"
    assert _run_hook(chat_gate, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": ".env"}}}))["decision"] == "deny"
    assert _run_hook(chat_gate, "{malformed")["decision"] == "deny"

    _, zero_gate = write_chat_pre_tool_hook(tmp_path / "zero_sb", allowed_tools=frozenset())
    assert _run_hook(zero_gate, json.dumps({"toolCall": {"name": "search_web", "args": {}}}))["decision"] == "deny"

    # 2. Image PreToolUse firewall (with staged inputs + symlink escape rejection)
    req_cwd = tmp_path / "img_req"
    inputs_dir = req_cwd / "inputs"
    inputs_dir.mkdir(parents=True)
    staged = inputs_dir / "ref_0.jpg"
    staged.write_bytes(_make_image_bytes("JPEG"))
    outside = tmp_path / "secret.env"
    outside.write_text("SECRET=1", encoding="utf-8")
    sym_escape = inputs_dir / "escape.jpg"
    sym_escape.symlink_to(outside)

    _, gate_no_in = write_image_pre_tool_hook(req_cwd, allow_staged_inputs=False)
    assert _run_hook(gate_no_in, json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "image-generator"}}}))["decision"] == "allow"
    assert _run_hook(gate_no_in, json.dumps({"toolCall": {"name": "invoke_subagent", "args": {"TypeName": "self"}}}))["decision"] == "deny"
    assert _run_hook(gate_no_in, json.dumps({"toolCall": {"name": "generate_image", "args": {"Prompt": "cat"}}}))["decision"] == "allow"
    assert _run_hook(gate_no_in, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(staged)}}}))["decision"] == "deny"
    assert _run_hook(gate_no_in, json.dumps({"toolCall": {"name": "run_command", "args": {}}}))["decision"] == "deny"

    _, gate_with_in = write_image_pre_tool_hook(req_cwd, allow_staged_inputs=True)
    assert _run_hook(gate_with_in, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(staged)}}}))["decision"] == "allow"
    assert _run_hook(gate_with_in, json.dumps({"toolCall": {"name": "generate_image", "args": {"ImagePaths": [str(staged)]}}}))["decision"] == "allow"
    assert _run_hook(gate_with_in, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(sym_escape)}}}))["decision"] == "deny"
    assert _run_hook(gate_with_in, json.dumps({"toolCall": {"name": "read_file", "args": {"Path": str(outside)}}}))["decision"] == "deny"


@pytest.mark.asyncio
async def test_cli_and_provider_generate_image_lifecycle_and_errors(tmp_path: Path):
    data_dir = tmp_path / "agy_home"
    out_dir = tmp_path / "img_tmp"
    conv_id = "conv-img-ok-01"
    jpg_bytes = _make_image_bytes("JPEG", (64, 48))
    ref_src = tmp_path / "ref.png"
    ref_src.write_bytes(_make_image_bytes("PNG", (20, 20)))
    captured: dict[str, Any] = {}

    def fake_ok_popen(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        req_cwd = Path(cwd)
        captured["cwd"] = req_cwd
        captured["hooks_ok"] = (req_cwd / ".agents" / "hooks.json").is_file()
        brain_dir = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "brain" / conv_id
        brain_dir.mkdir(parents=True, exist_ok=True)
        out_img = brain_dir / "astro_cat.jpg"
        out_img.write_bytes(jpg_bytes)
        captured["out_img"] = out_img
        return _MockPopen([
            json.dumps({"event": "init", "conversation_id": conv_id}),
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "conversation_id": conv_id,
                    "step_type": "subagent",
                    "tool_name": "invoke_subagent",
                    "subagent_info": {"subagents": [{"type_name": "image-generator"}]},
                },
            }),
            json.dumps({
                "event": "result",
                "result": {"conversation_id": conv_id, "status": "SUCCESS", "response": "astro_cat.jpg"},
            }),
        ])

    cli = AntigravityCLIClient(data_dir=data_dir, binary="/mock/bin/agy")
    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_ok_popen):
        res = await cli.generate_image(
            "astro cat",
            aspect_ratio="16:9",
            reference_images=[MediaAttachment(file_path=ref_src, mime_type="image/png")],
            output_dir=out_dir,
            cleanup_artifacts=True,
        )
    assert isinstance(res, GeneratedImageResult)
    assert res.image_bytes == jpg_bytes
    assert res.mime_type == "image/jpeg"
    assert (res.width, res.height) == (64, 48)
    assert captured["hooks_ok"] is True
    assert not captured["cwd"].exists()
    assert not captured["out_img"].exists()

    # Policy refusal
    with patch(
        "bot.antigravity.cli_client.popen_isolated_session",
        return_value=_MockPopen([
            json.dumps({"event": "init", "conversation_id": "conv-refuse"}),
            json.dumps({"event": "result", "result": {"conversation_id": "conv-refuse", "status": "SUCCESS", "response": "I cannot generate this due to safety policy."}}),
        ]),
    ):
        with pytest.raises(ImagePolicyRefusalError):
            await cli.generate_image("unsafe", output_dir=out_dir)

    # Timeout
    with patch(
        "bot.antigravity.cli_client.popen_isolated_session",
        return_value=_MockPopen([], block_until_killed=True),
    ):
        with pytest.raises(ImageTimeoutError):
            await cli.generate_image("slow", timeout_seconds=0.12, output_dir=out_dir)

    # Unauthorized subagent -> AntigravityNativeToolAbort
    with patch(
        "bot.antigravity.cli_client.popen_isolated_session",
        return_value=_MockPopen([
            json.dumps({"event": "init", "conversation_id": "conv-bad"}),
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "conversation_id": "conv-bad",
                    "step_type": "subagent",
                    "tool_name": "invoke_subagent",
                    "subagent_info": {"subagents": [{"type_name": "self"}]},
                },
            }),
        ]),
    ):
        with pytest.raises(AntigravityNativeToolAbort):
            await cli.generate_image("bad subagent", output_dir=out_dir)

    # Corrupt bytes -> ImageValidationError
    def fake_corrupt(cmd: list[str], *, cwd: str | Path, env: dict[str, str]):
        bdir = Path(env["HOME"]) / ".gemini" / "antigravity-cli" / "brain" / "conv-corrupt"
        bdir.mkdir(parents=True, exist_ok=True)
        (bdir / "bad.jpg").write_bytes(b"NOT_JPEG")
        return _MockPopen([
            json.dumps({"event": "init", "conversation_id": "conv-corrupt"}),
            json.dumps({"event": "result", "result": {"conversation_id": "conv-corrupt", "status": "SUCCESS", "response": "bad.jpg"}}),
        ])

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_corrupt):
        with pytest.raises(ImageValidationError):
            await cli.generate_image("corrupt", output_dir=out_dir)

    # Provider routing + circuit breaker on generate_image
    provider = AntigravityProvider(AntigravityConfig(data_dir=tmp_path / "p_img", enable_sdk=False))
    provider._cli_client = MagicMock()
    provider._cli_client.generate_image = AsyncMock(side_effect=AntigravityBillingError("credits balance is too low"))
    with pytest.raises(AntigravityBillingError):
        await provider.generate_image("planet", model="gemini-4-hydrogen")
    assert provider.is_rate_limited("gemini-4-hydrogen") is True
    provider._cli_client = None
    provider.reset_cooldown()
    with pytest.raises(ImageGenerationError):
        await provider.generate_image("cloud")
    cli.close()
