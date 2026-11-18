"""Tests for error classification, backoff policies, SDK client, CLI client, and AntigravityProvider."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.antigravity import (
    AntigravityCLIClient,
    AntigravityConfig,
    AntigravityProvider,
    AntigravitySDKClient,
    ErrorCategory,
    calculate_backoff,
    classify_error,
)
from bot.core.models import MediaAttachment, ProviderResponse


class _MockPopen:
    """Mock subprocess.Popen for CLI client testing."""

    def __init__(self, stdout_lines: list[str], returncode: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(stdout_lines) + "\n")
        self.stderr = io.StringIO("")
        self.pid = 4242
        self.returncode = returncode
        self._terminated = False

    def poll(self) -> int | None:
        return self.returncode if self._terminated else None

    def terminate(self) -> None:
        self._terminated = True

    def kill(self) -> None:
        self._terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self._terminated = True
        return self.returncode


def test_error_classification_pipeline():
    # 1. 429 Rate limit
    v1 = classify_error("429 Too Many Requests: Resource has been exhausted")
    assert v1.category == ErrorCategory.RATE_LIMIT
    assert v1.retryable is True
    assert v1.should_fallback is True

    # 2. Server error / output token cutoff
    v2 = classify_error("Previous response was cut off because it exceeded the output token limit")
    assert v2.category == ErrorCategory.SERVER_ERROR
    assert v2.retryable is True
    assert v2.should_compress is False

    # 3. Context overflow
    v3 = classify_error("max_trajectory_tokens exceeded: prompt is too long")
    assert v3.category == ErrorCategory.CONTEXT_OVERFLOW
    assert v3.retryable is True
    assert v3.should_compress is True

    # 4. Billing
    v4 = classify_error("AI credits balance is too low to process this request")
    assert v4.category == ErrorCategory.BILLING
    assert v4.retryable is False
    assert v4.should_fallback is True

    # 5. Auth permanent
    v5 = classify_error("OAuth token expired, please re-authenticate")
    assert v5.category == ErrorCategory.AUTH_PERMANENT
    assert v5.retryable is False
    assert v5.should_fallback is True


def test_exponential_backoff_calculation():
    d0 = calculate_backoff(0, base_delay=1.0, max_delay=30.0, jitter=False)
    d1 = calculate_backoff(1, base_delay=1.0, max_delay=30.0, jitter=False)
    d2 = calculate_backoff(2, base_delay=1.0, max_delay=30.0, jitter=False)
    d_max = calculate_backoff(10, base_delay=1.0, max_delay=30.0, jitter=False)

    assert d0 == 1.0
    assert d1 == 2.0
    assert d2 == 4.0
    assert d_max == 30.0


@pytest.mark.asyncio
async def test_cli_client_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = AntigravityCLIClient(data_dir=tmpdir, binary="/mock/bin/agy")

        stdout_lines = [
            json.dumps({"event": "init", "conversation_id": "conv_cli_01"}),
            json.dumps({
                "event": "step_update",
                "step_update": {
                    "text_delta": "Response from CLI fallback.",
                    "thinking_delta": "Thinking step...",
                    "usage": {"input_tokens": 50, "output_tokens": 15},
                },
            }),
            json.dumps({"event": "result", "result": {"status": "SUCCESS"}}),
        ]
        mock_proc = _MockPopen(stdout_lines)

        with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=mock_proc):
            resp = await cli.generate_response(prompt="Hello CLI")

        assert resp.text == "Response from CLI fallback."
        assert resp.thinking == "Thinking step..."
        assert resp.conversation_id == "conv_cli_01"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 50
        assert resp.usage.completion_tokens == 15
        cli.close()


@pytest.mark.asyncio
async def test_sdk_client_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        sdk_client = AntigravitySDKClient(data_dir=tmpdir, default_model="gemini-3.8-flash")

        # Mock Agent and ChatResponse
        mock_response = MagicMock()
        mock_response.text = AsyncMock(return_value="Direct SDK response text.")
        mock_response.stop_reason = "stop"

        async def _mock_thoughts():
            yield "SDK reasoning token 1"
            yield "SDK reasoning token 2"

        mock_response.thoughts = _mock_thoughts()
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=120,
            candidates_token_count=30,
            total_token_count=150,
            thoughts_token_count=10,
            cached_content_token_count=20,
        )

        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(return_value=mock_response)
        mock_agent.conversation_id = "sdk_conv_99"
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)

        with patch("bot.antigravity.sdk_client.Agent", return_value=mock_agent):
            resp = await sdk_client.generate_response(
                prompt="Hello SDK",
                system_instructions="You are a helpful assistant.",
            )

        assert resp.text == "Direct SDK response text."
        assert "SDK reasoning token 1" in (resp.thinking or "")
        assert resp.conversation_id == "sdk_conv_99"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 120
        assert resp.usage.completion_tokens == 30
        assert resp.usage.reasoning_tokens == 10


@pytest.mark.asyncio
async def test_antigravity_provider_sdk_primary_with_cli_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-api-key")
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AntigravityConfig(
            data_dir=Path(tmpdir),
            enable_sdk=True,
            enable_cli_fallback=True,
            max_retries=1,
        )
        provider = AntigravityProvider(config=cfg)
        assert provider._sdk_client is not None

        # 1. SDK Success path
        mock_sdk_resp = ProviderResponse(text="SDK Answer", model="gemini-3.8-flash")
        provider._sdk_client.generate_response = AsyncMock(return_value=mock_sdk_resp)

        resp1 = await provider.generate_response(prompt="Test Prompt")
        assert resp1.text == "SDK Answer"
        provider._sdk_client.generate_response.assert_awaited_once()

        # 2. SDK Failure -> CLI Fallback path
        provider._sdk_client.generate_response = AsyncMock(
            side_effect=RuntimeError("SDK service unavailable: 503")
        )
        mock_cli_resp = ProviderResponse(text="CLI Fallback Answer", model="gemini-3.8-flash")
        provider._cli_client.generate_response = AsyncMock(return_value=mock_cli_resp)

        resp2 = await provider.generate_response(prompt="Fallback Prompt")
        assert resp2.text == "CLI Fallback Answer"
        provider._cli_client.generate_response.assert_awaited_once()

        provider.close()


@pytest.mark.asyncio
async def test_antigravity_provider_skips_sdk_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AntigravityConfig(
            data_dir=Path(tmpdir),
            enable_sdk=True,
            enable_cli_fallback=True,
        )
        provider = AntigravityProvider(config=cfg)
        assert provider._sdk_client is None
        assert provider._cli_client is not None

        provider._cli_client.generate_response = AsyncMock(
            return_value=ProviderResponse(text="Direct CLI response", model="gemini-4-hydrogen")
        )
        resp = await provider.generate_response(prompt="Hello")
        assert resp.text == "Direct CLI response"
        provider._cli_client.generate_response.assert_awaited_once()
        provider.close()


@pytest.mark.asyncio
async def test_cli_client_stateless_turns_use_fresh_cascades_and_disable_slash_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        cli = AntigravityCLIClient(data_dir=tmpdir, binary="/mock/bin/agy")

        spawned_cmds: list[list[str]] = []
        turn_counter = {"n": 0}

        def fake_popen(cmd, *, cwd, env):
            spawned_cmds.append(list(cmd))
            turn_counter["n"] += 1
            cid = f"conv_fresh_{turn_counter['n']}"
            return _MockPopen([
                json.dumps({"event": "init", "conversation_id": cid}),
                json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": f"Reply {turn_counter['n']}"}}),
            ])

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
            r1 = await cli.generate_response(prompt="Turn 1", conversation_id=None)
            r2 = await cli.generate_response(prompt="Turn 2", conversation_id=None)

        assert r1.conversation_id == "conv_fresh_1"
        assert r2.conversation_id == "conv_fresh_2"
        assert len(spawned_cmds) == 2
        for cmd in spawned_cmds:
            assert "--disable-slash-commands" in cmd
            assert "--input-format" in cmd
            assert "--output-format" in cmd
        assert cli._worker_proc is None
        cli.close()


def test_cli_client_idle_worker_timeout_terminates_worker():
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        cli = AntigravityCLIClient(
            data_dir=tmpdir,
            binary="/mock/bin/agy",
            worker_idle_timeout_seconds=0.05,
        )
        mock_proc = _MockPopen([])
        spawned_cmds: list[list[str]] = []

        def fake_popen(cmd, *, cwd, env):
            spawned_cmds.append(list(cmd))
            return mock_proc

        with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
            proc, _drainer, _gdir = cli._ensure_worker("gemini-4-hydrogen", "low")
            assert proc is mock_proc
            assert "--disable-slash-commands" in spawned_cmds[0]
            assert mock_proc._terminated is False

            deadline = time.monotonic() + 2.0
            while cli._worker_proc is not None and time.monotonic() < deadline:
                time.sleep(0.02)

            assert cli._worker_proc is None
            assert mock_proc._terminated is True
        cli.close()


@pytest.mark.asyncio
async def test_antigravity_provider_rate_limit_circuit_breaker_and_no_hammering(monkeypatch):
    from bot.antigravity.errors import (
        AntigravityBillingError,
        AntigravityError,
        AntigravityRateLimitError,
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AntigravityConfig(
            data_dir=Path(tmpdir),
            enable_sdk=False,
            enable_cli_fallback=True,
            max_retries=3,
            rate_limit_max_retries=0,
            rate_limit_cooldown_seconds=300.0,
        )
        provider = AntigravityProvider(config=cfg)
        assert provider.is_rate_limited() is False
        assert provider.cooldown_remaining() == 0.0

        cli_mock = AsyncMock(
            side_effect=AntigravityRateLimitError(
                "429 Too Many Requests: RESOURCE_EXHAUSTED, retry_after=120s"
            )
        )
        provider._cli_client.generate_response = cli_mock

        # 1. First failure triggers 0 retries (1 attempt, not 3x!) and opens circuit breaker
        with pytest.raises(AntigravityRateLimitError):
            await provider.generate_response(prompt="Hello")
        assert cli_mock.await_count == 1
        assert provider.is_rate_limited() is True
        assert 100.0 <= provider.cooldown_remaining() <= 120.5

        # 2. Second call fails fast without touching CLI/network
        with pytest.raises(AntigravityError):
            await provider.generate_response(prompt="Hello again")
        assert cli_mock.await_count == 1

        # 3. Reset and verify BILLING also opens circuit breaker
        provider.reset_cooldown()
        assert provider.is_rate_limited() is False
        billing_mock = AsyncMock(
            side_effect=AntigravityBillingError("AI credits balance is too low")
        )
        provider._cli_client.generate_response = billing_mock

        with pytest.raises(AntigravityBillingError):
            await provider.generate_response(prompt="Billing check")
        assert billing_mock.await_count == 1
        assert provider.is_rate_limited() is True
        assert provider.cooldown_remaining() > 250.0

        with pytest.raises(AntigravityBillingError):
            await provider.generate_response(prompt="Blocked by billing cooldown")
        assert billing_mock.await_count == 1

        provider.close()


@pytest.mark.asyncio
async def test_antigravity_provider_media_attachments():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = AntigravityConfig(data_dir=Path(tmpdir), enable_sdk=False, enable_cli_fallback=True)
        provider = AntigravityProvider(config=cfg)

        mock_cli_resp = ProviderResponse(text="Processed photo and question.")
        provider._cli_client.generate_response = AsyncMock(return_value=mock_cli_resp)

        attachment = MediaAttachment(
            mime_type="image/jpeg",
            description="A photo of a white cat sitting on a table",
        )
        resp = await provider.generate_response(
            prompt="What is in the image?",
            media_attachments=[attachment],
        )

        assert resp.text == "Processed photo and question."
        called_prompt = provider._cli_client.generate_response.call_args[1]["prompt"]
        assert "A photo of a white cat sitting on a table" in called_prompt
        assert "What is in the image?" in called_prompt
        called_atts = provider._cli_client.generate_response.call_args[1]["media_attachments"]
        assert called_atts == [attachment]

        provider.close()


class _InteractiveAcpPopen:
    """Mock subprocess.Popen simulating an `agy acp --stdio` JSON-RPC 2.0 server."""

    def __init__(self, *, expected_mime: str = "image/jpeg") -> None:
        self.expected_mime = expected_mime
        self.sent_requests: list[dict] = []
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

        self.stdin = _StdinProxy()
        self.stdout = _StdoutProxy()

    def _handle_client_line(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent_requests.append(msg)
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            self._responses.append(
                json.dumps({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": 1}}) + "\n"
            )
        elif method == "session/new":
            self._responses.append(
                json.dumps({
                    "jsonrpc": "2.0",
                    "id": mid,
                    "result": {
                        "sessionId": "acp_sess_1",
                        "models": {
                            "currentModelId": "gemini-4-argon-medium",
                            "availableModels": [
                                {"modelId": "gemini-3.8-flash-low", "name": "Gemini 3.8 Flash (Low)"},
                                {"modelId": "gemini-4-hydrogen-medium", "name": "Gemini 4 Hydrogen (Medium)"},
                            ],
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


@pytest.mark.asyncio
async def test_cli_client_acp_image_vision_jpeg_and_webp(tmp_path: Path):
    from PIL import Image

    jpeg_path = tmp_path / "cat.jpg"
    Image.new("RGB", (16, 16), (255, 80, 0)).save(jpeg_path, format="JPEG")

    webp_path = tmp_path / "sticker.webp"
    Image.new("RGBA", (16, 16), (0, 200, 100, 255)).save(webp_path, format="WEBP")

    cli = AntigravityCLIClient(
        data_dir=tmp_path,
        binary="/mock/bin/agy",
        default_model="gemini-3.8-flash",
        default_effort="low",
    )

    # 1. JPEG image attachment routed to `agy acp --stdio`
    acp_jpeg = _InteractiveAcpPopen(expected_mime="image/jpeg")
    cmds: list[list[str]] = []

    def fake_popen(cmd, *, cwd, env):
        cmds.append(list(cmd))
        return acp_jpeg

    with patch("bot.antigravity.cli_client.popen_isolated_session", side_effect=fake_popen):
        resp1 = await cli.generate_response(
            prompt="Что на картинке?",
            system_instructions="Будь кратким",
            media_attachments=[MediaAttachment(file_path=jpeg_path, mime_type="image/jpeg")],
        )

    assert cmds[0] == ["/mock/bin/agy", "acp", "--stdio"]
    assert resp1.text == "На фото рыжий кот."
    assert resp1.conversation_id == "acp_sess_1"
    assert resp1.usage is not None and resp1.usage.prompt_tokens == 250

    prompt_req = [r for r in acp_jpeg.sent_requests if r.get("method") == "session/prompt"][0]
    blocks = prompt_req["params"]["prompt"]
    assert blocks[0] == {"type": "text", "text": "Что на картинке?"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["mimeType"] == "image/jpeg"
    assert len(blocks[1]["data"]) > 10

    new_req = [r for r in acp_jpeg.sent_requests if r.get("method") == "session/new"][0]
    assert new_req["params"]["_meta"]["antigravity"]["systemPrompt"]["append"] == "Будь кратким"

    set_model_req = [r for r in acp_jpeg.sent_requests if r.get("method") == "session/set_model"][0]
    assert set_model_req["params"]["modelId"] == "gemini-3.8-flash-low"

    # 2. WebP sticker normalized to PNG for ACP image compatibility
    acp_webp = _InteractiveAcpPopen(expected_mime="image/png")
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=acp_webp):
        await cli.generate_response(
            prompt="Оцени стикер",
            media_attachments=[MediaAttachment(file_path=webp_path, mime_type="image/webp")],
        )
    webp_req = [r for r in acp_webp.sent_requests if r.get("method") == "session/prompt"][0]
    assert webp_req["params"]["prompt"][1]["mimeType"] == "image/png"
    cli.close()
