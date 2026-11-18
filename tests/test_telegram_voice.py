"""Unit tests for Telegram voice sending via Bot API and Telethon Userbot."""

import json
import sys
import time
import urllib.request
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from telethon.tl.types import DocumentAttributeAudio

from bot.telegram.bot_api_client import TelegramBotAPIClient
from bot.telegram.client import TelegramUserbot


@pytest.mark.asyncio
async def test_bot_api_send_voice_multipart_and_record_voice_action(tmp_path, monkeypatch):
    voice_file = tmp_path / "reply.ogg"
    ogg_bytes = b"OggS\x00\x02fake-opus-voice-payload"
    voice_file.write_bytes(ogg_bytes)

    captured: dict = {}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"message_id": 501}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=60.0):
        captured["url"] = req.full_url
        captured["data"] = req.data
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    bot_client = TelegramBotAPIClient(
        bot_token="123456:TEST_TOKEN",
        typing_min_seconds=0.01,
        typing_max_seconds=0.02,
    )
    bot_client.call_api = AsyncMock(return_value={"ok": True, "result": True})

    res = await bot_client.send_voice(
        chat_id=-100123,
        voice_path=voice_file,
        caption="Голосовой ответ",
        reply_to_msg_id=55,
        duration=4.6,
        simulate_typing=True,
        timeout=45.0,
    )

    assert res["ok"] is True
    assert res["result"]["message_id"] == 501
    bot_client.call_api.assert_awaited_once_with(
        "sendChatAction",
        {"chat_id": -100123, "action": "record_voice"},
        timeout=10.0,
    )

    assert captured["url"] == "https://api.telegram.org/bot123456:TEST_TOKEN/sendVoice"
    assert captured["timeout"] == 45.0
    content_type = captured["headers"]["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1].encode("utf-8")

    body: bytes = captured["data"]
    assert b'name="chat_id"\r\n\r\n-100123\r\n' in body
    assert 'name="caption"\r\n\r\nГолосовой ответ\r\n'.encode("utf-8") in body
    assert b'name="reply_to_message_id"\r\n\r\n55\r\n' in body
    assert b'name="duration"\r\n\r\n5\r\n' in body
    assert b'name="voice"; filename="reply.ogg"\r\n' in body
    assert b"Content-Type: audio/ogg\r\n\r\n" in body
    assert ogg_bytes in body
    assert body.endswith(b"--" + boundary + b"--\r\n")


@pytest.mark.asyncio
async def test_bot_api_send_voice_retries_on_connection_reset(tmp_path, monkeypatch):
    voice_file = tmp_path / "voice.ogg"
    voice_file.write_bytes(b"OggS-retry-test")

    attempts = {"count": 0}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"message_id": 903}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=60.0):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")
    res = await bot_client.send_voice(
        chat_id=-100123,
        voice_path=voice_file,
        simulate_typing=False,
    )

    assert attempts["count"] == 2
    assert res["result"]["message_id"] == 903


@pytest.mark.asyncio
async def test_userbot_send_voice_telethon_mode(tmp_path):
    session_file = tmp_path / "test.session"
    voice_file = tmp_path / "voice.ogg"
    voice_file.write_bytes(b"OggS-telethon-voice")

    mock_raw_client = MagicMock()
    mock_raw_client.send_file = AsyncMock(return_value=MagicMock(id=889))

    action_cm = MagicMock()
    action_cm.__aenter__ = AsyncMock()
    action_cm.__aexit__ = AsyncMock()
    mock_raw_client.action = MagicMock(return_value=action_cm)

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        typing_min_seconds=0.01,
        typing_max_seconds=0.02,
        client=mock_raw_client,
    )

    res = await userbot.send_voice(
        chat_id=100,
        voice_path=voice_file,
        reply_to_msg_id=123,
        duration=3.2,
        simulate_typing=True,
    )

    assert res.id == 889
    mock_raw_client.action.assert_called_with(100, "record-voice")
    mock_raw_client.send_file.assert_awaited_once()
    call_args = mock_raw_client.send_file.await_args
    assert call_args.args == (100, str(voice_file.resolve()))
    assert call_args.kwargs["caption"] is None
    assert call_args.kwargs["reply_to"] == 123
    assert call_args.kwargs["voice_note"] is True
    attrs = call_args.kwargs["attributes"]
    assert len(attrs) == 1
    assert isinstance(attrs[0], DocumentAttributeAudio)
    assert attrs[0].duration == 3
    assert attrs[0].voice is True


@pytest.mark.asyncio
async def test_userbot_send_voice_bot_api_mode_routing(tmp_path):
    session_file = tmp_path / "test.session"
    voice_file = tmp_path / "voice.ogg"
    voice_file.write_bytes(b"OggS-botapi-route")

    mock_raw_client = MagicMock()
    mock_raw_client.send_file = AsyncMock()

    mock_bot_api = MagicMock()
    mock_bot_api.send_voice = AsyncMock(return_value={"ok": True, "result": {"message_id": 778}})

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[-100123],
        bot_names=["Jeff"],
        client=mock_raw_client,
        bot_api_client=mock_bot_api,
    )

    res = await userbot.send_voice(
        chat_id=-100123,
        voice_path=voice_file,
        caption=None,
        reply_to_msg_id=42,
        duration=4.0,
        simulate_typing=False,
        timeout=30.0,
    )

    assert res["result"]["message_id"] == 778
    mock_bot_api.send_voice.assert_awaited_once_with(
        chat_id=-100123,
        voice_path=voice_file.resolve(),
        caption=None,
        reply_to_msg_id=42,
        duration=4.0,
        simulate_typing=False,
        timeout=30.0,
    )
    mock_raw_client.send_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_voice_validation_errors(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()
    mock_raw_client.send_file = AsyncMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        client=mock_raw_client,
    )
    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")

    missing = tmp_path / "missing.ogg"
    with pytest.raises(FileNotFoundError):
        await userbot.send_voice(chat_id=100, voice_path=missing, simulate_typing=False)
    with pytest.raises(FileNotFoundError):
        await bot_client.send_voice(chat_id=100, voice_path=missing, simulate_typing=False)

    empty = tmp_path / "empty.ogg"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        await userbot.send_voice(chat_id=100, voice_path=empty, simulate_typing=False)
    with pytest.raises(ValueError, match="empty"):
        await bot_client.send_voice(chat_id=100, voice_path=empty, simulate_typing=False)

    mock_raw_client.send_file.assert_not_awaited()
