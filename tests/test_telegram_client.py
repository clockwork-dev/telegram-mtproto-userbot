"""Unit tests for Telegram MTProto Userbot client, permissions, and event handling."""

import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from telethon.tl.functions.messages import SendReactionRequest

from bot.telegram.auth import enforce_session_permissions
from bot.telegram.client import TelegramUserbot, calculate_typing_delay
from bot.telegram.handlers import IncomingMessage


def test_calculate_typing_delay():
    # 300 cpm = 5 chars/sec
    # Short text (10 chars -> 2.0s, clamped to min 1.5s + jitter)
    delay_short = calculate_typing_delay("Hello", cpm=300, min_seconds=1.5, max_seconds=8.0, jitter=0.0)
    assert delay_short == 1.5

    # Medium text (100 chars -> 20.0s, clamped to max 8.0s + jitter)
    delay_long = calculate_typing_delay("A" * 100, cpm=300, min_seconds=1.0, max_seconds=8.0, jitter=0.0)
    assert delay_long == 8.0

    # Intermediate text (20 chars -> 4.0s)
    delay_mid = calculate_typing_delay("A" * 20, cpm=300, min_seconds=1.0, max_seconds=8.0, jitter=0.5)
    assert delay_mid == 4.5


def test_session_permissions_enforcement(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(parents=True)
    session_file = session_dir / "userbot.session"
    session_file.write_text("dummy session data")

    # Give them permissive permissions first (0777 / 0666)
    os.chmod(session_dir, 0o777)
    os.chmod(session_file, 0o666)

    # Enforce secure permissions
    enforce_session_permissions(session_file)

    dir_mode = stat.S_IMODE(session_dir.stat().st_mode)
    file_mode = stat.S_IMODE(session_file.stat().st_mode)

    # Directory must be 0700 (rwx------)
    assert dir_mode == 0o700
    # Session file must be 0600 (rw-------)
    assert file_mode == 0o600


@pytest.mark.asyncio
async def test_client_lifecycle_connect_and_disconnect(tmp_path):
    session_file = tmp_path / "test.session"
    session_file.write_text("session")

    connected_state = [False]

    def fake_is_connected():
        return connected_state[0]

    async def fake_connect():
        connected_state[0] = True

    async def fake_disconnect():
        connected_state[0] = False

    mock_raw_client = MagicMock()
    mock_raw_client.is_connected = MagicMock(side_effect=fake_is_connected)
    mock_raw_client.connect = AsyncMock(side_effect=fake_connect)
    mock_raw_client.disconnect = AsyncMock(side_effect=fake_disconnect)
    mock_raw_client.is_user_authorized = AsyncMock(return_value=True)

    mock_me = MagicMock()
    mock_me.id = 123456
    mock_me.username = "test_userbot"
    mock_raw_client.get_me = AsyncMock(return_value=mock_me)

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )

    assert not userbot.is_connected()
    await userbot.start()

    assert userbot.bot_id == 123456
    assert userbot.bot_username == "test_userbot"
    mock_raw_client.connect.assert_awaited_once()

    await userbot.stop()
    mock_raw_client.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_client_start_raises_if_unauthorized(tmp_path):
    session_file = tmp_path / "unauth.session"
    session_file.write_text("session")

    mock_raw_client = MagicMock()
    mock_raw_client.is_connected = MagicMock(return_value=True)
    mock_raw_client.connect = AsyncMock()
    mock_raw_client.is_user_authorized = AsyncMock(return_value=False)

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )

    with pytest.raises(RuntimeError, match="not authorized"):
        await userbot.start()


@pytest.mark.asyncio
async def test_client_whitelist_filter_drops_unallowed_chats(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100, 200],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )
    userbot.bot_id = 999

    received_messages = []

    async def sample_handler(msg: IncomingMessage):
        received_messages.append(msg)

    userbot.register_message_handler(sample_handler)

    # Event in non-whitelisted chat
    event_blocked = MagicMock()
    event_blocked.chat_id = 555
    event_blocked.out = False
    event_blocked.sender_id = 111
    await userbot._on_new_message(event_blocked)
    assert len(received_messages) == 0

    # Event in whitelisted chat
    event_allowed = MagicMock()
    event_allowed.id = 42
    event_allowed.chat_id = 100
    event_allowed.out = False
    event_allowed.sender_id = 111
    event_allowed.raw_text = "Hello there"
    event_allowed.is_reply = False
    event_allowed.reply_to_msg_id = None
    event_allowed.media = None
    event_allowed.sender = None
    await userbot._on_new_message(event_allowed)

    assert len(received_messages) == 1
    assert received_messages[0].chat_id == 100
    assert received_messages[0].text == "Hello there"


@pytest.mark.asyncio
async def test_client_self_drop_filter(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )
    userbot.bot_id = 999

    received = []

    async def handler(msg: IncomingMessage):
        received.append(msg)

    userbot.register_message_handler(handler)

    # Outgoing message (event.out == True)
    event_out = MagicMock()
    event_out.chat_id = 100
    event_out.out = True
    event_out.sender_id = 999
    await userbot._on_new_message(event_out)
    assert len(received) == 0

    # Sender is bot ID
    event_self_sender = MagicMock()
    event_self_sender.chat_id = 100
    event_self_sender.out = False
    event_self_sender.sender_id = 999
    await userbot._on_new_message(event_self_sender)
    assert len(received) == 0


@pytest.mark.asyncio
async def test_client_send_reply_with_typing_and_quote(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()
    mock_raw_client.send_message = AsyncMock(return_value=MagicMock(id=777))

    # Action context manager mock
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
        typing_max_seconds=0.05,
        client=mock_raw_client,
    )

    result = await userbot.send_reply(
        chat_id=100,
        text="Quick answer",
        reply_to_msg_id=123,
        simulate_typing=True,
    )

    assert result.id == 777
    mock_raw_client.action.assert_called_with(100, "typing")
    mock_raw_client.send_message.assert_awaited_once_with(100, "Quick answer", reply_to=123)


@pytest.mark.asyncio
async def test_client_send_reaction_success(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = AsyncMock()
    mock_raw_client.get_input_entity = AsyncMock(return_value="peer_entity")

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )

    success = await userbot.send_reaction(chat_id=100, msg_id=456, emoji="🔥")
    assert success is True
    mock_raw_client.get_input_entity.assert_awaited_once_with(100)
    mock_raw_client.assert_awaited_once()

    # Verify SendReactionRequest was passed
    call_args = mock_raw_client.call_args[0][0]
    assert isinstance(call_args, SendReactionRequest)
    assert call_args.msg_id == 456
    assert call_args.reaction[0].emoticon == "🔥"


@pytest.mark.asyncio
async def test_client_send_reaction_failure_handled_gracefully(tmp_path):
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()
    mock_raw_client.get_input_entity = AsyncMock(side_effect=Exception("Network error"))

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[100],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )

    success = await userbot.send_reaction(chat_id=100, msg_id=456, emoji="🔥")
    assert success is False


@pytest.mark.asyncio
async def test_bot_api_send_reaction_emoji_normalization():
    """Verify TelegramBotAPIClient normalizes unsupported emojis like 😂 -> 🤣 and ❤️ -> ❤."""
    from bot.telegram.bot_api_client import TelegramBotAPIClient, normalize_reaction_emoji

    assert normalize_reaction_emoji("😂") == "🤣"
    assert normalize_reaction_emoji("❤️") == "❤"
    assert normalize_reaction_emoji("☺️") == "🥰"
    assert normalize_reaction_emoji("🔥") == "🔥"
    assert normalize_reaction_emoji("🫡") == "🫡"

    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")
    bot_client.call_api = AsyncMock(return_value={"ok": True, "result": True})

    ok1 = await bot_client.send_reaction(chat_id=-100123, msg_id=77, emoji="😂")
    assert ok1 is True
    bot_client.call_api.assert_awaited_once_with(
        "setMessageReaction",
        {
            "chat_id": -100123,
            "message_id": 77,
            "reaction": [{"type": "emoji", "emoji": "🤣"}],
        },
        timeout=10.0,
    )

    bot_client.call_api.reset_mock()
    ok2 = await bot_client.send_reaction(chat_id=-100123, msg_id=78, emoji="❤️")
    assert ok2 is True
    bot_client.call_api.assert_awaited_once_with(
        "setMessageReaction",
        {
            "chat_id": -100123,
            "message_id": 78,
            "reaction": [{"type": "emoji", "emoji": "❤"}],
        },
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_bot_api_send_document_multipart_payload_and_fields(tmp_path, monkeypatch):
    """Verify TelegramBotAPIClient.send_document encodes multipart/form-data correctly."""
    import json
    import urllib.request

    from bot.telegram.bot_api_client import TelegramBotAPIClient

    pdf_file = tmp_path / "Отчёт 2026.pdf"
    pdf_bytes = b"%PDF-1.4\n%fake-pdf-content\n"
    pdf_file.write_bytes(pdf_bytes)

    captured: dict = {}

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=120.0):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["data"] = req.data
        captured["timeout"] = timeout
        return FakeResponse(json.dumps({"ok": True, "result": {"message_id": 901}}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    bot_client = TelegramBotAPIClient(
        bot_token="123456:TEST_TOKEN",
        typing_min_seconds=0.01,
        typing_max_seconds=0.02,
    )
    bot_client.call_api = AsyncMock(return_value={"ok": True, "result": True})

    res = await bot_client.send_document(
        chat_id=-100123,
        file_path=pdf_file,
        caption="Вот PDF",
        reply_to_msg_id=55,
        simulate_typing=True,
        filename="final_report.pdf",
    )

    assert res["ok"] is True
    assert res["result"]["message_id"] == 901
    bot_client.call_api.assert_awaited_once_with(
        "sendChatAction",
        {"chat_id": -100123, "action": "upload_document"},
        timeout=10.0,
    )

    assert captured["url"] == "https://api.telegram.org/bot123456:TEST_TOKEN/sendDocument"
    assert captured["method"] == "POST"
    content_type = captured["headers"]["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1].encode("utf-8")

    body: bytes = captured["data"]
    assert b'name="chat_id"\r\n\r\n-100123\r\n' in body
    assert 'name="caption"\r\n\r\nВот PDF\r\n'.encode("utf-8") in body
    assert b'name="reply_to_message_id"\r\n\r\n55\r\n' in body
    assert b'name="document"; filename="final_report.pdf"\r\n' in body
    assert b"Content-Type: application/pdf\r\n\r\n" in body
    assert pdf_bytes in body
    assert body.endswith(b"--" + boundary + b"--\r\n")


@pytest.mark.asyncio
async def test_bot_api_send_document_retries_on_connection_reset(tmp_path, monkeypatch):
    """Verify send_document retries transient ConnectionResetError and succeeds."""
    import json
    import time
    import urllib.request

    from bot.telegram.bot_api_client import TelegramBotAPIClient

    doc_file = tmp_path / "notes.txt"
    doc_file.write_text("hello world", encoding="utf-8")

    attempts = {"count": 0}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps({"ok": True, "result": {"message_id": 902}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=120.0):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    bot_client = TelegramBotAPIClient(bot_token="123456:TEST_TOKEN")
    res = await bot_client.send_document(
        chat_id=-100123,
        file_path=doc_file,
        simulate_typing=False,
    )

    assert attempts["count"] == 2
    assert res["result"]["message_id"] == 902


@pytest.mark.asyncio
async def test_userbot_send_document_telethon_mode(tmp_path):
    """Verify TelegramUserbot.send_document uses Telethon send_file in Telethon mode."""
    session_file = tmp_path / "test.session"
    doc_file = tmp_path / "summary.docx"
    doc_file.write_bytes(b"PK\x03\x04fake-docx")

    mock_raw_client = MagicMock()
    mock_raw_client.send_file = AsyncMock(return_value=MagicMock(id=888))

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

    res = await userbot.send_document(
        chat_id=100,
        file_path=doc_file,
        caption="Готово",
        reply_to_msg_id=123,
        simulate_typing=True,
    )

    assert res.id == 888
    mock_raw_client.action.assert_called_with(100, "document")
    mock_raw_client.send_file.assert_awaited_once_with(
        100,
        str(doc_file.resolve()),
        caption="Готово",
        reply_to=123,
        force_document=True,
    )


@pytest.mark.asyncio
async def test_userbot_send_document_bot_api_mode_routing(tmp_path):
    """Verify TelegramUserbot.send_document routes to _bot_api_client.send_document when set."""
    session_file = tmp_path / "test.session"
    doc_file = tmp_path / "report.pdf"
    doc_file.write_bytes(b"%PDF-1.4\ncontent")

    mock_raw_client = MagicMock()
    mock_raw_client.send_file = AsyncMock()

    mock_bot_api = MagicMock()
    mock_bot_api.send_document = AsyncMock(return_value={"ok": True, "result": {"message_id": 777}})

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[-100123],
        bot_names=["Jeff"],
        client=mock_raw_client,
        bot_api_client=mock_bot_api,
    )

    res = await userbot.send_document(
        chat_id=-100123,
        file_path=doc_file,
        caption="Report",
        reply_to_msg_id=42,
        simulate_typing=False,
        filename="out.pdf",
    )

    assert res["result"]["message_id"] == 777
    mock_bot_api.send_document.assert_awaited_once_with(
        chat_id=-100123,
        file_path=doc_file.resolve(),
        caption="Report",
        reply_to_msg_id=42,
        simulate_typing=False,
        filename="out.pdf",
    )
    mock_raw_client.send_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_document_validation_errors(tmp_path):
    """Verify missing, directory, empty, and oversized files raise clear errors."""
    from bot.telegram.bot_api_client import TelegramBotAPIClient

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

    missing_file = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        await userbot.send_document(chat_id=100, file_path=missing_file, simulate_typing=False)
    with pytest.raises(FileNotFoundError):
        await bot_client.send_document(chat_id=100, file_path=missing_file, simulate_typing=False)

    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        await userbot.send_document(chat_id=100, file_path=empty_file, simulate_typing=False)

    with pytest.raises(ValueError, match="not a regular file"):
        await userbot.send_document(chat_id=100, file_path=tmp_path, simulate_typing=False)

    oversized_file = tmp_path / "huge.bin"
    with open(oversized_file, "wb") as f:
        f.truncate(46 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        await userbot.send_document(chat_id=100, file_path=oversized_file, simulate_typing=False)
    mock_raw_client.send_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_whitelisted_dm_accepted(tmp_path):
    """Verify whitelisted DM user (Rodion 382962362) is accepted and dispatched."""
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[],  # Groups disabled
        allowed_dm_user_ids=[382962362],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )
    userbot.bot_id = 999

    received_messages = []

    async def sample_handler(msg: IncomingMessage):
        received_messages.append(msg)

    userbot.register_message_handler(sample_handler)

    # DM event from whitelisted user Rodion
    event_rodion = MagicMock()
    event_rodion.id = 501
    event_rodion.chat_id = 382962362
    event_rodion.sender_id = 382962362
    event_rodion.out = False
    event_rodion.raw_text = "Привет Джефф"
    event_rodion.is_reply = False
    event_rodion.reply_to_msg_id = None
    event_rodion.media = None
    event_rodion.sender = MagicMock(bot=False, username="rodion", first_name="Родион")
    event_rodion.is_private = True
    event_rodion.sender_is_bot = False
    event_rodion.sender_username = "rodion"

    await userbot._on_new_message(event_rodion)

    assert len(received_messages) == 1
    assert received_messages[0].chat_id == 382962362
    assert received_messages[0].sender_id == 382962362
    assert received_messages[0].is_private is True
    assert received_messages[0].sender_is_bot is False


@pytest.mark.asyncio
async def test_client_unauthorized_dm_dropped(tmp_path):
    """Verify non-whitelisted DM user (999999999) is strictly dropped."""
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[],
        allowed_dm_user_ids=[382962362],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )
    userbot.bot_id = 999

    received = []
    userbot.register_message_handler(lambda m: received.append(m))

    event_stranger = MagicMock()
    event_stranger.id = 502
    event_stranger.chat_id = 999999999
    event_stranger.sender_id = 999999999
    event_stranger.out = False
    event_stranger.raw_text = "Hey bot"
    event_stranger.is_private = True
    event_stranger.sender = MagicMock(bot=False, username="stranger")
    event_stranger.sender_is_bot = False

    await userbot._on_new_message(event_stranger)
    assert len(received) == 0


@pytest.mark.asyncio
async def test_client_anti_loop_drops_hermes_and_bot_senders(tmp_path):
    """Verify bot senders and @hermes_kuk7yi6kfm3lcdfa_bot are dropped by client listener."""
    session_file = tmp_path / "test.session"
    mock_raw_client = MagicMock()

    userbot = TelegramUserbot(
        session_path=str(session_file),
        api_id=111,
        api_hash="hash",
        allowed_chat_ids=[-1003845062562],
        allowed_dm_user_ids=[382962362],
        ignored_sender_usernames=["hermes_kuk7yi6kfm3lcdfa_bot"],
        bot_names=["Jeff"],
        client=mock_raw_client,
    )
    userbot.bot_id = 999

    received = []
    userbot.register_message_handler(lambda m: received.append(m))

    # 1. Hermes bot sender
    event_hermes = MagicMock()
    event_hermes.id = 601
    event_hermes.chat_id = -1003845062562
    event_hermes.sender_id = 8710978944
    event_hermes.out = False
    event_hermes.is_private = False
    event_hermes.sender = MagicMock(bot=True, username="hermes_kuk7yi6kfm3lcdfa_bot")
    event_hermes.sender_is_bot = True
    event_hermes.sender_username = "hermes_kuk7yi6kfm3lcdfa_bot"

    await userbot._on_new_message(event_hermes)
    assert len(received) == 0

    # 2. Generic bot sender without username
    event_generic_bot = MagicMock()
    event_generic_bot.id = 602
    event_generic_bot.chat_id = -1003845062562
    event_generic_bot.sender_id = 777111
    event_generic_bot.out = False
    event_generic_bot.is_private = False
    event_generic_bot.sender = MagicMock(bot=True, username=None)
    event_generic_bot.sender_is_bot = True

    await userbot._on_new_message(event_generic_bot)
    assert len(received) == 0
