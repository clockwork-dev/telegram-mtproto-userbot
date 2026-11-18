"""End-to-End Smoke Tests, Orchestrator Verification, and Adversarial Regression Suite."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from bot.antigravity.auth import (
    _KEYRING_BYPASS_SSH_MARKER,
    build_clean_child_env,
    setup_isolated_home,
    sync_updated_isolated_tokens,
)
from bot.antigravity.errors import AntigravityNativeToolAbort, ErrorCategory, classify_error
from bot.antigravity.stream import AntigravityStreamParser
from bot.cli import build_parser, main
from bot.config.models import FriendProfile
from bot.main import run_dry_run_smoke
from bot.memory.compaction import CompactionEngine
from bot.memory.db import ChatMessage, MemoryStore
from bot.memory.fts import stem_russian_token
from bot.proactive.humor import HumorGenerator
from bot.telegram.bot_api_client import BotAPIMessageEvent, TelegramBotAPIClient
from bot.telegram.client import TelegramUserbot
from bot.telegram.heuristics import ReplyAction, ReplyHeuristics


def _load_smoke_config(tmp_path: Path):
    import shutil

    import yaml

    from bot.config.manager import ConfigManager

    cfg_dir = tmp_path / "_smoke_cfg"
    if not cfg_dir.exists():
        shutil.copytree(Path("config").resolve(), cfg_dir)
        ex = cfg_dir / "userbot.example.yaml"
        if ex.is_file():
            data = yaml.safe_load(ex.read_text(encoding="utf-8"))
            data["telegram"]["allowed_chat_ids"] = [-1001234567890]
            data["behavior"]["bot_names"] = ["піструнькиня", "Джефф", "jeff"]
            data["proactive"]["enabled"] = True
            (cfg_dir / "userbot.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        fr_ex = cfg_dir / "friends.example.yaml"
        if fr_ex.is_file() and not (cfg_dir / "friends.yaml").is_file():
            shutil.copy2(fr_ex, cfg_dir / "friends.yaml")
    return ConfigManager(config_dir=cfg_dir)


@pytest.mark.asyncio
async def test_full_pipeline_dry_run_smoke():
    """Verify full-pipeline dry-run smoke test (incoming msg + photo + FTS5 + friend profile + proactive)."""
    res = await run_dry_run_smoke("config/userbot.example.yaml")
    assert res["status"] == "ok"
    assert res["sent_messages_count"] == 1
    assert res["leftover_temp_dirs"] == 0
    assert res["reply_text"] is not None and "вложений: 1" in res["reply_text"]
    assert res["proactive_text"] is not None and len(res["proactive_text"]) > 5


def test_cli_dry_run_execution(capsys):
    """Verify `bot.cli main(['run', '--dry-run'])` executes cleanly and outputs JSON status."""
    rc = main(["run", "--config", "config/userbot.example.yaml", "--dry-run", "--log-level", "WARNING"])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "ok"
    assert data["leftover_temp_dirs"] == 0


def test_cli_auth_agy_invocation(tmp_path: Path):
    """Verify `bot.cli main(['auth-agy'])` invokes run_isolated_antigravity_login."""
    parser = build_parser()
    args = parser.parse_args(["auth-agy", "--data-dir", str(tmp_path), "--binary", "/bin/true"])
    assert args.command == "auth-agy"
    with patch("bot.cli.run_isolated_antigravity_login", return_value=0) as mock_login:
        rc = main(["auth-agy", "--data-dir", str(tmp_path), "--binary", "/bin/true"])
        assert rc == 0
        mock_login.assert_called_once_with(data_dir=str(tmp_path), binary="/bin/true")


def test_adversarial_m1_token_0600_and_atomic_sync(tmp_path: Path):
    """Regression test: loose 0o644 source token is tightened to 0o600 and synced atomically."""
    src_root = tmp_path / "src_home"
    src_gemini = src_root / ".gemini" / "antigravity-cli"
    src_gemini.mkdir(parents=True)
    src_token = src_gemini / "jetski-standalone-oauth-token"
    src_token.write_text("initial_token", encoding="utf-8")
    os.chmod(src_token, 0o644)

    iso_dir = tmp_path / "isolated"
    _, iso_gemini = setup_isolated_home(iso_dir, source_dir=src_root)
    iso_token = iso_gemini / "jetski-standalone-oauth-token"
    assert stat.S_IMODE(iso_token.stat().st_mode) == 0o600

    # Replace isolated token with a newer standalone regular file
    if iso_token.is_symlink() or iso_token.exists():
        iso_token.unlink()
    iso_token.write_text("refreshed_token_v2", encoding="utf-8")
    future_time = src_token.stat().st_mtime + 100.0
    os.utime(iso_token, (future_time, future_time))

    sync_updated_isolated_tokens(iso_gemini, data_dir=src_root)
    assert src_token.read_text(encoding="utf-8") == "refreshed_token_v2"
    assert stat.S_IMODE(src_token.stat().st_mode) == 0o600

    # Verify build_clean_child_env injects SSH_CONNECTION even on empty dir
    empty_dir = tmp_path / "empty_iso"
    empty_dir.mkdir()
    env = build_clean_child_env(empty_dir)
    assert env.get("SSH_CONNECTION") == _KEYRING_BYPASS_SSH_MARKER


def test_adversarial_m1_stream_interception_and_nested_errors():
    """Regression test: assistant block tool interception, split tag attributes, and nested exceptions."""
    parser = AntigravityStreamParser()
    with pytest.raises(AntigravityNativeToolAbort):
        parser.feed_line(
            json.dumps({"event": "assistant", "message": {"content": [{"step_type": "tool"}]}})
        )

    parser2 = AntigravityStreamParser()
    c1 = parser2.feed_line(
        json.dumps({"event": "step_update", "step_update": {"text_delta": "Hello <tool_"}})
    )
    assert "".join(ch.text or "" for ch in c1) == "Hello "
    c2 = parser2.feed_line(
        json.dumps({"event": "step_update", "step_update": {"text_delta": 'call id="1">'}})
    )
    assert "".join(ch.text or "" for ch in c2) == ""
    c3 = parser2.feed_line(
        json.dumps(
            {
                "event": "step_update",
                "step_update": {"text_delta": '{"name": "ping", "arguments": {}}</tool_call>'},
            }
        )
    )
    assert any(ch.tool_calls for ch in c3)

    # Nested exception classification
    try:
        try:
            raise ValueError("RESOURCE_EXHAUSTED: 429 too many requests")
        except ValueError as inner:
            raise RuntimeError("Wrapper failure") from inner
    except RuntimeError as outer:
        verdict = classify_error(outer)
        assert verdict.category == ErrorCategory.RATE_LIMIT


@pytest.mark.asyncio
async def test_adversarial_m2_m3_regressions(tmp_path: Path):
    """Regression test for M2 heuristics/whitelist coercion and M3 stemmer, compaction threshold & humor filter."""
    # M2: mention with colon and chat_id=None / string chat_id coercion
    heur = ReplyHeuristics()
    assert heur.is_mention("@custombot: привет", bot_names=["@custombot"]) is True

    class _NoChatEvent:
        chat_id = None
        out = False
        sender_id = 999
        raw_text = "Джефф, привет!"

    assert (
        heur.should_reply(_NoChatEvent(), bot_id=1, bot_names=["Джефф"], allowed_chat_ids=[-1001])
        == ReplyAction.IGNORE
    )

    ubot = TelegramUserbot(
        session_path=str(tmp_path / "s.session"),
        api_id=1,
        api_hash="h",
        allowed_chat_ids=["-100555"],  # type: ignore[list-item]
    )
    assert ubot.allowed_chat_ids == [-100555]

    # M3: 3-char Russian instrumental plural stemmer ("шаурмами" -> "шаурм")
    assert stem_russian_token("шаурмами") == "шаурм"
    store = MemoryStore(db_path=":memory:")
    await store.init_db()
    await store.add_fact("Алексей", "Обожает вкусную шаурму", "food")
    matches = await store.search_facts("шаурмами")
    assert len(matches) == 1

    # M3: CompactionEngine respects threshold when count < threshold
    for i in range(4):
        await store.save_message(
            ChatMessage(
                telegram_msg_id=i + 1,
                chat_id=-1001,
                user_id=10,
                timestamp=1000 + i,
                author_name="U",
                text=f"msg {i}",
            )
        )
    comp = CompactionEngine(memory_store=store, threshold=10, batch_size=3)
    assert await comp.run_compaction_for_chat(-1001) == 0
    await store.close()

    # M3: HumorGenerator filters banter_topics against sensitive_topics and forbidden_topics
    hgen = HumorGenerator(provider=None)
    friend = FriendProfile(
        name="Иван",
        banter_topics=["крипта и долги", "политика", "настройка Linux"],
        sensitive_topics=["крипта и долги"],
    )
    prompt = hgen.build_humor_prompt(
        friend=friend,
        forbidden_topics=["политика"],
    )
    allowed_line = [
        line for line in prompt.splitlines() if "Разрешенные темы для подколов:" in line
    ][0]
    assert "крипта" not in allowed_line
    assert "политика" not in allowed_line
    assert "настройка Linux" in allowed_line

    fallback_joke = await hgen.generate_banter(
        friend=friend,
        forbidden_topics=["политика"],
    )
    assert "крипта" not in fallback_joke
    assert "политика" not in fallback_joke
    assert "настройка Linux" in fallback_joke


@pytest.mark.asyncio
async def test_bot_api_client_adapter_and_polling():
    """Verify Classic Telegram Bot API adapter parses group messages, photos, and replies."""
    raw_msg = {
        "message_id": 19736,
        "from": {"id": 447164588, "is_bot": False, "first_name": "Анюта ♡", "username": "a_hlvn"},
        "chat": {"id": -1003845062562, "title": "летальний ісход", "type": "supergroup"},
        "date": 1794586333,
        "text": "піструнькиня, будемо в шось ґрати?",
    }
    ev = BotAPIMessageEvent(raw_msg, bot_id=8710978944)
    assert ev.chat_id == -1003845062562
    assert ev.sender_id == 447164588
    assert ev.out is False

    client = TelegramBotAPIClient(
        bot_token="123456:DUMMY_TOKEN",
        allowed_chat_ids=[-1003845062562],
        bot_names=["піструнькиня"],
    )
    client.bot_id = 8710978944
    client.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"

    inc = client._convert_msg_dict(raw_msg)
    assert inc is not None
    assert inc.is_mention is True
    assert inc.sender_id == 447164588


@pytest.mark.asyncio
async def test_persistent_worker_breaks_on_result_event_without_eof(tmp_path: Path) -> None:
    """Verify persistent agy worker returns immediately on result event without blocking for stdout EOF."""
    import io
    import json
    from unittest.mock import patch

    from bot.antigravity.cli_client import AntigravityCLIClient

    lines = [
        json.dumps({"event": "init", "conversation_id": "conv_persist_1"}) + "\n",
        json.dumps({"event": "result", "result": {"status": "SUCCESS", "response": "Instant reply!"}}) + "\n",
    ]
    call_idx = 0

    class _NonClosingStdout:
        def readline(self) -> str:
            nonlocal call_idx
            if call_idx < len(lines):
                line = lines[call_idx]
                call_idx += 1
                return line
            raise AssertionError("stdout.readline() was called after result event on persistent worker!")

    class _PersistentProc:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = _NonClosingStdout()
            self.stderr = io.StringIO("")
            self.pid = 99999

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

    cli = AntigravityCLIClient(data_dir=tmp_path, binary="/usr/bin/agy")
    mock_proc = _PersistentProc()
    with patch("bot.antigravity.cli_client.popen_isolated_session", return_value=mock_proc):
        resp = await cli.generate_response(prompt="Проверка")
    assert resp.text == "Instant reply!"
    assert resp.conversation_id == "conv_persist_1"
    cli.close()


@pytest.mark.asyncio
async def test_orchestrator_reactions_real_msg_id_and_active_window(tmp_path: Path) -> None:
    """Verify [REACT:🔥] parsing/sending/stripping, real message_id storage in SQLite, and 90s active window."""
    import time
    from unittest.mock import AsyncMock, MagicMock

    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_react_msg_id.sqlite"))
    await memory_store.init_db()

    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Ну это чистая база! [REACT:🔥]", model="test-model")
    )

    chat_id = (
        int(config_mgr.bot.telegram.allowed_chat_ids[0])
        if config_mgr.bot and config_mgr.bot.telegram.allowed_chat_ids
        else -1003845062562
    )

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock(return_value=True)
    mock_userbot.send_reply = AsyncMock(
        return_value={"ok": True, "result": {"message_id": 777888}}
    )

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    class _RawEv:
        id = 5001
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = "піструнькиня, глянь шо творится"
        text = raw_text
        photo = False
        media = False

    ev1 = _RawEv()
    ev1.chat_id = chat_id
    ev1.sender_id = 447164588

    inc1 = IncomingMessage(
        msg_id=5001,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev1.raw_text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=True,
        raw_event=ev1,
        media=None,
    )

    reply1 = await orchestrator.handle_incoming_message(inc1)
    assert reply1 == "Ну это чистая база!"
    mock_userbot.send_reaction.assert_awaited_once_with(chat_id, 5001, "🔥")
    mock_userbot.send_reply.assert_awaited_once()
    assert mock_userbot.send_reply.call_args.kwargs["text"] == "Ну это чистая база!"

    # Verify real telegram_msg_id (777888) was saved in SQLite messages
    recent = await memory_store.get_recent_history(chat_id, limit=10)
    bot_msgs = [m for m in recent if m.user_id == 8710978944]
    assert len(bot_msgs) == 1
    assert bot_msgs[0].telegram_msg_id == 777888
    assert bot_msgs[0].text == "Ну это чистая база!"

    # Now test follow-up within 60s active conversation window:
    # 1) Generic pronoun ("Че она не отвечает") at 20.0s must be IGNORED (no pronoun trigger + <25s cooldown)
    orchestrator.chat_states[chat_id]["last_bot_msg_time"] = time.time() - 20.0
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="[REACT:😂]", model="test-model")
    )
    mock_userbot.send_reaction.reset_mock()
    mock_userbot.send_reply.reset_mock()

    class _RawEv2:
        id = 5002
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = "Че она не отвечает"
        text = raw_text
        photo = False
        media = False

    ev2 = _RawEv2()
    ev2.chat_id = chat_id
    ev2.sender_id = 447164588

    inc2 = IncomingMessage(
        msg_id=5002,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev2.raw_text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=False,
        raw_event=ev2,
        media=None,
    )

    reply2 = await orchestrator.handle_incoming_message(inc2)
    assert reply2 is None
    mock_provider.generate_response.assert_not_awaited()
    mock_userbot.send_reaction.assert_not_awaited()

    # 2) Substantive follow-up past 25s active-window cooldown (30s) triggers SPONTANEOUS_REPLY,
    #    normalizes 😂 -> 🤣, updates last_bot_msg_time, and does not double-count max_per_hour
    stale_bot_ts = time.time() - 30.0
    orchestrator.chat_states[chat_id]["last_bot_msg_time"] = stale_bot_ts

    class _RawEv3:
        id = 50020
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = "Ну вообще реально странная ситуация получилась"
        text = raw_text
        photo = False
        media = False

    ev3 = _RawEv3()
    ev3.chat_id = chat_id
    ev3.sender_id = 447164588

    inc3 = IncomingMessage(
        msg_id=50020,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev3.raw_text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=False,
        raw_event=ev3,
        media=None,
    )

    with patch("bot.telegram.heuristics.random.random", return_value=0.0):
        reply3 = await orchestrator.handle_incoming_message(inc3)

    assert reply3 is None
    mock_userbot.send_reaction.assert_awaited_once_with(chat_id, 50020, "🤣")
    mock_userbot.send_reply.assert_not_awaited()
    assert orchestrator.chat_states[chat_id]["last_bot_msg_time"] > stale_bot_ts
    assert orchestrator.heuristics.recent_reply_count(chat_id) == 2

    await memory_store.close()


@pytest.mark.asyncio
async def test_orchestrator_document_generation_and_delivery_e2e(tmp_path: Path) -> None:
    """Verify [DOC:{...}] extraction, real Cyrillic PDF creation, send_document delivery, cleanup, and SQLite save."""
    from unittest.mock import AsyncMock, MagicMock

    from pypdf import PdfReader

    from bot.core.models import ProviderResponse
    from bot.documents.builder import DocumentBuilder
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_doc_e2e.sqlite"))
    await memory_store.init_db()

    doc_tmp_root = tmp_path / "doc_tmp"
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))
    doc_builder = DocumentBuilder(temp_dir_base=str(doc_tmp_root))

    doc_payload = json.dumps(
        {
            "format": "pdf",
            "title": "Сводка по Сумам",
            "subtitle": "Актуальные новости и погода",
            "filename": "сводка_сумы.pdf",
            "caption": "Держи свежий отчёт в PDF!",
            "sections": [
                {
                    "heading": "Главное",
                    "body": "Собрала свежую информацию по запросу и оформила в аккуратный PDF.",
                    "bullets": ["Кириллица отображается корректно", "Источники приложены ниже"],
                    "table": [["Параметр", "Значение"], ["Город", "Сумы"]],
                }
            ],
            "sources": ["https://example.org/news"],
        },
        ensure_ascii=False,
    )
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(
            text=f"Нашла всё нужное, лови файлик! [DOC:{doc_payload}]",
            model="test-model",
        )
    )

    chat_id = (
        int(config_mgr.bot.telegram.allowed_chat_ids[0])
        if config_mgr.bot and config_mgr.bot.telegram.allowed_chat_ids
        else -1003845062562
    )

    captured_uploads: list[dict] = []

    async def _fake_send_document(
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        reply_to_msg_id: int | None = None,
        simulate_typing: bool = True,
        filename: str | None = None,
    ) -> dict:
        p = Path(file_path)
        assert p.is_file()
        assert p.suffix == ".pdf"
        raw_bytes = p.read_bytes()
        assert len(raw_bytes) > 0
        assert raw_bytes.startswith(b"%PDF")
        extracted_text = "\n".join((page.extract_text() or "") for page in PdfReader(p).pages)
        assert "Сводка по Сумам" in extracted_text
        captured_uploads.append(
            {
                "chat_id": chat_id,
                "path": p,
                "size": len(raw_bytes),
                "caption": caption,
                "reply_to_msg_id": reply_to_msg_id,
                "filename": filename,
            }
        )
        return {"ok": True, "result": {"message_id": 888001}}

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock(return_value=True)
    mock_userbot.send_document = AsyncMock(side_effect=_fake_send_document)
    mock_userbot.send_reply = AsyncMock(
        return_value={"ok": True, "result": {"message_id": 888002}}
    )

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        document_builder=doc_builder,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    class _RawEv:
        id = 5003
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = "піструнькиня, найди инфу и сделай пдф"
        text = raw_text
        photo = False
        media = False

    ev = _RawEv()
    ev.chat_id = chat_id
    ev.sender_id = 447164588

    incoming = IncomingMessage(
        msg_id=5003,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev.raw_text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=True,
        raw_event=ev,
        media=None,
    )

    result_text = await orchestrator.handle_incoming_message(incoming)
    assert result_text == "Нашла всё нужное, лови файлик! [Документы: сводка_сумы.pdf]"

    # Prompt instructs search_web / read_url_content and [DOC:{...}] usage
    sent_prompt = mock_provider.generate_response.call_args.kwargs["prompt"]
    assert "search_web" in sent_prompt
    assert "read_url_content" in sent_prompt
    assert "[DOC:" in sent_prompt

    assert len(captured_uploads) == 1
    assert captured_uploads[0]["caption"] == "Нашла всё нужное, лови файлик!"
    assert captured_uploads[0]["reply_to_msg_id"] == 5003
    assert captured_uploads[0]["filename"] == "сводка_сумы.pdf"
    assert not captured_uploads[0]["path"].exists()
    assert list(doc_tmp_root.glob("bot_doc_*")) == []

    # Short reply text goes directly into document caption, avoiding a duplicate text message
    mock_userbot.send_reply.assert_not_awaited()

    recent = await memory_store.get_recent_history(chat_id, limit=10)
    bot_msgs = [m for m in recent if m.user_id == 8710978944]
    assert len(bot_msgs) == 1
    assert bot_msgs[0].telegram_msg_id == 888001
    assert bot_msgs[0].media_type == "document"
    assert "[Документы: сводка_сумы.pdf]" in bot_msgs[0].text
    assert "Нашла всё нужное, лови файлик!" in bot_msgs[0].text

    await memory_store.close()


@pytest.mark.asyncio
async def test_orchestrator_doc_only_reply_and_error_resilience(tmp_path: Path) -> None:
    """Verify doc-only [DOC:{...}] reply skips send_reply, updates history, and doc errors don't crash."""
    from unittest.mock import AsyncMock, MagicMock

    from bot.core.models import ProviderResponse
    from bot.documents.builder import DocumentBuilder
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_doc_only.sqlite"))
    await memory_store.init_db()

    doc_tmp_root = tmp_path / "doc_only_tmp"
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))
    doc_builder = DocumentBuilder(temp_dir_base=str(doc_tmp_root))

    doc_payload = json.dumps(
        {
            "format": "pdf",
            "title": "Краткая памятка",
            "filename": "памятка.pdf",
            "caption": "Только документ без лишних слов",
            "sections": [{"heading": "Пункты", "bullets": ["Первый пункт", "Второй пункт"]}],
        },
        ensure_ascii=False,
    )
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text=f"[DOC:{doc_payload}]", model="test-model")
    )

    chat_id = (
        int(config_mgr.bot.telegram.allowed_chat_ids[0])
        if config_mgr.bot and config_mgr.bot.telegram.allowed_chat_ids
        else -1003845062562
    )

    async def _fake_send_document(**kwargs) -> dict:
        p = Path(kwargs["file_path"])
        assert p.is_file() and p.stat().st_size > 0
        return {"ok": True, "result": {"message_id": 888100}}

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock(return_value=True)
    mock_userbot.send_document = AsyncMock(side_effect=_fake_send_document)
    mock_userbot.send_reply = AsyncMock(
        return_value={"ok": True, "result": {"message_id": 888101}}
    )

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        document_builder=doc_builder,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    class _RawEv:
        id = 5004
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = "піструнькиня, скинь пдф"
        text = raw_text
        photo = False
        media = False

    ev = _RawEv()
    ev.chat_id = chat_id
    ev.sender_id = 447164588

    inc = IncomingMessage(
        msg_id=5004,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text=ev.raw_text,
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=True,
        raw_event=ev,
        media=None,
    )

    res = await orchestrator.handle_incoming_message(inc)
    assert res == "[Документы: памятка.pdf]"
    mock_userbot.send_document.assert_awaited_once()
    assert mock_userbot.send_document.call_args.kwargs["caption"] == "Только документ без лишних слов"
    assert mock_userbot.send_document.call_args.kwargs["reply_to_msg_id"] == 5004
    mock_userbot.send_reply.assert_not_awaited()
    assert orchestrator.chat_states[chat_id]["last_bot_msg_time"] > 0
    assert list(doc_tmp_root.glob("bot_doc_*")) == []

    recent = await memory_store.get_recent_history(chat_id, limit=10)
    bot_msgs = [m for m in recent if m.user_id == 8710978944]
    assert len(bot_msgs) == 1
    assert bot_msgs[0].telegram_msg_id == 888100
    assert bot_msgs[0].text == "[Документы: памятка.pdf]"
    assert bot_msgs[0].media_type == "document"

    # Also verify send_document failure does not crash the message loop
    mock_userbot.send_document = AsyncMock(side_effect=RuntimeError("Telegram upload error"))
    mock_userbot.send_reply.reset_mock()
    ev.id = 5005
    inc_err = IncomingMessage(
        msg_id=5005,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text="піструнькиня, ещё пдф",
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=True,
        raw_event=ev,
        media=None,
    )
    err_res = await orchestrator.handle_incoming_message(inc_err)
    assert err_res is None
    mock_userbot.send_reply.assert_not_awaited()
    assert list(doc_tmp_root.glob("bot_doc_*")) == []

    await memory_store.close()


@pytest.mark.asyncio
async def test_orchestrator_proactive_cron_digest_delivery_and_memory_save(tmp_path: Path) -> None:
    """Verify orchestrator delivers cron digest with empty chat_states and persists to MemoryStore."""
    from datetime import datetime
    from unittest.mock import AsyncMock, MagicMock
    from zoneinfo import ZoneInfo

    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_cron_digest.sqlite"))
    await memory_store.init_db()

    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))
    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Утренний дайджест: мир всё ещё странный.", model="test-model")
    )

    chat_id = (
        int(config_mgr.bot.telegram.allowed_chat_ids[0])
        if config_mgr.bot and config_mgr.bot.telegram.allowed_chat_ids
        else -1003845062562
    )

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.start = AsyncMock()
    mock_userbot.stop = AsyncMock()
    mock_userbot.send_reply = AsyncMock(
        return_value={"ok": True, "result": {"message_id": 999111}}
    )

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
    )

    # Simulate empty chat_states after restart
    orchestrator.chat_states.clear()

    # Verify start() wires get_target_chat_ids into start_loop
    with patch.object(orchestrator.proactive_engine, "start_loop") as mock_start_loop:
        with patch.object(orchestrator.config_mgr, "start_watcher", new=AsyncMock()):
            await orchestrator.start()
        mock_start_loop.assert_called_once()
        kwargs = mock_start_loop.call_args.kwargs
        assert callable(kwargs["get_target_chat_ids"])
        assert chat_id in kwargs["get_target_chat_ids"]()

    kyiv = ZoneInfo("Europe/Kyiv")
    orchestrator.proactive_engine.init_cron_cursor(
        now_dt=datetime(2026, 11, 14, 8, 59, 30, tzinfo=kyiv)
    )
    await orchestrator.proactive_engine.run_proactive_tick(
        get_chat_states=lambda: orchestrator.chat_states,
        send_message_fn=orchestrator._send_proactive,
        get_target_chat_ids=orchestrator._get_proactive_target_chat_ids,
        now_dt=datetime(2026, 11, 14, 9, 0, 25, tzinfo=kyiv),
    )

    mock_userbot.send_reply.assert_awaited_once()
    assert mock_userbot.send_reply.call_args.kwargs["chat_id"] == chat_id
    assert (
        mock_userbot.send_reply.call_args.kwargs["text"]
        == "Утренний дайджест: мир всё ещё странный."
    )
    assert orchestrator.chat_states[chat_id]["last_bot_msg_time"] > 0

    recent = await memory_store.get_recent_history(chat_id, limit=10)
    bot_msgs = [m for m in recent if m.user_id == 8710978944]
    assert len(bot_msgs) == 1
    assert bot_msgs[0].telegram_msg_id == 999111
    assert bot_msgs[0].text == "Утренний дайджест: мир всё ещё странный."

    await memory_store.close()


def test_cli_run_digest_dry_run_and_validation(capsys) -> None:
    """Verify `bot.cli main(['run-digest', ...])` dry-run modes and unknown slot error handling."""
    rc = main(
        [
            "run-digest",
            "--config",
            "config/userbot.example.yaml",
            "--slot",
            "morning_digest",
            "--chat-id",
            "-1001234567890",
            "--dry-run",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "ok"
    assert data["dry_run"] is True
    assert len(data["results"]) >= 1
    assert data["results"][0]["job_id"] == "morning_digest"
    assert data["results"][0]["sent"] is False
    assert isinstance(data["results"][0]["text"], str)
    assert len(data["results"][0]["text"]) > 20

    # --all-due dry-run succeeds cleanly even when no slot is currently due
    rc_due = main(
        [
            "run-digest",
            "--config",
            "config/userbot.example.yaml",
            "--all-due",
            "--dry-run",
            "--log-level",
            "WARNING",
        ]
    )
    assert rc_due == 0
    due_data = json.loads(capsys.readouterr().out)
    assert due_data["status"] == "ok"
    assert due_data["dry_run"] is True
    assert isinstance(due_data["results"], list)

    # Unknown slot returns non-zero error code
    rc_bad = main(
        [
            "run-digest",
            "--config",
            "config/userbot.example.yaml",
            "--slot",
            "nonexistent_slot",
            "--dry-run",
        ]
    )
    assert rc_bad == 1


@pytest.mark.asyncio
async def test_orchestrator_skip_fact_cooldown_and_gated_prompts(tmp_path: Path) -> None:
    """Verify [SKIP], no author_display FTS pollution, fact cooldown, bot-history suppression, and gated tool prompts."""
    from unittest.mock import AsyncMock, MagicMock

    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_facts_skip.sqlite"))
    await memory_store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))

    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    friend_id = next(iter(config_mgr.friends.friends.keys()))
    friend_name = config_mgr.friends.friends[friend_id].name

    await memory_store.add_fact(
        subject=friend_name,
        fact="Обожает паять кастомные механические клавиатуры и свитчи",
        category="friend_bio",
        source_chat_id=chat_id,
    )

    mock_provider = MagicMock()
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Да, механика вещь хорошая.", model="test-model")
    )
    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock(return_value=True)
    mock_userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9101}})

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    def _make_inc(msg_id: int, text: str) -> IncomingMessage:
        class _Ev:
            id = msg_id
            out = False
            is_reply = False
            reply_to_msg_id = None
            raw_text = text
            photo = False
            media = False

        ev = _Ev()
        ev.chat_id = chat_id
        ev.sender_id = friend_id
        ev.text = text
        return IncomingMessage(
            msg_id=msg_id,
            chat_id=chat_id,
            sender_id=friend_id,
            sender_name=friend_name,
            text=text,
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=True,
            raw_event=ev,
            media=None,
        )

    # 1. Casual message unrelated to keyboards -> author_display must NOT trigger keyboard fact,
    #    and heavy tool instructions ([DOC], search_web) & banter_topics checklist must be omitted.
    await orchestrator.handle_incoming_message(_make_inc(1, "піструнькиня, привет как настроение вообще"))
    prompt1 = mock_provider.generate_response.call_args.kwargs["prompt"]
    assert "механические клавиатуры" not in prompt1
    assert "Разрешенные темы для шуток" not in prompt1
    assert "[DOC:" not in prompt1
    assert "search_web" not in prompt1

    # 2. Topic-matching substantive turn injects fact ONCE with passive framing
    await orchestrator.handle_incoming_message(
        _make_inc(2, "піструнькиня, посоветуй кастомные механические клавиатуры и свитчи")
    )
    prompt2 = mock_provider.generate_response.call_args.kwargs["prompt"]
    assert "ПАССИВНЫЙ КОНТЕКСТ" in prompt2
    assert "механические клавиатуры" in prompt2

    # 3. Repeating same topic on next turn is suppressed by fact cooldown + bot-history filter
    await orchestrator.handle_incoming_message(
        _make_inc(3, "піструнькиня, а какие еще механические клавиатуры и свитчи бывают?")
    )
    prompt3 = mock_provider.generate_response.call_args.kwargs["prompt"]
    assert "Обожает паять кастомные механические клавиатуры" not in prompt3

    # 4. [SKIP] response sends nothing and doesn't advance last_bot_msg_time
    prev_bot_time = orchestrator.chat_states[chat_id]["last_bot_msg_time"]
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="  [SKIP]  ", model="test-model")
    )
    mock_userbot.send_reply.reset_mock()
    mock_userbot.send_reaction.reset_mock()
    res_skip = await orchestrator.handle_incoming_message(
        _make_inc(4, "піструнькиня, ну ладно понятно")
    )
    assert res_skip is None
    mock_userbot.send_reply.assert_not_awaited()
    mock_userbot.send_reaction.assert_not_awaited()
    assert orchestrator.chat_states[chat_id]["last_bot_msg_time"] == prev_bot_time

    await memory_store.close()


@pytest.mark.asyncio
async def test_orchestrator_media_after_heuristics_and_cli_blind_guard(tmp_path: Path) -> None:
    """Verify IGNORE skips managed_media and CLI-only provider skips blind media-only LLM calls."""
    from unittest.mock import AsyncMock, MagicMock

    from bot.antigravity import AntigravityCLIClient
    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor, ProcessedMedia
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_media_gate.sqlite"))
    await memory_store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))
    managed_spy = MagicMock(wraps=media_proc.managed_media)
    media_proc.managed_media = managed_spy  # type: ignore[method-assign]

    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    mock_provider = MagicMock()
    mock_provider._sdk_client = None
    mock_provider._cli_client = MagicMock(spec=AntigravityCLIClient)
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="картинку не прогрузили", model="test-model")
    )

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reply = AsyncMock()

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    # 1. Spontaneous sticker (IGNORED by heuristics) must NEVER call managed_media
    class _StickerEv:
        id = 7001
        out = False
        is_reply = False
        reply_to_msg_id = None
        raw_text = ""
        text = ""
        sticker = True
        media = True

    ev_ign = _StickerEv()
    ev_ign.chat_id = chat_id
    ev_ign.sender_id = 447164588
    inc_ign = IncomingMessage(
        msg_id=7001,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text="",
        reply_to_msg_id=None,
        is_reply=False,
        is_mention=False,
        raw_event=ev_ign,
        media=True,
    )
    assert await orchestrator.handle_incoming_message(inc_ign) is None
    assert managed_spy.call_count == 0
    mock_provider.generate_response.assert_not_awaited()

    # 2. Direct reply-to-bot with photo-only (no caption) + CLI-only provider skips blind LLM call
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_managed_media(*args, **kwargs):
        yield ProcessedMedia(media_type="photo", mime_type="image/jpeg")

    media_proc.managed_media = _fake_managed_media  # type: ignore[method-assign]

    class _ReplyPhotoEv:
        id = 7002
        out = False
        is_reply = True
        is_reply_to_bot = True
        reply_to_msg_id = 100
        raw_text = ""
        text = ""
        photo = True
        media = True

    ev_photo = _ReplyPhotoEv()
    ev_photo.chat_id = chat_id
    ev_photo.sender_id = 447164588
    inc_photo = IncomingMessage(
        msg_id=7002,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text="",
        reply_to_msg_id=100,
        is_reply=True,
        is_mention=False,
        raw_event=ev_photo,
        media=True,
    )
    assert await orchestrator.handle_incoming_message(inc_photo) is None
    mock_provider.generate_response.assert_not_awaited()

    await memory_store.close()


@pytest.mark.asyncio
async def test_orchestrator_debounce_coalescing_and_startup_seeding(tmp_path: Path) -> None:
    """Verify burst message debouncing, rate-limit skip, and startup chat_states seeding."""
    import asyncio
    import time
    from unittest.mock import AsyncMock, MagicMock

    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor
    from bot.telegram.handlers import IncomingMessage

    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_debounce_seed.sqlite"))
    await memory_store.init_db()
    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))

    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])
    now = time.time()
    # Seed DB with recent human msg (25m ago) and bot msg (20m ago)
    await memory_store.save_message(
        ChatMessage(
            telegram_msg_id=11,
            chat_id=chat_id,
            user_id=447164588,
            timestamp=int(now - 1500),
            author_name="Анюта",
            text="Зацените прикол",
        )
    )
    await memory_store.save_message(
        ChatMessage(
            telegram_msg_id=12,
            chat_id=chat_id,
            user_id=8710978944,
            timestamp=int(now - 1200),
            author_name="піструнькиня",
            text="Хах, норм",
        )
    )

    mock_provider = MagicMock()
    mock_provider.is_rate_limited = MagicMock(return_value=False)
    mock_provider.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Вижу всю серию сразу!", model="test-model")
    )
    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.start = AsyncMock()
    mock_userbot.stop = AsyncMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock(return_value=True)
    mock_userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 9201}})

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=mock_provider,
        userbot=mock_userbot,
        debounce_seconds=0.05,
    )
    orchestrator.proactive_engine.is_in_quiet_hours = MagicMock(return_value=False)

    with patch.object(orchestrator.proactive_engine, "start_loop"):
        with patch.object(orchestrator.config_mgr, "start_watcher", new=AsyncMock()):
            await orchestrator.start()

    # Startup seeded timestamps from DB -> no unsolicited banter 60s after restart
    assert orchestrator.chat_states[chat_id]["last_bot_msg_time"] == pytest.approx(int(now - 1200))
    assert (
        orchestrator.proactive_engine.check_triggers(
            chat_id,
            orchestrator.chat_states[chat_id]["last_chat_msg_time"],
            orchestrator.chat_states[chat_id]["last_bot_msg_time"],
            current_time=now + 60,
        )
        is None
    )

    # Burst of 3 rapid messages coalesces into ONE provider call
    def _burst_msg(mid: int, txt: str, mention: bool = False) -> IncomingMessage:
        class _Ev:
            id = mid
            out = False
            is_reply = False
            reply_to_msg_id = None
            raw_text = txt
            photo = False
            media = False

        ev = _Ev()
        ev.chat_id = chat_id
        ev.sender_id = 447164588
        ev.text = txt
        return IncomingMessage(
            msg_id=mid,
            chat_id=chat_id,
            sender_id=447164588,
            sender_name="Анюта",
            text=txt,
            reply_to_msg_id=None,
            is_reply=False,
            is_mention=mention,
            raw_event=ev,
            media=None,
        )

    await orchestrator._on_incoming_message(_burst_msg(21, "Да"))
    await asyncio.sleep(0.01)
    await orchestrator._on_incoming_message(_burst_msg(22, "Уже"))
    await asyncio.sleep(0.01)
    await orchestrator._on_incoming_message(_burst_msg(23, "піструнькиня, глянь шо унесло", mention=True))

    await asyncio.sleep(0.12)
    mock_provider.generate_response.assert_awaited_once()
    coalesced_prompt = mock_provider.generate_response.call_args.kwargs["prompt"]
    assert "Да" in coalesced_prompt
    assert "Уже" in coalesced_prompt
    assert "глянь шо унесло" in coalesced_prompt

    # Verify rate-limit circuit breaker suppresses LLM call
    mock_provider.is_rate_limited = MagicMock(return_value=True)
    mock_provider.generate_response.reset_mock()
    assert (
        await orchestrator.handle_incoming_message(
            _burst_msg(24, "піструнькиня, ещё вопрос", mention=True)
        )
        is None
    )
    mock_provider.generate_response.assert_not_awaited()

    await orchestrator.stop()


@pytest.mark.asyncio
async def test_orchestrator_reply_image_vision_and_build_attachments(tmp_path: Path) -> None:
    """Verify build_media_attachments, captionless photo reply to bot, and text reply to someone's photo."""
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock

    from bot.antigravity import AntigravityConfig, AntigravityProvider
    from bot.core.models import ProviderResponse
    from bot.main import GroupBotOrchestrator
    from bot.media.processor import MediaProcessor, ProcessedMedia
    from bot.telegram.handlers import IncomingMessage

    photo_path = tmp_path / "photo.jpg"
    photo_path.write_bytes(b"\xff\xd8\xff\xe0dummy_jpeg")
    kf_path = tmp_path / "anim_kf.jpg"
    kf_path.write_bytes(b"\xff\xd8\xff\xe0dummy_kf")
    anim_path = tmp_path / "anim.mp4"
    anim_path.write_bytes(b"mp4bytes")

    # 1. Unit check on GroupBotOrchestrator.build_media_attachments
    photo_attachments = GroupBotOrchestrator.build_media_attachments(
        ProcessedMedia(
            media_type="photo",
            mime_type="image/jpeg",
            file_path=photo_path,
            metadata={"description": "Фотография"},
        )
    )
    anim_attachments = GroupBotOrchestrator.build_media_attachments(
        ProcessedMedia(
            media_type="animation",
            mime_type="video/mp4",
            file_path=anim_path,
            keyframe_path=kf_path,
            metadata={"description": "GIF"},
        )
    )
    assert len(photo_attachments) == 1
    assert photo_attachments[0].file_path == photo_path
    assert photo_attachments[0].mime_type == "image/jpeg"
    assert len(anim_attachments) == 1
    assert anim_attachments[0].file_path == kf_path
    assert anim_attachments[0].mime_type == "image/jpeg"

    # Set up orchestrator with CLI-only AntigravityProvider (no SDK key) supporting image vision
    config_mgr = _load_smoke_config(tmp_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "test_reply_vision.sqlite"))
    await memory_store.init_db()
    chat_id = int(config_mgr.bot.telegram.allowed_chat_ids[0])

    cli_stub = MagicMock()
    cli_stub.supports_image_vision = True
    cli_stub.generate_response = AsyncMock(
        return_value=ProviderResponse(text="Вижу на фото рыжего кота.", model="gemini-acp")
    )
    provider = AntigravityProvider(
        config=AntigravityConfig(data_dir=tmp_path / "agy", enable_sdk=False)
    )
    provider._sdk_client = None
    provider._cli_client = cli_stub

    mock_userbot = MagicMock()
    mock_userbot.bot_id = 8710978944
    mock_userbot.bot_username = "hermes_kuk7yi6kfm3lcdfa_bot"
    mock_userbot.client = MagicMock()
    mock_userbot.start = AsyncMock()
    mock_userbot.stop = AsyncMock()
    mock_userbot.send_typing_action = AsyncMock()
    mock_userbot.send_reaction = AsyncMock()
    mock_userbot.send_reply = AsyncMock(return_value={"ok": True, "result": {"message_id": 7771}})

    replied_photo_path = tmp_path / "replied_photo.jpg"
    replied_photo_path.write_bytes(b"\xff\xd8\xff\xe0replied_jpeg")

    direct_event = BotAPIMessageEvent(
        {
            "message_id": 601,
            "chat": {"id": chat_id, "type": "supergroup"},
            "from": {"id": 447164588, "first_name": "Анюта", "username": "anyuta"},
            "reply_to_message": {
                "message_id": 500,
                "from": {"id": 8710978944, "is_bot": True, "first_name": "піструнькиня"},
                "text": "покажи",
            },
            "photo": [{"file_id": "direct_photo_1", "width": 100, "height": 100, "file_size": 1024}],
        },
        bot_id=8710978944,
    )

    replied_media_event = BotAPIMessageEvent(
        {
            "message_id": 555,
            "chat": {"id": chat_id, "type": "supergroup"},
            "from": {"id": 999001, "first_name": "Вася"},
            "caption": "смотри",
            "photo": [{"file_id": "photo_file_1", "width": 100, "height": 100, "file_size": 1024}],
        },
        bot_id=8710978944,
    )

    media_proc = MediaProcessor(max_doc_size_mb=10, temp_dir_base=str(tmp_path / "tmp_media"))

    @asynccontextmanager
    async def fake_managed_media(message, client=None):
        if message is direct_event:
            yield ProcessedMedia(
                media_type="photo",
                mime_type="image/jpeg",
                file_path=photo_path,
                metadata={"description": "Фото пользователя"},
            )
        elif message is replied_media_event or getattr(message, "id", None) == 555:
            yield ProcessedMedia(
                media_type="photo",
                mime_type="image/jpeg",
                file_path=replied_photo_path,
                metadata={"description": "Фото из реплая"},
            )
        else:
            yield None

    media_proc.managed_media = fake_managed_media  # type: ignore[method-assign]

    orchestrator = GroupBotOrchestrator(
        config_mgr=config_mgr,
        memory_store=memory_store,
        media_processor=media_proc,
        provider=provider,
        userbot=mock_userbot,
    )
    orchestrator.coalesce_window_sec = 0.01

    # Seed bot message so is_reply_to_bot works if checked via DB too
    await memory_store.save_message(
        ChatMessage(
            chat_id=chat_id,
            telegram_msg_id=500,
            user_id=8710978944,
            author_name="піструнькиня",
            text="покажи",
            timestamp=1700000000,
        )
    )

    # 2. Captionless photo sent directly as a reply to the bot must NOT be skipped
    inc_direct = IncomingMessage(
        msg_id=601,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text="",
        reply_to_msg_id=500,
        is_reply=True,
        is_mention=False,
        raw_event=direct_event,
        media=direct_event.media,
    )
    res_direct = await orchestrator.handle_incoming_message(inc_direct)
    assert res_direct == "Вижу на фото рыжего кота."
    cli_stub.generate_response.assert_awaited_once()
    direct_kwargs = cli_stub.generate_response.call_args.kwargs
    assert len(direct_kwargs["media_attachments"]) == 1
    assert direct_kwargs["media_attachments"][0].file_path == photo_path

    # 3. Text reply referencing another user's photo downloads reply_to_media_event and passes it
    cli_stub.generate_response.reset_mock()
    reply_text_event = BotAPIMessageEvent(
        {
            "message_id": 602,
            "chat": {"id": chat_id, "type": "supergroup"},
            "from": {"id": 447164588, "first_name": "Анюта", "username": "anyuta"},
            "text": "піструнькиня, что на картинке?",
            "reply_to_message": {
                "message_id": 555,
                "from": {"id": 999001, "first_name": "Вася"},
                "caption": "смотри",
                "photo": [{"file_id": "photo_file_1", "width": 100, "height": 100, "file_size": 1024}],
            },
        },
        bot_id=8710978944,
    )

    inc_reply = IncomingMessage(
        msg_id=602,
        chat_id=chat_id,
        sender_id=447164588,
        sender_name="Анюта",
        text="піструнькиня, что на картинке?",
        reply_to_msg_id=555,
        is_reply=True,
        is_mention=True,
        raw_event=reply_text_event,
        media=None,
        reply_to_text=reply_text_event.reply_to_text,
        reply_to_author=reply_text_event.reply_to_author,
        reply_to_media_event=reply_text_event.reply_to_media_event,
    )
    res_reply = await orchestrator.handle_incoming_message(inc_reply)
    assert res_reply == "Вижу на фото рыжего кота."
    cli_stub.generate_response.assert_awaited_once()
    reply_kwargs = cli_stub.generate_response.call_args.kwargs
    reply_prompt = reply_kwargs.get("prompt") or (
        cli_stub.generate_response.call_args.args[0]
        if cli_stub.generate_response.call_args.args
        else ""
    )
    assert len(reply_kwargs["media_attachments"]) == 1
    assert reply_kwargs["media_attachments"][0].file_path == replied_photo_path
    assert "[Ответ на сообщение (Вася): смотри]" in reply_prompt
    assert "[Вложение из сообщения от Вася, на которое отвечают:" in reply_prompt

    await orchestrator.stop()
    await memory_store.close()






