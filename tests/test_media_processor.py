"""Unit tests for MediaProcessor multimodal routing, conversions, and deterministic cleanup."""

import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PIL import Image

from bot.media.audio import resolve_ffmpeg_path
from bot.media.processor import MediaProcessor


@pytest.fixture
def processor(tmp_path):
    base_tmp = tmp_path / "media_tmp"
    return MediaProcessor(
        max_doc_size_mb=10,
        temp_dir_base=str(base_tmp),
    )


@pytest.fixture
def sample_image_file(tmp_path) -> Path:
    img_path = tmp_path / "source.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def sample_webp_file(tmp_path) -> Path:
    webp_path = tmp_path / "source.webp"
    img = Image.new("RGBA", (64, 64), color="red")
    img.save(webp_path, format="WEBP")
    return webp_path


@pytest.mark.asyncio
async def test_process_photo(processor, sample_image_file):
    # Mock Telethon message with photo
    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = True
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = False

    async def fake_download(msg, file):
        shutil.copy2(sample_image_file, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with processor.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "photo"
        assert processed.mime_type == "image/jpeg"
        assert processed.file_path.exists()
        temp_dir = Path(processed.metadata["_temp_dir"])
        assert temp_dir.exists()

    # Once context manager exits, temp directory must be cleaned up
    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_webp_sticker_converted_to_png(processor, sample_webp_file):
    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = True
    mock_msg.document = False
    mock_msg.file = MagicMock(emoji="🎉")

    async def fake_download(msg, file):
        shutil.copy2(sample_webp_file, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with processor.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "sticker"
        assert processed.mime_type == "image/png"
        assert processed.file_path.suffix == ".png"
        assert processed.metadata.get("emoji") == "🎉"

        # Verify converted file is indeed PNG
        with Image.open(processed.file_path) as img:
            assert img.format == "PNG"

        temp_dir = Path(processed.metadata["_temp_dir"])
        assert temp_dir.exists()

    # Deterministic cleanup check
    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_document(processor, tmp_path):
    doc_src = tmp_path / "notes.txt"
    doc_src.write_text("Hello from processed document!", encoding="utf-8")

    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = True
    file_info = MagicMock()
    file_info.name = "notes.txt"
    file_info.size = len(doc_src.read_bytes())
    file_info.mime_type = "text/plain"
    mock_msg.file = file_info

    async def fake_download(msg, file):
        shutil.copy2(doc_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with processor.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "document"
        assert "Hello from processed document!" in processed.text_content
        temp_dir = Path(processed.metadata["_temp_dir"])
        assert temp_dir.exists()

    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_voice_note(processor, tmp_path):
    # Create genuine test ogg
    ffmpeg_bin = resolve_ffmpeg_path()
    ogg_src = tmp_path / "voice.ogg"
    import subprocess
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "libopus", str(ogg_src)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = True
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = False

    async def fake_download(msg, file):
        shutil.copy2(ogg_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with processor.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "voice"
        assert processed.mime_type == "audio/wav"
        assert processed.audio_path.exists()
        temp_dir = Path(processed.metadata["_temp_dir"])
        assert temp_dir.exists()

    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_media_cleanup_on_error(processor):
    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = True
    mock_msg.voice = False
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = False

    mock_client = MagicMock()
    # Download fails midway
    mock_client.download_media = AsyncMock(side_effect=RuntimeError("Connection reset"))

    # Before execution, count directories in temp_dir_base
    base = Path(processor.temp_dir_base)
    initial_dirs = list(base.glob("bot_media_*"))

    with pytest.raises(RuntimeError, match="Connection reset"):
        await processor.process_media(mock_msg, mock_client)

    # After error, any temporary directory created during the failed attempt must be cleaned up
    current_dirs = list(base.glob("bot_media_*"))
    assert len(current_dirs) == len(initial_dirs)


@pytest.mark.asyncio
async def test_process_message_with_no_media(processor):
    mock_msg = MagicMock()
    mock_msg.media = None
    mock_msg.photo = False
    mock_msg.voice = False

    result = await processor.process_media(mock_msg)
    assert result is None


def test_bot_api_message_event_animation_and_document_keys():
    from bot.telegram.bot_api_client import BotAPIMessageEvent

    msg_dict = {
        "message_id": 42,
        "chat": {"id": -100123456},
        "from": {"id": 999, "first_name": "Женя"},
        "animation": {
            "file_id": "anim_file_id_123",
            "file_name": "super_rjaka_demotivator_bot.mp4",
            "mime_type": "video/mp4",
            "file_size": 245760,
        },
        "document": {
            "file_id": "anim_file_id_123",
            "file_name": "super_rjaka_demotivator_bot.mp4",
            "mime_type": "video/mp4",
            "file_size": 245760,
        },
    }
    ev = BotAPIMessageEvent(msg_dict, bot_id=8710978944)
    assert ev.gif is True
    assert ev.video is True
    assert ev.document is False
    assert ev.media is True
    assert ev._file_id == "anim_file_id_123"
    assert ev.file_id == "anim_file_id_123"
    assert ev.file_name == "super_rjaka_demotivator_bot.mp4"
    assert ev.mime_type == "video/mp4"
    assert ev.file_size == 245760


@pytest.mark.asyncio
async def test_process_mp4_animation_extracts_jpeg_frame_via_ffmpeg(processor, tmp_path):
    import subprocess

    from bot.telegram.bot_api_client import BotAPIMessageEvent

    ffmpeg_bin = resolve_ffmpeg_path()
    mp4_src = tmp_path / "super_rjaka_demotivator_bot.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:d=0.5",
            "-pix_fmt",
            "yuv420p",
            str(mp4_src),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    msg_dict = {
        "message_id": 77,
        "chat": {"id": -100123456},
        "from": {"id": 999, "first_name": "Женя"},
        "animation": {
            "file_id": "anim_77",
            "file_name": "super_rjaka_demotivator_bot.mp4",
            "mime_type": "video/mp4",
            "file_size": mp4_src.stat().st_size,
        },
        "document": {
            "file_id": "anim_77",
            "file_name": "super_rjaka_demotivator_bot.mp4",
            "mime_type": "video/mp4",
            "file_size": mp4_src.stat().st_size,
        },
    }
    ev = BotAPIMessageEvent(msg_dict, bot_id=8710978944)

    async def fake_download(msg, file):
        shutil.copy2(mp4_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    processed = await processor.process_message(ev, mock_client)
    try:
        assert processed is not None
        assert processed.media_type in ("animation", "video")
        assert processed.mime_type == "image/jpeg"
        assert processed.image_path is not None and processed.image_path.exists()
        assert processed.file_path == processed.image_path
        with Image.open(processed.image_path) as img:
            assert img.format == "JPEG"
    finally:
        processor.cleanup(processed)


def test_sync_call_api_retries_on_connection_reset(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    from bot.telegram.bot_api_client import TelegramBotAPIClient

    client = TelegramBotAPIClient(bot_token="123456:ABCDEF")
    attempts = {"count": 0}

    class _DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"ok": true, "result": {"id": 123, "username": "test_bot"}}'

    def fake_urlopen(req, timeout=35.0):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        if attempts["count"] == 2:
            raise urllib.error.URLError(ConnectionResetError(104, "Connection reset by peer"))
        return _DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    res = client._sync_call_api("getMe", timeout=5.0)
    assert res["ok"] is True
    assert attempts["count"] == 3

    # Verify HTTPError raises immediately without retrying
    http_attempts = {"count": 0}

    def fake_http_err(req, timeout=35.0):
        http_attempts["count"] += 1
        raise urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"ok":false,"description":"Bad Request"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_http_err)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client._sync_call_api("sendMessage", {"chat_id": 1, "text": "hi"})
    assert http_attempts["count"] == 1


@pytest.mark.asyncio
async def test_process_voice_note_with_stt_transcriber(tmp_path):
    import subprocess

    from bot.media.stt import STTResult

    ffmpeg_bin = resolve_ffmpeg_path()
    ogg_src = tmp_path / "voice.ogg"
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "libopus", str(ogg_src)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    mock_stt = MagicMock()
    mock_stt.enabled = True
    mock_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Привет, как дела?",
            language="ru",
            duration=1.0,
            language_probability=0.99,
        )
    )

    proc = MediaProcessor(
        max_doc_size_mb=10,
        temp_dir_base=str(tmp_path / "media_tmp"),
        stt_transcriber=mock_stt,
    )

    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = True
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = False

    async def fake_download(msg, file):
        shutil.copy2(ogg_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with proc.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "voice"
        assert processed.audio_path is not None and processed.audio_path.exists()
        assert processed.text_content == "Привет, как дела?"
        assert processed.metadata["transcript"] == "Привет, как дела?"
        assert processed.metadata["language"] == "ru"
        assert processed.metadata["duration"] == 1.0
        mock_stt.transcribe.assert_awaited_once_with(processed.audio_path)
        temp_dir = Path(processed.metadata["_temp_dir"])
        assert temp_dir.exists()

    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_voice_note_stt_failure_graceful_fallback(tmp_path):
    import subprocess

    from bot.media.stt import STTTimeoutError

    ffmpeg_bin = resolve_ffmpeg_path()
    ogg_src = tmp_path / "voice.ogg"
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "libopus", str(ogg_src)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    mock_stt = MagicMock()
    mock_stt.enabled = True
    mock_stt.transcribe = AsyncMock(side_effect=STTTimeoutError("timed out"))

    proc = MediaProcessor(
        max_doc_size_mb=10,
        temp_dir_base=str(tmp_path / "media_tmp"),
        stt_transcriber=mock_stt,
    )

    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = True
    mock_msg.video_note = False
    mock_msg.sticker = False
    mock_msg.document = False

    async def fake_download(msg, file):
        shutil.copy2(ogg_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with proc.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "voice"
        assert processed.audio_path is not None and processed.audio_path.exists()
        assert processed.text_content is None
        assert "transcript" not in processed.metadata
        assert "timed out" in processed.metadata.get("stt_error", "")
        temp_dir = Path(processed.metadata["_temp_dir"])

    assert not temp_dir.exists()


@pytest.mark.asyncio
async def test_process_video_note_with_stt_transcriber(tmp_path):
    import subprocess

    from bot.media.stt import STTResult

    ffmpeg_bin = resolve_ffmpeg_path()
    mp4_src = tmp_path / "video_note.mp4"
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=240x240:d=0.6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.6",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            str(mp4_src),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    mock_stt = MagicMock()
    mock_stt.enabled = True
    mock_stt.transcribe = AsyncMock(
        return_value=STTResult(
            text="Привіт з кружечка",
            language="uk",
            duration=0.6,
            language_probability=0.97,
        )
    )

    proc = MediaProcessor(
        max_doc_size_mb=10,
        temp_dir_base=str(tmp_path / "media_tmp"),
        stt_transcriber=mock_stt,
    )

    mock_msg = MagicMock()
    mock_msg.media = MagicMock()
    mock_msg.photo = False
    mock_msg.voice = False
    mock_msg.video_note = True
    mock_msg.video = False
    mock_msg.gif = False
    mock_msg.animation = False
    mock_msg.sticker = False
    mock_msg.document = False
    mock_msg.file = MagicMock(name="video_note.mp4", size=mp4_src.stat().st_size, mime_type="video/mp4")
    mock_msg.file.name = "video_note.mp4"

    async def fake_download(msg, file):
        shutil.copy2(mp4_src, file)
        return file

    mock_client = MagicMock()
    mock_client.download_media = AsyncMock(side_effect=fake_download)

    async with proc.managed_media(mock_msg, mock_client) as processed:
        assert processed is not None
        assert processed.media_type == "video"
        assert processed.metadata.get("is_video_note") is True
        assert processed.audio_path is not None and processed.audio_path.exists()
        assert processed.keyframe_path is not None and processed.keyframe_path.exists()
        assert processed.text_content == "Привіт з кружечка"
        assert processed.metadata["transcript"] == "Привіт з кружечка"
        assert processed.metadata["language"] == "uk"
        mock_stt.transcribe.assert_awaited_once_with(processed.audio_path)
        temp_dir = Path(processed.metadata["_temp_dir"])

    assert not temp_dir.exists()
