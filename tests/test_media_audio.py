"""Unit tests for multimodal audio transcoding and video note processing with FFmpeg."""

import os
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PIL import Image

from bot.media.audio import (
    AudioProcessingError,
    extract_video_note_assets,
    resolve_ffmpeg_path,
    transcode_voice_to_wav,
)


def test_resolve_ffmpeg_path():
    # 1. Default static resolution
    path = resolve_ffmpeg_path()
    assert os.path.exists(path)
    assert os.access(path, os.X_OK)

    # 2. Custom path resolution
    custom_path = resolve_ffmpeg_path(path)
    assert custom_path == path

    # 3. Nonexistent path with mock fallback failure
    with patch("os.path.isfile", return_value=False), patch("shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="FFmpeg binary not found"):
            resolve_ffmpeg_path(custom_path="/nonexistent/ffmpeg")


@pytest.fixture
def synthetic_voice_ogg(tmp_path) -> Path:
    """Generate a genuine 1-second Opus .ogg audio file via FFmpeg."""
    ffmpeg_bin = resolve_ffmpeg_path()
    ogg_file = tmp_path / "voice.ogg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1",
        "-c:a",
        "libopus",
        str(ogg_file),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ogg_file


@pytest.fixture
def synthetic_video_mp4(tmp_path) -> Path:
    """Generate a genuine 2-second H.264/AAC .mp4 video note via FFmpeg."""
    ffmpeg_bin = resolve_ffmpeg_path()
    mp4_file = tmp_path / "video_note.mp4"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(mp4_file),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return mp4_file


@pytest.mark.asyncio
async def test_transcode_voice_to_wav(synthetic_voice_ogg, tmp_path):
    wav_out = tmp_path / "transcoded.wav"
    result = await transcode_voice_to_wav(synthetic_voice_ogg, wav_out)

    assert result == wav_out
    assert wav_out.exists()
    assert wav_out.stat().st_size > 0

    # Validate WAV properties using standard wave module
    with wave.open(str(wav_out), "rb") as wf:
        assert wf.getnchannels() == 1  # mono
        assert wf.getframerate() == 16000  # 16kHz
        assert wf.getsampwidth() == 2  # 16-bit PCM


@pytest.mark.asyncio
async def test_extract_video_note_assets(synthetic_video_mp4, tmp_path):
    audio_out = tmp_path / "audio.wav"
    frame_out = tmp_path / "keyframe.jpg"

    audio_res, frame_res = await extract_video_note_assets(
        input_path=synthetic_video_mp4,
        audio_output_path=audio_out,
        keyframe_output_path=frame_out,
    )

    assert audio_res == audio_out
    assert frame_res == frame_out
    assert audio_out.exists()
    assert frame_out.exists()

    # Validate audio WAV format
    with wave.open(str(audio_out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000

    # Validate keyframe JPEG format via Pillow
    with Image.open(frame_out) as img:
        assert img.format == "JPEG"
        assert img.size == (320, 240)


@pytest.mark.asyncio
async def test_transcode_missing_input_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        await transcode_voice_to_wav(tmp_path / "does_not_exist.ogg", tmp_path / "out.wav")


@pytest.mark.asyncio
async def test_transcode_corrupt_input_raises_audio_processing_error(tmp_path):
    corrupt_file = tmp_path / "corrupt.ogg"
    corrupt_file.write_bytes(b"NOT_A_VALID_OGG_HEADER_DATA_12345")

    with pytest.raises(AudioProcessingError, match="FFmpeg command exited with code"):
        await transcode_voice_to_wav(corrupt_file, tmp_path / "out.wav")
