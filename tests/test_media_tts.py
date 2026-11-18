"""Unit tests for TTS synthesis, language detection, sanitization, and OGG/Opus conversion."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from bot.media import tts as tts_module
from bot.media.audio import resolve_ffmpeg_path
from bot.media.tts import (
    TTSError,
    TTSSkippedError,
    TTSSynthesizer,
    detect_speech_language,
    probe_audio_duration,
    resolve_ffprobe_path,
    sanitize_text_for_tts,
)


def _create_sine_wav(dest: Path, duration: float = 1.2) -> None:
    """Create a valid WAV file via FFmpeg (no external mp3 encoder needed)."""
    ffmpeg_bin = resolve_ffmpeg_path()
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_detect_speech_language():
    assert detect_speech_language("Привіт, як твої справи?") == "uk"
    assert detect_speech_language("Їжак біля ґанку, є новини") == "uk"
    assert detect_speech_language("М'ясо та п'ять яблук") == "uk"
    assert detect_speech_language("Що це зараз, дуже добре, дякую!") == "uk"

    assert detect_speech_language("Привет, это отличный тёплый день!") == "ru"
    assert detect_speech_language("Объём данных вырос") == "ru"
    assert detect_speech_language("Окей, давай посмотрим") == "ru"
    assert detect_speech_language("OK 123") == "ru"


def test_sanitize_text_for_tts_cleans_markdown_and_urls():
    raw = (
        "# Заголовок ответа\n"
        "- **Первый** пункт с `inline_code` и _курсивом_\n"
        "2. Смотри [документацию](https://example.com/very/long/path?foo=bar&baz=123)\n"
        "Детали здесь: https://example.org/some/really/long/url/that/should/be/removed\n"
        "Идентификатор snake_case_var остаётся нетронутым."
    )
    cleaned = sanitize_text_for_tts(raw, max_chars=600)
    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "https://" not in cleaned
    assert "документацию" in cleaned
    assert "Первый пункт с inline_code и курсивом" in cleaned
    assert "snake_case_var" in cleaned


def test_sanitize_text_for_tts_rejections():
    with pytest.raises(TTSSkippedError) as exc_empty:
        sanitize_text_for_tts("   \n  ")
    assert exc_empty.value.reason == "empty"

    with pytest.raises(TTSSkippedError) as exc_url_only:
        sanitize_text_for_tts("https://example.com/only-a-url")
    assert exc_url_only.value.reason == "empty"

    with pytest.raises(TTSSkippedError) as exc_code:
        sanitize_text_for_tts("Вот код:\n```python\nprint('hi')\n```")
    assert exc_code.value.reason == "code_block"

    with pytest.raises(TTSSkippedError) as exc_table:
        sanitize_text_for_tts("| Name | Score |\n| --- | --- |\n| Alice | 10 |")
    assert exc_table.value.reason == "table"

    with pytest.raises(TTSSkippedError) as exc_long:
        sanitize_text_for_tts("А" * 120, max_chars=50)
    assert exc_long.value.reason == "too_long"


def test_sanitize_checks_length_after_stripping_long_urls():
    long_url = "https://example.com/" + ("a" * 300)
    raw = f"Вот [короткий ответ]({long_url})!"
    cleaned = sanitize_text_for_tts(raw, max_chars=50)
    assert cleaned == "Вот короткий ответ!"


@pytest.mark.asyncio
async def test_tts_synthesizer_produces_48khz_mono_ogg_opus_and_cleans_up(tmp_path, monkeypatch):
    captured_calls: list[dict[str, str]] = []
    sample_wav = tmp_path / "fixture.wav"
    _create_sine_wav(sample_wav, duration=1.5)

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz"):
            captured_calls.append({"text": text, "voice": voice, "rate": rate, "pitch": pitch})

        async def save(self, filename: str) -> None:
            Path(filename).write_bytes(sample_wav.read_bytes())

    monkeypatch.setattr(tts_module, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate))

    cfg = SimpleNamespace(
        enabled=True,
        ru_voice="ru-RU-DmitryNeural",
        uk_voice="uk-UA-OstapNeural",
        rate="+5%",
        pitch="-2Hz",
        max_chars=600,
        timeout_sec=20.0,
    )
    base_tmp = tmp_path / "tts_tmp"
    synth = TTSSynthesizer(config=cfg, temp_dir_base=str(base_tmp))

    # 1. Ukrainian text auto-detection
    async with synth.managed_speech("**Привіт!** Як твої справи? [лінk](https://x.com)") as result:
        assert result.language == "uk"
        assert result.voice == "uk-UA-OstapNeural"
        assert result.audio_path.exists()
        assert result.audio_path.suffix == ".ogg"
        assert result.duration >= 1.0
        assert "Привіт! Як твої справи? лінk" == result.sanitized_text

        # Verify actual OGG/Opus 48kHz mono properties via ffprobe (or ffmpeg -i fallback)
        ffmpeg_bin = resolve_ffmpeg_path()
        ffprobe_bin = resolve_ffprobe_path(ffmpeg_bin)
        if ffprobe_bin is not None:
            probe = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels",
                    "-of",
                    "json",
                    str(result.audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            assert stream["codec_name"] == "opus"
            assert int(stream["sample_rate"]) == 48000
            assert int(stream["channels"]) == 1
        else:
            inspect_proc = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-i", str(result.audio_path), "-f", "null", "-"],
                check=False,
                capture_output=True,
                text=True,
            )
            stderr_lower = inspect_proc.stderr.lower()
            assert "opus" in stderr_lower
            assert "48000 hz" in stderr_lower
            assert "mono" in stderr_lower

        managed_dir = result.temp_dir
        assert managed_dir is not None and managed_dir.exists()

    # Deterministic cleanup after managed_speech exits (no raw or ogg temp dirs left)
    assert list(base_tmp.glob("bot_tts_*")) == []

    # 2. Russian text auto-detection
    async with synth.managed_speech("Привет, всё работает отлично!") as ru_res:
        assert ru_res.language == "ru"
        assert ru_res.voice == "ru-RU-DmitryNeural"

    assert captured_calls[0]["rate"] == "+5%"
    assert captured_calls[0]["pitch"] == "-2Hz"
    assert list(base_tmp.glob("bot_tts_*")) == []


@pytest.mark.asyncio
async def test_tts_synthesizer_timeout_and_error_cleanup(tmp_path, monkeypatch):
    class HangingCommunicate:
        def __init__(self, **kwargs):
            pass

        async def save(self, filename: str) -> None:
            await asyncio.sleep(5.0)

    monkeypatch.setattr(tts_module, "edge_tts", SimpleNamespace(Communicate=HangingCommunicate))

    base_tmp = tmp_path / "tts_tmp"
    synth = TTSSynthesizer(timeout_sec=0.05, temp_dir_base=str(base_tmp))

    with pytest.raises(TTSError, match="timed out"):
        await synth.synthesize("Привет, тест таймаута")

    # No leaked temp dirs on error
    assert list(base_tmp.glob("bot_tts_*")) == []


@pytest.mark.asyncio
async def test_tts_synthesizer_disabled_or_missing_dependency(tmp_path, monkeypatch):
    disabled = TTSSynthesizer(enabled=False, temp_dir_base=str(tmp_path))
    with pytest.raises(TTSSkippedError) as exc_disabled:
        await disabled.synthesize("Привет")
    assert exc_disabled.value.reason == "disabled"

    monkeypatch.setattr(tts_module, "edge_tts", None)
    enabled = TTSSynthesizer(enabled=True, temp_dir_base=str(tmp_path))
    with pytest.raises(TTSError, match="edge-tts package is not installed"):
        await enabled.synthesize("Привет")


@pytest.mark.asyncio
async def test_probe_audio_duration_ffmpeg_fallback(tmp_path):
    wav_file = tmp_path / "sample.wav"
    _create_sine_wav(wav_file, duration=1.3)

    with patch("bot.media.tts.resolve_ffprobe_path", return_value=None):
        duration = await probe_audio_duration(wav_file)

    assert 1.1 <= duration <= 1.6
