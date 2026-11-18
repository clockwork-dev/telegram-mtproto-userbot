"""Unit tests for STTTranscriber, WAV loading, language filtering, timeouts, and fallbacks."""

import asyncio
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from bot.config.models import STTConfig
from bot.media import stt as stt_module
from bot.media.audio import resolve_ffmpeg_path
from bot.media.stt import (
    STTDurationError,
    STTLanguageError,
    STTResult,
    STTTimeoutError,
    STTTranscriber,
    STTUnavailableError,
    is_stt_available,
    load_wav_audio,
    probe_wav_duration,
)


def _write_pcm16_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
    nframes = int(duration_sec * sample_rate)
    silence = b"\x01\x00" * nframes
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)
    return path


def test_probe_and_load_wav_audio(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "sample.wav", duration_sec=1.5, sample_rate=16000)
    assert probe_wav_duration(wav_path) == pytest.approx(1.5, rel=1e-3)

    samples, duration = load_wav_audio(wav_path)
    assert duration == pytest.approx(1.5, rel=1e-3)
    assert len(samples) == 24000


@pytest.mark.asyncio
async def test_stt_graceful_degradation_when_package_missing(monkeypatch, tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "voice.wav", duration_sec=1.0)
    monkeypatch.setattr(stt_module, "WhisperModel", None)
    assert is_stt_available() is False

    transcriber = STTTranscriber()
    with pytest.raises(STTUnavailableError, match="not installed"):
        await transcriber.transcribe(wav_path)
    transcriber.close()


@pytest.mark.asyncio
async def test_stt_disabled_raises_unavailable(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "voice.wav", duration_sec=1.0)
    transcriber = STTTranscriber(enabled=False, model_factory=lambda *a, **kw: None)
    with pytest.raises(STTUnavailableError, match="disabled"):
        await transcriber.transcribe(wav_path)
    transcriber.close()


@pytest.mark.asyncio
async def test_stt_lazy_thread_safe_model_loading_and_config_binding(tmp_path):
    wav1 = _write_pcm16_wav(tmp_path / "v1.wav", duration_sec=1.0)
    wav2 = _write_pcm16_wav(tmp_path / "v2.wav", duration_sec=1.2)

    factory_calls: list[tuple[tuple, dict]] = []
    transcribe_calls: list[dict] = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            transcribe_calls.append({"audio": audio, "kwargs": kwargs})
            segs = [SimpleNamespace(text="Привет"), SimpleNamespace(text="мир")]
            info = SimpleNamespace(language="ru", language_probability=0.985)
            return iter(segs), info

    def fake_factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        time.sleep(0.03)
        return FakeModel()

    cfg = STTConfig(
        enabled=True,
        model_size="base",
        compute_type="int8",
        cpu_threads=2,
        max_duration_sec=120,
        timeout_sec=10.0,
        allowed_languages=["ru", "uk"],
        model_dir=str(tmp_path / "whisper_models"),
    )
    transcriber = STTTranscriber(config=cfg, model_factory=fake_factory)
    assert len(factory_calls) == 0

    res1, res2 = await asyncio.gather(
        transcriber.transcribe(wav1),
        transcriber.transcribe(wav2),
    )
    transcriber.close()

    assert len(factory_calls) == 1
    args, kwargs = factory_calls[0]
    assert args[0] == "base"
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"
    assert kwargs["cpu_threads"] == 2
    assert Path(kwargs["download_root"]) == (tmp_path / "whisper_models").resolve()

    assert res1 == STTResult(text="Привет мир", language="ru", duration=1.0, language_probability=0.985)
    assert res2 == STTResult(text="Привет мир", language="ru", duration=1.2, language_probability=0.985)
    # Audio must be passed as decoded samples (not path string) to bypass PyAV bug on Python 3.13
    assert not isinstance(transcribe_calls[0]["audio"], (str, Path))
    assert len(transcribe_calls[0]["audio"]) == 16000


@pytest.mark.asyncio
async def test_stt_ukrainian_auto_detect(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "uk.wav", duration_sec=2.0)

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            segs = [SimpleNamespace(text="Привіт, як твої справи?")]
            info = SimpleNamespace(language="uk", language_probability=0.962)
            return iter(segs), info

    transcriber = STTTranscriber(model_factory=lambda *a, **kw: FakeModel())
    res = await transcriber.transcribe(wav_path)
    transcriber.close()

    assert res.text == "Привіт, як твої справи?"
    assert res.language == "uk"
    assert res.duration == 2.0
    assert res.language_probability == 0.962


@pytest.mark.asyncio
async def test_stt_disallowed_language_retries_with_best_allowed_ru_or_uk(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "short_uk.wav", duration_sec=1.1)
    seen_languages: list[str | None] = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            lang = kwargs.get("language")
            seen_languages.append(lang)
            if lang is None:
                return (
                    iter([SimpleNamespace(text="pryvit")]),
                    SimpleNamespace(
                        language="pl",
                        language_probability=0.41,
                        all_language_probs=[("pl", 0.41), ("uk", 0.37), ("ru", 0.14)],
                    ),
                )
            assert lang == "uk"
            return (
                iter([SimpleNamespace(text="Привіт!")]),
                SimpleNamespace(language="uk", language_probability=0.88),
            )

    transcriber = STTTranscriber(
        allowed_languages=["ru", "uk"],
        model_factory=lambda *a, **kw: FakeModel(),
    )
    res = await transcriber.transcribe(wav_path)
    transcriber.close()

    assert seen_languages == [None, "uk"]
    assert res.language == "uk"
    assert res.text == "Привіт!"


@pytest.mark.asyncio
async def test_stt_strict_language_filter_raises(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "en.wav", duration_sec=1.0)

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return (
                iter([SimpleNamespace(text="Hello there")]),
                SimpleNamespace(language="en", language_probability=0.95),
            )

    transcriber = STTTranscriber(
        allowed_languages=["ru", "uk"],
        strict_language_filter=True,
        model_factory=lambda *a, **kw: FakeModel(),
    )
    with pytest.raises(STTLanguageError, match="en"):
        await transcriber.transcribe(wav_path)
    transcriber.close()


@pytest.mark.asyncio
async def test_stt_duration_limit_enforced_before_model_load(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "long.wav", duration_sec=6.0)
    factory_called = False

    def fake_factory(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        return None

    transcriber = STTTranscriber(max_duration_sec=5, model_factory=fake_factory)
    with pytest.raises(STTDurationError, match="exceeds max_duration_sec=5"):
        await transcriber.transcribe(wav_path)
    transcriber.close()
    assert factory_called is False


@pytest.mark.asyncio
async def test_stt_timeout_does_not_block_event_loop(tmp_path):
    wav_path = _write_pcm16_wav(tmp_path / "slow.wav", duration_sec=1.0)

    class SlowModel:
        def transcribe(self, audio, **kwargs):
            time.sleep(0.3)
            return iter([SimpleNamespace(text="slow")]), SimpleNamespace(language="ru", language_probability=0.9)

    transcriber = STTTranscriber(
        timeout_sec=0.05,
        model_factory=lambda *a, **kw: SlowModel(),
    )

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(10):
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    with pytest.raises(STTTimeoutError, match="timed out"):
        await transcriber.transcribe(wav_path)
    await ticker_task
    transcriber.close()
    assert ticks >= 3


@pytest.mark.asyncio
async def test_stt_transcodes_non_wav_input(tmp_path):
    ffmpeg_bin = resolve_ffmpeg_path()
    ogg_path = tmp_path / "voice.ogg"
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "libopus", str(ogg_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return (
                iter([SimpleNamespace(text="Тест из ogg")]),
                SimpleNamespace(language="ru", language_probability=0.99),
            )

    transcriber = STTTranscriber(model_factory=lambda *a, **kw: FakeModel())
    res = await transcriber.transcribe(ogg_path)
    transcriber.close()

    assert res.text == "Тест из ogg"
    assert res.language == "ru"
    assert res.duration == pytest.approx(1.0, abs=0.15)
