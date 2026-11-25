"""Speech-to-Text (STT) local transcription via faster-whisper.

Provides thread-safe lazy model loading, non-blocking async inference on a
dedicated single-worker executor, direct 16kHz PCM WAV loading via stdlib
``wave`` + ``numpy`` (bypassing PyAV issues on Python 3.13), duration and
timeout guards, language validation/constraining for Russian (``ru``) and
Ukrainian (``uk``), and graceful degradation when ``faster-whisper`` is absent.
"""

from __future__ import annotations

import array
import asyncio
import gc
import logging
import shutil
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    WhisperModel = None  # type: ignore[assignment,misc]

from bot.media.audio import transcode_voice_to_wav

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_CPU_THREADS = 2
DEFAULT_MAX_DURATION_SEC = 120
DEFAULT_TIMEOUT_SEC = 45.0
DEFAULT_ALLOWED_LANGUAGES = ("ru", "uk")
DEFAULT_MODEL_DIR = "data/models/whisper"
DEFAULT_IDLE_UNLOAD_SEC = 600.0
WHISPER_SAMPLE_RATE = 16000

_UNSET = object()


class STTError(Exception):
    """Base error raised during speech-to-text transcription."""


class STTUnavailableError(STTError):
    """Raised when STT is disabled or required packages are not installed."""


class STTDurationError(STTError):
    """Raised when audio is empty or exceeds the configured duration limit."""


class STTTimeoutError(STTError):
    """Raised when STT inference exceeds the configured timeout."""


class STTLanguageError(STTError):
    """Raised when detected language violates strict allowed_languages filtering."""


@dataclass
class STTResult:
    """Structured transcription result."""

    text: str
    language: str
    duration: float
    language_probability: Optional[float] = None


def is_stt_available() -> bool:
    """Return True if faster-whisper and numpy are importable in this environment."""
    return WhisperModel is not None and np is not None


def probe_wav_duration(wav_path: Path | str) -> float:
    """Read WAV header and return audio duration in seconds without decoding samples."""
    target = Path(wav_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Audio file not found: {target}")

    try:
        with wave.open(str(target), "rb") as wf:
            framerate = wf.getframerate()
            nframes = wf.getnframes()
    except wave.Error as exc:
        raise STTError(f"Invalid WAV file '{target.name}': {exc}") from exc

    if framerate <= 0 or nframes <= 0:
        raise STTDurationError(f"WAV file '{target.name}' is empty (frames={nframes}, rate={framerate}).")

    return nframes / float(framerate)


def load_wav_audio(wav_path: Path | str) -> tuple[Any, float]:
    """Load 16-bit PCM WAV into a normalized 16kHz mono float32 array and duration.

    Reads frames directly via Python's stdlib ``wave`` module so ``faster-whisper``
    does not invoke PyAV (which breaks under Python 3.13 + av>=19 metadata_errors).
    """
    target = Path(wav_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Audio file not found: {target}")

    try:
        with wave.open(str(target), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_frames = wf.readframes(nframes)
    except wave.Error as exc:
        raise STTError(f"Invalid WAV file '{target.name}': {exc}") from exc

    if sampwidth != 2:
        raise STTError(
            f"Unsupported WAV sample width {sampwidth} bytes in '{target.name}' (expected 16-bit PCM)."
        )
    if nchannels <= 0 or framerate <= 0 or nframes <= 0 or not raw_frames:
        raise STTDurationError(f"WAV file '{target.name}' contains no audio frames.")

    duration = nframes / float(framerate)

    if np is not None:
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
        if nchannels > 1:
            usable = (len(samples) // nchannels) * nchannels
            samples = samples[:usable].reshape(-1, nchannels).mean(axis=1).astype(np.float32)
        if framerate != WHISPER_SAMPLE_RATE and len(samples) > 1:
            target_len = max(1, int(round(len(samples) * WHISPER_SAMPLE_RATE / float(framerate))))
            x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            samples = np.interp(x_new, x_old, samples).astype(np.float32)
        return samples, duration

    # Pure-stdlib fallback when numpy is absent (e.g., unit tests with a mocked model).
    pcm = array.array("h")
    pcm.frombytes(raw_frames[: (len(raw_frames) // 2) * 2])
    if nchannels > 1:
        mono = [
            sum(pcm[i : i + nchannels]) / (32768.0 * nchannels)
            for i in range(0, (len(pcm) // nchannels) * nchannels, nchannels)
        ]
        return mono, duration
    return [sample / 32768.0 for sample in pcm], duration


def _normalize_languages(languages: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not languages:
        return ()
    normalized: list[str] = []
    for lang in languages:
        cleaned = str(lang).strip().lower()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _select_best_allowed_language(
    allowed_languages: tuple[str, ...],
    info: Any,
) -> tuple[str, Optional[float]]:
    """Pick the highest-probability allowed language from Whisper info, or first allowed."""
    all_probs = getattr(info, "all_language_probs", None)
    if all_probs:
        best_lang: Optional[str] = None
        best_prob = -1.0
        for item in all_probs:
            try:
                lang_code, prob = item[0], float(item[1])
            except (TypeError, IndexError, ValueError):
                continue
            lang_norm = str(lang_code).strip().lower()
            if lang_norm in allowed_languages and prob > best_prob:
                best_lang = lang_norm
                best_prob = prob
        if best_lang is not None:
            return best_lang, best_prob

    return allowed_languages[0], None


class STTTranscriber:
    """Async thread-safe Speech-to-Text transcriber for Russian and Ukrainian voice notes."""

    def __init__(
        self,
        config: Optional[Any] = None,
        *,
        enabled: Optional[bool] = None,
        model_size: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        max_duration_sec: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        allowed_languages: Optional[Sequence[str]] = None,
        model_dir: Any = _UNSET,
        idle_unload_seconds: Optional[float] = None,
        ffmpeg_path: Optional[str] = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        strict_language_filter: bool = False,
        model_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.enabled: bool = (
            enabled if enabled is not None else bool(getattr(config, "enabled", True))
        )
        self.model_size: str = (
            model_size
            if model_size is not None
            else str(getattr(config, "model_size", DEFAULT_MODEL_SIZE))
        )
        self.compute_type: str = (
            compute_type
            if compute_type is not None
            else str(getattr(config, "compute_type", DEFAULT_COMPUTE_TYPE))
        )
        self.cpu_threads: int = (
            cpu_threads
            if cpu_threads is not None
            else int(getattr(config, "cpu_threads", DEFAULT_CPU_THREADS))
        )
        self.max_duration_sec: int = (
            max_duration_sec
            if max_duration_sec is not None
            else int(getattr(config, "max_duration_sec", DEFAULT_MAX_DURATION_SEC))
        )
        self.timeout_sec: float = (
            timeout_sec
            if timeout_sec is not None
            else float(getattr(config, "timeout_sec", DEFAULT_TIMEOUT_SEC))
        )

        if allowed_languages is not None:
            self.allowed_languages: tuple[str, ...] = _normalize_languages(allowed_languages)
        else:
            cfg_langs = getattr(config, "allowed_languages", DEFAULT_ALLOWED_LANGUAGES)
            self.allowed_languages = _normalize_languages(cfg_langs)

        if model_dir is not _UNSET:
            self.model_dir: Optional[str] = str(model_dir) if model_dir else None
        else:
            cfg_dir = getattr(config, "model_dir", DEFAULT_MODEL_DIR)
            self.model_dir = str(cfg_dir) if cfg_dir else None

        raw_idle = (
            idle_unload_seconds
            if idle_unload_seconds is not None
            else getattr(config, "idle_unload_seconds", DEFAULT_IDLE_UNLOAD_SEC)
        )
        self.idle_unload_seconds: float = max(0.0, float(raw_idle or 0.0))

        self.ffmpeg_path = ffmpeg_path
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.strict_language_filter = strict_language_filter
        self._model_factory = model_factory

        self._model: Any = None
        self._model_lock = threading.Lock()
        self._active_jobs: int = 0
        self._last_used_mono: float = 0.0
        self._idle_timer: Optional[threading.Timer] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-whisper")
        self._warned_unavailable = False

    def _cancel_idle_timer_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_unload_locked(self) -> None:
        self._cancel_idle_timer_locked()
        if self._model is None or self.idle_unload_seconds <= 0.0 or self._active_jobs > 0:
            return
        timer = threading.Timer(self.idle_unload_seconds, self._on_idle_unload_timer)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _on_idle_unload_timer(self) -> None:
        unloaded = False
        with self._model_lock:
            self._idle_timer = None
            if self._model is None or self._active_jobs > 0 or self.idle_unload_seconds <= 0.0:
                return
            elapsed = time.monotonic() - self._last_used_mono
            if elapsed + 0.05 < self.idle_unload_seconds:
                self._schedule_idle_unload_locked()
                return
            self._model = None
            unloaded = True
        if unloaded:
            gc.collect()
            logger.info(
                "Unloaded idle faster-whisper model (size=%s) after %.0fs of inactivity.",
                self.model_size,
                self.idle_unload_seconds,
            )

    def unload_model(self) -> bool:
        """Explicitly release cached WhisperModel from RAM."""
        with self._model_lock:
            self._cancel_idle_timer_locked()
            if self._model is None or self._active_jobs > 0:
                return False
            self._model = None
        gc.collect()
        return True

    def _get_or_load_model(self) -> Any:
        """Lazily load and cache the WhisperModel in a thread-safe manner."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            factory = self._model_factory or WhisperModel
            if factory is None or (self._model_factory is None and np is None):
                if not self._warned_unavailable:
                    logger.warning(
                        "faster-whisper or numpy is not installed; local STT is disabled."
                    )
                    self._warned_unavailable = True
                raise STTUnavailableError(
                    "faster-whisper package is not installed; cannot transcribe audio."
                )

            download_root: Optional[str] = None
            if self.model_dir:
                model_path = Path(self.model_dir).resolve()
                model_path.mkdir(parents=True, exist_ok=True)
                download_root = str(model_path)

            logger.info(
                "Loading faster-whisper model size=%s compute_type=%s cpu_threads=%s model_dir=%s",
                self.model_size,
                self.compute_type,
                self.cpu_threads,
                download_root,
            )
            self._model = factory(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=download_root,
            )
            return self._model

    def _run_model_transcribe(
        self,
        model: Any,
        audio_array: Any,
        language: Optional[str],
    ) -> tuple[str, Any]:
        """Execute synchronous Whisper inference and materialize all segments."""
        kwargs: dict[str, Any] = {
            "language": language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        try:
            segments, info = model.transcribe(audio_array, **kwargs)
        except TypeError:
            # Fallback if a simplified mock or backend omits optional kwargs.
            segments, info = model.transcribe(audio_array, language=language)

        parts: list[str] = []
        for seg in segments:
            seg_text = getattr(seg, "text", "")
            if isinstance(seg_text, str):
                cleaned = seg_text.strip()
                if cleaned:
                    parts.append(cleaned)
        return " ".join(parts).strip(), info

    def _transcribe_sync(
        self,
        wav_path: Path,
        language: Optional[str] = None,
    ) -> STTResult:
        """Synchronous transcription workflow executed inside the worker thread."""
        with self._model_lock:
            self._active_jobs += 1
            self._cancel_idle_timer_locked()
        try:
            model = self._get_or_load_model()
            audio_array, duration = load_wav_audio(wav_path)

            forced_lang: Optional[str] = None
            if language is not None:
                forced_lang = language.strip().lower()
                if self.allowed_languages and forced_lang not in self.allowed_languages:
                    raise STTLanguageError(
                        f"Requested language '{forced_lang}' is not in allowed_languages={list(self.allowed_languages)}"
                    )
            elif len(self.allowed_languages) == 1:
                forced_lang = self.allowed_languages[0]

            text, info = self._run_model_transcribe(model, audio_array, language=forced_lang)

            detected_lang = (
                str(getattr(info, "language", "") or forced_lang or "").strip().lower()
            )
            raw_prob = getattr(info, "language_probability", None)
            lang_prob: Optional[float] = float(raw_prob) if raw_prob is not None else None

            if forced_lang is None and self.allowed_languages and detected_lang not in self.allowed_languages:
                if self.strict_language_filter:
                    raise STTLanguageError(
                        f"Detected language '{detected_lang}' is not in allowed_languages={list(self.allowed_languages)}"
                    )

                best_lang, best_prob = _select_best_allowed_language(self.allowed_languages, info)
                logger.debug(
                    "STT detected disallowed language '%s' (p=%s); retrying with '%s'",
                    detected_lang,
                    lang_prob,
                    best_lang,
                )
                text, retry_info = self._run_model_transcribe(model, audio_array, language=best_lang)
                detected_lang = best_lang
                retry_prob = getattr(retry_info, "language_probability", None)
                lang_prob = best_prob if best_prob is not None else (
                    float(retry_prob) if retry_prob is not None else None
                )

            return STTResult(
                text=text,
                language=detected_lang,
                duration=round(duration, 3),
                language_probability=round(lang_prob, 4) if lang_prob is not None else None,
            )
        finally:
            with self._model_lock:
                self._active_jobs = max(0, self._active_jobs - 1)
                self._last_used_mono = time.monotonic()
                self._schedule_idle_unload_locked()

    async def transcribe(
        self,
        audio_path: Path | str,
        *,
        language: Optional[str] = None,
    ) -> STTResult:
        """Asynchronously transcribe an audio file (16kHz PCM WAV or transcoded via FFmpeg)."""
        if not self.enabled:
            raise STTUnavailableError("STT transcription is disabled in configuration.")

        target = Path(audio_path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Audio file not found: {target}")

        if self._model_factory is None and not is_stt_available():
            if not self._warned_unavailable:
                logger.warning(
                    "faster-whisper or numpy is not installed; skipping STT for %s.",
                    target.name,
                )
                self._warned_unavailable = True
            raise STTUnavailableError(
                "faster-whisper package is not installed; cannot transcribe audio."
            )

        tmp_dir: Optional[Path] = None
        cleanup_in_caller = True
        wav_target = target
        try:
            if target.suffix.lower() != ".wav":
                tmp_dir = Path(tempfile.mkdtemp(prefix="bot_stt_wav_"))
                wav_target = await transcode_voice_to_wav(
                    input_path=target,
                    output_path=tmp_dir / "audio.wav",
                    ffmpeg_path=self.ffmpeg_path,
                )

            duration = probe_wav_duration(wav_target)
            if duration > self.max_duration_sec:
                raise STTDurationError(
                    f"Audio duration {duration:.1f}s exceeds max_duration_sec={self.max_duration_sec}s"
                )

            loop = asyncio.get_running_loop()

            def _run_and_cleanup() -> STTResult:
                try:
                    return self._transcribe_sync(wav_target, language)
                finally:
                    if tmp_dir is not None:
                        shutil.rmtree(tmp_dir, ignore_errors=True)

            cleanup_in_caller = False
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(self._executor, _run_and_cleanup),
                    timeout=self.timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                raise STTTimeoutError(
                    f"STT transcription timed out after {self.timeout_sec:.1f}s"
                ) from exc
        finally:
            if cleanup_in_caller and tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def close(self) -> None:
        """Shut down the underlying transcription worker thread and release cached model."""
        with self._model_lock:
            self._cancel_idle_timer_locked()
            self._model = None
        self._executor.shutdown(wait=False)
