"""Text-to-Speech (TTS) synthesis via edge-tts and FFmpeg (OGG/Opus).

Provides language auto-detection (Russian vs Ukrainian), markdown/URL speech
sanitization, eligibility guards for text fallback, async Edge TTS synthesis,
OGG/Opus (48 kHz mono) transcoding, audio duration probing, and deterministic
temporary file cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional

try:
    import edge_tts  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    edge_tts = None  # type: ignore[assignment]

from bot.media.audio import AudioProcessingError, resolve_ffmpeg_path

logger = logging.getLogger(__name__)

DEFAULT_RU_VOICE = "ru-RU-SvetlanaNeural"
DEFAULT_UK_VOICE = "uk-UA-PolinaNeural"
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_MAX_CHARS = 600
DEFAULT_TIMEOUT_SEC = 25.0

_UK_SPECIFIC_CHARS = frozenset("іїєґІЇЄҐ")
_RU_SPECIFIC_CHARS = frozenset("ыэёъЫЭЁЪ")

# Distinctly Ukrainian words (do not appear as standard Russian words)
_UK_LEXICAL_MARKERS = frozenset({
    "що",
    "це",
    "але",
    "або",
    "чи",
    "якщо",
    "коли",
    "тому",
    "дуже",
    "добре",
    "дякую",
    "дякувати",
    "будь",
    "ласка",
    "справи",
    "треба",
    "нема",
    "зараз",
    "вчора",
    "вже",
    "ще",
    "теж",
    "також",
    "трохи",
    "трошки",
    "гарно",
    "шо",
    "хто",
    "де",
    "куди",
    "чому",
    "робота",
    "бачу",
    "чую",
    "кажу",
    "його",
    "вона",
    "вони",
    "завжди",
    "мабуть",
    "напевно",
    "разом",
    "одразу",
    "швидко",
    "важко",
    "погано",
    "краще",
    "менше",
    "багато",
    "допомога",
    "вибач",
    "перепрошую",
    "будинок",
    "питання",
    "мене",
    "смачного",
    "добраніч",
})

_WORD_RE = re.compile(r"[а-яіїєґёa-z']+", re.IGNORECASE)
_UK_APOSTROPHE_RE = re.compile(r"[бвгґджзклмнпрстфхцчшщ]['’][яюєї]", re.IGNORECASE)

_FENCED_CODE_RE = re.compile(r"(^|\n)\s*(```|~~~)", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:https?://|www\.|mailto:|tg://|/)[^)]*\)")
_MD_REF_LINK_RE = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MD_REF_DEF_RE = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
_BARE_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]+(?:\([^\s<>()]*\))?", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_UNORDERED_LIST_RE = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`+([^`]+)`+")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_BOLD_ITALIC_STAR_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_BOLD_UNDER_RE = re.compile(r"(?<!\w)__(.+?)__(?!\w)")
_ITALIC_UNDER_RE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_DURATION_STDERR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class TTSError(Exception):
    """Raised when TTS synthesis, transcoding, or probing fails."""


class TTSSkippedError(TTSError):
    """Raised when text is unsuitable for speech and should fall back to text."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(slots=True)
class TTSResult:
    """Result of a TTS synthesis run."""

    audio_path: Path
    language: str
    voice: str
    duration: float
    sanitized_text: str
    temp_dir: Optional[Path] = None


def detect_speech_language(text: str) -> str:
    """Detect whether text should be spoken in Ukrainian ('uk') or Russian ('ru')."""
    if not text:
        return "ru"

    uk_chars = sum(1 for ch in text if ch in _UK_SPECIFIC_CHARS)
    ru_chars = sum(1 for ch in text if ch in _RU_SPECIFIC_CHARS)

    if uk_chars > 0 and ru_chars == 0:
        return "uk"
    if uk_chars > 0 and ru_chars > 0:
        return "uk" if uk_chars >= ru_chars else "ru"
    if ru_chars > 0:
        return "ru"

    if _UK_APOSTROPHE_RE.search(text):
        return "uk"

    words = [w.lower() for w in _WORD_RE.findall(text)]
    if any(w in _UK_LEXICAL_MARKERS for w in words):
        return "uk"

    return "ru"


def _contains_markdown_table(text: str) -> bool:
    """Return True if text contains a multi-line Markdown pipe table."""
    lines = text.splitlines()
    for idx in range(len(lines) - 1):
        curr_line = lines[idx].strip()
        next_line = lines[idx + 1].strip()
        if "|" in curr_line and _TABLE_SEP_RE.match(next_line):
            return True
        if _TABLE_ROW_RE.match(curr_line) and _TABLE_ROW_RE.match(next_line):
            return True
    return False


def sanitize_text_for_tts(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Sanitize response text for natural speech, rejecting un-speakable content.

    Raises:
        TTSSkippedError: If text is empty, contains code blocks or tables,
            has no speakable characters left after cleanup, or exceeds max_chars.
    """
    if not text or not text.strip():
        raise TTSSkippedError("empty", "Cannot synthesize empty text.")

    if _FENCED_CODE_RE.search(text):
        raise TTSSkippedError("code_block", "Text contains a fenced code block; falling back to text.")

    if _contains_markdown_table(text):
        raise TTSSkippedError("table", "Text contains a Markdown table; falling back to text.")

    cleaned = text
    cleaned = _MD_IMAGE_RE.sub(r"\1", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MD_REF_LINK_RE.sub(r"\1", cleaned)
    cleaned = _MD_REF_DEF_RE.sub("", cleaned)
    cleaned = _BARE_URL_RE.sub("", cleaned)

    cleaned = _HEADING_RE.sub("", cleaned)
    cleaned = _BLOCKQUOTE_RE.sub("", cleaned)
    cleaned = _HR_RE.sub("", cleaned)
    cleaned = _UNORDERED_LIST_RE.sub("", cleaned)
    cleaned = _ORDERED_LIST_RE.sub("", cleaned)

    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _STRIKE_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_ITALIC_STAR_RE.sub(r"\1", cleaned)
    cleaned = _BOLD_UNDER_RE.sub(r"\1", cleaned)
    cleaned = _ITALIC_UNDER_RE.sub(r"\1", cleaned)

    # Collapse empty parentheses left after stripping bare URLs like "site (https://...)"
    cleaned = re.sub(r"\(\s*\)", "", cleaned)

    # Join lines into natural spoken sentences
    paragraphs: list[str] = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if paragraphs:
            prev = paragraphs[-1]
            if prev[-1] not in ".!?…:;":
                paragraphs[-1] = f"{prev}."
        paragraphs.append(line)

    result = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
    if not result or not any(ch.isalnum() for ch in result):
        raise TTSSkippedError("empty", "Text has no speakable content after sanitization.")

    if len(result) > max_chars:
        raise TTSSkippedError(
            "too_long",
            f"Sanitized text length ({len(result)}) exceeds max_chars ({max_chars}).",
        )

    return result


def resolve_ffprobe_path(ffmpeg_bin: Optional[str] = None) -> Optional[str]:
    """Resolve ffprobe executable alongside ffmpeg or on PATH."""
    if ffmpeg_bin:
        sibling = Path(ffmpeg_bin).parent / "ffprobe"
        if os.path.isfile(str(sibling)) and os.access(str(sibling), os.X_OK):
            return str(sibling)
    return shutil.which("ffprobe")


async def _run_subprocess_with_timeout(
    cmd: list[str],
    timeout_sec: float,
) -> tuple[int, bytes, bytes]:
    """Run a subprocess asynchronously with guaranteed termination on timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        raise TTSError(f"Subprocess timed out after {timeout_sec:.1f}s: {cmd[0]}") from exc
    return proc.returncode or 0, stdout, stderr


async def probe_audio_duration(
    audio_path: Path | str,
    ffmpeg_path: Optional[str] = None,
    timeout_sec: float = 10.0,
) -> float:
    """Measure audio file duration in seconds using ffprobe, falling back to ffmpeg."""
    target = Path(audio_path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Audio file not found for probing: {target}")

    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)
    ffprobe_bin = resolve_ffprobe_path(ffmpeg_bin)

    if ffprobe_bin:
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(target),
        ]
        rc, stdout, stderr = await _run_subprocess_with_timeout(cmd, timeout_sec=timeout_sec)
        if rc == 0:
            raw_val = stdout.decode("utf-8", errors="replace").strip()
            try:
                duration = float(raw_val)
                if duration > 0:
                    return round(duration, 3)
            except ValueError:
                pass
        logger.debug(
            "ffprobe failed to parse duration for %s (rc=%s, stderr=%s), falling back to ffmpeg",
            target,
            rc,
            stderr.decode("utf-8", errors="replace").strip(),
        )

    cmd_ffmpeg = [
        ffmpeg_bin,
        "-hide_banner",
        "-i",
        str(target),
        "-f",
        "null",
        "-",
    ]
    _rc, _stdout, stderr = await _run_subprocess_with_timeout(cmd_ffmpeg, timeout_sec=timeout_sec)
    stderr_text = stderr.decode("utf-8", errors="replace")
    match = _DURATION_STDERR_RE.search(stderr_text)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if duration > 0:
            return round(duration, 3)

    raise TTSError(f"Failed to determine audio duration for {target}")


async def transcode_to_telegram_voice(
    input_path: Path | str,
    output_path: Path | str,
    ffmpeg_path: Optional[str] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> Path:
    """Convert audio file (e.g. MP3 from edge-tts) into Telegram OGG/Opus (48kHz mono)."""
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Input audio file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-c:a",
        "libopus",
        "-ar",
        "48000",
        "-ac",
        "1",
        "-b:a",
        "32k",
        "-vbr",
        "on",
        "-application",
        "voip",
        "-f",
        "ogg",
        str(dst),
    ]

    rc, _stdout, stderr = await _run_subprocess_with_timeout(cmd, timeout_sec=timeout_sec)
    if rc != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()
        raise AudioProcessingError(f"FFmpeg OGG/Opus transcode exited with code {rc}: {err_msg}")

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError(f"Transcoded OGG/Opus file is missing or empty: {dst}")

    return dst


class TTSSynthesizer:
    """Async TTS synthesizer producing Telegram-ready OGG/Opus voice messages."""

    def __init__(
        self,
        config: Optional[Any] = None,
        *,
        enabled: Optional[bool] = None,
        ru_voice: Optional[str] = None,
        uk_voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        max_chars: Optional[int] = None,
        timeout_sec: Optional[float] = None,
        ffmpeg_path: Optional[str] = None,
        temp_dir_base: str = "data/tmp",
    ) -> None:
        self.enabled: bool = (
            enabled if enabled is not None else bool(getattr(config, "enabled", True))
        )
        self.ru_voice: str = ru_voice or str(getattr(config, "ru_voice", DEFAULT_RU_VOICE))
        self.uk_voice: str = uk_voice or str(getattr(config, "uk_voice", DEFAULT_UK_VOICE))
        self.rate: str = rate or str(getattr(config, "rate", DEFAULT_RATE))
        self.pitch: str = pitch or str(getattr(config, "pitch", DEFAULT_PITCH))
        self.max_chars: int = (
            max_chars if max_chars is not None else int(getattr(config, "max_chars", DEFAULT_MAX_CHARS))
        )
        self.timeout_sec: float = (
            timeout_sec
            if timeout_sec is not None
            else float(getattr(config, "timeout_sec", DEFAULT_TIMEOUT_SEC))
        )
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir_base = temp_dir_base

    def _select_voice(self, language: str, voice_override: Optional[str] = None) -> str:
        if voice_override:
            return voice_override
        return self.uk_voice if language == "uk" else self.ru_voice

    async def synthesize(
        self,
        text: str,
        output_path: Optional[Path | str] = None,
        *,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize text into a 48kHz mono OGG/Opus voice note with duration."""
        if not self.enabled:
            raise TTSSkippedError("disabled", "TTS synthesis is disabled in configuration.")

        sanitized = sanitize_text_for_tts(text, max_chars=self.max_chars)
        detected_lang = language if language in {"ru", "uk"} else detect_speech_language(sanitized)
        selected_voice = self._select_voice(detected_lang, voice_override=voice)

        if edge_tts is None:
            raise TTSError("edge-tts package is not installed; cannot synthesize speech.")

        base_tmp = Path(self.temp_dir_base).resolve()
        base_tmp.mkdir(parents=True, exist_ok=True)

        owns_output_dir = output_path is None
        raw_tmp_dir = Path(tempfile.mkdtemp(prefix="bot_tts_raw_", dir=str(base_tmp)))
        out_tmp_dir: Optional[Path] = None

        try:
            if owns_output_dir:
                out_tmp_dir = Path(tempfile.mkdtemp(prefix="bot_tts_", dir=str(base_tmp)))
                target_ogg = (out_tmp_dir / "voice.ogg").resolve()
            else:
                target_ogg = Path(output_path).resolve()
                target_ogg.parent.mkdir(parents=True, exist_ok=True)

            raw_mp3 = raw_tmp_dir / "speech.mp3"

            communicate = edge_tts.Communicate(
                text=sanitized,
                voice=selected_voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            try:
                await asyncio.wait_for(
                    communicate.save(str(raw_mp3)),
                    timeout=self.timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                raise TTSError(
                    f"edge-tts synthesis timed out after {self.timeout_sec:.1f}s"
                ) from exc
            except Exception as exc:
                raise TTSError(f"edge-tts synthesis failed: {exc}") from exc

            if not raw_mp3.exists() or raw_mp3.stat().st_size == 0:
                raise TTSError("edge-tts produced an empty audio file.")

            try:
                await transcode_to_telegram_voice(
                    input_path=raw_mp3,
                    output_path=target_ogg,
                    ffmpeg_path=self.ffmpeg_path,
                    timeout_sec=self.timeout_sec,
                )
            except AudioProcessingError as exc:
                raise TTSError(f"Failed to convert TTS audio to OGG/Opus: {exc}") from exc

            duration = await probe_audio_duration(
                audio_path=target_ogg,
                ffmpeg_path=self.ffmpeg_path,
                timeout_sec=min(10.0, self.timeout_sec),
            )

            return TTSResult(
                audio_path=target_ogg,
                language=detected_lang,
                voice=selected_voice,
                duration=duration,
                sanitized_text=sanitized,
                temp_dir=out_tmp_dir,
            )
        except Exception:
            if out_tmp_dir is not None:
                shutil.rmtree(out_tmp_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(raw_tmp_dir, ignore_errors=True)

    @asynccontextmanager
    async def managed_speech(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> AsyncIterator[TTSResult]:
        """Synthesize voice into a managed temp directory and clean it up on exit."""
        base_tmp = Path(self.temp_dir_base).resolve()
        base_tmp.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="bot_tts_", dir=str(base_tmp)))
        try:
            result = await self.synthesize(
                text=text,
                output_path=temp_dir / "voice.ogg",
                language=language,
                voice=voice,
            )
            result.temp_dir = temp_dir
            yield result
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
