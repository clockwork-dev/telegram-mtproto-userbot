"""Text-to-Speech (TTS) synthesis via Gemini TTS / edge-tts and FFmpeg (OGG/Opus).

Provides language auto-detection (Russian vs Ukrainian), markdown/URL speech
sanitization, Gemini inline emotion tag handling, eligibility guards for text
fallback, async Gemini & Edge TTS synthesis, raw PCM & encoded audio to
OGG/Opus (48 kHz mono) transcoding, audio duration probing, secret redaction,
and deterministic temporary file cleanup.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence

try:
    import edge_tts  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    edge_tts = None  # type: ignore[assignment]

try:
    from google import genai  # type: ignore[import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

from bot.media.audio import AudioProcessingError, resolve_ffmpeg_path
from bot.security.sanitizer import SecretSanitizer

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "auto"
DEFAULT_FALLBACK_TO_EDGE = True
DEFAULT_RU_VOICE = "ru-RU-DmitryNeural"
DEFAULT_UK_VOICE = "uk-UA-OstapNeural"
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_MAX_CHARS = 600
DEFAULT_TIMEOUT_SEC = 25.0

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_RU_VOICE = "Fenrir"
DEFAULT_GEMINI_UK_VOICE = "Fenrir"
RECOMMENDED_GEMINI_MALE_VOICES: dict[str, str] = {
    "Fenrir": "Excitable, energetic, dynamic male voice — strong, punchy baseline for bold masculine banter (Potuzhnych).",
    "Charon": "Informative, calm, grounded middle-low male voice — great for composed explanations and dry wit.",
    "Orus": "Firm, clear, confident male voice — great for crisp, direct technical or everyday delivery.",
    "Puck": "Upbeat, lively, middle-register male voice — great for quick-witted, lighter conversational banter.",
}
RECOMMENDED_GEMINI_FEMALE_VOICES: dict[str, str] = {
    "Aoede": "Breezy, middle-register, confident and expressive — ideal baseline for warm conversational banter.",
    "Zephyr": "Bright, higher-register, energetic and crisp — great for lively, upbeat reactions.",
    "Leda": "Youthful, higher-register, vibrant and playful — great for expressive storytelling.",
    "Kore": "Firm, middle-register, composed and distinct — great for deadpan or grounded delivery.",
}
DEFAULT_GEMINI_STYLE_PROMPT = (
    "Grounded, confident, hyper-masculine guy (Potuzhnych) speaking naturally at normal "
    "conversational distance; resonant chest tone, composed stoic authority, dry ironic punch, "
    "natural breaths and unhurried spoken timing in crisp Russian/Ukrainian—never robotic, "
    "monotone, ASMR-whispering, flirtatious, or theatrical."
)
DEFAULT_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_PCM_SAMPLE_RATE = 24000
DEFAULT_PCM_CHANNELS = 1
_MAX_MOOD_DIRECTIVE_CHARS = 200

_VALID_PROVIDERS = frozenset({"auto", "gemini", "edge"})

_MIME_TO_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "audio/mp4": ".m4a",
}

_TTS_MOOD_TAG_RE = re.compile(
    r"\[\s*(?:MOOD|STYLE|TONE)\s*:\s*([^\]]*?)\s*\]",
    re.IGNORECASE,
)

_TTS_TAG_PREFIX_MODIFIER = (
    r"(?:(?:soft|quiet|light|deep|short|long|nervous|heavy|dry|warm|playful|"
    r"sarcastic|conspiratorial|dramatic|quick|faint|sudden|little|brief)\s+)?"
)
_TTS_TAG_SUFFIX_MODIFIER = (
    r"(?:\s+(?:softly|quietly|lightly|deeply|nervously|heavily|dryly|warmly|"
    r"playfully|sarcastically|conspiratorially|dramatically|gently|loudly|"
    r"sharply|briefly|bitterly|fondly))?"
)
_TTS_EMOTION_TAGS = (
    r"laughs?|laughing|laughter|chuckles?|chuckling|giggles?|giggling|snickers?|snickering|"
    r"smirks?|smirking|sighs?|sighing|whispers?|whispering|whispered|"
    r"excitement|excited(?:ly)?|gasp(?:s|ing)?|groan(?:s|ing)?|yawn(?:s|ing)?|"
    r"cough(?:s|ing)?|clears\s+throat|sniff(?:s|ing)?|snort(?:s|ing)?|scoff(?:s|ing)?|"
    r"shout(?:s|ing)?|scream(?:s|ing)?|sob(?:s|bing)?|cry(?:ing)?|hum(?:s|ming)?|"
    r"pant(?:s|ing)?|pause|beat|hesitates?|hesitating|voice\s+breaks|"
    r"uhm|um|uh|"
    r"sarcasm|sarcastically|sarcastic|deadpan|playful(?:ly)?|warm(?:ly)?|"
    r"happy|happily|sad(?:ly)?|angry|angrily|annoyed|surprised|curious(?:ly)?|"
    r"serious(?:ly)?|nervous(?:ly)?|confident(?:ly)?|dramatic(?:ally)?|"
    r"tired|bored|cheerful(?:ly)?|amused|hesitant(?:ly)?|teasing(?:ly)?|dry(?:ly)?|"
    r"smiling|mocking(?:ly)?|fond(?:ly)?|"
    r"breath(?:es|ing)?|exhale(?:s|ing)?|inhale(?:s|ing)?|"
    r"applause|shudder"
)
_TTS_EMOTION_TAG_RE = re.compile(
    rf"\[\s*{_TTS_TAG_PREFIX_MODIFIER}(?:{_TTS_EMOTION_TAGS}){_TTS_TAG_SUFFIX_MODIFIER}\s*\]",
    re.IGNORECASE,
)

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


def _redact_secret_text(text: Any, explicit_secrets: Sequence[Optional[str]] = ()) -> str:
    """Redact API keys and credential patterns from error messages before logging or raising."""
    secrets = [s.strip() for s in explicit_secrets if isinstance(s, str) and s.strip()]
    sanitizer = SecretSanitizer(extra_secrets=secrets, load_environ=False)
    return sanitizer.sanitize(text)


def _normalize_tts_mood(raw_mood: str) -> str:
    """Normalize an extracted [MOOD: ...] directive into a safe single-line note."""
    if not raw_mood:
        return ""
    # Strip markdown heading markers to prevent header injection into Gemini's director prompt.
    cleaned = re.sub(r"#+", " ", raw_mood)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n.,;:-")
    if len(cleaned) > _MAX_MOOD_DIRECTIVE_CHARS:
        cleaned = cleaned[:_MAX_MOOD_DIRECTIVE_CHARS].rstrip(" \t.,;:-")
    return cleaned


def _strip_tag_pattern_lines(text: str, pattern: re.Pattern[str]) -> str:
    """Strip a bracketed tag regex (including multi-line directives) cleanly."""
    if not text:
        return ""
    if not pattern.search(text):
        return text

    sentinel = "\x00TTS_TAG_REMOVED\x00"
    marked = pattern.sub(sentinel, text)
    out_lines: list[str] = []
    for raw_line in marked.splitlines():
        if sentinel in raw_line:
            cleaned = raw_line.replace(sentinel, " ")
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = re.sub(r"[ \t]+([,.!?;:…])", r"\1", cleaned).strip()
            if not cleaned:
                continue
            out_lines.append(cleaned)
        else:
            out_lines.append(raw_line.rstrip())

    return "\n".join(out_lines).strip()


def extract_tts_mood_and_text(text: str) -> tuple[Optional[str], str]:
    """Extract [MOOD: ...]/[STYLE: ...]/[TONE: ...] directives and return (mood, cleaned_text).

    Inline vocal cues (e.g. ``[laughs]``, ``[sigh]``, ``[chuckles]``, ``[pause]``)
    remain in ``cleaned_text`` for Gemini TTS while director mood directives are
    extracted and stripped cleanly.
    """
    if not text:
        return None, ""
    if not _TTS_MOOD_TAG_RE.search(text):
        return None, text

    mood_parts: list[str] = []
    seen_lower: set[str] = set()
    for match in _TTS_MOOD_TAG_RE.finditer(text):
        normalized = _normalize_tts_mood(match.group(1))
        if normalized:
            key = normalized.lower()
            if key not in seen_lower:
                seen_lower.add(key)
                mood_parts.append(normalized)

    merged_mood = _normalize_tts_mood("; ".join(mood_parts)) if mood_parts else ""
    cleaned_text = _strip_tag_pattern_lines(text, _TTS_MOOD_TAG_RE)
    return (merged_mood or None), cleaned_text


def strip_tts_emotion_tags(text: str) -> str:
    """Remove [MOOD:...]/[STYLE:...]/[TONE:...] and inline emotion tags (e.g. [laughs])."""
    if not text:
        return ""
    _, without_mood = extract_tts_mood_and_text(text)
    return _strip_tag_pattern_lines(without_mood, _TTS_EMOTION_TAG_RE)


def detect_speech_language(text: str) -> str:
    """Detect whether text should be spoken in Ukrainian ('uk') or Russian ('ru')."""
    probe = strip_tts_emotion_tags(text)
    if not probe:
        return "ru"

    uk_chars = sum(1 for ch in probe if ch in _UK_SPECIFIC_CHARS)
    ru_chars = sum(1 for ch in probe if ch in _RU_SPECIFIC_CHARS)

    if uk_chars > 0 and ru_chars == 0:
        return "uk"
    if uk_chars > 0 and ru_chars > 0:
        return "uk" if uk_chars >= ru_chars else "ru"
    if ru_chars > 0:
        return "ru"

    if _UK_APOSTROPHE_RE.search(probe):
        return "uk"

    words = [w.lower() for w in _WORD_RE.findall(probe)]
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


def sanitize_text_for_tts(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    preserve_emotion_tags: bool = False,
) -> str:
    """Sanitize response text for natural speech, rejecting un-speakable content.

    Director directives such as ``[MOOD: ...]``/``[STYLE: ...]``/``[TONE: ...]``
    are always stripped before length and speakability checks. When
    ``preserve_emotion_tags`` is False (default, e.g. for edge-tts), inline
    Gemini TTS emotion tags such as ``[laughs]`` or ``[sigh]`` are also stripped.

    Raises:
        TTSSkippedError: If text is empty, contains code blocks or tables,
            has no speakable characters left after cleanup, or exceeds max_chars.
    """
    if not text or not text.strip():
        raise TTSSkippedError("empty", "Cannot synthesize empty text.")

    _, cleaned = extract_tts_mood_and_text(text)
    if not cleaned or not cleaned.strip():
        raise TTSSkippedError("empty", "Text has no speakable content after mood extraction.")

    if _FENCED_CODE_RE.search(cleaned):
        raise TTSSkippedError("code_block", "Text contains a fenced code block; falling back to text.")

    if _contains_markdown_table(cleaned):
        raise TTSSkippedError("table", "Text contains a Markdown table; falling back to text.")

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

    if not preserve_emotion_tags:
        cleaned = strip_tts_emotion_tags(cleaned)

    # Collapse empty parentheses left after stripping bare URLs like "site (https://...)"
    cleaned = re.sub(r"\(\s*\)", "", cleaned)

    # Join lines into natural spoken sentences without punctuating standalone tag lines.
    paragraphs: list[str] = []
    for raw_line in cleaned.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if paragraphs:
            prev = paragraphs[-1]
            prev_spoken = strip_tts_emotion_tags(prev).strip() if preserve_emotion_tags else prev
            if prev_spoken and prev_spoken[-1] not in ".!?…:;":
                if prev[-1] not in ".!?…:;":
                    paragraphs[-1] = f"{prev}."
        paragraphs.append(line)

    result = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
    speakable_check = strip_tts_emotion_tags(result) if preserve_emotion_tags else result
    if not speakable_check or not any(ch.isalnum() for ch in speakable_check):
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


def _parse_audio_mime_type(mime_type: Optional[str]) -> tuple[bool, str, int, int]:
    """Parse Gemini audio mime_type into ``(is_raw_pcm, suffix_or_fmt, sample_rate, channels)``."""
    raw = (mime_type or "").strip()
    if not raw:
        return True, "s16le", DEFAULT_PCM_SAMPLE_RATE, DEFAULT_PCM_CHANNELS

    segments = [seg.strip() for seg in raw.split(";") if seg.strip()]
    base_mime = segments[0].lower()
    params: dict[str, str] = {}
    for seg in segments[1:]:
        if "=" in seg:
            k, v = seg.split("=", 1)
            params[k.strip().lower()] = v.strip().strip('"').strip("'")

    if base_mime in {"audio/l16", "audio/pcm"} or base_mime.startswith("audio/l16"):
        sample_rate = DEFAULT_PCM_SAMPLE_RATE
        channels = DEFAULT_PCM_CHANNELS
        if "rate" in params:
            try:
                parsed_rate = int(params["rate"])
                if parsed_rate > 0:
                    sample_rate = parsed_rate
            except ValueError:
                pass
        for ch_key in ("channels", "ch"):
            if ch_key in params:
                try:
                    parsed_ch = int(params[ch_key])
                    if parsed_ch > 0:
                        channels = parsed_ch
                except ValueError:
                    pass
                break
        return True, "s16le", sample_rate, channels

    suffix = _MIME_TO_SUFFIX.get(base_mime)
    if suffix is None:
        if "wav" in base_mime:
            suffix = ".wav"
        elif "mp3" in base_mime or "mpeg" in base_mime:
            suffix = ".mp3"
        elif "ogg" in base_mime or "opus" in base_mime:
            suffix = ".ogg"
        else:
            suffix = ".audio"
    return False, suffix, 0, 0


async def transcode_to_telegram_voice(
    input_path: Path | str,
    output_path: Path | str,
    ffmpeg_path: Optional[str] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    *,
    input_format: Optional[str] = None,
    input_sample_rate: Optional[int] = None,
    input_channels: Optional[int] = None,
) -> Path:
    """Convert audio file (raw PCM, WAV, MP3, etc.) into Telegram OGG/Opus (48kHz mono)."""
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Input audio file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)

    cmd = [ffmpeg_bin, "-y"]
    if input_format:
        cmd.extend(["-f", input_format])
        if input_sample_rate and input_sample_rate > 0:
            cmd.extend(["-ar", str(input_sample_rate)])
        if input_channels and input_channels > 0:
            cmd.extend(["-ac", str(input_channels)])
    cmd.extend([
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
    ])

    rc, _stdout, stderr = await _run_subprocess_with_timeout(cmd, timeout_sec=timeout_sec)
    if rc != 0:
        err_msg = stderr.decode("utf-8", errors="replace").strip()
        raise AudioProcessingError(f"FFmpeg OGG/Opus transcode exited with code {rc}: {err_msg}")

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError(f"Transcoded OGG/Opus file is missing or empty: {dst}")

    return dst


def _format_enum_name(value: Any) -> str:
    """Return clean enum/string name for Gemini feedback fields."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(value).strip()
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def _extract_gemini_audio_payload(
    response: Any,
    explicit_secrets: Sequence[Optional[str]] = (),
) -> tuple[bytes, Optional[str]]:
    """Extract raw audio bytes and mime_type from a Gemini GenerateContentResponse."""
    candidates = getattr(response, "candidates", None) or ()
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or ()
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if not data:
                continue
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data, validate=True)
                except Exception:
                    raise TTSError("Gemini TTS returned invalid base64 audio payload.") from None
            elif isinstance(data, bytearray):
                data = bytes(data)
            if isinstance(data, bytes) and data:
                mime_type = getattr(inline_data, "mime_type", None)
                return data, str(mime_type) if mime_type else None

    details: list[str] = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        block_reason = _format_enum_name(getattr(prompt_feedback, "block_reason", None))
        if block_reason and block_reason not in {"BLOCK_REASON_UNSPECIFIED", "0"}:
            details.append(f"block_reason={block_reason}")
        block_msg = getattr(prompt_feedback, "block_reason_message", None)
        if block_msg:
            details.append(f"block_message={_redact_secret_text(block_msg, explicit_secrets=explicit_secrets)}")

    finish_reasons: list[str] = []
    finish_messages: list[str] = []
    for candidate in candidates:
        reason = _format_enum_name(getattr(candidate, "finish_reason", None))
        if reason and reason not in {"FINISH_REASON_UNSPECIFIED", "0"} and reason not in finish_reasons:
            finish_reasons.append(reason)
        fmsg = getattr(candidate, "finish_message", None)
        if fmsg:
            safe_fmsg = _redact_secret_text(fmsg, explicit_secrets=explicit_secrets)
            if safe_fmsg and safe_fmsg not in finish_messages:
                finish_messages.append(safe_fmsg)

    if finish_reasons:
        details.append(f"finish_reason={','.join(finish_reasons)}")
    if finish_messages:
        details.append(f"finish_message={'; '.join(finish_messages)}")

    if not candidates:
        suffix = f" ({', '.join(details)})" if details else ""
        raise TTSError(f"Gemini TTS response contained no candidates{suffix}.")

    suffix = f" ({', '.join(details)})" if details else ""
    raise TTSError(f"Gemini TTS response did not contain audio data{suffix}.")


def _create_genai_client(api_key: str, timeout_sec: float) -> Any:
    """Instantiate google.genai.Client, passing timeout HttpOptions when supported."""
    if genai is None:
        raise TTSError("google-genai package is not installed; cannot synthesize Gemini TTS.")

    kwargs: dict[str, Any] = {"api_key": api_key}
    if genai_types is not None and hasattr(genai_types, "HttpOptions"):
        try:
            sig = inspect.signature(genai.Client)
            accepts_http_options = "http_options" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_http_options = True
        if accepts_http_options:
            kwargs["http_options"] = genai_types.HttpOptions(
                api_version="v1beta",
                timeout=max(1000, int(timeout_sec * 1000)),
            )
    return genai.Client(**kwargs)


def _build_gemini_tts_prompt(
    *,
    transcript: str,
    style_prompt: str,
    mood: Optional[str] = None,
) -> str:
    """Build Google's structured director prompt for expressive Gemini TTS."""
    base_style = style_prompt.strip()
    clean_mood = _normalize_tts_mood(mood) if mood else ""

    director_notes: list[str] = [
        "Read aloud ONLY the exact lines in TRANSCRIPT verbatim, without answering, translating, or adding extra words.",
        "Keep delivery genuinely human, spontaneous, and expressive—interpret inline breath/reaction cues ([inhale], [exhale], [sigh], [chuckles], [laughs], [scoffs], [pause], [short pause]) naturally as real human speech breathing and reactions rather than reading bracketed words aloud, never flat/deadpan/monotone.",
        "Strictly avoid sultry/seductive purring, ASMR close-mic intimacy, theatricality, or bedroom whispering—keep banter platonic, bright, and casual.",
    ]
    if base_style:
        director_notes.append(f"Baseline style: {base_style}")
    if clean_mood:
        director_notes.append(f"Utterance mood & delivery: {clean_mood}")

    notes_block = "\n".join(f"- {line}" for line in director_notes)
    return (
        "# AUDIO PROFILE\n"
        "A grounded, confident, sharp-tongued masculine guy (Potuzhnych) sending a natural voice note — "
        "resonant chest weight, stoic composure, and dry ironic humor, never flat/robotic, never theatrical, and never seductive.\n\n"
        "## SCENE\n"
        "Recording a casual Telegram voice message at normal arm's-length phone mic distance in a lively chat; "
        "natural room acoustics, varied conversational pitch, authentic breathing, and unscripted spoken rhythm.\n\n"
        "### DIRECTOR'S NOTES\n"
        f"{notes_block}\n\n"
        "#### TRANSCRIPT\n"
        f"{transcript}"
    )


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
        provider: Optional[str] = None,
        fallback_to_edge: Optional[bool] = None,
        gemini_model: Optional[str] = None,
        gemini_ru_voice: Optional[str] = None,
        gemini_uk_voice: Optional[str] = None,
        gemini_style_prompt: Optional[str] = None,
        gemini_api_key_env: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
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

        raw_provider = provider if provider is not None else getattr(config, "provider", DEFAULT_PROVIDER)
        self.provider: str = self._normalize_provider(raw_provider)
        self.fallback_to_edge: bool = bool(
            fallback_to_edge
            if fallback_to_edge is not None
            else getattr(config, "fallback_to_edge", DEFAULT_FALLBACK_TO_EDGE)
        )
        self.gemini_model: str = str(
            (gemini_model if gemini_model is not None else getattr(config, "gemini_model", None))
            or DEFAULT_GEMINI_MODEL
        )
        self.gemini_ru_voice: str = str(
            (gemini_ru_voice if gemini_ru_voice is not None else getattr(config, "gemini_ru_voice", None))
            or DEFAULT_GEMINI_RU_VOICE
        )
        self.gemini_uk_voice: str = str(
            (gemini_uk_voice if gemini_uk_voice is not None else getattr(config, "gemini_uk_voice", None))
            or DEFAULT_GEMINI_UK_VOICE
        )
        raw_style = (
            gemini_style_prompt
            if gemini_style_prompt is not None
            else getattr(config, "gemini_style_prompt", DEFAULT_GEMINI_STYLE_PROMPT)
        )
        self.gemini_style_prompt: str = str(DEFAULT_GEMINI_STYLE_PROMPT if raw_style is None else raw_style)
        self.gemini_api_key_env: str = str(
            (
                gemini_api_key_env
                if gemini_api_key_env is not None
                else getattr(config, "gemini_api_key_env", None)
            )
            or DEFAULT_GEMINI_API_KEY_ENV
        )
        raw_api_key = gemini_api_key if gemini_api_key is not None else getattr(config, "gemini_api_key", None)
        self.gemini_api_key: Optional[str] = str(raw_api_key) if raw_api_key is not None else None

    @staticmethod
    def _normalize_provider(value: Any) -> str:
        normalized = str(value or DEFAULT_PROVIDER).strip().lower()
        if normalized not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unsupported TTS provider {value!r}; expected one of {sorted(_VALID_PROVIDERS)}."
            )
        return normalized

    def _resolve_gemini_api_key(self) -> Optional[str]:
        candidates: list[Optional[str]] = [self.gemini_api_key]
        if self.gemini_api_key_env:
            candidates.append(os.environ.get(self.gemini_api_key_env))
        candidates.append(os.environ.get("GOOGLE_API_KEY"))
        for candidate in candidates:
            if isinstance(candidate, str):
                cleaned = candidate.strip()
                if cleaned:
                    return cleaned
        return None

    def _select_voice(self, language: str, voice_override: Optional[str] = None) -> str:
        if voice_override:
            return voice_override
        return self.uk_voice if language == "uk" else self.ru_voice

    def _select_gemini_voice(self, language: str, voice_override: Optional[str] = None) -> str:
        if voice_override:
            return voice_override
        return self.gemini_uk_voice if language == "uk" else self.gemini_ru_voice

    async def _synthesize_edge(
        self,
        *,
        sanitized_text: str,
        voice: str,
        raw_tmp_dir: Path,
        target_ogg: Path,
    ) -> None:
        if edge_tts is None:
            raise TTSError("edge-tts package is not installed; cannot synthesize speech.")

        raw_mp3 = raw_tmp_dir / "speech.mp3"
        communicate = edge_tts.Communicate(
            text=sanitized_text,
            voice=voice,
            rate=self.rate,
            pitch=self.pitch,
        )
        try:
            await asyncio.wait_for(
                communicate.save(str(raw_mp3)),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            raise TTSError(
                f"edge-tts synthesis timed out after {self.timeout_sec:.1f}s"
            ) from None
        except Exception as exc:
            raise TTSError(f"edge-tts synthesis failed: {_redact_secret_text(exc)}") from None

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
            raise TTSError(
                f"Failed to convert TTS audio to OGG/Opus: {_redact_secret_text(exc)}"
            ) from None

    async def _synthesize_gemini(
        self,
        *,
        sanitized_text: str,
        voice: str,
        api_key: str,
        raw_tmp_dir: Path,
        target_ogg: Path,
        mood: Optional[str] = None,
    ) -> None:
        if genai is None or genai_types is None:
            raise TTSError("google-genai package is not installed; cannot synthesize Gemini TTS.")

        prompt_text = _build_gemini_tts_prompt(
            transcript=sanitized_text,
            style_prompt=self.gemini_style_prompt,
            mood=mood,
        )
        gen_config = genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                )
            ),
        )

        client = _create_genai_client(api_key=api_key, timeout_sec=self.timeout_sec)
        try:
            aio_client = getattr(client, "aio", None)
            if aio_client is not None:
                if hasattr(aio_client, "__aenter__") and hasattr(aio_client, "__aexit__"):
                    async with aio_client as aclient:
                        response = await asyncio.wait_for(
                            aclient.models.generate_content(
                                model=self.gemini_model,
                                contents=prompt_text,
                                config=gen_config,
                            ),
                            timeout=self.timeout_sec,
                        )
                else:
                    try:
                        response = await asyncio.wait_for(
                            aio_client.models.generate_content(
                                model=self.gemini_model,
                                contents=prompt_text,
                                config=gen_config,
                            ),
                            timeout=self.timeout_sec,
                        )
                    finally:
                        aclose = getattr(aio_client, "aclose", None)
                        if callable(aclose):
                            try:
                                await aclose()
                            except Exception:
                                pass
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=self.gemini_model,
                        contents=prompt_text,
                        config=gen_config,
                    ),
                    timeout=self.timeout_sec,
                )
        except asyncio.TimeoutError:
            raise TTSError(
                f"Gemini TTS synthesis timed out after {self.timeout_sec:.1f}s"
            ) from None
        except TTSError:
            raise
        except Exception as exc:
            safe_msg = _redact_secret_text(exc, explicit_secrets=[api_key])
            raise TTSError(f"Gemini TTS synthesis failed: {safe_msg}") from None
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        audio_bytes, mime_type = _extract_gemini_audio_payload(
            response,
            explicit_secrets=[api_key],
        )
        is_pcm, fmt_or_suffix, sample_rate, channels = _parse_audio_mime_type(mime_type)

        if is_pcm:
            raw_audio = raw_tmp_dir / "gemini_speech.pcm"
            raw_audio.write_bytes(audio_bytes)
            transcode_kwargs: dict[str, Any] = {
                "input_format": fmt_or_suffix,
                "input_sample_rate": sample_rate,
                "input_channels": channels,
            }
        else:
            raw_audio = raw_tmp_dir / f"gemini_speech{fmt_or_suffix}"
            raw_audio.write_bytes(audio_bytes)
            transcode_kwargs = {}

        try:
            await transcode_to_telegram_voice(
                input_path=raw_audio,
                output_path=target_ogg,
                ffmpeg_path=self.ffmpeg_path,
                timeout_sec=self.timeout_sec,
                **transcode_kwargs,
            )
        except AudioProcessingError as exc:
            safe_msg = _redact_secret_text(exc, explicit_secrets=[api_key])
            raise TTSError(f"Failed to convert Gemini TTS audio to OGG/Opus: {safe_msg}") from None

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

        provider = self._normalize_provider(self.provider)
        resolved_key: Optional[str] = None
        use_gemini = False
        if provider == "gemini":
            resolved_key = self._resolve_gemini_api_key()
            if not resolved_key:
                raise TTSError(
                    f"Gemini TTS provider is selected, but no API key was found "
                    f"(checked gemini_api_key, {self.gemini_api_key_env}, GOOGLE_API_KEY)."
                )
            use_gemini = True
        elif provider == "auto":
            resolved_key = self._resolve_gemini_api_key()
            use_gemini = resolved_key is not None

        extracted_mood, body_text = extract_tts_mood_and_text(text)

        if use_gemini:
            sanitized = sanitize_text_for_tts(
                body_text,
                max_chars=self.max_chars,
                preserve_emotion_tags=True,
            )
            lang_probe_text = strip_tts_emotion_tags(sanitized)
        else:
            sanitized = sanitize_text_for_tts(
                body_text,
                max_chars=self.max_chars,
                preserve_emotion_tags=False,
            )
            lang_probe_text = sanitized

        detected_lang = language if language in {"ru", "uk"} else detect_speech_language(lang_probe_text)

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

            active_voice: str
            active_sanitized: str
            active_provider: str

            if use_gemini and resolved_key is not None:
                gemini_voice = self._select_gemini_voice(detected_lang, voice_override=voice)
                try:
                    await self._synthesize_gemini(
                        sanitized_text=sanitized,
                        voice=gemini_voice,
                        api_key=resolved_key,
                        raw_tmp_dir=raw_tmp_dir,
                        target_ogg=target_ogg,
                        mood=extracted_mood,
                    )
                    active_voice = gemini_voice
                    active_sanitized = sanitized
                    active_provider = "gemini"
                except TTSSkippedError:
                    raise
                except Exception as exc:
                    safe_err = _redact_secret_text(exc, explicit_secrets=[resolved_key])
                    if not self.fallback_to_edge:
                        if isinstance(exc, TTSError) and safe_err == str(exc) and exc.__cause__ is None:
                            raise
                        raise TTSError(safe_err) from None

                    logger.warning(
                        "Gemini TTS failed (%s); falling back to edge-tts.",
                        safe_err,
                    )
                    edge_sanitized = sanitize_text_for_tts(
                        body_text,
                        max_chars=self.max_chars,
                        preserve_emotion_tags=False,
                    )
                    edge_voice = self._select_voice(detected_lang, voice_override=voice)
                    await self._synthesize_edge(
                        sanitized_text=edge_sanitized,
                        voice=edge_voice,
                        raw_tmp_dir=raw_tmp_dir,
                        target_ogg=target_ogg,
                    )
                    active_voice = edge_voice
                    active_sanitized = edge_sanitized
                    active_provider = "edge"
            else:
                edge_voice = self._select_voice(detected_lang, voice_override=voice)
                await self._synthesize_edge(
                    sanitized_text=sanitized,
                    voice=edge_voice,
                    raw_tmp_dir=raw_tmp_dir,
                    target_ogg=target_ogg,
                )
                active_voice = edge_voice
                active_sanitized = sanitized
                active_provider = "edge"

            duration = await probe_audio_duration(
                audio_path=target_ogg,
                ffmpeg_path=self.ffmpeg_path,
                timeout_sec=min(10.0, self.timeout_sec),
            )

            safe_mood = (
                _redact_secret_text(
                    extracted_mood,
                    explicit_secrets=[resolved_key] if resolved_key else (),
                )
                if extracted_mood
                else "default"
            )
            logger.info(
                "TTS synthesized (provider=%s, voice=%s, lang=%s, duration=%.2fs, mood=%s)",
                active_provider,
                active_voice,
                detected_lang,
                duration,
                safe_mood,
            )

            return TTSResult(
                audio_path=target_ogg,
                language=detected_lang,
                voice=active_voice,
                duration=duration,
                sanitized_text=active_sanitized,
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
