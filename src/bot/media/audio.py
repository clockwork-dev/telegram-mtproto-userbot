"""Audio and video note processing with FFmpeg.

Transcodes voice notes (.ogg Opus) to 16kHz mono WAV and extracts both
audio and keyframe thumbnails from video notes (.mp4).
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Fallback known static FFmpeg path
DEFAULT_STATIC_FFMPEG = "/home/jeff/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"


class AudioProcessingError(Exception):
    """Raised when audio or video transcoding fails."""


def resolve_ffmpeg_path(custom_path: Optional[str] = None) -> str:
    """Resolve FFmpeg executable path via config, system PATH, standard paths, or static fallback."""
    if custom_path:
        if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
            return str(Path(custom_path).resolve())
        logger.debug("Configured ffmpeg_path '%s' not executable; trying fallback paths.", custom_path)

    system_which = shutil.which("ffmpeg")
    if system_which:
        return system_which

    for candidate in ("/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    if os.path.isfile(DEFAULT_STATIC_FFMPEG) and os.access(DEFAULT_STATIC_FFMPEG, os.X_OK):
        return DEFAULT_STATIC_FFMPEG

    try:
        for candidate in Path.home().glob(".hermes/tools/ffmpeg*/bin/ffmpeg"):
            if os.path.isfile(str(candidate)) and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
    except OSError:
        pass

    raise FileNotFoundError(
        "FFmpeg binary not found. Please configure a valid ffmpeg_path or install ffmpeg."
    )


DEFAULT_FFMPEG_TIMEOUT_SECONDS = 60.0


async def _run_ffmpeg_command(
    cmd: list[str],
    timeout_seconds: float = DEFAULT_FFMPEG_TIMEOUT_SECONDS,
) -> None:
    """Run an FFmpeg command asynchronously with timeout and subprocess cleanup."""
    logger.debug("Executing FFmpeg command: %s", " ".join(cmd))
    proc: Optional[asyncio.subprocess.Process] = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=max(1.0, float(timeout_seconds)),
        )
    except asyncio.TimeoutError as exc:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
        raise AudioProcessingError(
            f"FFmpeg command timed out after {timeout_seconds:.1f}s"
        ) from exc
    except asyncio.CancelledError:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
        raise
    except Exception as e:
        raise AudioProcessingError(f"Failed to start FFmpeg process: {e}") from e

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace")
        raise AudioProcessingError(
            f"FFmpeg command exited with code {proc.returncode}: {err_msg.strip()}"
        )


async def transcode_voice_to_wav(
    input_path: Path | str,
    output_path: Path | str,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    """Transcode a voice message (.ogg Opus) to 16kHz mono 16-bit PCM WAV."""
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
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]

    await _run_ffmpeg_command(cmd)

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError(f"Transcoded WAV file is missing or empty: {dst}")

    return dst


async def extract_audio_from_video(
    input_path: Path | str,
    output_path: Path | str,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    """Extract audio track from a video file into 16kHz mono WAV."""
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Input video file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]

    await _run_ffmpeg_command(cmd)

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError(f"Extracted audio file is missing or empty: {dst}")

    return dst


async def extract_keyframe_from_video(
    input_path: Path | str,
    output_path: Path | str,
    timestamp_sec: float = 1.0,
    ffmpeg_path: Optional[str] = None,
) -> Path:
    """Extract a keyframe JPEG thumbnail from video (attempting at timestamp_sec, falling back to 0s)."""
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()

    if not src.exists():
        raise FileNotFoundError(f"Input video file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg_path(ffmpeg_path)

    # First attempt: seek to requested timestamp
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"00:00:{timestamp_sec:05.2f}",
        "-i",
        str(src),
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(dst),
    ]

    try:
        await _run_ffmpeg_command(cmd)
    except AudioProcessingError:
        # Fallback: seek to beginning (0s) for very short videos
        logger.debug("Keyframe extraction at %ss failed, falling back to 0s", timestamp_sec)
        cmd_fallback = [
            ffmpeg_bin,
            "-y",
            "-ss",
            "00:00:00.00",
            "-i",
            str(src),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(dst),
        ]
        await _run_ffmpeg_command(cmd_fallback)

    if not dst.exists() or dst.stat().st_size == 0:
        raise AudioProcessingError(f"Extracted keyframe thumbnail is missing or empty: {dst}")

    return dst


async def extract_video_note_assets(
    input_path: Path | str,
    audio_output_path: Path | str,
    keyframe_output_path: Path | str,
    ffmpeg_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Extract both 16kHz mono audio WAV and keyframe thumbnail JPEG from a circular video note."""
    audio_dst = await extract_audio_from_video(
        input_path=input_path,
        output_path=audio_output_path,
        ffmpeg_path=ffmpeg_path,
    )
    keyframe_dst = await extract_keyframe_from_video(
        input_path=input_path,
        output_path=keyframe_output_path,
        timestamp_sec=1.0,
        ffmpeg_path=ffmpeg_path,
    )
    return audio_dst, keyframe_dst
