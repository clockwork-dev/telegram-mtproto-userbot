"""Multimodal Media Processor for Telegram Userbot.

Orchestrates downloading and conversion of photos, WebP stickers, documents,
voice notes, GIF animations, videos, and video notes with guaranteed temp file scrubber.
"""

import asyncio
import inspect
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from PIL import Image

from bot.media.audio import (
    AudioProcessingError,
    extract_audio_from_video,
    resolve_ffmpeg_path,
    transcode_voice_to_wav,
)
from bot.media.documents import (
    DocumentProcessingError,
    FileTooLargeError,
    extract_document_text,
)

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}
MAX_VIDEO_SIZE_MB = 15


def _has_media_flag(obj: Any, attr: str) -> bool:
    """Return True if attribute `attr` on `obj` is explicitly truthy (ignoring unset MagicMock children)."""
    if type(obj).__module__.startswith("unittest.mock"):
        children = getattr(obj, "_mock_children", {})
        if attr not in getattr(obj, "__dict__", {}) and attr not in children:
            return False
    return bool(getattr(obj, attr, None))


def _get_str_attr(obj: Any, attr: str) -> Optional[str]:
    """Return string attribute if it is a genuine non-empty str."""
    if obj is None:
        return None
    val = getattr(obj, attr, None)
    return val if isinstance(val, str) and val.strip() else None


def _get_int_attr(obj: Any, attr: str) -> int:
    """Return integer attribute if it is a genuine int/float (not bool or MagicMock)."""
    if obj is None:
        return 0
    val = getattr(obj, attr, 0)
    if isinstance(val, bool):
        return 0
    return int(val) if isinstance(val, (int, float)) else 0


@dataclass
class ProcessedMedia:
    """Structured representation of ingested multimodal content."""

    media_type: str  # "photo", "sticker", "document", "voice", "video_note", "video", "animation"
    mime_type: Optional[str] = None
    file_path: Optional[Path] = None
    image_path: Optional[Path] = None
    text_content: Optional[str] = None
    audio_path: Optional[Path] = None
    keyframe_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_path is not None and self.file_path is None:
            self.file_path = self.image_path
        elif (
            self.file_path is not None
            and self.image_path is None
            and self.mime_type
            and self.mime_type.startswith("image/")
        ):
            self.image_path = self.file_path


class MediaProcessor:
    """Orchestrator for Telegram multimodal media processing and cleanup."""

    def __init__(
        self,
        max_doc_size_mb: int = 10,
        ffmpeg_path: Optional[str] = None,
        temp_dir_base: Optional[str] = None,
        stt_transcriber: Optional[Any] = None,
    ) -> None:
        self.max_doc_size_mb = max_doc_size_mb
        self.ffmpeg_path = ffmpeg_path
        self.temp_dir_base = temp_dir_base
        self.stt_transcriber = stt_transcriber
        if temp_dir_base:
            Path(temp_dir_base).mkdir(parents=True, exist_ok=True)

    def _create_temp_dir(self) -> Path:
        """Create a dedicated scratch directory with 0700 permissions."""
        d = tempfile.mkdtemp(prefix="bot_media_", dir=self.temp_dir_base)
        return Path(d).resolve()

    def cleanup(self, processed: Optional[ProcessedMedia]) -> None:
        """Clean up all temporary files associated with a processed media object."""
        if not processed or not processed.metadata:
            return

        temp_dir = processed.metadata.get("_temp_dir")
        if temp_dir:
            path = Path(temp_dir)
            if path.exists():
                logger.debug("Cleaning up scratch media directory: %s", path)
                shutil.rmtree(path, ignore_errors=True)

    @asynccontextmanager
    async def managed_media(
        self,
        message: Any,
        client: Any = None,
    ) -> AsyncIterator[Optional[ProcessedMedia]]:
        """Context manager guaranteeing deterministic cleanup of temp assets in try...finally."""
        processed: Optional[ProcessedMedia] = None
        try:
            processed = await self.process_media(message, client)
            yield processed
        finally:
            if processed:
                self.cleanup(processed)

    async def _download_media_to(
        self,
        client: Any,
        message: Any,
        target_path: Path,
    ) -> Path:
        """Download media from Telethon message to a target path."""
        # 1. If message already points to a local file, copy directly
        existing_file = getattr(message, "file_path", None) or getattr(message, "local_path", None)
        if existing_file and Path(existing_file).exists():
            shutil.copy2(existing_file, target_path)
            return target_path

        # 2. Try client.download_media or client.client.download_media
        downloader = client
        if downloader and not callable(getattr(downloader, "download_media", None)):
            downloader = getattr(downloader, "client", None)

        if downloader and callable(getattr(downloader, "download_media", None)):
            res = downloader.download_media(message, file=str(target_path))
            if asyncio.iscoroutine(res):
                res = await res
            if res:
                return Path(res).resolve()

        # 3. Try message.download_media
        if callable(getattr(message, "download_media", None)):
            res = message.download_media(file=str(target_path))
            if asyncio.iscoroutine(res):
                res = await res
            if res:
                return Path(res).resolve()

        # 4. Try raw_event or message wrapper download_media
        for alt_obj in (getattr(message, "raw_event", None), getattr(message, "message", None)):
            if alt_obj is not None and not isinstance(alt_obj, str) and callable(getattr(alt_obj, "download_media", None)):
                res = alt_obj.download_media(file=str(target_path))
                if asyncio.iscoroutine(res):
                    res = await res
                if res:
                    return Path(res).resolve()

        raise RuntimeError("Neither client nor message provided a valid download mechanism.")

    async def _run_ffmpeg_frame_extract(self, cmd: list[str]) -> bool:
        """Run an FFmpeg command asynchronously and return True if it exited with 0."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as exc:
            logger.debug("FFmpeg frame extraction command failed: %s", exc)
            return False

    async def _extract_video_or_gif_frame(
        self,
        input_path: Path,
        output_jpg: Path,
    ) -> Path:
        """Extract a representative JPEG frame from a static/animated GIF or video file."""
        src = input_path.resolve()
        dst = output_jpg.resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        header = b""
        if src.exists():
            with open(src, "rb") as bf:
                header = bf.read(16)

        is_gif_file = src.suffix.lower() == ".gif" or header.startswith((b"GIF87a", b"GIF89a"))
        if is_gif_file:
            try:
                with Image.open(src) as img:
                    img.seek(0)
                    img.convert("RGB").save(dst, format="JPEG", quality=90)
                if dst.exists() and dst.stat().st_size > 0:
                    return dst
            except Exception as exc:
                logger.debug("Pillow GIF frame extraction failed, falling back to ffmpeg: %s", exc)

        ffmpeg_bin = resolve_ffmpeg_path(self.ffmpeg_path)
        cmd_primary = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(src),
            "-vf",
            "thumbnail,scale='min(1280,iw)':-1",
            "-frames:v",
            "1",
            str(dst),
        ]
        ok = await self._run_ffmpeg_frame_extract(cmd_primary)
        if not ok or not dst.exists() or dst.stat().st_size == 0:
            cmd_fallback = [
                ffmpeg_bin,
                "-y",
                "-ss",
                "00:00:00.1",
                "-i",
                str(src),
                "-frames:v",
                "1",
                str(dst),
            ]
            ok = await self._run_ffmpeg_frame_extract(cmd_fallback)

        if not ok or not dst.exists() or dst.stat().st_size == 0:
            cmd_zero = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src),
                "-frames:v",
                "1",
                str(dst),
            ]
            await self._run_ffmpeg_frame_extract(cmd_zero)

        if not dst.exists() or dst.stat().st_size == 0:
            raise AudioProcessingError(f"Failed to extract JPEG keyframe from {src.name}")

        return dst

    def _detect_media_kind(self, message: Any) -> Optional[str]:
        """Detect the media category from a Telethon or BotAPIMessageEvent message."""
        if message is None or isinstance(message, str):
            return None

        # Unwrap IncomingMessage if passed directly
        if type(message).__name__ == "IncomingMessage" and getattr(message, "raw_event", None) is not None:
            message = message.raw_event

        has_any_flag = any(
            _has_media_flag(message, attr)
            for attr in (
                "media",
                "photo",
                "voice",
                "video_note",
                "video",
                "gif",
                "animation",
                "sticker",
                "document",
            )
        )
        if not has_any_flag and not getattr(message, "media", None):
            return None

        if _has_media_flag(message, "voice"):
            return "voice"
        if _has_media_flag(message, "sticker"):
            return "sticker"
        if _has_media_flag(message, "photo"):
            return "photo"

        file_info = getattr(message, "file", None)
        raw_mime = (
            _get_str_attr(message, "mime_type")
            or _get_str_attr(file_info, "mime_type")
            or ""
        ).lower()
        raw_name = (
            _get_str_attr(message, "file_name")
            or _get_str_attr(file_info, "name")
            or ""
        )
        ext = Path(raw_name).suffix.lower() if raw_name else ""

        if _has_media_flag(message, "gif") or _has_media_flag(message, "animation"):
            return "animation"
        if _has_media_flag(message, "video_note"):
            return "video_note"
        if _has_media_flag(message, "video"):
            return "video"

        if raw_mime == "image/gif" or ext == ".gif":
            return "animation"
        if raw_mime.startswith("video/") or ext in {".mp4", ".webm", ".mov", ".mkv"}:
            return "video"

        if _has_media_flag(message, "document"):
            return "document"

        # Check media class name or attributes if standard properties are absent
        media = getattr(message, "media", None) or message
        media_type_name = type(media).__name__ if media else ""
        if "Photo" in media_type_name:
            return "photo"
        if "Document" in media_type_name:
            doc = getattr(media, "document", None)
            if doc:
                doc_mime = (_get_str_attr(doc, "mime_type") or "").lower()
                if doc_mime == "image/gif":
                    return "animation"
                attributes = getattr(doc, "attributes", [])
                for attr in attributes:
                    attr_name = type(attr).__name__
                    if "Sticker" in attr_name:
                        return "sticker"
                    if "Audio" in attr_name and getattr(attr, "voice", False):
                        return "voice"
                    if "Animated" in attr_name:
                        return "animation"
                    if "Video" in attr_name:
                        if getattr(attr, "round_message", False):
                            return "video_note"
                        return "video"
                if doc_mime.startswith("video/"):
                    return "video"
            return "document"

        return None

    async def _transcribe_audio(
        self,
        wav_path: Path,
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """Optionally transcribe WAV audio via stt_transcriber and populate metadata."""
        transcriber = self.stt_transcriber
        if transcriber is None:
            return None
        if hasattr(transcriber, "enabled") and not transcriber.enabled:
            return None

        transcribe_fn = getattr(transcriber, "transcribe", transcriber)
        if not callable(transcribe_fn):
            return None

        try:
            result = transcribe_fn(wav_path)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            logger.warning("Audio transcription failed for %s: %s", wav_path.name, exc)
            metadata["stt_error"] = str(exc)
            return None

        if result is None:
            return None

        if isinstance(result, str):
            transcript = result.strip()
            language = None
            duration = None
            lang_prob = None
        elif isinstance(result, dict):
            raw_text = result.get("text") or result.get("transcript") or ""
            transcript = str(raw_text).strip()
            language = result.get("language")
            duration = result.get("duration")
            lang_prob = result.get("language_probability")
        else:
            raw_text = getattr(result, "text", "") or ""
            transcript = str(raw_text).strip()
            language = getattr(result, "language", None)
            duration = getattr(result, "duration", None)
            lang_prob = getattr(result, "language_probability", None)

        if not transcript:
            if isinstance(language, str) and language.strip():
                metadata["language"] = language.strip()
            return None

        metadata["transcript"] = transcript
        if isinstance(language, str) and language.strip():
            metadata["language"] = language.strip()
        if isinstance(duration, (int, float)):
            metadata["duration"] = float(duration)
        if isinstance(lang_prob, (int, float)):
            metadata["language_probability"] = float(lang_prob)
        return transcript

    async def process_message(
        self,
        message: Any,
        client: Any = None,
    ) -> Optional[ProcessedMedia]:
        """Alias for process_media supporting GIF, animation, video, video_note, photo, sticker, voice, and documents."""
        return await self.process_media(message, client=client)

    async def process_media(
        self,
        message: Any,
        client: Any = None,
    ) -> Optional[ProcessedMedia]:
        """Download, transcode, and extract content with deterministic error cleanup."""
        kind = self._detect_media_kind(message)
        if not kind:
            return None

        temp_dir = self._create_temp_dir()

        try:
            # 1. Voice Note Processing
            if kind == "voice":
                raw_ogg = temp_dir / "voice.ogg"
                await self._download_media_to(client, message, raw_ogg)
                wav_path = temp_dir / "voice.wav"
                await transcode_voice_to_wav(
                    input_path=raw_ogg,
                    output_path=wav_path,
                    ffmpeg_path=self.ffmpeg_path,
                )
                voice_metadata: Dict[str, Any] = {
                    "_temp_dir": str(temp_dir),
                    "raw_path": str(raw_ogg),
                }
                transcript = await self._transcribe_audio(wav_path, voice_metadata)
                return ProcessedMedia(
                    media_type="voice",
                    mime_type="audio/wav",
                    audio_path=wav_path,
                    text_content=transcript,
                    metadata=voice_metadata,
                )

            # 2. GIF / Animation / Video / Video Note Processing (up to 15 MB)
            if kind in ("animation", "video", "video_note"):
                file_info = getattr(message, "file", None)
                raw_name = (
                    _get_str_attr(message, "file_name")
                    or _get_str_attr(file_info, "name")
                )
                raw_mime = (
                    _get_str_attr(message, "mime_type")
                    or _get_str_attr(file_info, "mime_type")
                )
                file_size = _get_int_attr(message, "file_size") or _get_int_attr(file_info, "size")

                if raw_name:
                    file_name = Path(raw_name).name
                elif raw_mime == "image/gif":
                    file_name = "animation.gif"
                elif kind == "video_note":
                    file_name = "video_note.mp4"
                elif kind == "animation":
                    file_name = "animation.mp4"
                else:
                    file_name = "video.mp4"

                if not Path(file_name).suffix:
                    file_name += ".gif" if raw_mime == "image/gif" else ".mp4"

                max_video_bytes = MAX_VIDEO_SIZE_MB * 1024 * 1024
                if file_size > max_video_bytes:
                    size_mb = file_size / (1024 * 1024)
                    return ProcessedMedia(
                        media_type="animation" if kind == "animation" else "video",
                        mime_type=raw_mime or "video/mp4",
                        text_content=(
                            f"[Video '{file_name}' ({size_mb:.2f} MB) exceeds maximum allowed size of {MAX_VIDEO_SIZE_MB} MB]"
                        ),
                        metadata={
                            "_temp_dir": str(temp_dir),
                            "oversized": True,
                            "file_name": file_name,
                            "file_size": file_size,
                        },
                    )

                raw_media_path = temp_dir / file_name
                await self._download_media_to(client, message, raw_media_path)

                frame_path = temp_dir / "frame.jpg"
                await self._extract_video_or_gif_frame(raw_media_path, frame_path)

                wav_path: Optional[Path] = None
                if kind == "video_note":
                    candidate_wav = temp_dir / "audio.wav"
                    try:
                        await extract_audio_from_video(
                            input_path=raw_media_path,
                            output_path=candidate_wav,
                            ffmpeg_path=self.ffmpeg_path,
                        )
                        if candidate_wav.exists() and candidate_wav.stat().st_size > 0:
                            wav_path = candidate_wav
                    except Exception as exc:
                        logger.debug("Optional audio extraction from video_note skipped: %s", exc)

                out_media_type = "animation" if kind == "animation" else "video"
                video_metadata: Dict[str, Any] = {
                    "_temp_dir": str(temp_dir),
                    "raw_path": str(raw_media_path),
                    "file_name": file_name,
                    "file_size": file_size,
                    "source_mime_type": raw_mime,
                    "is_video_note": kind == "video_note",
                }
                transcript: Optional[str] = None
                if kind == "video_note" and wav_path is not None:
                    transcript = await self._transcribe_audio(wav_path, video_metadata)
                return ProcessedMedia(
                    media_type=out_media_type,
                    mime_type="image/jpeg",
                    file_path=frame_path,
                    image_path=frame_path,
                    keyframe_path=frame_path,
                    audio_path=wav_path,
                    text_content=transcript,
                    metadata=video_metadata,
                )

            # 3. Photo Processing
            if kind == "photo":
                photo_path = temp_dir / "photo.jpg"
                await self._download_media_to(client, message, photo_path)
                return ProcessedMedia(
                    media_type="photo",
                    mime_type="image/jpeg",
                    file_path=photo_path,
                    image_path=photo_path,
                    metadata={"_temp_dir": str(temp_dir)},
                )

            # 4. Sticker Processing (WebP -> PNG conversion via Pillow)
            if kind == "sticker":
                webp_path = temp_dir / "sticker.webp"
                await self._download_media_to(client, message, webp_path)
                png_path = temp_dir / "sticker.png"

                # Convert static WebP sticker to PNG
                try:
                    with Image.open(webp_path) as img:
                        img.convert("RGBA").save(png_path, format="PNG")
                    target_file = png_path
                    target_mime = "image/png"
                except Exception as e:
                    logger.debug("Sticker conversion to PNG failed, using original: %s", e)
                    target_file = webp_path
                    target_mime = "image/webp"

                emoji = None
                file_info = getattr(message, "file", None)
                if file_info:
                    emoji = getattr(file_info, "emoji", None)

                return ProcessedMedia(
                    media_type="sticker",
                    mime_type=target_mime,
                    file_path=target_file,
                    image_path=target_file,
                    metadata={"_temp_dir": str(temp_dir), "emoji": emoji},
                )

            # 5. Document Processing with Size Guard & Text Extraction
            if kind == "document":
                file_info = getattr(message, "file", None)
                raw_name = getattr(file_info, "name", None)
                if isinstance(raw_name, str) and raw_name.strip():
                    file_name = Path(raw_name).name or "document"
                else:
                    file_name = "document"

                raw_size = getattr(file_info, "size", 0)
                file_size = int(raw_size) if isinstance(raw_size, (int, float)) else 0
                mime_type = getattr(file_info, "mime_type", None)
                if not isinstance(mime_type, str):
                    mime_type = None

                # Guard against oversized documents before downloading
                max_bytes = self.max_doc_size_mb * 1024 * 1024
                if file_size > max_bytes:
                    size_mb = file_size / (1024 * 1024)
                    return ProcessedMedia(
                        media_type="document",
                        mime_type=mime_type,
                        text_content=f"[Document '{file_name}' ({size_mb:.2f} MB) exceeds maximum allowed size of {self.max_doc_size_mb} MB]",
                        metadata={
                            "_temp_dir": str(temp_dir),
                            "oversized": True,
                            "file_name": file_name,
                            "file_size": file_size,
                        },
                    )

                doc_path = temp_dir / file_name
                await self._download_media_to(client, message, doc_path)

                # Check actual downloaded file size
                try:
                    text_content = extract_document_text(
                        doc_path,
                        mime_type=mime_type,
                        max_size_mb=self.max_doc_size_mb,
                    )
                except (FileTooLargeError, DocumentProcessingError) as err:
                    text_content = f"[{err}]"

                return ProcessedMedia(
                    media_type="document",
                    mime_type=mime_type,
                    file_path=doc_path,
                    text_content=text_content,
                    metadata={
                        "_temp_dir": str(temp_dir),
                        "file_name": file_name,
                        "file_size": file_size,
                    },
                )

            return None
        except Exception:
            # Clean up on error before propagating exception
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
