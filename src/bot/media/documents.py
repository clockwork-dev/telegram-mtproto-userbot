"""Document ingestion and text extraction for Multimodal Pipeline.

Supports PDF, DOCX, TXT, CSV, JSON, Markdown, and source code files with
strict size guards and context budget truncation.
"""

import logging
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Known text / code file extensions
TEXT_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".sh",
    ".bash",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".sql",
    ".ini",
    ".conf",
    ".log",
    ".env",
    ".toml",
}


class FileTooLargeError(Exception):
    """Raised when a document exceeds the maximum configured size limit."""


class DocumentProcessingError(Exception):
    """Raised when a document is a binary/media/archive file that cannot be decoded as text."""


# Binary media, audio, video, archive, and executable extensions that must never be decoded as UTF-8 text
BINARY_EXTENSIONS: Set[str] = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".gif",
    ".mp3",
    ".ogg",
    ".wav",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".exe",
    ".bin",
}


def _is_binary_header(header: bytes) -> bool:
    """Return True if the first 512 bytes contain binary magic signatures or null bytes."""
    if not header:
        return False
    # MP4 / QuickTime / ISO Base Media File Format ('ftyp' at bytes 4..8)
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return True
    # Common binary media/archive magic headers
    if header.startswith((
        b"GIF87a",
        b"GIF89a",
        b"\x1a\x45\xdf\xa3",  # WebM / MKV (EBML)
        b"RIFF",              # AVI / WAV / WebP
        b"OggS",              # Ogg
        b"ID3",               # MP3
        b"fLaC",              # FLAC
        b"PK\x03\x04",        # ZIP
        b"\x1f\x8b",          # GZIP
        b"7z\xbc\xaf\x27\x1c",  # 7z
        b"\x7fELF",           # ELF binary
        b"MZ",                # PE/EXE binary
    )):
        return True
    # Null bytes in the first 512 bytes indicate non-text binary data
    if b"\x00" in header:
        return True
    return False


def check_file_size(file_path: Path | str, max_size_mb: int = 10) -> bool:
    """Return True if file size is within limits; raises FileTooLargeError if exceeded."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_bytes = path.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise FileTooLargeError(
            f"Document '{path.name}' size ({size_mb:.2f} MB) exceeds maximum allowed {max_size_mb} MB"
        )
    return True


def extract_pdf_text(file_path: Path | str, max_chars: int = 50000) -> str:
    """Extract textual content from a PDF document using pypdf."""
    import pypdf

    path = Path(file_path)
    try:
        reader = pypdf.PdfReader(str(path))
        extracted_pages = []
        total_len = 0

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(f"--- Page {idx + 1} ---\n{page_text.strip()}")
                total_len += len(page_text)
                if total_len >= max_chars:
                    break

        combined = "\n\n".join(extracted_pages)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n... [Truncated due to context limit]"
        return combined
    except Exception as e:
        logger.warning("Failed to extract text from PDF %s: %s", path.name, e)
        raise


def extract_docx_text(file_path: Path | str, max_chars: int = 50000) -> str:
    """Extract textual content from a DOCX document using python-docx."""
    import docx

    path = Path(file_path)
    try:
        doc = docx.Document(str(path))
        lines = []

        for p in doc.paragraphs:
            if p.text.strip():
                lines.append(p.text.strip())

        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))

        combined = "\n".join(lines)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n... [Truncated due to context limit]"
        return combined
    except Exception as e:
        logger.warning("Failed to extract text from DOCX %s: %s", path.name, e)
        raise


def extract_plain_text(file_path: Path | str, max_chars: int = 50000) -> str:
    """Extract text from plain text or source code files with UTF-8 decoding."""
    path = Path(file_path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)

        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [Truncated due to context limit]"
        return content
    except Exception as e:
        logger.warning("Failed to read text file %s: %s", path.name, e)
        raise


def extract_document_text(
    file_path: Path | str,
    mime_type: Optional[str] = None,
    max_chars: int = 50000,
    max_size_mb: int = 10,
) -> str:
    """Extract readable text from a document based on extension or mime-type with size guards."""
    path = Path(file_path)
    check_file_size(path, max_size_mb=max_size_mb)

    ext = path.suffix.lower()
    mime_lower = (mime_type or "").lower().strip()

    if ext in BINARY_EXTENSIONS:
        raise DocumentProcessingError(
            f"Binary or media file '{path.name}' ({ext}) cannot be decoded as text."
        )

    if mime_lower.startswith(("video/", "audio/", "image/")):
        raise DocumentProcessingError(
            f"Binary media MIME type '{mime_type}' for '{path.name}' cannot be decoded as text."
        )

    with open(path, "rb") as bf:
        header = bf.read(512)

    if len(header) >= 8 and header[4:8] == b"ftyp":
        raise DocumentProcessingError(
            f"MP4/ISO media container header ('ftyp') detected in '{path.name}'; refusing UTF-8 decode."
        )

    if ext == ".pdf" or "pdf" in mime_lower:
        return extract_pdf_text(path, max_chars=max_chars)

    if ext in {".docx", ".doc"} or "wordprocessingml" in mime_lower:
        return extract_docx_text(path, max_chars=max_chars)

    if _is_binary_header(header):
        raise DocumentProcessingError(
            f"Binary data or magic bytes detected in '{path.name}'; refusing UTF-8 decode."
        )

    if ext in TEXT_EXTENSIONS or mime_lower.startswith("text/") or "json" in mime_lower:
        return extract_plain_text(path, max_chars=max_chars)

    # Fallback attempt as plain text only after verifying header is non-binary
    try:
        return extract_plain_text(path, max_chars=max_chars)
    except Exception as exc:
        logger.info("Unsupported or binary document type: %s (%s)", path.name, ext)
        raise DocumentProcessingError(
            f"Unsupported document format: {ext or mime_type or 'binary'}"
        ) from exc
