"""Deterministic document spec parsing and rendering subsystem."""

from bot.documents.builder import (
    MAX_DOCUMENT_SIZE_BYTES,
    DocumentBuilder,
    DocumentBuildError,
    DocumentFormat,
    DocumentSection,
    DocumentSpec,
    GeneratedDocument,
    extract_document_specs,
    find_cyrillic_font_pair,
    sanitize_filename,
)

__all__ = [
    "MAX_DOCUMENT_SIZE_BYTES",
    "DocumentBuildError",
    "DocumentBuilder",
    "DocumentFormat",
    "DocumentSection",
    "DocumentSpec",
    "GeneratedDocument",
    "extract_document_specs",
    "find_cyrillic_font_pair",
    "sanitize_filename",
]
