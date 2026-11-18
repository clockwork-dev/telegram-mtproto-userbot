"""Deterministic document rendering and structured spec extraction.

Supports PDF, DOCX, XLSX, and PPTX generation from structured LLM specs
inside isolated temporary directories, preferring headless ArtCraft CLIs
(``wordcraft-cli``, ``gridcraft-cli``, ``deckcraft-cli``) with Cyrillic-safe
pure-Python fallbacks (``reportlab``, ``python-docx``, ``openpyxl``/CSV).
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import logging
import os
import re
import shutil
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape as xml_escape

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

MAX_DOCUMENT_SIZE_BYTES = 45 * 1024 * 1024  # 45 MB (safe margin under Telegram 50 MB Bot API limit)

DocumentFormat = Literal["pdf", "docx", "xlsx", "pptx"]
_SUPPORTED_FORMATS = frozenset({"pdf", "docx", "xlsx", "pptx"})
_KNOWN_DOC_EXTENSIONS = frozenset({".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".txt", ".md", ".doc"})
_TAG_PREFIX_RE = re.compile(r"\[(?:DOC|DOCUMENT)\s*:", re.IGNORECASE)
_WINDOWS_FORBIDDEN_CHARS_RE = re.compile(r'[<>:"|?*\x00-\x1f]')
_SHEET_FORBIDDEN_CHARS_RE = re.compile(r"[:\\/?*\[\]\x00-\x1f]")

_FONT_SEARCH_ROOTS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
)
_PREFERRED_FONT_PAIRS = (
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    ("FreeSans.ttf", "FreeSansBold.ttf"),
)


class DocumentBuildError(Exception):
    """Raised when a document cannot be rendered or fails output validation."""


class DocumentSection(BaseModel):
    """Single logical section (or slide / sheet) of a structured document."""

    heading: Optional[str] = None
    paragraphs: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)
    table: Optional[List[List[Any]]] = None
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_variants(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)

        # Accept common LLM aliases for section heading
        if not coerced.get("heading"):
            for key in ("title", "name", "sheet"):
                if isinstance(coerced.get(key), str) and coerced[key].strip():
                    coerced["heading"] = coerced[key].strip()
                    break

        # Accept single string aliases (`body`, `text`, `content`) -> paragraphs
        paragraphs = list(coerced.get("paragraphs") or [])
        for key in ("body", "text", "content"):
            val = coerced.get(key)
            if isinstance(val, str) and val.strip():
                paragraphs.extend(p.strip() for p in val.splitlines() if p.strip())
            elif isinstance(val, list):
                paragraphs.extend(str(item).strip() for item in val if str(item).strip())
        coerced["paragraphs"] = [str(p) for p in paragraphs if str(p).strip()]

        if coerced.get("bullets"):
            coerced["bullets"] = [str(b).strip() for b in coerced["bullets"] if str(b).strip()]

        # Accept {"headers": [...], "rows": [[...]]} table shape
        table = coerced.get("table")
        if isinstance(table, dict):
            headers = table.get("headers") or table.get("columns")
            rows = table.get("rows") or []
            merged: List[List[Any]] = []
            if isinstance(headers, list) and headers:
                merged.append(headers)
            if isinstance(rows, list):
                merged.extend(rows)
            coerced["table"] = merged or None

        return coerced

    @field_validator("table")
    @classmethod
    def _normalize_table(cls, value: Optional[List[List[Any]]]) -> Optional[List[List[Any]]]:
        if not value:
            return None
        rows = [list(row) for row in value if isinstance(row, (list, tuple))]
        if not rows:
            return None
        max_cols = max((len(r) for r in rows), default=0)
        if max_cols == 0:
            return None
        return [r + [""] * (max_cols - len(r)) for r in rows]


class DocumentSpec(BaseModel):
    """Structured specification for generating an outbound document."""

    format: DocumentFormat = "pdf"
    title: str = "Document"
    subtitle: Optional[str] = None
    filename: Optional[str] = None
    caption: Optional[str] = None
    author: Optional[str] = None
    sections: List[DocumentSection] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_variants(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        coerced = dict(data)

        fmt = coerced.get("format")
        if isinstance(fmt, str):
            coerced["format"] = fmt.strip().lstrip(".").lower()

        if not coerced.get("title") or not str(coerced.get("title")).strip():
            coerced["title"] = str(coerced.get("filename") or "Document").strip() or "Document"

        # If LLM omits `sections` and places `body`/`content`/`paragraphs`/`bullets`/`table` at root
        if not coerced.get("sections"):
            root_sec: Dict[str, Any] = {}
            for key in ("paragraphs", "body", "text", "content", "bullets", "table"):
                if coerced.get(key):
                    root_sec[key] = coerced[key]
            if root_sec:
                coerced["sections"] = [root_sec]

        if coerced.get("sources"):
            coerced["sources"] = [str(s).strip() for s in coerced["sources"] if str(s).strip()]

        return coerced

    @field_validator("format")
    @classmethod
    def _validate_format(cls, value: str) -> str:
        normalized = value.strip().lstrip(".").lower()
        if normalized not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported document format: {value!r}")
        return normalized

    def resolved_filename(self) -> str:
        """Return a sanitized filename with an extension matching ``self.format``."""
        return sanitize_filename(self.filename, self.format, fallback_title=self.title)


@dataclass(frozen=True)
class GeneratedDocument:
    """Metadata and path for a rendered, validated output document."""

    spec: DocumentSpec
    path: Path
    filename: str
    format: str
    size_bytes: int
    backend: str
    caption: Optional[str] = None


def sanitize_filename(
    filename: Optional[str],
    fmt: str,
    fallback_title: Optional[str] = None,
) -> str:
    """Sanitize a user/LLM-supplied filename against traversal and ensure matching extension.

    Preserves Cyrillic and general UTF-8 characters while stripping path components,
    ``..`` traversal segments, control characters, and Windows-reserved symbols.
    """
    ext = fmt.strip().lstrip(".").lower()
    if ext not in _SUPPORTED_FORMATS and ext != "csv":
        ext = "pdf"

    def _clean_candidate(raw: Optional[str]) -> str:
        if not raw:
            return ""
        # Normalize both Unix and Windows separators, then take only the final segment
        normalized = raw.replace("\\", "/").strip()
        # Reject pure traversal segments explicitly before basename extraction
        parts = [p for p in normalized.split("/") if p and p not in {".", ".."}]
        candidate = parts[-1] if parts else ""
        candidate = candidate.replace("..", "")
        candidate = _WINDOWS_FORBIDDEN_CHARS_RE.sub("_", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .")
        return candidate

    cleaned = _clean_candidate(filename)
    if not cleaned:
        cleaned = _clean_candidate(fallback_title) or "document"

    # Cap stem length to stay well below 255-byte filesystem limits (Cyrillic = 2 bytes/char)
    target_suffix = f".{ext}"
    parsed = Path(cleaned)
    suffix_lower = parsed.suffix.lower()
    if suffix_lower == target_suffix:
        stem = parsed.stem
    elif suffix_lower in _KNOWN_DOC_EXTENSIONS:
        stem = parsed.stem
    else:
        stem = cleaned

    stem = stem.strip(" .")
    if not stem:
        stem = "document"
    stem = stem[:100].rstrip(" .")
    return f"{stem}{target_suffix}"


def _find_balanced_json_end(text: str, start_idx: int) -> Optional[int]:
    """Return index of matching ``}`` for the ``{`` at ``start_idx``, respecting JSON strings."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
    return None


def extract_document_specs(raw_text: str) -> Tuple[List[DocumentSpec], str]:
    """Extract ``[DOC:{...}]`` / ``[DOCUMENT:{...}]`` tags from LLM reply text.

    Supports multiline JSON and nested/string-embedded braces. Valid tags are parsed
    into ``DocumentSpec`` objects and stripped cleanly from the returned reply text.
    Malformed JSON or invalid spec tags are logged and ignored without raising.
    """
    if not raw_text:
        return [], ""

    specs: List[DocumentSpec] = []
    spans_to_strip: List[Tuple[int, int]] = []
    cursor = 0

    while cursor < len(raw_text):
        match = _TAG_PREFIX_RE.search(raw_text, cursor)
        if not match:
            break

        pos = match.end()
        while pos < len(raw_text) and raw_text[pos].isspace():
            pos += 1

        if pos >= len(raw_text) or raw_text[pos] != "{":
            cursor = match.end()
            continue

        json_end = _find_balanced_json_end(raw_text, pos)
        if json_end is None:
            logger.warning("Unclosed JSON object in [DOC:...] tag at offset %d", match.start())
            cursor = pos + 1
            continue

        close_pos = json_end + 1
        while close_pos < len(raw_text) and raw_text[close_pos].isspace():
            close_pos += 1

        if close_pos >= len(raw_text) or raw_text[close_pos] != "]":
            cursor = json_end + 1
            continue

        json_payload = raw_text[pos : json_end + 1]
        try:
            parsed = json.loads(json_payload)
            spec = DocumentSpec.model_validate(parsed)
        except Exception as exc:
            logger.warning("Ignoring invalid [DOC:...] spec at offset %d: %s", match.start(), exc)
            cursor = close_pos + 1
            continue

        specs.append(spec)
        spans_to_strip.append((match.start(), close_pos + 1))
        cursor = close_pos + 1

    if not spans_to_strip:
        return [], raw_text.strip()

    parts: List[str] = []
    last_idx = 0
    for start, end in spans_to_strip:
        parts.append(raw_text[last_idx:start])
        last_idx = end
    parts.append(raw_text[last_idx:])

    cleaned = "".join(parts)
    # Collapse whitespace gaps left behind by removed inline or standalone tags
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return specs, cleaned.strip()


def find_cyrillic_font_pair(
    search_roots: Sequence[Path] = _FONT_SEARCH_ROOTS,
) -> Optional[Tuple[Path, Path]]:
    """Locate a Cyrillic-capable (Regular, Bold) TrueType font pair on the system."""
    existing_roots = [root for root in search_roots if root.exists()]
    if not existing_roots:
        return None

    for regular_name, bold_name in _PREFERRED_FONT_PAIRS:
        regular_path: Optional[Path] = None
        bold_path: Optional[Path] = None
        for root in existing_roots:
            if regular_path is None:
                found = next(root.rglob(regular_name), None)
                if found and found.is_file():
                    regular_path = found
            if bold_path is None:
                found_bold = next(root.rglob(bold_name), None)
                if found_bold and found_bold.is_file():
                    bold_path = found_bold
        if regular_path is not None:
            return regular_path, (bold_path or regular_path)
    return None


def _escape_md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").strip()


def _spec_to_markdown(spec: DocumentSpec) -> str:
    """Convert a ``DocumentSpec`` into clean Markdown for ``wordcraft-cli convert``."""
    lines: List[str] = [f"# {spec.title.strip()}", ""]
    if spec.subtitle and spec.subtitle.strip():
        lines.extend([f"*{spec.subtitle.strip()}*", ""])

    for section in spec.sections:
        if section.heading and section.heading.strip():
            lines.extend([f"## {section.heading.strip()}", ""])

        for para in section.paragraphs:
            if para.strip():
                lines.extend([para.strip(), ""])

        if section.bullets:
            for item in section.bullets:
                lines.append(f"- {item.strip()}")
            lines.append("")

        if section.table:
            header = section.table[0]
            lines.append("| " + " | ".join(_escape_md_cell(c) for c in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in section.table[1:]:
                lines.append("| " + " | ".join(_escape_md_cell(c) for c in row) + " |")
            lines.append("")

        if section.notes and section.notes.strip():
            lines.extend([f"> {section.notes.strip()}", ""])

    if spec.sources:
        lines.extend(["## Источники", ""])
        for idx, src in enumerate(spec.sources, start=1):
            lines.append(f"{idx}. {src}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _col_letter(col_index_1based: int) -> str:
    """Convert 1-based column index to Excel column name (1 -> A, 27 -> AA)."""
    letters: List[str] = []
    n = col_index_1based
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord("A") + rem))
    return "".join(reversed(letters))


def _sanitize_sheet_name(name: str, used: Set[str]) -> str:
    """Produce a unique Excel-valid worksheet name (<=31 chars, no forbidden symbols)."""
    base = _SHEET_FORBIDDEN_CHARS_RE.sub("_", (name or "").strip()).strip("'")
    if not base:
        base = "Лист"
    base = base[:31]
    candidate = base
    counter = 2
    while candidate.lower() in used:
        suffix = f"_{counter}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        counter += 1
    used.add(candidate.lower())
    return candidate


async def _run_cli(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Optional[Dict[str, str]] = None,
    stdin_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    """Run an external CLI subprocess asynchronously with deterministic timeout cleanup."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=merged_env,
        stdin=asyncio.subprocess.PIPE if stdin_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8") if stdin_text is not None else None),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode if proc.returncode is not None else 1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


class DocumentBuilder:
    """Deterministic builder for PDF, DOCX, XLSX, and PPTX files inside isolated temp dirs."""

    def __init__(
        self,
        temp_dir_base: Optional[str | Path] = None,
        prefer_artcraft: bool = True,
        max_size_bytes: int = MAX_DOCUMENT_SIZE_BYTES,
        cli_timeout_seconds: float = 45.0,
        wordcraft_bin: Optional[str] = None,
        gridcraft_bin: Optional[str] = None,
        deckcraft_bin: Optional[str] = None,
    ) -> None:
        self.temp_dir_base = str(temp_dir_base) if temp_dir_base is not None else None
        if self.temp_dir_base:
            Path(self.temp_dir_base).mkdir(parents=True, exist_ok=True)
        self.prefer_artcraft = prefer_artcraft
        self.max_size_bytes = max_size_bytes
        self.cli_timeout_seconds = cli_timeout_seconds
        self.wordcraft_bin = wordcraft_bin or shutil.which("wordcraft-cli")
        self.gridcraft_bin = gridcraft_bin or shutil.which("gridcraft-cli")
        self.deckcraft_bin = deckcraft_bin or shutil.which("deckcraft-cli")

    def _create_temp_dir(self) -> Path:
        """Create an isolated temporary directory (0700 permissions via mkdtemp)."""
        d = tempfile.mkdtemp(prefix="bot_doc_", dir=self.temp_dir_base)
        return Path(d).resolve()

    @asynccontextmanager
    async def managed_generated_documents(
        self,
        specs: Sequence[DocumentSpec],
    ) -> AsyncIterator[List[GeneratedDocument]]:
        """Render ``specs`` inside an isolated temp dir and guarantee cleanup in ``finally``."""
        temp_dir = self._create_temp_dir()
        try:
            docs = await self.build_all(specs, output_dir=temp_dir)
            yield docs
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def build_all(
        self,
        specs: Sequence[DocumentSpec],
        output_dir: Path,
    ) -> List[GeneratedDocument]:
        """Render all specs into ``output_dir``, deduplicating colliding filenames."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        used_names: Set[str] = set()
        results: List[GeneratedDocument] = []
        for spec in specs:
            base_name = spec.resolved_filename()
            deduped_name = self._dedupe_filename(base_name, used_names)
            doc = await self.build_document(spec, output_dir=output_dir, filename_override=deduped_name)
            used_names.add(doc.filename.lower())
            results.append(doc)
        return results

    @staticmethod
    def _dedupe_filename(filename: str, used_names: Set[str]) -> str:
        if filename.lower() not in used_names:
            used_names.add(filename.lower())
            return filename
        path = Path(filename)
        counter = 2
        while True:
            candidate = f"{path.stem}_{counter}{path.suffix}"
            if candidate.lower() not in used_names:
                used_names.add(candidate.lower())
                return candidate
            counter += 1

    async def build_document(
        self,
        spec: DocumentSpec,
        output_dir: Path,
        filename_override: Optional[str] = None,
    ) -> GeneratedDocument:
        """Render a single ``DocumentSpec`` into ``output_dir`` and validate the output."""
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_name = sanitize_filename(
            filename_override or spec.filename,
            spec.format,
            fallback_title=spec.title,
        )
        target_path = (output_dir / safe_name).resolve()
        if target_path.parent != output_dir:
            raise DocumentBuildError(f"Unsafe target path resolved outside output_dir: {target_path}")

        if spec.format == "pdf":
            actual_path, actual_fmt, backend = await self._build_pdf(spec, output_dir, target_path)
        elif spec.format == "docx":
            actual_path, actual_fmt, backend = await self._build_docx(spec, output_dir, target_path)
        elif spec.format == "xlsx":
            actual_path, actual_fmt, backend = await self._build_xlsx(spec, output_dir, target_path)
        elif spec.format == "pptx":
            actual_path, actual_fmt, backend = await self._build_pptx(spec, output_dir, target_path)
        else:
            raise DocumentBuildError(f"Unsupported format: {spec.format}")

        size_bytes = self._validate_output_file(actual_path)
        return GeneratedDocument(
            spec=spec,
            path=actual_path,
            filename=actual_path.name,
            format=actual_fmt,
            size_bytes=size_bytes,
            backend=backend,
            caption=spec.caption,
        )

    def _validate_output_file(self, path: Path) -> int:
        if not path.exists() or not path.is_file():
            raise DocumentBuildError(f"Generated document does not exist: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise DocumentBuildError(f"Generated document is empty (0 bytes): {path}")
        if size > self.max_size_bytes:
            raise DocumentBuildError(
                f"Generated document ({size} bytes) exceeds {self.max_size_bytes} bytes limit: {path}"
            )
        return size

    async def _build_pdf(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> Tuple[Path, str, str]:
        if self.prefer_artcraft and self.wordcraft_bin:
            try:
                await self._render_with_wordcraft(spec, work_dir, target_path, strict=True)
                return target_path, "pdf", "wordcraft-cli"
            except Exception as exc:
                logger.warning("wordcraft-cli PDF failed, falling back to reportlab: %s", exc)
                target_path.unlink(missing_ok=True)

        await asyncio.to_thread(self._render_pdf_reportlab, spec, target_path)
        return target_path, "pdf", "reportlab"

    async def _build_docx(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> Tuple[Path, str, str]:
        if self.prefer_artcraft and self.wordcraft_bin:
            try:
                await self._render_with_wordcraft(spec, work_dir, target_path, strict=False)
                return target_path, "docx", "wordcraft-cli"
            except Exception as exc:
                logger.warning("wordcraft-cli DOCX failed, falling back to python-docx: %s", exc)
                target_path.unlink(missing_ok=True)

        await asyncio.to_thread(self._render_docx_python_docx, spec, target_path)
        return target_path, "docx", "python-docx"

    async def _build_xlsx(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> Tuple[Path, str, str]:
        if self.prefer_artcraft and self.gridcraft_bin:
            try:
                await self._render_with_gridcraft(spec, work_dir, target_path)
                return target_path, "xlsx", "gridcraft-cli"
            except Exception as exc:
                logger.warning("gridcraft-cli XLSX failed, falling back: %s", exc)
                target_path.unlink(missing_ok=True)

        if importlib.util.find_spec("openpyxl") is not None:
            await asyncio.to_thread(self._render_xlsx_openpyxl, spec, target_path)
            return target_path, "xlsx", "openpyxl"

        csv_path = target_path.with_suffix(".csv")
        await asyncio.to_thread(self._render_csv_fallback, spec, csv_path)
        return csv_path, "csv", "csv-fallback"

    async def _build_pptx(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> Tuple[Path, str, str]:
        if self.prefer_artcraft and self.deckcraft_bin:
            try:
                await self._render_with_deckcraft(spec, work_dir, target_path)
                return target_path, "pptx", "deckcraft-cli"
            except Exception as exc:
                logger.warning("deckcraft-cli PPTX failed: %s", exc)
                target_path.unlink(missing_ok=True)
                if importlib.util.find_spec("pptx") is None:
                    raise DocumentBuildError(f"deckcraft-cli PPTX rendering failed: {exc}") from exc

        if importlib.util.find_spec("pptx") is not None:
            await asyncio.to_thread(self._render_pptx_python_pptx, spec, target_path)
            return target_path, "pptx", "python-pptx"

        raise DocumentBuildError(
            "PPTX rendering unavailable: deckcraft-cli not found/failed and python-pptx not installed."
        )

    async def _render_with_wordcraft(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
        *,
        strict: bool,
    ) -> None:
        if not self.wordcraft_bin:
            raise DocumentBuildError("wordcraft-cli binary not found")
        md_file = work_dir / f"_input_{target_path.stem}.md"
        md_file.write_text(_spec_to_markdown(spec), encoding="utf-8")
        cmd = [self.wordcraft_bin, "convert", str(md_file), str(target_path)]
        if strict:
            cmd.append("--strict")
        try:
            rc, stdout, stderr = await _run_cli(
                cmd,
                cwd=work_dir,
                timeout=self.cli_timeout_seconds,
                env={"WORDCRAFT_REGION": "ru-RU"},
            )
            if rc != 0:
                raise DocumentBuildError(
                    f"wordcraft-cli exited with {rc}: {(stderr or stdout).strip()}"
                )
            self._validate_output_file(target_path)
        finally:
            md_file.unlink(missing_ok=True)

    async def _render_with_gridcraft(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> None:
        if not self.gridcraft_bin:
            raise DocumentBuildError("gridcraft-cli binary not found")

        commands: List[Dict[str, Any]] = [{"command": "file.new"}]
        used_sheets: Set[str] = set()
        first_sheet = True

        def _start_sheet(name: str) -> None:
            nonlocal first_sheet
            sheet_name = _sanitize_sheet_name(name, used_sheets)
            if first_sheet:
                commands.append({"command": "sheet.rename", "params": {"name": sheet_name}})
                first_sheet = False
            else:
                commands.append({"command": "home.insertSheet", "params": {"name": sheet_name}})

        # Build summary sheet if there are narrative sections or sources, or if no table sections exist
        narrative_rows: List[List[Any]] = [[spec.title]]
        if spec.subtitle:
            narrative_rows.append([spec.subtitle])
        narrative_rows.append([])

        table_sections: List[Tuple[str, List[List[Any]]]] = []
        for idx, section in enumerate(spec.sections, start=1):
            sec_title = section.heading or f"Раздел {idx}"
            if section.paragraphs or section.bullets:
                narrative_rows.append([sec_title])
                for p in section.paragraphs:
                    narrative_rows.append([p])
                for b in section.bullets:
                    narrative_rows.append([f"• {b}"])
                narrative_rows.append([])
            if section.table:
                table_sections.append((sec_title, section.table))

        if spec.sources:
            narrative_rows.append(["Источники"])
            for s in spec.sources:
                narrative_rows.append([s])

        if not table_sections or len(narrative_rows) > 3:
            _start_sheet("Обзор")
            commands.append(
                {
                    "command": "range.setValues",
                    "params": {"range": "A1", "values": narrative_rows, "parse": False},
                }
            )
            commands.append({"command": "home.bold", "params": {"range": "A1:A1", "on": True}})
            commands.append({"command": "home.autofitColumnWidth", "params": {}})

        for sheet_title, rows in table_sections:
            _start_sheet(sheet_title)
            num_rows = len(rows)
            num_cols = max((len(r) for r in rows), default=1)
            end_col = _col_letter(num_cols)
            commands.append(
                {
                    "command": "range.setValues",
                    "params": {"range": "A1", "values": rows, "parse": True},
                }
            )
            header_range = f"A1:{end_col}1"
            full_range = f"A1:{end_col}{num_rows}"
            commands.append({"command": "home.bold", "params": {"range": header_range, "on": True}})
            commands.append(
                {"command": "home.fillColor", "params": {"range": header_range, "color": "#DCE6F1"}}
            )
            commands.append(
                {"command": "home.borders", "params": {"range": full_range, "preset": "all"}}
            )
            commands.append({"command": "home.autofitColumnWidth", "params": {}})

        script_path = work_dir / f"_grid_{target_path.stem}.jsonl"
        script_path.write_text(
            "\n".join(json.dumps(cmd, ensure_ascii=False) for cmd in commands) + "\n",
            encoding="utf-8",
        )
        try:
            rc, stdout, stderr = await _run_cli(
                [
                    self.gridcraft_bin,
                    "run",
                    "--script",
                    str(script_path),
                    "--out",
                    str(target_path),
                ],
                cwd=work_dir,
                timeout=self.cli_timeout_seconds,
            )
            if rc != 0:
                raise DocumentBuildError(
                    f"gridcraft-cli exited with {rc}: {(stderr or stdout).strip()}"
                )
            self._validate_output_file(target_path)
        finally:
            script_path.unlink(missing_ok=True)

    async def _render_with_deckcraft(
        self,
        spec: DocumentSpec,
        work_dir: Path,
        target_path: Path,
    ) -> None:
        if not self.deckcraft_bin:
            raise DocumentBuildError("deckcraft-cli binary not found")

        lines: List[str] = [
            f"file.new={json.dumps({'blank': True}, ensure_ascii=False)}",
            "slide.new="
            + json.dumps(
                {
                    "layout": "title",
                    "title": spec.title,
                    "body": spec.subtitle or "",
                },
                ensure_ascii=False,
            ),
        ]

        for idx, section in enumerate(spec.sections, start=1):
            heading = section.heading or f"Раздел {idx}"
            body_parts: List[str] = list(section.paragraphs)
            body_parts.extend(f"• {b}" for b in section.bullets)
            body_text = "\n".join(body_parts)

            if body_text or not section.table:
                lines.append(
                    "slide.new="
                    + json.dumps(
                        {
                            "layout": "titleAndContent",
                            "title": heading,
                            "body": body_text,
                        },
                        ensure_ascii=False,
                    )
                )
                if section.notes:
                    lines.append(
                        f"slide.notes={json.dumps({'text': section.notes}, ensure_ascii=False)}"
                    )

            if section.table:
                num_rows = len(section.table)
                num_cols = max((len(r) for r in section.table), default=1)
                table_height = min(370, max(120, num_rows * 44))
                lines.append(
                    "slide.new="
                    + json.dumps(
                        {"layout": "titleOnly", "title": heading},
                        ensure_ascii=False,
                    )
                )
                lines.append(
                    "insert.table="
                    + json.dumps(
                        {
                            "rows": num_rows,
                            "cols": num_cols,
                            "rect": [50, 120, 860, table_height],
                        },
                        ensure_ascii=False,
                    )
                )
                for r_idx, row in enumerate(section.table):
                    for c_idx, cell_val in enumerate(row):
                        lines.append(
                            "text.set="
                            + json.dumps(
                                {
                                    "cell": [r_idx, c_idx],
                                    "text": "" if cell_val is None else str(cell_val),
                                },
                                ensure_ascii=False,
                            )
                        )
                if section.notes and not body_text:
                    lines.append(
                        f"slide.notes={json.dumps({'text': section.notes}, ensure_ascii=False)}"
                    )

        if spec.sources:
            lines.append(
                "slide.new="
                + json.dumps(
                    {
                        "layout": "titleAndContent",
                        "title": "Источники",
                        "body": "\n".join(f"{i}. {s}" for i, s in enumerate(spec.sources, start=1)),
                    },
                    ensure_ascii=False,
                )
            )

        cmd_file = work_dir / f"_deck_{target_path.stem}.txt"
        cmd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            rc, stdout, stderr = await _run_cli(
                [
                    self.deckcraft_bin,
                    "run",
                    "--cmd-file",
                    str(cmd_file),
                    "--save",
                    str(target_path),
                ],
                cwd=work_dir,
                timeout=self.cli_timeout_seconds,
            )
            if rc != 0:
                raise DocumentBuildError(
                    f"deckcraft-cli exited with {rc}: {(stderr or stdout).strip()}"
                )
            self._validate_output_file(target_path)
        finally:
            cmd_file.unlink(missing_ok=True)

    @staticmethod
    def _render_pdf_reportlab(spec: DocumentSpec, target_path: Path) -> None:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as exc:
            raise DocumentBuildError("reportlab is required for PDF fallback rendering") from exc

        font_pair = find_cyrillic_font_pair()
        if font_pair is None:
            raise DocumentBuildError(
                "No Cyrillic-capable TTF font (DejaVuSans/LiberationSans/NotoSans) found for reportlab."
            )
        regular_ttf, bold_ttf = font_pair
        regular_name = "BotDocSans"
        bold_name = "BotDocSans-Bold"

        pdfmetrics.registerFont(TTFont(regular_name, str(regular_ttf)))
        pdfmetrics.registerFont(TTFont(bold_name, str(bold_ttf)))
        pdfmetrics.registerFontFamily(
            regular_name,
            normal=regular_name,
            bold=bold_name,
            italic=regular_name,
            boldItalic=bold_name,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "BotDocTitle",
            parent=styles["Title"],
            fontName=bold_name,
            fontSize=18,
            leading=23,
            textColor=colors.HexColor("#1B2A4A"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "BotDocSubtitle",
            parent=styles["Normal"],
            fontName=regular_name,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "BotDocHeading",
            parent=styles["Heading2"],
            fontName=bold_name,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1B2A4A"),
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "BotDocBody",
            parent=styles["Normal"],
            fontName=regular_name,
            fontSize=10.5,
            leading=14.5,
            textColor=colors.HexColor("#1A202C"),
            spaceAfter=6,
        )
        bullet_style = ParagraphStyle(
            "BotDocBullet",
            parent=body_style,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=3,
        )
        cell_style = ParagraphStyle(
            "BotDocCell",
            parent=body_style,
            fontSize=9.5,
            leading=12.5,
            spaceAfter=0,
        )
        header_cell_style = ParagraphStyle(
            "BotDocHeaderCell",
            parent=cell_style,
            fontName=bold_name,
            textColor=colors.white,
        )

        doc = SimpleDocTemplate(
            str(target_path),
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=spec.title,
            author=spec.author or "Telegram Group Bot",
        )

        story: List[Any] = [Paragraph(xml_escape(spec.title), title_style)]
        if spec.subtitle:
            story.append(Paragraph(xml_escape(spec.subtitle), subtitle_style))
        else:
            story.append(Spacer(1, 4 * mm))

        for section in spec.sections:
            if section.heading:
                story.append(Paragraph(xml_escape(section.heading), heading_style))
            for para in section.paragraphs:
                story.append(Paragraph(xml_escape(para), body_style))
            for item in section.bullets:
                story.append(Paragraph(f"• {xml_escape(item)}", bullet_style))
            if section.table:
                num_cols = max((len(r) for r in section.table), default=1)
                avail_width = A4[0] - 36 * mm
                col_width = avail_width / num_cols
                table_data: List[List[Paragraph]] = []
                for r_idx, row in enumerate(section.table):
                    st = header_cell_style if r_idx == 0 else cell_style
                    table_data.append(
                        [Paragraph(xml_escape("" if c is None else str(c)), st) for c in row]
                    )
                tbl = Table(table_data, colWidths=[col_width] * num_cols, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B4C7E")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.extend([Spacer(1, 2 * mm), tbl, Spacer(1, 4 * mm)])
            if section.notes:
                story.append(Paragraph(xml_escape(section.notes), subtitle_style))

        if spec.sources:
            story.append(Paragraph("Источники", heading_style))
            for idx, src in enumerate(spec.sources, start=1):
                story.append(Paragraph(f"{idx}. {xml_escape(src)}", body_style))

        doc.build(story)

    @staticmethod
    def _render_docx_python_docx(spec: DocumentSpec, target_path: Path) -> None:
        try:
            import docx
        except ImportError as exc:
            raise DocumentBuildError("python-docx is required for DOCX fallback rendering") from exc

        document = docx.Document()
        document.core_properties.title = spec.title
        if spec.author:
            document.core_properties.author = spec.author

        normal_style = document.styles["Normal"]
        normal_style.font.name = "DejaVu Sans"

        document.add_heading(spec.title, level=0)
        if spec.subtitle:
            sub_p = document.add_paragraph(spec.subtitle)
            if sub_p.runs:
                sub_p.runs[0].italic = True

        for section in spec.sections:
            if section.heading:
                document.add_heading(section.heading, level=1)
            for para in section.paragraphs:
                document.add_paragraph(para)
            for item in section.bullets:
                document.add_paragraph(item, style="List Bullet")
            if section.table:
                num_rows = len(section.table)
                num_cols = max((len(r) for r in section.table), default=1)
                table = document.add_table(rows=num_rows, cols=num_cols)
                table.style = "Table Grid"
                for r_idx, row in enumerate(section.table):
                    for c_idx, val in enumerate(row):
                        cell = table.cell(r_idx, c_idx)
                        cell.text = "" if val is None else str(val)
                        if r_idx == 0:
                            for p in cell.paragraphs:
                                for run in p.runs:
                                    run.bold = True
            if section.notes:
                note_p = document.add_paragraph(section.notes)
                if note_p.runs:
                    note_p.runs[0].italic = True

        if spec.sources:
            document.add_heading("Источники", level=1)
            for idx, src in enumerate(spec.sources, start=1):
                document.add_paragraph(f"{idx}. {src}")

        document.save(str(target_path))

    @staticmethod
    def _render_xlsx_openpyxl(spec: DocumentSpec, target_path: Path) -> None:
        import openpyxl  # type: ignore[import-not-found]
        from openpyxl.styles import Font  # type: ignore[import-not-found]

        wb = openpyxl.Workbook()
        ws = wb.active
        used_sheets: Set[str] = set()
        ws.title = _sanitize_sheet_name("Обзор", used_sheets)
        ws.append([spec.title])
        ws["A1"].font = Font(bold=True, size=14)
        if spec.subtitle:
            ws.append([spec.subtitle])
        ws.append([])

        for idx, section in enumerate(spec.sections, start=1):
            heading = section.heading or f"Раздел {idx}"
            if section.paragraphs or section.bullets:
                ws.append([heading])
                for p in section.paragraphs:
                    ws.append([p])
                for b in section.bullets:
                    ws.append([f"• {b}"])
                ws.append([])
            if section.table:
                table_ws = wb.create_sheet(title=_sanitize_sheet_name(heading, used_sheets))
                for r_idx, row in enumerate(section.table, start=1):
                    table_ws.append(row)
                    if r_idx == 1:
                        for cell in table_ws[1]:
                            cell.font = Font(bold=True)

        if spec.sources:
            ws.append(["Источники"])
            for s in spec.sources:
                ws.append([s])

        wb.save(str(target_path))

    @staticmethod
    def _render_csv_fallback(spec: DocumentSpec, target_path: Path) -> None:
        with target_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([spec.title])
            if spec.subtitle:
                writer.writerow([spec.subtitle])
            writer.writerow([])
            for section in spec.sections:
                if section.heading:
                    writer.writerow([section.heading])
                for p in section.paragraphs:
                    writer.writerow([p])
                for b in section.bullets:
                    writer.writerow([f"• {b}"])
                if section.table:
                    for row in section.table:
                        writer.writerow(["" if c is None else str(c) for c in row])
                writer.writerow([])
            if spec.sources:
                writer.writerow(["Источники"])
                for s in spec.sources:
                    writer.writerow([s])

    @staticmethod
    def _render_pptx_python_pptx(spec: DocumentSpec, target_path: Path) -> None:
        from pptx import Presentation  # type: ignore[import-not-found]

        prs = Presentation()
        title_slide = prs.slides.add_slide(prs.slide_layouts[0])
        title_slide.shapes.title.text = spec.title
        if len(title_slide.placeholders) > 1 and spec.subtitle:
            title_slide.placeholders[1].text = spec.subtitle

        for idx, section in enumerate(spec.sections, start=1):
            heading = section.heading or f"Раздел {idx}"
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = heading
            body_lines = list(section.paragraphs) + [f"• {b}" for b in section.bullets]
            if section.table:
                body_lines.extend(" | ".join(str(c) for c in row) for row in section.table)
            if len(slide.placeholders) > 1:
                slide.placeholders[1].text = "\n".join(body_lines)

        prs.save(str(target_path))
