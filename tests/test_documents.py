"""Tests for deterministic document spec extraction and rendering."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import docx
import pytest
from pypdf import PdfReader

from bot.documents import (
    DocumentBuilder,
    DocumentBuildError,
    DocumentSection,
    DocumentSpec,
    extract_document_specs,
    find_cyrillic_font_pair,
    sanitize_filename,
)


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class TestSpecExtractionAndSanitization:
    def test_extract_single_line_doc_tag(self) -> None:
        raw = (
            "Вот краткий итог исследования.\n"
            '[DOC:{"format":"pdf","title":"Отчёт","filename":"отчёт.pdf","sections":[{"heading":"Введение","paragraphs":["Текст"]}]}]\n'
            "Если нужно дополнить — напишите."
        )
        specs, clean = extract_document_specs(raw)
        assert len(specs) == 1
        assert specs[0].format == "pdf"
        assert specs[0].title == "Отчёт"
        assert specs[0].resolved_filename() == "отчёт.pdf"
        assert "[DOC:" not in clean
        assert "Вот краткий итог исследования." in clean
        assert "Если нужно дополнить — напишите." in clean

    def test_extract_multiline_document_tag_with_braces_in_strings(self) -> None:
        raw = """Готовлю таблицу с показателями.

[DOCUMENT:
{
  "format": "DOCX",
  "title": "Анализ {Q3} -> {Q4}",
  "subtitle": "Фигурные скобки } внутри строки { не ломают парсер",
  "filename": "..\\\\..\\\\evil\\\\анализ.txt",
  "sections": [
    {
      "heading": "Данные",
      "body": "Первый абзац\\nВторой абзац",
      "bullets": ["Пункт A {1}", "Пункт B }2{"],
      "table": {
        "headers": ["Метрика", "Значение"],
        "rows": [["Выручка", "100%"]]
      }
    }
  ],
  "sources": ["https://example.com/a?x={1}"]
}
]

Готово!"""
        specs, clean = extract_document_specs(raw)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.format == "docx"
        assert spec.title == "Анализ {Q3} -> {Q4}"
        assert spec.sections[0].paragraphs == ["Первый абзац", "Второй абзац"]
        assert spec.sections[0].table == [["Метрика", "Значение"], ["Выручка", "100%"]]
        assert spec.resolved_filename() == "анализ.docx"
        assert "[DOCUMENT:" not in clean
        assert "Готовлю таблицу с показателями." in clean
        assert "Готово!" in clean

    def test_extract_multiple_tags_and_invalid_json_resilience(self) -> None:
        raw = (
            "Начало.\n"
            '[DOC:{"format":"pdf","title":"Первый","sections":[{"paragraphs":["OK"]}]}]\n'
            "[DOC:{broken json: not valid}]\n"
            '[DOC:{"format":"exe","title":"BadFormat"}]\n'
            "[DOC:{unclosed\n"
            '[DOCUMENT:{"format":"xlsx","title":"Второй"}]\n'
            "Конец."
        )
        specs, clean = extract_document_specs(raw)
        assert [s.title for s in specs] == ["Первый", "Второй"]
        assert [s.format for s in specs] == ["pdf", "xlsx"]
        assert "Начало." in clean
        assert "Конец." in clean
        assert '"Первый"' not in clean
        assert '"Второй"' not in clean

    @pytest.mark.parametrize(
        ("raw_name", "fmt", "fallback", "expected"),
        [
            ("../../etc/passwd", "pdf", "Fallback", "passwd.pdf"),
            ("..\\..\\secret\\отчёт.docx", "pdf", "Fallback", "отчёт.pdf"),
            ("../..", "xlsx", "Сводка за год", "Сводка за год.xlsx"),
            ("   ", "pptx", None, "document.pptx"),
            ("my.report.v2", "pdf", None, "my.report.v2.pdf"),
            ("Кириллица: тест <1>.PDF", "pdf", None, "Кириллица_ тест _1_.pdf"),
        ],
    )
    def test_sanitize_filename(
        self, raw_name: str, fmt: str, fallback: str | None, expected: str
    ) -> None:
        cleaned = sanitize_filename(raw_name, fmt, fallback_title=fallback)
        assert cleaned == expected
        assert ".." not in cleaned
        assert "/" not in cleaned
        assert "\\" not in cleaned


class TestDocumentRendering:
    def _sample_spec(self, fmt: str = "pdf") -> DocumentSpec:
        return DocumentSpec(
            format=fmt,  # type: ignore[arg-type]
            title="Прямой тест кириллицы",
            subtitle="Подзаголовок документа — проверка шрифтов",
            filename=f"тест_кириллицы.{fmt}",
            sections=[
                DocumentSection(
                    heading="Раздел 1: Основные сведения",
                    paragraphs=["Прямой тест кириллицы в абзаце: ёжик, щука, №100, ₽."],
                    bullets=["Первый пункт списка", "Второй пункт списка"],
                    table=[
                        ["Показатель", "Значение", "Комментарий"],
                        ["Выручка", 1500, "Кириллица в таблице"],
                        ["Итого", "=SUM(B2:B2)", "Расчёт"],
                    ],
                    notes="Примечание докладчика для проверки.",
                )
            ],
            sources=["https://example.com/ru-report"],
        )

    async def test_pdf_reportlab_fallback_cyrillic_verified_with_pypdf(self, tmp_path: Path) -> None:
        assert find_cyrillic_font_pair() is not None, "Expected Cyrillic TTF font on system"
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=False)
        spec = self._sample_spec("pdf")
        doc = await builder.build_document(spec, output_dir=tmp_path)

        assert doc.backend == "reportlab"
        assert doc.format == "pdf"
        assert doc.path.exists() and doc.size_bytes > 0
        text = _extract_pdf_text(doc.path)
        assert "Прямой тест кириллицы" in text
        assert "Кириллица в таблице" in text
        assert "Источники" in text

    async def test_pdf_wordcraft_cli_falls_back_to_reportlab_on_failure(
        self, tmp_path: Path
    ) -> None:
        bad_bin = tmp_path / "fake-wordcraft-cli"
        bad_bin.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
        bad_bin.chmod(0o755)

        builder = DocumentBuilder(
            temp_dir_base=tmp_path,
            prefer_artcraft=True,
            wordcraft_bin=str(bad_bin),
        )
        doc = await builder.build_document(self._sample_spec("pdf"), output_dir=tmp_path)
        assert doc.backend == "reportlab"
        assert "Прямой тест кириллицы" in _extract_pdf_text(doc.path)

    async def test_docx_python_docx_fallback_cyrillic_verified(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=False)
        doc = await builder.build_document(self._sample_spec("docx"), output_dir=tmp_path)

        assert doc.backend == "python-docx"
        assert doc.path.exists() and doc.size_bytes > 0

        loaded = docx.Document(str(doc.path))
        all_text = "\n".join(p.text for p in loaded.paragraphs)
        assert "Прямой тест кириллицы" in all_text
        assert "Первый пункт списка" in all_text
        assert len(loaded.tables) == 1
        assert loaded.tables[0].cell(1, 2).text == "Кириллица в таблице"

    async def test_artcraft_wordcraft_cli_pdf_and_docx(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=True)
        if not builder.wordcraft_bin:
            pytest.skip("wordcraft-cli not installed in test environment")

        pdf_doc = await builder.build_document(self._sample_spec("pdf"), output_dir=tmp_path)
        assert pdf_doc.backend == "wordcraft-cli"
        assert pdf_doc.path.exists() and pdf_doc.size_bytes > 0
        pdf_text = _extract_pdf_text(pdf_doc.path)
        assert "Прямой тест кириллицы" in pdf_text
        assert "Кириллица в таблице" in pdf_text

        docx_doc = await builder.build_document(self._sample_spec("docx"), output_dir=tmp_path)
        assert docx_doc.backend == "wordcraft-cli"
        assert docx_doc.path.exists() and docx_doc.size_bytes > 0
        loaded = docx.Document(str(docx_doc.path))
        assert any("Прямой тест кириллицы" in p.text for p in loaded.paragraphs)
        assert len(loaded.tables) == 1

    async def test_artcraft_gridcraft_cli_xlsx(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=True)
        if not builder.gridcraft_bin:
            pytest.skip("gridcraft-cli not installed in test environment")

        xlsx_doc = await builder.build_document(self._sample_spec("xlsx"), output_dir=tmp_path)
        assert xlsx_doc.backend == "gridcraft-cli"
        assert xlsx_doc.format == "xlsx"
        assert xlsx_doc.path.exists() and xlsx_doc.size_bytes > 0

        with zipfile.ZipFile(xlsx_doc.path, "r") as zf:
            names = set(zf.namelist())
            assert "xl/workbook.xml" in names

        cat = subprocess.run(
            [builder.gridcraft_bin, "cat", str(xlsx_doc.path), "--sheet", "Раздел 1_ Основные сведения", "--csv"],
            text=True,
            capture_output=True,
            check=True,
        )
        assert "Показатель,Значение,Комментарий" in cat.stdout
        assert "1500" in cat.stdout
        assert "Кириллица в таблице" in cat.stdout

    async def test_artcraft_deckcraft_cli_pptx(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=True)
        if not builder.deckcraft_bin:
            pytest.skip("deckcraft-cli not installed in test environment")

        pptx_doc = await builder.build_document(self._sample_spec("pptx"), output_dir=tmp_path)
        assert pptx_doc.backend == "deckcraft-cli"
        assert pptx_doc.format == "pptx"
        assert pptx_doc.path.exists() and pptx_doc.size_bytes > 0

        info = subprocess.run(
            [builder.deckcraft_bin, "info", str(pptx_doc.path)],
            text=True,
            capture_output=True,
            check=True,
        )
        meta = json.loads(info.stdout)
        # Title slide + content slide + table slide + sources slide = 4 slides
        assert len(meta["slides"]) == 4

        with zipfile.ZipFile(pptx_doc.path, "r") as zf:
            slide_xml = b"".join(
                zf.read(name) for name in zf.namelist() if name.startswith("ppt/slides/slide")
            ).decode("utf-8", errors="ignore")
            assert "Прямой тест кириллицы" in slide_xml
            assert "Кириллица в таблице" in slide_xml

    async def test_xlsx_csv_fallback_when_no_gridcraft_or_openpyxl(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(temp_dir_base=tmp_path, prefer_artcraft=False)
        doc = await builder.build_document(self._sample_spec("xlsx"), output_dir=tmp_path)
        if doc.backend == "openpyxl":
            assert doc.format == "xlsx"
        else:
            assert doc.backend == "csv-fallback"
            assert doc.format == "csv"
            content = doc.path.read_text(encoding="utf-8-sig")
            assert "Прямой тест кириллицы" in content
            assert "Кириллица в таблице" in content

    async def test_managed_generated_documents_dedupes_and_cleans_up(self, tmp_path: Path) -> None:
        scratch_root = tmp_path / "scratch"
        builder = DocumentBuilder(temp_dir_base=scratch_root, prefer_artcraft=False)
        specs = [
            DocumentSpec(format="pdf", title="Первый", filename="same.pdf"),
            DocumentSpec(format="pdf", title="Второй", filename="same.pdf"),
        ]

        captured_dir: Path | None = None
        async with builder.managed_generated_documents(specs) as docs:
            assert len(docs) == 2
            assert docs[0].filename == "same.pdf"
            assert docs[1].filename == "same_2.pdf"
            assert all(d.path.exists() and d.size_bytes > 0 for d in docs)
            captured_dir = docs[0].path.parent
            assert captured_dir.parent == scratch_root.resolve()

        assert captured_dir is not None
        assert not captured_dir.exists()

    async def test_managed_generated_documents_cleans_up_on_exception(self, tmp_path: Path) -> None:
        scratch_root = tmp_path / "scratch"
        builder = DocumentBuilder(temp_dir_base=scratch_root, prefer_artcraft=False)
        spec = DocumentSpec(format="pdf", title="Тест очистки", filename="cleanup.pdf")

        captured_dir: Path | None = None
        with pytest.raises(RuntimeError, match="simulated send error"):
            async with builder.managed_generated_documents([spec]) as docs:
                captured_dir = docs[0].path.parent
                assert captured_dir.exists()
                raise RuntimeError("simulated send error")

        assert captured_dir is not None
        assert not captured_dir.exists()

    async def test_size_limit_validation_raises_document_build_error(self, tmp_path: Path) -> None:
        builder = DocumentBuilder(
            temp_dir_base=tmp_path,
            prefer_artcraft=False,
            max_size_bytes=10,
        )
        with pytest.raises(DocumentBuildError, match="exceeds 10 bytes limit"):
            await builder.build_document(self._sample_spec("pdf"), output_dir=tmp_path)
