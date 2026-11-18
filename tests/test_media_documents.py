"""Unit tests for multimodal document text extraction and size guards."""

import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import docx
import pytest
from reportlab.pdfgen import canvas

from bot.media.documents import (
    DocumentProcessingError,
    FileTooLargeError,
    check_file_size,
    extract_document_text,
    extract_docx_text,
    extract_pdf_text,
    extract_plain_text,
)


def test_pdf_text_extraction(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Autonomous Telegram Userbot PDF Content")
    c.drawString(100, 700, "Second line of test information.")
    c.save()

    extracted = extract_pdf_text(pdf_path)
    assert "Autonomous Telegram Userbot PDF Content" in extracted
    assert "Second line of test information." in extracted


def test_docx_text_extraction(tmp_path):
    docx_path = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_heading("Architecture Overview", level=1)
    doc.add_paragraph("This is a paragraph inside the DOCX document.")

    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Header 1"
    table.rows[0].cells[1].text = "Header 2"
    table.rows[1].cells[0].text = "Val A"
    table.rows[1].cells[1].text = "Val B"

    doc.save(str(docx_path))

    extracted = extract_docx_text(docx_path)
    assert "Architecture Overview" in extracted
    assert "This is a paragraph inside the DOCX document." in extracted
    assert "Header 1 | Header 2" in extracted
    assert "Val A | Val B" in extracted


def test_plain_text_and_code_extraction(tmp_path):
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Hello from plain text note.\nSecond line.", encoding="utf-8")

    code_path = tmp_path / "script.py"
    code_path.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,role\nAlice,Admin\nBob,Dev\n", encoding="utf-8")

    json_path = tmp_path / "config.json"
    json_path.write_text('{"bot": "telethon", "active": true}', encoding="utf-8")

    assert "Hello from plain text note." in extract_plain_text(txt_path)
    assert "def hello():" in extract_document_text(code_path)
    assert "Alice,Admin" in extract_document_text(csv_path)
    assert '"active": true' in extract_document_text(json_path)


def test_oversized_document_guard(tmp_path):
    large_file = tmp_path / "large.txt"
    # Create 11 MB sparse file
    with open(large_file, "wb") as f:
        f.seek(11 * 1024 * 1024 - 1)
        f.write(b"\0")

    assert large_file.stat().st_size >= 11 * 1024 * 1024

    with pytest.raises(FileTooLargeError, match="exceeds maximum allowed 10 MB"):
        check_file_size(large_file, max_size_mb=10)

    with pytest.raises(FileTooLargeError):
        extract_document_text(large_file, max_size_mb=10)


def test_context_budget_truncation(tmp_path):
    long_file = tmp_path / "long.txt"
    long_file.write_text("X" * 1000, encoding="utf-8")

    extracted = extract_plain_text(long_file, max_chars=100)
    assert len(extracted) < 200
    assert "Truncated due to context limit" in extracted


def test_reject_mp4_ftyp_and_binary_files(tmp_path):
    mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x01\x00mdat"

    # 1. Standard .mp4 extension
    mp4_file = tmp_path / "super_rjaka_demotivator_bot.mp4"
    mp4_file.write_bytes(mp4_bytes)
    with pytest.raises(DocumentProcessingError):
        extract_document_text(mp4_file, mime_type="video/mp4")

    # 2. Disguised MP4 header without .mp4 extension (e.g. 'document' or '.txt')
    disguised_mp4 = tmp_path / "disguised_video.txt"
    disguised_mp4.write_bytes(mp4_bytes)
    with pytest.raises(DocumentProcessingError):
        extract_document_text(disguised_mp4)

    # 3. Null bytes in generic binary file
    raw_bin = tmp_path / "unknown_payload"
    raw_bin.write_bytes(b"header\x00\x00binary_payload\xff\xfe")
    with pytest.raises(DocumentProcessingError):
        extract_document_text(raw_bin)
