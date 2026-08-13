from __future__ import annotations

from pathlib import Path

import pytest

from audiobook.document_ingestion import ChapterTextLoadResult, load_chapter_text


def test_load_chapter_text_reads_plain_text_files(tmp_path: Path) -> None:
    chapter_path = tmp_path / "chapter.txt"
    chapter_path.write_text("Hello from a text chapter.", encoding="utf-8")

    result = load_chapter_text(chapter_path)

    assert isinstance(result, ChapterTextLoadResult)
    assert result.source_kind == "text"
    assert result.used_ocr is False
    assert result.text == "Hello from a text chapter."


def test_load_chapter_text_rejects_unknown_suffix(tmp_path: Path) -> None:
    chapter_path = tmp_path / "chapter.docx"
    chapter_path.write_text("Not supported.", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported chapter file format"):
        load_chapter_text(chapter_path)
