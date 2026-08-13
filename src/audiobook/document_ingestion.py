from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .text_chunker import normalize_text

SUPPORTED_TEXT_SUFFIXES = (".txt", ".text", ".md")
SUPPORTED_CHAPTER_SUFFIXES = SUPPORTED_TEXT_SUFFIXES + (".pdf",)


@dataclass(frozen=True)
class PageExtractionResult:
    page_number: int
    text: str
    word_count: int
    alnum_count: int
    kept: bool


@dataclass(frozen=True)
class ChapterTextLoadResult:
    source_path: Path
    source_kind: str
    text: str
    used_ocr: bool
    page_results: tuple[PageExtractionResult, ...]

    @property
    def kept_page_count(self) -> int:
        return sum(1 for page in self.page_results if page.kept)

    @property
    def skipped_page_count(self) -> int:
        return sum(1 for page in self.page_results if not page.kept)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _alnum_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]", text))


def _word_count(text: str) -> int:
    return len(text.split())


def _ensure_tool_exists(tool_name: str) -> None:
    if shutil.which(tool_name) is None:
        raise RuntimeError(
            f"Required system tool '{tool_name}' is not available. "
            "Install the Colab OCR dependencies first."
        )


def read_text_chapter(chapter_path: str | Path) -> ChapterTextLoadResult:
    path = Path(chapter_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    return ChapterTextLoadResult(
        source_path=path,
        source_kind="text",
        text=text,
        used_ocr=False,
        page_results=(),
    )


def _extract_direct_pdf_text(pdf_path: Path) -> str:
    _ensure_tool_exists("pdftotext")
    result = _run_command(["pdftotext", str(pdf_path), "-"])
    return result.stdout


def _render_pdf_pages(pdf_path: Path, image_root: Path) -> list[Path]:
    _ensure_tool_exists("pdftoppm")
    image_root.mkdir(parents=True, exist_ok=True)
    _run_command(["pdftoppm", "-png", str(pdf_path), str(image_root / "page")])
    pages = sorted(image_root.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"No page images were rendered from PDF: {pdf_path}")
    return pages


def _ocr_image(image_path: Path) -> str:
    _ensure_tool_exists("tesseract")
    result = _run_command(["tesseract", str(image_path), "stdout"])
    return result.stdout


def extract_pdf_text(
    pdf_path: str | Path,
    *,
    minimum_direct_text_alnum: int = 800,
    minimum_ocr_page_words: int = 30,
    keep_low_text_pages: bool = False,
    force_ocr: bool = False,
    scratch_directory: str | Path | None = None,
) -> ChapterTextLoadResult:
    path = Path(pdf_path).expanduser().resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path}")

    direct_text = _extract_direct_pdf_text(path)
    if not force_ocr and _alnum_count(direct_text) >= minimum_direct_text_alnum:
        return ChapterTextLoadResult(
            source_path=path,
            source_kind="pdf",
            text=direct_text,
            used_ocr=False,
            page_results=(),
        )

    if scratch_directory is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="audiobook-pdf-ocr-")
        scratch_root = Path(temp_dir.name)
    else:
        temp_dir = None
        scratch_root = Path(scratch_directory).expanduser().resolve()
        scratch_root.mkdir(parents=True, exist_ok=True)

    try:
        images = _render_pdf_pages(path, scratch_root / "pages")
        page_results: list[PageExtractionResult] = []
        for page_index, image_path in enumerate(images, start=1):
            text = _ocr_image(image_path)
            normalized = normalize_text(text)
            words = _word_count(normalized)
            alnum = _alnum_count(normalized)
            keep_page = keep_low_text_pages or words >= minimum_ocr_page_words
            page_results.append(
                PageExtractionResult(
                    page_number=page_index,
                    text=normalized,
                    word_count=words,
                    alnum_count=alnum,
                    kept=keep_page,
                )
            )

        kept_pages = [page for page in page_results if page.kept and page.text]
        if not kept_pages:
            kept_pages = [page for page in page_results if page.text]

        combined_text = "\n\n".join(page.text for page in kept_pages if page.text).strip()
        return ChapterTextLoadResult(
            source_path=path,
            source_kind="pdf",
            text=combined_text,
            used_ocr=True,
            page_results=tuple(page_results),
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def load_chapter_text(
    chapter_path: str | Path,
    *,
    force_ocr: bool = False,
    minimum_direct_text_alnum: int = 800,
    minimum_ocr_page_words: int = 30,
    keep_low_text_pages: bool = False,
    scratch_directory: str | Path | None = None,
) -> ChapterTextLoadResult:
    path = Path(chapter_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return read_text_chapter(path)
    if suffix == ".pdf":
        return extract_pdf_text(
            path,
            force_ocr=force_ocr,
            minimum_direct_text_alnum=minimum_direct_text_alnum,
            minimum_ocr_page_words=minimum_ocr_page_words,
            keep_low_text_pages=keep_low_text_pages,
            scratch_directory=scratch_directory,
        )
    supported = ", ".join(SUPPORTED_CHAPTER_SUFFIXES)
    raise ValueError(
        f"Unsupported chapter file format '{suffix}'. Supported formats: {supported}"
    )
