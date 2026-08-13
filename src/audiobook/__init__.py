"""Utilities for generating audiobooks with official Chatterbox models."""

from .chatterbox_engine import ChatterboxEngine, VoiceReferenceValidation
from .config import (
    SUPPORTED_LANGUAGE_CODES,
    SUPPORTED_LANGUAGE_LABELS,
    AudiobookJobConfig,
    build_job_config,
)
from .document_ingestion import (
    ChapterTextLoadResult,
    PageExtractionResult,
    SUPPORTED_CHAPTER_SUFFIXES,
    load_chapter_text,
)
from .manifest import ChapterManifest, SegmentRecord
from .text_chunker import TextSegment, chunk_text
from .workflow import (
    ChapterRenderResult,
    export_optional_chapter_formats,
    prepare_or_resume_manifest,
    render_chapter_audio,
    synthesize_manifest_segments,
)

__all__ = [
    "AudiobookJobConfig",
    "ChapterTextLoadResult",
    "ChapterManifest",
    "ChapterRenderResult",
    "ChatterboxEngine",
    "PageExtractionResult",
    "SegmentRecord",
    "SUPPORTED_LANGUAGE_CODES",
    "SUPPORTED_LANGUAGE_LABELS",
    "SUPPORTED_CHAPTER_SUFFIXES",
    "TextSegment",
    "VoiceReferenceValidation",
    "build_job_config",
    "chunk_text",
    "export_optional_chapter_formats",
    "load_chapter_text",
    "prepare_or_resume_manifest",
    "render_chapter_audio",
    "synthesize_manifest_segments",
]
