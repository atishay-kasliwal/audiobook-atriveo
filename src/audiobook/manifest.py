from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import hashlib

from .text_chunker import TextSegment


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_text_payload(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class SegmentRecord:
    id: str
    index: int
    paragraph_index: int
    text: str
    char_count: int
    starts_paragraph: bool
    ends_paragraph: bool
    status: str = "pending"
    audio_path: str | None = None
    attempts: int = 0
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    error: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SegmentRecord":
        return cls(**payload)


@dataclass
class ChapterManifest:
    book_id: str
    chapter_number: int
    language: str
    chapter_source: str
    source_text_hash: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    segments: list[SegmentRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "book_id": self.book_id,
            "chapter_number": self.chapter_number,
            "language": self.language,
            "chapter_source": self.chapter_source,
            "source_text_hash": self.source_text_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChapterManifest":
        return cls(
            book_id=payload["book_id"],
            chapter_number=payload["chapter_number"],
            language=payload["language"],
            chapter_source=payload["chapter_source"],
            source_text_hash=payload["source_text_hash"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            config=payload["config"],
            segments=[SegmentRecord.from_dict(item) for item in payload["segments"]],
        )

    def segment_map(self) -> dict[str, SegmentRecord]:
        return {segment.id: segment for segment in self.segments}


def build_manifest(
    *,
    book_id: str,
    chapter_number: int,
    language: str,
    chapter_source: str,
    source_text: str,
    config: dict[str, Any],
    segments: list[TextSegment],
) -> ChapterManifest:
    timestamp = utc_timestamp()
    records = [
        SegmentRecord(
            id=segment.segment_id,
            index=segment.index,
            paragraph_index=segment.paragraph_index,
            text=segment.text,
            char_count=segment.char_count,
            starts_paragraph=segment.starts_paragraph,
            ends_paragraph=segment.ends_paragraph,
            updated_at=timestamp,
        )
        for segment in segments
    ]
    return ChapterManifest(
        book_id=book_id,
        chapter_number=chapter_number,
        language=language,
        chapter_source=chapter_source,
        source_text_hash=hash_text_payload(source_text),
        created_at=timestamp,
        updated_at=timestamp,
        config=config,
        segments=records,
    )


def save_manifest(manifest: ChapterManifest, manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = utc_timestamp()
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_manifest(manifest_path: str | Path) -> ChapterManifest:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return ChapterManifest.from_dict(payload)

