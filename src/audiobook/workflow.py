from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio_assembler import (
    assemble_chapter_audio,
    export_with_ffmpeg,
    ffmpeg_available,
    validate_rendered_audio,
    write_wav,
)
from .config import AudiobookJobConfig
from .manifest import ChapterManifest, SegmentRecord, build_manifest, load_manifest, save_manifest
from .text_chunker import chunk_text


class SegmentGenerator(Protocol):
    sample_rate: int | None

    def generate_segment(
        self,
        text: str,
        *,
        language_code: str,
        audio_prompt_path: str | Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ChapterRenderResult:
    manifest_path: Path
    chapter_wav_path: Path
    segment_paths: list[Path]
    failed_segment_ids: list[str]
    mp3_path: Path | None = None
    m4b_path: Path | None = None


def _run_log_path(config: AudiobookJobConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        config.paths.logs_dir
        / f"{config.chapter.book_id}_chapter_{config.chapter.chapter_number:03d}_{timestamp}.log"
    )


def append_log(log_path: Path, message: str, **payload: object) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def manifest_path_for(config: AudiobookJobConfig) -> Path:
    return (
        config.paths.manifests_dir
        / f"{config.chapter.book_id}_chapter_{config.chapter.chapter_number:03d}.json"
    )


def segment_audio_path_for(config: AudiobookJobConfig, segment_id: str) -> Path:
    return (
        config.paths.segments_dir
        / config.chapter.book_id
        / f"chapter_{config.chapter.chapter_number:03d}"
        / f"{segment_id}.wav"
    )


def chapter_audio_path_for(config: AudiobookJobConfig) -> Path:
    return (
        config.paths.chapters_dir
        / config.chapter.book_id
        / f"{config.chapter.book_id}_chapter_{config.chapter.chapter_number:03d}.wav"
    )


def _compare_segment_layout(expected: ChapterManifest, actual: ChapterManifest) -> None:
    if expected.source_text_hash != actual.source_text_hash:
        raise ValueError(
            "The chapter text has changed since this manifest was created. "
            "Use a new chapter file, a new book id, or delete/regenerate the old manifest intentionally."
        )
    expected_pairs = [(segment.id, segment.text) for segment in expected.segments]
    actual_pairs = [(segment.id, segment.text) for segment in actual.segments]
    if expected_pairs != actual_pairs:
        raise ValueError(
            "The existing manifest no longer matches the current chunking layout. "
            "Refusing to reuse stale segment ids."
        )


def refresh_manifest_for_resume(
    manifest: ChapterManifest,
    *,
    force_regeneration: bool,
) -> ChapterManifest:
    for segment in manifest.segments:
        audio_exists = bool(segment.audio_path) and Path(segment.audio_path).exists()
        if force_regeneration:
            segment.status = "pending"
            segment.attempts = 0
            segment.error = None
            continue
        if segment.status == "completed" and audio_exists:
            continue
        if segment.status == "completed" and not audio_exists:
            segment.status = "pending"
            segment.error = "Completed audio file was missing during resume."
            continue
        if segment.status == "failed" and audio_exists:
            segment.status = "completed"
            segment.error = None
    return manifest


def prepare_or_resume_manifest(
    config: AudiobookJobConfig,
    *,
    chapter_text: str,
) -> tuple[ChapterManifest, Path]:
    segments = chunk_text(
        chapter_text,
        chapter_number=config.chapter.chapter_number,
        language_code=config.chapter.target_language,
        minimum_chunk_characters=config.generation.minimum_chunk_characters,
        maximum_chunk_characters=config.generation.maximum_chunk_characters,
    )
    draft_manifest = build_manifest(
        book_id=config.chapter.book_id,
        chapter_number=config.chapter.chapter_number,
        language=config.chapter.target_language,
        chapter_source=str(config.chapter.chapter_file),
        source_text=chapter_text,
        config=config.to_manifest_config(),
        segments=segments,
    )

    path = manifest_path_for(config)
    if path.exists():
        existing = load_manifest(path)
        _compare_segment_layout(existing, draft_manifest)
        manifest = refresh_manifest_for_resume(
            existing,
            force_regeneration=config.generation.force_regeneration,
        )
    else:
        manifest = draft_manifest
    save_manifest(manifest, path)
    return manifest, path


def synthesize_manifest_segments(
    config: AudiobookJobConfig,
    manifest: ChapterManifest,
    engine: SegmentGenerator,
) -> ChapterManifest:
    log_path = _run_log_path(config)
    append_log(log_path, "segment_generation_started", manifest_path=str(manifest_path_for(config)))

    if config.generation.voice_file:
        audio_prompt_path: str | None = str(config.generation.voice_file)
    else:
        audio_prompt_path = None

    for segment in manifest.segments:
        target_path = segment_audio_path_for(config, segment.id)
        existing_completed_path = Path(segment.audio_path) if segment.audio_path else target_path
        if (
            segment.status == "completed"
            and existing_completed_path.exists()
            and not config.generation.force_regeneration
        ):
            append_log(log_path, "segment_skipped", segment_id=segment.id, reason="already_completed")
            continue

        success = False
        for attempt in range(segment.attempts + 1, config.generation.maximum_retries + 1):
            try:
                audio = engine.generate_segment(
                    segment.text,
                    language_code=config.chapter.target_language,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=config.generation.exaggeration,
                    cfg_weight=config.generation.cfg_weight,
                    temperature=config.generation.temperature,
                )
                validate_rendered_audio(audio)
                sample_rate = engine.sample_rate
                if sample_rate is None:
                    raise RuntimeError("The engine did not expose a sample rate after generation.")
                write_wav(target_path, audio, sample_rate)
                segment.audio_path = str(target_path)
                segment.sample_rate = sample_rate
                segment.channels = 1
                segment.duration_seconds = float(len(audio) / sample_rate)
                segment.status = "completed"
                segment.error = None
                segment.attempts = attempt
                segment.updated_at = datetime.now(timezone.utc).isoformat()
                save_manifest(manifest, manifest_path_for(config))
                append_log(log_path, "segment_completed", segment_id=segment.id, attempt=attempt)
                success = True
                break
            except Exception as exc:
                segment.status = "failed"
                segment.error = str(exc)
                segment.attempts = attempt
                segment.updated_at = datetime.now(timezone.utc).isoformat()
                save_manifest(manifest, manifest_path_for(config))
                append_log(
                    log_path,
                    "segment_failed",
                    segment_id=segment.id,
                    attempt=attempt,
                    error=str(exc),
                )
        if not success:
            append_log(log_path, "segment_exhausted_retries", segment_id=segment.id)
    return manifest


def render_chapter_audio(
    config: AudiobookJobConfig,
    manifest: ChapterManifest,
) -> ChapterRenderResult:
    completed_segments: list[SegmentRecord] = []
    segment_paths: list[Path] = []
    failed_segment_ids: list[str] = []
    for segment in manifest.segments:
        if segment.status == "completed" and segment.audio_path and Path(segment.audio_path).exists():
            completed_segments.append(segment)
            segment_paths.append(Path(segment.audio_path))
            continue
        failed_segment_ids.append(segment.id)

    if failed_segment_ids and not config.generation.partial_export:
        raise RuntimeError(
            "Refusing to assemble the chapter because some segments are incomplete: "
            + ", ".join(failed_segment_ids)
        )
    if not completed_segments:
        raise RuntimeError("No completed segments are available for chapter assembly.")

    assembly = assemble_chapter_audio(
        [str(path) for path in segment_paths],
        completed_segments,
        silence=config.silence,
        export=config.export,
    )
    chapter_wav_path = chapter_audio_path_for(config)
    write_wav(chapter_wav_path, assembly.audio, assembly.sample_rate)

    return ChapterRenderResult(
        manifest_path=manifest_path_for(config),
        chapter_wav_path=chapter_wav_path,
        segment_paths=segment_paths,
        failed_segment_ids=failed_segment_ids,
    )


def export_optional_chapter_formats(
    config: AudiobookJobConfig,
    chapter_wav_path: str | Path,
) -> tuple[Path | None, Path | None]:
    chapter_wav = Path(chapter_wav_path)
    mp3_path: Path | None = None
    if config.export.export_mp3 and ffmpeg_available():
        mp3_path = chapter_wav.with_suffix(".mp3")
        export_with_ffmpeg(chapter_wav, mp3_path, bitrate=config.export.mp3_bitrate)

    m4b_path: Path | None = None
    if config.export.export_m4b and ffmpeg_available():
        m4b_path = chapter_wav.with_suffix(".m4b")
        export_with_ffmpeg(chapter_wav, m4b_path, bitrate=config.export.m4b_bitrate)

    return mp3_path, m4b_path
