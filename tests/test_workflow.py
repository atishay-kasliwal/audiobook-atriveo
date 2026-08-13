from __future__ import annotations

from pathlib import Path

import numpy as np

from audiobook.audio_assembler import write_wav
from audiobook.chatterbox_engine import ChatterboxEngine
from audiobook.config import build_job_config
from audiobook.manifest import load_manifest, save_manifest
from audiobook.workflow import (
    prepare_or_resume_manifest,
    render_chapter_audio,
    synthesize_manifest_segments,
)


class FakeEngine:
    def __init__(self) -> None:
        self.sample_rate = 24000
        self.calls: list[str] = []

    def generate_segment(self, text: str, **_: object) -> np.ndarray:
        self.calls.append(text)
        duration = self.sample_rate // 10
        return np.linspace(0.0, 0.2, duration, dtype=np.float32)


def build_test_config(tmp_path: Path, *, partial_export: bool = False):
    chapter_file = tmp_path / "chapter.txt"
    chapter_file.write_text(
        "First paragraph starts here. It continues with another sentence.\n\n"
        "Second paragraph adds more narration for the test case.",
        encoding="utf-8",
    )
    return build_job_config(
        repository_url="https://github.com/example/book-to-audio",
        branch_name="main",
        repository_directory_name="book-to-audio",
        private_repository=False,
        book_id="demo-book",
        chapter_number=1,
        chapter_file=chapter_file,
        source_language="en",
        target_language="en",
        voice_file=None,
        output_directory=tmp_path / "drive",
        minimum_chunk_characters=40,
        maximum_chunk_characters=90,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        maximum_retries=2,
        force_regeneration=False,
        allow_cpu_fallback=False,
        partial_export=partial_export,
        export_mp3=False,
        export_m4b=False,
    )


def test_manifest_creation_round_trip(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    chapter_text = config.chapter.chapter_file.read_text(encoding="utf-8")

    manifest, manifest_path = prepare_or_resume_manifest(config, chapter_text=chapter_text)
    reloaded = load_manifest(manifest_path)

    assert manifest_path.exists()
    assert reloaded.book_id == "demo-book"
    assert reloaded.segments[0].id == "chapter_001_segment_0001"
    assert reloaded.segments[0].status == "pending"


def test_resume_marks_missing_completed_audio_as_pending(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    chapter_text = config.chapter.chapter_file.read_text(encoding="utf-8")
    manifest, manifest_path = prepare_or_resume_manifest(config, chapter_text=chapter_text)

    manifest.segments[0].status = "completed"
    manifest.segments[0].audio_path = str(tmp_path / "missing.wav")
    save_manifest(manifest, manifest_path)

    resumed_manifest, _ = prepare_or_resume_manifest(config, chapter_text=chapter_text)
    assert resumed_manifest.segments[0].status == "pending"
    assert "missing" in (resumed_manifest.segments[0].error or "").lower()


def test_completed_segment_protection_skips_regeneration(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    chapter_text = config.chapter.chapter_file.read_text(encoding="utf-8")
    manifest, manifest_path = prepare_or_resume_manifest(config, chapter_text=chapter_text)

    prebuilt_audio = tmp_path / "existing.wav"
    write_wav(prebuilt_audio, np.ones(2400, dtype=np.float32) * 0.1, 24000)
    manifest.segments[0].status = "completed"
    manifest.segments[0].audio_path = str(prebuilt_audio)
    save_manifest(manifest, manifest_path)

    engine = FakeEngine()
    updated_manifest = synthesize_manifest_segments(config, manifest, engine)

    assert updated_manifest.segments[0].status == "completed"
    assert len(engine.calls) == len(updated_manifest.segments) - 1


def test_render_chapter_audio_refuses_incomplete_manifest_without_partial_export(
    tmp_path: Path,
) -> None:
    config = build_test_config(tmp_path, partial_export=False)
    chapter_text = config.chapter.chapter_file.read_text(encoding="utf-8")
    manifest, _ = prepare_or_resume_manifest(config, chapter_text=chapter_text)

    engine = FakeEngine()
    synthesize_manifest_segments(config, manifest, engine)

    manifest.segments[-1].status = "failed"
    manifest.segments[-1].audio_path = None

    try:
        render_chapter_audio(config, manifest)
    except RuntimeError as exc:
        assert "Refusing to assemble" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("render_chapter_audio should fail when required segments are missing.")


def test_missing_voice_file_handling() -> None:
    engine = ChatterboxEngine(device="cpu", allow_cpu_fallback=True, model_cls=object())
    try:
        engine.validate_voice_reference("does-not-exist.wav")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a missing voice file to raise FileNotFoundError.")
