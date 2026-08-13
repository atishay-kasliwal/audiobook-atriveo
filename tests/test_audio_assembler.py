from __future__ import annotations

from pathlib import Path

import numpy as np

from audiobook.audio_assembler import assemble_chapter_audio, write_wav
from audiobook.config import ExportSettings, SilenceSettings
from audiobook.manifest import SegmentRecord


def test_assemble_chapter_audio_inserts_expected_silences(tmp_path: Path) -> None:
    sample_rate = 1000
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    write_wav(first_path, np.ones(sample_rate // 10, dtype=np.float32) * 0.2, sample_rate)
    write_wav(second_path, np.ones(sample_rate // 5, dtype=np.float32) * 0.4, sample_rate)

    records = [
        SegmentRecord(
            id="chapter_001_segment_0001",
            index=1,
            paragraph_index=1,
            text="First segment.",
            char_count=14,
            starts_paragraph=True,
            ends_paragraph=True,
            status="completed",
            audio_path=str(first_path),
        ),
        SegmentRecord(
            id="chapter_001_segment_0002",
            index=2,
            paragraph_index=2,
            text="Second segment.",
            char_count=15,
            starts_paragraph=True,
            ends_paragraph=True,
            status="completed",
            audio_path=str(second_path),
        ),
    ]

    result = assemble_chapter_audio(
        [first_path, second_path],
        records,
        silence=SilenceSettings(
            between_segments_ms=100,
            between_paragraphs_ms=300,
            chapter_prefix_ms=200,
            chapter_suffix_ms=400,
        ),
        export=ExportSettings(normalize_peak=0.98),
    )

    expected_samples = 200 + 100 + 300 + 200 + 400
    assert result.sample_rate == sample_rate
    assert result.channels == 1
    assert result.audio.shape[0] == expected_samples

