from __future__ import annotations

import pytest

from audiobook.text_chunker import chunk_text


def test_chunk_text_preserves_abbreviations_and_paragraph_boundaries() -> None:
    text = (
        "Dr. Rao waited outside the station. Then she checked the time and stepped inside.\n\n"
        "\"Hello there,\" he said. She smiled and nodded before the meeting began."
    )

    segments = chunk_text(
        text,
        chapter_number=1,
        language_code="en",
        minimum_chunk_characters=40,
        maximum_chunk_characters=90,
    )

    assert segments[0].text.startswith("Dr. Rao waited outside the station.")
    assert segments[0].starts_paragraph is True
    assert segments[0].paragraph_index == 1
    assert segments[-1].paragraph_index == 2
    assert all("\n\n" not in segment.text for segment in segments)


def test_segment_ids_are_stable_for_identical_input() -> None:
    text = "One sentence follows another. A third sentence closes the paragraph."

    first_run = chunk_text(
        text,
        chapter_number=4,
        language_code="en",
        minimum_chunk_characters=30,
        maximum_chunk_characters=60,
    )
    second_run = chunk_text(
        text,
        chapter_number=4,
        language_code="en",
        minimum_chunk_characters=30,
        maximum_chunk_characters=60,
    )

    assert [segment.segment_id for segment in first_run] == [
        segment.segment_id for segment in second_run
    ]


def test_unsupported_language_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported target language code"):
        chunk_text(
            "Hello world.",
            chapter_number=1,
            language_code="xx",
            minimum_chunk_characters=50,
            maximum_chunk_characters=100,
        )

