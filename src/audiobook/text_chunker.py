from __future__ import annotations

from dataclasses import dataclass
import re

from .config import SUPPORTED_LANGUAGE_CODES, ensure_supported_target_language

_ABBREVIATIONS = (
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "No.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "U.S.",
    "U.K.",
)
_HARD_STOP_PUNCTUATION = ".!?…。！？"
_SOFT_BREAK_PATTERN = re.compile(r"(?<=[,;:،，、؛])\s+")
_SENTENCE_PATTERN = re.compile(r".+?(?:[.!?…。！？]+(?:['\"”’)\]]+)?|$)", re.S)


@dataclass(frozen=True)
class TextSegment:
    segment_id: str
    index: int
    paragraph_index: int
    text: str
    char_count: int
    starts_paragraph: bool
    ends_paragraph: bool


def stable_segment_id(chapter_number: int, segment_index: int) -> str:
    return f"chapter_{chapter_number:03d}_segment_{segment_index:04d}"


def normalize_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if not cleaned:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(cleaned)
        previous_blank = False
    normalized = "\n".join(normalized_lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _protect_abbreviations(text: str) -> str:
    protected = text
    for abbreviation in _ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", "<DOT>"))
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DECIMAL>", protected)
    return protected


def _restore_protected_tokens(text: str) -> str:
    return text.replace("<DOT>", ".").replace("<DECIMAL>", ".")


def split_into_paragraphs(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]


def split_into_sentences(paragraph: str) -> list[str]:
    protected = _protect_abbreviations(paragraph)
    sentences: list[str] = []
    for match in _SENTENCE_PATTERN.finditer(protected):
        candidate = _restore_protected_tokens(match.group(0)).strip()
        if candidate:
            sentences.append(candidate)
    return sentences or [paragraph.strip()]


def _split_dense_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    if len(words) == 1:
        return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        next_len = len(word) if not current else current_len + 1 + len(word)
        if current and next_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]

    soft_parts = [part.strip() for part in _SOFT_BREAK_PATTERN.split(sentence) if part.strip()]
    if len(soft_parts) == 1:
        return _split_dense_text(sentence, max_chars)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in soft_parts:
        next_len = len(part) if not current else current_len + 1 + len(part)
        if current and next_len > max_chars:
            chunks.append(" ".join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len = next_len
    if current:
        chunks.append(" ".join(current))

    normalized_chunks: list[str] = []
    for chunk in chunks:
        normalized_chunks.extend(_split_dense_text(chunk, max_chars))
    return normalized_chunks


def chunk_text(
    text: str,
    *,
    chapter_number: int,
    language_code: str,
    minimum_chunk_characters: int = 200,
    maximum_chunk_characters: int = 500,
) -> list[TextSegment]:
    if minimum_chunk_characters < 1:
        raise ValueError("minimum_chunk_characters must be at least 1.")
    if maximum_chunk_characters < minimum_chunk_characters:
        raise ValueError(
            "maximum_chunk_characters must be greater than or equal to "
            "minimum_chunk_characters."
        )

    normalized_language = ensure_supported_target_language(language_code)
    if normalized_language not in SUPPORTED_LANGUAGE_CODES:
        raise ValueError(f"Unsupported language code: {language_code}")

    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    segments: list[TextSegment] = []
    segment_index = 1
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        units: list[str] = []
        for sentence in split_into_sentences(paragraph):
            units.extend(split_oversized_sentence(sentence, maximum_chunk_characters))

        chunk_texts: list[str] = []
        current_units: list[str] = []
        current_len = 0
        for unit in units:
            next_len = len(unit) if not current_units else current_len + 1 + len(unit)
            if current_units and next_len > maximum_chunk_characters:
                chunk_texts.append(" ".join(current_units).strip())
                current_units = [unit]
                current_len = len(unit)
                continue
            current_units.append(unit)
            current_len = next_len
        if current_units:
            chunk_texts.append(" ".join(current_units).strip())

        if len(chunk_texts) >= 2:
            merged_chunks: list[str] = [chunk_texts[0]]
            for chunk in chunk_texts[1:]:
                previous = merged_chunks[-1]
                if len(chunk) < minimum_chunk_characters and len(previous) + 1 + len(chunk) <= maximum_chunk_characters:
                    merged_chunks[-1] = f"{previous} {chunk}".strip()
                else:
                    merged_chunks.append(chunk)
            chunk_texts = merged_chunks

        for index_in_paragraph, chunk in enumerate(chunk_texts, start=1):
            segments.append(
                TextSegment(
                    segment_id=stable_segment_id(chapter_number, segment_index),
                    index=segment_index,
                    paragraph_index=paragraph_index,
                    text=chunk,
                    char_count=len(chunk),
                    starts_paragraph=index_in_paragraph == 1,
                    ends_paragraph=index_in_paragraph == len(chunk_texts),
                )
            )
            segment_index += 1

    return segments
