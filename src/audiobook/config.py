from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

SUPPORTED_LANGUAGE_LABELS: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}
SUPPORTED_LANGUAGE_CODES = tuple(sorted(SUPPORTED_LANGUAGE_LABELS.keys()))


def ensure_supported_target_language(language_code: str) -> str:
    normalized = language_code.strip().lower()
    if normalized not in SUPPORTED_LANGUAGE_LABELS:
        supported = ", ".join(SUPPORTED_LANGUAGE_CODES)
        raise ValueError(
            f"Unsupported target language code '{language_code}'. "
            f"Supported values: {supported}"
        )
    return normalized


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not sanitized:
        raise ValueError("Identifiers must contain at least one letter or number.")
    return sanitized


@dataclass(frozen=True)
class GitSyncSettings:
    repository_url: str
    branch_name: str = "main"
    repository_directory_name: str = "book-to-audio-workspace"
    private_repository: bool = False

    def __post_init__(self) -> None:
        if not self.repository_url.strip():
            raise ValueError("repository_url is required.")
        if not self.branch_name.strip():
            raise ValueError("branch_name is required.")
        if not self.repository_directory_name.strip():
            raise ValueError("repository_directory_name is required.")


@dataclass(frozen=True)
class ChapterSettings:
    book_id: str
    chapter_number: int
    chapter_file: Path
    source_language: str
    target_language: str

    def __post_init__(self) -> None:
        if self.chapter_number < 1:
            raise ValueError("chapter_number must be 1 or greater.")
        if not str(self.chapter_file).strip():
            raise ValueError("chapter_file is required.")
        ensure_supported_target_language(self.target_language)


@dataclass(frozen=True)
class GenerationSettings:
    voice_file: Path | None = None
    minimum_chunk_characters: int = 200
    maximum_chunk_characters: int = 500
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    maximum_retries: int = 3
    force_regeneration: bool = False
    allow_cpu_fallback: bool = False
    partial_export: bool = False
    smoke_test_text: str = "This is a quick smoke test for the audiobook pipeline."

    def __post_init__(self) -> None:
        if self.minimum_chunk_characters < 1:
            raise ValueError("minimum_chunk_characters must be at least 1.")
        if self.maximum_chunk_characters < self.minimum_chunk_characters:
            raise ValueError(
                "maximum_chunk_characters must be greater than or equal to "
                "minimum_chunk_characters."
            )
        if not 0.25 <= self.exaggeration <= 2.0:
            raise ValueError("exaggeration must be between 0.25 and 2.0.")
        if not 0.0 <= self.cfg_weight <= 1.5:
            raise ValueError("cfg_weight must be between 0.0 and 1.5.")
        if not 0.05 <= self.temperature <= 5.0:
            raise ValueError("temperature must be between 0.05 and 5.0.")
        if self.maximum_retries < 1:
            raise ValueError("maximum_retries must be at least 1.")


@dataclass(frozen=True)
class SilenceSettings:
    between_segments_ms: int = 250
    between_paragraphs_ms: int = 500
    chapter_prefix_ms: int = 400
    chapter_suffix_ms: int = 800

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if value < 0:
                raise ValueError(f"{field_name} must be 0 or greater.")


@dataclass(frozen=True)
class ExportSettings:
    export_mp3: bool = True
    export_m4b: bool = False
    mp3_bitrate: str = "192k"
    m4b_bitrate: str = "96k"
    normalize_peak: float = 0.98

    def __post_init__(self) -> None:
        if not 0.1 <= self.normalize_peak <= 1.0:
            raise ValueError("normalize_peak must be between 0.1 and 1.0.")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    books_dir: Path
    voices_dir: Path
    segments_dir: Path
    chapters_dir: Path
    manifests_dir: Path
    logs_dir: Path

    def ensure_directories(self) -> None:
        for path in (
            self.root,
            self.books_dir,
            self.voices_dir,
            self.segments_dir,
            self.chapters_dir,
            self.manifests_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "books_dir": str(self.books_dir),
            "voices_dir": str(self.voices_dir),
            "segments_dir": str(self.segments_dir),
            "chapters_dir": str(self.chapters_dir),
            "manifests_dir": str(self.manifests_dir),
            "logs_dir": str(self.logs_dir),
        }


@dataclass(frozen=True)
class AudiobookJobConfig:
    git: GitSyncSettings
    chapter: ChapterSettings
    generation: GenerationSettings
    silence: SilenceSettings
    export: ExportSettings
    paths: ProjectPaths
    t3_model: str = "v3"

    def to_manifest_config(self) -> dict[str, object]:
        return {
            "git": asdict(self.git),
            "chapter": {
                "book_id": self.chapter.book_id,
                "chapter_number": self.chapter.chapter_number,
                "chapter_file": str(self.chapter.chapter_file),
                "source_language": self.chapter.source_language,
                "target_language": self.chapter.target_language,
            },
            "generation": {
                "voice_file": str(self.generation.voice_file)
                if self.generation.voice_file
                else None,
                "minimum_chunk_characters": self.generation.minimum_chunk_characters,
                "maximum_chunk_characters": self.generation.maximum_chunk_characters,
                "exaggeration": self.generation.exaggeration,
                "cfg_weight": self.generation.cfg_weight,
                "temperature": self.generation.temperature,
                "maximum_retries": self.generation.maximum_retries,
                "force_regeneration": self.generation.force_regeneration,
                "allow_cpu_fallback": self.generation.allow_cpu_fallback,
                "partial_export": self.generation.partial_export,
            },
            "silence": asdict(self.silence),
            "export": asdict(self.export),
            "paths": self.paths.to_dict(),
            "t3_model": self.t3_model,
        }


def build_project_paths(root_directory: str | Path) -> ProjectPaths:
    root = Path(root_directory).expanduser().resolve()
    return ProjectPaths(
        root=root,
        books_dir=root / "books",
        voices_dir=root / "voices",
        segments_dir=root / "segments",
        chapters_dir=root / "chapters",
        manifests_dir=root / "manifests",
        logs_dir=root / "logs",
    )


def build_job_config(
    *,
    repository_url: str,
    branch_name: str,
    repository_directory_name: str,
    private_repository: bool,
    book_id: str,
    chapter_number: int,
    chapter_file: str | Path,
    source_language: str,
    target_language: str,
    voice_file: str | Path | None,
    output_directory: str | Path,
    minimum_chunk_characters: int,
    maximum_chunk_characters: int,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    maximum_retries: int,
    force_regeneration: bool,
    allow_cpu_fallback: bool,
    partial_export: bool,
    export_mp3: bool,
    export_m4b: bool,
    t3_model: str = "v3",
) -> AudiobookJobConfig:
    paths = build_project_paths(output_directory)
    paths.ensure_directories()
    return AudiobookJobConfig(
        git=GitSyncSettings(
            repository_url=repository_url,
            branch_name=branch_name,
            repository_directory_name=repository_directory_name,
            private_repository=private_repository,
        ),
        chapter=ChapterSettings(
            book_id=sanitize_identifier(book_id),
            chapter_number=chapter_number,
            chapter_file=Path(chapter_file),
            source_language=source_language.strip(),
            target_language=ensure_supported_target_language(target_language),
        ),
        generation=GenerationSettings(
            voice_file=Path(voice_file).expanduser() if voice_file else None,
            minimum_chunk_characters=minimum_chunk_characters,
            maximum_chunk_characters=maximum_chunk_characters,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            maximum_retries=maximum_retries,
            force_regeneration=force_regeneration,
            allow_cpu_fallback=allow_cpu_fallback,
            partial_export=partial_export,
        ),
        silence=SilenceSettings(),
        export=ExportSettings(export_mp3=export_mp3, export_m4b=export_m4b),
        paths=paths,
        t3_model=t3_model,
    )
