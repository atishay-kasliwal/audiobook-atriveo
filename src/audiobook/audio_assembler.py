from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np

from .config import ExportSettings, SilenceSettings
from .manifest import SegmentRecord


@dataclass(frozen=True)
class AssemblyResult:
    audio: np.ndarray
    sample_rate: int
    channels: int


def _to_2d(audio: np.ndarray) -> np.ndarray:
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 1:
        return normalized[:, None]
    if normalized.ndim == 2:
        return normalized
    raise ValueError("Audio must be one-dimensional mono data or a 2D array.")


def read_wav(audio_path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(audio_path)
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        frames = handle.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}.")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    else:
        audio = audio.reshape(-1, 1)
    return audio, sample_rate


def write_wav(audio_path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    path = Path(audio_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = np.clip(_to_2d(audio), -1.0, 1.0)
    pcm = (normalized * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def build_silence(duration_ms: int, sample_rate: int, channels: int) -> np.ndarray:
    samples = int(sample_rate * (duration_ms / 1000.0))
    return np.zeros((samples, channels), dtype=np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current_peak <= 0.0:
        return audio
    scale = min(target_peak / current_peak, 1.0 / current_peak)
    return (audio * scale).astype(np.float32)


def validate_rendered_audio(audio: np.ndarray) -> None:
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Rendered audio is empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Rendered audio contains non-finite values.")
    if float(np.max(np.abs(array))) < 1e-5:
        raise ValueError("Rendered audio is effectively silent.")


def assemble_chapter_audio(
    segment_paths: list[str | Path],
    segment_records: list[SegmentRecord],
    *,
    silence: SilenceSettings,
    export: ExportSettings,
) -> AssemblyResult:
    if len(segment_paths) != len(segment_records):
        raise ValueError("segment_paths and segment_records must be the same length.")
    if not segment_paths:
        raise ValueError("At least one segment is required to assemble chapter audio.")

    loaded_audio: list[np.ndarray] = []
    sample_rate: int | None = None
    channels: int | None = None
    for segment_path in segment_paths:
        audio, current_sample_rate = read_wav(segment_path)
        if sample_rate is None:
            sample_rate = current_sample_rate
            channels = audio.shape[1]
        elif current_sample_rate != sample_rate:
            raise ValueError("All segments must share the same sample rate.")
        elif audio.shape[1] != channels:
            raise ValueError("All segments must share the same channel layout.")
        loaded_audio.append(audio)

    assert sample_rate is not None
    assert channels is not None

    pieces: list[np.ndarray] = [build_silence(silence.chapter_prefix_ms, sample_rate, channels)]
    for index, (audio, record) in enumerate(zip(loaded_audio, segment_records)):
        pieces.append(audio)
        if index == len(loaded_audio) - 1:
            continue
        pause_ms = (
            silence.between_paragraphs_ms
            if record.ends_paragraph
            else silence.between_segments_ms
        )
        pieces.append(build_silence(pause_ms, sample_rate, channels))
    pieces.append(build_silence(silence.chapter_suffix_ms, sample_rate, channels))
    assembled = np.concatenate(pieces, axis=0)
    assembled = normalize_peak(assembled, export.normalize_peak)
    return AssemblyResult(audio=assembled, sample_rate=sample_rate, channels=channels)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def export_with_ffmpeg(
    source_wav: str | Path,
    output_path: str | Path,
    *,
    bitrate: str,
) -> Path:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on this system.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix == ".mp3":
        codec = "libmp3lame"
        extra_args: list[str] = []
    elif suffix == ".m4b":
        codec = "aac"
        extra_args = ["-movflags", "+faststart", "-f", "mp4"]
    else:
        raise ValueError(f"Unsupported ffmpeg export format: {destination.suffix}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_wav),
        "-vn",
        "-c:a",
        codec,
        "-b:a",
        bitrate,
        *extra_args,
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return destination
