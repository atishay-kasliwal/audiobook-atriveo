from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import warnings
import wave

import numpy as np

from .config import SUPPORTED_LANGUAGE_LABELS, ensure_supported_target_language


@dataclass(frozen=True)
class VoiceReferenceValidation:
    path: Path
    sample_rate: int | None
    channels: int | None
    duration_seconds: float
    clipped_ratio: float
    warnings: tuple[str, ...]


class ChatterboxEngine:
    def __init__(
        self,
        *,
        device: str | None = None,
        t3_model: str = "v3",
        allow_cpu_fallback: bool = False,
        model_cls: object | None = None,
    ) -> None:
        self.allow_cpu_fallback = allow_cpu_fallback
        self.device = device or self.resolve_device(allow_cpu_fallback=allow_cpu_fallback)
        self.t3_model = t3_model
        self._model_cls = model_cls
        self._model = None

    @staticmethod
    def resolve_device(*, allow_cpu_fallback: bool) -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if allow_cpu_fallback:
            warnings.warn(
                "CUDA is unavailable. Falling back to CPU generation, which will be much slower.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "cpu"
        raise RuntimeError(
            "CUDA is unavailable. Select a GPU runtime in Colab or enable "
            "allow_cpu_fallback explicitly."
        )

    @staticmethod
    def supported_languages() -> dict[str, str]:
        return dict(SUPPORTED_LANGUAGE_LABELS)

    @property
    def sample_rate(self) -> int | None:
        if self._model is None:
            return None
        return int(self._model.sr)

    def _get_model_class(self):
        if self._model_cls is not None:
            return self._model_cls
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS

    def load_model(self):
        if self._model is None:
            model_cls = self._get_model_class()
            self._model = model_cls.from_pretrained(device=self.device, t3_model=self.t3_model)
        return self._model

    def validate_language(self, language_code: str) -> str:
        return ensure_supported_target_language(language_code)

    def _load_voice_audio(self, voice_path: Path) -> tuple[np.ndarray, int | None, int | None]:
        if voice_path.suffix.lower() == ".wav":
            with wave.open(str(voice_path), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                frame_count = handle.getnframes()
                frames = handle.readframes(frame_count)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio, sample_rate, channels

        try:
            import librosa
        except ImportError as exc:  # pragma: no cover - exercised in Colab
            raise RuntimeError(
                "Non-WAV voice validation requires librosa. Install Colab dependencies first."
            ) from exc

        audio, sample_rate = librosa.load(str(voice_path), sr=None, mono=True)
        return np.asarray(audio, dtype=np.float32), int(sample_rate), 1

    def validate_voice_reference(self, voice_path: str | Path) -> VoiceReferenceValidation:
        path = Path(voice_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Voice reference file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Voice reference path must be a file: {path}")

        supported_suffixes = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}
        if path.suffix.lower() not in supported_suffixes:
            raise ValueError(
                f"Unsupported voice reference format '{path.suffix}'. "
                "Supported formats: .wav, .flac, .ogg, .mp3, .m4a, .aac"
            )

        audio, sample_rate, channels = self._load_voice_audio(path)
        duration_seconds = len(audio) / sample_rate if sample_rate else 0.0
        if duration_seconds < 3.0:
            raise ValueError("Voice reference must be at least 3 seconds long.")
        if duration_seconds > 60.0:
            raise ValueError("Voice reference must be 60 seconds or shorter.")
        if not np.any(np.abs(audio) > 1e-4):
            raise ValueError("Voice reference appears to contain no usable audio.")

        clipped_ratio = float(np.mean(np.abs(audio) >= 0.999))
        warning_messages: list[str] = []
        if duration_seconds < 10.0 or duration_seconds > 20.0:
            warning_messages.append(
                "A clean 10 to 20 second reference clip usually produces the best results."
            )
        if clipped_ratio > 0.01:
            warning_messages.append(
                "The reference clip appears heavily clipped. A cleaner recording is recommended."
            )
        return VoiceReferenceValidation(
            path=path,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            clipped_ratio=clipped_ratio,
            warnings=tuple(warning_messages),
        )

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
    ) -> np.ndarray:
        normalized_language = self.validate_language(language_code)
        model = self.load_model()

        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        kwargs = {
            "language_id": normalized_language,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        }
        if audio_prompt_path:
            kwargs["audio_prompt_path"] = str(Path(audio_prompt_path).expanduser())

        try:
            waveform = model.generate(text, **kwargs)
            if hasattr(waveform, "detach"):
                waveform = waveform.detach()
            if hasattr(waveform, "cpu"):
                waveform = waveform.cpu()
            audio = np.asarray(waveform, dtype=np.float32).squeeze()
            if audio.ndim != 1:
                audio = audio.reshape(-1)
            return audio
        finally:
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

