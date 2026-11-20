"""Utterance language identification from audio.

Telugu script and Latin spelling are not inputs. te-en is not predicted
until a segment model exists; code_mix_audio stays unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable

import yaml

from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.audio.preprocessing import prepare_utterance
from telugu_speech.audio.vad import energy_vad

ProbabilityFn = Callable[[Path], dict[str, float]]


@dataclass(frozen=True)
class DetectionConfig:
    sample_rate_hz: int
    min_speech_seconds: float
    te_confidence_min: float
    other_confidence_min: float
    backend: str
    model_path: str
    code_mix_audio: str


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float | None
    backend: str
    code_mix_audio: str
    speech_seconds: float
    closest: tuple[tuple[str, float], ...] = ()


def load_detection_config(path: Path) -> DetectionConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detection config must be a mapping")
    required = (
        "sample_rate_hz",
        "min_speech_seconds",
        "te_confidence_min",
        "other_confidence_min",
        "backend",
        "model_path",
        "code_mix_audio",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"detection config missing {missing}")
    if payload["code_mix_audio"] != "unsupported":
        raise ValueError("code_mix_audio must stay unsupported until a segment model exists")
    return DetectionConfig(
        sample_rate_hz=int(payload["sample_rate_hz"]),
        min_speech_seconds=float(payload["min_speech_seconds"]),
        te_confidence_min=float(payload["te_confidence_min"]),
        other_confidence_min=float(payload["other_confidence_min"]),
        backend=str(payload["backend"]),
        model_path=str(payload["model_path"]),
        code_mix_audio="unsupported",
    )


def speech_seconds(path: Path, sample_rate_hz: int) -> float:
    audio, _spans = prepare_utterance(path, sample_rate_hz)
    spans = energy_vad(audio, sample_rate_hz)
    samples = sum(span.end_sample - span.start_sample for span in spans)
    return samples / float(sample_rate_hz)


def _closest(probabilities: dict[str, float]) -> tuple[tuple[str, float], ...]:
    ranked = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)[:3]
    return tuple((str(name), float(score)) for name, score in ranked)


def classify_probabilities(
    probabilities: dict[str, float],
    spoken_seconds: float,
    config: DetectionConfig,
    backend_name: str,
) -> Detection:
    closest = _closest(probabilities)
    if spoken_seconds < config.min_speech_seconds:
        return Detection("uncertain", None, "energy-vad", "unsupported", spoken_seconds)
    if not probabilities:
        return Detection("uncertain", None, backend_name, "unsupported", spoken_seconds)
    best_language, best = closest[0]
    telugu = float(probabilities.get("te", 0.0))
    if best_language == "te" and telugu >= config.te_confidence_min:
        return Detection("te-IN", telugu, backend_name, "unsupported", spoken_seconds, closest)
    if best_language != "te" and best >= config.other_confidence_min:
        return Detection("other", best, backend_name, "unsupported", spoken_seconds, closest)
    return Detection("uncertain", best, backend_name, "unsupported", spoken_seconds, closest)


class FasterWhisperLanguageDetector:
    """Multilingual Whisper language probabilities. Not the Telugu ASR checkpoint."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.name = f"faster-whisper-lid:{model_path}"
        self._model = None

    def language_probabilities(self, audio_path: Path) -> dict[str, float]:
        if not Path(self.model_path).exists():
            raise ModelUnavailable(
                f"Multilingual LID weights are not present at {self.model_path}. "
                "Telugu ASR weights are not used for detection."
            )
        model = self._load()
        audio, _spans = prepare_utterance(audio_path, 16000)
        with _lid_lock(self.model_path):
            _language, _probability, all_probs = model.detect_language(audio, vad_filter=False)
        return {str(name): float(score) for name, score in all_probs}

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise ModelUnavailable("faster-whisper is not installed") from exc
        from telugu_speech.tts.runtime import shared_engine

        def build() -> object:
            from faster_whisper import WhisperModel

            return WhisperModel(
                self.model_path,
                device="cpu",
                compute_type="int8",
                cpu_threads=2,
                num_workers=1,
                local_files_only=True,
            )

        self._model = shared_engine(("lid-int8", self.model_path, 2), build)
        return self._model


_LID_LOCKS: dict[str, threading.Lock] = {}
_LID_LOCKS_GUARD = threading.Lock()


def _lid_lock(model_path: str) -> threading.Lock:
    with _LID_LOCKS_GUARD:
        lock = _LID_LOCKS.get(model_path)
        if lock is None:
            lock = threading.Lock()
            _LID_LOCKS[model_path] = lock
        return lock


def detect_path(
    path: Path,
    config: DetectionConfig,
    probability_fn: ProbabilityFn | None = None,
    backend_name: str | None = None,
) -> Detection:
    spoken = speech_seconds(path, config.sample_rate_hz)
    if spoken < config.min_speech_seconds:
        return Detection("uncertain", None, "energy-vad", "unsupported", spoken)
    scorer = probability_fn or FasterWhisperLanguageDetector(config.model_path).language_probabilities
    name = backend_name or config.backend
    probabilities = scorer(path)
    return classify_probabilities(probabilities, spoken, config, name)
