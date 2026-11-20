"""Shared audio preparation for manifests and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from telugu_speech.audio.resampling import read_wav, resample, write_wav
from telugu_speech.audio.vad import SpeechSpan, energy_vad

TARGET_SAMPLE_RATE_HZ = 16000
PEAK_TARGET = 0.95


def peak_normalize(audio: np.ndarray, peak: float = PEAK_TARGET) -> np.ndarray:
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum == 0.0:
        return audio.astype(np.float32, copy=False)
    return (audio * (peak / maximum)).astype(np.float32)


def prepare_utterance(path: Path, target_rate: int = TARGET_SAMPLE_RATE_HZ) -> tuple[np.ndarray, list[SpeechSpan]]:
    audio, source_rate = read_wav(path)
    audio = resample(audio, source_rate, target_rate)
    audio = peak_normalize(audio)
    return audio, energy_vad(audio, target_rate)


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(audio.shape[0]) / float(sample_rate)


def write_prepared(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE_HZ) -> None:
    write_wav(path, audio, sample_rate)
