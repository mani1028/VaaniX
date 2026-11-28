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


def detect_clipping(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at (or above) the clipping threshold.

    A high fraction means the recording is distorted at the source —
    no amount of decoding tuning fixes it, and the clip should be
    flagged in error analysis (condition: clipped) instead of blamed
    on the model. Returns 0.0 for empty audio.
    """
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading/trailing silence using the energy VAD spans.

    Interior pauses are kept: only the dead air before the first and
    after the last speech span is removed. Returns the input unchanged
    when no speech span is found.
    """
    spans = energy_vad(audio, sample_rate)
    if not spans:
        return audio
    return audio[spans[0].start_sample : spans[-1].end_sample]


def prepare_utterance(
    path: Path,
    target_rate: int = TARGET_SAMPLE_RATE_HZ,
    trim_silence_: bool = False,
    peak: float = PEAK_TARGET,
) -> tuple[np.ndarray, list[SpeechSpan]]:
    """Resample to 16 kHz, peak-normalise, return (audio, VAD spans).

    Defaults are unchanged from before this change, so existing bench
    numbers stay comparable. Pass ``trim_silence_=True`` only when the
    bench run records it — trimming changes RTF and can change WER, so
    it must never be silent. Clipping is measured by callers via
    :func:`detect_clipping` on the raw audio, before normalisation.
    """
    audio, source_rate = read_wav(path)
    audio = resample(audio, source_rate, target_rate)
    if trim_silence_:
        audio = trim_silence(audio, target_rate)
    audio = peak_normalize(audio, peak=peak)
    return audio, energy_vad(audio, target_rate)


def duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return float(audio.shape[0]) / float(sample_rate)


def write_prepared(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE_HZ) -> None:
    write_wav(path, audio, sample_rate)
