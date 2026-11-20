"""Train-only waveform augmentations. Do not apply these to evaluation audio."""

from __future__ import annotations

import numpy as np

from telugu_speech.audio.resampling import resample


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.square(audio)) + 1e-12)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return audio + noise


def speed_perturb(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("speed factor must be positive")
    changed = resample(audio, sample_rate, max(1, int(round(sample_rate * factor))))
    return resample(changed, max(1, int(round(sample_rate * factor))), sample_rate)


def telephone_bandwidth(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Approximate an 8 kHz telephone path and return to the original rate."""
    narrow = resample(audio, sample_rate, 8000)
    return resample(narrow, 8000, sample_rate)
