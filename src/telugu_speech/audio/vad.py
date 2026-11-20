"""Voice activity detection.

Energy VAD is the offline default so tests do not download Silero weights.
Silero is available when torch and the Silero package are installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechSpan:
    start_sample: int
    end_sample: int


def energy_vad(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    energy_threshold: float = 0.01,
) -> list[SpeechSpan]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    frame = max(1, int(sample_rate * frame_ms / 1000))
    spans: list[SpeechSpan] = []
    start: int | None = None
    for offset in range(0, audio.shape[0], frame):
        chunk = audio[offset:offset + frame]
        active = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12)) >= energy_threshold
        if active and start is None:
            start = offset
        elif not active and start is not None:
            spans.append(SpeechSpan(start, offset))
            start = None
    if start is not None:
        spans.append(SpeechSpan(start, audio.shape[0]))
    return spans


def silero_vad(audio: np.ndarray, sample_rate: int) -> list[SpeechSpan]:
    """Lazy Silero entry point. Raises if the optional stack is absent."""
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Silero VAD is not installed") from exc
    get_speech_timestamps = utils[0]
    tensor = torch.from_numpy(audio)
    stamps = get_speech_timestamps(tensor, model, sampling_rate=sample_rate)
    return [SpeechSpan(int(item["start"]), int(item["end"])) for item in stamps]
