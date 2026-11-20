"""PCM loading and resampling without a required native audio library."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 2:
            array = array.mean(axis=1)
        return np.ascontiguousarray(array, dtype=np.float32), int(sample_rate)
    except ImportError:
        return _read_pcm_wav(path)


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError("only 8-bit and 16-bit PCM WAV files are supported")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def tighten_pauses(audio: np.ndarray, sample_rate: int, max_pause_s: float = 0.16) -> np.ndarray:
    """Shorten dead-air gaps. Soft speech and pauses under about 0.28 s stay."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0 or sample_rate <= 0:
        return audio
    frame = max(1, int(0.02 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    if audio.size < frame * 4:
        return audio
    windows = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
    rms = np.sqrt(np.mean(windows.astype(np.float64) * windows, axis=1))
    silent = np.zeros(audio.size, dtype=bool)
    starts = np.flatnonzero(rms < 0.008) * hop
    if starts.size:
        silent[(starts[:, None] + np.arange(frame)).ravel()] = True
    max_keep = int(max_pause_s * sample_rate)
    edge_keep = int(0.04 * sample_rate)
    pieces: list[np.ndarray] = []
    index = 0
    while index < audio.size:
        quiet = bool(silent[index])
        end = index + 1
        while end < audio.size and bool(silent[end]) == quiet:
            end += 1
        chunk = audio[index:end]
        if quiet and chunk.size > max(max_keep, int(0.28 * sample_rate)):
            if index == 0:
                chunk = chunk[-edge_keep:]
            elif end == audio.size:
                chunk = chunk[:edge_keep]
            else:
                head = max_keep // 2
                chunk = np.concatenate([chunk[:head], chunk[-(max_keep - head) :]])
        pieces.append(chunk)
        index = end
    if not pieces:
        return audio
    return np.concatenate(pieces).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0.0, audio.shape[0] - 1, num=target_length, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)
