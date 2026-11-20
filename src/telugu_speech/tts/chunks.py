"""Split long Telugu lines before a voice model sees them.

Hima's text embedding accepts 1,024 positions. A line with no sentence
break can be longer than that and the ONNX step then fails.
"""

from __future__ import annotations

from pathlib import Path
import re

SPEECH_PIECE_LIMIT = 180
_SENTENCE = re.compile(r"(?<=[.!?।\u0964])\s*")


def speech_pieces(text: str, limit: int = SPEECH_PIECE_LIMIT) -> tuple[str, ...]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ()
    parts = [part.strip() for part in _SENTENCE.split(cleaned) if part.strip()]
    pieces: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            pieces.append(current)
            current = ""

    for part in parts:
        if len(part) <= limit:
            if current and len(current) + 1 + len(part) <= limit:
                current = f"{current} {part}"
            else:
                flush()
                current = part
            continue
        flush()
        for word in part.split(" "):
            if len(word) > limit:
                flush()
                pieces.extend(word[index:index + limit] for index in range(0, len(word), limit))
                continue
            if current and len(current) + 1 + len(word) > limit:
                flush()
            current = f"{current} {word}".strip()
    flush()
    return tuple(pieces)


def write_chunked_speech(chunks: list[object], sample_rate_hz: int, output_path: Path) -> Path:
    import numpy as np

    from telugu_speech.audio.resampling import tighten_pauses, write_wav

    gap = np.zeros(int(0.12 * sample_rate_hz), np.float32)
    parts: list[np.ndarray] = []
    for chunk in chunks:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            continue
        if parts:
            parts.append(gap)
        parts.append(audio)
    combined = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 1e-4:
        combined = combined * (0.9 / peak)
    write_wav(output_path, tighten_pauses(combined, sample_rate_hz), sample_rate_hz)
    return output_path
