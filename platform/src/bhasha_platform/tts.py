"""TTS facade: compact MMS Telugu or IndicF5 when available."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from telugu_speech.asr.infer import ModelUnavailable
from telugu_speech.tts.infer import CompactMmsTtsBackend, IndicF5Backend, MmsTtsBackend


def resolve_tts_dir() -> Path:
    for path in (
        REPO / "lite" / "models" / "telugu-tts-micro",
        REPO / "models" / "telugu-tts-micro",
        REPO / "models" / "telugu-tts",
    ):
        if path.is_dir():
            return path
    raise ModelUnavailable("No local Telugu TTS folder found")


def synthesize(
    text: str,
    output: Path,
    backend: str = "telugu_micro",
    reference_audio: Path | None = None,
    reference_text: str = "",
) -> dict:
    if backend == "indicf5":
        engine = IndicF5Backend()
        written = engine.synthesize(text, reference_audio, reference_text, output)
        return {
            "audio_path": str(written),
            "backend": engine.name,
            "sample_rate_hz": engine.sample_rate_hz,
            "license": "MIT (IndicF5) + consent required for reference voice",
        }
    path = resolve_tts_dir()
    if (path / "weights.int8.npz").is_file():
        engine = CompactMmsTtsBackend(str(path))
    else:
        engine = MmsTtsBackend(str(path))
    written = engine.synthesize(text, None, "", output)
    return {
        "audio_path": str(written),
        "backend": engine.name,
        "sample_rate_hz": engine.sample_rate_hz,
        "license": "CC-BY-NC-4.0 until replaced by IndicF5 or consented MIT/Apache voice",
        "model_path": str(path),
    }
