"""ASR facade: prefer local micro Telugu, then lab backends."""

from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for candidate in (REPO / "src",):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from telugu_speech.asr.infer import FasterWhisperBackend, IndicConformerBackend, ModelUnavailable


def resolve_asr_path(preferred: str | None = None) -> str:
    if preferred and Path(preferred).is_dir():
        return preferred
    for path in (
        REPO / "lite" / "models" / "telugu-asr-micro",
        REPO / "models" / "telugu-asr-micro",
        REPO / "models" / "telugu-asr",
    ):
        if path.is_dir():
            return str(path)
    raise ModelUnavailable("No local Telugu ASR folder found under lite/models or models/")


def transcribe(audio: Path, backend: str = "telugu_micro", language: str = "te") -> dict:
    if language != "te" and backend.startswith("telugu"):
        raise ModelUnavailable(
            f"Language {language} needs IndicConformer multi or another multilingual open ASR."
        )
    if backend in {"indicconformer_te", "indicconformer"}:
        engine = IndicConformerBackend(
            "ai4bharat/indicconformer_stt_te_hybrid_ctc_rnnt_large",
            "ctc",
        )
        text = engine.transcribe(audio)
        return {"text": text, "backend": engine.name, "language": language}
    path = resolve_asr_path()
    engine = FasterWhisperBackend(path)
    text = engine.transcribe(audio)
    return {"text": text, "backend": engine.name, "language": language, "model_path": path}
