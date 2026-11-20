"""TTS quantization is separate from the vocoder and is not run in Phase 1."""

from __future__ import annotations

from telugu_speech.tts.model import CompactTtsSpec


def quantize_plan(spec: CompactTtsSpec) -> dict[str, object]:
    if spec.package_bytes_min >= spec.package_bytes_max:
        raise ValueError("package size bounds are inverted")
    return {
        "model": spec.name,
        "acoustic": spec.acoustic,
        "vocoder": spec.vocoder,
        "quantize_vocoder_separately": True,
        "status": "planned",
    }
