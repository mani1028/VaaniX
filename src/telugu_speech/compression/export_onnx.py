"""ONNX export is refused until a real checkpoint and torch are present."""

from __future__ import annotations

from pathlib import Path

from telugu_speech.compression.quantize_asr import require_checkpoint


def export_onnx(checkpoint: Path, destination: Path) -> Path:
    require_checkpoint(checkpoint)
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError("ONNX export requires torch") from exc
    raise RuntimeError(
        f"ONNX export for {checkpoint.name} is not implemented until the student checkpoint exists. "
        f"Destination would be {destination}."
    )
