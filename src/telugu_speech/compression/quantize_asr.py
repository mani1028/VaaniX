"""Compression helpers record intent and refuse missing checkpoints."""

from __future__ import annotations

from pathlib import Path

from telugu_speech.asr.model import CompactConformerSpec


def estimate_int8_bytes(parameter_count: int) -> int:
    return parameter_count


def quantize_plan(spec: CompactConformerSpec) -> dict[str, int | str]:
    parameters = spec.estimated_parameters()
    return {
        "model": spec.name,
        "parameters": parameters,
        "int8_weight_bytes": estimate_int8_bytes(parameters),
        "status": "planned",
    }


def require_checkpoint(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path
