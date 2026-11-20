"""TTS training is deferred until a consented single-speaker corpus exists."""

from __future__ import annotations

from pathlib import Path

import yaml

from telugu_speech.asr.train import TrainingDeferred


def load_tts_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TTS config must be a mapping")
    if payload.get("consent_required") is False:
        raise ValueError("TTS training requires consent_required: true")
    return payload


def train(config_path: Path) -> None:
    load_tts_config(config_path)
    raise TrainingDeferred("Phase 1 does not train a TTS model.")
