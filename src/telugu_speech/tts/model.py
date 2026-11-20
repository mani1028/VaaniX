"""Compact TTS package bounds. No weights are trained in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactTtsSpec:
    name: str = "TeluguTinyTTS"
    acoustic: str = "fastpitch"
    vocoder: str = "hifigan"
    speakers: int = 1
    package_bytes_min: int = 50_000_000
    package_bytes_max: int = 150_000_000
