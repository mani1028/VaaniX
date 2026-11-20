"""Vocoder selection is recorded here and executed only after acoustic training."""

from __future__ import annotations

VOCODERS = ("hifi_gan", "vocos")


def select_vocoder(name: str) -> str:
    if name not in VOCODERS:
        raise ValueError(f"vocoder must be one of {VOCODERS}")
    return name
