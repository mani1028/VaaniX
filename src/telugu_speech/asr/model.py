"""Compact student description. Weights are not trained in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompactConformerSpec:
    name: str = "TeluguTinyASR"
    blocks: int = 8
    width: int = 256
    features: int = 80
    sample_rate_hz: int = 16000
    decoder: str = "ctc"

    def estimated_parameters(self) -> int:
        # Order-of-magnitude encoder estimate used for config review, not a measured count.
        per_block = self.width * self.width * 8
        return self.blocks * per_block + self.features * self.width
