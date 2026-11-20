"""Orthographic akshara segmentation used until a phonetic G2P exists."""

from __future__ import annotations

import re

from telugu_speech.text.unicode import VIRAMA, to_nfc

_AKSHARA = re.compile(
    r"(?:[\u0c15-\u0c39](?:" + VIRAMA + r"[\u0c15-\u0c39])*[\u0c3e-\u0c4c\u0c56\u0c62\u0c63]?|"
    r"[\u0c05-\u0c14])"
)


def graphemes_to_aksharas(text: str) -> list[str]:
    """Split Telugu letters into orthographic syllables. Non-Telugu runs stay intact."""
    normalized = to_nfc(text)
    units: list[str] = []
    index = 0
    for match in _AKSHARA.finditer(normalized):
        if match.start() > index:
            gap = normalized[index:match.start()]
            if gap.strip():
                units.append(gap.strip())
        units.append(match.group(0))
        index = match.end()
    tail = normalized[index:].strip()
    if tail:
        units.append(tail)
    return units
