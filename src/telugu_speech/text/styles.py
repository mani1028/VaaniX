"""STT output styles. These post-process a hypothesis; they do not invent words."""

from __future__ import annotations

import re

from telugu_speech.text.normalization import NormalizationConfig, normalize_text

STYLES = ("telugu", "codemix", "verbatim")


def apply_output_style(hypothesis: str, style: str) -> tuple[str, str, str]:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    raw_text, norm_text = normalize_text(hypothesis, NormalizationConfig())
    if style == "verbatim":
        text = re.sub(r"\s+", " ", hypothesis).strip()
    elif style == "codemix":
        text = raw_text
    else:
        text = norm_text
    return text, raw_text, norm_text
