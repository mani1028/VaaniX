"""Test-set counts and the rule that keeps the current engine until a fair rerun."""

from __future__ import annotations

import json
from pathlib import Path


CATEGORIES = (
    "clean_conversational",
    "telangana",
    "andhra",
    "code_mix",
    "noisy",
    "speakers_and_speeds",
)
TARGET_PER_CATEGORY = 100


def dataset_counts(manifest: Path) -> list[dict[str, object]]:
    counts = {category: 0 for category in CATEGORIES}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            category = str(row.get("category", ""))
            if category in counts:
                counts[category] += 1
            if row.get("code_mix") is True and category != "code_mix":
                counts["code_mix"] += 1
            if row.get("noisy") is True and category != "noisy":
                counts["noisy"] += 1
    return [
        {"category": category, "clips": counts[category], "target": TARGET_PER_CATEGORY}
        for category in CATEGORIES
    ]


def release_decision(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    """Recommend only. This never changes the production model files."""
    if baseline.get("status") != "measured" or candidate.get("status") != "measured":
        return {
            "replace_production": False,
            "quality": "not_compared",
            "deployment": "not_compared",
            "reason": "Both models must be measured on the same clips and the same machine.",
        }
    if baseline.get("clips") != candidate.get("clips") or not baseline.get("clips"):
        return {
            "replace_production": False,
            "quality": "not_compared",
            "deployment": "not_compared",
            "reason": "The clip counts differ, so this is not the same test.",
        }
    base_wer = _number(baseline.get("wer"))
    cand_wer = _number(candidate.get("wer"))
    base_rtf = _number(baseline.get("rtf"))
    cand_rtf = _number(candidate.get("rtf"))
    base_ram = _number(baseline.get("peak_rss_mb"))
    cand_ram = _number(candidate.get("peak_rss_mb"))
    if None in (base_wer, cand_wer, base_rtf, cand_rtf, base_ram, cand_ram):
        return {
            "replace_production": False,
            "quality": "not_compared",
            "deployment": "not_compared",
            "reason": "WER, real-time factor, and memory are all required.",
        }
    quality = "candidate" if cand_wer <= base_wer - 0.02 else "baseline"
    deployment = "candidate" if cand_rtf <= base_rtf and cand_ram <= base_ram * 1.1 else "baseline"
    return {
        "replace_production": False,
        "quality": quality,
        "deployment": deployment,
        "reason": "A category lead is recorded. Production files stay unchanged until a separate decision.",
    }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
