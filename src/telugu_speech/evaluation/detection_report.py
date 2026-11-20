"""Detection gate status. Metrics are written only when lid_label rows exist."""

from __future__ import annotations

import json
from pathlib import Path


def write_detection_report(manifest: Path, report_path: Path) -> str:
    labeled = 0
    total = 0
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if isinstance(row, dict) and row.get("lid_label"):
                labeled += 1
    if labeled == 0:
        text = (
            "# Telugu detection benchmark\n\n"
            "Detection gate: not evaluable.\n\n"
            f"Manifest `{manifest}` has {total} rows and {labeled} rows with `lid_label`.\n"
            "Accuracy, macro F1, and code-switch boundary F1 are not measured.\n"
            "`te-en` audio detection is unsupported until a segment model and labeled spans exist.\n"
        )
    else:
        text = (
            "# Telugu detection benchmark\n\n"
            f"Labeled rows: {labeled} of {total}.\n\n"
            "A scorer is not wired until the segment model exists. "
            "Do not treat row counts as accuracy.\n"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text
