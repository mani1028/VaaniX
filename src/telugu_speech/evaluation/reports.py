"""Markdown reports grouped by region, condition, and code-mix."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from telugu_speech.evaluation.asr_metrics import ErrorCounts


def _mean(counts: list[ErrorCounts]) -> float | None:
    if not counts:
        return None
    total_errors = sum(item.substitutions + item.deletions + item.insertions for item in counts)
    total_ref = sum(item.reference_length for item in counts)
    if total_ref == 0:
        return 0.0
    return total_errors / total_ref


def render_report(model_results: list[dict[str, object]]) -> str:
    lines = [
        "# Telugu Speech Benchmark",
        "",
        "Slice rates are micro-averaged over reference tokens in that slice.",
        "External published numbers are not included here; see docs/benchmark_results.md.",
        "",
    ]
    for result in model_results:
        lines.append(f"## {result['model_id']}")
        lines.append("")
        if result.get("error"):
            status = str(result["error"])
            if status.startswith("not measured"):
                lines.append(f"Status: {status}")
            else:
                lines.append(f"Status: unavailable. {status}")
            lines.append("")
            continue
        lines.append(f"- Utterances: {result['utterances']}")
        lines.append(f"- Real-time factor: {_format(result.get('rtf'))}")
        lines.append(f"- Peak RSS bytes: {result.get('peak_rss_bytes')}")
        lines.append(f"- Model bytes: {result.get('model_bytes')}")
        lines.append("")
        lines.append("| Slice | Kind | WER | CER |")
        lines.append("| --- | --- | ---: | ---: |")
        for slice_row in result["slices"]:  # type: ignore[index]
            lines.append(
                f"| {slice_row['name']} | {slice_row['kind']} | "
                f"{_format(slice_row['wer'])} | {_format(slice_row['cer'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def group_slices(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[("region", str(row["region"]))].append(row)
        groups[("condition", str(row["condition"]))].append(row)
        groups[("code_mix", "yes" if row["code_mix"] else "no")].append(row)
        groups[("overall", "all")].append(row)
    rendered: list[dict[str, object]] = []
    for (kind, name), members in groups.items():
        rendered.append(
            {
                "kind": kind,
                "name": name,
                "wer": _mean([member["wer"] for member in members]),  # type: ignore[misc]
                "cer": _mean([member["cer"] for member in members]),  # type: ignore[misc]
            }
        )
    return rendered


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
