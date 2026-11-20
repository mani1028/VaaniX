"""Reproducible scorecard. Missing models stay not measured."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50_s": percentile(values, 50),
        "p95_s": percentile(values, 95),
        "p99_s": percentile(values, 99),
    }


def directory_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and ".cache" not in item.parts:
            total += item.stat().st_size
    return total


def silence_stats(samples: list[float], sample_rate: int, threshold: float = 0.008) -> dict[str, float | int | None]:
    if not samples or sample_rate <= 0:
        return {"silence_fraction": None, "longest_silence_s": None}
    longest = 0
    run = 0
    silent = 0
    for sample in samples:
        if abs(sample) < threshold:
            silent += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return {
        "silence_fraction": silent / len(samples),
        "longest_silence_s": longest / sample_rate,
    }


def model_catalog(root: Path) -> list[dict[str, object]]:
    """Same-hardware catalog. A path that is absent is not a score."""
    rows = [
        ("stt", "telugu-asr-micro", root / "models/telugu-asr-micro", "Our CPU recognizer. Same weights as vasista22/whisper-telugu-tiny, int8, faster-whisper."),
        ("stt", "telugu-asr-small", root / "models/telugu-asr", "Optional larger Telugu Whisper. Scored only when the folder is present."),
        ("stt", "whisper.cpp", root / "models/whisper-cpp-telugu", "Not bundled. A missing folder is not a result."),
        ("stt", "sherpa-onnx", root / "models/sherpa-onnx-telugu", "Not bundled. A missing folder is not a result."),
        ("tts", "hima-int8", root / "lite/models/hima-tts-nano-te", "Our voices Srinivas and Sravani. This is the Hima Nano int8 pack, not a second engine."),
        ("tts", "chatterbox", root / "models/chatterbox-telugu", "Not bundled. Not scored."),
        ("tts", "laya", root / "models/laya-telugu", "Not bundled. Not scored."),
        ("tts", "indic-tiny", root / "models/tts/indic-tiny-tts-telugu", "Removed from the product. Not scored."),
    ]
    catalog = []
    for role, name, path, note in rows:
        present = path.is_dir() and any(path.rglob("*"))
        catalog.append(
            {
                "role": role,
                "model": name,
                "path": str(path),
                "status": "present" if present else "not_installed",
                "bytes": directory_bytes(path) if present else None,
                "note": note,
            }
        )
    return catalog


def write_report(directory: Path, report: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_csv(directory / "models.csv", list(report.get("models") or []))
    _write_csv(directory / "stt.csv", list(report.get("stt") or []))
    _write_csv(directory / "tts.csv", list(report.get("tts") or []))
    _write_csv(directory / "concurrency.csv", list(report.get("concurrency") or []))
    _write_csv(directory / "competitors.csv", list(report.get("competitors") or []))
    _write_csv(directory / "dataset.csv", list(report.get("dataset") or []))
    (directory / "report.html").write_text(render_html(report), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in (fields or ["empty"])})


def render_html(report: dict[str, object]) -> str:
    hardware = report.get("hardware") or {}
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>Telugu speech benchmark</title>",
        "<style>body{font-family:Georgia,serif;max-width:52rem;margin:2rem auto;color:#1c1917;line-height:1.45}",
        "table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border-bottom:1px solid #e7e5e4;padding:0.35rem;text-align:left}",
        ".bar{height:0.7rem;background:#7c3aed;border-radius:4px}</style></head><body>",
        "<h1>Telugu speech benchmark</h1>",
        "<p>Measured on this machine only. A blank cell was not run. MOS was not collected.</p>",
        f"<p>Host: {_esc(hardware.get('platform'))}. CPUs: {_esc(hardware.get('cpu_count'))}. "
        f"Python: {_esc(hardware.get('python'))}. Threads under test: {_esc(hardware.get('threads'))}.</p>",
        "<h2>Models</h2>",
        _table(list(report.get("models") or []), ("role", "model", "status", "bytes", "note")),
        "<h2>Speech recognition</h2>",
        _table(list(report.get("stt") or []), ("model", "clips", "wer", "cer", "rtf", "load_s", "peak_rss_mb")),
        _bars(list(report.get("stt") or []), "model", "rtf", "Real-time factor, lower is faster"),
        "<h2>Speech generation</h2>",
        _table(list(report.get("tts") or []), ("voice", "threads", "rtf", "first_audio_s", "audio_s", "peak_rss_mb", "silence_fraction")),
        "<h2>Concurrent calls</h2>",
        _table(list(report.get("concurrency") or []), ("kind", "workers", "p50_s", "p95_s", "p99_s", "errors", "wall_s")),
        "<h2>Competitors</h2>",
        "<p>A row marked not_installed was not scored. Production voices and the recognizer are unchanged.</p>",
        _table(list(report.get("competitors") or []), ("role", "model", "status", "license", "telugu")),
        "<h2>Test set still to collect</h2>",
        _table(list(report.get("dataset") or []), ("category", "clips", "target")),
        "<h2>Release check</h2>",
        f"<p>{_esc((report.get('decision') or {}).get('reason'))}</p>",
        "<h2>Not a ranking against other products</h2>",
        "<p>Listener scores stay blank until native speakers fill configs/mos_listeners.csv. "
        "An empty average is not a MOS.</p>",
        "</body></html>",
    ]
    return "\n".join(lines)


def _bars(rows: list[dict[str, object]], label: str, field: str, title: str) -> str:
    usable = [(row.get(label), row.get(field)) for row in rows if isinstance(row.get(field), (int, float))]
    if not usable:
        return ""
    peak = max(float(value) for _name, value in usable) or 1.0
    parts = [f"<h3>{_esc(title)}</h3>"]
    for name, value in usable:
        width = max(2, int(100 * float(value) / peak))
        parts.append(
            f"<p>{_esc(name)} {_esc(round(float(value), 4))}<br>"
            f"<span class=\"bar\" style=\"width:{width}%\"></span></p>"
        )
    return "\n".join(parts)


def _table(rows: list[dict[str, object]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    head = "".join(f"<th>{_esc(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_esc(row.get(column))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
