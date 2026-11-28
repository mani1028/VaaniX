"""Real-world ASR bench: per-condition WER/CER/AER on local recordings.

This extends the frozen-set bench (``telugu_speech.bench``) instead of
replacing it. The frozen bench scores curated sets (FLEURS-te, Kathbath-te,
cv-te) behind an n>=200 / >=20-speaker gate; this module scores whatever
real recordings Mani has — his own voice, background noise, phone-quality
audio, code-mixed speech, fast speech, several speakers — with no gate
and no invented data. A small real-world set is reported as what it is:
per-condition numbers with n and speaker counts, never a single blended
headline.

Conditions (exactly these six; anything else is rejected):
    clean          quiet room, read Telugu, 16 kHz
    noise          real background noise (street, fan, TV, market)
    phone_8k       8 kHz audio (phone call / 8 kHz export)
    code_mix       Telugu-English mixed speech (Latin words kept as spoken)
    fast           fast / hurried speech
    multi_speaker  more than one speaker in the clip

Manifest: CSV or JSONL. CSV columns: ``path,reference_text,condition,
speaker,region,duration_s`` (only the first three are required).
JSONL keys: ``path`` (or ``audio``), ``reference_text`` (or ``raw_text``),
``condition``, ``speaker``, ``region``, ``duration_s``, ``utt_id``.
Audio paths are resolved against the working directory first, then
against the manifest's folder. A missing audio file fails the run and
names the path — clips are never silently skipped.

Scoring reuses the frozen bench's pieces: the same normalizer
(``NormalizationConfig`` defaults, digit policy spoken), the same
WER/CER/AER functions, the same 16 kHz ``prepare_utterance`` audio prep,
and the same RTF / peak-RSS measurement. The only difference is the
grouping: results are aggregated per real-world condition, and the
pooled row is labelled pooled and printed last — it is not the result.

Usage (from the repo root):
    .venv311/bin/python -m telugu_speech.evaluation.realworld \
        --asr-model telugu-asr-micro --manifest data/realworld/sample_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys

from telugu_speech.asr.infer import ModelUnavailable, build_backend
from telugu_speech.audio.preprocessing import (
    detect_clipping,
    duration_seconds,
    prepare_utterance,
)
from telugu_speech.audio.resampling import read_wav
from telugu_speech.bench import resolve_asr_model
from telugu_speech.evaluation.asr_metrics import (
    ErrorCounts,
    akshara_error_rate,
    character_error_rate,
    micro_average,
    segment_scores,
    word_error_rate,
)
from telugu_speech.evaluation.benchmark import default_result_path, write_benchmark_json
from telugu_speech.evaluation.tts_metrics import Stopwatch, peak_rss_bytes, real_time_factor
from telugu_speech.text.normalization import NormalizationConfig, normalize_text

CONDITIONS = ("clean", "noise", "phone_8k", "code_mix", "fast", "multi_speaker")

CONDITION_NOTES = {
    "clean": "quiet room, read Telugu, 16 kHz",
    "noise": "real background noise (street, fan, TV, market)",
    "phone_8k": "8 kHz audio (phone call / 8 kHz export)",
    "code_mix": "Telugu-English mixed speech (Latin words kept as spoken)",
    "fast": "fast / hurried speech",
    "multi_speaker": "more than one speaker in the clip",
}

# Warn (do not gate) when a condition has fewer clips / speakers than this:
# a real-world set is small by nature, but a 2-clip condition is not a result.
WARN_MIN_CLIPS = 10
WARN_MIN_SPEAKERS = 3


@dataclass(frozen=True)
class RealWorldClip:
    utt_id: str
    audio: str
    reference_text: str
    condition: str
    speaker: str
    region: str = "unspecified"
    duration_s: float | None = None


def _resolve_audio(raw: str, manifest_dir: Path) -> str:
    """Working directory first, then the manifest's folder."""
    candidate = Path(raw)
    if candidate.is_file():
        return str(candidate)
    nested = manifest_dir / raw
    if nested.is_file():
        return str(nested)
    return str(candidate)


def _clip_from_mapping(payload: dict[str, object], source: str, line_no: int) -> RealWorldClip:
    def _str(key: str, *aliases: str) -> str | None:
        for name in (key, *aliases):
            value = payload.get(name)
            if value is not None and str(value).strip() != "":
                return str(value)
        return None

    audio = _str("path", "audio")
    reference = _str("reference_text", "raw_text")
    condition = _str("condition")
    if audio is None or reference is None or condition is None:
        raise ValueError(f"{source}:{line_no} needs path, reference_text and condition")
    if condition not in CONDITIONS:
        raise ValueError(
            f"{source}:{line_no} unknown condition {condition!r}; "
            f"allowed: {', '.join(CONDITIONS)}"
        )
    duration = payload.get("duration_s")
    return RealWorldClip(
        utt_id=_str("utt_id") or Path(audio).stem,
        audio=audio,
        reference_text=reference,
        condition=condition,
        speaker=_str("speaker") or "unknown",
        region=_str("region") or "unspecified",
        duration_s=None if duration is None or str(duration).strip() == "" else float(duration),
    )


def load_manifest(path: Path) -> list[RealWorldClip]:
    """Read a CSV or JSONL real-world manifest. Missing audio fails loudly."""
    if not path.is_file():
        raise SystemExit(f"manifest not found: `{path}` (see docs/REALWORLD.md)")
    rows: list[RealWorldClip] = []
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            for line_no, payload in enumerate(csv.DictReader(handle), start=2):
                rows.append(_clip_from_mapping(payload, str(path), line_no))
    elif path.suffix.lower() in {".jsonl", ".json"}:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be an object")
            rows.append(_clip_from_mapping(payload, str(path), line_no))
    else:
        raise SystemExit(f"manifest must be .csv or .jsonl, got `{path}`")
    if not rows:
        raise SystemExit(f"manifest is empty: `{path}`")
    missing = [row.audio for row in rows if not Path(row.audio).is_file()]
    if missing:
        # Resolve first: a path may exist relative to the manifest folder.
        resolved = [_resolve_audio(row.audio, path.parent) for row in rows]
        rows = [RealWorldClip(**{**row.__dict__, "audio": audio}) for row, audio in zip(rows, resolved)]
        missing = [row.audio for row in rows if not Path(row.audio).is_file()]
        if missing:
            raise FileNotFoundError(
                "real-world manifest references missing audio — nothing was scored:\n"
                + "\n".join(f"  {item}" for item in missing)
            )
    return rows


def _dir_bytes(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def condition_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Micro-averaged WER/CER/AER per condition, in CONDITIONS order.

    ``rows`` carry ErrorCounts under ``wer``/``cer``/``aer`` and
    ``numerals``/``latin`` (or None), plus ``elapsed_s``/``duration_s``.
    Pure function: no model, no audio, no I/O.
    """
    ordered: list[dict[str, object]] = []
    for condition in CONDITIONS:
        members = [row for row in rows if row["condition"] == condition]
        if not members:
            continue
        audio_s = sum(float(row["duration_s"]) for row in members)
        elapsed_s = sum(float(row["elapsed_s"]) for row in members)
        ordered.append(
            {
                "condition": condition,
                "clips": len(members),
                "speakers": len({str(row["speaker"]) for row in members}),
                "audio_s": audio_s,
                "wer": micro_average([row["wer"] for row in members]),  # type: ignore[misc]
                "cer": micro_average([row["cer"] for row in members]),  # type: ignore[misc]
                "aer": micro_average([row["aer"] for row in members]),  # type: ignore[misc]
                "numeral_wer": micro_average(
                    [row["numerals"] for row in members if row["numerals"] is not None]  # type: ignore[misc]
                ),
                "latin_wer": micro_average(
                    [row["latin"] for row in members if row["latin"] is not None]  # type: ignore[misc]
                ),
                "rtf": real_time_factor(elapsed_s, audio_s),
            }
        )
    return ordered


def pooled_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    """One pooled row over all conditions. Labelled pooled, never the headline."""
    if not rows:
        return {"condition": "pooled", "clips": 0, "speakers": 0, "audio_s": 0.0,
                "wer": None, "cer": None, "aer": None,
                "numeral_wer": None, "latin_wer": None, "rtf": None}
    audio_s = sum(float(row["duration_s"]) for row in rows)
    elapsed_s = sum(float(row["elapsed_s"]) for row in rows)
    return {
        "condition": "pooled",
        "clips": len(rows),
        "speakers": len({str(row["speaker"]) for row in rows}),
        "audio_s": audio_s,
        "wer": micro_average([row["wer"] for row in rows]),  # type: ignore[misc]
        "cer": micro_average([row["cer"] for row in rows]),  # type: ignore[misc]
        "aer": micro_average([row["aer"] for row in rows]),  # type: ignore[misc]
        "numeral_wer": micro_average(
            [row["numerals"] for row in rows if row["numerals"] is not None]  # type: ignore[misc]
        ),
        "latin_wer": micro_average(
            [row["latin"] for row in rows if row["latin"] is not None]  # type: ignore[misc]
        ),
        "rtf": real_time_factor(elapsed_s, audio_s),
    }


def _rate(counts: ErrorCounts | None) -> float | None:
    return None if counts is None else counts.rate


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_condition_table(
    stats: list[dict[str, object]], pooled: dict[str, object], model_id: str
) -> str:
    """Per-condition table. The pooled row is last and labelled as pooled."""
    lines = [
        f"Real-world ASR — {model_id} ({date.today().isoformat()})",
        "",
        "condition      n  speakers  audio_s    WER     CER     AER     numWER  latWER  RTF",
    ]
    for row in stats:
        lines.append(
            f"{str(row['condition']):<14} {int(row['clips']):>2}  {int(row['speakers']):>8}  "
            f"{float(row['audio_s']):>7.1f}  {_fmt(row['wer']):>7}  {_fmt(row['cer']):>7}  "
            f"{_fmt(row['aer']):>7}  {_fmt(row['numeral_wer']):>7}  {_fmt(row['latin_wer']):>7}  {_fmt(row['rtf']):>5}"
        )
    lines.append(
        f"{'pooled *':<14} {int(pooled['clips']):>2}  {int(pooled['speakers']):>8}  "
        f"{float(pooled['audio_s']):>7.1f}  {_fmt(pooled['wer']):>7}  {_fmt(pooled['cer']):>7}  "
        f"{_fmt(pooled['aer']):>7}  {_fmt(pooled['numeral_wer']):>7}  {_fmt(pooled['latin_wer']):>7}  {_fmt(pooled['rtf']):>5}"
    )
    lines.append("")
    lines.append("* pooled = all conditions mixed. Do not quote it alone; quote per-condition rows.")
    small = [row for row in stats if int(row["clips"]) < WARN_MIN_CLIPS or int(row["speakers"]) < WARN_MIN_SPEAKERS]
    if small:
        names = ", ".join(str(row["condition"]) for row in small)
        lines.append(f"Small-n warning (n<{WARN_MIN_CLIPS} clips or <{WARN_MIN_SPEAKERS} speakers): {names}. Not quotable yet.")
    return "\n".join(lines) + "\n"


def run_realworld(
    manifest: Path,
    model_path: Path,
    model_id: str,
    revision: str | None = None,
    max_per_condition: int | None = None,
    out_path: Path | None = None,
    report_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Score a real-world manifest. Returns (json_payload, report_text)."""
    clips = load_manifest(manifest)
    if max_per_condition is not None:
        kept: list[RealWorldClip] = []
        seen: dict[str, int] = {}
        for clip in clips:
            if seen.get(clip.condition, 0) >= max_per_condition:
                continue
            seen[clip.condition] = seen.get(clip.condition, 0) + 1
            kept.append(clip)
        clips = kept
    config = NormalizationConfig()
    backend = build_backend({"backend": "faster_whisper", "model_size": str(model_path)})
    scored: list[dict[str, object]] = []
    error: str | None = None
    try:
        for clip in clips:
            audio_path = Path(clip.audio)
            try:
                raw, _rate_hz = read_wav(audio_path)
                clip_fraction = detect_clipping(raw)
            except Exception:
                clip_fraction = None
            audio, _spans = prepare_utterance(audio_path)
            duration = clip.duration_s
            if duration is None:
                duration = duration_seconds(audio, 16000)
            with Stopwatch() as watch:
                hypothesis = backend.transcribe(audio_path)
            _raw_ref, norm_ref = normalize_text(clip.reference_text, config)
            _raw_hyp, norm_hyp = normalize_text(hypothesis, config)
            segments = segment_scores(norm_ref, norm_hyp)
            scored.append(
                {
                    "utt_id": clip.utt_id,
                    "audio": clip.audio,
                    "condition": clip.condition,
                    "speaker": clip.speaker,
                    "region": clip.region,
                    "duration_s": duration,
                    "elapsed_s": watch.elapsed_s,
                    "rtf": real_time_factor(watch.elapsed_s, duration),
                    "clipping_fraction": clip_fraction,
                    "hypothesis": hypothesis,
                    "wer": word_error_rate(norm_ref, norm_hyp),
                    "cer": character_error_rate(norm_ref, norm_hyp),
                    "aer": akshara_error_rate(norm_ref, norm_hyp),
                    "numerals": segments.numerals,
                    "latin": segments.latin,
                }
            )
    except ModelUnavailable as exc:
        error = f"not measured. {exc}"
        scored = []
    except (OSError, RuntimeError) as exc:
        error = str(exc)
        scored = []
    stats = condition_stats(scored)
    pooled = pooled_stats(scored)
    text = render_condition_table(stats, pooled, model_id)
    if error is not None:
        text += f"\nERROR: {error}\n"
    payload: dict[str, object] = {
        "tool": "realworld",
        "version": 1,
        "date": date.today().isoformat(),
        "model_id": model_id,
        "model_path": str(model_path),
        "model_bytes": _dir_bytes(model_path),
        "model_revision_sha": revision,
        "manifest": str(manifest),
        "normalizer": {"digit_policy": config.digit_policy},
        "error": error,
        "clips": len(scored),
        "speakers": len({str(row["speaker"]) for row in scored}),
        "peak_rss_bytes": peak_rss_bytes(),
        "conditions": [
            {key: row[key] for key in ("condition", "clips", "speakers", "audio_s", "wer", "cer", "aer", "numeral_wer", "latin_wer", "rtf")}
            for row in stats
        ],
        "pooled": {key: pooled[key] for key in ("clips", "speakers", "audio_s", "wer", "cer", "aer", "numeral_wer", "latin_wer", "rtf")},
        "per_clip": [
            {
                "utt_id": row["utt_id"],
                "audio": row["audio"],
                "condition": row["condition"],
                "speaker": row["speaker"],
                "region": row["region"],
                "duration_s": row["duration_s"],
                "rtf": row["rtf"],
                "clipping_fraction": row["clipping_fraction"],
                "wer": _rate(row["wer"]),  # type: ignore[arg-type]
                "cer": _rate(row["cer"]),  # type: ignore[arg-type]
                "aer": _rate(row["aer"]),  # type: ignore[arg-type]
                "numeral_wer": _rate(row["numerals"]),  # type: ignore[arg-type]
                "latin_wer": _rate(row["latin"]),  # type: ignore[arg-type]
                "hypothesis": row["hypothesis"],
            }
            for row in scored
        ],
    }
    out = out_path or default_result_path(f"{model_id}-realworld")
    report = report_path or out.with_suffix(".md")
    write_benchmark_json(out, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    return payload, text + f"\nJSON: {out}\nReport: {report}\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="telugu_speech.evaluation.realworld",
        description="Score ASR on real-world recordings, per condition (never one blended number).",
    )
    parser.add_argument("--asr-model", required=True, help="Model id under models/ or a path to a model folder.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV or JSONL manifest (see docs/REALWORLD.md).")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path (default bench/results/<model>-realworld/<date>.json).")
    parser.add_argument("--report", type=Path, default=None, help="Markdown report path (default: same stem as --out, .md).")
    parser.add_argument("--revision", default=None, help="Model revision SHA to record (never guessed; omit if unknown).")
    parser.add_argument("--max-per-condition", type=int, default=None, help="Cap clips per condition (smoke test only).")
    parser.add_argument("--list-conditions", action="store_true", help="Print the six conditions and exit.")
    args = parser.parse_args(argv)
    if args.list_conditions:
        for name in CONDITIONS:
            print(f"{name}: {CONDITION_NOTES[name]}")
        return
    if args.max_per_condition is not None and args.max_per_condition < 1:
        raise SystemExit("--max-per-condition must be at least 1")
    model_path = resolve_asr_model(args.asr_model)
    try:
        _payload, text = run_realworld(
            args.manifest,
            model_path,
            args.asr_model,
            revision=args.revision,
            max_per_condition=args.max_per_condition,
            out_path=args.out,
            report_path=args.report,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
