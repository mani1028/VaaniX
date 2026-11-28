"""Run ASR backends over a manifest and write slice + JSON reports.

A benchmark result is only comparable when it names the set, the speaker
count, the model revision, and the normalizer. One pooled WER is not a
release metric: every result here carries WER + CER + AER (akshara error
rate), raw and normalized, plus per-region / per-condition / code-mix slices.

Telugu is agglutinative, so WER runs roughly 3-4x the akshara error rate on
the same transcript. Never publish the WER without the CER and AER beside it.

JSON results are written to ``bench/results/<model>/<date>.json`` and are
never hand-edited; re-run the bench to change a number.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path

import yaml

from telugu_speech.asr.dataset import Utterance, read_manifest
from telugu_speech.asr.infer import AsrBackend, ModelUnavailable, build_backend
from telugu_speech.audio.preprocessing import duration_seconds, prepare_utterance
from telugu_speech.evaluation.asr_metrics import (
    ErrorCounts,
    akshara_error_rate,
    character_error_rate,
    micro_average,
    segment_scores,
    word_error_rate,
)
from telugu_speech.evaluation.reports import group_slices, render_report, write_report
from telugu_speech.evaluation.tts_metrics import Stopwatch, file_size_bytes, peak_rss_bytes, real_time_factor
from telugu_speech.text.normalization import NormalizationConfig, normalize_text

MIN_CLIPS_PER_SET = 200
MIN_SPEAKERS_PER_SET = 20


def load_eval_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a mapping")
    for key in ("manifest", "report", "models", "max_clips"):
        if key not in payload:
            raise ValueError(f"evaluation config missing {key}")
    return payload


def validate_benchmark_set(
    utterances: list[Utterance],
    min_clips: int = MIN_CLIPS_PER_SET,
    min_speakers: int = MIN_SPEAKERS_PER_SET,
) -> dict[str, int]:
    """Check a frozen test set is big enough to quote. Raises ValueError."""
    speakers = {row.speaker_id for row in utterances}
    problems: list[str] = []
    if len(utterances) < min_clips:
        problems.append(f"{len(utterances)} clips, need at least {min_clips}")
    if len(speakers) < min_speakers:
        problems.append(f"{len(speakers)} speakers, need at least {min_speakers}")
    if problems:
        raise ValueError("benchmark set is too small to quote: " + "; ".join(problems))
    return {"clips": len(utterances), "speakers": len(speakers)}


def assert_speakers_disjoint_from(test: list[Utterance], train: list[Utterance]) -> None:
    """Fail if any test speaker also appears in the train manifest."""
    overlap = {row.speaker_id for row in test} & {row.speaker_id for row in train}
    if overlap:
        raise ValueError(f"test/train speaker overlap: {sorted(overlap)}")


def _rate(counts: ErrorCounts | None) -> float | None:
    return None if counts is None else counts.rate


def slice_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Micro-averaged WER/CER/AER per slice, with clip and speaker counts."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[("region", str(row["region"]))].append(row)
        groups[("condition", str(row["condition"]))].append(row)
        groups[("code_mix", "yes" if row["code_mix"] else "no")].append(row)
        groups[("overall", "all")].append(row)
    rendered: list[dict[str, object]] = []
    for (kind, name), members in sorted(groups.items()):
        rendered.append(
            {
                "kind": kind,
                "name": name,
                "clips": len(members),
                "speakers": len({str(member["speaker"]) for member in members}),
                "wer": micro_average([member["wer"] for member in members]),  # type: ignore[misc]
                "cer": micro_average([member["cer"] for member in members]),  # type: ignore[misc]
                "aer": micro_average([member["aer"] for member in members]),  # type: ignore[misc]
                "wer_raw": micro_average([member["wer_raw"] for member in members]),  # type: ignore[misc]
                "cer_raw": micro_average([member["cer_raw"] for member in members]),  # type: ignore[misc]
                "aer_raw": micro_average([member["aer_raw"] for member in members]),  # type: ignore[misc]
                "numeral_wer": micro_average(
                    [member["numerals"] for member in members if member["numerals"] is not None]  # type: ignore[misc]
                ),
                "latin_wer": micro_average(
                    [member["latin"] for member in members if member["latin"] is not None]  # type: ignore[misc]
                ),
            }
        )
    return rendered


def default_result_path(model_id: str, root: Path | None = None) -> Path:
    """bench/results/<model>/<date>.json — the only place JSON results live."""
    base = root or Path("bench/results")
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in model_id)
    return base / safe / f"{date.today().isoformat()}.json"


def write_benchmark_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_benchmark(
    manifest: Path,
    report_path: Path,
    backends: list[tuple[str, AsrBackend]],
    max_clips: int,
    model_bytes: dict[str, int | None] | None = None,
    model_revisions: dict[str, str] | None = None,
    json_path: Path | None = None,
    require_frozen: bool = False,
) -> str:
    utterances = read_manifest(manifest)[:max_clips]
    if require_frozen:
        validate_benchmark_set(utterances)
    config = NormalizationConfig()
    model_results: list[dict[str, object]] = []
    json_results: list[dict[str, object]] = []
    sizes = model_bytes or {}
    revisions = model_revisions or {}
    for model_id, backend in backends:
        scored: list[dict[str, object]] = []
        elapsed = 0.0
        audio_s = 0.0
        error = None
        try:
            for utterance in utterances:
                audio_path = Path(utterance.audio)
                if not audio_path.is_file():
                    raise FileNotFoundError(f"missing audio {audio_path}")
                audio, _spans = prepare_utterance(audio_path)
                with Stopwatch() as watch:
                    hypothesis = backend.transcribe(audio_path)
                elapsed += watch.elapsed_s
                duration = utterance.duration_s
                if duration is None:
                    duration = duration_seconds(audio, 16000)
                audio_s += duration
                _raw, normalized_hypothesis = normalize_text(hypothesis, config)
                segments = segment_scores(utterance.norm_text, normalized_hypothesis)
                clip_rtf = real_time_factor(watch.elapsed_s, duration)
                scored.append(
                    {
                        "utt_id": utterance.utt_id,
                        "speaker": utterance.speaker_id,
                        "region": utterance.region,
                        "condition": utterance.condition,
                        "code_mix": utterance.code_mix,
                        "duration_s": duration,
                        "rtf": clip_rtf,
                        "wer": word_error_rate(utterance.norm_text, normalized_hypothesis),
                        "cer": character_error_rate(utterance.norm_text, normalized_hypothesis),
                        "aer": akshara_error_rate(utterance.norm_text, normalized_hypothesis),
                        "wer_raw": word_error_rate(utterance.raw_text, hypothesis),
                        "cer_raw": character_error_rate(utterance.raw_text, hypothesis),
                        "aer_raw": akshara_error_rate(utterance.raw_text, hypothesis),
                        "numerals": segments.numerals,
                        "latin": segments.latin,
                    }
                )
        except ModelUnavailable as exc:
            error = f"not measured. {exc}"
            scored = []
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            error = str(exc)
            scored = []
        slices = slice_stats(scored)
        overall = next((row for row in slices if row["kind"] == "overall"), None)
        model_results.append(
            {
                "model_id": model_id,
                "error": error,
                "utterances": len(scored),
                "rtf": real_time_factor(elapsed, audio_s),
                "peak_rss_bytes": peak_rss_bytes(),
                "model_bytes": sizes.get(model_id),
                "slices": group_slices(scored),
            }
        )
        json_results.append(
            {
                "model_id": model_id,
                "model_revision_sha": revisions.get(model_id),
                "date": date.today().isoformat(),
                "manifest": str(manifest),
                "normalizer": {"digit_policy": config.digit_policy},
                "error": error,
                "clips": len(scored),
                "speakers": len({str(row["speaker"]) for row in scored}),
                "wer": None if overall is None else overall["wer"],
                "cer": None if overall is None else overall["cer"],
                "aer": None if overall is None else overall["aer"],
                "wer_raw": None if overall is None else overall["wer_raw"],
                "cer_raw": None if overall is None else overall["cer_raw"],
                "aer_raw": None if overall is None else overall["aer_raw"],
                "rtf": real_time_factor(elapsed, audio_s),
                "peak_rss_bytes": peak_rss_bytes(),
                "model_bytes": sizes.get(model_id),
                "slices": slices,
                "utterances": [
                    {
                        **{key: row[key] for key in ("utt_id", "speaker", "region", "condition", "code_mix", "duration_s", "rtf")},
                        "wer": _rate(row["wer"]),  # type: ignore[arg-type]
                        "cer": _rate(row["cer"]),  # type: ignore[arg-type]
                        "aer": _rate(row["aer"]),  # type: ignore[arg-type]
                        "wer_raw": _rate(row["wer_raw"]),  # type: ignore[arg-type]
                        "cer_raw": _rate(row["cer_raw"]),  # type: ignore[arg-type]
                        "aer_raw": _rate(row["aer_raw"]),  # type: ignore[arg-type]
                    }
                    for row in scored
                ],
            }
        )
    text = render_report(model_results)
    extra = ["", "## Accuracy (WER + CER + AER)", ""]
    for result in json_results:
        revision = result.get("model_revision_sha") or "not recorded"
        extra.append(
            f"- {result['model_id']}: WER {result['wer']}, CER {result['cer']}, "
            f"AER {result['aer']} (normalized, n={result['clips']}), "
            f"revision {revision}."
        )
    text = text + "\n".join(extra) + "\n"
    write_report(report_path, text)
    if json_path is not None:
        write_benchmark_json(json_path, {"models": json_results})
    return text


def backends_from_config(config: dict[str, object]) -> list[tuple[str, AsrBackend]]:
    models = config["models"]
    if not isinstance(models, list):
        raise ValueError("models must be a list")
    built: list[tuple[str, AsrBackend]] = []
    for spec in models:
        if not isinstance(spec, dict):
            raise ValueError("each model must be a mapping")
        built.append((str(spec.get("id", spec.get("backend"))), build_backend(spec)))
    return built


def checkpoint_bytes(path: Path | None) -> int | None:
    return file_size_bytes(path)
