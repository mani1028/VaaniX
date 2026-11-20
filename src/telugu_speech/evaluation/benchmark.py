"""Run ASR backends over a manifest and write a slice report."""

from __future__ import annotations

from pathlib import Path

import yaml

from telugu_speech.asr.dataset import read_manifest
from telugu_speech.asr.infer import AsrBackend, ModelUnavailable, build_backend
from telugu_speech.audio.preprocessing import duration_seconds, prepare_utterance
from telugu_speech.evaluation.asr_metrics import character_error_rate, word_error_rate
from telugu_speech.evaluation.reports import group_slices, render_report, write_report
from telugu_speech.evaluation.tts_metrics import Stopwatch, file_size_bytes, peak_rss_bytes, real_time_factor
from telugu_speech.text.normalization import NormalizationConfig, normalize_text


def load_eval_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation config must be a mapping")
    for key in ("manifest", "report", "models", "max_clips"):
        if key not in payload:
            raise ValueError(f"evaluation config missing {key}")
    return payload


def run_benchmark(
    manifest: Path,
    report_path: Path,
    backends: list[tuple[str, AsrBackend]],
    max_clips: int,
    model_bytes: dict[str, int | None] | None = None,
) -> str:
    utterances = read_manifest(manifest)[:max_clips]
    config = NormalizationConfig()
    model_results: list[dict[str, object]] = []
    sizes = model_bytes or {}
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
                scored.append(
                    {
                        "region": utterance.region,
                        "condition": utterance.condition,
                        "code_mix": utterance.code_mix,
                        "wer": word_error_rate(utterance.norm_text, normalized_hypothesis),
                        "cer": character_error_rate(utterance.norm_text, normalized_hypothesis),
                    }
                )
        except ModelUnavailable as exc:
            error = f"not measured. {exc}"
            scored = []
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            error = str(exc)
            scored = []
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
    text = render_report(model_results)
    write_report(report_path, text)
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
