"""Frozen-set benchmark CLI.

Usage:
    python -m telugu_speech.bench --asr-model telugu-asr-micro \\
        --sets fleurs-te kathbath-te cv-te --n 200

Datasets are NOT in this repo and are never invented here. Each set must
already exist as a manifest at:

    data/benchmarks/<set>/manifest.jsonl

(audio paths inside the manifest are used as written — absolute, or
relative to the directory you run from, matching run_benchmark. They are
NOT resolved against the set folder.) If a manifest is missing, this
exits 2 and
names the exact path it expected. Use ``--allow-small`` only for a smoke
test; a quoted result needs n>=200 clips from >=20 speakers per set.

TTS intelligibility is opt-in: pass ``--tts-voice telugu_srinivas`` and
``--tts-sentences <file>`` (one Telugu sentence per line). Each sentence is
synthesized, transcribed back with the ASR model, and scored. Without those
flags the TTS section is reported as ``not_measured``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from telugu_speech.asr.dataset import read_manifest
from telugu_speech.asr.infer import ModelUnavailable, build_backend
from telugu_speech.evaluation.benchmark import (
    assert_speakers_disjoint_from,
    MIN_CLIPS_PER_SET,
    MIN_SPEAKERS_PER_SET,
    default_result_path,
    run_benchmark,
    validate_benchmark_set,
    write_benchmark_json,
)
from telugu_speech.evaluation.asr_metrics import (
    akshara_error_rate,
    character_error_rate,
    micro_average,
    word_error_rate,
)
from telugu_speech.text.normalization import NormalizationConfig, normalize_text

SETS = ("fleurs-te", "kathbath-te", "cv-te")
DEFAULT_DATA_ROOT = Path("data/benchmarks")


def manifest_for(data_root: Path, dataset: str) -> Path:
    return data_root / dataset / "manifest.jsonl"


def resolve_asr_model(spec: str) -> Path:
    """A model id (``models/<id>``) or a path. Fails naming what was tried."""
    candidate = Path(spec)
    if candidate.is_dir():
        return candidate
    under_models = Path("models") / spec
    if under_models.is_dir():
        return under_models
    raise SystemExit(
        f"asr model not found: tried `{candidate}` and `{under_models}`. "
        "Place the model folder there first; no model is downloaded by the bench."
    )


def _fail_missing(missing: list[Path]) -> "SystemExit":
    lines = ["benchmark dataset missing — nothing was scored:"]
    for path in missing:
        lines.append(f"  expected manifest at `{path}`")
    lines.append("Datasets are not in this repo. Prepare each set first (see docs/benchmark_results.md),")
    lines.append("then re-run. No data was invented.")
    return SystemExit("\n".join(lines))


def score_tts_intelligibility(
    asr_backend, voice: str, sentences_path: Path, out_path: Path
) -> dict[str, object]:
    """Synthesize sentences, transcribe them back, score the round trip."""
    from telugu_speech.tts.voices import backend_for_voice

    sentences = [line.strip() for line in sentences_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not sentences:
        raise SystemExit(f"no sentences in `{sentences_path}`")
    tts = backend_for_voice(voice)
    if tts is None:
        raise SystemExit(f"unknown voice `{voice}`; known voices are telugu_srinivas and telugu_sravani")
    import tempfile

    config = NormalizationConfig()
    wers, cers, aers = [], [], []
    try:
        with tempfile.TemporaryDirectory() as directory:
            for index, sentence in enumerate(sentences):
                out = Path(directory) / f"tts-{index}.wav"
                tts.synthesize(sentence, None, "", out)
                hypothesis = asr_backend.transcribe(out)
                _raw, norm_hyp = normalize_text(hypothesis, config)
                _raw_ref, norm_ref = normalize_text(sentence, config)
                wers.append(word_error_rate(norm_ref, norm_hyp))
                cers.append(character_error_rate(norm_ref, norm_hyp))
                aers.append(akshara_error_rate(norm_ref, norm_hyp))
    except ModelUnavailable as exc:
        return {"status": "not_measured", "error": str(exc)}
    payload: dict[str, object] = {
        "status": "measured",
        "voice": voice,
        "sentences": len(sentences),
        "wer": micro_average(wers),
        "cer": micro_average(cers),
        "aer": micro_average(aers),
        "date": date.today().isoformat(),
    }
    write_benchmark_json(out_path, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="telugu_speech.bench", description="Score ASR on frozen Telugu test sets.")
    parser.add_argument("--asr-model", required=True, help="Model id under models/ or a path to a model folder.")
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS), help="Test sets to score.")
    parser.add_argument("--n", type=int, default=MIN_CLIPS_PER_SET, help="Max clips per set (default 200).")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root holding <set>/manifest.jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path (default bench/results/<model>/<date>.json).")
    parser.add_argument("--min-speakers", type=int, default=MIN_SPEAKERS_PER_SET)
    parser.add_argument("--allow-small", action="store_true", help="Skip the n>=200 / >=20 speakers gate (smoke test only).")
    parser.add_argument("--train-manifest", type=Path, default=None, help="Optional train manifest to check speaker-disjointness against.")
    parser.add_argument("--revision", default=None, help="Model revision SHA to record (never guessed; omit if unknown).")
    parser.add_argument("--tts-voice", default=None, help="Opt-in TTS intelligibility voice, e.g. telugu_srinivas.")
    parser.add_argument("--tts-sentences", type=Path, default=None, help="One Telugu sentence per line, for TTS intelligibility.")
    parser.add_argument(
        "--crosscheck-jiwer",
        action="store_true",
        help="Also print jiwer WER for comparison (needs pip install '.[bench]').",
    )
    args = parser.parse_args(argv)

    model_path = resolve_asr_model(args.asr_model)
    missing = [manifest_for(args.data_root, name) for name in args.sets if not manifest_for(args.data_root, name).is_file()]
    if missing:
        raise _fail_missing(missing)
    if args.n < 1:
        raise SystemExit("--n must be at least 1")

    try:
        backend = build_backend({"backend": "faster_whisper", "model_size": str(model_path)})
    except ModelUnavailable as exc:
        raise SystemExit(f"asr model unavailable: {exc}") from exc

    out = args.out or default_result_path(args.asr_model)
    for name in args.sets:
        manifest = manifest_for(args.data_root, name)
        utterances = read_manifest(manifest)
        if not args.allow_small:
            try:
                validate_benchmark_set(utterances, min_clips=args.n, min_speakers=args.min_speakers)
            except ValueError as exc:
                raise SystemExit(f"{name}: {exc} (use --allow-small only for a smoke test)") from exc
        if args.train_manifest is not None:
            train = read_manifest(args.train_manifest)
            try:
                assert_speakers_disjoint_from(utterances, train)
            except ValueError as exc:
                raise SystemExit(f"{name}: {exc}") from exc
        report = Path(f"bench/results/{args.asr_model}/{name}.md")
        text = run_benchmark(
            manifest,
            report,
            [(args.asr_model, backend)],
            max_clips=args.n,
            model_revisions={args.asr_model: args.revision} if args.revision else None,
            json_path=out if len(args.sets) == 1 else out.parent / f"{name}.json",
            require_frozen=not args.allow_small,
        )
        print(text)
        print(f"JSON: {out if len(args.sets) == 1 else out.parent / f'{name}.json'}")
        if args.crosscheck_jiwer:
            try:
                import jiwer  # type: ignore[import-not-found]
            except ImportError:
                raise SystemExit("jiwer is not installed; run pip install '.[bench]' first") from None
            # Second pass, same clips: jiwer WER must match our WER on the
            # same normalized texts. A mismatch means a normalizer bug.
            refs, hyps = [], []
            for row in utterances[: args.n]:
                hyp = backend.transcribe(Path(row.audio))
                _raw, norm_hyp = normalize_text(hyp, NormalizationConfig())
                refs.append(row.norm_text)
                hyps.append(norm_hyp)
            measures = jiwer.compute_measures(refs, hyps)
            print(f"{name}: jiwer WER {measures['wer']:.4f} (cross-check, same clips)") 

    if args.tts_voice is not None or args.tts_sentences is not None:
        if args.tts_voice is None or args.tts_sentences is None:
            raise SystemExit("--tts-voice and --tts-sentences must be given together")
        if not args.tts_sentences.is_file():
            raise SystemExit(f"tts sentences file missing: `{args.tts_sentences}`")
        result = score_tts_intelligibility(backend, args.tts_voice, args.tts_sentences, out.parent / "tts-intelligibility.json")
        print(f"TTS intelligibility: {result}")
    else:
        print("TTS intelligibility: not_measured (pass --tts-voice and --tts-sentences to measure it)")


if __name__ == "__main__":
    main()
